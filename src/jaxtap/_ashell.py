# Copyright 2026- The jax-tap Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
A-shell: the ``with tap.record():`` context manager form.

Monkeypatches ``jax.lax.scan`` (and ``jax.lax.while_loop``) for the duration
of the context, so that any call to those primitives inside the ``with`` block
is automatically instrumented — zero change to user code.

The patch machinery is ported from the hardened #964 patch in
``reference-964/progress_bar.py`` (tqdm / display-thread / output_file
stripped; jaxtap is emission-only).

Thread safety
-------------
A single registry lock (``_registry_lock``) is held ONLY around the
empty→non-empty and non-empty→empty registry transitions (enter/exit critical
sections).  It is NEVER held across the trace hot path.

Attribution rules (mirroring the #964 decision table):
  - Exactly ONE active context: any calling thread is attributed to it.
    This is the "delegate" pattern — enter on main, run work on a worker
    thread, still gets taps.  CONTRACT (L4): with n=1, bystander pollution
    is indistinguishable from legitimate delegation — documented boundary,
    not fixed.
  - TWO or more active contexts: attribution requires an exact match between
    the calling thread and the context's ``_owner_thread`` (the thread that
    called ``__enter__``).  A bystander thread passes through untapped rather
    than risk guessing wrong.

Re-entrant (nested) contexts
-----------------------------
Nesting DIFFERENT ``tap.record()`` objects is supported.  With ≥2 simultaneous
contexts on the same owner thread, the INNERMOST context (last-entered) wins
for taps fired during its scope.  This is the natural outcome of the
``owned[-1]`` selection in ``_select_ctx``.

Reusing the SAME context object in nested ``with`` blocks raises RuntimeError
immediately on the inner ``__enter__`` (L1 guard).

Depth counter
-------------
A single thread-local ``_depth`` counter is incremented by EVERY patched
primitive (_patched_scan AND _patched_while).  This prevents double-
instrumentation across primitives: when ``_intercept_scan`` calls
``verbose(g)(init, xs)`` and the B-core rewrites re-enter ``jax.lax.scan``
(or ``jax.lax.while_loop``), they see depth > 0 and pass through unchanged.

In addition, a separate thread-local flag ``_in_interpret`` suppresses A-shell
interception for the current thread while ``interpret()`` (the B-core walker)
is executing its ``make_jaxpr`` trace.  This prevents double-instrumentation
when ``verbose()`` or ``record(f)`` is called inside an active context (L5).

Foreign patch chaining
----------------------
On the empty→non-empty transition, the current value of ``jax.lax.scan`` is
captured as ``_session_scan`` (analogously for while_loop) — unless it IS
already ``_patched_scan`` (self-capture guard).  On the non-empty→empty
transition, we restore ONLY if ``jax.lax.scan is _patched_scan``; if a
foreign patch was installed OVER us during our session, we leave it alone and
warn once.

Known boundaries
----------------
L4 — Thread delegation: with ONE active context, any thread's scans are
     attributed to it.  At n=1, bystander pollution is indistinguishable from
     legitimate delegation (same as #964 precedent).  Documented; not fixed.

L7 — If a foreign patch lands OVER us and we exit, the pre-session foreign
     chain is forgotten.  The restored value is the pre-our-session state,
     not what existed before the foreign patch was installed.

THEORETIC — async-backend (GPU/TPU) callback-thread attribution may misroute
            under ≥2 contexts: the callback thread may differ from the originating
            thread, causing the fallback "no match" path.  Documented; not fixed.

Dynamic host routing (jit-cache staleness fix)
-----------------------------------------------
When ``verbose(g, on_step=..., ...)`` is called, the ``on_step`` callable
gets baked into the XLA compiled artifact via ``jax.debug.callback``.  If the
baked ``on_step`` closes over a SPECIFIC recorder, that recorder is written to
forever — even after ``__exit__`` (phantom emission), and even when the
compiled artifact is reused inside a DIFFERENT context (wrong-recorder).

Fix: the baked ``on_step`` is always ``_dynamic_router``, a module-level
singleton that performs a RUNTIME lookup of ``_context_registry`` on each
firing.  This ensures:
  - After exit: no active context → event dropped (no phantom emission).
  - Cache-hit in new context: new context's recorder receives the event.
  - The device-side ``select`` / ``taps`` config IS baked at trace time
    (device-side; cannot be changed dynamically) — document if a cache-hit
    occurs in a context with a DIFFERENT ``select``, the trace-time select
    wins for the device-side computation; only host routing is live.

GC self-heal (L2)
-----------------
The registry stores ``weakref.ref(self)`` rather than ``self``, so a dropped
(never-exited) context object is GC-collectable.  A ``weakref.finalize``
callback (``_cleanup``) is registered on entry and deregisters the context
plus restores the patched primitives when the object is collected.  The
finalizer is detached on successful ``__exit__`` so it does NOT fire for the
normal with-block flow.

jit-cache boundary (third direction — documented, not fixed)
-------------------------------------------------------------
A function compiled BEFORE any context is entered has NO callback baked in.
Subsequent calls inside a context produce 0 events (no retrace → no
callbacks).  Workaround: ``jax.clear_caches()`` before entering the context.
"""

from __future__ import annotations

import threading
import uuid
import warnings
import weakref
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import jax

if TYPE_CHECKING:
    from .collectors import FlightRecorder

__all__ = ["_RecordContext", "suppress_interception", "emergency_restore"]

# ---------------------------------------------------------------------------
# Captured import-time originals (never changed after module load)
# ---------------------------------------------------------------------------

_original_scan: Callable = jax.lax.scan
_original_while: Callable = jax.lax.while_loop

# ---------------------------------------------------------------------------
# Thread-local counters and guards
# ---------------------------------------------------------------------------

# Depth counter: incremented for EVERY entry into _patched_scan or _patched_while.
# Ensures re-entries from the B-core rewrites (via make_jaxpr tracing) see
# depth > 0 and pass through, preventing double-instrumentation across both
# primitives.
_depth = threading.local()

# Interception-suppression flag: set True while interpret()/make_jaxpr is running.
# When True, _patched_scan/_patched_while pass through without intercepting, so
# calling verbose()/record(f) inside an active context does NOT double-instrument
# the function's scans (L5 fix).
_in_interpret = threading.local()

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Values are weakref.ref[_RecordContext] so the registry does NOT prevent GC.
# A dropped (never-exited) context can be collected; _cleanup fires, cleans up.
_context_registry: "dict[str, weakref.ref[_RecordContext]]" = {}

# Guards the empty<->non-empty transition of _context_registry.
# NEVER held across the trace hot path or the ``yield`` of a context.
_registry_lock = threading.Lock()

# Captured on empty->non-empty: whatever jax.lax.scan / while_loop pointed
# to BEFORE we first installed _patched_scan / _patched_while.  May be a
# foreign patch rather than the import-time original.  None means no session.
_session_scan: Callable | None = None
_session_while: Callable | None = None

# Warn-once flags: raised when a foreign patch is detected over us on exit.
# Session-scoped (L6): reset on the registry empty→non-empty transition so a
# genuine clobber in a NEW session is always warned, even if a prior session
# (or a bogus exit) had already raised the flag.
_clobber_scan_warned: bool = False
_clobber_while_warned: bool = False


# ---------------------------------------------------------------------------
# GC self-heal callback (L2)
# ---------------------------------------------------------------------------


def _cleanup(key: str) -> None:
    """Weakref finalize callback: deregister *key* and restore if this was the last context.

    Runs when a ``_RecordContext`` is GC-collected without ``__exit__`` being called
    (e.g. manual ``__enter__()`` on a context that subsequently fell out of scope).
    Silent — no clobber warning (the user dropped the object; they cannot recover it).

    The finalizer is detached in ``__exit__`` so it does NOT fire for the normal
    ``with``-block flow.
    """
    global _session_scan, _session_while
    with _registry_lock:
        if key not in _context_registry:
            return  # already cleaned up by __exit__
        _context_registry.pop(key, None)
        if not _context_registry:
            if jax.lax.scan is _patched_scan:
                jax.lax.scan = (
                    _session_scan if _session_scan is not None else _original_scan
                )
            _session_scan = None
            if jax.lax.while_loop is _patched_while:
                jax.lax.while_loop = (
                    _session_while if _session_while is not None else _original_while
                )
            _session_while = None


# ---------------------------------------------------------------------------
# Emergency restore (public API exported via __init__.py)
# ---------------------------------------------------------------------------


def emergency_restore() -> None:
    """Restore ``jax.lax.scan`` and ``jax.lax.while_loop`` to session-captured originals.

    For recovering a corrupted interactive session — e.g. after a crash inside a
    ``tap.record()`` block or an interrupted kernel.

    Behaviour
    ---------
    - If our patch is on top: restores to the pre-session value (or the import-time
      original when no pre-session value was captured), clears the registry, resets
      session state and session-scoped warn-once flags.
    - If a foreign patch is on top (we were clobbered): warns and leaves the foreign
      patch in place; only clears our internal state.

    This is a last-resort mechanism.  Well-behaved code using ``with tap.record():``
    never needs it.
    """
    global _session_scan, _session_while, _clobber_scan_warned, _clobber_while_warned
    with _registry_lock:
        _context_registry.clear()
        if jax.lax.scan is _patched_scan:
            jax.lax.scan = (
                _session_scan if _session_scan is not None else _original_scan
            )
        else:
            warnings.warn(
                "jaxtap.emergency_restore: jax.lax.scan is not our patch "
                "(possibly clobbered by a foreign patch); leaving it alone.",
                UserWarning,
                stacklevel=2,
            )
        if jax.lax.while_loop is _patched_while:
            jax.lax.while_loop = (
                _session_while if _session_while is not None else _original_while
            )
        else:
            warnings.warn(
                "jaxtap.emergency_restore: jax.lax.while_loop is not our patch "
                "(possibly clobbered by a foreign patch); leaving it alone.",
                UserWarning,
                stacklevel=2,
            )
        _session_scan = None
        _session_while = None
        _clobber_scan_warned = False
        _clobber_while_warned = False


# ---------------------------------------------------------------------------
# L5 guard: suppress interception during interpret()/make_jaxpr
# ---------------------------------------------------------------------------


@contextmanager
def suppress_interception() -> Generator[None, None, None]:
    """Context manager: suppress A-shell interception for the current thread.

    Used by ``_walker.interpret()`` around its ``jax.make_jaxpr`` call so that
    any ``jax.lax.scan`` / ``jax.lax.while_loop`` resolved during tracing passes
    through the A-shell patch without triggering a context intercept.

    Without this guard, ``verbose(f)`` called inside a ``with tap.record():`` block
    would double-instrument ``f``'s scans: once via the B-core rewrite and once via
    the A-shell, causing the context recorder to also capture events from the user's
    explicitly-tapped call.
    """
    old = getattr(_in_interpret, "value", False)
    _in_interpret.value = True
    try:
        yield
    finally:
        _in_interpret.value = old


# ---------------------------------------------------------------------------
# Delegation helpers
# ---------------------------------------------------------------------------


def _underlying_scan() -> Callable:
    """Return the scan implementation to delegate pass-through calls to.

    If a foreign patch was installed before our session, we chain to it
    (true chaining, not just non-clobbering on exit).
    """
    return _session_scan if _session_scan is not None else _original_scan


def _underlying_while() -> Callable:
    """Analogous to ``_underlying_scan`` for while_loop."""
    return _session_while if _session_while is not None else _original_while


def _active_contexts() -> "list[_RecordContext]":
    """Snapshot the currently active contexts, dereferencing weak refs.

    Takes a lock-free snapshot of registry values and drops any dead refs
    (contexts that were GC'd between the snapshot and the dereference).
    """
    return [
        ctx
        for ctx in (r() for r in list(_context_registry.values()))
        if ctx is not None
    ]


def _select_ctx(active: "list[_RecordContext]") -> "_RecordContext | None":
    """Select which context owns the current call.

    With 1 active context: any thread is attributed to it (delegate pattern).
    With >=2: only the innermost context whose owner_thread matches the calling
    thread is selected; bystander threads return None (pass-through).
    """
    if len(active) == 1:
        return active[0]
    here = threading.get_ident()
    owned = [ctx for ctx in active if ctx._owner_thread == here]
    return owned[-1] if owned else None  # innermost wins for re-entrant


# ---------------------------------------------------------------------------
# Dynamic router — the singleton on_step baked into every XLA artifact
# ---------------------------------------------------------------------------


def _dynamic_router(event: Any) -> None:
    """Route a TapEvent to whichever context is active at CALL TIME.

    This function (not a specific recorder) is what gets baked into XLA
    compiled artifacts via ``jax.debug.callback``.  Because it performs a
    runtime lookup of ``_context_registry``, it correctly handles:

    - After-exit calls (phantom emission): no context active → event dropped.
    - Cache-hit inside a new context: new context's recorder receives the event.

    Both the FlightRecorder and the optional user ``on_step`` callback are
    routed here, both ``_guard``-wrapped.

    Limitation (document, not fix): if a cache-hit occurs in a context with a
    DIFFERENT ``select`` or ``taps`` config than what was baked at trace time,
    the trace-time device-side select wins for the on-device computation;
    only the host routing is live.
    """
    active = _active_contexts()
    if not active:
        return  # no active context → drop (covers post-exit phantom case)
    ctx = _select_ctx(active)
    if ctx is None or ctx._recorder is None:
        return
    from . import _guard  # lazy to avoid circular import at module load

    _guard(ctx._recorder, event)
    if ctx._extra_on_step is not None:
        _guard(ctx._extra_on_step, event)


# ---------------------------------------------------------------------------
# Patched primitives
# ---------------------------------------------------------------------------


def _patched_scan(
    f: Callable,
    init: Any,
    xs: Any = None,
    length: int | None = None,
    reverse: bool = False,
    unroll: "int | bool" = 1,
    _split_transpose: bool = False,
) -> Any:
    """Drop-in replacement for ``jax.lax.scan`` while any _RecordContext is active.

    Signature matches ``jax.lax.scan`` exactly — including the positional
    ``reverse``, ``unroll``, and ``_split_transpose`` parameters — so that
    callers passing these positionally (e.g. ``scan(f, init, xs, None, True)``)
    work identically inside and outside an active context.

    Only the outermost call (depth == 0) is instrumented; nested calls
    (from the B-core rewrites or from user-code scans discovered during
    ``make_jaxpr`` tracing) pass through unchanged.

    When ``_in_interpret`` is set (i.e. we are inside ``interpret()``'s
    ``make_jaxpr`` call), all calls pass through without interception (L5 guard).
    """
    # L5: suppress interception during interpret()/make_jaxpr tracing.
    if getattr(_in_interpret, "value", False):
        return _underlying_scan()(
            f,
            init,
            xs=xs,
            length=length,
            reverse=reverse,
            unroll=unroll,
            _split_transpose=_split_transpose,
        )
    depth = getattr(_depth, "value", 0)
    _depth.value = depth + 1
    try:
        active = _active_contexts()
        if depth == 0 and active:
            ctx = _select_ctx(active)
            if ctx is None:
                return _underlying_scan()(
                    f,
                    init,
                    xs=xs,
                    length=length,
                    reverse=reverse,
                    unroll=unroll,
                    _split_transpose=_split_transpose,
                )
            k = ctx._get_next_idx()
            return ctx._intercept_scan(
                f,
                init,
                xs,
                length,
                reverse=reverse,
                unroll=unroll,
                _split_transpose=_split_transpose,
                _start_idx=k,
            )
        return _underlying_scan()(
            f,
            init,
            xs=xs,
            length=length,
            reverse=reverse,
            unroll=unroll,
            _split_transpose=_split_transpose,
        )
    finally:
        _depth.value = depth


def _patched_while(cond_fun: Callable, body_fun: Callable, init_val: Any) -> Any:
    """Drop-in replacement for ``jax.lax.while_loop`` while any _RecordContext is active.

    When ``_in_interpret`` is set, passes through without interception (L5 guard).
    """
    # L5: suppress interception during interpret()/make_jaxpr tracing.
    if getattr(_in_interpret, "value", False):
        return _underlying_while()(cond_fun, body_fun, init_val)
    depth = getattr(_depth, "value", 0)
    _depth.value = depth + 1
    try:
        active = _active_contexts()
        if depth == 0 and active:
            ctx = _select_ctx(active)
            if ctx is None:
                return _underlying_while()(cond_fun, body_fun, init_val)
            k = ctx._get_next_idx()
            return ctx._intercept_while(cond_fun, body_fun, init_val, _start_idx=k)
        return _underlying_while()(cond_fun, body_fun, init_val)
    finally:
        _depth.value = depth


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


class _RecordContext:
    """
    Context manager returned by ``tap.record(...)`` (no-callable form).

    Monkeypatches ``jax.lax.scan`` and ``jax.lax.while_loop`` on entry,
    collects TapEvents from any scan/while that runs inside the block, and
    restores the original primitives on exit — even if the body raised.

    See module docstring for thread-safety, re-entrant, foreign-patch, and
    jit-cache-boundary contracts.

    The host callback baked into compiled artifacts is always the module-level
    singleton ``_dynamic_router``, not a closure over ``self._recorder``.
    This means:
      - After exit: events are dropped, not appended to the dead recorder.
      - Cache-hit in a new context: events route to the new context's recorder.

    The with-form's transform contract (MAJOR-2)
    -------------------------------------------
    Events emitted by the context reflect the program AS TRACED at the call
    site — specifically:

    - ``grad`` around code inside the context fires forward-pass events only
      (a FEATURE: no fwd+bwd double-fire from the forward pass tape).
    - ``vmap`` around a scan inside the context produces per-lane events.
    - An enclosing ``jit``'s path prefix is NOT visible because the A-shell
      intercepts the un-jitted call at the outermost boundary.
    - Full equivalence with ``verbose()`` holds for un-transformed, non-jitted
      calls: the same paths, the same step sequence, the same carry values.

    Addressing contract (MAJOR-1)
    -----------------------------
    Top-level path indices are UNIQUE and MONOTONIC per context: the first
    outermost scan/while intercepted in the block is ``scan[0]``/``while[0]``,
    the second is ``scan[1]``/``while[1]``, etc.  The counter is per-context
    (each new ``tap.record()`` object starts at 0).

    CONTRACT: equality with ``verbose()`` numbering holds when the function
    passed to ``verbose()`` contains exactly the same top-level scan/while
    calls as the with-block body.  A top-level ``cond`` or ``remat`` between
    scans is INVISIBLE to the A-shell patch (it only intercepts scan/while),
    so ``verbose()`` counts it in ``n_cf`` while the A-shell does not.  This
    causes a numbering divergence for programs where non-scan/while boundaries
    appear at the outermost level.  Document; do not fix.
    """

    def __init__(
        self,
        *,
        select: "Callable | None" = None,
        ops: "tuple[str, ...]" = ("scan", "while_loop"),
        sample_every: int = 1,
        where: "Callable[[str], bool] | None" = None,
        max_depth: "int | None" = None,
        taps: "Sequence[Any]" = (),
        on_step: "Callable | None" = None,
        alert: "Callable | None" = None,
        alert_once: bool = False,
    ) -> None:
        self._select = select
        self._ops = ops
        self._sample_every = sample_every
        self._where = where
        self._max_depth = max_depth
        self._taps = taps
        self._extra_on_step = on_step  # optional live-stream callback
        self._alert = alert
        self._alert_once = alert_once
        self._recorder: "FlightRecorder | None" = None
        self._key: str | None = None
        self._owner_thread: int | None = None
        # Top-level interception counter: each outermost intercepted scan/while
        # consumes the next index (MAJOR-1 — unique monotonic addressing).
        self._next_toplevel_idx: int = 0
        self._idx_lock = threading.Lock()
        # GC self-heal finalizer (L2).  Set in __enter__, detached in __exit__.
        self._finalizer: "weakref.finalize | None" = None

    def _get_next_idx(self) -> int:
        """Atomically read-and-increment the top-level interception counter."""
        with self._idx_lock:
            k = self._next_toplevel_idx
            self._next_toplevel_idx += 1
        return k

    def __enter__(self) -> "FlightRecorder":
        global \
            _session_scan, \
            _session_while, \
            _clobber_scan_warned, \
            _clobber_while_warned

        # L1: re-entrancy guard — reusing the SAME context object in nested
        # with-blocks overwrites self._key, orphaning the first registry entry
        # and permanently corrupting global state.  Raise immediately.
        if self._key is not None:
            raise RuntimeError(
                "tap.record() context is not re-entrant: a context object cannot be "
                "entered while it is already active.  Create a new tap.record() for "
                "each with-block."
            )

        from .collectors import FlightRecorder as _FR

        self._recorder = _FR()
        self._key = str(uuid.uuid4())
        self._owner_thread = threading.get_ident()
        self._next_toplevel_idx = 0

        with _registry_lock:
            if not _context_registry:
                # empty -> non-empty: capture whatever is installed NOW.
                # Self-capture guard: if jax.lax.scan is already _patched_scan
                # (shouldn't happen when registry is empty, but be safe) skip.
                candidate_scan = jax.lax.scan
                if candidate_scan is not _patched_scan:
                    _session_scan = candidate_scan
                candidate_while = jax.lax.while_loop
                if candidate_while is not _patched_while:
                    _session_while = candidate_while
                # L6: reset session-scoped warn-once flags.  A new session
                # (registry empty → non-empty) gets fresh warn budget so a
                # genuine clobber is always reported.
                _clobber_scan_warned = False
                _clobber_while_warned = False
            _context_registry[self._key] = weakref.ref(self)
            jax.lax.scan = _patched_scan
            jax.lax.while_loop = _patched_while

        # L2: GC self-heal — attach a finalize that runs _cleanup(key) when
        # self is collected without __exit__ being called.  Because the
        # registry stores weakref.ref(self) (not self), the registry does NOT
        # keep self alive; a dropped context is GC-eligible.
        # detach() in __exit__ ensures the finalizer is a no-op for the normal
        # with-block flow.
        self._finalizer = weakref.finalize(self, _cleanup, self._key)

        return self._recorder

    def __exit__(self, *exc: Any) -> None:
        global \
            _session_scan, \
            _session_while, \
            _clobber_scan_warned, \
            _clobber_while_warned

        # L2: detach the GC finalizer — normal exit path handles cleanup below.
        if self._finalizer is not None:
            self._finalizer.detach()
            self._finalizer = None

        with _registry_lock:
            # L3/L4: track whether THIS call actually deregistered something.
            # Double-exit and exit-without-enter have self._key = None → pop
            # does nothing → did_deregister = False → restore/clobber block
            # is skipped entirely, so no bogus clobber warning is emitted and
            # the warn-once flag is not poisoned.
            did_deregister = False
            if self._key is not None:
                _context_registry.pop(self._key, None)
                self._key = None
                did_deregister = True

            if did_deregister and not _context_registry:
                # non-empty -> empty: restore.

                # scan restore
                if jax.lax.scan is _patched_scan:
                    jax.lax.scan = (
                        _session_scan if _session_scan is not None else _original_scan
                    )
                elif not _clobber_scan_warned:
                    # Foreign patch installed OVER us: leave it, warn once.
                    _clobber_scan_warned = True
                    try:
                        warnings.warn(
                            "jaxtap: jax.lax.scan was replaced by a foreign patch while "
                            "a tap.record() context was active. "
                            "jaxtap will not restore it to avoid clobbering the foreign patch.",
                            UserWarning,
                            stacklevel=2,
                        )
                    except Exception:
                        pass
                _session_scan = None

                # while_loop restore
                if jax.lax.while_loop is _patched_while:
                    jax.lax.while_loop = (
                        _session_while
                        if _session_while is not None
                        else _original_while
                    )
                elif not _clobber_while_warned:
                    _clobber_while_warned = True
                    try:
                        warnings.warn(
                            "jaxtap: jax.lax.while_loop was replaced by a foreign patch while "
                            "a tap.record() context was active. "
                            "jaxtap will not restore it to avoid clobbering the foreign patch.",
                            UserWarning,
                            stacklevel=2,
                        )
                    except Exception:
                        pass
                _session_while = None

    def _intercept_scan(
        self,
        body: Callable,
        init: Any,
        xs: Any,
        length: int | None,
        reverse: bool = False,
        unroll: "int | bool" = 1,
        _split_transpose: bool = False,
        _start_idx: int = 0,
    ) -> Any:
        """Apply the B-core verbose() transform to a scan call intercepted at depth 0.

        Builds a wrapper ``g`` that calls the underlying scan with the original
        body and kwargs, then runs ``verbose(g, on_step=_dynamic_router, ...)``
        on it.

        The ``on_step`` is always the module-level ``_dynamic_router`` singleton —
        NOT ``self._recorder``.  This ensures the baked XLA callback resolves
        the live active context at EXECUTION TIME, preventing phantom emission
        and enabling correct routing on cache-hits in new contexts.

        The depth counter (already incremented to 1 in ``_patched_scan``) prevents
        the B-core's own ``jax.lax.scan`` calls from being re-intercepted.

        ``_start_idx`` is the top-level counter value consumed for this call
        (from ``_get_next_idx()``).  It is forwarded to ``verbose`` as the
        private ``_start_cf_index`` kwarg so that the B-core walker addresses
        this scan as ``scan[_start_idx]`` rather than always ``scan[0]``.
        """
        from . import verbose as _verbose

        underlying = _underlying_scan()

        def g(init_: Any, xs_: Any) -> Any:
            return underlying(
                body,
                init_,
                xs_,
                length=length,
                reverse=reverse,
                unroll=unroll,
                _split_transpose=_split_transpose,
            )

        return _verbose(
            g,
            on_step=_dynamic_router,
            select=self._select,
            ops=self._ops,
            sample_every=self._sample_every,
            where=self._where,
            max_depth=self._max_depth,
            taps=list(self._taps),
            alert=self._alert,
            alert_once=self._alert_once,
            _start_cf_index=_start_idx,
        )(init, xs)

    def _intercept_while(
        self,
        cond_fun: Callable,
        body_fun: Callable,
        init_val: Any,
        _start_idx: int = 0,
    ) -> Any:
        """Apply the B-core verbose() transform to a while_loop call intercepted at depth 0.

        Uses ``_dynamic_router`` as ``on_step`` for the same reason as
        ``_intercept_scan`` — runtime context resolution, no phantom emission.

        ``_start_idx`` addresses this while_loop as ``while[_start_idx]``
        rather than always ``while[0]``, ensuring unique monotonic addressing
        across sequential top-level while_loop calls in the same context.
        """
        from . import verbose as _verbose

        underlying = _underlying_while()

        def g(init_: Any) -> Any:
            return underlying(cond_fun, body_fun, init_)

        return _verbose(
            g,
            on_step=_dynamic_router,
            select=self._select,
            ops=self._ops,
            sample_every=self._sample_every,
            where=self._where,
            max_depth=self._max_depth,
            taps=list(self._taps),
            alert=self._alert,
            alert_once=self._alert_once,
            _start_cf_index=_start_idx,
        )(init_val)
