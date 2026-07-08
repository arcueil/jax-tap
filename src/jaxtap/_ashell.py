"""
A-shell: the ``with tap.record():`` context manager form.

Monkeypatches ``jax.lax.scan`` (and ``jax.lax.while_loop``) for the duration
of the context, so that any call to those primitives inside the ``with`` block
is automatically instrumented — zero change to user code.

The patch machinery is ported from the hardened #964 patch in
``blackjax/blackjax/progress_bar.py`` (tqdm / display-thread / output_file
stripped; jaxtap is emission-only).

Thread safety
-------------
A single registry lock (``_registry_lock``) is held ONLY around the
empty→non-empty and non-empty→empty registry transitions (enter/exit critical
sections).  It is NEVER held across the trace hot path.

Attribution rules (mirroring the #964 decision table):
  - Exactly ONE active context: any calling thread is attributed to it.
    This is the "delegate" pattern — enter on main, run work on a worker
    thread, still gets taps.
  - TWO or more active contexts: attribution requires an exact match between
    the calling thread and the context's ``_owner_thread`` (the thread that
    called ``__enter__``).  A bystander thread passes through untapped rather
    than risk guessing wrong.

Re-entrant (nested) contexts
-----------------------------
Nested ``with tap.record()`` blocks are supported.  With ≥2 simultaneous
contexts on the same owner thread, the INNERMOST context (last-entered) wins
for taps fired during its scope.  This is the natural outcome of the
``owned[-1]`` selection in ``_select_ctx``.

Depth counter
-------------
A single thread-local ``_depth`` counter is incremented by EVERY patched
primitive (_patched_scan AND _patched_while).  This prevents double-
instrumentation across primitives: when ``_intercept_scan`` calls
``verbose(g)(init, xs)`` and the B-core rewrites re-enter ``jax.lax.scan``
(or ``jax.lax.while_loop``), they see depth > 0 and pass through unchanged.

Foreign patch chaining
----------------------
On the empty→non-empty transition, the current value of ``jax.lax.scan`` is
captured as ``_session_scan`` (analogously for while_loop) — unless it IS
already ``_patched_scan`` (self-capture guard).  On the non-empty→empty
transition, we restore ONLY if ``jax.lax.scan is _patched_scan``; if a
foreign patch was installed OVER us during our session, we leave it alone and
warn once.

jit-cache boundary (documented, not fixed)
-------------------------------------------
A function JIT-compiled BEFORE entering the context reuses the cached trace
on subsequent calls and emits nothing inside the context — no retrace means
no new callbacks are baked in.  Workaround: ``jax.clear_caches()`` before
entering the context.
"""

from __future__ import annotations

import threading
import uuid
import warnings
from typing import TYPE_CHECKING, Any, Callable, Sequence

import jax

if TYPE_CHECKING:
    from .collectors import FlightRecorder

__all__ = ["_RecordContext"]

# ---------------------------------------------------------------------------
# Captured import-time originals (never changed after module load)
# ---------------------------------------------------------------------------

_original_scan: Callable = jax.lax.scan
_original_while: Callable = jax.lax.while_loop

# ---------------------------------------------------------------------------
# Thread-local depth counter (shared across ALL patched primitives)
# ---------------------------------------------------------------------------

# Increment for EVERY entry into _patched_scan or _patched_while.
# This ensures that re-entries from _intercept_scan / _intercept_while
# (via jax.make_jaxpr tracing and the B-core rewrites) see depth > 0 and
# pass through, preventing double-instrumentation across both primitives.
_depth = threading.local()

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_context_registry: dict[str, "_RecordContext"] = {}

# Guards the empty<->non-empty transition of _context_registry.
# NEVER held across the trace hot path or the ``yield`` of a context.
_registry_lock = threading.Lock()

# Captured on empty->non-empty: whatever jax.lax.scan / while_loop pointed
# to BEFORE we first installed _patched_scan / _patched_while.  May be a
# foreign patch rather than the import-time original.  None means no session.
_session_scan: Callable | None = None
_session_while: Callable | None = None

# Warn-once flags: raised when a foreign patch is detected over us on exit.
_clobber_scan_warned: bool = False
_clobber_while_warned: bool = False


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


def _select_ctx(active: list["_RecordContext"]) -> "_RecordContext | None":
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
# Patched primitives
# ---------------------------------------------------------------------------


def _patched_scan(
    f: Callable, init: Any, xs: Any = None, length: int | None = None, **kwargs: Any
) -> Any:
    """Drop-in replacement for ``jax.lax.scan`` while any _RecordContext is active.

    Only the outermost call (depth == 0) is instrumented; nested calls
    (from the B-core rewrites or from user-code scans discovered during
    ``make_jaxpr`` tracing) pass through unchanged.
    """
    depth = getattr(_depth, "value", 0)
    _depth.value = depth + 1
    try:
        active = list(_context_registry.values())
        if depth == 0 and active:
            ctx = _select_ctx(active)
            if ctx is None:
                return _underlying_scan()(f, init, xs=xs, length=length, **kwargs)
            return ctx._intercept_scan(f, init, xs, length, **kwargs)
        return _underlying_scan()(f, init, xs=xs, length=length, **kwargs)
    finally:
        _depth.value = depth


def _patched_while(cond_fun: Callable, body_fun: Callable, init_val: Any) -> Any:
    """Drop-in replacement for ``jax.lax.while_loop`` while any _RecordContext is active."""
    depth = getattr(_depth, "value", 0)
    _depth.value = depth + 1
    try:
        active = list(_context_registry.values())
        if depth == 0 and active:
            ctx = _select_ctx(active)
            if ctx is None:
                return _underlying_while()(cond_fun, body_fun, init_val)
            return ctx._intercept_while(cond_fun, body_fun, init_val)
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
    jit-cache-boundary contract.
    """

    def __init__(
        self,
        *,
        select: "Callable | None" = None,
        ops: tuple[str, ...] = ("scan", "while_loop"),
        sample_every: int = 1,
        where: "Callable[[str], bool] | None" = None,
        max_depth: "int | None" = None,
        taps: "Sequence[Any]" = (),
    ) -> None:
        self._select = select
        self._ops = ops
        self._sample_every = sample_every
        self._where = where
        self._max_depth = max_depth
        self._taps = taps
        self._recorder: "FlightRecorder | None" = None
        self._key: str | None = None
        self._owner_thread: int | None = None

    def __enter__(self) -> "FlightRecorder":
        global _session_scan, _session_while

        from .collectors import FlightRecorder as _FR

        self._recorder = _FR()
        self._key = str(uuid.uuid4())
        self._owner_thread = threading.get_ident()

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
            _context_registry[self._key] = self
            jax.lax.scan = _patched_scan
            jax.lax.while_loop = _patched_while

        return self._recorder  # type: ignore[return-value]

    def __exit__(self, *exc: Any) -> None:
        global _session_scan, _session_while, _clobber_scan_warned, _clobber_while_warned

        with _registry_lock:
            if self._key is not None:
                _context_registry.pop(self._key, None)
                self._key = None

            if not _context_registry:
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
                        _session_while if _session_while is not None else _original_while
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
        **kwargs: Any,
    ) -> Any:
        """Apply the B-core verbose() transform to a scan call intercepted at depth 0.

        Builds a wrapper ``g`` that calls the underlying scan with the original
        body and kwargs, then runs ``verbose(g, ...)`` on it.  The depth counter
        (already incremented to 1 in ``_patched_scan``) prevents the B-core's
        own ``jax.lax.scan`` calls from being re-intercepted.
        """
        from . import verbose as _verbose

        underlying = _underlying_scan()

        def g(init_: Any, xs_: Any) -> Any:
            return underlying(body, init_, xs_, length=length, **kwargs)

        return _verbose(
            g,
            on_step=self._recorder,
            select=self._select,
            ops=self._ops,
            sample_every=self._sample_every,
            where=self._where,
            max_depth=self._max_depth,
            taps=list(self._taps),
        )(init, xs)

    def _intercept_while(
        self,
        cond_fun: Callable,
        body_fun: Callable,
        init_val: Any,
    ) -> Any:
        """Apply the B-core verbose() transform to a while_loop call intercepted at depth 0."""
        from . import verbose as _verbose

        underlying = _underlying_while()

        def g(init_: Any) -> Any:
            return underlying(cond_fun, body_fun, init_)

        return _verbose(
            g,
            on_step=self._recorder,
            select=self._select,
            ops=self._ops,
            sample_every=self._sample_every,
            where=self._where,
            max_depth=self._max_depth,
            taps=list(self._taps),
        )(init_val)
