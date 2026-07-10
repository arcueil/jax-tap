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
jaxtap — zero-code-change runtime telemetry for JAX control flow.

Usage::

    import jaxtap as tap

    events = []

    def on_step(event: tap.TapEvent) -> None:
        events.append(event)

    tapped_f = tap.verbose(f, on_step=on_step)
    result = tapped_f(*args)   # bitwise-identical to f(*args)

The ``select`` parameter runs inside the traced program (on-device) to reduce
what crosses the host boundary::

    tap.verbose(f, on_step=cb, select=lambda carry: carry[0].mean())

Path-aware select (M1d): if ``select`` accepts a ``path`` kwarg or 2nd positional
parameter, jaxtap passes the stable node address at call time::

    tap.verbose(f, on_step=cb, select=lambda carry, *, path: {"node": path, "v": carry[0]})

Primitive taps observe named JAX primitives by kind, with zero modification
to the user's code::

    tapped_f = tap.verbose(
        f,
        on_step=cb,
        taps=[tap.on("cholesky", select=lambda outs: jnp.all(jnp.isfinite(outs[0])))],
    )
    # cb receives TapEvent(path="scan[0]/jit[0]/cholesky[0]", step=<scan step>, value=<bool>)

Ergonomic collector helper::

    g, rec = tap.record(f)
    g(*args)
    rec.df()
"""

from __future__ import annotations

import dataclasses

try:
    from ._version import __version__
except ImportError:  # package not installed (editable install without build step)
    __version__ = "unknown"
import inspect
import sys
import warnings
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from ._ashell import _original_scan as original_scan
from ._walker import interpret

if TYPE_CHECKING:
    from ._ashell import _RecordContext
    from .collectors import FlightRecorder

__all__ = [
    "__version__",
    "TapEvent",
    "PrimitiveTap",
    "on",
    "watch_nan",
    "print",
    "primitives",
    "verbose",
    "record",
    "emergency_restore",
    "original_scan",
    "FlightRecorder",
    "JSONLWriter",
    "read_jsonl",
]


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TapEvent:
    """A single telemetry emission from a tapped control-flow operation.

    TapEvent objects are delivered to the host via the ``on_step`` callback
    and represent one snapshot from inside a scan, while_loop, or primitive
    tap. All fields are immutable.

    Parameters
    ----------
    path : str
        Stable hierarchical address of the tapped node, e.g.
        ``"scan[0]/jit[0]/cholesky[0]"``. For carry taps, this is the
        control-flow node. For primitive taps, this includes the primitive
        name and occurrence index.
    step : int
        Enclosing loop iteration index (0-based). For primitive taps that
        fire outside any loop, this is ``-1`` (sentinel).
    value : Any
        The selected value delivered to the host (after applying the ``select``
        function, if given). When ``select`` is ``None``, this is a flat tuple
        of the carry's leaf elements (pytree structure is erased by tracing).
    total : int or None, optional
        The enclosing loop's total iteration count when known. For ``scan``,
        this is the loop length ``N``. For ``while_loop``, outside any loop,
        or for primitive taps outside any loop, this is ``None``. Default:
        ``None``.
    """

    path: str  # stable address, e.g. "scan[0]/while[0]"
    step: int  # iteration index (0-based); -1 for primitive taps outside any loop
    value: Any  # selected carry payload delivered to the host
    total: "int | None" = (
        None  # enclosing loop length when known (scan: N; while/outside: None)
    )


@dataclasses.dataclass(frozen=True)
class PrimitiveTap:
    """Spec for tapping a specific JAX primitive by name.

    Created via :func:`on`::

        tap.on("cholesky")
        tap.on("cholesky", select=lambda outs: jnp.all(jnp.isfinite(outs[0])))
        tap.on("cholesky",
               select=lambda outs: jnp.all(jnp.isfinite(outs[0])),
               alert=lambda ok: not ok,
               label="NaN/Inf")

    Parameters
    ----------
    prim_name:
        The JAX primitive name to match (e.g. ``"cholesky"``, ``"dot_general"``).
        This is the name that appears in ``jax.make_jaxpr`` output.
    select:
        Optional traced-side callable applied to the primitive's **output tuple**
        on-device before the host callback.  Receives a tuple of the primitive's
        output arrays; only the selector's return value crosses the host boundary.
        Default: ``None`` — the full output tuple is delivered as ``TapEvent.value``.

        **Path-aware form** (M1d FIX 3): if the callable accepts a ``path``
        parameter (keyword or 2nd positional), jaxtap calls
        ``select(eff_outs, path=path)`` where ``path`` is the primitive tap's
        stable address (e.g. ``"scan[0]/jit[0]/cholesky[0]"``).  Inspected once
        at ``verbose()`` time; no per-step overhead.  Single-argument selectors
        unchanged.
    alert:
        Optional HOST-side predicate called with the (host-side) ``TapEvent.value``.
        When it returns truthy, jaxtap emits one terse line to stderr::

            [tap] FAIL {path} {step}/{total_or_'?'}: {label}

        The predicate runs inside the ``_guard`` discipline (never propagates).
        Alert firing is independent of ``on_step`` / the recorder — both still
        receive every event.
    label:
        Short string used in the alert line.  Default: the primitive name.
    output:
        Select a single primitive output by index before passing to ``select``
        (or before delivering the value when ``select`` is ``None``).
        ``None`` (default) passes the full output tuple.  Out-of-range indices
        raise ``IndexError`` at trace time.

        .. warning:: **Primitive output order ≠ Python API order.**
            The ``output=k`` index refers to the JAX *primitive's* output list,
            not the Python-level return tuple of the high-level function.  These
            can differ.  For example, ``jnp.linalg.eigh`` returns
            ``(eigenvalues, eigenvectors)`` in Python, but the underlying
            ``eigh`` primitive emits ``(eigenvectors, eigenvalues)`` — so
            ``output=0`` gives eigenvectors, not eigenvalues.  Use
            ``tap.print(prim_name)`` (no ``output=``) first to inspect the
            actual output layout before relying on a specific index.
    once:
        When ``True``, the alert / print fires at most once per :func:`verbose`
        call (B-form) or per trace (A-form).  Subsequent truthy events from the
        same spec instance are silently dropped.  Default: ``False`` (every
        truthy event fires).  Useful to suppress repetitive alert lines when only
        the *first* occurrence matters.

    Scope
    -----
    The walker ALWAYS descends into scan/while bodies, so primitive taps fire
    inside loops regardless of ``ops``/``where``/``max_depth`` filtering.
    Those filters control only whether the loop node EMITS carry taps — they do
    not suppress prim-tap coverage inside the body.
    AD-opaque primitives (``custom_jvp_call``/``custom_vjp_call`` interiors)
    are not descended (v1 policy unchanged).

    ``sample_every`` gates primitive taps inside loops with the same device-side
    ``lax.cond(step % se == 0, fire, noop)`` pattern as carry taps.  Primitive
    taps that fire OUTSIDE any loop (``TapEvent.step == -1``) are always ungated
    regardless of ``sample_every``.  ``once=`` and ``alert`` operate on the
    events that survive the gate.  For se≥10 the callback cost amortises to
    ~1 µs/step (see benchmark); se=10 is the recommended monitoring baseline.

    Step context
    ------------
    ``TapEvent.step`` is the enclosing loop's live step value.  When a primitive
    tap fires outside any scan/while loop, ``step == -1`` (the sentinel).
    ``TapEvent.total`` is the enclosing scan's length, or ``None`` for while loops
    and for primitive taps outside any loop.
    """

    prim_name: str
    select: "Callable | None" = None
    alert: "Callable | None" = None
    label: "str | None" = None
    output: "int | None" = None
    once: bool = False
    # _printer: set by tap.print() — uses value format instead of FAIL label format.
    _printer: bool = dataclasses.field(default=False, repr=False)


def on(
    prim_name: str,
    select: "Callable | None" = None,
    alert: "Callable | None" = None,
    label: "str | None" = None,
    output: "int | None" = None,
    once: bool = False,
) -> PrimitiveTap:
    """Create a :class:`PrimitiveTap` spec.

    Parameters
    ----------
    prim_name:
        Name of the JAX primitive to tap.
    select:
        Optional on-device reducer applied to the primitive's output tuple
        (or to the single output when ``output=k`` is given).
    alert:
        Optional HOST-side predicate on the event value; when truthy, emits one
        terse line to stderr: ``[tap] FAIL {path} {step}/{total}: {label}``.
        Runs inside the ``_guard`` discipline (never propagates).
    label:
        Short label for the alert line.  Default: ``prim_name``.
    output:
        Select a single primitive output by index before calling ``select``
        (or before delivering the value when ``select`` is ``None``).
        ``None`` (default) passes the full output tuple.  Out-of-range indices
        raise ``IndexError`` at trace time.

        Note: indices refer to the JAX *primitive's* output order, which can
        differ from the Python API's return order.  Use ``tap.print(prim_name)``
        first to inspect the actual layout before relying on a specific index.
    once:
        When ``True``, the alert fires at most once per :func:`verbose` call.
        Default: ``False``.

    Returns
    -------
    PrimitiveTap

    Examples
    --------
    Create a basic primitive tap for cholesky::

        spec = tap.on("cholesky")

    Tap cholesky with a finiteness check (select returns bool)::

        spec = tap.on("cholesky",
                      select=lambda outs: jnp.all(jnp.isfinite(outs[0])),
                      alert=lambda ok: not ok,
                      label="NaN/Inf")

    Use a predicate label that is returned as a string::

        spec = tap.on("cholesky",
                      alert=lambda ok: "matrix singular" if not ok else False)
    """
    return PrimitiveTap(
        prim_name=prim_name,
        select=select,
        alert=alert,
        label=label,
        output=output,
        once=once,
    )


# ---------------------------------------------------------------------------
# Path-aware select inspection (FIX 3)
# ---------------------------------------------------------------------------


def _accepts_path(fn: "Callable") -> bool:
    """Return True if ``fn`` accepts ``path`` as a keyword argument or 2nd positional.

    Inspected once at trace time (Python level) when ``verbose()`` is called so
    the branch is resolved statically — no per-step overhead.

    Accepts:
    - ``select(leaves, *, path)``  — keyword-only 'path'
    - ``select(leaves, path)``     — 2nd positional-or-keyword parameter named 'path'
    - ``select(leaves, path_str)`` — any name works if it is the 2nd positional

    Does NOT accept ``path`` through **kwargs to avoid false positives.
    Returns False on any introspection failure (e.g. built-in callables).
    """
    try:
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        # Named 'path' anywhere in the signature (kwarg or positional)
        if "path" in sig.parameters:
            param = sig.parameters["path"]
            # Exclude **kwargs — that would match everything
            if param.kind != inspect.Parameter.VAR_KEYWORD:
                return True
        # 2nd positional parameter (any name) — caller passes path=
        positional = [
            p
            for p in params
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        if len(positional) >= 2:
            return True
        return False
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Warn-once guards
# ---------------------------------------------------------------------------

_warned: set[int] = set()  # tracks id(on_step) that have warned once this session
_alert_warned: set[int] = set()  # tracks id(alert_fn) that have warned once
# NOTE: id() values can be reused after GC; warn-once dedup is best-effort for M0.
# Shared by both _fire_alert (primitive taps) and _fire_carry_alert (carry taps).


def _guard(on_step: Callable[[TapEvent], None], event: TapEvent) -> None:
    """Call ``on_step(event)`` with a never-raise guarantee."""
    cb_id = id(on_step)
    try:
        on_step(event)
    except Exception as exc:  # noqa: BLE001
        if cb_id not in _warned:
            _warned.add(cb_id)
            msg = (
                f"jaxtap: on_step callback raised {type(exc).__name__}: {exc!s}. "
                "Telemetry suppressed for this callback to preserve program behaviour."
            )
            try:
                warnings.warn(msg, UserWarning, stacklevel=1)
            except Exception:  # noqa: BLE001  -- handles python -W error
                pass


def _format_value(value: Any) -> str:
    """Compact single-line repr of a host-side value for :func:`tap.print` output.

    Uses numpy with ``printoptions(precision=4, threshold=8, edgeitems=2)`` so
    large arrays truncate rather than flooding stderr.  Internal newlines are
    collapsed to spaces so the result fits on one physical line.
    """
    import numpy as _np  # lazy — only imported when a tap.print actually fires

    try:
        arr = _np.asarray(value)
        with _np.printoptions(precision=4, threshold=8, edgeitems=2):
            raw = repr(arr)
        return " ".join(raw.split())  # collapse internal newlines / extra spaces
    except Exception:  # noqa: BLE001
        try:
            return " ".join(repr(value).split())
        except Exception:  # noqa: BLE001
            return "<unprintable>"


def _fire_alert(
    spec: PrimitiveTap, event: TapEvent, _once_fired: "set[int] | None" = None
) -> None:
    """Fire ``spec.alert(event.value)`` and write the terse line to stderr if truthy.

    Runs inside the ``_guard`` discipline: a raising alert predicate is caught,
    warned once, and suppressed — it never propagates to the user.

    When ``spec._printer`` is True (set by :func:`print`), the line uses the
    value format ``[tap] {path} {step}/{total}: {value}`` instead of the FAIL
    format ``[tap] FAIL {path} {step}/{total}: {label}``.

    ``_once_fired`` is a per-:func:`verbose`-call set of spec ``id``\\ s that have
    already emitted at least one line.  When ``spec.once`` is True and the spec's
    id is already in the set, the call is a no-op (the once budget is spent).
    """
    alert_fn = spec.alert
    assert alert_fn is not None  # caller must check
    alert_id = id(alert_fn)
    try:
        should_alert = alert_fn(event.value)
    except Exception as exc:  # noqa: BLE001
        if alert_id not in _alert_warned:
            _alert_warned.add(alert_id)
            msg = (
                f"jaxtap: alert predicate raised {type(exc).__name__}: {exc!s}. "
                "Alert suppressed for this predicate to preserve program behaviour."
            )
            try:
                warnings.warn(msg, UserWarning, stacklevel=1)
            except Exception:  # noqa: BLE001
                pass
        return
    if should_alert:
        # once=True: fire at most once per verbose() call / per trace.
        spec_id = id(spec)
        if spec.once and _once_fired is not None:
            if spec_id in _once_fired:
                return
            _once_fired.add(spec_id)
        total_str = str(event.total) if event.total is not None else "?"
        try:
            if spec._printer:
                # tap.print format: [tap] {path} {step}/{total}: {value_repr}
                value_repr = _format_value(event.value)
                sys.stderr.write(
                    f"[tap] {event.path} {event.step}/{total_str}: {value_repr}\n"
                )
            else:
                # alert format: [tap] FAIL {path} {step}/{total}: {label}
                label = spec.label if spec.label is not None else spec.prim_name
                sys.stderr.write(
                    f"[tap] FAIL {event.path} {event.step}/{total_str}: {label}\n"
                )
        except Exception:  # noqa: BLE001
            pass


def _fire_carry_alert(
    alert_fn: "Callable[[TapEvent], Any]",
    event: TapEvent,
    alert_once: bool,
    _carry_once_fired: "set[str]",
) -> None:
    """Fire a carry-tap alert callable on a TapEvent; write terse FAIL line if truthy.

    Extends the ``_fire_alert`` machinery to carry taps:

    - ``alert_fn`` receives the full :class:`TapEvent` (not just ``event.value``).
    - ``msg`` in the output line is ``str(result)`` when ``result`` is a ``str``,
      otherwise the fixed label ``"alert"``.
    - ``_carry_once_fired`` is a per-:func:`verbose`-call ``set[str]`` tracking
      which paths have already fired; when ``alert_once`` is True and the path
      is already in the set, the call is a no-op.
    - Runs inside the ``_guard`` discipline: a raising callable warns once via
      ``_alert_warned`` (shared with primitive-tap alerts) and is suppressed.
    """
    alert_id = id(alert_fn)
    try:
        result = alert_fn(event)
    except Exception as exc:  # noqa: BLE001
        if alert_id not in _alert_warned:
            _alert_warned.add(alert_id)
            msg = (
                f"jaxtap: carry alert callable raised {type(exc).__name__}: {exc!s}. "
                "Alert suppressed for this callable to preserve program behaviour."
            )
            try:
                warnings.warn(msg, UserWarning, stacklevel=1)
            except Exception:  # noqa: BLE001
                pass
        return
    if result:
        if alert_once:
            if event.path in _carry_once_fired:
                return
            _carry_once_fired.add(event.path)
        total_str = str(event.total) if event.total is not None else "?"
        line_msg = result if isinstance(result, str) else "alert"
        try:
            sys.stderr.write(
                f"[tap] FAIL {event.path} {event.step}/{total_str}: {line_msg}\n"
            )
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Mapping from public-API op names to internal jaxpr primitive names.
_OP_NAME_MAP: dict[str, str] = {
    "scan": "scan",
    "while_loop": "while",
    "while": "while",  # allow the short form too
}


def verbose(
    f: Callable,
    *,
    on_step: "Callable[[TapEvent], None] | None" = None,
    select: Callable | None = None,
    ops: tuple[str, ...] = ("scan", "while_loop"),
    sample_every: int = 1,
    where: "Callable[[str], bool] | None" = None,
    max_depth: "int | None" = None,
    taps: "Sequence[PrimitiveTap]" = (),
    alert: "Callable[[TapEvent], Any] | None" = None,
    alert_once: bool = False,
    _start_cf_index: int = 0,
) -> Callable:
    """
    Return a function bitwise-identical to ``f`` that emits telemetry from
    inside ``lax.scan`` / ``lax.while_loop`` without any modification to ``f``.

    Parameters
    ----------
    f:
        The function to instrument.
    on_step:
        Optional host callback called with a :class:`TapEvent` after each
        control-flow iteration or matched primitive.  If provided, it must
        never raise (failures are caught, warned once, and suppressed to
        preserve program behavior).  When ``None``, a bare verbose trace
        is created with zero runtime cost (no debug_callback baked into
        jitted code).  Default: ``None``.  Both ``on_step`` and ``alert``
        may be ``None`` (bare trace, no callbacks).
    alert:
        Optional HOST-side callable evaluated on each carry-tap :class:`TapEvent`
        (the same event that ``on_step`` would receive).  When truthy, emits one
        terse line to stderr::

            [tap] FAIL {path} {step}/{total}: {msg}

        where ``msg`` is the returned value when it is a ``str``, otherwise the
        fixed label ``"alert"``.  Runs inside the ``_guard`` discipline (never
        propagates).  Fires BEFORE ``on_step``; both run; their results are
        independent.  Alert without ``on_step`` is legal (alert-only tap).
        Default: ``None``.
    alert_once:
        When ``True``, the alert fires at most once per path per :func:`verbose`
        call.  Subsequent truthy events from the same path are silently dropped.
        Mirrors the ``once=`` semantics of :func:`on`.  Default: ``False``.
    select:
        Optional traced-side callable applied to the carry **on-device** before
        the host boundary crossing. Receives a **flat tuple of carry leaves**
        (pytree structure is erased by JAX tracing — jaxprs are flat).
        Only the selector's return value crosses to the host; its pytree
        structure is captured at trace time and reconstructed on the host.
        Default: ``None`` — the full flat carry tuple is delivered as
        ``TapEvent.value``.

        **CRITICAL**: the ``select`` function receives **flat leaves**, not the
        original carry pytree. To use the carry's structure, reshape inside
        ``select`` or use a path-aware form (see below).

        Example: if carry is ``{"a": x, "b": [y, z]}``, ``select`` receives
        a flat tuple ``(x, y, z)`` and must index by position:
        ``select=lambda leaves: leaves[0] + leaves[1]`` (selects ``x + y``).

        **Path-aware form**: if the callable accepts a ``path`` parameter
        (keyword or 2nd positional), jaxtap calls
        ``select(carry_leaves, path=path)`` where ``path`` is the stable
        node-address string (e.g. ``"scan[0]"``).  Inspected once at
        ``verbose()`` call time — no per-step overhead.  Back-compat:
        single-argument selectors unchanged.
    ops:
        Which control-flow operators to tap.  Accepted values: ``"scan"``,
        ``"while_loop"``.  Default: both.
    sample_every:
        Fire taps only on steps 0, k, 2k, … (device-side gate via
        ``lax.cond``).  Default: 1 (every step).  Must be ≥ 1.
        Gates BOTH carry taps and primitive taps that fire inside a loop.
        Primitive taps outside any loop (step == -1) are always ungated.
        At se=10 the callback cost amortises to ~1 µs/step — recommended
        for semi-production monitoring (see bench/README.md).
    where:
        Optional path predicate: only CF nodes whose path satisfies
        ``where(path)`` EMIT carry taps.  The address counter advances for
        filtered-out nodes so addressing remains stable.  The walker still
        DESCENDS into filtered bodies, so primitive taps and nested carry taps
        inside them fire normally.  Default: all nodes emit.
    max_depth:
        Optional depth limit.  CF nodes at depth > ``max_depth`` (depth =
        number of ``/`` segments in the path) do not emit carry taps.
        Deeper primitive taps and nested loops still fire.  Default: no limit.
    taps:
        Sequence of :class:`PrimitiveTap` specs created via :func:`on`.  After
        the walker binds any non-boundary primitive whose name matches a spec,
        the spec's ``select`` is applied on-device and ``on_step`` is called
        with a :class:`TapEvent`.  Path format:
        ``{enclosing_path}{prim_name}[{j}]`` where ``j`` counts occurrences of
        the primitive at this level.  ``TapEvent.step`` is the enclosing loop's
        live step, or ``-1`` when firing outside any loop.

    Returns
    -------
    Callable
        A function with the same signature as ``f``, returning bitwise-identical
        results, that emits a :class:`TapEvent` via ``on_step`` for each
        control-flow iteration and/or matched primitive.

    Examples
    --------
    Record all carry-tap events::

        events = []
        tapped = tap.verbose(f, on_step=lambda e: events.append(e))
        result = tapped(*args)
        print(f"Recorded {len(events)} events")

    Alert on every event (prints to stderr)::

        tapped = tap.verbose(f, alert=lambda e: True)
        result = tapped(*args)
        # stderr output:
        # [tap] FAIL scan[0] 0/5: alert
        # [tap] FAIL scan[0] 1/5: alert
        # ...

    Alert with custom message when step > 0::

        tapped = tap.verbose(
            f,
            alert=lambda e: "step > 0" if e.step > 0 else False
        )
        # stderr (on step 1):
        # [tap] FAIL scan[0] 1/5: step > 0

    Alert once per path (suppresses repeated lines)::

        tapped = tap.verbose(f, alert=lambda e: True, alert_once=True)
        # Only the first event per path path prints to stderr

    Bare verbose trace with no callback (alert-only)::

        tapped = tap.verbose(f, alert=lambda e: e.step == 0)
        result = tapped(*args)  # no on_step callback, only alert fires

    Notes
    -----
    **vmap × while_loop carry taps**: when ``f`` contains a ``while_loop``
    and is wrapped in ``jax.vmap``, carry taps deliver exactly one event per
    real per-lane iteration (not one per joint iteration).  jaxtap re-evaluates
    the while cond inside the body to compute a per-lane active mask, then
    sign-encodes it into the step argument: real iterations ship ``step`` as-is
    (non-negative); ghost lanes from finished vmap lanes ship ``-(step+1)``
    (always negative).  The host closure checks the sign and drops ghosts before
    constructing ``TapEvent`` — so ``on_step`` and ``alert`` only see events
    from lanes where the cond was still True.

    Note: ``lax.cond(active, tap, noop)`` was considered but does **not** work
    under ``vmap`` — a per-lane predicate causes ``lax.cond`` to evaluate both
    branches for all lanes; ``debug.callback`` fires unconditionally regardless
    of the predicate (fundamental JAX vmap+effects semantics).  The sign-encode
    host-drop is the correct mechanism.

    **Residual prim-tap ghost-firing**: primitive taps (``taps=[tap.on(...)]``)
    inside a vmapped while body still fire for all joint iterations, including
    ghost ones from finished lanes.  Carry-tap ghost suppression is exact;
    prim-tap suppression is a future arc.

    **While-loop overhead**: the A1 mitigation adds one extra cond evaluation
    per body iteration (XLA-compiled, not a Python callback) plus a sign-bit
    decode in the host closure (~+4 µs/iter vs. no-A1 baseline, measured at
    N=2000 K=25 in ``bench/a1_decompose.py``).  For trivial conds (counter
    < N) the cond re-eval is near-zero.  For expensive convergence-check
    conds (e.g. ``norm(carry) > tol``) the extra eval doubles the cond XLA
    work but the host-callback overhead is unchanged
    (see ``bench/while_cond_overhead.py``).

    **Carry boundary & pytree erasure**: Jaxprs are flat — tracing erases the
    carry's pytree structure and it is NOT recoverable from the scan equation.
    When ``select`` is ``None``, ``TapEvent.value`` is a *flat tuple of carry
    leaves* (not the original pytree). Use ``select`` to reshape them::

        # If carry is {"a": x, "b": y} (2 leaves), select receives (x, y)
        tap.verbose(f, on_step=cb, select=lambda leaves: {"a": leaves[0], "b": leaves[1]})

    The ``select`` function receives the flat leaf tuple and returns any
    pytree, which is captured at trace time and reconstructed on the host
    side with its structure intact.
    """
    if sample_every < 1:
        raise ValueError(f"sample_every must be >= 1, got {sample_every}")

    # Per-verbose()-call set tracking which paths have already fired a carry alert
    # with alert_once=True.  Mutable so all _host closures for this verbose() call
    # share the same once-budget across every loop site.
    _carry_once_fired: set[str] = set()

    internal_ops: frozenset[str] = frozenset(
        _OP_NAME_MAP[op] for op in ops if op in _OP_NAME_MAP
    )

    if on_step is None and alert is None:
        # No carry-tap observers at all.  Skip jax.debug.callback entirely so no
        # no-op callback (~33 µs/event on CPU) is traced into the compiled artifact.
        # Primitive-tap callbacks (prim_tap_fn below) are unaffected.
        # _while_active kwarg accepted (and ignored) so rewrite_while can always
        # call tap_cb(..., _while_active=active) unconditionally.
        def tap_cb(
            path: str,
            step: Any,
            *carry_leaves: Any,
            total: "int | None" = None,
            _while_active: Any = None,
        ) -> None:
            pass

    else:
        if select is not None:
            # FIX 3: inspect select once at verbose() call time.
            # If it accepts 'path' (kwarg or 2nd positional), call select(leaves, path=path);
            # otherwise call select(leaves) for backward compatibility.
            _select_wants_path: bool = _accepts_path(select)

            # tap_cb is called INSIDE the traced computation (scan/while body),
            # so ``select`` runs on-device before the host-boundary crossing.
            # ``total`` is a Python int (or None) passed as a kwarg from the rewrites;
            # it is captured in ``_host`` via closure — not a JAX argument.
            # tap_cb is called INSIDE the traced computation (scan/while body),
            # so ``select`` runs on-device before the host-boundary crossing.
            # ``total`` is a Python int (or None) passed as a kwarg from the rewrites;
            # it is captured in ``_host`` via closure — not a JAX argument.
            # ``_while_active`` is a JAX bool (or None) from rewrite_while.  When not
            # None, we encode active into the step arg using the sign bit: real steps
            # ship as-is (non-negative); ghost lanes ship -(step+1) (always negative).
            # The host decodes: raw<0 means ghost → drop silently.  This avoids the
            # ~16-19 µs/iter overhead of shipping one extra scalar operand through
            # debug.callback (measured in bench/a1_decompose.py arm (a), N=2000 K=25).
            def _base_tap_cb(
                path: str,
                step: Any,
                *carry_leaves: Any,
                total: "int | None" = None,
                _while_active: Any = None,
            ) -> None:
                # _select_wants_path is a Python bool: this branch resolves at trace time.
                selected = (
                    select(carry_leaves, path=path)
                    if _select_wants_path
                    else select(carry_leaves)
                )
                flat_selected = jax.tree_util.tree_leaves(selected)
                # Capture pytree structure at Python (trace) time for host-side recon.
                sel_tree = jax.tree_util.tree_structure(selected)

                if _while_active is not None:
                    # A1 mitigation: encode active into the step sign bit so no extra
                    # callback arg is needed.  ghost lane → step becomes -(step+1).
                    # INT32_MAX bound: -(step+1) overflows at step == 2^31-1; unreachable.
                    active_step = jax.lax.select(
                        _while_active, step, -(step + jnp.int32(1))
                    )

                    def _host(step_: Any, *flat_vals: Any) -> None:
                        raw = step_.item()
                        if raw < 0:
                            return  # ghost vmap+while lane — drop silently
                        value = jax.tree_util.tree_unflatten(sel_tree, list(flat_vals))
                        event = TapEvent(path=path, step=raw, value=value, total=total)
                        # alert fires BEFORE on_step; both are independent.
                        if alert is not None:
                            _fire_carry_alert(
                                alert, event, alert_once, _carry_once_fired
                            )
                        if on_step is not None:
                            _guard(on_step, event)

                    jax.debug.callback(
                        _host, active_step, *flat_selected, ordered=False
                    )
                else:

                    def _host(step_: Any, *flat_vals: Any) -> None:
                        value = jax.tree_util.tree_unflatten(sel_tree, list(flat_vals))
                        event = TapEvent(
                            path=path, step=step_.item(), value=value, total=total
                        )
                        # alert fires BEFORE on_step; both are independent.
                        if alert is not None:
                            _fire_carry_alert(
                                alert, event, alert_once, _carry_once_fired
                            )
                        if on_step is not None:
                            _guard(on_step, event)

                    jax.debug.callback(_host, step, *flat_selected, ordered=False)

        else:
            # tap_cb is called INSIDE the traced computation; ``jax.debug.callback``
            # ships the carry leaves to the host.  ``path`` is a static Python string
            # captured in the closure.
            # ``_while_active`` is a JAX bool (or None) from rewrite_while.  When not
            # None, active is encoded into the step sign bit (see select= variant above).
            def _base_tap_cb(
                path: str,
                step: Any,
                *carry_leaves: Any,
                total: "int | None" = None,
                _while_active: Any = None,
            ) -> None:
                if _while_active is not None:
                    # A1 mitigation: sign-encode active into step; no extra callback arg.
                    # INT32_MAX bound: -(step+1) overflows at step == 2^31-1; unreachable.
                    active_step = jax.lax.select(
                        _while_active, step, -(step + jnp.int32(1))
                    )

                    def _host(step_: Any, *leaves: Any) -> None:
                        raw = step_.item()
                        if raw < 0:
                            return  # ghost vmap+while lane — drop silently
                        event = TapEvent(path=path, step=raw, value=leaves, total=total)
                        # alert fires BEFORE on_step; both are independent.
                        if alert is not None:
                            _fire_carry_alert(
                                alert, event, alert_once, _carry_once_fired
                            )
                        if on_step is not None:
                            _guard(on_step, event)

                    jax.debug.callback(_host, active_step, *carry_leaves, ordered=False)
                else:

                    def _host(step_: Any, *leaves: Any) -> None:
                        event = TapEvent(
                            path=path, step=step_.item(), value=leaves, total=total
                        )
                        # alert fires BEFORE on_step; both are independent.
                        if alert is not None:
                            _fire_carry_alert(
                                alert, event, alert_once, _carry_once_fired
                            )
                        if on_step is not None:
                            _guard(on_step, event)

                    jax.debug.callback(_host, step, *carry_leaves, ordered=False)

        # Wrap with sample_every gate (device-side lax.cond) when k > 1.
        # Both branches return None (empty pytree) so lax.cond type-checks;
        # JAX's effects system ensures the debug callback fires only in the true branch.
        # ``total`` is forwarded as a Python kwarg (not a JAX argument — it is a
        # static int captured per-scan-call at trace time).
        # ``_while_active`` is threaded through so ghost-drop still applies when
        # sample_every gating is also active.
        if sample_every > 1:
            _uncapped = _base_tap_cb

            def tap_cb(
                path: str,
                step: Any,
                *carry_leaves: Any,
                total: "int | None" = None,
                _while_active: Any = None,
            ) -> None:
                jax.lax.cond(
                    step % sample_every == 0,
                    lambda _: _uncapped(
                        path,
                        step,
                        *carry_leaves,
                        total=total,
                        _while_active=_while_active,
                    ),
                    lambda _: None,
                    step,
                )

        else:
            tap_cb = _base_tap_cb

    # Build the primitive-tap callback if any specs were provided.
    # This function is called from _interp after binding a matched primitive.
    # It fires jax.debug.callback → _guard-wrapped on_step with a TapEvent,
    # then fires the spec's alert predicate (if any) on the host-side value.
    # path: Python string (static, determined at trace time)
    # step: JAX int32 (live, from enclosing loop) or jnp.int32(-1) (outside loop)
    # outvals: tuple of the primitive's output arrays (device-side)
    # spec: the matching PrimitiveTap
    # total: Python int or None — enclosing scan length (None for while / outside loop)
    #
    # _once_fired tracks which specs have already fired once this verbose() call;
    # created fresh here so each verbose() invocation has its own independent set.
    # Scoped here (not module-global) so the once budget resets on each verbose() call.
    _once_fired: set[int] = set()
    # FIX 3: precompute per-spec path-awareness flag once at verbose() call time.
    _spec_path_flags: dict[int, bool] = {}
    for _spec_item in taps:
        if _spec_item.select is not None:
            _spec_path_flags[id(_spec_item)] = _accepts_path(_spec_item.select)
    prim_tap_fn: Callable | None = None
    if taps:

        def prim_tap_fn(
            path: str,
            step: Any,
            outvals: tuple,
            spec: PrimitiveTap,
            total: "int | None" = None,
            _in_loop: bool = False,
        ) -> None:
            # Apply output index selection first (trace-time bounds check).
            if spec.output is not None:
                k = spec.output
                n = len(outvals)
                if k < 0 or k >= n:
                    raise IndexError(
                        f"jaxtap: tap.on({spec.prim_name!r}, output={k}) — primitive has"
                        f" {n} output(s) (valid indices: 0..{n - 1})"
                    )
                eff_outs: Any = outvals[k]
            else:
                eff_outs = outvals
            if spec.select is not None:
                # FIX 3: pass path= if the per-tap select accepts it.
                _pspec_wants_path = _spec_path_flags.get(id(spec), False)
                selected = (
                    spec.select(eff_outs, path=path)
                    if _pspec_wants_path
                    else spec.select(eff_outs)
                )
            else:
                selected = eff_outs
            flat_selected = jax.tree_util.tree_leaves(selected)
            sel_tree = jax.tree_util.tree_structure(selected)
            _spec = spec  # capture for host closure
            _of = _once_fired  # capture per-verbose() set by reference

            def _host(step_: Any, *flat_vals: Any) -> None:
                value = jax.tree_util.tree_unflatten(sel_tree, list(flat_vals))
                event = TapEvent(path=path, step=step_.item(), value=value, total=total)
                if on_step is not None:
                    _guard(on_step, event)
                if _spec.alert is not None:
                    _fire_alert(_spec, event, _of)

            # M1d FIX 1: gate primitive taps with sample_every when inside a loop.
            # Prim taps OUTSIDE any loop (_in_loop=False / step sentinel -1) are
            # always ungated so they always fire regardless of sample_every.
            if sample_every > 1 and _in_loop:
                jax.lax.cond(
                    step % sample_every == 0,
                    lambda _: jax.debug.callback(
                        _host, step, *flat_selected, ordered=False
                    ),
                    lambda _: None,
                    step,
                )
            else:
                jax.debug.callback(_host, step, *flat_selected, ordered=False)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if kwargs:
            raise TypeError("jaxtap.verbose does not support keyword arguments to f")
        return interpret(
            f,
            args,
            tap_cb,
            internal_ops,
            where=where,
            max_depth=max_depth,
            prim_taps=taps,
            prim_tap_fn=prim_tap_fn,
            _start_cf_index=_start_cf_index,
        )

    return wrapped


def record(
    f: "Callable | None" = None,
    *,
    select: "Callable | None" = None,
    ops: "tuple[str, ...]" = ("scan", "while_loop"),
    sample_every: int = 1,
    where: "Callable[[str], bool] | None" = None,
    max_depth: "int | None" = None,
    taps: "Sequence[PrimitiveTap]" = (),
    on_step: "Callable[[TapEvent], None] | None" = None,
    alert: "Callable[[TapEvent], Any] | None" = None,
    alert_once: bool = False,
) -> "tuple[Callable, FlightRecorder] | _RecordContext":
    """
    Dual-form recorder for zero-code-change telemetry.

    **B-form** (callable given): wire a :class:`FlightRecorder` to
    ``verbose(f, ...)`` and return the ``(tapped_fn, recorder)`` pair::

        g, rec = tap.record(f)
        g(*args)
        rec.df()

        # With live-streaming callback:
        g, rec = tap.record(f, on_step=announce)
        g(*args)   # announce() AND rec both receive every TapEvent

    **A-form** (no callable): return a context manager that monkeypatches
    ``jax.lax.scan`` / ``jax.lax.while_loop`` for the duration of the block,
    collecting events from any user code that runs inside — unmodified::

        with tap.record(select=..., taps=[tap.on("cholesky")]) as rec:
            result = anything(...)   # UNMODIFIED user code
        rec.events  # list[TapEvent]
        rec.df()    # pandas DataFrame

        # Delete the ``with`` line → nothing was ever there.

        # With live-streaming callback:
        with tap.record(on_step=announce) as rec:
            result = anything(...)  # announce() fires live; rec collects all

    All keyword arguments except ``on_step`` are identical in both forms and
    are forwarded to :func:`verbose`.

    Parameters
    ----------
    f : Callable or None
        Function to instrument (B-form), or ``None`` for A-form (context manager).
    select : Callable or None, optional
        Traced-side reducer applied to carry before host boundary crossing
        (see :func:`verbose`). Default: ``None``.
    ops : tuple of str, optional
        Control-flow operators to tap (``"scan"``, ``"while_loop"``).
        Default: ``("scan", "while_loop")``.
    sample_every : int, optional
        Fire taps only on steps 0, k, 2k, … (device-side gate).
        Default: 1 (every step).
    where : Callable or None, optional
        Path predicate; only CF nodes whose path satisfies ``where(path)`` emit
        carry taps. Default: ``None`` (all nodes emit).
    max_depth : int or None, optional
        Depth limit; CF nodes at depth > ``max_depth`` do not emit carry taps.
        Default: ``None`` (no limit).
    taps : Sequence of PrimitiveTap, optional
        Primitive taps (created via :func:`on`) to fire on named primitives.
        Default: ``()``.
    alert : Callable or None, optional
        HOST-side predicate evaluated on each carry-tap :class:`TapEvent`.
        When truthy, emits one terse line to stderr::

            [tap] FAIL {path} {step}/{total}: {msg}

        where ``msg`` is the returned value when it is a ``str``, otherwise
        ``"alert"``. Runs inside ``_guard`` discipline (never propagates).
        Fires BEFORE ``on_step``. Default: ``None``.
    alert_once : bool, optional
        When ``True``, alert fires at most once per path per :func:`record`
        call. Default: ``False``.
    on_step : Callable or None, optional
        Optional additional host callback.  When given, every :class:`TapEvent`
        is delivered to BOTH the :class:`FlightRecorder` (``rec``) AND
        ``on_step``, in that order, both ``_guard``-wrapped (never-raise).
        For the A-form this callback is dynamically resolved at event-fire time
        (see ``_dynamic_router``), so it respects the same post-exit and
        cache-hit routing as the recorder itself. Default: ``None``.

    A-form notes
    ------------
    The host callback baked into JIT-compiled artifacts is always the
    module-level ``_dynamic_router`` singleton.  This means:

    - **Phantom emission prevented**: after ``__exit__`` events are dropped,
      not appended to the closed recorder.
    - **Cache-hit routing**: if the same jitted function is called inside a
      NEW context, events route to the new context's recorder — even though
      the compiled artifact was baked inside a prior context.
    - **Trace-time config**: ``select`` and ``taps`` are baked at trace time
      (device-side).  On a cache-hit in a new context with different
      ``select``/``taps``, the trace-time config applies; only host routing
      is live.  Document this to users when mixing configs across contexts.
    - **Pre-context compilation**: functions compiled BEFORE any context was
      entered have no callback baked in → 0 events inside a context.
      Workaround: ``jax.clear_caches()`` before entering.

    Thread delegation: with ONE context active, any calling thread's scan/while
    is attributed to it.  With >=2 simultaneous contexts, only the context
    whose owner thread matches receives events; bystanders pass through.

    vmap × while_loop carry taps: ghost events from finished lanes are suppressed
    (see :func:`verbose` Notes).  Prim-tap ghost-firing in vmapped while bodies
    is a residual known boundary.
    """
    if f is None:
        from ._ashell import _RecordContext as _RC

        return _RC(
            select=select,
            ops=ops,
            sample_every=sample_every,
            where=where,
            max_depth=max_depth,
            taps=taps,
            on_step=on_step,
            alert=alert,
            alert_once=alert_once,
        )

    from .collectors import FlightRecorder as _FlightRecorder

    recorder = _FlightRecorder()

    if on_step is not None:
        # Combine recorder + user callback into a single on_step for verbose().
        _user_cb = on_step

        def _combined(event: TapEvent) -> None:
            _guard(recorder, event)
            _guard(_user_cb, event)

        effective_on_step: Callable[[TapEvent], None] = _combined
    else:
        effective_on_step = recorder

    tapped = verbose(
        f,
        on_step=effective_on_step,
        select=select,
        ops=ops,
        sample_every=sample_every,
        where=where,
        max_depth=max_depth,
        taps=taps,
        alert=alert,
        alert_once=alert_once,
    )
    return tapped, recorder


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def watch_nan(
    prim_name: str,
    label: str = "NaN/Inf",
    output: "int | None" = None,
    once: bool = False,
) -> PrimitiveTap:
    """Create a :class:`PrimitiveTap` that alerts when any float output is non-finite.

    Equivalent to::

        tap.on(prim_name,
               select=<device-side all-isfinite over all float outputs>,
               alert=lambda ok: not ok,
               label=label)

    The ``select`` reduces over every float output leaf via ``jnp.isfinite``;
    non-float outputs are skipped safely.  The ``alert`` fires when the result
    is False (i.e. at least one NaN or Inf was found).

    Parameters
    ----------
    prim_name:
        JAX primitive name to watch (e.g. ``"cholesky"``).
        Use :func:`primitives` to discover the correct string.
    label:
        Short label shown in the alert line.  Default: ``"NaN/Inf"``.
    output:
        Optional single-output index.  When given, the finiteness check applies
        to that one output array (not a tuple) — useful for primitives such as
        ``eigh`` which return multiple outputs and you only care about one.
        ``None`` (default) checks all float outputs in the output tuple.

        Note: indices refer to the JAX *primitive's* output order, which can
        differ from the Python API's return order.  Use ``tap.print(prim_name)``
        first to inspect the actual layout before relying on a specific index.
    once:
        When ``True``, the alert fires at most once per :func:`verbose` call.
        Useful when only the first non-finite occurrence matters and you want to
        suppress the flood of repeated FAIL lines for every subsequent step.
        Default: ``False``.

    Returns
    -------
    PrimitiveTap

    Examples
    --------
    Watch for NaN/Inf in primitive outputs inside a scan::

        import jax.lax as lax

        spec = tap.watch_nan("add")
        events = []

        def f(x):
            def body(carry, _):
                # Force NaN on step 1
                return jnp.where(jnp.arange(2) == 1, jnp.nan, carry + 1), None
            return lax.scan(body, x, jnp.arange(2))[0]

        tapped = tap.verbose(f, taps=[spec])
        result = tapped(jnp.array([1.0, 2.0]))
        # stderr output:
        # [tap] FAIL scan[0]/add[0] 1/2: NaN/Inf

    Watch a single output from a multi-output primitive::

        # For eigh which returns (eigenvalues, eigenvectors) in Python
        # but the primitive outputs (eigenvectors, eigenvalues),
        # watch the eigenvalues (output=1 in primitive order)
        spec = tap.watch_nan("eigh", output=1)
    """
    if output is not None:
        # Single-array mode: select receives one array, not a tuple.
        def _select_finite_single(o: Any) -> Any:
            try:
                if jnp.issubdtype(o.dtype, jnp.floating):
                    return jnp.all(jnp.isfinite(o))
            except (AttributeError, TypeError):
                pass
            return jnp.bool_(True)

        select_fn: Callable[..., Any] = _select_finite_single
    else:
        # Tuple mode: select receives the full output tuple.
        def _select_finite_tuple(outs: tuple) -> Any:
            checks = []
            for o in outs:
                try:
                    if jnp.issubdtype(o.dtype, jnp.floating):
                        checks.append(jnp.all(jnp.isfinite(o)))
                except (AttributeError, TypeError):
                    pass
            if not checks:
                return jnp.bool_(True)  # no float outputs — trivially ok
            if len(checks) == 1:
                return checks[0]
            return jnp.all(jnp.stack(checks))

        select_fn = _select_finite_tuple

    return on(
        prim_name,
        select=select_fn,
        alert=lambda ok: not ok,
        label=label,
        output=output,
        once=once,
    )


def print(  # noqa: A001 — intentionally shadows builtin; internal code uses sys.stderr.write
    prim_name: str,
    output: "int | None" = None,
    select: "Callable | None" = None,
    label: "str | None" = None,
    once: bool = False,
) -> PrimitiveTap:
    """Create a :class:`PrimitiveTap` that always prints the tapped value to stderr.

    Every time the named primitive fires, emits one terse line to stderr::

        [tap] {path} {step}/{total_or_'?'}: {value}

    where ``value`` is formatted with
    ``numpy.printoptions(precision=4, threshold=8, edgeitems=2)`` so large
    arrays are truncated rather than flooding the terminal.  Internal newlines
    in the numpy repr are collapsed to spaces so the event always fits on one
    physical line.

    This is the simplest diagnostic tool: add it while debugging, remove it
    when done — no other code changes needed.

    Parameters
    ----------
    prim_name:
        JAX primitive name to tap (e.g. ``"dot_general"``).
    output:
        Optional single-output index.  When given, only that output array is
        printed.  ``None`` prints the full output tuple.

        Note: indices refer to the JAX *primitive's* output order, which can
        differ from the Python API's return order.  Use ``tap.print(prim_name)``
        without an index first to inspect the actual output layout.
    select:
        Optional on-device reducer applied before printing.  Receives the
        output (single array if ``output=k``, tuple otherwise).
    label:
        Stored on the spec for composition with an additional ``alert``;
        not shown in the print format itself.
    once:
        When ``True``, print only on the first fire per :func:`verbose` call.
        Default: ``False``.

    Returns
    -------
    PrimitiveTap

    Examples
    --------
    Print primitive outputs to stderr::

        spec = tap.print("dot_general")
        tapped = tap.verbose(lambda x: jnp.dot(x, x), taps=[spec])
        result = tapped(jnp.array([1.0, 2.0, 3.0]))
        # stderr output:
        # [tap] dot_general[0] -1/?: array([14.], dtype=float32)

    Print only the first occurrence (``once=True``)::

        spec = tap.print("add", once=True)
        def f(x):
            def body(carry, _):
                return carry + 1, None
            return lax.scan(body, x, jnp.arange(3))[0]

        tapped = tap.verbose(f, taps=[spec])
        result = tapped(0)
        # stderr (fires only on step 0):
        # [tap] scan[0]/add[0] 0/3: array(1, dtype=int32)
    """
    return PrimitiveTap(
        prim_name=prim_name,
        output=output,
        select=select,
        alert=lambda v: True,  # always fire
        label=label,
        once=once,
        _printer=True,
    )


# ---------------------------------------------------------------------------
# Discovery helper
# ---------------------------------------------------------------------------


def _count_prims(jaxpr: Any, counts: "dict[str, int]") -> None:
    """Recursively count all primitives in a Jaxpr, descending into sub-jaxprs."""
    for eqn in jaxpr.eqns:
        name = eqn.primitive.name
        counts[name] = counts.get(name, 0) + 1
        p = eqn.params
        # scan, jit/pjit/closed_call: params["jaxpr"] is a ClosedJaxpr.
        # remat2: params["jaxpr"] is a bare Jaxpr (no .jaxpr attr).
        if "jaxpr" in p:
            sub = p["jaxpr"]
            if hasattr(sub, "jaxpr"):  # ClosedJaxpr
                _count_prims(sub.jaxpr, counts)
            elif hasattr(sub, "eqns"):  # bare Jaxpr
                _count_prims(sub, counts)
        # while_loop: params["cond_jaxpr"] and params["body_jaxpr"] are ClosedJaxprs.
        if "cond_jaxpr" in p:
            cj = p["cond_jaxpr"]
            if hasattr(cj, "jaxpr"):
                _count_prims(cj.jaxpr, counts)
        if "body_jaxpr" in p:
            bj = p["body_jaxpr"]
            if hasattr(bj, "jaxpr"):
                _count_prims(bj.jaxpr, counts)
        # cond/switch: params["branches"] is a tuple of ClosedJaxprs.
        if "branches" in p:
            for branch in p["branches"]:
                if hasattr(branch, "jaxpr"):
                    _count_prims(branch.jaxpr, counts)
        # custom_jvp_call / custom_vjp_call: params["call_jaxpr"].
        if "call_jaxpr" in p:
            cj = p["call_jaxpr"]
            if hasattr(cj, "jaxpr"):
                _count_prims(cj.jaxpr, counts)
            elif hasattr(cj, "eqns"):
                _count_prims(cj, counts)


def primitives(f: Callable, *args: Any) -> "dict[str, int]":
    """Find the string to pass to tap.on().

    Traces ``f(*args)`` once (via ``jax.make_jaxpr``) and returns a dict
    ``{primitive_name: count}`` for every primitive in the program, including
    inside nested sub-jaxprs (scan/while bodies, jit/pjit boundaries, cond
    branches, remat regions).  Read-only — no instrumentation, no execution.

    Parameters
    ----------
    f:
        The function to trace.
    *args:
        Example arguments (shapes / dtypes determine the trace; values are unused).

    Returns
    -------
    dict[str, int]
        Mapping ``primitive_name → total occurrence count`` across all nesting levels.

    Examples
    --------
    Discover primitives in a computation::

        def model(x):
            def body(carry, _):
                return jnp.linalg.cholesky(carry)
            return lax.scan(body, x, jnp.arange(5))[0]

        prims = tap.primitives(model, jnp.eye(2))
        # Returns: {'scan': 1, 'cholesky': 5, 'integer_pow': 1, ...}

    Use the discovered names with ``tap.on()``::

        spec = tap.on("cholesky", alert=lambda ok: not ok)
        with tap.record(taps=[spec]) as rec:
            result = model(jnp.eye(2))
    """
    closed = jax.make_jaxpr(f)(*args)
    counts: dict[str, int] = {}
    _count_prims(closed.jaxpr, counts)
    return counts


# ---------------------------------------------------------------------------
# Re-export collectors for convenience
# ---------------------------------------------------------------------------


def emergency_restore() -> None:
    """Restore ``jax.lax.scan`` / ``jax.lax.while_loop`` to session-captured originals.

    Use this function to recover a corrupted interactive session after a crash
    inside a ``tap.record()`` context manager or an interrupted kernel that
    left jax-tap's monkeypatch in place.

    Behavior:

    - If jax-tap's patch is on top: restores the original implementations,
      clears the event registry, and resets all session state.
    - If a foreign patch is on top: warns but leaves it alone; still clears
      jax-tap's internal state.

    Notes
    -----
    Well-behaved code using ``with tap.record():`` (context manager form) never
    needs this — the context manager ensures cleanup on ``__exit__``, even after
    exceptions.  This is a recovery tool for interactive debugging only.

    Examples
    --------
    Restore after an interrupted session::

        # Your kernel crashed inside tap.record(); JAX is now patched.
        # In a new cell:
        import jaxtap as tap
        tap.emergency_restore()
        # jax.lax.scan and jax.lax.while_loop are now unpatched
    """
    from ._ashell import emergency_restore as _er

    _er()


def __getattr__(name: str) -> Any:
    """Lazy re-export of collectors to avoid circular imports at module load."""
    if name in ("FlightRecorder", "JSONLWriter", "read_jsonl"):
        from . import collectors as _collectors

        return getattr(_collectors, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
