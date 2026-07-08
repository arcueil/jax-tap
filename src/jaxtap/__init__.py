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
import warnings
from typing import TYPE_CHECKING, Any, Callable, Sequence

import jax

from ._walker import interpret

if TYPE_CHECKING:
    from .collectors import FlightRecorder

__all__ = [
    "TapEvent",
    "PrimitiveTap",
    "on",
    "verbose",
    "record",
    "FlightRecorder",
    "JSONLWriter",
    "read_jsonl",
]


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TapEvent:
    """A single telemetry emission from inside a tapped control-flow operator."""

    path: str  # stable address, e.g. "scan[0]/while[0]"
    step: int  # iteration index (0-based); -1 for primitive taps outside any loop
    value: Any  # selected carry payload delivered to the host


@dataclasses.dataclass(frozen=True)
class PrimitiveTap:
    """Spec for tapping a specific JAX primitive by name.

    Created via :func:`on`::

        tap.on("cholesky")
        tap.on("cholesky", select=lambda outs: jnp.all(jnp.isfinite(outs[0])))

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

    Scope
    -----
    Primitive taps only fire in code the walker descends into.  Loops filtered by
    ``ops``/``where``/``max_depth`` are not descended, so primitives inside them
    are silent.  ``sample_every`` does NOT gate primitive taps (loop-carry taps
    only).  AD-opaque primitives (``custom_jvp_call``/``custom_vjp_call``
    interiors) are not descended.

    Step context
    ------------
    ``TapEvent.step`` is the enclosing loop's live step value.  When a primitive
    tap fires outside any scan/while loop, ``step == -1`` (the sentinel).
    """

    prim_name: str
    select: Callable | None = None


def on(prim_name: str, select: Callable | None = None) -> PrimitiveTap:
    """Create a :class:`PrimitiveTap` spec.

    Parameters
    ----------
    prim_name:
        Name of the JAX primitive to tap.
    select:
        Optional on-device reducer applied to the primitive's output tuple.

    Returns
    -------
    PrimitiveTap
    """
    return PrimitiveTap(prim_name=prim_name, select=select)


# ---------------------------------------------------------------------------
# Warn-once guard
# ---------------------------------------------------------------------------

_warned: set[int] = set()  # tracks id(on_step) that have warned once this session
# NOTE: id() values can be reused after GC; warn-once dedup is best-effort for M0.


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
    on_step: Callable[[TapEvent], None],
    select: Callable | None = None,
    ops: tuple[str, ...] = ("scan", "while_loop"),
    sample_every: int = 1,
    where: "Callable[[str], bool] | None" = None,
    max_depth: "int | None" = None,
    taps: "Sequence[PrimitiveTap]" = (),
) -> Callable:
    """
    Return a function bitwise-identical to ``f`` that emits telemetry from
    inside ``lax.scan`` / ``lax.while_loop`` without any modification to ``f``.

    Parameters
    ----------
    f:
        The function to instrument.
    on_step:
        Host callback called with a :class:`TapEvent` after each control-flow
        iteration or matched primitive.  Must never raise (failures are caught
        and warned once).
    select:
        Optional traced-side callable applied to the carry tuple INSIDE the
        traced program before the host callback.  Receives a tuple of carry
        leaves; only the selector's output crosses the host boundary.
        Default: the full carry tuple.
    ops:
        Which control-flow operators to tap.  Accepted values: ``"scan"``,
        ``"while_loop"``.  Default: both.
    sample_every:
        Fire taps only on steps 0, k, 2k, … (device-side gate via
        ``lax.cond``).  Default: 1 (every step).  Must be ≥ 1.
        Note: ``sample_every`` does NOT gate primitive taps (``taps=``).
    where:
        Optional path predicate: only CF nodes whose path satisfies
        ``where(path)`` are instrumented.  The address counter advances for
        filtered-out nodes so addressing remains stable.  Default: all nodes.
    max_depth:
        Optional depth limit.  CF nodes at depth > ``max_depth`` (depth =
        number of ``/`` segments in the path) are bound opaquely.
        Default: no limit.
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

    Notes
    -----
    **Carry boundary**: jaxprs are flat — pytree structure of the carry is erased
    by tracing and is NOT recoverable from the scan equation.  When ``select`` is
    ``None``, ``TapEvent.value`` is a *flat tuple of carry leaves* (not the
    original pytree).  Use ``select`` to reshape them into the desired structure::

        tap.verbose(f, on_step=cb, select=lambda leaves: {"a": leaves[0], "b": leaves[1]})

    The ``select`` function receives the tuple of carry leaves and its return value
    is delivered to the host with its pytree structure preserved via trace-time
    treedef capture.
    """
    if sample_every < 1:
        raise ValueError(f"sample_every must be >= 1, got {sample_every}")

    internal_ops: frozenset[str] = frozenset(_OP_NAME_MAP[op] for op in ops if op in _OP_NAME_MAP)

    if select is not None:
        # tap_cb is called INSIDE the traced computation (scan/while body),
        # so ``select`` runs on-device before the host-boundary crossing.
        def _base_tap_cb(path: str, step: Any, *carry_leaves: Any) -> None:
            selected = select(carry_leaves)
            flat_selected = jax.tree_util.tree_leaves(selected)
            # Capture pytree structure at Python (trace) time for host-side recon.
            sel_tree = jax.tree_util.tree_structure(selected)

            def _host(step_: Any, *flat_vals: Any) -> None:
                value = jax.tree_util.tree_unflatten(sel_tree, list(flat_vals))
                _guard(on_step, TapEvent(path=path, step=int(step_), value=value))

            jax.debug.callback(_host, step, *flat_selected, ordered=False)

    else:
        # tap_cb is called INSIDE the traced computation; ``jax.debug.callback``
        # ships the carry leaves to the host.  ``path`` is a static Python string
        # captured in the closure.
        def _base_tap_cb(path: str, step: Any, *carry_leaves: Any) -> None:  # type: ignore[misc]
            def _host(step_: Any, *leaves: Any) -> None:
                _guard(on_step, TapEvent(path=path, step=int(step_), value=leaves))

            jax.debug.callback(_host, step, *carry_leaves, ordered=False)

    # Wrap with sample_every gate (device-side lax.cond) when k > 1.
    # Both branches return None (empty pytree) so lax.cond type-checks;
    # JAX's effects system ensures the debug callback fires only in the true branch.
    if sample_every > 1:
        _uncapped = _base_tap_cb

        def tap_cb(path: str, step: Any, *carry_leaves: Any) -> None:
            jax.lax.cond(
                step % sample_every == 0,
                lambda _: _uncapped(path, step, *carry_leaves),
                lambda _: None,
                step,
            )

    else:
        tap_cb = _base_tap_cb

    # Build the primitive-tap callback if any specs were provided.
    # This function is called from _interp after binding a matched primitive.
    # It fires jax.debug.callback → _guard-wrapped on_step with a TapEvent.
    # path: Python string (static, determined at trace time)
    # step: JAX int32 (live, from enclosing loop) or jnp.int32(-1) (outside loop)
    # outvals: tuple of the primitive's output arrays (device-side)
    # spec: the matching PrimitiveTap
    prim_tap_fn: Callable | None = None
    if taps:

        def prim_tap_fn(path: str, step: Any, outvals: tuple, spec: PrimitiveTap) -> None:  # type: ignore[misc]
            if spec.select is not None:
                selected = spec.select(outvals)
            else:
                selected = outvals
            flat_selected = jax.tree_util.tree_leaves(selected)
            sel_tree = jax.tree_util.tree_structure(selected)

            def _host(step_: Any, *flat_vals: Any) -> None:
                value = jax.tree_util.tree_unflatten(sel_tree, list(flat_vals))
                _guard(on_step, TapEvent(path=path, step=int(step_), value=value))

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
        )

    return wrapped


def record(
    f: Callable,
    *,
    select: "Callable | None" = None,
    ops: "tuple[str, ...]" = ("scan", "while_loop"),
    sample_every: int = 1,
    where: "Callable[[str], bool] | None" = None,
    max_depth: "int | None" = None,
    taps: "Sequence[PrimitiveTap]" = (),
) -> "tuple[Callable, FlightRecorder]":
    """
    Wire a :class:`FlightRecorder` to ``verbose(f, ...)`` and return the pair.

    Usage::

        g, rec = tap.record(f)
        g(*args)
        rec.df()

    All keyword arguments are forwarded to :func:`verbose`.
    """
    from .collectors import FlightRecorder as _FlightRecorder

    recorder = _FlightRecorder()
    tapped = verbose(
        f,
        on_step=recorder,
        select=select,
        ops=ops,
        sample_every=sample_every,
        where=where,
        max_depth=max_depth,
        taps=taps,
    )
    return tapped, recorder


# ---------------------------------------------------------------------------
# Re-export collectors for convenience
# ---------------------------------------------------------------------------


def __getattr__(name: str) -> Any:
    """Lazy re-export of collectors to avoid circular imports at module load."""
    if name in ("FlightRecorder", "JSONLWriter", "read_jsonl"):
        from . import collectors as _collectors

        return getattr(_collectors, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
