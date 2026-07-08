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

Ergonomic collector helper::

    g, rec = tap.record(f)
    g(*args)
    rec.df()
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import TYPE_CHECKING, Any, Callable

import jax

from ._walker import interpret

if TYPE_CHECKING:
    from .collectors import FlightRecorder

__all__ = [
    "TapEvent",
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
    step: int  # iteration index (0-based)
    value: Any  # selected carry payload delivered to the host


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
        iteration.  Must never raise (failures are caught and warned once).
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
    where:
        Optional path predicate: only CF nodes whose path satisfies
        ``where(path)`` are instrumented.  The address counter advances for
        filtered-out nodes so addressing remains stable.  Default: all nodes.
    max_depth:
        Optional depth limit.  CF nodes at depth > ``max_depth`` (depth =
        number of ``/`` segments in the path) are bound opaquely.
        Default: no limit.

    Returns
    -------
    Callable
        A function with the same signature as ``f``, returning bitwise-identical
        results, that emits a :class:`TapEvent` via ``on_step`` for each
        control-flow iteration.

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

    tap_cb = _base_tap_cb

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if kwargs:
            raise TypeError("jaxtap.verbose does not support keyword arguments to f")
        return interpret(
            f,
            args,
            tap_cb,
            internal_ops,
            sample_every=sample_every,
            where=where,
            max_depth=max_depth,
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
