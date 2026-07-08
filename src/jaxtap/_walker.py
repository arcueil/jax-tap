"""
jax-tap B-core: recursive jaxpr-walker transform.

Stable addressing scheme
------------------------
Each nesting level maintains a single counter ``n_cf`` that increments for
every control-flow equation encountered at that level, regardless of kind.
This produces paths like::

    scan[0]           -- first CF eqn at top level is a scan
    scan[0]/while[1]  -- second CF eqn inside scan[0] body is a while
    scan[0]/scan[0]   -- first CF eqn inside scan[0] body is itself a scan

The counter increments for ALL CF primitives at a level (whether or not they
are in ``ops``), so addresses are stable when ops filtering changes.

Call-primitive dispatch
-----------------------
``jit`` / ``pjit`` / ``closed_call`` carry the sub-jaxpr under ``params["jaxpr"]``.
``custom_jvp_call`` / ``custom_vjp_call`` carry it under ``params["call_jaxpr"]``.

v1 policy for ``custom_jvp_call`` / ``custom_vjp_call``: bind opaquely, do NOT
recurse inside.  Inserting callbacks through AD boundaries can silently alter
gradient semantics (same concern that deprecated ``host_callback`` raised).
These primitives are bound via their original ``eqn.primitive.bind`` so autodiff
correctness is fully preserved.

v1 policy for ``jit`` / ``pjit`` / ``closed_call``: recurse into the inner jaxpr
and re-wrap the interpreted sub-call in a fresh ``jax.jit`` so that the compile
boundary is preserved when the transformed function is called without an outer jit.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
from jax.extend import core as jax_core

from ._rewrites import rewrite_scan, rewrite_while

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Primitives whose sub-jaxpr lives under params["jaxpr"].
_JIT_PRIMS: frozenset[str] = frozenset({"jit", "pjit", "closed_call"})

# AD-boundary primitives: bind opaquely (see module docstring).
_AD_PRIMS: frozenset[str] = frozenset({"custom_jvp_call", "custom_vjp_call"})

# All control-flow primitives (counted for stable addressing regardless of ops).
_CF_PRIMS: frozenset[str] = frozenset({"scan", "while"})

TapCallback = Callable[..., None]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def interpret(
    f: Callable,
    args: tuple,
    tap_cb: TapCallback,
    ops: frozenset[str],
) -> Any:
    """
    Trace ``f(*args)`` once with ``make_jaxpr(return_shape=True)`` and run
    the recursive interpreter, emitting taps for primitives in ``ops``.

    Returns the output pytree of ``f(*args)``.
    """
    closed, out_shapes = jax.make_jaxpr(f, return_shape=True)(*args)
    out_tree = jax.tree_util.tree_structure(out_shapes)
    flat_args = jax.tree_util.tree_leaves(args)
    out_flat = _interp(closed.jaxpr, closed.consts, flat_args, tap_cb, ops, path="")
    return jax.tree_util.tree_unflatten(out_tree, out_flat)


# ---------------------------------------------------------------------------
# Recursive interpreter
# ---------------------------------------------------------------------------


def _read(env: dict, a: Any) -> Any:
    if isinstance(a, jax_core.Literal):
        return a.val
    return env[a]


def _interp(
    jaxpr: jax_core.Jaxpr,
    consts: list,
    args: list,
    tap_cb: TapCallback,
    ops: frozenset[str],
    path: str,
) -> list:
    """Evaluate ``jaxpr`` against ``args``, rewriting CF primitives in ``ops``."""
    env: dict = {}

    for v, val in zip(jaxpr.constvars, consts):
        env[v] = val
    for v, val in zip(jaxpr.invars, args):
        env[v] = val

    n_cf = 0  # per-level CF counter for stable path addressing

    for eqn in jaxpr.eqns:
        invals = [_read(env, a) for a in eqn.invars]
        prim_name = eqn.primitive.name

        if prim_name in _CF_PRIMS:
            cf_index = n_cf
            n_cf += 1

            if prim_name == "scan" and "scan" in ops:
                here = f"{path}scan[{cf_index}]"
                outvals = rewrite_scan(eqn, invals, tap_cb, ops, here, _interp)

            elif prim_name == "while" and "while" in ops:
                here = f"{path}while[{cf_index}]"
                outvals = rewrite_while(eqn, invals, tap_cb, ops, here, _interp)

            else:
                # CF primitive not in ops: bind opaquely (still counted above).
                outvals = eqn.primitive.bind(*invals, **eqn.params)
                if not eqn.primitive.multiple_results:
                    outvals = [outvals]

        elif prim_name in _JIT_PRIMS:
            inner = eqn.params["jaxpr"]

            def _inner_call(*flat_in, _j=inner, _p=path):
                return _interp(_j.jaxpr, _j.consts, list(flat_in), tap_cb, ops, _p)

            result = jax.jit(_inner_call)(*invals)
            # Normalise to list; jit preserves the pytree structure of the return.
            outvals = list(result) if isinstance(result, (list, tuple)) else [result]

        elif prim_name in _AD_PRIMS:
            # Bind opaquely — do NOT recurse (see module docstring).
            outvals = eqn.primitive.bind(*invals, **eqn.params)
            if not eqn.primitive.multiple_results:
                outvals = [outvals]

        else:
            outvals = eqn.primitive.bind(*invals, **eqn.params)
            if not eqn.primitive.multiple_results:
                outvals = [outvals]

        for v, val in zip(eqn.outvars, outvals):
            env[v] = val

    return [_read(env, v) for v in jaxpr.outvars]
