"""
jax-tap B-core: recursive jaxpr-walker transform.

Stable addressing scheme
------------------------
Each nesting level maintains a single counter ``n_cf`` that increments for
every higher-order primitive encountered at that level, regardless of kind.
This produces paths like::

    scan[0]                    -- first boundary at top level is a scan
    scan[0]/while[1]           -- second boundary inside scan[0] body is a while
    scan[0]/scan[0]            -- first boundary inside scan[0] body is itself a scan
    jit[0]/scan[0]             -- first boundary inside a jit is a scan
    cond[0]/b1/scan[0]         -- scan in branch-1 of the first cond
    remat[0]/scan[0]           -- scan inside a checkpoint boundary

The counter increments for ALL higher-order primitives at a level (whether or
not they are in ``ops``), so addresses are stable when ops filtering changes.

Boundary-visible addressing (M3 remediation)
---------------------------------------------
Every sub-jaxpr-carrying higher-order primitive gets a path segment:
``jit[k]``, ``cond[k]/b{j}``, ``remat[k]`` — alongside ``scan[k]``/``while[k]``.
This makes paths structurally UNIQUE across jit/cond/remat boundaries (fixes
F2) and honest (the address shows where the tap actually lives, fixes F1).

F1: instrumentation was silently dropped inside ``cond``/``switch``/``remat2``
    because the walker bound them opaquely without recursing.  Fix: recurse
    into every branch (cond/switch) or sub-jaxpr (remat2).

F2: jit boundary passed the path unchanged and _interp reset n_cf=0, causing
    a jit-nested scan to collide with a sibling top-level scan.  Fix: jit
    counts in n_cf and appends ``jit[k]/`` to the path before recursing.

Call-primitive dispatch
-----------------------
``jit`` / ``pjit`` / ``closed_call`` carry the sub-jaxpr under ``params["jaxpr"]``
(a ClosedJaxpr).
``remat2`` carries the sub-jaxpr under ``params["jaxpr"]`` (a bare Jaxpr with
empty constvars; all inputs come from eqn.invars).
``cond`` (used for both ``lax.cond`` and ``lax.switch``) carries branch jaxprs
under ``params["branches"]`` (tuple of ClosedJaxpr); ``invals[0]`` is the int32
branch selector; ``invals[1:]`` are the shared branch operands.
``custom_jvp_call`` / ``custom_vjp_call`` carry it under ``params["call_jaxpr"]``.

v1 policy for ``custom_jvp_call`` / ``custom_vjp_call``: bind opaquely, do NOT
recurse inside.  Inserting callbacks through AD boundaries can silently alter
gradient semantics (same concern that deprecated ``host_callback`` raised).
These primitives (and all other primitives) are bound using the canonical
``get_bind_params`` pattern from ``jax.core.eval_jaxpr``::

    bind_params = eqn.primitive.get_bind_params(eqn.params)
    eqn.primitive.bind(*invals, **bind_params)

This is REQUIRED for ``custom_jvp_call`` / ``custom_vjp_call``: their params
carry ``call_jaxpr`` / ``jvp_jaxpr_fun`` which ``get_bind_params`` converts
into a ``subfuns`` keyword argument that ``bind`` expects.  Naively passing
``**eqn.params`` raises ``KeyError: 'subfuns'``.  The ``get_bind_params``
pattern preserves the custom JVP/VJP rule so ``jax.grad(verbose(f))``
differentiates correctly through instrumented programs.

v1 policy for ``jit`` / ``pjit`` / ``closed_call``: recurse into the inner jaxpr
with an extended path (``jit[k]/``), then re-wrap in a fresh ``jax.jit`` to
preserve the compile boundary.  NOTE: this fresh jit does not preserve the
original ``donated_invars``/``in_shardings``/``out_shardings`` params — a known
limitation acceptable for pre-v1.  For ``remat2``, ``jax.checkpoint`` is used
with ``prevent_cse`` and ``policy`` threaded from the original eqn params.

Step threading (M1a)
--------------------
``_interp`` now carries a ``step`` parameter — the live enclosing loop step
value, threaded from the rewrites.  This enables primitive taps to report
the correct enclosing step even when the primitive is nested several boundaries
deep.  Outside any loop the sentinel ``jnp.int32(-1)`` is used (callers see
``TapEvent.step == -1``).

For ``jit``/``pjit``/``closed_call`` boundaries the step is passed as an
**explicit argument** to the fresh ``jax.jit`` wrapper because JIT creates a
new compilation context — closed-over abstract tracers from an outer scan body
are not automatically forwarded across a JIT boundary.  For ``cond``/``switch``
and ``remat2`` the step is captured via Python closure, which is safe because
those primitives trace their sub-functions in the current (outer) tracing
context.

Primitive taps (M1a)
--------------------
After binding any NON-boundary, NON-AD eqn, ``_interp`` checks the eqn's
primitive name against the ``prim_taps`` sequence.  On a match it fires
``prim_tap_fn(path, step, outvals_tuple, spec)`` which lives in ``__init__.py``
and wraps ``jax.debug.callback`` → ``_guard``-wrapped ``on_step``.

Primitive tap addressing: ``{enclosing_path}{prim_name}[{j}]`` where ``j`` is a
per-level, per-prim-name counter (separate from the boundary counter ``n_cf``).
``sample_every`` does NOT gate primitive taps in M1a (loop-carry taps only).
Primitive taps only fire inside code the walker descends into; loops filtered
by ``ops``/``where``/``max_depth`` are not descended, so primitive taps inside
them are silent.

Known v1 boundaries (not fixed here, documented)
-------------------------------------------------
A1 — vmap(while_loop): ``while_loop`` inside a ``vmap`` emits ghost events
     (one per vmap lane in addition to the real ones) because the vmap
     transformation copies the body trace.  This is inherent to how JAX's
     vmap-while interacts with ``jax.debug.callback``.

A3 — remat + grad double-fire: a scan inside a ``jax.checkpoint`` region fires
     its tap once on the forward pass and once on the backward pass (remat
     re-executes the forward body during differentiation).  Both firings carry
     correct carry values; the duplication is inherent to rematerialisation.

Filter hooks (M2)
-----------------
``where`` and ``max_depth`` are evaluated at Python (trace) time for each CF node
before deciding whether to rewrite it.  If a node is filtered out, it is bound
opaquely (same semantics as a primitive not in ``ops``): the address counter still
advances (addressing stability), but the node's body is NOT recursed into and no
per-step tap_cb calls are emitted for it.  This is intentional and mirrors the
``ops`` filtering contract.

Note: with boundary-visible addressing, ``max_depth`` now counts ALL higher-order
boundaries (jit, cond, remat, scan, while) — a scan inside a jit at
``jit[0]/scan[0]`` has depth 1, not 0.

``sample_every`` gating lives one level up, in the ``tap_cb`` closure built by
``verbose()`` in ``__init__.py``.  That closure wraps the callback in a device-side
``lax.cond(step % k == 0, ...)`` before passing it here, so the rewrites always
call ``tap_cb`` unconditionally and correctness is preserved.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
from jax.extend import core as jax_core

from ._ashell import suppress_interception
from ._rewrites import rewrite_scan, rewrite_while

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# All higher-order primitives that get a boundary counter slot in n_cf.
# This makes addressing stable and boundary-visible regardless of ops filtering.
_BOUNDARY_PRIMS: frozenset[str] = frozenset(
    {"scan", "while", "cond", "remat2", "jit", "pjit", "closed_call"}
)

# jit-family: the sub-jaxpr lives under params["jaxpr"] (a ClosedJaxpr).
_JIT_PRIMS: frozenset[str] = frozenset({"jit", "pjit", "closed_call"})

# AD-boundary primitives: bind opaquely (see module docstring).
_AD_PRIMS: frozenset[str] = frozenset({"custom_jvp_call", "custom_vjp_call"})

TapCallback = Callable[..., None]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def interpret(
    f: Callable,
    args: tuple,
    tap_cb: TapCallback,
    ops: frozenset[str],
    where: "Callable[[str], bool] | None" = None,
    max_depth: "int | None" = None,
    prim_taps: Sequence[Any] = (),
    prim_tap_fn: "Callable | None" = None,
    _start_cf_index: int = 0,
) -> Any:
    """
    Trace ``f(*args)`` once with ``make_jaxpr(return_shape=True)`` and run
    the recursive interpreter, emitting taps for primitives in ``ops``.

    Parameters
    ----------
    where:
        Optional path predicate; only CF nodes whose path satisfies ``where``
        are instrumented.  The address counter still advances for filtered-out
        nodes (addressing stability).
    max_depth:
        Optional depth limit.  CF nodes at depth > max_depth (depth = number
        of ``/`` in the path) are bound opaquely.  With boundary-visible
        addressing, jit/cond/remat boundaries each add 1 to the depth.
    prim_taps:
        Sequence of ``PrimitiveTap`` specs.  After binding any non-boundary eqn
        whose ``primitive.name`` matches a spec, fire ``prim_tap_fn``.
    prim_tap_fn:
        Callable ``(path, step, outvals_tuple, spec)`` that fires the host
        callback.  Required when ``prim_taps`` is non-empty.
    _start_cf_index:
        Private kwarg.  Initial value for the top-level boundary counter
        ``n_cf`` in ``_interp``.  Default 0 (standard verbose() usage).
        Set by the A-shell to ensure sequential top-level intercepts are
        addressed as ``scan[k]``/``while[k]`` with k monotonically increasing
        per context rather than always restarting at 0.

    Returns the output pytree of ``f(*args)``.
    """
    # L5: suppress A-shell interception for the ENTIRE interpret() call.
    #
    # Two interception sites must be guarded:
    #
    # 1. make_jaxpr tracing: jax.lax.scan/while_loop inside f resolve to
    #    _patched_scan/_patched_while. Without suppression the active context
    #    would intercept them during tracing, before the B-core even runs.
    #
    # 2. _interp execution (rewrite_scan / rewrite_while): the B-core emits
    #    jax.lax.scan(body_fn, ...) to run the user loop with callbacks baked
    #    in. These calls also resolve to _patched_scan at depth 0 (no outer
    #    _patched_scan on the call stack, since verbose() was called directly
    #    rather than through the A-shell intercept path). Without suppression
    #    the active context intercepts them again — double-instrumentation.
    #
    # The depth counter alone cannot guard site 2 when verbose()/record(f) is
    # invoked directly by the user rather than via _patched_scan (which would
    # have depth > 0). suppress_interception() covers both sites cleanly.
    with suppress_interception():
        closed, out_shapes = jax.make_jaxpr(f, return_shape=True)(*args)
        out_tree = jax.tree_util.tree_structure(out_shapes)
        flat_args = jax.tree_util.tree_leaves(args)
        out_flat = _interp(
            closed.jaxpr,
            closed.consts,
            flat_args,
            tap_cb,
            ops,
            path="",
            where=where,
            max_depth=max_depth,
            step=None,
            prim_taps=prim_taps,
            prim_tap_fn=prim_tap_fn,
            _start_cf_index=_start_cf_index,
        )
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
    where: "Callable[[str], bool] | None" = None,
    max_depth: "int | None" = None,
    step: Any = None,
    prim_taps: Sequence[Any] = (),
    prim_tap_fn: "Callable | None" = None,
    _start_cf_index: int = 0,
) -> list:
    """Evaluate ``jaxpr`` against ``args``, rewriting CF primitives in ``ops``.

    Parameters
    ----------
    step:
        Live step value from the enclosing loop.  ``None`` on entry from
        ``interpret()`` — normalised to ``jnp.int32(-1)`` (outside-any-loop
        sentinel) at the start of this function.
    prim_taps:
        Sequence of ``PrimitiveTap`` specs for primitive-name tapping.
    prim_tap_fn:
        Host callback builder; called as ``prim_tap_fn(path, step, outvals, spec)``
        for each matched primitive.
    _start_cf_index:
        Private kwarg.  Initial value for the top-level boundary counter.
        Forwarded from ``interpret()``; zero for all recursive calls (the counter
        starts fresh at each nesting level).
    """
    # Normalise sentinel: None means "not inside any loop".
    if step is None:
        step = jnp.int32(-1)

    env: dict = {}

    for v, val in zip(jaxpr.constvars, consts):
        env[v] = val
    for v, val in zip(jaxpr.invars, args):
        env[v] = val

    # _start_cf_index: used at the TOP level only (forwarded from interpret()).
    # Recursive calls (rewrite_scan, rewrite_while, jit/cond/remat) call
    # _recurse with path_ already extended — they always start their OWN n_cf
    # at 0 via the default _start_cf_index=0.  Only the outermost _interp call
    # (from interpret()) may receive a non-zero start (A-shell MAJOR-1 fix).
    n_cf = _start_cf_index  # per-level boundary counter for stable path addressing
    # Per-level, per-prim-name occurrence counter for primitive tap addressing.
    n_prim_tap: dict[str, int] = {}

    # Closure that propagates where/max_depth/step/prim_taps through recursive
    # calls; _rewrites.py calls interp_fn with a 7-argument signature
    # (the 7th being the live step).
    def _recurse(
        jaxpr_: Any,
        consts_: Any,
        args_: Any,
        tap_cb_: Any,
        ops_: Any,
        path_: Any,
        step_: Any = None,
    ) -> Any:
        return _interp(
            jaxpr_,
            consts_,
            args_,
            tap_cb_,
            ops_,
            path_,
            where=where,
            max_depth=max_depth,
            step=step_,
            prim_taps=prim_taps,
            prim_tap_fn=prim_tap_fn,
        )

    for eqn in jaxpr.eqns:
        invals = [_read(env, a) for a in eqn.invars]
        prim_name = eqn.primitive.name

        if prim_name in _BOUNDARY_PRIMS:
            # ALL higher-order primitives get a boundary counter slot, whether
            # or not they are in ops (addressing stability + boundary-visible paths).
            cf_index = n_cf
            n_cf += 1

            if prim_name == "scan" and "scan" in ops:
                here = f"{path}scan[{cf_index}]"
                depth = here.count("/")
                # Filter hooks: where- and max_depth-filtered nodes are bound
                # opaquely (addressing counter already advanced above).
                if (where is None or where(here)) and (max_depth is None or depth <= max_depth):
                    outvals = rewrite_scan(eqn, invals, tap_cb, ops, here, _recurse, step)
                else:
                    bind_params = eqn.primitive.get_bind_params(eqn.params)
                    outvals = eqn.primitive.bind(*invals, **bind_params)
                    if not eqn.primitive.multiple_results:
                        outvals = [outvals]

            elif prim_name == "while" and "while" in ops:
                here = f"{path}while[{cf_index}]"
                depth = here.count("/")
                if (where is None or where(here)) and (max_depth is None or depth <= max_depth):
                    outvals = rewrite_while(eqn, invals, tap_cb, ops, here, _recurse, step)
                else:
                    bind_params = eqn.primitive.get_bind_params(eqn.params)
                    outvals = eqn.primitive.bind(*invals, **bind_params)
                    if not eqn.primitive.multiple_results:
                        outvals = [outvals]

            elif prim_name == "cond":
                # F1 fix: recurse into all branches (cond and switch both use the
                # same primitive; branches tuple has 2 or N ClosedJaxpr objects).
                # invals[0] is the int32 branch selector; invals[1:] are the shared
                # operands.  Re-emit through jax.lax.switch (handles N branches).
                branches = eqn.params["branches"]
                index = invals[0]  # int32 branch selector
                operands = invals[1:]  # shared operands for all branches

                def _make_branch(branch_jaxpr: "jax_core.ClosedJaxpr", branch_idx: int) -> Callable:
                    branch_path = f"{path}cond[{cf_index}]/b{branch_idx}/"
                    # Capture step via default arg so each branch closure holds
                    # the correct enclosing step at definition time.
                    _step = step

                    def branch_fn(*ops_tuple: Any) -> tuple:
                        return tuple(
                            _recurse(
                                branch_jaxpr.jaxpr,
                                branch_jaxpr.consts,
                                list(ops_tuple),
                                tap_cb,
                                ops,
                                branch_path,
                                _step,
                            )
                        )

                    return branch_fn

                instrumented_branches = [_make_branch(b, j) for j, b in enumerate(branches)]
                result = jax.lax.switch(index, instrumented_branches, *operands)
                outvals = list(result) if isinstance(result, (list, tuple)) else [result]

            elif prim_name == "remat2":
                # F1 fix: recurse inside remat2 sub-jaxpr, re-emit through
                # jax.checkpoint with original prevent_cse and policy preserved.
                # NOTE: params["jaxpr"] for remat2 is a bare Jaxpr (not ClosedJaxpr);
                # constvars are empty — all inputs come through eqn.invars.
                inner_jaxpr = eqn.params["jaxpr"]  # bare Jaxpr
                prevent_cse = eqn.params["prevent_cse"]
                policy = eqn.params["policy"]
                new_path = f"{path}remat[{cf_index}]/"
                # Capture step via default arg for safe closure semantics.
                _step = step

                def _inner_remat(
                    *flat_in: Any, _j: Any = inner_jaxpr, _p: str = new_path, _s: Any = _step
                ) -> tuple:
                    return tuple(_recurse(_j, [], list(flat_in), tap_cb, ops, _p, _s))

                result = jax.checkpoint(_inner_remat, prevent_cse=prevent_cse, policy=policy)(
                    *invals
                )
                outvals = list(result) if isinstance(result, (list, tuple)) else [result]

            elif prim_name in _JIT_PRIMS:
                # F2 fix: make jit boundary visible in path (jit[k]/ prefix).
                # The inner jaxpr is a ClosedJaxpr; recurse into it with the
                # extended path, then re-wrap in jax.jit to preserve the compile
                # boundary.
                # Step threading: step is passed as an explicit ARGUMENT to the
                # fresh jax.jit wrapper (not captured via closure) because jit
                # creates a new compilation context.  Closed-over abstract tracers
                # from an enclosing scan body trace are not forwarded across a
                # JIT boundary, causing incorrect constant-folding.  Passing step
                # explicitly makes it a proper JIT input that changes per iteration.
                # Limitation: the fresh jax.jit does not thread the original
                # donated_invars/in_shardings/out_shardings params (acceptable pre-v1).
                inner = eqn.params["jaxpr"]  # ClosedJaxpr
                new_path = f"{path}jit[{cf_index}]/"

                def _inner_jit(*flat_in_and_step: Any, _j: Any = inner, _p: str = new_path) -> Any:
                    # step is the last positional argument.
                    *flat_in, step_in = flat_in_and_step
                    return _recurse(_j.jaxpr, _j.consts, list(flat_in), tap_cb, ops, _p, step_in)

                result = jax.jit(_inner_jit)(*invals, step)
                outvals = list(result) if isinstance(result, (list, tuple)) else [result]

            else:
                # Higher-order primitive not handled (e.g. scan/while not in ops):
                # bind opaquely — boundary counter already advanced above.
                bind_params = eqn.primitive.get_bind_params(eqn.params)
                outvals = eqn.primitive.bind(*invals, **bind_params)
                if not eqn.primitive.multiple_results:
                    outvals = [outvals]

        elif prim_name in _AD_PRIMS:
            # Bind opaquely — do NOT recurse (see module docstring).
            # MUST use get_bind_params: custom_jvp_call/custom_vjp_call carry
            # call_jaxpr + jvp_jaxpr_fun in params; get_bind_params converts
            # them to the subfuns= kwarg that bind expects.  Direct **eqn.params
            # raises KeyError: 'subfuns'.  This pattern also preserves the custom
            # JVP/VJP rule so jax.grad(verbose(f)) differentiates correctly.
            bind_params = eqn.primitive.get_bind_params(eqn.params)
            outvals = eqn.primitive.bind(*invals, **bind_params)
            if not eqn.primitive.multiple_results:
                outvals = [outvals]

        else:
            # Use the eval_jaxpr pattern for all remaining primitives.
            bind_params = eqn.primitive.get_bind_params(eqn.params)
            outvals = eqn.primitive.bind(*invals, **bind_params)
            if not eqn.primitive.multiple_results:
                outvals = [outvals]

            # Primitive tap: check if any spec matches this primitive name.
            # Fires on EVERY matched non-boundary, non-AD primitive.
            if prim_taps and prim_tap_fn is not None:
                matching = [spec for spec in prim_taps if spec.prim_name == prim_name]
                if matching:
                    j = n_prim_tap.get(prim_name, 0)
                    n_prim_tap[prim_name] = j + 1
                    tap_path = f"{path}{prim_name}[{j}]"
                    for spec in matching:
                        prim_tap_fn(tap_path, step, tuple(outvals), spec)

        for v, val in zip(eqn.outvars, outvals):
            env[v] = val

    return [_read(env, v) for v in jaxpr.outvars]
