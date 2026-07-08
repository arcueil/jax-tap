"""SCOPING PROTOTYPE — can jax-tap tap a body-LOCAL (L = cholesky(M)) directly,
by primitive KIND, with the scan body UNMODIFIED, and stream it in real time?

This is NOT the shipped jaxtap API — it's a ~55-line mini-walker to prove the
'just define L' idea before we design M1's tap-class surface. It:
  - recurses into the scan body (rebuilding the scan),
  - when it hits the `cholesky` primitive eqn, binds it normally to get L,
    then fires jax.debug.callback on L (real-time, mid-loop),
  - so the user's body just writes `L = jnp.linalg.cholesky(M)` and never
    mentions logging.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from jax.extend import core as jax_core


def primitive_tap(f, prim_name, sel, on_event):
    """Return g(*args) == f(*args) that fires on_event(step, sel(out)) every time
    `prim_name` executes inside a (possibly nested) scan body."""

    def g(*args):
        closed = jax.make_jaxpr(f)(*args)
        flat = jax.tree_util.tree_leaves(args)
        out = _interp(closed.jaxpr, closed.consts, flat, step_ref=[jnp.int32(0)])
        return out[0] if len(out) == 1 else out

    def _read(env, a):
        return a.val if isinstance(a, jax_core.Literal) else env[a]

    def _interp(jaxpr, consts, args, step_ref):
        env = {}
        for v, val in zip(jaxpr.constvars, consts):
            env[v] = val
        for v, val in zip(jaxpr.invars, args):
            env[v] = val
        for eqn in jaxpr.eqns:
            invals = [_read(env, a) for a in eqn.invars]
            name = eqn.primitive.name
            if name == "scan":
                outvals = _rewrite_scan(eqn, invals, step_ref)
            elif name in ("jit", "pjit", "closed_call"):
                # jnp.linalg.cholesky etc. wrap themselves in jit -> recurse
                inner = eqn.params["jaxpr"]
                outvals = _interp(inner.jaxpr, inner.consts, invals, step_ref)
            else:
                bp = eqn.primitive.get_bind_params(eqn.params)
                outvals = eqn.primitive.bind(*invals, **bp)
                outvals = outvals if eqn.primitive.multiple_results else [outvals]
                if name == prim_name:  # <-- PRIMITIVE TAP: fire on the output
                    jax.debug.callback(on_event, step_ref[0], sel(outvals[0]),
                                       ordered=False)
            for v, val in zip(eqn.outvars, outvals):
                env[v] = val
        return [_read(env, v) for v in jaxpr.outvars]

    def _rewrite_scan(eqn, invals, step_ref):
        p = eqn.params
        body, nc, ncar = p["jaxpr"], p["num_consts"], p["num_carry"]
        consts, init, xs = invals[:nc], invals[nc:nc+ncar], invals[nc+ncar:]
        rest = {k: v for k, v in p.items()
                if k not in ("jaxpr", "num_consts", "num_carry")}

        def body_fn(carry_step, x):
            carry, step = carry_step
            step_ref[0] = step  # expose the live step to the primitive tap
            xf = list(x) if isinstance(x, (list, tuple)) else ([] if x is None else [x])
            outs = _interp(body.jaxpr, body.consts, [*consts, *carry, *xf], step_ref)
            return (outs[:ncar], step + 1), outs[ncar:]

        (carry, _), ys = jax.lax.scan(body_fn, (init, jnp.int32(0)), xs, **rest)
        return [*carry, *ys]

    return g


# ------- demo-01 body, SIMPLIFIED: it just defines L. No logging code at all. -------
def sampler(log_step0):
    def step(carry, _):
        log_step, k = carry
        c = 1.0 - 10.0 ** (-k)
        M = jnp.array([[1.0, c], [c, 1.0]], dtype=c.dtype)
        L = jnp.linalg.cholesky(M)                     # <-- the only line that matters
        logdens = -0.5 * 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        finite = jnp.isfinite(logdens)
        return (jnp.where(finite, log_step + 0.05, log_step - 1.0), k + 1.0), logdens
    (log_step, _), _ = jax.lax.scan(step, (log_step0, 1.0), None, length=25)
    return log_step


first_bad = [None]
def announce(step, ok):
    # Real-time: this prints DURING the scan, before it finishes.
    if not bool(ok) and first_bad[0] is None:
        first_bad[0] = int(step)
        print(f"    [live] cholesky output went NON-FINITE at scan step {int(step)}")

g = primitive_tap(sampler, "cholesky", lambda L: jnp.all(jnp.isfinite(L)), announce)
print("running instrumented sampler (body was NEVER modified)...")
jax.block_until_ready(g(jnp.float32(0.0)))
print(f"RESULT: primitive tap on 'cholesky' caught first-bad step = {first_bad[0]} "
      f"[{'PASS' if first_bad[0] is not None else 'FAIL'}]")
