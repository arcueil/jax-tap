"""AYS round 2 on arm-A A2/A3 (remat) + the A1 mitigation question.

A2 claim: remat2 primitive is unhandled -> walker binds opaquely -> 0 events
  inside jax.checkpoint. Verify remat2 carries a recursable jaxpr param (=> a
  cheap fix: add remat2 to the recursion set, like jit).
A1 mitigation: inside a vmapped while body, is the per-lane cond predicate
  computable so jaxtap COULD gate the callback to suppress ghosts (something the
  raw baseline cannot do)? Probe whether the active-mask is recoverable.
"""
import jax
import jax.numpy as jnp

# ---- A2: what primitive does jax.checkpoint produce, and is it recursable? ----
def inner(x):
    c, _ = jax.lax.scan(lambda a, b: (a + b, a), x, jnp.arange(4.0))
    return c

def f(x):
    return jax.checkpoint(inner)(x)

closed = jax.make_jaxpr(f)(jnp.float32(1.0))
top = closed.jaxpr.eqns
print("A2: top-level eqns:", [e.primitive.name for e in top])
for e in top:
    if "remat" in e.primitive.name or "checkpoint" in e.primitive.name:
        print(f"  remat prim = '{e.primitive.name}', params keys = {sorted(e.params)}")
        # does it carry a jaxpr we can recurse into (like jit)?
        for key in ("jaxpr", "call_jaxpr"):
            if key in e.params:
                sub = e.params[key]
                subj = sub.jaxpr if hasattr(sub, "jaxpr") else sub
                print(f"    -> params['{key}'] inner eqns = {[ie.primitive.name for ie in subj.eqns]}")
                print("    => RECURSABLE: adding remat2 to the recursion set would reach the inner scan")

# ---- A1 mitigation probe: is the cond predicate per-lane available in the body? ----
# In jaxtap's rewrite_while, body_fn has access to `carry`; the cond_jaxpr can be
# evaluated on that carry to get the per-lane active flag. Confirm eval works and
# yields the right mask for a batched carry under vmap.
LIM = jnp.float32(10.0)

def cond_pred(c):
    return c < LIM

def check(v0):
    # emulate: inside the (batched) body, evaluate cond on the current carry
    return cond_pred(v0)

vprobe = jax.vmap(check)(jnp.array([9.0, 10.0, 11.0], dtype=jnp.float32))
print("\nA1 mitigation: cond predicate evaluable per-lane on carry ->", vprobe,
      "(True=active). jaxtap COULD gate the callback on this to drop ghosts;",
      "raw baseline cannot. => v1.x mitigation candidate, not a v1 blocker.")
