# jax-tap sketch extension: while_loop rebuild with heartbeat tap.
# Subtlety under test: augmenting the carry with a step counter changes the
# carry structure that BOTH cond and body see -> wrap both to unpack.
# while params on 0.10.1: body_jaxpr, body_nconsts, cond_jaxpr, cond_nconsts
import jax
import jax.numpy as jnp
import numpy as np


def rewrite_while(eqn, invals, tap_cb, here):
    p = eqn.params
    cj, cn = p["cond_jaxpr"], p["cond_nconsts"]
    bj, bn = p["body_jaxpr"], p["body_nconsts"]
    cconsts, bconsts, init = invals[:cn], invals[cn:cn + bn], invals[cn + bn:]

    def cond_fn(carry_step):
        carry, _ = carry_step
        (pred,) = jax.core.eval_jaxpr(cj.jaxpr, cj.consts, *cconsts, *carry)
        return pred

    def body_fn(carry_step):
        carry, step = carry_step
        new_carry = jax.core.eval_jaxpr(bj.jaxpr, bj.consts, *bconsts, *carry)
        jax.debug.callback(tap_cb, here, step, new_carry, ordered=False)
        return (new_carry, step + 1)

    carry, _ = jax.lax.while_loop(cond_fn, body_fn, (init, jnp.int32(0)))
    return list(carry)


# ---- harness: closed-over consts in cond AND body, multi-leaf carry ----
LIM = jnp.float32(37.0)   # cond const
INC = jnp.float32(1.7)    # body const

def f(v0, k0):
    def cond(c):
        return c[0] < LIM
    def body(c):
        return (c[0] + INC, c[1] * 2)
    return jax.lax.while_loop(cond, body, (v0, k0))

v0, k0 = jnp.float32(0.3), jnp.int32(1)
ref = f(v0, k0)

closed = jax.make_jaxpr(f)(v0, k0)
(weqn,) = [e for e in closed.jaxpr.eqns if e.primitive.name == "while"]
print("while eqn found; cond_nconsts:", weqn.params["cond_nconsts"],
      "body_nconsts:", weqn.params["body_nconsts"])

events = []
def cb(pth, step, carry):
    events.append((pth, int(step)))

# interpret the top-level jaxpr minimally: literals/consts + the while eqn
env = {}
for v, val in zip(closed.jaxpr.constvars, closed.consts):
    env[v] = val
for v, val in zip(closed.jaxpr.invars, [v0, k0]):
    env[v] = val
def read(a):
    return a.val if type(a).__name__ == "Literal" else env[a]
for eqn in closed.jaxpr.eqns:
    invals = [read(a) for a in eqn.invars]
    if eqn.primitive.name == "while":
        outvals = rewrite_while(eqn, invals, cb, "while[0]")
    else:
        outvals = eqn.primitive.bind(*invals, **eqn.params)
        if not eqn.primitive.multiple_results:
            outvals = [outvals]
    for v, val in zip(eqn.outvars, outvals):
        env[v] = val
got = tuple(read(v) for v in closed.jaxpr.outvars)
jax.block_until_ready(got)

ok = all(np.asarray(r).tobytes() == np.asarray(g).tobytes() for r, g in zip(ref, got))
steps = [s for _, s in events]
print("bitwise identical:", ok)
print("heartbeat events:", len(events), "steps:", steps[:5], "...")
print("expected iters:", int(np.ceil((37.0 - 0.3) / 1.7)))
print("ALL CHECKS PASSED" if ok and steps == list(range(len(steps))) and len(steps) > 0
      else "FAILURES ABOVE")
