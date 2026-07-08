"""ATTACK: a scan/while nested inside jax.lax.cond (or lax.switch) branch is
SILENTLY not instrumented. cond lowers to the `cond` primitive, which the walker
routes to the generic else-branch (get_bind_params + bind) WITHOUT walking the
branch sub-jaxprs. Any scan inside a branch produces ZERO events, while a
top-level scan is tapped normally.

Symptom to prove: result bitwise-correct, but events for the in-cond scan are MISSING.
"""
import jax
import jax.numpy as jnp
import numpy as np
import jaxtap as tap


def _bytes(x):
    return [np.asarray(v).tobytes() for v in jax.tree_util.tree_leaves(x)]


def bitwise_eq(a, b):
    return _bytes(a) == _bytes(b)


# ---- Case 1: scan inside a cond branch ----
def f_cond(pred, x0, xs):
    def true_branch(c):
        # This scan should be tapped but is inside a cond -> NOT walked.
        out, _ = jax.lax.scan(lambda a, b: (a + b, a), c, xs)
        return out

    def false_branch(c):
        return c

    return jax.lax.cond(pred, true_branch, false_branch, x0)


pred = jnp.bool_(True)
x0 = jnp.float32(1.0)
xs = jnp.arange(5.0, dtype=jnp.float32)

ref = f_cond(pred, x0, xs)
events = []
got = tap.verbose(f_cond, on_step=lambda e: events.append(e))(pred, x0, xs)
jax.block_until_ready(got)

print("=== Case 1: scan inside lax.cond ===")
print("bitwise identical :", bitwise_eq(ref, got))
print("events emitted    :", len(events), "  <-- EXPECTED 5 (scan runs 5 steps)")
print("paths             :", sorted({e.path for e in events}))
print("VERDICT: dropped instrumentation" if len(events) == 0 else "instrumented OK")


# ---- Case 2: scan inside a lax.switch branch ----
def f_switch(i, x0, xs):
    def make(mult):
        def branch(c):
            out, _ = jax.lax.scan(lambda a, b: (a + b * mult, a), c, xs)
            return out
        return branch
    return jax.lax.switch(i, [make(1.0), make(2.0), make(3.0)], x0)


i = jnp.int32(1)
ref2 = f_switch(i, x0, xs)
events2 = []
got2 = tap.verbose(f_switch, on_step=lambda e: events2.append(e))(i, x0, xs)
jax.block_until_ready(got2)

print("\n=== Case 2: scan inside lax.switch ===")
print("bitwise identical :", bitwise_eq(ref2, got2))
print("events emitted    :", len(events2), "  <-- EXPECTED 5")
print("VERDICT: dropped instrumentation" if len(events2) == 0 else "instrumented OK")


# ---- Case 3: control -- the SAME scan at top level IS tapped (proves the
# gap is specifically the cond boundary, not the scan itself) ----
def f_plain(x0, xs):
    out, _ = jax.lax.scan(lambda a, b: (a + b, a), x0, xs)
    return out


events3 = []
got3 = tap.verbose(f_plain, on_step=lambda e: events3.append(e))(x0, xs)
jax.block_until_ready(got3)
print("\n=== Case 3 (control): same scan at top level ===")
print("events emitted    :", len(events3), "  <-- top-level scan IS tapped")
