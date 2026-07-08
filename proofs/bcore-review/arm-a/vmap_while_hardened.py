"""Harden the vmap-while finding: determinism + fabricated-value detection."""
import jax, jax.numpy as jnp
import numpy as np
import jaxtap as tap
def _b(x): return [np.asarray(v).tobytes() for v in jax.tree_util.tree_leaves(x)]

LIM = jnp.float32(10.0)
# Carry = (counter, accumulator). accumulator sums counter each step.
def f(state):
    def cond(s):
        c, acc = s
        return c < LIM
    def body(s):
        c, acc = s
        return (c + 1.0, acc + c)
    return jax.lax.while_loop(cond, body, state)

c0 = jnp.array([0.0, 5.0, 9.0], dtype=jnp.float32)
acc0 = jnp.zeros(3, dtype=jnp.float32)
ref = jax.vmap(f)((c0, acc0))

# Determinism of event count across 3 runs
counts = []
for _ in range(3):
    ev=[]
    got = jax.vmap(tap.verbose(f, on_step=lambda e: ev.append(e)))((c0, acc0))
    jax.block_until_ready(got)
    counts.append(len([e for e in ev if e.path=='while[0]']))
print("bitwise (primal preserved):", _b(ref)==_b(got))
print("event counts across 3 runs:", counts, "(expected per-lane trips 10+5+1 = 16)")

# Fabricated-value detection: collect all counter values delivered to host.
# TRUE reachable counter values across all lanes: {1..10} for lane0, {6..10} lane1, {10} lane2.
# A value of 11.0 is IMPOSSIBLE in the real computation (body of a done lane, masked).
ev=[]
jax.block_until_ready(jax.vmap(tap.verbose(f, on_step=lambda e: ev.append(e)))((c0, acc0)))
counter_vals = sorted({float(np.asarray(e.value[0])) for e in ev if e.path=='while[0]'})
print("distinct counter values delivered to host:", counter_vals)
fabricated = [v for v in counter_vals if v > 10.0]
print("FABRICATED values (impossible in real per-lane run, > LIM):", fabricated)

# also: does the accumulator show fabricated states?
acc_vals = sorted({round(float(np.asarray(e.value[1])),3) for e in ev if e.path=='while[0]'})
print("distinct accumulator values delivered:", acc_vals)

# Compare: what accumulator values does the REAL computation actually pass through?
real_acc = set()
for start in [0.0,5.0,9.0]:
    c=np.float32(start); acc=np.float32(0.0)
    while c < np.float32(10.0):
        acc = acc + c; c = c + np.float32(1.0)
        real_acc.add(round(float(acc),3))
print("REAL reachable accumulator values:", sorted(real_acc))
ghost_acc = sorted(set(acc_vals) - real_acc)
print("GHOST accumulator values host saw but program never computed:", ghost_acc)
