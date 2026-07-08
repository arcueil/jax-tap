"""AYS on arm-A A1: is the vmap-while over-fire jaxtap-specific or inherent?

Baseline: a hand-written vmapped while_loop with a RAW jax.debug.callback in the
body — NO jaxtap. If this ALSO fires 30x with ghost values, A1 is an inherent
property of vmap+while_loop+debug.callback (documented boundary), not a jaxtap
defect. If the raw baseline fires 16x cleanly, jaxtap is making it worse.
"""
import numpy as np
import jax
import jax.numpy as jnp

LIM = jnp.float32(10.0)
seen = []

def cb(counter, acc):
    seen.append((float(counter), float(acc)))

def f(v0, acc0):
    def cond(state):
        c, a = state
        return c < LIM
    def body(state):
        c, a = state
        jax.debug.callback(cb, c + 1.0, a + (c + 1.0), ordered=False)  # tap the would-be new carry
        return (c + 1.0, a + (c + 1.0))
    return jax.lax.while_loop(cond, body, (v0, acc0))

v0 = jnp.array([0.0, 5.0, 9.0], dtype=jnp.float32)   # per-lane trips: 10, 5, 1 = 16
acc0 = jnp.zeros(3, dtype=jnp.float32)

out = jax.vmap(f)(v0, acc0)
jax.block_until_ready(out)

counters = sorted({c for c, _ in seen})
print("RAW baseline (no jaxtap):")
print("  event count:", len(seen), "(expected clean per-lane = 10+5+1 = 16)")
print("  distinct counters delivered:", counters)
print("  fabricated counters > LIM(10):", [c for c in counters if c > 10.0])
print("  VERDICT:", "INHERENT (raw baseline also over-fires + ghosts)"
      if len(seen) != 16 or any(c > 10.0 for c in counters)
      else "JAXTAP-SPECIFIC (raw baseline clean)")
