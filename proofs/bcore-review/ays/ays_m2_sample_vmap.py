"""AYS on M2: does sample_every=k actually throttle UNDER vmap?

swe-m2 honestly flagged: under vmap, `lax.cond(step % k == 0, fire, noop)` has a
BATCHED predicate -> JAX runs BOTH branches (select semantics). If the fire
branch's jax.debug.callback executes unconditionally under that, sample_every
is silently defeated under vmap (fires every step, not every k-th).

Compare event counts: unbatched (should throttle) vs vmapped (does it?).
"""
import jax
import jax.numpy as jnp
import jaxtap as tap

N = 12
K = 3
LANES = 4

def f(x0, xs):
    return jax.lax.scan(lambda c, x: (c + x, c), x0, xs)

# --- unbatched baseline: sample_every=K should fire on steps 0,3,6,9 = 4 events ---
ev_unbatched = []
g = tap.verbose(f, on_step=ev_unbatched.append, sample_every=K)
x0 = jnp.float32(0.0)
xs = jnp.arange(float(N), dtype=jnp.float32)
jax.block_until_ready(g(x0, xs))
steps_unbatched = sorted({e.step for e in ev_unbatched})
print(f"UNBATCHED sample_every={K}: {len(ev_unbatched)} events, steps={steps_unbatched}")
print(f"  expected: steps 0,3,6,9 -> 4 events. {'OK' if steps_unbatched == [0,3,6,9] else 'WRONG'}")

# --- vmapped: does the throttle hold? ---
ev_vmap = []
gv = tap.verbose(f, on_step=ev_vmap.append, sample_every=K)
x0b = jnp.zeros(LANES, dtype=jnp.float32)
xsb = jnp.tile(jnp.arange(float(N), dtype=jnp.float32), (LANES, 1))
jax.block_until_ready(jax.vmap(gv)(x0b, xsb))
steps_vmap = sorted({e.step for e in ev_vmap})
# If throttle holds: 4 sampled steps * 4 lanes = 16 events, steps {0,3,6,9}.
# If defeated: 12 steps * 4 lanes = 48 events, steps {0..11}.
print(f"\nVMAP({LANES} lanes) sample_every={K}: {len(ev_vmap)} events, distinct steps={steps_vmap}")
if len(ev_vmap) == 4 * LANES and steps_vmap == [0, 3, 6, 9]:
    print("  => THROTTLE HOLDS under vmap (16 events, only sampled steps)")
elif len(ev_vmap) == N * LANES or steps_vmap == list(range(N)):
    print(f"  => THROTTLE DEFEATED under vmap: fired every step ({len(ev_vmap)} events, all {N} steps).")
    print("     sample_every is SILENTLY IGNORED under vmap -- MAJOR instrumentation-contract gap.")
else:
    print(f"  => UNEXPECTED: {len(ev_vmap)} events, steps {steps_vmap} -- investigate.")
