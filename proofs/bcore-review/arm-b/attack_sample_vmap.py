"""ATTACK: the M2 `sample_every` gate wraps tap_cb in jax.lax.cond
(step % k == 0 ? fire : None). Under vmap, lax.cond with a batched predicate is
lowered to `select` and BOTH branches execute -> the debug.callback effect in the
'fire' branch may run for EVERY step regardless of the gate. Test whether
sample_every actually reduces events (a) without vmap and (b) under vmap."""
import jax
import jax.numpy as jnp
import numpy as np
import jaxtap as tap


def _bytes(x):
    return [np.asarray(v).tobytes() for v in jax.tree_util.tree_leaves(x)]


def bitwise_eq(a, b):
    return _bytes(a) == _bytes(b)


N = 8
xs = jnp.arange(float(N), dtype=jnp.float32)

def scan_f(c, xs_):
    return jax.lax.scan(lambda a, x: (a + x, a), c, xs_)

print("=== sample_every WITHOUT vmap (k=4, N=8) ===")
ev = []
got = tap.verbose(scan_f, on_step=lambda e: ev.append(e), sample_every=4)(jnp.float32(0.0), xs)
jax.block_until_ready(got)
print("steps fired:", sorted(e.step for e in ev), " <-- EXPECTED [0, 4]")
print("count:", len(ev), "(expected 2)")

print("\n=== sample_every UNDER vmap (k=4, N=8, LANES=3) ===")
LANES = 3
carry_b = jnp.zeros(LANES, dtype=jnp.float32)
xs_b = jnp.tile(xs, (LANES, 1))
ref = jax.vmap(scan_f)(carry_b, xs_b)
ev2 = []
try:
    got2 = jax.vmap(tap.verbose(scan_f, on_step=lambda e: ev2.append(e), sample_every=4))(carry_b, xs_b)
    jax.block_until_ready(got2)
    print("bitwise identical:", bitwise_eq(ref, got2))
    from collections import Counter
    step_counts = Counter(e.step for e in ev2)
    print("events total:", len(ev2), " <-- EXPECTED 2*LANES =", 2 * LANES, "if gate holds")
    print("steps fired (with multiplicity):", dict(sorted(step_counts.items())))
    fired_steps = sorted(set(e.step for e in ev2))
    print("distinct steps fired:", fired_steps, " <-- EXPECTED [0, 4]")
    if set(fired_steps) - {0, 4}:
        print("VERDICT: sample_every GATE DEFEATED under vmap -- fired on non-multiples of k")
    elif len(ev2) != 2 * LANES:
        print(f"VERDICT: event count off ({len(ev2)} != {2*LANES})")
    else:
        print("VERDICT: gate holds under vmap")
except Exception as exc:
    import traceback
    traceback.print_exc()
    print("VERDICT: CRASH under vmap:", type(exc).__name__, str(exc)[:160])
