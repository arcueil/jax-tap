"""ATTACK: vmap over while_loop with data-dependent, per-lane trip counts.
Design promise: bitwise-identical outputs + per-lane event counts as designed.
Hypothesis: the batched while runs max_trip_count joint iterations; the tap
fires every joint iteration for every lane, so early-finishing lanes get
EXTRA (stale) events -> over-counting + wrong values.
"""
import jax, jax.numpy as jnp
import numpy as np
import jaxtap as tap

# Each lane counts up from v0 to >= LIM by +1.0. Different v0 => different trip count.
LIM = jnp.float32(10.0)
def f(v0):
    return jax.lax.while_loop(lambda c: c < LIM, lambda c: c + 1.0, v0)

# 3 lanes with very different trip counts: 10, 5, 1 iterations
v0 = jnp.array([0.0, 5.0, 9.0], dtype=jnp.float32)

# baseline (untapped) vmap
ref = jax.vmap(f)(v0)

events = []
got = jax.vmap(tap.verbose(f, on_step=lambda e: events.append(e)))(v0)
jax.block_until_ready(got)

def _bytes(x):
    return [np.asarray(v).tobytes() for v in jax.tree_util.tree_leaves(x)]
bitwise = _bytes(ref) == _bytes(got)
print("ref             :", np.asarray(ref))
print("got (tapped)    :", np.asarray(got))
print("BITWISE IDENTICAL:", bitwise)

# Expected per-lane trip counts (float32 arithmetic)
expected_counts = []
for x in np.asarray(v0):
    c = np.float32(x); n = 0
    while c < np.float32(LIM):
        c = c + np.float32(1.0); n += 1
    expected_counts.append(n)
total_expected = sum(expected_counts)
print("\nExpected per-lane trip counts:", expected_counts, "-> total events SHOULD be", total_expected)

while_events = [e for e in events if e.path == "while[0]"]
print("ACTUAL total while[0] events:", len(while_events))

# Histogram of step indices seen
from collections import Counter
step_hist = Counter(e.step for e in while_events)
print("Step-index histogram (step -> count):", dict(sorted(step_hist.items())))
max_step = max(step_hist) if step_hist else -1
print("Max step index observed:", max_step, "(max trip count =", max(expected_counts), ")")

# Look at the VALUES delivered per step: are there stale/duplicated carry values?
print("\nPer-event (step, value) sample:")
for e in sorted(while_events, key=lambda e:(e.step,)):
    v = np.asarray(e.value[0]) if isinstance(e.value, tuple) else np.asarray(e.value)
    print(f"  step={e.step}  value={v}")

print("\nVERDICT:",
      "OVER-COUNT" if len(while_events) != total_expected else "counts-match",
      "| bitwise", "OK" if bitwise else "BROKEN")
