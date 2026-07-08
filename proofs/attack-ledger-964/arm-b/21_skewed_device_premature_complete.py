"""AYS (d) part 2: attempt an EMPIRICAL demonstration of 'the bar reaches
100% while a slower shard is still computing' -- previously only reasoned
from source (THEORETIC). Skew per-device work using axis_index-conditioned
extra compute on device 0 so device 1 finishes its 50 steps well before
device 0, and observe state.current_step / whether device 0's still-valid,
in-flight (smaller) step indices get silently dropped by the monotonic
clamp once device 1's fast finish has already pushed current_step to
n_steps-1.
"""
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

import time

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P
from jax.sharding import Mesh

import blackjax

mesh = Mesh(jax.devices(), axis_names=("i",))
N_STEPS = 60

def per_device_scan(x0, xs):
    dev = jax.lax.axis_index("i")

    def body(carry, x):
        # Device 0 does much heavier per-step work (many extra matmuls);
        # device 1 does almost none -- a large, deliberate timing skew.
        def heavy(c):
            m = jnp.ones((120, 120)) * (c + 1.0)
            for _ in range(15):
                m = jnp.tanh(m @ m) * 0.999
            return c + jnp.sum(m) * 1e-6
        def light(c):
            return c + x

        new_c = jax.lax.cond(dev == 0, heavy, light, carry)
        return new_c, new_c

    final, ys = jax.lax.scan(body, x0, xs)
    return final, ys

sharded_fn = shard_map(
    per_device_scan, mesh=mesh,
    in_specs=(P("i"), P(None, "i")), out_specs=(P("i"), P(None, "i")),
)

x0 = jnp.zeros((2,))
xs = jnp.ones((N_STEPS, 2))

timeline = []  # (wall_time, idx) as observed by the callback, in arrival order

with blackjax.progress_bar(label="skewed", print_rate=1) as state:
    orig_cb = state._step_callback
    t_start = time.monotonic()
    def cb(idx):
        timeline.append((time.monotonic() - t_start, int(idx)))
        orig_cb(idx)
    state._step_callback = cb

    jit_fn = jax.jit(sharded_fn)
    final, ys = jit_fn(x0, xs)
    jax.block_until_ready(final)

print("total callback fires:", len(timeline), "(expect 2 *", N_STEPS, "=", 2 * N_STEPS, ")")
print("final state.current_step:", state.current_step, "/ n_steps:", state.n_steps)

# Reconstruct: what's the arrival-order sequence of idx values, and does
# current_step (per the REAL monotonic-clamp logic in _step_callback) ever
# reach n_steps-1 while a smaller idx arrives afterward (evidence that a
# slower device's still-legitimate progress became invisible)?
max_seen = -1
first_reach_top_at = None
late_smaller_after_top = []
for t, idx in timeline:
    if idx > max_seen:
        max_seen = idx
    if max_seen == N_STEPS - 1 and first_reach_top_at is None:
        first_reach_top_at = t
    if first_reach_top_at is not None and t > first_reach_top_at and idx < N_STEPS - 1:
        late_smaller_after_top.append((t, idx))

print("wall-clock time bar first showed 100% (current_step==n_steps-1):",
      first_reach_top_at)
print("total wall-clock time until BOTH devices' callbacks fully drained:",
      timeline[-1][0] if timeline else None)
print("number of callback events AFTER the bar first hit 100% that carried "
      "a SMALLER, still-legitimate step index (silently invisible progress "
      "from the slower device):", len(late_smaller_after_top))
if late_smaller_after_top:
    print("  e.g. first few:", late_smaller_after_top[:5])
print()
print("CONCLUSION:",
      "EMPIRICALLY CONFIRMED -- bar showed 100% while the slow shard was still "
      "emitting legitimate (now-invisible) progress"
      if late_smaller_after_top else
      "NOT reproduced with this workload/skew -- keeping as THEORETIC")
