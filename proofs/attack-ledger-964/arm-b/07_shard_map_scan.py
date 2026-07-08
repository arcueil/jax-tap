"""Attack #3: jax.debug.callback inside shard_map, wrapped by
progress_bar()'s scan-patching. 2-device CPU repro. This composition is
untested in the PR (blackjax's own eca.py/LAPS runs scans inside
shard_map). Check for: crash, per-device duplicated callback fires
(inflated apparent step count), or silent no-instrumentation.
"""
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P
from jax.sharding import Mesh

import blackjax

print("device count:", jax.device_count())
assert jax.device_count() == 2, "need 2 CPU devices for this repro"

mesh = Mesh(jax.devices(), axis_names=("i",))

N_STEPS = 50

def per_device_scan(x0, xs):
    # x0: local carry, shape () per device.
    # xs: local xs, shape (N_STEPS,) per device -- scan axis (0) is NOT the
    # sharded axis (that's axis 1 of the un-sharded array).
    def body(carry, x):
        return carry + x, carry
    final, ys = jax.lax.scan(body, x0, xs)
    return final, ys

sharded_fn = shard_map(
    per_device_scan,
    mesh=mesh,
    in_specs=(P("i"), P(None, "i")),
    out_specs=(P("i"), P(None, "i")),
)

x0 = jnp.zeros((2,))
xs = jnp.ones((N_STEPS, 2))  # per-device local: (N_STEPS,) after sharding axis 1

events = []
with blackjax.progress_bar(label="shard_map") as state:
    orig_cb = state._step_callback
    def cb(idx):
        events.append(int(idx))
        orig_cb(idx)
    state._step_callback = cb

    try:
        final, ys = jax.jit(sharded_fn)(x0, xs)
        jax.block_until_ready(final)
        print("shard_map+scan completed without crash")
        print("final:", final)
    except Exception as e:
        print("CRASH:", type(e).__name__, e)

print("total callback fires observed:", len(events))
print("expected if instrumented once per logical step (not doubled):", N_STEPS)
print("expected if fired once PER DEVICE per step (doubled/duplicated):", 2 * N_STEPS)
print("state.n_steps recorded:", state.n_steps)
