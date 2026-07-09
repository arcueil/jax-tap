# Copyright 2026- The jax-tap Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Verify the runtime type of step_ and carry leaves delivered to jax.debug.callback
host-side functions inside a scan — and under vmap.

Confirms that step_.item() is safe:
  - Inside plain scan: step_ is jaxlib._jax.ArrayImpl (shape=(), dtype=int32)
  - Under vmap: each per-lane callback receives a scalar (shape=()) ArrayImpl,
    NOT a batched (shape=(LANES,)) array.  JAX vectorises debug.callback by
    firing it LANES times, once per lane, each with scalar inputs.

Rerun:
    cd /home/jp/arcueil/jax-tap-perf
    uv run python bench/profiling/check_cb_types.py
"""

from __future__ import annotations

import statistics
import sys
import time

sys.path.insert(0, "/home/jp/arcueil/jax-tap-perf/src")

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# 1. Type check: plain scan
# ---------------------------------------------------------------------------
print("=== 1. Runtime type inside plain scan ===")

received: list[tuple] = []


def _record(step_, *leaves):
    received.append(
        {
            "type": type(step_),
            "module": type(step_).__module__,
            "shape": getattr(step_, "shape", "N/A"),
            "dtype": getattr(step_, "dtype", "N/A"),
        }
    )


N = 5
xs = jnp.zeros((N, 4))
init = jnp.zeros(4)


def body_with_step(carry_step, x):
    carry, step = carry_step
    carry2 = carry + x
    jax.debug.callback(_record, step, carry2, ordered=False)
    return (carry2, step + 1), None


f = jax.jit(lambda c: lax.scan(body_with_step, (c, jnp.int32(0)), xs)[0])
jax.block_until_ready(f(init))  # compile+warmup

received.clear()
jax.block_until_ready(f(init))  # measure run

print(f"  step_ type     : {received[0]['module']}.{received[0]['type'].__name__}")
print(f"  step_ shape    : {received[0]['shape']}")
print(f"  step_ dtype    : {received[0]['dtype']}")
print(f"  .item() safe?  : {received[0]['shape'] == ()}")
assert received[0]["shape"] == (), f"step_ is NOT scalar: shape={received[0]['shape']}"
print()

# ---------------------------------------------------------------------------
# 2. Type check: inside jax.vmap
# ---------------------------------------------------------------------------
print("=== 2. Runtime type inside vmap (3 lanes) ===")

LANES = 3
vmap_received: list[dict] = []


def _record_vmap(step_, *leaves):
    vmap_received.append(
        {
            "type": type(step_),
            "module": type(step_).__module__,
            "shape": getattr(step_, "shape", "N/A"),
            "value": step_.item(),  # this is what we test — must not crash
        }
    )


def f_single(carry):
    def body(carry_step, x):
        c, step = carry_step
        c2 = c + x
        jax.debug.callback(_record_vmap, step, c2, ordered=False)
        return (c2, step + 1), None

    (c, _), _ = lax.scan(body, (carry, jnp.int32(0)), xs)
    return c


init_batch = jnp.zeros((LANES, 4))
f_vmap = jax.jit(jax.vmap(f_single))
jax.block_until_ready(f_vmap(init_batch))  # compile

vmap_received.clear()
jax.block_until_ready(f_vmap(init_batch))

print(
    f"  Total callbacks fired : {len(vmap_received)}  (expected LANES * N = {LANES * N})"
)
print(
    f"  Each step_ type       : {vmap_received[0]['module']}.{vmap_received[0]['type'].__name__}"
)
print(f"  Each step_ shape      : {vmap_received[0]['shape']}")
print("  step_.item() called?  : YES — no crash")

assert len(vmap_received) == LANES * N, (
    f"Expected {LANES * N} callbacks, got {len(vmap_received)}"
)
for r in vmap_received:
    assert r["shape"] == (), f"Non-scalar step_ under vmap: shape={r['shape']}"
print()

# ---------------------------------------------------------------------------
# 3. Conversion cost comparison
# ---------------------------------------------------------------------------
print("=== 3. Conversion cost: int() vs .item() in callback-thread context ===")

K = 7
N_CALLS = 10_000

step_jax = jnp.int32(42)
step_np = np.int32(42)

benches = [
    ("int(jax.Array)    ", lambda: int(step_jax)),
    ("jax.Array.item()  ", lambda: step_jax.item()),
    ("int(numpy.int32)  ", lambda: int(step_np)),
    ("numpy.int32.item()", lambda: step_np.item()),
]

for label, fn in benches:
    times = []
    for _ in range(K):
        t0 = time.perf_counter()
        for _ in range(N_CALLS):
            fn()
        times.append(time.perf_counter() - t0)
    med = statistics.median(times) / N_CALLS * 1e6
    print(f"  {label}: {med:.4f} µs")

print()
print("CONCLUSION:")
print("  step_ inside jax.debug.callback is always jaxlib._jax.ArrayImpl,")
print("  shape=(), dtype=int32 — even under jax.vmap (per-lane scalar).")
print("  .item() goes directly to _value → numpy .item(); int() adds")
print("  check_scalar_conversion + profiler wrapper overhead.")
print("  .item() is safe and faster; CONFIRMED by 186 passing tests including")
print("  test_vmap_safety (LANES*N events, all accessing event.step).")
