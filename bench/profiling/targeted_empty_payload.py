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
Targeted bench: empty-payload select=() at se=1 vs full-carry at se=1.

Hypothesis under test: does select=lambda _: () save measurable µs/step
at se=1 compared to the default (full carry shipped)?

At se=1 every step fires a callback.  With select=():
  - device-side: carry is NOT shipped across host boundary (0 bytes)
  - host-side:   _host still runs: step_.item() + TapEvent(value=()) + _guard
So the only saving is the data-transfer cost (carry bytes → host buffer).

Rerun:
    cd /home/jp/arcueil/jax-tap-perf
    uv run python bench/profiling/targeted_empty_payload.py
"""

from __future__ import annotations

import statistics
import sys
import time

import jax
import jax.lax as lax
import jax.numpy as jnp

import jaxtap as tap

sys.path.insert(0, "/home/jp/arcueil/jax-tap-perf/src")

N = 10_000
DIM = 8
K = 9
SEED = 42
xs = jax.random.normal(jax.random.PRNGKey(SEED), (N, DIM))
init_carry = jnp.zeros(DIM)


def body_bare(carry, x):
    return carry * 1.01 + jnp.sin(x), None


def f_inner(c):
    return lax.scan(body_bare, c, xs)[0]


def noop(event):
    pass


def time_fn(f, init, label, baseline=None):
    jax.block_until_ready(f(init))
    times = []
    for _ in range(K):
        t0 = time.perf_counter()
        jax.block_until_ready(f(init))
        times.append(time.perf_counter() - t0)
    med = statistics.median(times) / N * 1e6
    mn = min(times) / N * 1e6
    delta = f"  (+{med - baseline:.3f} µs vs baseline)" if baseline is not None else ""
    print(f"  {label:<55}: {med:.3f} µs/step (min={mn:.3f}){delta}")
    return med


# --- manual-payload (same payload as verbose default): step int32 + DIM floats ---
init_mp = (jnp.zeros(DIM), jnp.int32(0))


def body_mp(carry, x):
    c, i = carry
    c2 = c * 1.01 + jnp.sin(x)
    jax.debug.callback(lambda i_, v_: None, i, c2, ordered=False)
    return (c2, i + 1), None


def f_mp(carry):
    (c, _), _ = lax.scan(body_mp, carry, xs)
    return c


print(f"N={N:,}, DIM={DIM}, K={K}\n")
mp = time_fn(jax.jit(f_mp), init_mp, "manual-payload (step+carry, noop host)")


# --- manual-progress (step only, no carry): the absolute data-transfer floor ---
def body_mstep(carry_step, x):
    carry, step = carry_step
    carry2 = carry * 1.01 + jnp.sin(x)
    jax.debug.callback(lambda s_: None, step, ordered=False)
    return (carry2, step + 1), None


def f_mstep(init):
    (c, _), _ = lax.scan(body_mstep, init, xs)
    return c


init_mstep = (jnp.zeros(DIM), jnp.int32(0))
mstep = time_fn(jax.jit(f_mstep), init_mstep, "manual-progress  (step only, noop host)")

# --- verbose(se=1) with full carry (default) ---
f_full = jax.jit(tap.verbose(f_inner, on_step=noop, sample_every=1))
full = time_fn(f_full, init_carry, "verbose(se=1, default)           [full carry]", mp)

# --- verbose(se=1) with select=lambda _: () — empty payload ---
f_empty = jax.jit(
    tap.verbose(f_inner, on_step=noop, sample_every=1, select=lambda _: ())
)
empty = time_fn(
    f_empty, init_carry, "verbose(se=1, select=lambda _: ()) [no carry]", mp
)

print()
print("=" * 70)
print("FINDINGS (se=1 empty-payload hypothesis):")
print(
    f"  manual-payload vs manual-progress delta: {mp - mstep:.3f} µs  (carry transfer cost)"
)
print(
    f"  verbose full-carry machinery:            {full - mp:.3f} µs above manual-payload"
)
print(
    f"  verbose empty-payload machinery:         {empty - mstep:.3f} µs above manual-progress"
)
print(f"  empty-payload saving vs full-carry:      {full - empty:.3f} µs at se=1")

# ---------------------------------------------------------------------------
# Skip-unflatten fast-path: measure the SPECIFIC hypothesis.
#
# The hypothesis: add a code branch `if not flat_vals: value = ()` in the
# select-branch _host closure, skipping tree_unflatten(empty_tree, []).
# This is the ACTUAL code fast-path, distinct from the data-transfer saving.
# ---------------------------------------------------------------------------
print()
print("--- Skip-unflatten fast-path hypothesis (the specific code change) ---")
_K = 9
_N_CALLS = 100_000
_empty_tree = jax.tree_util.tree_structure(())
_empty_flat: list = []

# Cost of tree_unflatten with empty pytree
_times_unflatten = []
for _ in range(_K):
    _t0 = time.perf_counter()
    for _ in range(_N_CALLS):
        _ = jax.tree_util.tree_unflatten(_empty_tree, _empty_flat)
    _times_unflatten.append(time.perf_counter() - _t0)
cost_unflatten = statistics.median(_times_unflatten) / _N_CALLS * 1e6

# Cost of direct value = () assignment (the fast path)
_times_skip = []
for _ in range(_K):
    _t0 = time.perf_counter()
    for _ in range(_N_CALLS):
        _ = ()
    _times_skip.append(time.perf_counter() - _t0)
cost_skip = statistics.median(_times_skip) / _N_CALLS * 1e6

skip_saving = cost_unflatten - cost_skip
print(f"  tree_unflatten(empty_tree, []):  {cost_unflatten:.4f} µs")
print(f"  value = () fast path:            {cost_skip:.4f} µs")
print(f"  skip-unflatten saving:           {skip_saving:.4f} µs")
print(
    f"  [MEASURED — {skip_saving:.2f} µs saving — "
    f"{'REJECTED' if skip_saving < 1.0 else 'CANDIDATE'} (gate: 1 µs)]"
)
