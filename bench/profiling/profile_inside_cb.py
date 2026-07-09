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
Decompose where the per-event host-callback overhead actually goes.

Measures each step of the host-side _host() closure independently using
jax.debug.callback arms (A–H).  Establishes the per-component µs cost
that motivated OPT 1 (int → .item()) and OPT 2 (router fast-path).

Key findings (N=10,000, DIM=8, K=9):
  - ARM A (noop, same structure as verbose): ≈ 0 µs above manual-payload
    → callback dispatch structure is NOT the cost; ALL overhead is in _host body
  - ARM C (int(step_)):  ~14 µs above noop → int() through JAX __int__ is expensive
  - ARM B (step_.item()): ~1.2 µs above noop → .item() is near-minimal
  - ARM D (item + TapEvent): ~3 µs above noop
  - ARM F (item + TapEvent + _guard): ~4.5 µs above noop → irreducible floor
  - ARM H (actual tap.verbose): must match ARM F closely (validation)

Rerun:
    cd /home/jp/arcueil/jax-tap-perf
    uv run python bench/profiling/profile_inside_cb.py
"""

from __future__ import annotations

import statistics
import time

import jax
import jax.lax as lax
import jax.numpy as jnp

import jaxtap as tap
from jaxtap import TapEvent, _guard
from jaxtap._walker import interpret

N = 10_000
DIM = 8
K = 9
SEED = 42
xs = jax.random.normal(jax.random.PRNGKey(SEED), (N, DIM))
init_carry = jnp.zeros(DIM)
internal_ops = frozenset(["scan"])


def body_bare(carry, x):
    return carry * 1.01 + jnp.sin(x), None


def f_inner(c):
    return lax.scan(body_bare, c, xs)[0]


def noop_on_step(event):
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
    vs = f"  [+{med - baseline:.2f} µs vs baseline]" if baseline is not None else ""
    print(f"  {label:<45}: {med:.3f} µs/step (min={mn:.3f}){vs}")
    return med


def make_verbose_with_cb(host_fn):
    """Build a verbose-like function that fires jax.debug.callback(host_fn, ...)."""

    def _base_cb(path_, step, *carry_leaves, total=None):
        jax.debug.callback(host_fn, step, *carry_leaves, ordered=False)

    def wrapped(*args):
        return interpret(f_inner, args, _base_cb, internal_ops)

    return jax.jit(wrapped)


# BASELINE: manual-payload (step int32 + DIM floats, noop host)
init_mp = (jnp.zeros(DIM), jnp.int32(0))


def body_mp(carry, x):
    c, i = carry
    c2 = c * 1.01 + jnp.sin(x)
    jax.debug.callback(lambda i_, v_: None, i, c2, ordered=False)
    return (c2, i + 1), None


def f_mp(carry):
    (c, _), _ = lax.scan(body_mp, carry, xs)
    return c


mp = time_fn(jax.jit(f_mp), init_mp, "BASELINE: manual-payload (noop lambda)")
print()

# ARM A: noop (confirming structural overhead ≈ 0)
a_noop = time_fn(
    make_verbose_with_cb(lambda s, *leaves_: None),
    init_carry,
    "ARM A: noop lambda (same struct as verbose)",
    mp,
)


# ARM B: step_.item() only
def b_fn(step_, *leaves):
    _ = step_.item()


a_item = time_fn(make_verbose_with_cb(b_fn), init_carry, "ARM B: step_.item() only", mp)


# ARM C: int(step_) only
def c_fn(step_, *leaves):
    _ = int(step_)


a_int = time_fn(make_verbose_with_cb(c_fn), init_carry, "ARM C: int(step_) only", mp)


# ARM D: step_.item() + TapEvent construction (no _guard, no on_step call)
def d_fn(step_, *leaves):
    _ = TapEvent(path="scan[0]", step=step_.item(), value=leaves, total=N)


a_item_tap = time_fn(
    make_verbose_with_cb(d_fn),
    init_carry,
    "ARM D: step_.item() + TapEvent (no _guard)",
    mp,
)


# ARM E: int(step_) + TapEvent construction (no _guard, no on_step call)
def e_fn(step_, *leaves):
    _ = TapEvent(path="scan[0]", step=int(step_), value=leaves, total=N)


a_int_tap = time_fn(
    make_verbose_with_cb(e_fn),
    init_carry,
    "ARM E: int(step_) + TapEvent (no _guard)",
    mp,
)


# ARM F: step_.item() + TapEvent + _guard(noop) — what verbose does after OPT 1
def f_fn(step_, *leaves):
    _guard(
        noop_on_step, TapEvent(path="scan[0]", step=step_.item(), value=leaves, total=N)
    )


a_item_full = time_fn(
    make_verbose_with_cb(f_fn),
    init_carry,
    "ARM F: step_.item() + TapEvent + _guard",
    mp,
)


# ARM G: int(step_) + TapEvent + _guard(noop) — what verbose did BEFORE OPT 1
def g_fn(step_, *leaves):
    _guard(
        noop_on_step, TapEvent(path="scan[0]", step=int(step_), value=leaves, total=N)
    )


a_int_full = time_fn(
    make_verbose_with_cb(g_fn),
    init_carry,
    "ARM G: int(step_) + TapEvent + _guard (PRE-OPT 1)",
    mp,
)

# ARM H: actual verbose (benchmark validation — should match ARM F closely)
f_v = jax.jit(tap.verbose(f_inner, on_step=noop_on_step, sample_every=1))
actual_v = time_fn(f_v, init_carry, "ARM H: actual tap.verbose (reference)", mp)

print()
print("=" * 75)
print("BREAKDOWN SUMMARY (vs manual-payload baseline):")
print(
    f"  ARM A noop structural overhead:     {a_noop - mp:.2f} µs  (dispatch struct, not host body)"
)
print(f"  ARM B step_.item() cost:            {a_item - mp:.2f} µs")
print(f"  ARM C int(step_) cost:              {a_int - mp:.2f} µs")
print(f"  OPT 1 saving (C - B):               {a_int - a_item:.2f} µs")
print(f"  ARM D TapEvent cost (with item()):  {a_item_tap - a_item:.2f} µs")
print(f"  ARM F _guard cost:                  {a_item_full - a_item_tap:.2f} µs")
print(
    f"  ARM F total machinery (item path):  {a_item_full - mp:.2f} µs  ← irreducible floor"
)
print(
    f"  ARM G total machinery (int path):   {a_int_full - mp:.2f} µs  ← PRE-OPT 1 cost"
)
print(f"  ARM H actual verbose:               {actual_v - mp:.2f} µs  ← should ≈ ARM F")
