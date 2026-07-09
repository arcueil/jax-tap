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
bench/while_cond_overhead.py — A1 mitigation: cost of double cond evaluation.

The A1 cond-gating mitigation re-evaluates the while cond jaxpr INSIDE every
body iteration to obtain a per-lane active mask.  For a trivial cond like
``counter < N`` the re-evaluation is a single comparison — negligible.  For a
convergence-check cond that involves real floating-point work (e.g.
``jnp.linalg.norm(carry) > tol``) the extra evaluation doubles the cond work
per iteration and the overhead may be noticeable.

This script measures both cases and prints before/after numbers (run it once
before applying the mitigation patch and once after).

Both arms are designed to run EXACTLY N_ITERS iterations (the counter cond is
always the binding constraint) so per-iteration costs are directly comparable.

Arms
----
  bare-trivial      : plain jax.lax.while_loop, trivial cond (counter < N)
  tapped-trivial    : tap.verbose with trivial cond
  bare-expensive    : plain jax.lax.while_loop, expensive cond
                      (norm(carry[0]) > tol AND counter < N; norm is real work
                       but the counter is the binding stopper → always N iters)
  tapped-expensive  : tap.verbose with expensive cond

Usage
-----
  uv run python bench/while_cond_overhead.py           # N=2000, K=25
  uv run python bench/while_cond_overhead.py --smoke   # N=2000, K=7 (same iters, fewer repeats)
"""

from __future__ import annotations

import argparse
import statistics
import time

import jax
import jax.lax as lax
import jax.numpy as jnp

import jaxtap as tap

DIM = 16
N_ITERS = 2000  # fixed trip count for all arms (≥2000 for stable medians)
# norm threshold set far below any reachable value so the counter always binds.
# init_v is ones → norm ~= 4.0; tol = 0.01 is never reached in 200 iters with lr=0.001.
TOL = jnp.float32(0.01)


def noop_on_step(event: tap.TapEvent) -> None:
    pass


# ---------------------------------------------------------------------------
# Shared carries
# ---------------------------------------------------------------------------

INIT_TRIVIAL = (jnp.ones(DIM, dtype=jnp.float32), jnp.int32(0))
INIT_EXPENSIVE = (jnp.ones(DIM, dtype=jnp.float32), jnp.int32(0))


# ---------------------------------------------------------------------------
# While-loop bodies and conds (always terminates at N_ITERS)
# ---------------------------------------------------------------------------


def trivial_cond(carry):
    """Trivial cond: single int comparison."""
    _v, counter = carry
    return counter < jnp.array(N_ITERS, dtype=jnp.int32)


def trivial_body(carry):
    v, counter = carry
    return (v * jnp.float32(1.001) + jnp.sin(v) * jnp.float32(0.001), counter + 1)


def expensive_cond(carry):
    """Expensive cond: norm computation PLUS counter check.

    The counter is the binding stopper (tol never reached), so the loop always
    runs exactly N_ITERS.  The norm() call adds real floating-point work to
    the cond path — representing a real convergence check.
    """
    v, counter = carry
    norm_ok = jnp.linalg.norm(v) > TOL  # O(DIM) work; always True in this setup
    count_ok = counter < jnp.array(N_ITERS, dtype=jnp.int32)
    return norm_ok & count_ok


def expensive_body(carry):
    v, counter = carry
    v2 = v * jnp.float32(1.001) + jnp.sin(v) * jnp.float32(0.001)
    return (v2, counter + 1)


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------


def warmup_and_time(jit_fn, init, K: int) -> tuple[float, float]:
    """1 warmup call then K timed repeats. Returns (median µs/iter, min µs/iter)."""
    jax.block_until_ready(jit_fn(init))
    times = []
    for _ in range(K):
        t0 = time.perf_counter()
        jax.block_until_ready(jit_fn(init))
        times.append(time.perf_counter() - t0)
    return statistics.median(times) / N_ITERS * 1e6, min(times) / N_ITERS * 1e6


# ---------------------------------------------------------------------------
# Arm factories
# ---------------------------------------------------------------------------


def arm_bare_trivial():
    def f(carry):
        return lax.while_loop(trivial_cond, trivial_body, carry)

    return jax.jit(f), INIT_TRIVIAL


def arm_tapped_trivial():
    def f(carry):
        return lax.while_loop(trivial_cond, trivial_body, carry)

    ft = tap.verbose(f, on_step=noop_on_step)
    return jax.jit(ft), INIT_TRIVIAL


def arm_bare_expensive():
    def f(carry):
        return lax.while_loop(expensive_cond, expensive_body, carry)

    return jax.jit(f), INIT_EXPENSIVE


def arm_tapped_expensive():
    def f(carry):
        return lax.while_loop(expensive_cond, expensive_body, carry)

    ft = tap.verbose(f, on_step=noop_on_step)
    return jax.jit(ft), INIT_EXPENSIVE


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A1 mitigation: while cond overhead bench"
    )
    parser.add_argument("--smoke", action="store_true", help="Smoke run (K=3 repeats)")
    args = parser.parse_args()

    K = 7 if args.smoke else 25
    smoke_tag = " *(smoke)*" if args.smoke else ""

    print()
    print(f"## A1 cond-overhead bench — while_loop, dim={DIM}, N={N_ITERS}{smoke_tag}")
    print()

    rows = []
    for label, arm_fn in [
        ("bare-trivial", arm_bare_trivial),
        ("tapped-trivial", arm_tapped_trivial),
        ("bare-expensive", arm_bare_expensive),
        ("tapped-expensive", arm_tapped_expensive),
    ]:
        fn, init = arm_fn()
        med, mn = warmup_and_time(fn, init, K)
        rows.append((label, med, mn))

    bare_trivial_med = rows[0][1]
    bare_exp_med = rows[2][1]

    print("| arm | median µs/iter | min µs/iter | overhead vs bare |")
    print("|-----|---------------|-------------|-----------------|")
    for label, med, mn in rows:
        ref = bare_trivial_med if "trivial" in label else bare_exp_med
        ovhd = med - ref
        sign = "+" if ovhd >= 0 else ""
        print(f"| {label} | {med:.2f} | {mn:.2f} | {sign}{ovhd:.2f} µs/iter |")

    print()
    delta_trivial = rows[1][1] - rows[0][1]
    delta_expensive = rows[3][1] - rows[2][1]
    print(f"  tap overhead (trivial cond):   +{delta_trivial:.2f} µs/iter")
    print(f"  tap overhead (expensive cond): +{delta_expensive:.2f} µs/iter")
    extra = delta_expensive - delta_trivial
    print(f"  extra vs trivial (double-cond cost): {extra:+.2f} µs/iter")
    if extra > 5.0:
        print(
            "  WARN: double-cond cost is significant (>5 µs/iter); document prominently."
        )
    else:
        print("  OK: double-cond cost is within noise / negligible.")
    print()


if __name__ == "__main__":
    main()
