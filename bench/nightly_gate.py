#!/usr/bin/env python3
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
bench/nightly_gate.py — CI-safe benchmark gate for machinery regression detection.

Measures the irreducible jaxtap machinery overhead (carrying payload above the
host-callback floor) on the same machine in one process. The design isolates
machinery cost from payload-transit cost by computing:

    MACHINERY = verbose(se=1) − manual-payload (µs/step)

This difference is self-normalizing across machines — most noise from different
hardware cancels out because both arms run in the same process. The verbose arm
includes full jaxtap machinery (step routing, event construction); the manual-payload
arm mirrors the exact payload (step int32 + dim-8 float32 carry) without jaxtap,
measuring only the host-callback floor.

CRITICAL: The self-normalizing property only holds at full N (≥10,000 steps).
At small N (smoke mode, N=100), trace/dispatch overhead does not amortize and
machinery numbers are ~7× inflated and not comparable; smoke mode is report-only.

Rationale for the 15.0 µs threshold:
  - Current machinery ≈ 8.2 µs (post perf/emission-machinery)
  - Known int()-regression class measured 21 µs
  - 15 µs catches structural regressions while tolerating shared-runner noise
  - Absolute floor numbers (machinery near 0) are REPORTED, never gated

Usage
-----
  uv run python bench/nightly_gate.py           # full run (~5-10 s)
  uv run python bench/nightly_gate.py --smoke   # quick check (<2 s, N=100)

Environment
-----------
  JAXTAP_BENCH_GATE_US      — override machinery threshold (default: 15.0)
  GITHUB_STEP_SUMMARY       — append table to GitHub Actions job summary
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import jax
import jax.lax as lax
import jax.numpy as jnp

import jaxtap as tap

DIM = 8
SEED = 42


def make_xs(N: int) -> jax.Array:
    """Generate N random input vectors."""
    return jax.random.normal(jax.random.PRNGKey(SEED), (N, DIM))


def scan_body(carry: jax.Array, x: jax.Array):
    """Simple scan body: c = c * 1.01 + sin(x)."""
    return carry * 1.01 + jnp.sin(x), None


def noop_on_step(event: tap.TapEvent) -> None:
    """No-op tap callback."""
    pass


def warmup_and_time(jit_fn, carry: jax.Array, N: int, K: int) -> float:
    """Warmup (1 call, compilation excluded) then K timed repeats.

    Returns median µs/step.
    """
    jax.block_until_ready(jit_fn(carry))
    times = []
    for _ in range(K):
        t0 = time.perf_counter()
        jax.block_until_ready(jit_fn(carry))
        times.append(time.perf_counter() - t0)
    return statistics.median(times) / N * 1e6


def arm_manual_payload(N: int) -> tuple:
    """
    Manual baseline arm: jax.debug.callback with step int32 + carry float32.

    This mirrors the exact payload of the verbose arm (step int32 + dim-8 carry)
    without jaxtap machinery, establishing the irreducible host-callback floor.
    """
    xs = make_xs(N)
    init = (jnp.zeros(DIM), jnp.int32(0))

    def body(carry, x):
        c, i = carry
        c2 = c * 1.01 + jnp.sin(x)
        jax.debug.callback(lambda i_, v_: None, i, c2, ordered=False)
        return (c2, i + 1), None

    def f(carry):
        (c, _), _ = lax.scan(body, carry, xs)
        return c

    return jax.jit(f), init


def arm_verbose(N: int, sample_every: int = 1) -> tuple:
    """
    Verbose arm: tap.verbose with on_step=noop, measuring full machinery.
    """
    xs = make_xs(N)
    init = jnp.zeros(DIM)

    def f(c):
        return lax.scan(scan_body, c, xs)[0]

    ft = tap.verbose(f, on_step=noop_on_step, sample_every=sample_every)
    return jax.jit(ft), init


def print_results(
    manual_payload_us: float,
    verbose_us: float,
    machinery_us: float,
    passed: bool,
    threshold_us: float,
) -> str:
    """Generate markdown table for output."""
    status = "✓ PASS" if passed else "✗ FAIL"

    lines = []
    lines.append("")
    lines.append("## Machinery Regression Gate")
    lines.append("")
    lines.append("| Arm | µs/step |")
    lines.append("|-----|---------|")
    lines.append(f"| manual-payload | {manual_payload_us:.3f} |")
    lines.append(f"| verbose(se=1) | {verbose_us:.3f} |")
    lines.append(f"| **MACHINERY** | **{machinery_us:.3f}** |")
    lines.append("")
    lines.append(f"Threshold: {threshold_us:.1f} µs/step")
    lines.append(f"Status: {status}")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Machinery regression gate for jaxtap CI."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke run: N=100, K=3 — quick sanity check",
    )
    args = parser.parse_args()

    if args.smoke:
        N = 100
        K = 3
    else:
        N = 10_000
        K = 5

    # Get threshold from environment or use default
    threshold_us = float(os.environ.get("JAXTAP_BENCH_GATE_US", "15.0"))

    print(
        f"jax {jax.__version__} | device: {jax.devices()[0]}",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"N={N:,} | K={K} | threshold={threshold_us:.1f} µs",
        file=sys.stderr,
        flush=True,
    )

    # Measure manual-payload arm
    print("Measuring manual-payload...", file=sys.stderr, flush=True)
    fn, init = arm_manual_payload(N)
    manual_payload_us = warmup_and_time(fn, init, N, K)
    print(
        f"  manual-payload: {manual_payload_us:.3f} µs/step",
        file=sys.stderr,
        flush=True,
    )

    # Measure verbose arm
    print("Measuring verbose(se=1)...", file=sys.stderr, flush=True)
    fn, init = arm_verbose(N, sample_every=1)
    verbose_us = warmup_and_time(fn, init, N, K)
    print(f"  verbose(se=1):  {verbose_us:.3f} µs/step", file=sys.stderr, flush=True)

    # Compute machinery (self-normalizing difference)
    machinery_us = verbose_us - manual_payload_us
    print(
        f"  MACHINERY:      {machinery_us:.3f} µs/step",
        file=sys.stderr,
        flush=True,
    )

    # Determine pass/fail
    passed = machinery_us <= threshold_us

    # Generate and print output
    output = print_results(
        manual_payload_us, verbose_us, machinery_us, passed, threshold_us
    )
    print(output)

    # Append to GitHub Actions job summary if set
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        try:
            with open(summary_file, "a") as f:
                f.write(output)
                f.write("\n")
        except Exception as e:
            print(
                f"Warning: could not write to GITHUB_STEP_SUMMARY: {e}", file=sys.stderr
            )

    # Exit with appropriate code
    if args.smoke:
        # Smoke mode is report-only: trace/dispatch overhead does not amortize at small N
        print(
            "\n⊘ Smoke mode (N=100): threshold not applied — use full run (N=10,000) to gate.",
            file=sys.stderr,
        )
        sys.exit(0)
    elif passed:
        print(
            f"\n✓ Machinery {machinery_us:.3f} µs ≤ {threshold_us:.1f} µs threshold",
            file=sys.stderr,
        )
        sys.exit(0)
    else:
        print(
            f"\n✗ REGRESSION: Machinery {machinery_us:.3f} µs > {threshold_us:.1f} µs threshold",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
