# Copyright 2026 The jax-tap Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
bench/progress_bar.py — "What does the lens cost for a progress bar on a realistic workload?"

Headline benchmark: semi-production progress-bar monitoring of a leapfrog sweep
(dim-100 Gaussian target, L_STEPS=15 sub-steps per scan step, nested-scan body).

The body contains a NESTED scan: the outer scan drives N sampler steps; each outer
step runs an inner lax.scan of L_STEPS leapfrog sub-steps.  All jaxtap arms tap the
OUTER scan only, via ``where=lambda p: p == "scan[0]"``, so inner heartbeats from the
leapfrog sub-steps do not fire.

Arms
----
  bare                   — plain outer lax.scan, jitted, no callbacks
  manual-progress        — step-only jax.debug.callback every step (hand-rolled bar)
  jaxtap-se10            — tap.verbose(f, se=10, where=outer-only); full (q,p) carry
  jaxtap-se100           — tap.verbose(f, se=100, where=outer-only); full (q,p) carry
  jaxtap-se10-progress   — tap.verbose(f, se=10, where=outer, select=empty); 0 bytes
  jaxtap-se100-progress  — tap.verbose(f, se=100, where=outer, select=empty); 0 bytes

The ``-progress`` rows use ``select=lambda _: ()`` — zero bytes cross the host
boundary.  TapEvent.value=(); callback cost ≈ the step-only callback floor (~33-40 µs).

Change 3 — manual floor sanity check
-------------------------------------
The manual-progress arm should sit near the ~33-37 µs/callback floor documented in
callback_floor.py.  If the measured delta vs bare deviates wildly (<25 or >50 µs),
it is flagged as a finding in the printed output.

Usage
-----
  uv run python bench/progress_bar.py           # full run (N=10 000, K=7)
  uv run python bench/progress_bar.py --smoke   # smoke at N=100, K=3 (<60 s)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `uv run python bench/progress_bar.py` from the repo root.
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.lax as lax
import jax.numpy as jnp

import jaxtap as tap
from bench._common import (
    DIM,
    L_STEPS,
    STEP_SIZE,
    make_body,
    make_init,
    noop_on_step,
    print_markdown_table,
    warmup_and_time,
)

# outer-scan-only emission filter: tap only the outermost scan (scan[0]);
# the walker still descends into the inner leapfrog scan but does not emit there.
_OUTER_ONLY = lambda p: p == "scan[0]"  # noqa: E731


# ---------------------------------------------------------------------------
# Arm factories
# ---------------------------------------------------------------------------


def arm_bare(N: int) -> tuple:
    body = make_body(L_STEPS)

    def f(state):
        return lax.scan(body, state, None, length=N)[0]

    return jax.jit(f), make_init(1)


def arm_manual_progress(N: int) -> tuple:
    """Step-only jax.debug.callback every step — minimal hand-rolled progress bar.

    Callback: single int32 step index, no carry shipped across the host boundary.
    Serves as the reference floor for single-step-only callback cost.
    """
    body = make_body(L_STEPS)
    steps = jnp.arange(N, dtype=jnp.int32)

    def body_with_cb(state, step):
        new_state, _ = body(state, step)
        jax.debug.callback(lambda s: None, step, ordered=False)
        return new_state, None

    def f(state):
        return lax.scan(body_with_cb, state, steps)[0]

    return jax.jit(f), make_init(1)


def arm_jaxtap_carry(N: int, sample_every: int = 10) -> tuple:
    """tap.verbose carry tap; outer-scan-only (where="scan[0]"); full (q,p) payload."""
    body = make_body(L_STEPS)

    def f(state):
        return lax.scan(body, state, None, length=N)[0]

    ft = tap.verbose(
        f,
        on_step=noop_on_step,
        sample_every=sample_every,
        where=_OUTER_ONLY,  # outer scan only — inner leapfrog scan not emitted
    )
    return jax.jit(ft), make_init(1)


def arm_jaxtap_progress(N: int, sample_every: int = 10) -> tuple:
    """tap.verbose with empty-payload select — the recommended progress-bar idiom.

    ``select=lambda _: ()`` ships ZERO carry bytes across the host boundary.
    The TapEvent still carries path/step/total metadata for tqdm-style display;
    it simply has an empty value field.  Callback cost ≈ step-only floor (~33-40 µs).
    Outer-scan-only via ``where=lambda p: p == "scan[0]"``.
    """
    body = make_body(L_STEPS)

    def f(state):
        return lax.scan(body, state, None, length=N)[0]

    ft = tap.verbose(
        f,
        on_step=noop_on_step,
        sample_every=sample_every,
        where=_OUTER_ONLY,  # outer scan only — inner leapfrog scan not emitted
        select=lambda _: (),  # zero bytes cross the host boundary
    )
    return jax.jit(ft), make_init(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="jaxtap progress-bar benchmark — nested-scan leapfrog body"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke run at N=100, K=3 to verify all code paths before full run",
    )
    args = parser.parse_args()

    if args.smoke:
        N = 100
        K = 3
    else:
        N = 10_000
        K = 7

    print(f"jax {jax.__version__} | device: {jax.devices()[0]}", file=sys.stderr, flush=True)
    print(
        f"progress_bar | smoke={args.smoke} | N={N:,} | K={K} | DIM={DIM} | L_STEPS={L_STEPS}",
        file=sys.stderr,
        flush=True,
    )

    rows: list[dict] = []

    # --- bare ---
    print("\n  bare ...", file=sys.stderr, flush=True)
    fn, init = arm_bare(N)
    bare_med, bare_min = warmup_and_time(fn, init, N, K)
    rows.append(dict(arm="bare", se="-", lanes=1, med=bare_med, mn=bare_min))
    print(f"  bare: {bare_med:.3f} µs/step (BODY BASELINE)", file=sys.stderr, flush=True)

    # --- manual-progress ---
    print("  manual-progress ...", file=sys.stderr, flush=True)
    fn, init = arm_manual_progress(N)
    med, mn = warmup_and_time(fn, init, N, K)
    rows.append(dict(arm="manual-progress", se=1, lanes=1, med=med, mn=mn))
    delta = med - bare_med
    print(
        f"  manual-progress: {med:.3f} µs/step  (+{delta:.2f} µs vs bare)",
        file=sys.stderr,
        flush=True,
    )

    # sanity check: manual floor should be near ~33-37 µs/callback floor
    if delta < 25 or delta > 50:
        print(
            f"  *** FINDING: manual-progress delta={delta:.2f} µs OUTSIDE expected 25-50 µs range ***",
            file=sys.stderr,
            flush=True,
        )

    # --- jaxtap carry (full payload) ---
    for se in [10, 100]:
        print(f"  jaxtap-se{se} (full carry, outer-only) ...", file=sys.stderr, flush=True)
        fn, init = arm_jaxtap_carry(N, sample_every=se)
        med, mn = warmup_and_time(fn, init, N, K)
        rows.append(dict(arm=f"jaxtap-se{se}", se=se, lanes=1, med=med, mn=mn))
        print(
            f"  jaxtap-se{se}: {med:.3f} µs/step  (+{med - bare_med:.2f} µs = {(med - bare_med) / bare_med * 100:.1f}%)",
            file=sys.stderr,
            flush=True,
        )

    # --- jaxtap progress idiom (empty payload) ---
    for se in [10, 100]:
        print(
            f"  jaxtap-se{se}-progress (empty payload, outer-only) ...", file=sys.stderr, flush=True
        )
        fn, init = arm_jaxtap_progress(N, sample_every=se)
        med, mn = warmup_and_time(fn, init, N, K)
        rows.append(dict(arm=f"jaxtap-se{se}-progress", se=se, lanes=1, med=med, mn=mn))
        print(
            f"  jaxtap-se{se}-progress: {med:.3f} µs/step  (+{med - bare_med:.2f} µs = {(med - bare_med) / bare_med * 100:.1f}%)",
            file=sys.stderr,
            flush=True,
        )

    print_markdown_table(
        rows,
        bare_med,
        bare_min,
        title="progress_bar — nested-scan leapfrog, outer-scan-only tapping",
        N=N,
        K=K,
        smoke=args.smoke,
    )

    # Config notes
    print("### Config notes (progress_bar.py)")
    print()
    print(
        f"- **body**: nested-scan leapfrog — outer `lax.scan` of N steps;"
        f" each outer step runs `lax.scan(leapfrog_step, carry, None, length={L_STEPS})`;"
        f" dim={DIM}, step_size={STEP_SIZE}; carry = (q, p) ∈ ℝ^{DIM} × ℝ^{DIM};"
        " M_PREC = (A Aᵀ)/DIM + I (fixed, seeded)"
    )
    print(
        '- **outer-scan-only tapping**: all jaxtap arms use `where=lambda p: p == "scan[0]"`;'
        " the walker descends into the inner leapfrog scan but does not emit carry taps there;"
        " inner heartbeats do not fire"
    )
    print(
        "- **bare**: `lax.scan(body, init, None, length=N)`, jitted — no callbacks;"
        " body contains inner lax.scan"
    )
    print(
        "- **manual-progress**: same nested-scan body + `jax.debug.callback(λ s: None, step, ordered=False)`"
        " every step; step int32 only, no carry shipped; reference ceiling for step-only callback cost"
    )
    print(
        "- **jaxtap-se10/100**: `tap.verbose(f, on_step=noop, sample_every=k, where=outer)`"
        " — carry tap; full (q, p) carry shipped on fire (800 bytes per event)"
    )
    print(
        "- **jaxtap-se10/100-progress**: `tap.verbose(f, on_step=noop, sample_every=k,"
        " where=outer, select=lambda _: ())` — progress-bar idiom; ZERO bytes cross"
        " the host boundary; TapEvent.value=(); callback cost ≈ step-only floor (~33-40 µs)"
    )
    print()
    if args.smoke:
        print(
            "*Full run: `PYTHONUNBUFFERED=1 uv run python bench/progress_bar.py 2>&1 | tee bench/progress_bar_run.log`*"
        )


if __name__ == "__main__":
    main()
