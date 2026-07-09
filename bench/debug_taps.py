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
bench/debug_taps.py — "What does the lens cost in debugging configurations?"

Arms
----
  debug-carry-se1       — carry tap se=1 (fully open lens); scalar select; isolates frequency cost
  debug-prim-se10       — se-gated dot_general prim tap (M1d gating demo) on simple body (L_STEPS=1)
  vmap-se10-l8          — vmap 8 lanes, jaxtap se=10, nested-scan body, outer-only tapping
  nested-outer-only     — nested-scan body, jaxtap se=10, where=outer-only (FIRST nested-scan datapoint)
  nested-both-levels    — nested-scan body, jaxtap se=10, no where (both outer + inner scan tapped)

The last two arms ("nested-tap volume") are the first datapoint for the deferred nested-scan
benchmarking task.  They isolate the cost of tapping both scan levels vs outer-only.
The µs difference between nested-outer-only and nested-both-levels quantifies inner-scan
emission overhead.  Labelled as [nested-scan bench datapoint] for future reference.

Usage
-----
  uv run python bench/debug_taps.py           # full run (N=10 000, K=7)
  uv run python bench/debug_taps.py --smoke   # smoke at N=100, K=3 (<60 s)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `uv run python bench/debug_taps.py` from the repo root.
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.lax as lax

import jaxtap as tap
from bench._common import (
    DIM,
    L_STEPS,
    leapfrog_body_simple,
    make_body,
    make_init,
    noop_on_step,
    print_markdown_table,
    warmup_and_time,
)

_OUTER_ONLY = lambda p: p == "scan[0]"  # noqa: E731


# ---------------------------------------------------------------------------
# Arm factories
# ---------------------------------------------------------------------------


def arm_bare(N: int, lanes: int = 1) -> tuple:
    body = make_body(L_STEPS)

    def f(state):
        return lax.scan(body, state, None, length=N)[0]

    fn = jax.vmap(f) if lanes > 1 else f
    return jax.jit(fn), make_init(lanes)


def arm_debug_carry_se1(N: int) -> tuple:
    """Carry tap every step with a scalar select.

    Scalar select (q[0]) minimises host-boundary transit so the number isolates
    callback FREQUENCY cost (not data-size cost).  se=1 fires N callbacks per sweep.
    Reference: callback_floor.py verbose(se=1) row (~73 µs/step on empty body).
    """
    body = make_body(L_STEPS)

    def f(state):
        return lax.scan(body, state, None, length=N)[0]

    ft = tap.verbose(
        f,
        on_step=noop_on_step,
        sample_every=1,
        where=_OUTER_ONLY,
        select=lambda leaves: leaves[0][0],  # q[0] — scalar float32
    )
    return jax.jit(ft), make_init(1)


def arm_debug_prim_se10(N: int) -> tuple:
    """Carry tap + dot_general prim tap at se=10 — demonstrates M1d gating.

    Uses leapfrog_body_simple (L_STEPS=1, 2 matvecs/step) rather than the full
    nested-scan body.  Reason: with L_STEPS>1 the walker inserts 2*L_STEPS
    gated lax.cond checks per scan step (one per dot_general instance), which
    adds substantial device-side overhead that swamps the gating demonstration.
    With L_STEPS=1 there are exactly 2 prim taps per gated step.

    Before M1d: dot_general fired 2*N times regardless of sample_every.
    After M1d: fires only 2*(N/se) = 2*(N/10) times — confirmed by this arm.
    """

    def f(state):
        return lax.scan(leapfrog_body_simple, state, None, length=N)[0]

    ft = tap.verbose(
        f,
        on_step=noop_on_step,
        sample_every=10,
        taps=[tap.on("dot_general", select=lambda o: o[0][0])],
    )
    return jax.jit(ft), make_init(1)


def arm_vmap_se10(N: int, lanes: int = 8) -> tuple:
    """vmap(lanes=8) + jaxtap se=10; nested-scan body; outer-only tapping."""
    body = make_body(L_STEPS)

    def f(state):
        return lax.scan(body, state, None, length=N)[0]

    ft = tap.verbose(
        f,
        on_step=noop_on_step,
        sample_every=10,
        where=_OUTER_ONLY,
    )
    fn = jax.vmap(ft)
    return jax.jit(fn), make_init(lanes)


def arm_nested_outer_only(N: int) -> tuple:
    """[nested-scan bench datapoint] se=10, outer-scan-only emission.

    The body contains a nested scan.  where=outer-only means only the outer
    scan[0] emits carry taps; the inner leapfrog scan is walked but silent.
    """
    body = make_body(L_STEPS)

    def f(state):
        return lax.scan(body, state, None, length=N)[0]

    ft = tap.verbose(
        f,
        on_step=noop_on_step,
        sample_every=10,
        where=_OUTER_ONLY,  # outer scan only
    )
    return jax.jit(ft), make_init(1)


def arm_nested_both_levels(N: int) -> tuple:
    """[nested-scan bench datapoint] se=10, both outer AND inner scan tapped.

    No where= filter, so jaxtap instruments both scan levels.  Inner heartbeats
    fire per leapfrog sub-step (L_STEPS=15 inner steps per outer step).
    The delta vs nested-outer-only quantifies the inner-scan emission overhead.
    """
    body = make_body(L_STEPS)

    def f(state):
        return lax.scan(body, state, None, length=N)[0]

    ft = tap.verbose(
        f,
        on_step=noop_on_step,
        sample_every=10,
        # no where= : both outer and inner scan emit
    )
    return jax.jit(ft), make_init(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="jaxtap debug-taps benchmark — carry, prim, vmap, nested-scan"
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

    print(
        f"jax {jax.__version__} | device: {jax.devices()[0]}",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"debug_taps | smoke={args.smoke} | N={N:,} | K={K} | DIM={DIM} | L_STEPS={L_STEPS}",
        file=sys.stderr,
        flush=True,
    )

    rows: list[dict] = []

    # --- bare baseline (nested-scan body) ---
    print("\n  bare ...", file=sys.stderr, flush=True)
    fn, init = arm_bare(N)
    bare_med, bare_min = warmup_and_time(fn, init, N, K)
    rows.append(dict(arm="bare", se="-", lanes=1, med=bare_med, mn=bare_min))
    print(
        f"  bare: {bare_med:.3f} µs/step (BODY BASELINE)", file=sys.stderr, flush=True
    )

    # --- debug-carry-se1 ---
    print("  debug-carry-se1 ...", file=sys.stderr, flush=True)
    fn, init = arm_debug_carry_se1(N)
    med, mn = warmup_and_time(fn, init, N, K)
    rows.append(dict(arm="debug-carry-se1", se=1, lanes=1, med=med, mn=mn))
    print(
        f"  debug-carry-se1: {med:.3f} µs/step  (+{med - bare_med:.2f} µs = {(med - bare_med) / bare_med * 100:.1f}%)",
        file=sys.stderr,
        flush=True,
    )

    # --- debug-prim-se10 (simple body, L_STEPS=1) ---
    # Note: bare_prim is the bare arm for this SIMPLE body, not the nested body
    print("  debug-prim-se10 (simple body, L_STEPS=1) ...", file=sys.stderr, flush=True)

    def bare_simple_f(state):
        return lax.scan(leapfrog_body_simple, state, None, length=N)[0]

    bare_simple_fn = jax.jit(bare_simple_f)
    bare_simple_med, bare_simple_min = warmup_and_time(
        bare_simple_fn, make_init(1), N, K
    )
    rows.append(
        dict(
            arm="bare-simple (L_STEPS=1)",
            se="-",
            lanes=1,
            med=bare_simple_med,
            mn=bare_simple_min,
        )
    )
    print(
        f"  bare-simple: {bare_simple_med:.3f} µs/step (simple body baseline)",
        file=sys.stderr,
        flush=True,
    )

    fn, init = arm_debug_prim_se10(N)
    med, mn = warmup_and_time(fn, init, N, K)
    rows.append(dict(arm="debug-prim-se10", se=10, lanes=1, med=med, mn=mn))
    print(
        f"  debug-prim-se10: {med:.3f} µs/step  (+{med - bare_simple_med:.2f} µs vs simple-bare = {(med - bare_simple_med) / bare_simple_med * 100:.1f}%)",
        file=sys.stderr,
        flush=True,
    )

    # --- vmap-se10 (8 lanes) ---
    print("  vmap-se10 (lanes=8) ...", file=sys.stderr, flush=True)
    fn_bare8, init8 = arm_bare(N, lanes=8)
    bare8_med, bare8_min = warmup_and_time(fn_bare8, init8, N, K)
    rows.append(dict(arm="bare-l8", se="-", lanes=8, med=bare8_med, mn=bare8_min))
    print(
        f"  bare-l8: {bare8_med:.3f} µs/step (vmap bare baseline)",
        file=sys.stderr,
        flush=True,
    )

    fn, init = arm_vmap_se10(N, lanes=8)
    med, mn = warmup_and_time(fn, init, N, K)
    rows.append(dict(arm="vmap-se10", se=10, lanes=8, med=med, mn=mn))
    print(
        f"  vmap-se10: {med:.3f} µs/step  (+{med - bare8_med:.2f} µs vs vmap-bare = {(med - bare8_med) / bare8_med * 100:.1f}%)",
        file=sys.stderr,
        flush=True,
    )

    # --- nested-tap volume (first nested-scan bench datapoint) ---
    print("\n  [nested-scan bench datapoint]", file=sys.stderr, flush=True)

    print("  nested-outer-only (se=10, where=outer) ...", file=sys.stderr, flush=True)
    fn, init = arm_nested_outer_only(N)
    nested_outer_med, nested_outer_min = warmup_and_time(fn, init, N, K)
    rows.append(
        dict(
            arm="nested-outer-only",
            se=10,
            lanes=1,
            med=nested_outer_med,
            mn=nested_outer_min,
        )
    )
    print(
        f"  nested-outer-only: {nested_outer_med:.3f} µs/step  (+{nested_outer_med - bare_med:.2f} µs)",
        file=sys.stderr,
        flush=True,
    )

    print("  nested-both-levels (se=10, no where) ...", file=sys.stderr, flush=True)
    fn, init = arm_nested_both_levels(N)
    nested_both_med, nested_both_min = warmup_and_time(fn, init, N, K)
    rows.append(
        dict(
            arm="nested-both-levels",
            se=10,
            lanes=1,
            med=nested_both_med,
            mn=nested_both_min,
        )
    )
    print(
        f"  nested-both-levels: {nested_both_med:.3f} µs/step  (+{nested_both_med - bare_med:.2f} µs)",
        file=sys.stderr,
        flush=True,
    )

    nested_delta = nested_both_med - nested_outer_med
    print(
        f"\n  [nested-scan bench datapoint] outer-only={nested_outer_med:.3f} µs/step,"
        f" both-levels={nested_both_med:.3f} µs/step,"
        f" delta={nested_delta:+.3f} µs/step (inner-scan emission cost at se=10)",
        file=sys.stderr,
        flush=True,
    )

    print_markdown_table(
        rows,
        bare_med,
        bare_min,
        title="debug_taps — nested-scan leapfrog, debugging configurations",
        N=N,
        K=K,
        smoke=args.smoke,
    )

    # Nested-scan datapoint summary
    print("### Nested-scan tap volume (first datapoint — deferred nested-scan bench)")
    print()
    print("| arm | se | µs/step (med) | vs bare (µs) |")
    print("|-----|----|----|-----|")
    print(
        f"| nested-outer-only | 10 | {nested_outer_med:.3f} | +{nested_outer_med - bare_med:.2f} |"
    )
    print(
        f"| nested-both-levels | 10 | {nested_both_med:.3f} | +{nested_both_med - bare_med:.2f} |"
    )
    print(f"| **delta (both − outer)** | — | — | **{nested_delta:+.3f}** µs/step |")
    print()
    print(
        f"Inner-scan emission overhead: **{nested_delta:+.3f} µs/step** at se=10 with L_STEPS={L_STEPS}."
        " This is the first datapoint for the deferred nested-scan benchmarking task."
        f" Inner scan fires L_STEPS={L_STEPS} heartbeats per outer step; at se=10"
        " the inner callbacks fire on ~N/10 outer steps."
    )
    print()

    # Config notes
    print("### Config notes (debug_taps.py)")
    print()
    print(
        "- **debug-carry-se1**: `tap.verbose(f, se=1, where=outer, select=lambda l: l[0][0])`"
        " — scalar select (q[0]) isolates FREQUENCY cost (not carry-size cost);"
        " se=1 fires N callbacks per sweep; reference for always-on monitoring floor"
    )
    print(
        "- **debug-prim-se10**: `tap.verbose(f_simple, se=10, taps=[tap.on('dot_general', ...)])`"
        " — simple body (L_STEPS=1, 2 matvecs/step); M1d gating demo; before M1d:"
        " 2N firings; after M1d: 2×(N/10) firings"
    )
    print(
        f"- **vmap-se10**: `jax.vmap(tap.verbose(f, se=10, where=outer))` — {8} lanes;"
        " nested-scan body; outer-scan-only; each lane fires own callbacks"
    )
    print(
        '- **nested-outer-only**: `tap.verbose(f, se=10, where=lambda p: p == "scan[0]")`'
        " — nested-scan body; only outer scan emits; inner leapfrog scan silent"
    )
    print(
        "- **nested-both-levels**: `tap.verbose(f, se=10)` — nested-scan body; no where="
        " filter; both outer scan and inner leapfrog scan emit carry taps"
    )
    print()

    sys.stdout.flush()


if __name__ == "__main__":
    main()
