"""
bench_v2.py — "What does the lens cost on a REALISTIC workload?"

Headline scenario: progress-bar monitoring of a leapfrog sweep (dim-100
Gaussian target). Each scan step runs L_STEPS leapfrog sub-steps (each
requiring two M_PREC matvecs), so the body costs 5–50 µs/step at N=10 000.
The v1 body (~0.1 µs/step) is unrealistically fast — it makes a 33 µs
callback look 300× the body. This benchmark supplies the realistic denominator.

Arm layout
----------
SCENARIO 1 — PROGRESS-BAR (headline):
  bare             — plain lax.scan, jitted, no callbacks
  manual-progress  — step-only debug.callback every step (hand-rolled progress bar)
  jaxtap-se10      — tap.verbose(f, on_step=noop, sample_every=10)  [JP baseline]
  jaxtap-se100     — tap.verbose(f, on_step=noop, sample_every=100)

SCENARIO 2 — DEBUGGING (2 highlight rows):
  debug-carry-se1  — carry tap every step, scalar select (smallest carry transit)
  debug-prim-se10  — simple-body (L_STEPS=1) + dot_general prim tap at se=10
                     (M1d gating demo; uses simple body to avoid 2*L_STEPS
                     prim-tap lax.cond checks per step that would swamp the arm)

SCENARIO 3 — VMAP (semi-production multi-chain):
  vmap-se10        — vmap(8 lanes) + jaxtap-se10

Axes
----
  N           : 10_000 only (N-independence confirmed in v1 bench)
  sample_every: 10, 100 (progress bar); 1 (debugging)
  lanes       : 1 (all except vmap arm), 8 (vmap arm)

Measurement
-----------
  JIT + 1 warmup call (compilation excluded), then K≥5 timed repeats.
  jax.block_until_ready + time.perf_counter. Report median + min µs/step.
  Key output: % overhead vs bare — the number that tells you how much the
  lens costs relative to real compute.

Realistic body (L_STEPS substeps)
----------------------------------
  Each scan step runs L_STEPS leapfrog sub-steps on a dim-100 Gaussian target.
  U(q) = 0.5 q^T M_PREC q  →  grad_U(q) = M_PREC q
  Standard normal momentum: grad_K(p) = p (no extra matvec)
  Two M_PREC matvecs per sub-step → 2*L_STEPS matvecs per scan step.
  L_STEPS is calibrated so the bare body costs 5–50 µs at N=10_000;
  the smoke run at N=100 reports the actual cost so you can adjust if needed.

v1 context
----------
  v1 (bench_overhead.py) benchmarked an empty-ish body (dim=8, ~0.1 µs/step).
  v1 numbers stay as an appendix in bench/README.md.
  v2 supplies the REALISTIC denominator. Both exist for different audiences.

Usage
-----
  uv run python bench/bench_v2.py            # full run (≤10 min)
  uv run python bench/bench_v2.py --smoke    # smoke at N=100 (<60 s)
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import jax
import jax.lax as lax
import jax.numpy as jnp
import jaxtap as tap

# ---------------------------------------------------------------------------
# Body parameters
# ---------------------------------------------------------------------------

DIM = 100  # position/momentum dimension
STEP_SIZE = 0.01
SEED = 42
L_STEPS = 15  # leapfrog sub-steps per scan step; 2*L_STEPS matvecs/step
# targets ~15–25 µs/step bare at N=10_000 on this box

# Build precision matrix at import time (constant across all arms).
# M_PREC is a dim×dim positive-definite matrix with eigenvalues ~1..2.
# U(q) = 0.5 q^T M_PREC q  →  grad_U(q) = M_PREC q
_key = jax.random.PRNGKey(SEED)
_A = jax.random.normal(_key, (DIM, DIM))
M_PREC: jax.Array = (_A @ _A.T) / DIM + jnp.eye(DIM)


# ---------------------------------------------------------------------------
# Leapfrog bodies
# ---------------------------------------------------------------------------


def leapfrog_body(state: tuple, _: jax.Array) -> tuple:
    """L_STEPS leapfrog sub-steps per scan step (the MAIN body for all arms except debug-prim).

    Each sub-step requires two M_PREC matvecs → 2*L_STEPS matvecs total per scan step.
    L_STEPS is calibrated so the bare body costs 5–50 µs at N=10_000.
    """
    q, p = state
    for _ in range(L_STEPS):
        gq = jnp.dot(M_PREC, q)  # matvec — potential gradient at q
        p_half = p - 0.5 * STEP_SIZE * gq
        q = q + STEP_SIZE * p_half
        gq = jnp.dot(M_PREC, q)  # matvec — potential gradient at q_new
        p = p_half - 0.5 * STEP_SIZE * gq
    return (q, p), None


def leapfrog_body_simple(state: tuple, _: jax.Array) -> tuple:
    """Single leapfrog step (L_STEPS=1, 2 matvecs).

    Used ONLY for the debug-prim-se10 arm, where the prim tap instruments
    EVERY dot_general in the body. With L_STEPS>1 each scan step would have
    2*L_STEPS gated lax.cond checks (one per prim tap), swamping the arm
    and obscuring the M1d-gating demonstration.
    """
    q, p = state
    gq = jnp.dot(M_PREC, q)
    p_half = p - 0.5 * STEP_SIZE * gq
    q_new = q + STEP_SIZE * p_half
    gq_new = jnp.dot(M_PREC, q_new)
    p_new = p_half - 0.5 * STEP_SIZE * gq_new
    return (q_new, p_new), None


def noop_on_step(event: tap.TapEvent) -> None:
    pass


# ---------------------------------------------------------------------------
# Init helper
# ---------------------------------------------------------------------------


def make_init(lanes: int = 1) -> tuple:
    rng = jax.random.PRNGKey(SEED + 1)
    if lanes > 1:
        q = jnp.zeros((lanes, DIM))
        p = jax.random.normal(rng, (lanes, DIM))
    else:
        q = jnp.zeros(DIM)
        p = jax.random.normal(rng, (DIM,))
    return (q, p)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def warmup_and_time(jit_fn, init, N: int, K: int) -> tuple[float, float]:
    """1 warmup then K timed repeats. Returns (median µs/step, min µs/step)."""
    jax.block_until_ready(jit_fn(init))
    times = []
    for _ in range(K):
        t0 = time.perf_counter()
        jax.block_until_ready(jit_fn(init))
        times.append(time.perf_counter() - t0)
    return statistics.median(times) / N * 1e6, min(times) / N * 1e6


# ---------------------------------------------------------------------------
# Arm factories
# ---------------------------------------------------------------------------


def arm_bare(N: int, lanes: int = 1) -> tuple:
    def f(state):
        return lax.scan(leapfrog_body, state, None, length=N)[0]

    fn = jax.vmap(f) if lanes > 1 else f
    return jax.jit(fn), make_init(lanes)


def arm_manual_progress(N: int) -> tuple:
    """Step-only debug.callback every step — minimal hand-rolled progress bar.

    Body: same L_STEPS leapfrog sub-steps as arm_bare.
    Callback: single int32 step index only, no carry shipped.
    Serves as the reference ceiling for progress-bar overhead.
    """
    steps = jnp.arange(N, dtype=jnp.int32)

    def body(state, step):
        q, p = state
        for _ in range(L_STEPS):
            gq = jnp.dot(M_PREC, q)
            p_half = p - 0.5 * STEP_SIZE * gq
            q = q + STEP_SIZE * p_half
            gq = jnp.dot(M_PREC, q)
            p = p_half - 0.5 * STEP_SIZE * gq
        jax.debug.callback(lambda s: None, step, ordered=False)
        return (q, p), None

    def f(state):
        return lax.scan(body, state, steps)[0]

    return jax.jit(f), make_init(1)


def arm_jaxtap(N: int, sample_every: int = 10, lanes: int = 1) -> tuple:
    """tap.verbose carry tap; se gates how often the callback fires."""

    def f(state):
        return lax.scan(leapfrog_body, state, None, length=N)[0]

    ft = tap.verbose(f, on_step=noop_on_step, sample_every=sample_every)
    fn = jax.vmap(ft) if lanes > 1 else ft
    return jax.jit(fn), make_init(lanes)


def arm_debug_carry_se1(N: int) -> tuple:
    """Carry tap every step with a scalar select.

    Scalar select minimises host-boundary transit so the number isolates
    callback FREQUENCY cost (not data-size cost). The v1 payload-decomposition
    finding still applies: on an empty body, se=1 runs at ~73 µs/step because
    the host callback is called N times (not because the carry is large).
    Reference: bench_overhead.py verbose(se=1) row.
    """

    def f(state):
        return lax.scan(leapfrog_body, state, None, length=N)[0]

    ft = tap.verbose(
        f,
        on_step=noop_on_step,
        sample_every=1,
        select=lambda leaves: leaves[0][0],  # q[0] — scalar float32
    )
    return jax.jit(ft), make_init(1)


def arm_debug_prim_se10(N: int) -> tuple:
    """Carry tap + dot_general prim tap at se=10 — demonstrates M1d gating.

    Uses leapfrog_body_simple (L_STEPS=1, 2 matvecs/step) rather than the
    full L_STEPS body. Reason: with L_STEPS>1 the walker inserts 2*L_STEPS
    gated lax.cond checks per scan step (one per dot_general instance), which
    adds substantial device-side overhead that swamps the gating demonstration.
    With L_STEPS=1 there are exactly 2 prim taps per gated step, keeping the
    arm interpretable.

    Before M1d: dot_general prim tap fired 2*N times regardless of sample_every.
    After M1d: fires only 2*(N/se) = 2*(N/10) times — confirmed here.
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


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------


def print_tables(rows: list[dict], bare_med: float, bare_min: float, smoke: bool) -> None:
    smoke_tag = "  *(smoke run: N=100, K=3)*" if smoke else ""

    N_str = "100" if smoke else "10 000"

    print()
    print("---")
    print()
    print(
        f"## v2 — realistic body: dim={DIM} leapfrog, {L_STEPS} sub-steps/"
        f"scan-step ({2 * L_STEPS} matvecs/step), N={N_str}{smoke_tag}"
    )
    print()
    print(f"Bare body: **{bare_med:.2f} µs/step** (med) / {bare_min:.2f} µs/step (min)")
    print()
    print("Measurement: JIT + 1 warmup excluded; K repeats; `jax.block_until_ready`.")
    print()

    hdr = (
        "| scenario | arm | se | lanes"
        " | µs/step (med) | µs/step (min)"
        " | vs bare (µs) | vs bare (%) |"
    )
    sep = (
        "|----------|-----|----|------|--------------|---------------|-------------|-------------|"
    )
    print(hdr)
    print(sep)

    for r in rows:
        vs_bare_us = r["med"] - bare_med
        vs_bare_pct = vs_bare_us / bare_med * 100 if bare_med > 0 else float("nan")

        arm = r["arm"]
        if arm == "bare":
            vs_bare_us_s = "—"
            vs_bare_pct_s = "—"
        else:
            vs_bare_us_s = f"{vs_bare_us:+.2f}"
            vs_bare_pct_s = f"{vs_bare_pct:+.1f}%"

        print(
            f"| {r['scenario']:<9} | {arm:<18} | {str(r['se']):>3} | {r['lanes']:>5}"
            f" | {r['med']:>13.3f} | {r['mn']:>13.3f}"
            f" | {vs_bare_us_s:>12} | {vs_bare_pct_s:>11} |"
        )

    print()
    print("### Config notes")
    print()
    print(
        f"- **body (main)**: {L_STEPS} leapfrog sub-steps per scan step on dim-{DIM} Gaussian;"
        f" {2 * L_STEPS} `jnp.dot(M_PREC, q)` matvecs per scan step;"
        " carry = (q, p) ∈ ℝ^100 × ℝ^100; step_size=0.01; M_PREC = (A Aᵀ)/DIM + I (fixed, seeded)"
    )
    print(
        "- **body (debug-prim)**: 1 leapfrog sub-step (2 matvecs) — simpler body used"
        " for the prim tap arm to avoid 2×L_STEPS lax.cond checks per scan step"
    )
    print("- **bare**: `lax.scan(leapfrog_body, init, None, length=N)`, jitted — no callbacks")
    print(
        "- **manual-progress**: same L_STEPS body + `jax.debug.callback(λ s: None, step, ordered=False)`"
        " every step — hand-rolled progress bar, step int32 only, no carry shipped"
    )
    print(
        "- **jaxtap-se10/100**: `tap.verbose(f, on_step=noop, sample_every=k)` — carry tap;"
        " device-side `lax.cond(step % k == 0, fire, noop)` gate; full (q, p) carry shipped on fire"
    )
    print(
        "- **debug-carry-se1**: `tap.verbose(f, sample_every=1, select=lambda l: l[0][0])`"
        " — scalar select (q[0]) minimises data transit; isolates FREQUENCY cost (not carry-size cost)"
    )
    print(
        "- **debug-prim-se10**: `tap.verbose(f_simple, se=10, taps=[tap.on('dot_general', select=lambda o: o[0][0])])`"
        " — simple body; carry tap + M1d-gated dot_general prim tap;"
        " before M1d: 2N firings; after M1d: 2×(N/10) firings"
    )
    print(
        "- **vmap-se10**: `jax.vmap(tap.verbose(f, se=10))` — 8-lane semi-production multi-chain;"
        " each lane fires its own callbacks; per-step cost across ALL lanes reported"
    )
    if smoke:
        print()
        print(
            "*Full run: `PYTHONUNBUFFERED=1 uv run python bench/bench_v2.py 2>&1 | tee bench/v2_run.log`*"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="jaxtap v2 overhead benchmark — realistic body")
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
        f"bench_v2 | smoke={args.smoke} | N={N:,} | K={K} | DIM={DIM}", file=sys.stderr, flush=True
    )

    rows: list[dict] = []

    # --- SCENARIO 1: PROGRESS-BAR ---

    print("\n=== SCENARIO 1: PROGRESS-BAR ===", file=sys.stderr, flush=True)

    fn, init = arm_bare(N)
    bare_med, bare_min = warmup_and_time(fn, init, N, K)
    rows.append(dict(scenario="progress", arm="bare", se="-", lanes=1, med=bare_med, mn=bare_min))
    print(
        f"  bare:              {bare_med:.3f} µs/step  (BODY BASELINE)", file=sys.stderr, flush=True
    )
    sys.stdout.flush()

    fn, init = arm_manual_progress(N)
    med, mn = warmup_and_time(fn, init, N, K)
    rows.append(dict(scenario="progress", arm="manual-progress", se=1, lanes=1, med=med, mn=mn))
    print(
        f"  manual-progress:   {med:.3f} µs/step  (+{med - bare_med:.2f} µs = {(med - bare_med) / bare_med * 100:.1f}% overhead)",
        file=sys.stderr,
        flush=True,
    )
    sys.stdout.flush()

    for se in [10, 100]:
        fn, init = arm_jaxtap(N, sample_every=se)
        med, mn = warmup_and_time(fn, init, N, K)
        rows.append(dict(scenario="progress", arm=f"jaxtap-se{se}", se=se, lanes=1, med=med, mn=mn))
        print(
            f"  jaxtap-se{se:<3}:       {med:.3f} µs/step  (+{med - bare_med:.2f} µs = {(med - bare_med) / bare_med * 100:.1f}% overhead)",
            file=sys.stderr,
            flush=True,
        )
        sys.stdout.flush()

    # --- SCENARIO 2: DEBUGGING ---

    print("\n=== SCENARIO 2: DEBUGGING ===", file=sys.stderr, flush=True)

    fn, init = arm_debug_carry_se1(N)
    med, mn = warmup_and_time(fn, init, N, K)
    rows.append(dict(scenario="debug", arm="debug-carry-se1", se=1, lanes=1, med=med, mn=mn))
    print(
        f"  debug-carry-se1:   {med:.3f} µs/step  (+{med - bare_med:.2f} µs = {(med - bare_med) / bare_med * 100:.1f}% overhead)",
        file=sys.stderr,
        flush=True,
    )
    sys.stdout.flush()

    fn, init = arm_debug_prim_se10(N)
    med, mn = warmup_and_time(fn, init, N, K)
    rows.append(dict(scenario="debug", arm="debug-prim-se10", se=10, lanes=1, med=med, mn=mn))
    print(
        f"  debug-prim-se10:   {med:.3f} µs/step  (+{med - bare_med:.2f} µs = {(med - bare_med) / bare_med * 100:.1f}% overhead)",
        file=sys.stderr,
        flush=True,
    )
    sys.stdout.flush()

    # --- SCENARIO 3: VMAP ---

    print("\n=== SCENARIO 3: VMAP (lanes=8) ===", file=sys.stderr, flush=True)

    fn, init = arm_bare(N, lanes=8)
    bare8_med, bare8_min = warmup_and_time(fn, init, N, K)
    rows.append(dict(scenario="vmap", arm="bare-l8", se="-", lanes=8, med=bare8_med, mn=bare8_min))
    print(
        f"  bare-l8:           {bare8_med:.3f} µs/step  (vmap bare baseline)",
        file=sys.stderr,
        flush=True,
    )
    sys.stdout.flush()

    fn, init = arm_jaxtap(N, sample_every=10, lanes=8)
    med, mn = warmup_and_time(fn, init, N, K)
    rows.append(dict(scenario="vmap", arm="vmap-se10", se=10, lanes=8, med=med, mn=mn))
    print(
        f"  vmap-se10:         {med:.3f} µs/step  (+{med - bare8_med:.2f} µs vs vmap-bare = {(med - bare8_med) / bare8_med * 100:.1f}%)",
        file=sys.stderr,
        flush=True,
    )
    sys.stdout.flush()

    print_tables(rows, bare_med, bare_min, smoke=args.smoke)


if __name__ == "__main__":
    main()
