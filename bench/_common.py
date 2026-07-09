"""
bench/_common.py — shared infrastructure for the jax-tap bench suite.

Exports:
  - DIM, STEP_SIZE, SEED, L_STEPS, M_PREC  — body constants
  - leapfrog_step(state, _)               — standalone inner step (nested-scan ready)
  - make_body(L)                           — return body(carry, x) using inner lax.scan
  - leapfrog_body_simple(state, _)         — single-step body (L_STEPS=1) for prim-tap arms
  - noop_on_step(event)                    — no-op tap callback
  - make_init(lanes)                       — initial (q, p) state
  - warmup_and_time(jit_fn, init, N, K)   — timed harness (warmup excluded, K repeats)
  - print_markdown_table(...)              — standard bench table printer
"""

from __future__ import annotations

import statistics
import sys
import time

import jax
import jax.lax as lax
import jax.numpy as jnp
import jaxtap as tap

# ---------------------------------------------------------------------------
# Body constants
# ---------------------------------------------------------------------------

DIM = 100
STEP_SIZE = 0.005  # per-leapfrog-sub-step size
SEED = 42
L_STEPS = 15  # leapfrog sub-steps per outer scan step; 2*L_STEPS matvecs/step

# Precision matrix: M_PREC ~ (A Aᵀ)/DIM + I, eigenvalues ~1..2.
# U(q) = 0.5 q^T M_PREC q  →  grad_U(q) = M_PREC q
_key = jax.random.PRNGKey(SEED)
_A = jax.random.normal(_key, (DIM, DIM))
M_PREC: jax.Array = (_A @ _A.T) / DIM + jnp.eye(DIM)


# ---------------------------------------------------------------------------
# Leapfrog step (factored for nested scan)
# ---------------------------------------------------------------------------


def leapfrog_step(state: tuple, _) -> tuple:
    """One leapfrog sub-step: momentum half-kick, position full-step, momentum half-kick.

    Factored as a standalone function so make_body() can embed it in an inner
    lax.scan, giving the outer jaxpr a NESTED scan structure.

    L is static here; for randomised per-step L later, swap the inner scan for
    lax.fori_loop/while_loop with a traced bound — the step fn is already factored
    for it.
    """
    q, p = state
    p = p - STEP_SIZE * jnp.dot(M_PREC, q)
    q = q + STEP_SIZE * p
    return (q, p), None


def make_body(L: int = L_STEPS):
    """Return body(carry, x) that runs L leapfrog sub-steps via an inner lax.scan.

    The outer scan step calls:
        state, _ = jax.lax.scan(leapfrog_step, carry, None, length=L)
    so the outer jaxpr contains a nested scan.  jaxtap's walker instruments BOTH
    levels by default — use ``where=lambda p: p == "scan[0]"`` in tap.verbose() to
    restrict emission to the outer scan only.
    """

    def body(carry, _x):
        state, _ = lax.scan(leapfrog_step, carry, None, length=L)
        return state, None

    return body


def leapfrog_body_simple(state: tuple, _) -> tuple:
    """Single leapfrog sub-step (2 matvecs) — used for prim-tap arms.

    With L_STEPS>1 the walker inserts 2*L_STEPS gated lax.cond checks per scan
    step (one per dot_general prim tap), which swamps the arm and obscures the M1d
    gating demonstration.  This body keeps exactly 2 prim-tap sites per step.
    """
    q, p = state
    p = p - STEP_SIZE * jnp.dot(M_PREC, q)
    q = q + STEP_SIZE * p
    return (q, p), None


# ---------------------------------------------------------------------------
# Shared callbacks / helpers
# ---------------------------------------------------------------------------


def noop_on_step(event: tap.TapEvent) -> None:
    pass


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
# Timing harness
# ---------------------------------------------------------------------------


def warmup_and_time(jit_fn, init, N: int, K: int) -> tuple[float, float]:
    """Warmup (1 call, compilation excluded) then K timed repeats.

    Returns (median µs/step, min µs/step).  Uses jax.block_until_ready to
    ensure device execution completes before stopping the timer.
    """
    jax.block_until_ready(jit_fn(init))
    times = []
    for _ in range(K):
        t0 = time.perf_counter()
        jax.block_until_ready(jit_fn(init))
        times.append(time.perf_counter() - t0)
    return statistics.median(times) / N * 1e6, min(times) / N * 1e6


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------


def print_markdown_table(
    rows: list[dict],
    bare_med: float,
    bare_min: float,
    title: str,
    N: int,
    K: int,
    smoke: bool = False,
) -> None:
    """Print a standard bench markdown table.

    Row dict keys: arm (str), se (str|int), lanes (int), med (float), mn (float).
    bare_med/bare_min: the no-callback baseline for computing vs-bare columns.
    """
    smoke_tag = "  *(smoke run: N=100, K=3)*" if smoke else ""
    N_str = f"{N:,}"

    print()
    print("---")
    print()
    print(f"## {title} — N={N_str}, K={K}{smoke_tag}")
    print()
    print(
        f"Bare body: **{bare_med:.2f} µs/step** (med) / {bare_min:.2f} µs/step (min). "
        f"Body: nested-scan leapfrog, dim={DIM}, L_STEPS={L_STEPS}, "
        f"step_size={STEP_SIZE}."
    )
    print()
    print("Measurement: JIT + 1 warmup excluded; K repeats; `jax.block_until_ready`.")
    print()

    hdr = "| arm | se | lanes | µs/step (med) | µs/step (min) | vs bare (µs) | vs bare (%) |"
    sep = "|-----|----|----|--------------|--------------|-------------|------------|"
    print(hdr)
    print(sep)

    for r in rows:
        vs_us = r["med"] - bare_med
        vs_pct = vs_us / bare_med * 100 if bare_med > 0 else float("nan")

        arm = r["arm"]
        if arm == "bare":
            vs_us_s = "—"
            vs_pct_s = "—"
        else:
            vs_us_s = f"{vs_us:+.2f}"
            vs_pct_s = f"{vs_pct:+.1f}%"

        print(
            f"| {arm:<28} | {str(r['se']):>3} | {r['lanes']:>5}"
            f" | {r['med']:>13.3f} | {r['mn']:>13.3f}"
            f" | {vs_us_s:>12} | {vs_pct_s:>10} |"
        )

    print()
    sys.stdout.flush()
