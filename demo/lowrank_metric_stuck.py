"""demo — the metric that never moved (#949, parked SEVEN WEEKS).

ORIGINAL BUG (class: control-flow / carry tap on adaptation state)
------------------------------------------------------------------
A low-rank mass-matrix adaptation consumed the SCORE covariance without
inverting it. For a Gaussian target N(0, S): scores are s = -S^{-1} x, so
cov(scores) = S^{-1} — and the un-inverted factor cancels the position
covariance: the learned metric collapses to cov(x) @ cov(s) ~ S S^{-1} = I.
The adaptation RAN, returned a metric, raised nothing. The sampler just mixed
badly. Every run "worked." The bug sat parked for SEVEN WEEKS and fell only
to a dedicated 10-D MVN control experiment plus a code read.

WHAT THIS DEMO SHOWS
--------------------
The warmup loop below contains ZERO logging code. A carry tap whose ``select``
computes the running metric's eigenvalue range ON-DEVICE makes the failure
visible in ONE run, within the FIRST sampled window: the buggy metric's
eigenvalues sit pinned at ~1.0 while the true target covariance spans
[0.1, 10]. Minutes instead of seven weeks — the tap turns "adaptation ran"
into "adaptation LEARNED NOTHING," which is the actual question.

Run:  uv run python demo/lowrank_metric_stuck.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import jaxtap as tap

DIM = 10
N_STEPS = 4000
EVERY = 500  # tap sampling stride (volume knob)

# True target covariance: eigenvalues spread over [0.1, 10] — plenty to learn.
TRUE_EIGS = jnp.geomspace(0.1, 10.0, DIM)
SIGMA = jnp.diag(TRUE_EIGS)


def make_warmup(buggy: bool):
    """A toy warmup: stream draws, accumulate position/score covariances,
    derive the metric. NOTE: no logging/telemetry code anywhere in here."""

    def logdensity_grad(x):
        # the model's score: for N(0, S), grad log p = -S^{-1} x
        return -jnp.linalg.solve(SIGMA, x)

    def step(carry, x):
        s_x, s_s, k = carry
        s = logdensity_grad(x)
        s_x = s_x + jnp.outer(x, x)          # position covariance accumulator
        s_s = s_s + jnp.outer(s, s)          # score covariance accumulator
        return (s_x, s_s, k + 1.0), None

    def run(draws):
        init = (jnp.zeros((DIM, DIM)), jnp.zeros((DIM, DIM)), 0.0)
        (s_x, s_s, k), _ = jax.lax.scan(step, init, draws)
        cov_x, cov_s = s_x / k, s_s / k
        if buggy:
            # THE BUG: score covariance used UN-INVERTED -> S @ S^{-1} ~ I.
            # The whitening factor cancels everything the metric should learn.
            return cov_x @ cov_s
        # the fix (#949): the position covariance IS the learned metric ~ S
        return cov_x

    return run


def metric_eig_range(leaves):
    """Carry-tap ``select``: derive the CURRENT metric from the running
    accumulators and reduce to its eigenvalue range — all on-device; only two
    scalars cross to the host per sampled step."""
    s_x, s_s, k = leaves
    kk = jnp.maximum(k, 1.0)
    m = (s_x / kk) @ (s_s / kk)  # the buggy pipeline's metric, as it evolves
    # m is a product of two SPD estimates: not symmetric, but similar to an
    # SPD matrix -> its eigenvalues are real and positive. eigvalsh would be
    # WRONG here (garbage on non-symmetric input); use eigvals + real part.
    eigs = jnp.sort(jnp.real(jnp.linalg.eigvals(m)))
    return {"eig_min": eigs[0], "eig_max": eigs[-1]}


def main() -> None:
    print("=" * 70)
    key = jax.random.key(949)
    draws = jax.random.multivariate_normal(key, jnp.zeros(DIM), SIGMA, (N_STEPS,))

    # ---------------- WITHOUT jax-tap: what a user sees ----------------
    m_buggy = make_warmup(buggy=True)(draws)
    m_fixed = make_warmup(buggy=False)(draws)
    print("\nwithout jax-tap: warmup 'succeeded' in both pipelines —")
    print("  a metric came back, nothing raised, nothing looked wrong.")

    # ---------------- WITH jax-tap ----------------
    # The with-block is the only addition; the warmup code is UNMODIFIED.
    with tap.record(select=metric_eig_range, sample_every=EVERY) as rec:
        make_warmup(buggy=True)(draws)

    print(f"\nwith jax-tap (carry tap, every {EVERY} steps): the metric's own story —")
    print(f"  {'step':>5}  {'eig_min':>8}  {'eig_max':>8}   (true covariance spans "
          f"[{float(TRUE_EIGS[0]):.1f}, {float(TRUE_EIGS[-1]):.1f}])")
    win = sorted((e for e in rec.events if e.path == "scan[0]"), key=lambda e: e.step)
    for e in win:
        if e.step == 0:
            continue  # k=1: single-sample estimate, not informative
        print(f"  {e.step:>5}  {float(e.value['eig_min']):>8.3f}  "
              f"{float(e.value['eig_max']):>8.3f}")
    print("  -> eig-ratio stays ~1x from the FIRST window: the metric NEVER moved off")
    print("     identity. Visible in one run — this bug sat parked for 7 weeks.")

    # ---------------- verdict ----------------
    be = jnp.sort(jnp.real(jnp.linalg.eigvals(m_buggy)))
    fe = jnp.linalg.eigvalsh(m_fixed)  # cov_x IS symmetric
    print("\nfinal metrics (true eig-ratio to learn: 100x):")
    print(f"  buggy: eigs [{float(be[0]):.3f}, {float(be[-1]):.3f}] "
          f"ratio {float(be[-1] / be[0]):.1f}x  (identity-shaped)")
    print(f"  fixed: eigs [{float(fe[0]):.3f}, {float(fe[-1]):.3f}] "
          f"ratio {float(fe[-1] / fe[0]):.1f}x  (learned the target)")

    # Verdicts by eigenvalue RATIO (robust to finite-sample noise):
    # the true covariance has ratio 100; an identity-shaped metric has ~1.
    be = jnp.sort(jnp.real(jnp.linalg.eigvals(m_buggy)))
    ratio_b = float(be[-1] / be[0])
    ratio_f = float(fe[-1] / fe[0])
    stuck = ratio_b < 5.0            # buggy: never learned the 100x spread
    learned = ratio_f > 25.0         # fixed: clearly did
    tapped = bool(win) and all(
        float(e.value["eig_max"]) / max(float(e.value["eig_min"]), 1e-6) < 6.0
        for e in win if e.step > 0
    )
    print("\n" + "=" * 70)
    print(f"RESULT: tap exposes 'metric never moved off identity' in one run "
          f"[{'PASS' if stuck and tapped else 'FAIL'}]")
    print(f"        fixed pipeline learns the true covariance "
          f"[{'PASS' if learned else 'FAIL'}]")


if __name__ == "__main__":
    main()
