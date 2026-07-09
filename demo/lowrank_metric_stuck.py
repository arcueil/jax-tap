"""The metric that never moved.

BUG: a mass-matrix adaptation uses the score covariance WITHOUT inverting it.
For a Gaussian target, cov(x) @ cov(scores) ~ S @ S^{-1} = I — the metric
collapses to identity and learns nothing.
HARD TO DETECT: adaptation runs, returns a metric, raises nothing; the sampler
just mixes badly. Bugs of this shape sit undiagnosed for weeks. Inspired by a
real fix: https://github.com/blackjax-devs/blackjax/pull/949
WITH TAP: a carry-tap ``select`` computes the evolving metric's eigenvalue
range on-device — "learned nothing" is visible in the FIRST window, with zero
changes to the warmup loop.

Run:  uv run python demo/lowrank_metric_stuck.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import jaxtap as tap

DIM, N_STEPS, EVERY = 10, 4000, 500
TRUE_EIGS = jnp.geomspace(0.1, 10.0, DIM)  # target covariance spans 100x
SIGMA = jnp.diag(TRUE_EIGS)


def make_warmup(buggy: bool):
    def step(carry, x):
        s_x, s_s, k = carry
        s = -jnp.linalg.solve(SIGMA, x)  # the model's score
        # ╔═ jax-tap virtual injection ═══════════════════════════════════╗
        # ║ print(eig_range(<the carry this body RETURNS>))                ║
        # ╚═ fires at this return, every sampled step; body never edited ══╝
        return (s_x + jnp.outer(x, x), s_s + jnp.outer(s, s), k + 1.0), None

    def run(draws):
        init = (jnp.zeros((DIM, DIM)), jnp.zeros((DIM, DIM)), 0.0)
        (s_x, s_s, k), _ = jax.lax.scan(step, init, draws)
        cov_x, cov_s = s_x / k, s_s / k
        if buggy:
            return cov_x @ cov_s  # <-- BUG LIVES HERE: cov_s not inverted -> ~I
        return cov_x  # the fix: the position covariance is the learned metric

    return run


def eig_range(leaves):
    """Carry-tap select: metric eig-range, computed ON-DEVICE from the running
    accumulators; only two scalars per sampled step cross to the host."""
    s_x, s_s, k = leaves
    kk = jnp.maximum(k, 1.0)
    m = (s_x / kk) @ (s_s / kk)
    eigs = jnp.sort(jnp.real(jnp.linalg.eigvals(m)))  # product of SPDs: real eigs
    return {"lo": eigs[0], "hi": eigs[-1]}


def main() -> None:
    draws = jax.random.multivariate_normal(
        jax.random.key(0), jnp.zeros(DIM), SIGMA, (N_STEPS,))

    m_buggy = make_warmup(buggy=True)(draws)   # without tap: "succeeds",
    m_fixed = make_warmup(buggy=False)(draws)  # nothing looks wrong

    with tap.record(select=eig_range, sample_every=EVERY) as rec:
        make_warmup(buggy=True)(draws)  # unmodified — delete `with` after debugging

    print("metric eig-range while adapting (true spread: 100x):")
    for e in sorted((e for e in rec.events if e.path == "scan[0]"), key=lambda e: e.step):
        if e.step:
            print(f"  step {e.step:>4}: [{float(e.value['lo']):.2f}, "
                  f"{float(e.value['hi']):.2f}]  ratio ~"
                  f"{float(e.value['hi'] / e.value['lo']):.1f}x")
    print("  -> flat at ~1x from the first window: it is learning NOTHING.")

    be = jnp.sort(jnp.real(jnp.linalg.eigvals(m_buggy)))
    fe = jnp.linalg.eigvalsh(m_fixed)
    rb, rf = float(be[-1] / be[0]), float(fe[-1] / fe[0])
    print(f"\nfinal: buggy ratio {rb:.1f}x (identity-shaped) | "
          f"fixed ratio {rf:.1f}x (learned the target)")
    print(f"\nRESULT: 'adaptation learned nothing' visible in one run "
          f"[{'PASS' if rb < 5.0 and rf > 25.0 else 'FAIL'}]")


if __name__ == "__main__":
    main()
