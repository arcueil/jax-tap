"""demo — the float32 Cholesky trap (GP regression, 2026-05-12).

ORIGINAL BUG (class: primitive / silent-NaN)
--------------------------------------------
A Gaussian-process log-density did a Cholesky of a kernel matrix in float32.
As warmup drove the length-scale into an ill-conditioned regime, the float32
Cholesky silently produced a NON-FINITE factor -> NaN log-density -> NaN
gradient. Dual-averaging step-size adaptation "dodged" the NaN by shrinking the
step size toward zero, so the chain FROZE in place. Every finite diagnostic
(R-hat approx 1, no divergences) said "converged." Cost: a multi-day probe
cascade, because the failure was invisible from the outside — the run *looked*
healthy.

THE TAP THAT WOULD HAVE CAUGHT IT
---------------------------------
A per-step tap that reduces the Cholesky factor to a single finite/not-finite
bool on-device (reduce-on-device rule) and ships it to the host. The first step
whose factor is non-finite is the bug, localized instantly — instead of a
multi-day cascade.

WHAT THIS DEMO SHOWS
--------------------
1. A pure-JAX "sampler loop" (`lax.scan`) whose per-step log-density needs a
   Cholesky of an increasingly ill-conditioned 2x2 matrix.
2. WITHOUT jaxtap: the loop finishes, the step size has frozen, and a naive
   "did it run to completion?" check passes — the NaN is invisible.
3. WITH jaxtap (`tap.verbose` + a `select` that ships only `isfinite(L)` and the
   frozen step size): the exact first-bad step is localized from the recorder.
4. The trap is FLOAT32-specific: rerun under float64 and the first-bad step moves
   far later (or never within the horizon) — this is why it hid in production.

Run:  uv run python demo/cholesky_float32_trap.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import jaxtap as tap


def make_sampler(n_steps: int):
    """A toy 'sampler loop': each step Choleskys a 2x2 kernel whose conditioning
    worsens geometrically, then does a dual-averaging-style step-size update that
    freezes when the log-density is non-finite (mimicking the real DA dodge)."""

    def step(carry, _):
        log_step, k = carry
        # Correlation c_k -> 1 geometrically: lambda_min(M) = 1 - c_k = 10**(-k).
        # In float32, once 10**(-k) drops below eps (~1.2e-7, around k=7), c_k
        # rounds to exactly 1.0 -> M is numerically singular -> Cholesky factor
        # non-finite. In float64 the same happens far later (~k=16).
        c = 1.0 - 10.0 ** (-k)
        M = jnp.array([[1.0, c], [c, 1.0]], dtype=c.dtype)
        L = jnp.linalg.cholesky(M)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        logdens = -0.5 * logdet  # stand-in GP log-density

        finite = jnp.isfinite(logdens)
        # DA "dodge": grow the step while healthy, collapse it on a non-finite value.
        new_log_step = jnp.where(finite, log_step + 0.05, log_step - 1.0)
        return (new_log_step, k + 1.0), logdens

    def run(log_step0):
        (log_step, _), _ = jax.lax.scan(step, (log_step0, 1.0), None, length=n_steps)
        return log_step  # "final tuned step size" — the only thing a user inspects

    return run


def cholesky_factor_finite(carry):
    """select: reduce the per-step state to what crosses the host boundary.
    Recomputes L cheaply on-device and ships ONE bool + the step size."""
    log_step, k = carry
    c = 1.0 - 10.0 ** (-k)
    M = jnp.array([[1.0, c], [c, 1.0]], dtype=c.dtype)
    L = jnp.linalg.cholesky(M)
    return {"factor_finite": jnp.all(jnp.isfinite(L)), "log_step": log_step}


def run_demo(x64: bool) -> dict:
    jax.config.update("jax_enable_x64", x64)
    n_steps = 25
    run = make_sampler(n_steps)
    log_step0 = jnp.asarray(0.0, dtype=jnp.float64 if x64 else jnp.float32)

    # --- WITHOUT jaxtap: what a user sees ---
    final_log_step = float(run(log_step0))

    # --- WITH jaxtap: localize the first non-finite Cholesky step ---
    # Uses rec.events directly (no pandas needed); rec.df() is the pandas view.
    g, rec = tap.record(run, select=cholesky_factor_finite)
    jax.block_until_ready(g(log_step0))
    first_bad = next(
        (e.step for e in sorted(rec.events, key=lambda e: e.step)
         if not bool(e.value["factor_finite"])),
        None,
    )
    return {"final_log_step": final_log_step, "first_bad_step": first_bad, "n_steps": n_steps}


def main() -> None:
    print(__doc__)
    print("=" * 70)

    r32 = run_demo(x64=False)
    print("\nFLOAT32 (production default):")
    print(f"  what the user sees:  final tuned log-step = {r32['final_log_step']:.3f}")
    print("    -> the loop 'completed'; a large-negative frozen step size is the")
    print("       only clue, and R-hat/divergence checks would still look fine.")
    if r32["first_bad_step"] is not None:
        print(f"  jaxtap CAUGHT IT:    first non-finite Cholesky at step "
              f"{r32['first_bad_step']} / {r32['n_steps']}")
    else:
        print("  jaxtap: no non-finite step within horizon")

    r64 = run_demo(x64=True)
    print("\nFLOAT64 (the fix):")
    if r64["first_bad_step"] is not None:
        print(f"  first non-finite Cholesky at step {r64['first_bad_step']} / {r64['n_steps']}")
    else:
        print(f"  NO non-finite step within {r64['n_steps']} steps — the trap is gone.")

    print("\n" + "=" * 70)
    caught = r32["first_bad_step"] is not None
    moved = (r64["first_bad_step"] is None) or (
        r32["first_bad_step"] is not None and r64["first_bad_step"] > r32["first_bad_step"]
    )
    print(f"RESULT: jaxtap localized the silent float32 NaN at its first-bad step "
          f"[{'PASS' if caught else 'FAIL'}]")
    print(f"        the trap is float32-specific (float64 defers/avoids it) "
          f"[{'PASS' if moved else 'FAIL'}]")


if __name__ == "__main__":
    main()
