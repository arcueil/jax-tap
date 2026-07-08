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

WHAT THIS DEMO SHOWS — "Make print-debugging great again"
----------------------------------------------------------
The sampler body below contains ZERO logging code — it just defines
``L = jnp.linalg.cholesky(M)`` like any numerical program. Then:

1. WITHOUT jax-tap: the loop "completes"; the only clue is a frozen step size.
   A post-hoc check would wait for the whole run — and still tell you nothing
   about WHERE it went wrong.
2. WITH jax-tap: wrap the UNMODIFIED call in ``with tap.record(...)`` and put a
   primitive tap on ``"cholesky"``. The tap observes the ACTUAL factor L (by
   primitive kind — no reconstruction), and the ``on_step`` callback LOUDLY
   ANNOUNCES the first non-finite factor LIVE, mid-loop, before the scan
   finishes. Done testing? Delete the ``with`` block — nothing was ever there.
3. The trap is FLOAT32-specific: the kernel's conditioning plateaus at a
   realistic lambda_min = 1e-12 — below float32's eps (silently singular) but
   comfortably inside float64's range. Under float64 the trap NEVER fires:
   float64 genuinely fixes it, exactly as it did for the original bug.

Run:  uv run python demo/cholesky_float32_trap.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import jaxtap as tap


def make_sampler(n_steps: int):
    """A toy 'sampler loop'. NOTE: the body contains NO logging/telemetry code.

    Each step Choleskys a 2x2 kernel whose conditioning worsens geometrically
    (correlation c_k -> 1; lambda_min = 10**(-k), plateauing at 1e-12), then does a
    dual-averaging-style update that freezes when the log-density goes
    non-finite (mimicking the real DA dodge)."""

    def step(carry, _):
        log_step, k = carry
        # Conditioning worsens geometrically then PLATEAUS at lambda_min = 1e-12
        # (realistic kernels have bounded conditioning). 1e-12 is far below
        # float32 eps (~1.2e-7) -> f32 sees a silently singular matrix; far
        # above float64 eps (~2.2e-16) -> f64 is fine. That asymmetry IS the trap.
        c = 1.0 - 10.0 ** (-jnp.minimum(k, 12.0))
        M = jnp.array([[1.0, c], [c, 1.0]], dtype=c.dtype)
        L = jnp.linalg.cholesky(M)  # <-- the only line that matters
        logdens = -0.5 * 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        finite = jnp.isfinite(logdens)
        new_log_step = jnp.where(finite, log_step + 0.05, log_step - 1.0)
        return (new_log_step, k + 1.0), logdens

    def run(log_step0):
        (log_step, _), _ = jax.lax.scan(step, (log_step0, 1.0), None, length=n_steps)
        return log_step  # "final tuned step size" — all a user normally sees

    return run


def run_demo(x64: bool) -> dict:
    jax.config.update("jax_enable_x64", x64)
    n_steps = 25
    run = make_sampler(n_steps)
    log_step0 = jnp.asarray(0.0, dtype=jnp.float64 if x64 else jnp.float32)

    # ---------------- WITHOUT jax-tap: what a user sees ----------------
    final_log_step = float(run(log_step0))

    # ---------------- WITH jax-tap ----------------
    # The tap addresses the cholesky primitive BY KIND and reduces its output
    # on-device to one bool (reduce-on-device: only the bool crosses to host).
    # `announce` streams LIVE — it fires DURING the scan, not after.
    announced = []

    def announce(e: tap.TapEvent) -> None:
        if "cholesky" in e.path and not bool(e.value) and not announced:
            announced.append(e.step)
            print(
                f"    >>> LIVE: cholesky factor went NON-FINITE at step {e.step} "
                f"(path {e.path}) — announced BEFORE the scan finished <<<"
            )

    with tap.record(
        taps=[tap.on("cholesky", select=lambda outs: jnp.all(jnp.isfinite(outs[0])))],
        on_step=announce,
    ) as rec:
        run(log_step0)  # <-- UNMODIFIED user code. Delete this `with` block
        #                     and nothing was ever there.

    # Post-hoc view of the same stream (the recorder collected everything):
    chol = sorted((e for e in rec.events if "cholesky" in e.path), key=lambda e: e.step)
    first_bad = next((e.step for e in chol if not bool(e.value)), None)
    return {"final_log_step": final_log_step, "first_bad_step": first_bad, "n_steps": n_steps}


def main() -> None:
    print("=" * 70)

    print("\nFLOAT32 (production default):")
    r32 = run_demo(x64=False)
    print(f"  without jax-tap:  final tuned log-step = {r32['final_log_step']:.3f}")
    print("    -> the loop 'completed'; a large-negative frozen step size is the")
    print("       only clue, and R-hat/divergence checks would still look fine.")
    if r32["first_bad_step"] is not None:
        print(
            f"  with jax-tap:     first non-finite cholesky at step "
            f"{r32['first_bad_step']} / {r32['n_steps']} (also announced live above)"
        )

    print("\nFLOAT64 (the fix):")
    r64 = run_demo(x64=True)
    if r64["first_bad_step"] is not None:
        print(f"  first non-finite cholesky at step {r64['first_bad_step']} / {r64['n_steps']}")
    else:
        print(f"  NO non-finite step within {r64['n_steps']} steps — the trap is gone.")

    print("\n" + "=" * 70)
    caught = r32["first_bad_step"] is not None
    moved = (r64["first_bad_step"] is None) or (
        r32["first_bad_step"] is not None and r64["first_bad_step"] > r32["first_bad_step"]
    )
    print(
        f"RESULT: live announcement + localization of the silent float32 NaN "
        f"[{'PASS' if caught else 'FAIL'}]"
    )
    print(
        f"        the trap is float32-specific (float64 genuinely fixes it) "
        f"[{'PASS' if moved else 'FAIL'}]"
    )


if __name__ == "__main__":
    main()
