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

"""The float32 Cholesky trap.

BUG: a float32 Cholesky silently produces a non-finite factor once the matrix
becomes ill-conditioned; step-size adaptation "dodges" the NaN, so the loop
completes and *looks* converged.
HARD TO DETECT: nothing raises. The only symptom is a frozen step size —
diagnostics (R-hat, divergences) still look fine. Real episodes of this class
cost days.
WITH TAP: ``tap.watch_nan("cholesky")`` announces the first bad factor live,
mid-loop, with zero changes to the sampler.

Run:  uv run python demo/cholesky_float32_trap.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import jaxtap as tap


def make_sampler(n_steps: int):
    def step(carry, _):
        log_step, k = carry
        # conditioning worsens, then plateaus at lambda_min = 1e-12:
        # below float32 eps (~1.2e-7), far above float64 eps (~2.2e-16).
        c = 1.0 - 10.0 ** (-jnp.minimum(k, 12.0))
        M = jnp.array([[1.0, c], [c, 1.0]], dtype=c.dtype)
        L = jnp.linalg.cholesky(M)  # <-- BUG LIVES HERE: silently non-finite in f32
        # ╔═ jax-tap virtual injection ════════════════════════════════════╗
        # ║ if not isfinite(L).all(): print(step, "NaN/Inf")               ║
        # ╚═ fires as if written HERE — this function is never edited ═════╝
        logdens = -0.5 * 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        new_log_step = jnp.where(jnp.isfinite(logdens), log_step + 0.05, log_step - 1.0)
        return (new_log_step, k + 1.0), logdens

    def run(log_step0):
        (log_step, _), _ = jax.lax.scan(step, (log_step0, 1.0), None, length=n_steps)
        return log_step

    return run


def run_demo(x64: bool):
    jax.config.update("jax_enable_x64", x64)
    run = make_sampler(25)
    x0 = jnp.asarray(0.0, dtype=jnp.float64 if x64 else jnp.float32)

    final = float(run(x0))  # without tap: completes "fine"

    with tap.record(taps=[tap.watch_nan("cholesky", once=True)]) as rec:
        run(x0)  # unmodified call — delete this `with` block after debugging

    bad = [e.step for e in rec.events if "cholesky" in e.path and not bool(e.value)]
    return final, (min(bad) if bad else None)


def main() -> None:
    print("float32:", end=" ")
    final32, bad32 = run_demo(x64=False)
    print(f"loop 'completed' (frozen log-step {final32:.1f}); "
          f"tap caught first bad cholesky at step {bad32}/25 (live line above)")

    print("float64:", end=" ")
    final64, bad64 = run_demo(x64=True)
    print("no non-finite step — the trap is float32-specific"
          if bad64 is None else f"bad at {bad64}??")

    ok = bad32 is not None and bad64 is None
    print(f"\nRESULT: silent NaN localized live, zero code changes "
          f"[{'PASS' if ok else 'FAIL'}]")


if __name__ == "__main__":
    main()
