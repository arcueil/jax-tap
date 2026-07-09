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

"""The acceptance statistic that was secretly bimodal.

BUG: a step-size controller (dual-averaging) tunes toward a target MEAN
acceptance of 0.8 — but the per-step acceptance it consumes is secretly
bimodal: trajectories either fly (~0.95) or fail (~0.02), with almost nothing
in between. The mean can sit near the target while describing NO actual step.
HARD TO DETECT: the controller runs, the mean acceptance looks close to
target, nothing errors — the step size just oscillates forever and sampling
quality silently degrades. Only the DISTRIBUTION of the per-step values gives
it away, and nobody looks at a distribution the code never surfaces.
WITH TAP: the controller's own carry holds the last acceptance — a carry tap
streams it, and a five-bucket histogram of the collected values shows the
{~0, ~0.95} split on sight.

Run:  uv run python demo/multinomial_da_bimodal.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import jaxtap as tap

N_STEPS = 2000
TARGET = 0.8


def make_controller(n_steps: int):
    """A toy dual-averaging driver. Per step: run a 'trajectory' whose
    acceptance depends on the current step size vs the trajectory's own
    tolerance (two populations: easy / hard), then adapt the step size toward
    TARGET mean acceptance. The carry keeps the last acceptance — as real
    controller states do."""

    def step(carry, key):
        log_eps, _ = carry
        eps = jnp.exp(log_eps)
        # two trajectory populations: 45% 'hard' (tolerate eps < 0.25),
        # 55% 'easy' (tolerate eps < 1.2)
        hard = jax.random.bernoulli(key, 0.45)
        tolerance = jnp.where(hard, 0.25, 1.2)
        # <-- BUG LIVES HERE: acceptance is effectively binary per trajectory
        #     ({~0.95, ~0.02}); the controller below only ever sees its MEAN.
        accept = jnp.where(eps < tolerance, 0.95, 0.02)
        log_eps = log_eps + 0.3 * (accept - TARGET)  # dual-averaging-ish
        # ╔═ jax-tap virtual injection ═════════════════════════════════════╗
        # ║ print(accept)  — the carry this body RETURNS, every step        ║
        # ╚═ fires at this return; the controller is never edited ══════════╝
        return (log_eps, accept), log_eps

    def run(keys):
        (log_eps, _), eps_trace = jax.lax.scan(step, (jnp.log(0.5), 0.0), keys)
        return jnp.exp(eps_trace)

    return run


def main() -> None:
    keys = jax.random.split(jax.random.key(5), N_STEPS)
    run = make_controller(N_STEPS)

    # ---------------- without jax-tap: looks almost fine ----------------
    eps_trace = run(keys)
    tail = eps_trace[N_STEPS // 2 :]
    swing = float(tail.max() / tail.min())

    # ---------------- with jax-tap ----------------
    with tap.record(select=lambda leaves: {"accept": leaves[1]}) as rec:
        run(keys)  # unmodified — delete `with` after debugging

    acc = jnp.array([float(e.value["accept"]) for e in rec.events if e.path == "scan[0]"])
    mean_acc = float(acc.mean())
    print(f"without jax-tap: mean acceptance {mean_acc:.2f} (target {TARGET}) — 'close enough!'")
    print(f"  but the tuned step size never settles: max/min over the last half = {swing:.1f}x")

    edges = [0.0, 0.1, 0.3, 0.7, 0.9, 1.0]
    counts = [int(((acc >= lo) & (acc < hi)).sum()) for lo, hi in zip(edges[:-1], edges[1:])]
    print("\nwith jax-tap, histogram of the per-step acceptance the controller consumed:")
    for (lo, hi), c in zip(zip(edges[:-1], edges[1:]), counts):
        bar = "#" * (c // 25)
        print(f"  [{lo:.1f},{hi:.1f}): {c:>5}  {bar}")
    print("  -> BIMODAL: the 'mean acceptance' near target describes NO actual step;")
    print("     mean-targeting dual averaging hunts forever between the two modes.")

    lo_frac = counts[0] / len(acc)
    hi_frac = counts[-1] / len(acc)
    mid_frac = sum(counts[1:-1]) / len(acc)
    bimodal = lo_frac > 0.1 and hi_frac > 0.5 and mid_frac < 0.02
    mean_near = abs(mean_acc - TARGET) < 0.25
    print(f"\nRESULT: bimodality visible on sight from the tapped stream "
          f"[{'PASS' if bimodal else 'FAIL'}]")
    print(f"        while the mean looked plausible ({mean_acc:.2f}) and eps swung {swing:.1f}x "
          f"[{'PASS' if mean_near and swing > 1.5 else 'FAIL'}]")


if __name__ == "__main__":
    main()
