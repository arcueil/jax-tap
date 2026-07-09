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

"""Eight chains, one progress bar — vmap and the tap duality.

WHAT THIS SHOWS: jax-tap under `jax.vmap` follows one rule with two useful
faces. Whether a tap fires once or per-lane depends on whether the value it
ships is BATCHED:
  - `select=lambda _: ()` ships only the step counter — identical across
    lanes (unbatched) -> the callback fires ONCE per sampled step. Eight
    chains, ONE clean progress bar.
  - `select` on the carry ships batched values -> the callback fires PER
    LANE. Eight chains, eight telemetry streams (per-chain statistics).
Same mechanism; `select` chooses which face you get.
HARD PART IT SOLVES: multi-chain progress normally needs the chain code to
cooperate; here the vmapped sampler is UNMODIFIED and the duality gives both
the single bar and the per-chain values for free.

Run:  uv run python demo/vmap_chains_progress.py
"""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp

import jaxtap as tap

N_CHAINS = 8
N_STEPS = 400
EVERY = 40


def make_sampler():
    def step(carry, key):
        x = carry
        x = 0.9 * x + 0.4 * jax.random.normal(key)
        # ╔═ jax-tap virtual injection ═══════════════════════════════════╗
        # ║ print(step)      — unbatched -> fires ONCE per step (the bar)  ║
        # ║ print(x)         — batched   -> fires PER CHAIN (telemetry)    ║
        # ╚═ same return, two faces; `select` picks which value ships ═════╝
        return x, None

    def run(x0, keys):
        x, _ = jax.lax.scan(step, x0, keys)
        return x

    return run


def main() -> None:
    run = make_sampler()
    keys = jax.random.split(jax.random.key(8), (N_CHAINS, N_STEPS))
    x0 = jnp.zeros(N_CHAINS)

    # ---------------- face 1: ONE progress bar for 8 chains ----------------
    ticks = []

    def bar(e: tap.TapEvent) -> None:
        ticks.append(e.step)
        n = int((e.step + 1) / N_STEPS * 40)
        sys.stderr.write(f"\r8 chains [{'#' * n}{'.' * (40 - n)}] step {e.step + 1}/{e.total}")

    with tap.record(select=lambda _: (), sample_every=EVERY, on_step=bar):
        out = jax.vmap(run)(x0, keys)  # UNMODIFIED vmapped sampler
    sys.stderr.write("\n")
    n_bar_events = len(ticks)

    # ---------------- face 2: per-chain telemetry from the same run ----------------
    with tap.record(select=lambda leaves: leaves[0], sample_every=EVERY) as rec:
        jax.vmap(run)(x0, keys)
    n_tel_events = len([e for e in rec.events if e.path == "scan[0]"])
    finals = [round(float(e.value), 2) for e in rec.events][-N_CHAINS:]

    print(
        f"face 1 (select=()): {n_bar_events} events for {N_CHAINS} chains x "
        f"{N_STEPS // EVERY} sampled steps -> ONE bar, lane-independent"
    )
    print(
        f"face 2 (select=carry): {n_tel_events} events "
        f"(= {N_CHAINS} chains x {N_STEPS // EVERY}) -> per-chain values,"
    )
    print(f"  e.g. last sampled window per chain: {finals}")

    ok = (
        n_bar_events == N_STEPS // EVERY
        and n_tel_events == N_CHAINS * (N_STEPS // EVERY)
        and out.shape == (N_CHAINS,)
    )
    print(
        f"\nRESULT: vmap duality — one bar (unbatched) AND per-chain telemetry "
        f"(batched) from the same unmodified sampler [{'PASS' if ok else 'FAIL'}]"
    )


if __name__ == "__main__":
    main()
