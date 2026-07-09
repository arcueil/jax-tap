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

"""Execution hidden inside "compile time".

BUG: the first call of a jitted function pays trace + compile +
execute in one opaque block. Naive profiling ("the first call is compilation")
attributes ALL of it to compilation — and on asynchronously-dispatching
backends the same conflation smears execution into whatever phase happens to
block (see https://docs.jax.dev/en/latest/async_dispatch.html). Real episodes
have hidden minutes of execution inside a reported "tracing" number,
misdirecting the optimization effort entirely.
HARD TO DETECT: the first call is one indivisible wall-time measurement —
nothing in user code marks where compilation ends and execution begins.
WITH TAP: tap events are emitted by the RUNNING program, so the FIRST event's
arrival timestamp IS the compile/execute boundary. One tap splits the opaque
first-call window into (trace+compile) vs (execution) — which doubles as a
FEATURE: from a single first call you get the true compile cost AND a free
forecast of steady-state runtime (the execution part), before ever running
the compiled program again. NOTE: the ideal is a dedicated jit-event tap
class (trace/compile/execute timestamps — roadmap); event-arrival timing is
the shipped approximation.

Run:  uv run python demo/async_dispatch_compile_blowup.py
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp

import jaxtap as tap

N_STEPS = 4_000_000
DIM = 16


def make_model():
    def body(c, _):
        # ╔═ jax-tap virtual injection ══════════════════════════════════════╗
        # ║ (any tap here doubles as an execution heartbeat: its arrival     ║
        # ║  timestamp proves the program is EXECUTING, not compiling)       ║
        # ╚═ fires at this return, gated by sample_every ════════════════════╝
        return c * 0.99999 + jnp.sin(c) * 1e-5, None

    def run(x):
        c, _ = jax.lax.scan(body, x, None, length=N_STEPS)
        return c

    return run


def main() -> None:
    x = jnp.ones(DIM)

    # ---------------- without jax-tap: the naive first-call profile ----------------
    run = make_model()
    t0 = time.perf_counter()
    jax.block_until_ready(jax.jit(run)(x))
    t1 = time.perf_counter()
    first_call = t1 - t0
    print(
        f"without jax-tap: first call took {first_call:.2f}s — naive profiling "
        f"reports\n  'compilation: {first_call:.2f}s' and moves on to optimize tracing."
    )

    # ---------------- with jax-tap: split the opaque window ----------------
    arrivals = []
    g = jax.jit(
        tap.verbose(
            make_model(),
            on_step=lambda e: arrivals.append(time.perf_counter()),
            sample_every=500_000,
            select=lambda leaves: (),
        )
    )
    T0 = time.perf_counter()
    jax.block_until_ready(g(x))
    T1 = time.perf_counter()
    compile_part = arrivals[0] - T0
    exec_part = T1 - arrivals[0]
    hidden = exec_part / (T1 - T0)

    # cross-check against the SAME (tapped) program's steady-state execution —
    # the tap's se-gate adds a small per-step cost, so tapped-vs-tapped is the
    # honest comparison; the BOUNDARY location is what transfers to your code.
    t2 = time.perf_counter()
    jax.block_until_ready(g(x))
    steady_exec = time.perf_counter() - t2

    print("\nwith jax-tap (first event = compile/execute boundary):")
    print(f"  trace+compile : {compile_part:.2f}s")
    print(f"  execution     : {exec_part:.2f}s  <- was HIDDEN inside 'compilation'")
    print(f"  ({hidden:.0%} of the reported first-call number was actually execution)")
    print(f"  -> free forecast from ONE call: steady-state runtime ~{exec_part:.2f}s")
    print(f"     (validated: the actual second run measures {steady_exec:.2f}s)")

    ok = (abs(exec_part - steady_exec) / steady_exec < 0.5) and hidden > 0.3
    print(
        f"\nRESULT: tap arrival time splits compile vs execution " f"[{'PASS' if ok else 'FAIL'}]"
    )


if __name__ == "__main__":
    main()
