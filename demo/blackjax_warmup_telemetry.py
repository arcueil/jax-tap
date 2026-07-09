"""Watch a real BlackJAX warmup adapt — with zero changes to BlackJAX.

WHAT THIS SHOWS: the flagship use case. `blackjax.window_adaptation` tunes a
NUTS sampler's step size and mass matrix inside a compiled scan — normally a
black box until it returns. One `with tap.record(...)` around the UNMODIFIED
`warmup.run(...)` streams the adaptation state as it evolves: step size,
average acceptance, and the mass matrix, every sampled window. "Why did my
warmup go wrong at step 700?" becomes a question you can answer from data.
HOW: a carry tap's `select` picks the informative leaves of the warmup scan's
carry ON-DEVICE (three scalars + a small vector per sampled step cross to the
host). Leaf indices below were identified for blackjax 1.5 with a one-off
diagnostic select — the flat-leaves contract is the documented boundary; see
demo/README.md.
REQUIRES: blackjax (not a jax-tap dependency).

Run:  uv run --with blackjax python demo/blackjax_warmup_telemetry.py
"""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp

import jaxtap as tap

try:
    import blackjax
except ImportError:
    print("this demo needs blackjax:  uv run --with blackjax python "
          "demo/blackjax_warmup_telemetry.py")
    sys.exit(0)

N_STEPS = 600
EVERY = 100
TARGET_ACC = 0.8

# an anisotropic Gaussian: precisions [1, 10] -> true inverse mass ~ [0.9, 0.1]
PRECISIONS = jnp.array([1.0, 10.0])


def logdensity(x):
    return -0.5 * jnp.sum(x**2 * PRECISIONS)


def adaptation_view(leaves, path=None):
    """Carry-tap `select` for blackjax 1.5's window_adaptation scan.
    Indices found by probing (see docstring): 12 = step size, 6 = the dual-
    averaging error average (target - avg acceptance), 13 = inverse mass
    matrix diagonal. Computed on-device; only these values cross to host."""
    if path != "scan[0]":
        return ()
    return {"step_size": leaves[12],
            "acc_avg": TARGET_ACC - leaves[6],
            "imm": leaves[13]}


def main() -> None:
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity,
                                        target_acceptance_rate=TARGET_ACC)
    key = jax.random.key(0)

    # ╔═ jax-tap virtual injection ════════════════════════════════════════╗
    # ║ print(step_size, acc_avg, inverse_mass_matrix)  — inside blackjax's ║
    # ║ warmup scan, every sampled step, with blackjax UNMODIFIED           ║
    # ╚═ the whole point: this line exists nowhere in blackjax ═════════════╝
    with tap.record(select=adaptation_view, sample_every=EVERY) as rec:
        results, _info = warmup.run(key, jnp.ones(2), num_steps=N_STEPS)
    jax.block_until_ready(jax.tree_util.tree_leaves(results))

    print("blackjax window_adaptation, live from inside the scan:")
    print(f"  {'step':>5}  {'step_size':>9}  {'acc_avg':>7}  imm diagonal")
    rows = sorted((e for e in rec.events if e.path == "scan[0]"), key=lambda e: e.step)
    for e in rows:
        v = e.value
        print(f"  {e.step:>5}  {float(v['step_size']):>9.3f}  "
              f"{float(v['acc_avg']):>7.2f}  "
              f"[{float(v['imm'][0]):.3f}, {float(v['imm'][1]):.3f}]")

    # cross-check the tapped stream against what warmup actually returned
    final = results.parameters
    ret_ss = float(final["step_size"])
    ret_imm = [float(x) for x in final["inverse_mass_matrix"]]
    last = rows[-1].value
    print(f"\nwarmup returned: step_size {ret_ss:.3f}, imm "
          f"[{ret_imm[0]:.3f}, {ret_imm[1]:.3f}]")
    print(f"tap's last window saw:      {float(last['step_size']):.3f}, imm "
          f"[{float(last['imm'][0]):.3f}, {float(last['imm'][1]):.3f}]")

    imm_ok = all(abs(float(last["imm"][i]) - ret_imm[i]) / max(ret_imm[i], 1e-6) < 0.3
                 for i in range(2))
    ok = len(rows) >= 4 and imm_ok
    print(f"\nRESULT: adaptation observed live inside an unmodified blackjax warmup "
          f"[{'PASS' if ok else 'FAIL'}]")


if __name__ == "__main__":
    main()
