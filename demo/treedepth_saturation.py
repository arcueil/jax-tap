"""The saturated tree depth nobody was watching.

BUG: a NUTS-style sampler caps its trajectory doubling at MAX_TREEDEPTH. In a
stiff region of state space the dynamics NEED more doublings — the cap binds,
silently. Every draw still returns; summary diagnostics look fine; the
sampler just quietly stops exploring properly in exactly the region where it
matters.
HARD TO DETECT: saturation is a PER-DRAW event. Means, R-hats, and acceptance
summaries wash it out; unless something watches every draw's tree depth, the
signature can go unnoticed for months.
WITH TAP: the sampler's carry keeps the last tree depth (as real NUTS info
does) — a carry tap becomes a live tripwire the moment depth == MAX, plus a
post-hoc saturation fraction from the same stream.

Run:  uv run python demo/treedepth_saturation.py
"""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp

import jaxtap as tap

MAX_TREEDEPTH = 10
N_DRAWS = 2000


def make_sampler(n_draws: int):
    """A toy NUTS-ish driver. Per draw: position evolves as a slow AR(1) with
    occasional excursions into the tails; the doublings the trajectory needs
    grow with |x| (stiff tails). In real NUTS `depth` is the doubling loop's
    exit count — here its essence, capped identically."""

    def draw(carry, key):
        x, _ = carry
        x = 0.97 * x + 0.35 * jax.random.normal(key)
        needed = jnp.ceil(jnp.log2(1.0 + jnp.abs(x) * 3.0)) + 6.0
        depth = jnp.minimum(needed, MAX_TREEDEPTH)  # <-- BUG LIVES HERE: the
        #     cap binds silently; the draw returns either way.
        # ╔═ jax-tap virtual injection ════════════════════════════════════╗
        # ║ print(depth)  — the carry this body RETURNS, every draw         ║
        # ╚═ fires at this return; the sampler is never edited ═════════════╝
        return (x, depth), x

    def run(x0, keys):
        _, xs = jax.lax.scan(draw, (x0, 0.0), keys)
        return xs

    return run


def main() -> None:
    keys = jax.random.split(jax.random.key(6), N_DRAWS)
    run = make_sampler(N_DRAWS)

    # ---------------- without jax-tap: looks fine ----------------
    xs = run(jnp.float32(0.0), keys)
    print(f"without jax-tap: {N_DRAWS} draws completed, mean |x| = "
          f"{float(jnp.abs(xs).mean()):.2f}, no warnings anywhere.")

    # ---------------- with jax-tap: live tripwire + post-hoc fraction ----------------
    tripped = []

    def tripwire(e: tap.TapEvent) -> None:
        if e.value >= MAX_TREEDEPTH and not tripped:
            tripped.append(e.step)
            sys.stderr.write(f"[tap] FAIL {e.path} {e.step}/{e.total}: "
                             f"treedepth=={MAX_TREEDEPTH} (saturated)\n")

    with tap.record(select=lambda leaves: leaves[1], on_step=tripwire) as rec:
        run(jnp.float32(0.0), keys)  # unmodified — delete `with` after debugging

    depths = jnp.array([float(e.value) for e in rec.events if e.path == "scan[0]"])
    sat = depths >= MAX_TREEDEPTH
    frac = float(sat.mean())
    mean_depth = float(depths.mean())
    # longest consecutive saturated stretch (the 'stuck in the tail' signature)
    runs, cur = 0, 0
    for s in [bool(v) for v in sat]:
        cur = cur + 1 if s else 0
        runs = max(runs, cur)

    print(f"\nwith jax-tap: first saturation announced live at draw {tripped[0] if tripped else '—'}")
    print(f"  mean depth {mean_depth:.1f} (innocuous) — but {frac:.0%} of draws SATURATED,")
    print(f"  including a {runs}-draw consecutive stretch (an excursion the sampler")
    print("  could not explore at the depth it needed).")

    ok = tripped and 0.03 < frac < 0.6 and mean_depth < MAX_TREEDEPTH - 1 and runs >= 10
    print(f"\nRESULT: per-draw tripwire + saturation fraction from one tapped stream "
          f"[{'PASS' if ok else 'FAIL'}]")


if __name__ == "__main__":
    main()
