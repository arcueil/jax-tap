"""The inner loop that quit early.

BUG: an inner optimizer buried inside an outer sampling loop hits its
iteration cap without converging. The curvature evaluated at the non-converged
exit point is inflated by an order of magnitude, silently collapsing the outer
step size.
HARD TO DETECT: every value is finite, nothing raises, and the failing
``while_loop`` sits levels deep (the real episode: solver inside a Laplace
approximation inside an integrator inside a sampler scan — four levels down).
The outer loop just gets mysteriously slow.
WITH TAP: the inner solver's own carry already holds (iterations, gradient) —
a ``where``-targeted carry tap streams its exit state per outer step, with the
address (``scan[0]/while[0]``) showing exactly WHERE, and zero changes to any
level of the nest.

Run:  uv run python demo/lbfgs_maxiter_curvature.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import jaxtap as tap

MAXITER = 30
TOL = 1e-3
N_OUTER = 12


def inner_solve(m, z0):
    """Damped-Newton minimization of f(z) = cosh(z - m); curvature f'' = cosh."""

    def cond(c):
        z, it, g = c
        return (jnp.abs(g) > TOL) & (it < MAXITER)  # <-- the iteration cap

    def body(c):
        z, it, _ = c
        z = z - 0.6 * jnp.tanh(z - m)  # damped Newton step (bounded move)
        # the while-loop tap acts AS IF we injected `print(it, |grad|)` right
        # here, every iteration, four levels deep — without editing anything.
        return (z, it + 1, jnp.sinh(z - m))

    z, it, g = jax.lax.while_loop(cond, body, (z0, 0, jnp.sinh(z0 - m)))
    # <-- BUG LIVES HERE: if the cap was hit, z is far from m and the
    #     curvature cosh(z - m) is INFLATED (cosh(0)=1 at the true optimum).
    return z, jnp.cosh(z - m)


def sampler(targets):
    """Outer loop: warm-started inner solves; step size from curvature."""

    def step(carry, m):
        z, _ = carry
        z, curvature = inner_solve(m, z)
        eps = 1.0 / jnp.sqrt(curvature)  # collapses when curvature inflates
        return (z, eps), eps

    (_, _), eps_trace = jax.lax.scan(step, (jnp.float32(0.0), 1.0), targets)
    return eps_trace


def exit_state(leaves):
    """Carry-tap select for the inner solver: ship (iterations, |grad|) —
    two scalars per iteration. `where` below targets ONLY the while loop,
    so this select never sees any other carry."""
    z, it, g = leaves
    return {"it": it, "g": jnp.abs(g)}


def main() -> None:
    # target drifts gently, then JUMPS mid-run — later solves exceed the cap
    # gentle drift with TWO isolated jumps of 26 (at t=5 and t=9): 30
    # damped-Newton steps (move <= 0.6 each) cover ~18 units, so the cap BINDS
    # on the jump steps only -- the solver exits ~8 units short, curvature
    # cosh(8) inflates ~1500x, then the next warm start recovers. Transient,
    # finite, silent. (Offsets keep every solve nonzero-distance: a solve that
    # needs 0 iterations emits no heartbeat at all.)
    t = jnp.arange(N_OUTER, dtype=jnp.float32)
    targets = (t + 1.0) * 0.8 + jnp.where(t >= 5, 26.0, 0.0) + jnp.where(t >= 9, 26.0, 0.0)

    eps = sampler(targets)  # without tap: all finite, just mysteriously tiny

    with tap.record(select=exit_state, where=lambda p: "while" in p) as rec:
        sampler(targets)  # unmodified — delete `with` after debugging

    # inner heartbeats arrive per iteration; a step-counter reset marks the
    # next outer solve — split into episodes and keep each exit state
    beats = rec.events  # `where` already limited taps to the while loop
    exits, last = [], None
    for e in beats:
        if last is not None and e.step <= last.step:
            exits.append(last)
        last = e
    exits.append(last)

    print(f"inner-solver exit state per outer step (cap={MAXITER}, path "
          f"{beats[0].path}):")
    capped = 0
    for t, e in enumerate(exits):
        hit = e.value["it"] >= MAXITER and e.value["g"] > TOL
        capped += bool(hit)
        flag = "  <-- hit cap, NOT converged (silent)" if hit else ""
        print(f"  outer {t:>2}: {int(e.value['it']):>2} iters, "
              f"|grad| {float(e.value['g']):>7.3f}{flag}")

    print(f"\nstep size: healthy {float(eps[0]):.3f} -> collapsed "
          f"{float(eps.min()):.6f}  ({float(eps[0] / eps.min()):.0f}x smaller, "
          f"all values finite, no warning anywhere)")

    ok = (len(exits) == N_OUTER and capped == 2
          and any(e.value["it"] < MAXITER for e in exits[:4]))
    print(f"\nRESULT: silent maxiter exits localized at their address "
          f"[{'PASS' if ok else 'FAIL'}]")


if __name__ == "__main__":
    main()
