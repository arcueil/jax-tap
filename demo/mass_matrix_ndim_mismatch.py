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

"""The dense config that silently ran diagonal.

BUG: a sampler kernel dispatches on the mass matrix's ndim (2-D = dense
algorithm, 1-D = diagonal). A plumbing bug hands it a 1-D matrix even though
the user CONFIGURED dense — so the kernel silently runs the wrong algorithm.
HARD TO DETECT: nothing errors, nothing is non-finite; samples still flow.
The only symptom is quietly degraded mixing on correlated targets — a
statistics-level bias that can take a multi-day investigation to trace back
to a shape.
WITH TAP: a Python-level `if/else` (like the ndim dispatch) runs at TRACE
time — only the taken branch is baked into the compiled program; the other
path simply does not exist in the jaxpr. That is exactly why this class is
hard to debug at runtime. But the baked-in branch leaves a fingerprint: the
dense path contains a `dot_general`, the diagonal path only a `mul`.
`tap.primitives()` reads the traced jaxpr directly, so if you know the op's
name (`dot_general` here), checking whether that code path was baked in at
all is a one-liner — trace time, zero runtime cost. And
`tap.print("dot_general", once=True)` confirms it live — SILENCE on the buggy
run is itself the symptom. NOTE: the ideal
here is a dedicated trace-time SHAPE tap (roadmap); `tap.primitives()` is the
shipped approximation.

Run:  uv run python demo/mass_matrix_ndim_mismatch.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import jaxtap as tap

DIM = 2
RHO = 0.9
SIGMA = jnp.array([[1.0, RHO], [RHO, 1.0]])
N_STEPS = 2000


def make_sampler(inverse_mass_matrix):
    """A toy preconditioned MALA-ish sampler. The kernel dispatches on the
    mass matrix's ndim — the library convention the bug slips through."""

    def grad_logp(x):
        return -jnp.linalg.solve(SIGMA, x)

    def apply_imm(v):
        if inverse_mass_matrix.ndim == 2:
            return jnp.dot(inverse_mass_matrix, v)  # dense algorithm
        # ╔═ jax-tap virtual injection ════════════════════════════════════╗
        # ║ print(inverse_mass_matrix.ndim)  — which algebra actually ran  ║
        # ╚═ read at TRACE time via tap.primitives(); nothing edited ══════╝
        return inverse_mass_matrix * v  # diagonal algorithm (elementwise)

    def step(carry, key):
        x = carry
        drift = 0.15 * apply_imm(grad_logp(x))
        noise = 0.4 * jax.random.normal(key, (DIM,))
        return x + drift + noise, x

    def run(x0, keys):
        _, xs = jax.lax.scan(step, x0, keys)
        return xs

    return run


def lag1_autocorr(xs):
    u = (xs[:, 0] + xs[:, 1]) / jnp.sqrt(2.0)  # the correlated direction
    u = u - u.mean()
    return float(jnp.dot(u[:-1], u[1:]) / jnp.dot(u, u))


def main() -> None:
    keys = jax.random.split(jax.random.key(4), N_STEPS)
    x0 = jnp.zeros(DIM)

    # inverse mass matrix convention: M^{-1} ~ target covariance SIGMA
    dense_imm = SIGMA                    # what the user CONFIGURED (2-D)
    buggy_imm = jnp.diag(SIGMA)          # <-- BUG LIVES HERE: the plumbing
    #   returns the DIAGONAL (1-D) of the matrix; downstream dispatch-on-ndim
    #   silently switches to the diagonal algorithm.

    # ---------------- without jax-tap: both "work" ----------------
    xs_buggy = make_sampler(buggy_imm)(x0, keys)
    xs_dense = make_sampler(dense_imm)(x0, keys)
    ac_buggy, ac_dense = lag1_autocorr(xs_buggy), lag1_autocorr(xs_dense)
    print(f"without jax-tap: both runs 'work'. lag-1 autocorr: "
          f"configured-dense-but-buggy {ac_buggy:.3f} vs true-dense {ac_dense:.3f}")
    print("  -> a quiet mixing degradation; nothing raised, nothing NaN.")

    # ---------------- with jax-tap: read the primitive fingerprint ----------------
    prims_buggy = tap.primitives(make_sampler(buggy_imm), x0, keys)
    prims_dense = tap.primitives(make_sampler(dense_imm), x0, keys)
    print("\ntap.primitives (trace-time, zero runtime cost):")
    print(f"  buggy:  dot_general={prims_buggy.get('dot_general', 0)}  "
          f"(dense algebra ABSENT — your dense config is not running dense!)")
    print(f"  fixed:  dot_general={prims_dense.get('dot_general', 0)}  (dense algebra present)")

    # Runtime confirmation: tap the DISTINCTIVE primitive (dot_general -- the
    # dense algebra). `mul` would be too generic (PRNG internals are full of
    # them). On the buggy run the tap stays SILENT -- absence IS the symptom.
    print("\nruntime check, tap.print('dot_general', once=True):")
    print("  buggy run:  ", end="", flush=True)
    with tap.record(taps=[tap.print("dot_general", once=True)]):
        make_sampler(buggy_imm)(x0, keys)
    print("(silence -- no dense algebra executed)")
    print("  fixed run:  ", end="", flush=True)
    with tap.record(taps=[tap.print("dot_general", once=True)]):
        make_sampler(dense_imm)(x0, keys)

    caught = prims_buggy.get("dot_general", 0) == 0 and prims_dense.get("dot_general", 0) > 0
    degraded = ac_buggy > ac_dense
    print(f"\nRESULT: ndim mismatch caught from the primitive fingerprint "
          f"[{'PASS' if caught else 'FAIL'}]")
    print(f"        the silent symptom is real (mixing degraded) "
          f"[{'PASS' if degraded else 'FAIL'}]")


if __name__ == "__main__":
    main()
