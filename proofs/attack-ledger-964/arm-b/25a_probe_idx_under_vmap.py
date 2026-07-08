"""Probe: under jax.vmap, what does `idx` actually look like when it
reaches _step_callback? If jax.debug.callback's vmap rule delivers a
batched (size>1) array instead of looping per-element, `int(idx)` would
raise TypeError."""
import jax
import jax.numpy as jnp

import blackjax
from blackjax.util import run_inference_algorithm
from tests.fixtures import std_normal_logdensity

observed = []

algorithm = blackjax.hmc(
    std_normal_logdensity, step_size=0.1, inverse_mass_matrix=jnp.eye(2),
    num_integration_steps=5,
)

def run(key, position):
    return run_inference_algorithm(
        rng_key=key, inference_algorithm=algorithm, num_steps=10,
        initial_position=position,
    )

keys = jax.random.split(jax.random.key(0), 3)
positions = jnp.zeros((3, 2))

with blackjax.progress_bar(label="vmap-probe") as state:
    orig_cb = state._step_callback
    def probing_cb(idx):
        observed.append((type(idx).__name__, getattr(idx, "shape", None), getattr(idx, "dtype", None)))
        orig_cb(idx)
    state._step_callback = probing_cb

    (final_state, _) = jax.vmap(run)(keys, positions)
    jax.block_until_ready(final_state)

print("number of callback invocations:", len(observed))
print("distinct (type, shape, dtype) tuples observed:", set(observed))
print("any shape with size > 1 (would break int(idx))?",
      any(s is not None and getattr(s, "__len__", lambda: 0)() > 0 and
          __import__("math").prod(s) > 1 for _, s, _ in observed))
