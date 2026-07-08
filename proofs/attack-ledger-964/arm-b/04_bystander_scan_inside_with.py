"""Attack #1: within a correctly-used `with blackjax.progress_bar(...)`
block, ANY depth-0 jax.lax.scan in the process gets swept in -- not just the
blackjax call. Simulate a user who does some of their own top-level-scan
preprocessing (e.g. a rolling window transform) before invoking blackjax,
all inside the same `with` block (a very natural pattern: 'let me wrap my
whole pipeline in one progress bar').
"""
import jax
import jax.numpy as jnp

import blackjax
from blackjax.util import run_inference_algorithm
from tests.fixtures import std_normal_logdensity

def user_preprocessing(data):
    """Pretend third-party / user preprocessing: a top-level lax.scan the
    user wrote with NO knowledge of blackjax.progress_bar's existence."""
    def body(carry, x):
        return carry + x, carry + x
    _, windowed = jax.lax.scan(body, 0.0, data)
    return windowed

algorithm = blackjax.nuts(
    std_normal_logdensity, step_size=0.1, inverse_mass_matrix=jnp.eye(2)
)

events = []
with blackjax.progress_bar(label="MCMC") as state:
    orig_cb = state._step_callback
    def logging_cb(idx):
        events.append((state.n_steps, int(idx)))
        orig_cb(idx)
    state._step_callback = logging_cb

    # user's own unrelated 300-step preprocessing scan -- nothing to do with
    # sampling -- runs INSIDE the "MCMC" progress-bar context.
    data = user_preprocessing(jnp.arange(300, dtype=jnp.float32))
    jax.block_until_ready(data)

    # NOW the actual, unrelated, 100-step MCMC scan.
    final_state, _ = run_inference_algorithm(
        rng_key=jax.random.key(0),
        inference_algorithm=algorithm,
        num_steps=100,
        initial_position=jnp.zeros(2),
    )
    jax.block_until_ready(final_state)

phases = sorted(set(n for n, _ in events))
print("distinct phase lengths instrumented under the single 'MCMC' bar:", phases)
print("300-step user preprocessing counted as 'MCMC' progress:", 300 in phases)
print("total callback fires:", len(events), "(300 preprocessing + 100 sampling expected)")
print("final state.n_steps (whatever ran LAST wins, regardless of relevance):",
      state.n_steps)
