"""Real-world confirmation: sample TWO independent chains sequentially,
each wrapped in ITS OWN progress_bar() context, reusing the SAME algorithm
object / run_inference_algorithm call signature -- exactly what a user
looping over chains or re-running a notebook cell would naturally do.
"""
import jax
import jax.numpy as jnp

import blackjax
from blackjax.util import run_inference_algorithm


def std_normal_logdensity(x):
    return -0.5 * jnp.sum(x**2)


algorithm = blackjax.nuts(
    std_normal_logdensity, step_size=0.1, inverse_mass_matrix=jnp.eye(2)
)


def run_one_chain(key, label):
    with blackjax.progress_bar(label=label) as state:
        final_state, _ = run_inference_algorithm(
            rng_key=key,
            inference_algorithm=algorithm,
            num_steps=100,
            initial_position=jnp.zeros(2),
        )
        jax.block_until_ready(final_state)
    return state


key1, key2, key3 = jax.random.split(jax.random.key(0), 3)

s1 = run_one_chain(key1, "chain-1")
print(f"chain 1: n_steps={s1.n_steps} current_step={s1.current_step}")

s2 = run_one_chain(key2, "chain-2")
print(f"chain 2 (SAME algorithm/shapes, NEW ctx): n_steps={s2.n_steps} current_step={s2.current_step}")

s3 = run_one_chain(key3, "chain-3")
print(f"chain 3 (SAME algorithm/shapes, NEW ctx): n_steps={s3.n_steps} current_step={s3.current_step}")
