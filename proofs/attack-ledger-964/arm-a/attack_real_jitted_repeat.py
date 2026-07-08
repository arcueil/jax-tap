"""The natural performance-optimization pattern: pre-jax.jit the sampling
entry point ONCE (to avoid recompiling for every chain/run), then reuse
that SAME jitted callable across multiple independent progress_bar()
blocks (e.g. looping over chains, or re-running a notebook cell that holds
a module-level jitted function). Does the SECOND (and later) run's bar go
silent?
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


@jax.jit
def sample_chain(key, position):
    final_state, _ = run_inference_algorithm(
        rng_key=key,
        inference_algorithm=algorithm,
        num_steps=100,
        initial_position=position,
    )
    return final_state


def run_one_chain(key, label):
    with blackjax.progress_bar(label=label) as state:
        out = sample_chain(key, jnp.zeros(2))
        jax.block_until_ready(out)
    return state


key1, key2, key3 = jax.random.split(jax.random.key(0), 3)

s1 = run_one_chain(key1, "pre-jit-chain-1")
print(f"chain 1 (first ever call of sample_chain): n_steps={s1.n_steps} current_step={s1.current_step}")
print(f"  s1 state right after its own ctx exit: n_steps={s1.n_steps} current_step={s1.current_step}")

s2 = run_one_chain(key2, "pre-jit-chain-2")
print(f"chain 2 (SAME jitted sample_chain, NEW progress_bar ctx): n_steps={s2.n_steps} current_step={s2.current_step}")
print(f"  DID chain 2 secretly mutate DEAD s1?  s1.n_steps={s1.n_steps} s1.current_step={s1.current_step}")

s3 = run_one_chain(key3, "pre-jit-chain-3")
print(f"chain 3 (SAME jitted sample_chain, NEW progress_bar ctx): n_steps={s3.n_steps} current_step={s3.current_step}")
print(f"  DID chain 3 secretly mutate DEAD s1?  s1.n_steps={s1.n_steps} s1.current_step={s1.current_step}")
print(f"  DID chain 3 secretly mutate DEAD s2?  s2.n_steps={s2.n_steps} s2.current_step={s2.current_step}")
