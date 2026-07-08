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


key1, key2, key3 = jax.random.split(jax.random.key(0), 3)

s1_call_count = [0]
with blackjax.progress_bar(label="c1") as state1:
    orig_cb1 = state1._step_callback

    def counting_cb1(idx):
        s1_call_count[0] += 1
        orig_cb1(idx)

    state1._step_callback = counting_cb1
    out1 = sample_chain(key1, jnp.zeros(2))
    jax.block_until_ready(out1)

print("chain1: n_steps=", state1.n_steps, "callback fired", s1_call_count[0], "times")

with blackjax.progress_bar(label="c2") as state2:
    out2 = sample_chain(key2, jnp.zeros(2))
    jax.block_until_ready(out2)

print("chain2 (cache-hit expected): n_steps=", state2.n_steps)
print(
    "did the STALE state1 callback fire again during chain2's cache-hit run?",
    "count now =", s1_call_count[0], "(should stay 100 if it did NOT fire again)",
)

with blackjax.progress_bar(label="c3") as state3:
    out3 = sample_chain(key3, jnp.zeros(2))
    jax.block_until_ready(out3)

print("chain3 (cache-hit expected): n_steps=", state3.n_steps)
print("s1_call_count after chain3 =", s1_call_count[0])
