"""AYS (i): control case -- ONE progress_bar() context wraps the WHOLE
chain loop (not a fresh context per chain). Does chain 2's cache-hit
callback fire into the SAME still-live state, restoring the bar via the
step==0 phase-reset heuristic? Also test jax.clear_caches() between
per-chain fresh contexts as a user-side workaround.
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


keys = jax.random.split(jax.random.key(0), 3)

print("=== Case (i)-A: ONE progress_bar() context around the whole loop ===")
calls = []
with blackjax.progress_bar(label="one-ctx-loop") as state:
    orig_cb = state._step_callback

    def counting_cb(idx):
        calls.append(int(idx))
        orig_cb(idx)

    state._step_callback = counting_cb

    for i, key in enumerate(keys):
        out = sample_chain(key, jnp.zeros(2))
        jax.block_until_ready(out)
        print(
            f"  after chain {i}: state.n_steps={state.n_steps} "
            f"state.current_step={state.current_step} total_calls_so_far={len(calls)}"
        )

print("full call sequence length:", len(calls))
print("count of idx==0 occurrences (should be 3, one per chain, if each chain re-fires 0..99):", calls.count(0))
print("first 5 calls of each chain (chunks of 100):", [calls[i * 100 : i * 100 + 5] for i in range(3)])

print()
print("=== Case (i)-B: per-chain FRESH context + jax.clear_caches() between chains ===")


@jax.jit
def sample_chain_b(key, position):
    final_state, _ = run_inference_algorithm(
        rng_key=key,
        inference_algorithm=algorithm,
        num_steps=100,
        initial_position=position,
    )
    return final_state


keys_b = jax.random.split(jax.random.key(1), 3)
for i, key in enumerate(keys_b):
    with blackjax.progress_bar(label=f"clearcache-chain-{i}") as state_b:
        out = sample_chain_b(key, jnp.zeros(2))
        jax.block_until_ready(out)
    print(f"chain {i} (before clear_caches, this call already happened): n_steps={state_b.n_steps}")
    jax.clear_caches()
