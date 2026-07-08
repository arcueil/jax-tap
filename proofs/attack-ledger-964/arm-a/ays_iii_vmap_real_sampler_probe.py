import jax
import jax.numpy as jnp

import blackjax
from blackjax.util import run_inference_algorithm

received = []


def std_normal_logdensity(x):
    return -0.5 * jnp.sum(x**2)


algorithm = blackjax.hmc(
    std_normal_logdensity,
    step_size=0.1,
    inverse_mass_matrix=jnp.eye(2),
    num_integration_steps=5,
)


def run(key, position):
    return run_inference_algorithm(
        rng_key=key,
        inference_algorithm=algorithm,
        num_steps=8,
        initial_position=position,
    )


with blackjax.progress_bar(label="real-vmap-probe") as state:
    orig_cb = state._step_callback

    def probing_cb(idx):
        received.append((getattr(idx, "shape", None), int(jnp.asarray(idx).max()) if hasattr(idx, "shape") and idx.shape else int(idx)))
        orig_cb(idx)

    state._step_callback = probing_cb

    keys = jax.random.split(jax.random.key(0), 4)  # 4 chains
    positions = jnp.zeros((4, 2))
    (final_state, _) = jax.vmap(run)(keys, positions)
    jax.block_until_ready(final_state)

print("n_steps:", state.n_steps)
print("total callback invocations:", len(received))
print("expected if 1-call-per-step (idx shared/unbatched):", 8)
print("expected if 1-call-per-(chain,step) i.e. batched idx delivered once per lane:", 4 * 8)
for r in received:
    print("  shape,val:", r)
print("final current_step:", state.current_step)
