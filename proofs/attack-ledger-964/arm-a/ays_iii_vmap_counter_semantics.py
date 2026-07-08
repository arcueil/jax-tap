"""AYS (iii): what does the callback actually RECEIVE under vmap -- a
scalar idx per lane (multiple calls), or a batched array (one call)? This
determines whether a monotonic call-counter would overrun under vmap, as
the TL claims.
"""
import jax
import jax.numpy as jnp

import blackjax  # noqa: F401
from blackjax.progress_bar import progress_bar

received = []


def body(carry, x):
    return carry + x, carry


def run(key):
    return jax.lax.scan(body, 0.0, jnp.arange(5.0))


with progress_bar(label="vmap-shape-probe") as state:
    orig_cb = state._step_callback

    def probing_cb(idx):
        received.append((type(idx).__name__, getattr(idx, "shape", None), idx))
        orig_cb(idx)

    state._step_callback = probing_cb

    keys = jax.random.split(jax.random.key(0), 3)  # 3 chains
    out = jax.vmap(run)(keys)
    jax.block_until_ready(out)

print("n_steps:", state.n_steps)
print("total callback invocations:", len(received))
print("expected if 1-call-per-step (batched idx):", 5)
print("expected if 1-call-per-(chain,step):", 3 * 5)
for r in received:
    print("  received:", r)
