import jax
import jax.numpy as jnp

import blackjax  # noqa: F401
from blackjax.progress_bar import progress_bar


def raw_loss(theta):
    def body(carry, x):
        new_carry = jnp.sin(carry * theta + x)
        return new_carry, new_carry

    final, _ = jax.lax.scan(body, 0.0, jnp.arange(20.0))
    return final**2


remat_loss = jax.checkpoint(raw_loss)


def count(label, fn, *args):
    calls = []
    with progress_bar(label=label) as state:
        orig_cb = state._step_callback

        def counting_cb(idx):
            calls.append(int(idx))
            orig_cb(idx)

        state._step_callback = counting_cb
        out = fn(*args)
        jax.block_until_ready(out)
    print(f"{label}: n_steps={state.n_steps} calls={len(calls)} out={out}")
    return calls


count("plain-forward-no-checkpoint", raw_loss, 0.7)
count("checkpoint-forward-only-no-grad", remat_loss, 0.7)
count("checkpoint-forward-jitted-no-grad", jax.jit(remat_loss), 0.7)
count("checkpoint-under-grad", lambda t: jax.grad(remat_loss)(t), 0.7)
count("checkpoint-under-jit-grad", jax.jit(lambda t: jax.grad(remat_loss)(t)), 0.7)
