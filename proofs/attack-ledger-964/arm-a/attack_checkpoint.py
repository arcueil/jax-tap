"""Attack: jax.checkpoint (remat) around a scan-based loss, differentiated.

If the scan body is rematerialized during the backward pass, does the
embedded jax.debug.callback fire again (double-counting steps in the
displayed progress), and does the grad VALUE still match the unpatched
(no remat, no progress_bar) baseline exactly?
"""
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


@jax.checkpoint
def remat_loss(theta):
    return raw_loss(theta)


theta0 = 0.7

val0, grad0 = jax.value_and_grad(raw_loss)(theta0)
val_r0, grad_r0 = jax.value_and_grad(remat_loss)(theta0)
print("no-remat baseline   val,grad:", val0, grad0)
print("remat (no ctx)      val,grad:", val_r0, grad_r0)
print("remat matches no-remat:", bool(val0 == val_r0), bool(grad0 == grad_r0))

calls = []
with progress_bar(label="remat-test") as state:
    orig_cb = state._step_callback

    def counting_cb(idx):
        calls.append(int(idx))
        orig_cb(idx)

    state._step_callback = counting_cb
    val1, grad1 = jax.value_and_grad(remat_loss)(theta0)
    jax.block_until_ready((val1, grad1))

print("remat (in ctx)      val,grad:", val1, grad1)
print("in-ctx matches no-remat baseline:", bool(val0 == val1), bool(grad0 == grad1))
print("callback fire count (n_steps=20, expect 20 if single-fire):", len(calls))
print("raw call sequence:", calls)
