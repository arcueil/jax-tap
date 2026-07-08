"""Attack: jax.grad through a scan traced inside progress_bar().

Checks:
1. Gradient VALUE is bitwise-identical with/without the context.
2. Whether the debug.callback fires extra times during backward pass
   (double-counting steps -- would corrupt bar display, not the math).
"""
import jax
import jax.numpy as jnp

import blackjax  # noqa: F401  (registers progress_bar on the package)
from blackjax.progress_bar import progress_bar

calls_no_ctx = []
calls_ctx = []


def loss(theta):
    def body(carry, x):
        new_carry = carry * theta + x
        return new_carry, new_carry

    final, _ = jax.lax.scan(body, 0.0, jnp.arange(10.0))
    return final**2


theta0 = 2.0

# Baseline: unpatched.
val0, grad0 = jax.value_and_grad(loss)(theta0)

# Inside progress_bar context.
with progress_bar(label="grad-test") as state:
    orig_cb = state._step_callback

    def counting_cb(idx):
        calls_ctx.append(int(idx))
        orig_cb(idx)

    state._step_callback = counting_cb
    val1, grad1 = jax.value_and_grad(loss)(theta0)
    jax.block_until_ready((val1, grad1))

print("baseline  val,grad:", val0, grad0)
print("in-context val,grad:", val1, grad1)
print("bitwise identical value:", bool(val0 == val1))
print("bitwise identical grad :", bool(grad0 == grad1))
print("num callback fires (n_steps=10):", len(calls_ctx), calls_ctx)
