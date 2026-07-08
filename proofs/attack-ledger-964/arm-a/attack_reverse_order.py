"""Attack: does the progress bar count DOWN (or otherwise misreport order)
under reverse=True, even though the final numeric output is correct?

The augmented xs is (original_xs, jnp.arange(n)); under reverse=True, JAX
scan traverses xs from the END first. Since indices=arange(n) is just
another xs leaf, it gets reversed in lockstep -- so the *value* of idx
delivered to the callback on the FIRST iteration executed would be n-1,
not 0. Check the exact sequence of idx values seen by the callback.
"""
import jax
import jax.numpy as jnp

import blackjax  # noqa: F401
from blackjax.progress_bar import progress_bar

xs = jnp.arange(6.0)


def body(carry, x):
    return carry + x, carry


calls = []
with progress_bar(label="reverse-test", print_rate=1) as state:
    orig_cb = state._step_callback

    def counting_cb(idx):
        calls.append(int(idx))
        orig_cb(idx)

    state._step_callback = counting_cb
    final, ys = jax.lax.scan(body, 0.0, xs, reverse=True)
    jax.block_until_ready(final)

expected_final, expected_ys = jax.lax.scan(body, 0.0, xs, reverse=True)

print("callback idx sequence (execution order):", calls)
print("final matches:", bool(final == expected_final))
print("ys matches:", bool(jnp.allclose(ys, expected_ys)))
print("current_step after run (bar's final displayed value):", state.current_step)
print("n_steps:", state.n_steps)
