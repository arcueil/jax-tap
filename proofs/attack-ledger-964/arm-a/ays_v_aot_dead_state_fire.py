"""AYS (v): for jax.jit(f).lower() done INSIDE the ctx, then .compile()+run
AFTER exit -- does the baked-in callback actually FIRE (into the dead
state) at run time, or was it dropped/never wired at compile/lower time?
Settle with a counting probe attached BEFORE exit.
"""
import jax
import jax.numpy as jnp

import blackjax  # noqa: F401
from blackjax.progress_bar import progress_bar

call_count = [0]


def f(x):
    def body(carry, xi):
        return carry + xi, carry

    final, ys = jax.lax.scan(body, x, jnp.arange(10.0))
    return final, ys


with progress_bar(label="lower-test") as state:
    orig_cb = state._step_callback

    def counting_cb(idx):
        call_count[0] += 1
        orig_cb(idx)

    state._step_callback = counting_cb

    lowered = jax.jit(f).lower(0.0)
    compiled = lowered.compile()
    print("compiled INSIDE ctx (not yet run). call_count so far:", call_count[0])

print("ctx exited. call_count so far (should still be 0, nothing executed yet):", call_count[0])
print("state.output_file after exit:", state.output_file)
print("state fields right after exit: n_steps=", state.n_steps, "current_step=", state.current_step)

out = compiled(0.0)
jax.block_until_ready(out)
print("ran compiled executable AFTER ctx exit. result:", out)
print("call_count AFTER post-exit run (did the dead callback fire?):", call_count[0])
print("state.current_step after post-exit run (mutated on the dead object?):", state.current_step)

# run it again to make sure it's not a one-off
out2 = compiled(5.0)
jax.block_until_ready(out2)
print("ran a SECOND time after exit. call_count now:", call_count[0], "result:", out2)
