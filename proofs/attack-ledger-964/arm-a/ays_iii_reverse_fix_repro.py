"""Repro for the proposed one-liner fix: display_idx = n-1-idx if reverse
else idx. Simulated by post-processing the raw idx sequence observed from
the CURRENT implementation (proves correctness of the transform without
editing the module)."""
import jax
import jax.numpy as jnp

import blackjax  # noqa: F401
from blackjax.progress_bar import progress_bar

n = 6
raw_calls = []


def body(carry, x):
    return carry + x, carry


with progress_bar(label="reverse-raw", print_rate=1) as state:
    orig_cb = state._step_callback

    def recording_cb(idx):
        raw_calls.append(int(idx))
        orig_cb(idx)

    state._step_callback = recording_cb
    final, ys = jax.lax.scan(body, 0.0, jnp.arange(float(n)), reverse=True)
    jax.block_until_ready(final)

print("raw idx sequence actually delivered by the CURRENT implementation:", raw_calls)
print("current (buggy) final current_step:", state.current_step)

# Now simulate applying the proposed fix as a post-hoc transform on the same
# raw idx stream, to prove correctness of the one-liner before touching the
# module.
reverse = True


class FakeState:
    def __init__(self, n_steps):
        self.n_steps = n_steps
        self.current_step = 0

    def step_callback_fixed(self, raw_idx):
        display_idx = self.n_steps - 1 - raw_idx if reverse else raw_idx
        step = int(display_idx)
        if step > self.current_step or step == 0:
            self.current_step = step
        return step


fake = FakeState(n_steps=n)
displayed = [fake.step_callback_fixed(i) for i in raw_calls]
print("displayed sequence under the fix (should be ascending 0..n-1):", displayed)
print("final current_step under the fix (should be n-1 = %d, i.e. 100%% complete):" % (n - 1), fake.current_step)
print("monotonically ascending:", displayed == sorted(displayed))
print("does NOT spuriously trigger the step==0 phase-reset mid-run:", displayed.count(0) == 1)
