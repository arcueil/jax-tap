"""Attack #6c: JAX dispatch is asynchronous -- a jit'd call issued inside the
`with` body returns as soon as it is DISPATCHED, not when it COMPLETES. If
the user's own code inside the `with` block doesn't call
block_until_ready() before the block ends (an easy omission -- the 8
existing unit tests all call it explicitly, but real user code often
won't), __exit__ runs immediately: it sets the stop event, joins the
display thread (closing/removing the visible bar), sets output_file=None,
and deletes the file -- all while the traced computation is STILL RUNNING
on the device in the background. Net effect: the bar visually reports
"done" (or just vanishes) while the sampling is still in flight with no
progress indication at all, and any output_file-based external reader
(e.g. progress_reader.py) also loses its signal mid-run.
"""
import threading
import time

import jax
import jax.numpy as jnp

import blackjax

# Deliberately heavy per-step work (repeated matmuls) so that overall
# dispatch clearly outlives the (nearly-instant) Python-level call that
# issues it.
D = 300
N_STEPS = 400

def body(carry, x):
    m = carry
    for _ in range(6):
        m = jnp.tanh(m @ m) * 0.999
    return m, jnp.sum(m)

x0 = jnp.eye(D) * 0.01

callback_timestamps = []
exit_time = {}

with blackjax.progress_bar(label="async", print_rate=1) as state:
    orig_cb = state._step_callback
    def cb(idx):
        callback_timestamps.append((time.monotonic(), int(idx)))
        orig_cb(idx)
    state._step_callback = cb

    t_issue = time.monotonic()
    final, _ = jax.lax.scan(body, x0, jnp.arange(N_STEPS))
    t_returned = time.monotonic()
    print(f"jax.lax.scan(...) call returned after {t_returned - t_issue:.4f}s "
          f"(no block_until_ready called -- may still be dispatching async)")
    # BUG UNDER TEST: no block_until_ready() here -- context exits now.

exit_time["t"] = time.monotonic()
print(f"'with' block exited at t={exit_time['t'] - t_issue:.4f}s after issue")
print("state._display_thread alive immediately after exit:",
      state._display_thread.is_alive())

# Now actually wait for the real computation to finish, to see how much
# further compute happened AFTER the bar was already torn down.
jax.block_until_ready(final)
t_finished = time.monotonic()
print(f"underlying computation actually finished at t={t_finished - t_issue:.4f}s "
      f"({t_finished - exit_time['t']:.4f}s AFTER the context/bar had already exited)")

late_callbacks = [ts for ts, idx in callback_timestamps if ts > exit_time["t"]]
print("callback fires that landed AFTER the context had already exited "
      "(bar closed, file deleted, thread joined):", len(late_callbacks))
print("total callback fires:", len(callback_timestamps), "expected:", N_STEPS)
