"""Attack #4b: user calls progress_bar() and manually invokes __enter__()
(e.g. in a notebook cell, intending to call __exit__ in a later cell), then
an exception/kernel-interrupt happens before __exit__ is ever called. This
is a realistic Jupyter pattern: 'cm = blackjax.progress_bar(); cm.__enter__()'
in cell N, '...run sampling...' in cell N+1, 'cm.__exit__(None,None,None)'
in cell N+2 -- if cell N+1 errors, __exit__ never runs.

Consequence under test: jax.lax.scan stays monkeypatched FOREVER (module
global), so an UNRELATED, LATER scan -- run by code that has never heard of
progress_bar -- gets silently instrumented: a background daemon thread
spins up rendering a tqdm bar for code that never asked for one, and the
registry is never cleaned up.
"""
import threading
import time

import jax
import jax.numpy as jnp

import blackjax
from blackjax.progress_bar import _original_scan, _progress_registry

cm = blackjax.progress_bar(label="leaked")
state = cm.__enter__()  # forgot the `with`; __exit__ never called
print("after manual __enter__: scan patched?", jax.lax.scan is not _original_scan)
print("registry size:", len(_progress_registry))

# Now, much later, completely unrelated code -- that has never seen
# blackjax.progress_bar -- calls jax.lax.scan directly.
def unrelated_body(carry, x):
    return carry + x, carry

final, ys = jax.lax.scan(unrelated_body, 0.0, jnp.arange(1000))
jax.block_until_ready(final)
time.sleep(0.3)  # let the daemon thread render at least once

print("unrelated scan result correct:", float(final) == float(jnp.arange(1000).sum()))
print("state.n_steps hijacked to unrelated scan's length:", state.n_steps)
print("state.current_step advanced by unrelated code:", state.current_step)
print("display thread alive, rendering a bar nobody asked for:",
      state._display_thread.is_alive())
print("active thread count (leaked daemon):", threading.active_count())

# Process never recovers on its own: scan stays patched indefinitely.
print("scan STILL patched after unrelated call completes:",
      jax.lax.scan is not _original_scan)
