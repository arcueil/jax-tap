"""Attack #4c: functools.partial(jax.lax.scan, f) captured BEFORE the
context is entered, then called INSIDE the context. Since the partial binds
the function object at creation time (the true original), calling it inside
the context silently bypasses the patch entirely -- no bar, no callback, no
error. Confirm this is truly silent (not even a warning)."""
import functools
import io
import contextlib

import jax
import jax.numpy as jnp

import blackjax

def body(carry, x):
    return carry + x, carry

# Captured BEFORE any progress_bar context exists.
pre_bound_scan = functools.partial(jax.lax.scan, body)

buf = io.StringIO()
with contextlib.redirect_stderr(buf):
    with blackjax.progress_bar(label="should-not-appear") as state:
        final, _ = pre_bound_scan(0.0, jnp.arange(1000))
        jax.block_until_ready(final)

stderr_output = buf.getvalue()
print("scan executed correctly despite bypass:", float(final) == float(jnp.arange(1000).sum()))
print("state.n_steps ever set (bar would have appeared):", state.n_steps)
print("any stderr/warning emitted about the silent bypass:", bool(stderr_output.strip()))
print("stderr content:", repr(stderr_output[:200]))
