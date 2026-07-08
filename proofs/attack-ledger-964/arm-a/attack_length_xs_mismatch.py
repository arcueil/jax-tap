"""Attack: length= given together with xs= of a DIFFERENT leading length.

First establish vanilla (unpatched) jax.lax.scan semantics for this
combination, then compare to the patched behavior inside progress_bar().
"""
import traceback

import jax
import jax.numpy as jnp

import blackjax  # noqa: F401
from blackjax.progress_bar import _original_scan, progress_bar


def body(carry, x):
    return carry + x, carry


xs10 = jnp.arange(10.0)

print("=== vanilla scan: length=5, xs has 10 elements ===")
try:
    final, ys = _original_scan(body, 0.0, xs10, length=5)
    print("no error. final=", final, "ys=", ys)
except Exception as e:
    print("raised:", type(e).__name__, str(e)[:300])

print()
print("=== vanilla scan: length=15, xs has 10 elements ===")
try:
    final, ys = _original_scan(body, 0.0, xs10, length=15)
    print("no error. final=", final, "ys=", ys)
except Exception as e:
    print("raised:", type(e).__name__, str(e)[:300])

print()
print("=== patched scan (inside progress_bar): length=5, xs has 10 elements ===")
with progress_bar(label="mismatch-test") as state:
    try:
        final, ys = jax.lax.scan(body, 0.0, xs10, length=5)
        jax.block_until_ready(final)
        print("no error. final=", final, "ys=", ys, "n_steps=", state.n_steps)
    except Exception:
        print("raised:")
        traceback.print_exc(limit=3)

print()
print("=== patched scan (inside progress_bar): length=15, xs has 10 elements ===")
with progress_bar(label="mismatch-test2") as state:
    try:
        final, ys = jax.lax.scan(body, 0.0, xs10, length=15)
        jax.block_until_ready(final)
        print("no error. final=", final, "ys=", ys, "n_steps=", state.n_steps)
    except Exception:
        print("raised:")
        traceback.print_exc(limit=3)
