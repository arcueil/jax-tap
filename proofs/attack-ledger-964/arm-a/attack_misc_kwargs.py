import traceback

import jax
import jax.numpy as jnp

import blackjax  # noqa: F401
from blackjax.progress_bar import progress_bar


def body(carry, x):
    return carry + x, carry


xs = jnp.arange(8.0)

print("=== unroll=True (bool, not int) ===")
try:
    expected = jax.lax.scan(body, 0.0, xs, unroll=True)
    with progress_bar(label="unroll-bool") as state:
        got = jax.lax.scan(body, 0.0, xs, unroll=True)
        jax.block_until_ready(got)
    print("match:", bool(got[0] == expected[0]), "n_steps=", state.n_steps)
except Exception:
    traceback.print_exc(limit=4)

print()
print("=== f= / init= / xs= passed as keyword args ===")
try:
    expected = jax.lax.scan(f=body, init=0.0, xs=xs)
    with progress_bar(label="all-kwargs") as state:
        got = jax.lax.scan(f=body, init=0.0, xs=xs)
        jax.block_until_ready(got)
    print("match:", bool(got[0] == expected[0]), "n_steps=", state.n_steps)
except Exception:
    traceback.print_exc(limit=4)

print()
print("=== positional f, init, xs (no keywords) ===")
try:
    expected = jax.lax.scan(body, 0.0, xs)
    with progress_bar(label="positional") as state:
        got = jax.lax.scan(body, 0.0, xs)
        jax.block_until_ready(got)
    print("match:", bool(got[0] == expected[0]), "n_steps=", state.n_steps)
except Exception:
    traceback.print_exc(limit=4)

print()
print("=== carry-structure-mismatch error message (user bug in f) ===")


def bad_body(carry, x):
    # user bug: returns a 2-tuple structure instead of matching init's scalar
    return (carry + x, carry), carry


print("--- vanilla ---")
try:
    jax.lax.scan(bad_body, 0.0, xs)
except Exception as e:
    print(type(e).__name__, str(e)[:400])

print("--- patched ---")
with progress_bar(label="bad-body"):
    try:
        jax.lax.scan(bad_body, 0.0, xs)
    except Exception as e:
        print(type(e).__name__, str(e)[:400])
