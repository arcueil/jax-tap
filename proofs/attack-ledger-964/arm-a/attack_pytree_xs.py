"""Attack: xs as dict / namedtuple / pytree with a None subtree; and xs with
leaves of different leading lengths (mismatched lengths).

Compares vanilla vs patched scan for each shape.
"""
import collections
import traceback

import jax
import jax.numpy as jnp

import blackjax  # noqa: F401
from blackjax.progress_bar import _original_scan, progress_bar

Point = collections.namedtuple("Point", ["a", "b"])


def run(label, xs, body, use_patch):
    print(f"--- {label} (patched={use_patch}) ---")
    try:
        if use_patch:
            with progress_bar(label=label) as state:
                final, ys = jax.lax.scan(body, 0.0, xs)
                jax.block_until_ready(final)
                print("n_steps recorded:", state.n_steps)
        else:
            final, ys = _original_scan(body, 0.0, xs)
        print("OK final=", final)
        return final, ys, None
    except Exception as e:
        print("raised:", type(e).__name__, str(e)[:300])
        return None, None, e


# 1. dict xs
def body_dict(carry, x):
    return carry + x["a"] + x["b"], carry


xs_dict = {"a": jnp.arange(5.0), "b": jnp.arange(5.0) * 2}
f0 = run("dict-xs", xs_dict, body_dict, use_patch=False)
f1 = run("dict-xs", xs_dict, body_dict, use_patch=True)
print("dict-xs match:", f0[0] == f1[0])
print()

# 2. namedtuple xs
def body_nt(carry, x):
    return carry + x.a + x.b, carry


xs_nt = Point(a=jnp.arange(5.0), b=jnp.arange(5.0) * 3)
f0 = run("namedtuple-xs", xs_nt, body_nt, use_patch=False)
f1 = run("namedtuple-xs", xs_nt, body_nt, use_patch=True)
print("namedtuple-xs match:", f0[0] == f1[0])
print()

# 3. namedtuple xs with a None field
def body_nt_none(carry, x):
    # x.b is None -- only touch x.a
    return carry + x.a, carry


xs_nt_none = Point(a=jnp.arange(5.0), b=None)
f0 = run("namedtuple-xs-with-None-field", xs_nt_none, body_nt_none, use_patch=False)
f1 = run("namedtuple-xs-with-None-field", xs_nt_none, body_nt_none, use_patch=True)
print("namedtuple-None match:", f0[0] == f1[0])
print()

# 4. xs=() empty pytree (zero leaves), with explicit length
def body_empty(carry, x):
    return carry + 1.0, carry


print("--- xs=() empty pytree + length=7 ---")
try:
    f0 = _original_scan(body_empty, 0.0, (), length=7)
    print("vanilla OK:", f0[0])
except Exception as e:
    print("vanilla raised:", e)

calls = []
with progress_bar(label="empty-xs") as state:
    orig_cb = state._step_callback

    def counting_cb(idx):
        calls.append(int(idx))
        orig_cb(idx)

    state._step_callback = counting_cb
    try:
        f1 = jax.lax.scan(body_empty, 0.0, (), length=7)
        jax.block_until_ready(f1)
        print("patched OK:", f1[0], "n_steps=", state.n_steps, "callback fires=", len(calls))
    except Exception as e:
        print("patched raised:", type(e).__name__, str(e)[:300])
print()

# 5. mismatched leading lengths across xs leaves (should be an error either way)
def body_mismatch(carry, x):
    return carry + x[0] + x[1], carry


xs_mismatch = (jnp.arange(5.0), jnp.arange(7.0))
print("--- mismatched-length xs leaves (5 vs 7) ---")
try:
    _original_scan(body_mismatch, 0.0, xs_mismatch)
    print("vanilla: no error (unexpected)")
except Exception as e:
    print("vanilla raised:", type(e).__name__, str(e)[:200])

with progress_bar(label="mismatch-leaves"):
    try:
        jax.lax.scan(body_mismatch, 0.0, xs_mismatch)
        print("patched: no error (unexpected)")
    except Exception as e:
        print("patched raised:", type(e).__name__, str(e)[:200])
