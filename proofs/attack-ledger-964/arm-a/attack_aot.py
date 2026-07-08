"""Attack: AOT / introspection tooling on scan-containing code traced
INSIDE the progress_bar context: jax.jit(...).lower(), jax.make_jaxpr,
jax.eval_shape. Does the extra debug.callback change what these produce,
or crash where unpatched code works?
"""
import traceback

import jax
import jax.numpy as jnp

import blackjax  # noqa: F401
from blackjax.progress_bar import progress_bar


def f(x):
    def body(carry, xi):
        return carry + xi, carry

    final, ys = jax.lax.scan(body, x, jnp.arange(10.0))
    return final, ys


print("=== jax.make_jaxpr, unpatched ===")
jaxpr0 = jax.make_jaxpr(f)(0.0)
print(jaxpr0)

print()
print("=== jax.make_jaxpr, INSIDE progress_bar ===")
with progress_bar(label="jaxpr-test") as state:
    try:
        jaxpr1 = jax.make_jaxpr(f)(0.0)
        print(jaxpr1)
        print("n_steps recorded (should be 0 -- no actual execution):", state.n_steps)
    except Exception:
        traceback.print_exc(limit=4)

print()
print("=== jax.eval_shape, INSIDE progress_bar ===")
with progress_bar(label="eval_shape-test") as state:
    try:
        out = jax.eval_shape(f, 0.0)
        print("OK:", out, "n_steps:", state.n_steps)
    except Exception:
        traceback.print_exc(limit=4)

print()
print("=== jax.jit(f).lower(0.0), INSIDE progress_bar, then .compile() AFTER exit ===")
with progress_bar(label="lower-test") as state:
    try:
        lowered = jax.jit(f).lower(0.0)
        print("lowered OK inside ctx")
    except Exception:
        traceback.print_exc(limit=4)
        lowered = None

if lowered is not None:
    try:
        compiled = lowered.compile()
        out = compiled(0.0)
        jax.block_until_ready(out)
        print("compiled+run AFTER ctx exit OK:", out)
        print("jax.lax.scan currently:", jax.lax.scan)
    except Exception:
        traceback.print_exc(limit=4)
