"""Does the SAME staleness generalize to a PLAIN scan (no cond, no
checkpoint) when wrapped in a FRESH jax.jit(...) call each time (a new
PjitFunction object each call), inside a brand-new progress_bar() context
each time? This determines whether the docstring's caveat ("a function
traced once outside this context... will not show a bar") is actually
about "this literal jitted object" (narrow) or "this python function +
shape signature, forever, across any number of distinct jax.jit() wrapper
objects" (much broader, much more surprising).
"""
import jax
import jax.numpy as jnp

import blackjax  # noqa: F401
from blackjax.progress_bar import progress_bar


def make_plain():
    def scan_body(carry, x):
        return carry + x, carry

    def plain_fn(x):
        final, _ = jax.lax.scan(scan_body, x, jnp.arange(10.0))
        return final

    return plain_fn


plain_fn = make_plain()

with progress_bar(label="plain-1st") as s1:
    out1 = jax.jit(plain_fn)(1.0)  # fresh PjitFunction object
    jax.block_until_ready(out1)
    print("1st call (fresh jit wrapper #1), in ctx: n_steps=", s1.n_steps, "out=", out1)

with progress_bar(label="plain-2nd") as s2:
    out2 = jax.jit(plain_fn)(1.0)  # ANOTHER fresh PjitFunction object, same fn+shape
    jax.block_until_ready(out2)
    print("2nd call (fresh jit wrapper #2, SAME fn+shape), NEW ctx: n_steps=", s2.n_steps, "out=", out2)

with progress_bar(label="plain-3rd-diffshape") as s3:
    out3 = jax.jit(plain_fn)(jnp.float32(1.0))  # different input, same shape/dtype though
    jax.block_until_ready(out3)
    print("3rd call (same shape/dtype signature), NEW ctx: n_steps=", s3.n_steps, "out=", out3)

# Now try a genuinely different shape (should force retrace).
with progress_bar(label="plain-4th-diffshape") as s4:
    out4 = jax.jit(plain_fn)(jnp.zeros((3,)))
    jax.block_until_ready(out4)
    print("4th call (DIFFERENT shape -> forces retrace), NEW ctx: n_steps=", s4.n_steps, "out=", out4)
