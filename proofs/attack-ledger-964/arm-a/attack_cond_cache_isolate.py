"""Isolate: is the n_steps=0 under lax.cond/while_loop a real "effects
inside cond/while forbid the callback" issue, or a jaxpr-caching issue
where the branch function was already traced (with unpatched scan) BEFORE
ever entering a progress_bar context -- so the callback is silently never
attached, for the rest of the process, regardless of new contexts?
"""
import jax
import jax.numpy as jnp

import blackjax  # noqa: F401
from blackjax.progress_bar import progress_bar


def make_cond_case():
    def scan_body(carry, x):
        return carry + x, carry

    def branch_true(x):
        final, _ = jax.lax.scan(scan_body, 0.0, jnp.arange(10.0))
        return final + x

    def branch_false(x):
        return x * 2.0

    def cond_fn(pred, x):
        return jax.lax.cond(pred, branch_true, branch_false, x)

    return cond_fn


# Case 1: NEVER call cond_fn/branch_true anywhere before entering the ctx.
cond_fn_1 = make_cond_case()
with progress_bar(label="cond-fresh") as state:
    out = jax.jit(cond_fn_1)(True, 1.0)
    jax.block_until_ready(out)
    print("FRESH cond, first-ever call INSIDE ctx: out=", out, "n_steps=", state.n_steps)

print()

# Case 2: same fresh function, call it a SECOND time, in a brand NEW ctx.
with progress_bar(label="cond-fresh-2nd") as state2:
    out2 = jax.jit(cond_fn_1)(True, 1.0)
    jax.block_until_ready(out2)
    print("SAME cond fn, second call in a NEW ctx: out=", out2, "n_steps=", state2.n_steps)

print()

# Case 3: a function traced once OUTSIDE any context, then used INSIDE one.
cond_fn_2 = make_cond_case()
warm = jax.jit(cond_fn_2)(True, 1.0)
jax.block_until_ready(warm)
print("warmup call outside ctx done:", warm)
with progress_bar(label="cond-prewarmed") as state3:
    out3 = jax.jit(cond_fn_2)(True, 1.0)
    jax.block_until_ready(out3)
    print("PRE-WARMED cond fn, called INSIDE ctx: out=", out3, "n_steps=", state3.n_steps)

print()

# Case 4: plain (non-cond) scan for comparison -- fresh function, first call inside ctx.
def make_plain():
    def scan_body(carry, x):
        return carry + x, carry

    def plain_fn(x):
        final, _ = jax.lax.scan(scan_body, x, jnp.arange(10.0))
        return final

    return plain_fn


plain_fn = make_plain()
with progress_bar(label="plain-fresh") as state4:
    out4 = jax.jit(plain_fn)(1.0)
    jax.block_until_ready(out4)
    print("FRESH plain (no cond) call INSIDE ctx: out=", out4, "n_steps=", state4.n_steps)
