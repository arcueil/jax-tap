"""Does vmap+cond+scan+debug.callback ever successfully instrument (not
just avoid crashing), when the function's first-ever call happens inside
the context (avoiding the cache-staleness confound)?"""
import jax
import jax.numpy as jnp

import blackjax  # noqa: F401
from blackjax.progress_bar import progress_bar


def make_case():
    def scan_body(carry, x):
        return carry + x, carry

    def branch_true(x):
        final, _ = jax.lax.scan(scan_body, 0.0, jnp.arange(10.0))
        return final + x

    def branch_false(x):
        return x * 2.0

    def cond_fn(pred, x):
        return jax.lax.cond(pred, branch_true, branch_false, x)

    def vmapped_cond(preds, xs):
        return jax.vmap(cond_fn)(preds, xs)

    return vmapped_cond


vmapped_cond = make_case()
preds = jnp.array([True, False, True, True])
xs = jnp.array([1.0, 2.0, 3.0, 4.0])

with progress_bar(label="fresh-vmap-cond") as state:
    out = jax.jit(vmapped_cond)(preds, xs)
    jax.block_until_ready(out)
    print("FRESH vmap(cond(scan)), first-ever call INSIDE ctx: out=", out, "n_steps=", state.n_steps)

expected = jnp.array([46.0, 4.0, 48.0, 50.0])
print("matches expected math:", bool(jnp.allclose(out, expected)))
