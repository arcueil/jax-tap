import jax
import jax.numpy as jnp

import blackjax  # noqa: F401
from blackjax.progress_bar import progress_bar


def make_case():
    def scan_body(carry, x):
        return carry + x, carry

    def while_cond(carry):
        i, acc = carry
        return i < 3

    def while_body(carry):
        i, acc = carry
        final, _ = jax.lax.scan(scan_body, 0.0, jnp.arange(10.0))
        return i + 1, acc + final

    def run_while(x0):
        return jax.lax.while_loop(while_cond, while_body, (0, x0))

    def vmapped_while(x0s):
        return jax.vmap(run_while)(x0s)

    return vmapped_while


vmapped_while = make_case()
x0s = jnp.array([0.0, 1.0, 2.0, 3.0])

with progress_bar(label="fresh-vmap-while") as state:
    out = jax.jit(vmapped_while)(x0s)
    jax.block_until_ready(out)
    print("FRESH vmap(while(scan)), first-ever call INSIDE ctx: out=", out, "n_steps=", state.n_steps)

# expected: each lane runs while_body 3 times (i:0->3), each time adding 45 (sum 0..9)
expected_acc = x0s + 3 * 45.0
print("matches expected math:", bool(jnp.allclose(out[1], expected_acc)), "expected:", expected_acc)
