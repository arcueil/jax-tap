"""AYS (iv): vmap over a function whose cond branch contains a depth-0
scan traced inside the context; and vmap-of-while-of-scan. Does either
crash on JAX 0.10.0 (the historical #927 minefield: vmap + effectful
control flow)?
"""
import traceback

import jax
import jax.numpy as jnp

import blackjax  # noqa: F401
from blackjax.progress_bar import progress_bar


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


print("=== vmap(cond(scan)) unpatched ===")
preds = jnp.array([True, False, True])
xs = jnp.array([1.0, 2.0, 3.0])
try:
    out0 = jax.jit(vmapped_cond)(preds, xs)
    print("OK:", out0)
except Exception:
    traceback.print_exc(limit=6)

print()
print("=== vmap(cond(scan)) INSIDE progress_bar ===")
with progress_bar(label="vmap-cond") as state:
    try:
        out1 = jax.jit(vmapped_cond)(preds, xs)
        jax.block_until_ready(out1)
        print("OK:", out1, "n_steps=", state.n_steps, "current_step=", state.current_step)
        print("matches unpatched:", bool(jnp.allclose(out0, out1)))
    except Exception:
        traceback.print_exc(limit=6)


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


x0s = jnp.array([0.0, 1.0, 2.0])

print()
print("=== vmap(while(scan)) unpatched ===")
try:
    outw0 = jax.jit(vmapped_while)(x0s)
    print("OK:", outw0)
except Exception:
    traceback.print_exc(limit=6)

print()
print("=== vmap(while(scan)) INSIDE progress_bar ===")
with progress_bar(label="vmap-while") as state2:
    try:
        outw1 = jax.jit(vmapped_while)(x0s)
        jax.block_until_ready(outw1)
        print("OK:", outw1, "n_steps=", state2.n_steps)
        print(
            "matches unpatched:",
            bool(jnp.array_equal(outw0[0], outw1[0])) and bool(jnp.allclose(outw0[1], outw1[1])),
        )
    except Exception:
        traceback.print_exc(limit=6)
