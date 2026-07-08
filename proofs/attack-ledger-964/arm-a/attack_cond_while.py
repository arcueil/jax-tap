"""Attack: an outermost (depth-0) scan traced INSIDE a lax.cond branch or a
lax.while_loop body. JAX has effect-consistency constraints across cond
branches / while bodies; does instrumenting the scan break something that
works fine unpatched?
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


print("=== scan inside lax.cond branch, unpatched ===")
try:
    out = jax.jit(cond_fn)(True, 1.0)
    print("OK:", out)
except Exception:
    traceback.print_exc(limit=4)

print()
print("=== scan inside lax.cond branch, INSIDE progress_bar ===")
with progress_bar(label="cond-test") as state:
    try:
        out = jax.jit(cond_fn)(True, 1.0)
        jax.block_until_ready(out)
        print("OK:", out, "n_steps=", state.n_steps, "current_step=", state.current_step)
    except Exception:
        traceback.print_exc(limit=4)

print()
print("=== scan inside lax.cond FALSE branch selected, INSIDE progress_bar ===")
with progress_bar(label="cond-test-false") as state:
    try:
        out = jax.jit(cond_fn)(False, 1.0)
        jax.block_until_ready(out)
        print("OK:", out, "n_steps=", state.n_steps, "current_step=", state.current_step)
    except Exception:
        traceback.print_exc(limit=4)


def while_cond(carry):
    i, acc = carry
    return i < 3


def while_body(carry):
    i, acc = carry
    final, _ = jax.lax.scan(scan_body, 0.0, jnp.arange(10.0))
    return i + 1, acc + final


def run_while():
    return jax.lax.while_loop(while_cond, while_body, (0, 0.0))


print()
print("=== scan inside lax.while_loop body, unpatched ===")
try:
    out = jax.jit(run_while)()
    print("OK:", out)
except Exception:
    traceback.print_exc(limit=4)

print()
print("=== scan inside lax.while_loop body, INSIDE progress_bar ===")
with progress_bar(label="while-test") as state:
    try:
        out = jax.jit(run_while)()
        jax.block_until_ready(out)
        print("OK:", out, "n_steps=", state.n_steps)
    except Exception:
        traceback.print_exc(limit=4)
