"""AYS probe: the three untested seams — vmap, grad, custom_jvp opaque-bind.

For each: does tap.verbose preserve the invariant the library PROMISES
(bitwise identity / autodiff correctness / vmap-safety)?
"""
import numpy as np
import jax
import jax.numpy as jnp
import jaxtap as tap


def report(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def bytes_eq(a, b):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    if len(la) != len(lb):
        return False
    return all(np.asarray(x).tobytes() == np.asarray(y).tobytes() for x, y in zip(la, lb))


# ---- Seam 1: custom_jvp primitive in the traced program (AD-prims opaque bind) ----
# jax.nn.relu / jax.nn.softplus carry custom_jvp. Put one inside a scan body.
def f_customjvp(x0, xs):
    def body(c, x):
        c2 = jax.nn.softplus(c + x)   # softplus has a custom_jvp rule
        return c2, c2
    return jax.lax.scan(body, x0, xs)

x0 = jnp.float32(0.5)
xs = jnp.linspace(0.1, 1.0, 5, dtype=jnp.float32)
try:
    ref = f_customjvp(x0, xs)
    events = []
    got = tap.verbose(f_customjvp, on_step=events.append)(x0, xs)
    jax.block_until_ready(got)
    report("custom_jvp-in-scan bitwise", bytes_eq(ref, got), f"({len(events)} events)")
except Exception as e:
    report("custom_jvp-in-scan bitwise", False, f"RAISED {type(e).__name__}: {e}")


# ---- Seam 2: grad through the transform ----
# The library inherits #964's claim that grad is bitwise-exact. Prove it.
def scalar_loss(theta):
    def body(c, x):
        return c * theta + x, c
    final, _ = jax.lax.scan(body, jnp.float32(0.0), jnp.arange(5.0, dtype=jnp.float32))
    return final

try:
    g_ref = jax.grad(scalar_loss)(jnp.float32(1.3))
    # grad of the tapped version (on_step is a no-op host callback)
    tapped = tap.verbose(scalar_loss, on_step=lambda e: None)
    g_got = jax.grad(tapped)(jnp.float32(1.3))
    jax.block_until_ready(g_got)
    report("grad(verbose(f)) bitwise", bytes_eq(g_ref, g_got),
           f"(ref={float(g_ref):.6f} got={float(g_got):.6f})")
except Exception as e:
    report("grad(verbose(f)) bitwise", False, f"RAISED {type(e).__name__}: {e}")


# ---- Seam 3: vmap — the headline vmap-safety property ----
def f_vmap(x0, xs):
    return jax.lax.scan(lambda c, x: (c + x, c * x), x0, xs)

x0b = jnp.arange(3, dtype=jnp.float32)                     # 3 lanes
xsb = jnp.arange(3 * 4, dtype=jnp.float32).reshape(3, 4)   # 3 lanes x 4 steps
try:
    ref_v = jax.vmap(f_vmap)(x0b, xsb)
    events_v = []
    got_v = jax.vmap(tap.verbose(f_vmap, on_step=events_v.append))(x0b, xsb)
    jax.block_until_ready(got_v)
    report("vmap(verbose(f)) bitwise", bytes_eq(ref_v, got_v),
           f"({len(events_v)} events; 3 lanes x 4 steps)")
except Exception as e:
    report("vmap(verbose(f)) bitwise", False, f"RAISED {type(e).__name__}: {e}")
