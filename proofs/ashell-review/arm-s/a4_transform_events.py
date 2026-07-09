"""ARM-S battery 4: do transforms AROUND the context drop/diverge events vs verbose()?"""
from collections import Counter
import numpy as np
import jax
import jax.numpy as jnp
import jaxtap as tap

def bw(a, b):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return len(la) == len(lb) and all(
        np.asarray(x).tobytes() == np.asarray(y).tobytes() for x, y in zip(la, lb))

xs = jnp.arange(5.0, dtype=jnp.float32)
def scanfn(x0):
    return jax.lax.scan(lambda c, x: (c * 1.01 + jnp.sin(x), c), x0, xs)[0]

def compare(name, transform, arg):
    """transform is a function taking the base fn; we apply verbose vs A-shell equivalently.
    For verbose we must instrument the INNERMOST fn (scanfn), then apply transform.
    That is NOT what A-shell does (A-shell patches globally), so instead we compare
    the natural usage: verbose(whole_transformed_fn) vs with-context running it."""
    whole = lambda a: transform(scanfn)(a)
    vb = []
    r_vb = tap.verbose(whole, on_step=vb.append)(arg)
    jax.block_until_ready(r_vb)
    with tap.record() as rec:
        r_ash = whole(arg)
    jax.block_until_ready(r_ash)
    vb_paths = Counter(e.path for e in vb)
    ash_paths = Counter(e.path for e in rec.events)
    n_vb, n_ash = len(vb), len(rec.events)
    print(f"[{name}]")
    print(f"    verbose: {n_vb} events {dict(vb_paths)}")
    print(f"    ashell : {n_ash} events {dict(ash_paths)}")
    print(f"    result bitwise: {bw(r_vb, r_ash)}   counts match: {n_vb == n_ash}   "
          f"paths match: {vb_paths == ash_paths}")

compare("grad(scan)", lambda f: jax.grad(f), jnp.float32(0.5))
compare("jit(scan)", lambda f: jax.jit(f), jnp.float32(0.5))
compare("grad(jit(scan))", lambda f: jax.grad(lambda x: jax.jit(f)(x)), jnp.float32(0.5))
compare("jit(grad(scan))", lambda f: jax.jit(jax.grad(f)), jnp.float32(0.5))
compare("vmap(scan)", lambda f: jax.vmap(f), jnp.arange(3.0, dtype=jnp.float32))
compare("vmap(grad(scan))", lambda f: jax.vmap(jax.grad(f)), jnp.arange(3.0, dtype=jnp.float32))
compare("hessian(scan)", lambda f: jax.hessian(f), jnp.float32(0.5))
