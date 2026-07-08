"""ARM-S battery 5: separate jit-cache artifacts from genuine transform divergence.
Run A-shell with a FRESH cache each time (clear_caches) so no pre-context compile leaks."""
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

def ashell_events(whole, arg):
    jax.clear_caches()
    with tap.record() as rec:
        r = whole(arg)
    jax.block_until_ready(r)
    return len(rec.events), Counter(e.path for e in rec.events), r

def verbose_events(whole, arg):
    jax.clear_caches()
    vb = []
    r = tap.verbose(whole, on_step=vb.append)(arg)
    jax.block_until_ready(r)
    return len(vb), Counter(e.path for e in vb), r

def run(name, transform, arg):
    whole = lambda a: transform(scanfn)(a)
    # A-shell FIRST with cleared cache, THEN verbose with cleared cache
    n_ash, p_ash, r_ash = ashell_events(whole, arg)
    n_vb, p_vb, r_vb = verbose_events(whole, arg)
    print(f"[{name}]  A-shell={n_ash} {dict(p_ash)}   verbose={n_vb} {dict(p_vb)}"
          f"   bitwise={bw(r_ash, r_vb)}   MATCH={n_ash==n_vb and p_ash==p_vb}")

print("clear_caches BEFORE each A-shell run (no pre-context compile leak):")
run("jit(scan)          ", lambda f: jax.jit(f), jnp.float32(0.5))
run("grad(jit(scan))    ", lambda f: jax.grad(lambda x: jax.jit(f)(x)), jnp.float32(0.5))
run("grad(scan)         ", lambda f: jax.grad(f), jnp.float32(0.5))
run("jit(grad(scan))    ", lambda f: jax.jit(jax.grad(f)), jnp.float32(0.5))
run("vmap(scan)         ", lambda f: jax.vmap(f), jnp.arange(3.0, dtype=jnp.float32))
run("vmap(grad(scan))   ", lambda f: jax.vmap(jax.grad(f)), jnp.arange(3.0, dtype=jnp.float32))
run("hessian(scan)      ", lambda f: jax.hessian(f), jnp.float32(0.5))

print()
print("Cross-check: plain jit(scan) inside ctx with FRESH cache -> should NOT be 0")
jax.clear_caches()
fj = jax.jit(scanfn)
with tap.record() as rec:
    r = fj(jnp.float32(0.5))
    jax.block_until_ready(r)
print(f"   fresh jit(scan) inside ctx: {len(rec.events)} events (0 would be a silent drop)")
