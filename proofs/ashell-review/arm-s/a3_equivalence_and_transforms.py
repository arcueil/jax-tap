"""ARM-S battery 3: event-equivalence vs verbose() for SINGLE top-level programs,
plus deep transforms and the jit-boundary path question."""
from collections import Counter
import numpy as np
import jax
import jax.numpy as jnp
import jaxtap as tap

def bw(a, b):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return len(la) == len(lb) and all(
        np.asarray(x).dtype == np.asarray(y).dtype and
        np.asarray(x).tobytes() == np.asarray(y).tobytes() for x, y in zip(la, lb))

def norm_val(v):
    # normalise a TapEvent.value (tuple of leaves) to comparable bytes
    return tuple(np.asarray(x).tobytes() for x in jax.tree_util.tree_leaves(v))

def event_key(e):
    return (e.path, e.step, norm_val(e.value))

def compare(name, fn, arg, **kw):
    """Run fn under verbose() and under the A-shell; compare full event sets + result."""
    vb = []
    r_vb = tap.verbose(fn, on_step=vb.append, **kw)(arg)
    jax.block_until_ready(r_vb)

    ctx_kw = {k: v for k, v in kw.items()}
    with tap.record(**ctx_kw) as rec:
        r_ash = fn(arg)
    jax.block_until_ready(r_ash)

    vb_keys = Counter(event_key(e) for e in vb)
    ash_keys = Counter(event_key(e) for e in rec.events)
    paths_vb = Counter(e.path for e in vb)
    paths_ash = Counter(e.path for e in rec.events)

    events_equal = vb_keys == ash_keys
    result_bw = bw(r_vb, r_ash)
    tag = "PASS" if (events_equal and result_bw) else "FAIL"
    print(f"[{tag}] {name}: events_equal={events_equal} result_bitwise={result_bw}")
    if not events_equal:
        print(f"        verbose paths: {dict(paths_vb)}")
        print(f"        ashell  paths: {dict(paths_ash)}")
        # show a couple of differing keys
        only_vb = list((vb_keys - ash_keys).elements())[:2]
        only_ash = list((ash_keys - vb_keys).elements())[:2]
        if only_vb:
            print(f"        only-in-verbose (paths/steps): {[(k[0],k[1]) for k in only_vb]}")
        if only_ash:
            print(f"        only-in-ashell  (paths/steps): {[(k[0],k[1]) for k in only_ash]}")
    return events_equal and result_bw

xs = jnp.arange(5.0, dtype=jnp.float32)
INNER = jnp.arange(3.0, dtype=jnp.float32)

print("=== SINGLE top-level CF op: should be equivalent to verbose() ===")

# nested scan (one top-level scan containing an inner scan)
def nested(x0):
    def outer(c, x):
        c2, _ = jax.lax.scan(lambda a, b: (a * 1.001 + jnp.sin(b), a), c + x, INNER)
        return c2, c2
    return jax.lax.scan(outer, x0, xs)[0]
compare("nested scan", nested, jnp.float32(0.5))

# while inside scan
def while_in_scan(x0):
    def outer(c, x):
        w = jax.lax.while_loop(lambda s: s < c + 5.0, lambda s: s + 1.0, c)
        return c + x, w
    return jax.lax.scan(outer, x0, xs)[0]
compare("while-in-scan", while_in_scan, jnp.float32(0.5))

# cond inside scan
def cond_in_scan(x0):
    def outer(c, x):
        c2 = jax.lax.cond(x > 2.0, lambda a: a * 2.0, lambda a: a + 1.0, c)
        return c2, c2
    return jax.lax.scan(outer, x0, xs)[0]
compare("cond-in-scan", cond_in_scan, jnp.float32(0.5))

# cond in scan + select
compare("cond-in-scan + select(sum)", cond_in_scan, jnp.float32(0.5),
        select=lambda leaves: sum(jax.tree_util.tree_leaves(leaves)))

# sample_every
compare("nested scan + sample_every=2", nested, jnp.float32(0.5), sample_every=2)

# where filter
compare("nested scan + where(only outer)", nested, jnp.float32(0.5),
        where=lambda p: p == "scan[0]")

# max_depth
compare("nested scan + max_depth=0", nested, jnp.float32(0.5), max_depth=0)

# primitive taps (cholesky inside scan)
def chol(x0):
    def body(carry, _):
        c = 1.0 - 10.0 ** (-carry)
        M = jnp.array([[1.0, c], [c, 1.0]], dtype=c.dtype)
        L = jnp.linalg.cholesky(M)
        return carry + 1.0, jnp.sum(jnp.diag(L))
    return jax.lax.scan(body, x0, None, length=5)[0]
compare("primitive tap cholesky", chol, jnp.float32(1.0),
        taps=[tap.on("cholesky")])

print()
print("=== JIT-BOUNDARY path: scan inside a TOP-LEVEL jit ===")
def jit_scan(x0):
    return jax.jit(lambda y: jax.lax.scan(lambda c, x: (c + x, c), y, xs)[0])(x0)
compare("scan inside top-level jit", jit_scan, jnp.float32(0.5))

print()
print("=== TRANSFORMS AROUND the context (bitwise focus) ===")

def scanfn(x0):
    return jax.lax.scan(lambda c, x: (c * 1.01 + jnp.sin(x), c), x0, xs)[0]

def tcheck(name, ref_fn, ctx_fn):
    ref = ref_fn()
    with tap.record() as rec:
        got = ctx_fn()
    jax.block_until_ready(got)
    ok = bw(ref, got)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: bitwise={ok} events={len(rec.events)}")

# grad of jit of scan
tcheck("grad(jit(scan))",
       lambda: jax.grad(lambda x: jax.jit(scanfn)(x))(jnp.float32(0.5)),
       lambda: jax.grad(lambda x: jax.jit(scanfn)(x))(jnp.float32(0.5)))
# vmap of grad
tcheck("vmap(grad(scan))",
       lambda: jax.vmap(jax.grad(scanfn))(jnp.arange(3.0, dtype=jnp.float32)),
       lambda: jax.vmap(jax.grad(scanfn))(jnp.arange(3.0, dtype=jnp.float32)))
# hessian
tcheck("hessian(scan)",
       lambda: jax.hessian(scanfn)(jnp.float32(0.5)),
       lambda: jax.hessian(scanfn)(jnp.float32(0.5)))
# grad of grad
tcheck("grad(grad(scan))",
       lambda: jax.grad(jax.grad(scanfn))(jnp.float32(0.5)),
       lambda: jax.grad(jax.grad(scanfn))(jnp.float32(0.5)))
# jit(grad(scan))
tcheck("jit(grad(scan))",
       lambda: jax.jit(jax.grad(scanfn))(jnp.float32(0.5)),
       lambda: jax.jit(jax.grad(scanfn))(jnp.float32(0.5)))
