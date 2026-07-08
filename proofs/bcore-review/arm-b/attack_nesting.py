"""ATTACK: addressing under structure. Sibling scans, a sub-jaxpr referenced
twice, 4-level nesting, and scan-inside-jit-inside-scan (does the jit boundary
collision propagate into nested addressing?)."""
import jax
import jax.numpy as jnp
import jaxtap as tap
from collections import Counter


def paths_of(f, *args):
    ev = []
    got = tap.verbose(f, on_step=lambda e: ev.append(e))(*args)
    jax.block_until_ready(got)
    return Counter(e.path for e in ev)


xs = jnp.arange(3.0, dtype=jnp.float32)

print("=== 1. two sibling scans at top level -> scan[0], scan[1] ===")
def f_sibling(x0):
    a, _ = jax.lax.scan(lambda c, x: (c + x, c), x0, xs)
    b, _ = jax.lax.scan(lambda c, x: (c * 1.0 + x, c), x0, xs)
    return a + b
print(dict(paths_of(f_sibling, jnp.float32(0.0))))

print("\n=== 2. a body containing two SIBLING inner scans -> scan[0]/scan[0], scan[0]/scan[1] ===")
def f_two_inner(x0):
    def outer(c, x):
        a, _ = jax.lax.scan(lambda cc, xx: (cc + xx, cc), c, xs)
        b, _ = jax.lax.scan(lambda cc, xx: (cc * 1.0, cc), a, xs)
        return b, a
    return jax.lax.scan(outer, x0, xs)
print(dict(paths_of(f_two_inner, jnp.float32(0.0))))

print("\n=== 3. same helper called twice (shared sub-jaxpr referenced twice) ===")
def helper(c0):
    out, _ = jax.lax.scan(lambda c, x: (c + x, c), c0, xs)
    return out
def f_shared(x0):
    return helper(x0) + helper(x0 + 1.0)
print(dict(paths_of(f_shared, jnp.float32(0.0))))

print("\n=== 4. 4-level nesting: scan/while/scan/while ===")
def f_4level(x0):
    def l1(c1, x1):                      # scan[0]
        def cond2(c2): return c2 < c1 + 3.0
        def body2(c2):                   # while[0] inside scan[0]
            def l3(c3, x3):              # scan[0] inside .../while[0]
                def cond4(c4): return c4 < c3 + 2.0
                def body4(c4):           # while[0] deepest
                    return c4 + 1.0
                c3b = jax.lax.while_loop(cond4, body4, c3)
                return c3b, c3b
            c2b, _ = jax.lax.scan(l3, c2, xs)
            return c2b
        c1b = jax.lax.while_loop(cond2, body2, c1)
        return c1b, c1b
    out, _ = jax.lax.scan(l1, x0, xs)
    return out
p4 = paths_of(f_4level, jnp.float32(0.0))
for k in sorted(p4):
    print(f"   {k!r}: {p4[k]}")

print("\n=== 5. scan inside jit inside scan -- does jit-boundary collision propagate? ===")
def f_jit_in_scan(x0):
    def outer(c, x):
        @jax.jit
        def inner(v):
            out, _ = jax.lax.scan(lambda cc, xx: (cc + xx, cc), v, xs)
            return out
        return inner(c), c
    out, _ = jax.lax.scan(outer, x0, xs)
    return out
p5 = paths_of(f_jit_in_scan, jnp.float32(0.0))
print("   paths:", dict(p5))
print("   NOTE: the inner scan lives under a jit inside scan[0]; jit passes _p unchanged")
print("   so its address is scan[0]/scan[0] -- same as a *direct* inner scan (jit invisible)")
