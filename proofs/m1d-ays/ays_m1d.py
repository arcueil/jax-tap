"""AYS M1d R1: se-gate under vmap; descend-always bitwise/grad on cond+remat;
gating visible in us; prim tap in cond-in-scan gated; while-loop gating."""
import time, io, contextlib
import numpy as np
import jax, jax.numpy as jnp
import jaxtap as tap

FAILS = []
def check(name, ok, d=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {d}")
    if not ok: FAILS.append(name)
def bw(a, b):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return len(la) == len(lb) and all(np.asarray(x).tobytes() == np.asarray(y).tobytes() for x, y in zip(la, lb))

# 1: se-gate under vmap (per-lane)
def f(x0):
    def body(c, x):
        return c + jnp.sin(x) * 1.01, None
    c, _ = jax.lax.scan(body, x0, jnp.arange(30.0))
    return c
ev = []
g = jax.vmap(tap.verbose(f, on_step=lambda e: None, sample_every=10,
                         taps=[tap.on("sin", select=lambda o: o[0][()] if o[0].ndim==0 else o[0]) ]))
ref_v = jax.vmap(f)(jnp.arange(3.0))
evs = []
g2 = jax.vmap(tap.verbose(f, on_step=evs.append, sample_every=10,
                          taps=[tap.on("sin")]))
out = g2(jnp.arange(3.0)); jax.block_until_ready(out)
sin_ev = [e for e in evs if "sin" in e.path]
check("vmap se-gate: prim taps = lanes*(N/se)", len(sin_ev) == 3 * 3, f"({len(sin_ev)} vs 9)")
check("vmap se-gate bitwise", bw(ref_v, out))

# 2: descend-always bitwise + grad on cond/remat-heavy program, filters active
def deep(c0):
    def inner(z):
        s, _ = jax.lax.scan(lambda a, b: (a * 1.01 + b, None), z, jnp.arange(4.0))
        return s
    def body(c, x):
        c1 = jax.lax.cond(x > 1.0, lambda z: jax.checkpoint(inner)(z), lambda z: z * 1.1, c)
        return c1, None
    c, _ = jax.lax.scan(body, c0, jnp.arange(4.0))
    return c
ref = deep(jnp.float32(0.5))
got = tap.verbose(deep, on_step=lambda e: None, where=lambda p: False, max_depth=0)(jnp.float32(0.5))
check("descend-always: cond+remat, all-filtered, bitwise", bw(ref, got))
gr = jax.grad(deep)(jnp.float32(0.5))
gg = jax.grad(tap.verbose(deep, on_step=lambda e: None, where=lambda p: False))(jnp.float32(0.5))
check("descend-always grad bitwise", bw(gr, gg), f"(ref={float(gr):.5f} got={float(gg):.5f})")

# 3: gating visible in wall time (prim tap, N=10k)
xs10k = jnp.linspace(0., 1., 10_000)
def fb(c0):
    def body(c, x):
        return c * 1.01 + jnp.sin(x), None
    c, _ = jax.lax.scan(body, c0, xs10k)
    return c
def timeit(gfn):
    jax.block_until_ready(gfn(jnp.float32(0.))) 
    ts = []
    for _ in range(3):
        t0 = time.perf_counter(); jax.block_until_ready(gfn(jnp.float32(0.))); ts.append(time.perf_counter()-t0)
    return np.median(ts) / 10_000 * 1e6
t_se1 = timeit(jax.jit(tap.verbose(fb, on_step=lambda e: None, taps=[tap.on("sin")])))
t_se100 = timeit(jax.jit(tap.verbose(fb, on_step=lambda e: None, sample_every=100, taps=[tap.on("sin")])))
print(f"[INFO] primtap us/step: se=1 {t_se1:.2f}, se=100 {t_se100:.2f}")
check("se-gating shows up in wall time (>=10x)", t_se1 / max(t_se100, 1e-9) > 10, f"({t_se1/t_se100:.1f}x)")

# 4: prim tap inside cond-in-scan, gated
def fc(c0):
    def body(c, x):
        c1 = jax.lax.cond(x > 0.5, lambda z: jnp.sin(z) + z, lambda z: z, c)
        return c1, None
    c, _ = jax.lax.scan(body, c0, jnp.linspace(0., 1., 20))
    return c
evc = []
gc = tap.verbose(fc, on_step=evc.append, sample_every=5, taps=[tap.on("sin")])
refc = fc(jnp.float32(0.3)); gotc = gc(jnp.float32(0.3)); jax.block_until_ready(gotc)
sin_c = [e for e in evc if "sin" in e.path]
# taken-branch steps where x>0.5: steps 10..19 (10 steps); gated se=5 -> steps in {10,15} = 2
check("cond-in-scan prim tap gated", len(sin_c) == 2 and bw(refc, gotc),
      f"({len(sin_c)} events, steps={sorted(e.step for e in sin_c)})")

# 5: while-loop prim tap gated
def fw(v0):
    def cond_fn(c): return c[0] < 25.0
    def body_fn(c):
        v, acc = c
        return (v + 1.0, acc + jnp.sin(v))
    return jax.lax.while_loop(cond_fn, body_fn, (v0, jnp.float32(0.)))
evw = []
gw = tap.verbose(fw, on_step=evw.append, sample_every=10, taps=[tap.on("sin")])
refw = fw(jnp.float32(0.)); gotw = gw(jnp.float32(0.)); jax.block_until_ready(gotw)
sin_w = [e for e in evw if "sin" in e.path]
check("while prim tap gated (25 iters, se=10 -> 3)", len(sin_w) == 3 and bw(refw, gotw),
      f"({len(sin_w)}, steps={sorted(e.step for e in sin_w)})")

print("\n" + ("M1D AYS R1: ALL PASS" if not FAILS else f"FAILURES: {FAILS}"))
