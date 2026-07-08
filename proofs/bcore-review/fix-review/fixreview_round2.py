"""FIX-REVIEW round 2 — different angle from round 1.

1. Does where/max_depth filtering now work CORRECTLY across the newly-visible
   jit/cond boundaries? (F2 said the bug corrupted M2 filters across jit.)
2. grad-of-grad (grad2) through a recursed cond — higher-order AD survives?
3. deeper nesting: cond inside jit inside scan — addressing + bitwise.
"""
import numpy as np
import jax
import jax.numpy as jnp
import jaxtap as tap

FAILS = []
def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok: FAILS.append(name)

def bw(a, b):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return len(la) == len(lb) and all(
        np.asarray(x).tobytes() == np.asarray(y).tobytes() for x, y in zip(la, lb))

xs = jnp.arange(5.0, dtype=jnp.float32)
def scanfn(c0):
    c, _ = jax.lax.scan(lambda c, x: (c * 1.01 + x, c), c0, xs)
    return c

# ---------- 1: max_depth / where across a jit boundary ----------
def f_jit_nested(c0):
    a = scanfn(c0)                   # scan[0]  (depth 0)
    b = jax.jit(scanfn)(c0 + 1.0)   # jit[1]/scan[0]  (depth 1)
    return a + b

# max_depth=0 should tap only the top-level scan (depth 0), NOT the jit-nested one
ev0 = []
jax.block_until_ready(tap.verbose(f_jit_nested, on_step=ev0.append, max_depth=0)(jnp.float32(0.5)))
paths0 = sorted({e.path for e in ev0})
check("max_depth=0 excludes jit-nested scan", paths0 == ["scan[0]"], f"(paths={paths0})")

# where selecting only the jit-nested scan
evw = []
jax.block_until_ready(tap.verbose(
    f_jit_nested, on_step=evw.append, where=lambda p: p.startswith("jit"))(jnp.float32(0.5)))
pathsw = sorted({e.path for e in evw})
check("where selects only jit-nested scan", pathsw == ["jit[1]/scan[0]"], f"(paths={pathsw})")

# bitwise identity must still hold with filters active
ref = f_jit_nested(jnp.float32(0.5))
got = tap.verbose(f_jit_nested, on_step=lambda e: None, max_depth=0)(jnp.float32(0.5))
check("filtered run still bitwise", bw(ref, got))

# ---------- 2: grad2 through a recursed cond ----------
def f_cond(c0):
    return jax.lax.cond(c0 > 0, lambda z: scanfn(z), lambda z: scanfn(-z), c0)
g2ref = jax.grad(jax.grad(f_cond))(jnp.float32(0.5))
g2got = jax.grad(jax.grad(tap.verbose(f_cond, on_step=lambda e: None)))(jnp.float32(0.5))
jax.block_until_ready(g2got)
check("grad2(cond)", bw(g2ref, g2got), f"(ref={float(g2ref):.6f} got={float(g2got):.6f})")

# ---------- 3: cond inside jit inside scan (deep nesting) ----------
def f_deep(c0):
    def body(c, x):
        inner = jax.jit(lambda z: jax.lax.cond(x > 2.0, lambda w: w + 1.0, lambda w: w * 1.1, z))
        return inner(c), c
    c, _ = jax.lax.scan(body, c0, xs)
    return c
ref_d = f_deep(jnp.float32(0.5))
ev_d = []
got_d = tap.verbose(f_deep, on_step=ev_d.append)(jnp.float32(0.5))
jax.block_until_ready(got_d)
paths_d = sorted({e.path for e in ev_d})
check("deep cond-in-jit-in-scan bitwise", bw(ref_d, got_d), f"(paths={paths_d})")
gref_d = jax.grad(f_deep)(jnp.float32(0.5))
ggot_d = jax.grad(tap.verbose(f_deep, on_step=lambda e: None))(jnp.float32(0.5))
check("deep nesting grad", bw(gref_d, ggot_d), f"(ref={float(gref_d):.5f} got={float(ggot_d):.5f})")

print("\n" + ("ROUND 2 ALL PASS" if not FAILS else f"ROUND 2 FAILURES: {FAILS}"))
