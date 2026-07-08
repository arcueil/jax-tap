"""AYS on M1b A-shell — probing the seams the 12 tests may not cover.

R1: 1. the promise: with-form bitwise + delete-the-with -> zero events + restored attr
    2. PHANTOM EMISSION: f jitted+called INSIDE ctx, called again AFTER exit
       -> do events leak into the dead recorder? (trace-cache boundary, reverse)
    3. jax.jit-wrapped user fn inside the ctx (interception during jit tracing)
    4. grad THROUGH the with-form; vmap THROUGH the with-form
"""
import numpy as np
import jax
import jax.numpy as jnp
import jaxtap as tap

FAILS = []
def check(name, ok, d=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {d}")
    if not ok: FAILS.append(name)
def bw(a, b):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return len(la) == len(lb) and all(
        np.asarray(x).tobytes() == np.asarray(y).tobytes() for x, y in zip(la, lb))

xs = jnp.arange(5.0, dtype=jnp.float32)
def f(x0):
    def body(c, x):
        return c * 1.01 + jnp.sin(x), c
    c, _ = jax.lax.scan(body, x0, xs)
    return c

orig_scan = jax.lax.scan
ref = f(jnp.float32(0.5))

# --- 1: the promise ---
with tap.record() as rec:
    got = f(jnp.float32(0.5))
jax.block_until_ready(got)
check("with-form bitwise", bw(ref, got))
check("with-form collected 5 events", len(rec.events) == 5, f"({len(rec.events)})")
check("exit restored lax.scan", jax.lax.scan is orig_scan)
post = f(jnp.float32(0.5))   # delete-the-with: same call, no context
check("after-with call bitwise + no new events",
      bw(ref, post) and len(rec.events) == 5, f"(events still {len(rec.events)})")

# --- 2: PHANTOM EMISSION (jitted inside, called after) ---
fj = jax.jit(f)
with tap.record() as rec2:
    r_in = fj(jnp.float32(0.5))     # traced INSIDE ctx -> callbacks baked into cache
    jax.block_until_ready(r_in)
n_inside = len(rec2.events)
r_out = fj(jnp.float32(0.5))        # SAME compiled artifact, ctx exited
jax.block_until_ready(r_out)
n_after = len(rec2.events)
check("jit-inside-ctx collected during ctx", n_inside > 0, f"({n_inside})")
print(f"[INFO] phantom emission after exit: events {n_inside} -> {n_after} "
      f"({'PHANTOM +' + str(n_after - n_inside) if n_after > n_inside else 'none'})")
check("jit-inside bitwise both calls", bw(ref, r_in) and bw(ref, r_out))

# --- 3: jitted call INSIDE ctx (fresh trace under jit) ---
def g2(x0):
    return jax.jit(f)(x0)
with tap.record() as rec3:
    got3 = g2(jnp.float32(0.5))
jax.block_until_ready(got3)
check("jit-wrapped call inside ctx: bitwise", bw(ref, got3))
check("jit-wrapped call inside ctx: events", len(rec3.events) == 5, f"({len(rec3.events)})")

# --- 4: grad / vmap THROUGH the with-form ---
gref = jax.grad(f)(jnp.float32(0.5))
with tap.record() as rec4:
    ggot = jax.grad(f)(jnp.float32(0.5))
jax.block_until_ready(ggot)
check("grad through with-form bitwise", bw(gref, ggot),
      f"(ref={float(gref):.6f} got={float(ggot):.6f})")

x0b = jnp.arange(3, dtype=jnp.float32)
vref = jax.vmap(f)(x0b)
with tap.record() as rec5:
    vgot = jax.vmap(f)(x0b)
jax.block_until_ready(vgot)
check("vmap through with-form bitwise", bw(vref, vgot), f"(events={len(rec5.events)})")

check("lax.scan restored at end of all probes", jax.lax.scan is orig_scan)
print("\n" + ("M1B AYS R1: ALL PASS" if not FAILS else f"FAILURES: {FAILS}"))
