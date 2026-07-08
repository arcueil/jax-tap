"""FIX-REVIEW (protocol item 4) of the F1+F2 remediation (a7b9658).

The fix RECURSES into cond/switch/remat2 and re-emits via jax.lax.switch /
jax.checkpoint. Hostile questions:
  1. Does it preserve BITWISE IDENTITY on cond/switch/remat2/jit programs?
  2. Does it preserve GRAD (the fix recurses into AD-relevant higher-order prims)?
  3. Does it preserve VMAP?
  4. Is F1 actually fixed (events now fire inside cond/switch/remat)?
  5. Is F2 actually fixed (unique paths across jit)?
  6. Compositions: cond-in-scan, grad over the lot.
A fix that corrupts a VALUE or GRAD is worse than the F1 it cures -> BLOCKER.
"""
import numpy as np
import jax
import jax.numpy as jnp
import jaxtap as tap

FAILS = []
def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILS.append(name)

def bw(a, b):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return len(la) == len(lb) and all(
        np.asarray(x).tobytes() == np.asarray(y).tobytes() for x, y in zip(la, lb))

xs = jnp.arange(5.0, dtype=jnp.float32)

# ---------- programs with a scan nested inside each higher-order prim ----------
def scanfn(c0):
    c, _ = jax.lax.scan(lambda c, x: (c * 1.01 + x, c), c0, xs)
    return c

def f_cond(p, c0):
    return jax.lax.cond(p > 0, lambda z: scanfn(z), lambda z: scanfn(z * 2.0), c0)

def f_switch(i, c0):
    return jax.lax.switch(i, [lambda z: scanfn(z), lambda z: scanfn(z + 1.0),
                              lambda z: scanfn(z * 3.0)], c0)

def f_remat(c0):
    return jax.checkpoint(scanfn)(c0)

def f_jit_siblings(c0):
    a = scanfn(c0)                    # top-level scan -> scan[0]
    b = jax.jit(scanfn)(c0 + 1.0)    # jit-nested scan -> jit[1]/scan[0]
    return a + b

# ---------- 1: bitwise identity forward ----------
for name, f, args in [
    ("cond forward", f_cond, (jnp.float32(1.0), jnp.float32(0.5))),
    ("switch forward", f_switch, (jnp.int32(1), jnp.float32(0.5))),
    ("remat forward", f_remat, (jnp.float32(0.5),)),
    ("jit-siblings forward", f_jit_siblings, (jnp.float32(0.5),)),
]:
    ref = f(*args)
    got = tap.verbose(f, on_step=lambda e: None)(*args)
    jax.block_until_ready(got)
    check(name, bw(ref, got), f"(ref={float(ref):.5f} got={float(got):.5f})")

# ---------- 2: grad correctness through the fix ----------
for name, f, arg in [
    ("grad(cond)", lambda c: f_cond(jnp.float32(1.0), c), jnp.float32(0.5)),
    ("grad(switch)", lambda c: f_switch(jnp.int32(2), c), jnp.float32(0.5)),
    ("grad(remat)", f_remat, jnp.float32(0.5)),
    ("grad(jit-siblings)", f_jit_siblings, jnp.float32(0.5)),
]:
    gref = jax.grad(f)(arg)
    ggot = jax.grad(tap.verbose(f, on_step=lambda e: None))(arg)
    jax.block_until_ready(ggot)
    check(name, bw(gref, ggot), f"(ref={float(gref):.5f} got={float(ggot):.5f})")

# ---------- 3: vmap over cond-with-scan ----------
pv = jnp.array([1.0, -1.0, 1.0], dtype=jnp.float32)
cv = jnp.array([0.5, 0.7, 0.9], dtype=jnp.float32)
ref_v = jax.vmap(f_cond)(pv, cv)
got_v = jax.vmap(tap.verbose(f_cond, on_step=lambda e: None))(pv, cv)
jax.block_until_ready(got_v)
check("vmap(cond)", bw(ref_v, got_v))

# ---------- 4: F1 actually fixed (events fire inside cond/switch/remat) ----------
for name, f, args, expect in [
    ("F1 cond events", f_cond, (jnp.float32(1.0), jnp.float32(0.5)), 5),
    ("F1 switch events", f_switch, (jnp.int32(1), jnp.float32(0.5)), 5),
    ("F1 remat events", f_remat, (jnp.float32(0.5),), 5),
]:
    ev = []
    jax.block_until_ready(tap.verbose(f, on_step=ev.append)(*args))
    paths = sorted({e.path for e in ev})
    check(name, len(ev) == expect, f"({len(ev)} events, paths={paths})")

# ---------- 5: F2 fixed (unique paths across jit) ----------
ev = []
jax.block_until_ready(tap.verbose(f_jit_siblings, on_step=ev.append)(jnp.float32(0.5)))
paths = sorted({e.path for e in ev})
check("F2 unique jit paths", len(paths) == 2 and any("jit" in p for p in paths),
      f"(paths={paths})")

# ---------- 6: composition — cond inside scan, grad over it ----------
def f_cond_in_scan(c0):
    def body(c, x):
        c2 = jax.lax.cond(x > 2.0, lambda z: z + 1.0, lambda z: z * 1.1, c)
        return c2, c2
    c, _ = jax.lax.scan(body, c0, xs)
    return c
gref = jax.grad(f_cond_in_scan)(jnp.float32(0.5))
ggot = jax.grad(tap.verbose(f_cond_in_scan, on_step=lambda e: None))(jnp.float32(0.5))
jax.block_until_ready(ggot)
check("grad(cond-in-scan)", bw(gref, ggot), f"(ref={float(gref):.5f} got={float(ggot):.5f})")

print("\n" + ("ALL FIX-REVIEW CHECKS PASSED" if not FAILS else f"FAILURES: {FAILS}"))
