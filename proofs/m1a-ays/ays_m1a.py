"""AYS on M1a primitive taps — probing the seams the tests DON'T cover.

1. vmap x primitive tap (agent's reasoned-only claim: fires LANES*N, no crash)
2. step threading through a COND branch inside a scan (closure-captured claim)
3. step threading inside a WHILE loop (step from while's own counter?)
4. live step values actually vary 0..N-1 through the jit boundary (the
   constant-fold risk the agent claims to have fixed with explicit-arg)
5. reverse=True: step = iteration index (0..N-1), documented semantics
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

N = 5
xs = jnp.arange(float(N), dtype=jnp.float32)

# ---- 4: live step varies through the jit boundary (cholesky case) ----
def f_chol(c0):
    def body(c, x):
        M = jnp.eye(2, dtype=jnp.float32) * (c + x + 1.0)
        L = jnp.linalg.cholesky(M)
        return c + jnp.sum(L) * 0.01, c
    c, _ = jax.lax.scan(body, c0, xs)
    return c

ev = []
g = tap.verbose(f_chol, on_step=ev.append,
                taps=[tap.on("cholesky", select=lambda outs: jnp.sum(outs[0]))])
ref = f_chol(jnp.float32(1.0)); got = g(jnp.float32(1.0)); jax.block_until_ready(got)
steps = sorted(e.step for e in ev if "cholesky" in e.path)
vals = [float(e.value) for e in sorted(ev, key=lambda e: e.step) if "cholesky" in e.path]
check("live step varies through jit", steps == list(range(N)), f"(steps={steps})")
check("cholesky values vary per step (not constant-folded)",
      len(set(np.round(vals, 4))) == N, f"(vals={np.round(vals,3).tolist()})")
check("bitwise with prim tap", bw(ref, got))

# ---- 1: vmap x primitive tap (the reasoned-only claim) ----
ev.clear()
LANES = 3
c0b = jnp.arange(1.0, 1.0 + LANES, dtype=jnp.float32)
try:
    gv = jax.vmap(tap.verbose(f_chol, on_step=ev.append,
                              taps=[tap.on("cholesky", select=lambda o: jnp.sum(o[0]))]))
    ref_v = jax.vmap(f_chol)(c0b)
    got_v = gv(c0b); jax.block_until_ready(got_v)
    n_chol = sum(1 for e in ev if "cholesky" in e.path)
    check("vmap prim-tap bitwise", bw(ref_v, got_v))
    check("vmap prim-tap fires LANES*N", n_chol == LANES * N, f"({n_chol} vs {LANES*N})")
except Exception as e:
    check("vmap prim-tap", False, f"RAISED {type(e).__name__}: {e}")

# ---- 2: prim tap inside a cond branch inside a scan (step via closure claim) ----
def f_cond(c0):
    def body(c, x):
        c2 = jax.lax.cond(x > 2.0,
                          lambda z: jnp.sum(jnp.linalg.cholesky(jnp.eye(2) * (z + 1.0))) + z,
                          lambda z: z * 1.1,
                          c)
        return c2, c2
    c, _ = jax.lax.scan(body, c0, xs)
    return c

ev.clear()
gc = tap.verbose(f_cond, on_step=ev.append,
                 taps=[tap.on("cholesky", select=lambda o: jnp.sum(o[0]))])
refc = f_cond(jnp.float32(0.5)); gotc = gc(jnp.float32(0.5)); jax.block_until_ready(gotc)
chol_ev = [e for e in ev if "cholesky" in e.path]
# xs = 0..4; x>2 for x=3,4 -> the true branch RUNS at steps 3,4. But cond under
# the walker instruments BOTH branches; only the taken branch executes at runtime.
steps_c = sorted(e.step for e in chol_ev)
paths_c = sorted({e.path for e in chol_ev})
check("cond prim-tap bitwise", bw(refc, gotc))
check("cond prim-tap fires only on taken-branch steps", steps_c == [3, 4],
      f"(steps={steps_c}, paths={paths_c})")

# ---- 3: prim tap inside a while loop (step from while's counter) ----
def f_while(v0):
    def cond_fn(c): return c[0] < 5.0
    def body_fn(c):
        v, acc = c
        L = jnp.linalg.cholesky(jnp.eye(2) * (v + 1.0))
        return (v + 1.0, acc + jnp.sum(L))
    return jax.lax.while_loop(cond_fn, body_fn, (v0, jnp.float32(0.0)))

ev.clear()
gw = tap.verbose(f_while, on_step=ev.append,
                 taps=[tap.on("cholesky", select=lambda o: jnp.sum(o[0]))])
refw = f_while(jnp.float32(0.0)); gotw = gw(jnp.float32(0.0)); jax.block_until_ready(gotw)
chol_w = sorted(e.step for e in ev if "cholesky" in e.path)
check("while prim-tap bitwise", bw(refw, gotw))
check("while prim-tap live steps 0..4", chol_w == list(range(5)), f"(steps={chol_w})")

# ---- 5: reverse=True step semantics ----
def f_rev(c0):
    def body(c, x):
        L = jnp.linalg.cholesky(jnp.eye(2) * (x + 1.0))
        return c + jnp.sum(L), c
    c, _ = jax.lax.scan(body, c0, xs, reverse=True)
    return c

ev.clear()
gr = tap.verbose(f_rev, on_step=ev.append,
                 taps=[tap.on("cholesky", select=lambda o: o[0][0, 0])])
refr = f_rev(jnp.float32(0.0)); gotr = gr(jnp.float32(0.0)); jax.block_until_ready(gotr)
rev_ev = sorted([e for e in ev if "cholesky" in e.path], key=lambda e: e.step)
# iteration 0 should process x=4 (reverse) -> L[0,0]=sqrt(5)~2.236
first_val = float(rev_ev[0].value) if rev_ev else None
check("reverse: step=iteration idx, first iter sees LAST x",
      rev_ev and abs(first_val - np.sqrt(5.0)) < 1e-5,
      f"(step0 L00={first_val:.4f}, sqrt(5)={np.sqrt(5.0):.4f})")
check("reverse bitwise", bw(refr, gotr))

print("\n" + ("M1A AYS ROUND 1: ALL PASS" if not FAILS else f"FAILURES: {FAILS}"))
