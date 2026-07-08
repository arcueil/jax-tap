import numpy as np, warnings
import jax, jax.numpy as jnp
import jaxtap as tap

FAILS = []
def check(name, ok, d=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {d}")
    if not ok: FAILS.append(name)
def bw(a, b):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return all(np.asarray(x).tobytes() == np.asarray(y).tobytes() for x, y in zip(la, lb))

xs = jnp.arange(5.0, dtype=jnp.float32)

# grad THROUGH a tapped cholesky in the loss path
def loss(theta):
    def body(c, x):
        L = jnp.linalg.cholesky(jnp.eye(2) * (c * theta + x + 1.0))
        return c + jnp.sum(L) * 0.1, c
    c, _ = jax.lax.scan(body, jnp.float32(1.0), xs)
    return c
gref = jax.grad(loss)(jnp.float32(1.3))
ggot = jax.grad(tap.verbose(loss, on_step=lambda e: None,
                            taps=[tap.on("cholesky", select=lambda o: jnp.sum(o[0]))]))(jnp.float32(1.3))
jax.block_until_ready(ggot)
check("grad THROUGH tapped cholesky", bw(gref, ggot), f"(ref={float(gref):.6f} got={float(ggot):.6f})")

# totality: raising on_step w/ prim taps, incl -W error
def boom(e): raise ValueError("boom")
ref = loss(jnp.float32(1.3))
with warnings.catch_warnings():
    warnings.simplefilter("error")
    got = tap.verbose(loss, on_step=boom,
                      taps=[tap.on("cholesky", select=lambda o: jnp.sum(o[0]))])(jnp.float32(1.3))
    jax.block_until_ready(got)
check("prim-tap totality under -W error", bw(ref, got))

# sample_every gates loop taps but NOT prim taps; both coexist
ev = []
g = tap.verbose(loss, on_step=ev.append, sample_every=2,
                taps=[tap.on("cholesky", select=lambda o: jnp.sum(o[0]))])
got2 = g(jnp.float32(1.3)); jax.block_until_ready(got2)
loop_steps = sorted(e.step for e in ev if e.path == "scan[0]")
prim_steps = sorted(e.step for e in ev if "cholesky" in e.path)
check("sample_every=2 gates loop taps", loop_steps == [0, 2, 4], f"({loop_steps})")
check("prim taps ungated (documented)", prim_steps == [0, 1, 2, 3, 4], f"({prim_steps})")
check("coexistence bitwise", bw(ref, got2))

print("\n" + ("M1A AYS ROUND 2: ALL PASS" if not FAILS else f"FAILURES: {FAILS}"))
