import jax, jax.numpy as jnp
import numpy as np
import jaxtap as tap
def _b(x): return [np.asarray(v).tobytes() for v in jax.tree_util.tree_leaves(x)]

N=4
def scan_f(c, xs):
    return jax.lax.scan(lambda a,x:(jax.nn.softplus(a+x), a), c, xs)

# ---- nested vmap over scan: vmap(vmap(verbose(f))) ----
L1, L2 = 2, 3
carry = jnp.ones((L1, L2), dtype=jnp.float32)
xsb = jnp.broadcast_to(jnp.arange(float(N), dtype=jnp.float32), (L1, L2, N))
ref = jax.vmap(jax.vmap(scan_f))(carry, xsb)
ev=[]
got = jax.vmap(jax.vmap(tap.verbose(scan_f, on_step=lambda e: ev.append(e))))(carry, xsb)
jax.block_until_ready(got)
print("nested vmap(vmap) scan bitwise:", _b(ref)==_b(got),
      "| events:", len(ev), "expected", L1*L2*N)

# ---- batched CARRY only (xs shared/unbatched via in_axes) ----
carryb = jnp.arange(1.0, 1.0+3, dtype=jnp.float32)  # 3 lanes
xs_shared = jnp.arange(float(N), dtype=jnp.float32)
ref2 = jax.vmap(scan_f, in_axes=(0, None))(carryb, xs_shared)
ev=[]
got2 = jax.vmap(tap.verbose(scan_f, on_step=lambda e: ev.append(e)), in_axes=(0, None))(carryb, xs_shared)
jax.block_until_ready(got2)
print("batched-carry-only scan bitwise:", _b(ref2)==_b(got2), "| events:", len(ev), "expected", 3*N)

# ---- batched XS only (carry shared) ----
xsb2 = jnp.stack([jnp.arange(float(N))+k for k in range(3)]).astype(jnp.float32)
ref3 = jax.vmap(scan_f, in_axes=(None, 0))(jnp.float32(1.0), xsb2)
ev=[]
got3 = jax.vmap(tap.verbose(scan_f, on_step=lambda e: ev.append(e)), in_axes=(None,0))(jnp.float32(1.0), xsb2)
jax.block_until_ready(got3)
print("batched-xs-only scan bitwise:", _b(ref3)==_b(got3), "| events:", len(ev), "expected", 3*N)

# ---- vmap grad (per-lane grad through transform) ----
def loss(c, xs):
    _, ys = scan_f(c, xs)
    return jnp.sum(ys)
gref = jax.vmap(jax.grad(loss), in_axes=(0,None))(carryb, xs_shared)
ggot = jax.vmap(jax.grad(tap.verbose(loss, on_step=lambda e:None)), in_axes=(0,None))(carryb, xs_shared)
jax.block_until_ready(ggot)
print("vmap(grad) bitwise:", _b(gref)==_b(ggot))

# ---- select with a LOSSY reduction: on-device, must not touch primal ----
def simple(c, xs):
    return jax.lax.scan(lambda a,x:(a+x, a*x), c, xs)
x0 = jnp.float32(1.0); xs = jnp.arange(float(N), dtype=jnp.float32)
ref_val = simple(x0, xs)
ev=[]
got_val = tap.verbose(simple, on_step=lambda e: ev.append(e),
                      select=lambda leaves: leaves[0].mean() + leaves[0].sum())(x0, xs)
jax.block_until_ready(got_val)
print("\nselect(lossy sum/mean) primal bitwise:", _b(ref_val)==_b(got_val), "| events:", len(ev))
print("  selected values:", [float(np.asarray(e.value)) for e in ev])

# ---- select returning integer/bool ----
ev=[]
got_i = tap.verbose(simple, on_step=lambda e: ev.append(e),
                    select=lambda leaves: (leaves[0] > 2.0))(x0, xs)
jax.block_until_ready(got_i)
print("select->bool primal bitwise:", _b(ref_val)==_b(got_i),
      "| value dtypes:", [np.asarray(e.value).dtype for e in ev])

# ---- complex grad through transform (holomorphic) ----
def cplx_loss(theta):  # theta complex
    final,_ = jax.lax.scan(lambda c,x:(c*theta + x, c), theta, jnp.arange(3, dtype=jnp.complex64)+1j)
    return final
th = jnp.complex64(0.3+0.2j)
gr = jax.grad(cplx_loss, holomorphic=True)(th)
gg = jax.grad(tap.verbose(cplx_loss, on_step=lambda e:None), holomorphic=True)(th)
print("\ncomplex holomorphic grad bitwise:", _b(gr)==_b(gg), "| ref", complex(gr), "got", complex(gg))
