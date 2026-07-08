import jax, jax.numpy as jnp
import numpy as np
import jaxtap as tap
def _b(x): return [np.asarray(v).tobytes() for v in jax.tree_util.tree_leaves(x)]
def show(name, ref, got, ev=None):
    ok = _b(ref)==_b(got)
    extra = f" | events={len(ev)}" if ev is not None else ""
    print(f"{name:38s} bitwise={ok}{extra}")
    if not ok:
        print("   ref:", jax.tree_util.tree_leaves(ref))
        print("   got:", jax.tree_util.tree_leaves(got))

# ---- integer carry ----
def int_scan(c0, xs):
    return jax.lax.scan(lambda c,x:(c+x, c*x), c0, xs)
c0 = jnp.int32(1); xsi = jnp.arange(5, dtype=jnp.int32)
ev=[]; got = tap.verbose(int_scan, on_step=lambda e: ev.append(e))(c0, xsi)
show("int32 carry", int_scan(c0,xsi), got, ev)

# ---- bool carry ----
def bool_scan(c0, xs):
    return jax.lax.scan(lambda c,x:(jnp.logical_xor(c, x), c), c0, xs)
b0 = jnp.bool_(True); xsb = jnp.array([True,False,True,True], dtype=jnp.bool_)
ev=[]; got = tap.verbose(bool_scan, on_step=lambda e: ev.append(e))(b0, xsb)
show("bool carry", bool_scan(b0,xsb), got, ev)

# ---- complex64 carry ----
def cplx_scan(c0, xs):
    return jax.lax.scan(lambda c,x:(c*x + 1j, c), c0, xs)
cc0 = jnp.complex64(0.5+0.5j); xsc = jnp.arange(4, dtype=jnp.complex64)+1j
ev=[]; got = tap.verbose(cplx_scan, on_step=lambda e: ev.append(e))(cc0, xsc)
show("complex64 carry", cplx_scan(cc0,xsc), got, ev)

# ---- mixed-dtype carry (int32 + float32 + bool) ----
def mixed_scan(carry, xs):
    def body(c, x):
        i, f, b = c
        return (i + x.astype(jnp.int32), f + jnp.sin(f), jnp.logical_not(b)), f
    return jax.lax.scan(body, carry, xs)
mc = (jnp.int32(0), jnp.float32(1.0), jnp.bool_(False)); xsm = jnp.arange(5.0, dtype=jnp.float32)
ev=[]; got = tap.verbose(mixed_scan, on_step=lambda e: ev.append(e))(mc, xsm)
show("mixed int/float/bool carry", mixed_scan(mc,xsm), got, ev)

# ---- weak-typed python-scalar carry ----
def weak_scan(c0, xs):
    return jax.lax.scan(lambda c,x:(c+x, c), c0, xs)
ev=[]; got = tap.verbose(weak_scan, on_step=lambda e: ev.append(e))(1.0, jnp.arange(4.0))  # c0 is python float
show("weak-typed python-scalar carry", weak_scan(1.0, jnp.arange(4.0)), got, ev)

# check output dtype exactly matches (weak type perturbation?)
r = weak_scan(1.0, jnp.arange(4.0)); g = tap.verbose(weak_scan, on_step=lambda e:None)(1.0, jnp.arange(4.0))
print("   weak dtypes ref:", [l.dtype for l in jax.tree_util.tree_leaves(r)], "got:", [l.dtype for l in jax.tree_util.tree_leaves(g)])

# ---- DEGENERATE ----
print("\n--- degenerate ---")
# length-0 scan
def len0(c0, xs):
    return jax.lax.scan(lambda c,x:(c+x, c*x), c0, xs)
xs0 = jnp.zeros((0,), dtype=jnp.float32)
ev=[]; got = tap.verbose(len0, on_step=lambda e: ev.append(e))(jnp.float32(3.0), xs0)
show("length-0 scan", len0(jnp.float32(3.0), xs0), got, ev)

# single-element scan
xs1 = jnp.array([2.0], dtype=jnp.float32)
ev=[]; got = tap.verbose(len0, on_step=lambda e: ev.append(e))(jnp.float32(3.0), xs1)
show("single-element scan", len0(jnp.float32(3.0), xs1), got, ev)

# 0-iteration while
def w0(v0):
    return jax.lax.while_loop(lambda c: c < 0.0, lambda c: c+1.0, v0)
ev=[]; got = tap.verbose(w0, on_step=lambda e: ev.append(e))(jnp.float32(5.0))
show("0-iteration while", w0(jnp.float32(5.0)), got, ev)

# length via `length=` with xs=None
def lenN(c0):
    return jax.lax.scan(lambda c,_:(c*1.5, c), c0, None, length=4)
ev=[]; got = tap.verbose(lenN, on_step=lambda e: ev.append(e))(jnp.float32(2.0))
show("scan xs=None length=4", lenN(jnp.float32(2.0)), got, ev)
