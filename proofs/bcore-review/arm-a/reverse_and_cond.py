import jax, jax.numpy as jnp
import numpy as np
import jaxtap as tap
def _b(x): return [np.asarray(v).tobytes() for v in jax.tree_util.tree_leaves(x)]

xs = jnp.arange(1.0, 6.0, dtype=jnp.float32)

# --- reverse=True grad ---
def rev_scan(theta):
    final,_ = jax.lax.scan(lambda c,x:(c*1.1 + theta*jnp.sin(x), c), theta, xs, reverse=True)
    return final
vr = tap.verbose(rev_scan, on_step=lambda e: None)
print("reverse fwd bitwise:", _b(rev_scan(0.7))==_b(vr(jnp.float32(0.7))))
gr, gg = jax.grad(rev_scan)(jnp.float32(0.7)), jax.grad(vr)(jnp.float32(0.7))
print("reverse grad:", float(gr), float(gg), "bitwise:", _b(gr)==_b(gg))
g2r, g2g = jax.grad(jax.grad(rev_scan))(jnp.float32(0.7)), jax.grad(jax.grad(vr))(jnp.float32(0.7))
print("reverse grad^2:", float(g2r), float(g2g), "bitwise:", _b(g2r)==_b(g2g))

# --- reverse event step-order: check steps still 0..N-1 ---
evs = []
tap.verbose(rev_scan, on_step=lambda e: evs.append(e))(jnp.float32(0.7))
jax.block_until_ready(None)
print("reverse event steps:", [e.step for e in evs if e.path=='scan[0]'])

# --- lax.cond inside scan body ---
def cond_scan(theta):
    def body(c, x):
        c2 = jax.lax.cond(x > 2.5, lambda z: z*2.0, lambda z: z+theta, c)
        return c2, c2
    final, ys = jax.lax.scan(body, theta, xs)
    return jnp.sum(ys)
vc = tap.verbose(cond_scan, on_step=lambda e: None)
print("\ncond-in-scan fwd bitwise:", _b(cond_scan(0.7))==_b(vc(jnp.float32(0.7))))
gcr, gcg = jax.grad(cond_scan)(jnp.float32(0.7)), jax.grad(vc)(jnp.float32(0.7))
print("cond-in-scan grad:", float(gcr), float(gcg), "bitwise:", _b(gcr)==_b(gcg))

# --- lax.switch inside scan body ---
def switch_scan(theta):
    def body(c, x):
        idx = (x.astype(jnp.int32)) % 3
        c2 = jax.lax.switch(idx, [lambda z: z+theta, lambda z: z*1.5, lambda z: z-0.1], c)
        return c2, c2
    _, ys = jax.lax.scan(body, theta, xs)
    return jnp.sum(ys)
vs = tap.verbose(switch_scan, on_step=lambda e: None)
print("\nswitch-in-scan fwd bitwise:", _b(switch_scan(0.7))==_b(vs(jnp.float32(0.7))))
gsr, gsg = jax.grad(switch_scan)(jnp.float32(0.7)), jax.grad(vs)(jnp.float32(0.7))
print("switch-in-scan grad:", float(gsr), float(gsg), "bitwise:", _b(gsr)==_b(gsg))
