import jax, jax.numpy as jnp
import numpy as np
import jaxtap as tap
def _b(x): return [np.asarray(v).tobytes() for v in jax.tree_util.tree_leaves(x)]

xs = jnp.arange(1.0, 6.0, dtype=jnp.float32)   # length 5
def scan_loss(theta):
    final,_ = jax.lax.scan(lambda c,x:(jnp.sin(c)+theta*x, c), theta, xs)
    return final

theta = jnp.float32(0.7)

# ---- checkpoint(verbose(f)): does it double-fire under grad? ----
print("=== checkpoint(verbose(f)) ===")
ev = []
cv = jax.checkpoint(tap.verbose(scan_loss, on_step=lambda e: ev.append(e)))
# forward only
r_fwd = cv(theta); jax.block_until_ready(r_fwd)
print("fwd bitwise:", _b(scan_loss(theta))==_b(r_fwd), "| events after fwd-only:", len(ev))
ev.clear()
# under grad (checkpoint recomputes forward in backward)
g = jax.grad(cv)(theta); jax.block_until_ready(g)
gref = jax.grad(scan_loss)(theta)
print("grad bitwise:", _b(gref)==_b(g), "| events during grad:", len(ev), "(scan length =", len(xs), ")")

# ---- verbose(checkpoint(f)): are inner-scan taps present at all? ----
print("\n=== verbose(checkpoint(f)) ===")
ckf = jax.checkpoint(scan_loss)
ev2 = []
vc = tap.verbose(ckf, on_step=lambda e: ev2.append(e))
r2 = vc(theta); jax.block_until_ready(r2)
print("fwd bitwise:", _b(scan_loss(theta))==_b(r2), "| events (scan[0] should be 5):", len([e for e in ev2 if 'scan' in e.path]))
print("  all paths seen:", sorted({e.path for e in ev2}) or "(NONE)")
g2 = jax.grad(vc)(theta); jax.block_until_ready(g2)
print("grad bitwise:", _b(jax.grad(ckf)(theta))==_b(g2))
