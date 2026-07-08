"""ATTACK: higher-order autodiff through verbose(scan)."""
import jax, jax.numpy as jnp
import numpy as np
import jaxtap as tap

xs = jnp.arange(1.0, 6.0, dtype=jnp.float32)
def scan_f(theta):
    # carry accumulates theta*x each step; return final carry (nonlinear in theta)
    final, _ = jax.lax.scan(lambda c, x: (c*jnp.sin(c) + theta*x, c), theta, xs)
    return final

theta = jnp.float32(0.7)
v = tap.verbose(scan_f, on_step=lambda e: None)

def _b(x): return np.asarray(x).tobytes()

# value
print("f      :", float(scan_f(theta)), "| verbose:", float(v(theta)),
      "| bitwise:", _b(scan_f(theta))==_b(v(theta)))
# grad
g_ref = jax.grad(scan_f)(theta); g_got = jax.grad(v)(theta)
print("grad   :", float(g_ref), "| verbose:", float(g_got), "| bitwise:", _b(g_ref)==_b(g_got))
# grad^2
g2_ref = jax.grad(jax.grad(scan_f))(theta); g2_got = jax.grad(jax.grad(v))(theta)
print("grad^2 :", float(g2_ref), "| verbose:", float(g2_got), "| bitwise:", _b(g2_ref)==_b(g2_got))
# grad^3
g3_ref = jax.grad(jax.grad(jax.grad(scan_f)))(theta)
g3_got = jax.grad(jax.grad(jax.grad(v)))(theta)
print("grad^3 :", float(g3_ref), "| verbose:", float(g3_got), "| bitwise:", _b(g3_ref)==_b(g3_got))

# hessian on a vector-input scan
xs2 = jnp.arange(1.0,4.0,dtype=jnp.float32)
def scan_vec(p):  # p is a 2-vector
    final,_ = jax.lax.scan(lambda c,x:(c*c + p[0]*x + p[1], c), p[0]+p[1], xs2)
    return final
vv = tap.verbose(scan_vec, on_step=lambda e: None)
p = jnp.array([0.3, 0.5], dtype=jnp.float32)
H_ref = jax.hessian(scan_vec)(p); H_got = jax.hessian(vv)(p)
print("hessian bitwise:", _b(H_ref)==_b(H_got))
print("  H_ref=\n", np.asarray(H_ref), "\n  H_got=\n", np.asarray(H_got))
