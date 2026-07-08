"""AYS round 2 — attack the FIX. Fix-code gets the same hostility (protocol item 4).

The R1 fix + tests only exercise custom_JVP. Round 2 probes:
  A. custom_VJP primitive (different eqn: custom_vjp_call) — untested by R1.
  B. a custom_jvp whose rule is DELIBERATELY distinct from the primal derivative
     -> proves the CUSTOM rule survives verbose(), not just that grad happens to
     match primal autodiff.
  C. confirm get_bind_params returns a flat dict (the agent's correction of the TL).
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax import custom_jvp, custom_vjp
import jaxtap as tap


def close(a, b, tol=0):
    a, b = np.asarray(a), np.asarray(b)
    return (a.tobytes() == b.tobytes()) if tol == 0 else bool(np.allclose(a, b, atol=tol))


# ---- C: get_bind_params shape (verify the agent's arity correction) ----
def _prog(x0, xs):
    return jax.lax.scan(lambda c, x: (jax.nn.softplus(c + x), c), x0, xs)
closed = jax.make_jaxpr(_prog)(jnp.float32(0.5), jnp.arange(3.0, dtype=jnp.float32))
scan_body = [e for e in closed.jaxpr.eqns if e.primitive.name == "scan"][0].params["jaxpr"]
jit_inner = [e for e in scan_body.jaxpr.eqns if e.primitive.name == "jit"][0].params["jaxpr"]
cjc = [e for e in jit_inner.jaxpr.eqns if e.primitive.name == "custom_jvp_call"][0]
gbp = cjc.primitive.get_bind_params(cjc.params)
print(f"[C] get_bind_params returns {type(gbp).__name__} keys={sorted(gbp)} "
      f"(flat-dict claim {'CONFIRMED' if isinstance(gbp, dict) else 'REFUTED'})")


# ---- B: custom_jvp with a SENTINEL derivative distinct from the primal ----
@custom_jvp
def f_sentinel(x):
    return x * x                      # primal derivative would be 2x

@f_sentinel.defjvp
def _f_sentinel_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    return f_sentinel(x), jnp.float32(42.0) * dx    # sentinel: derivative == 42, not 2x

def loss_sentinel(theta):
    final, _ = jax.lax.scan(lambda c, x: (f_sentinel(c + x), c),
                            theta, jnp.arange(3.0, dtype=jnp.float32))
    return final

g_ref = jax.grad(loss_sentinel)(jnp.float32(0.7))
g_got = jax.grad(tap.verbose(loss_sentinel, on_step=lambda e: None))(jnp.float32(0.7))
jax.block_until_ready(g_got)
# If the custom rule survives, both use the sentinel; both should be bitwise equal
# AND the gradient must reflect the 42-sentinel chain, not the 2x primal.
primal_autodiff = jax.grad(lambda t: (lambda c: c)(  # what 2x-primal grad would give
    __import__("functools").reduce(lambda c, x: (c + x) ** 2, [0.0, 1.0, 2.0], t)))(jnp.float32(0.7))
print(f"[B] custom_jvp sentinel rule survives verbose:  "
      f"bitwise_eq(grad_ref,grad_verbose)={close(g_ref, g_got)}  "
      f"(ref={float(g_ref):.4f} verbose={float(g_got):.4f}; primal-2x-grad would be {float(primal_autodiff):.4f})")


# ---- A: custom_VJP (reverse-mode custom rule; different primitive) ----
@custom_vjp
def f_vjp(x):
    return jnp.sin(x)

def _f_vjp_fwd(x):
    return f_vjp(x), x

def _f_vjp_bwd(res, g):
    x = res
    return (jnp.float32(7.0) * g,)    # sentinel VJP: cotangent == 7, not cos(x)

f_vjp.defvjp(_f_vjp_fwd, _f_vjp_bwd)

def loss_vjp(theta):
    final, _ = jax.lax.scan(lambda c, x: (f_vjp(c + x), c),
                            theta, jnp.arange(3.0, dtype=jnp.float32))
    return final

try:
    fwd_ref = loss_vjp(jnp.float32(0.5))
    fwd_got = tap.verbose(loss_vjp, on_step=lambda e: None)(jnp.float32(0.5))
    jax.block_until_ready(fwd_got)
    gv_ref = jax.grad(loss_vjp)(jnp.float32(0.5))
    gv_got = jax.grad(tap.verbose(loss_vjp, on_step=lambda e: None))(jnp.float32(0.5))
    jax.block_until_ready(gv_got)
    print(f"[A] custom_vjp forward bitwise={close(fwd_ref, fwd_got)}  "
          f"grad-through-verbose bitwise={close(gv_ref, gv_got)}  "
          f"(ref={float(gv_ref):.4f} verbose={float(gv_got):.4f})")
except Exception as e:
    print(f"[A] custom_vjp RAISED {type(e).__name__}: {e}")
