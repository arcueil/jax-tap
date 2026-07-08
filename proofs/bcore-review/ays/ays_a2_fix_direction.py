"""Is A2's fix really 'add remat2 to _JIT_PRIMS'? That branch re-wraps the body
in jax.jit — which would DROP remat2's checkpoint semantics (prevent_cse /
the don't-save-intermediates boundary). Probe the correct fix direction.

Question 1: does remat2 need special handling vs jit? (params differ)
Question 2: can we recurse into remat2.jaxpr and RE-BIND via remat2 (preserving
            the primitive) rather than re-wrapping in jit?
"""
import jax
import jax.numpy as jnp

def inner(x):
    c, _ = jax.lax.scan(lambda a, b: (a + b, a), x, jnp.arange(4.0))
    return c

def f(x):
    return jax.checkpoint(inner)(x)

closed = jax.make_jaxpr(f)(jnp.float32(1.0))
remat_eqn = [e for e in closed.jaxpr.eqns if e.primitive.name == "remat2"][0]
print("remat2 params:", sorted(remat_eqn.params))
print("  prevent_cse:", remat_eqn.params.get("prevent_cse"),
      " policy:", remat_eqn.params.get("policy"),
      " differentiated:", remat_eqn.params.get("differentiated"))

# get_bind_params for remat2 — can we recurse-then-rebind preserving the primitive?
gbp = remat_eqn.primitive.get_bind_params(remat_eqn.params)
print("  get_bind_params ->", type(gbp).__name__, "keys:", sorted(gbp) if isinstance(gbp, dict) else gbp)

# Contrast with jit's params (what _JIT_PRIMS re-wrap would impose):
jit_closed = jax.make_jaxpr(jax.jit(inner))(jnp.float32(1.0))
jit_eqn = [e for e in jit_closed.jaxpr.eqns if e.primitive.name in ("jit", "pjit")][0]
print("\njit params (what re-wrap uses):", sorted(jit_eqn.params))
print("=> remat2 has prevent_cse/policy/differentiated that jit LACKS.")
print("=> Naive 'add remat2 to _JIT_PRIMS' (jax.jit re-wrap) DROPS those =>",
      "loses the checkpoint boundary (memory regression). WRONG fix.")
print("=> Correct: recurse into remat2.jaxpr, rebuild instrumented, RE-BIND via",
      "remat2.bind with get_bind_params (preserve prevent_cse/policy). Verify at fix time.")
