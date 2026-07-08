# Probe the JAX interpreter surface on the installed version.
# What we need for the B-core (jaxpr-walker) transform:
#   1. make_jaxpr -> ClosedJaxpr with eqns we can walk
#   2. identify scan/while/cond/pjit eqns + their sub-jaxpr params
#   3. re-evaluate a (body) jaxpr as a callable: eval_jaxpr or equivalent
import jax
import jax.numpy as jnp

print("jax", jax.__version__)

# --- symbol availability across the historic API shuffle ---
candidates = [
    ("jax.core.eval_jaxpr", lambda: jax.core.eval_jaxpr),
    ("jax.core.ClosedJaxpr", lambda: jax.core.ClosedJaxpr),
    ("jax.core.Jaxpr", lambda: jax.core.Jaxpr),
    ("jax.core.jaxpr_as_fun", lambda: jax.core.jaxpr_as_fun),
    ("jax.extend.core.ClosedJaxpr", lambda: __import__("jax.extend.core", fromlist=["x"]).ClosedJaxpr),
    ("jax.extend.core.Jaxpr", lambda: __import__("jax.extend.core", fromlist=["x"]).Jaxpr),
    ("jax.extend.core.jaxpr_as_fun", lambda: __import__("jax.extend.core", fromlist=["x"]).jaxpr_as_fun),
    ("jax.extend.core.primitives", lambda: __import__("jax.extend.core", fromlist=["x"]).primitives),
]
for name, get in candidates:
    try:
        get()
        print(f"  OK       {name}")
    except Exception as e:
        print(f"  MISSING  {name}  ({type(e).__name__})")

# --- what does a jitted scan-containing program's jaxpr look like? ---
def inner(c, x):
    return c + x, c * x

def f(x0, xs):
    c, ys = jax.lax.scan(inner, x0, xs)
    # nested control flow: a while inside, to see its eqn shape too
    c2 = jax.lax.while_loop(lambda v: v < 10.0, lambda v: v + 1.0, c)
    return c2, ys

x0 = jnp.float32(0.0)
xs = jnp.arange(5, dtype=jnp.float32)

closed = jax.make_jaxpr(f)(x0, xs)
print("\ntop-level eqn primitives:", [e.primitive.name for e in closed.jaxpr.eqns])
for eqn in closed.jaxpr.eqns:
    if eqn.primitive.name == "scan":
        print("scan params keys:", sorted(eqn.params.keys()))
        print("  num_consts:", eqn.params["num_consts"], " num_carry:", eqn.params["num_carry"],
              " length:", eqn.params["length"])
        print("  body type:", type(eqn.params["jaxpr"]).__name__)
    if eqn.primitive.name == "while":
        print("while params keys:", sorted(eqn.params.keys()))

# --- under jit: is everything wrapped in a pjit eqn? ---
closed_jit = jax.make_jaxpr(jax.jit(f))(x0, xs)
print("\njitted top-level eqns:", [e.primitive.name for e in closed_jit.jaxpr.eqns])
pj = closed_jit.jaxpr.eqns[0]
if pj.primitive.name in ("pjit", "jit", "closed_call"):
    print("pjit params keys:", sorted(pj.params.keys()))
    print("  inner eqns:", [e.primitive.name for e in pj.params["jaxpr"].jaxpr.eqns])
