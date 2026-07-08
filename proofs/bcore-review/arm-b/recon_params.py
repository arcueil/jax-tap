"""Reconnaissance: dump the params of scan/while/jit/pjit/cond eqns and the
public jax.lax.scan signature. Establishes ground truth for the attacks."""
import inspect
import jax
import jax.numpy as jnp


def show(name, closed):
    print(f"\n=== {name} ===")
    for eqn in closed.jaxpr.eqns:
        print(f"  prim={eqn.primitive.name!r} multiple_results={eqn.primitive.multiple_results}")
        for k, v in eqn.params.items():
            r = repr(v)
            if len(r) > 90:
                r = r[:90] + "..."
            print(f"      param {k!r}: {r}")


# scan
def f_scan(x0, xs):
    return jax.lax.scan(lambda c, x: (c + x, c * x), x0, xs)
show("scan", jax.make_jaxpr(f_scan)(jnp.float32(0.5), jnp.arange(3.0, dtype=jnp.float32)))

# while
def f_while(v0):
    return jax.lax.while_loop(lambda c: c < 5.0, lambda c: c + 1.0, v0)
show("while", jax.make_jaxpr(f_while)(jnp.float32(0.0)))

# jit
def f_jit(x):
    return jax.jit(lambda y: y * 2.0)(x)
show("jit", jax.make_jaxpr(f_jit)(jnp.float32(1.0)))

# pjit w/ shardings
def f_pjit(x):
    g = jax.jit(lambda y: y + 1.0, donate_argnums=0, inline=True)
    return g(x)
show("jit-donate-inline", jax.make_jaxpr(f_pjit)(jnp.float32(1.0)))

# cond
def f_cond(x):
    return jax.lax.cond(x > 0, lambda a: a * 2, lambda a: a * 3, x)
show("cond", jax.make_jaxpr(f_cond)(jnp.float32(1.0)))

# switch
def f_switch(i, x):
    return jax.lax.switch(i, [lambda a: a + 1, lambda a: a + 2, lambda a: a + 3], x)
show("switch", jax.make_jaxpr(f_switch)(jnp.int32(1), jnp.float32(1.0)))

# scan public signature
print("\n=== jax.lax.scan signature ===")
print(inspect.signature(jax.lax.scan))
print("\n=== jax.lax.while_loop signature ===")
print(inspect.signature(jax.lax.while_loop))
