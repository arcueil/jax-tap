import jax, jax.numpy as jnp
import inspect

# What params does a scan eqn carry, and does jax.lax.scan accept them?
def f(x0, xs):
    return jax.lax.scan(lambda c,x:(c+x, c*x), x0, xs)

cj = jax.make_jaxpr(f)(jnp.float32(0.5), jnp.arange(4.0, dtype=jnp.float32))
for eqn in cj.jaxpr.eqns:
    if eqn.primitive.name == 'scan':
        print("SCAN PARAMS keys:", sorted(eqn.params.keys()))
        print("  linear =", eqn.params.get('linear'))
        print("  num_consts =", eqn.params.get('num_consts'), "num_carry =", eqn.params.get('num_carry'))
        print("  length =", eqn.params.get('length'))

print("\njax.lax.scan signature:", inspect.signature(jax.lax.scan))

# while params
def g(v0):
    return jax.lax.while_loop(lambda c: c < 5.0, lambda c: c+1.0, v0)
cjg = jax.make_jaxpr(g)(jnp.float32(0.0))
for eqn in cjg.jaxpr.eqns:
    if eqn.primitive.name == 'while':
        print("\nWHILE PARAMS keys:", sorted(eqn.params.keys()))
