import jax, jax.numpy as jnp
def scan_loss(theta):
    final,_ = jax.lax.scan(lambda c,x:(jnp.sin(c)+theta*x, c), theta, jnp.arange(1.0,6.0,dtype=jnp.float32))
    return final
cj = jax.make_jaxpr(jax.checkpoint(scan_loss))(jnp.float32(0.7))
names = [e.primitive.name for e in cj.jaxpr.eqns]
print("top-level primitives under jax.checkpoint(f):", names)
# Is there a scan directly, or is it wrapped in remat?
print("scan present at top level?:", 'scan' in names)
