import jax
import jax.numpy as jnp


def f(x0, xs):
    def body(c, x):
        return jax.nn.softplus(c + x), c
    return jax.lax.scan(body, x0, xs)


x0 = jnp.float32(0.5)
xs = jnp.linspace(0.1, 1.0, 5, dtype=jnp.float32)
closed = jax.make_jaxpr(f)(x0, xs)
scan_eqn = [e for e in closed.jaxpr.eqns if e.primitive.name == "scan"][0]
body = scan_eqn.params["jaxpr"]
jit_eqn = [e for e in body.jaxpr.eqns if e.primitive.name == "jit"][0]
inner = jit_eqn.params["jaxpr"]
print("softplus jit inner eqns:", [e.primitive.name for e in inner.jaxpr.eqns])

for e in inner.jaxpr.eqns:
    if "custom" in e.primitive.name:
        print(f"\nFOUND: {e.primitive.name}, params={sorted(e.params)}")
        dummy = [jnp.float32(1.0)] * len(e.invars)
        # naive bind (walker's current _AD_PRIMS path)
        try:
            e.primitive.bind(*dummy, **e.params)
            print("  naive bind: OK")
        except Exception as ex:
            print(f"  naive bind: RAISED {type(ex).__name__}: {ex}   <-- THE CRASH")
        # canonical fix
        subfuns, bind_params = e.primitive.get_bind_params(e.params)
        print(f"  get_bind_params: subfuns={len(subfuns)}, bind_params keys={sorted(bind_params)}")
        try:
            e.primitive.bind(*subfuns, *dummy, **bind_params)
            print("  bind(*subfuns,*invals,**bind_params): OK  <-- THE FIX")
        except Exception as ex:
            print(f"  fixed bind: RAISED {type(ex).__name__}: {ex}")
