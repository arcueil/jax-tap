"""R2: was the vmap 'failure' my probe's unbatched tap value? sin(c*x) is
carry-dependent (batched) -> expect per-lane firing = lanes * N/se."""
import numpy as np
import jax, jax.numpy as jnp
import jaxtap as tap

def f(x0):
    def body(c, x):
        return c + jnp.sin(c * x) * 0.01, None   # sin arg BATCHED (depends on carry)
    c, _ = jax.lax.scan(body, x0, jnp.arange(30.0))
    return c

evs = []
g = jax.vmap(tap.verbose(f, on_step=evs.append, sample_every=10, taps=[tap.on("sin")]))
ref = jax.vmap(f)(jnp.arange(3.0))
out = g(jnp.arange(3.0)); jax.block_until_ready(out)
sin_ev = [e for e in evs if "sin" in e.path]
bw = all(np.asarray(r).tobytes() == np.asarray(o).tobytes()
         for r, o in zip(jax.tree_util.tree_leaves(ref), jax.tree_util.tree_leaves(out)))
print(f"batched tap value: {len(sin_ev)} events (expect 9 = 3 lanes x 3 sampled), bitwise={bw}")
print("VERDICT:", "PROBE BUG — per-lane firing works; unbatched values single-fire (documented #964 duality)"
      if len(sin_ev) == 9 and bw else "REAL DEFECT — investigate")
