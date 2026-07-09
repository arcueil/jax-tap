"""Semantic progress bar for an UNBOUNDED while_loop: the carry's own
tempering lambda IS the progress fraction. No loop index could do this."""
import sys
import jax, jax.numpy as jnp
import jaxtap as tap

def adaptive_smc_ish(key):
    # unbounded loop: temper from lam=0 to 1, data-dependent step sizes
    def cond(c): return c[0] < 1.0
    def body(c):
        lam, key = c
        key, k = jax.random.split(key)
        dlam = 0.01 + 0.05 * jax.random.uniform(k)   # adaptive increment
        return (jnp.minimum(lam + dlam, 1.0), key)
    return jax.lax.while_loop(cond, body, (jnp.float32(0.), key))

def bar(e):  # host-side consumer: 40-char semantic progress bar
    lam = float(e.value)
    n = int(lam * 40)
    sys.stderr.write(f"\rtempering [{'#'*n}{'.'*(40-n)}] {lam*100:5.1f}%")

with tap.record(select=lambda leaves: leaves[0], on_step=bar):
    out = adaptive_smc_ish(jax.random.key(0))   # UNMODIFIED, UNBOUNDED loop
sys.stderr.write("\n")
print(f"done at lambda={float(out[0]):.3f} — a real 0-100% bar on a loop with total='?'")
