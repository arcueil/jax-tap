"""Does the L2 GC self-heal actually RESTORE the patch (not just drop events)?"""
import gc
import jax
import jax.numpy as jnp
import jaxtap as tap

orig_scan = jax.lax.scan
orig_while = jax.lax.while_loop

ctx = tap.record()
ctx.__enter__()
print("patched after manual enter:", jax.lax.scan is not orig_scan)

del ctx
gc.collect()

healed_scan = jax.lax.scan is orig_scan
healed_while = jax.lax.while_loop is orig_while
print("scan restored after del+gc:", healed_scan)
print("while restored after del+gc:", healed_while)

# and the machinery still works afterward
def f(x0):
    c, _ = jax.lax.scan(lambda c, x: (c + x, c), x0, jnp.arange(3.0))
    return c
with tap.record() as rec:
    r = f(jnp.float32(0.0))
print("fresh context after heal works:", len(rec.events) == 3, f"({len(rec.events)} events)")
print("VERDICT:", "SELF-HEAL COMPLETE" if healed_scan and healed_while else "PATCH STILL LEAKED (heal incomplete)")
