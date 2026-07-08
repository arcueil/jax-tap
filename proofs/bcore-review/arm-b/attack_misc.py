"""ATTACK: additional structural edge cases hunting for a BLOCKER (crash or
silent-wrong) and more instances of the opaque-primitive-hides-CF bug class."""
import jax
import jax.numpy as jnp
import numpy as np
import jaxtap as tap
from collections import Counter


def _bytes(x):
    return [np.asarray(v).tobytes() for v in jax.tree_util.tree_leaves(x)]


def bitwise_eq(a, b):
    return _bytes(a) == _bytes(b)


xs = jnp.arange(4.0, dtype=jnp.float32)

print("=== 1. jit with ZERO array outputs (side-effect only) inside tapped scan ===")
def f_zero_out(x0):
    def body(c, x):
        @jax.jit
        def sidefx(v):
            jax.debug.print("")  # no return value used
            return ()
        sidefx(c)
        return c + x, c
    return jax.lax.scan(body, x0, xs)
try:
    ref = f_zero_out(jnp.float32(0.0))
    ev = []
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        got = tap.verbose(f_zero_out, on_step=lambda e: ev.append(e))(jnp.float32(0.0))
        jax.block_until_ready(got)
    print("  bitwise:", bitwise_eq(ref, got), "events:", len(ev), "-> clean" )
except Exception as exc:
    import traceback; traceback.print_exc()
    print("  VERDICT: CRASH", type(exc).__name__)


print("\n=== 2. remat/checkpoint HIDING a scan (opaque-primitive-hides-CF class) ===")
def f_remat(x0):
    @jax.checkpoint
    def inner(c0):
        out, _ = jax.lax.scan(lambda c, x: (c + x, c), c0, xs)
        return out
    return inner(x0)
try:
    ref = f_remat(jnp.float32(0.0))
    ev = []
    got = tap.verbose(f_remat, on_step=lambda e: ev.append(e))(jnp.float32(0.0))
    jax.block_until_ready(got)
    # what primitive does remat lower to?
    prims = {e.primitive.name for e in jax.make_jaxpr(f_remat)(jnp.float32(0.0)).jaxpr.eqns}
    print("  top-level prims:", prims)
    print("  bitwise:", bitwise_eq(ref, got), " events:", len(ev),
          " <-- EXPECTED 4; if 0 -> instrumentation DROPPED inside remat")
    print("  VERDICT:", "DROPPED instrumentation inside remat" if len(ev) == 0 else "walked")
except Exception as exc:
    import traceback; traceback.print_exc()
    print("  VERDICT: CRASH", type(exc).__name__)


print("\n=== 3. f returning a Python scalar / None / mixed pytree ===")
def f_pyscalar(x0):
    out, _ = jax.lax.scan(lambda c, x: (c + x, c), x0, xs)
    return out, 7, None   # python int + None alongside an array
try:
    ref = f_pyscalar(jnp.float32(0.0))
    ev = []
    got = tap.verbose(f_pyscalar, on_step=lambda e: ev.append(e))(jnp.float32(0.0))
    jax.block_until_ready(got)
    print("  ref:", ref)
    print("  got:", got)
    print("  arrays bitwise:", bitwise_eq(ref, got), " events:", len(ev))
    print("  structure match:", jax.tree_util.tree_structure(ref) == jax.tree_util.tree_structure(got))
except Exception as exc:
    import traceback; traceback.print_exc()
    print("  VERDICT: CRASH", type(exc).__name__)


print("\n=== 4. SHARPENED collision: a DIRECT inner scan next to a JIT-wrapped inner scan ===")
print("    (siblings in the same body -> both resolve to scan[0]/scan[0])")
def f_collide(x0):
    def outer(c, x):
        a, _ = jax.lax.scan(lambda cc, xx: (cc + xx, cc), c, xs)     # direct: scan[0]/scan[0]
        @jax.jit
        def wrapped(v):
            b, _ = jax.lax.scan(lambda cc, xx: (cc * 1.0, cc), v, xs)  # jit-hidden: scan[0]/scan[0] TOO
            return b
        return wrapped(a), a
    out, _ = jax.lax.scan(outer, x0, xs)
    return out
ev = []
got = tap.verbose(f_collide, on_step=lambda e: ev.append(e))(jnp.float32(0.0))
jax.block_until_ready(got)
c = Counter(e.path for e in ev)
print("   path -> event count:", dict(c))
# Outer runs 4 steps; each inner scan runs 4 steps -> 16 events per inner scan.
# If the two distinct inner scans collide, scan[0]/scan[0] holds 32 and there is
# NO scan[0]/scan[1].
inner0 = c.get("scan[0]/scan[0]", 0)
if "scan[0]/scan[1]" not in c and inner0 == 32:
    print(f"   VERDICT: COLLISION -- two DISTINCT inner scans (direct + jit-hidden) merged")
    print(f"            into scan[0]/scan[0] carrying {inner0} events; scan[0]/scan[1] MISSING")
else:
    print("   VERDICT: distinct:", dict(c))
