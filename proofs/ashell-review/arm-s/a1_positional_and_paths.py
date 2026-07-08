"""ARM-S battery 1: positional-arg forwarding + multi-top-level path divergence."""
import traceback
import numpy as np
import jax
import jax.numpy as jnp
import jaxtap as tap

def bw(a, b):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return len(la) == len(lb) and all(
        np.asarray(x).tobytes() == np.asarray(y).tobytes() for x, y in zip(la, lb))

print("=" * 70)
print("ATTACK 1: scan called with `reverse` as a POSITIONAL arg")
print("=" * 70)
xs = jnp.arange(5.0, dtype=jnp.float32)
def body(c, x): return c + x, c

# Valid JAX: scan(f, init, xs, length, reverse) -- reverse positional
ref = jax.lax.scan(body, jnp.float32(0.0), xs, None, True)  # reverse=True positionally
print("OUTSIDE context: scan(body, init, xs, None, True) ->", ref[0], "OK")

try:
    with tap.record() as rec:
        got = jax.lax.scan(body, jnp.float32(0.0), xs, None, True)
    print("INSIDE context: got =", got[0])
    print("bitwise:", bw(ref, got), " events:", len(rec.events))
except Exception as e:
    print("INSIDE context RAISED:", type(e).__name__, "->", str(e)[:200])
    print("   (outside works, inside raises => divergence)")

print()
print("=" * 70)
print("ATTACK 1b: scan called with `unroll` positional too")
print("=" * 70)
ref2 = jax.lax.scan(body, jnp.float32(0.0), xs, None, False, 2)  # unroll=2 positionally
print("OUTSIDE: OK ->", ref2[0])
try:
    with tap.record() as rec:
        got2 = jax.lax.scan(body, jnp.float32(0.0), xs, None, False, 2)
    print("INSIDE: bitwise:", bw(ref2, got2), " events:", len(rec.events))
except Exception as e:
    print("INSIDE RAISED:", type(e).__name__, "->", str(e)[:200])

print()
print("=" * 70)
print("ATTACK 2: two SEQUENTIAL top-level scans -- path divergence vs verbose()")
print("=" * 70)

def two_scans(x0):
    a, _ = jax.lax.scan(lambda c, x: (c + x, c), x0, xs)
    b, _ = jax.lax.scan(lambda c, x: (c * x, c), a, xs)
    return b

# verbose() reference: ONE trace, shared counter
vb_events = []
r_vb = tap.verbose(two_scans, on_step=vb_events.append)(jnp.float32(1.0))
jax.block_until_ready(r_vb)
vb_paths = sorted(set(e.path for e in vb_events))
print("verbose() paths:", vb_paths)

# A-shell
with tap.record() as rec:
    r_ash = two_scans(jnp.float32(1.0))
jax.block_until_ready(r_ash)
ash_paths = sorted(set(e.path for e in rec.events))
print("A-shell  paths:", ash_paths)
print("bitwise results match:", bw(r_vb, r_ash))
print("PATHS EQUIVALENT:", vb_paths == ash_paths,
      "  <-- contract requires same paths" )
# count per path
from collections import Counter
print("verbose count/path:", dict(Counter(e.path for e in vb_events)))
print("A-shell count/path:", dict(Counter(e.path for e in rec.events)))

print()
print("=" * 70)
print("ATTACK 3: top-level scan THEN while -- shared counter divergence")
print("=" * 70)
def scan_then_while(x0):
    a, _ = jax.lax.scan(lambda c, x: (c + x, c), x0, xs)
    b = jax.lax.while_loop(lambda c: c < 50.0, lambda c: c + 1.0, a)
    return b

vb2 = []
r_vb2 = tap.verbose(scan_then_while, on_step=vb2.append)(jnp.float32(1.0))
jax.block_until_ready(r_vb2)
print("verbose() paths:", sorted(set(e.path for e in vb2)))
with tap.record() as rec2:
    r_ash2 = scan_then_while(jnp.float32(1.0))
jax.block_until_ready(r_ash2)
print("A-shell  paths:", sorted(set(e.path for e in rec2.events)))
print("bitwise:", bw(r_vb2, r_ash2))
