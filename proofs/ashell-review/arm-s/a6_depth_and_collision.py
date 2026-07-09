"""ARM-S battery 6: depth-counter escapes, multi-call addressing, value integrity
under path collision, and 'trace-time helper calls scan' constructions."""
from collections import Counter
import numpy as np
import jax
import jax.numpy as jnp
import jaxtap as tap

def bw(a, b):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return len(la) == len(lb) and all(
        np.asarray(x).tobytes() == np.asarray(y).tobytes() for x, y in zip(la, lb))

xs = jnp.arange(5.0, dtype=jnp.float32)

print("=== A: scan inside a PYTHON loop (5 separate top-level calls) ===")
def py_loop(x0):
    outs = []
    for i in range(5):
        r, _ = jax.lax.scan(lambda c, x: (c + x + i, c), x0, xs)
        outs.append(r)
    return jnp.stack(outs)

vb = []
r_vb = tap.verbose(py_loop, on_step=vb.append)(jnp.float32(0.0))
jax.block_until_ready(r_vb)
with tap.record() as rec:
    r_ash = py_loop(jnp.float32(0.0))
jax.block_until_ready(r_ash)
print("verbose paths:", dict(Counter(e.path for e in vb)))
print("ashell  paths:", dict(Counter(e.path for e in rec.events)))
print("total events verbose=%d ashell=%d  bitwise=%s" %
      (len(vb), len(rec.events), bw(r_vb, r_ash)))
print("--> 5 distinct scans; verbose addresses scan[0..4], ashell collapses to scan[0]")

print()
print("=== B: are ALL 5 python-loop calls actually instrumented? (no drop) ===")
# each call fires 5 steps; if all 5 calls instrumented -> 25 events total
print("ashell total events:", len(rec.events), "(expected 25 if all 5 calls instrumented)")

print()
print("=== C: VALUE integrity under collision -- two scans w/ different bodies ===")
def two_diff(x0):
    a, _ = jax.lax.scan(lambda c, x: (c + 100.0, c), x0, xs)   # +100 each
    b, _ = jax.lax.scan(lambda c, x: (c * 2.0, c), x0, xs)     # *2 each
    return a, b

with tap.record() as rec_c:
    two_diff(jnp.float32(1.0))
# both scans land on scan[0]; steps 0..4 appear TWICE. Can a consumer tell them apart?
by_step = {}
for e in rec_c.events:
    by_step.setdefault(e.step, []).append(float(np.asarray(e.value[0])))
print("scan[0] events grouped by step (each step has TWO different scans' values):")
for s in sorted(by_step):
    print(f"   step {s}: values={by_step[s]}")
print("--> same (path, step) key maps to TWO different scans -> telemetry ambiguous")

print()
print("=== D: trace-time helper that itself calls lax.scan (nested via closure) ===")
def helper(v):
    # called during tracing of the outer scan body; issues its OWN scan
    return jax.lax.scan(lambda c, _: (c * 1.1, c), v, None, length=3)[0]

def outer_with_helper(x0):
    def body(c, x):
        return helper(c + x), c
    return jax.lax.scan(body, x0, xs)[0]

vb_d = []
r_vb_d = tap.verbose(outer_with_helper, on_step=vb_d.append)(jnp.float32(0.5))
jax.block_until_ready(r_vb_d)
with tap.record() as rec_d:
    r_ash_d = outer_with_helper(jnp.float32(0.5))
jax.block_until_ready(r_ash_d)
print("verbose paths:", dict(Counter(e.path for e in vb_d)))
print("ashell  paths:", dict(Counter(e.path for e in rec_d.events)))
print("bitwise:", bw(r_vb_d, r_ash_d),
      " event-count match:", len(vb_d) == len(rec_d.events))

print()
print("=== E: two INDEPENDENT top-level scans, second built from first's output ===")
def chained(x0):
    a, _ = jax.lax.scan(lambda c, x: (c + x, c), x0, xs)
    b, _ = jax.lax.scan(lambda c, x: (c - x, c), a, xs)
    return b
with tap.record() as rec_e:
    chained(jnp.float32(0.0))
print("ashell paths (both should be instrumented, collide at scan[0]):",
      dict(Counter(e.path for e in rec_e.events)))
