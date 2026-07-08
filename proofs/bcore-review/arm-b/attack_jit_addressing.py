"""ATTACK: the jit branch recurses with `_p=path` UNCHANGED and _interp resets
`n_cf = 0` at entry. So a scan inside a jit gets the same path as a sibling
scan at the enclosing level -> DUPLICATE, non-unique addresses. The module
docstring claims "addresses are stable [and] unique". This breaks uniqueness
across a jit boundary.
"""
import jax
import jax.numpy as jnp
import jaxtap as tap


# A top-level scan, then a jit containing another scan. Both should be
# distinguishable, but both land on "scan[0]".
def f(x0, xs):
    # top-level scan -> should be scan[0]
    a, _ = jax.lax.scan(lambda c, x: (c + x, c), x0, xs)

    # scan inside a (non-inlined) jit -> gets path "" + "scan[0]" == collision
    @jax.jit
    def inner(c):
        b, _ = jax.lax.scan(lambda cc, x: (cc * 1.0 + x, cc), c, xs)
        return b

    return a + inner(x0)


x0 = jnp.float32(1.0)
xs = jnp.arange(4.0, dtype=jnp.float32)

events = []
got = tap.verbose(f, on_step=lambda e: events.append(e))(x0, xs)
jax.block_until_ready(got)

from collections import Counter
paths = Counter(e.path for e in events)
print("=== jit-boundary addressing ===")
print("distinct paths and event counts:")
for p, n in sorted(paths.items()):
    print(f"   {p!r}: {n} events")
print()
print("Total events:", len(events))
distinct = set(paths)
print("Distinct path strings:", sorted(distinct))
# Two DIFFERENT scans (top-level + in-jit). If addressing were unique there
# would be two distinct path strings. If they collide, only one string.
if len(distinct) == 1:
    print("VERDICT: ADDRESS COLLISION -- two distinct scans share one path,",
          list(distinct)[0], "-> cannot distinguish which scan a tap came from")
else:
    print("VERDICT: distinct addresses:", sorted(distinct))
