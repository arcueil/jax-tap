"""ARM L / two probes:

(A) Generator crossing a context boundary: a `with tap.record()` that SUSPENDS
    at a yield keeps the patch installed while suspended. Scans run by the caller
    during suspension are captured (single-ctx delegate). Closing/dropping the
    generator DOES self-heal (GeneratorExit unwinds the with) -- contrast with the
    direct-manual-__enter__ leak (repro 02), which does NOT self-heal.

(B) Nested-context async misrouting attempt: try to make an OUTER scan's debug
    callback fire while an INNER context is active, so _dynamic_router's
    innermost-wins routing misattributes outer's events to inner. (On CPU,
    dispatch is synchronous so this is expected to be non-reproducible ->
    THEORETIC for async GPU/TPU backends.)
"""
import gc

import jax
import jax.numpy as jnp
import jaxtap as tap
from jaxtap._ashell import _context_registry, _original_scan, _original_while, _patched_scan


def clean():
    _context_registry.clear()
    jax.lax.scan = _original_scan
    jax.lax.while_loop = _original_while


xs = jnp.arange(5.0, dtype=jnp.float32)

print("=== (A) generator suspended inside `with tap.record()` ===")
clean()


def gen():
    with tap.record() as rec:
        yield rec
        yield "done"


g = gen()
rec = next(g)  # enters context; SUSPENDS inside the with
print("patch active while generator suspended?", jax.lax.scan is _patched_scan)
print("registry size while suspended:", len(_context_registry))
# caller runs a scan while the context is suspended -> captured by delegate rule
r = jax.lax.scan(lambda c, x: (c + x, c), jnp.float32(0.0), xs)
jax.block_until_ready(r)
print("caller's scan captured while generator suspended? events =", len(rec.events))

# Drop the generator without resuming -> GeneratorExit unwinds the with -> __exit__.
del g
gc.collect()
print("after dropping generator + gc: patch restored (self-heal)?",
      jax.lax.scan is _original_scan)
print("registry size after gc:", len(_context_registry))

print()
print("=== (B) nested async misrouting attempt (jit, no block until inner active) ===")
clean()
jax.clear_caches()


def scan_fn(x0):
    return jax.lax.scan(lambda c, x: (c * 1.01 + jnp.sin(x), c), x0, xs)


fj = jax.jit(scan_fn)

with tap.record() as outer:
    r_out = fj(jnp.float32(0.5))          # dispatch under outer; do NOT block
    with tap.record() as inner:
        # if outer's callbacks were deferred to here, they'd misroute to inner
        jax.block_until_ready(r_out)      # force outer's callbacks to fire NOW
        r_in = fj(jnp.float32(0.7))       # cache-hit under inner
        jax.block_until_ready(r_in)

print("outer recorder events:", len(outer.events), "(want 5 from outer's fj)")
print("inner recorder events:", len(inner.events), "(want 5 from inner's fj)")
if len(outer.events) < 5:
    print(">>> MISROUTING: some of outer's events leaked to inner")
elif len(outer.events) == 5 and len(inner.events) == 5:
    print(">>> no misrouting on CPU (synchronous dispatch); THEORETIC for async backends")
else:
    print(">>> unexpected:", len(outer.events), len(inner.events))
clean()
