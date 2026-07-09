"""ARM L / Router-vs-verbose interaction: call an EXPLICITLY tapped function
(tap.verbose / tap.record B-form) INSIDE a `with tap.record()` context.

The user's verbose() runs interpret(), which does make_jaxpr(f). While tracing f,
f's `jax.lax.scan` resolves to the PATCHED scan; _depth is 0 (the user called
verbose directly, not via _patched_scan) -> the CONTEXT intercepts it too.

Question: does the same scan get double-instrumented?  Do the context recorder
AND the user's callback both receive events?  Are counts/results right?
"""
import jax
import jax.numpy as jnp
import jaxtap as tap
from jaxtap._ashell import _context_registry, _original_scan, _original_while


def clean():
    _context_registry.clear()
    jax.lax.scan = _original_scan
    jax.lax.while_loop = _original_while


clean()

N = 5
x0 = jnp.float32(1.0)
xs = jnp.arange(float(N), dtype=jnp.float32)


def f(a, b):
    return jax.lax.scan(lambda c, x: (c + x, c * x), a, b)


ref = f(x0, xs)
jax.block_until_ready(ref)

# Baseline: verbose OUTSIDE any context.
base_events = []
tap.verbose(f, on_step=lambda e: base_events.append(e))(x0, xs)
jax.block_until_ready(None)
print("baseline verbose() outside context: user events =", len(base_events))

# Now: verbose INSIDE a context.
clean()
user_events = []
with tap.record() as rec_ctx:
    g = tap.verbose(f, on_step=lambda e: user_events.append(e))
    got = g(x0, xs)
    jax.block_until_ready(got)

print("\n--- explicit verbose() called INSIDE `with tap.record()` ---")
print("user callback events:", len(user_events), "(baseline was", len(base_events), ")")
print("context recorder events:", len(rec_ctx.events))
print("context recorder paths:", sorted({e.path for e in rec_ctx.events}))
print("result bitwise-correct?",
      [bytes(__import__('numpy').asarray(v)) for v in jax.tree_util.tree_leaves(got)]
      == [bytes(__import__('numpy').asarray(v)) for v in jax.tree_util.tree_leaves(ref)])
print()
if len(user_events) != len(base_events):
    print(f">>> user event count changed inside context: {len(base_events)} -> {len(user_events)}")
if len(rec_ctx.events) > 0:
    print(f">>> DOUBLE INSTRUMENTATION: context recorder ALSO captured "
          f"{len(rec_ctx.events)} events from the user's explicitly-tapped call")
else:
    print(">>> context recorder saw 0 events (no double instrumentation)")
clean()
