"""ARM L / Finding: with EXACTLY ONE active context, the "delegate" rule
attributes ANY thread's scan to that context -- including a genuinely unrelated
background thread the user never intended to trace.  Result: the user's recorder
is POLLUTED with telemetry from foreign code (wrong paths/steps interleaved).

This is the dark side of the single-context delegate rule (_select_ctx: len==1
-> active[0] regardless of caller).  Distinct from the TL's "worker delegation
works" probe: here the worker is a BYSTANDER, and the pollution corrupts the
event stream the user is collecting for their OWN scan.
"""
import threading

import jax
import jax.numpy as jnp
import jaxtap as tap
from jaxtap._ashell import _context_registry, _original_scan, _original_while


def clean():
    _context_registry.clear()
    jax.lax.scan = _original_scan
    jax.lax.while_loop = _original_while


clean()

# The user's own scan: length 5.
xs_user = jnp.arange(5.0, dtype=jnp.float32)

# An UNRELATED background thread's scan: length 100 (imagine a data pipeline,
# a different library's worker, an async prefetcher -- never asked to be traced).
xs_bystander = jnp.arange(100.0, dtype=jnp.float32)

bystander_done = threading.Event()
start = threading.Event()


def bystander():
    start.wait()
    r = jax.lax.scan(lambda c, x: (c + x, c), 0.0, xs_bystander)
    jax.block_until_ready(r)
    bystander_done.set()


t = threading.Thread(target=bystander)
t.start()

with tap.record() as rec:
    start.set()                       # release the bystander while ONE ctx active
    r_user = jax.lax.scan(lambda c, x: (c + x, c), jnp.float32(0.0), xs_user)
    jax.block_until_ready(r_user)
    bystander_done.wait(timeout=30)   # ensure bystander's scan runs inside ctx

paths = {}
for e in rec.events:
    paths[e.path] = paths.get(e.path, 0) + 1

print("user's own scan length:", len(xs_user))
print("bystander (unrelated) scan length:", len(xs_bystander))
print("total events captured by the user's recorder:", len(rec.events))
print("event counts by path:", paths)
print()
pollution = len(rec.events) - len(xs_user)
if pollution > 0:
    print(f">>> POLLUTION: {pollution} events from UNRELATED background-thread code")
    print(">>> landed in the user's recorder (single-context delegate mis-attribution).")
else:
    print("no pollution observed")

clean()
