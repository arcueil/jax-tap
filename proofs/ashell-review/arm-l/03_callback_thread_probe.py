"""ARM L / Probe: which THREAD does the baked host callback (_dynamic_router)
run on?

_dynamic_router calls _select_ctx(active), whose >=2-context branch keys on
threading.get_ident().  That is only correct if the callback runs on the
context's OWNER thread.  If jax.debug.callback dispatches on a runtime/callback
thread, the >=2-context attribution is computed against the WRONG ident.

We capture the ident seen inside on_step and compare to the owner thread ident.
Tested for: eager scan, jitted scan, jitted+blocked, jitted async (no block).
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


MAIN = threading.get_ident()
print("main/owner thread ident:", MAIN)

xs = jnp.arange(5.0, dtype=jnp.float32)


def scan_fn(x0):
    return jax.lax.scan(lambda c, x: (c * 1.01 + jnp.sin(x), c), x0, xs)


for label, fn, block in [
    ("EAGER scan", scan_fn, True),
    ("JITTED scan (blocked)", jax.jit(scan_fn), True),
    ("JITTED scan (async, no block until after exit)", jax.jit(scan_fn), False),
]:
    clean()
    jax.clear_caches()
    seen = []

    def on_step(e, _seen=seen):
        _seen.append(threading.get_ident())

    with tap.record(on_step=on_step) as rec:
        r = fn(jnp.float32(0.5))
        if block:
            jax.block_until_ready(r)
    if not block:
        jax.block_until_ready(r)

    idents = set(seen)
    print(f"\n[{label}] on_step fired {len(seen)} times")
    print(f"  distinct callback-thread idents: {idents}")
    print(f"  callback ran on OWNER thread? {idents == {MAIN} if idents else 'no events'}")

clean()
