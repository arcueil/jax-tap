"""ARM L / Finding: manual __enter__ without __exit__ (notebook reality) leaks
the patch FOREVER, and there is NO GC self-heal.

The old #964 form was a generator @contextmanager: dropping the generator ran
its `finally` via GeneratorExit -> self-heal.  The new _RecordContext is a plain
class with (a) NO __del__ finalizer and (b) the registry holds a STRONG ref to
self -> the object is never even eligible for collection.  Double whammy.

Consequence: a later, completely unrelated scan -- from code that never heard of
jaxtap -- is silently intercepted and its telemetry is appended to the leaked
recorder (mis-attribution of a bystander to a dead-but-not-collected context).
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


clean()
print("=== manual __enter__, never __exit__, then drop the ref + gc.collect() ===")


def notebook_cell_N():
    cm = tap.record(label="leaked") if False else tap.record()
    rec = cm.__enter__()  # forgot `with`; __exit__ never called
    return rec  # cm itself falls out of scope here


rec = notebook_cell_N()  # `cm` local is now unreferenced by user code
print("scan patched after __enter__ (no exit)? ", jax.lax.scan is _patched_scan)
print("registry size:", len(_context_registry))

gc.collect()
print("--- after gc.collect() (user dropped every handle to the context obj) ---")
print("scan STILL patched? ", jax.lax.scan is _patched_scan)
print("registry size (leaked):", len(_context_registry))
print("registry holds a strong ref to the ctx, so it can NEVER be collected.")

# Prove the bystander mis-attribution: unrelated later code, single active
# (leaked) context => delegate rule attributes it to the dead context's recorder.
print()
print("--- unrelated later scan is silently instrumented into the leaked recorder ---")
n_before = len(rec.events)
final, _ = jax.lax.scan(lambda c, x: (c + x, c), 0.0, jnp.arange(7.0))
jax.block_until_ready(final)
n_after = len(rec.events)
print(f"leaked recorder grew from {n_before} to {n_after} events from UNRELATED code")
print("result still correct?", float(final) == 21.0)

clean()
print()
print("BUG CONFIRMED: no GC self-heal; leaked patch mis-attributes bystander scans")
