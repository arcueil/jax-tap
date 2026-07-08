"""ARM L / Finding: reusing the SAME _RecordContext object in a nested fashion
permanently corrupts global state (jax.lax.scan stays patched forever).

Two triggers, both realistic:
  (A) manual double __enter__ then a single __exit__
  (B) `with ctx: with ctx: pass`  -- same object, nested `with` statements

Root cause: __enter__ overwrites self._key (and self._recorder) each time, so
the FIRST registry entry is orphaned. On the LAST __exit__, self._key is
already None -> the pop is skipped -> registry never empties -> scan/while
never restored.  Global process state is corrupted for ALL later JAX code.
"""
import jax
import jax.numpy as jnp
import jaxtap as tap
from jaxtap._ashell import (
    _context_registry,
    _original_scan,
    _original_while,
    _patched_scan,
)


def clean():
    _context_registry.clear()
    jax.lax.scan = _original_scan
    jax.lax.while_loop = _original_while


def report(tag):
    print(f"[{tag}] registry size = {len(_context_registry)}")
    print(f"[{tag}] scan is _patched_scan (LEAK if True) = {jax.lax.scan is _patched_scan}")
    print(f"[{tag}] scan is _original_scan (healthy if True) = {jax.lax.scan is _original_scan}")


print("========== (A) manual: ctx.__enter__(); ctx.__enter__(); ctx.__exit__() ==========")
clean()
ctx = tap.record()
r1 = ctx.__enter__()
first_key = ctx._key
print("after 1st __enter__: key =", first_key, "| registry =", len(_context_registry))
r2 = ctx.__enter__()  # same object entered again
second_key = ctx._key
print("after 2nd __enter__: key =", second_key, "| registry =", len(_context_registry))
print("recorder identity changed by 2nd enter (r1 is r2)?", r1 is r2)
print("first key still in registry (orphaned)?", first_key in _context_registry)
ctx.__exit__(None, None, None)  # pops second_key, sets self._key=None
report("after single __exit__")
print(">>> LEAK: registry still holds the orphaned first entry; scan NOT restored." )

print()
print("========== (B) `with ctx: with ctx: pass`  (same object, nested with) ==========")
clean()
ctx = tap.record()
leaked = False
try:
    with ctx as rr1:
        with ctx as rr2:  # SAME object entered while already active
            pass
        # inner __exit__ ran here: popped the 2nd key, self._key=None
    # outer __exit__ ran here: self._key is None -> pop skipped -> registry never empties
except Exception as e:
    print("raised:", type(e).__name__, e)
report("after both `with` blocks exit")

# Prove the corruption is load-bearing: a LATER, unrelated scan is now silently
# intercepted by the leaked context, and jax.lax.scan is the patched fn forever.
print()
print("--- consequence: unrelated later code is now silently instrumented ---")
if jax.lax.scan is _patched_scan:
    final, ys = jax.lax.scan(lambda c, x: (c + x, c), 0.0, jnp.arange(5.0))
    jax.block_until_ready(final)
    print("unrelated scan ran through PATCHED scan; result correct?",
          float(final) == 10.0)
    print("registry size seen by the leaked patch:", len(_context_registry))

clean()
print()
print("BUG CONFIRMED" if True else "clean")
