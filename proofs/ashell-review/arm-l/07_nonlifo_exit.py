"""ARM L / Verification: non-LIFO exit order restores correctly and scan stays
usable BETWEEN the two exits.

  order 1: a.enter, b.enter, a.exit (first), b.exit (last)
  order 2: a.enter, b.enter, b.exit (first), a.exit (last)

Between the first and second exit, jax.lax.scan must still be patched (one ctx
still active) and must run correctly.  After the last exit, restored to original.
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


xs = jnp.arange(4.0, dtype=jnp.float32)
ref = jax.lax.scan(lambda c, x: (c + x, c), 0.0, xs)
jax.block_until_ready(ref)


def run_order(name, exit_first):
    clean()
    a = tap.record()
    b = tap.record()
    ra = a.__enter__()
    rb = b.__enter__()
    print(f"\n[{name}] after both enter: patched? {jax.lax.scan is _patched_scan}, "
          f"registry={len(_context_registry)}")

    first, second = (a, b) if exit_first == "a" else (b, a)
    first.__exit__(None, None, None)
    print(f"[{name}] after {exit_first} exits: patched? {jax.lax.scan is _patched_scan} "
          f"(expect True), registry={len(_context_registry)}")

    # scan must still work between exits
    r_mid = jax.lax.scan(lambda c, x: (c + x, c), jnp.float32(0.0), xs)
    jax.block_until_ready(r_mid)
    ok_mid = float(r_mid[0]) == float(ref[0])
    print(f"[{name}] scan between exits bitwise-correct? {ok_mid}")

    second.__exit__(None, None, None)
    restored = jax.lax.scan is _original_scan and jax.lax.while_loop is _original_while
    print(f"[{name}] after last exit: restored to original? {restored}, "
          f"registry={len(_context_registry)}")
    return restored and ok_mid


ok1 = run_order("order1 a-first", "a")
ok2 = run_order("order2 b-first", "b")
clean()
print()
print(">>> CLEAN" if (ok1 and ok2) else ">>> BUG in non-LIFO restore")
