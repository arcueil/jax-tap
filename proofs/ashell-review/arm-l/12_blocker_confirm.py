"""ARM L / Tight re-confirmation of the top BLOCKER for interrogation.

Reusing a single _RecordContext object in nested fashion (`with ctx: with ctx:`)
permanently leaves BOTH jax.lax.scan AND jax.lax.while_loop patched, with a
leaked registry entry and NO recovery path.

Minimal, self-contained, asserts hard.
"""
import jax
import jaxtap as tap
from jaxtap._ashell import (
    _context_registry,
    _original_scan,
    _original_while,
    _patched_scan,
    _patched_while,
)

# start from a known-clean state
_context_registry.clear()
jax.lax.scan = _original_scan
jax.lax.while_loop = _original_while
assert jax.lax.scan is _original_scan
assert jax.lax.while_loop is _original_while
assert len(_context_registry) == 0

ctx = tap.record()  # ONE context object, reused nested
with ctx:
    with ctx:
        pass

# After BOTH with-blocks exit, healthy code expects full restoration.
scan_leaked = jax.lax.scan is _patched_scan
while_leaked = jax.lax.while_loop is _patched_while
registry_leaked = len(_context_registry)

print("scan   leaked (still _patched_scan)? ", scan_leaked)
print("while  leaked (still _patched_while)?", while_leaked)
print("registry entries leaked:            ", registry_leaked)

# Heal for any subsequent process use, then report.
_context_registry.clear()
jax.lax.scan = _original_scan
jax.lax.while_loop = _original_while

assert scan_leaked and while_leaked and registry_leaked == 1, "BLOCKER did not reproduce"
print("\nBLOCKER REPRODUCED: nested reuse of one context object leaks BOTH "
      "primitives + registry, no self-heal.")
