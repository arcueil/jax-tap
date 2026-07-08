"""Attack #5: patcher composition. Another library monkeypatches
jax.lax.scan AFTER our context enters; when our context exits (registry
empties), we restore `_original_scan` -- the TRUE original captured at
blackjax IMPORT time -- silently destroying the other library's patch,
which was never restored.
"""
import jax
import jax.numpy as jnp

import blackjax
from blackjax.progress_bar import _original_scan

true_original = jax.lax.scan
assert _original_scan is true_original, "sanity: blackjax captured the true original at import"

other_lib_patch_calls = []
def other_library_patched_scan(f, init, xs=None, length=None, **kwargs):
    other_lib_patch_calls.append(1)
    return true_original(f, init, xs=xs, length=length, **kwargs)

with blackjax.progress_bar(label="outer"):
    # A different tool patches jax.lax.scan WHILE our context is active
    # (e.g. a debugging/tracing tool, another instrumentation library).
    jax.lax.scan = other_library_patched_scan
    print("other library's patch installed while blackjax context active:",
          jax.lax.scan is other_library_patched_scan)
    # blackjax's own context does NOT know about this and doesn't re-wrap it
    # -- it already swapped jax.lax.scan to _patched_scan on __enter__ and
    # never looks at it again until __exit__.

print("after blackjax context exits, jax.lax.scan is:",
      "TRUE ORIGINAL (other library's patch silently destroyed)"
      if jax.lax.scan is true_original
      else "other library's patch (survived)" if jax.lax.scan is other_library_patched_scan
      else "blackjax's patched_scan (still active, bug)")

# Confirm the other library's patch is gone for good -- a scan call now
# bypasses it entirely, invisibly to that library.
def body(carry, x):
    return carry + x, carry
final, _ = jax.lax.scan(body, 0.0, jnp.arange(5))
jax.block_until_ready(final)
print("other library's patched_scan called after blackjax exit:",
      len(other_lib_patch_calls) > 0, "(calls recorded:", other_lib_patch_calls, ")")
