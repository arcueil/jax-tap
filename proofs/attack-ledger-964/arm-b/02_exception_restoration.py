"""Attack #4 (restoration integrity): exception raised mid-trace inside the
`with` body -- confirm the patch is restored and the display thread joined
via the `finally` block."""
import threading

import jax
import jax.numpy as jnp

import blackjax
from blackjax.progress_bar import _original_scan, _progress_registry

pre_threads = threading.active_count()

try:
    with blackjax.progress_bar(label="boom") as state:
        def body(carry, x):
            if x == 3:
                raise RuntimeError("simulated failure mid-trace")
            return carry + x, carry

        # NOTE: the exception fires during *tracing* (Python-level), since
        # lax.scan traces body with a tracer x, so `x == 3` on a tracer needs
        # care. Instead raise inside a callback fired after a real dispatch,
        # or simply raise before calling scan at all inside the `with`.
        raise RuntimeError("simulated failure mid-trace (pre-scan)")
except RuntimeError as e:
    print("caught:", e)

print("scan restored:", jax.lax.scan is _original_scan)
print("registry empty:", _progress_registry == {})
print("thread count back to baseline:", threading.active_count(), "vs pre:", pre_threads)

# Second scenario: exception raised *during* a real scan's tracing (Python
# exception from within f, raised eagerly at trace time since scan traces f
# once with abstract values -- a Python-level raise not depending on tracer
# values will fire).
try:
    with blackjax.progress_bar(label="boom2") as state:
        def body2(carry, x):
            raise ValueError("trace-time failure")

        jax.lax.scan(body2, 0.0, jnp.arange(5))
except ValueError as e:
    print("caught2:", e)

print("scan restored after trace-time failure:", jax.lax.scan is _original_scan)
print("registry empty after trace-time failure:", _progress_registry == {})
print("final thread count:", threading.active_count())
