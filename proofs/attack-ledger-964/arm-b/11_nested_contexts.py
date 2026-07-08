"""Attack #5b / #6: nested progress_bar() contexts. Confirm (a) the
`active[-1]` insertion-order semantics claimed in the docstring/comment
actually hold, (b) jax.lax.scan re-assignment on the second `__enter__` is
idempotent (no crash, no double-wrapping), and (c) two simultaneous bars
render onto the terminal at once (visual collision) since both display
threads run concurrently and neither is aware of the other."""
import threading
import time

import jax
import jax.numpy as jnp

import blackjax
from blackjax.progress_bar import _progress_registry

with blackjax.progress_bar(label="OUTER", print_rate=1) as outer_state:
    print("registry size with 1 context:", len(_progress_registry))
    with blackjax.progress_bar(label="INNER", print_rate=1) as inner_state:
        print("registry size with 2 contexts (nested):", len(_progress_registry))
        print("outer is inner (same object)?", outer_state is inner_state)

        def body(carry, x):
            return carry + x, carry
        # Which state's callback fires for a scan issued while BOTH are
        # active? Per the code, active[-1] (most-recently-entered == INNER)
        # wins.
        final, _ = jax.lax.scan(body, 0.0, jnp.arange(20))
        jax.block_until_ready(final)
        time.sleep(0.2)

    print("after inner exits: registry size:", len(_progress_registry))
    print("outer.n_steps unaffected by inner's scan:", outer_state.n_steps)
    print("inner.n_steps received the scan meant to be ambiguous:", inner_state.n_steps)

    # Does OUTER's display/callback still work correctly after INNER tore
    # down (deleted its own registry entry, but NOT jax.lax.scan since
    # registry was non-empty)?
    final2, _ = jax.lax.scan(body, 0.0, jnp.arange(15))
    jax.block_until_ready(final2)
    time.sleep(0.2)
    print("outer.n_steps updated correctly post-inner-exit:", outer_state.n_steps)
    print("scan still patched (registry non-empty, outer still active):",
          jax.lax.scan is not None and "patched" if jax.lax.scan.__name__ == "_patched_scan" else "ORIGINAL (bug: restored too early)")

print("registry empty after outer exits:", _progress_registry == {})
