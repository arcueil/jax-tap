"""Attack #2c: two threads, EACH opening its OWN progress_bar() context
(simulating two independent users/requests, each correctly using the
context manager as documented). Because `active[-1]` is a single global
LIFO slot with zero thread/request affinity, thread A's own scan can get
routed to thread B's ProgressState -- even though thread A's own context is
still perfectly valid and present in the registry. This is a stronger claim
than plain cross-talk (attack 05): it shows CORRECT usage by BOTH parties
still produces cross-contamination purely due to the registry's global
last-entered-wins ordering, with no owner/thread-affinity concept at all.
"""
import threading
import time

import jax
import jax.numpy as jnp

import blackjax

a_ready = threading.Event()
b_entered = threading.Event()
a_scan_done = threading.Event()
results = {}

def thread_a():
    with blackjax.progress_bar(label="A-OWN-BAR", print_rate=1) as state_a:
        results["state_a"] = state_a
        a_ready.set()
        # Wait until B has ALSO entered its own (separate) context before A
        # issues A's own scan -- A did everything right: opened its own
        # context, is calling jax.lax.scan from within its own `with` block.
        b_entered.wait(timeout=5)

        def body(carry, x):
            return carry + x, carry
        final, _ = jax.lax.scan(body, 0.0, jnp.arange(123))
        jax.block_until_ready(final)
        a_scan_done.set()
        time.sleep(0.3)

def thread_b():
    a_ready.wait(timeout=5)
    with blackjax.progress_bar(label="B-OWN-BAR", print_rate=1) as state_b:
        results["state_b"] = state_b
        b_entered.set()
        a_scan_done.wait(timeout=5)

th_a = threading.Thread(target=thread_a)
th_b = threading.Thread(target=thread_b)
th_a.start()
th_b.start()
th_a.join()
th_b.join()

state_a = results["state_a"]
state_b = results["state_b"]
print("state_a.n_steps (should be 123 if A's own scan went to A's own bar):", state_a.n_steps)
print("state_b.n_steps (nonzero here means A's scan was misrouted to B's bar):", state_b.n_steps)
print("A's 123-step scan was misrouted to B's context despite A using its OWN "
      "`with` block correctly:", state_b.n_steps == 123 and state_a.n_steps == 0)
