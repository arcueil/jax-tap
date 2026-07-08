"""Attack #2a: _scan_depth is thread-local but the registry + the monkeypatch
itself are process-global. Thread T1 holds a progress_bar() context; Thread
T2 (which never touches progress_bar) runs its own plain jitted scan
concurrently. Since T2's _scan_depth thread-local starts fresh at 0 on that
thread, _patched_scan sees depth==0 and active-nonempty -> T2's scan gets
swept into T1's ProgressState. Simulates e.g. a web-service thread pool
where one request thread wraps sampling in a progress bar while another
request thread (serving a totally different user) runs unrelated JAX code.
"""
import threading
import time

import jax
import jax.numpy as jnp

import blackjax

barrier = threading.Barrier(2)
t1_events = []
t2_ready = threading.Event()
t2_done = threading.Event()

def t1_worker():
    with blackjax.progress_bar(label="T1-user-A") as state:
        orig_cb = state._step_callback
        def cb(idx):
            t1_events.append((threading.get_ident(), state.n_steps, int(idx)))
            orig_cb(idx)
        state._step_callback = cb

        barrier.wait()  # sync with T2 so both scans overlap in wall time
        t2_ready.set()

        def body(carry, x):
            return carry + x, carry
        # Deliberately slow this down a touch relative to T2 so the two
        # scans' dispatches interleave.
        final, _ = jax.lax.scan(body, 0.0, jnp.arange(2000))
        jax.block_until_ready(final)
        t2_done.wait(timeout=5)

def t2_worker():
    barrier.wait()
    t2_ready.wait()

    def body(carry, x):
        return carry + x, carry
    # T2 has NO progress_bar context of its own.
    final, _ = jax.lax.scan(body, 0.0, jnp.arange(777))
    jax.block_until_ready(final)
    t2_done.set()

th1 = threading.Thread(target=t1_worker)
th2 = threading.Thread(target=t2_worker)
th1.start()
th2.start()
th1.join()
th2.join()

thread_ids = set(tid for tid, _, _ in t1_events)
n_steps_seen = sorted(set(n for _, n, _ in t1_events))
print("distinct thread-ids whose scan callbacks landed in T1's ProgressState:", len(thread_ids))
print("distinct scan lengths (n_steps) recorded in T1's bar:", n_steps_seen)
print("T2's 777-step scan leaked into T1's 'T1-user-A' bar:", 777 in n_steps_seen)
print("total callback fires observed by T1's state:", len(t1_events),
      "(2000 + 777 = 2777 expected if fully crosstalked)")
