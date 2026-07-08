"""AYS (e): I did NOT actually run the reader-races-the-delete consequence
in round 1 -- I only reasoned about progress_reader.py's break-after-3-
misses logic from source. Actually running it now: a real polling loop
using the verbatim logic from progress_reader.py's main(), racing against
the two-context-shared-path scenario from attack #10.
"""
import os
import tempfile
import threading
import time

import jax
import jax.numpy as jnp

import blackjax
from blackjax.progress_reader import read_progress

tmpdir = tempfile.mkdtemp()
shared_path = os.path.join(tmpdir, "shared_progress.txt")

reader_concluded_finished_at = {"t": None}
reader_log = []

def reader_loop(interval=0.05, patience=3):
    """Verbatim logic transplanted from progress_reader.py's main(), just
    parameterized for a faster interval/lower patience so this test doesn't
    need to run for minutes."""
    bar_started = False
    last = -1
    missing_after_start = 0
    t0 = time.monotonic()
    while time.monotonic() - t0 < 3.0:
        result = read_progress(shared_path)
        if result is not None:
            step, total = result
            reader_log.append((time.monotonic() - t0, "seen", step, total))
            bar_started = True
            missing_after_start = 0
            last = step
        elif bar_started:
            missing_after_start += 1
            reader_log.append((time.monotonic() - t0, "missing", missing_after_start))
            if missing_after_start > patience:
                reader_concluded_finished_at["t"] = time.monotonic() - t0
                return
        time.sleep(interval)

ctx1_started = threading.Event()
ctx1_should_exit = threading.Event()
worker2_finished = threading.Event()

def worker1():
    with blackjax.progress_bar(label="A", output_file=shared_path, print_rate=1) as s:
        def body(c, x):
            return c + x, c
        final, _ = jax.lax.scan(body, 0.0, jnp.arange(5))
        jax.block_until_ready(final)
        ctx1_started.set()
        ctx1_should_exit.wait(timeout=5)
    # __exit__ deletes shared_path unconditionally here.

def worker2():
    ctx1_started.wait(timeout=5)
    with blackjax.progress_bar(label="B", output_file=shared_path, print_rate=1) as s:
        def body(c, x):
            return c + x, c
        # A LONG-RUNNING job (many slow steps) that is very much still
        # alive when worker1 deletes the shared file out from under it.
        for chunk in range(30):
            final, _ = jax.lax.scan(body, float(chunk), jnp.arange(20))
            jax.block_until_ready(final)
            if chunk == 2:
                ctx1_should_exit.set()  # let worker1 exit + delete now
            time.sleep(0.08)
    worker2_finished.set()

t_reader = threading.Thread(target=reader_loop, kwargs={"interval": 0.05, "patience": 3})
t1 = threading.Thread(target=worker1)
t2 = threading.Thread(target=worker2)
t_reader.start()
t1.start()
t2.start()
t1.join()
t2.join()
t_reader.join(timeout=5)

print("worker2 (the still-running, legitimate job) actually finished at "
      "wall time (relative):", "yes -- worker2_finished.is_set() =", worker2_finished.is_set())
print("reader concluded 'run finished' (patience exceeded) at t=",
      reader_concluded_finished_at["t"])
print()
print("reader event log (t, event, ...):")
for entry in reader_log:
    print("  ", entry)
print()
if reader_concluded_finished_at["t"] is not None:
    print("REPRODUCED: external reader gave up and exited BEFORE worker2's "
          "legitimate run actually completed, due solely to the shared-path "
          "collision with worker1's unrelated context's teardown.")
else:
    print("NOT reproduced at these timings/patience -- reader's window "
          "tolerated the transient gap (patience=3 polls survived it). "
          "State this outcome plainly rather than assuming the theoretic "
          "escalation holds.")
