"""AYS (e) follow-up: rather than chasing the exact reader-threshold
crossing (timing-sensitive, borderline in test 22), measure the ACTUAL
wall-clock duration the shared file spends missing, at high poll
resolution (1ms), to give a clean, threshold-independent number. Compare
against progress_reader.py's REAL defaults (interval=0.2s, patience>3 i.e.
needs > 0.6s of continuous absence to conclude 'finished')."""
import os
import tempfile
import threading
import time

import jax
import jax.numpy as jnp

import blackjax

tmpdir = tempfile.mkdtemp()
shared_path = os.path.join(tmpdir, "shared_progress.txt")

gap_log = []  # (t, exists)
stop_measuring = threading.Event()

def measure():
    t0 = time.monotonic()
    while not stop_measuring.is_set():
        gap_log.append((time.monotonic() - t0, os.path.exists(shared_path)))
        time.sleep(0.001)

ctx1_started = threading.Event()
ctx1_should_exit = threading.Event()

def worker1():
    with blackjax.progress_bar(label="A", output_file=shared_path, print_rate=1) as s:
        def body(c, x):
            return c + x, c
        final, _ = jax.lax.scan(body, 0.0, jnp.arange(5))
        jax.block_until_ready(final)
        ctx1_started.set()
        ctx1_should_exit.wait(timeout=5)

def worker2():
    ctx1_started.wait(timeout=5)
    with blackjax.progress_bar(label="B", output_file=shared_path, print_rate=1) as s:
        def body(c, x):
            return c + x, c
        for chunk in range(15):
            final, _ = jax.lax.scan(body, float(chunk), jnp.arange(20))
            jax.block_until_ready(final)
            if chunk == 2:
                ctx1_should_exit.set()
            time.sleep(0.1)

tm = threading.Thread(target=measure)
t1 = threading.Thread(target=worker1)
t2 = threading.Thread(target=worker2)
tm.start()
t1.start()
t2.start()
t1.join()
t2.join()
stop_measuring.set()
tm.join()

# find the longest continuous run of `exists == False` AFTER the file first
# appeared (i.e. after worker1's first write).
first_seen = next((i for i, (_, e) in enumerate(gap_log) if e), None)
longest_gap = 0.0
cur_gap_start = None
for t, exists in gap_log[first_seen:]:
    if not exists:
        if cur_gap_start is None:
            cur_gap_start = t
    else:
        if cur_gap_start is not None:
            longest_gap = max(longest_gap, t - cur_gap_start)
            cur_gap_start = None
if cur_gap_start is not None:
    longest_gap = max(longest_gap, gap_log[-1][0] - cur_gap_start)

print("longest continuous file-missing gap measured (seconds):", longest_gap)
print("progress_reader.py's REAL default: interval=0.2s, breaks after "
      ">3 consecutive misses -> needs a gap EXCEEDING 0.2*4=0.8s to trip")
print("verdict:", "gap EXCEEDS the real-default trip threshold -> reproducible with default settings"
      if longest_gap > 0.8 else
      "gap is SHORTER than the real-default trip threshold -> NOT reliably reproducible "
      "with progress_reader.py's actual default interval/patience; only a risk under "
      "faster custom polling intervals")
