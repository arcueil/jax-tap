"""Attack #7: two progress_bar() contexts (different threads / interleaved
lifetimes) sharing the SAME output_file path. The first context's __exit__
deletes the file (`os.remove`) unconditionally on exit -- even if a second,
still-active context is concurrently writing to and depending on that exact
same path. This simulates a careless default (e.g. both call sites in a
codebase use the same hardcoded /tmp/bjx_progress.txt)."""
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

ctx2_saw_missing_file = threading.Event()
ctx1_started = threading.Event()
ctx1_should_exit = threading.Event()

def worker1():
    with blackjax.progress_bar(label="A", output_file=shared_path, print_rate=1) as s:
        def body(c, x):
            return c + x, c
        # Write a handful of steps then idle so worker2 can overlap.
        final, _ = jax.lax.scan(body, 0.0, jnp.arange(5))
        jax.block_until_ready(final)
        ctx1_started.set()
        ctx1_should_exit.wait(timeout=5)
    # __exit__ has now run os.remove(shared_path) unconditionally.

def worker2():
    ctx1_started.wait(timeout=5)
    with blackjax.progress_bar(label="B", output_file=shared_path, print_rate=1) as s:
        def body(c, x):
            return c + x, c
        final, _ = jax.lax.scan(body, 0.0, jnp.arange(200))
        jax.block_until_ready(final)
        # Let worker1 exit (and delete the shared file) WHILE we are still
        # inside our own (still-active) context using the same path.
        ctx1_should_exit.set()
        time.sleep(0.5)
        exists_after_other_exit = os.path.exists(shared_path)
        print("shared_path still exists after OTHER context's __exit__ ran:",
              exists_after_other_exit)
        if not exists_after_other_exit:
            ctx2_saw_missing_file.set()
        # Now run more scan steps -- does worker2's own state resurrect the
        # file, or is it permanently gone until worker2's own __exit__?
        final2, _ = jax.lax.scan(body, final, jnp.arange(5))
        jax.block_until_ready(final2)
        print("file exists after worker2 ran more steps post-deletion:",
              os.path.exists(shared_path))

t1 = threading.Thread(target=worker1)
t2 = threading.Thread(target=worker2)
t1.start()
t2.start()
t1.join()
t2.join()

print("worker2 (still holding its OWN active context) observed the file "
      "vanish out from under it:", ctx2_saw_missing_file.is_set())
