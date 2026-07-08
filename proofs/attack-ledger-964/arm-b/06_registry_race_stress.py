"""Attack #2b: registry dict mutation race. Many threads rapidly open/close
progress_bar() contexts (mutating the global `_progress_registry` dict via
insert/delete) while another thread continuously calls the patched
jax.lax.scan (which does `list(_progress_registry.values())` on every
call). Look for `RuntimeError: dictionary changed size during iteration`
or any other corruption/crash over many iterations.
"""
import threading
import time
import traceback

import jax
import jax.numpy as jnp

import blackjax

errors = []
stop = threading.Event()

def churner(idx):
    while not stop.is_set():
        try:
            with blackjax.progress_bar(label=f"churn-{idx}"):
                pass
        except Exception:
            errors.append(("churner", idx, traceback.format_exc()))
            return

def scanner():
    def body(carry, x):
        return carry + x, carry
    n = 0
    while not stop.is_set():
        try:
            final, _ = jax.lax.scan(body, 0.0, jnp.arange(10))
            jax.block_until_ready(final)
            n += 1
        except Exception:
            errors.append(("scanner", n, traceback.format_exc()))
            return
    print("scanner completed", n, "scans")

threads = [threading.Thread(target=churner, args=(i,)) for i in range(8)]
threads.append(threading.Thread(target=scanner))
for t in threads:
    t.start()
time.sleep(5.0)
stop.set()
for t in threads:
    t.join(timeout=5)

print("total errors:", len(errors))
for who, idx, tb in errors[:5]:
    print("----", who, idx)
    print(tb)

from blackjax.progress_bar import _progress_registry, _original_scan
print("final registry state (should be empty):", _progress_registry)
print("scan restored to original:", jax.lax.scan is _original_scan)
