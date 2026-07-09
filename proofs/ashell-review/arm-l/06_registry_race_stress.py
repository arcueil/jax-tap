"""ARM L / Stress: many threads concurrently enter/exit contexts (each running a
scan inside) in a tight loop.  Hunt for:
  - RuntimeError from the LOCKLESS list(_context_registry.values()) read in
    _patched_scan racing dict mutation under the lock.
  - leaked registry entries at the end.
  - jax.lax.scan not restored to original at the end (torn transition).
  - wrong results (bitwise).
"""
import threading

import jax
import jax.numpy as jnp
import jaxtap as tap
from jaxtap._ashell import _context_registry, _original_scan, _original_while


def clean():
    _context_registry.clear()
    jax.lax.scan = _original_scan
    jax.lax.while_loop = _original_while


clean()

N_THREADS = 8
N_ITERS = 40
xs = jnp.arange(4.0, dtype=jnp.float32)
ref = jax.lax.scan(lambda c, x: (c + x, c), 0.0, xs)
jax.block_until_ready(ref)

errors = []
bad_results = [0]
max_registry_seen = [0]
lock = threading.Lock()
stop_probe = threading.Event()


def probe_registry():
    # Continuously read the registry lock-free, mimicking the hot path read.
    while not stop_probe.is_set():
        try:
            snap = list(_context_registry.values())
            with lock:
                if len(snap) > max_registry_seen[0]:
                    max_registry_seen[0] = len(snap)
        except Exception as e:  # noqa: BLE001
            errors.append(("probe", repr(e)))


def worker(tid):
    try:
        for _ in range(N_ITERS):
            with tap.record() as rec:
                r = jax.lax.scan(lambda c, x: (c + x, c), jnp.float32(0.0), xs)
                jax.block_until_ready(r)
                if float(r[0]) != float(ref[0]):
                    with lock:
                        bad_results[0] += 1
    except Exception as e:  # noqa: BLE001
        errors.append((tid, repr(e)))


probe = threading.Thread(target=probe_registry)
probe.start()
threads = [threading.Thread(target=worker, args=(i,)) for i in range(N_THREADS)]
for t in threads:
    t.start()
for t in threads:
    t.join(timeout=120)
stop_probe.set()
probe.join(timeout=5)

print("threads:", N_THREADS, "iters each:", N_ITERS)
print("max concurrent registry size observed:", max_registry_seen[0])
print("errors:", errors[:10] or "none")
print("bad (non-bitwise) results:", bad_results[0])
print("final registry size (should be 0):", len(_context_registry))
print("scan restored to original?", jax.lax.scan is _original_scan)
print("while restored to original?", jax.lax.while_loop is _original_while)
print()
leaked = len(_context_registry) != 0 or jax.lax.scan is not _original_scan
print(">>> LEAK/TORN STATE" if (leaked or errors or bad_results[0]) else ">>> CLEAN under stress")
clean()
