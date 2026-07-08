"""AYS ROUND 2 on the M1b fix — attack the dynamic router itself.

1. Sequential contexts w/ different on_step: cache-hit in ctx B routes to B's
   recorder AND B's on_step (not A's).
2. Worker-thread delegation on a CACHE-HIT artifact (traced in an earlier ctx
   on main thread; called from a worker thread inside a single active ctx —
   owner-affinity: 1 active => any thread attributed).
3. Nested contexts + cache hit: inner-wins routing.
4. No-context cache-hit: dropped (re-confirm within this process).
"""
import threading
import numpy as np
import jax
import jax.numpy as jnp
import jaxtap as tap

FAILS = []
def check(name, ok, d=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {d}")
    if not ok: FAILS.append(name)

xs = jnp.arange(5.0, dtype=jnp.float32)
def f(x0):
    def body(c, x):
        return c * 1.01 + jnp.sin(x), c
    c, _ = jax.lax.scan(body, x0, xs)
    return c

fj = jax.jit(f)

# --- prime the cache inside ctx A (with its own on_step) ---
live_a, live_b = [], []
with tap.record(on_step=live_a.append) as rec_a:
    jax.block_until_ready(fj(jnp.float32(0.5)))
n_a = len(rec_a.events)
check("ctx A collected + live on_step", n_a == 5 and len(live_a) == 5,
      f"(rec={n_a}, live={len(live_a)})")

# --- 1: cache-hit in ctx B routes to B (recorder AND on_step), not A ---
with tap.record(on_step=live_b.append) as rec_b:
    jax.block_until_ready(fj(jnp.float32(0.5)))   # cache hit
check("cache-hit routes to B recorder", len(rec_b.events) == 5, f"({len(rec_b.events)})")
check("cache-hit routes to B on_step", len(live_b) == 5, f"({len(live_b)})")
check("A saw nothing new", len(rec_a.events) == n_a and len(live_a) == 5,
      f"(rec_a={len(rec_a.events)})")

# --- 2: worker-thread delegation on cache-hit artifact ---
res = {}
def worker():
    res["out"] = fj(jnp.float32(0.5))
    jax.block_until_ready(res["out"])
with tap.record() as rec_w:
    t = threading.Thread(target=worker)
    t.start(); t.join()
check("worker-thread delegation routes (1 active ctx)", len(rec_w.events) == 5,
      f"({len(rec_w.events)})")

# --- 3: nested contexts + cache hit -> inner wins ---
with tap.record() as rec_outer:
    with tap.record() as rec_inner:
        jax.block_until_ready(fj(jnp.float32(0.5)))
    n_inner_only = len(rec_inner.events)
    jax.block_until_ready(fj(jnp.float32(0.5)))    # now only outer active
check("nested: inner won during inner scope", n_inner_only == 5,
      f"(inner={n_inner_only})")
check("nested: outer got the post-inner call", len(rec_outer.events) == 5,
      f"(outer={len(rec_outer.events)})")

# --- 4: no context -> dropped ---
before = (len(rec_a.events), len(rec_b.events), len(rec_w.events), len(rec_outer.events))
jax.block_until_ready(fj(jnp.float32(0.5)))
after = (len(rec_a.events), len(rec_b.events), len(rec_w.events), len(rec_outer.events))
check("no-ctx cache-hit dropped everywhere", before == after, f"({before} -> {after})")

print("\n" + ("M1B AYS R2: ALL PASS" if not FAILS else f"FAILURES: {FAILS}"))
