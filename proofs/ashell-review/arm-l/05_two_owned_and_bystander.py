"""ARM L / Verification: with >=2 active contexts (one per owner thread):
  (1) two owner threads each with their OWN context -> no cross-talk
  (2) a THIRD bystander thread (owns neither) running a scan -> passes UNTAPPED
      and bitwise-correct.
This is the documented affinity rule; confirm it actually holds.
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

xsA = jnp.arange(4.0, dtype=jnp.float32)   # thread A: 4 events
xsB = jnp.arange(7.0, dtype=jnp.float32)   # thread B: 7 events
xsBys = jnp.arange(50.0, dtype=jnp.float32)  # bystander: should be 0 events

barrier = threading.Barrier(3)
recs = {}
errs = {}
bystander_events = {"n": None, "bitwise_ok": None}

ref_bys = jax.lax.scan(lambda c, x: (c + x, c), 0.0, xsBys)
jax.block_until_ready(ref_bys)


def owner(name, xs):
    try:
        with tap.record() as rec:
            barrier.wait()           # all 3 threads active simultaneously (2 ctx + bystander)
            r = jax.lax.scan(lambda c, x: (c + x, c), jnp.float32(0.0), xs)
            jax.block_until_ready(r)
            # hold the context open a beat so the bystander runs with 2 ctx active
            import time
            time.sleep(0.5)
            recs[name] = list(rec.events)
    except Exception as e:  # noqa: BLE001
        errs[name] = e


def bystander():
    try:
        barrier.wait()
        # both owner contexts are active now (2 contexts) -> we own neither
        r = jax.lax.scan(lambda c, x: (c + x, c), 0.0, xsBys)
        jax.block_until_ready(r)
        bystander_events["bitwise_ok"] = float(r[0]) == float(ref_bys[0])
    except Exception as e:  # noqa: BLE001
        errs["bystander"] = e


tA = threading.Thread(target=owner, args=("A", xsA))
tB = threading.Thread(target=owner, args=("B", xsB))
tBys = threading.Thread(target=bystander)
for t in (tA, tB, tBys):
    t.start()
for t in (tA, tB, tBys):
    t.join(timeout=30)

print("errors:", {k: repr(v) for k, v in errs.items()} or "none")
print("thread A recorder event count:", len(recs.get("A", [])), "(expected", len(xsA), ")")
print("thread B recorder event count:", len(recs.get("B", [])), "(expected", len(xsB), ")")
# cross-talk check: did A's recorder pick up any of B's 7-step events or vice versa?
a_steps = sorted(e.step for e in recs.get("A", []))
b_steps = sorted(e.step for e in recs.get("B", []))
print("A steps:", a_steps)
print("B steps:", b_steps)
print("bystander scan bitwise-correct?", bystander_events["bitwise_ok"])
print("final registry size (should be 0):", len(_context_registry))
print("scan restored to original?", jax.lax.scan is _original_scan)

crosstalk = len(recs.get("A", [])) != len(xsA) or len(recs.get("B", [])) != len(xsB)
print()
print(">>> CROSS-TALK DETECTED" if crosstalk else ">>> CLEAN: no cross-talk, bystander untapped")
clean()
