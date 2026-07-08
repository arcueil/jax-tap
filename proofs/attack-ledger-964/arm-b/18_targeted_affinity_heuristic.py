"""AYS (b) resolution: a narrower heuristic than blanket thread-affinity.
Key insight from 16+17: attack #3 (bystander thread, ZERO context of its
own, exactly ONE context globally active) is *information-theoretically
indistinguishable* from the legitimate 'main thread's own delegated worker'
pattern (also exactly one context active) -- neither raw Thread nor
ThreadPoolExecutor propagate any identity token, so `_patched_scan` cannot
tell these apart without an explicit opt-in API change. NOT claiming to fix
#3 here.

But attack #4 (TWO contexts simultaneously active, each on its own owner
thread) IS distinguishable: when len(active) >= 2, record each state's
*owner* thread (the thread that called __enter__) and prefer the state
whose owner matches the CURRENT thread over blind `active[-1]`. This only
engages when ambiguity exists (>=2 candidates) so it cannot regress the
single-context delegation patterns (which always have len(active)==1).

Prototype implemented by monkeypatching a local copy of `_patched_scan`'s
selection logic (NOT editing the repo) and swapping it in at runtime for
this test process only.
"""
import sys
import threading

import jax
import jax.numpy as jnp

import blackjax
import blackjax.progress_bar  # noqa: F401 -- populate sys.modules
# NOTE: `blackjax/__init__.py` does `from .progress_bar import progress_bar`,
# which shadows the *submodule* name with the *function* on the `blackjax`
# package object -- so `import blackjax.progress_bar as pb` would silently
# bind `pb` to the function (dotted-import-as resolves via attribute access
# post-import), not the module. Pull the real submodule out of sys.modules
# instead, which is unaffected by that shadowing.
pb = sys.modules["blackjax.progress_bar"]

# --- capture owner thread on ProgressState (prototype-only monkeypatch) ---
_orig_init = pb.ProgressState.__init__
def patched_init(self, *a, **kw):
    _orig_init(self, *a, **kw)
    self.owner_thread = threading.get_ident()
pb.ProgressState.__init__ = patched_init

def targeted_patched_scan(f, init, xs=None, length=None, **kwargs):
    depth = getattr(pb._scan_depth, "value", 0)
    pb._scan_depth.value = depth + 1
    try:
        active = list(pb._progress_registry.values())
        if depth == 0 and active:
            if len(active) >= 2:
                here = threading.get_ident()
                owned = [s for s in active if getattr(s, "owner_thread", None) == here]
                state = owned[-1] if owned else active[-1]
            else:
                state = active[-1]
            return pb._inject_progress(f, init, xs, length, state, **kwargs)
        else:
            return pb._original_scan(f, init, xs=xs, length=length, **kwargs)
    finally:
        pb._scan_depth.value = depth

pb._patched_scan = targeted_patched_scan


def body(carry, x):
    return carry + x, carry


print("=== Regression check: Pattern 1 (single ctx, raw-Thread delegation) still works ===")
with blackjax.progress_bar(label="p1") as s:
    res = {}
    def w():
        f, _ = jax.lax.scan(body, 0.0, jnp.arange(321))
        jax.block_until_ready(f)
    t = threading.Thread(target=w)
    t.start(); t.join()
print("n_steps == 321 (delegation preserved):", s.n_steps == 321)

print()
print("=== Fix check: attack #14 pattern, two threads each owning their own ctx ===")
a_ready = threading.Event()
b_entered = threading.Event()
a_done = threading.Event()
results = {}

def thread_a():
    with blackjax.progress_bar(label="A") as sa:
        results["a"] = sa
        a_ready.set()
        b_entered.wait(timeout=5)
        f, _ = jax.lax.scan(body, 0.0, jnp.arange(123))
        jax.block_until_ready(f)
        a_done.set()

def thread_b():
    a_ready.wait(timeout=5)
    with blackjax.progress_bar(label="B") as sb:
        results["b"] = sb
        b_entered.set()
        a_done.wait(timeout=5)

ta = threading.Thread(target=thread_a)
tb = threading.Thread(target=thread_b)
ta.start(); tb.start()
ta.join(); tb.join()

print("state_a.n_steps (should be 123 now, WAS 0 under the old code):", results["a"].n_steps)
print("state_b.n_steps (should be 0 now, WAS 123 -- the misattribution -- under old code):",
      results["b"].n_steps)
print("FIXED:", results["a"].n_steps == 123 and results["b"].n_steps == 0)

print()
print("=== Disclosed residual gap: 2 contexts active AND one of them delegates to a sub-worker ===")
a_ready.clear(); b_entered.clear(); a_done.clear()
results2 = {}

def thread_a2():
    with blackjax.progress_bar(label="A2") as sa:
        results2["a"] = sa
        a_ready.set()
        b_entered.wait(timeout=5)
        # A delegates its OWN work to a sub-worker thread (legitimate
        # pattern per Pattern 1) WHILE B's context is also active.
        def sub_worker():
            f, _ = jax.lax.scan(body, 0.0, jnp.arange(55))
            jax.block_until_ready(f)
        subt = threading.Thread(target=sub_worker)
        subt.start(); subt.join()
        a_done.set()

def thread_b2():
    a_ready.wait(timeout=5)
    with blackjax.progress_bar(label="B2") as sb:
        results2["b"] = sb
        b_entered.set()
        a_done.wait(timeout=5)

ta2 = threading.Thread(target=thread_a2)
tb2 = threading.Thread(target=thread_b2)
ta2.start(); tb2.start()
ta2.join(); tb2.join()
print("A2's sub-worker's 55-step scan landed on A2's own bar:", results2["a"].n_steps == 55)
print("A2's sub-worker's scan misattributed to B2 instead (residual gap):",
      results2["b"].n_steps == 55)
