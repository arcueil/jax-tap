"""AYS (2): complete decision table for a REFINED heuristic. Round-1's
heuristic (owned = states whose owner_thread matches; else active[-1]) was
tested only for owner-vs-owner. Here: does a genuine bystander T3 (owns
NOTHING) fall through to _original_scan (no bar, safe) or get misattributed
via the `active[-1]` fallback?

REFINEMENT under test: when len(active) >= 2, if the calling thread owns
NONE of the active contexts, fall through to `_original_scan` (no bar)
INSTEAD of guessing `active[-1]`. This trades "some multi-context delegation
patterns lose their bar" for "eliminates misattribution entirely once
ambiguity exists" -- disclosed as an intentional, documented mode
discontinuity vs the n_active==1 case (where ANY thread, owner or not,
gets attributed -- that's Pattern 1's delegation support).
"""
import sys
import threading

import jax
import jax.numpy as jnp

import blackjax
pb = sys.modules["blackjax.progress_bar"]

_orig_init = pb.ProgressState.__init__
def patched_init(self, *a, **kw):
    _orig_init(self, *a, **kw)
    self.owner_thread = threading.get_ident()
pb.ProgressState.__init__ = patched_init

def refined_patched_scan(f, init, xs=None, length=None, **kwargs):
    depth = getattr(pb._scan_depth, "value", 0)
    pb._scan_depth.value = depth + 1
    try:
        active = list(pb._progress_registry.values())
        if depth == 0 and active:
            if len(active) >= 2:
                here = threading.get_ident()
                owned = [s for s in active if getattr(s, "owner_thread", None) == here]
                if not owned:
                    # Ambiguous AND unattributable -- do not guess.
                    return pb._original_scan(f, init, xs=xs, length=length, **kwargs)
                state = owned[-1]
            else:
                state = active[-1]
            return pb._inject_progress(f, init, xs, length, state, **kwargs)
        else:
            return pb._original_scan(f, init, xs=xs, length=length, **kwargs)
    finally:
        pb._scan_depth.value = depth

pb._patched_scan = refined_patched_scan


def body(carry, x):
    return carry + x, carry


def run_scan_get_n(n_steps):
    f, _ = jax.lax.scan(body, 0.0, jnp.arange(n_steps))
    jax.block_until_ready(f)


print("=" * 70)
print("CELL 1: n_active=1, caller = the owner")
print("=" * 70)
with blackjax.progress_bar(label="c1") as s:
    run_scan_get_n(11)
print("s.n_steps == 11 (correct, trivial):", s.n_steps == 11)

print()
print("=" * 70)
print("CELL 2: n_active=1, caller = foreign thread (delegate OR bystander -- "
      "indistinguishable, both get the single active state)")
print("=" * 70)
with blackjax.progress_bar(label="c2") as s:
    t = threading.Thread(target=lambda: run_scan_get_n(22))
    t.start(); t.join()
print("s.n_steps == 22 (Pattern-1 delegation preserved):", s.n_steps == 22)

print()
print("=" * 70)
print("CELL 3: n_active=2, caller = owner of S1")
print("CELL 4: n_active=2, caller = owner of S2")
print("=" * 70)
ready1 = threading.Event(); ready2 = threading.Event(); done1 = threading.Event()
res = {}
def th1():
    with blackjax.progress_bar(label="S1") as s1:
        res["s1"] = s1
        ready1.set()
        ready2.wait(timeout=5)
        run_scan_get_n(31)
        done1.set()
def th2():
    ready1.wait(timeout=5)
    with blackjax.progress_bar(label="S2") as s2:
        res["s2"] = s2
        ready2.set()
        done1.wait(timeout=5)
        run_scan_get_n(32)
a = threading.Thread(target=th1); b = threading.Thread(target=th2)
a.start(); b.start(); a.join(); b.join()
print("S1 owner's scan landed on S1 (n_steps==31):", res["s1"].n_steps == 31)
print("S2 owner's scan landed on S2 (n_steps==32):", res["s2"].n_steps == 32)

print()
print("=" * 70)
print("CELL 5: n_active=2, caller = T3, a TRUE BYSTANDER owning NEITHER "
      "context -- must fall through to _original_scan (no bar), not "
      "get misattributed to active[-1]")
print("=" * 70)
ready1.clear(); ready2.clear(); done1.clear()
res2 = {}
t3_ran_uninstrumented = {"ok": False}
def th1b():
    with blackjax.progress_bar(label="S1b") as s1:
        res2["s1"] = s1
        ready1.set()
        ready2.wait(timeout=5)
        # T3: genuinely unrelated thread, owns no context at all, started
        # from HERE only for test plumbing -- it is not S1's delegate, it
        # doesn't touch s1 or report to it.
        t3 = threading.Thread(target=lambda: run_scan_get_n(41))
        t3.start(); t3.join()
        t3_ran_uninstrumented["ok"] = True
        done1.set()
def th2b():
    ready1.wait(timeout=5)
    with blackjax.progress_bar(label="S2b") as s2:
        res2["s2"] = s2
        ready2.set()
        done1.wait(timeout=5)
a2 = threading.Thread(target=th1b); b2 = threading.Thread(target=th2b)
a2.start(); b2.start(); a2.join(); b2.join()
print("T3's 41-step scan did NOT land on S1 (n_steps stays whatever S1 had "
      "before, i.e. 0):", res2["s1"].n_steps)
print("T3's 41-step scan did NOT land on S2 (n_steps stays 0):", res2["s2"].n_steps)
print("=> T3 correctly fell through to _original_scan, no bar, no "
      "misattribution:", res2["s1"].n_steps == 0 and res2["s2"].n_steps == 0)

print()
print("=" * 70)
print("CELL 6: n_active=2, caller = S1's OWN delegate sub-worker (legitimate "
      "Pattern-1 delegation attempted WHILE S2 is also active) -- "
      "documented mode discontinuity: this now ALSO gets no bar, same "
      "mechanism as cell 5, NOT misattributed to S2 either")
print("=" * 70)
ready1.clear(); ready2.clear(); done1.clear()
res3 = {}
def th1c():
    with blackjax.progress_bar(label="S1c") as s1:
        res3["s1"] = s1
        ready1.set()
        ready2.wait(timeout=5)
        sub = threading.Thread(target=lambda: run_scan_get_n(55))
        sub.start(); sub.join()
        done1.set()
def th2c():
    ready1.wait(timeout=5)
    with blackjax.progress_bar(label="S2c") as s2:
        res3["s2"] = s2
        ready2.set()
        done1.wait(timeout=5)
a3 = threading.Thread(target=th1c); b3 = threading.Thread(target=th2c)
a3.start(); b3.start(); a3.join(); b3.join()
print("S1's own delegate's 55-step scan: S1.n_steps =", res3["s1"].n_steps,
      "(0 = no bar, NOT 55)")
print("S1's own delegate's 55-step scan: S2.n_steps =", res3["s2"].n_steps,
      "(0 = confirms NOT misattributed to S2 either)")
