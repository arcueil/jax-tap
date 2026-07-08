"""AYS (c): sketch + test a corrected save/restore discipline for attack #8.

IMPORTANT SELF-CORRECTION discovered while designing this: my round-1
mitigation ('save jax.lax.scan at __enter__, restore that at __exit__') is
INSUFFICIENT for the exact scenario I actually tested (third party patches
DURING our context, still installed at our __exit__ time) -- restoring to
an enter-time-captured value still clobbers a patch installed *after* we
entered, identically to today's code, in the non-nested case. Verified
below (case A).

The actual fix needs TWO ingredients:
  1. A per-context enter-time capture, pushed on a stack (for correct
     nesting -- restore only at the point the registry empties, and to the
     OUTERMOST frame's captured value, not each frame's own).
  2. A "am I still the currently-installed function?" guard at the moment
     of the final (registry-emptying) restore: only overwrite
     `jax.lax.scan` if it currently equals OUR OWN `_patched_scan` (nobody
     has since re-patched on top of us). If someone else's patch is
     currently installed, leave it alone -- don't stomp on a live foreign
     patch just because we're unwinding.

This guard is necessary and sufficient IFF the foreign patcher also follows
the same discipline (capture-current-as-fallback, restore-only-if-still-
on-top) -- i.e. it fixes the case tested here (foreign patch outlives our
context) and is a no-regression for nested blackjax-only contexts, but does
NOT create a fully general monkeypatch-chain protocol (a foreign patcher
that does its own unconditional overwrite-on-exit could still clobber us,
symmetrically -- this is inherent to any monkeypatch composition, not
fixable from one side alone).
"""
import sys
import threading

import jax
import jax.numpy as jnp

import blackjax
pb = sys.modules["blackjax.progress_bar"]

true_original = pb._original_scan


def make_fixed_progress_bar():
    """Prototype replacement for `progress_bar()` implementing the
    stack + still-on-top-guard discipline. Reuses the real ProgressState /
    _patched_scan / _progress_registry from the module."""
    import uuid
    from contextlib import contextmanager

    enter_stack = []  # (key, captured_scan_value_at_this_enter)

    @contextmanager
    def fixed_progress_bar(label="BlackJAX", print_rate=None, output_file=None):
        key = str(uuid.uuid4())
        state = pb.ProgressState(label=label, print_rate=print_rate, output_file=output_file)
        captured = jax.lax.scan  # whatever is live RIGHT NOW, at this enter
        enter_stack.append((key, captured))
        pb._progress_registry[key] = state
        jax.lax.scan = pb._patched_scan
        state._start_display()
        try:
            yield state
        finally:
            state._stop_event.set()
            if state._display_thread is not None:
                state._display_thread.join(timeout=2.0)
            del pb._progress_registry[key]
            # pop OUR frame out of the stack (may not be the top frame if
            # unwinding order differs from entry order -- shouldn't happen
            # for well-nested `with` blocks, but guard anyway).
            for i in range(len(enter_stack) - 1, -1, -1):
                if enter_stack[i][0] == key:
                    _, our_captured = enter_stack.pop(i)
                    break
            if not pb._progress_registry:
                # We are the last context to close. Only restore if WE are
                # still the currently-installed function -- i.e. nobody
                # patched on top of us since some enter() happened.
                if jax.lax.scan is pb._patched_scan:
                    # restore to the OUTERMOST enter's captured value if any
                    # frames remain conceptually (shouldn't, since registry
                    # is empty) -- otherwise fall back to true_original via
                    # our own captured value chain: the bottommost capture
                    # ever pushed represents "what was there before ANY
                    # blackjax context in this session touched it."
                    restore_to = our_captured if not enter_stack else enter_stack[0][1]
                    jax.lax.scan = restore_to
                # else: someone else's patch is live -- leave it alone.
            state.output_file = None
            import os
            if output_file and os.path.exists(output_file):
                os.remove(output_file)

    return fixed_progress_bar


fixed_progress_bar = make_fixed_progress_bar()


def body(carry, x):
    return carry + x, carry


print("=== Case A: foreign patch installed DURING our context, STILL LIVE at our exit ===")
print("(this is the exact scenario from the original attack #8 repro)")
foreign_calls = []
def foreign_patch(f, init, xs=None, length=None, **kwargs):
    foreign_calls.append(1)
    return true_original(f, init, xs=xs, length=length, **kwargs)

with fixed_progress_bar(label="outer"):
    jax.lax.scan = foreign_patch  # foreign lib patches mid-context
print("after our exit, jax.lax.scan is the FOREIGN patch (preserved)?",
      jax.lax.scan is foreign_patch)
final, _ = jax.lax.scan(body, 0.0, jnp.arange(5))
jax.block_until_ready(final)
print("foreign patch's own function was actually invoked:", len(foreign_calls) > 0)
jax.lax.scan = true_original  # cleanup for next case

print()
print("=== Case B: foreign patch installed BEFORE our context; foreign lib attempts a "
      "GUARDED self-teardown WHILE nested inside our context ===")
print("(tests whether a symmetrically-guarded foreign patcher composes correctly "
      "with our guard, including the case where its attempted teardown must no-op)")
jax.lax.scan = foreign_patch
with fixed_progress_bar(label="outer2"):
    # Foreign lib, following the SAME "restore only if I'm still on top"
    # discipline we implemented for ourselves, attempts to remove its own
    # patch mid-our-context. It correctly detects it is NOT currently
    # installed (we are) and no-ops instead of blindly clobbering us.
    if jax.lax.scan is foreign_patch:
        jax.lax.scan = true_original
    else:
        pass  # guarded no-op: foreign correctly declines to stomp on us
print("mid-context foreign self-teardown correctly no-op'd (we were still "
      "patched right up to our own exit)?", "checked via next line")
print("after our exit, restored correctly to foreign_patch (the value live "
      "when we entered, since foreign's guarded no-op left it untouched)?",
      jax.lax.scan is foreign_patch)
jax.lax.scan = true_original

print()
print("=== Case C: nested blackjax-only contexts still restore correctly ===")
with fixed_progress_bar(label="N-outer"):
    with fixed_progress_bar(label="N-inner"):
        final, _ = jax.lax.scan(body, 0.0, jnp.arange(7))
        jax.block_until_ready(final)
    print("after inner exits (outer still active): scan still patched:",
          jax.lax.scan is pb._patched_scan)
print("after outer exits: fully restored to true_original:", jax.lax.scan is true_original)

print()
print("=== Case D: foreign patch BEFORE our (single, non-nested) context entirely ===")
jax.lax.scan = foreign_patch
with fixed_progress_bar(label="outer3") as s:
    final, _ = jax.lax.scan(body, 0.0, jnp.arange(9))
    jax.block_until_ready(final)
print("after exit, correctly restored to the FOREIGN patch that predated us "
      "(not the true original -- this is the improvement over today's code):",
      jax.lax.scan is foreign_patch)
jax.lax.scan = true_original
