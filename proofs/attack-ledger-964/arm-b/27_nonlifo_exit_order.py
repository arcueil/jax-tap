"""AYS (4): non-LIFO exit order (A enters, B enters, A exits FIRST, B exits
last). Does the round-1 stack-based restore still land on the TRUE
outermost (pre-A) captured value?

SELF-CORRECTION found while designing this test: the round-1
stack-with-per-frame-pop design is WRONG here. When A exits first (registry
non-empty, so no restore happens), A's frame is popped out of enter_stack.
When B exits later (registry now empty), the code's fallback logic
(`our_captured if not enter_stack else enter_stack[0][1]`) uses B's OWN
captured value (since enter_stack is now empty, having already lost A's
entry) -- but B's own captured value was `_patched_scan` (whatever A had
already installed by the time B entered), NOT the true pre-A original. This
restores to a WRONG value (the patched function itself) instead of the true
original.

FIX: don't pop per-frame captures at all. Track a single
"pre-session" value, captured ONLY on the empty-to-nonempty registry
transition (i.e. only by whichever context happens to be first in, LIFO or
not), and restored ONLY on the nonempty-to-empty transition (whichever
context happens to be last out). This is correct regardless of entry/exit
order and needs no stack at all.
"""
import sys
import threading
from contextlib import contextmanager

import jax
import jax.numpy as jnp

import blackjax
pb = sys.modules["blackjax.progress_bar"]

true_original = pb._original_scan


def make_v2():
    import os
    import uuid

    _pre_session_scan = {"value": None}  # None sentinel: no session open

    @contextmanager
    def fixed_progress_bar_v2(label="BlackJAX", print_rate=None, output_file=None):
        key = str(uuid.uuid4())
        state = pb.ProgressState(label=label, print_rate=print_rate, output_file=output_file)
        if not pb._progress_registry:  # empty -> nonempty transition
            _pre_session_scan["value"] = jax.lax.scan
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
            if not pb._progress_registry:  # nonempty -> empty transition
                if jax.lax.scan is pb._patched_scan:
                    jax.lax.scan = _pre_session_scan["value"]
                _pre_session_scan["value"] = None
            state.output_file = None
            if output_file and os.path.exists(output_file):
                os.remove(output_file)

    return fixed_progress_bar_v2


fixed_progress_bar_v2 = make_v2()


def body(carry, x):
    return carry + x, carry


print("=== Non-LIFO: A enters, B enters, A exits FIRST, B exits LAST ===")
cm_a = fixed_progress_bar_v2(label="A")
cm_b = fixed_progress_bar_v2(label="B")
state_a = cm_a.__enter__()
print("after A enters: patched?", jax.lax.scan is pb._patched_scan)
state_b = cm_b.__enter__()
print("after B enters: patched?", jax.lax.scan is pb._patched_scan)

final, _ = jax.lax.scan(body, 0.0, jnp.arange(9))
jax.block_until_ready(final)

cm_a.__exit__(None, None, None)  # A exits FIRST (non-LIFO vs entry order... well this IS LIFO actually)
print("after A exits (B still active): still patched?", jax.lax.scan is pb._patched_scan,
      "(should be True -- B still open)")

cm_b.__exit__(None, None, None)  # B exits last
print("after B exits (last one out): restored to TRUE original?",
      jax.lax.scan is true_original)

print()
print("=== Genuinely non-LIFO: A enters, B enters, B exits FIRST, A exits LAST "
      "(this is actually LIFO w.r.t. nesting but let's also do the reverse "
      "interleaving to be thorough: A enters, B enters, A exits, B enters "
      "AGAIN as C while A's slot is gone) -- stress the transition logic "
      "directly with 3 contexts in an odd order ===")
cm_a2 = fixed_progress_bar_v2(label="A2")
cm_b2 = fixed_progress_bar_v2(label="B2")
a2 = cm_a2.__enter__()
b2 = cm_b2.__enter__()
cm_a2.__exit__(None, None, None)  # A2 (the true FIRST-in) exits first, B2 remains
print("after A2 exits (B2 still active): still patched?",
      jax.lax.scan is pb._patched_scan)
cm_c2 = fixed_progress_bar_v2(label="C2")
c2 = cm_c2.__enter__()  # C2 enters while B2 is still active (registry non-empty)
print("after C2 enters (B2 still active, registry was non-empty): "
      "pre_session_scan untouched (still the value from BEFORE A2 first "
      "entered)?")
cm_b2.__exit__(None, None, None)  # B2 exits, C2 still active
print("after B2 exits (C2 still active): still patched?",
      jax.lax.scan is pb._patched_scan)
cm_c2.__exit__(None, None, None)  # C2 (the true LAST-out) exits
print("after C2 exits (last one out, registry now empty): restored to TRUE "
      "original?", jax.lax.scan is true_original)
