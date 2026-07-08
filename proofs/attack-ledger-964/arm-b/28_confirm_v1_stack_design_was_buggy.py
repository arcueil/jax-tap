"""Direct confirmation: run the ROUND-1 stack-based design (from
19_patch_stack_fix.py, per-frame pop + fallback to own capture) against the
EXACT non-LIFO scenario from 27, to show it actually produces the wrong
restore value -- not just a predicted failure."""
import sys
import threading
from contextlib import contextmanager

import jax
import jax.numpy as jnp

import blackjax
pb = sys.modules["blackjax.progress_bar"]

true_original = pb._original_scan


def make_v1_buggy():
    import os
    import uuid

    enter_stack = []

    @contextmanager
    def fixed_progress_bar_v1(label="BlackJAX", print_rate=None, output_file=None):
        key = str(uuid.uuid4())
        state = pb.ProgressState(label=label, print_rate=print_rate, output_file=output_file)
        captured = jax.lax.scan
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
            for i in range(len(enter_stack) - 1, -1, -1):
                if enter_stack[i][0] == key:
                    _, our_captured = enter_stack.pop(i)
                    break
            if not pb._progress_registry:
                if jax.lax.scan is pb._patched_scan:
                    restore_to = our_captured if not enter_stack else enter_stack[0][1]
                    jax.lax.scan = restore_to
            state.output_file = None
            if output_file and os.path.exists(output_file):
                os.remove(output_file)

    return fixed_progress_bar_v1


fixed_progress_bar_v1 = make_v1_buggy()

cm_a = fixed_progress_bar_v1(label="A")
cm_b = fixed_progress_bar_v1(label="B")
cm_a.__enter__()
cm_b.__enter__()
cm_a.__exit__(None, None, None)  # A exits first; B remains active
cm_b.__exit__(None, None, None)  # B exits last (registry now empty)

print("round-1 (v1) stack design: final jax.lax.scan after both exit:")
print("  is true_original?", jax.lax.scan is true_original)
print("  is _patched_scan (itself -- the bug)?", jax.lax.scan is pb._patched_scan)
if jax.lax.scan is pb._patched_scan:
    print("CONFIRMED BUG: v1's non-LIFO restore left jax.lax.scan pointing "
          "at its OWN patched function -- scan is now permanently "
          "instrumented with an empty registry (silent, no active context, "
          "yet every future scan still routes through _patched_scan's dead "
          "'active=[] -> fall through' branch -- functionally harmless per-call "
          "since it always falls through to _original_scan when registry is "
          "empty, but jax.lax.scan is no longer IDENTICAL to the true "
          "original, which breaks any code doing an `is` check, e.g. "
          "test_restoration's `assertIs(jax.lax.scan, _original_scan)`).")
# restore global state for cleanliness
jax.lax.scan = true_original
