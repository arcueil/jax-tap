"""AYS (1): harden _step_callback so it is TOTAL (never raises) under any
input JAX can deliver, and confirm this actually prevents the JaxRuntimeError
crash from #14. Sweep for every raise path found:
  - OSError family from the file write/replace (the original #14 finding)
  - ZeroDivisionError from print_rate=0 (step % rate with rate==0)
  - int(idx) on a weird value -- probed clean under vmap in 25a, kept as a
    defensive catch-all since we cannot enumerate every future JAX delivery
    mode
Design: an OUTER broad `except Exception` (the correctness invariant is
"never raises to the JAX runtime", so a narrower catch cannot be a complete
fix -- unanticipated failure modes must not crash the whole computation
either) wrapping an INNER `except OSError` that specifically disables
output_file (the recurring, retryable-looking failure) so we don't keep
paying a doomed write on every remaining step. Both warn ONCE.
"""
import sys
import threading
import warnings

import jax
import jax.numpy as jnp

import blackjax
pb = sys.modules["blackjax.progress_bar"]


def hardened_step_callback(self, idx):
    """Runs on the host once per scan step (outermost scan only).

    INVARIANT: this function MUST NEVER RAISE, under any input JAX can
    deliver. It executes inside a `jax.debug.callback`; an exception here
    crosses back into the XLA runtime as a fatal JaxRuntimeError and kills
    the ENTIRE traced computation -- not just the progress display -- so a
    bad `output_file` path (permission change, full disk, NFS hiccup) would
    otherwise crash an in-progress MCMC run. Every code path is defensive;
    the outer `except Exception` is intentionally broad because the
    invariant is unconditional and the specific ways a host callback can
    fail cannot be enumerated in advance.
    """
    try:
        step = int(idx)
        rate = self._resolved_print_rate()
        if rate <= 0:
            rate = 1
        if step % rate == 0 or step == self.n_steps - 1:
            if step > self.current_step or step == 0:
                self.current_step = step
            if self.output_file:
                try:
                    tmp = self.output_file + ".tmp"
                    with open(tmp, "w") as fh:
                        fh.write(f"{step} {self.n_steps}")
                    import os
                    os.replace(tmp, self.output_file)
                except OSError as e:
                    if not getattr(self, "_output_file_warned", False):
                        warnings.warn(
                            f"blackjax.progress_bar: disabling output_file "
                            f"after a write failure ({e!r}); the progress "
                            f"bar itself is unaffected.",
                            stacklevel=2,
                        )
                        self._output_file_warned = True
                    self.output_file = None
    except Exception as e:  # noqa: BLE001 -- see invariant in the docstring
        if not getattr(self, "_callback_warned", False):
            warnings.warn(
                f"blackjax.progress_bar: internal callback error "
                f"({e!r}); further progress updates for this context are "
                f"disabled, but the underlying computation is unaffected.",
                stacklevel=2,
            )
            self._callback_warned = True


pb.ProgressState._step_callback = hardened_step_callback


def body(carry, x):
    return carry + x, carry


print("=== Test 1: permission-denied output_file dir -- must NOT crash ===")
import os
import stat
import tempfile

tmpdir = tempfile.mkdtemp()
readonly_dir = os.path.join(tmpdir, "readonly")
os.makedirs(readonly_dir)
os.chmod(readonly_dir, stat.S_IREAD | stat.S_IEXEC)
bad_path = os.path.join(readonly_dir, "progress.txt")

crashed = False
warned = []
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    try:
        with blackjax.progress_bar(label="perm", output_file=bad_path, print_rate=1) as state:
            final, _ = jax.lax.scan(body, 0.0, jnp.arange(50))
            jax.block_until_ready(final)
        print("computation completed WITHOUT crashing; final result correct:",
              float(final) == float(jnp.arange(50).sum()))
        print("state.current_step reached the end despite the write failures:",
              state.current_step)
    except Exception as e:
        crashed = True
        print("STILL CRASHED:", type(e).__name__, e)
    warned = [str(x.message) for x in w]
print("crashed:", crashed)
print("number of warnings emitted (should be exactly 1, not 50 -- warn-once):",
      len(warned))
if warned:
    print("warning text:", warned[0][:120])
os.chmod(readonly_dir, stat.S_IRWXU)

print()
print("=== Test 2: print_rate=0 -- must NOT ZeroDivisionError ===")
crashed2 = False
try:
    with blackjax.progress_bar(label="zero-rate", print_rate=0) as state2:
        final2, _ = jax.lax.scan(body, 0.0, jnp.arange(30))
        jax.block_until_ready(final2)
    print("completed without crash; current_step:", state2.current_step)
except Exception as e:
    crashed2 = True
    print("STILL CRASHED:", type(e).__name__, e)
print("crashed:", crashed2)
