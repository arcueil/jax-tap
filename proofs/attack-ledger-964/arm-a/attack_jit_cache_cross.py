"""Attack: jit-cache cross-contamination.

Case A: function traced INSIDE the context, called AFTER exit. Docstring
admits the callback stays baked in. Does the callback firing into a dead
ProgressState crash, or corrupt anything beyond display (e.g. resurrect a
deleted output_file, per the b335f6f18 fix -- verify it actually holds)?

Case B (reverse): function traced OUTSIDE, called INSIDE a *later*
progress_bar context (should be silent no-bar; confirm the OUTER-most-frame
computed VALUE is still correct, and it doesn't crash or wrongly attach to
the new context's state).
"""
import os
import tempfile
import time
import traceback

import jax
import jax.numpy as jnp

import blackjax  # noqa: F401
from blackjax.progress_bar import progress_bar


def make_runner():
    def body(carry, x):
        return carry + x, carry

    @jax.jit
    def run(x0):
        final, ys = jax.lax.scan(body, x0, jnp.arange(15.0))
        return final, ys

    return run


tmpdir = tempfile.mkdtemp()
path = os.path.join(tmpdir, "progress.txt")

print("=== Case A: trace inside ctx (with output_file), call after exit ===")
run_a = make_runner()
with progress_bar(label="A", output_file=path, print_rate=1) as state:
    out = run_a(0.0)
    jax.block_until_ready(out)
    print("inside-ctx result:", out)

print("output_file exists after ctx exit (should be False):", os.path.exists(path))
print("state.output_file after exit (should be None):", state.output_file)

# Now call the SAME jit-cached function after exit -- callback still baked in.
try:
    out2 = run_a(100.0)
    jax.block_until_ready(out2)
    print("post-exit call result (correct math expected):", out2)
except Exception:
    print("post-exit call CRASHED:")
    traceback.print_exc(limit=4)

time.sleep(0.3)  # let any stray callback-driven file write happen if it will
print("output_file resurrected after post-exit call?:", os.path.exists(path))
if os.path.exists(path):
    with open(path) as fh:
        print("  resurrected contents:", fh.read())

print()
print("=== Case A2: post-exit call happens WHILE a brand-new ctx is open (different output_file) ===")
path2 = os.path.join(tmpdir, "progress2.txt")
with progress_bar(label="A2", output_file=path2, print_rate=1) as state2:
    try:
        out3 = run_a(5.0)  # still the OLD jit cache from ctx "A", stale callback
        jax.block_until_ready(out3)
        print("stale-cached call result inside NEW ctx:", out3)
        print("new ctx state2.current_step (should be untouched by stale callback):", state2.current_step)
        print("did stale callback write into path2 (wrong file)?:", os.path.exists(path2))
    except Exception:
        traceback.print_exc(limit=4)

print()
print("=== Case B: trace OUTSIDE any ctx, then call INSIDE a later ctx ===")


def make_runner_plain():
    def body(carry, x):
        return carry + x, carry

    @jax.jit
    def run(x0):
        final, ys = jax.lax.scan(body, x0, jnp.arange(8.0))
        return final, ys

    return run


run_b = make_runner_plain()
baseline = run_b(0.0)
jax.block_until_ready(baseline)
print("baseline (outside any ctx):", baseline)

with progress_bar(label="B") as state3:
    out_b = run_b(0.0)  # same jit cache, traced with unpatched scan
    jax.block_until_ready(out_b)
    print("same call INSIDE new ctx:", out_b)
    print("match baseline:", bool(out_b[0] == baseline[0]))
    print("state3.n_steps (should stay 0 -- no bar attached):", state3.n_steps)
