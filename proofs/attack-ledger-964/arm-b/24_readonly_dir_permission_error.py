"""AYS (e), stated plainly: the NFS/cross-device os.replace attack listed in
the original brief's line 7 was NOT run in round 1. Running the closest
locally-reproducible adjacent case now: output_file pointing at a directory
this process cannot write to (permission denied), to see whether the
resulting exception inside jax.debug.callback crashes the whole traced
computation or fails silently.
"""
import os
import stat
import tempfile

import jax
import jax.numpy as jnp

import blackjax

tmpdir = tempfile.mkdtemp()
readonly_dir = os.path.join(tmpdir, "readonly")
os.makedirs(readonly_dir)
os.chmod(readonly_dir, stat.S_IREAD | stat.S_IEXEC)  # r-x, no write
bad_path = os.path.join(readonly_dir, "progress.txt")

def body(carry, x):
    return carry + x, carry

crashed = False
try:
    with blackjax.progress_bar(label="perm", output_file=bad_path, print_rate=1) as state:
        final, _ = jax.lax.scan(body, 0.0, jnp.arange(20))
        jax.block_until_ready(final)
    print("context exited normally; final scan result correct:",
          float(final) == float(jnp.arange(20).sum()))
except Exception as e:
    crashed = True
    print("CRASHED:", type(e).__name__, e)

print("whole computation crashed due to a PermissionError inside the "
      "callback:", crashed)

os.chmod(readonly_dir, stat.S_IRWXU)  # restore for cleanup
