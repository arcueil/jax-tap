"""AYS (1)(iii): does the display thread's tqdm output crash anything
beyond itself when stderr is closed/redirected mid-run (pytest capture
teardown, Jupyter cell boundary)? Confirm: does it spew an uncaught
traceback, and does the MAIN thread / computation survive regardless?
"""
import io
import sys
import threading
import time

import jax
import jax.numpy as jnp

import blackjax

# Capture anything threading's default excepthook would print (that's where
# "Exception in thread ...:" tracebacks from an unhandled thread exception
# go by default in Python 3.8+).
thread_exceptions = []
def custom_hook(args):
    thread_exceptions.append(args)
old_hook = threading.excepthook
threading.excepthook = custom_hook

def body(carry, x):
    time.sleep(0.02)  # slow enough that the display thread ticks a few times
    return carry + x, carry

# Redirect stderr to a stream, then CLOSE it mid-run (simulating pytest's
# capsys teardown or a Jupyter kernel restart closing the underlying fd)
# while the display thread is actively writing to it.
fake_stderr = io.StringIO()
real_stderr = sys.stderr
sys.stderr = fake_stderr

crashed = False
try:
    with blackjax.progress_bar(label="closed-stderr", print_rate=1) as state:
        def close_stderr_soon():
            time.sleep(0.05)
            fake_stderr.close()  # closed WHILE the render loop may be mid-write
        t = threading.Thread(target=close_stderr_soon)
        t.start()
        final, _ = jax.lax.scan(body, 0.0, jnp.arange(15))
        jax.block_until_ready(final)
        t.join()
except Exception as e:
    crashed = True
    sys.stderr = real_stderr
    print("MAIN THREAD / COMPUTATION CRASHED:", type(e).__name__, e)
finally:
    sys.stderr = real_stderr

threading.excepthook = old_hook

print("main thread computation crashed:", crashed)
print("final result still correct (computation unaffected by the display "
      "thread's stderr trouble):", float(final) == float(jnp.arange(15).sum()))
print("number of uncaught exceptions surfaced from background threads:",
      len(thread_exceptions))
for exc in thread_exceptions:
    print("  thread:", exc.thread.name if exc.thread else None,
          "exc_type:", exc.exc_type.__name__ if exc.exc_type else None,
          "exc_value:", exc.exc_value)
