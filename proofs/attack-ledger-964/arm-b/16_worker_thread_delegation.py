"""AYS (b): does the legitimate pattern -- context entered on the MAIN
thread, the actual scan executed on a WORKER thread (ThreadPoolExecutor /
threading.Thread, a common 'run sampling in the background so the notebook
UI stays responsive' pattern) -- work TODAY? If yes, a naive thread-id
affinity fix would regress it.
"""
import concurrent.futures
import threading

import jax
import jax.numpy as jnp

import blackjax

def body(carry, x):
    return carry + x, carry

print("=== Pattern 1: raw threading.Thread, context entered on main thread ===")
with blackjax.progress_bar(label="main-ctx-raw-thread", print_rate=1) as state:
    result = {}
    def worker():
        final, _ = jax.lax.scan(body, 0.0, jnp.arange(321))
        jax.block_until_ready(final)
        result["final"] = final
    t = threading.Thread(target=worker)
    t.start()
    t.join()
print("raw-Thread: bar advanced (n_steps==321)?", state.n_steps == 321,
      "n_steps=", state.n_steps, "current_step=", state.current_step)

print()
print("=== Pattern 2: concurrent.futures.ThreadPoolExecutor.submit ===")
with blackjax.progress_bar(label="main-ctx-executor", print_rate=1) as state2:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(lambda: jax.block_until_ready(
            jax.lax.scan(body, 0.0, jnp.arange(654))))
        fut.result()
print("ThreadPoolExecutor: bar advanced (n_steps==654)?", state2.n_steps == 654,
      "n_steps=", state2.n_steps, "current_step=", state2.current_step)
