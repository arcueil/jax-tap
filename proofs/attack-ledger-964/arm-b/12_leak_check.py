"""Attack #6b/#8b: process-hygiene leak check over N=100 sequential
(non-overlapping) contexts -- daemon thread count and open-fd count should
return to baseline after each context tears down."""
import os
import threading

import jax
import jax.numpy as jnp

import blackjax

def fd_count():
    try:
        return len(os.listdir(f"/proc/{os.getpid()}/fd"))
    except FileNotFoundError:
        return -1

base_threads = threading.active_count()
base_fds = fd_count()

def body(carry, x):
    return carry + x, carry

for i in range(100):
    with blackjax.progress_bar(label=f"iter-{i}", print_rate=1000000) as state:
        final, _ = jax.lax.scan(body, 0.0, jnp.arange(10))
        jax.block_until_ready(final)

end_threads = threading.active_count()
end_fds = fd_count()

print(f"threads: base={base_threads} end={end_threads} (delta={end_threads - base_threads})")
print(f"fds:     base={base_fds} end={end_fds} (delta={end_fds - base_fds})")
