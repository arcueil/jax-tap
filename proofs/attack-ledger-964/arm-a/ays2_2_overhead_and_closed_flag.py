"""AYS round 2, (2): quantify the per-step overhead of a phantom (orphaned)
debug.callback baked into a compiled scan, and check whether a cheap
`closed` guard changes that overhead at all (it shouldn't -- the host
round-trip dispatch is what costs time, not the python-side body)."""
import timeit

import jax
import jax.numpy as jnp

N_STEPS = 1000
N_REPEAT = 20


def make_plain_scan():
    def body(carry, x):
        return carry + x, carry

    def f(x0):
        final, _ = jax.lax.scan(body, x0, jnp.arange(float(N_STEPS)))
        return final

    return jax.jit(f)


def make_scan_with_live_callback():
    call_count = [0]

    def cb(idx):
        call_count[0] += 1

    def body(carry, x):
        orig_x, idx = x
        jax.debug.callback(cb, idx, ordered=False)
        return carry + orig_x, carry

    def f(x0):
        final, _ = jax.lax.scan(
            body, x0, (jnp.arange(float(N_STEPS)), jnp.arange(N_STEPS))
        )
        return final

    return jax.jit(f), call_count


def make_scan_with_closed_guard_callback():
    call_count = [0]
    closed = [True]  # simulate "already exited" -- the realistic phantom case

    def cb(idx):
        if closed[0]:
            return
        call_count[0] += 1  # pragma: no cover -- guard should always trigger here

    def body(carry, x):
        orig_x, idx = x
        jax.debug.callback(cb, idx, ordered=False)
        return carry + orig_x, carry

    def f(x0):
        final, _ = jax.lax.scan(
            body, x0, (jnp.arange(float(N_STEPS)), jnp.arange(N_STEPS))
        )
        return final

    return jax.jit(f), call_count


plain = make_plain_scan()
with_cb, cb_count = make_scan_with_live_callback()
with_guarded_cb, guarded_count = make_scan_with_closed_guard_callback()

# warm up (compile) each once, outside timing.
jax.block_until_ready(plain(0.0))
jax.block_until_ready(with_cb(0.0))
jax.block_until_ready(with_guarded_cb(0.0))


def run_plain():
    jax.block_until_ready(plain(0.0))


def run_with_cb():
    jax.block_until_ready(with_cb(0.0))


def run_with_guarded_cb():
    jax.block_until_ready(with_guarded_cb(0.0))


t_plain = timeit.timeit(run_plain, number=N_REPEAT) / N_REPEAT
t_cb = timeit.timeit(run_with_cb, number=N_REPEAT) / N_REPEAT
t_guarded = timeit.timeit(run_with_guarded_cb, number=N_REPEAT) / N_REPEAT

print(f"N_STEPS={N_STEPS}, averaged over {N_REPEAT} runs")
print(f"plain scan, no callback at all       : {t_plain*1e6:10.1f} us/run -> {t_plain/N_STEPS*1e6:6.2f} us/step")
print(f"scan w/ LIVE (unguarded) callback     : {t_cb*1e6:10.1f} us/run -> {t_cb/N_STEPS*1e6:6.2f} us/step")
print(f"scan w/ CLOSED-GUARDED callback        : {t_guarded*1e6:10.1f} us/run -> {t_guarded/N_STEPS*1e6:6.2f} us/step")
print()
print(f"overhead of ANY debug.callback vs plain scan : {(t_cb - t_plain)/N_STEPS*1e6:6.2f} us/step")
print(f"overhead saved by closed-guard vs live body   : {(t_cb - t_guarded)/N_STEPS*1e6:6.2f} us/step")
print(f"callback fire counts -- live: {cb_count[0]}, closed-guarded (should be 0 real work): {guarded_count[0]}")
