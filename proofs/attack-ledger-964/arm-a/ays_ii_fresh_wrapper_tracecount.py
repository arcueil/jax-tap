"""AYS (ii): pin down whether a FRESH jax.jit(fn) wrapper object (distinct
Python object, same underlying fn) actually skips re-tracing (cache hit)
or genuinely re-traces. Use a python-level side effect INSIDE the traced
function (executed only when the tracer runs the function body, i.e. at
trace time, not at every call) to detect real re-tracing independent of
the progress_bar callback mechanism entirely.
"""
import jax
import jax.numpy as jnp

import blackjax  # noqa: F401
from blackjax.progress_bar import progress_bar

trace_events = []


def scan_body(carry, x):
    return carry + x, carry


def plain_fn(x):
    trace_events.append("TRACED")  # only runs when the python fn is called by the tracer
    final, _ = jax.lax.scan(scan_body, x, jnp.arange(10.0))
    return final


wrapper1 = jax.jit(plain_fn)
print("wrapper1 id:", id(wrapper1))
with progress_bar(label="w1") as s1:
    out1 = wrapper1(1.0)
    jax.block_until_ready(out1)
print("after wrapper1 call: trace_events=", trace_events, "n_steps=", s1.n_steps)

wrapper2 = jax.jit(plain_fn)  # a BRAND NEW PjitFunction object, same underlying plain_fn
print("wrapper2 id:", id(wrapper2), "is wrapper1 is wrapper2:", wrapper1 is wrapper2)
with progress_bar(label="w2") as s2:
    out2 = wrapper2(1.0)
    jax.block_until_ready(out2)
print("after wrapper2 (FRESH object) call: trace_events=", trace_events, "n_steps=", s2.n_steps)

print()
print("=== control: genuinely different function object (same code) ===")


def make_fresh_plain_fn():
    def scan_body2(carry, x):
        return carry + x, carry

    def plain_fn2(x):
        trace_events.append("TRACED2")
        final, _ = jax.lax.scan(scan_body2, x, jnp.arange(10.0))
        return final

    return plain_fn2


plain_fn2 = make_fresh_plain_fn()
wrapper3 = jax.jit(plain_fn2)
with progress_bar(label="w3-fresh-fn") as s3:
    out3 = wrapper3(1.0)
    jax.block_until_ready(out3)
print("after wrapper3 (genuinely NEW fn object) call: trace_events=", trace_events, "n_steps=", s3.n_steps)
