"""AYS round 2, (1): construct the counterexample -- pass ONE genuinely
batched value (derived from the vmapped input) into a debug.callback
alongside an unbatched trace-time arange idx. Does the batched arg force
an "unroll" (multiple interleaved fires, once per batch element) while an
all-unbatched call stays a single fire? This isolates the invariant the
progress_bar safety currently depends on.
"""
import jax
import jax.numpy as jnp

received_unbatched_only = []
received_with_batched_arg = []


def cb_unbatched_only(idx):
    received_unbatched_only.append(int(idx))


def cb_with_batched(idx, batched_val):
    # record what we actually receive -- scalar (unrolled per-lane call) or
    # an array (single vectorized call carrying the whole batch)?
    received_with_batched_arg.append(
        (int(idx), type(batched_val).__name__, tuple(getattr(batched_val, "shape", ())), batched_val)
    )


def make_case_a():
    """xs = arange(n) only -- never touches the vmapped input. Mirrors the
    actual progress_bar mechanism exactly."""

    def body(carry, x):
        idx = x
        jax.debug.callback(cb_unbatched_only, idx, ordered=False)
        return carry + 1.0, carry

    def run(key):
        return jax.lax.scan(body, 0.0, jnp.arange(4))

    return run


def make_case_b():
    """xs = (arange(n), key-derived-per-lane-value) -- the callback now
    also receives a value that DOES depend on the vmapped axis."""

    def body(carry, x):
        idx, batched_val = x
        jax.debug.callback(cb_with_batched, idx, batched_val, ordered=False)
        return carry + 1.0, carry

    def run(key):
        # per-lane-varying leaf: depends on `key`, so it IS batched under vmap
        per_lane = key * jnp.ones(4, dtype=jnp.float32)
        return jax.lax.scan(body, 0.0, (jnp.arange(4), per_lane))

    return run


print("=== Case A: all-unbatched xs (arange only), vmapped over 3 lanes ===")
run_a = make_case_a()
keys = jnp.arange(3, dtype=jnp.float32)  # stand-in "keys", just need per-lane variation
out_a = jax.jit(jax.vmap(run_a))(keys)
jax.block_until_ready(out_a)
print("total callback invocations:", len(received_unbatched_only))
print("expected if single-fire-per-step:", 4, " expected if per-lane-unroll:", 3 * 4)
print("sequence:", received_unbatched_only)

print()
print("=== Case B: xs includes a genuinely batched (key-derived) leaf ===")
received_with_batched_arg.clear()
run_b = make_case_b()
out_b = jax.jit(jax.vmap(run_b))(keys)
jax.block_until_ready(out_b)
print("total callback invocations:", len(received_with_batched_arg))
print("expected if single-fire-per-step (vectorized arg):", 4, " expected if per-lane-unroll:", 3 * 4)
for r in received_with_batched_arg:
    print("  received:", r)
