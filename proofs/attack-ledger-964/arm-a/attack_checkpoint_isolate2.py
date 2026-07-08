import jax
import jax.numpy as jnp

import blackjax  # noqa: F401
from blackjax.progress_bar import progress_bar


def make_fns():
    def raw_loss(theta):
        def body(carry, x):
            new_carry = jnp.sin(carry * theta + x)
            return new_carry, new_carry

        final, _ = jax.lax.scan(body, 0.0, jnp.arange(20.0))
        return final**2

    return raw_loss, jax.checkpoint(raw_loss)


def count(label, fn, *args):
    calls = []
    with progress_bar(label=label) as state:
        orig_cb = state._step_callback

        def counting_cb(idx):
            calls.append(int(idx))
            orig_cb(idx)

        state._step_callback = counting_cb
        out = fn(*args)
        jax.block_until_ready(out)
    print(f"{label}: n_steps={state.n_steps} calls={len(calls)} out={out}")
    return calls


# Case 1: brand new function, FIRST ever call is jit(checkpoint(...)), no grad.
raw1, remat1 = make_fns()
count("FRESH-jit-checkpoint-no-grad-FIRST-CALL", jax.jit(remat1), 0.7)
# Second call of the SAME jitted fn, same context style -- does it now fire (recompiled/cached) or still 0?
count("FRESH-jit-checkpoint-no-grad-SECOND-CALL-same-fn", jax.jit(remat1), 0.7)

print()

# Case 2: brand new function, FIRST ever call is jit(grad(checkpoint(...))).
raw2, remat2 = make_fns()
count("FRESH-jit-grad-checkpoint-FIRST-CALL", jax.jit(lambda t: jax.grad(remat2)(t)), 0.7)

print()

# Case 3: brand new function, FIRST ever call is plain grad(checkpoint(...)) (no explicit jit).
raw3, remat3 = make_fns()
count("FRESH-grad-checkpoint-no-jit-FIRST-CALL", lambda t: jax.grad(remat3)(t), 0.7)

print()

# Case 4: does calling checkpoint(fn) once WITHOUT jit "warm" it so a LATER
# jit(checkpoint(fn)) call in a brand new progress_bar context also gets 0?
raw4, remat4 = make_fns()
count("FRESH-eager-checkpoint-warmup-call", remat4, 0.7)
count("FRESH-then-jit-SAME-fn-new-ctx", jax.jit(remat4), 0.7)
