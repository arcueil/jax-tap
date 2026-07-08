"""ATTACK: effects / ordered primitives and the user's OWN callbacks inside a
tapped program. The walker re-binds every non-CF primitive via get_bind_params;
does the user's io_callback (ordered=True) / debug.print survive the walk and
keep firing? Does an ordered effect token get corrupted?"""
import jax
import jax.numpy as jnp
import numpy as np
import jaxtap as tap


def _bytes(x):
    return [np.asarray(v).tobytes() for v in jax.tree_util.tree_leaves(x)]


def bitwise_eq(a, b):
    return _bytes(a) == _bytes(b)


xs = jnp.arange(4.0, dtype=jnp.float32)

print("=" * 66)
print("CASE 1: user's jax.debug.print INSIDE a tapped scan body")
print("=" * 66)
user_prints = []

def f_userprint(x0, xs_):
    def body(c, x):
        jax.debug.print("user-sees c={c}", c=c)
        return c + x, c
    return jax.lax.scan(body, x0, xs_)

ref = f_userprint(jnp.float32(0.0), xs)
ev = []
# capture stdout to count user debug prints
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    got = tap.verbose(f_userprint, on_step=lambda e: ev.append(e))(jnp.float32(0.0), xs)
    jax.block_until_ready(got)
n_user_prints = buf.getvalue().count("user-sees")
print("bitwise identical  :", bitwise_eq(ref, got))
print("jaxtap events      :", len(ev))
print("user debug.prints  :", n_user_prints, " <-- EXPECTED 4 (user's own callback must survive)")
print("VERDICT:", "user callback SURVIVED" if n_user_prints == 4 else "user callback DROPPED/BROKEN")


print("\n" + "=" * 66)
print("CASE 2: user's io_callback(ordered=True) INSIDE a tapped scan body")
print("=" * 66)
seen = []

def f_ordered(x0, xs_):
    def sink(v):
        seen.append(float(v))
    def body(c, x):
        jax.experimental.io_callback(sink, None, c, ordered=True)
        return c + x, c
    return jax.lax.scan(body, x0, xs_)

import jax.experimental
try:
    ref2 = f_ordered(jnp.float32(0.0), xs)
    seen.clear()
    ev2 = []
    got2 = tap.verbose(f_ordered, on_step=lambda e: ev2.append(e))(jnp.float32(0.0), xs)
    jax.block_until_ready(got2)
    print("bitwise identical    :", bitwise_eq(ref2, got2))
    print("ordered cb fires seen:", len(seen), " <-- EXPECTED 4")
    print("order preserved      :", seen == sorted(seen), seen)
    print("VERDICT:", "ordered effect SURVIVED" if len(seen) == 4 else "ordered effect BROKEN")
except Exception as exc:
    import traceback
    traceback.print_exc()
    print("VERDICT: ordered io_callback CRASHED through walker:", type(exc).__name__)


print("\n" + "=" * 66)
print("CASE 3: user's pure_callback INSIDE a tapped scan body")
print("=" * 66)
def f_pure(x0, xs_):
    def add_host(a, b):
        return np.asarray(a) + np.asarray(b)
    def body(c, x):
        c2 = jax.pure_callback(add_host, jax.ShapeDtypeStruct((), jnp.float32), c, x)
        return c2, c
    return jax.lax.scan(body, x0, xs_)

try:
    ref3 = f_pure(jnp.float32(0.0), xs)
    ev3 = []
    got3 = tap.verbose(f_pure, on_step=lambda e: ev3.append(e))(jnp.float32(0.0), xs)
    jax.block_until_ready(got3)
    print("bitwise identical:", bitwise_eq(ref3, got3))
    print("jaxtap events    :", len(ev3))
    print("VERDICT:", "pure_callback SURVIVED" if bitwise_eq(ref3, got3) else "pure_callback BROKEN")
except Exception as exc:
    import traceback
    traceback.print_exc()
    print("VERDICT: pure_callback CRASHED:", type(exc).__name__)
