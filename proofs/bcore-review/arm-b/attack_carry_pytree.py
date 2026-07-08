"""ATTACK: carry-augmentation and pytree edge cases.
The rewrite adds a jnp.int32(0) step counter to the carry as (carry_list, step).
Attack carries that stress the flat-list assumption: None leaves, empty carry,
xs-less scan (length-only), deeply nested pytree, and int32-scalar carry leaf."""
import jax
import jax.numpy as jnp
import numpy as np
import jaxtap as tap


def _bytes(x):
    return [np.asarray(v).tobytes() for v in jax.tree_util.tree_leaves(x)]


def bitwise_eq(a, b):
    return _bytes(a) == _bytes(b)


def run(name, f, *args, expect_events=None):
    print(f"\n=== {name} ===")
    try:
        ref = f(*args)
    except Exception as exc:
        print(f"  reference itself raised {type(exc).__name__}: {exc}")
        return
    try:
        ev = []
        got = tap.verbose(f, on_step=lambda e: ev.append(e))(*args)
        jax.block_until_ready(got)
        ok = bitwise_eq(ref, got)
        print(f"  bitwise identical: {ok}   events: {len(ev)}"
              + (f" (expected {expect_events})" if expect_events is not None else ""))
        if not ok:
            print("  VERDICT: NON-BITWISE / structural corruption")
        elif expect_events is not None and len(ev) != expect_events:
            print(f"  VERDICT: EVENT COUNT WRONG (got {len(ev)}, expected {expect_events})")
        else:
            print("  VERDICT: clean")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"  VERDICT: CRASH {type(exc).__name__}: {exc}")


xs = jnp.arange(5.0, dtype=jnp.float32)

# 1. xs-less scan (length only, xs=None)
def f_lengthonly(x0):
    return jax.lax.scan(lambda c, _: (c + 1.0, c), x0, None, length=5)
run("xs-less scan (length=5, xs=None)", f_lengthonly, jnp.float32(0.0), expect_events=5)

# 2. carry containing a None leaf (pytree with None)
def f_none_carry(x0):
    # carry is a dict with a None value -> None is NOT a leaf (pytree node)
    def body(c, x):
        return {"a": c["a"] + x, "b": None}, c["a"]
    return jax.lax.scan(body, {"a": x0, "b": None}, xs)
run("carry with None leaf in pytree", f_none_carry, jnp.float32(0.0), expect_events=5)

# 3. empty-ish carry: carry is an empty tuple, all state via xs->ys
def f_empty_carry(xs_):
    def body(c, x):
        return (), x * 2.0
    _, ys = jax.lax.scan(body, (), xs_)
    return ys
run("empty-tuple carry ()", f_empty_carry, xs, expect_events=5)

# 4. deeply nested pytree carry
def f_nested_carry(x0):
    init = {"outer": (x0, {"inner": [x0, x0]})}
    def body(c, x):
        a = c["outer"][0] + x
        lst = c["outer"][1]["inner"]
        return {"outer": (a, {"inner": [lst[0] * 1.0, lst[1] + x]})}, a
    return jax.lax.scan(body, init, xs)
run("deeply nested pytree carry", f_nested_carry, jnp.float32(1.0), expect_events=5)

# 5. carry leaf that is itself an int32 scalar (collision w/ step counter type?)
def f_int_carry(x0):
    # carry has an int32 counter leaf plus a float leaf
    def body(c, x):
        i, v = c
        return (i + jnp.int32(1), v + x), v
    return jax.lax.scan(body, (jnp.int32(0), x0), xs)
run("carry containing an int32 scalar leaf", f_int_carry, jnp.float32(0.0), expect_events=5)

# 6. while_loop with a nested-pytree carry (bare tuple)
def f_while_pytree(v0):
    def cond(c):
        return c[0] < 10.0
    def body(c):
        return (c[0] + 1.0, c[1] * 1.0)
    return jax.lax.while_loop(cond, body, (v0, jnp.float32(2.0)))
run("while_loop with tuple-pytree carry", f_while_pytree, jnp.float32(0.0))

# 7. scan whose ys output is a pytree (not flat)
def f_pytree_ys(x0):
    def body(c, x):
        return c + x, {"double": c * 2.0, "orig": c}
    return jax.lax.scan(body, x0, xs)
run("scan with pytree ys output", f_pytree_ys, jnp.float32(0.0), expect_events=5)
