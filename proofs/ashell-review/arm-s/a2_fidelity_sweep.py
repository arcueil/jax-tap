"""ARM-S battery 2: bitwise fidelity across API surface, dtypes, pytrees, errors."""
import traceback
import numpy as np
import jax
import jax.numpy as jnp
import jaxtap as tap

FAILS = []
def bw(a, b):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    if len(la) != len(lb):
        return False
    for x, y in zip(la, lb):
        xa, ya = np.asarray(x), np.asarray(y)
        if xa.dtype != ya.dtype or xa.tobytes() != ya.tobytes():
            return False
    return True

def weak_of(x):
    # report weak_type of leaves
    return [getattr(l, "weak_type", None) for l in jax.tree_util.tree_leaves(x)]

def check(name, fn):
    ref = fn()
    try:
        with tap.record() as rec:
            got = fn()
        jax.block_until_ready(got)
        ok = bw(ref, got)
        wt_ref = weak_of(ref)
        wt_got = weak_of(got)
        wtok = wt_ref == wt_got
        tag = "PASS" if (ok and wtok) else "FAIL"
        if not (ok and wtok):
            FAILS.append(name)
        print(f"[{tag}] {name}: bitwise={ok} weaktype_match={wtok} "
              f"(ref_wt={wt_ref} got_wt={wt_got}) events={len(rec.events)}")
    except Exception as e:
        FAILS.append(name)
        print(f"[FAIL] {name}: RAISED {type(e).__name__}: {str(e)[:150]}")

xs = jnp.arange(5.0, dtype=jnp.float32)

# --- dtype / weak-type fidelity ---
check("int32 carry",
      lambda: jax.lax.scan(lambda c, x: (c + x, c), jnp.int32(0), jnp.arange(5, dtype=jnp.int32)))
check("weak-type float init (python float)",
      lambda: jax.lax.scan(lambda c, x: (c + x, c), 1.0, xs))
check("complex64 carry",
      lambda: jax.lax.scan(lambda c, x: (c + x, c),
                           jnp.complex64(1 + 1j),
                           jnp.arange(5, dtype=jnp.complex64)))
check("float64 (if enabled) / else float32",
      lambda: jax.lax.scan(lambda c, x: (c + x, c), jnp.float32(0.0), xs))
check("bool carry",
      lambda: jax.lax.scan(lambda c, x: (jnp.logical_xor(c, x > 2), c),
                           jnp.bool_(False), xs))

# --- scan API surface through interceptor (keyword variants) ---
check("xs=None + length= keyword",
      lambda: jax.lax.scan(lambda c, _: (c + 1.0, c), jnp.float32(0.0), None, length=5))
check("reverse=True keyword",
      lambda: jax.lax.scan(lambda c, x: (c + x, c), jnp.float32(0.0), xs, reverse=True))
check("unroll=2 keyword",
      lambda: jax.lax.scan(lambda c, x: (c + x, c), jnp.float32(0.0), xs, unroll=2))
check("length=len(xs) redundant keyword",
      lambda: jax.lax.scan(lambda c, x: (c + x, c), jnp.float32(0.0), xs, length=5))
check("f= init= xs= all keyword",
      lambda: jax.lax.scan(f=lambda c, x: (c + x, c), init=jnp.float32(0.0), xs=xs))

# --- pytree carries ---
check("dict carry",
      lambda: jax.lax.scan(lambda c, x: ({"a": c["a"] + x, "b": c["b"] * x}, c["a"]),
                           {"a": jnp.float32(0.0), "b": jnp.float32(1.0)}, xs))
check("carry with None leaf",
      lambda: jax.lax.scan(lambda c, x: ((c[0] + x, None), c[0]),
                           (jnp.float32(0.0), None), xs))
check("carry with empty tuple",
      lambda: jax.lax.scan(lambda c, x: ((c[0] + x, ()), c[0]),
                           (jnp.float32(0.0), ()), xs))
check("nested pytree xs (dict of arrays)",
      lambda: jax.lax.scan(lambda c, x: (c + x["u"] + x["v"], c),
                           jnp.float32(0.0),
                           {"u": xs, "v": xs * 2}))

# --- while_loop multi-leaf / pytree carries ---
check("while multi-leaf tuple carry",
      lambda: jax.lax.while_loop(lambda cs: cs[0] < 10.0,
                                 lambda cs: (cs[0] + 1.0, cs[1] * 2.0),
                                 (jnp.float32(0.0), jnp.float32(1.0))))
check("while dict carry",
      lambda: jax.lax.while_loop(lambda c: c["i"] < 5,
                                 lambda c: {"i": c["i"] + 1, "acc": c["acc"] + c["i"]},
                                 {"i": jnp.int32(0), "acc": jnp.int32(0)}))

# --- PRNG key in carry ---
def prng_scan():
    def body(key, _):
        key, sub = jax.random.split(key)
        return key, jax.random.normal(sub)
    return jax.lax.scan(body, jax.random.PRNGKey(0), None, length=5)
check("PRNG-key carry scan", prng_scan)

print()
print("=" * 70)
print("ERROR TRANSPARENCY: user shape-mismatch inside scan body")
print("=" * 70)

def bad_scan():
    # carry/output shape mismatch: body returns wrong carry shape
    return jax.lax.scan(lambda c, x: (jnp.stack([c, c]), c), jnp.float32(0.0), xs)

# outside
outside_exc = None
try:
    bad_scan()
except Exception as e:
    outside_exc = e
    print("OUTSIDE raised:", type(e).__name__)

inside_exc = None
try:
    with tap.record():
        bad_scan()
except Exception as e:
    inside_exc = e
    print("INSIDE  raised:", type(e).__name__)

print("Same exception KIND:", type(outside_exc) == type(inside_exc)
      if (outside_exc and inside_exc) else "N/A")
if inside_exc is not None:
    tb = "".join(traceback.format_exception(type(inside_exc), inside_exc, inside_exc.__traceback__))
    jaxtap_frames = [ln for ln in tb.splitlines() if "jaxtap" in ln and "File" in ln]
    print("jaxtap frames in INSIDE traceback:", len(jaxtap_frames))
    for ln in jaxtap_frames:
        print("   ", ln.strip())

print()
print("FAILURES:", FAILS if FAILS else "NONE")
