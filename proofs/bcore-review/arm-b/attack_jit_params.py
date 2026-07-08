"""ATTACK: the jit/pjit re-wrap `jax.jit(_inner_call)(*invals)` forwards NONE of
the original jit eqn's params. Enumerate every dropped param and classify each
as CORRECTNESS vs PERF/COSMETIC by empirical test on CPU single-device.
"""
import warnings
import jax
import jax.numpy as jnp
import numpy as np
import jaxtap as tap


def _bytes(x):
    return [np.asarray(v).tobytes() for v in jax.tree_util.tree_leaves(x)]


def bitwise_eq(a, b):
    return _bytes(a) == _bytes(b)


print("=" * 70)
print("PART 1: enumerate params carried on a jit eqn and what the re-wrap keeps")
print("=" * 70)

def f_probe(x):
    @jax.jit
    def inner(y):
        return y * 2.0
    return inner(x)

closed = jax.make_jaxpr(f_probe)(jnp.float32(1.0))
jit_eqn = [e for e in closed.jaxpr.eqns if e.primitive.name == "jit"][0]
orig_params = set(jit_eqn.params.keys())
# The re-wrap builds jax.jit(_inner_call) with defaults; only "jaxpr" content is
# reused (walked). Every OTHER param is reset to jax.jit defaults.
kept = {"jaxpr"}
dropped = orig_params - kept
print("jit eqn params :", sorted(orig_params))
print("re-wrap keeps  :", sorted(kept), "(walks the sub-jaxpr; all else -> jit defaults)")
print("DROPPED         :", sorted(dropped))


print("\n" + "=" * 70)
print("PART 2: donated_invars -- does dropping donation change behavior?")
print("=" * 70)

def f_donate(x, xs):
    # non-inlined nested jit with donation, called inside a tapped scan-bearing fn
    inner = jax.jit(lambda a: a + 1.0, donate_argnums=0)
    y = inner(x)
    out, _ = jax.lax.scan(lambda c, b: (c + b, c), y, xs)
    return out

x = jnp.float32(1.0)
xs = jnp.arange(4.0, dtype=jnp.float32)

# inspect donated_invars on the nested jit eqn
cj = jax.make_jaxpr(f_donate)(x, xs)
for e in cj.jaxpr.eqns:
    if e.primitive.name in ("jit", "pjit"):
        print("nested jit donated_invars param:", e.params.get("donated_invars"))

ref = f_donate(x, xs)
ev = []
with warnings.catch_warnings(record=True) as wl:
    warnings.simplefilter("always")
    got = tap.verbose(f_donate, on_step=lambda e: ev.append(e))(x, xs)
    jax.block_until_ready(got)
donate_warnings = [w for w in wl if "donat" in str(w.message).lower()]
print("bitwise identical:", bitwise_eq(ref, got))
print("donation-related warnings during tapped run:", len(donate_warnings))
print("=> donation dropped: correctness-neutral" if bitwise_eq(ref, got) else "=> CORRECTNESS BREAK")


print("\n" + "=" * 70)
print("PART 3: compiler_options_kvs -- dropped compile flags")
print("=" * 70)
def f_copts(x):
    inner = jax.jit(lambda y: y * 3.0, compiler_options={"xla_cpu_enable_fast_math": True})
    return inner(x)
try:
    cj = jax.make_jaxpr(f_copts)(jnp.float32(2.0))
    for e in cj.jaxpr.eqns:
        if e.primitive.name in ("jit", "pjit"):
            print("nested jit compiler_options_kvs:", e.params.get("compiler_options_kvs"))
    ref = f_copts(jnp.float32(2.0))
    got = tap.verbose(f_copts, on_step=lambda e: None)(jnp.float32(2.0))
    print("bitwise identical:", bitwise_eq(ref, got))
    print("=> compiler_options dropped from re-wrap (nested-level effect is XLA-dependent)")
except Exception as exc:
    print("compiler_options probe raised:", type(exc).__name__, exc)


print("\n" + "=" * 70)
print("PART 4: name -- profiling/telemetry attribution lost")
print("=" * 70)
def f_named(x):
    @jax.jit
    def my_special_kernel(y):
        return y - 1.0
    return my_special_kernel(x)
cj = jax.make_jaxpr(f_named)(jnp.float32(5.0))
for e in cj.jaxpr.eqns:
    if e.primitive.name == "jit":
        print("original jit name param:", e.params.get("name"))
print("re-wrap name will be '_inner_call' (the walker's closure) -> user kernel name lost")


print("\n" + "=" * 70)
print("PART 5: keep_unused -- dropped-unused-input semantics")
print("=" * 70)
def f_keep(x, unused):
    inner = jax.jit(lambda a, b: a * 2.0, keep_unused=True)
    y = inner(x, unused)
    out, _ = jax.lax.scan(lambda c, z: (c + z, c), y, jnp.arange(3.0, dtype=jnp.float32))
    return out
x2, u2 = jnp.float32(1.0), jnp.float32(99.0)
cj = jax.make_jaxpr(f_keep)(x2, u2)
for e in cj.jaxpr.eqns:
    if e.primitive.name in ("jit", "pjit"):
        print("nested jit keep_unused:", e.params.get("keep_unused"))
ref = f_keep(x2, u2)
got = tap.verbose(f_keep, on_step=lambda e: None)(x2, u2)
print("bitwise identical:", bitwise_eq(ref, got))
