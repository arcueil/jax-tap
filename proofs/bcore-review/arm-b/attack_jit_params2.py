"""ATTACK (isolated): classify each dropped jit param as CORRECTNESS vs PERF.
Uses FRESH arrays per call so a donating reference run cannot poison the input."""
import warnings
import jax
import jax.numpy as jnp
import numpy as np
import jaxtap as tap


def _bytes(x):
    return [np.asarray(v).tobytes() for v in jax.tree_util.tree_leaves(x)]


def bitwise_eq(a, b):
    return _bytes(a) == _bytes(b)


def fresh():
    return jnp.arange(4.0, dtype=jnp.float32)  # new buffer each call


print("=" * 66)
print("PART 2b: donation dropped -> walker is conservative (no donate).")
print("Isolated with fresh buffers so no prior call deletes the input.")
print("=" * 66)

def f_donate(xs):
    inner = jax.jit(lambda a: a + 1.0, donate_argnums=0)
    y = inner(xs)                       # donates its own fresh input
    out, _ = jax.lax.scan(lambda c, b: (c + b, c), jnp.float32(0.0), y)
    return out

# reference computed on its OWN fresh buffer
ref = f_donate(fresh())
ev = []
with warnings.catch_warnings(record=True) as wl:
    warnings.simplefilter("always")
    got = tap.verbose(f_donate, on_step=lambda e: ev.append(e))(fresh())
    jax.block_until_ready(got)
dwarn = [w for w in wl if "delet" in str(w.message).lower() or "donat" in str(w.message).lower()]
print("bitwise identical      :", bitwise_eq(ref, got))
print("events                 :", len(ev))
print("donation/deletion warns:", len(dwarn))
print("CLASSIFY: donation drop is CORRECTNESS-SAFE (walker keeps buffer alive)"
      if bitwise_eq(ref, got) else "CLASSIFY: CORRECTNESS BREAK")


print("\n" + "=" * 66)
print("PART 3b: compiler_options_kvs dropped")
print("=" * 66)
def f_copts(xs):
    inner = jax.jit(lambda a: a * 3.0, compiler_options={"xla_cpu_enable_fast_math": True})
    y = inner(xs)
    out, _ = jax.lax.scan(lambda c, b: (c + b, c), jnp.float32(0.0), y)
    return out
try:
    cj = jax.make_jaxpr(f_copts)(fresh())
    for e in cj.jaxpr.eqns:
        if e.primitive.name in ("jit", "pjit"):
            print("nested jit compiler_options_kvs:", e.params.get("compiler_options_kvs"))
    ref = f_copts(fresh())
    got = tap.verbose(f_copts, on_step=lambda e: None)(fresh())
    print("bitwise identical:", bitwise_eq(ref, got),
          "(fast_math flag dropped; XLA respects it only at outermost compile)")
except Exception as exc:
    print("compiler_options probe raised:", type(exc).__name__, str(exc)[:120])


print("\n" + "=" * 66)
print("PART 5b: keep_unused dropped")
print("=" * 66)
def f_keep(xs, unused):
    inner = jax.jit(lambda a, b: a * 2.0, keep_unused=True)
    y = inner(xs, unused)
    out, _ = jax.lax.scan(lambda c, z: (c + z, c), jnp.float32(0.0), y)
    return out
u = jnp.float32(99.0)
cj = jax.make_jaxpr(f_keep)(fresh(), u)
for e in cj.jaxpr.eqns:
    if e.primitive.name in ("jit", "pjit"):
        print("nested jit keep_unused:", e.params.get("keep_unused"))
ref = f_keep(fresh(), u)
got = tap.verbose(f_keep, on_step=lambda e: None)(fresh(), u)
print("bitwise identical:", bitwise_eq(ref, got))


print("\n" + "=" * 66)
print("PART 6: in_shardings / out_shardings dropped (single CPU device)")
print("=" * 66)
from jax.sharding import SingleDeviceSharding
dev = jax.devices()[0]
shard = SingleDeviceSharding(dev)
def f_shard(xs):
    inner = jax.jit(lambda a: a + 1.0, in_shardings=shard, out_shardings=shard)
    y = inner(xs)
    out, _ = jax.lax.scan(lambda c, b: (c + b, c), jnp.float32(0.0), y)
    return out
try:
    cj = jax.make_jaxpr(f_shard)(fresh())
    for e in cj.jaxpr.eqns:
        if e.primitive.name in ("jit", "pjit"):
            print("nested jit in_shardings:", e.params.get("in_shardings"),
                  "out_shardings:", e.params.get("out_shardings"))
    ref = f_shard(fresh())
    got = tap.verbose(f_shard, on_step=lambda e: None)(fresh())
    print("bitwise identical:", bitwise_eq(ref, got),
          "(single device: sharding masked -> THEORETIC on multi-device)")
except Exception as exc:
    print("sharding probe raised:", type(exc).__name__, str(exc)[:120])
