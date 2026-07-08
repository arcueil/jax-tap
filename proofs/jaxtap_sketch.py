# jax-tap B-core minimal sketch: jaxpr-walker transform on JAX 0.10.1.
# Contract to prove:
#   1. tapped(f) outputs are bitwise-identical to f
#   2. taps fire per scan step with (path, step, selected carry value)
#   3. nested scans get stable path addresses (scan[0]/scan[0])
#   4. the transform composes with jax.jit (recurse through 'jit' eqns)
import jax
import jax.numpy as jnp
import numpy as np

CONTROL_FLOW = {"scan"}
CALL_PRIMS = {"jit", "pjit", "closed_call", "custom_jvp_call", "custom_vjp_call"}


def tapped(f, tap_cb):
    """Return g s.t. g(*args) == f(*args) bitwise, emitting carry taps."""

    def wrapped(*args):
        closed = jax.make_jaxpr(f)(*args)
        flat = jax.tree_util.tree_leaves(args)
        out_flat = _interp(closed.jaxpr, closed.consts, flat, tap_cb, path="")
        return jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(jax.eval_shape(f, *args)), out_flat)

    return wrapped


def _interp(jaxpr, consts, args, tap_cb, path):
    env = {}

    def read(a):
        return a.val if type(a).__name__ == "Literal" else env[a]

    def write(v, val):
        env[v] = val

    assert len(jaxpr.constvars) == len(consts) and len(jaxpr.invars) == len(args)
    for v, val in zip(jaxpr.constvars, consts):
        write(v, val)
    for v, val in zip(jaxpr.invars, args):
        write(v, val)
    n_cf = 0  # per-level control-flow counter -> stable addresses
    for eqn in jaxpr.eqns:
        invals = [read(a) for a in eqn.invars]
        prim = eqn.primitive.name
        if prim == "scan":
            here = f"{path}scan[{n_cf}]"; n_cf += 1
            outvals = _rewrite_scan(eqn, invals, tap_cb, here)
        elif prim in CALL_PRIMS:
            here = path  # transparent: recurse, keep addressing level
            inner = eqn.params["jaxpr"]
            outvals = _interp(inner.jaxpr, inner.consts, invals, tap_cb, here)
        else:
            outvals = eqn.primitive.bind(*invals, **eqn.params)
            if not eqn.primitive.multiple_results:
                outvals = [outvals]
        assert len(eqn.outvars) == len(outvals)
        for v, val in zip(eqn.outvars, outvals):
            write(v, val)
    return [read(v) for v in jaxpr.outvars]


def _rewrite_scan(eqn, invals, tap_cb, here):
    p = eqn.params
    body, nc, ncar = p["jaxpr"], p["num_consts"], p["num_carry"]
    consts, init, xs = invals[:nc], invals[nc:nc + ncar], invals[nc + ncar:]

    def body_fn(carry_step, x):
        carry, step = carry_step
        # recurse: nested control flow inside the body is instrumented too
        outs = _interp(body.jaxpr, body.consts, [*consts, *carry, *x],
                       tap_cb, path=here + "/")
        new_carry, ys = outs[:ncar], outs[ncar:]
        jax.debug.callback(tap_cb, here, step, new_carry, ordered=False)
        return (new_carry, step + 1), ys

    (carry, _), ys = jax.lax.scan(body_fn, (init, jnp.int32(0)), xs,
                                  length=p["length"], reverse=p["reverse"],
                                  unroll=p["unroll"])
    return [*carry, *ys]


# ---------------- proof harness ----------------
def inner_step(c, x):
    return c * 1.001 + jnp.sin(x), c


def nested(c, x):
    c2, _ = jax.lax.scan(inner_step, c + x, jnp.arange(3.0))  # inner scan
    return c2, c2 * 2.0


def f(x0, xs):
    c, ys = jax.lax.scan(nested, x0, xs)
    return c, ys


x0, xs = jnp.float32(0.5), jnp.linspace(0., 1., 4, dtype=jnp.float32)

events = []
def cb(pth, step, carry):
    events.append((pth, int(step), [np.asarray(v) for v in carry]))

# --- 1+2+3: identity, taps, nesting ---
ref = f(x0, xs)
got = tapped(f, cb)(x0, xs)
jax.block_until_ready(got)
for r, g in zip(jax.tree_util.tree_leaves(ref), jax.tree_util.tree_leaves(got)):
    assert np.asarray(r).tobytes() == np.asarray(g).tobytes(), "NOT bitwise identical"
paths = sorted({e[0] for e in events})
print("BITWISE IDENTICAL: OK")
print("tap paths seen:", paths)
print("event count:", len(events), "(expect 4 outer + 4*3 inner = 16)")

# --- 4: under jit (transform of a jitted fn; and jit of the transformed fn) ---
events.clear()
got_j = tapped(jax.jit(f), cb)(x0, xs)          # walker recurses through 'jit' eqn
jax.block_until_ready(got_j)
n1 = len(events)
events.clear()
got_jt = jax.jit(tapped(f, cb))(x0, xs)          # transformed fn is itself jittable
jax.block_until_ready(got_jt)
n2 = len(events)
for r, g in zip(jax.tree_util.tree_leaves(ref),
                jax.tree_util.tree_leaves(got_j) + jax.tree_util.tree_leaves(got_jt)):
    pass
ok_j = all(np.asarray(r).tobytes() == np.asarray(g).tobytes()
           for r, g in zip(jax.tree_util.tree_leaves(ref), jax.tree_util.tree_leaves(got_j)))
ok_jt = all(np.asarray(r).tobytes() == np.asarray(g).tobytes()
            for r, g in zip(jax.tree_util.tree_leaves(ref), jax.tree_util.tree_leaves(got_jt)))
print(f"tapped(jit(f)): bitwise={ok_j}, events={n1}")
print(f"jit(tapped(f)): bitwise={ok_jt}, events={n2}")
print("ALL CHECKS PASSED" if ok_j and ok_jt and n1 == n2 == 16 else "FAILURES ABOVE")
