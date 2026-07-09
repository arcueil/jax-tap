"""
B-core conformance tests — ports the checks from the bcore-review corpora
that are NOT yet covered by tests/test_jaxtap.py or tests/test_collectors.py.

Sources:
  proofs/bcore-review/fix-review/fixreview_bcore.py  (checks 5,6,8,9,14)
  proofs/bcore-review/fix-review/fixreview_round2.py (checks 1,2,4,5,6)
  proofs/bcore-review/arm-a/grad2_hessian.py
  proofs/bcore-review/arm-a/dtypes_degenerate.py
  proofs/m1a-ays/ays_m1a.py              (vmap prim-tap, cond prim-tap, while prim-tap)
  proofs/m1d-ays/ays_m1d.py              (cond-in-scan prim tap gated, while prim tap gated)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jaxtap as tap
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

XS5 = jnp.arange(5.0, dtype=jnp.float32)


def _bw(a, b) -> bool:
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return len(la) == len(lb) and all(
        np.asarray(x).tobytes() == np.asarray(y).tobytes() for x, y in zip(la, lb)
    )


def _collect(f, *args, **kw):
    events: list = []
    got = tap.verbose(f, on_step=events.append, **kw)(*args)
    jax.block_until_ready(got)
    return got, events


# Shared scan function used across multiple tests.
def _scanfn(c0):
    c, _ = jax.lax.scan(lambda c, x: (c * 1.01 + x, c), c0, XS5)
    return c


# ---------------------------------------------------------------------------
# fixreview_bcore.py — grad through higher-order primitives
# ---------------------------------------------------------------------------


def test_grad_cond_bitwise():
    """grad(verbose(cond-with-scan)) must be bitwise identical to grad(cond-with-scan).

    Ports fixreview_bcore.py check 5: grad(cond).
    """

    def f_cond(pred, c0):
        return jax.lax.cond(pred > 0, _scanfn, lambda z: _scanfn(z * 2.0), c0)

    c0 = jnp.float32(0.5)
    gref = jax.grad(lambda c: f_cond(jnp.float32(1.0), c))(c0)
    ggot = jax.grad(lambda c: tap.verbose(f_cond, on_step=lambda e: None)(jnp.float32(1.0), c))(c0)
    jax.block_until_ready(ggot)
    assert _bw(
        gref, ggot
    ), f"grad(cond) not bitwise identical: ref={float(gref):.6f} got={float(ggot):.6f}"


def test_grad_switch_bitwise():
    """grad(verbose(switch-with-scan)) must be bitwise identical.

    Ports fixreview_bcore.py check 6: grad(switch).
    """

    def f_switch(i, c0):
        return jax.lax.switch(
            i, [_scanfn, lambda z: _scanfn(z + 1.0), lambda z: _scanfn(z * 3.0)], c0
        )

    c0 = jnp.float32(0.5)
    gref = jax.grad(lambda c: f_switch(jnp.int32(2), c))(c0)
    ggot = jax.grad(lambda c: tap.verbose(f_switch, on_step=lambda e: None)(jnp.int32(2), c))(c0)
    jax.block_until_ready(ggot)
    assert _bw(
        gref, ggot
    ), f"grad(switch) not bitwise identical: ref={float(gref):.6f} got={float(ggot):.6f}"


def test_grad_jit_siblings_bitwise():
    """grad(verbose(jit-siblings)) must be bitwise identical.

    jit-siblings: top-level scan + jit-nested scan.
    Ports fixreview_bcore.py check 8: grad(jit-siblings).
    """

    def f_jit_siblings(c0):
        a = _scanfn(c0)
        b = jax.jit(_scanfn)(c0 + 1.0)
        return a + b

    c0 = jnp.float32(0.5)
    gref = jax.grad(f_jit_siblings)(c0)
    ggot = jax.grad(tap.verbose(f_jit_siblings, on_step=lambda e: None))(c0)
    jax.block_until_ready(ggot)
    assert _bw(
        gref, ggot
    ), f"grad(jit-siblings) not bitwise identical: ref={float(gref):.6f} got={float(ggot):.6f}"


def test_vmap_cond_bitwise():
    """vmap(verbose(cond-with-scan)) must be bitwise identical.

    Ports fixreview_bcore.py check 9: vmap(cond).
    """

    def f_cond(pred, c0):
        return jax.lax.cond(pred > 0, _scanfn, lambda z: _scanfn(z * 2.0), c0)

    pv = jnp.array([1.0, -1.0, 1.0], dtype=jnp.float32)
    cv = jnp.array([0.5, 0.7, 0.9], dtype=jnp.float32)

    ref = jax.vmap(f_cond)(pv, cv)
    got = jax.vmap(tap.verbose(f_cond, on_step=lambda e: None))(pv, cv)
    jax.block_until_ready(got)
    assert _bw(ref, got), "vmap(cond-with-scan) not bitwise identical"


def test_grad_cond_in_scan_bitwise():
    """grad(verbose(scan-body-with-cond)) must be bitwise identical.

    Ports fixreview_bcore.py check 14: grad(cond-in-scan).
    """

    def f_cond_in_scan(c0):
        def body(c, x):
            c2 = jax.lax.cond(x > 2.0, lambda z: z + 1.0, lambda z: z * 1.1, c)
            return c2, c2

        c, _ = jax.lax.scan(body, c0, XS5)
        return c

    c0 = jnp.float32(0.5)
    gref = jax.grad(f_cond_in_scan)(c0)
    ggot = jax.grad(tap.verbose(f_cond_in_scan, on_step=lambda e: None))(c0)
    jax.block_until_ready(ggot)
    assert _bw(
        gref, ggot
    ), f"grad(cond-in-scan) not bitwise identical: ref={float(gref):.6f} got={float(ggot):.6f}"


# ---------------------------------------------------------------------------
# fixreview_round2.py — max_depth / where across jit boundary + grad2
# ---------------------------------------------------------------------------


def _jit_nested(c0):
    """top-level scan[0] + jit-nested scan."""
    a = _scanfn(c0)
    b = jax.jit(_scanfn)(c0 + 1.0)
    return a + b


def test_max_depth_across_jit_boundary():
    """max_depth=0 keeps only the top-level scan[0] and drops jit[1]/scan[0].

    Ports fixreview_round2.py check 1.
    Note: existing test_max_depth_0 uses nested scan (no jit). This test covers
    the JIT-boundary variant where depth is introduced by a jit() wrapper.
    """
    c0 = jnp.float32(0.5)
    ref = _jit_nested(c0)
    events: list = []
    got = tap.verbose(_jit_nested, on_step=events.append, max_depth=0)(c0)
    jax.block_until_ready(got)

    assert _bw(ref, got), "max_depth=0 across jit boundary must not perturb output"
    paths = sorted({e.path for e in events})
    assert paths == ["scan[0]"], f"max_depth=0 should yield only scan[0], got {paths}"


def test_where_across_jit_boundary():
    """where=startswith('jit') selects only the jit-nested scan.

    Ports fixreview_round2.py check 2.
    """
    c0 = jnp.float32(0.5)
    ref = _jit_nested(c0)
    events: list = []
    got = tap.verbose(_jit_nested, on_step=events.append, where=lambda p: p.startswith("jit"))(c0)
    jax.block_until_ready(got)

    assert _bw(ref, got), "where filter across jit boundary must not perturb output"
    paths = sorted({e.path for e in events})
    assert paths == ["jit[1]/scan[0]"], f"where(jit) should select only jit[1]/scan[0], got {paths}"


def test_grad2_cond_bitwise():
    """grad(grad(verbose(cond-with-scan))) must be bitwise identical.

    Ports fixreview_round2.py check 4: grad2(cond).
    """

    def f_cond(c0):
        return jax.lax.cond(c0 > 0, _scanfn, lambda z: _scanfn(-z), c0)

    c0 = jnp.float32(0.5)
    g2ref = jax.grad(jax.grad(f_cond))(c0)
    g2got = jax.grad(jax.grad(tap.verbose(f_cond, on_step=lambda e: None)))(c0)
    jax.block_until_ready(g2got)
    assert _bw(
        g2ref, g2got
    ), f"grad2(cond) not bitwise identical: ref={float(g2ref):.6f} got={float(g2got):.6f}"


def test_deep_cond_in_jit_in_scan_bitwise():
    """cond inside jit inside scan: result bitwise identical under verbose().

    Ports fixreview_round2.py check 5.
    """

    def f_deep(c0):
        def body(c, x):
            inner = jax.jit(
                lambda z: jax.lax.cond(x > 2.0, lambda w: w + 1.0, lambda w: w * 1.1, z)
            )
            return inner(c), c

        c, _ = jax.lax.scan(body, c0, XS5)
        return c

    c0 = jnp.float32(0.5)
    ref = f_deep(c0)
    got, _ = _collect(f_deep, c0)
    assert _bw(ref, got), "deep cond-in-jit-in-scan not bitwise identical"


def test_deep_nesting_grad_bitwise():
    """grad through deep cond-in-jit-in-scan must be bitwise identical.

    Ports fixreview_round2.py check 6.
    """

    def f_deep(c0):
        def body(c, x):
            inner = jax.jit(
                lambda z: jax.lax.cond(x > 2.0, lambda w: w + 1.0, lambda w: w * 1.1, z)
            )
            return inner(c), c

        c, _ = jax.lax.scan(body, c0, XS5)
        return c

    c0 = jnp.float32(0.5)
    gref = jax.grad(f_deep)(c0)
    ggot = jax.grad(tap.verbose(f_deep, on_step=lambda e: None))(c0)
    jax.block_until_ready(ggot)
    assert _bw(
        gref, ggot
    ), f"deep nesting grad not bitwise: ref={float(gref):.6f} got={float(ggot):.6f}"


# ---------------------------------------------------------------------------
# bcore arm-a/grad2_hessian.py — higher-order autodiff
# ---------------------------------------------------------------------------


def test_higher_order_autodiff():
    """grad^2, grad^3, and Hessian through verbose(scan) are bitwise identical.

    Ports proofs/bcore-review/arm-a/grad2_hessian.py.
    """
    xs3 = jnp.arange(1.0, 4.0, dtype=jnp.float32)

    def scan_f(theta):
        final, _ = jax.lax.scan(lambda c, x: (c * jnp.sin(c) + theta * x, c), theta, xs3)
        return final

    theta = jnp.float32(0.7)
    v = tap.verbose(scan_f, on_step=lambda e: None)

    g_ref = jax.grad(scan_f)(theta)
    g_got = jax.grad(v)(theta)
    jax.block_until_ready(g_got)
    assert _bw(g_ref, g_got), f"grad not bitwise: ref={float(g_ref):.6f} got={float(g_got):.6f}"

    g2_ref = jax.grad(jax.grad(scan_f))(theta)
    g2_got = jax.grad(jax.grad(v))(theta)
    jax.block_until_ready(g2_got)
    assert _bw(
        g2_ref, g2_got
    ), f"grad^2 not bitwise: ref={float(g2_ref):.6f} got={float(g2_got):.6f}"

    g3_ref = jax.grad(jax.grad(jax.grad(scan_f)))(theta)
    g3_got = jax.grad(jax.grad(jax.grad(v)))(theta)
    jax.block_until_ready(g3_got)
    assert _bw(
        g3_ref, g3_got
    ), f"grad^3 not bitwise: ref={float(g3_ref):.6f} got={float(g3_got):.6f}"


def test_hessian_through_verbose():
    """Hessian of a vector-input scan is bitwise identical through verbose().

    Ports proofs/bcore-review/arm-a/grad2_hessian.py Hessian case.
    """
    xs2 = jnp.arange(1.0, 4.0, dtype=jnp.float32)

    def scan_vec(p):
        final, _ = jax.lax.scan(lambda c, x: (c * c + p[0] * x + p[1], c), p[0] + p[1], xs2)
        return final

    p = jnp.array([0.3, 0.5], dtype=jnp.float32)
    h_ref = jax.hessian(scan_vec)(p)
    h_got = jax.hessian(tap.verbose(scan_vec, on_step=lambda e: None))(p)
    jax.block_until_ready(h_got)
    assert _bw(h_ref, h_got), "Hessian through verbose(scan_vec) not bitwise identical"


# ---------------------------------------------------------------------------
# bcore arm-a/dtypes_degenerate.py — dtype coverage for verbose()
# ---------------------------------------------------------------------------


def test_int32_carry_bitwise():
    """int32 carry scan is bitwise identical through verbose().

    Ports proofs/bcore-review/arm-a/dtypes_degenerate.py int32 case.
    """

    def f(c0, xs):
        return jax.lax.scan(lambda c, x: (c + x, c * x), c0, xs)

    c0 = jnp.int32(1)
    xs = jnp.arange(5, dtype=jnp.int32)
    ref = f(c0, xs)
    got, events = _collect(f, c0, xs)
    assert _bw(ref, got), "int32 carry not bitwise identical through verbose()"
    assert len(events) == 5, f"expected 5 events, got {len(events)}"


def test_complex64_carry_bitwise():
    """complex64 carry scan is bitwise identical through verbose().

    Ports proofs/bcore-review/arm-a/dtypes_degenerate.py complex64 case.
    """

    def f(c0, xs):
        return jax.lax.scan(lambda c, x: (c * x + 1j, c), c0, xs)

    c0 = jnp.complex64(0.5 + 0.5j)
    xs = jnp.arange(4, dtype=jnp.complex64) + 1j
    ref = f(c0, xs)
    got, events = _collect(f, c0, xs)
    assert _bw(ref, got), "complex64 carry not bitwise identical through verbose()"
    assert len(events) == 4, f"expected 4 events, got {len(events)}"


def test_mixed_dtype_carry_bitwise():
    """Mixed int32+float32+bool carry scan is bitwise identical through verbose().

    Ports proofs/bcore-review/arm-a/dtypes_degenerate.py mixed carry case.
    """

    def body(c, x):
        i, f, b = c
        return (i + x.astype(jnp.int32), f + jnp.sin(f), jnp.logical_not(b)), f

    mc = (jnp.int32(0), jnp.float32(1.0), jnp.bool_(False))
    xs = jnp.arange(5.0, dtype=jnp.float32)

    def f(c0, xs_):
        return jax.lax.scan(body, c0, xs_)

    ref = f(mc, xs)
    got, events = _collect(f, mc, xs)
    assert _bw(ref, got), "mixed-dtype carry not bitwise identical through verbose()"
    assert len(events) == 5, f"expected 5 events, got {len(events)}"


# ---------------------------------------------------------------------------
# m1a-ays/ays_m1a.py — vmap prim-tap, cond prim-tap, while prim-tap
# ---------------------------------------------------------------------------

N_CHOL = 5
XS_CHOL = jnp.arange(float(N_CHOL), dtype=jnp.float32)


def _f_chol(c0):
    def body(c, x):
        M = jnp.eye(2, dtype=jnp.float32) * (c + x + 1.0)
        L = jnp.linalg.cholesky(M)
        return c + jnp.sum(L) * 0.01, c

    c, _ = jax.lax.scan(body, c0, XS_CHOL)
    return c


def test_vmap_prim_tap_fires_lanes_times_n():
    """vmap over verbose(f_chol) with prim tap fires LANES*N events.

    Ports proofs/m1a-ays/ays_m1a.py check 1 (vmap × primitive tap).
    """
    LANES = 3
    c0b = jnp.arange(1.0, 1.0 + LANES, dtype=jnp.float32)

    events: list = []
    gv = jax.vmap(
        tap.verbose(
            _f_chol,
            on_step=events.append,
            taps=[tap.on("cholesky", select=lambda o: jnp.sum(o[0]))],
        )
    )
    ref = jax.vmap(_f_chol)(c0b)
    got = gv(c0b)
    jax.block_until_ready(got)

    chol_events = [e for e in events if "cholesky" in e.path]
    assert _bw(ref, got), "vmap prim-tap result not bitwise identical"
    assert (
        len(chol_events) == LANES * N_CHOL
    ), f"vmap prim-tap: expected {LANES * N_CHOL} events, got {len(chol_events)}"


def test_cond_prim_tap_taken_branch_only():
    """Prim tap inside a cond branch fires only on steps where the branch is taken.

    Ports proofs/m1a-ays/ays_m1a.py check 2 (cond prim-tap).
    xs = 0..4; x > 2 at steps 3, 4 → cholesky fires at steps 3 and 4 only.
    """

    def f_cond(c0):
        def body(c, x):
            c2 = jax.lax.cond(
                x > 2.0,
                lambda z: jnp.sum(jnp.linalg.cholesky(jnp.eye(2) * (z + 1.0))) + z,
                lambda z: z * 1.1,
                c,
            )
            return c2, c2

        c, _ = jax.lax.scan(body, c0, XS_CHOL)
        return c

    events: list = []
    gc = tap.verbose(
        f_cond,
        on_step=events.append,
        taps=[tap.on("cholesky", select=lambda o: jnp.sum(o[0]))],
    )
    ref = f_cond(jnp.float32(0.5))
    got = gc(jnp.float32(0.5))
    jax.block_until_ready(got)

    assert _bw(ref, got), "cond prim-tap result not bitwise identical"
    chol_steps = sorted(e.step for e in events if "cholesky" in e.path)
    assert chol_steps == [
        3,
        4,
    ], f"cond prim-tap should fire at steps 3,4 (x>2); got steps={chol_steps}"


def test_while_prim_tap_live_steps():
    """Prim tap inside while_loop fires at live step indices 0..N-1.

    Ports proofs/m1a-ays/ays_m1a.py check 3 (while prim-tap).
    """

    def f_while(v0):
        def cond_fn(c):
            return c[0] < 5.0

        def body_fn(c):
            v, acc = c
            L = jnp.linalg.cholesky(jnp.eye(2) * (v + 1.0))
            return (v + 1.0, acc + jnp.sum(L))

        return jax.lax.while_loop(cond_fn, body_fn, (v0, jnp.float32(0.0)))

    events: list = []
    gw = tap.verbose(
        f_while,
        on_step=events.append,
        taps=[tap.on("cholesky", select=lambda o: jnp.sum(o[0]))],
    )
    ref = f_while(jnp.float32(0.0))
    got = gw(jnp.float32(0.0))
    jax.block_until_ready(got)

    assert _bw(ref, got), "while prim-tap result not bitwise identical"
    chol_steps = sorted(e.step for e in events if "cholesky" in e.path)
    assert chol_steps == list(range(5)), f"while prim-tap: expected steps 0..4, got {chol_steps}"


# ---------------------------------------------------------------------------
# m1d-ays/ays_m1d.py — sample_every gating of prim taps in cond and while
# ---------------------------------------------------------------------------


def test_cond_in_scan_prim_tap_gated():
    """Prim tap inside cond-in-scan is gated by sample_every.

    Ports proofs/m1d-ays/ays_m1d.py check 4.
    linspace(0,1,20): x>0.5 at steps 10..19 (10 steps); se=5 → steps 10,15 fire.
    """

    def fc(c0):
        def body(c, x):
            c1 = jax.lax.cond(x > 0.5, lambda z: jnp.sin(z) + z, lambda z: z, c)
            return c1, None

        c, _ = jax.lax.scan(body, c0, jnp.linspace(0.0, 1.0, 20))
        return c

    events: list = []
    gc = tap.verbose(fc, on_step=events.append, sample_every=5, taps=[tap.on("sin")])
    ref = fc(jnp.float32(0.3))
    got = gc(jnp.float32(0.3))
    jax.block_until_ready(got)

    sin_events = [e for e in events if "sin" in e.path]
    assert _bw(ref, got), "cond-in-scan prim tap gated: result not bitwise identical"
    assert len(sin_events) == 2, (
        f"expected 2 sin events (steps 10,15 with se=5); got {len(sin_events)}, "
        f"steps={sorted(e.step for e in sin_events)}"
    )


def test_while_prim_tap_gated():
    """Prim tap inside while_loop is gated by sample_every.

    Ports proofs/m1d-ays/ays_m1d.py check 5.
    while_loop runs 25 iterations (0..24); se=10 → fires at steps 0, 10, 20 = 3 events.
    """

    def fw(v0):
        def cond_fn(c):
            return c[0] < 25.0

        def body_fn(c):
            v, acc = c
            return (v + 1.0, acc + jnp.sin(v))

        return jax.lax.while_loop(cond_fn, body_fn, (v0, jnp.float32(0.0)))

    events: list = []
    gw = tap.verbose(fw, on_step=events.append, sample_every=10, taps=[tap.on("sin")])
    ref = fw(jnp.float32(0.0))
    got = gw(jnp.float32(0.0))
    jax.block_until_ready(got)

    sin_events = [e for e in events if "sin" in e.path]
    assert _bw(ref, got), "while prim-tap gated: result not bitwise identical"
    assert len(sin_events) == 3, (
        f"expected 3 sin events (steps 0,10,20 with se=10, 25 iters); "
        f"got {len(sin_events)}, steps={sorted(e.step for e in sin_events)}"
    )
