# Copyright 2026- The jax-tap Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
import numpy as np

import jaxtap as tap

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
    ggot = jax.grad(
        lambda c: tap.verbose(f_cond, on_step=lambda e: None)(jnp.float32(1.0), c)
    )(c0)
    jax.block_until_ready(ggot)
    assert _bw(gref, ggot), (
        f"grad(cond) not bitwise identical: ref={float(gref):.6f} got={float(ggot):.6f}"
    )


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
    ggot = jax.grad(
        lambda c: tap.verbose(f_switch, on_step=lambda e: None)(jnp.int32(2), c)
    )(c0)
    jax.block_until_ready(ggot)
    assert _bw(gref, ggot), (
        f"grad(switch) not bitwise identical: ref={float(gref):.6f} got={float(ggot):.6f}"
    )


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
    assert _bw(gref, ggot), (
        f"grad(jit-siblings) not bitwise identical: ref={float(gref):.6f} got={float(ggot):.6f}"
    )


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
    assert _bw(gref, ggot), (
        f"grad(cond-in-scan) not bitwise identical: ref={float(gref):.6f} got={float(ggot):.6f}"
    )


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
    got = tap.verbose(
        _jit_nested, on_step=events.append, where=lambda p: p.startswith("jit")
    )(c0)
    jax.block_until_ready(got)

    assert _bw(ref, got), "where filter across jit boundary must not perturb output"
    paths = sorted({e.path for e in events})
    assert paths == ["jit[1]/scan[0]"], (
        f"where(jit) should select only jit[1]/scan[0], got {paths}"
    )


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
    assert _bw(g2ref, g2got), (
        f"grad2(cond) not bitwise identical: ref={float(g2ref):.6f} got={float(g2got):.6f}"
    )


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
    assert _bw(gref, ggot), (
        f"deep nesting grad not bitwise: ref={float(gref):.6f} got={float(ggot):.6f}"
    )


# ---------------------------------------------------------------------------
# bcore arm-a/grad2_hessian.py — higher-order autodiff
# ---------------------------------------------------------------------------


def test_higher_order_autodiff():
    """grad^2, grad^3, and Hessian through verbose(scan) are bitwise identical.

    Ports proofs/bcore-review/arm-a/grad2_hessian.py.
    """
    xs3 = jnp.arange(1.0, 4.0, dtype=jnp.float32)

    def scan_f(theta):
        final, _ = jax.lax.scan(
            lambda c, x: (c * jnp.sin(c) + theta * x, c), theta, xs3
        )
        return final

    theta = jnp.float32(0.7)
    v = tap.verbose(scan_f, on_step=lambda e: None)

    g_ref = jax.grad(scan_f)(theta)
    g_got = jax.grad(v)(theta)
    jax.block_until_ready(g_got)
    assert _bw(g_ref, g_got), (
        f"grad not bitwise: ref={float(g_ref):.6f} got={float(g_got):.6f}"
    )

    g2_ref = jax.grad(jax.grad(scan_f))(theta)
    g2_got = jax.grad(jax.grad(v))(theta)
    jax.block_until_ready(g2_got)
    assert _bw(g2_ref, g2_got), (
        f"grad^2 not bitwise: ref={float(g2_ref):.6f} got={float(g2_got):.6f}"
    )

    g3_ref = jax.grad(jax.grad(jax.grad(scan_f)))(theta)
    g3_got = jax.grad(jax.grad(jax.grad(v)))(theta)
    jax.block_until_ready(g3_got)
    assert _bw(g3_ref, g3_got), (
        f"grad^3 not bitwise: ref={float(g3_ref):.6f} got={float(g3_got):.6f}"
    )


def test_hessian_through_verbose():
    """Hessian of a vector-input scan is bitwise identical through verbose().

    Ports proofs/bcore-review/arm-a/grad2_hessian.py Hessian case.
    """
    xs2 = jnp.arange(1.0, 4.0, dtype=jnp.float32)

    def scan_vec(p):
        final, _ = jax.lax.scan(
            lambda c, x: (c * c + p[0] * x + p[1], c), p[0] + p[1], xs2
        )
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
    assert len(chol_events) == LANES * N_CHOL, (
        f"vmap prim-tap: expected {LANES * N_CHOL} events, got {len(chol_events)}"
    )


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
    assert chol_steps == list(range(5)), (
        f"while prim-tap: expected steps 0..4, got {chol_steps}"
    )


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


# ---------------------------------------------------------------------------
# A1 mitigation: vmap(while_loop) ghost-event suppression
#
# Ports proofs/bcore-review/arm-a/vmap_while.py and
# proofs/bcore-review/arm-a/vmap_while_hardened.py.
#
# Under vmap+while_loop, JAX runs max(trip_counts) joint iterations for all
# lanes.  Before A1 mitigation, debug.callback fired for ghost iterations
# (lanes already finished), delivering fabricated carry values to the host.
# After mitigation: carry taps emit only for active lanes (cond was True on
# the pre-body carry); ghost iterations are silently dropped before TapEvent
# construction.  Prim taps inside the body still ghost-fire — see
# test_vmap_while_prim_tap_residual_ghost below.
# ---------------------------------------------------------------------------

# Shared setup: 3 lanes, trip counts 10, 5, 1 → 16 real carry-tap events.
# max(10, 5, 1) = 10 joint iterations × 3 lanes = 30 raw callback fires.
# Mitigation should deliver exactly 16 to on_step.
_LIM = jnp.float32(10.0)
_V0 = jnp.array([0.0, 5.0, 9.0], dtype=jnp.float32)
_EXPECTED_REAL = 10 + 5 + 1  # 16


def _vmap_while_f(v0):
    """A single-scalar while_loop; when vmapped gives per-lane trip counts."""
    return jax.lax.while_loop(lambda c: c < _LIM, lambda c: c + jnp.float32(1.0), v0)


def test_vmap_while_carry_ghost_suppression():
    """After A1 mitigation, vmap+while carry taps fire exactly once per REAL step.

    3 lanes with trip counts 10, 5, 1 → exactly 16 carry-tap events, not 30.
    Output is bitwise identical to the untapped reference.

    Ports proofs/bcore-review/arm-a/vmap_while.py.
    """
    ref = jax.vmap(_vmap_while_f)(_V0)

    events: list = []
    got = jax.vmap(tap.verbose(_vmap_while_f, on_step=events.append))(_V0)
    jax.block_until_ready(got)

    assert _bw(ref, got), "vmap+while A1: output not bitwise identical"

    while_events = [e for e in events if e.path == "while[0]"]
    assert len(while_events) == _EXPECTED_REAL, (
        f"A1 ghost suppression: expected {_EXPECTED_REAL} carry-tap events, "
        f"got {len(while_events)} (30 = 10 joint iters × 3 lanes before mitigation)"
    )


def test_vmap_while_carry_no_fabricated_values():
    """After A1 mitigation, no impossible (fabricated) carry values reach on_step.

    Lane 0 counts 0→10, lane 1 counts 5→10, lane 2 counts 9→10.
    Any counter value > 10.0 is impossible in the real per-lane computation.

    Ports proofs/bcore-review/arm-a/vmap_while_hardened.py.
    """
    events: list = []
    got = jax.vmap(tap.verbose(_vmap_while_f, on_step=events.append))(_V0)
    jax.block_until_ready(got)

    while_events = [e for e in events if e.path == "while[0]"]
    counter_vals = [float(np.asarray(e.value[0])) for e in while_events]
    fabricated = [v for v in counter_vals if v > float(_LIM)]
    assert not fabricated, (
        f"A1 ghost suppression: fabricated counter values > LIM delivered: {fabricated}"
    )


def test_vmap_while_carry_ghost_suppression_with_select():
    """A1 mitigation works through the select= path.

    Applies select=lambda leaves: leaves[0] so the host receives the counter
    directly.  Ghost suppression must apply in the select branch too.
    """
    ref = jax.vmap(_vmap_while_f)(_V0)

    events: list = []
    got = jax.vmap(
        tap.verbose(
            _vmap_while_f, on_step=events.append, select=lambda leaves: leaves[0]
        )
    )(_V0)
    jax.block_until_ready(got)

    assert _bw(ref, got), "vmap+while A1 (select): output not bitwise identical"
    while_events = [e for e in events if e.path == "while[0]"]
    assert len(while_events) == _EXPECTED_REAL, (
        f"A1 ghost suppression (select): expected {_EXPECTED_REAL} events, "
        f"got {len(while_events)}"
    )
    fabricated = [
        float(np.asarray(e.value))
        for e in while_events
        if float(np.asarray(e.value)) > float(_LIM)
    ]
    assert not fabricated, f"A1 (select): fabricated values: {fabricated}"


def test_vmap_while_alert_no_ghost_alerts():
    """A1 ghost drop must happen BEFORE alert evaluation.

    An alert that triggers only on ghost-only values (counter > LIM) must
    fire zero times after the mitigation.  Before the fix it would fire once
    per ghost event (false alarms on stale carry).

    Rider #3 regression: ghost events must never reach alert= or on_step.
    """
    import io
    import sys

    events: list = []
    alert_fires: list = []

    def alert_fn(event):
        val = float(np.asarray(event.value[0]))
        if val > float(_LIM):
            alert_fires.append(val)
            return f"ghost! val={val}"
        return False

    stderr_buf = io.StringIO()
    old_stderr = sys.stderr
    sys.stderr = stderr_buf
    try:
        got = jax.vmap(
            tap.verbose(_vmap_while_f, on_step=events.append, alert=alert_fn)
        )(_V0)
        jax.block_until_ready(got)
    finally:
        sys.stderr = old_stderr

    fail_lines = [ln for ln in stderr_buf.getvalue().splitlines() if "FAIL" in ln]

    assert not alert_fires, (
        f"A1: alert fired on ghost values (should be 0 fires): {alert_fires}"
    )
    assert not fail_lines, (
        f"A1: ghost FAIL lines emitted to stderr (should be 0): {fail_lines}"
    )
    while_events = [e for e in events if e.path == "while[0]"]
    assert len(while_events) == _EXPECTED_REAL, (
        f"A1 alert test: expected {_EXPECTED_REAL} on_step events, got {len(while_events)}"
    )


def test_vmap_while_carry_sample_every_no_corruption():
    """sign-encode doesn't corrupt sample_every gate or leak ghost values.

    Two lanes: start values [0.0, 8.0], LIM=10.0.
      - Lane 0: 10 real iterations (joint steps 0-9).
      - Lane 1: 2 real iterations (joint steps 0-1), then ghost from step 2.
    joint while_loop runs max(10, 2) = 10 joint steps.

    Code invariant: the sample_every gate ``lax.cond(step % k == 0, _uncapped, noop, step)``
    operates on ``step`` (the real joint counter, always >=0).  Sign-encoding of the
    active mask happens INSIDE ``_uncapped``, AFTER the gate has already fired — so the
    gate arithmetic cannot be corrupted by sign-encode.

    Known boundary: under jax.vmap + while_loop, JAX's batching rule broadcasts the
    scalar joint step counter to per-lane shape (B,) in the batched carry.  This makes
    ``step % k == 0`` a per-lane bool, triggering lax.cond's "evaluate both branches"
    behaviour — the debug.callback fires for all real iterations regardless of sample_every.
    This is a pre-existing JAX vmap+lax.cond+effects limitation, NOT introduced by sign-encode.

    What this test freezes:
      1. All delivered step values are non-negative (sign-encode never leaks negative raw).
      2. No ghost values are delivered (sign-encode drops all ghost lanes host-side).
      3. The event count equals the total real iterations (10 + 2 = 12) — sample_every
         does not suppress under vmap+while (see boundary above).
      4. Output is bitwise-identical.
    """
    _v0_se = jnp.array([0.0, 8.0], dtype=jnp.float32)
    _lim_se = jnp.float32(10.0)

    def _f(v0):
        return jax.lax.while_loop(
            lambda c: c < _lim_se,
            lambda c: c + jnp.float32(1.0),
            v0,
        )

    events: list = []
    got = jax.vmap(tap.verbose(_f, on_step=events.append, sample_every=2))(_v0_se)
    jax.block_until_ready(got)

    while_events = [e for e in events if e.path == "while[0]"]
    steps = [e.step for e in while_events]
    carry_vals = [float(e.value[0]) for e in while_events]

    # 1. All delivered steps non-negative: sign-encode never leaks -(step+1) to host.
    bad_steps = [s for s in steps if s < 0]
    assert not bad_steps, (
        f"sign-encode corruption: negative raw step values reached host: {bad_steps}"
    )
    # 2. No ghost values: carry must not exceed LIM (ghosts produce carry > LIM).
    max_carry = max(carry_vals)
    assert max_carry <= float(_lim_se), (
        f"ghost leak: max carry {max_carry} > {float(_lim_se)}"
    )
    # 3. Event count == 12 (all real iterations; sample_every doesn't suppress under
    #    vmap+while due to lax.cond+effects boundary — see docstring).
    assert len(while_events) == 12, (
        f"expected 12 real events (10+2 lanes), got {len(while_events)}"
    )
    # 4. Bitwise-identical output.
    np.testing.assert_array_equal(got, np.array([10.0, 10.0]))


def test_vmap_while_prim_tap_residual_ghost():
    """Primitive taps inside a vmapped while body STILL ghost-fire (known boundary).

    The A1 mitigation applies to CARRY TAPS only.  Primitive taps inside the
    while body do not receive the active mask and therefore still fire for
    ghost iterations under vmap.  This test documents and freezes the current
    known behaviour (documented boundary — extend to prim taps in a future arc).

    With 3 lanes (trip counts 10, 5, 1) and a prim tap on the 'add' inside the
    body, the raw while_loop fires 10 joint iterations × 3 lanes = 30 prim-tap
    events.  Carry taps are still correctly limited to 16.
    """
    events: list = []
    got = jax.vmap(
        tap.verbose(
            _vmap_while_f,
            on_step=events.append,
            taps=[tap.on("add")],
        )
    )(_V0)
    jax.block_until_ready(got)

    carry_events = [e for e in events if e.path == "while[0]"]
    prim_events = [e for e in events if "add" in e.path]

    assert len(carry_events) == _EXPECTED_REAL, (
        f"carry taps: expected {_EXPECTED_REAL}, got {len(carry_events)}"
    )
    # Prim taps still ghost-fire: the count should be > _EXPECTED_REAL.
    # Exact count depends on JAX internals; we just assert it is MORE than 16.
    assert len(prim_events) > _EXPECTED_REAL, (
        f"prim taps expected to ghost-fire (>16 events) but got {len(prim_events)}; "
        "this test documents the known A1 residual boundary for prim taps"
    )


# ---------------------------------------------------------------------------
# Fix/vmap-while-crash: regression tests for GitHub issue #5
#
# Both bugs triggered in the B-form: verbose(vmap(f)) where f contains a
# while_loop.  make_jaxpr(vmap(f)) produces a while primitive with cond_jaxpr
# returning bool[n] (JAX's vmap batching rule stores the unreduced predicate).
# rewrite_while crashed because it passed this batched bool to lax.select
# (which requires scalar 'which').
#
# Fix: detect the batched cond (cond_jaxpr.outvars[0].aval.ndim > 0) and bind
# the while opaquely, preserving bitwise-identical outputs without crashing.
#
# Consumer note: outer SCAN carry taps fire normally with batched carry — this
# is what tuningfork needs for NUTS diagnostics (treedepth lives in the scan,
# not the inner while).  The inner while taps are suppressed (opaque bind);
# per-lane while telemetry is a future arc.
# ---------------------------------------------------------------------------


def test_bform_vmap_while_no_crash_and_bitwise():
    """B-form verbose(vmap(while)) must not crash and output must be bitwise identical.

    Regression for GitHub issue #5 Bug 2: tap.record() / verbose() on a
    vmapped while_loop crashed with 'TypeError: select which must be scalar'.

    After fix: no crash; 0 while carry taps (opaque bind); output identical.
    This is the MINIMAL repro from the issue report.

    Tests both integer and float carry: float catch rounding differences that
    integer equality would mask.  Uses a float while that actually runs
    (cond: c > 0.1, body: c * 0.5) so the output is non-trivial.
    """

    def f_int(s):
        def cond(c):
            return c < 5

        def body(c):
            return c + 1

        return jax.lax.while_loop(cond, body, s)

    vmapped_int = jax.vmap(f_int)
    init_int = jnp.zeros(4, dtype=jnp.int32)
    ref_int = vmapped_int(init_int)
    events_int: list = []
    got_int = tap.verbose(vmapped_int, on_step=events_int.append)(init_int)
    jax.block_until_ready(got_int)

    assert np.asarray(ref_int).tobytes() == np.asarray(got_int).tobytes(), (
        "B-form vmap×while (int): output not bitwise identical"
    )
    assert len([e for e in events_int if "while" in e.path]) == 0, (
        "B-form vmap×while (int): expected 0 while carry taps (opaque bind)"
    )

    # Float carry: different init values per lane, while runs non-trivially.
    # Any spurious computation inside the opaque-bound while would change bits.
    def f_float(s):
        def cond(c):
            return c > 0.1

        def body(c):
            return c * 0.5

        return jax.lax.while_loop(cond, body, s)

    vmapped_float = jax.vmap(f_float)
    init_float = jnp.array([2.0, 4.0, 0.5, 8.0], dtype=jnp.float32)
    ref_float = vmapped_float(init_float)
    events_float: list = []
    got_float = tap.verbose(vmapped_float, on_step=events_float.append)(init_float)
    jax.block_until_ready(got_float)

    assert np.asarray(ref_float).tobytes() == np.asarray(got_float).tobytes(), (
        f"B-form vmap×while (float): output not bitwise identical; "
        f"ref={np.asarray(ref_float).tolist()} got={np.asarray(got_float).tolist()}"
    )
    assert len([e for e in events_float if "while" in e.path]) == 0, (
        "B-form vmap×while (float): expected 0 while carry taps (opaque bind)"
    )


def test_bform_vmap_scan_while_no_crash_and_scan_taps():
    """B-form verbose(vmap(scan(while))) must not crash; scan taps must fire.

    Regression for GitHub issue #5 Bug 1: verbose(vmap_f) where vmap_f contains
    scan(while) crashed with the same 'select which' error when _interp descended
    into the batched scan body and encountered the batched while.

    After fix: no crash; scan carry taps fire with batched carry values; inner
    while taps are suppressed (opaque bind); output bitwise identical.

    This is the NUTS-shaped consumer test: NUTS/HMC uses vmap(scan(while)) —
    the leapfrog integrator is a while_loop nested in a trajectory scan.  Scan-
    level taps deliver the high-value diagnostics (treedepth in the carry).
    """
    scan_length = 3

    def outer(s):
        def body(state, _):
            def inner_body(c):
                return c + 1

            def inner_cond(c):
                return c < 4

            return jax.lax.while_loop(inner_cond, inner_body, state), None

        return jax.lax.scan(body, s, None, length=scan_length)

    vmapped_outer = jax.vmap(outer)
    init = jnp.zeros(4, dtype=jnp.int32)

    ref = vmapped_outer(init)
    events: list = []
    got = tap.verbose(vmapped_outer, on_step=events.append)(init)
    jax.block_until_ready(got)

    np.testing.assert_array_equal(
        got[0], ref[0], err_msg="B-form vmap×scan(while) output not bitwise identical"
    )
    scan_events = [e for e in events if "scan" in e.path]
    while_events = [e for e in events if "while" in e.path]

    # Outer scan emits one carry tap per step, batched carry values.
    assert len(scan_events) == scan_length, (
        f"Expected {scan_length} scan carry taps (one per step), got {len(scan_events)}"
    )
    # Batched carry: each event's value should be an array of shape (4,).
    for ev in scan_events:
        carry_val = ev.value[0]
        assert hasattr(carry_val, "shape") and carry_val.shape == (4,), (
            f"Scan carry tap value should be batched shape (4,), got {carry_val!r}"
        )
    # Inner while is opaque-bound: no while carry taps.
    assert len(while_events) == 0, (
        f"B-form vmap×scan(while): expected 0 while taps (opaque bind), got {len(while_events)}"
    )


def test_bform_nested_vmap_while_no_crash():
    """B-form verbose(vmap(vmap(while))) detects ndim=2 batched cond and binds opaquely.

    The cond detection predicate checks ndim > 0, not ndim == 1, so nested
    vmap (producing bool[n, m] cond output) is also handled correctly.
    """

    def f(s):
        def cond(c):
            return c < 3

        def body(c):
            return c + 1

        return jax.lax.while_loop(cond, body, s)

    nested_vmap_f = jax.vmap(jax.vmap(f))
    init = jnp.zeros((3, 4), dtype=jnp.int32)

    ref = nested_vmap_f(init)
    events: list = []
    got = tap.verbose(nested_vmap_f, on_step=events.append)(init)
    jax.block_until_ready(got)

    np.testing.assert_array_equal(
        got, ref, err_msg="B-form vmap(vmap(while)) output not bitwise identical"
    )
    # Verify the jaxpr's cond has ndim=2 (confirms nested-vmap detection)
    closed = jax.make_jaxpr(nested_vmap_f)(init)
    cond_aval = closed.jaxpr.eqns[0].params["cond_jaxpr"].jaxpr.outvars[0].aval
    assert cond_aval.ndim == 2, (
        f"Expected nested-vmap cond to have ndim=2, got ndim={cond_aval.ndim}"
    )
    while_events = [e for e in events if "while" in e.path]
    assert len(while_events) == 0, (
        f"B-form nested-vmap×while: expected 0 while taps, got {len(while_events)}"
    )


def test_bform_vmap_while_opaque_bind_blast_radius_prim_tap():
    """Opaque bind suppresses ALL taps inside the vmap-batched while subtree.

    The opaque bind doesn't just suppress while carry taps — it suppresses the
    entire B-form descent into the while's body and cond jaxprs.  A prim-tap
    on 'add' (the primitive inside the while body) fires in a normal (non-vmap)
    while but fires 0 times inside a vmap-batched while.

    Control: tap.on('add') fires in a plain (scalar) while.
    Subject: tap.on('add') is silent when the while is vmap-batched.
    """

    def f(s):
        def cond(c):
            return c < 5

        def body(c):
            return c + 1

        return jax.lax.while_loop(cond, body, s)

    # Control: add prim fires in the non-vmap while (proves the API works).
    events_scalar: list = []
    tap.verbose(f, on_step=events_scalar.append, taps=[tap.on("add")])(jnp.int32(0))
    assert len(events_scalar) > 0, (
        "Control: tap.on('add') should fire in a plain while_loop"
    )

    # Subject: same function, but vmapped — add prim is inside the opaque-bound subtree.
    vmapped_f = jax.vmap(f)
    init = jnp.zeros(4, dtype=jnp.int32)
    ref = vmapped_f(init)
    events_vmap: list = []
    got = tap.verbose(vmapped_f, on_step=events_vmap.append, taps=[tap.on("add")])(init)
    jax.block_until_ready(got)

    assert np.asarray(ref).tobytes() == np.asarray(got).tobytes(), (
        "Blast radius test: output must be bitwise identical"
    )
    assert len(events_vmap) == 0, (
        f"Opaque bind blast radius: expected 0 events (all taps in vmap-batched "
        f"while subtree suppressed), got {len(events_vmap)}"
    )


def test_bform_vmap_while_opaque_bind_blast_radius_nested_scan():
    """Opaque bind blast radius: a scan nested INSIDE a vmapped while body is suppressed.

    A scan inside a normal while body would produce carry-tap events.  When
    the enclosing while is vmap-batched, the entire subtree — including the
    inner scan — is opaquely bound and emits no events.

    Note: the test function f(while(scan)) nests scan INSIDE while; compare
    with test_bform_vmap_scan_while_no_crash_and_scan_taps which nests while
    INSIDE scan.  The suppression direction matters: the opaque-bound node is
    always the outermost while with a vmap-batched cond.
    """

    def f(s):
        def cond(c):
            return c < 6

        def body(c):
            # scan inside the while body — would fire events in a normal while
            result, _ = jax.lax.scan(lambda a, _: (a + 1, None), c, None, length=2)
            return result

        return jax.lax.while_loop(cond, body, s)

    vmapped_f = jax.vmap(f)
    init = jnp.zeros(4, dtype=jnp.int32)
    ref = vmapped_f(init)
    events: list = []
    got = tap.verbose(vmapped_f, on_step=events.append)(init)
    jax.block_until_ready(got)

    assert np.asarray(ref).tobytes() == np.asarray(got).tobytes(), (
        "Blast radius (nested scan): output must be bitwise identical"
    )
    scan_in_while = [e for e in events if "scan" in e.path]
    while_ev = [e for e in events if "while" in e.path]
    assert len(scan_in_while) == 0, (
        f"Blast radius: nested scan inside vmap-batched while emitted {len(scan_in_while)} events"
    )
    assert len(while_ev) == 0, (
        f"Blast radius: vmap-batched while emitted {len(while_ev)} carry events"
    )


def test_aform_vmap_scan_while_consumer_path():
    """A-form `with tap.record():` around jax.vmap(scan(while)) — NOT pre-jitted.

    This is the primary consumer path for tuningfork: the vmapped HMC/NUTS
    sampler is called INSIDE a record() context before any jit-cache is
    populated.  The A-form intercepts at the scalar-in-vmap level, so scan
    AND while taps both fire per-lane (4 lanes × 3 steps = 12 scan events;
    4 iterations × 4 lanes = 16 while events at scan-step 0).  Output is
    bitwise identical.

    Pre-jit caveat: if the vmapped function is called with jax.jit() OUTSIDE
    a record() context first, the jit-compiled artifact has no callbacks and
    subsequent calls inside record() emit 0 events.  Consumers must ensure the
    first compilation happens inside a record() context — or use the B-form
    instead (tap.verbose(jax.vmap(f))), which taps at the batched-jaxpr level
    and always fires scan carry taps regardless of prior jit compilation.
    """

    scan_length = 3
    batch_size = 4

    def outer(s):
        def body(state, _):
            def inner_body(c):
                return c + 1

            def inner_cond(c):
                return c < 4

            return jax.lax.while_loop(inner_cond, inner_body, state), None

        return jax.lax.scan(body, s, None, length=scan_length)

    vmapped_outer = jax.vmap(outer)
    init = jnp.zeros(batch_size, dtype=jnp.int32)
    ref = vmapped_outer(init)

    events_aform: list = []
    with tap.record(on_step=events_aform.append):
        got = vmapped_outer(init)  # first call: traced with A-form active
    jax.block_until_ready(got)

    # Output must be bitwise identical.
    assert np.asarray(ref[0]).tobytes() == np.asarray(got[0]).tobytes(), (
        f"A-form vmap×scan(while): output not bitwise identical; "
        f"ref={np.asarray(ref[0]).tolist()} got={np.asarray(got[0]).tolist()}"
    )

    # A-form fires per-lane (scalar intercept inside vmap trace):
    #   scan: scan_length × batch_size = 3×4 = 12 events, each scalar carry
    #   while: batch_size × (while iterations at step 0) = 4×4 = 16 events
    scan_events = [e for e in events_aform if e.path == "scan[0]"]
    while_events = [e for e in events_aform if "while" in e.path]
    assert len(scan_events) == scan_length * batch_size, (
        f"A-form: expected {scan_length * batch_size} per-lane scan events, "
        f"got {len(scan_events)}"
    )
    # Scan carry values are scalars (per-lane), not batched arrays.
    for ev in scan_events:
        carry = ev.value[0]
        assert not hasattr(carry, "shape") or carry.shape == (), (
            f"A-form scan carry tap should be scalar (per-lane), got shape {carry.shape}"
        )
    # While taps fire per-lane during scan step 0 (all 4 lanes iterate 4 times:
    # c in {0,1,2,3} → 4 while-carry events × 4 lanes = 16).  Steps 1 and 2
    # have carry=4 so the while exits immediately (0 events).
    expected_while = 4 * batch_size  # 4 iterations × 4 lanes at step 0
    assert len(while_events) == expected_while, (
        f"A-form: expected {expected_while} per-lane while events, got {len(while_events)}"
    )


# ---------------------------------------------------------------------------
# AYS-2: predicate false-positive / false-negative boundary tests
# ---------------------------------------------------------------------------


def test_predicate_false_positive_vector_carry_scalar_cond():
    """FALSE-POSITIVE guard: non-vmap while with vector carry + scalar cond taps fire.

    The detection predicate keys on `cond_jaxpr.outvars[0].aval.ndim > 0` —
    the COND OUTPUT shape, NOT the carry shape.  A while with vector carry
    (e.g. shape (8,)) and a scalar reducing cond (jnp.linalg.norm(carry) < tol)
    has ndim=0 → predicate False → rewrite_while runs → A1 carry-taps fire.

    If the predicate mistakenly keyed on carry shape, these taps would be
    wrongly suppressed (opaque bind) — this test would fail.

    Also tests `c[0] < 5` (index into vector carry → scalar output) to confirm
    the boundary: it's always the cond OUTPUT that matters.
    """

    # Case A: vector carry, norm-based scalar cond.
    def f_norm(c):
        def cond(carry):
            return jnp.linalg.norm(carry) < 5.0

        def body(carry):
            return carry + 0.5

        return jax.lax.while_loop(cond, body, c)

    init_norm = jnp.zeros(8, dtype=jnp.float32)
    ref_norm = f_norm(init_norm)
    events_norm: list = []
    got_norm = tap.verbose(f_norm, on_step=events_norm.append)(init_norm)
    jax.block_until_ready(got_norm)

    assert np.asarray(ref_norm).tobytes() == np.asarray(got_norm).tobytes(), (
        "vector-carry norm-cond: output not bitwise identical"
    )
    while_ev_norm = [e for e in events_norm if "while" in e.path]
    assert len(while_ev_norm) > 0, (
        "FALSE-POSITIVE: vector-carry norm-cond while suppressed (predicate keyed on carry?)"
    )
    # Carry events should have batched shape (8,) — the full carry, not a scalar.
    for ev in while_ev_norm:
        assert ev.value[0].shape == (8,), (
            f"norm-cond carry tap shape should be (8,), got {ev.value[0].shape}"
        )

    # Case B: vector carry, index-into-carry scalar cond.
    def f_index(c):
        def cond(carry):
            return carry[0] < 5.0

        def body(carry):
            return carry + 1.0

        return jax.lax.while_loop(cond, body, c)

    init_idx = jnp.zeros(4, dtype=jnp.float32)
    ref_idx = f_index(init_idx)
    events_idx: list = []
    got_idx = tap.verbose(f_index, on_step=events_idx.append)(init_idx)
    jax.block_until_ready(got_idx)

    assert np.asarray(ref_idx).tobytes() == np.asarray(got_idx).tobytes(), (
        "vector-carry index-cond: output not bitwise identical"
    )
    while_ev_idx = [e for e in events_idx if "while" in e.path]
    assert len(while_ev_idx) > 0, (
        "FALSE-POSITIVE: vector-carry index-cond while suppressed"
    )

    # Verify ndim=0 for both conds (documents WHY these are not suppressed).
    def _cond_ndim(f, *args):
        closed = jax.make_jaxpr(f)(*args)
        eqns = [e for e in closed.jaxpr.eqns if str(e.primitive) == "while"]
        return eqns[0].params["cond_jaxpr"].jaxpr.outvars[0].aval.ndim

    assert _cond_ndim(f_norm, init_norm) == 0, (
        "norm-cond should have scalar output (ndim=0)"
    )
    assert _cond_ndim(f_index, init_idx) == 0, (
        "index-cond should have scalar output (ndim=0)"
    )


def test_predicate_false_negative_impossible():
    """FALSE-NEGATIVE hunt: a vmap-batched while with scalar cond routes to rewrite_while.

    The detection predicate `ndim > 0` is NECESSARY AND SUFFICIENT to identify
    cases where rewrite_while would crash:
    - rewrite_while crashes when `active = eval_jaxpr(cond_jaxpr)` returns bool[n],
      because `lax.select(bool[n], scalar_step, ...)` fails.
    - If cond_jaxpr output is scalar (ndim=0), `active` is scalar → lax.select
      succeeds → no crash.

    Therefore: there is no escape path.  A while with ndim=0 cond is either:
    (a) truly non-vmap-batched → rewrite_while always correct, or
    (b) vmap-batched-with-scalar-cond (cond reads only non-batched carry elements)
        → rewrite_while handles it correctly (scalar active, batched carry processed).

    This test proves case (b): vmap with in_axes=None on a shared counter,
    cond reads only that counter.  JAX does NOT batch the cond (cond_jaxpr output
    stays scalar, ndim=0).  The predicate routes to rewrite_while, which handles
    it correctly because `active` is scalar.  Output is bitwise identical and
    carry taps fire with batched carry values.
    """

    def f(x, count):
        """carry = (f32[4] batched across lanes, i32 scalar shared counter)."""

        def cond(carry):
            return carry[1] < 5  # reads scalar counter only

        def body(carry):
            return (carry[0] + 1.0, carry[1] + 1)

        return jax.lax.while_loop(cond, body, (x, count))

    vmapped_f = jax.vmap(f, in_axes=(0, None))
    init_x = jnp.zeros(4, dtype=jnp.float32)
    init_count = jnp.int32(0)

    # Verify the jaxpr: cond reads only the scalar counter → ndim=0.
    closed = jax.make_jaxpr(vmapped_f)(init_x, init_count)
    while_eqn = next(e for e in closed.jaxpr.eqns if str(e.primitive) == "while")
    cj = while_eqn.params["cond_jaxpr"]
    cond_ndim = cj.jaxpr.outvars[0].aval.ndim
    assert cond_ndim == 0, (
        f"Expected cond_jaxpr ndim=0 (scalar cond reads non-batched counter), "
        f"got ndim={cond_ndim}"
    )

    # Run through verbose(): predicate says False → rewrite_while path.
    ref = vmapped_f(init_x, init_count)
    events: list = []
    got = tap.verbose(vmapped_f, on_step=events.append)(init_x, init_count)
    jax.block_until_ready(got)

    # Output is bitwise identical.
    assert np.asarray(ref[0]).tobytes() == np.asarray(got[0]).tobytes(), (
        "false-negative test: output not bitwise identical"
    )

    # rewrite_while fired: carry taps appear (5 iterations × scalar active → no crash).
    while_events = [e for e in events if "while" in e.path]
    assert len(while_events) == 5, (
        f"Expected 5 while carry taps (5 iterations of scalar cond), "
        f"got {len(while_events)}"
    )
    # Carry values are batched f32[4] (x) paired with scalar i32 counter.
    # value[0] is the first leaf of the carry pytree, i.e. the float array.
    for ev in while_events:
        assert ev.value[0].shape == (4,), (
            f"Carry tap should show batched float (shape (4,)), got {ev.value[0].shape}"
        )
