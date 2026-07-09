"""
M1d API-polish tests: sample_every gating for prim taps (FIX 1),
emission-only filters / descend-always (FIX 2), path-aware select (FIX 3).

Run with: uv run pytest tests/test_m1d_api_polish.py -v
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jaxtap as tap
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect(f, *args, **verbose_kwargs):
    events: list[tap.TapEvent] = []
    tapped = tap.verbose(f, on_step=lambda e: events.append(e), **verbose_kwargs)
    result = tapped(*args)
    jax.block_until_ready(result)
    return result, events


def _bytes(x):
    return [np.asarray(v).tobytes() for v in jax.tree_util.tree_leaves(x)]


def bitwise_eq(a, b) -> bool:
    return _bytes(a) == _bytes(b)


# ---------------------------------------------------------------------------
# Shared programs
# ---------------------------------------------------------------------------


def _sin_scan(x0, n):
    """Simple scan that calls sin in the body (non-jit-wrapped)."""

    def body(carry, _):
        v = jnp.sin(carry)
        return v, v

    return jax.lax.scan(body, x0, None, length=n)


def _scan_wrapping_while(targets):
    """Scan whose body contains a while_loop — the demo-03 archetype."""

    def inner_solve(m, z0):
        def cond(c):
            z, it = c
            return (jnp.abs(z - m) > 1e-2) & (it < 20)

        def body(c):
            z, it = c
            return (z - 0.5 * jnp.tanh(z - m), it + 1)

        z, it = jax.lax.while_loop(cond, body, (z0, 0))
        return z, it

    def step(carry, m):
        z, _ = carry
        z, it = inner_solve(m, z)
        return (z, it), it

    (_, _), it_trace = jax.lax.scan(step, (jnp.float32(0.0), 0), targets)
    return it_trace


# ---------------------------------------------------------------------------
# FIX 2 tests: emission-only filters, descend-always
# ---------------------------------------------------------------------------


class TestFix2EmissionOnly:
    """
    M1d FIX 2: ops/where/max_depth control EMISSION of carry taps only.
    The walker always descends into scan/while bodies so prim taps fire.
    """

    def test_prim_tap_inside_ops_empty_fires(self):
        """
        ops=() suppresses all carry taps but prim taps inside loops still fire.
        """
        N = 5

        def f(x):
            def body(carry, _):
                return jnp.sin(carry) + 0.1, carry

            return jax.lax.scan(body, x, None, length=N)

        x = jnp.float32(1.0)
        ref = f(x)
        events: list[tap.TapEvent] = []
        got = tap.verbose(
            f,
            on_step=lambda e: events.append(e),
            taps=[tap.on("sin")],
            ops=(),  # no carry taps at all
        )(x)
        jax.block_until_ready(got)

        assert bitwise_eq(ref, got), "ops=() result not bitwise identical"

        # No carry tap events (ops=() suppresses them)
        carry_events = [e for e in events if "sin" not in e.path]
        assert len(carry_events) == 0, f"expected 0 carry events, got {len(carry_events)}"

        # Prim taps FIRE inside the scan body even though ops=()
        sin_events = [e for e in events if "sin" in e.path]
        assert len(sin_events) == N, (
            f"expected {N} sin events inside ops=() scan, got {len(sin_events)}"
        )

    def test_prim_tap_inside_where_filtered_loop_fires(self):
        """
        where=lambda p: 'scan[1]' in p filters scan[0] carry taps,
        but prim taps inside scan[0] still fire.
        """
        N = 4

        def f(x):
            def body_chol(carry, _):
                M = jnp.array([[1.0, 0.5], [0.5, 1.0]])
                L = jnp.linalg.cholesky(M)
                return carry + L[0, 0], L[0, 0]

            def body_simple(carry, _):
                return carry + 1.0, carry

            out1, _ = jax.lax.scan(body_chol, x, None, length=N)
            out2, _ = jax.lax.scan(body_simple, out1, None, length=N)
            return out2

        x = jnp.float32(0.0)
        ref = f(x)
        events: list[tap.TapEvent] = []
        got = tap.verbose(
            f,
            on_step=lambda e: events.append(e),
            taps=[tap.on("cholesky")],
            where=lambda p: "scan[1]" in p,
        )(x)
        jax.block_until_ready(got)

        assert bitwise_eq(ref, got), "where-filtered result not bitwise identical"

        # scan[0] carry taps suppressed
        scan0_carry = [e for e in events if e.path == "scan[0]"]
        assert len(scan0_carry) == 0, f"scan[0] carry should be silent, got {len(scan0_carry)}"

        # cholesky FIRES inside where-filtered scan[0]
        chol_events = [e for e in events if "cholesky" in e.path]
        assert len(chol_events) == N, (
            f"expected {N} cholesky events inside filtered scan[0], got {len(chol_events)}"
        )

        # scan[1] carry taps fire as normal
        scan1_carry = [e for e in events if "scan[1]" in e.path]
        assert len(scan1_carry) == N, f"expected {N} scan[1] carry events, got {len(scan1_carry)}"

    def test_max_depth_emission_only_prim_fires_deeper(self):
        """
        max_depth=0 suppresses carry taps for nodes with depth > 0,
        but prim taps deeper than max_depth still fire.
        """
        OUTER, INNER = 3, 4

        def f(x):
            def outer_body(carry, _):
                def inner_body(ic, __):
                    v = jnp.sin(ic)  # exactly one sin call per inner step
                    return ic + v, v

                ic_out, _ = jax.lax.scan(inner_body, carry, None, length=INNER)
                return ic_out, ic_out

            return jax.lax.scan(outer_body, x, None, length=OUTER)

        x = jnp.float32(0.0)
        ref = f(x)
        events: list[tap.TapEvent] = []
        got = tap.verbose(
            f,
            on_step=lambda e: events.append(e),
            taps=[tap.on("sin")],
            max_depth=0,  # only depth-0 loop emits carry; inner scan is at depth 1
        )(x)
        jax.block_until_ready(got)

        assert bitwise_eq(ref, got), "max_depth=0 result not bitwise identical"

        # Outer scan (depth 0) emits carry taps
        outer_carry = [e for e in events if e.path == "scan[0]"]
        assert len(outer_carry) == OUTER, (
            f"expected {OUTER} outer carry events, got {len(outer_carry)}"
        )

        # Inner scan (depth 1 > max_depth=0) does NOT emit carry taps
        inner_carry = [e for e in events if "scan[0]/scan[0]" in e.path and "sin" not in e.path]
        assert len(inner_carry) == 0, (
            f"inner carry should be suppressed at depth 1, got {len(inner_carry)}"
        )

        # sin prim taps inside the inner scan DO fire (descend-always)
        sin_events = [e for e in events if "sin" in e.path]
        assert len(sin_events) == OUTER * INNER, (
            f"expected {OUTER * INNER} sin events (inside inner scan), got {len(sin_events)}"
        )

    def test_demo03_while_in_scan_where_while(self):
        """
        THE demo-03 acceptance test.

        scan wrapping while; where=lambda p: 'while' in p.
        Expected: while carry taps fire; scan carry taps silent; bitwise holds.
        This was broken pre-FIX-2 because the scan failed the predicate and the
        inner while was unreachable (opaque bind).
        """
        targets = jnp.linspace(0.1, 0.8, 8, dtype=jnp.float32)
        ref = _scan_wrapping_while(targets)

        events: list[tap.TapEvent] = []
        got = tap.verbose(
            _scan_wrapping_while,
            on_step=lambda e: events.append(e),
            where=lambda p: "while" in p,
        )(targets)
        jax.block_until_ready(got)

        assert bitwise_eq(ref, got), "scan-wrapping-while result not bitwise identical"

        # scan carry taps are silent (where does not match scan[0])
        scan_carry = [e for e in events if e.path == "scan[0]"]
        assert len(scan_carry) == 0, (
            f"scan carry should be silent with 'while' filter, got {len(scan_carry)}"
        )

        # while carry taps fire (path contains 'while')
        while_carry = [e for e in events if "while" in e.path]
        assert len(while_carry) > 0, (
            "while carry taps must fire — this was the demo-03 bug pre-FIX-2"
        )

    def test_addressing_stable_under_ops_filter(self):
        """
        With ops=('scan',), while's address counter still advances so scan
        address is unchanged.  Same as the pre-existing test_ops_filtering
        but with a prim tap verifying descent into the while body.
        """

        def f(carry):
            def cond(c):
                return c < 3.0

            def wbody(c):
                return c + 1.0

            c1 = jax.lax.while_loop(cond, wbody, carry[0])

            def sbody(c, x):
                return c + x + jnp.sin(c), c

            c2, _ = jax.lax.scan(sbody, carry[1], jnp.arange(4.0, dtype=jnp.float32))
            return (c1, c2)

        carry0 = (jnp.float32(0.0), jnp.float32(0.0))
        ref = f(carry0)
        events: list[tap.TapEvent] = []
        got = tap.verbose(
            f,
            on_step=lambda e: events.append(e),
            taps=[tap.on("sin")],
            ops=("scan",),  # while NOT in ops — no while carry taps
        )(carry0)
        jax.block_until_ready(got)

        assert bitwise_eq(ref, got), "addressing-stable ops filter result not bitwise identical"

        paths = {e.path for e in events}

        # scan[1] (while[0] incremented counter so scan is index 1)
        assert "scan[1]" in paths, f"scan[1] missing; got {sorted(paths)}"

        # while carry taps absent
        assert not any("while" in p and "sin" not in p for p in paths), (
            f"while carry taps should not fire, paths: {sorted(paths)}"
        )

        # sin fires inside scan body (scan IS descended)
        sin_events = [e for e in events if "sin" in e.path]
        assert len(sin_events) > 0, "sin should fire inside descended scan body"


# ---------------------------------------------------------------------------
# FIX 1 tests: sample_every gates primitive taps
# ---------------------------------------------------------------------------


class TestFix1SampleEveryGatesPrimTaps:
    """
    M1d FIX 1: sample_every gates prim taps inside loops with the same
    lax.cond(step % se == 0, ...) pattern as carry taps.
    Prim taps outside any loop are always ungated.
    """

    def test_prim_tap_counts_se_1_10_100(self):
        """
        N=30 scan, sin prim tap: expected event counts are
        se=1 → 30, se=10 → 3, se=100 → 1.
        """
        N = 30

        def f(x):
            def body(carry, _):
                return jnp.sin(carry) + 0.01, carry

            return jax.lax.scan(body, x, None, length=N)

        x = jnp.float32(1.0)

        for se, expected in [(1, 30), (10, 3), (100, 1)]:
            events: list[tap.TapEvent] = []
            got = tap.verbose(
                f,
                on_step=lambda e: events.append(e),
                taps=[tap.on("sin")],
                sample_every=se,
            )(x)
            jax.block_until_ready(got)

            sin_events = [e for e in events if "sin" in e.path]
            assert len(sin_events) == expected, (
                f"se={se}: expected {expected} sin events, got {len(sin_events)}"
            )

    def test_prim_tap_outside_loop_always_fires(self):
        """
        A prim tap outside any loop fires regardless of sample_every (ungated).
        """

        def f(x):
            M = jnp.array([[1.0, 0.5], [0.5, 1.0]])
            L = jnp.linalg.cholesky(M)
            return L[0, 0] + x

        x = jnp.float32(1.0)
        ref = f(x)

        for se in [1, 10, 100]:
            events: list[tap.TapEvent] = []
            got = tap.verbose(
                f,
                on_step=lambda e: events.append(e),
                taps=[tap.on("cholesky")],
                sample_every=se,
            )(x)
            jax.block_until_ready(got)

            chol_events = [e for e in events if "cholesky" in e.path]
            assert len(chol_events) == 1, (
                f"se={se}: outside-loop prim tap should always fire (got {len(chol_events)})"
            )
            assert chol_events[0].step == -1, "outside-loop step must be -1"
            assert bitwise_eq(ref, got), f"se={se}: outside-loop result not bitwise identical"

    def test_prim_tap_gating_bitwise_identity(self):
        """
        sample_every gating on prim taps must not alter computation results.
        """
        N = 20

        def f(x):
            def body(carry, _):
                return jnp.sin(carry), carry

            return jax.lax.scan(body, x, None, length=N)

        x = jnp.float32(0.7)
        ref = f(x)

        for se in [1, 5, 20, 100]:
            events: list[tap.TapEvent] = []
            got = tap.verbose(
                f,
                on_step=lambda e: events.append(e),
                taps=[tap.on("sin")],
                sample_every=se,
            )(x)
            jax.block_until_ready(got)

            assert bitwise_eq(ref, got), f"se={se}: prim tap gating broke bitwise identity"

    def test_once_composes_with_gate(self):
        """
        once=True fires at most once across events that SURVIVE the gate.
        Gated-out events (step % se != 0) do not consume the once budget.
        """
        N = 30

        def f(x):
            def body(carry, _):
                return jnp.sin(carry) + 0.01, carry

            return jax.lax.scan(body, x, None, length=N)

        x = jnp.float32(1.0)

        # With se=10, N=30: events fire at steps 0, 10, 20 (3 events).
        # once=True: alert fires once (at step 0, the first event that survives).
        # The alert predicate is "always True" so it fires on the first surviving event.
        import sys
        from io import StringIO

        old_stderr = sys.stderr
        sys.stderr = buf = StringIO()
        try:
            events: list[tap.TapEvent] = []
            got = tap.verbose(
                f,
                on_step=lambda e: events.append(e),
                taps=[tap.on("sin", alert=lambda v: True, label="test-once", once=True)],
                sample_every=10,
            )(x)
            jax.block_until_ready(got)
        finally:
            sys.stderr = old_stderr

        fail_lines = [ln for ln in buf.getvalue().splitlines() if "FAIL" in ln]
        # once=True: exactly ONE alert line across all surviving events
        assert len(fail_lines) == 1, (
            f"once=True with se=10 should produce 1 FAIL line, got {len(fail_lines)}: {fail_lines}"
        )

        # on_step receives 3 events (steps 0, 10, 20)
        sin_events = [e for e in events if "sin" in e.path]
        assert len(sin_events) == 3, (
            f"expected 3 sin events at se=10 N=30, got {len(sin_events)}"
        )

    def test_carry_tap_and_prim_tap_both_gated(self):
        """
        With sample_every=k both carry and prim taps are gated consistently.
        """
        N = 10

        def f(x):
            def body(carry, _):
                return jnp.sin(carry) + 0.01, carry

            return jax.lax.scan(body, x, None, length=N)

        x = jnp.float32(1.0)
        se = 5  # expect N/se = 2 events of each kind

        events: list[tap.TapEvent] = []
        got = tap.verbose(
            f,
            on_step=lambda e: events.append(e),
            taps=[tap.on("sin")],
            sample_every=se,
        )(x)
        jax.block_until_ready(got)

        carry_events = [e for e in events if e.path == "scan[0]"]
        sin_events = [e for e in events if "sin" in e.path]

        assert len(carry_events) == N // se, (
            f"expected {N // se} carry events, got {len(carry_events)}"
        )
        assert len(sin_events) == N // se, (
            f"expected {N // se} sin events, got {len(sin_events)}"
        )

    def test_prim_tap_se_gating_fired_steps_are_correct(self):
        """
        With se=k the prim tap fires at steps 0, k, 2k, ... (not other steps).
        """
        N = 30
        se = 10

        def f(x):
            def body(carry, _):
                return jnp.sin(carry) + 0.01, carry

            return jax.lax.scan(body, x, None, length=N)

        x = jnp.float32(1.0)
        events: list[tap.TapEvent] = []
        got = tap.verbose(
            f,
            on_step=lambda e: events.append(e),
            taps=[tap.on("sin")],
            sample_every=se,
        )(x)
        jax.block_until_ready(got)

        sin_events = [e for e in events if "sin" in e.path]
        fired_steps = sorted(e.step for e in sin_events)
        expected_steps = list(range(0, N, se))
        assert fired_steps == expected_steps, (
            f"se={se}: expected steps {expected_steps}, got {fired_steps}"
        )


# ---------------------------------------------------------------------------
# FIX 3 tests: path-aware select
# ---------------------------------------------------------------------------


class TestFix3PathAwareSelect:
    """
    M1d FIX 3: if a select callable accepts a 'path' kwarg or 2nd positional,
    jaxtap calls select(leaves, path=path) instead of select(leaves).
    Same for per-tap select on PrimitiveTap.
    """

    def test_select_without_path_back_compat(self):
        """
        select(leaves) — single-arg form — still works as before.
        """
        N = 4
        x0 = jnp.float32(1.0)
        xs = jnp.arange(float(N), dtype=jnp.float32)

        def f(x, xs_):
            return jax.lax.scan(lambda c, x_: (c + x_, c), x, xs_)

        events: list[tap.TapEvent] = []
        got = tap.verbose(
            f,
            on_step=lambda e: events.append(e),
            select=lambda leaves: leaves[0],
        )(x0, xs)
        jax.block_until_ready(got)

        scan_events = [e for e in events if e.path == "scan[0]"]
        assert len(scan_events) == N
        for e in scan_events:
            assert np.asarray(e.value).ndim == 0, "select(leaves) must give scalar"

    def test_select_with_path_kwarg_receives_path(self):
        """
        select(leaves, path=path) — path is the stable node address string.
        """
        N = 6

        def f(x0, xs):
            return jax.lax.scan(lambda c, x: (c + x, c), x0, xs)

        x0 = jnp.float32(0.0)
        xs = jnp.arange(float(N), dtype=jnp.float32)

        received_paths: list[str] = []

        def path_select(leaves, *, path):
            received_paths.append(path)
            return leaves[0]

        events: list[tap.TapEvent] = []
        got = tap.verbose(
            f,
            on_step=lambda e: events.append(e),
            select=path_select,
        )(x0, xs)
        jax.block_until_ready(got)

        scan_events = [e for e in events if e.path == "scan[0]"]
        assert len(scan_events) == N

        # path is "scan[0]" for each call
        assert len(received_paths) == N, f"expected {N} path calls, got {len(received_paths)}"
        assert all(p == "scan[0]" for p in received_paths), (
            f"unexpected paths: {set(received_paths)}"
        )

    def test_select_with_path_positional_2nd_arg(self):
        """
        select(leaves, path) — path as 2nd positional — also accepted.
        """
        N = 4

        def f(x0, xs):
            return jax.lax.scan(lambda c, x: (c + x, c), x0, xs)

        x0 = jnp.float32(0.0)
        xs = jnp.arange(float(N), dtype=jnp.float32)

        received_paths: list[str] = []

        def positional_select(leaves, path):  # 2nd positional, not kwarg-only
            received_paths.append(path)
            return leaves[0]

        events: list[tap.TapEvent] = []
        got = tap.verbose(
            f,
            on_step=lambda e: events.append(e),
            select=positional_select,
        )(x0, xs)
        jax.block_until_ready(got)

        scan_events = [e for e in events if e.path == "scan[0]"]
        assert len(scan_events) == N
        assert len(received_paths) == N
        assert all(p == "scan[0]" for p in received_paths)

    def test_select_path_receives_correct_path_per_node(self):
        """
        Nested scan: outer select receives 'scan[0]'; inner receives 'scan[0]/scan[0]'.
        """
        OUTER, INNER = 3, 4

        def f(x0, xs_outer):
            INNER_XS = jnp.arange(float(INNER), dtype=jnp.float32)

            def outer_body(c, x):
                c2, _ = jax.lax.scan(
                    lambda ic, xi: (ic + xi, ic),
                    c + x,
                    INNER_XS,
                )
                return c2, c2

            return jax.lax.scan(outer_body, x0, xs_outer)

        x0 = jnp.float32(0.0)
        xs_outer = jnp.arange(float(OUTER), dtype=jnp.float32)

        path_by_event: list[tuple[str, str]] = []  # (event.path, select_path)

        def path_select(leaves, *, path):
            # Return a sentinel dict so we can track which select call matched
            path_by_event.append(("(select called)", path))
            return leaves[0]

        events: list[tap.TapEvent] = []
        got = tap.verbose(
            f,
            on_step=lambda e: events.append(e),
            select=path_select,
        )(x0, xs_outer)
        jax.block_until_ready(got)

        # Outer scan fires OUTER times; inner fires OUTER*INNER times
        outer_events = [e for e in events if e.path == "scan[0]"]
        inner_events = [e for e in events if e.path == "scan[0]/scan[0]"]
        assert len(outer_events) == OUTER
        assert len(inner_events) == OUTER * INNER

        # select was called with correct paths
        outer_paths = [p for (_, p) in path_by_event if p == "scan[0]"]
        inner_paths = [p for (_, p) in path_by_event if p == "scan[0]/scan[0]"]
        assert len(outer_paths) == OUTER, f"expected {OUTER} outer path calls"
        assert len(inner_paths) == OUTER * INNER, f"expected {OUTER * INNER} inner path calls"

    def test_per_tap_select_with_path(self):
        """
        PrimitiveTap.select with path kwarg receives the prim tap path.
        """
        N = 5

        def f(x):
            def body(carry, _):
                return jnp.sin(carry) + 0.01, carry

            return jax.lax.scan(body, x, None, length=N)

        x = jnp.float32(1.0)

        received_prim_paths: list[str] = []

        def path_aware_select(v, *, path):
            received_prim_paths.append(path)
            return v[0] if isinstance(v, tuple) else v

        events: list[tap.TapEvent] = []
        got = tap.verbose(
            f,
            on_step=lambda e: events.append(e),
            taps=[tap.on("sin", select=path_aware_select)],
        )(x)
        jax.block_until_ready(got)

        sin_events = [e for e in events if "sin" in e.path]
        assert len(sin_events) == N

        # prim path looks like "scan[0]/sin[0]"
        assert len(received_prim_paths) == N
        for p in received_prim_paths:
            assert "sin" in p, f"expected 'sin' in prim path, got {p!r}"

    def test_per_tap_select_without_path_back_compat(self):
        """
        PrimitiveTap with select(v) — single-arg — still works (back-compat).
        """
        N = 4

        def f(x):
            def body(carry, _):
                return jnp.sin(carry) + 0.01, carry

            return jax.lax.scan(body, x, None, length=N)

        x = jnp.float32(1.0)
        events: list[tap.TapEvent] = []
        got = tap.verbose(
            f,
            on_step=lambda e: events.append(e),
            taps=[tap.on("sin", select=lambda v: v[0] if isinstance(v, tuple) else v)],
        )(x)
        jax.block_until_ready(got)

        sin_events = [e for e in events if "sin" in e.path]
        assert len(sin_events) == N
        for e in sin_events:
            assert np.asarray(e.value).ndim == 0 or hasattr(e.value, "shape"), (
                "select(v) result should be array-like"
            )
