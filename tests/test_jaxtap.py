"""
M0 invariant tests for jaxtap B-core walker.

Ports every invariant from proofs/jaxtap_sketch.py and
proofs/jaxtap_while_sketch.py, plus the M0-specific requirements.

Run with: uv run pytest
"""
from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxtap as tap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect(f, *args, **verbose_kwargs):
    """Run tap.verbose(f, ...) on args; return (result, events_list)."""
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
# Small programs used across multiple tests
# ---------------------------------------------------------------------------


def _simple_scan(x0, xs):
    """Flat scalar carry; no nesting."""
    return jax.lax.scan(lambda c, x: (c + x, c * x), x0, xs)


def _nested_scan(x0, xs):
    """Outer scan whose body contains an inner scan + a closed-over const array."""
    INNER_XS = jnp.arange(3.0, dtype=jnp.float32)

    def outer_body(c, x):
        c2, _ = jax.lax.scan(
            lambda c_, xi: (c_ * 1.001 + jnp.sin(xi), c_),
            c + x,
            INNER_XS,
        )
        return c2, c2 * 2.0

    return jax.lax.scan(outer_body, x0, xs)


def _simple_while(v0):
    """while_loop with closed-over consts in BOTH cond and body."""
    LIM = jnp.float32(37.0)
    INC = jnp.float32(1.7)

    def cond(c):
        return c < LIM

    def body(c):
        return c + INC

    return jax.lax.while_loop(cond, body, v0)


# ---------------------------------------------------------------------------
# test_identity_bitwise
# ---------------------------------------------------------------------------


def test_identity_bitwise():
    """Scan, nested scan, and while_loop all produce bitwise-identical outputs."""
    x0 = jnp.float32(0.5)
    xs = jnp.linspace(0.0, 1.0, 4, dtype=jnp.float32)

    ref_s = _simple_scan(x0, xs)
    got_s, _ = _collect(_simple_scan, x0, xs)
    assert bitwise_eq(ref_s, got_s), "plain scan not bitwise identical"

    ref_n = _nested_scan(x0, xs)
    got_n, _ = _collect(_nested_scan, x0, xs)
    assert bitwise_eq(ref_n, got_n), "nested scan not bitwise identical"

    v0 = jnp.float32(0.3)
    ref_w = _simple_while(v0)
    got_w, _ = _collect(_simple_while, v0)
    assert bitwise_eq(ref_w, got_w), "while_loop not bitwise identical"


# ---------------------------------------------------------------------------
# test_scan_taps
# ---------------------------------------------------------------------------


def test_scan_taps():
    """Per-step events are emitted in order 0..N-1 on the correct path."""
    N = 6
    x0 = jnp.float32(1.0)
    xs = jnp.arange(float(N), dtype=jnp.float32)

    _, events = _collect(_simple_scan, x0, xs)

    scan_events = [e for e in events if e.path == "scan[0]"]
    assert len(scan_events) == N
    assert [e.step for e in scan_events] == list(range(N))


# ---------------------------------------------------------------------------
# test_nested_addressing
# ---------------------------------------------------------------------------


def test_nested_addressing():
    """
    Nested scans get stable addresses (scan[0]/scan[0]).
    Mixed scan+while at one level share a single counter: scan[0], while[1].
    """
    x0 = jnp.float32(0.5)
    xs = jnp.linspace(0.0, 1.0, 4, dtype=jnp.float32)

    # --- nested scans ---
    _, events_nested = _collect(_nested_scan, x0, xs)
    paths_nested = {e.path for e in events_nested}
    assert "scan[0]" in paths_nested, "outer scan path missing"
    assert "scan[0]/scan[0]" in paths_nested, "inner scan path missing"

    # --- mixed scan + while at the same (top) level ---
    def f_mixed(carry):
        # scan is the 0th CF eqn → "scan[0]"; while is the 1st → "while[1]"
        c1, _ = jax.lax.scan(
            lambda c, x: (c + x, c),
            carry[0],
            jnp.arange(3.0, dtype=jnp.float32),
        )

        def cond(c):
            return c < 5.0

        def body(c):
            return c + 1.0

        c2 = jax.lax.while_loop(cond, body, carry[1])
        return (c1, c2)

    carry0 = (jnp.float32(0.0), jnp.float32(0.0))
    _, events_mixed = _collect(f_mixed, carry0)
    paths_mixed = {e.path for e in events_mixed}
    assert "scan[0]" in paths_mixed, "scan[0] missing from mixed program"
    assert "while[1]" in paths_mixed, "while[1] missing — counter not shared"


# ---------------------------------------------------------------------------
# test_jit_composition
# ---------------------------------------------------------------------------


def test_jit_composition():
    """verbose(jit(f)) and jit(verbose(f)) are both bitwise-correct and emit equal event counts."""
    x0 = jnp.float32(0.5)
    xs = jnp.linspace(0.0, 1.0, 4, dtype=jnp.float32)
    ref = _nested_scan(x0, xs)

    # verbose(jit(f)) — walker recurses through the jit eqn
    events1: list[tap.TapEvent] = []
    got1 = tap.verbose(jax.jit(_nested_scan), on_step=lambda e: events1.append(e))(x0, xs)
    jax.block_until_ready(got1)

    # jit(verbose(f)) — the instrumented function is itself jittable
    events2: list[tap.TapEvent] = []
    got2 = jax.jit(tap.verbose(_nested_scan, on_step=lambda e: events2.append(e)))(x0, xs)
    jax.block_until_ready(got2)

    assert bitwise_eq(ref, got1), "verbose(jit(f)) not bitwise identical"
    assert bitwise_eq(ref, got2), "jit(verbose(f)) not bitwise identical"
    assert len(events1) == len(events2), "event counts differ between compositions"


# ---------------------------------------------------------------------------
# test_while_heartbeat
# ---------------------------------------------------------------------------


def test_while_heartbeat():
    """Event count matches the number of while iterations (heartbeat)."""
    v0 = jnp.float32(0.3)
    LIM, INC = np.float32(37.0), np.float32(1.7)

    # Count expected iterations in float32 arithmetic to match XLA.
    c = np.float32(v0)
    expected_iters = 0
    while c < LIM:
        c = c + INC
        expected_iters += 1

    _, events = _collect(_simple_while, v0)
    while_events = [e for e in events if e.path == "while[0]"]
    assert len(while_events) == expected_iters
    assert [e.step for e in while_events] == list(range(expected_iters))


# ---------------------------------------------------------------------------
# test_params_passthrough
# ---------------------------------------------------------------------------


def test_params_passthrough():
    """scan with reverse=True and unroll=2 remain bitwise-correct with correct event counts."""
    N = 5
    x0 = jnp.float32(1.0)
    xs = jnp.arange(float(N), dtype=jnp.float32)

    def scan_reverse(x0_, xs_):
        return jax.lax.scan(lambda c, x: (c + x, c * x), x0_, xs_, reverse=True)

    def scan_unroll2(x0_, xs_):
        return jax.lax.scan(lambda c, x: (c + x, c * x), x0_, xs_, unroll=2)

    ref_rev = scan_reverse(x0, xs)
    got_rev, ev_rev = _collect(scan_reverse, x0, xs)
    assert bitwise_eq(ref_rev, got_rev), "reverse=True not bitwise identical"
    assert len(ev_rev) == N

    ref_u2 = scan_unroll2(x0, xs)
    got_u2, ev_u2 = _collect(scan_unroll2, x0, xs)
    assert bitwise_eq(ref_u2, got_u2), "unroll=2 not bitwise identical"
    assert len(ev_u2) == N


# ---------------------------------------------------------------------------
# test_select_reduce_on_device
# ---------------------------------------------------------------------------


def test_select_reduce_on_device():
    """select reduces carry on-device; TapEvent.value carries correct pytree."""
    N = 4
    x0 = jnp.float32(1.0)
    xs = jnp.arange(float(N), dtype=jnp.float32)

    # Scalar selector — value should be a 0-d array
    events_scalar: list[tap.TapEvent] = []
    tapped_scalar = tap.verbose(
        _simple_scan,
        on_step=lambda e: events_scalar.append(e),
        select=lambda leaves: leaves[0].mean(),
    )
    jax.block_until_ready(tapped_scalar(x0, xs))
    assert len(events_scalar) == N
    for e in events_scalar:
        assert np.asarray(e.value).ndim == 0, "scalar selector must yield a 0-d value"

    # Dict-returning selector — value must be a dict
    events_dict: list[tap.TapEvent] = []
    tapped_dict = tap.verbose(
        _simple_scan,
        on_step=lambda e: events_dict.append(e),
        select=lambda leaves: {"carry": leaves[0]},
    )
    jax.block_until_ready(tapped_dict(x0, xs))
    assert len(events_dict) == N
    for e in events_dict:
        assert isinstance(e.value, dict), "dict selector must yield a dict value"
        assert "carry" in e.value


# ---------------------------------------------------------------------------
# test_ops_filtering
# ---------------------------------------------------------------------------


def test_ops_filtering():
    """ops=('scan',) suppresses while events; scan addresses are stable (counter not reset)."""

    def f(carry):
        # while is the 0th CF eqn; scan is the 1st.
        def cond(c):
            return c < 3.0

        def body_fn(c):
            return c + 1.0

        c1 = jax.lax.while_loop(cond, body_fn, carry[0])
        c2, _ = jax.lax.scan(
            lambda c, x: (c + x, c),
            carry[1],
            jnp.arange(4.0, dtype=jnp.float32),
        )
        return (c1, c2)

    carry0 = (jnp.float32(0.0), jnp.float32(0.0))

    # Both ops — while[0] and scan[1]
    _, ev_all = _collect(f, carry0)
    paths_all = {e.path for e in ev_all}
    assert "while[0]" in paths_all
    assert "scan[1]" in paths_all

    # Scan only — while suppressed; scan address is UNCHANGED (still index 1)
    _, ev_scan = _collect(f, carry0, ops=("scan",))
    paths_scan = {e.path for e in ev_scan}
    assert "scan[1]" in paths_scan, "scan index changed when while was filtered — counter reset"
    assert not any("while" in p for p in paths_scan), "while events appeared despite filtering"


# ---------------------------------------------------------------------------
# test_callback_totality
# ---------------------------------------------------------------------------


def test_callback_totality():
    """A raising on_step never corrupts results; warns exactly once; -W error is handled."""
    N = 5
    x0 = jnp.float32(0.0)
    xs = jnp.arange(float(N), dtype=jnp.float32)
    ref = _simple_scan(x0, xs)

    # --- Part 1: raises → correct results + exactly 1 UserWarning ---
    call_count = [0]

    def raising_cb(event: tap.TapEvent) -> None:
        call_count[0] += 1
        raise ValueError("boom")

    tap._warned.discard(id(raising_cb))  # ensure fresh warn-once state

    with pytest.warns(UserWarning, match="jaxtap") as warn_list:
        got = tap.verbose(_simple_scan, on_step=raising_cb)(x0, xs)
        jax.block_until_ready(got)

    assert bitwise_eq(ref, got), "result corrupted by raising callback"
    assert len(warn_list.list) == 1, f"expected exactly 1 warning, got {len(warn_list.list)}"
    assert call_count[0] == N, "callback must be attempted every step"

    # --- Part 2: under warnings.simplefilter("error") no exception propagates ---
    def raising_cb2(event: tap.TapEvent) -> None:
        raise RuntimeError("boom2")

    tap._warned.discard(id(raising_cb2))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got2 = tap.verbose(_simple_scan, on_step=raising_cb2)(x0, xs)
        jax.block_until_ready(got2)

    assert bitwise_eq(ref, got2), "result corrupted under -W error"


# ---------------------------------------------------------------------------
# test_carry_leaves_contract
# ---------------------------------------------------------------------------


def test_carry_leaves_contract():
    """Dict-carry scan: without select TapEvent.value is a flat tuple; select reshapes it."""

    def step_fn(carry, x):
        return {"a": carry["a"] + x, "b": carry["b"] * 2.0}, x

    carry0 = {"a": jnp.float32(1.0), "b": jnp.float32(2.0)}
    xs = jnp.arange(3.0, dtype=jnp.float32)

    def scan_f(carry_, xs_):
        return jax.lax.scan(step_fn, carry_, xs_)

    # Without select: flat tuple of carry leaves (dict flattens alphabetically → [a, b])
    events_flat: list[tap.TapEvent] = []
    got = tap.verbose(scan_f, on_step=lambda e: events_flat.append(e))(carry0, xs)
    jax.block_until_ready(got)

    assert len(events_flat) == 3
    for e in events_flat:
        assert isinstance(e.value, tuple), "without select, value must be a tuple"
        assert len(e.value) == 2, "dict with 2 keys must flatten to 2 leaves"

    # With select reshaping: value is a dict
    events_shaped: list[tap.TapEvent] = []
    got2 = tap.verbose(
        scan_f,
        on_step=lambda e: events_shaped.append(e),
        select=lambda leaves: {"a": leaves[0], "b": leaves[1]},
    )(carry0, xs)
    jax.block_until_ready(got2)

    assert len(events_shaped) == 3
    for e in events_shaped:
        assert isinstance(e.value, dict), "select must return structured value"
        assert set(e.value.keys()) == {"a", "b"}


# ---------------------------------------------------------------------------
# test_literal_outvar
# ---------------------------------------------------------------------------


def test_literal_outvar():
    """A function returning a constant is handled correctly (_read Literal branch)."""

    def f(x):
        # The scan body returns a constant zero as ys; _read must handle Literals in outvars.
        c, ys = jax.lax.scan(
            lambda c, _: (c + 1.0, jnp.zeros((), jnp.float32)),
            x,
            jnp.arange(3.0, dtype=jnp.float32),
        )
        return c, ys

    x = jnp.float32(0.0)
    ref = f(x)
    got, events = _collect(f, x)
    assert bitwise_eq(ref, got), "constant-returning function not bitwise identical"
    assert len(events) == 3, "expected 3 scan-step events"


# ---------------------------------------------------------------------------
# test_kwargs_rejected
# ---------------------------------------------------------------------------


def test_kwargs_rejected():
    """verbose(f) raises TypeError when called with keyword arguments."""

    def f(x):
        return x

    tapped = tap.verbose(f, on_step=lambda e: None)
    with pytest.raises(TypeError):
        tapped(jnp.float32(1.0), k=1)
