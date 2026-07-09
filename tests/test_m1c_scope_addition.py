"""
M1c scope-addition tests: output=k indexing, tap.print, composition.

Covers:
- output=k on PrimitiveTap / tap.on() selects single output before select
- out-of-range output index raises IndexError at trace time
- tap.print writes [tap] {path} {step}/{total}: {value} format to stderr
- tap.print respects numpy printoptions truncation
- tap.print with-form (A-shell)
- watch_nan with output=k (single-array mode)
- composition: watch_nan + tap.print on same primitive

Run with: uv run pytest tests/test_m1c_scope_addition.py -v
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


_EIGH_N = 3  # concrete Python int — scan length must be static at trace time


def _eigh_scan(x0):
    """Scan body that calls jnp.linalg.eigh; returns (eigenvalues, eigenvectors)."""

    def body(carry, _):
        k = carry
        M = jnp.array([[k, 0.0], [0.0, k + 1.0]], dtype=jnp.float32)
        vals, vecs = jnp.linalg.eigh(M)
        return carry + 1.0, vals

    return jax.lax.scan(body, x0, None, length=_EIGH_N)


# ---------------------------------------------------------------------------
# output=k: single-output selection
# ---------------------------------------------------------------------------


def test_output_k_delivers_single_array_not_tuple():
    """output=0 — value is a single array (not a tuple) with a concrete ndim."""
    prim_counts = tap.primitives(_eigh_scan, jnp.float32(1.0))
    eigh_name = next((k for k in prim_counts if "eigh" in k), None)
    assert eigh_name is not None, f"no eigh prim found; got {list(prim_counts)}"

    _, events = _collect(
        _eigh_scan,
        jnp.float32(1.0),
        taps=[tap.on(eigh_name, output=0)],
    )
    prim_events = [e for e in events if eigh_name in e.path]
    assert len(prim_events) == _EIGH_N
    for e in prim_events:
        # Without output=k, value would be a tuple; with it, a single array.
        assert not isinstance(e.value, tuple), f"expected array not tuple, got {type(e.value)}"
        assert hasattr(e.value, "shape"), f"expected array-like, got {type(e.value)}"
        # output=0 of eigh is one of the two 2×2 or (2,) outputs (whichever JAX puts first)
        assert len(np.asarray(e.value).shape) >= 1, "expected at least 1-d"


def test_output_k_with_select_receives_single_array():
    """output=0 + select=sum — select receives a single array, result is scalar."""
    prim_counts = tap.primitives(_eigh_scan, jnp.float32(1.0))
    eigh_name = next((k for k in prim_counts if "eigh" in k), None)
    assert eigh_name is not None

    events = []
    tapped = tap.verbose(
        _eigh_scan,
        on_step=lambda e: events.append(e),
        taps=[tap.on(eigh_name, output=0, select=lambda arr: arr.sum())],
    )
    result = tapped(jnp.float32(1.0))
    jax.block_until_ready(result)

    prim_events = [e for e in events if eigh_name in e.path]
    assert len(prim_events) == _EIGH_N
    for e in prim_events:
        # sum of any 1-d or 2-d array is a scalar
        assert np.asarray(e.value).shape == (), f"expected scalar, got {np.asarray(e.value).shape}"


def test_output_k_out_of_range_raises_index_error():
    """output=99 raises IndexError at trace time (first call to wrapped)."""
    prim_counts = tap.primitives(_eigh_scan, jnp.float32(1.0))
    eigh_name = next((k for k in prim_counts if "eigh" in k), None)
    assert eigh_name is not None

    tapped = tap.verbose(
        _eigh_scan,
        on_step=lambda e: None,
        taps=[tap.on(eigh_name, output=99)],
    )
    with pytest.raises(IndexError, match="output=99"):
        tapped(jnp.float32(1.0))


def test_output_k_negative_raises_index_error():
    """output=-1 raises IndexError at trace time."""
    prim_counts = tap.primitives(_eigh_scan, jnp.float32(1.0))
    eigh_name = next((k for k in prim_counts if "eigh" in k), None)
    assert eigh_name is not None

    tapped = tap.verbose(
        _eigh_scan,
        on_step=lambda e: None,
        taps=[tap.on(eigh_name, output=-1)],
    )
    with pytest.raises(IndexError):
        tapped(jnp.float32(1.0))


# ---------------------------------------------------------------------------
# watch_nan with output=k (single-array mode)
# ---------------------------------------------------------------------------


def _sqrt_nan_scan(x0):
    """Scan that passes negative values to sqrt — produces NaN for steps where carry < 0."""

    def body(carry, _):
        v = jnp.sqrt(carry)  # NaN when carry < 0
        return carry - 1.0, v

    return jax.lax.scan(body, x0, None, length=4)  # carry: x0, x0-1, x0-2, x0-3


def test_watch_nan_output_k_fires_on_nan_in_selected_output(capsys):
    """watch_nan(output=0) in single-array mode fires when the selected output is NaN.

    Uses sqrt with a negative carry (x0=1.5 → carry goes 1.5, 0.5, −0.5, −1.5;
    sqrt(negative) = NaN for steps 2 and 3).
    """
    prim_counts = tap.primitives(_sqrt_nan_scan, jnp.float32(1.5))
    sqrt_name = next((k for k in prim_counts if "sqrt" in k), None)
    assert sqrt_name is not None, f"no sqrt prim; got {list(prim_counts)}"

    tapped = tap.verbose(
        _sqrt_nan_scan,
        on_step=lambda e: None,
        taps=[tap.watch_nan(sqrt_name, output=0, label="sqrt NaN")],
    )
    result = tapped(jnp.float32(1.5))
    jax.block_until_ready(result)
    captured = capsys.readouterr()
    assert "FAIL" in captured.err, f"expected FAIL on stderr, got: {captured.err!r}"
    assert "sqrt NaN" in captured.err


def test_watch_nan_output_k_silent_when_finite(capsys):
    """watch_nan(output=0) in single-array mode is silent when output is finite."""
    prim_counts = tap.primitives(_sqrt_nan_scan, jnp.float32(1.5))
    sqrt_name = next((k for k in prim_counts if "sqrt" in k), None)
    assert sqrt_name is not None

    tapped = tap.verbose(
        _sqrt_nan_scan,
        on_step=lambda e: None,
        # All sqrt inputs are positive when x0=10.0 (carry: 10, 9, 8, 7)
        taps=[tap.watch_nan(sqrt_name, output=0)],
    )
    result = tapped(jnp.float32(10.0))
    jax.block_until_ready(result)
    captured = capsys.readouterr()
    assert "FAIL" not in captured.err


# ---------------------------------------------------------------------------
# tap.print: format and truncation
# ---------------------------------------------------------------------------


def _sin_scan(x0):
    """Scan body that computes sin(carry) 3 times."""

    def body(carry, _):
        v = jnp.sin(carry)
        return v, v

    return jax.lax.scan(body, x0, None, length=3)


def _cos_scan(x0):
    """Scan body that computes cos(carry) 2 times."""

    def body(carry, _):
        return jnp.cos(carry), carry

    return jax.lax.scan(body, x0, None, length=2)


def _large_array_scan(x0):
    """Scan body that calls sin on a 64-element vector."""

    def body(carry, _):
        v = jnp.ones(64, dtype=jnp.float32) * carry
        w = jnp.sin(v)
        return carry + 1.0, w.sum()

    return jax.lax.scan(body, x0, None, length=2)


def test_tap_print_format(capsys):
    """tap.print emits [tap] {path} {step}/{total}: {value} to stderr."""
    prim_counts = tap.primitives(_sin_scan, jnp.float32(1.0))
    sin_name = next((k for k in prim_counts if k == "sin"), None)
    assert sin_name is not None, f"no sin found; prims={list(prim_counts)}"

    tapped = tap.verbose(
        _sin_scan,
        on_step=lambda e: None,
        taps=[tap.print(sin_name)],
    )
    result = tapped(jnp.float32(1.0))
    jax.block_until_ready(result)
    captured = capsys.readouterr()

    lines = [line for line in captured.err.splitlines() if "[tap]" in line]
    assert len(lines) == 3, f"expected 3 lines, got {len(lines)}: {captured.err!r}"
    for line in lines:
        # Must NOT contain "FAIL" (that's the alert format)
        assert "FAIL" not in line, f"unexpected FAIL in tap.print line: {line!r}"
        # Must start with "[tap] "
        assert line.startswith("[tap] "), f"unexpected prefix: {line!r}"
        # Must contain "/" for step/total
        assert "/" in line, f"missing step/total separator: {line!r}"


def test_tap_print_no_fail_prefix(capsys):
    """tap.print lines do NOT contain 'FAIL'."""
    prim_counts = tap.primitives(_cos_scan, jnp.float32(0.5))
    cos_name = next((k for k in prim_counts if k == "cos"), None)
    assert cos_name is not None

    tapped = tap.verbose(
        _cos_scan,
        on_step=lambda e: None,
        taps=[tap.print(cos_name)],
    )
    result = tapped(jnp.float32(0.5))
    jax.block_until_ready(result)
    captured = capsys.readouterr()
    assert "FAIL" not in captured.err
    assert "[tap]" in captured.err


def test_tap_print_truncation_large_array(capsys):
    """tap.print truncates large arrays via numpy printoptions threshold=8."""
    prim_counts = tap.primitives(_large_array_scan, jnp.float32(1.0))
    sin_name = next((k for k in prim_counts if k == "sin"), None)
    assert sin_name is not None

    tapped = tap.verbose(
        _large_array_scan,
        on_step=lambda e: None,
        taps=[tap.print(sin_name)],
    )
    result = tapped(jnp.float32(1.0))
    jax.block_until_ready(result)
    captured = capsys.readouterr()
    lines = [line for line in captured.err.splitlines() if "[tap]" in line]
    assert len(lines) == 2
    for line in lines:
        # numpy threshold=8 inserts "..." for arrays longer than 8 elements
        assert "..." in line, f"expected truncation in: {line!r}"


def _tanh_scan(x0):
    """Scan body that computes tanh(carry) 3 times."""

    def body(carry, _):
        return jnp.tanh(carry), carry

    return jax.lax.scan(body, x0, None, length=3)


def test_tap_print_with_form(capsys):
    """tap.print works via the A-form (with tap.record)."""
    prim_counts = tap.primitives(_tanh_scan, jnp.float32(0.3))
    tanh_name = next((k for k in prim_counts if "tanh" in k), None)
    assert tanh_name is not None, f"no tanh found; prims={list(prim_counts)}"

    with tap.record(taps=[tap.print(tanh_name)]):
        result = _tanh_scan(jnp.float32(0.3))
        jax.block_until_ready(result)

    captured = capsys.readouterr()
    lines = [line for line in captured.err.splitlines() if "[tap]" in line]
    assert len(lines) == 3, f"expected 3 lines, got {len(lines)}: {captured.err!r}"
    for line in lines:
        assert "FAIL" not in line
        assert "[tap]" in line


# ---------------------------------------------------------------------------
# Composition: watch_nan + tap.print on same function
# ---------------------------------------------------------------------------


def _mixed_scan(x0):
    """Scan body using sin; 4 steps."""

    def body(carry, _):
        v = jnp.sin(carry)
        return carry + 1.0, v

    return jax.lax.scan(body, x0, None, length=4)


def test_watch_nan_and_print_compose(capsys):
    """watch_nan and tap.print can both be in taps= for the same function."""
    prim_counts = tap.primitives(_mixed_scan, jnp.float32(0.0))
    sin_name = next((k for k in prim_counts if k == "sin"), None)
    assert sin_name is not None

    events = []
    tapped = tap.verbose(
        _mixed_scan,
        on_step=lambda e: events.append(e),
        taps=[
            tap.watch_nan(sin_name, label="sin_nan"),
            tap.print(sin_name),
        ],
    )
    result = tapped(jnp.float32(0.0))
    jax.block_until_ready(result)
    captured = capsys.readouterr()

    # tap.print lines (no FAIL) — should have 4 lines (one per step)
    tap_lines = [
        line for line in captured.err.splitlines() if "[tap]" in line and "FAIL" not in line
    ]
    assert len(tap_lines) == 4, f"expected 4 tap.print lines, got {len(tap_lines)}"

    # watch_nan: sin with finite inputs produces finite outputs — no FAIL expected.
    # The key assertion is that both specs coexist without raising.
    fail_lines = [line for line in captured.err.splitlines() if "FAIL" in line]
    assert len(fail_lines) == 0, f"unexpected FAIL from finite sin: {fail_lines}"
