"""
M2 tests for jaxtap: tap-spec layer (sample_every, where, max_depth)
and collectors (FlightRecorder, JSONLWriter, read_jsonl, record).

Run with: uv run pytest tests/test_collectors.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxtap as tap
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers (mirror of test_jaxtap.py helpers)
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
# Small programs
# ---------------------------------------------------------------------------


def _simple_scan(x0, xs):
    return jax.lax.scan(lambda c, x: (c + x, c * x), x0, xs)


def _nested_scan(x0, xs):
    INNER_XS = jnp.arange(3.0, dtype=jnp.float32)

    def outer_body(c, x):
        c2, _ = jax.lax.scan(
            lambda c_, xi: (c_ * 1.001 + jnp.sin(xi), c_),
            c + x,
            INNER_XS,
        )
        return c2, c2 * 2.0

    return jax.lax.scan(outer_body, x0, xs)


def _mixed_cf(carry):
    """scan[0] followed by while[1] at the same (top) level."""
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


# ===========================================================================
# Tap-spec layer
# ===========================================================================

# ---------------------------------------------------------------------------
# sample_every
# ---------------------------------------------------------------------------


def test_sample_every_k2():
    """sample_every=2 fires on steps 0, 2, 4, … and nowhere else."""
    N = 6
    xs = jnp.arange(float(N), dtype=jnp.float32)
    ref, _ = _collect(_simple_scan, jnp.float32(1.0), xs)
    got, events = _collect(_simple_scan, jnp.float32(1.0), xs, sample_every=2)

    assert bitwise_eq(ref, got), "sample_every=2 must not perturb output"
    scan_events = [e for e in events if e.path == "scan[0]"]
    fired_steps = sorted(e.step for e in scan_events)
    assert fired_steps == [0, 2, 4], f"expected [0,2,4], got {fired_steps}"


def test_sample_every_k3():
    """sample_every=3 fires on steps 0, 3 for N=6."""
    N = 6
    xs = jnp.arange(float(N), dtype=jnp.float32)
    ref, _ = _collect(_simple_scan, jnp.float32(1.0), xs)
    got, events = _collect(_simple_scan, jnp.float32(1.0), xs, sample_every=3)

    assert bitwise_eq(ref, got), "sample_every=3 must not perturb output"
    scan_events = [e for e in events if e.path == "scan[0]"]
    fired_steps = sorted(e.step for e in scan_events)
    assert fired_steps == [0, 3], f"expected [0,3], got {fired_steps}"


def test_sample_every_1_is_default():
    """sample_every=1 fires on every step (same as the default)."""
    N = 5
    xs = jnp.arange(float(N), dtype=jnp.float32)
    _, events_default = _collect(_simple_scan, jnp.float32(1.0), xs)
    _, events_k1 = _collect(_simple_scan, jnp.float32(1.0), xs, sample_every=1)
    assert len(events_default) == len(events_k1) == N


def test_sample_every_invalid():
    """sample_every=0 raises ValueError."""
    with pytest.raises(ValueError, match="sample_every"):
        tap.verbose(lambda x: x, on_step=lambda e: None, sample_every=0)


# ---------------------------------------------------------------------------
# where predicate
# ---------------------------------------------------------------------------


def test_where_outer_only():
    """where filters to only the outer scan; inner scan is never fired."""
    x0 = jnp.float32(0.5)
    xs = jnp.linspace(0.0, 1.0, 4, dtype=jnp.float32)

    ref, _ = _collect(_nested_scan, x0, xs)
    got, events = _collect(_nested_scan, x0, xs, where=lambda p: p == "scan[0]")

    assert bitwise_eq(ref, got), "where filter must not perturb output"
    paths = {e.path for e in events}
    assert paths == {"scan[0]"}, f"expected only scan[0], got {paths}"


def test_where_no_match():
    """where predicate that matches nothing → zero events, bitwise identical output."""
    x0 = jnp.float32(0.5)
    xs = jnp.linspace(0.0, 1.0, 4, dtype=jnp.float32)

    ref, _ = _collect(_nested_scan, x0, xs)
    got, events = _collect(_nested_scan, x0, xs, where=lambda p: False)

    assert bitwise_eq(ref, got), "empty where must not perturb output"
    assert events == [], "expected no events when where always returns False"


def test_where_addressing_stable():
    """
    Filtering out scan[0] must not change the index of while[1].
    Addresses are assigned by a per-level counter that advances regardless
    of the where filter.
    """
    carry0 = (jnp.float32(0.0), jnp.float32(0.0))

    # Full instrumentation: both scan[0] and while[1]
    ref, full_events = _collect(_mixed_cf, carry0)
    full_paths = {e.path for e in full_events}
    assert "scan[0]" in full_paths
    assert "while[1]" in full_paths

    # Filter to while[1] only; scan[0] is skipped but counter still advances
    _, filtered_events = _collect(_mixed_cf, carry0, where=lambda p: p == "while[1]")
    filtered_paths = {e.path for e in filtered_events}
    assert "while[1]" in filtered_paths, "while[1] must still be addressable when scan[0] filtered"
    assert "scan[0]" not in filtered_paths


# ---------------------------------------------------------------------------
# max_depth
# ---------------------------------------------------------------------------


def test_max_depth_0():
    """max_depth=0 → only top-level CF nodes (depth 0) are tapped."""
    x0 = jnp.float32(0.5)
    xs = jnp.linspace(0.0, 1.0, 4, dtype=jnp.float32)

    ref, _ = _collect(_nested_scan, x0, xs)
    got, events = _collect(_nested_scan, x0, xs, max_depth=0)

    assert bitwise_eq(ref, got), "max_depth=0 must not perturb output"
    paths = {e.path for e in events}
    for p in paths:
        assert "/" not in p, f"max_depth=0 should not yield nested paths, got {p!r}"
    assert "scan[0]" in paths


def test_max_depth_1():
    """max_depth=1 → top-level and one-deep nodes (depth 0 and 1)."""
    x0 = jnp.float32(0.5)
    xs = jnp.linspace(0.0, 1.0, 4, dtype=jnp.float32)

    ref, _ = _collect(_nested_scan, x0, xs)
    got, events = _collect(_nested_scan, x0, xs, max_depth=1)

    assert bitwise_eq(ref, got), "max_depth=1 must not perturb output"
    paths = {e.path for e in events}
    # Top-level scan fires
    assert "scan[0]" in paths
    # One-deep inner scan fires
    assert "scan[0]/scan[0]" in paths
    # Nothing with two slashes (depth 2) fires
    for p in paths:
        assert p.count("/") <= 1, f"depth-2 path leaked through max_depth=1: {p!r}"


def test_max_depth_none_is_unlimited():
    """max_depth=None (default) is equivalent to no depth restriction."""
    x0 = jnp.float32(0.5)
    xs = jnp.linspace(0.0, 1.0, 4, dtype=jnp.float32)
    _, events_none = _collect(_nested_scan, x0, xs, max_depth=None)
    _, events_default = _collect(_nested_scan, x0, xs)
    assert len(events_none) == len(events_default)


# ---------------------------------------------------------------------------
# M0 invariants still hold after M2 changes
# ---------------------------------------------------------------------------


def test_m0_bitwise_still_holds():
    """All M0 bitwise-identity invariants survive the M2 changes."""
    x0 = jnp.float32(0.5)
    xs = jnp.linspace(0.0, 1.0, 4, dtype=jnp.float32)

    ref = _simple_scan(x0, xs)
    got, _ = _collect(_simple_scan, x0, xs)
    assert bitwise_eq(ref, got)

    ref2 = _nested_scan(x0, xs)
    got2, _ = _collect(_nested_scan, x0, xs)
    assert bitwise_eq(ref2, got2)


# ===========================================================================
# Collectors
# ===========================================================================

# ---------------------------------------------------------------------------
# FlightRecorder
# ---------------------------------------------------------------------------


def test_flight_recorder_accumulates():
    """FlightRecorder.events accumulates all fired TapEvents."""
    from jaxtap.collectors import FlightRecorder

    rec = FlightRecorder()
    tapped = tap.verbose(_simple_scan, on_step=rec)
    N = 5
    xs = jnp.arange(float(N), dtype=jnp.float32)
    result = tapped(jnp.float32(1.0), xs)
    jax.block_until_ready(result)

    assert len(rec.events) == N
    assert all(e.path == "scan[0]" for e in rec.events)
    assert [e.step for e in rec.events] == list(range(N))


def test_flight_recorder_df_scalar_select():
    """FlightRecorder.df() with a scalar select → one 'value' column."""
    pytest.importorskip("pandas")
    from jaxtap.collectors import FlightRecorder

    rec = FlightRecorder()
    N = 4
    xs = jnp.arange(float(N), dtype=jnp.float32)
    tapped = tap.verbose(_simple_scan, on_step=rec, select=lambda c: c[0])
    result = tapped(jnp.float32(1.0), xs)
    jax.block_until_ready(result)

    df = rec.df()
    assert list(df.columns) == ["path", "step", "value"], f"columns: {list(df.columns)}"
    assert len(df) == N
    assert list(df["path"]) == ["scan[0]"] * N
    assert list(df["step"]) == list(range(N))


def test_flight_recorder_df_dict_select():
    """FlightRecorder.df() with a dict select → one column per key."""
    pytest.importorskip("pandas")
    from jaxtap.collectors import FlightRecorder

    rec = FlightRecorder()
    N = 3
    xs = jnp.arange(float(N), dtype=jnp.float32)

    def _simple_scan2(x0, xs):
        return jax.lax.scan(lambda c, x: ((c[0] + x, c[1] * 2.0), c[0]), (x0, x0), xs)

    tapped = tap.verbose(
        _simple_scan2,
        on_step=rec,
        select=lambda leaves: {"a": leaves[0], "b": leaves[1]},
    )
    result = tapped(jnp.float32(1.0), xs)
    jax.block_until_ready(result)

    df = rec.df()
    assert "a" in df.columns and "b" in df.columns, f"columns: {list(df.columns)}"
    assert len(df) == N


def test_flight_recorder_df_no_pandas(monkeypatch):
    """FlightRecorder.df() raises ImportError with install hint when pandas absent."""
    from jaxtap.collectors import FlightRecorder

    rec = FlightRecorder()

    # Temporarily hide pandas from sys.modules
    monkeypatch.setitem(sys.modules, "pandas", None)  # type: ignore[arg-type]
    with pytest.raises((ImportError, SystemError)):
        rec.df()


# ---------------------------------------------------------------------------
# JSONLWriter / read_jsonl
# ---------------------------------------------------------------------------


def test_jsonl_roundtrip_scalar_carry():
    """Write events with scalar carry, read back and check path/step/value."""
    from jaxtap.collectors import JSONLWriter, read_jsonl

    N = 4
    xs = jnp.arange(float(N), dtype=jnp.float32)

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        fpath = Path(f.name)

    try:
        with JSONLWriter(fpath) as w:
            tapped = tap.verbose(_simple_scan, on_step=w)
            result = tapped(jnp.float32(1.0), xs)
            jax.block_until_ready(result)

        events = read_jsonl(fpath)
        assert len(events) == N
        assert all(e.path == "scan[0]" for e in events)
        assert [e.step for e in events] == list(range(N))
        # value round-trip: should be a tuple of scalars (or close to it)
        for e in events:
            assert isinstance(e.value, tuple), f"expected tuple, got {type(e.value)}"
    finally:
        fpath.unlink(missing_ok=True)


def test_jsonl_roundtrip_dict_select():
    """Write events with dict-select value, read back and verify structure."""
    from jaxtap.collectors import JSONLWriter, read_jsonl

    N = 3
    xs = jnp.arange(float(N), dtype=jnp.float32)

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        fpath = Path(f.name)

    try:
        with JSONLWriter(fpath) as w:
            tapped = tap.verbose(
                _simple_scan,
                on_step=w,
                select=lambda c: {"carry": c[0]},
            )
            result = tapped(jnp.float32(1.0), xs)
            jax.block_until_ready(result)

        events = read_jsonl(fpath)
        assert len(events) == N
        for e in events:
            assert isinstance(e.value, dict), f"expected dict, got {type(e.value)}"
            assert "carry" in e.value
    finally:
        fpath.unlink(missing_ok=True)


def test_jsonl_roundtrip_values_match():
    """Values from read_jsonl are numerically consistent with the originals."""
    from jaxtap.collectors import FlightRecorder, JSONLWriter, read_jsonl

    N = 4
    xs = jnp.arange(float(N), dtype=jnp.float32)

    rec = FlightRecorder()

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        fpath = Path(f.name)

    try:
        # Collect both ways simultaneously
        def _dual(event):
            rec(event)

        with JSONLWriter(fpath) as w:

            def _both(event):
                _dual(event)
                w(event)

            tapped = tap.verbose(_simple_scan, on_step=_both, select=lambda c: c[0])
            result = tapped(jnp.float32(1.0), xs)
            jax.block_until_ready(result)

        read_events = read_jsonl(fpath)
        assert len(read_events) == len(rec.events) == N
        for live, disk in zip(rec.events, read_events):
            assert live.path == disk.path
            assert live.step == disk.step
            # Values: live has JAX scalar, disk has numpy scalar
            assert np.isclose(float(np.asarray(live.value)), float(np.asarray(disk.value)))
    finally:
        fpath.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# record() helper
# ---------------------------------------------------------------------------


def test_record_helper_basic():
    """tap.record(f) returns (tapped_fn, recorder) and populates recorder.events."""
    N = 5
    xs = jnp.arange(float(N), dtype=jnp.float32)

    g, rec = tap.record(_simple_scan)
    result = g(jnp.float32(1.0), xs)
    jax.block_until_ready(result)

    assert len(rec.events) == N
    assert all(e.path == "scan[0]" for e in rec.events)


def test_record_helper_bitwise():
    """result from tap.record(f) is bitwise-identical to f."""
    x0 = jnp.float32(1.0)
    xs = jnp.arange(5.0, dtype=jnp.float32)

    ref = _simple_scan(x0, xs)
    g, _ = tap.record(_simple_scan)
    got = g(x0, xs)
    jax.block_until_ready(got)
    assert bitwise_eq(ref, got)


def test_record_helper_with_select():
    """tap.record(f, select=...) passes select through to verbose."""
    pytest.importorskip("pandas")

    N = 4
    xs = jnp.arange(float(N), dtype=jnp.float32)

    g, rec = tap.record(_simple_scan, select=lambda c: c[0])
    result = g(jnp.float32(1.0), xs)
    jax.block_until_ready(result)

    df = rec.df()
    assert "value" in df.columns
    assert len(df) == N


# ---------------------------------------------------------------------------
# reduce-on-device preserved (select only ships selector output)
# ---------------------------------------------------------------------------


def test_select_reduce_preserved():
    """
    With a select, only the selector output crosses the host boundary.
    TapEvent.value has the shape/structure of the selector's return.
    """
    N = 5
    xs = jnp.arange(float(N), dtype=jnp.float32)

    _, events = _collect(_simple_scan, jnp.float32(1.0), xs, select=lambda c: c[0])
    for e in events:
        # value should be the scalar (0-dim) result of c[0], not the full carry
        assert (
            np.asarray(e.value).ndim == 0
        ), f"expected scalar select output, got shape {np.asarray(e.value).shape}"


def test_select_dict_structure():
    """dict select → value is a dict with the expected keys."""
    N = 3
    xs = jnp.arange(float(N), dtype=jnp.float32)

    _, events = _collect(
        _simple_scan,
        jnp.float32(1.0),
        xs,
        select=lambda c: {"carry": c[0]},
    )
    for e in events:
        assert isinstance(e.value, dict)
        assert "carry" in e.value
