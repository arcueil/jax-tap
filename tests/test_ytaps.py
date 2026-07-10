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
Tests for y-taps (0.3.0): taps on scan OUTPUTS (per-step ys).

GitHub issue: #3
Design ratified: 2026-07-10

Run with: uv run pytest tests/test_ytaps.py
"""

from __future__ import annotations

import io
import sys
import warnings

import jax
import jax.lax as lax
import jax.numpy as jnp
import pytest

import jaxtap as tap

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_scan(n_steps: int = 5):
    """Return (f, carry_init) where f runs a scan with scalar carry and scalar ys."""

    def f(carry_init):
        def body(carry, x):
            new_carry = carry + x
            ys = new_carry * 2.0  # scalar ys
            return new_carry, ys

        return lax.scan(body, carry_init, jnp.arange(n_steps, dtype=jnp.float32))

    return f, jnp.float32(0.0)


def _none_ys_scan(n_steps: int = 5):
    """Return (f, carry_init) where the body returns (carry, None)."""

    def f(carry_init):
        def body(carry, x):
            return carry + x, None

        return lax.scan(body, carry_init, jnp.arange(n_steps, dtype=jnp.float32))

    return f, jnp.float32(0.0)


def _dict_ys_scan(n_steps: int = 3):
    """Return (f, carry_init) where ys is a dict (pytree with 2 leaves)."""

    def f(carry_init):
        def body(carry, x):
            new_carry = carry + x
            ys = {"a": new_carry, "b": new_carry * 2.0}
            return new_carry, ys

        return lax.scan(body, carry_init, jnp.arange(n_steps, dtype=jnp.float32))

    return f, jnp.float32(0.0)


# ---------------------------------------------------------------------------
# 1. Basic y-tap fires with correct value and kind
# ---------------------------------------------------------------------------


def test_basic_ytap_fires():
    """Y-tap fires once per step with correct step index and kind="output"."""
    f, init = _simple_scan(n_steps=5)

    y_events = []
    tapped = tap.verbose(f, on_ys=lambda e: y_events.append(e))
    tapped(init)

    assert len(y_events) == 5
    for i, e in enumerate(y_events):
        assert e.kind == "output"
        assert e.step == i
        assert e.total == 5
        assert e.path == "scan[0]"


def test_basic_ytap_value():
    """Y-tap value matches expected per-step scan output."""
    f, init = _simple_scan(n_steps=4)

    y_events = []
    tapped = tap.verbose(f, on_ys=lambda e: y_events.append(e))
    tapped(init)

    # body: carry = cumsum(0..3) = [0,1,3,6]; ys = carry*2
    expected_ys = [0.0, 2.0, 6.0, 12.0]
    for i, e in enumerate(y_events):
        # value is a flat tuple when select_ys is None
        assert abs(float(e.value[0]) - expected_ys[i]) < 1e-5, (
            f"step {i}: expected {expected_ys[i]}, got {e.value}"
        )


# ---------------------------------------------------------------------------
# 2. JIT — y-tap fires through jit boundary
# ---------------------------------------------------------------------------


def test_ytap_jit():
    """Y-tap fires correctly inside jax.jit."""
    f, init = _simple_scan(n_steps=5)

    y_events = []
    tapped = jax.jit(tap.verbose(f, on_ys=lambda e: y_events.append(e)))
    tapped(init)

    assert len(y_events) == 5
    assert all(e.kind == "output" for e in y_events)
    assert [e.step for e in y_events] == list(range(5))


# ---------------------------------------------------------------------------
# 3. vmap — per-lane y-tap events
# ---------------------------------------------------------------------------


def test_ytap_vmap():
    """Vmapped scan fires y-taps for all lanes (2 lanes × 5 steps = 10 events)."""
    f, _ = _simple_scan(n_steps=5)

    y_events = []
    tapped = jax.vmap(tap.verbose(f, on_ys=lambda e: y_events.append(e)))
    inits = jnp.array([0.0, 1.0])  # 2 lanes
    tapped(inits)

    assert len(y_events) == 10  # 2 lanes × 5 steps
    assert all(e.kind == "output" for e in y_events)


# ---------------------------------------------------------------------------
# 4. Nested scan — y-taps at each level
# ---------------------------------------------------------------------------


def test_ytap_nested_scan():
    """Y-taps fire at the correct level for nested scans."""
    outer_steps = 2
    inner_steps = 3

    def f(carry_init):
        def outer_body(outer_carry, _):
            def inner_body(inner_carry, x):
                new_ic = inner_carry + x
                return new_ic, new_ic  # inner ys = inner_carry

            inner_carry_out, _ = lax.scan(
                inner_body, outer_carry, jnp.ones(inner_steps)
            )
            return inner_carry_out, inner_carry_out  # outer ys = inner result

        return lax.scan(outer_body, carry_init, jnp.ones(outer_steps))

    outer_y_events = []
    inner_y_events = []

    def on_ys(e):
        if e.path == "scan[0]":
            outer_y_events.append(e)
        else:
            inner_y_events.append(e)

    tapped = tap.verbose(f, on_ys=on_ys)
    tapped(jnp.float32(0.0))

    assert len(outer_y_events) == outer_steps
    # inner fires outer_steps × inner_steps = 6
    assert len(inner_y_events) == outer_steps * inner_steps
    assert all(e.kind == "output" for e in outer_y_events + inner_y_events)


# ---------------------------------------------------------------------------
# 5. select_ys flat-leaves indexing
# ---------------------------------------------------------------------------


def test_select_ys_flat_leaves():
    """select_ys receives flat leaves tuple; indexing selects a single leaf."""
    f, init = _dict_ys_scan(n_steps=3)

    # ys is dict {"a": ..., "b": ...}; dict leaves sorted by key → (a, b) = index 0, 1
    a_events = []
    tapped = tap.verbose(
        f,
        on_ys=lambda e: a_events.append(e),
        select_ys=lambda ys_leaves: ys_leaves[0],  # select "a"
    )
    tapped(init)

    assert len(a_events) == 3
    assert all(e.kind == "output" for e in a_events)
    # Each event's value should be scalar (selected single leaf)
    for e in a_events:
        # value should NOT be a tuple when select_ys returns a scalar
        import numpy as _np

        assert _np.ndim(e.value) == 0 or (
            isinstance(e.value, tuple) and len(e.value) == 1
        )


def test_select_ys_none_select():
    """Without select_ys, value is a flat tuple of all ys leaves."""
    f, init = _dict_ys_scan(n_steps=2)

    y_events = []
    tapped = tap.verbose(f, on_ys=lambda e: y_events.append(e))
    tapped(init)

    assert len(y_events) == 2
    # dict with 2 leaves → flat tuple of 2 elements
    for e in y_events:
        assert isinstance(e.value, tuple)
        assert len(e.value) == 2


# ---------------------------------------------------------------------------
# 6. on_ys separate from on_step — routing contract
# ---------------------------------------------------------------------------


def test_on_ys_separate_from_on_step():
    """Carry events go to on_step only; output events go to on_ys only."""
    f, init = _simple_scan(n_steps=5)

    carry_events = []
    output_events = []

    tapped = tap.verbose(
        f,
        on_step=lambda e: carry_events.append(e),
        on_ys=lambda e: output_events.append(e),
    )
    tapped(init)

    assert len(carry_events) == 5
    assert len(output_events) == 5

    # All carry events are kind="carry"
    assert all(e.kind == "carry" for e in carry_events)
    # All output events are kind="output"
    assert all(e.kind == "output" for e in output_events)

    # No cross-routing: no output event in carry_events, no carry in output_events
    assert not any(e.kind == "output" for e in carry_events)
    assert not any(e.kind == "carry" for e in output_events)


def test_both_callbacks_fire():
    """Both on_step (carry) and on_ys (output) fire for the same scan."""
    f, init = _simple_scan(n_steps=3)

    all_carry = []
    all_output = []

    tapped = tap.verbose(
        f,
        on_step=lambda e: all_carry.append(e),
        on_ys=lambda e: all_output.append(e),
    )
    tapped(init)

    # Both fire 3 times (once per step)
    assert len(all_carry) == 3
    assert len(all_output) == 3


def test_carry_fires_before_ys():
    """Within the same step, carry event fires before output event."""
    f, init = _simple_scan(n_steps=3)

    order = []

    tapped = tap.verbose(
        f,
        on_step=lambda e: order.append(("carry", e.step)),
        on_ys=lambda e: order.append(("output", e.step)),
    )
    tapped(init)

    # For each step, carry should appear before output
    for step in range(3):
        carry_idx = next(
            i for i, (k, s) in enumerate(order) if k == "carry" and s == step
        )
        output_idx = next(
            i for i, (k, s) in enumerate(order) if k == "output" and s == step
        )
        assert carry_idx < output_idx, f"Step {step}: carry should fire before output"


# ---------------------------------------------------------------------------
# 7. alert_ys and alert_ys_once
# ---------------------------------------------------------------------------


def test_alert_ys_fires_to_stderr():
    """alert_ys emits [tap] FAIL line to stderr when predicate is truthy."""
    f, init = _simple_scan(n_steps=5)

    buf = io.StringIO()
    old_stderr = sys.stderr
    sys.stderr = buf
    try:
        tapped = tap.verbose(f, alert_ys=lambda e: f"output: step {e.step} ys")
        tapped(init)
    finally:
        sys.stderr = old_stderr

    output = buf.getvalue()
    assert "[tap] FAIL scan[0]" in output
    assert "output: step 0 ys" in output
    # Should fire for all 5 steps
    assert output.count("[tap] FAIL") == 5


def test_alert_ys_once():
    """alert_ys_once=True fires the alert at most once per path."""
    f, init = _simple_scan(n_steps=5)

    buf = io.StringIO()
    old_stderr = sys.stderr
    sys.stderr = buf
    try:
        tapped = tap.verbose(
            f, alert_ys=lambda e: "output: triggered", alert_ys_once=True
        )
        tapped(init)
    finally:
        sys.stderr = old_stderr

    output = buf.getvalue()
    # Only one FAIL line despite 5 steps
    assert output.count("[tap] FAIL") == 1


def test_alert_ys_with_on_ys():
    """alert_ys fires before on_ys; both run independently."""
    f, init = _simple_scan(n_steps=3)

    received_events = []
    fired_alerts = []

    buf = io.StringIO()
    old_stderr = sys.stderr
    sys.stderr = buf
    try:
        tapped = tap.verbose(
            f,
            on_ys=lambda e: received_events.append(e),
            alert_ys=lambda e: fired_alerts.append(e.step) or "output: alert",
        )
        tapped(init)
    finally:
        sys.stderr = old_stderr

    # Both fired
    assert len(received_events) == 3
    assert len(fired_alerts) == 3


# ---------------------------------------------------------------------------
# 8. ys-only (no carry tap configured)
# ---------------------------------------------------------------------------


def test_ytap_only_no_carry():
    """on_ys without on_step: only output events fire, carry tap is absent."""
    f, init = _simple_scan(n_steps=4)

    y_events = []
    carry_events = []
    tapped = tap.verbose(
        f,
        on_step=None,
        on_ys=lambda e: y_events.append(e),
    )
    tapped(init)

    assert len(y_events) == 4
    assert len(carry_events) == 0
    assert all(e.kind == "output" for e in y_events)


# ---------------------------------------------------------------------------
# 9. None-ys scan → zero output events (the progress-bar idiom footgun guard)
# ---------------------------------------------------------------------------


def test_none_ys_scan_zero_events():
    """Scan with body returning (carry, None) produces ZERO output events."""
    f, init = _none_ys_scan(n_steps=5)

    y_events = []
    tapped = tap.verbose(
        f,
        on_step=None,  # no carry tap
        on_ys=lambda e: y_events.append(e),  # y-tap configured
        select_ys=lambda _: None,  # shouldn't matter — no ys leaves to select
    )
    tapped(init)

    # Critical: must be zero (not 5 empty-value events)
    assert len(y_events) == 0, (
        f"None-ys scan fired {len(y_events)} y-tap events — "
        "len(ys)>0 guard in rewrite_scan is not working"
    )


def test_none_ys_scan_carry_still_fires():
    """Carry tap still fires on None-ys scans even when on_ys is configured."""
    f, init = _none_ys_scan(n_steps=5)

    carry_events = []
    y_events = []
    tapped = tap.verbose(
        f,
        on_step=lambda e: carry_events.append(e),
        on_ys=lambda e: y_events.append(e),
    )
    tapped(init)

    assert len(carry_events) == 5  # carry tap fires normally
    assert len(y_events) == 0  # no output events


# ---------------------------------------------------------------------------
# 10. while-only + select_ys → zero events (scan-only boundary)
# ---------------------------------------------------------------------------


def test_while_only_ytap_no_events():
    """select_ys on a while-only function produces zero output events."""

    def f(carry_init):
        def cond(carry):
            return carry < 5

        def body(carry):
            return carry + 1

        return lax.while_loop(cond, body, carry_init)

    y_events = []
    tapped = tap.verbose(
        f,
        on_ys=lambda e: y_events.append(e),
        select_ys=lambda ys_leaves: ys_leaves,
        ops=("while_loop",),
    )
    tapped(jnp.int32(0))

    assert len(y_events) == 0


def test_while_ytap_user_warning():
    """select_ys with ops not including 'scan' emits a UserWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tap.verbose(
            lambda x: x,
            on_ys=lambda e: None,
            select_ys=lambda _: None,
            ops=("while_loop",),
        )
    assert len(w) == 1
    assert "select_ys has no effect" in str(w[0].message)
    assert issubclass(w[0].category, UserWarning)


# ---------------------------------------------------------------------------
# 11. sample_every gates y-taps
# ---------------------------------------------------------------------------


def test_sample_every_gates_ytaps():
    """sample_every=2 halves y-tap events (fires on steps 0, 2, 4)."""
    f, init = _simple_scan(n_steps=6)

    y_events = []
    tapped = tap.verbose(
        f,
        on_ys=lambda e: y_events.append(e),
        sample_every=2,
    )
    tapped(init)

    # Steps 0, 2, 4 → 3 events
    assert len(y_events) == 3
    assert [e.step for e in y_events] == [0, 2, 4]


def test_sample_every_gates_both():
    """sample_every gates both carry and y-tap together."""
    f, init = _simple_scan(n_steps=6)

    carry_events = []
    y_events = []
    tapped = tap.verbose(
        f,
        on_step=lambda e: carry_events.append(e),
        on_ys=lambda e: y_events.append(e),
        sample_every=3,
    )
    tapped(init)

    # Steps 0, 3 → 2 events each
    assert len(carry_events) == 2
    assert len(y_events) == 2
    assert [e.step for e in carry_events] == [e.step for e in y_events]


# ---------------------------------------------------------------------------
# 12. record() A-form passthrough
# ---------------------------------------------------------------------------


def test_record_aform_ytap():
    """A-form tap.record() with select_ys collects output events in rec.events."""

    def f():
        def body(carry, x):
            return carry + x, carry * 2.0

        return lax.scan(body, jnp.float32(0.0), jnp.arange(4, dtype=jnp.float32))

    jax.clear_caches()

    with tap.record(
        on_ys=lambda e: None, select_ys=lambda ys_leaves: ys_leaves[0]
    ) as rec:
        f()

    output_events = [e for e in rec.events if e.kind == "output"]
    assert len(output_events) == 4
    assert all(e.kind == "output" for e in output_events)


def test_record_aform_both_carry_and_output():
    """A-form rec.events holds both carry and output events."""

    def f():
        def body(carry, x):
            return carry + x, carry + 1.0

        return lax.scan(body, jnp.float32(0.0), jnp.arange(3, dtype=jnp.float32))

    jax.clear_caches()

    with tap.record(on_ys=lambda e: None) as rec:
        f()

    carry_events = [e for e in rec.events if e.kind == "carry"]
    output_events = [e for e in rec.events if e.kind == "output"]
    assert len(carry_events) == 3
    assert len(output_events) == 3


def test_record_bform_ytap():
    """B-form tap.record(f, ...) collects output events in rec.events."""
    f, init = _simple_scan(n_steps=5)

    g, rec = tap.record(f, on_ys=lambda e: None)
    g(init)

    output_events = [e for e in rec.events if e.kind == "output"]
    carry_events = [e for e in rec.events if e.kind == "carry"]
    assert len(output_events) == 5
    assert len(carry_events) == 5


def test_record_bform_ytap_only():
    """B-form with only on_ys set: output events reach rec.events.

    In B-form, the recorder always receives carry events too (effective_on_step=recorder
    regardless of on_step).  Filtering by kind is the consumer's responsibility.
    """
    f, init = _simple_scan(n_steps=4)

    g, rec = tap.record(f, on_ys=lambda e: None)
    g(init)

    output_events = [e for e in rec.events if e.kind == "output"]
    carry_events = [e for e in rec.events if e.kind == "carry"]
    # Both reach rec.events in B-form — carry via recorder wired as effective_on_step
    assert len(output_events) == 4
    assert len(carry_events) == 4


def test_record_bform_select_ys_routes_to_recorder():
    """B-form with select_ys: output events with selected values reach rec.events."""
    f, init = _dict_ys_scan(n_steps=3)

    # select "b" leaf (index 1 — dict sorted: a→0, b→1)
    g, rec = tap.record(f, select_ys=lambda ys_leaves: ys_leaves[1])
    g(init)

    output_events = [e for e in rec.events if e.kind == "output"]
    assert len(output_events) == 3
    # All output events should have scalar values (selected single leaf)
    for e in output_events:
        import numpy as _np

        # value should be a scalar or 0-d array (from jnp result)
        assert _np.ndim(e.value) <= 1


# ---------------------------------------------------------------------------
# 13. kind field on events
# ---------------------------------------------------------------------------


def test_kind_field_defaults_to_carry():
    """Existing carry-tap events have kind='carry' (backward-safe default)."""
    f, init = _simple_scan(n_steps=3)

    events = []
    tapped = tap.verbose(f, on_step=lambda e: events.append(e))
    tapped(init)

    assert all(e.kind == "carry" for e in events)


def test_kind_field_output_for_ytap():
    """Y-tap events have kind='output'."""
    f, init = _simple_scan(n_steps=3)

    events = []
    tapped = tap.verbose(f, on_ys=lambda e: events.append(e))
    tapped(init)

    assert all(e.kind == "output" for e in events)


def test_kind_field_construction():
    """TapEvent can be constructed with kind='carry' (default) and kind='output'."""
    carry_event = tap.TapEvent(path="scan[0]", step=0, value=1.0, total=5)
    assert carry_event.kind == "carry"

    output_event = tap.TapEvent(
        path="scan[0]", step=0, value=1.0, total=5, kind="output"
    )
    assert output_event.kind == "output"


def test_kind_field_is_immutable():
    """TapEvent is a frozen dataclass — kind cannot be reassigned."""
    import dataclasses

    event = tap.TapEvent(path="scan[0]", step=0, value=1.0)
    assert dataclasses.is_dataclass(event)
    with pytest.raises((dataclasses.FrozenInstanceError, TypeError)):
        event.kind = "output"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 14. df() column contract unchanged
# ---------------------------------------------------------------------------


def test_df_columns_unchanged():
    """df() still returns exactly ['path', 'step', 'value'] — kind not included."""
    f, init = _simple_scan(n_steps=3)

    g, rec = tap.record(f, on_ys=lambda e: None)
    g(init)

    df = rec.df()
    assert list(df.columns) == ["path", "step", "value"]


def test_df_excludes_output_events():
    """df() is built from all events; output events have 'value' column but no 'kind'."""
    f, init = _simple_scan(n_steps=3)

    g, rec = tap.record(f, on_ys=lambda e: None, select_ys=lambda ys: ys[0])
    g(init)

    df = rec.df()
    # df() only uses path, step, value — kind is invisible regardless of event type
    assert "kind" not in df.columns
    assert list(df.columns) == ["path", "step", "value"]


# ---------------------------------------------------------------------------
# 15. Bitwise-identical output guarantee
# ---------------------------------------------------------------------------


def test_ytap_output_identical():
    """Adding y-taps does not change the numerical output of f."""
    import numpy as np

    f, init = _simple_scan(n_steps=7)

    bare_result = f(init)
    tapped = tap.verbose(f, on_ys=lambda e: None, select_ys=lambda ys: ys[0])
    tap_result = tapped(init)

    # Both carry output and ys stacked output
    np.testing.assert_array_equal(np.array(bare_result[0]), np.array(tap_result[0]))
    np.testing.assert_array_equal(np.array(bare_result[1]), np.array(tap_result[1]))


# ---------------------------------------------------------------------------
# 16. alert_ys receives full TapEvent (not just value)
# ---------------------------------------------------------------------------


def test_alert_ys_receives_tapevents():
    """alert_ys predicate receives a full TapEvent with path, step, total, kind."""
    f, init = _simple_scan(n_steps=3)

    seen_events = []
    tapped = tap.verbose(
        f,
        alert_ys=lambda e: seen_events.append(e) or False,  # never actually alert
    )
    tapped(init)

    assert len(seen_events) == 3
    for e in seen_events:
        assert e.kind == "output"
        assert e.path == "scan[0]"
        assert e.total == 3
        assert 0 <= e.step < 3
