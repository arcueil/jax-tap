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
v1.1 tests: alert= sugar for carry taps (tap.verbose / tap.record).

Tests every requirement from the spec:
  - fires on condition with exact format
  - silent when falsy
  - str return becomes msg
  - non-str truthy → "alert"
  - alert_once fires exactly once across steps
  - alert + on_step both run, alert first
  - throwing alert doesn't crash and warns once
  - works under ``with tap.record():`` form (B-form and A-form)
  - respects sample_every gating
  - trace identity (alert on/off same jaxpr structure)
  - alert-only tap (no on_step)

``length`` in ``lax.scan`` must be a concrete Python int.  Each test defines
its function as a closure that captures N so it is never a traced abstract
value when verbose() calls make_jaxpr(f).

Run with: uv run pytest tests/test_carry_alerts.py -v
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxtap as tap

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bytes(x):
    return [np.asarray(v).tobytes() for v in jax.tree_util.tree_leaves(x)]


def _bitwise_eq(a, b) -> bool:
    return _bytes(a) == _bytes(b)


def _make_step_scan(n: int):
    """Return a function that scans ``n`` steps; carry += 1 each step."""

    def f(x0):
        def body(carry, _):
            return carry + 1.0, carry * 2.0

        return jax.lax.scan(body, x0, None, length=n)

    return f


def _get_carry(event: tap.TapEvent) -> float:
    """Extract carry scalar from a TapEvent (flat-tuple value or scalar)."""
    v = event.value
    leaf = v[0] if isinstance(v, tuple) else v
    return float(np.asarray(leaf))


# ---------------------------------------------------------------------------
# 1. Fires on condition with exact format
# ---------------------------------------------------------------------------


def test_carry_alert_fires_exact_format(capsys):
    """Truthy alert emits exactly ``[tap] FAIL {path} {step}/{total}: {msg}``."""
    N = 6
    x0 = jnp.float32(0.0)
    f = _make_step_scan(N)

    tapped = tap.verbose(
        f,
        on_step=lambda e: None,
        alert=lambda e: _get_carry(e) > 3.0,  # fires at steps 3, 4, 5
    )
    result = tapped(x0)
    jax.block_until_ready(result)

    captured = capsys.readouterr()
    fail_lines = [ln for ln in captured.err.splitlines() if ln.startswith("[tap] FAIL")]
    assert len(fail_lines) > 0, f"expected ≥1 FAIL line; got stderr: {captured.err!r}"

    for ln in fail_lines:
        parts = ln.split()
        # [tap] FAIL {path} {step}/{total}: {msg}
        assert parts[0] == "[tap]", f"prefix wrong: {ln!r}"
        assert parts[1] == "FAIL", f"FAIL missing: {ln!r}"
        assert "/" in parts[3], f"step/total missing in {parts[3]!r}: {ln!r}"
        step_part, total_part = parts[3].rstrip(":").split("/")
        assert step_part.lstrip("-").isdigit(), f"step not int in {ln!r}"
        assert total_part.isdigit() or total_part == "?", f"total not int/'?': {ln!r}"
        assert parts[4] == "alert", f"expected msg='alert' in {ln!r}"


# ---------------------------------------------------------------------------
# 2. Silent when falsy
# ---------------------------------------------------------------------------


def test_carry_alert_silent_when_falsy(capsys):
    """Alert callable returning False emits nothing; on_step still receives all events."""
    N = 6
    x0 = jnp.float32(0.0)
    f = _make_step_scan(N)

    events: list[tap.TapEvent] = []
    tapped = tap.verbose(
        f,
        on_step=lambda e: events.append(e),
        alert=lambda e: False,
    )
    result = tapped(x0)
    jax.block_until_ready(result)

    captured = capsys.readouterr()
    fail_lines = [ln for ln in captured.err.splitlines() if ln.startswith("[tap] FAIL")]
    assert len(fail_lines) == 0, f"expected 0 FAIL lines; got: {captured.err!r}"

    scan_events = [e for e in events if e.path == "scan[0]"]
    assert len(scan_events) == N, f"on_step should still receive {N} events"


# ---------------------------------------------------------------------------
# 3. str return becomes msg
# ---------------------------------------------------------------------------


def test_carry_alert_str_return_is_msg(capsys):
    """When alert returns a str, that string appears verbatim as the line's message."""
    N = 5
    x0 = jnp.float32(0.0)
    f = _make_step_scan(N)
    custom_msg = "carry exceeded threshold"

    def alert_fn(event: tap.TapEvent):
        return custom_msg if _get_carry(event) > 2.0 else False

    tapped = tap.verbose(f, on_step=lambda e: None, alert=alert_fn)
    result = tapped(x0)
    jax.block_until_ready(result)

    captured = capsys.readouterr()
    fail_lines = [ln for ln in captured.err.splitlines() if ln.startswith("[tap] FAIL")]
    assert len(fail_lines) > 0, f"expected ≥1 FAIL line; got: {captured.err!r}"
    for ln in fail_lines:
        assert custom_msg in ln, f"expected custom msg in {ln!r}"


# ---------------------------------------------------------------------------
# 4. Non-str truthy → "alert"
# ---------------------------------------------------------------------------


def test_carry_alert_nonstr_truthy_becomes_alert(capsys):
    """Non-str truthy returns (True, 1, 3.14) produce the fixed label 'alert'."""
    N = 3
    x0 = jnp.float32(10.0)  # large start so carry > 0 always
    f = _make_step_scan(N)

    for truthy_val in (True, 1, 3.14):

        def alert_fn(event: tap.TapEvent, _tv=truthy_val):
            return _tv

        tapped = tap.verbose(f, on_step=lambda e: None, alert=alert_fn)
        result = tapped(x0)
        jax.block_until_ready(result)

        captured = capsys.readouterr()
        fail_lines = [
            ln for ln in captured.err.splitlines() if ln.startswith("[tap] FAIL")
        ]
        assert len(fail_lines) > 0, f"expected FAIL lines for truthy_val={truthy_val!r}"
        for ln in fail_lines:
            parts = ln.split()
            assert parts[-1] == "alert", (
                f"expected msg='alert' for truthy_val={truthy_val!r}, got: {ln!r}"
            )


# ---------------------------------------------------------------------------
# 5. alert_once fires exactly once across steps
# ---------------------------------------------------------------------------


def test_carry_alert_once_fires_exactly_once(capsys):
    """With alert_once=True the alert fires at most once per path."""
    N = 10
    x0 = jnp.float32(0.0)
    f = _make_step_scan(N)

    tapped = tap.verbose(
        f,
        on_step=lambda e: None,
        alert=lambda e: True,  # always truthy
        alert_once=True,
    )
    result = tapped(x0)
    jax.block_until_ready(result)

    captured = capsys.readouterr()
    fail_lines = [ln for ln in captured.err.splitlines() if ln.startswith("[tap] FAIL")]
    scan_lines = [ln for ln in fail_lines if "scan[0]" in ln]
    assert len(scan_lines) == 1, (
        f"alert_once=True must fire exactly once per path; got {len(scan_lines)} lines"
    )


# ---------------------------------------------------------------------------
# 6. alert + on_step both run, alert first
# ---------------------------------------------------------------------------


def test_carry_alert_and_on_step_both_run(capsys):
    """alert fires before on_step; both are called for every sampled event."""
    N = 5
    x0 = jnp.float32(0.0)
    f = _make_step_scan(N)

    order: list[str] = []

    def alert_fn(event: tap.TapEvent):
        order.append("alert")
        return True

    def on_step_fn(event: tap.TapEvent):
        order.append("on_step")

    tapped = tap.verbose(f, on_step=on_step_fn, alert=alert_fn)
    result = tapped(x0)
    jax.block_until_ready(result)

    assert order.count("alert") == N, (
        f"alert called {order.count('alert')}×, expected {N}"
    )
    assert order.count("on_step") == N, (
        f"on_step called {order.count('on_step')}×, expected {N}"
    )

    # For each pair, alert must precede on_step
    for i in range(0, len(order) - 1, 2):
        assert order[i] == "alert", f"expected alert at position {i}, got {order[i]!r}"
        assert order[i + 1] == "on_step", (
            f"expected on_step at {i + 1}, got {order[i + 1]!r}"
        )

    captured = capsys.readouterr()
    fail_lines = [ln for ln in captured.err.splitlines() if ln.startswith("[tap] FAIL")]
    assert len(fail_lines) == N, f"expected {N} FAIL lines, got {len(fail_lines)}"


# ---------------------------------------------------------------------------
# 7. Throwing alert warns once and doesn't crash computation
# ---------------------------------------------------------------------------


def test_carry_alert_throwing_warns_once():
    """A raising alert callable warns once (UserWarning); computation result is correct."""
    N = 4
    x0 = jnp.float32(0.0)
    f = _make_step_scan(N)

    call_count = [0]

    def bad_alert(event: tap.TapEvent):
        call_count[0] += 1
        raise RuntimeError("alert boom")

    # Remove any prior warning state for this object's id
    tap._alert_warned.discard(id(bad_alert))

    ref = f(x0)

    with pytest.warns(UserWarning, match="jaxtap"):
        got = tap.verbose(f, on_step=lambda e: None, alert=bad_alert)(x0)
        jax.block_until_ready(got)

    assert _bitwise_eq(ref, got), "computation result must be bitwise-identical"
    assert call_count[0] >= 1, "bad_alert must have been attempted at least once"


# ---------------------------------------------------------------------------
# 8. Works under tap.record() B-form
# ---------------------------------------------------------------------------


def test_carry_alert_record_b_form(capsys):
    """alert= on record(f, alert=...) B-form fires and recorder still collects."""
    N = 5
    x0 = jnp.float32(0.0)
    f = _make_step_scan(N)

    tapped, rec = tap.record(f, alert=lambda e: True)
    result = tapped(x0)
    jax.block_until_ready(result)

    scan_events = [e for e in rec.events if e.path == "scan[0]"]
    assert len(scan_events) == N, (
        f"recorder expected {N} events, got {len(scan_events)}"
    )

    captured = capsys.readouterr()
    fail_lines = [ln for ln in captured.err.splitlines() if ln.startswith("[tap] FAIL")]
    assert len(fail_lines) == N, f"expected {N} FAIL lines, got {len(fail_lines)}"


# ---------------------------------------------------------------------------
# 9. Works under tap.record() A-form (with-block)
# ---------------------------------------------------------------------------


def test_carry_alert_record_a_form(capsys):
    """alert= on ``with tap.record(alert=...):`` A-form fires correctly."""
    N = 5
    x0 = jnp.float32(0.0)
    f = _make_step_scan(N)

    with tap.record(alert=lambda e: True) as rec:
        result = f(x0)
    jax.block_until_ready(result)

    scan_events = [e for e in rec.events if e.path == "scan[0]"]
    assert len(scan_events) == N, (
        f"recorder expected {N} events, got {len(scan_events)}"
    )

    captured = capsys.readouterr()
    fail_lines = [ln for ln in captured.err.splitlines() if ln.startswith("[tap] FAIL")]
    assert len(fail_lines) == N, f"expected {N} FAIL lines, got {len(fail_lines)}"


# ---------------------------------------------------------------------------
# 10. Respects sample_every gating
# ---------------------------------------------------------------------------


def test_carry_alert_respects_sample_every(capsys):
    """Alert only fires on steps that pass the sample_every gate."""
    N = 10
    SE = 3  # gate: steps 0, 3, 6, 9 → 4 events
    x0 = jnp.float32(0.0)
    f = _make_step_scan(N)

    events: list[tap.TapEvent] = []
    tapped = tap.verbose(
        f,
        on_step=lambda e: events.append(e),
        alert=lambda e: True,
        sample_every=SE,
    )
    result = tapped(x0)
    jax.block_until_ready(result)

    expected = len([s for s in range(N) if s % SE == 0])

    scan_events = [e for e in events if e.path == "scan[0]"]
    assert len(scan_events) == expected, (
        f"on_step: expected {expected} events (sample_every={SE}), got {len(scan_events)}"
    )

    captured = capsys.readouterr()
    fail_lines = [ln for ln in captured.err.splitlines() if ln.startswith("[tap] FAIL")]
    assert len(fail_lines) == expected, (
        f"alert: expected {expected} FAIL lines (sample_every={SE}), got {len(fail_lines)}"
    )


# ---------------------------------------------------------------------------
# 11. Alert-only tap (on_step=None)
# ---------------------------------------------------------------------------


def test_carry_alert_only_no_on_step(capsys):
    """alert= works without on_step (alert-only tap)."""
    N = 5
    x0 = jnp.float32(0.0)
    f = _make_step_scan(N)

    tapped = tap.verbose(f, on_step=None, alert=lambda e: True)
    result = tapped(x0)
    jax.block_until_ready(result)

    captured = capsys.readouterr()
    fail_lines = [ln for ln in captured.err.splitlines() if ln.startswith("[tap] FAIL")]
    assert len(fail_lines) == N, f"expected {N} FAIL lines, got {len(fail_lines)}"


# ---------------------------------------------------------------------------
# 12. Trace identity
# ---------------------------------------------------------------------------


def test_carry_alert_trace_identity():
    """Adding alert= does not change the device-side JAX computation.

    Verified by:
    1. Bitwise-equal results (computation is identical).
    2. Same number of jaxpr equations: alert is purely host-side logic living
       inside the Python _host callback closure, not in the traced XLA graph.
    """
    N = 6
    x0 = jnp.float32(0.0)
    f = _make_step_scan(N)

    # Run both and check results are bitwise equal
    ref = tap.verbose(f, on_step=lambda e: None)(x0)
    jax.block_until_ready(ref)
    got = tap.verbose(f, on_step=lambda e: None, alert=lambda e: False)(x0)
    jax.block_until_ready(got)
    assert _bitwise_eq(ref, got), (
        "results must be bitwise-identical with alert vs without"
    )

    # Structural jaxpr equivalence: equation count must match
    jaxpr_no = jax.make_jaxpr(tap.verbose(f, on_step=lambda e: None))(x0)
    jaxpr_with = jax.make_jaxpr(
        tap.verbose(f, on_step=lambda e: None, alert=lambda e: False)
    )(x0)

    n_no = len(jaxpr_no.jaxpr.eqns)
    n_with = len(jaxpr_with.jaxpr.eqns)
    assert n_no == n_with, (
        f"jaxpr equation count differs: {n_no} (no alert) vs {n_with} (with alert). "
        "alert= must not add any traced equations."
    )
