"""
M1c tests: TapEvent.total, alert/label, watch_nan, tap.primitives.

All 86 prior tests remain green; this file adds ~10 new tests.

Run with: uv run pytest tests/test_m1c_alerts.py -v
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


def _chol_scan_n(x0, n):
    """Scan body computing a Cholesky; conditioning worsens → f32 goes NaN at step 7."""

    def body(carry, _):
        k = carry
        c = 1.0 - 10.0 ** (-jnp.minimum(k, 12.0))
        M = jnp.array([[1.0, c], [c, 1.0]], dtype=c.dtype)
        L = jnp.linalg.cholesky(M)
        logdens = -0.5 * 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        return carry + 1.0, logdens

    return jax.lax.scan(body, x0, None, length=n)


# ---------------------------------------------------------------------------
# TapEvent.total: scan-carry events
# ---------------------------------------------------------------------------


def test_total_scan_carry_events():
    """Scan-carry events have total == scan length."""
    N = 7
    x0 = jnp.float32(0.0)
    xs = jnp.arange(float(N), dtype=jnp.float32)

    def f(x, xs_):
        return jax.lax.scan(lambda c, x_: (c + x_, c * x_), x, xs_)

    _, events = _collect(f, x0, xs)
    scan_events = [e for e in events if e.path == "scan[0]"]
    assert len(scan_events) == N
    for e in scan_events:
        assert e.total == N, f"expected total={N}, got {e.total}"


def test_total_while_events_none():
    """While-carry events have total == None (length unknown at trace time)."""
    v0 = jnp.float32(0.0)

    def f(v):
        return jax.lax.while_loop(lambda c: c < 5.0, lambda c: c + 1.0, v)

    _, events = _collect(f, v0)
    while_events = [e for e in events if e.path == "while[0]"]
    assert len(while_events) > 0
    for e in while_events:
        assert e.total is None, f"while total must be None, got {e.total}"


def test_total_prim_tap_in_scan():
    """Primitive tap inside a scan carries total == scan length."""
    N = 8
    x0 = jnp.float32(1.0)

    def f(x):
        return _chol_scan_n(x, N)

    events: list[tap.TapEvent] = []
    got = tap.verbose(f, on_step=lambda e: events.append(e), taps=[tap.on("cholesky")])(x0)
    jax.block_until_ready(got)

    chol_events = [e for e in events if "cholesky" in e.path]
    assert len(chol_events) == N
    for e in chol_events:
        assert e.total == N, f"prim-tap in scan must have total={N}, got {e.total}"


def test_total_prim_tap_outside_loop():
    """Primitive tap outside any loop has total == None."""

    def f(x):
        M = jnp.array([[1.0, 0.5], [0.5, 1.0]])
        L = jnp.linalg.cholesky(M)
        return L[0, 0] + x

    x = jnp.float32(1.0)
    events: list[tap.TapEvent] = []
    got = tap.verbose(f, on_step=lambda e: events.append(e), taps=[tap.on("cholesky")])(x)
    jax.block_until_ready(got)

    chol_events = [e for e in events if "cholesky" in e.path]
    assert len(chol_events) == 1
    assert (
        chol_events[0].total is None
    ), f"outside-loop total must be None, got {chol_events[0].total}"


def test_total_prim_tap_while_body():
    """Primitive tap inside a while loop has total == None."""

    def f(x):
        def cond(c):
            return c[0] < 4.0

        def body(c):
            k, acc = c
            M = jnp.array([[1.0, 0.5], [0.5, 1.0]])
            L = jnp.linalg.cholesky(M)
            return (k + 1.0, acc + L[0, 0])

        return jax.lax.while_loop(cond, body, (x, jnp.float32(0.0)))

    x = jnp.float32(0.0)
    events: list[tap.TapEvent] = []
    got = tap.verbose(f, on_step=lambda e: events.append(e), taps=[tap.on("cholesky")])(x)
    jax.block_until_ready(got)

    chol_events = [e for e in events if "cholesky" in e.path]
    assert len(chol_events) > 0
    for e in chol_events:
        assert e.total is None, f"prim-tap in while must have total=None, got {e.total}"


# ---------------------------------------------------------------------------
# alert: terse line to stderr
# ---------------------------------------------------------------------------


def test_alert_fires_terse_line(capsys):
    """alert=... emits exactly one [tap] FAIL line per firing event to stderr."""
    N = 10
    x0 = jnp.float32(1.0)  # f32 → NaN at step ~7

    def f(x):
        return _chol_scan_n(x, N)

    events: list[tap.TapEvent] = []
    got = tap.verbose(
        f,
        on_step=lambda e: events.append(e),
        taps=[
            tap.on(
                "cholesky",
                select=lambda outs: jnp.all(jnp.isfinite(outs[0])),
                alert=lambda ok: not ok,
                label="NaN/Inf",
            )
        ],
    )(x0)
    jax.block_until_ready(got)

    captured = capsys.readouterr()
    stderr_lines = [ln for ln in captured.err.splitlines() if ln.strip()]

    # At least one alert line must appear (first non-finite step)
    alert_lines = [ln for ln in stderr_lines if ln.startswith("[tap] FAIL")]
    assert (
        len(alert_lines) > 0
    ), f"expected at least one [tap] FAIL line; got stderr: {captured.err!r}"

    # Format check: [tap] FAIL {path} {step}/{total}: {label}
    for ln in alert_lines:
        parts = ln.split()
        assert parts[0] == "[tap]", f"bad prefix: {ln!r}"
        assert parts[1] == "FAIL", f"bad keyword: {ln!r}"
        # parts[2] = path, parts[3] = step/total:, parts[4] = label
        assert "/" in parts[3], f"expected step/total in {parts[3]!r}: {ln!r}"
        step_part, total_part = parts[3].rstrip(":").split("/")
        assert step_part.lstrip("-").isdigit(), f"step not an int in {ln!r}"
        # total is an int (scan) or '?' (while/outside)
        assert total_part.isdigit() or total_part == "?", f"total not int or '?': {ln!r}"
        assert parts[4] == "NaN/Inf", f"label wrong: {ln!r}"


def test_alert_silent_when_finite(capsys):
    """alert does NOT fire when predicate returns False (all outputs finite)."""
    N = 4
    x0 = jnp.float32(1.0)

    def f(x):
        # Simple finite scan body — no cholesky singularity
        return jax.lax.scan(lambda c, _: (c + 1.0, c), x, None, length=N)

    # Tap add_any primitive with an alert that NEVER fires (condition always False)
    events: list[tap.TapEvent] = []
    got = tap.verbose(
        f,
        on_step=lambda e: events.append(e),
        taps=[tap.on("integer_pow", alert=lambda v: False)],
    )(x0)
    jax.block_until_ready(got)

    captured = capsys.readouterr()
    alert_lines = [ln for ln in captured.err.splitlines() if ln.startswith("[tap] FAIL")]
    assert (
        len(alert_lines) == 0
    ), f"expected 0 alert lines for always-False predicate; got: {captured.err!r}"

    # on_step still receives every event
    assert len([e for e in events if e.path == "scan[0]"]) == N


def test_alert_independent_of_on_step(capsys):
    """alert fires AND on_step receives every event (they are independent)."""
    N = 10
    x0 = jnp.float32(1.0)

    def f(x):
        return _chol_scan_n(x, N)

    events: list[tap.TapEvent] = []
    got = tap.verbose(
        f,
        on_step=lambda e: events.append(e),
        taps=[
            tap.on(
                "cholesky",
                select=lambda outs: jnp.all(jnp.isfinite(outs[0])),
                alert=lambda ok: not ok,
                label="NaN/Inf",
            )
        ],
    )(x0)
    jax.block_until_ready(got)

    chol_events = [e for e in events if "cholesky" in e.path]
    # on_step receives ALL N cholesky events
    assert len(chol_events) == N, f"on_step should receive all {N} events, got {len(chol_events)}"

    captured = capsys.readouterr()
    alert_lines = [ln for ln in captured.err.splitlines() if ln.startswith("[tap] FAIL")]
    # alert fires only for non-finite ones (f32 breaks at step ~7)
    non_finite_count = sum(1 for e in chol_events if not bool(e.value))
    assert (
        len(alert_lines) == non_finite_count
    ), f"expected {non_finite_count} alert lines, got {len(alert_lines)}"


def test_alert_raising_predicate_warn_once():
    """A raising alert predicate warns once, never propagates; results are correct."""
    N = 4
    x0 = jnp.float32(0.0)

    # Use a scan whose body directly calls sin (non-boundary, non-AD primitive).
    def f(x):
        def body(c, _):
            return jnp.sin(c) + 1.0, c

        return jax.lax.scan(body, x, None, length=N)

    call_count = [0]

    def bad_alert(v):
        call_count[0] += 1
        raise RuntimeError("alert boom")

    tap._alert_warned.discard(id(bad_alert))

    ref = f(x0)
    with pytest.warns(UserWarning, match="jaxtap"):
        got = tap.verbose(
            f,
            on_step=lambda e: None,
            taps=[tap.on("sin", alert=bad_alert)],
        )(x0)
        jax.block_until_ready(got)

    # Results bitwise correct
    ref_val = np.asarray(ref[0])
    got_val = np.asarray(got[0])
    assert np.array_equal(ref_val, got_val)
    # Alert callback was attempted at least once
    assert call_count[0] >= 1


def test_alert_while_loop_question_mark(capsys):
    """While-loop prim taps use '?' as the total marker in the alert line."""
    x0 = jnp.float32(0.0)

    def f(x):
        def cond(c):
            return c[0] < 4.0

        def body(c):
            k, acc = c
            M = jnp.array([[1.0, 0.5], [0.5, 1.0]])
            L = jnp.linalg.cholesky(M)
            return (k + 1.0, acc + L[0, 0])

        return jax.lax.while_loop(cond, body, (x, jnp.float32(0.0)))

    got = tap.verbose(
        f,
        on_step=lambda e: None,
        taps=[
            tap.on(
                "cholesky",
                select=lambda outs: jnp.bool_(False),  # always alert
                alert=lambda ok: not ok,
                label="test-label",
            )
        ],
    )(x0)
    jax.block_until_ready(got)

    captured = capsys.readouterr()
    alert_lines = [ln for ln in captured.err.splitlines() if ln.startswith("[tap] FAIL")]
    assert len(alert_lines) > 0, "expected at least one alert line"
    for ln in alert_lines:
        # While loop total should appear as '?'
        step_total_part = ln.split()[3]  # e.g. "0/?:"
        assert "?" in step_total_part, f"while total should be '?' in {ln!r}"


# ---------------------------------------------------------------------------
# watch_nan
# ---------------------------------------------------------------------------


def test_watch_nan_catches_nan_in_f32(capsys):
    """watch_nan catches non-finite cholesky in f32; silent in f64."""
    jax.config.update("jax_enable_x64", False)
    N = 25
    x0 = jnp.float32(1.0)

    def f(x):
        return _chol_scan_n(x, N)

    events: list[tap.TapEvent] = []
    got = tap.verbose(f, on_step=lambda e: events.append(e), taps=[tap.watch_nan("cholesky")])(x0)
    jax.block_until_ready(got)

    chol_events = [e for e in events if "cholesky" in e.path]
    assert len(chol_events) == N

    # All events have bool values (the select produces all-isfinite)
    first_bad = next((e.step for e in chol_events if not bool(e.value)), None)
    assert first_bad is not None, "f32 cholesky should go NaN"

    captured = capsys.readouterr()
    alert_lines = [ln for ln in captured.err.splitlines() if ln.startswith("[tap] FAIL")]
    assert len(alert_lines) > 0, "watch_nan should emit at least one alert line"
    # Label should be 'NaN/Inf'
    assert "NaN/Inf" in alert_lines[0], f"expected 'NaN/Inf' label in {alert_lines[0]!r}"
    # total should be the scan length
    step_total = alert_lines[0].split()[3]  # "7/25:"
    assert (
        step_total.endswith(f"/{N}:") or step_total == f"{first_bad}/{N}:"
    ), f"expected step/{N}: in {step_total!r}"


def test_watch_nan_silent_when_finite(capsys):
    """watch_nan does NOT alert when all outputs are finite."""
    jax.config.update("jax_enable_x64", True)
    N = 25
    x0 = jnp.float64(1.0)

    def f(x):
        return _chol_scan_n(x, N)

    tap.verbose(f, on_step=lambda e: None, taps=[tap.watch_nan("cholesky")])(x0)
    jax.block_until_ready(None)

    captured = capsys.readouterr()
    alert_lines = [ln for ln in captured.err.splitlines() if ln.startswith("[tap] FAIL")]
    assert len(alert_lines) == 0, f"f64 should produce no alerts; got: {captured.err!r}"

    # Reset to f32 for other tests
    jax.config.update("jax_enable_x64", False)


def test_watch_nan_with_form(capsys):
    """watch_nan works through the A-shell with-form."""
    jax.config.update("jax_enable_x64", False)
    N = 25
    x0 = jnp.float32(1.0)

    def f(x):
        return _chol_scan_n(x, N)

    with tap.record(taps=[tap.watch_nan("cholesky")]) as rec:
        f(x0)

    jax.block_until_ready(None)

    chol_events = [e for e in rec.events if "cholesky" in e.path]
    assert len(chol_events) == N

    first_bad = next((e.step for e in chol_events if not bool(e.value)), None)
    assert first_bad is not None, "f32 cholesky should have non-finite events"

    captured = capsys.readouterr()
    alert_lines = [ln for ln in captured.err.splitlines() if ln.startswith("[tap] FAIL")]
    assert len(alert_lines) > 0, "watch_nan with-form should emit alert lines"


# ---------------------------------------------------------------------------
# tap.primitives
# ---------------------------------------------------------------------------


def test_primitives_scan_and_cholesky():
    """primitives() finds 'scan' and 'cholesky' in a scan-cholesky program."""
    N = 5
    x0 = jnp.float32(1.0)

    def f(x):
        return _chol_scan_n(x, N)

    result = tap.primitives(f, x0)
    assert isinstance(result, dict)
    assert "scan" in result, f"'scan' not found in {sorted(result)}"
    assert "cholesky" in result, f"'cholesky' not found in {sorted(result)}"
    assert result["cholesky"] >= 1, f"cholesky count should be >= 1, got {result['cholesky']}"


def test_primitives_nested_and_cond():
    """primitives() descends into cond branches and nested scans."""
    N = 4
    xs = jnp.arange(float(N), dtype=jnp.float32)
    x0 = jnp.float32(1.0)

    def f(pred, x0_, xs_):
        def true_branch(c):
            out, _ = jax.lax.scan(lambda a, b: (a + b, a), c, xs_)
            return out

        def false_branch(c):
            return c * 2.0

        c1 = jax.lax.cond(pred, true_branch, false_branch, x0_)
        # outer scan wrapping the cond
        c2, _ = jax.lax.scan(lambda c, _: (c + 1.0, c), c1, None, length=3)
        return c2

    pred = jnp.bool_(True)
    result = tap.primitives(f, pred, x0, xs)

    # scan must appear (at least the outer one and the one in cond branch)
    assert "scan" in result, f"'scan' not in {sorted(result)}"
    assert result["scan"] >= 2, f"expected >= 2 scan occurrences, got {result['scan']}"
    # cond must appear
    assert "cond" in result, f"'cond' not in {sorted(result)}"


def test_primitives_returns_dict_of_ints():
    """primitives() return type is dict[str, int] with positive counts."""
    x0 = jnp.float32(0.5)
    xs = jnp.arange(4.0, dtype=jnp.float32)

    def f(x, xs_):
        return jax.lax.scan(lambda c, x_: (c + x_, c * x_), x, xs_)

    result = tap.primitives(f, x0, xs)
    assert isinstance(result, dict)
    for k, v in result.items():
        assert isinstance(k, str), f"key {k!r} is not a str"
        assert isinstance(v, int) and v > 0, f"count for {k!r} is {v!r}, expected positive int"
