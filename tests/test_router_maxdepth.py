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
Router max_depth filter regression tests (issue #2).

The dynamic router must apply the RECEIVING context's max_depth to events that
arrive via a jit-cache hit from a foreign trace baked at a different max_depth.

Bug: a function compiled under ``tap.record(max_depth=None)`` bakes the router
at the original (no-limit) depth.  Calling that compiled artifact inside a
second ``tap.record(max_depth=0)`` used to route ALL-depth events to the
receiving recorder — ~50× host-callback waste for NUTS-style programs.

Fix location: ``_dynamic_router`` in src/jaxtap/_ashell.py — a host-side filter
  ``if ctx._max_depth is not None and event.path.count("/") > ctx._max_depth``
applied BEFORE alert/recorder/on_step firing.

Depth measure: ``event.path.count("/")`` — exactly the measure the walker uses
(_walker.py, lines 423, 444: ``depth = here.count("/")``, filter ``depth <= max_depth``).

Run with: uv run pytest tests/test_router_maxdepth.py -v
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import jaxtap as tap

# ---------------------------------------------------------------------------
# Shared helper: a two-level nested scan
# ---------------------------------------------------------------------------

N_OUTER = 4
N_INNER = 3


def _nested_scan(x0):
    """Outer scan (N_OUTER steps) with an inner scan (N_INNER steps) in the body.

    Path layout:
      depth-0 events: ``scan[0]``          (N_OUTER per call)
      depth-1 events: ``scan[0]/scan[0]``  (N_OUTER * N_INNER per call)
    """
    inner_xs = jnp.arange(float(N_INNER), dtype=jnp.float32)

    def outer_body(c, _):
        c2, _ = jax.lax.scan(lambda ci, xi: (ci + xi, ci), c, inner_xs)
        return c2, c2

    return jax.lax.scan(outer_body, x0, jnp.zeros(N_OUTER, dtype=jnp.float32))


# ---------------------------------------------------------------------------
# 1. Repro — regression test for issue #2
#
# Compile under max_depth=None, call inside max_depth=0.
# Receiving recorder must see ONLY depth-0 events (N_OUTER), not the inflated
# all-depth count (N_OUTER + N_OUTER * N_INNER = 4 + 12 = 16).
# ---------------------------------------------------------------------------


def test_router_maxdepth_cache_hit_filters_deep_events():
    """Cache-hit events from a max_depth=None trace are filtered by the receiving
    context's max_depth=0: only outer-scan (depth-0) events reach the recorder."""
    x0 = jnp.float32(1.0)

    # Compile and warm up under an unconstrained context (max_depth=None).
    with tap.record() as _compile_rec:
        f_jit = jax.jit(_nested_scan)
        jax.block_until_ready(f_jit(x0))
    # After exit, flush any pending callbacks from the compilation context.
    jax.effects_barrier()

    # Now call the already-compiled artifact inside a context with max_depth=0.
    with tap.record(max_depth=0) as rec:
        jax.block_until_ready(f_jit(x0))
    jax.effects_barrier()

    all_events = rec.events
    depth0 = [e for e in all_events if e.path == "scan[0]"]
    deep = [e for e in all_events if e.path.count("/") > 0]

    assert len(deep) == 0, (
        f"max_depth=0 receiving context must receive 0 deep events; "
        f"got {len(deep)}: {[e.path for e in deep]}"
    )
    assert len(depth0) == N_OUTER, (
        f"receiving context must still see {N_OUTER} depth-0 events; got {len(depth0)}"
    )


# ---------------------------------------------------------------------------
# 2. Cross-consumer alert variant
#
# Receiving context has max_depth=0 AND an alert that would fire on deep-path
# events if the filter were absent.  The alert must NOT fire for filtered events.
# ---------------------------------------------------------------------------


def test_router_maxdepth_alert_not_fired_for_filtered_events(capsys):
    """Alert registered on the receiving context (max_depth=0) must not fire for
    deep events that are filtered out by the max_depth host-side filter."""
    x0 = jnp.float32(1.0)
    alert_fired: list[str] = []

    def alert_fn(event: tap.TapEvent):
        # Would fire on ANY event that slips through; we record the path.
        alert_fired.append(event.path)
        return True  # always "truthy" → would emit a FAIL line

    # Compile under unconstrained context.
    with tap.record():
        f_jit = jax.jit(_nested_scan)
        jax.block_until_ready(f_jit(x0))
    jax.effects_barrier()

    # Receiving context: max_depth=0, alert fires on anything it sees.
    with tap.record(max_depth=0, alert=alert_fn):
        jax.block_until_ready(f_jit(x0))
    jax.effects_barrier()

    deep_alerts = [p for p in alert_fired if p.count("/") > 0]
    assert len(deep_alerts) == 0, (
        f"alert must not fire for filtered deep events; fired on: {deep_alerts}"
    )

    # Alert MAY fire for depth-0 events (not our concern here), but it must
    # not fire for anything deeper.
    captured = capsys.readouterr()
    fail_lines = [
        ln
        for ln in captured.err.splitlines()
        if ln.startswith("[tap] FAIL") and "/" in ln.split()[2]
    ]
    # FAIL lines contain the path; check none reference a deep path.
    for ln in fail_lines:
        path_token = ln.split()[2]  # third word is the path
        assert path_token.count("/") == 0, (
            f"FAIL line references a deep path that should have been filtered: {ln!r}"
        )


# ---------------------------------------------------------------------------
# 3a. No-regression: single context max_depth=None — all events delivered
#
# The None-check short-circuits; no events should be newly dropped.
# ---------------------------------------------------------------------------


def test_router_maxdepth_none_delivers_all_events():
    """Single context with max_depth=None: depth-0 AND depth-1 events are all delivered
    (filter is never triggered)."""
    x0 = jnp.float32(1.0)

    with tap.record(max_depth=None) as rec:
        f_jit = jax.jit(_nested_scan)
        jax.block_until_ready(f_jit(x0))
    jax.effects_barrier()

    depth0 = [e for e in rec.events if e.path == "scan[0]"]
    depth1 = [e for e in rec.events if e.path == "scan[0]/scan[0]"]

    assert len(depth0) == N_OUTER, (
        f"expected {N_OUTER} depth-0 events; got {len(depth0)}"
    )
    assert len(depth1) == N_OUTER * N_INNER, (
        f"expected {N_OUTER * N_INNER} depth-1 events; got {len(depth1)}"
    )


# ---------------------------------------------------------------------------
# 3b. No-regression: trace and receiving max_depth agree — count unchanged
#
# When compiled AND called under the same max_depth=0, only depth-0 events
# are emitted device-side; the host filter adds nothing new and the count
# must match what the device emits (N_OUTER).
# ---------------------------------------------------------------------------


def test_router_maxdepth_same_context_count_unchanged():
    """When trace-time and receiving max_depth both equal 0, event count is the
    same as without the router fix — no events are newly dropped."""
    x0 = jnp.float32(1.0)

    # Clear any prior cached artifact to ensure a fresh trace at max_depth=0.
    jax.clear_caches()

    with tap.record(max_depth=0) as rec:
        f_jit = jax.jit(_nested_scan)
        jax.block_until_ready(f_jit(x0))
    jax.effects_barrier()

    depth0 = [e for e in rec.events if e.path == "scan[0]"]
    deep = [e for e in rec.events if e.path.count("/") > 0]

    assert len(depth0) == N_OUTER, (
        f"expected {N_OUTER} depth-0 events; got {len(depth0)}"
    )
    assert len(deep) == 0, (
        f"max_depth=0 baked device-side must emit 0 deep events; got {len(deep)}"
    )


# ---------------------------------------------------------------------------
# 4. Alert-once budget: filtered events must not consume _carry_once_fired
#
# alert_once limits the FAIL stderr line to one per path.  alert_fn itself is
# called for each delivered event; _carry_once_fired is populated on the first
# truthy return.  The early `return` in _dynamic_router (before _carry_alert_fn)
# means filtered deep events never call alert_fn and never touch the once-set.
# A legit depth-0 event must therefore still produce its FAIL line even when
# deep events arrive on the same run.
# ---------------------------------------------------------------------------


def test_router_maxdepth_filtered_events_do_not_consume_alert_once_budget(capsys):
    """alert_once budget is not pre-consumed by filtered deep events.

    Compile under max_depth=None so the XLA artifact bakes full-depth callbacks.
    Call under max_depth=0, alert_once=True, always-truthy alert.

    Expected:
      - alert_fn is NOT called for any filtered deep event (early return before
        _carry_alert_fn — the once budget is never touched by them).
      - alert_fn IS called for each depth-0 event (N_OUTER times).
      - exactly 1 FAIL line is written to stderr (alert_once limits output).
    """
    x0 = jnp.float32(1.0)
    alert_calls: list[str] = []

    def alert_fn(event: tap.TapEvent) -> bool:
        alert_calls.append(event.path)
        return True

    # Compile under no depth limit so the artifact emits at all depths.
    jax.clear_caches()
    with tap.record():
        f_jit = jax.jit(_nested_scan)
        jax.block_until_ready(f_jit(x0))
    jax.effects_barrier()

    with tap.record(max_depth=0, alert=alert_fn, alert_once=True) as rec:
        jax.block_until_ready(f_jit(x0))
    jax.effects_barrier()

    depth0_events = [e for e in rec.events if e.path == "scan[0]"]
    deep_events = [e for e in rec.events if e.path.count("/") > 0]
    deep_alert_calls = [p for p in alert_calls if p.count("/") > 0]
    depth0_alert_calls = [p for p in alert_calls if p.count("/") == 0]

    # Filtered deep events must not reach alert_fn at all.
    assert len(deep_events) == 0, "deep events must be filtered from recorder"
    assert len(deep_alert_calls) == 0, (
        f"alert_fn must not be called for filtered deep events; called on: {deep_alert_calls}"
    )
    # Depth-0 events are delivered; alert_fn fires N_OUTER times.
    assert len(depth0_events) == N_OUTER, (
        f"expected {N_OUTER} depth-0 events; got {len(depth0_events)}"
    )
    assert len(depth0_alert_calls) == N_OUTER, (
        f"alert_fn must be called for each depth-0 event; got {len(depth0_alert_calls)}"
    )
    # alert_once: exactly 1 FAIL line for the depth-0 path.
    captured = capsys.readouterr()
    fail_lines = [ln for ln in captured.err.splitlines() if ln.startswith("[tap] FAIL")]
    assert len(fail_lines) == 1, (
        f"alert_once must produce exactly 1 FAIL line; got: {fail_lines}"
    )
