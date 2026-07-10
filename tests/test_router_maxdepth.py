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

The dynamic router must apply the RECEIVING context's max_depth to carry-tap
events that arrive via a jit-cache hit from a foreign trace baked at a
different max_depth.

Bug: a function compiled under ``tap.record(max_depth=None)`` bakes the router
at the original (no-limit) depth.  Calling that compiled artifact inside a
second ``tap.record(max_depth=0)`` used to route ALL-depth carry events to the
receiving recorder — ~50x host-callback waste for NUTS-style programs.

Fix: ``_dynamic_router`` applies the receiving context's max_depth host-side,
but ONLY to carry events (last path segment ``scan[k]`` or ``while[k]``).
Primitive-tap events (``taps=[tap.on(...)]``, ``tap.watch_nan()``) pass through
regardless of depth — the walker has no device-side max_depth gate for prim-taps
(walker.py:596-609), so the router must not add one either.  Silently filtering
prim-taps would kill NaN tripwires for consumers that combine max_depth=0
(blackjax issue-#5 dodge) with watch_nan.

Depth measure: ``event.path.count("/")`` — exactly the measure the walker uses
(_walker.py, lines 423, 444: ``depth = here.count("/")``, filter
``depth <= max_depth``).

Run with: uv run pytest tests/test_router_maxdepth.py -v
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import jaxtap as tap

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

N_OUTER = 4
N_INNER = 3


def _nested_scan(x0):
    """Outer scan (N_OUTER steps) with an inner scan (N_INNER steps) in the body.

    Path layout (carry events only — no prim taps in this helper):
      depth-0 carry: ``scan[0]``          (N_OUTER per call)
      depth-1 carry: ``scan[0]/scan[0]``  (N_OUTER * N_INNER per call)
    """
    inner_xs = jnp.arange(float(N_INNER), dtype=jnp.float32)

    def outer_body(c, _):
        c2, _ = jax.lax.scan(lambda ci, xi: (ci + xi, ci), c, inner_xs)
        return c2, c2

    return jax.lax.scan(outer_body, x0, jnp.zeros(N_OUTER, dtype=jnp.float32))


N_STEPS = 4  # scan length for prim-tap helpers


def _scan_with_sin(x0):
    """Scan with a sin primitive in the body.

    With taps=[tap.on("sin")]:
      depth-0 carry: ``scan[0]``       (N_STEPS per call)
      depth-1 prim:  ``scan[0]/sin[0]`` (N_STEPS per call)
    """

    def body(c, x):
        return c * 1.01 + jnp.sin(x), None

    return jax.lax.scan(body, x0, jnp.arange(float(N_STEPS)))[0]


# ---------------------------------------------------------------------------
# 1. Repro — regression test for issue #2
#
# Compile under max_depth=None, call inside max_depth=0.
# Receiving recorder must see ONLY depth-0 CARRY events (N_OUTER), not the
# inflated all-depth count (N_OUTER + N_OUTER * N_INNER = 4 + 12 = 16).
# ---------------------------------------------------------------------------


def test_router_maxdepth_cache_hit_filters_deep_carry_events():
    """Cache-hit carry events from a max_depth=None trace are filtered by the
    receiving context's max_depth=0: only outer-scan (depth-0) carry events
    reach the recorder."""
    x0 = jnp.float32(1.0)

    # Compile and warm up under an unconstrained context (max_depth=None).
    with tap.record() as _compile_rec:
        f_jit = jax.jit(_nested_scan)
        jax.block_until_ready(f_jit(x0))
    jax.effects_barrier()

    # Now call the already-compiled artifact inside a context with max_depth=0.
    with tap.record(max_depth=0) as rec:
        jax.block_until_ready(f_jit(x0))
    jax.effects_barrier()

    all_events = rec.events
    depth0 = [e for e in all_events if e.path == "scan[0]"]
    deep_carry = [
        e for e in all_events if e.path.count("/") > 0 and e.path.endswith("]")
    ]

    assert len(deep_carry) == 0, (
        f"max_depth=0 receiving context must receive 0 deep carry events; "
        f"got {len(deep_carry)}: {[e.path for e in deep_carry]}"
    )
    assert len(depth0) == N_OUTER, (
        f"receiving context must still see {N_OUTER} depth-0 events; got {len(depth0)}"
    )


# ---------------------------------------------------------------------------
# 2. Cross-consumer alert variant
#
# Receiving context has max_depth=0 AND an alert that would fire on deep-path
# carry events if the filter were absent.  The alert must NOT fire for
# filtered carry events.
# ---------------------------------------------------------------------------


def test_router_maxdepth_alert_not_fired_for_filtered_carry_events(capsys):
    """Alert registered on the receiving context (max_depth=0) must not fire for
    deep carry events that are filtered out by the max_depth host-side filter."""
    x0 = jnp.float32(1.0)
    alert_fired: list[str] = []

    def alert_fn(event: tap.TapEvent):
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

    deep_carry_alerts = [
        p
        for p in alert_fired
        if p.count("/") > 0 and p.rsplit("/", 1)[-1].startswith(("scan[", "while["))
    ]
    assert len(deep_carry_alerts) == 0, (
        f"alert must not fire for filtered deep carry events; fired on: {deep_carry_alerts}"
    )

    captured = capsys.readouterr()
    # No FAIL line should reference a deep carry path.
    for ln in captured.err.splitlines():
        if not ln.startswith("[tap] FAIL"):
            continue
        path_token = ln.split()[2]
        if path_token.rsplit("/", 1)[-1].startswith(("scan[", "while[")):
            assert path_token.count("/") == 0, (
                f"FAIL line references a deep carry path that should have been filtered: {ln!r}"
            )


# ---------------------------------------------------------------------------
# 3a. No-regression: single context max_depth=None — all events delivered
#
# The None-check short-circuits; no events should be newly dropped.
# ---------------------------------------------------------------------------


def test_router_maxdepth_none_delivers_all_events():
    """Single context with max_depth=None: depth-0 AND depth-1 carry events are
    all delivered (filter is never triggered)."""
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
# are emitted device-side; the host filter adds nothing new.
# ---------------------------------------------------------------------------


def test_router_maxdepth_same_context_count_unchanged():
    """When trace-time and receiving max_depth both equal 0, event count is the
    same as without the router fix — no carry events are newly dropped."""
    x0 = jnp.float32(1.0)

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
# 4. Alert-once budget: filtered carry events must not consume _carry_once_fired
# ---------------------------------------------------------------------------


def test_router_maxdepth_filtered_events_do_not_consume_alert_once_budget(capsys):
    """alert_once budget is not pre-consumed by filtered deep carry events.

    Compile under max_depth=None. Call under max_depth=0, alert_once=True,
    always-truthy alert.  Filtered deep carry events must not call alert_fn
    (early return before _carry_alert_fn) and must not touch _carry_once_fired.
    A legit depth-0 event must still produce its FAIL line.
    """
    x0 = jnp.float32(1.0)
    alert_calls: list[str] = []

    def alert_fn(event: tap.TapEvent) -> bool:
        alert_calls.append(event.path)
        return True

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

    assert len(deep_events) == 0, "deep carry events must be filtered from recorder"
    assert len(deep_alert_calls) == 0, (
        f"alert_fn must not be called for filtered deep carry events; called on: {deep_alert_calls}"
    )
    assert len(depth0_events) == N_OUTER, (
        f"expected {N_OUTER} depth-0 events; got {len(depth0_events)}"
    )
    assert len(depth0_alert_calls) == N_OUTER, (
        f"alert_fn must be called for each depth-0 event; got {len(depth0_alert_calls)}"
    )
    captured = capsys.readouterr()
    fail_lines = [ln for ln in captured.err.splitlines() if ln.startswith("[tap] FAIL")]
    assert len(fail_lines) == 1, (
        f"alert_once must produce exactly 1 FAIL line; got: {fail_lines}"
    )


# ---------------------------------------------------------------------------
# 5. Prim-tap survival: same-context max_depth=0 + tap.on(...)
#
# Regression guard for the AYS-2 finding: the router filter MUST NOT apply
# to primitive-tap events.  The walker has no device-side max_depth gate for
# prim-taps (walker.py:596-609); filtering them host-side would silently kill
# NaN tripwires for consumers that pair max_depth=0 with tap.watch_nan / tap.on.
#
# Before the scoped fix, the router applied max_depth to ALL events; a sin
# prim at scan[0]/sin[0] (depth=1) under max_depth=0 was silently dropped.
# ---------------------------------------------------------------------------


def test_router_prim_tap_survives_same_context_max_depth():
    """Prim-tap events (tap.on) at depth > max_depth must still reach the recorder
    in a same-context trace — max_depth only gates carry events, not prim-taps."""
    x0 = jnp.float32(1.0)

    jax.clear_caches()

    with tap.record(max_depth=0, taps=[tap.on("sin")]) as rec:
        f_jit = jax.jit(_scan_with_sin)
        jax.block_until_ready(f_jit(x0))
    jax.effects_barrier()

    carry = [e for e in rec.events if e.path == "scan[0]"]
    prim = [e for e in rec.events if "sin" in e.path]

    assert len(carry) == N_STEPS, (
        f"expected {N_STEPS} carry events at scan[0]; got {len(carry)}"
    )
    assert len(prim) == N_STEPS, (
        f"prim-tap events at scan[0]/sin[0] must NOT be filtered by max_depth=0; "
        f"got {len(prim)} (expected {N_STEPS}). "
        f"This is the AYS-2 regression: router was incorrectly dropping prim-taps."
    )
    # Confirm the prim-tap path is exactly what we expect (depth=1, past max_depth).
    prim_paths = {e.path for e in prim}
    assert prim_paths == {"scan[0]/sin[0]"}, f"unexpected prim-tap paths: {prim_paths}"


# ---------------------------------------------------------------------------
# 6. Prim-tap survival: cache-hit under max_depth=0
#
# The baked prim-tap callback routes through _dynamic_router at call time.
# A prim-tap event (depth=1) on a foreign-trace cache-hit under max_depth=0
# must still reach the receiving recorder.
# ---------------------------------------------------------------------------


def test_router_prim_tap_survives_cache_hit_max_depth():
    """Prim-tap events from a foreign-trace cache-hit must not be filtered by the
    receiving context's max_depth=0 — carry-only scoping applies to the router."""
    x0 = jnp.float32(1.0)

    # Compile under no depth limit with prim-taps baked in.
    jax.clear_caches()
    with tap.record(taps=[tap.on("sin")]):
        f_jit = jax.jit(_scan_with_sin)
        jax.block_until_ready(f_jit(x0))
    jax.effects_barrier()

    # Call the cached artifact under a stricter max_depth=0 context.
    with tap.record(max_depth=0, taps=[tap.on("sin")]) as rec:
        jax.block_until_ready(f_jit(x0))
    jax.effects_barrier()

    carry = [e for e in rec.events if e.path == "scan[0]"]
    prim = [e for e in rec.events if "sin" in e.path]

    assert len(carry) == N_STEPS, f"expected {N_STEPS} carry events; got {len(carry)}"
    assert len(prim) == N_STEPS, (
        f"prim-tap events must survive cache-hit under max_depth=0; "
        f"got {len(prim)} (expected {N_STEPS})"
    )
