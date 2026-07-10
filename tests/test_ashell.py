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
M1b: A-shell tests — ``with tap.record():`` context manager form.

Tests for the patch machinery (registry, depth counter, foreign-patch
chaining, thread delegation) and for the user-facing UX contract
("remove the ``with`` line and nothing changes").

Run with: uv run pytest tests/test_ashell.py
"""

from __future__ import annotations

import threading

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxtap as tap

# ---------------------------------------------------------------------------
# Helpers (shared with test_jaxtap.py style)
# ---------------------------------------------------------------------------


def _bytes(x):
    return [np.asarray(v).tobytes() for v in jax.tree_util.tree_leaves(x)]


def bitwise_eq(a, b) -> bool:
    return _bytes(a) == _bytes(b)


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


def _simple_while(v0):
    LIM = jnp.float32(37.0)
    INC = jnp.float32(1.7)
    return jax.lax.while_loop(lambda c: c < LIM, lambda c: c + INC, v0)


def _chol_scan_n(x0, n):
    def body(carry, _):
        c = 1.0 - 10.0 ** (-carry)
        M = jnp.array([[1.0, c], [c, 1.0]], dtype=c.dtype)
        L = jnp.linalg.cholesky(M)
        logdens = -0.5 * 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        return carry + 1.0, logdens

    return jax.lax.scan(body, x0, None, length=n)


# ---------------------------------------------------------------------------
# 1. Basic with-form: scan → events collected, result bitwise-identical
# ---------------------------------------------------------------------------


def test_ashell_basic_scan():
    """Scan inside with block: events collected, result bitwise-identical to bare call."""
    N = 6
    x0 = jnp.float32(1.0)
    xs = jnp.arange(float(N), dtype=jnp.float32)

    ref = _simple_scan(x0, xs)

    with tap.record() as rec:
        got = _simple_scan(x0, xs)

    jax.block_until_ready(got)
    assert bitwise_eq(ref, got), "A-shell scan not bitwise identical"
    scan_events = [e for e in rec.events if e.path == "scan[0]"]
    assert len(scan_events) == N, f"expected {N} events, got {len(scan_events)}"
    assert [e.step for e in scan_events] == list(range(N))


# ---------------------------------------------------------------------------
# 2. Delete-the-with equivalence
# ---------------------------------------------------------------------------


def test_ashell_delete_with():
    """Outside any context: zero events, lax.scan and while_loop are originals."""
    from jaxtap._ashell import _original_scan, _original_while

    x0 = jnp.float32(1.0)
    xs = jnp.arange(5.0, dtype=jnp.float32)

    # Enter then exit — primitives must be back to original
    with tap.record() as rec_inner:
        pass  # no scan called

    assert jax.lax.scan is _original_scan, "lax.scan not restored after empty with"
    assert jax.lax.while_loop is _original_while, (
        "while_loop not restored after empty with"
    )
    assert len(rec_inner.events) == 0

    # Calling the scan outside a context emits nothing and returns correct value
    ref = _simple_scan(x0, xs)
    got = _simple_scan(x0, xs)
    assert bitwise_eq(ref, got)
    assert jax.lax.scan is _original_scan


# ---------------------------------------------------------------------------
# 3. No double instrumentation: nested scan event count == verbose() count
# ---------------------------------------------------------------------------


def test_ashell_no_double_instrumentation():
    """Nested scan inside context: event count must exactly match verbose()."""
    N_OUTER = 4
    INNER_N = 3
    x0 = jnp.float32(0.5)
    xs = jnp.linspace(0.0, 1.0, N_OUTER, dtype=jnp.float32)

    # Count events from the with-form
    with tap.record() as rec_ctx:
        _ = _nested_scan(x0, xs)

    jax.block_until_ready(None)

    # Count events from verbose() (the reference transform)
    events_verbose = []
    tap.verbose(_nested_scan, on_step=lambda e: events_verbose.append(e))(x0, xs)
    jax.block_until_ready(None)

    ctx_outer = len([e for e in rec_ctx.events if e.path == "scan[0]"])
    ctx_inner = len([e for e in rec_ctx.events if e.path == "scan[0]/scan[0]"])
    vb_outer = len([e for e in events_verbose if e.path == "scan[0]"])
    vb_inner = len([e for e in events_verbose if e.path == "scan[0]/scan[0]"])

    assert ctx_outer == vb_outer, (
        f"outer event count mismatch: with-form={ctx_outer}, verbose={vb_outer}"
    )
    assert ctx_inner == vb_inner, (
        f"inner event count mismatch: with-form={ctx_inner}, verbose={vb_inner}"
    )
    # Concrete values for the report
    assert ctx_outer == N_OUTER, f"outer expected {N_OUTER}, got {ctx_outer}"
    assert ctx_inner == N_OUTER * INNER_N, (
        f"inner expected {N_OUTER * INNER_N}, got {ctx_inner}"
    )


# ---------------------------------------------------------------------------
# 4. while_loop inside context: heartbeat events
# ---------------------------------------------------------------------------


def test_ashell_while_loop():
    """While loop inside context: heartbeat events with correct step count."""
    v0 = jnp.float32(0.3)
    LIM, INC = np.float32(37.0), np.float32(1.7)

    c = np.float32(v0)
    expected_iters = 0
    while c < LIM:
        c = c + INC
        expected_iters += 1

    ref = _simple_while(v0)

    with tap.record() as rec:
        got = _simple_while(v0)

    jax.block_until_ready(got)
    assert bitwise_eq(ref, got), "while_loop A-shell not bitwise identical"
    while_events = [e for e in rec.events if e.path == "while[0]"]
    assert len(while_events) == expected_iters, (
        f"expected {expected_iters} while events, got {len(while_events)}"
    )


# ---------------------------------------------------------------------------
# 5. Primitive taps via context
# ---------------------------------------------------------------------------


def test_ashell_primitive_taps():
    """tap.on('cholesky') inside context fires with correct step values."""
    N = 5
    x0 = jnp.float32(1.0)

    def f(x):
        return _chol_scan_n(x, N)

    ref = f(x0)

    with tap.record(taps=[tap.on("cholesky")]) as rec:
        got = f(x0)

    jax.block_until_ready(got)
    assert bitwise_eq(ref, got), "primitive tap broke bitwise identity in A-shell"
    chol_events = [e for e in rec.events if "cholesky" in e.path]
    assert len(chol_events) == N, (
        f"expected {N} cholesky events, got {len(chol_events)}"
    )
    assert sorted(e.step for e in chol_events) == list(range(N))


# ---------------------------------------------------------------------------
# 6. Exception in body: lax.scan restored, context deregistered
# ---------------------------------------------------------------------------


def test_ashell_exception_restores():
    """Exception inside the with block: lax.scan is restored and registry is clean."""
    from jaxtap._ashell import _context_registry, _original_scan, _original_while

    x0 = jnp.float32(1.0)
    xs = jnp.arange(4.0, dtype=jnp.float32)

    with pytest.raises(RuntimeError, match="intentional"):
        with tap.record() as rec:
            _ = _simple_scan(x0, xs)
            raise RuntimeError("intentional")

    assert jax.lax.scan is _original_scan, "lax.scan not restored after exception"
    assert jax.lax.while_loop is _original_while, (
        "while_loop not restored after exception"
    )
    assert len(_context_registry) == 0, "registry not empty after exception"
    # Events collected before the exception are accessible
    assert len(rec.events) > 0, "expected events before the exception"


# ---------------------------------------------------------------------------
# 7. Exit restores: after with, jax.lax.scan is pre-enter value
# ---------------------------------------------------------------------------


def test_ashell_exit_restores():
    """After ``with`` exits cleanly, jax.lax.scan and while_loop are restored."""
    from jaxtap._ashell import _original_scan, _original_while

    pre_scan = jax.lax.scan
    pre_while = jax.lax.while_loop

    with tap.record() as _:
        # Inside: primitives are patched
        from jaxtap._ashell import _patched_scan, _patched_while

        assert jax.lax.scan is _patched_scan
        assert jax.lax.while_loop is _patched_while

    # Outside: primitives are restored
    assert jax.lax.scan is pre_scan, "lax.scan not restored on clean exit"
    assert jax.lax.while_loop is pre_while, "while_loop not restored on clean exit"
    assert jax.lax.scan is _original_scan
    assert jax.lax.while_loop is _original_while


# ---------------------------------------------------------------------------
# 8. Foreign patch installed BEFORE our enter: we chain to it, restore to it
# ---------------------------------------------------------------------------


def test_ashell_foreign_patch_chain():
    """A foreign patch installed before enter is chained through and restored on exit."""
    from jaxtap._ashell import _original_scan

    foreign_calls: list[str] = []

    def _foreign_scan(f, init, xs=None, length=None, **kwargs):
        foreign_calls.append("foreign")
        return _original_scan(f, init, xs=xs, length=length, **kwargs)

    jax.lax.scan = _foreign_scan

    try:
        x0 = jnp.float32(1.0)
        xs = jnp.arange(5.0, dtype=jnp.float32)
        ref = _foreign_scan(lambda c, x: (c + x, c * x), x0, xs)
        foreign_calls.clear()

        with tap.record() as rec:
            got = _simple_scan(x0, xs)

        jax.block_until_ready(got)

        # After context: restored to foreign patch (not original)
        assert jax.lax.scan is _foreign_scan, (
            "lax.scan should be restored to foreign patch, not original"
        )
        # Events collected
        assert len(rec.events) > 0, "expected scan events"
        # Result correct
        assert bitwise_eq(ref, got), "foreign-patch-chain not bitwise identical"
    finally:
        jax.lax.scan = _original_scan


# ---------------------------------------------------------------------------
# 9. Foreign patch installed OVER us during context: exit does not clobber
# ---------------------------------------------------------------------------


def test_ashell_foreign_patch_over_us():
    """Foreign patch installed while context is active: our exit leaves it alone."""
    from jaxtap._ashell import _original_scan

    clobber_calls: list[str] = []

    def _clobber_scan(f, init, xs=None, length=None, **kwargs):
        clobber_calls.append("clobber")
        return _original_scan(f, init, xs=xs, length=length, **kwargs)

    # Reset warn-once flag from prior test runs so we can observe the warning
    import jaxtap._ashell as _ashell_mod

    _ashell_mod._clobber_scan_warned = False

    with pytest.warns(UserWarning, match="jaxtap"):
        with tap.record() as _:
            # Patch over us
            jax.lax.scan = _clobber_scan
        # __exit__ should see that jax.lax.scan is NOT _patched_scan → warn + leave alone

    assert jax.lax.scan is _clobber_scan, (
        "exit should NOT have clobbered the foreign patch installed over us"
    )
    # Cleanup
    jax.lax.scan = _original_scan
    # Reset warn flag for subsequent tests
    _ashell_mod._clobber_scan_warned = False


# ---------------------------------------------------------------------------
# 10. Two sequential contexts: both work, both restore
# ---------------------------------------------------------------------------


def test_ashell_sequential_contexts():
    """Two sequential contexts: each collects independently, both restore correctly."""
    from jaxtap._ashell import _original_scan

    x0 = jnp.float32(1.0)
    xs = jnp.arange(4.0, dtype=jnp.float32)
    ref = _simple_scan(x0, xs)

    with tap.record() as rec1:
        got1 = _simple_scan(x0, xs)

    with tap.record() as rec2:
        got2 = _simple_scan(x0, xs)

    jax.block_until_ready(got1)
    jax.block_until_ready(got2)

    assert bitwise_eq(ref, got1), "first context not bitwise identical"
    assert bitwise_eq(ref, got2), "second context not bitwise identical"
    assert len(rec1.events) == len(xs), f"rec1 expected {len(xs)} events"
    assert len(rec2.events) == len(xs), f"rec2 expected {len(xs)} events"
    assert jax.lax.scan is _original_scan, "lax.scan not restored after second context"


# ---------------------------------------------------------------------------
# 11. Re-entrant / nested contexts: inner context wins
# ---------------------------------------------------------------------------


def test_ashell_reentrant_contexts():
    """Nested with blocks: inner context wins for taps fired during its scope."""
    x0 = jnp.float32(1.0)
    xs_outer = jnp.arange(5.0, dtype=jnp.float32)
    xs_inner = jnp.arange(3.0, dtype=jnp.float32)

    with tap.record() as rec_outer:
        _ = _simple_scan(x0, xs_outer)  # 5 events → rec_outer

        with tap.record() as rec_inner:
            _ = _simple_scan(x0, xs_inner)  # 3 events → rec_inner (inner wins)

        # After inner exits, outer still active
        _ = _simple_scan(x0, xs_outer)  # 5 more events → rec_outer

    jax.block_until_ready(None)

    # Inner context collected the scan that ran while it was active
    assert len(rec_inner.events) == len(xs_inner), (
        f"inner expected {len(xs_inner)} events, got {len(rec_inner.events)}"
    )
    # Outer collected the scans that ran before and after the inner context
    assert len(rec_outer.events) == 2 * len(xs_outer), (
        f"outer expected {2 * len(xs_outer)} events, got {len(rec_outer.events)}"
    )


# ---------------------------------------------------------------------------
# 12. Thread delegation: enter on main, scan on worker → events attributed
# ---------------------------------------------------------------------------


def test_ashell_thread_delegation():
    """With ONE context active, a scan run on a worker thread is attributed to it."""
    x0 = jnp.float32(1.0)
    xs = jnp.arange(6.0, dtype=jnp.float32)
    ref = _simple_scan(x0, xs)

    worker_result = [None]
    worker_error = [None]

    def worker():
        try:
            worker_result[0] = _simple_scan(x0, xs)
            jax.block_until_ready(worker_result[0])
        except Exception as e:
            worker_error[0] = e

    with tap.record() as rec:
        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=30.0)

    if worker_error[0] is not None:
        raise worker_error[0]

    assert t.is_alive() is False, "worker thread did not finish"
    assert worker_result[0] is not None, "worker did not produce a result"
    assert bitwise_eq(ref, worker_result[0]), "worker result not bitwise identical"
    assert len(rec.events) > 0, (
        "expected events attributed from worker thread to single active context"
    )
    scan_events = [e for e in rec.events if e.path == "scan[0]"]
    assert len(scan_events) == len(xs), (
        f"expected {len(xs)} scan events from worker, got {len(scan_events)}"
    )


# ---------------------------------------------------------------------------
# 13. Phantom emission regression (AYS R1 probe 2)
# ---------------------------------------------------------------------------


def test_ashell_no_phantom_after_exit():
    """Callbacks baked into a jitted artifact must NOT append to the recorder after exit.

    Root cause (fixed): the baked on_step used to close over self._recorder.
    After __exit__, the compiled XLA artifact still fired → phantom emission.
    Fix: _dynamic_router is baked instead; it drops events when no context active.
    """
    N = 5
    xs_local = jnp.arange(float(N), dtype=jnp.float32)

    def _fresh_fn(x0):
        return jax.lax.scan(lambda c, x: (c * 1.01 + jnp.sin(x), c), x0, xs_local)

    fj = jax.jit(_fresh_fn)

    with tap.record() as rec:
        r_in = fj(jnp.float32(0.5))
        jax.block_until_ready(r_in)

    n_inside = len(rec.events)
    assert n_inside > 0, "expected events collected during context"

    # Call the same compiled artifact AFTER exit — must produce 0 new events
    r_out = fj(jnp.float32(0.5))
    jax.block_until_ready(r_out)
    n_after = len(rec.events)

    assert n_after == n_inside, (
        f"phantom emission: events grew from {n_inside} to {n_after} after context exit"
    )
    assert bitwise_eq(r_in, r_out), "jit call after exit not bitwise identical"


# ---------------------------------------------------------------------------
# 14. Cache-hit inside new context routes to the new recorder (AYS R1 probe 3)
# ---------------------------------------------------------------------------


def test_ashell_cache_hit_new_context():
    """A jit cache-hit inside a NEW context routes events to the NEW recorder.

    Root cause (fixed): baked on_step closed over the FIRST context's recorder.
    A cache-hit in a second context would append to the first (now-dead) recorder;
    the second recorder saw 0 events.
    Fix: _dynamic_router is baked; it resolves the active context at call time.
    """
    N = 5
    xs_local = jnp.arange(float(N), dtype=jnp.float32)

    def _fresh_fn2(x0):
        return jax.lax.scan(lambda c, x: (c * 1.01 + jnp.sin(x), c), x0, xs_local)

    fj2 = jax.jit(_fresh_fn2)
    ref = _fresh_fn2(jnp.float32(0.5))

    # Context A: bake the compiled artifact with dynamic router
    with tap.record() as rec_a:
        r_a = fj2(jnp.float32(0.5))
        jax.block_until_ready(r_a)

    assert len(rec_a.events) > 0, "expected events in context A"

    # Context B: cache-hit on the same compiled artifact → must route to rec_b
    with tap.record() as rec_b:
        r_b = fj2(jnp.float32(0.5))
        jax.block_until_ready(r_b)

    n_b = len(rec_b.events)
    assert n_b == len(rec_a.events), (
        f"cache-hit in new context: expected {len(rec_a.events)} events in rec_b, got {n_b}"
    )
    assert bitwise_eq(ref, r_b), "cache-hit result not bitwise identical"
    # rec_a must not grow (no phantom into dead recorder)
    assert len(rec_a.events) == len(rec_a.events), "rec_a grew after its context exited"


# ---------------------------------------------------------------------------
# 15. on_step passthrough — A-form
# ---------------------------------------------------------------------------


def test_ashell_on_step_aform():
    """A-form on_step: live callback fires alongside the recorder."""
    N = 6
    x0 = jnp.float32(1.0)
    xs = jnp.arange(float(N), dtype=jnp.float32)
    ref = _simple_scan(x0, xs)

    live_events: list[tap.TapEvent] = []

    with tap.record(on_step=lambda e: live_events.append(e)) as rec:
        got = _simple_scan(x0, xs)

    jax.block_until_ready(got)

    assert bitwise_eq(ref, got), "on_step passthrough broke bitwise identity"
    assert len(rec.events) == N, f"recorder expected {N} events, got {len(rec.events)}"
    assert len(live_events) == N, (
        f"live callback expected {N} events, got {len(live_events)}"
    )
    # Both should have seen the same events (same steps, same paths)
    assert [e.step for e in rec.events] == [e.step for e in live_events]


# ---------------------------------------------------------------------------
# 16. Raising on_step does not corrupt results or the recorder
# ---------------------------------------------------------------------------


def test_ashell_raising_on_step_aform():
    """A-form: raising on_step does not corrupt results; recorder still collects."""

    N = 5
    x0 = jnp.float32(0.0)
    xs = jnp.arange(float(N), dtype=jnp.float32)
    ref = _simple_scan(x0, xs)

    call_count = [0]

    def raising_cb(e: tap.TapEvent) -> None:
        call_count[0] += 1
        raise ValueError("boom")

    # Reset warn-once state
    tap._warned.discard(id(raising_cb))

    with pytest.warns(UserWarning, match="jaxtap"):
        with tap.record(on_step=raising_cb) as rec:
            got = _simple_scan(x0, xs)
            jax.block_until_ready(got)

    assert bitwise_eq(ref, got), "result corrupted by raising on_step in A-form"
    assert len(rec.events) == N, f"recorder expected {N} events, got {len(rec.events)}"
    assert call_count[0] == N, "raising callback was not attempted every step"


# ---------------------------------------------------------------------------
# 17. on_step passthrough — B-form
# ---------------------------------------------------------------------------


def test_ashell_on_step_bform():
    """B-form on_step: live callback fires alongside the recorder for record(f)."""
    N = 4
    x0 = jnp.float32(1.0)
    xs = jnp.arange(float(N), dtype=jnp.float32)
    ref = _simple_scan(x0, xs)

    live_events: list[tap.TapEvent] = []

    g, rec = tap.record(_simple_scan, on_step=lambda e: live_events.append(e))
    got = g(x0, xs)
    jax.block_until_ready(got)

    assert bitwise_eq(ref, got), "B-form on_step broke bitwise identity"
    assert len(rec.events) == N, f"recorder expected {N} events, got {len(rec.events)}"
    assert len(live_events) == N, (
        f"live callback expected {N} events, got {len(live_events)}"
    )


# ===========================================================================
# REMEDIATION regression tests (arm-S + arm-L findings)
# ===========================================================================

# ---------------------------------------------------------------------------
# 18. BLOCKER-1: positional args battery (arm-S)
# ---------------------------------------------------------------------------


def test_ashell_scan_positional_reverse():
    """scan(f, init, xs, None, True) — reverse positional — works inside context."""
    xs = jnp.arange(5.0, dtype=jnp.float32)

    def body(c, x):
        return c + x, c

    ref = jax.lax.scan(body, jnp.float32(0.0), xs, None, True)  # reverse positional

    with tap.record() as rec:
        got = jax.lax.scan(body, jnp.float32(0.0), xs, None, True)

    jax.block_until_ready(got)
    assert bitwise_eq(ref, got), "reverse=True positional: result wrong inside context"
    assert len(rec.events) == len(xs), "reverse=True positional: wrong event count"


def test_ashell_scan_positional_unroll():
    """scan(f, init, xs, None, False, 2) — unroll positional — works inside context."""
    xs = jnp.arange(5.0, dtype=jnp.float32)

    def body(c, x):
        return c + x, c

    ref = jax.lax.scan(
        body, jnp.float32(0.0), xs, None, False, 2
    )  # unroll=2 positional

    with tap.record() as rec:
        got = jax.lax.scan(body, jnp.float32(0.0), xs, None, False, 2)

    jax.block_until_ready(got)
    assert bitwise_eq(ref, got), "unroll=2 positional: result wrong inside context"
    assert len(rec.events) == len(xs), "unroll=2 positional: wrong event count"


# ---------------------------------------------------------------------------
# 19. MAJOR-1: sequential top-level scans get unique paths (arm-S)
# ---------------------------------------------------------------------------


def test_ashell_sequential_scan_paths():
    """Two sequential top-level scans in one context: paths must be scan[0], scan[1]."""
    xs = jnp.arange(5.0, dtype=jnp.float32)

    def two_scans(x0):
        a, _ = jax.lax.scan(lambda c, x: (c + x, c), x0, xs)
        b, _ = jax.lax.scan(lambda c, x: (c * x + 1, c), a, xs)
        return b

    ref = two_scans(jnp.float32(1.0))

    with tap.record() as rec:
        got = two_scans(jnp.float32(1.0))

    jax.block_until_ready(got)
    assert bitwise_eq(ref, got), "sequential scans: result wrong"
    paths = sorted({e.path for e in rec.events})
    assert paths == ["scan[0]", "scan[1]"], f"expected [scan[0], scan[1]], got {paths}"
    assert sum(1 for e in rec.events if e.path == "scan[0]") == len(xs)
    assert sum(1 for e in rec.events if e.path == "scan[1]") == len(xs)


def test_ashell_sequential_scan_while_paths():
    """Sequential scan then while in one context: paths must be scan[0], while[1]."""
    xs = jnp.arange(5.0, dtype=jnp.float32)

    def scan_then_while(x0):
        a, _ = jax.lax.scan(lambda c, x: (c + x, c), x0, xs)
        b = jax.lax.while_loop(lambda c: c < 50.0, lambda c: c + 1.0, a)
        return b

    ref = scan_then_while(jnp.float32(1.0))

    with tap.record() as rec:
        got = scan_then_while(jnp.float32(1.0))

    jax.block_until_ready(got)
    assert bitwise_eq(ref, got), "scan+while: result wrong"
    paths = sorted({e.path for e in rec.events})
    assert "scan[0]" in paths, f"scan[0] missing from {paths}"
    assert "while[1]" in paths, f"while[1] missing from {paths}"


def test_ashell_python_loop_five_scans():
    """Five separate top-level scans via Python loop: paths scan[0..4], 5*N events total."""
    N = 5
    xs = jnp.arange(float(N), dtype=jnp.float32)

    def py_loop(x0):
        outs = []
        for i in range(N):
            r, _ = jax.lax.scan(lambda c, x: (c + x + i, c), x0, xs)
            outs.append(r)
        return jnp.stack(outs)

    ref = py_loop(jnp.float32(0.0))

    with tap.record() as rec:
        got = py_loop(jnp.float32(0.0))

    jax.block_until_ready(got)
    assert bitwise_eq(ref, got), "python-loop 5 scans: result wrong"
    paths = sorted({e.path for e in rec.events})
    expected_paths = [f"scan[{i}]" for i in range(N)]
    assert paths == expected_paths, f"expected {expected_paths}, got {paths}"
    assert len(rec.events) == N * N, (
        f"expected {N * N} total events, got {len(rec.events)}"
    )


def test_ashell_scan_cond_scan_divergence_pin():
    """Conformance pin: scan/cond/scan diverges from verbose() addressing.

    verbose() addresses the top-level jaxpr equations: scan[0], cond[1], scan[2].
    The A-shell only intercepts scan/while_loop calls, so cond is invisible to it.
    A-shell: scan[0], scan[1] (two intercepts, consecutive indices).
    verbose(): scan[0], scan[2] (cond increments the counter to 1, scan gets 2).

    This is a documented boundary, not a bug.  The test pins the behavior.
    """
    xs = jnp.arange(4.0, dtype=jnp.float32)

    def scan_cond_scan(x0):
        # top-level: scan, then cond (invisible to A-shell), then scan
        a, _ = jax.lax.scan(lambda c, x: (c + x, c), x0, xs)
        b = jax.lax.cond(a > 10.0, lambda _: a * 2.0, lambda _: a * 0.5, a)
        c, _ = jax.lax.scan(lambda carry, x: (carry + x, carry), b, xs)
        return c

    ref = scan_cond_scan(jnp.float32(0.0))

    # A-shell: two intercepted calls -> scan[0], scan[1]
    with tap.record() as rec:
        got = scan_cond_scan(jnp.float32(0.0))
    jax.block_until_ready(got)
    assert bitwise_eq(ref, got), "scan/cond/scan: result wrong in A-shell"
    ash_paths = sorted({e.path for e in rec.events})
    assert ash_paths == [
        "scan[0]",
        "scan[1]",
    ], f"A-shell: expected [scan[0], scan[1]], got {ash_paths}"

    # verbose(): n_cf advances through scan[0], cond[1], scan[2]
    vb_events: list = []
    tap.verbose(scan_cond_scan, on_step=vb_events.append)(jnp.float32(0.0))
    jax.block_until_ready(None)
    vb_paths = sorted({e.path for e in vb_events})
    assert vb_paths == [
        "scan[0]",
        "scan[2]",
    ], f"verbose: expected [scan[0], scan[2]], got {vb_paths}"


# ---------------------------------------------------------------------------
# 20. L1: re-enter same context raises RuntimeError
# ---------------------------------------------------------------------------


def test_ashell_reenter_raises():
    """Reusing the same context object in nested with-blocks raises RuntimeError."""
    ctx = tap.record()
    with pytest.raises(RuntimeError, match="not re-entrant"):
        with ctx:
            with ctx:  # second enter on same object must raise
                pass
    # After the exception, primitives must be restored.
    from jaxtap._ashell import _context_registry, _original_scan, _original_while

    assert jax.lax.scan is _original_scan, "lax.scan not restored after re-enter error"
    assert jax.lax.while_loop is _original_while, (
        "while_loop not restored after re-enter error"
    )
    assert len(_context_registry) == 0, "registry not empty after re-enter error"


# ---------------------------------------------------------------------------
# 21. emergency_restore restores correctly
# ---------------------------------------------------------------------------


def test_ashell_emergency_restore():
    """jaxtap.emergency_restore() restores patched primitives and clears state."""
    import jaxtap
    from jaxtap._ashell import _original_scan, _original_while, _patched_scan

    # Force a leaking state (simulate a crash: manually enter, never exit).
    ctx = tap.record()
    ctx.__enter__()
    assert jax.lax.scan is _patched_scan, "scan should be patched before restore"

    jaxtap.emergency_restore()

    assert jax.lax.scan is _original_scan, "scan not restored by emergency_restore"
    assert jax.lax.while_loop is _original_while, (
        "while_loop not restored by emergency_restore"
    )
    from jaxtap._ashell import _context_registry, _session_scan, _session_while

    assert len(_context_registry) == 0, "registry not empty after emergency_restore"
    assert _session_scan is None, "_session_scan not cleared"
    assert _session_while is None, "_session_while not cleared"

    # Detach the orphaned finalizer to avoid spurious cleanup later.
    if ctx._finalizer is not None:
        ctx._finalizer.detach()
        ctx._finalizer = None


# ---------------------------------------------------------------------------
# 22. L3/L4: double-exit no-op + no warning poisoning
# ---------------------------------------------------------------------------


def test_ashell_double_exit_noop():
    """Double __exit__ is a harmless no-op: no warning, scan still original."""
    import warnings

    from jaxtap._ashell import _original_scan

    ctx = tap.record()
    with ctx:
        pass
    # self._key is now None; second exit should be silent no-op
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ctx.__exit__(None, None, None)

    jaxtap_warns = [x for x in w if "jaxtap" in str(x.message)]
    assert len(jaxtap_warns) == 0, (
        f"double-exit emitted unexpected warning: {jaxtap_warns}"
    )
    assert jax.lax.scan is _original_scan, "scan was corrupted by double-exit"


def test_ashell_bogus_exit_does_not_poison_warnonce():
    """exit-without-enter must not poison warn-once flag; real clobber still warns."""
    import warnings

    import jaxtap._ashell as A

    # Reset state.
    A._clobber_scan_warned = False
    A._clobber_while_warned = False

    # Bogus exit: never entered, _key is None.
    ctx = tap.record()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ctx.__exit__(None, None, None)
    assert not w, f"bogus exit emitted unexpected warnings: {w}"
    assert A._clobber_scan_warned is False, "bogus exit poisoned warn-once flag"

    # Genuine clobber: should warn.
    with warnings.catch_warnings(record=True) as w_real:
        warnings.simplefilter("always")
        with tap.record():
            jax.lax.scan = A._original_scan  # foreign clobber during context
        # exit sees non-patched scan → warns
    jax.lax.scan = A._original_scan  # cleanup
    A._clobber_scan_warned = False  # cleanup

    real_warned = any("jaxtap" in str(x.message) for x in w_real)
    assert real_warned, "genuine clobber warning was silenced"


# ---------------------------------------------------------------------------
# 23. L6: session-scoped warn-once (two independent clobbers both warn)
# ---------------------------------------------------------------------------


def test_ashell_session_scoped_warnonce():
    """Two independent sessions each with a clobber: both emit a warning."""
    import warnings

    from jaxtap._ashell import _original_scan

    # Session 1 clobber.
    with warnings.catch_warnings(record=True) as w1:
        warnings.simplefilter("always")
        with tap.record():
            jax.lax.scan = lambda f, init, xs=None, length=None, **kw: _original_scan(
                f, init, xs=xs, length=length, **kw
            )
    jax.lax.scan = _original_scan
    n1 = sum("jaxtap" in str(x.message) for x in w1)

    # Session 2 clobber — flags reset on new session start.
    with warnings.catch_warnings(record=True) as w2:
        warnings.simplefilter("always")
        with tap.record():
            jax.lax.scan = lambda f, init, xs=None, length=None, **kw: _original_scan(
                f, init, xs=xs, length=length, **kw
            )
    jax.lax.scan = _original_scan
    n2 = sum("jaxtap" in str(x.message) for x in w2)

    assert n1 >= 1, "first clobber should have warned"
    assert n2 >= 1, "second clobber should also warn (session-scoped reset)"


# ---------------------------------------------------------------------------
# 24. L2: manual-enter GC self-heal
# ---------------------------------------------------------------------------


def test_ashell_gc_selfheal():
    """Manual __enter__ without __exit__: dropping the context allows GC to restore."""
    import gc

    from jaxtap._ashell import (
        _context_registry,
        _original_scan,
        _original_while,
        _patched_scan,
    )

    # Clean state.
    assert jax.lax.scan is _original_scan
    assert len(_context_registry) == 0

    def leaky_enter():
        ctx = tap.record()
        ctx.__enter__()
        # ctx falls out of scope here without __exit__

    leaky_enter()
    gc.collect()  # CPython ref-counting handles it immediately; gc.collect for PyPy

    assert jax.lax.scan is not _patched_scan, "GC self-heal: scan still patched"
    assert jax.lax.scan is _original_scan, "GC self-heal: scan not restored to original"
    assert jax.lax.while_loop is _original_while, (
        "GC self-heal: while_loop not restored"
    )
    assert len(_context_registry) == 0, "GC self-heal: registry not empty"


# ---------------------------------------------------------------------------
# 25. L5: verbose()/record(f) inside context — single instrumentation
# ---------------------------------------------------------------------------


def test_ashell_verbose_inside_context_no_double():
    """verbose(f) called inside an active context: user callback counts correct,
    context recorder sees 0 events from the explicitly-tapped call."""
    N = 5
    x0 = jnp.float32(1.0)
    xs = jnp.arange(float(N), dtype=jnp.float32)
    ref = _simple_scan(x0, xs)

    user_events: list = []

    with tap.record() as rec_ctx:
        g = tap.verbose(_simple_scan, on_step=lambda e: user_events.append(e))
        got = g(x0, xs)
        jax.block_until_ready(got)

    assert bitwise_eq(ref, got), "verbose inside context: result wrong"
    assert len(user_events) == N, (
        f"user callback: expected {N} events, got {len(user_events)}"
    )
    assert len(rec_ctx.events) == 0, (
        f"context recorder: expected 0 events from explicit verbose() call, "
        f"got {len(rec_ctx.events)}"
    )


def test_ashell_record_bform_inside_context_no_double():
    """record(f) (B-form) called inside an active context: no double instrumentation."""
    N = 4
    x0 = jnp.float32(1.0)
    xs = jnp.arange(float(N), dtype=jnp.float32)
    ref = _simple_scan(x0, xs)

    with tap.record() as rec_ctx:
        g, rec_inner = tap.record(_simple_scan)
        got = g(x0, xs)
        jax.block_until_ready(got)

    assert bitwise_eq(ref, got), "record(f) inside context: result wrong"
    assert len(rec_inner.events) == N, (
        f"inner recorder: expected {N} events, got {len(rec_inner.events)}"
    )
    assert len(rec_ctx.events) == 0, (
        f"outer context: expected 0 events from record(f) call, got {len(rec_ctx.events)}"
    )


# ---------------------------------------------------------------------------
# 26. Public API: original_scan export
# ---------------------------------------------------------------------------


def test_original_scan_export():
    """jaxtap.original_scan is the pristine jax.lax.scan, unchanged after context."""
    from jaxtap._ashell import _original_scan

    # original_scan must be importable from jaxtap namespace
    assert hasattr(tap, "original_scan"), "tap.original_scan not found"

    # At import time, before any patching, original_scan must equal the pristine scan
    assert tap.original_scan is _original_scan, (
        "tap.original_scan should reference the captured pristine jax.lax.scan"
    )

    # jax.lax.scan must equal original_scan at session start (no active context)
    assert jax.lax.scan is tap.original_scan, (
        "At session start, jax.lax.scan should be the pristine original_scan"
    )

    # After a context enters and exits, original_scan must remain unchanged
    x0 = jnp.float32(1.0)
    xs = jnp.arange(5.0, dtype=jnp.float32)

    with tap.record():
        _simple_scan(x0, xs)

    jax.block_until_ready(None)

    # After exit, jax.lax.scan should be restored to original_scan
    assert jax.lax.scan is tap.original_scan, (
        "After context exit, jax.lax.scan should be restored to original_scan"
    )

    # original_scan itself must never change
    assert tap.original_scan is _original_scan, (
        "original_scan reference must remain unchanged throughout session"
    )
