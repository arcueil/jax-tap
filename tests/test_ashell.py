"""
M1b: A-shell tests — ``with tap.record():`` context manager form.

Tests for the patch machinery (registry, depth counter, foreign-patch
chaining, thread delegation) and for the user-facing UX contract
("remove the ``with`` line and nothing changes").

Run with: uv run pytest tests/test_ashell.py
"""

from __future__ import annotations

import threading
from collections import Counter

import jax
import jax.numpy as jnp
import jaxtap as tap
import numpy as np
import pytest

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
    assert jax.lax.while_loop is _original_while, "while_loop not restored after empty with"
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
    assert len(chol_events) == N, f"expected {N} cholesky events, got {len(chol_events)}"
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
    assert jax.lax.while_loop is _original_while, "while_loop not restored after exception"
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
    from jaxtap._ashell import _original_scan, _patched_scan

    clobber_calls: list[str] = []

    def _clobber_scan(f, init, xs=None, length=None, **kwargs):
        clobber_calls.append("clobber")
        return _original_scan(f, init, xs=xs, length=length, **kwargs)

    # Reset warn-once flag from prior test runs so we can observe the warning
    import jaxtap._ashell as _ashell_mod
    _ashell_mod._clobber_scan_warned = False

    x0 = jnp.float32(1.0)
    xs = jnp.arange(3.0, dtype=jnp.float32)

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
