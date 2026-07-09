"""
A-shell conformance tests — ports the checks from the ashell-review corpora
that are NOT yet covered by tests/test_ashell.py.

Sources:
  proofs/ashell-review/arm-s/a2_fidelity_sweep.py   (dtype+API+pytree+error)
  proofs/ashell-review/arm-s/a3_equivalence_and_transforms.py  (verbose vs ashell)
  proofs/ashell-review/arm-l/05_two_owned_and_bystander.py
  proofs/ashell-review/arm-l/06_registry_race_stress.py  (bounded)
  proofs/ashell-review/arm-l/07_nonlifo_exit.py
"""

from __future__ import annotations

import threading
import warnings

import jax
import jax.numpy as jnp
import numpy as np

import jaxtap as tap
from jaxtap._ashell import (
    _context_registry,
    _original_scan,
    _original_while,
    _patched_scan,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bw(a, b) -> bool:
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return len(la) == len(lb) and all(
        np.asarray(x).tobytes() == np.asarray(y).tobytes() for x, y in zip(la, lb)
    )


def _clean():
    """Reset A-shell global state to a known-clean baseline."""
    _context_registry.clear()
    jax.lax.scan = _original_scan
    jax.lax.while_loop = _original_while


XS5 = jnp.arange(5.0, dtype=jnp.float32)


def _simple_scan(x0):
    return jax.lax.scan(lambda c, x: (c + x, c * x), x0, XS5)


# ---------------------------------------------------------------------------
# a2_fidelity_sweep.py — dtype + API surface + pytrees + error transparency
# ---------------------------------------------------------------------------


def test_ashell_int32_carry_bitwise():
    """int32 carry scan is bitwise identical through A-shell with correct event count.

    Ports proofs/ashell-review/arm-s/a2_fidelity_sweep.py int32 case.
    """
    xs_i = jnp.arange(5, dtype=jnp.int32)
    ref = jax.lax.scan(lambda c, x: (c + x, c), jnp.int32(0), xs_i)
    with tap.record() as rec:
        got = jax.lax.scan(lambda c, x: (c + x, c), jnp.int32(0), xs_i)
    jax.block_until_ready(got)
    assert _bw(ref, got), "int32 carry: A-shell result not bitwise identical"
    assert len(rec.events) == 5, f"int32 carry: expected 5 events, got {len(rec.events)}"


def test_ashell_complex64_carry_bitwise():
    """complex64 carry scan is bitwise identical through A-shell.

    Ports proofs/ashell-review/arm-s/a2_fidelity_sweep.py complex64 case.
    """
    xs_c = jnp.arange(5, dtype=jnp.complex64)
    ref = jax.lax.scan(lambda c, x: (c + x, c), jnp.complex64(1 + 1j), xs_c)
    with tap.record() as rec:
        got = jax.lax.scan(lambda c, x: (c + x, c), jnp.complex64(1 + 1j), xs_c)
    jax.block_until_ready(got)
    assert _bw(ref, got), "complex64 carry: A-shell result not bitwise identical"
    assert len(rec.events) == 5, f"complex64 carry: expected 5 events, got {len(rec.events)}"


def test_ashell_dict_carry_bitwise():
    """Dict carry scan is bitwise identical through A-shell.

    Ports proofs/ashell-review/arm-s/a2_fidelity_sweep.py dict carry case.
    """

    def body(c, x):
        return {"a": c["a"] + x, "b": c["b"] * x}, c["a"]

    init = {"a": jnp.float32(0.0), "b": jnp.float32(1.0)}
    ref = jax.lax.scan(body, init, XS5)
    with tap.record() as rec:
        got = jax.lax.scan(body, init, XS5)
    jax.block_until_ready(got)
    assert _bw(ref, got), "dict carry: A-shell result not bitwise identical"
    assert len(rec.events) == 5, f"dict carry: expected 5 events, got {len(rec.events)}"


def test_ashell_pytree_xs_dict():
    """Pytree xs (dict of arrays) is handled bitwise-identically by A-shell.

    Ports proofs/ashell-review/arm-s/a2_fidelity_sweep.py 'nested pytree xs' case.
    """
    xs_dict = {"u": XS5, "v": XS5 * 2}
    ref = jax.lax.scan(lambda c, x: (c + x["u"] + x["v"], c), jnp.float32(0.0), xs_dict)
    with tap.record() as rec:
        got = jax.lax.scan(lambda c, x: (c + x["u"] + x["v"], c), jnp.float32(0.0), xs_dict)
    jax.block_until_ready(got)
    assert _bw(ref, got), "pytree xs dict: A-shell result not bitwise identical"
    assert len(rec.events) == 5, f"pytree xs dict: expected 5 events, got {len(rec.events)}"


def test_ashell_prng_key_carry():
    """PRNG key carry scan is bitwise identical through A-shell.

    Ports proofs/ashell-review/arm-s/a2_fidelity_sweep.py PRNG-key carry case.
    """

    def body(key, _):
        key, sub = jax.random.split(key)
        return key, jax.random.normal(sub)

    ref = jax.lax.scan(body, jax.random.PRNGKey(0), None, length=5)
    with tap.record() as rec:
        got = jax.lax.scan(body, jax.random.PRNGKey(0), None, length=5)
    jax.block_until_ready(got)
    assert _bw(ref, got), "PRNG-key carry: A-shell result not bitwise identical"
    assert len(rec.events) == 5, f"PRNG-key carry: expected 5 events, got {len(rec.events)}"


def test_ashell_all_keyword_form():
    """scan called with all-keyword f= init= xs= form works through A-shell.

    Ports proofs/ashell-review/arm-s/a2_fidelity_sweep.py 'f= init= xs= all keyword' case.
    """
    body = lambda c, x: (c + x, c)  # noqa: E731
    ref = jax.lax.scan(f=body, init=jnp.float32(0.0), xs=XS5)
    with tap.record() as rec:
        got = jax.lax.scan(f=body, init=jnp.float32(0.0), xs=XS5)
    jax.block_until_ready(got)
    assert _bw(ref, got), "all-keyword form: A-shell result not bitwise identical"
    assert len(rec.events) == 5, f"all-keyword form: expected 5 events, got {len(rec.events)}"


def test_ashell_error_transparency():
    """User shape-mismatch inside scan raises same exception type inside A-shell as outside.

    Ports proofs/ashell-review/arm-s/a2_fidelity_sweep.py error transparency case.
    """

    def bad_scan():
        return jax.lax.scan(lambda c, x: (jnp.stack([c, c]), c), jnp.float32(0.0), XS5)

    outside_exc = None
    try:
        bad_scan()
    except Exception as e:
        outside_exc = type(e)

    inside_exc = None
    try:
        with tap.record():
            bad_scan()
    except Exception as e:
        inside_exc = type(e)

    assert outside_exc is not None, "expected bad_scan to raise outside context"
    assert inside_exc is not None, "expected bad_scan to raise inside A-shell context"
    assert outside_exc == inside_exc, (
        f"error transparency: outside={outside_exc.__name__!r} inside={inside_exc.__name__!r}"
    )


# ---------------------------------------------------------------------------
# a3_equivalence_and_transforms.py — verbose() vs A-shell event equivalence
# ---------------------------------------------------------------------------


def _events_equivalent(f, arg, **kw):
    """Return (verbose_events, ashell_events, results_bitwise)."""
    from collections import Counter

    vb: list = []
    r_vb = tap.verbose(f, on_step=vb.append, **kw)(arg)
    jax.block_until_ready(r_vb)

    ctx_kw = dict(kw)
    with tap.record(**ctx_kw) as rec:
        r_ash = f(arg)
    jax.block_until_ready(r_ash)

    def _key(e):
        return (
            e.path,
            e.step,
            tuple(np.asarray(v).tobytes() for v in jax.tree_util.tree_leaves(e.value)),
        )

    vb_counts = Counter(_key(e) for e in vb)
    ash_counts = Counter(_key(e) for e in rec.events)
    return vb_counts, ash_counts, _bw(r_vb, r_ash)


def test_ashell_equivalence_nested_scan():
    """verbose() and A-shell produce identical events for a nested scan.

    Ports proofs/ashell-review/arm-s/a3_equivalence_and_transforms.py nested scan case.
    """
    INNER = jnp.arange(3.0, dtype=jnp.float32)

    def nested(x0):
        def outer(c, x):
            c2, _ = jax.lax.scan(lambda a, b: (a * 1.001 + jnp.sin(b), a), c + x, INNER)
            return c2, c2

        return jax.lax.scan(outer, x0, XS5)[0]

    vb, ash, bw = _events_equivalent(nested, jnp.float32(0.5))
    assert bw, "nested scan: verbose vs ashell result not bitwise identical"
    assert vb == ash, (
        f"nested scan: event mismatch verbose={sum(vb.values())} ash={sum(ash.values())}"
    )


def test_ashell_equivalence_cond_in_scan():
    """verbose() and A-shell produce identical events for cond-in-scan.

    Ports proofs/ashell-review/arm-s/a3_equivalence_and_transforms.py cond-in-scan case.
    """

    def cond_in_scan(x0):
        def outer(c, x):
            c2 = jax.lax.cond(x > 2.0, lambda a: a * 2.0, lambda a: a + 1.0, c)
            return c2, c2

        return jax.lax.scan(outer, x0, XS5)[0]

    vb, ash, bw = _events_equivalent(cond_in_scan, jnp.float32(0.5))
    assert bw, "cond-in-scan: verbose vs ashell result not bitwise identical"
    assert vb == ash, (
        f"cond-in-scan: event mismatch verbose={sum(vb.values())} ash={sum(ash.values())}"
    )


def test_ashell_equivalence_primitive_taps():
    """verbose() and A-shell produce identical events for prim tap (cholesky in scan).

    Ports proofs/ashell-review/arm-s/a3_equivalence_and_transforms.py primitive tap case.
    """

    def chol(x0):
        def body(carry, _):
            c = 1.0 - 10.0 ** (-carry)
            M = jnp.array([[1.0, c], [c, 1.0]], dtype=c.dtype)
            L = jnp.linalg.cholesky(M)
            return carry + 1.0, jnp.sum(jnp.diag(L))

        return jax.lax.scan(body, x0, None, length=5)[0]

    vb, ash, bw = _events_equivalent(chol, jnp.float32(1.0), taps=[tap.on("cholesky")])
    assert bw, "prim-tap: verbose vs ashell result not bitwise identical"
    assert vb == ash, f"prim-tap: event mismatch verbose={sum(vb.values())} ash={sum(ash.values())}"


# ---------------------------------------------------------------------------
# arm-l/07_nonlifo_exit.py — non-LIFO exit order
# ---------------------------------------------------------------------------


def test_ashell_nonlifo_exit_order():
    """Non-LIFO exit order restores correctly and scan stays usable between exits.

    Two orderings tested:
      order 1: a.enter, b.enter, a.exits FIRST, b.exits LAST
      order 2: a.enter, b.enter, b.exits FIRST, a.exits LAST

    After the FIRST exit, scan must still be patched (one ctx still active)
    and must run correctly. After the LAST exit, scan must be restored.

    Ports proofs/ashell-review/arm-l/07_nonlifo_exit.py.
    """
    xs = jnp.arange(4.0, dtype=jnp.float32)
    ref = jax.lax.scan(lambda c, x: (c + x, c), 0.0, xs)
    jax.block_until_ready(ref)

    def _run_order(exit_first: str):
        _clean()
        a = tap.record()
        b = tap.record()
        a.__enter__()
        b.__enter__()

        assert jax.lax.scan is _patched_scan, f"{exit_first}: scan not patched after both enter"
        assert len(_context_registry) == 2

        first, second = (a, b) if exit_first == "a" else (b, a)
        first.__exit__(None, None, None)

        # After first exit: one context still active; scan must still be patched
        assert jax.lax.scan is _patched_scan, (
            f"order({exit_first}-first): scan not still patched after first exit"
        )
        assert len(_context_registry) == 1

        # Scan must still work correctly between exits
        r_mid = jax.lax.scan(lambda c, x: (c + x, c), jnp.float32(0.0), xs)
        jax.block_until_ready(r_mid)
        assert _bw(ref, r_mid), f"order({exit_first}-first): scan between exits not bitwise correct"

        second.__exit__(None, None, None)

        # After last exit: both restored
        assert jax.lax.scan is _original_scan, (
            f"order({exit_first}-first): scan not restored after last exit"
        )
        assert jax.lax.while_loop is _original_while, (
            f"order({exit_first}-first): while_loop not restored after last exit"
        )
        assert len(_context_registry) == 0, (
            f"order({exit_first}-first): registry not empty after last exit"
        )

    try:
        _run_order("a")
        _run_order("b")
    finally:
        _clean()


# ---------------------------------------------------------------------------
# arm-l/05_two_owned_and_bystander.py — two owner threads + bystander
# ---------------------------------------------------------------------------


def test_ashell_two_owned_contexts_bystander():
    """Two owner threads each with their own context: no cross-talk, bystander untapped.

    When >=2 contexts are active, each thread's scan is attributed to its own
    context and a bystander thread (owning neither) passes untapped and bitwise-correct.

    Ports proofs/ashell-review/arm-l/05_two_owned_and_bystander.py.
    """
    _clean()

    xs_a = jnp.arange(4.0, dtype=jnp.float32)  # 4 events in ctx A
    xs_b = jnp.arange(7.0, dtype=jnp.float32)  # 7 events in ctx B
    xs_bys = jnp.arange(12.0, dtype=jnp.float32)  # bystander: should be 0 events

    ref_bys = jax.lax.scan(lambda c, x: (c + x, c), 0.0, xs_bys)
    jax.block_until_ready(ref_bys)

    barrier = threading.Barrier(3)
    recs: dict = {}
    bys_result: dict = {}
    errors: dict = {}

    def _owner(name, xs):
        try:
            with tap.record() as rec:
                barrier.wait()  # all 3 threads start simultaneously
                r = jax.lax.scan(lambda c, x: (c + x, c), jnp.float32(0.0), xs)
                jax.block_until_ready(r)
            recs[name] = list(rec.events)
        except Exception as e:  # noqa: BLE001
            errors[name] = e

    def _bystander():
        try:
            barrier.wait()  # join the simultaneous run (>=2 contexts active)
            r = jax.lax.scan(lambda c, x: (c + x, c), 0.0, xs_bys)
            jax.block_until_ready(r)
            bys_result["ok"] = _bw(ref_bys, r)
        except Exception as e:  # noqa: BLE001
            errors["bystander"] = e

    ta = threading.Thread(target=_owner, args=("A", xs_a))
    tb = threading.Thread(target=_owner, args=("B", xs_b))
    tby = threading.Thread(target=_bystander)
    for t in (ta, tb, tby):
        t.start()
    for t in (ta, tb, tby):
        t.join(timeout=30)

    try:
        assert not errors, f"thread errors: {errors}"
        assert len(recs.get("A", [])) == len(xs_a), (
            f"thread A: expected {len(xs_a)} events, got {len(recs.get('A', []))}"
        )
        assert len(recs.get("B", [])) == len(xs_b), (
            f"thread B: expected {len(xs_b)} events, got {len(recs.get('B', []))}"
        )
        assert bys_result.get("ok", False), "bystander scan not bitwise-correct"
        assert len(_context_registry) == 0, "registry not empty after all threads exit"
        assert jax.lax.scan is _original_scan, "scan not restored after all threads exit"
    finally:
        _clean()


# ---------------------------------------------------------------------------
# arm-l/06_registry_race_stress.py — bounded concurrent enter/exit stress
# ---------------------------------------------------------------------------


def test_ashell_registry_race_bounded():
    """Concurrent enter/exit with lock-free registry reads: no errors, clean finish.

    Bounded version of proofs/ashell-review/arm-l/06_registry_race_stress.py:
    4 threads × 20 iters (vs original 8 × 40) to keep runtime < 10s while
    exercising the same race between _context_registry mutation and the
    lock-free list() snapshot in _patched_scan.
    """
    _clean()

    N_THREADS = 4
    N_ITERS = 20
    xs = jnp.arange(4.0, dtype=jnp.float32)
    ref = jax.lax.scan(lambda c, x: (c + x, c), 0.0, xs)
    jax.block_until_ready(ref)

    errors: list = []
    bad_results = [0]
    lock = threading.Lock()
    stop_probe = threading.Event()

    def _probe_registry():
        while not stop_probe.is_set():
            try:
                list(_context_registry.values())  # lock-free snapshot
            except Exception as e:  # noqa: BLE001
                errors.append(("probe", repr(e)))

    def _worker(tid):
        try:
            for _ in range(N_ITERS):
                with tap.record():
                    r = jax.lax.scan(lambda c, x: (c + x, c), jnp.float32(0.0), xs)
                    jax.block_until_ready(r)
                    if float(r[0]) != float(ref[0]):
                        with lock:
                            bad_results[0] += 1
        except Exception as e:  # noqa: BLE001
            errors.append((tid, repr(e)))

    probe = threading.Thread(target=_probe_registry)
    probe.start()
    workers = [threading.Thread(target=_worker, args=(i,)) for i in range(N_THREADS)]
    for w in workers:
        w.start()
    for w in workers:
        w.join(timeout=60)
    stop_probe.set()
    probe.join(timeout=5)

    try:
        assert not errors, f"race stress: thread errors: {errors[:5]}"
        assert bad_results[0] == 0, f"race stress: {bad_results[0]} non-bitwise results"
        assert len(_context_registry) == 0, (
            f"race stress: registry leaked ({len(_context_registry)} entries)"
        )
        assert jax.lax.scan is _original_scan, "race stress: scan not restored to original"
        assert jax.lax.while_loop is _original_while, "race stress: while_loop not restored"
    finally:
        _clean()


# ---------------------------------------------------------------------------
# Additional ashell warning-related conformance
# ---------------------------------------------------------------------------


def test_ashell_warnonce_resets_each_session():
    """Each new session resets the warn-once flag: two independent clobbers both warn.

    Supplementary check from arm-l/08_foreign_and_warnonce.py part A
    (already covered by test_ashell_session_scoped_warnonce — kept here for
    completeness in the conformance map).
    """
    import jaxtap._ashell as _A

    _A._clobber_scan_warned = False
    _A._clobber_while_warned = False

    def _make_foreign():
        return lambda *a, **k: _original_scan(*a, **k)

    # Session 1
    with warnings.catch_warnings(record=True) as w1:
        warnings.simplefilter("always")
        with tap.record():
            jax.lax.scan = _make_foreign()
    jax.lax.scan = _original_scan
    n1 = sum("jaxtap" in str(x.message) for x in w1)

    # Session 2
    with warnings.catch_warnings(record=True) as w2:
        warnings.simplefilter("always")
        with tap.record():
            jax.lax.scan = _make_foreign()
    jax.lax.scan = _original_scan
    n2 = sum("jaxtap" in str(x.message) for x in w2)

    assert n1 >= 1, "first clobber did not warn"
    assert n2 >= 1, "second clobber did not warn (session-scoped reset broken)"
    _clean()
