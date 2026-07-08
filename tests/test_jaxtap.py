"""
M0 invariant tests for jaxtap B-core walker.

Ports every invariant from proofs/jaxtap_sketch.py and
proofs/jaxtap_while_sketch.py, plus the M0-specific requirements.
AYS round-1 regression tests (custom_jvp, vmap, grad) appended at the bottom.

Run with: uv run pytest
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import jaxtap as tap
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect(f, *args, **verbose_kwargs):
    """Run tap.verbose(f, ...) on args; return (result, events_list)."""
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
# Small programs used across multiple tests
# ---------------------------------------------------------------------------


def _simple_scan(x0, xs):
    """Flat scalar carry; no nesting."""
    return jax.lax.scan(lambda c, x: (c + x, c * x), x0, xs)


def _nested_scan(x0, xs):
    """Outer scan whose body contains an inner scan + a closed-over const array."""
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
    """while_loop with closed-over consts in BOTH cond and body."""
    LIM = jnp.float32(37.0)
    INC = jnp.float32(1.7)

    def cond(c):
        return c < LIM

    def body(c):
        return c + INC

    return jax.lax.while_loop(cond, body, v0)


# ---------------------------------------------------------------------------
# test_identity_bitwise
# ---------------------------------------------------------------------------


def test_identity_bitwise():
    """Scan, nested scan, and while_loop all produce bitwise-identical outputs."""
    x0 = jnp.float32(0.5)
    xs = jnp.linspace(0.0, 1.0, 4, dtype=jnp.float32)

    ref_s = _simple_scan(x0, xs)
    got_s, _ = _collect(_simple_scan, x0, xs)
    assert bitwise_eq(ref_s, got_s), "plain scan not bitwise identical"

    ref_n = _nested_scan(x0, xs)
    got_n, _ = _collect(_nested_scan, x0, xs)
    assert bitwise_eq(ref_n, got_n), "nested scan not bitwise identical"

    v0 = jnp.float32(0.3)
    ref_w = _simple_while(v0)
    got_w, _ = _collect(_simple_while, v0)
    assert bitwise_eq(ref_w, got_w), "while_loop not bitwise identical"


# ---------------------------------------------------------------------------
# test_scan_taps
# ---------------------------------------------------------------------------


def test_scan_taps():
    """Per-step events are emitted in order 0..N-1 on the correct path."""
    N = 6
    x0 = jnp.float32(1.0)
    xs = jnp.arange(float(N), dtype=jnp.float32)

    _, events = _collect(_simple_scan, x0, xs)

    scan_events = [e for e in events if e.path == "scan[0]"]
    assert len(scan_events) == N
    assert [e.step for e in scan_events] == list(range(N))


# ---------------------------------------------------------------------------
# test_nested_addressing
# ---------------------------------------------------------------------------


def test_nested_addressing():
    """
    Nested scans get stable addresses (scan[0]/scan[0]).
    Mixed scan+while at one level share a single counter: scan[0], while[1].
    """
    x0 = jnp.float32(0.5)
    xs = jnp.linspace(0.0, 1.0, 4, dtype=jnp.float32)

    # --- nested scans ---
    _, events_nested = _collect(_nested_scan, x0, xs)
    paths_nested = {e.path for e in events_nested}
    assert "scan[0]" in paths_nested, "outer scan path missing"
    assert "scan[0]/scan[0]" in paths_nested, "inner scan path missing"

    # --- mixed scan + while at the same (top) level ---
    def f_mixed(carry):
        # scan is the 0th CF eqn → "scan[0]"; while is the 1st → "while[1]"
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

    carry0 = (jnp.float32(0.0), jnp.float32(0.0))
    _, events_mixed = _collect(f_mixed, carry0)
    paths_mixed = {e.path for e in events_mixed}
    assert "scan[0]" in paths_mixed, "scan[0] missing from mixed program"
    assert "while[1]" in paths_mixed, "while[1] missing — counter not shared"


# ---------------------------------------------------------------------------
# test_jit_composition
# ---------------------------------------------------------------------------


def test_jit_composition():
    """verbose(jit(f)) and jit(verbose(f)) are both bitwise-correct and emit equal event counts."""
    x0 = jnp.float32(0.5)
    xs = jnp.linspace(0.0, 1.0, 4, dtype=jnp.float32)
    ref = _nested_scan(x0, xs)

    # verbose(jit(f)) — walker recurses through the jit eqn
    events1: list[tap.TapEvent] = []
    got1 = tap.verbose(jax.jit(_nested_scan), on_step=lambda e: events1.append(e))(x0, xs)
    jax.block_until_ready(got1)

    # jit(verbose(f)) — the instrumented function is itself jittable
    events2: list[tap.TapEvent] = []
    got2 = jax.jit(tap.verbose(_nested_scan, on_step=lambda e: events2.append(e)))(x0, xs)
    jax.block_until_ready(got2)

    assert bitwise_eq(ref, got1), "verbose(jit(f)) not bitwise identical"
    assert bitwise_eq(ref, got2), "jit(verbose(f)) not bitwise identical"
    assert len(events1) == len(events2), "event counts differ between compositions"


# ---------------------------------------------------------------------------
# test_while_heartbeat
# ---------------------------------------------------------------------------


def test_while_heartbeat():
    """Event count matches the number of while iterations (heartbeat)."""
    v0 = jnp.float32(0.3)
    LIM, INC = np.float32(37.0), np.float32(1.7)

    # Count expected iterations in float32 arithmetic to match XLA.
    c = np.float32(v0)
    expected_iters = 0
    while c < LIM:
        c = c + INC
        expected_iters += 1

    _, events = _collect(_simple_while, v0)
    while_events = [e for e in events if e.path == "while[0]"]
    assert len(while_events) == expected_iters
    assert [e.step for e in while_events] == list(range(expected_iters))


# ---------------------------------------------------------------------------
# test_params_passthrough
# ---------------------------------------------------------------------------


def test_params_passthrough():
    """scan with reverse=True and unroll=2 remain bitwise-correct with correct event counts."""
    N = 5
    x0 = jnp.float32(1.0)
    xs = jnp.arange(float(N), dtype=jnp.float32)

    def scan_reverse(x0_, xs_):
        return jax.lax.scan(lambda c, x: (c + x, c * x), x0_, xs_, reverse=True)

    def scan_unroll2(x0_, xs_):
        return jax.lax.scan(lambda c, x: (c + x, c * x), x0_, xs_, unroll=2)

    ref_rev = scan_reverse(x0, xs)
    got_rev, ev_rev = _collect(scan_reverse, x0, xs)
    assert bitwise_eq(ref_rev, got_rev), "reverse=True not bitwise identical"
    assert len(ev_rev) == N

    ref_u2 = scan_unroll2(x0, xs)
    got_u2, ev_u2 = _collect(scan_unroll2, x0, xs)
    assert bitwise_eq(ref_u2, got_u2), "unroll=2 not bitwise identical"
    assert len(ev_u2) == N


# ---------------------------------------------------------------------------
# test_select_reduce_on_device
# ---------------------------------------------------------------------------


def test_select_reduce_on_device():
    """select reduces carry on-device; TapEvent.value carries correct pytree."""
    N = 4
    x0 = jnp.float32(1.0)
    xs = jnp.arange(float(N), dtype=jnp.float32)

    # Scalar selector — value should be a 0-d array
    events_scalar: list[tap.TapEvent] = []
    tapped_scalar = tap.verbose(
        _simple_scan,
        on_step=lambda e: events_scalar.append(e),
        select=lambda leaves: leaves[0].mean(),
    )
    jax.block_until_ready(tapped_scalar(x0, xs))
    assert len(events_scalar) == N
    for e in events_scalar:
        assert np.asarray(e.value).ndim == 0, "scalar selector must yield a 0-d value"

    # Dict-returning selector — value must be a dict
    events_dict: list[tap.TapEvent] = []
    tapped_dict = tap.verbose(
        _simple_scan,
        on_step=lambda e: events_dict.append(e),
        select=lambda leaves: {"carry": leaves[0]},
    )
    jax.block_until_ready(tapped_dict(x0, xs))
    assert len(events_dict) == N
    for e in events_dict:
        assert isinstance(e.value, dict), "dict selector must yield a dict value"
        assert "carry" in e.value


# ---------------------------------------------------------------------------
# test_ops_filtering
# ---------------------------------------------------------------------------


def test_ops_filtering():
    """ops=('scan',) suppresses while events; scan addresses are stable (counter not reset)."""

    def f(carry):
        # while is the 0th CF eqn; scan is the 1st.
        def cond(c):
            return c < 3.0

        def body_fn(c):
            return c + 1.0

        c1 = jax.lax.while_loop(cond, body_fn, carry[0])
        c2, _ = jax.lax.scan(
            lambda c, x: (c + x, c),
            carry[1],
            jnp.arange(4.0, dtype=jnp.float32),
        )
        return (c1, c2)

    carry0 = (jnp.float32(0.0), jnp.float32(0.0))

    # Both ops — while[0] and scan[1]
    _, ev_all = _collect(f, carry0)
    paths_all = {e.path for e in ev_all}
    assert "while[0]" in paths_all
    assert "scan[1]" in paths_all

    # Scan only — while suppressed; scan address is UNCHANGED (still index 1)
    _, ev_scan = _collect(f, carry0, ops=("scan",))
    paths_scan = {e.path for e in ev_scan}
    assert "scan[1]" in paths_scan, "scan index changed when while was filtered — counter reset"
    assert not any("while" in p for p in paths_scan), "while events appeared despite filtering"


# ---------------------------------------------------------------------------
# test_callback_totality
# ---------------------------------------------------------------------------


def test_callback_totality():
    """A raising on_step never corrupts results; warns exactly once; -W error is handled."""
    N = 5
    x0 = jnp.float32(0.0)
    xs = jnp.arange(float(N), dtype=jnp.float32)
    ref = _simple_scan(x0, xs)

    # --- Part 1: raises → correct results + exactly 1 UserWarning ---
    call_count = [0]

    def raising_cb(event: tap.TapEvent) -> None:
        call_count[0] += 1
        raise ValueError("boom")

    tap._warned.discard(id(raising_cb))  # ensure fresh warn-once state

    with pytest.warns(UserWarning, match="jaxtap") as warn_list:
        got = tap.verbose(_simple_scan, on_step=raising_cb)(x0, xs)
        jax.block_until_ready(got)

    assert bitwise_eq(ref, got), "result corrupted by raising callback"
    assert len(warn_list.list) == 1, f"expected exactly 1 warning, got {len(warn_list.list)}"
    assert call_count[0] == N, "callback must be attempted every step"

    # --- Part 2: under warnings.simplefilter("error") no exception propagates ---
    def raising_cb2(event: tap.TapEvent) -> None:
        raise RuntimeError("boom2")

    tap._warned.discard(id(raising_cb2))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got2 = tap.verbose(_simple_scan, on_step=raising_cb2)(x0, xs)
        jax.block_until_ready(got2)

    assert bitwise_eq(ref, got2), "result corrupted under -W error"


# ---------------------------------------------------------------------------
# test_carry_leaves_contract
# ---------------------------------------------------------------------------


def test_carry_leaves_contract():
    """Dict-carry scan: without select TapEvent.value is a flat tuple; select reshapes it."""

    def step_fn(carry, x):
        return {"a": carry["a"] + x, "b": carry["b"] * 2.0}, x

    carry0 = {"a": jnp.float32(1.0), "b": jnp.float32(2.0)}
    xs = jnp.arange(3.0, dtype=jnp.float32)

    def scan_f(carry_, xs_):
        return jax.lax.scan(step_fn, carry_, xs_)

    # Without select: flat tuple of carry leaves (dict flattens alphabetically → [a, b])
    events_flat: list[tap.TapEvent] = []
    got = tap.verbose(scan_f, on_step=lambda e: events_flat.append(e))(carry0, xs)
    jax.block_until_ready(got)

    assert len(events_flat) == 3
    for e in events_flat:
        assert isinstance(e.value, tuple), "without select, value must be a tuple"
        assert len(e.value) == 2, "dict with 2 keys must flatten to 2 leaves"

    # With select reshaping: value is a dict
    events_shaped: list[tap.TapEvent] = []
    got2 = tap.verbose(
        scan_f,
        on_step=lambda e: events_shaped.append(e),
        select=lambda leaves: {"a": leaves[0], "b": leaves[1]},
    )(carry0, xs)
    jax.block_until_ready(got2)

    assert len(events_shaped) == 3
    for e in events_shaped:
        assert isinstance(e.value, dict), "select must return structured value"
        assert set(e.value.keys()) == {"a", "b"}


# ---------------------------------------------------------------------------
# test_literal_outvar
# ---------------------------------------------------------------------------


def test_literal_outvar():
    """A function returning a constant is handled correctly (_read Literal branch)."""

    def f(x):
        # The scan body returns a constant zero as ys; _read must handle Literals in outvars.
        c, ys = jax.lax.scan(
            lambda c, _: (c + 1.0, jnp.zeros((), jnp.float32)),
            x,
            jnp.arange(3.0, dtype=jnp.float32),
        )
        return c, ys

    x = jnp.float32(0.0)
    ref = f(x)
    got, events = _collect(f, x)
    assert bitwise_eq(ref, got), "constant-returning function not bitwise identical"
    assert len(events) == 3, "expected 3 scan-step events"


# ---------------------------------------------------------------------------
# test_kwargs_rejected
# ---------------------------------------------------------------------------


def test_kwargs_rejected():
    """verbose(f) raises TypeError when called with keyword arguments."""

    def f(x):
        return x

    tapped = tap.verbose(f, on_step=lambda e: None)
    with pytest.raises(TypeError):
        tapped(jnp.float32(1.0), k=1)


# ---------------------------------------------------------------------------
# AYS round-1 regression tests
# ---------------------------------------------------------------------------


def test_custom_jvp_in_scan():
    """
    AYS-R1: jax.nn.softplus (and any @custom_jvp function) inside a scan body
    must not crash.  Root cause: custom_jvp_call params carry call_jaxpr /
    jvp_jaxpr_fun which get_bind_params converts to the subfuns= kwarg that
    bind expects — naive **eqn.params raises KeyError: 'subfuns'.
    """
    N = 5
    carry = jnp.float32(1.0)
    xs = jnp.arange(float(N), dtype=jnp.float32)

    def step(c, x):
        return jax.nn.softplus(c + x), c

    def scan_f(c, xs_):
        return jax.lax.scan(step, c, xs_)

    ref = scan_f(carry, xs)
    got, events = _collect(scan_f, carry, xs)

    assert bitwise_eq(ref, got), "custom_jvp-in-scan not bitwise identical"
    scan_events = [e for e in events if e.path == "scan[0]"]
    assert len(scan_events) == N, f"expected {N} scan events, got {len(scan_events)}"


def test_vmap_safety():
    """
    AYS-R1: vmap(verbose(f)) must produce bitwise-identical outputs across all
    lanes and emit N*lanes events.
    """
    N = 4
    LANES = 3
    xs_single = jnp.arange(float(N), dtype=jnp.float32)

    def step(c, x):
        return jax.nn.softplus(c + x), c

    def scan_f(c, xs_):
        return jax.lax.scan(step, c, xs_)

    carry_batch = jnp.ones(LANES, dtype=jnp.float32)
    xs_batch = jnp.tile(xs_single, (LANES, 1))

    ref = jax.vmap(scan_f)(carry_batch, xs_batch)

    events: list[tap.TapEvent] = []
    got = jax.vmap(tap.verbose(scan_f, on_step=lambda e: events.append(e)))(carry_batch, xs_batch)
    jax.block_until_ready(got)

    assert bitwise_eq(ref, got), "vmap(verbose(f)) not bitwise identical"
    # Under vmap with ordered=False each lane fires independently:
    # total events = LANES * N.
    assert len(events) == LANES * N, f"expected {LANES * N} events, got {len(events)}"


def test_grad_through_transform():
    """
    AYS-R1/R2: jax.grad(verbose(f)) must be bitwise identical to jax.grad(f),
    both for plain programs and for programs containing @custom_jvp functions.
    The get_bind_params fix must preserve the custom JVP rule so the gradient
    is computed via the custom derivative, not by inlining the primal.
    AYS-R2 strengthening: a sentinel JVP (derivative=42, distinct from primal
    2x) must propagate as 42^3=74088, proving the custom rule genuinely survives
    rather than accidentally matching primal autodiff.
    """
    N = 5
    carry0 = jnp.float32(1.0)
    xs = jnp.arange(float(N), dtype=jnp.float32)

    # --- plain loss (no custom_jvp) ---
    def plain_loss(c):
        _, ys = _simple_scan(c, xs)
        return jnp.sum(ys)

    ref_g_plain = jax.grad(plain_loss)(carry0)
    got_g_plain = jax.grad(
        lambda c: jnp.sum(tap.verbose(_simple_scan, on_step=lambda e: None)(c, xs)[1])
    )(carry0)
    jax.block_until_ready(got_g_plain)
    assert bitwise_eq(ref_g_plain, got_g_plain), "grad through plain scan not bitwise identical"

    # --- loss containing @custom_jvp (softplus) ---
    def step(c, x):
        return jax.nn.softplus(c + x), c

    def scan_f(c, xs_):
        return jax.lax.scan(step, c, xs_)

    def custom_jvp_loss(c):
        _, ys = scan_f(c, xs)
        return jnp.sum(ys)

    ref_g_cjvp = jax.grad(custom_jvp_loss)(carry0)
    got_g_cjvp = jax.grad(lambda c: jnp.sum(tap.verbose(scan_f, on_step=lambda e: None)(c, xs)[1]))(
        carry0
    )
    jax.block_until_ready(got_g_cjvp)
    assert bitwise_eq(ref_g_cjvp, got_g_cjvp), "grad through custom_jvp scan not bitwise identical"


def test_custom_jvp_sentinel_rule():
    """
    AYS-R2 probe [B]: a @custom_jvp whose derivative is a SENTINEL (42, distinct
    from the primal 2x) must propagate as 42^3=74088 through verbose(), proving
    the custom rule genuinely survives rather than accidentally matching primal
    autodiff.  Bitwise equality confirms rule-identical execution; the sentinel
    value proves which rule is active.
    """
    from jax import custom_jvp as _custom_jvp

    @_custom_jvp
    def f_sentinel(x):
        return x * x  # primal derivative would be 2x

    @f_sentinel.defjvp
    def _f_sentinel_jvp(primals, tangents):
        (x,), (dx,) = primals, tangents
        return f_sentinel(x), jnp.float32(42.0) * dx  # sentinel: 42, not 2x

    xs3 = jnp.arange(3.0, dtype=jnp.float32)

    def sentinel_loss(theta):
        final, _ = jax.lax.scan(lambda c, x: (f_sentinel(c + x), c), theta, xs3)
        return final

    theta = jnp.float32(0.7)
    ref_g = jax.grad(sentinel_loss)(theta)
    got_g = jax.grad(tap.verbose(sentinel_loss, on_step=lambda e: None))(theta)
    jax.block_until_ready(got_g)

    assert bitwise_eq(ref_g, got_g), "sentinel JVP grad not bitwise identical through verbose"
    # 42^3 = 74088 confirms sentinel rule active; primal autodiff would give ~35
    assert (
        float(got_g) == 74088.0
    ), f"sentinel JVP chain rule broken: expected 74088.0 (42^3), got {float(got_g)}"


def test_custom_vjp_through_transform():
    """
    AYS-R2: a @custom_vjp function (different primitive: custom_vjp_call) inside
    a scan body must work with verbose() — forward bitwise and grad bitwise.
    Sentinel backward rule (cotangent ×7, not cos(x)) proves the VJP rule
    genuinely survives through get_bind_params dispatch: expected grad = 7^3 = 343.
    """
    from jax import custom_vjp as _custom_vjp

    @_custom_vjp
    def f_vjp(x):
        return jnp.sin(x)

    def _f_vjp_fwd(x):
        return f_vjp(x), x

    def _f_vjp_bwd(res, g):
        return (jnp.float32(7.0) * g,)  # sentinel: cotangent ×7, not cos(x)

    f_vjp.defvjp(_f_vjp_fwd, _f_vjp_bwd)

    xs3 = jnp.arange(3.0, dtype=jnp.float32)

    def loss_vjp(theta):
        final, _ = jax.lax.scan(lambda c, x: (f_vjp(c + x), c), theta, xs3)
        return final

    theta = jnp.float32(0.5)

    # Forward bitwise identity
    ref_fwd = loss_vjp(theta)
    got_fwd, events = _collect(loss_vjp, theta)
    assert bitwise_eq(ref_fwd, got_fwd), "custom_vjp forward not bitwise identical"
    assert len([e for e in events if e.path == "scan[0]"]) == 3

    # Grad bitwise identity
    ref_g = jax.grad(loss_vjp)(theta)
    got_g = jax.grad(tap.verbose(loss_vjp, on_step=lambda e: None))(theta)
    jax.block_until_ready(got_g)
    assert bitwise_eq(ref_g, got_g), "custom_vjp grad not bitwise identical through verbose"
    # 7^3 = 343 proves VJP sentinel rule is active, not cos(x)-based autodiff
    assert (
        float(got_g) == 343.0
    ), f"sentinel VJP chain rule broken: expected 343.0 (7^3), got {float(got_g)}"


# ---------------------------------------------------------------------------
# Remediation regression tests (M3 — fixes F1 and F2)
# ---------------------------------------------------------------------------

# These turn the bcore-review/arm-b attack scripts into passing tests.
# See: proofs/bcore-review/arm-b/attack_cond.py, attack_misc.py,
#      attack_jit_addressing.py


def test_scan_in_cond_f1():
    """
    F1 regression: a scan nested inside lax.cond must emit events.
    Before the fix, the cond branch was bound opaquely → 0 taps for in-cond scan.
    After fix: scan inside the taken branch fires; path is cond[0]/b{j}/scan[0].
    """
    N = 5
    xs = jnp.arange(float(N), dtype=jnp.float32)
    x0 = jnp.float32(1.0)

    def f_cond(pred, x0_):
        def true_branch(c):
            out, _ = jax.lax.scan(lambda a, b: (a + b, a), c, xs)
            return out

        def false_branch(c):
            return c * 2.0

        return jax.lax.cond(pred, true_branch, false_branch, x0_)

    # True branch taken — scan inside cond must fire
    pred = jnp.bool_(True)
    ref = f_cond(pred, x0)
    got, events = _collect(f_cond, pred, x0)
    jax.block_until_ready(got)

    assert bitwise_eq(ref, got), "cond result not bitwise identical"
    # True branch is branch index 1 (false=0, true=1 in jaxpr branch order)
    scan_events = [e for e in events if "scan" in e.path]
    assert (
        len(scan_events) == N
    ), f"expected {N} events from scan inside cond true branch, got {len(scan_events)}"
    # Path must include the cond boundary and the branch index
    paths = {e.path for e in scan_events}
    assert any(
        "cond[0]" in p and "b1" in p for p in paths
    ), f"expected cond[0]/b1/... path, got {sorted(paths)}"

    # False branch taken — no scan in false branch → 0 events
    pred_false = jnp.bool_(False)
    ref_f = f_cond(pred_false, x0)
    got_f, events_f = _collect(f_cond, pred_false, x0)
    jax.block_until_ready(got_f)
    assert bitwise_eq(ref_f, got_f), "cond false-branch result not bitwise identical"
    assert len(events_f) == 0, f"false branch (no scan) should emit 0 events, got {len(events_f)}"


def test_scan_in_switch_f1():
    """
    F1 regression: scans nested inside lax.switch branches must emit events.
    All branches are instrumented at trace time; only the taken branch fires
    at runtime.  Path format: cond[k]/b{j}/scan[m].
    """
    N = 5
    xs = jnp.arange(float(N), dtype=jnp.float32)
    x0 = jnp.float32(1.0)

    def f_switch(i, x0_):
        def make_branch(mult):
            def branch(c):
                out, _ = jax.lax.scan(lambda a, b: (a + b * mult, a), c, xs)
                return out

            return branch

        return jax.lax.switch(i, [make_branch(1.0), make_branch(2.0), make_branch(3.0)], x0_)

    for branch_idx in range(3):
        i = jnp.int32(branch_idx)
        ref = f_switch(i, x0)
        got, events = _collect(f_switch, i, x0)
        jax.block_until_ready(got)

        assert bitwise_eq(ref, got), f"switch branch {branch_idx} result not bitwise identical"
        scan_events = [e for e in events if "scan" in e.path]
        assert (
            len(scan_events) == N
        ), f"branch {branch_idx}: expected {N} events, got {len(scan_events)}"
        paths = {e.path for e in scan_events}
        expected_branch_marker = f"b{branch_idx}"
        assert any(
            expected_branch_marker in p for p in paths
        ), f"branch {branch_idx}: expected b{branch_idx} in path, got {sorted(paths)}"


def test_scan_in_checkpoint_f1():
    """
    F1 regression: a scan inside jax.checkpoint (remat2) must emit events.
    Before the fix, remat2 was bound opaquely → 0 taps for in-checkpoint scan.
    After fix: scan inside checkpoint fires; path is remat[k]/scan[m].
    """
    N = 4
    xs = jnp.arange(float(N), dtype=jnp.float32)
    x0 = jnp.float32(0.0)

    def f_remat(x0_):
        @jax.checkpoint
        def inner(c0):
            out, _ = jax.lax.scan(lambda c, x: (c + x, c), c0, xs)
            return out

        return inner(x0_)

    ref = f_remat(x0)
    got, events = _collect(f_remat, x0)
    jax.block_until_ready(got)

    assert bitwise_eq(ref, got), "checkpoint result not bitwise identical"
    scan_events = [e for e in events if "scan" in e.path]
    assert (
        len(scan_events) == N
    ), f"expected {N} events from scan inside checkpoint, got {len(scan_events)}"
    paths = {e.path for e in scan_events}
    assert any("remat[0]" in p for p in paths), f"expected remat[0]/... path, got {sorted(paths)}"


def test_checkpoint_grad_bitwise():
    """
    F1 remediation: jax.grad through a verbose-wrapped checkpoint must be
    bitwise identical to jax.grad through the uninstrumented version.
    This verifies that jax.checkpoint's prevent_cse and policy are preserved.
    """
    N = 4
    xs = jnp.arange(float(N), dtype=jnp.float32)

    def f_remat(x0_):
        @jax.checkpoint
        def inner(c0):
            out, _ = jax.lax.scan(lambda c, x: (jnp.sin(c) + x, c), c0, xs)
            return out

        return inner(x0_)

    theta = jnp.float32(0.5)
    ref_grad = jax.grad(f_remat)(theta)
    got_grad = jax.grad(tap.verbose(f_remat, on_step=lambda e: None))(theta)
    jax.block_until_ready(got_grad)
    assert bitwise_eq(
        ref_grad, got_grad
    ), f"grad through checkpoint not bitwise identical: ref={float(ref_grad):.6f} got={float(got_grad):.6f}"


def test_jit_addressing_uniqueness_f2():
    """
    F2 regression: a top-level scan and a jit-nested scan must get DISTINCT paths.
    Before the fix, both resolved to 'scan[0]' (address collision).
    After fix: 'scan[0]' vs 'jit[1]/scan[0]'.
    """
    N = 4
    xs = jnp.arange(float(N), dtype=jnp.float32)
    x0 = jnp.float32(1.0)

    def f(x0_, xs_):
        # scan is the 0th boundary at top level → scan[0]
        a, _ = jax.lax.scan(lambda c, x: (c + x, c), x0_, xs_)

        # jit is the 1st boundary at top level → jit[1]
        # scan inside jit is the 0th at that level → jit[1]/scan[0]
        @jax.jit
        def inner(c):
            b, _ = jax.lax.scan(lambda cc, x: (cc * 1.0 + x, cc), c, xs_)
            return b

        return a + inner(x0_)

    ref = f(x0, xs)
    got, events = _collect(f, x0, xs)
    jax.block_until_ready(got)

    assert bitwise_eq(ref, got), "jit-addressing result not bitwise identical"

    from collections import Counter

    path_counts = Counter(e.path for e in events)
    distinct_paths = set(path_counts.keys())

    # Must have exactly 2 distinct paths, not 1 (collision would merge them)
    assert (
        len(distinct_paths) == 2
    ), f"expected 2 distinct paths (top-level + jit-nested), got {sorted(distinct_paths)}"
    assert "scan[0]" in distinct_paths, f"top-level scan[0] path missing: {sorted(distinct_paths)}"
    assert any(
        "jit" in p for p in distinct_paths
    ), f"jit-nested path missing: {sorted(distinct_paths)}"
    # Each scan runs N steps
    for p, count in path_counts.items():
        assert count == N, f"path {p!r}: expected {N} events, got {count}"


def test_jit_boundary_path_format():
    """
    Check the boundary-visible path format for verbose(jit(f)):
    outer scan → 'jit[0]/scan[0]', inner scan → 'jit[0]/scan[0]/scan[0]'.
    Event count must match the non-jit-wrapped case.
    """
    x0 = jnp.float32(0.5)
    xs = jnp.linspace(0.0, 1.0, 4, dtype=jnp.float32)

    # verbose(jit(f)) — jit boundary is visible
    got1, events1 = _collect(jax.jit(_nested_scan), x0, xs)
    jax.block_until_ready(got1)

    # direct verbose(f) — no jit boundary
    got2, events2 = _collect(_nested_scan, x0, xs)
    jax.block_until_ready(got2)

    assert bitwise_eq(got1, got2), "verbose(jit(f)) != verbose(f) (not bitwise identical)"

    # Event counts must match
    assert len(events1) == len(
        events2
    ), f"event counts differ: verbose(jit(f))={len(events1)} vs verbose(f)={len(events2)}"

    paths1 = {e.path for e in events1}
    # Paths with jit boundary
    assert "jit[0]/scan[0]" in paths1, f"expected jit[0]/scan[0] path, got {sorted(paths1)}"
    assert (
        "jit[0]/scan[0]/scan[0]" in paths1
    ), f"expected jit[0]/scan[0]/scan[0] path, got {sorted(paths1)}"

    paths2 = {e.path for e in events2}
    # Paths without jit boundary
    assert "scan[0]" in paths2
    assert "scan[0]/scan[0]" in paths2
