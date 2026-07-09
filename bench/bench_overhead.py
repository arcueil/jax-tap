"""
bench_overhead.py — "What does the jaxtap lens cost?"

Three arms for a representative scan body (dim=8 float32 carry,
body: c = c * 1.01 + sin(x)):

  bare      — plain lax.scan, jitted, steady-state baseline
  manual    — same scan + jax.debug.callback(noop, carry) per step (irreducible cost)
  verbose   — tap.verbose(f, on_step=noop, sample_every=k) carry tap only
  record-A  — with tap.record(on_step=noop, sample_every=k): f_jit() A-form context
  primtap   — tap.verbose(..., taps=[tap.on("sin", ...)]) carry+prim tap combined

Axes
----
  N           : 1_000, 10_000, 100_000 (scan length)
  sample_every: 1, 10, 100  (jaxtap arms; prim tap fires every step regardless of se)
  vmap_lanes  : 1, 8  (bare/manual/verbose; lanes=8 only at N=10_000 to cap wall time)

Measurement
-----------
  JIT + 1 warmup call (compilation excluded from timing), then K=7 timed repeats
  using jax.block_until_ready + time.perf_counter.
  Compile time (first call) reported separately for N=10_000, all arms.
  Host callbacks make timing noisy: report median + min wall/step in µs.

Prim-tap-only caveat
--------------------
  ops=() would prevent walker descent into scan, silencing prim taps inside the body.
  The primtap arm therefore uses default ops=(scan, while_loop), measuring carry+prim
  combined.  Prim-tap-only isolation is not achievable with the current API.

Usage
-----
  uv run python bench/bench_overhead.py           # full run (~10 min)
  uv run python bench/bench_overhead.py --smoke   # smoke run at N=100 (<30 s)
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

import jax
import jax.lax as lax
import jax.numpy as jnp

import jaxtap as tap

DIM = 8
SEED = 42


# ---------------------------------------------------------------------------
# Deterministic helpers
# ---------------------------------------------------------------------------


def make_xs(N: int) -> jax.Array:
    return jax.random.normal(jax.random.PRNGKey(SEED), (N, DIM))


def scan_body(carry: jax.Array, x: jax.Array):
    return carry * 1.01 + jnp.sin(x), None


def noop_on_step(event: tap.TapEvent) -> None:
    pass


def noop_cb(carry: jax.Array) -> None:
    pass


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def warmup_and_time(jit_fn, carry, N: int, K: int) -> tuple[float, float]:
    """1 warmup call then K timed repeats. Returns (median µs/step, min µs/step)."""
    jax.block_until_ready(jit_fn(carry))
    times = []
    for _ in range(K):
        t0 = time.perf_counter()
        jax.block_until_ready(jit_fn(carry))
        times.append(time.perf_counter() - t0)
    return statistics.median(times) / N * 1e6, min(times) / N * 1e6


def first_call_us(jit_fn, carry) -> float:
    """First-call (compile + exec) wall time in µs."""
    t0 = time.perf_counter()
    jax.block_until_ready(jit_fn(carry))
    return (time.perf_counter() - t0) * 1e6


# ---------------------------------------------------------------------------
# Arm factories
# ---------------------------------------------------------------------------


def arm_bare(N: int, lanes: int = 1):
    xs = make_xs(N)
    init = jnp.zeros((lanes, DIM)) if lanes > 1 else jnp.zeros(DIM)

    def f(c):
        return lax.scan(scan_body, c, xs)[0]

    fn = jax.vmap(f) if lanes > 1 else f
    return jax.jit(fn), init


def arm_manual(N: int, lanes: int = 1):
    xs = make_xs(N)
    init = jnp.zeros((lanes, DIM)) if lanes > 1 else jnp.zeros(DIM)

    def body(carry, x):
        jax.debug.callback(noop_cb, carry, ordered=False)
        return carry * 1.01 + jnp.sin(x), None

    def f(c):
        return lax.scan(body, c, xs)[0]

    fn = jax.vmap(f) if lanes > 1 else f
    return jax.jit(fn), init


def arm_verbose(N: int, sample_every: int = 1, lanes: int = 1):
    xs = make_xs(N)
    init = jnp.zeros((lanes, DIM)) if lanes > 1 else jnp.zeros(DIM)

    def f(c):
        return lax.scan(scan_body, c, xs)[0]

    ft = tap.verbose(f, on_step=noop_on_step, sample_every=sample_every)
    fn = jax.vmap(ft) if lanes > 1 else ft
    return jax.jit(fn), init


def arm_record_aform(N: int, sample_every: int = 1, K: int = 7) -> tuple[float, float]:
    """
    A-form (context manager) arm.

    The function is compiled inside the first context so the dynamic router
    is baked into the XLA artifact.  Subsequent calls inside fresh contexts
    route events via the module-level _dynamic_router to the current context.
    Context enter/exit overhead is measured outside the timing window.
    """
    xs = make_xs(N)
    init = jnp.zeros(DIM)

    def f(c):
        return lax.scan(scan_body, c, xs)[0]

    with tap.record(on_step=noop_on_step, sample_every=sample_every):
        f_jit = jax.jit(f)
        jax.block_until_ready(f_jit(init))  # compile + warmup inside context

    times = []
    for _ in range(K):
        with tap.record(on_step=noop_on_step, sample_every=sample_every) as rec:
            t0 = time.perf_counter()
            jax.block_until_ready(f_jit(init))
            times.append(time.perf_counter() - t0)
        rec.events.clear()

    return statistics.median(times) / N * 1e6, min(times) / N * 1e6


def arm_primtap(N: int, sample_every: int = 1):
    """
    Carry tap + sin primitive tap (combined measurement).

    sample_every gates the carry tap only; the sin prim tap fires every step.
    Uses default ops=(scan, while_loop) so the walker descends into scan and
    prim taps fire.  ops=() would prevent descent, silencing prim taps.
    """
    xs = make_xs(N)
    init = jnp.zeros(DIM)

    def f(c):
        return lax.scan(scan_body, c, xs)[0]

    ft = tap.verbose(
        f,
        on_step=noop_on_step,
        sample_every=sample_every,
        taps=[tap.on("sin", select=lambda o: o[0][0])],
    )
    return jax.jit(ft), init


def compile_time_record_aform(N: int, sample_every: int = 1) -> float:
    """First-call time for A-form record (compile inside context)."""
    xs = make_xs(N)
    init = jnp.zeros(DIM)

    def f(c):
        return lax.scan(scan_body, c, xs)[0]

    with tap.record(on_step=noop_on_step, sample_every=sample_every):
        f_jit = jax.jit(f)
        t0 = time.perf_counter()
        jax.block_until_ready(f_jit(init))
        return (time.perf_counter() - t0) * 1e6


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------


def print_tables(rows: list[dict], compile_rows: list[tuple], smoke: bool) -> None:
    smoke_tag = " *(smoke run: N=100, K=3)*" if smoke else ""

    print()
    print("---")
    print()

    # Compile time table
    print(f"## Compile time (first call, N=10 000, lanes=1){smoke_tag}")
    print()
    print("| arm | compile (ms) |")
    print("|-----|-------------|")
    for label, ct_us in compile_rows:
        print(f"| {label} | {ct_us / 1000:.0f} |")
    print()

    # Baselines per (N, lanes)
    bare_map = {(r["N"], r["lanes"]): r["med"] for r in rows if r["arm"] == "bare"}
    manual_map = {(r["N"], r["lanes"]): r["med"] for r in rows if r["arm"] == "manual"}

    print(f"## Steady-state timing (median + min wall/step in µs){smoke_tag}")
    print()
    print("Overhead columns = arm median minus baseline at same (N, lanes).")
    print("`vs manual` isolates jaxtap machinery above the irreducible host-callback floor.")
    print()
    hdr = (
        "| arm | N | se | lanes"
        " | median (µs/step) | min (µs/step)"
        " | vs bare (µs) | vs manual (µs) |"
    )
    sep = (
        "|-----|---|----|------"
        "|-----------------|---------------"
        "|-------------|----------------|"
    )
    print(hdr)
    print(sep)

    for r in rows:
        key = (r["N"], r["lanes"])
        bare_b = bare_map.get(key, float("nan"))
        man_b = manual_map.get(key, float("nan"))
        vs_bare = r["med"] - bare_b
        vs_man = r["med"] - man_b

        arm = r["arm"]
        vs_bare_s = f"{vs_bare:+.2f}" if arm != "bare" else "—"
        vs_man_s = f"{vs_man:+.2f}" if arm not in ("bare", "manual") else "—"

        print(
            f"| {arm:<9} | {r['N']:>7,} | {str(r['se']):>4} | {r['lanes']:>5}"
            f" | {r['med']:>15.3f} | {r['mn']:>13.3f}"
            f" | {vs_bare_s:>12} | {vs_man_s:>14} |"
        )

    print()
    print("### Config notes")
    print()
    print("- **body**: `c = c * 1.01 + jnp.sin(x)`, carry dim=8 float32, xs ~ N(0,1)")
    print("- **bare**: plain `lax.scan`, jitted — no callbacks")
    print("- **manual**: `jax.debug.callback(noop, carry, ordered=False)` per step — irreducible host-callback floor")
    print("- **verbose**: `tap.verbose(f, on_step=noop, sample_every=k)` — carry tap only; ops=(scan, while_loop)")
    print("- **record-A**: `with tap.record(on_step=noop, sample_every=k): f_jit()` — A-form; function compiled inside first context; context enter/exit excluded from timing window")
    print("- **primtap**: `tap.verbose(f, on_step=noop, se=k, taps=[tap.on('sin', select=...)])` — carry+prim combined; `se` gates carry tap only; sin prim fires every step")
    print("- **vmap lanes=8**: only at N=10 000 (wall budget). Each debug.callback multiplies by lanes per step.")
    print("- **prim-tap-only**: not measurable; ops=() prevents walker descent into scan, silencing prim taps. primtap arm = carry+prim combined.")
    if smoke:
        print()
        print("*Full run: `PYTHONUNBUFFERED=1 uv run python bench/bench_overhead.py 2>&1 | tee bench/run.log`*")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="jaxtap overhead benchmark")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke run: N=100, se=[1,10], K=3 — verifies all code paths in <30 s",
    )
    args = parser.parse_args()

    if args.smoke:
        N_VALUES = [100]
        SE_VALUES = [1, 10]
        K = 3
        N_COMPILE = 100
        VMAP_N = None
    else:
        N_VALUES = [1_000, 10_000, 100_000]
        SE_VALUES = [1, 10, 100]
        K = 7
        N_COMPILE = 10_000
        VMAP_N = 10_000

    print(f"jax {jax.__version__} | device: {jax.devices()[0]}", file=sys.stderr, flush=True)
    print(f"smoke={args.smoke} | K={K} | DIM={DIM} | SEED={SEED}", file=sys.stderr, flush=True)

    rows: list[dict] = []
    compile_rows: list[tuple] = []

    # -----------------------------------------------------------------------
    # Compile times
    # -----------------------------------------------------------------------
    print(f"\nCompile times N={N_COMPILE}...", file=sys.stderr, flush=True)

    for label, factory, kwargs in [
        ("bare", arm_bare, {"N": N_COMPILE}),
        ("manual", arm_manual, {"N": N_COMPILE}),
        ("verbose(se=1)", arm_verbose, {"N": N_COMPILE, "sample_every": 1}),
        ("primtap(se=1)", arm_primtap, {"N": N_COMPILE, "sample_every": 1}),
    ]:
        fn, init = factory(**kwargs)
        ct = first_call_us(fn, init)
        compile_rows.append((label, ct))
        print(f"  {label}: {ct / 1000:.1f} ms", file=sys.stderr, flush=True)

    ct_rec = compile_time_record_aform(N_COMPILE, sample_every=1)
    compile_rows.append(("record-A(se=1)", ct_rec))
    print(f"  record-A(se=1): {ct_rec / 1000:.1f} ms", file=sys.stderr, flush=True)

    # -----------------------------------------------------------------------
    # Steady-state timing
    # -----------------------------------------------------------------------
    for N in N_VALUES:
        print(f"\nN={N:,}...", file=sys.stderr, flush=True)

        # bare (lanes=1)
        fn, init = arm_bare(N, lanes=1)
        bare_med, bare_min = warmup_and_time(fn, init, N, K)
        rows.append(dict(arm="bare", N=N, se="-", lanes=1, med=bare_med, mn=bare_min))
        print(f"  bare:          {bare_med:.3f} µs/step", file=sys.stderr, flush=True)

        # manual (lanes=1)
        fn, init = arm_manual(N, lanes=1)
        man_med, man_min = warmup_and_time(fn, init, N, K)
        rows.append(dict(arm="manual", N=N, se="-", lanes=1, med=man_med, mn=man_min))
        print(f"  manual:        {man_med:.3f} µs/step", file=sys.stderr, flush=True)

        # verbose (lanes=1, varying se)
        for se in SE_VALUES:
            fn, init = arm_verbose(N, sample_every=se)
            med, mn = warmup_and_time(fn, init, N, K)
            rows.append(dict(arm="verbose", N=N, se=se, lanes=1, med=med, mn=mn))
            print(f"  verbose(se={se:>3}): {med:.3f} µs/step", file=sys.stderr, flush=True)

        # record-A (lanes=1, varying se)
        for se in SE_VALUES:
            med, mn = arm_record_aform(N, sample_every=se, K=K)
            rows.append(dict(arm="record-A", N=N, se=se, lanes=1, med=med, mn=mn))
            print(f"  record-A(se={se:>3}): {med:.3f} µs/step", file=sys.stderr, flush=True)

        # primtap (lanes=1, varying se)
        for se in SE_VALUES:
            fn, init = arm_primtap(N, sample_every=se)
            med, mn = warmup_and_time(fn, init, N, K)
            rows.append(dict(arm="primtap", N=N, se=se, lanes=1, med=med, mn=mn))
            print(f"  primtap(se={se:>3}): {med:.3f} µs/step", file=sys.stderr, flush=True)

        # vmap lanes=8 (only at VMAP_N to cap wall time)
        if N == VMAP_N:
            lanes = 8
            print(f"  --- vmap lanes={lanes} ---", file=sys.stderr, flush=True)

            fn, init = arm_bare(N, lanes=lanes)
            med, mn = warmup_and_time(fn, init, N, K)
            rows.append(dict(arm="bare", N=N, se="-", lanes=lanes, med=med, mn=mn))
            print(f"  bare(l={lanes}):    {med:.3f} µs/step", file=sys.stderr, flush=True)

            fn, init = arm_manual(N, lanes=lanes)
            med, mn = warmup_and_time(fn, init, N, K)
            rows.append(dict(arm="manual", N=N, se="-", lanes=lanes, med=med, mn=mn))
            print(f"  manual(l={lanes}):  {med:.3f} µs/step", file=sys.stderr, flush=True)

            for se in SE_VALUES:
                fn, init = arm_verbose(N, sample_every=se, lanes=lanes)
                med, mn = warmup_and_time(fn, init, N, K)
                rows.append(dict(arm="verbose", N=N, se=se, lanes=lanes, med=med, mn=mn))
                print(f"  verbose(se={se:>3},l={lanes}): {med:.3f} µs/step", file=sys.stderr, flush=True)

    print_tables(rows, compile_rows, smoke=args.smoke)


if __name__ == "__main__":
    main()
