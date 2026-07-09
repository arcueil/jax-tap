# jax-tap overhead benchmark

**"What does the lens cost?"** — results from `bench_overhead.py` on this box.

Platform: CPU (cpu:0), JAX 0.10.2. Total wall: ~16 min.

---

## Compile time (first call, N=10 000, lanes=1)

| arm | compile (ms) |
|-----|-------------|
| bare | 58 |
| manual | 423 |
| verbose(se=1) | 859 |
| primtap(se=1) | 1552 |
| record-A(se=1) | 969 |

The jaxtap arms (verbose, record-A, primtap) take 10–25× longer to compile than
bare because the walker re-traces and re-interprets the jaxpr at every `verbose()` call.
Compile cost is paid once per `(function, input_shape)` pair; subsequent calls use the
XLA cache.

---

## Steady-state timing (median + min wall/step in µs)

Overhead columns = arm median minus baseline at same (N, lanes).
`vs manual` isolates jaxtap machinery above the irreducible host-callback floor.
Measurement: JIT + 1 warmup call excluded, then K=7 repeats with `jax.block_until_ready`.

| arm | N | se | lanes | median (µs/step) | min (µs/step) | vs bare (µs) | vs manual (µs) |
|-----|---|----|-------|-----------------|---------------|-------------|----------------|
| bare      |   1,000 |    - |     1 |           0.033 |         0.026 |            — |              — |
| manual    |   1,000 |    - |     1 |          31.987 |        31.690 |       +31.95 |              — |
| verbose   |   1,000 |    1 |     1 |          73.351 |        69.056 |       +73.32 |         +41.36 |
| verbose   |   1,000 |   10 |     1 |           8.379 |         7.539 |        +8.35 |         -23.61 |
| verbose   |   1,000 |  100 |     1 |           1.379 |         1.114 |        +1.35 |         -30.61 |
| record-A  |   1,000 |    1 |     1 |          80.661 |        77.952 |       +80.63 |         +48.67 |
| record-A  |   1,000 |   10 |     1 |           8.329 |         8.079 |        +8.30 |         -23.66 |
| record-A  |   1,000 |  100 |     1 |           3.078 |         2.365 |        +3.04 |         -28.91 |
| primtap   |   1,000 |    1 |     1 |         147.057 |       144.704 |      +147.02 |        +115.07 |
| primtap   |   1,000 |   10 |     1 |          83.236 |        78.723 |       +83.20 |         +51.25 |
| primtap   |   1,000 |  100 |     1 |          72.039 |        68.478 |       +72.01 |         +40.05 |
| bare      |  10,000 |    - |     1 |           0.106 |         0.056 |            — |              — |
| manual    |  10,000 |    - |     1 |          32.901 |        31.646 |       +32.80 |              — |
| verbose   |  10,000 |    1 |     1 |          73.381 |        71.069 |       +73.28 |         +40.48 |
| verbose   |  10,000 |   10 |     1 |           7.256 |         7.202 |        +7.15 |         -25.65 |
| verbose   |  10,000 |  100 |     1 |           1.034 |         0.884 |        +0.93 |         -31.87 |
| record-A  |  10,000 |    1 |     1 |          85.171 |        79.579 |       +85.06 |         +52.27 |
| record-A  |  10,000 |   10 |     1 |           7.943 |         7.836 |        +7.84 |         -24.96 |
| record-A  |  10,000 |  100 |     1 |           0.926 |         0.911 |        +0.82 |         -31.98 |
| primtap   |  10,000 |    1 |     1 |         151.539 |       144.322 |      +151.43 |        +118.64 |
| primtap   |  10,000 |   10 |     1 |          77.612 |        75.631 |       +77.51 |         +44.71 |
| primtap   |  10,000 |  100 |     1 |          69.808 |        67.704 |       +69.70 |         +36.91 |
| bare      |  10,000 |    - |     8 |           0.055 |         0.053 |            — |              — |
| manual    |  10,000 |    - |     8 |       1,877.620 |     1,739.041 |    +1,877.56 |              — |
| verbose   |  10,000 |    1 |     8 |       3,152.036 |     2,917.787 |    +3,151.98 |      +1,274.42 |
| verbose   |  10,000 |   10 |     8 |          59.249 |        56.007 |       +59.19 |              * |
| verbose   |  10,000 |  100 |     8 |           6.099 |         6.043 |        +6.04 |              * |
| bare      | 100,000 |    - |     1 |           0.058 |         0.058 |            — |              — |
| manual    | 100,000 |    - |     1 |          31.804 |        31.403 |       +31.75 |              — |
| verbose   | 100,000 |    1 |     1 |          71.894 |        70.763 |       +71.84 |         +40.09 |
| verbose   | 100,000 |   10 |     1 |           7.607 |         7.230 |        +7.55 |         -24.20 |
| verbose   | 100,000 |  100 |     1 |           0.911 |         0.866 |        +0.85 |         -30.89 |
| record-A  | 100,000 |    1 |     1 |          84.962 |        83.942 |       +84.90 |         +53.16 |
| record-A  | 100,000 |   10 |     1 |           8.348 |         8.043 |        +8.29 |         -23.46 |
| record-A  | 100,000 |  100 |     1 |           0.993 |         0.963 |        +0.93 |         -30.81 |
| primtap   | 100,000 |    1 |     1 |         144.354 |       142.079 |      +144.30 |        +112.55 |
| primtap   | 100,000 |   10 |     1 |          78.375 |        75.883 |       +78.32 |         +46.57 |
| primtap   | 100,000 |  100 |     1 |          72.498 |        70.587 |       +72.44 |         +40.69 |

\* `vs manual` for vmap se>1 is not comparable: manual(l=8) fires 8 callbacks/step
(at se=1 cost), while verbose(l=8, se>1) fires far fewer.  See vmap note below.

### Config notes

- **body**: `c = c * 1.01 + jnp.sin(x)`, carry dim=8 float32, xs ~ N(0,1), seed=42
- **bare**: plain `lax.scan`, jitted — no callbacks
- **manual**: `jax.debug.callback(noop, carry, ordered=False)` per step — irreducible host-callback floor
- **verbose**: `tap.verbose(f, on_step=noop, sample_every=k)` — carry tap only; ops=(scan, while_loop)
- **record-A**: `with tap.record(on_step=noop, sample_every=k): f_jit()` — A-form context manager; function compiled inside first context; context enter/exit overhead excluded from timing window
- **primtap**: `tap.verbose(f, on_step=noop, se=k, taps=[tap.on("sin", select=lambda o: o[0][0])])` — carry tap + sin primitive tap combined; `se` gates carry tap only; **sin prim tap fires every step regardless of se**
- **vmap lanes=8**: only at N=10 000 to stay within ~15 min wall budget. lanes=8 at N=1 000 and N=100 000 skipped.
- **prim-tap-only**: not measurable; `ops=()` prevents walker descent into scan, silencing prim taps inside the body. The primtap arm = carry+prim combined.

---

## Interpretation

### 1. Overhead vs the manual arm (jaxtap machinery cost)

At N=10 000, se=1 (every step):
- **bare**: 0.1 µs/step
- **manual**: 32.9 µs/step (+32.8 µs vs bare — irreducible `jax.debug.callback` dispatch)
- **verbose**: 73.4 µs/step (+40.5 µs vs manual) — **FINDING: 2.23× manual**
- **record-A**: 85.2 µs/step (+52.3 µs vs manual) — **FINDING: 2.59× manual**
- **primtap**: 151.5 µs/step (+118.6 µs vs manual) — **FINDING: 4.61× manual**

All three jaxtap arms exceed the 2× manual threshold flagged in the spec.  The gap
comes from jaxtap's added work on each event: an extra `step` argument crossing the
host boundary, `TapEvent` dataclass construction, and the `_guard` wrapper call.
For primtap, the sin prim tap fires an additional host callback on top of the carry
tap, roughly doubling the callback cost per step.

These overheads are real; a careful user who hand-rolls `jax.debug.callback` will
pay ~33 µs/step; jaxtap verbose costs ~73 µs/step at se=1.

### 2. sample_every amortization curve

For **verbose** (carry tap only), amortization is near-linear:
- se=1 → se=10: 73 → 7.3 µs/step ≈ 10× reduction
- se=1 → se=100: 73 → 1.0 µs/step ≈ 73× reduction

A floor of ~0.9–1.0 µs/step remains at se=100 because the device-side `lax.cond`
(modulo check) runs every step even when the callback is skipped.

For **primtap**, amortization is poor because the sin primitive tap fires every step
regardless of se:
- se=1 → se=10: 152 → 78 µs/step (2× reduction, not 10×)
- se=1 → se=100: 152 → 70 µs/step (2.2× reduction, not 100×)

The prim tap dominates at large se.  Users should treat primitive taps as always-on
cost equal to carry tap at se=1, regardless of the sample_every setting.

### 3. vmap per-lane callback multiplication

With vmap lanes=8 at N=10 000:
- manual: 32.9 → 1877.6 µs/step (57× the single-lane cost, not 8×)
- verbose(se=1): 73.4 → 3152 µs/step (43× the single-lane cost, not 8×)

The super-linear scaling (57× vs 8×) indicates that host callbacks under vmap do not
dispatch in parallel.  Each of the 8 lanes serialises its callback through the JAX
callback dispatch, with additional per-callback OS and Python dispatch overhead that
compounds with lane count.  Under verbose(se=10) the per-step overhead drops to
59.2 µs (still measurable but far cheaper), suggesting that reducing callback
frequency is the only effective mitigation for large vmap batches.

### 4. ~32 µs anchor comparison

The prior-art measurement (proofs/attack-ledger-964, arm-a) found ~32 µs/step for
an unguarded `jax.debug.callback` on this box.  This benchmark's manual arm
reproduces that: 31.8–32.9 µs/step across N=1 000, 10 000, 100 000.  The anchor
is confirmed; per-step callback cost is stable across scan lengths.
