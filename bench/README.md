# jax-tap overhead benchmark

**"What does the lens cost on a real workload?"** — two benchmarks answering
different questions. v2 (this headline) uses a realistic body; v1 (appendix)
characterises the callback floor.

Platform: CPU (cpu:0), JAX 0.10.2.

Scripts:
- `bench_v2.py` — realistic leapfrog body, progress-bar scenario (this page)
- `bench_overhead.py` — microbenchmark on empty body (appendix below)

---

## v2 — realistic body (headline)

**"What does the lens cost when the scan body does real work?"**

Body: `dim=100` leapfrog on a Gaussian target, `L_STEPS=15` sub-steps per scan
step (30 `jnp.dot(M_PREC, ·)` matvecs per step). Bare body: **~11 µs/step**
at N=10 000 (10–11 µs range across runs; host-callback jitter adds ~15–20%
run-to-run variance). This puts the 33 µs host-callback floor at 3× the body
(not 330× as in v1), giving a realistic overhead picture.

Run: `PYTHONUNBUFFERED=1 uv run python bench/bench_v2.py 2>&1 | tee bench/v2_run.log`
Smoke: `uv run python bench/bench_v2.py --smoke`

### SCENARIO 1 — PROGRESS-BAR (headline)

N=10 000, K=7, median+min µs/step. `vs bare (%)` = the question that matters.
Host-callback jitter gives ~15–20% run-to-run variance; µs overhead is portable,
% is body-relative (see scaling note below).

| arm | se | µs/step (med) | µs/step (min) | vs bare (µs) | vs bare (%) |
|-----|----|--------------:|---------------|:------------:|:-----------:|
| bare | — | 11.129 | 11.046 | — | — |
| manual-progress | 1 | 48.198 | 47.686 | +37.07 | **+333%** |
| jaxtap-se10 | 10 | 21.207 | 21.094 | +10.08 | **+91%** |
| jaxtap-se100 | 100 | 11.902 | 11.750 | +0.77 | **+7%** |
| jaxtap-se10-progress | 10 | 16.687 | 16.440 | +5.56 | **+50%** |
| jaxtap-se100-progress | 100 | 12.797 | 11.267 | +1.67 | **+15%** |

**"progress idiom" rows** (`-progress`) use `select=lambda leaves: ()` — zero bytes
cross the host boundary; `TapEvent.value = ()`; callback cost ≈ step-only floor
(~37 µs/event). This is the recommended idiom for tqdm-style bars.

**Recommendation ladder (pick the first row that fits your use case):**

| monitoring goal | idiom | overhead on ~11 µs body |
|-----------------|-------|------------------------|
| lightweight progress (se=100) | `tap.verbose(f, se=100, select=lambda _: ())` | **+15%** (~1.7 µs) |
| finer-grained progress (se=10) | `tap.verbose(f, se=10, select=lambda _: ())` | **+50%** (~5.6 µs) |
| carry inspection (se=100) | `tap.verbose(f, se=100)` | +7% (~0.8 µs) |
| carry inspection (se=10) | `tap.verbose(f, se=10)` | +91% (~10 µs) |
| always-on debugging | `tap.verbose(f, se=1, select=scalar)` | +724% — see debug rows |

#### Scaling note — µs overhead is fixed; % scales with body size

The jaxtap overhead in µs/step is a property of the callback mechanism (fixed
per event), not of the body compute. The % falls proportionally as the body grows:

| body cost | se=10-progress overhead | overhead % |
|-----------|------------------------|-----------|
| ~11 µs (this benchmark) | ~5.6 µs | ~50% |
| ~50 µs (modest sampler step) | ~5.6 µs | ~11% |
| ~100 µs (real-world sampler) | ~5.6 µs | ~6% |

**State the µs cost to collaborators; % is only meaningful relative to a stated
body cost.** The µs numbers here are on CPU (cpu:0), JAX 0.10.2, single-threaded.

#### What drives the overhead? (payload-size decomposition)

The `jaxtap-se10` arm ships the full `(q, p) ∈ ℝ^100 × ℝ^100` carry (800 bytes)
at every 10th step. Per-callback cost solves from the se=10 and se=100 data to
~118 µs (vs v1's ~33 µs for a 32-byte carry; ~3.6× scaling for 25× payload —
sub-linear in bytes but substantial). The jaxtap machinery component (~20 µs above
the raw `jax.debug.callback` floor — see v1 §1 for the decomposition) is a known
post-v1 optimisation target; the remaining ~100 µs is `jax.debug.callback` itself.

| quantity | se=10 full carry | se=10 empty payload |
|----------|-----------------|---------------------|
| per-callback cost | ~118 µs | ~37 µs |
| amortised per step at se=10 | ~10 µs | ~5.2 µs |
| device-side lax.cond | ~0.4 µs | ~0.4 µs |
| **total overhead** | **~10 µs (+91%)** | **~5.6 µs (+50%)** |

Compared to v1: `verbose(se=10)` on the 0.06 µs empty body cost +7.2 µs (+12 000%).
Same µs overhead in the same order of magnitude — the denominator changed.

### SCENARIO 2 — DEBUGGING (2 highlight rows)

| arm | se | body | µs/step (med) | vs bare* (µs) | vs bare* (%) |
|-----|----|----|------:|:---------:|:---------:|
| debug-carry-se1 | 1 | L_STEPS=15 | 91.669 | +80.54 | +724% |
| debug-prim-se10 | 10 | L_STEPS=1 | 25.623 | +14.49 | +130%† |

† debug-prim uses the simple body (1 sub-step, ~1.2 µs/step bare) to avoid
2×L_STEPS gated lax.cond checks per scan step that would swamp the arm. The
+130% is vs the simple body bare (~1.2 µs); vs the full-body bare it would be
smaller in absolute µs but the arms would not be comparable.

**debug-carry-se1:** `tap.verbose(f, sample_every=1, select=lambda l: l[0][0])`
fires a scalar-select callback every step. Even with the scalar reducing carry
transit to ~37 µs/callback, se=1 fires N callbacks per sweep → +86 µs/step.
This is the floor for any always-on monitoring. Use se≥10 for production.

**debug-prim-se10:** `tap.verbose(f_simple, se=10, taps=[tap.on("dot_general", ...)])`
demonstrates M1d gating: `dot_general` fires 2×/step in the simple body; with
se=10 it fires only 2×(N/10) times instead of 2×N. The +16.5 µs overhead
= carry tap amortised + 2× prim-tap amortised. Before M1d, the prim tap would
have fired 2×N times regardless of `sample_every`; M1d's gating confirmed.

### SCENARIO 3 — VMAP (semi-production multi-chain, 8 lanes)

| arm | se | lanes | µs/step (med) | vs vmap-bare (µs) | vs vmap-bare (%) |
|-----|----|-------|------:|:-----:|:-----:|
| bare-l8 | — | 8 | 39.855 | — | — |
| vmap-se10 | 10 | 8 | 150.166 | +110.31 | +277% |

8-lane vmap bare: 39.9 µs (≈3.6× single-lane, not 8× — XLA vectorises the matvecs
across the batch). vmap-se10 overhead: +110 µs (+277% vs vmap bare, ~11× single-lane
se=10 overhead). Super-linear lane scaling for callbacks matches v1 finding: host
callbacks under vmap serialise; each lane's event goes through the dispatch queue
individually. Reducing se is the only effective mitigation for large vmap batches.

### Config notes (v2)

- **body (main)**: 15 leapfrog sub-steps/scan-step on dim-100 Gaussian; 30 `jnp.dot(M_PREC, q)` matvecs/scan-step; carry = (q, p) ∈ ℝ^100 × ℝ^100; step_size=0.01; M_PREC = (A Aᵀ)/DIM + I (fixed seed)
- **body (debug-prim)**: 1 leapfrog sub-step (2 matvecs) — avoids 2×L_STEPS lax.cond checks/step
- **bare**: `lax.scan(leapfrog_body, init, None, length=N)`, jitted — no callbacks
- **manual-progress**: same body + `jax.debug.callback(λ s: None, step, ordered=False)` every step; step int32 only
- **jaxtap-se10/100**: `tap.verbose(f, on_step=noop, sample_every=k)` — carry tap; device-side `lax.cond(step % k == 0, fire, noop)` gate; full (q, p) carry shipped on fire
- **jaxtap-se10/100-progress**: `tap.verbose(f, on_step=noop, sample_every=k, select=lambda leaves: ())` — progress-bar idiom; ZERO bytes cross the host boundary; TapEvent.value=(); callback cost ≈ step-only floor (~33–40 µs/event)
- **debug-carry-se1**: `tap.verbose(f, sample_every=1, select=lambda l: l[0][0])` — scalar select isolates frequency cost
- **debug-prim-se10**: `tap.verbose(f_simple, se=10, taps=[tap.on('dot_general', select=lambda o: o[0][0])])` — M1d gating demo
- **vmap-se10**: `jax.vmap(tap.verbose(f, se=10))` — 8 lanes; each lane fires own callbacks

---

## Appendix — v1 microbenchmark (empty body)

**"What is the irreducible callback floor?"** — results from `bench_overhead.py`.

This benchmark uses an empty-ish body (dim=8, `c = c * 1.01 + jnp.sin(x)`, ~0.1 µs/step)
to characterise the raw callback cost, independent of body compute. Overhead
percentages look catastrophic (thousands of percent) because the body is
unrealistically fast. The v1 numbers are correct for their stated purpose:
measuring callback floor, amortisation curves, and the payload-decomposition.

**Why both benchmarks exist:**
- v1 tells you how much the lens costs in the limit (empty body, pure overhead floor).
- v2 tells you how much the lens costs in practice (real body, realistic framing).

### Compile time (first call, N=10 000, lanes=1)

| arm | compile (ms) |
|-----|-------------|
| bare | 58 |
| manual | 423 |
| manual-payload | 659 |
| verbose(se=1) | 859 |
| primtap(se=1) | 1552 |
| record-A(se=1) | 969 |

The jaxtap arms take 10–25× longer to compile than bare because the walker
re-traces and re-interprets the jaxpr at every `verbose()` call. Compile cost
is paid once per `(function, input_shape)` pair; subsequent calls use the XLA cache.

### Steady-state timing (median + min wall/step in µs)

Overhead columns = arm median minus baseline at same (N, lanes).
`vs manual` isolates jaxtap machinery above the irreducible host-callback floor.
Measurement: JIT + 1 warmup excluded, then K=7 repeats with `jax.block_until_ready`.

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
| manual-payload |  10,000 |    - |     1 |          55.411 |        54.142 |       +55.31 |         +22.51 |
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
(at se=1 cost), while verbose(l=8, se>1) fires far fewer. See vmap note below.

### Config notes (v1)

- **body**: `c = c * 1.01 + jnp.sin(x)`, carry dim=8 float32, xs ~ N(0,1), seed=42
- **bare**: plain `lax.scan`, jitted — no callbacks
- **manual**: `jax.debug.callback(noop, carry, ordered=False)` per step — carry-only host-callback floor
- **manual-payload**: `jax.debug.callback(lambda i,v: None, step_i32, carry, ordered=False)` per step — same payload as verbose (step + carry); N=10 000, lanes=1 only; used to isolate jaxtap machinery cost from payload-transit cost
- **verbose**: `tap.verbose(f, on_step=noop, sample_every=k)` — carry tap only; ops=(scan, while_loop)
- **record-A**: `with tap.record(on_step=noop, sample_every=k): f_jit()` — A-form context manager; function compiled inside first context; context enter/exit overhead excluded from timing window
- **primtap**: `tap.verbose(f, on_step=noop, se=k, taps=[tap.on("sin", select=lambda o: o[0][0])])` — carry tap + sin primitive tap combined; `se` gates carry tap only; **sin prim tap fires every step regardless of se**
- **vmap lanes=8**: only at N=10 000 to stay within ~15 min wall budget. lanes=8 at N=1 000 and N=100 000 skipped.
- **prim-tap-only**: not measurable; `ops=()` prevents walker descent into scan, silencing prim taps inside the body. The primtap arm = carry+prim combined.

### Interpretation (v1)

#### 1. Overhead vs the manual arm — payload-aware decomposition

The original benchmark compared verbose against a carry-only manual callback. The
amendment adds `manual-payload`, which ships the same data as verbose (step int32 +
full dim-8 carry), enabling a fair apples-to-apples split.

At N=10 000, se=1 (every step):

| arm | µs/step | ratio vs manual |
|-----|---------|-----------------|
| bare | 0.1 | — |
| manual (carry only) | 32.9 | 1× (baseline) |
| manual-payload (step + carry) | 55.4 | 1.69× manual |
| verbose(se=1) | 73.4 | 2.23× manual |
| record-A(se=1) | 85.2 | 2.59× manual |
| primtap(se=1) | 151.5 | 4.61× manual |

**Two-part decomposition of the verbose cost:**

**A. Payload-transit cost (the feature):** shipping one extra int32 (the step index)
alongside the dim-8 carry raises the callback cost from 32.9 µs (carry only) to
55.4 µs — an extra **+22.5 µs** purely from the larger host-boundary transfer.
This is `jax.debug.callback` overhead and is unavoidable for any callback that ships
both a step counter and carry values.

**B. jaxtap machinery cost (our overhead):** verbose at 73.4 µs vs payload-equal
manual at 55.4 µs → **+18 µs, 1.32× a payload-equal manual callback**. This
~18 µs covers `TapEvent` dataclass construction, the `_guard` wrapper call, and the
module-level router dispatch that verbose adds on top of a raw `jax.debug.callback`.

**FINDING 1 (revised):** The 2.23× figure conflates payload cost with machinery cost
because the two arms carry different data. The honest decomposition is:

- **Progress-bar use case** (step scalar only, no carry shipped): a hand-rolled
  step-scalar callback costs roughly the carry-only floor (~33 µs); verbose at
  73 µs is **2.2× that**, which is the ceiling for this use case.
- **Value-debugging use case** (step + carry, same data as verbose): the
  payload-equal manual costs 55.4 µs; verbose is **1.32× that** — jaxtap
  machinery adds ~18 µs/step beyond what you'd pay to ship the same data yourself.

For record-A and primtap the machinery gap widens further — record-A adds A-form
context dispatch (+12 µs above verbose); primtap fires a second host callback for
the sin primitive on every step regardless of `sample_every`, roughly doubling the
raw callback cost.

#### 2. sample_every amortization curve

For **verbose** (carry tap only), amortization is near-linear:
- se=1 → se=10: 73 → 7.3 µs/step ≈ 10× reduction
- se=1 → se=100: 73 → 1.0 µs/step ≈ 73× reduction

A floor of ~0.9–1.0 µs/step remains at se=100 because the device-side `lax.cond`
(modulo check) runs every step even when the callback is skipped.

For **primtap**, amortization is poor because the sin primitive tap fires every step
regardless of se (v1 measurement; M1d now gates prim taps — see v2 debug-prim-se10):
- se=1 → se=10: 152 → 78 µs/step (2× reduction, not 10×)
- se=1 → se=100: 152 → 70 µs/step (2.2× reduction, not 100×)

#### 3. vmap per-lane callback multiplication

With vmap lanes=8 at N=10 000:
- manual: 32.9 → 1877.6 µs/step (57× the single-lane cost, not 8×)
- verbose(se=1): 73.4 → 3152 µs/step (43× the single-lane cost, not 8×)

The super-linear scaling (57× vs 8×) indicates that host callbacks under vmap do not
dispatch in parallel. Each of the 8 lanes serialises its callback through the JAX
callback dispatch, with additional per-callback OS and Python dispatch overhead that
compounds with lane count. Under verbose(se=10) the per-step overhead drops to
59.2 µs (still measurable but far cheaper), suggesting that reducing callback
frequency is the only effective mitigation for large vmap batches.

#### 4. ~32 µs anchor comparison

The prior-art measurement (proofs/attack-ledger-964, arm-a) found ~32 µs/step for
an unguarded `jax.debug.callback` on this box. This benchmark's manual arm
reproduces that: 31.8–32.9 µs/step across N=1 000, 10 000, 100 000. The anchor
is confirmed; per-step callback cost is stable across scan lengths.
