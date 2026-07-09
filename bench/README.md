# jax-tap overhead benchmark

**"What does the lens cost on a real workload?"** — three benchmark files answering
different questions. `progress_bar.py` is the headline; `debug_taps.py` covers
debugging configurations and the first nested-scan datapoint; `callback_floor.py`
characterises the raw callback floor.

Platform: CPU (cpu:0), JAX 0.10.2.

Scripts:
- `bench/progress_bar.py` — progress-bar monitoring, nested-scan body (headline)
- `bench/debug_taps.py` — debugging scenarios + nested-tap volume datapoint
- `bench/callback_floor.py` — microbenchmark on empty body (callback floor isolation)

Body construction changed in the restructure: the leapfrog body now uses an inner
`lax.scan` (nested scan) instead of a Python `for` loop.  Bare body: **~8.7 µs/step**
(nested-scan, N=10 000) vs ~11.1 µs/step (old unrolled).  The nested-scan version
compiles to a more efficient XLA plan: **−2.4 µs/step** vs old unrolled.

---

## progress_bar.py — nested-scan leapfrog, outer-scan-only tapping (headline)

**"What does the lens cost for a progress bar on a realistic workload?"**

Body: dim=100 leapfrog on a Gaussian target, `L_STEPS=15` sub-steps per outer scan
step via a **nested `lax.scan`**.  All jaxtap arms tap the **outer scan only**
(`where=lambda p: p == "scan[0]"`); the inner leapfrog scan is walked but silent.

Run: `PYTHONUNBUFFERED=1 uv run python bench/progress_bar.py 2>&1 | tee bench/progress_bar_run.log`
Smoke: `uv run python bench/progress_bar.py --smoke`

### SCENARIO 1 — PROGRESS-BAR (headline)

N=10 000, K=7, median+min µs/step.  `vs bare (%)` = the overhead-relative-to-body number.
Host-callback jitter gives ~15–20% run-to-run variance; µs overhead is portable,
% is body-relative.

| arm | se | µs/step (med) | µs/step (min) | vs bare (µs) | vs bare (%) |
|-----|----|--------------:|--------------:|:------------:|:-----------:|
| bare | — | 8.718 | 8.709 | — | — |
| manual-progress | 1 | 40.266 | 38.920 | +31.55 | **+362%** |
| jaxtap-se10 | 10 | 19.199 | 18.845 | +10.48 | **+120%** |
| jaxtap-se100 | 100 | 10.292 | 10.215 | +1.57 | **+18%** |
| jaxtap-se10-progress | 10 | 14.539 | 14.126 | +5.82 | **+67%** |
| jaxtap-se100-progress | 100 | 9.897 | 9.743 | +1.18 | **+14%** |

**"progress idiom" rows** (`-progress`) use `select=lambda _: ()` — zero bytes cross
the host boundary; `TapEvent.value = ()`; callback cost ≈ step-only floor.

**Manual-progress floor sanity check (Change 3):** the manual-progress delta
(+31.55 µs) sits inside the 25–50 µs expected range and is consistent with the
~33 µs/step raw-callback floor measured in `callback_floor.py`.  No anomaly.

**Body change delta:** nested-scan bare = 8.72 µs/step vs old unrolled bare = 11.13 µs/step
→ **−2.41 µs/step** (−22%).  The nested lax.scan compiles more efficiently than the
unrolled Python loop.  All `vs bare (%)` numbers are relative to the new body.

**Recommendation ladder (pick the first row that fits):**

| monitoring goal | idiom | overhead on ~8.7 µs body |
|-----------------|-------|--------------------------|
| lightweight progress (se=100) | `tap.verbose(f, se=100, where=outer, select=lambda _: ())` | **+14%** (~1.2 µs) |
| finer-grained progress (se=10) | `tap.verbose(f, se=10, where=outer, select=lambda _: ())` | **+67%** (~5.8 µs) |
| carry inspection (se=100) | `tap.verbose(f, se=100, where=outer)` | +18% (~1.6 µs) |
| carry inspection (se=10) | `tap.verbose(f, se=10, where=outer)` | +120% (~10.5 µs) |
| always-on debugging | `tap.verbose(f, se=1, select=scalar)` | see debug rows |

#### Scaling note — µs overhead is fixed; % scales with body size

The jaxtap overhead in µs/step is a property of the callback mechanism (fixed per
event), not the body compute.  The % falls proportionally as the body grows:

| body cost | se=10-progress overhead | overhead % |
|-----------|------------------------|-----------|
| ~8.7 µs (this benchmark, nested-scan) | ~5.8 µs | ~67% |
| ~50 µs (modest sampler step) | ~5.8 µs | ~12% |
| ~100 µs (real-world sampler) | ~5.8 µs | ~6% |

**State the µs cost to collaborators; % is only meaningful relative to a stated body cost.**

#### Payload-size decomposition

The `jaxtap-se10` arm ships the full `(q, p) ∈ ℝ^100 × ℝ^100` carry (800 bytes)
at every 10th step.  Per-callback cost from se=10 vs se=100 data: ~109 µs/event
(amortised to ~10.5 µs/step at se=10).  The progress idiom (`select=()`) reduces
that to ~37 µs/event (amortised to ~5.8 µs/step).

| quantity | se=10 full carry | se=10 empty payload |
|----------|-----------------|---------------------|
| per-callback cost | ~109 µs | ~37 µs |
| amortised per step at se=10 | ~10.5 µs | ~5.8 µs |
| device-side lax.cond | ~0.4 µs | ~0.4 µs |
| **total overhead** | **~10.5 µs (+120%)** | **~5.8 µs (+67%)** |

### Config notes (progress_bar.py)

- **body**: nested-scan leapfrog — outer `lax.scan` of N steps; each outer step runs `lax.scan(leapfrog_step, carry, None, length=15)`; dim=100, step_size=0.005; carry = (q, p) ∈ ℝ^100 × ℝ^100; M_PREC = (A Aᵀ)/DIM + I (fixed, seeded)
- **outer-scan-only tapping**: all jaxtap arms use `where=lambda p: p == "scan[0]"`; walker descends into inner leapfrog scan but emits no taps there
- **bare**: `lax.scan(body, init, None, length=N)`, jitted — no callbacks; body contains inner lax.scan
- **manual-progress**: same nested-scan body + `jax.debug.callback(λ s: None, step, ordered=False)` every step; step int32 only, no carry shipped
- **jaxtap-se10/100**: `tap.verbose(f, on_step=noop, sample_every=k, where=outer)` — carry tap; full (q, p) carry (800 bytes) shipped on fire
- **jaxtap-se10/100-progress**: `tap.verbose(f, on_step=noop, sample_every=k, where=outer, select=lambda _: ())` — progress-bar idiom; ZERO bytes cross the host boundary; TapEvent.value=()

---

## debug_taps.py — debugging configurations + nested-scan datapoint

**"What does the lens cost in debugging configurations, and how expensive is tapping a nested scan?"**

Run: `PYTHONUNBUFFERED=1 uv run python bench/debug_taps.py 2>&1 | tee bench/debug_taps_run.log`
Smoke: `uv run python bench/debug_taps.py --smoke`

### SCENARIO 2 — DEBUGGING

N=10 000, K=7.  Bare body (nested-scan): 8.75 µs/step.

| arm | se | body | µs/step (med) | vs bare (µs) | vs bare (%) |
|-----|----|----|------:|:---------:|:---------:|
| debug-carry-se1 | 1 | nested L_STEPS=15 | 82.703 | +73.95 | +845% |
| bare-simple | — | L_STEPS=1 | 0.971 | — | — |
| debug-prim-se10 | 10 | simple L_STEPS=1 | 18.337 | +17.37† | +1788%† |
| bare-l8 (vmap) | — | nested L_STEPS=15 | 36.381 | — | — |
| vmap-se10 | 10 | nested L_STEPS=15 | 145.629 | +109.25‡ | +300%‡ |

† vs simple-bare (0.971 µs/step).  ‡ vs vmap-bare (36.381 µs/step).

**debug-carry-se1:** `tap.verbose(f, se=1, where=outer, select=lambda l: l[0][0])`
fires a scalar-select callback every step.  +73.95 µs/step is the always-on monitoring
floor on this body.  Use se≥10 for production.

**debug-prim-se10:** `tap.verbose(f_simple, se=10, taps=[tap.on("dot_general", ...)])`
on the simple body (L_STEPS=1, 2 matvecs/step).  M1d gating confirmed: with se=10,
dot_general fires 2×(N/10) times instead of 2×N.  Simple body used to avoid
2×L_STEPS lax.cond checks per scan step that would swamp the arm.

**vmap-se10:** 8-lane vmap on the nested-scan body.  vmap bare = 36.4 µs (≈4.2× single-lane).
vmap-se10 overhead: +109 µs vs vmap-bare (+300%).  Super-linear lane scaling for
callbacks (same pattern as v1 finding): host callbacks under vmap serialise.

### SCENARIO 3 — NESTED-TAP VOLUME (first datapoint, deferred nested-scan bench)

Same nested-scan body, se=10.  Arm A: outer-only (`where=lambda p: p == "scan[0]"`).
Arm B: no filter (both outer + inner leapfrog scan tapped).

| arm | se | µs/step (med) | vs bare (µs) |
|-----|----|--------------:|:------------:|
| nested-outer-only | 10 | 22.017 | +13.27 |
| nested-both-levels | 10 | 212.055 | +203.31 |
| **delta (both − outer)** | — | **—** | **+190.04 µs/step** |

**Inner-scan emission overhead: +190 µs/step** at se=10 with L_STEPS=15.  The inner
scan fires L_STEPS=15 heartbeats per outer step; at se=10 the inner carry taps fire
on N/10 outer steps, adding 15 callbacks per outer-step-tapped.  This is the first
datapoint for the deferred nested-scan benchmarking task.

**Consequence for users:** if you call `tap.verbose(f)` without `where=` on a function
containing a nested scan, the inner scan fires heartbeats and the overhead can be
substantial.  Always use `where=lambda p: p == "scan[0]"` (or a depth limit) for
outer-only monitoring on functions with nested control flow.

### Config notes (debug_taps.py)

- **debug-carry-se1**: `tap.verbose(f, se=1, where=outer, select=lambda l: l[0][0])` — scalar select (q[0]) isolates FREQUENCY cost; se=1 fires N callbacks/sweep
- **debug-prim-se10**: `tap.verbose(f_simple, se=10, taps=[tap.on('dot_general', select=lambda o: o[0][0])])` — simple body (L_STEPS=1); M1d gating demo
- **vmap-se10**: `jax.vmap(tap.verbose(f, se=10, where=outer))` — 8 lanes; nested-scan body; outer-only
- **nested-outer-only**: `tap.verbose(f, se=10, where=lambda p: p == "scan[0]")` — nested body; outer scan only
- **nested-both-levels**: `tap.verbose(f, se=10)` — nested body; no filter; both scan levels emit

---

## callback_floor.py — microbenchmark (empty body, callback floor)

**"What is the irreducible callback floor?"** — renamed from `bench_overhead.py`.

Empty-ish body (dim=8, `c = c * 1.01 + jnp.sin(x)`, ~0.1 µs/step) to characterise
raw callback cost independent of body compute.  Overhead percentages look large
because the body is unrealistically fast — these numbers isolate the floor, not
production overhead.

Run: `PYTHONUNBUFFERED=1 uv run python bench/callback_floor.py 2>&1 | tee bench/callback_floor_run.log`
Smoke: `uv run python bench/callback_floor.py --smoke`

**Why both benchmarks exist:**
- `callback_floor.py` tells you the irreducible per-callback cost (~33 µs/step floor).
- `progress_bar.py` tells you what that cost looks like in practice (body ~8.7 µs,
  callbacks from +14% to +120% overhead).

### Compile time (first call, N=10 000, lanes=1)

| arm | compile (ms) |
|-----|-------------|
| bare | 58 |
| manual | 423 |
| manual-payload | 659 |
| verbose(se=1) | 859 |
| primtap(se=1) | 1552 |
| record-A(se=1) | 969 |

### Steady-state timing (median + min wall/step in µs)

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

\* `vs manual` for vmap se>1 not comparable: manual(l=8) fires 8 callbacks/step,
verbose(l=8, se>1) fires far fewer.

### Payload-size decomposition (v1 finding)

At N=10 000, se=1:

| arm | µs/step | ratio vs manual |
|-----|---------|-----------------|
| bare | 0.1 | — |
| manual (carry only) | 32.9 | 1× baseline |
| manual-payload (step + carry) | 55.4 | 1.69× |
| verbose(se=1) | 73.4 | 2.23× |
| record-A(se=1) | 85.2 | 2.59× |
| primtap(se=1) | 151.5 | 4.61× |

**Two-part decomposition:**
- **Payload-transit**: shipping one extra int32 raises cost from 32.9 → 55.4 µs (+22.5 µs from `jax.debug.callback` itself, unavoidable for step+carry).
- **jaxtap machinery**: verbose 73.4 µs vs payload-equal manual 55.4 µs → **+18 µs** (1.32× a payload-equal callback). Covers TapEvent construction, `_guard` wrapper, and router dispatch.

**Sample-every amortisation (verbose, carry only):**
- se=1 → 73 µs/step; se=10 → 7.3 µs/step (~10× reduction); se=100 → 1.0 µs/step (~73×). Floor ~0.9–1.0 µs/step at se=100 (device-side lax.cond runs every step).

### Config notes (callback_floor.py)

- **body**: `c = c * 1.01 + jnp.sin(x)`, carry dim=8 float32, xs ~ N(0,1), seed=42
- **bare**: plain `lax.scan`, jitted — no callbacks
- **manual**: `jax.debug.callback(noop, carry, ordered=False)` per step — raw callback floor
- **manual-payload**: `jax.debug.callback(lambda i,v: None, step_i32, carry, ordered=False)` — step+carry payload; isolates jaxtap machinery cost; N=10 000, lanes=1 only
- **verbose**: `tap.verbose(f, on_step=noop, sample_every=k)` — carry tap; ops=(scan, while_loop)
- **record-A**: A-form context manager; function compiled inside first context; enter/exit overhead excluded
- **primtap**: carry+sin prim tap combined; se gates carry tap only; sin prim fires every step (pre-M1d; M1d now gates prim taps — see debug-prim-se10 in debug_taps.py)
- **vmap lanes=8**: only at N=10 000 (wall budget)
- **prim-tap-only**: not measurable; `ops=()` prevents walker descent into scan. primtap arm = carry+prim combined.
