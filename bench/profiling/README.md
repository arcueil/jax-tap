# bench/profiling — measurement corpus for perf/emission-machinery

These scripts document the measurement evidence behind each optimization
on the `perf/emission-machinery` branch.  Run from the project root:

```
cd /home/jp/arcueil/jax-tap-perf
uv run python bench/profiling/<script>.py
```

---

## Scripts

### `profile_inside_cb.py`
**What it measures**: Decomposes the per-event host-callback overhead into
individual components using synthetic `jax.debug.callback` arms (A–H).

Arms:
- A: noop lambda (establishes that JAX dispatch structure costs ≈ 0 µs)
- B: `step_.item()` only (cost of scalar extraction via `.item()`)
- C: `int(step_)` only (cost of scalar extraction via JAX `__int__`)
- D: B + TapEvent construction (no guard)
- E: C + TapEvent construction (no guard)
- F: B + TapEvent + `_guard` — what `verbose()` does **after OPT 1**
- G: C + TapEvent + `_guard` — what `verbose()` did **before OPT 1**
- H: actual `tap.verbose(f, on_step=noop)` — validation arm (should ≈ F)

Key finding: `int(jax.Array)` costs ~16 µs inside the JAX callback thread
(via `check_scalar_conversion` → profiler wrapper → `_value`); `.item()`
costs ~5 µs (direct `_value` → numpy `.item()`).  OPT 1 saves ~11 µs/event.

### `check_cb_types.py`
**What it measures**: Runtime type and shape of `step_` (and carry leaves)
delivered to `jax.debug.callback` host functions — inside a plain scan AND
under `jax.vmap`.

Key findings:
- `step_` is always `jaxlib._jax.ArrayImpl`, shape=(), dtype=int32
- Under `jax.vmap(f)` with LANES lanes: JAX fires the callback LANES×N times,
  each invocation receiving a **scalar** (shape=()) step_ — NOT a batched
  (shape=(LANES,)) array.  `.item()` is safe under vmap.
- Conversion costs in direct Python context: `int(jax.Array)` = 2.3 µs;
  `jax.Array.item()` = 1.1 µs (callback thread context costs ~16 µs vs ~5 µs
  due to JAX profiler and GIL overhead — see profile_inside_cb.py ARM B vs C).

### `targeted_empty_payload.py`
**What it measures**: Two things — (1) the data-transfer saving from
`select=lambda _: ()` vs full carry at se=1; (2) the cost of
`jax.tree_util.tree_unflatten(empty_tree, [])` vs a direct `value = ()`
assignment, to quantify the skip-unflatten fast-path hypothesis.

Key finding: `verbose(se=1, select=lambda _: ())` saves **~26 µs/step** vs
`verbose(se=1)` with full carry at DIM=8 — but this saving is inherent
(no carry shipped across host boundary).  The specific skip-unflatten
hypothesis (`if not flat_vals: value = ()`) saves **0.12 µs**
(`tree_unflatten(empty_tree, [])` = 0.155 µs; direct `()` = 0.031 µs).
Below the 1 µs gate.  **[MEASURED — 0.12 µs saving — hypothesis rejected]**

---

## Durable knowledge from the perf/emission-machinery arc

### Root cause: where the ~18–21 µs machinery came from

All overhead is in the **host-side `_host()` closure body**, not in JAX
dispatch structure.  ARM A (noop with identical structure) shows +0.9 µs vs
manual-payload — structural overhead is negligible.

The two components that mattered:

**1. `int(step_)` — the dominant cost (~11–16 µs/event in-callback)**

`jax.Array.__int__()` goes through:
```
check_scalar_conversion()  →  JAX profiler wrapper  →  _value  →  Python int()
```
The profiler wrapper acquires a lock and calls into C++ on every invocation.
`jax.Array.item()` bypasses this: `_value → numpy_scalar.item()`.

| call | direct Python µs | callback thread µs |
|------|------------------|--------------------|
| `int(jax.Array)` | 2.34 | ~16 |
| `jax.Array.item()` | 1.07 | ~5 |
| `int(numpy.int32)` | 0.056 | — |
| `numpy.int32.item()` | 0.24 | — |

The callback thread cost is ~7× higher than direct Python due to JAX
dispatching callbacks from a C++ thread with Python GIL acquire on each call.

**Fix (OPT 1):** all three `_host` closures in `__init__.py` changed from
`int(step_)` to `step_.item()`.  Saving: ~11 µs/event.

**2. Per-call imports in `_dynamic_router` (~1.6 µs/event)**

The main 307394d carry-alert merge added `from . import _fire_carry_alert, _guard`
as a lazy per-call import inside `_dynamic_router` (noqa'd to avoid circular import
at module load).  `importlib._handle_fromlist` costs ~0.5 µs per call even when
the module is already loaded (dict lookup + attribute check).

Additionally, `_active_contexts()` (list allocation + weakref genexpr) costs
~1.1 µs, and the n=1 case (overwhelmingly the common case) doesn't need it.

**Fix (OPT 2):** module-level `_guard_fn` and `_carry_alert_fn` cache references
on first call; n=1 fast path inlines the single weakref deref and skips list
allocation + `_select_ctx()` entirely.  Saving: ~1.6 µs/event.

### Irreducible floor after OPT 1+2

At se=1, single context, no alert, DIM=8 carry:

| component | µs |
|-----------|-----|
| JAX callback dispatch overhead | ~0.9 |
| `step_.item()` | ~5.0 |
| `TapEvent` construction | ~1.6 |
| `_guard` try/except | ~4.8 |
| **total above manual-payload** | **~11–12** |

ARM H (actual `tap.verbose`) measured at ~7 µs above manual-payload after
OPT 1+2 — lower than the synthetic F arm sum because OPT 2 reduces router
overhead not captured in the ARM F measurement.

### Candidates tested and rejected (all below 1 µs gate)

| hypothesis | µs saving (measured) | status |
|------------|---------------------|--------|
| `TapEvent.__slots__` (FastEvent) | 0.17 | **REJECTED** — direct Python context only; negligible in callback thread |
| `_guard` try/except elimination | < 0.3 | **REJECTED** — guard is contractual; semantics |
| skip-unflatten for `select=()` | 0.12 | **REJECTED** — `tree_unflatten(empty_tree,[])` costs 0.155 µs; direct `()` is 0.031 µs |
| empty-payload TapEvent skip (`value==()`) | < 0.3 | **REJECTED** — TapEvent with empty tuple is already cheap |

### se=100 arm noise characterisation

At se=100, N=10,000: only 100 callbacks fire per run.  The K=7 bench shows
high variance (~0.4 µs peak-to-trough on a 0.9 µs median) because 100 events
× ~63 µs/event = 6.3 ms amortized over a 9-second wall run — scheduling jitter
dominates.

At K=25:
- main 307394d:       0.889 µs/step ± 0.048 µs  (stdev)
- perf/emission-mach: 0.795 µs/step ± 0.022 µs  (stdev)

The K=7 "regression" (0.906 → 1.338 µs) was sampling noise — at K=25 the
perf branch is actually slightly faster (−0.09 µs; ~10%), consistent with
OPT 2 savings amortized over fewer firings.  **Not a regression.**

---

## Baseline numbers referenced in commit messages

All measurements: CPU, N=10,000, K=7 (bench/callback_floor.py), jax 0.10.2.

| checkpoint | verbose(se=1) µs/step | manual-payload µs/step | machinery µs |
|------------|----------------------|----------------------|-------------|
| main 4d62fa4 (pre-alert) | 75.2 | 54.1 | 21.1 |
| main 307394d (post-alert) | 72.7 | 54.6 | 18.1 |
| perf/emission-machinery   | 61.4 | 53.2 | 8.2 |
