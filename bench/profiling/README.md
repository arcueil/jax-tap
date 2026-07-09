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

Key finding: `int(jax.Array)` costs ~14 µs inside the JAX callback thread
(via `check_scalar_conversion` → profiler wrapper → `_value`); `.item()`
costs ~1.2 µs (direct `_value` → numpy `.item()`).  OPT 1 saves ~12 µs/event.

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
  `jax.Array.item()` = 1.1 µs (callback thread context costs ~14 µs vs ~1.2 µs
  due to JAX profiler and GIL overhead — see profile_inside_cb.py ARM B vs C).

### `targeted_empty_payload.py`
**What it measures**: Whether an empty-payload (`select=lambda _: ()`) code
fast-path is worth implementing at `se=1`.

Key finding: `verbose(se=1, select=lambda _: ())` saves **~26 µs/step** vs
`verbose(se=1)` with full carry at DIM=8 — but this saving is inherent
(no carry shipped across host boundary, not a code optimisation).  The
machinery overhead for the empty case is **~6 µs** above the step-only
manual-progress floor, already close to the `step_.item() + TapEvent(value=()) + _guard`
minimum.  A hypothetical code fast-path skipping TapEvent for `value=()` would
save < 0.5 µs — below the 1 µs gate.  **Hypothesis correctly rejected.**

---

## Baseline numbers referenced in commit messages

All measurements: CPU, N=10,000, K=7 (bench/callback_floor.py), jax 0.10.2.

| checkpoint | verbose(se=1) µs/step | manual-payload µs/step | machinery µs |
|------------|----------------------|----------------------|-------------|
| main 4d62fa4 (pre-alert) | 75.2 | 54.1 | 21.1 |
| main 307394d (post-alert) | 72.7 | 54.6 | 18.1 |
| perf/emission-machinery | 61.4 | 53.2 | 8.2 |
