# Changelog

## Unreleased — 0.3.0

### New: y-taps — taps on scan OUTPUTS (per-step `ys`) (GitHub #3)

Zero-code-change telemetry for the per-step `ys` returned by `lax.scan`.
Enables NUTS treedepth-saturation tripwires and other per-step diagnostics
without modifying the sampler.

**New parameters on `tap.verbose()` and `tap.record()`:**

- `select_ys` — on-device selector applied to ys flat leaves before the host
  boundary crossing.  Receives a **flat tuple of ys leaves** (same jaxpr-
  boundary contract as `select` for carry).
- `on_ys` — separate live host callback for output events (`kind="output"`),
  parallel to `on_step` for carry events.  Carry events go to `on_step`;
  output events go to `on_ys`.  Both may be set; both fire independently.
- `alert_ys` — host-side alert predicate on output events.  When truthy,
  emits `[tap] FAIL {path} {step}/{total}: {msg}` to stderr.  Messages
  should be self-identifying (include "output:" or a field name) to
  distinguish from carry-tap alerts in log grep.
- `alert_ys_once` — fire `alert_ys` at most once per path per call.

**`TapEvent.kind` field (new, backward-safe):**
- Default `"carry"` — all pre-0.3.0 code that constructs or inspects
  `TapEvent` without `kind=` continues to work unchanged.
- Y-tap events carry `kind="output"`.
- `df()` is unchanged (does not include `kind` or `total` columns).
- Both carry and output events land in the single `rec.events` stream;
  filter with `[e for e in rec.events if e.kind == "output"]`.

**Scope:** scan-only.  `while_loop` has no per-step ys; `rewrite_while` is
not touched.  Setting `select_ys` on a function with only while_loops
produces zero output events (silent no-op).  A `UserWarning` is emitted at
`verbose()`/`record()` call time when `select_ys` is set but `"scan"` is
not in `ops`.

**None-ys guard:** scans where the body returns `(carry, None)` (the
progress-bar idiom) produce zero output events — the `len(ys) > 0` guard in
`rewrite_scan.body_fn` is a Python trace-time check with zero device overhead.

**JSONL note:** `JSONLWriter` does not serialize `kind`; round-tripped events
default to `kind="carry"`.  Consumers who need `kind` in a JSONL stream
should add it to their serialization layer.

**Canonical treedepth tripwire:**

```python
with tap.record(
    select_ys=lambda ys_leaves: ys_leaves[treedepth_field_idx],
    on_ys=lambda e: print(f"step {e.step}: treedepth={float(e.value):.0f}"),
    alert_ys=lambda e: (
        f"output: treedepth saturated at step {e.step}"
        if float(e.value) >= 10.0 else False
    ),
    alert_ys_once=True,
) as rec:
    result = sampler.run(key, x0, n_steps)

output_events = [e for e in rec.events if e.kind == "output"]
```

## 0.2.1 (2026-07-10)

Consumer-driven fixes from the blackjax / tuningfork integration.

- **Fix: `max_depth` over-routed on cross-context cache hits, and is now
  correctly scoped to carry taps (GitHub #2).** When a function compiled under
  one `tap.record()` context was called inside a second context with a stricter
  `max_depth`, the dynamic router delivered all-depth carry events to the
  receiving recorder (measured ~50× host-callback waste on a 15-step NUTS run).
  The router now applies the receiving context's `max_depth` to foreign-trace
  cache-hit events. Scoped to **carry** taps only (matching the documented
  semantics): primitive taps (`tap.on`, `tap.watch_nan`) are **not** filtered
  by `max_depth`, so NaN tripwires survive under `max_depth=0`.
- **`jaxtap.original_scan` public export + cross-consumer docs (GitHub #4).**
  A stable public reference to the pristine `jax.lax.scan` (captured at import)
  for tools that assert scan restoration without importing privates. Plus a
  Known-Boundaries example documenting that a compiled instrumented artifact
  carries its trace-time `select`/callback — a second consumer reusing it
  receives no events (instrument under your own context instead).
- **Fix: B-form `verbose(vmap(f))` crash on vmapped while_loop (GitHub #5)**.
  `tap.verbose(jax.vmap(f))` (and equivalent A-form usage against a pre-vmapped
  function) crashed with `TypeError: select 'which' must be scalar` when `f`
  contains a `while_loop`.  Root cause: JAX's vmap batching rule stores the
  unreduced per-lane predicate (`bool[n]`) in `cond_jaxpr`; `rewrite_while`
  passed this to `lax.select` which requires a scalar `which`.

  Fix: detect vmap-batched while_loops in the walker by checking
  `cond_jaxpr.outvars[0].aval.ndim > 0` and bind them opaquely (re-emit through
  the original primitive).  The `ndim > 0` predicate catches all batching depths
  (`bool[n]` from single vmap, `bool[n,m]` from nested vmap, etc.).  A normal
  `while_loop` cond must return scalar bool, so `ndim > 0` can only arise from
  vmap batching.

  Delivered semantics:

  | context | while carry taps | notes |
  |---|---|---|
  | non-vmap while | yes — A1 ghost-drop intact | unchanged |
  | `jax.vmap(tap.verbose(f))` | yes — A1 per-lane ghost-drop | unchanged; A-form scalar intercept was never broken |
  | `tap.verbose(jax.vmap(f))` | **suppressed** (opaque bind) | crash fixed; bitwise-identical output |

  **Opaque-bind blast radius**: the opaque bind suppresses the entire B-form
  walker descent into the batched while's subtree — not just the while's own
  carry tap.  Prim-taps (`tap.on("add")` etc.) and scans nested inside the
  vmap-batched while body are also suppressed.  Parent taps (e.g. a scan that
  *wraps* the vmapped while) fire normally.  This is documented and tested.

  **A-form consumer path** (`with tap.record(): jax.vmap(f)(batch)`): the
  A-form monkeypatches `jax.lax.while_loop` before vmap's batching rule applies,
  so it intercepts at the scalar-in-vmap level — both the inner while and outer
  scan taps fire per-lane.  **Critical caveat**: if the vmapped function is
  compiled by `jax.jit` *outside* a `record()` context first (cache populated
  before record is active), the compiled artifact has no callbacks and subsequent
  calls inside `record()` emit 0 events.  Consumer guidance:

  - **B-form** (`tap.verbose(jax.vmap(f))`): recommended when the function may
    already be jit-compiled.  Scan taps fire (batched carry).  While taps
    suppressed (opaque bind).  No pre-jit hazard.
  - **A-form** (`with tap.record(): vmapped_f(batch)`): fires per-lane scan AND
    while taps.  Ensure the *first call* (jit compilation) happens inside the
    `record()` context.

  Outer **scan** carry taps around the vmapped while fire correctly with batched
  carry values in B-form — this is the high-value diagnostic for NUTS/HMC
  samplers (treedepth etc. live in the scan carry, not the inner while).
  Per-lane while telemetry under B-form vmap is a future arc.

## 0.2.0 (2026-07-09)

- **Carry-tap alerts**: `alert=` / `alert_once=` on `tap.verbose` and
  `tap.record` — host-side tripwires on carry events with the standard terse
  `[tap] FAIL {path} {step}/{total}: {msg}` line. Zero device-side cost
  (trace-identical with and without). A-form alerts resolve from the ACTIVE
  context on jit-cache hits (same live routing as `on_step`). `on_step` is now
  optional: a bare `tap.verbose(f)` traces no callback at all.
- **Emission machinery −55%** (18.1 → 8.2 µs/event on the reference CPU):
  `step_.item()` replaces `int(step_)` in the host path (JAX's
  `__int__`/`check_scalar_conversion` route costs 6× `.item()`), plus a
  single-context fast path in the dynamic router. The progress idiom at
  `sample_every=100` now costs ≈ +0.6 µs/step. Profiling evidence and the
  rejected-candidates table live in `bench/profiling/`.
- **vmap×while ghost events eliminated (carry taps)**: lanes that finish early
  no longer deliver fabricated carry values — per-lane event streams are exact
  (previously 30 events fired where 16 were real). Active-lane mask is
  sign-encoded into the step operand (~4–8 µs/iter; `bench/a1_decompose.py`).
  Ghost drop happens before `alert=`, so tripwires never fire on impossible
  values. Residual: primitive taps inside vmapped while bodies still fire on
  ghost iterations; `sample_every` is effectively ungated under vmap×while
  (JAX broadcasts the joint step, making the gate per-lane — a
  `vmap`+`lax.cond`+effects limitation).
- **NumPy-style API docs**: full Parameters/Returns/Examples (executed) for the
  public surface, including the flat-leaves `select` contract.
- **Nightly canary hardened**: runs the test suite against JAX nightly with
  warnings-as-errors AND a machinery-regression bench gate
  (`bench/nightly_gate.py`, machine-independent machinery/floor ratio ≤ 0.25).
- **GPU artifact validation**: `tools/gpu_pypi_validate.sh` installs the
  published wheel from PyPI on a CUDA box and runs suite + demos + bench.

## 0.1.0 (2026-07-09)

Initial release. Zero-code-change runtime telemetry for JAX control flow.

- **B-core**: `tap.verbose(f)` — jaxpr-walker transform; bitwise-identical
  outputs and gradients through `jit`, `vmap`, `grad`/`grad²`, `checkpoint`,
  `custom_jvp`/`custom_vjp`; stable boundary-visible addressing
  (`scan[0]/jit[0]/cholesky[0]`).
- **A-shell**: `with tap.record():` — monkeypatch context manager over the
  same transform; hardened lifecycle (re-enter guard, GC self-heal,
  `emergency_restore()`), dynamic host-side event routing across contexts.
- **Tap toolkit**: carry taps with device-side `select` (path-aware),
  primitive taps by kind (`tap.on`), `tap.watch_nan`, `tap.print`,
  `tap.primitives()` discovery, `once=`, `output=`, `sample_every` gating
  (carry AND primitive taps), `where`/`max_depth` emission filters.
- **Collectors**: in-memory `FlightRecorder` (`.df()` with optional pandas),
  JSONL writer/reader, `record()` helper.
- **Benchmarks** (`bench/`): callback floor ~33 µs/event on the reference
  CPU; the progress idiom at `sample_every=100` costs ≈ +1–15% on realistic
  bodies; honest recommendation ladders and payload decomposition in
  `bench/README.md`.
- **Demos** (`demo/`): ten runnable real-bug reproductions — the project's
  primary documentation for now.
- Adversarially reviewed: two 2-arm hostile reviews (B-core, A-shell) with
  remediation and hostile fix-reviews; the full attack/repro corpus is frozen
  under `proofs/`.
- GPU-validated (CUDA 13, RTX 5090): full test suite + demos.

### Known boundaries (documented)

- vmap×while_loop emits per-joint-iteration events including masked lanes
  (inherent to JAX's batched while lowering). [Mitigated for carry taps in
  0.2.0 — see above.]
- Taps riding along `grad` observe the forward pass only; tap the
  differentiated function (`tap.verbose(jax.grad(f))`) to observe the
  backward pass.
- Trace-time tap configuration travels with the compiled artifact on
  jit-cache hits; host-side event routing is live.
- The jit re-wrap does not thread donation/shardings (correctness-neutral on
  CPU; multi-device is future work).
