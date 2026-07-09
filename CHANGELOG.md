# Changelog

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
