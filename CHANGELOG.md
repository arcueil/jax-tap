# Changelog

## 0.1.0 (unreleased)

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
- **Demos** (`demo/`): nine runnable real-bug reproductions — the project's
  primary documentation for now.
- Adversarially reviewed: two 2-arm hostile reviews (B-core, A-shell) with
  remediation and hostile fix-reviews; the full attack/repro corpus is frozen
  under `proofs/`.
- GPU-validated (CUDA 13, RTX 5090): full test suite + demos.

### Known boundaries (documented)

- vmap×while_loop emits per-joint-iteration events including masked lanes
  (inherent to JAX's batched while lowering).
- Taps riding along `grad` observe the forward pass only; tap the
  differentiated function (`tap.verbose(jax.grad(f))`) to observe the
  backward pass.
- Trace-time tap configuration travels with the compiled artifact on
  jit-cache hits; host-side event routing is live.
- The jit re-wrap does not thread donation/shardings (correctness-neutral on
  CPU; multi-device is future work).
