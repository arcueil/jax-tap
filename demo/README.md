# demo/ — jaxtap re-run against our own bugs

Each file reproduces the *essence* of a real bug we hit (in pure JAX, standalone
— no blackjax), shows the **silent symptom a user actually saw**, then shows
**jaxtap localizing it**. Every file self-reports `PASS`/`FAIL` and runs on the
base install:

```
uv run python demo/<name>.py
```

Files are named by topic. The sections below explain what each one shows and
which jaxtap capability it exercises.

## Honesty about tap classes

jaxtap v1 ships the **control-flow / carry tap** class (`tap.verbose` + `select`
at scan/while seams). The bug ledger spans more classes; each file states which
it uses, and where the ideal class is not yet built, says so plainly.

| Legend | meaning |
|--------|---------|
| ✅ | demonstrated with shipped v1 capability |
| ⚠️ | approximated with current capability; the *ideal* tap class is roadmap |
| 🚧 | documented boundary — jaxtap intentionally does NOT catch this |

## The files

### ✅ `cholesky_float32_trap.py` — silent NaN that *looks converged*
**Bug (GP regression, multi-day to find):** a float32 Cholesky in a GP
log-density silently produced a non-finite factor as warmup drove the kernel
ill-conditioned; dual-averaging shrank the step size to "dodge" the NaN, so the
chain froze while R-hat and divergence checks still said "converged."
**What it shows:** the sampler body contains ZERO logging code — it just
defines `L = jnp.linalg.cholesky(M)`. The unmodified call is wrapped in
`with tap.record(taps=[tap.on("cholesky", ...)], on_step=announce)`: the tap
observes the ACTUAL factor by primitive kind, and `announce` fires **live,
mid-loop** — the non-finite factor is loudly reported BEFORE the scan finishes,
at its true step and address (`scan[0]/jit[0]/cholesky[0]`). Delete the `with`
block and nothing was ever there. Also shows the trap is float32-specific
(float64 defers it far later).
**Tap class:** primitive tap (`tap.on("cholesky")`, reduce-on-device to one
bool) via the A-shell `with` form + live `on_step` streaming.

### ✅ `lowrank_metric_stuck.py` — the metric that never moved
**Bug (#949, parked 7 weeks):** an un-inverted score covariance cancelled the
position covariance (cov(x)·cov(s) ≈ Σ·Σ⁻¹ = I), leaving the learned metric
identity-shaped — adaptation ran, returned a metric, raised nothing; the
sampler just mixed badly.
**What it shows:** the warmup loop has zero logging code; a carry-tap `select`
derives the evolving metric's eigenvalue range ON-DEVICE from the running
accumulators (two scalars/window cross to host, `sample_every=500`). The
eig-ratio sits at ~1× from the FIRST window while the target spans 100× —
"adaptation LEARNED NOTHING" visible in one run vs seven weeks parked. The
fixed pipeline's ratio reaches 100×.
**Tap class:** carry tap on adaptation state (with-form + `select` +
`sample_every`).

### ✅ `lbfgs_maxiter_curvature.py` — the inner loop that quit early *(planned)*
**Bug (multi-day):** an L-BFGS inner solve hit `maxiter=30` without converging;
the returned curvature was 18–54× inflated, collapsing the step size ~670× — all
values finite, no warning, four control-flow levels deep.
**Will show:** a tap on the inner `while_loop` exit state (`iters == maxiter`,
grad-norm) flags the non-converged exit — showcasing inner-loop taps and the
boundary-visible addressing from the F1/F2 remediation.
**Tap class:** inner while-loop carry tap.

### ✅ `multinomial_da_bimodal.py` — acceptance secretly bimodal *(planned)*
**Bug:** a trajectory-weight change made dual-averaging never converge because
the acceptance signal was secretly bimodal ({≈0, ≈0.95}), which the scalar mean
hid.
**Will show:** a carry-tap on the DA acceptance per step; the collected values
reveal the bimodality on sight.
**Tap class:** carry tap on DA state.

### ✅ `treedepth_saturation.py` — the saturated-chain blind spot *(planned)*
**Bug (found months later):** reference chains saturated the max tree depth; the
certification gate was blind to it.
**Will show:** a per-draw carry-tap on tree depth acts as a live tripwire that
fires the moment depth hits its ceiling.
**Tap class:** per-draw event / carry tap.

### ⚠️ `mass_matrix_ndim_mismatch.py` — dense recipe ran diagonal *(planned)*
**Bug (5-PR arc, 3 days):** a dense mass-matrix recipe silently ran with a
1-D (diagonal) `inverse_mass_matrix` — a shape mismatch, not a value bug.
**Will show:** a runtime tap on `inverse_mass_matrix.ndim` catches the mismatch.
**Caveat:** the *ideal* here is a **trace-time shape tap** (static, zero runtime
cost); jaxtap v1 has no trace-time tap class yet, so this file approximates with
a runtime tap and marks the zero-cost version as roadmap.

### ⚠️ `async_dispatch_compile_blowup.py` — execution hidden as "tracing" *(planned)*
**Bug:** async dispatch made 427 s of execution appear inside a reported "19 s
tracing" — a measurement artifact that misdirected optimization.
**Will show:** the concept of jit trace-vs-execute event timestamps separating
real compile time from execution.
**Caveat:** jaxtap v1 has no **jit-event tap** class; this file demonstrates the
idea and marks the tap class as roadmap.

### 🚧 `backward_pass_vjp_nan.py` — the NaN jaxtap does NOT catch *(planned)*
**Bug (probdiffeq):** a `hypot(0, 0)` produced a NaN only in the **backward
(gradient) pass**; the forward pass was clean.
**What it shows — honestly:** a forward-pass tap does NOT and CANNOT see this;
backward-pass taps are grad-transform territory (oryx has undefined semantics
here too). The file proves the boundary rather than faking a catch — jaxtap
states its limits.
**Tap class:** documented boundary (out of scope by design).

## Why this directory exists

It is the empirical answer to "does the zero-code-change tap promise pay off on
real bugs?" — 5 clean wins, 2 honest roadmap markers, 1 honest boundary. The
files double as the acceptance corpus for each tap class as it ships.
