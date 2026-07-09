# demo/ — real debugging stories, re-run against the tap promise

Each file reproduces the *essence* of a real class of numerical-computing bug
(in pure JAX, standalone), shows the **silent symptom a user actually sees**,
then shows **jax-tap localizing it**. Every file self-reports `PASS`/`FAIL`
and runs on the base install:

```
uv run python demo/<name>.py
```

The bugs are distilled from real debugging episodes in probabilistic-programming
and scientific-JAX development. Where a public reference exists, it is linked;
the demos themselves are self-contained and assume no familiarity with any
particular library.

## Suggested reading order

For a newcomer, start with this sequence:

1. **`cholesky_float32_trap.py`** — a primitive tap on a single operation
2. **`lowrank_metric_stuck.py`** — a carry tap with on-device `select` aggregation
3. **`lbfgs_maxiter_curvature.py`** — a carry tap with address-specific `where` targeting
4. **`multinomial_da_bimodal.py`** — streaming a carry's evolution to see its distribution
5. **`treedepth_saturation.py`** — live tripwire + post-hoc statistics from one tapped stream
6. **`mass_matrix_ndim_mismatch.py`** — reading trace-time fingerprints with `tap.primitives()`
7. **`async_dispatch_compile_blowup.py`** — using event timestamps to locate a boundary
8. **`backward_pass_vjp_nan.py`** — tapping the differentiated function to see the backward pass
9. **`vmap_chains_progress.py`** — the vmap duality: one bar for 8 chains, or per-chain telemetry

Each builds on the previous one's concepts, and together they show every major tap class and the patterns they address.

## Honesty about tap classes

Each file states which tap class it uses, and where the ideal class is not yet
built, says so plainly.

| Legend | meaning |
|--------|---------|
| ✅ | demonstrated with shipped capability |
| ⚠️ | approximated with current capability; the *ideal* tap class is roadmap |
| 🚧 | documented boundary — jax-tap intentionally does NOT catch this |

## The files

### ✅ `cholesky_float32_trap.py` — silent NaN that *looks converged*
**Bug pattern:** a float32 Cholesky of an ill-conditioned matrix silently
produces a non-finite factor deep inside a sampler's warmup loop. The step-size
adaptation "dodges" the NaN by shrinking toward zero, so the chain freezes in
place — while convergence diagnostics still look fine. This class of bug can
absorb days: the run *completes* and nothing raises.
**What it shows:** the loop body contains zero logging code — it just defines
`L = jnp.linalg.cholesky(M)`. Wrapping the unmodified call in
`with tap.record(taps=[tap.watch_nan("cholesky", once=True)])` produces one
live line the moment the factor goes bad — before the loop finishes:
`[tap] FAIL scan[0]/jit[0]/cholesky[0] 7/25: NaN/Inf`. Also shows the trap is
float32-specific (float64 genuinely fixes it).
**Tap class:** primitive tap (`tap.watch_nan`) via the `with` form.

### ✅ `lowrank_metric_stuck.py` — the metric that never moved
**Bug pattern:** a mass-matrix adaptation consumes the score covariance
*without inverting it*. For a Gaussian target the un-inverted factor exactly
cancels the position covariance (cov(x)·cov(scores) ≈ Σ·Σ⁻¹ = I), so the
learned metric collapses to identity: adaptation runs, returns a metric,
raises nothing — the sampler just mixes badly. Bugs of this shape can sit
undiagnosed for weeks because every run "works."
Inspired by a real fix in BlackJAX:
[blackjax-devs/blackjax#949](https://github.com/blackjax-devs/blackjax/pull/949).
**What it shows:** a carry-tap `select` derives the evolving metric's
eigenvalue range ON-DEVICE from the running accumulators (two scalars per
window cross to host, `sample_every=500`). The eigenvalue ratio sits at ~1×
from the FIRST window while the target spans 100× — "adaptation learned
nothing" is visible in one run. The fixed pipeline reaches 100×.
**Tap class:** carry tap on adaptation state (`with` form + `select` +
`sample_every`).

### ✅ `lbfgs_maxiter_curvature.py` — the inner loop that quit early
**Bug pattern:** an inner optimizer (e.g. L-BFGS inside a Laplace
approximation inside a sampler step) hits its iteration cap without
converging. The curvature at the non-converged exit is silently inflated,
collapsing the outer step size — all values finite, no warning, and the
failing loop is several control-flow levels deep.
**What it shows:** the inner solver's own carry already holds
(iterations, gradient) — a carry tap streams its per-iteration heartbeat;
splitting the stream at step-counter resets yields the exit state per outer
step. Two transient cap-hits (30 iters, |grad| ~3e3) stand out instantly at
their address `scan[0]/while[0]`, explaining a silent 58× step-size collapse.
Also demonstrates two real seams: a solve needing 0 iterations emits no
heartbeat, and one `select` serves every tapped node (branch on carry arity).
**Tap class:** inner while-loop carry tap (heartbeat + exit state).

### ✅ `multinomial_da_bimodal.py` — acceptance secretly bimodal
**Bug pattern:** a step-size controller tunes toward a target MEAN acceptance,
but the per-step values it consumes are secretly bimodal ({~0.02, ~0.95},
nothing between). The mean sits exactly at target while describing NO actual
step — tuning hunts between the modes forever, and nothing errors.
**What it shows:** the controller's own carry holds the last acceptance; a
carry-tap streams it and a five-bucket histogram shows the split on sight
(mean 0.80 'on target' vs 325/0/0/0/1675 buckets; ε swings ~2× forever).
The lesson: the tap surfaces the DISTRIBUTION the code only ever averaged.
**Tap class:** carry tap on controller state.

### ✅ `treedepth_saturation.py` — the saturated-chain blind spot
**Bug pattern:** a NUTS-style sampler silently saturates its max tree depth in
stiff regions — every draw still returns; means and summary diagnostics wash
the per-draw signature out, so it can go unnoticed for months.
**What it shows:** the carry keeps the last tree depth (as real NUTS info
does); one tapped stream gives BOTH a live tripwire (`[tap] FAIL scan[0]
203/2000: treedepth==10`) AND the post-hoc picture: mean depth 8.4 looks
innocuous while 7% of draws saturated, including a 20-draw consecutive
excursion the sampler couldn't explore at the depth it needed.
**Tap class:** per-draw event / carry tap (live tripwire + recorder from the
same stream).

### ⚠️ `mass_matrix_ndim_mismatch.py` — dense config ran diagonal
**Bug pattern:** a sampler kernel dispatches on the mass matrix's `ndim`; a
plumbing bug hands it a 1-D matrix although the user configured dense — the
wrong algorithm runs silently (nothing errors; mixing quietly degrades on
correlated targets).
**What it shows:** the executed PRIMITIVES are the fingerprint — the dense
path contains `dot_general`, the diagonal path only `mul`. `tap.primitives()`
reads it at trace time (zero runtime cost): buggy `dot_general=0` vs fixed
`=1`. A `tap.print("dot_general", once=True)` runtime check makes ABSENCE the
live symptom (buggy run: silence; fixed run: one line). Also a practical tip:
tap the DISTINCTIVE primitive — generic ones like `mul` match PRNG internals.
**Caveat:** the *ideal* is a dedicated **trace-time shape tap** (roadmap);
`tap.primitives()` is the shipped approximation.

### ⚠️ `async_dispatch_compile_blowup.py` — execution hidden as "compilation"
**Bug pattern:** the first call of a jitted function pays trace + compile +
execute in one opaque wall-time block; naive profiling attributes it all to
compilation (and on async backends the conflation smears execution into
whatever phase blocks — see JAX's
[async dispatch documentation](https://docs.jax.dev/en/latest/async_dispatch.html)).
Real episodes hid minutes of execution inside a reported "tracing" number.
**What it shows:** tap events are emitted by the RUNNING program, so the
FIRST event's arrival timestamp IS the compile/execute boundary: the demo
splits an opaque first call into trace+compile vs execution (75–80% of the
"compilation" number was actually execution), cross-checked against the same
program's steady-state run for consistency.
**Caveat:** the *ideal* is a **jit-event tap** class (trace/compile/execute
timestamps — roadmap); event-arrival timing is the shipped approximation.

### 🚧 `backward_pass_vjp_nan.py` — the NaN born in the backward pass
**Bug pattern:** a hand-rolled `sqrt(c**2 + x**2)` is finite forward, but its
derivative `c/sqrt(...)` is 0/0 = NaN at the origin — the NaN exists only in
the backward pass. (Modern `jnp.hypot` ships a guarded VJP for exactly this
reason — the demo reproduces the pre-fix form.)
**What it shows — honestly, in two acts:** (1) **the boundary**: forward taps
on `loss()` stay SILENT — taps riding along a grad transform observe the
forward pass only, by documented contract; jax-tap cannot see this bug from
the forward side. (2) **the escape hatch**: `jax.grad(loss)` is itself just a
function whose jaxpr contains the backward pass as ordinary primitives — tap
THE DIFFERENTIATED FUNCTION and `watch_nan("div")` fires at the 0/0's birth
site with an address: `[tap] FAIL scan[0]/div[0] 0/3: NaN/Inf`.
**Tap class:** documented boundary + the tap-grad(f)-itself workaround.

### ✅ `vmap_chains_progress.py` — eight chains, one progress bar
**What it shows:** under `jax.vmap`, whether a tap fires once or per-lane
depends on whether the shipped value is batched. `select=lambda _: ()` ships
only the (unbatched) step counter → ONE progress bar for 8 chains (10
events); `select` on the carry ships batched values → per-chain telemetry
(80 = 8×10). Same unmodified sampler; `select` picks the face.
**Tap class:** carry tap under `vmap` (the documented per-lane duality, used
as a feature).

## Why this directory exists

It is the empirical answer to "does the zero-code-change tap promise pay off
on real bugs?" — clean wins where the tap classes ship today, honest roadmap
markers where they don't, and one honest boundary. The files double as the
acceptance corpus for each tap class as it ships.
