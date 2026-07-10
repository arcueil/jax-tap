# jax-tap (pronounced "just tap")

**Make print-debugging great again.**

Zero-code-change runtime telemetry for JAX control flow. Hold a lens over
unmodified JAX code, watch the values mutating *inside* `lax.scan` /
`lax.while_loop` — live, while it runs — then remove the lens and nothing was
ever there. The observed and production programs are **bitwise-identical**; the
lens is the only difference.

## The problem

Debugging JAX control flow is hard in a very specific way: your bug lives
**in the middle of a (nested) loop**, in a carry that mutates every step. You
can't `print` inside `jit`. Errors surface — if at all — long after the step
that caused them, and the loop may be four levels deep
(`scan → body → while → solver`). The classic workarounds are all bad:

- **Record into the carry, inspect post-hoc** — you must *edit the program*
  (add debug fields to the carry and thread them out through every nesting
  level), you pay O(steps × size) device memory stacking values you'll throw
  away, and you wait for the whole run to finish. Runs with these bugs often
  *don't* usefully finish: a NaN-frozen chain "completes" looking healthy.
- **`jax.debug.print` sprinkled in the body** — code changes you must remember
  to remove, no step/loop context, and nothing structured to analyse.

jax-tap's answer: **`jax.debug.callback` streaming as a first-class feature.**
Taps announce values *live, mid-loop* — the moment a Cholesky factor goes
non-finite you hear about it, at the exact step and loop address it happened,
while your original code contains **zero logging lines**.

## The promise

```python
import jaxtap as tap

with tap.record(select=lambda leaves: leaves[1].mean()) as rec:
    result = warmup.run(key, x0, num_steps)   # your code, UNMODIFIED
    # `leaves` = the loop carry's flat leaves; pick what ships to the host

rec.df()    # step-indexed telemetry → pandas
```

Done testing? Delete the `with` line. Nothing else changes — because nothing
else was ever touched.

The `with` form is shipped and ready to use. (The original wrapping API
`g, rec = tap.record(f)` is also available if you prefer.)

## Install

```bash
pip install jax-tap                 # imports as `jaxtap`
pip install "jax-tap[pandas]"       # optional: rec.df() → pandas
```

Requires `jax >= 0.10`. Use it as `import jaxtap as tap` (pronounced "just tap").

## What you can tap

| Tap class | What it observes | Status |
|-----------|------------------|--------|
| **Control-flow / carry taps** | the mutating carry at every `scan`/`while` step, at any nesting depth, with stable addresses (`scan[0]/while[1]`) | ✅ shipped |
| **Primitive taps** — *"just define L"* | outputs of named primitives (`tap.on("cholesky", ...)`): your body just writes `L = jnp.linalg.cholesky(M)`; the tap observes the actual `L` by primitive *kind* | ✅ shipped |
| **Output / y-taps** | per-step scan **outputs** (the `ys`), not just the carry — e.g. NUTS `treedepth` each step, for live saturation tripwires (`select_ys`/`on_ys`/`alert_ys`) | ✅ shipped (0.3.0) |
| Trace-time taps | shapes/dtypes/retrace events, at trace time, zero runtime cost | 🗺 roadmap |
| jit-event taps | trace-vs-execute timestamps ("why is this recompiling / where did 400 s go") | 🗺 roadmap |
| Backward-pass values | NaNs that exist only in the gradient pass | 🚫 documented boundary — grad-transform territory, out of scope by design |

## Design principles (the guarantees)

1. **Bitwise identity.** `tap.verbose(f)(*args)` returns exactly `f(*args)` —
   values *and* gradients, through `jit`, `vmap`, `grad`, `grad²`, `checkpoint`,
   `custom_jvp`/`custom_vjp`. This is the core CI gate, adversarially reviewed
   (see `proofs/`).
2. **Live streaming.** Telemetry crosses to the host as it happens, not after
   the loop ends. A hung or frozen loop still tells you where it froze.
3. **Reduce on device, ship scalars.** Your `select` runs inside the traced
   program; only its (small) output crosses the host boundary.
4. **Callback totality.** A raising/broken logging callback can never change or
   crash your program — telemetry failures warn once and step aside (holds even
   under `python -W error`).
5. **Emission only.** jax-tap ships data (an in-memory recorder → `.df()`,
   JSONL). No progress bars, no display frontends — consumers own read-out.

## Quick start

```python
import jax, jax.numpy as jnp
import jaxtap as tap

def f(x0, xs):
    def body(c, x):                    # your code — no logging lines
        return c * 1.01 + jnp.sin(x), c
    return jax.lax.scan(body, x0, xs)

x0, xs = jnp.float32(0.0), jnp.linspace(0.0, 1.0, 100)

# THE with-form (the promise): wrap the CALL, never the code
with tap.record(select=lambda leaves: {"c": leaves[0]}) as rec:
    f(x0, xs)                          # unmodified; bitwise-identical
rec.events[-1]      # TapEvent(path='scan[0]', step=99, total=100, value={'c': ...})
rec.df()            # pandas view (optional extra)

# toolkit one-liners compose the same way:
with tap.record(taps=[tap.watch_nan("cholesky", once=True)]):
    f(x0, xs)       # -> a live "[tap] FAIL ...: NaN/Inf" line, if it ever fires

# wrap-the-callable form (same power, when you have the function in hand):
g = tap.verbose(f, on_step=print, sample_every=10)
g(x0, xs)

# volume control on big loops:
#   sample_every=100 · where=lambda p: p == "scan[0]" · max_depth=1
```

## How `select` works

`select` is the device-side half of your print statement; `on_step` is the host-side half.

**Input**: the flat tuple of carry leaves returned by the loop body on each iteration.
**Output**: any pytree; its structure is preserved to the host.

The `select` function runs inside the traced program (on-device) to minimize what crosses the host boundary. Here is the intuition ladder with five one-line examples:

```python
# 1. select=None (default): full carry leaf tuple crosses host
g = tap.verbose(f, on_step=lambda e: print(e.value))
#   → TapEvent.value = (carry_leaf_0, carry_leaf_1, ...)

# 2. select one leaf: only e.g. the position vector
select=lambda leaves: leaves[0]
#   → TapEvent.value = array(...)

# 3. select with structure: build a dict on-device with computation
select=lambda leaves: {"pos": leaves[0], "sin_pos": jnp.sin(leaves[0])}
#   → TapEvent.value = {"pos": ..., "sin_pos": ...}

# 4. select with reduction: check finiteness before crossing
select=lambda leaves: jnp.isfinite(leaves[0]).all()
#   → TapEvent.value = True or False (one bool crosses host)

# 5. select with empty payload: progress idiom (step index only)
select=lambda _: ()
#   → TapEvent.value = ()  (nothing crosses host except step counter)
```

**Two traced consequences**:

1. **JAX-traceable only**: `select` must be traced-compatible (numpy operations only, no host-side Python logic).
2. **Gated by `sample_every`**: unsampled steps pay zero cost — the `select` function is never called on skipped steps.

Path-aware select (optional): if your `select` accepts a keyword argument named `path` or is a 2-parameter function, jaxtap passes the stable node address at call time (inspected once at trace time, zero per-step overhead):

```python
select=lambda leaves, *, path: {"node": path, "value": leaves[0]}
#   → TapEvent.value = {"node": "scan[0]", "value": ...}
```

## One emission primitive (how `record` and `verbose` relate)

Everything that ever leaves the device goes through a single primitive:
`jax.debug.callback`. The `with` form contains no emission code of its own —
it is lifecycle machinery (patch, registry, restore, routing) that applies
the same `verbose()` transform at the intercepted call site. The full
topography, for a progress bar:

```text
with tap.record(on_step=bar_update):     patches jax.lax.scan on __enter__
    run(...)                             your code, unmodified
      └─ your code calls jax.lax.scan(body, ...)
           └─ the PATCHED scan fires → interceptor (outermost call only)
                └─ builds g = λ init, xs: original_scan(body, init, xs)
                └─ calls verbose(g, on_step=<router>, select=..., sample_every=...)
                     └─ verbose builds the tap_cb closures — the ONLY place
                        jax.debug.callback exists in the codebase
                          └─ the walker rebuilds the scan; the rewrites inject
                             tap_cb(path, step, *carry) at the body's return
                               └─ DEVICE, per sampled step:
                                    lax.cond(step % se == 0,
                                       jax.debug.callback(host_fn, step, *selected))
                                     └─ HOST: router → the ACTIVE context
                                          → recorder.append(TapEvent)
                                          → your bar_update(TapEvent)
```

Two consequences of this shape: both forms are event-equivalent by
construction (same transform, applied at a different moment), and the
trace-time-config boundary applies to both identically (the callback and
`select` are baked by `verbose`'s trace either way) while the HOST routing
stays live (events always go to the currently-active context).

## The debugging toolkit

jax-tap ships four ergonomic helpers for the most common debugging patterns.

### Watch for NaNs: `tap.watch_nan()`

The most common trap: a primitive silently produces a non-finite output, and
your loop looks converged even though it has frozen or is computing garbage.
`tap.watch_nan("prim_name")` creates a tap that alerts **live** the moment a
NaN or Inf appears:

```python
with tap.record(taps=[tap.watch_nan("cholesky", once=True)]) as rec:
    result = sampler(x0, n_steps)   # your code, unmodified
```

Output (to stderr, live):
```
[tap] FAIL scan[0]/jit[0]/cholesky[0] 7/25: NaN/Inf
```

The `once=True` argument fires the alert only once per run — useful when a
single occurrence matters and you want to suppress the flood of repeated lines
for every subsequent step.

### Print values: `tap.print()`

For one-line diagnostic output of a primitive's values, `tap.print()` streams
to stderr with truncated array formatting:

```python
with tap.record(taps=[tap.print("mul")]) as rec:
    result = f(x0, xs)
```

Output (to stderr, one line per primitive firing):
```
[tap] scan[0]/mul[0] 0/5: array([0.], dtype=float32)
[tap] scan[0]/mul[0] 1/5: array([0.8499], dtype=float32)
[tap] scan[0]/mul[0] 2/5: array([1.7768], dtype=float32)
```

Arrays are printed with `numpy.printoptions(precision=4, threshold=8, edgeitems=2)`
so large arrays truncate cleanly without flooding your terminal.

### Custom alerts and selectors: `tap.on()`

For fine-grained control, `tap.on()` combines a device-side `select` reducer
(to minimize host-boundary traffic) with a host-side `alert` predicate:

```python
with tap.record(
    taps=[
        tap.on(
            "sin",
            select=lambda outs: outs[0],    # device-side: extract first output
            alert=lambda v: v > 0.8,        # host-side: when to alert
            label="sin exceeded 0.8"        # label shown in [tap] FAIL line
        )
    ]
) as rec:
    result = f(x0, xs)
```

Output (to stderr, only when the alert predicate is truthy):
```
[tap] FAIL scan[0]/sin[0] 1/5: sin exceeded 0.8
[tap] FAIL scan[0]/sin[0] 2/5: sin exceeded 0.8
```

### Discover primitive names: `tap.primitives()`

Unsure which primitive you want to tap? `tap.primitives(f, *args)` traces the
function once (with `jax.make_jaxpr`) and returns a dict of all primitives:

```python
prims = tap.primitives(f, x0, xs)
# {'scan': 1, 'mul': 1, 'sin': 1, 'convert_element_type': 1, 'add': 1}
```

Pass any string you see as the `prim_name` argument to `tap.on()`, `tap.print()`,
or `tap.watch_nan()`.

### A gotcha: `output=` index matches JAX primitive order, not Python API order

When tapping a primitive with multiple outputs, **indices refer to the JAX
primitive's output order, which can differ from the Python API's return order.**

For example, `jnp.linalg.eigh` returns `(eigenvalues, eigenvectors)` in Python,
but the underlying primitive emits `(eigenvectors, eigenvalues)` — so `output=0`
gives eigenvectors, not eigenvalues. Before relying on a specific `output=` index,
use `tap.print(prim_name)` (without an index) to inspect the actual layout.

### Emergency recovery: `tap.emergency_restore()`

If a session crashes inside a `tap.record()` block, the monkey-patched
`jax.lax.scan` and `jax.lax.while_loop` may not be restored. Call
`tap.emergency_restore()` to reset the library state to a clean slate.

## Progress recipes

jax-tap ships no display frontends — display is the consumer's responsibility.

### Simple progress bar (semi-production baseline)

The progress idiom (`select=lambda _: ()`) costs the least: only the step counter crosses the host boundary, zero bytes of carry data.

```python
from tqdm import tqdm
bar = tqdm(total=1000)
with tap.record(on_step=lambda e: e.path == "scan[0]" and bar.update(10),
                sample_every=10, select=lambda _: ()):
    result = run(...)
bar.close()
```

**Overhead**: at `sample_every=10` on a ~100 µs body, this costs ≈ **+6%**. At `sample_every=100`, overhead drops to **≈ +1%** (see the recommendation ladder in `bench/README.md` for other body sizes).

### Semantic progress (unbounded loop)

For an unbounded `while_loop`, the loop has no known total — but the *carry itself* can encode progress. Here, a tempering exponent drives the loop and doubles as a progress fraction:

```python
def bar(e):
    lam = float(e.value)  # the carry's tempering parameter, 0 to 1
    n = int(lam * 40)
    sys.stderr.write(f"\rtempering [{'#'*n}{'.'*(40-n)}] {lam*100:5.1f}%")

with tap.record(select=lambda leaves: leaves[0], on_step=bar):
    result = sampler(...)  # unbounded while_loop, unmodified
```

This pattern requires no total or manual step accounting — the tap streams the carry value that IS the progress metric. See `proofs/semantic-progress/semantic_progress.py` for a live example.

### Carry alerts

The `alert=` parameter on `tap.verbose` (and `tap.record`) turns a carry tap
into a live tripwire.  The callable runs host-side on every sampled
`TapEvent`; a truthy return emits one terse line to stderr with no other
action required.

```python
import jax
import jax.numpy as jnp
import jaxtap as tap

THRESHOLD = 5.0   # alert when carry exceeds this value

def accumulator(x0):
    """Carry increments each step; bug: no guard once it exceeds threshold."""
    def body(carry, step_frac):
        return carry + step_frac, carry
    xs = jnp.linspace(0.0, 2.0, 10, dtype=jnp.float32)
    return jax.lax.scan(body, x0, xs)

# One-line tripwire — delete the with-block when done debugging.
# alert receives the full TapEvent; return a str for a custom message
# or True for the default "alert" label.
with tap.record(
    select=lambda leaves: leaves[0],          # ship carry scalar to host
    alert=lambda e: (                         # host-side predicate
        f"carry={float(e.value):.2f} exceeded {THRESHOLD}"
        if float(e.value) > THRESHOLD else False
    ),
    alert_once=True,                          # silence after first hit
) as rec:
    accumulator(jnp.float32(0.0))  # unmodified

# Output to stderr (live, on first crossing):
# [tap] FAIL scan[0] 7/10: carry=6.22 exceeded 5.0
```

**Rules of thumb**

- `alert` receives the same `TapEvent` as `on_step`.  Return a `str` for a
  custom message, any other truthy value for the fixed label `"alert"`.
- `alert_once=True` fires at most once per path — useful to silence the flood
  from a stuck loop while still catching the first occurrence.
- alert runs before `on_step`; both run; a raising alert warns once and steps
  aside (never touches the computation).
- `sample_every` gates carry taps equally: `alert` only sees the events that
  cross the host boundary.
- Zero device-side cost: `alert` is purely host-side.  The compiled XLA
  artifact is identical whether or not `alert=` is set.

### Y-taps: tapping scan outputs

**New in 0.3.0**: `lax.scan` body functions return per-step output values (`ys`)
in addition to the carry. Y-taps observe these outputs live, using the same
`select` / `alert` machinery as carry taps, but with separate callbacks
(`on_ys` / `alert_ys`). A common use: live tripwires on adaptive sampler
info (e.g., NUTS tree depth saturation).

```python
import jax
import jax.lax as lax
import jax.numpy as jnp
import jaxtap as tap
from collections import namedtuple

# Example: NUTS sampler with live tree-depth monitoring
NUTSInfo = namedtuple("NUTSInfo", ["is_divergent", "treedepth"])

def sampler(init_x, n_steps):
    """Adaptive sampler returning per-step NUTSInfo as ys."""
    def body(chain_state, step_idx):
        # chain_state: the carry (updated each step)
        # ys: NUTSInfo (per-step sampler info)
        new_state = chain_state + 0.1
        is_div = step_idx > n_steps // 2  # divergence flag
        td = 2.0 ** (step_idx % 5)  # tree depth
        return new_state, NUTSInfo(is_divergent=is_div, treedepth=td)

    return lax.scan(body, init_x, jnp.arange(n_steps))

# Canonical recipe: live tripwire on saturation
# NUTSInfo leaves (alphabetical dict order / namedtuple field order):
# [0] = is_divergent, [1] = treedepth
with tap.record(
    select_ys=lambda ys_leaves: ys_leaves[1],  # extract treedepth
    alert_ys=lambda e: (
        f"output: treedepth saturated at step {e.step}"
        if float(e.value) >= 8.0 else False
    ),
    alert_ys_once=True,  # alert only on first occurrence
) as rec:
    result, ys_stream = sampler(jnp.float32(0.0), 100)

# Output to stderr (live, on first saturation):
# [tap] FAIL scan[0] 23/100: output: treedepth saturated at step 23

# Inspect captured output events
output_events = [e for e in rec.events if e.kind == "output"]
print(f"Recorded {len(output_events)} per-step outputs")
```

**Key points**

- **Separate from carry taps**: `on_ys` receives output events only;
  `on_step` receives carry events only. Both may be set; they fire independently.
- **Flat-leaves indexing**: `select_ys` receives a flat tuple of ys leaves
  (pytree structure is erased by tracing, same as `select` for carry).
  To index into `NUTSInfo`: leaves follow field order
  (namedtuple fields in declaration order; dict keys sorted alphabetically).
  Inspect with `jax.tree_util.tree_leaves(example_y)`.
- **Scan-only**: y-taps apply to `lax.scan` only (which has per-step outputs);
  `lax.while_loop` has no ys and produces no y-tap events.
- **Mixed streams**: `rec.events` holds both carry and output events in a single
  list. Filter with `e.kind == "output"` to separate them.
- **`df()` unchanged**: `.df()` continues to skip `kind` / `total` columns.
  To include them, build a DataFrame directly from `rec.events`.
- **Self-identifying messages**: carry and output alerts share the same stderr
  format `[tap] FAIL {path} {step}/{total}: {msg}`. Include `"output:"` or a
  field name in `alert_ys` messages to distinguish from carry alerts in log grep.

## What the first call tells you

The first call to a jitted function pays trace + compile + execute in one opaque wall-time block. Naive profiling attributes it all to compilation — but **the first tap event's arrival timestamp IS the compile/execute boundary**.

From a single first call, measure *true* compile cost (trace + compile) separately from execution cost, giving you a free steady-state runtime forecast before ever running the compiled program again. This is the "compile split" capability; see `demo/async_dispatch_compile_blowup.py` for a worked example (7-demo in the suggested reading order).

## demo/ — Learn by example

**demo/ is the primary documentation.** It holds ten runnable files demonstrating real bugs from our own history (silent float32 Cholesky NaNs, adaptation metrics that never moved, inner loops that quit early, ...) — each shows the silent symptom, then jax-tap localizing it. A suggested reading order and context are in `demo/README.md`.

The flagship: `demo/blackjax_warmup_telemetry.py` instruments a real BlackJAX warmup unmodified, streaming its step size and mass matrix as the algorithm adapts — no changes to BlackJAX, zero logging code in the warmup itself.

## Known boundaries

See `CHANGELOG.md` under "Known boundaries" for the complete documented list. In brief: `vmap×while_loop` includes masked lanes; taps riding `grad` observe the forward pass only (tap the differentiated function to observe backward); trace-time config travels with compiled artifacts on cache hits (host routing is live); the jit re-wrap does not thread donation/shardings.

### `vmap`×`while_loop`: which call patterns are transparent

A `while_loop` inside `jax.vmap` is walked **transparently** (its inner taps
fire) when its cond depends **only on a non-vmapped value** — e.g. a step
counter (`s[0] < N`). Under vmap the cond output then stays **scalar** (`ndim
0`), and jax-tap recurses into the loop body normally.

It is **opaque** (only the surrounding `scan`/carry events surface, no inner
`while` taps) when the cond depends on a **per-chain computed value** — e.g. a
convergence/divergence flag set inside the body (NUTS's `do_keep_integrating`,
built from `has_terminated` / `is_diverging`). JAX's vmap batching rule then
keeps the cond output **batched** (`ndim 1`), which trips jax-tap's
`_cond_is_batched` guard: the `while` is bound opaquely to preserve
bitwise-identical outputs (JAX's reduction + lane-masking happen at XLA
lowering, not in the jaxpr, so the walker can't safely reconstruct them). This
is the same mechanism as the vmap-batched-`while` note above — surfaced by the
cond's dtype/shape, not by any jit-cache state.

Practically: NUTS/dynamic-trajectory samplers under `vmap` fall in the opaque
case, but the pathology still reaches you — a NaN inside the trajectory loop
propagates through the carry into the outer `scan[0]` events, where a carry tap
(or `watch_nan`) catches it. Inner per-step trajectory telemetry under vmap is
future work.

### Cross-consumer select: compiled artifacts lock selection

When you compile an instrumented function under consumer A (via `tap.record(f)`,
`tap.verbose(f)`, or wrapped in `jax.jit`), A's `select` config is baked into
the compiled artifact. Later calls by consumer B with a different `select` will
NOT see B's selection—B's context receives **zero events** because A's baked
callback owns the artifact. This is the sharp edge: the `select` function runs
on-device (inside the traced program) and is frozen at compile time.

**Under the `with tap.record():` A-form**, each distinct `select` re-traces the
function, so you get your own selection; this is safe. The boundary applies
when reusing a **compiled instrumented artifact** (B-form function, or JIT
boundary) across consumers with different selects.

**Example** (sharp edge: sharing compiled B-form across consumers):
```python
import jax, jax.numpy as jnp
import jaxtap as tap

def scan_fn(x0, xs):
    def body(carry, x):
        return carry + x, carry * 2.0
    return jax.lax.scan(body, x0, xs)

jax.clear_caches()
x0, xs = jnp.float32(1.0), jnp.arange(3.0, dtype=jnp.float32)

# Consumer A: compile with empty select (B-form)
g_a, rec_a = tap.record(scan_fn, select=lambda _: ())
result_a = g_a(x0, xs)  # Compiled; A's select baked at trace time
jax.block_until_ready(result_a)
print(len(rec_a.events))  # 3

# Consumer B: call cached g_a in B's own context with different select
with tap.record(select=lambda c: c.mean()) as rec_b:
    result_b = g_a(x0, xs)  # Cache hit: A's baked callback, not B's
    jax.block_until_ready(result_b)

# A's instrumented callback was baked into the compiled g_a artifact.
# When B calls g_a, A's callback fires; B's context is invisible to the
# compiled code. B's select never runs; result: B receives no events.
print(len(rec_b.events))  # 0  (not 3, and not the mean values B requested)
```

Output:
```
3
0
```

**Recommendation:** Instrument each consumer's function under its own `select`
configuration. Avoid sharing compiled `tap.record(f)`, `tap.verbose(f)`, or
JIT-wrapped instrumented artifacts across consumers with different `select`
configs. If you need the same function under different selects, compile it
separately for each consumer (or re-wrap it under the A-form
`with tap.record(select=...):`).

## Naming

Distribution `jax-tap`, import `jaxtap`, documented alias `import jaxtap as tap`.
Lineage: JAX's deprecated `host_callback.id_tap` — the "tap" family is idiomatic
JAX history.
