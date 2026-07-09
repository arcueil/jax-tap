# jax-tap

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

with tap.record(select=lambda carry: carry.logdensity.mean()) as rec:
    result = warmup.run(key, x0, num_steps)   # your code, UNMODIFIED

rec.df()    # step-indexed telemetry → pandas
```

Done testing? Delete the `with` line. Nothing else changes — because nothing
else was ever touched.

The `with` form is shipped and ready to use. (The original wrapping API
`g, rec = tap.record(f)` is also available if you prefer.)

## What you can tap

| Tap class | What it observes | Status |
|-----------|------------------|--------|
| **Control-flow / carry taps** | the mutating carry at every `scan`/`while` step, at any nesting depth, with stable addresses (`scan[0]/while[1]`) | ✅ shipped |
| **Primitive taps** — *"just define L"* | outputs of named primitives (`tap.on("cholesky", ...)`): your body just writes `L = jnp.linalg.cholesky(M)`; the tap observes the actual `L` by primitive *kind* | ✅ shipped |
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

## Quick start (today's API)

```python
import jax, jax.numpy as jnp
import jaxtap as tap

def f(x0, xs):
    def body(c, x):                       # your code — no logging lines
        return c * 1.01 + jnp.sin(x), c
    return jax.lax.scan(body, x0, xs)

# live stream:
g = tap.verbose(f, on_step=print)                    # TapEvent(path='scan[0]', step=3, value=...)
g(x0, xs)                                            # bitwise-identical result

# or record + analyse:
g, rec = tap.record(f, select=lambda leaves: {"c": leaves[0]})
g(x0, xs)
rec.df()                                             # path | step | c

# volume control on big loops:
tap.verbose(f, on_step=cb, sample_every=100, max_depth=1, where=lambda p: "while" in p)
```

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

## Progress bar in four lines

jax-tap ships no display frontends — display is the consumer's responsibility.
To add a progress bar, wire a simple `on_step` callback that updates your bar
when the enclosing loop fires:

```python
from tqdm import tqdm
bar = tqdm(total=1000)
with tap.record(on_step=lambda e: e.path == "scan[0]" and bar.update(1)):
    run(...)
```

The `sample_every=` parameter is your rate-limiting knob: pass it to `tap.record()`
to emit events only every *k* steps, reducing host-boundary traffic if your loop is
very tight.

## demo/ — real bugs, re-run against the tap promise

`demo/` holds one runnable file per real bug from our own history (silent
float32 Cholesky NaNs, adaptation metrics that never moved, inner loops that
quit early, ...) — each shows the silent symptom, then jax-tap localizing it.
Start with `demo/cholesky_float32_trap.py`.

## Status

Pre-release; not yet on PyPI. The foundation (jaxpr-walker core, tap-spec
layer, collectors) is built and has passed a 2-arm adversarial review
(numerics: zero corruption found; structure: both findings fixed and
re-reviewed — evidence frozen under `proofs/`).

| Milestone | Scope | State |
|-----------|-------|-------|
| M0 | jaxpr-walker core (`tap.verbose`) | ✅ merged |
| M2 | tap specs (`sample_every`/`where`/`max_depth`) + collectors (`record`, `.df()`, JSONL) | ✅ merged |
| B-core review | 2-arm adversarial review + remediation | ✅ merged |
| M1a | primitive taps (`tap.on("cholesky", ...)`) | ✅ merged |
| M1b | the `with tap.record():` context form | ✅ merged |
| A-shell review | adversarial review of A-form context-manager semantics | ✅ merged |
| M1c | alert sugar (`watch_nan`, `tap.print`, `output=`, `once=`) | ✅ merged |
| M3/M4/M5 | conformance suite · demos · docs | queued |

## Naming

Distribution `jax-tap`, import `jaxtap`, documented alias `import jaxtap as tap`.
Lineage: JAX's deprecated `host_callback.id_tap` — the "tap" family is idiomatic
JAX history.
