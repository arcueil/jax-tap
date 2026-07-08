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

*(The `with` form is landing in M1b — see Status. Today the same power is
available by wrapping the callable: `g, rec = tap.record(f)`.)*

## What you can tap

| Tap class | What it observes | Status |
|-----------|------------------|--------|
| **Control-flow / carry taps** | the mutating carry at every `scan`/`while` step, at any nesting depth, with stable addresses (`scan[0]/while[1]`) | ✅ shipped |
| **Primitive taps** — *"just define L"* | outputs of named primitives (`tap.on("cholesky", ...)`): your body just writes `L = jnp.linalg.cholesky(M)`; the tap observes the actual `L` by primitive *kind* | 🔨 in progress (M1a) |
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
| M1a | primitive taps (`tap.on("cholesky", ...)`) | 🔨 in progress |
| M1b | the `with tap.record():` context form | next (own adversarial review) |
| M3/M4/M5 | conformance suite · demos · docs | queued |

## Naming

Distribution `jax-tap`, import `jaxtap`, documented alias `import jaxtap as tap`.
Lineage: JAX's deprecated `host_callback.id_tap` — the "tap" family is idiomatic
JAX history.
