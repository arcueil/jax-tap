# y-taps design proposal: taps on scan outputs (ys / per-step info)

**jax-tap issue:** #3
**Branch:** `spike/y-taps-design`
**Status:** RATIFIED 2026-07-10 — deferred to 0.3.0 (build after #2/#5 ship in 0.2.1)
**Author:** SWE agent (spike #3, AYS-1 revisions, JP ratification)
**Date:** 2026-07-10

---

## Ratified design — build brief for the 0.3.0 implementer

This section is the turnkey spec. The rest of the document is the spike/feasibility
record. Do not implement until 0.2.1 (#2/#5) has shipped.

### Ratified API signature

```python
tap.verbose(
    f,
    # --- carry tap (existing, unchanged) ---
    on_step: Callable[[TapEvent], None] | None = None,
    select: Callable | None = None,
    alert: Callable[[TapEvent], Any] | None = None,
    alert_once: bool = False,
    # --- y-tap (new) ---
    on_ys: Callable[[TapEvent], None] | None = None,   # ← live callback for output events
    select_ys: Callable | None = None,                 # ← on-device selector for ys leaves
    alert_ys: Callable[[TapEvent], Any] | None = None, # ← host-side alert predicate
    alert_ys_once: bool = False,                       # ← fire alert at most once per path
    # --- shared (unchanged) ---
    ops: tuple[str, ...] = ("scan", "while_loop"),
    sample_every: int = 1,
    where: Callable[[str], bool] | None = None,
    max_depth: int | None = None,
    taps: Sequence[PrimitiveTap] = (),
)
```

**Key design decisions (JP ratified):**

- `on_ys` is a **separate** live callback for output events, parallel to `on_step`
  for carry events. Carry events go to `on_step`; output events go to `on_ys`.
  They are independent: both, one, or neither may be set.
- `rec.events` (FlightRecorder) still collects **both** carry and output events in
  a single stream. Output events have `TapEvent.kind == "output"`; carry events
  have `kind == "carry"` (the default). Consumers filter with
  `[e for e in rec.events if e.kind == "output"]`.
- `df()` is unchanged — it does not include `kind` or `total` columns.

### Canonical tuningfork treedepth tripwire

```python
with tap.record(
    select_ys=lambda ys_leaves: ys_leaves[treedepth_field_idx],
    on_ys=lambda e: announce(f"step {e.step}: treedepth={float(e.value):.0f}"),
    alert_ys=lambda e: (
        f"treedepth saturated at step {e.step}"
        if float(e.value) >= 10.0 else False
    ),
    alert_ys_once=True,
) as rec:
    result = sampler.run(key, x0, n_steps)

output_events = [e for e in rec.events if e.kind == "output"]
```

### TapEvent change (backward-safe, confirmed by execution)

```python
@dataclasses.dataclass(frozen=True)
class TapEvent:
    path: str
    step: int
    value: Any
    total: "int | None" = None
    kind: str = "carry"   # "carry" | "output"  — new; default preserves compat
```

### Files and ~LOC

| File | Change | ~LOC |
|------|--------|------|
| `src/jaxtap/_rewrites.py` | Add `y_tap_cb` param to `rewrite_scan`; `len(ys)>0` guard; no change to `rewrite_while` | ~25 |
| `src/jaxtap/__init__.py` | Add `on_ys`, `select_ys`, `alert_ys`, `alert_ys_once` to `verbose()` and `record()`; build `y_tap_cb` closure; add `kind` to `TapEvent` | ~90 |
| `src/jaxtap/_walker.py` | Thread `y_tap_cb` through `interpret()` / `_interp()` to `rewrite_scan` | ~15 |
| `src/jaxtap/_ashell.py` | Forward new kwargs in `_RecordContext` | ~20 |
| `tests/test_ytaps.py` | New: basic, jit, vmap, nested, on_ys, select_ys, alert_ys, ys-only, while no-op, None-ys no-op, sample_every | ~200 |

**Total production LOC: ~150. Total with tests: ~350.**

### Critical implementation notes

1. **`rewrite_scan.body_fn` change** — insert after existing carry tap:
   ```python
   # ys is the flat list outs[ncar:]; len==0 when body returns (carry, None).
   # The len(ys)>0 guard is Python-time (trace-time) — zero device overhead.
   if y_tap_cb is not None and len(ys) > 0:
       y_tap_cb(here, step, *ys, total=total)
   ```

2. **`y_tap_cb` closure in `verbose()`** — mirrors `tap_cb` build exactly:
   apply `select_ys` on-device, capture `sel_ys_tree` at trace time, reconstruct
   pytree on host, fire `_fire_carry_alert(alert_ys, event, alert_ys_once, ...)`
   then `_guard(on_ys, event)` with `kind="output"`. Apply `sample_every` wrapper
   identically to `tap_cb`.

3. **`on_ys` is NOT delivered through `on_step`** — the FlightRecorder's
   `__call__` receives output events as a side-effect of `_guard(recorder, event)`
   in the y_tap_cb closure (same pattern as carry events). Both carry and output
   events land in `rec.events`.

4. **`select_ys` receives flat leaves** — same jaxpr-boundary contract as `select`
   for carry. Leaf order: namedtuple fields in declaration order; dict keys sorted
   alphabetically. Document with an example in the docstring.

5. **y-taps are scan-only** — `rewrite_while` is not touched; no y_tap_cb plumbing
   into while rewrites.

6. **alert_ys messages should be self-identifying** — carry and output alerts share
   the same `[tap] FAIL {path} {step}/{total}: {msg}` format; no format change.
   Docstring note: "Include 'output:' or a field name to distinguish from carry-tap
   alerts in log grep."

7. **JSONLWriter does not serialize `kind`** — round-tripped events default to
   `kind="carry"`. Acceptable for 0.3.0; note in release notes if y-tap events
   are likely to be JSONL-streamed.

---

## Summary (spike record)

This document answers the five feasibility questions posed by the TL, provides
executed PoC evidence for all mechanical claims, finalises the API surface, and
gives an implementation sketch so JP could size and ratify the work.

**Verdict: feasible with minimal surgery.** The ys from a scan body's return
are already available at the exact point where the carry tap fires in
`rewrite_scan`. A y-tap callback can be inserted there with no structural
change to the rewrite or the walker.

**AYS-1 revisions (closed before ratification):**
- `TapEvent.kind` compat confirmed with executed evidence; no positional-unpack risk
- `None`-ys scan footgun identified and fixed: guard y-tap call with `len(ys) > 0`
- `df()` does NOT gain `kind` column for free (draft doc was wrong); corrected
- Alert format: no change; self-identifying messages are the convention

---

## 1. Feasibility

### Where the carry tap fires today

In `src/jaxtap/_rewrites.py`, `rewrite_scan`, the body wrapper is:

```python
def body_fn(carry_step, x):
    carry, step = carry_step
    outs = interp_fn(body.jaxpr, ...)
    new_carry = outs[:ncar]          # line 126
    ys = outs[ncar:]                 # line 127 — ys already available here
    if emit_carry:
        tap_cb(here, step, *new_carry, total=total)  # line 134 — carry tap fires here
    return (new_carry, step + 1), ys
```

`ys` is a flat list of arrays computed on line 127. The carry tap fires on
line 134. A y-tap call can be inserted immediately after — no restructuring
needed. The ys value is in scope, the step counter is in scope, and the total
is captured from the enclosing `rewrite_scan` frame.

### PoC evidence (executed)

Script: `scratch/poc_ytaps.py`. Output captured below.

**Q1a — basic carry+y tap (non-jitted):**
```
carry_out: 10.0
carry events (5): [(0, 0.0), (1, 1.0), (2, 3.0), (3, 6.0), (4, 10.0)]
y events (5): [(0, {'is_div': False, 'treedepth': 0.0}), ...]
PASS: both carry and y taps fire; correct step indices and kinds
```

**Q1b — jitted scan:**
```
carry events: 5, y events: 5
PASS: jitted scan fires both carry and y taps
```

**Q1c — vmapped scan (2 chains):**
```
carry events: 10, y events: 10   (2 lanes × 5 steps)
PASS: vmapped scan fires carry and y taps for all lanes
```

**Q1d — nested scan (outer steps=2, inner steps=3):**
```
outer y events: 2
inner y events: 6   (2 outer × 3 inner)
PASS: nested scan fires y taps at each level correctly
```

All assertions passed. The `jax.debug.callback` mechanism for ys is identical
to the carry mechanism: flat-leaf tuple shipped through the host boundary, step
scalar included, pytree structure reconstructed on the host from a captured
`tree_structure` snapshot.

### Does it compose with the carry tap in the same scan?

Yes. Both callbacks fire within the same `body_fn` wrapper: the carry tap fires
first, then the y tap. They share the same `step` value and `total`. The
existing `emit_carry` guard suppresses the carry callback independently; a
parallel `emit_ys` guard (default True) would gate the y tap.

### Does it compose with the A1 sign-encode step machinery?

The A1 mitigation (ghost-lane suppression for `vmap × while_loop`) encodes an
active mask into the step sign bit in `rewrite_while`. **y-taps are
scan-only** (see Section 5). `rewrite_scan` does not use sign-encode — scan
has no ghost-lane problem because JAX's scan always runs exactly N steps per
lane. No interaction.

---

## 2. Semantics

### Pytree ys and the flat-leaves boundary

`lax.scan`'s ys are a pytree — any pytree the body returns as its second
element. At the jaxpr boundary, JAX flattens this pytree into a list of arrays.
Inside `body_fn` in the rewrite, `ys = outs[ncar:]` is already a flat list.

The y-tap callback receives **flat leaves** exactly as the carry tap does. The
pytree structure is NOT recoverable from the jaxpr without user input. This is
the same documented boundary as for `select=` on carry.

`select_ys` receives a flat tuple of y leaves. To index into NUTSInfo
specifically, the user must know the leaf order. The leaf order follows JAX's
default pytree traversal: for a dict, keys are sorted alphabetically; for a
namedtuple, fields appear in declaration order. Users can inspect with
`jax.tree_util.tree_leaves(example_y)`.

For the tuningfork use case, `NUTSInfo` is a namedtuple. Leaf order follows
namedtuple field order; users should consult tuningfork's type definition. An
alternative is to use a `select_ys` that applies field extraction before the
host boundary:

```python
select_ys=lambda ys_leaves: ys_leaves[treedepth_index]
```

### `TapEvent` for a y-tap (AYS-1 compat verdict)

Proposed: add a `kind` field to `TapEvent` with default `"carry"` for backward
compatibility. Y-tap events carry `kind="output"`.

```python
@dataclasses.dataclass(frozen=True)
class TapEvent:
    path: str
    step: int
    value: Any
    total: "int | None" = None
    kind: str = "carry"   # "carry" | "output"  — new field; default preserves compat
```

**Executed backward-compat audit (scratch/poc_ays1.py):**

(a) `TapEvent` is a `dataclasses.dataclass(frozen=True)` — NOT a NamedTuple.
    Confirmed: `dataclasses.is_dataclass(TapEvent) == True`, no `._fields`.

(b) Positional unpack is impossible. Verified by execution:
    ```
    path, step, value, total = event  →  TypeError: cannot unpack non-iterable TapEvent object
    ```
    Grep of `src/`, `tests/`, `demo/` found zero positional unpack sites.
    All access is attribute-based (`e.path`, `e.step`, `e.value`, `e.total`).

(c) `FlightRecorder.df()` does NOT gain a `kind` column for free.
    The draft doc was wrong on this point. `df()` (`collectors.py:161-164`)
    builds rows explicitly as `{"path": event.path, "step": int(event.step),
    **_value_to_columns(event.value)}` — it reads only `path`, `step`, and
    `value`; `total` and `kind` are not included. Adding `kind` to `TapEvent`
    does not change `df()` output. Existing test
    `test_collectors.py:305` (`assert list(df.columns) == ["path", "step", "value"]`)
    continues to pass unchanged.

    Decision: do NOT add `kind` to `df()` in this arc. Consumers who need it
    can build their own frame:
    ```python
    import pandas as pd
    df = pd.DataFrame([{"kind": e.kind, "path": e.path, "step": e.step,
                        "value": e.value} for e in rec.events])
    ```
    Adding `kind` to `df()` is a separate forward arc (requires updating the
    existing column-assertion test and documenting the new column).

(d) All construction sites in `src/jaxtap/` use keyword arguments only:
    `TapEvent(path=path, step=raw, value=value, total=total)`.
    The one site without `total=` (`collectors.py:346`, in `read_jsonl`) also
    uses keywords; it continues to work unchanged (total defaults to None,
    kind defaults to "carry").

**Verdict: `kind` field is backward-safe. No positional-unpack risk, no test
breakage, no df() column-order change.**

**TapEvent shape for a y-tap is identical to a carry tap:**
- `path`: same path string as the enclosing scan (`"scan[0]"`, etc.)
- `step`: the loop step index (0-based)
- `value`: the selected y value (after `select_ys`, if given)
- `total`: the scan length (same int as carry events for the same scan)
- `kind`: `"output"`

### Single stream vs. separate streams

**Decision: single `rec.events` stream with the `kind` discriminator.**

Rationale:
- A single stream preserves step-ordering across carry and output events; a
  dual-stream design hides the interleaving.
- Consumers that only want carry events already had a flat list; they add a
  trivial `.kind == "carry"` filter. The empty-filter path (no y-taps configured)
  produces zero output events — zero overhead, no API change.
- Separate `rec.carry_events` / `rec.output_events` streams add surface area
  and make it harder to answer "what happened at step 42?" in sequence.
- `rec.events` filtering by `kind` is O(n) at read time; no overhead at event
  delivery time. `df()` does not change (see AYS-1 compat note in section above).

**Ordering — NO cross-callback guarantee.** `new_carry` is emitted before `ys`
in `body_fn`, but both cross the host boundary via separate
`jax.debug.callback(ordered=False)` calls, which are NOT serialized against each
other. So the carry and output events for the *same step* may arrive in either
order — CPU happens to preserve emission order (carry then output), GPU may not.
(Corrected 2026-07-10: an earlier draft claimed carry-fires-first as a guarantee;
the published-wheel GPU validation disproved it. Do not rely on carry/output
relative order within a step, nor on cross-step order under `ordered=False`.)

---

## 3. API surface

### Ratified parameter names (JP decision)

```python
tap.verbose(
    f,
    # --- carry tap (existing, unchanged) ---
    on_step: Callable[[TapEvent], None] | None = None,
    select: Callable | None = None,
    alert: Callable[[TapEvent], Any] | None = None,
    alert_once: bool = False,
    # --- y-tap (new, ratified) ---
    on_ys: Callable[[TapEvent], None] | None = None,    # ← separate live output callback
    select_ys: Callable | None = None,                  # ← on-device ys selector
    alert_ys: Callable[[TapEvent], Any] | None = None,  # ← host-side alert predicate
    alert_ys_once: bool = False,                        # ← once-per-path alert gate
    # --- shared (unchanged) ---
    ops: tuple[str, ...] = ("scan", "while_loop"),
    sample_every: int = 1,
    where: Callable[[str], bool] | None = None,
    max_depth: int | None = None,
    taps: Sequence[PrimitiveTap] = (),
)
```

**Name rationale**

- `select_ys` / `alert_ys` / `alert_ys_once` mirror `select` / `alert` /
  `alert_once` for carry — consistent, learnable pattern; `_ys` suffix matches
  `jax.lax.scan`'s own `ys` naming.
- `on_ys` parallels `on_step` — carries the same semantics (live host callback,
  _guard-wrapped, never-raise) but fires on output events only. This is a JP
  decision: carry events go to `on_step`, output events go to `on_ys`, keeping
  the two streams cleanly separated at the live-callback level.
- `select_output` / `on_output` were rejected: "output" is ambiguous (scan output
  includes both carry and ys); `_ys` is unambiguous.

The `record()` context manager and B-form accept the same new kwargs, forwarded
to `verbose()`.

### Callback routing: `on_step` vs `on_ys` vs `rec.events`

- **`on_step`**: live host callback receiving carry-tap `TapEvent` objects
  (`kind=="carry"`). Unchanged from 0.2.0.
- **`on_ys`**: live host callback receiving output-tap `TapEvent` objects
  (`kind=="output"`). New. Independent of `on_step`.
- **`rec.events`** (FlightRecorder): single list containing **both** carry and
  output events. The FlightRecorder's `__call__` is wired to both `tap_cb` and
  `y_tap_cb` closures, so it collects all events regardless of `on_step`/`on_ys`.
  Filter with `e.kind`:
  ```python
  output_events = [e for e in rec.events if e.kind == "output"]
  carry_events  = [e for e in rec.events if e.kind == "carry"]
  ```

### Does `sample_every` apply to y-taps?

Yes. `sample_every` gates both carry and y-tap callbacks via the same device-side
`lax.cond(step % sample_every == 0, ...)` mechanism. The gate is applied once in
`body_fn`; if the step is gated out, neither callback fires for that step. This
preserves the semantic consistency: both events from the same step either fire
together or are both suppressed.

### Does a y-tap without a carry tap make sense?

Yes. Setting `on_ys` / `select_ys` without `on_step` / `select` is valid.
The carry-tap block (`if emit_carry: tap_cb(...)`) is independently guarded.
With `on_step=None, select=None, alert=None, on_ys=<fn>, select_ys=<fn>`,
only output events appear in `rec.events`. The carry callback bakes a no-op
into the compiled artifact (existing behavior when `on_step=None and alert=None`).

The y-tap callback similarly bakes a no-op when
`on_ys=None, select_ys=None, alert_ys=None` — zero `jax.debug.callback`
overhead for y-taps when none are configured.

### Interaction with `ops` and `where` / `max_depth`

`select_ys` applies to ALL scans that pass the carry-tap emission filter
(`ops` / `where` / `max_depth`). It does not apply to scans that are filtered
out by `ops` (e.g. `ops=("while_loop",)` → no scan y-taps). This matches the
behavior of `select` for carry taps.

An additional `ops_ys` parameter is **not** proposed at this stage. The
common case (tap the same scans for both carry and output) is handled by the
shared `ops` / `where` / `max_depth`. Advanced differentiation can be added
later.

---

## 4. Cost

Each y-tap event calls `jax.debug.callback` once per step with:
- 1 scalar (step)
- N arrays from `select_ys(ys_leaves)` (or all ys_leaves if `select_ys=None`)

The benchmark baseline from `bench/callback_floor.py` and the A1 decomposition
study is ~15–19 µs per operand crossing the host boundary. This is the same
cost as carry taps.

**Per-step y-tap overhead:**
- `select_ys=None`: all ys leaves shipped. For NUTSInfo (~4 scalar fields):
  4 × 15–19 µs ≈ 60–76 µs/step (on top of carry-tap cost).
- `select_ys=lambda leaves: leaves[treedepth_idx]`: 1 scalar shipped ≈
  15–19 µs/step.
- `select_ys=lambda _: ()`: empty select, 0 array operands shipped. Only the
  step scalar crosses. Fixed cost ≈ 15–19 µs/step (callback invocation floor).
  Useful as a "y-step heartbeat" to count ys-side events without data transfer.

**`sample_every` mitigation:** at `sample_every=10`, per-step amortised cost
drops to ~1.5–8 µs (consistent with carry-tap measurements in bench/README.md).
For tuningfork NUTS at 1000 warmup steps with `sample_every=10`, total y-tap
overhead ≈ 1.5–2 ms — negligible against typical NUTS runtimes.

**Recommendation:** default to `select_ys=<specific field selector>` (not
`None`) to avoid shipping all NUTSInfo fields when only treedepth is needed.
Gate with `sample_every=10` for semi-production monitoring.

---

## 5. Scope boundary

**y-taps are scan-only.**

`jax.lax.while_loop` has the signature:

```python
jax.lax.while_loop(cond_fun, body_fun, init_val) -> carry
```

The body returns only a carry (`carry -> carry`). There is no per-step output
accumulation in while_loop. Setting `select_ys` on a function that contains
only while_loops produces **zero output events** — the y-tap callback is never
inserted by `rewrite_while` (which has no `ys` variable and no y-tap call site).

This is a **no-op, not an error**, by design:
- Raising an error at trace time would require inspecting the jaxpr before
  entering the rewrite, adding latency.
- A `UserWarning` at `verbose()`/`record()` call time is the recommended
  safeguard: emit it when `select_ys` is set but `"scan"` is not in `ops`.
  (A stronger check would trace the function and confirm no scan primitives
  are present, but that doubles the trace cost — not worth it at API entry.)

**Proposed warning text:**
```
jaxtap: select_ys has no effect when ops does not include "scan" (while_loop
has no per-step ys accumulation). Remove select_ys or add "scan" to ops.
```

### None-ys scans — the progress-bar idiom (AYS-1 must-fix)

A large fraction of real scans use `lax.scan(body, init, None, length=N)` where
the body returns `(carry, None)` — the progress-bar pattern, used in jaxtap's
own demos and many BlackJAX loops. In these scans there are **no per-step
outputs**: `jax.tree_util.tree_leaves(None) == []`.

**Executed evidence (scratch/poc_ays1.py):**
```
Case A: body returns (carry, None)
  [trace] y_leaves = []  (len=0)
  [trace] y_tree = PyTreeDef(None)
none_y_events collected: 5   ← FOOTGUN: 5 empty-value events fired
  first event: step=0, val=None
```

Without a fix, setting `select_ys` on a None-output scan fires a stream of
`TapEvent(kind="output", value=None)` events every step — useless noise.

**Required fix:** in `body_fn` of `rewrite_scan`, guard the y-tap call:

```python
# ys is already a flat list (outs[ncar:])
if y_tap_cb is not None and len(ys) > 0:
    y_tap_cb(here, step, *ys, total=total)
```

`len(ys) > 0` is a Python (trace-time) check on the flat list length, evaluated
once at trace time — zero device-side overhead. When `ys` is empty (body returns
`(carry, None)` or any ys pytree with no leaves), the y-tap call is skipped
entirely: no `jax.debug.callback` baked in, zero events.

This means `select_ys` on a None-output scan is a **silent no-op** (zero events,
no error, no warning beyond the already-proposed `ops` check). This is the right
behavior: the scan has nothing to tap, so nothing fires. Users writing
`select_ys=lambda _: treedepth_idx` against a None-ys scan get silence, not spam.

**Note:** `select_ys=lambda _: ()` as a "ys-side heartbeat" only makes sense
when the scan body actually produces ys. On a None-output scan it would also
produce silence after the `len(ys) > 0` guard (since `ys=[]` before any selector
is applied). This is correct — there is no per-step ys to heartbeat on.

---

## 6. Implementation sketch

### Files to change

| File | Change | Estimated LOC |
|------|--------|---------------|
| `src/jaxtap/_rewrites.py` | Add `y_tap_cb` param to `rewrite_scan`; insert y-tap call after carry tap; no change to `rewrite_while` | ~25 |
| `src/jaxtap/__init__.py` | Add `on_ys`, `select_ys`, `alert_ys`, `alert_ys_once` to `verbose()` and `record()`; build `y_tap_cb` closure; add `kind` to `TapEvent`; thread `y_tap_cb` to `interpret()` | ~90 |
| `src/jaxtap/_walker.py` | Thread `y_tap_cb` through `interpret()` and `_interp()` to `rewrite_scan` | ~15 |
| `src/jaxtap/_ashell.py` | Forward `on_ys`, `select_ys`, `alert_ys`, `alert_ys_once` in `_RecordContext` | ~20 |
| `tests/test_ytaps.py` | New: basic, jit, vmap, nested, on_ys, select_ys, alert_ys, alert_ys_once, ys-only, while no-op, None-ys no-op, sample_every | ~200 |

**Total new production LOC: ~150. Total including tests: ~350.**

### What `rewrite_scan` change looks like

```python
def rewrite_scan(
    eqn,
    invals,
    tap_cb,
    ops,
    here,
    interp_fn,
    outer_step=None,
    *,
    emit_carry: bool = True,
    y_tap_cb: TapCallback | None = None,   # ← new parameter
) -> list:
    ...

    def body_fn(carry_step, x):
        carry, step = carry_step
        ...
        outs = interp_fn(...)
        new_carry = outs[:ncar]
        ys = outs[ncar:]   # already flat; len==0 when body returns (carry, None)
        if emit_carry:
            tap_cb(here, step, *new_carry, total=total)
        # NEW: y-tap fires after carry tap, same step value.
        # AYS-1 fix: len(ys) > 0 guard prevents empty-event spam on None-output scans.
        # This is a Python (trace-time) check — zero device-side overhead.
        if y_tap_cb is not None and len(ys) > 0:
            y_tap_cb(here, step, *ys, total=total)
        return (new_carry, step + 1), ys
```

The `y_tap_cb` closure is built in `verbose()` analogously to `tap_cb`:
- Applies `select_ys` on-device (before the debug.callback)
- Captures `sel_ys_tree` at trace time for host-side pytree reconstruction
- On the host: calls `_fire_carry_alert(alert_ys, event, alert_ys_once, ...)`,
  then `_guard(on_ys, event)` (the separate output callback), then
  `_guard(recorder, event)` so `rec.events` captures it — all with
  `kind="output"` on the TapEvent. `on_step` is NOT called for output events.

The `sample_every` gate applies to the y-tap via the same `lax.cond` wrapper
already used for `tap_cb`: build `y_tap_cb` as `_base_y_tap_cb`, then wrap it
if `sample_every > 1`. Both `tap_cb` and `y_tap_cb` are wrapped independently
but with the same `sample_every` value, so they gate in sync.

### `TapEvent` change

```python
@dataclasses.dataclass(frozen=True)
class TapEvent:
    path: str
    step: int
    value: Any
    total: "int | None" = None
    kind: str = "carry"   # "carry" | "output"  — new; default preserves compat
```

The `kind="carry"` default means all existing code that constructs or inspects
`TapEvent` without the `kind` field continues to work.

### `_RecordContext` (_ashell.py) change

`_RecordContext.__init__` gains `select_ys`, `alert_ys`, `alert_ys_once` kwargs
forwarded to `verbose()` via `_make_verbose_kwargs()`. The A-form
(`with tap.record(select_ys=...) as rec`) then works transparently.

---

## 7. Alert line format (AYS-1 minor)

Both carry and y-tap alerts produce the same terse format:
```
[tap] FAIL scan[0] 5/1000: <msg>
```
A carry alert and a y-tap alert on the same scan are indistinguishable if the
message is generic (e.g. `alert_ys=lambda e: True` → `"alert"`).

**Recommendation: no format change.** Self-identifying messages are the
existing convention for carry taps too — a generic `alert=lambda e: True` on
carry produces the same undifferentiated line. The tuningfork use case already
writes a self-identifying message:
```python
alert_ys=lambda e: f"treedepth saturated at step {e.step}" if float(e.value) >= 10.0 else False
```
This is the correct idiom for both carry and output alerts. Document in the
`alert_ys` docstring: "Messages should be self-identifying; include 'output:'
or a field name to distinguish from carry-tap alerts in log grep."

An alternative (`scan[0](ys)` suffix in the path column of the alert line) was
considered and rejected: it adds a format divergence between alert lines and
`TapEvent.path`, creating confusion when users cross-reference the two.

---

## 8. Ratified decisions record

These were open questions before JP ratification; all are now closed.

1. **`kind` field backward-safe** — confirmed by execution (frozen dataclass,
   no positional-unpack sites, all keyword construction, `test_collectors.py:305`
   unaffected). No action needed at implementation time.

2. **`df()` unchanged** — `FlightRecorder.df()` does not include `kind` or
   `total` columns; consumers who need `kind` in a DataFrame build it from
   `rec.events` directly. No `df()` change in this arc.

3. **None-ys no-op via `len(ys) > 0` guard** — trace-time Python check in
   `rewrite_scan.body_fn`; zero device overhead. Setting `on_ys`/`select_ys` on
   a scan that returns `(carry, None)` produces zero events silently.

4. **ys leaf order** — flat-leaves order follows JAX pytree convention (namedtuple
   field order; dict sorted keys). Document with an example in `select_ys` docstring.

5. **`alert_ys` receives a full `TapEvent`** — consistent with `alert` for carry.
   Lets predicates use `e.step` and `e.total` (e.g. suppress alerts during warmup).

6. **JSONLWriter drops `kind`** — acceptable for 0.3.0; note in release notes if
   y-tap JSONL streaming is likely.

7. **`on_ys` separate from `on_step`** (JP decision) — carry events → `on_step`;
   output events → `on_ys`. Both feed into `rec.events` via the FlightRecorder
   side-channel. This is the ratified routing model.

8. **Schedule: deferred to 0.3.0** — do not start implementation until #2/#5
   have shipped in 0.2.1.

---

## Appendix A: original PoC output (scratch/poc_ytaps.py)

```
JAX version: 0.10.2

============================================================
Q1a: basic carry+y tap, non-jitted
============================================================
carry_out: 10.0
carry events (5): [(0, Array(0., ...)), (1, Array(1., ...)), (2, Array(3., ...)), (3, Array(6., ...)), (4, Array(10., ...))]
y events (5): [(0, {'is_div': Array(False, ...), 'treedepth': Array(0., ...)}), (1, ...), ...]
PASS: both carry and y taps fire; both have correct step indices and kinds

============================================================
Q1b: jitted scan — carry+y taps both fire
============================================================
carry events: 5, y events: 5
PASS: jitted scan fires both carry and y taps

============================================================
Q1c: vmapped scan (2 chains) — carry+y taps per lane
============================================================
carry events: 10, y events: 10   (2 lanes × 5 steps)
PASS: vmapped scan fires carry and y taps for all lanes

============================================================
Q1d: nested scan (outer/inner) — y taps at each level
============================================================
outer y events: 2
inner y events: 6   (2 outer × 3 inner)
PASS: nested scan fires y taps at each level correctly

============================================================
Q2: SEMANTICS — pytree ys, flat leaves, TapEvent shape
============================================================
pytree y events: 12
treedepths observed: [1.0, 2.0, 3.0, 4.0, 5.0] ...
saturation events (treedepth >= 10): [(9, 10.0)]
PASS: pytree ys with select → TapEvent shape consistent; treedepth tripwire works

============================================================
Q3: API — single rec.events stream with kind discriminator
============================================================
total events: 6 (carry=3, output=3)
PASS: single event stream; kind discriminator separates carry/output; carry fires first

============================================================
Q4: COST — empty select_ys idiom for heartbeat with zero payload
============================================================
heartbeat count: 100 (expected 100)
PASS: empty-select heartbeat idiom confirmed

============================================================
Q5: SCOPE — while_loop has no ys (scan-only boundary confirmed)
============================================================
while_loop result: 5 (scalar carry, no ys)
PASS: scope boundary confirmed — y-taps are scan-only by construction

ALL PoC CHECKS PASSED
```

## Appendix B: AYS-1 gap analysis output (scratch/poc_ays1.py)

```
JAX version: 0.10.2

============================================================
Q1: TapEvent type + backward-compat audit
============================================================
dataclasses.is_dataclass(TapEvent): True
has NamedTuple ._fields: False
positional unpack raises TypeError: cannot unpack non-iterable TapEvent object
=> No positional unpack risk — frozen dataclass is not iterable

TapEvent fields: [('path', MISSING, ...), ('step', MISSING, ...), ('value', MISSING, ...),
                  ('total', None, ...)]

All construction sites in src/jaxtap/ use keyword arguments only.
kind='carry' default means every existing site continues to work unchanged.
PASS Q1(a,b,d): dataclass, no positional unpack, all keyword construction sites

df() builds rows as {path, step, **_value_to_columns(value)} explicitly.
kind is NOT in df() unless we explicitly add it.
Claim in draft doc 'df() gains kind column for free' is WRONG — corrected.
Recommendation: do NOT add kind to df() in this arc; no test breakage.

============================================================
Q2: None-ys scans — what does select_ys do on (carry, None) body?
============================================================
Case A: body returns (carry, None)
  [trace] y_leaves = []  (len=0)
  [trace] y_tree = PyTreeDef(None)
none_y_events collected: 5   ← FOOTGUN (without fix)
  first event: step=0, val=None

Case B: what does jax.tree_util.tree_leaves(None) return?
  tree_leaves(None) = []
  tree_structure(None) = PyTreeDef(None)

FIX: guard with len(ys) > 0 in body_fn — Python trace-time check, zero overhead.

Case C: body returns (carry, scalar) — scalar ys fires normally
  scalar_y_events: 3 (expected 3)
  first: step=0, val=0.0

============================================================
Q3: alert line ambiguity
============================================================
RECOMMENDATION: no format change; document self-identifying message convention.
```
