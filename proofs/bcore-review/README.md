# bcore-review — 2-arm adversarial review of the B-core (2026-07-08)

The full 2-arm adversarial review of the jax-tap B-core (`tap.verbose` walker),
brought forward from M3 to review the load-bearing foundation while small/fresh
(JP: "small trunk in the beginning is critical so we build on a good
foundation"). Disjoint lanes: arm A = semantics/numerics, arm B = jaxpr
structure. TL 2×AYS each. Repros frozen here as conformance-suite seed.

## Arm A (semantics/numerics) — VERDICT: foundation SOLID, no BLOCKERs

Headline: **zero value / gradient / dtype corruption** on any valid program —
grad³, hessian, complex64 holomorphic, dtype edges, degenerate loops, reverse,
cond/switch-in-scan, vmap-over-scan all bitwise-identical. Every defect is in
the instrumentation CONTRACT, not the numerics.

| # | Finding | TL 2×AYS verdict |
|---|---------|------------------|
| A1 | `vmap(while_loop)` over-fires (30 vs 16) + delivers ghost carry values (counter=11, acc=19) | **RECLASSIFIED: inherent JAX boundary, NOT a jax-tap bug.** `ays/ays_a1_baseline.py` proves a raw `jax.debug.callback` in a hand-written vmapped while (no jaxtap) fires the identical 30 + identical ghost. Batched while runs max-trip joint iters; masked lanes' bodies still execute+fire. Response: DOCUMENT (ratified design point #2). v1.x mitigation candidate: gate the callback on the per-lane cond predicate (evaluable — `ays/ays_a2a3_round2.py` shows `[True,False,False]`; raw baseline can't do this). |
| A2 | `verbose(jax.checkpoint(f))` emits ZERO events (silent) | **CONFIRMED real gap.** `remat2` carries `params['jaxpr']` (the inner scan); walker binds it opaquely instead of recursing. **FIX DIRECTION CORRECTED** (TL self-AYS, `ays/ays_a2_fix_direction.py`): NOT a one-line add to `_JIT_PRIMS` — that branch re-wraps in `jax.jit`, which would DROP `remat2`'s `prevent_cse`/`policy`/`differentiated` (the checkpoint memory boundary → silent memory regression). Correct fix = a DISTINCT branch: recurse into `remat2.jaxpr`, rebuild instrumented, RE-BIND via `remat2.bind` + `get_bind_params` (preserve the remat params). Note the A3 consequence: instrumenting inside a checkpoint inherently double-fires under grad (document). |
| A3 | `checkpoint(verbose(f))` double-fires under grad (10 vs 5) | **inherent** — remat recomputes forward during backward. Document with A1. |
| T1 | int32 step-counter overflow at >2³¹ iters | THEORETIC; corrupts only `TapEvent.step`, never primal (separate carry slot). |

Env note (arm A): JAX is **0.10.2** (not 0.10.1); scan params still carry no
`linear`/`_split_transpose`, so `rewrite_scan` param-forwarding remains safe.

## Arm B (jaxpr structure) — pending

## AYS counterexamples (TL)

- `ays/ays_a1_baseline.py` — raw vmapped-while + debug.callback (no jaxtap):
  proves A1 is inherent (30 events + ghost 11.0, matching jaxtap exactly).
- `ays/ays_a2a3_round2.py` — proves `remat2` is recursable (A2 fixable) and the
  A1 cond-gating mitigation is feasible.
