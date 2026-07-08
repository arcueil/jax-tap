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

## Arm B (jaxpr structure) — VERDICT: no BLOCKER, 2 MAJOR structural bugs

Headline: numerics bitwise-correct everywhere; 2 MAJOR instrumentation bugs,
both CONFIRMED in clean M0 (TL re-ran in an isolated worktree).

| # | Finding | TL 2×AYS verdict |
|---|---------|------------------|
| F1 | instrumentation silently dropped inside `cond`/`switch`/`remat2` (walker's `else` binds opaquely, never recurses) | **CONFIRMED real bug.** scan-in-cond/switch → 0 events (top-level → 5). Subsumes arm A's A2. Undocumented, large blast radius. Remediate. |
| F2 | jit boundary collapses addressing → non-unique paths (jit branch keeps path + `_interp` resets `n_cf=0`) | **CONFIRMED real bug.** top-level + jit-nested scan both `scan[0]`, 8 events merged. Mislabels; also corrupts M2 `where=`/`max_depth=` across jit. Remediate. |
| jit/pjit param-drop | re-wrap forwards only the sub-jaxpr | CPU correctness-neutral (arm B retracted its own "Array deleted" false positive); THEORETIC multi-device sharding only. Document. |
| — MINOR | re-wrap name `_inner_call` lost from profiles; non-array output → TypedInt | inherent to make_jaxpr interpreters. |

CLEAN (bounds remediation): effects/ordered callbacks, io_callback(ordered), 7
carry/pytree edges, intra-jaxpr nesting, **M2 sample_every×vmap gate holds**.

### Unified remediation invariant (arm B closing insight + TL)

**Descend into the sub-jaxpr but re-emit through the ORIGINAL primitive's own
`bind` with its params threaded — NEVER re-wrap in a different primitive.** A
fresh `jax.jit(_inner_call)` synthesizes a NEW higher-order eqn whose params
default, dropping `prevent_cse`/`donation`/`shardings`/`name`. One rule yields:
jit-transparent-with-continuous-CF-counter (fixes F2), cond/switch
branch-visible addressing (fixes F1), remat-preserving (subsumes A2).

### M2 AYS (host-side layer, proportionate review)

`ays/ays_m2_sample_vmap.py`: `sample_every=k` throttle HOLDS under vmap (16
events = 4 sampled steps × 4 lanes, steps {0,3,6,9}) — swe-m2's flagged concern
(batched-cond runs both branches → fire every step) did NOT materialize;
`jax.debug.callback`'s vmap rule masks per-lane. Corroborated by arm B's
independent `attack_sample_vmap.py`.

## AYS counterexamples (TL)

- `ays/ays_a1_baseline.py` — raw vmapped-while + debug.callback (no jaxtap):
  proves A1 is inherent (30 events + ghost 11.0, matching jaxtap exactly).
- `ays/ays_a2a3_round2.py` — proves `remat2` is recursable (A2 fixable) and the
  A1 cond-gating mitigation is feasible.
