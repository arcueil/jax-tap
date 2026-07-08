# ays-m0 — TL adversarial interrogation of the M0 deliverable (2026-07-08)

Frozen evidence from the TL's 2×AYS pass on the M0 walker (tl.md protocol
item 3: interrogate initial implementations, not just review arms). The M0
suite was 12/12 green but three seams flagged in the M0 brief were untested;
AYS probed them empirically.

| Script | Seam | Verdict |
|--------|------|---------|
| `ays_seams_vmap_grad_customjvp.py` | vmap-safety, grad-through-transform, custom_jvp opaque-bind | **1 DEFECT**: custom_jvp inside a scan body → `KeyError: 'subfuns'`. vmap + grad PASS (but were untested → regression gap). |
| `rootcause_customjvp_subfuns.py` | root-cause of the crash | `jax.nn.softplus` → nested `jit` → `custom_jvp_call` eqn; the walker's naive `bind(*invals, **eqn.params)` cannot bind a primitive carrying `subfuns`. |
| `round2_customvjp_sentinel_rule.py` | AYS **round 2** — attack the FIX (item 4) | 3 probes, all PASS: [C] `get_bind_params` returns a flat dict (arity correction confirmed); [B] a custom_jvp with a sentinel derivative (=42, ≠ primal 2x) propagates as 42³ through verbose → custom rule genuinely survives; [A] `custom_vjp` (untested by R1) forward+grad bitwise, sentinel cotangent 7³ survives. |

## The defect

Any program using `jax.nn.*` / `logsumexp` / `erf` / any `@custom_jvp` function
inside tapped control flow crashes. This is the exact seam the M0 brief flagged
("custom_jvp_call uses call_jaxpr, not jaxpr") — the "bind opaquely" answer does
not actually work for primitives with closed sub-functions.

## Fix constraint (why it is NOT a one-liner)

The canonical `subfuns, bind_params = prim.get_bind_params(eqn.params); prim.bind(
*subfuns, *invals, **bind_params)` pattern (as in `jax.core.eval_jaxpr`) is the
direction — BUT a blunt blanket application broke `grad(verbose(f))`: the fix
must **preserve the custom_jvp/custom_vjp rule** so autodiff through the
reconstructed program still differentiates correctly. Forward-eval correctness
AND grad correctness must both hold. See the M1-adjacent fix commit for the
validated resolution + regression tests (custom_jvp-in-scan, vmap, grad).
