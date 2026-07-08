# proofs/ — plan-mode risk-retirement sketches (2026-07-08, JAX 0.10.1)

Runnable artifacts from the TL plan-mode pass that ratified **B-core-at-v1**
(see `arcueil-config/project/worklog/decisions/2026-07-08-jax-tap-b-core-at-v1.md`).

| File | Proves |
|------|--------|
| `jaxtap_probe.py` | interpreter-surface availability on 0.10.1: `jax.extend.core` types, `jax.core.eval_jaxpr`, scan/while/jit eqn param shapes |
| `jaxtap_sketch.py` | recursive eqn-walker: bitwise identity, per-step carry taps, nested `scan[0]/scan[0]` addressing, jit-composition both directions |
| `jaxtap_while_sketch.py` | while_loop rebuild: cond/body wrapping around the step-augmented carry, closed-over consts in both, heartbeat taps, bitwise identity |

These are frozen evidence, not library code — the walker in `src/jaxtap/` is the
maintained implementation. Run with `uv run python proofs/<file>.py`; each prints
`ALL CHECKS PASSED`.

Known version facts they encode (do not re-learn these the hard way):

- `jax.util` is REMOVED in 0.10.1.
- scan params are `{jaxpr, length, num_carry, num_consts, reverse, unroll}` —
  no `linear`/`_split_transpose`; pass params through generically.
- jitted programs wrap bodies in a `jit`-named eqn (not `pjit`) with
  `params["jaxpr"]`.
- while params: `{body_jaxpr, body_nconsts, cond_jaxpr, cond_nconsts}`.
