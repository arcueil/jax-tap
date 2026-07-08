# attack-ledger-964 — rescued adversarial-review repro scripts (frozen evidence)

Rescued 2026-07-08 from `/tmp/arm-a/` (27 files) and `/tmp/arm-b/` (32 files)
before tmp cleanup — the repro scripts from the 2-arm adversarial review of
blackjax PR #964 (the mechanism jax-tap generalizes). Provenance: the archived
thread `blackjax-devs/claude-config/project/worklog/threads/_archive/`
`feat-progress-bar-context-manager.md` (§ Adversarial review record).

- **arm-a/** — semantics attacks: grad/remat/checkpoint, AOT, jit-cache
  staleness, vmap counter semantics + batched-arg unroll counterexamples
  (`ays2_1_batched_arg_unroll.py` is THE arm-A counterexample for the
  args-stay-unbatched invariant), reverse order, pytree xs, kwargs passthrough,
  length mismatch, phantom-callback dead-state fire (`ays_v_aot_dead_state_fire.py`),
  overhead measurement (`ays2_2_overhead_and_closed_flag.py`, the ~32 µs/step floor).
- **arm-b/** — global-state attacks: patch clobber matrix, foreign-patch
  chaining, non-LIFO exits, registry races, thread cross-talk + owner-affinity
  decision table (`26_full_decision_table.py`), manual-enter leaks, GC self-heal,
  shard_map multi-device fire, output_file failure modes (the #B14 BLOCKER),
  contextvars propagation, bystander absorption.

Review headline: **zero numeric corruption in any constructible case**; every
defect was display, hygiene, or one crash path. Final defect list + verdicts in
the archived thread.

## Status in this repo

These scripts target `blackjax.progress_bar` and are NOT runnable against
jaxtap as-is. They are the **source material for M3** (conformance suite):
each attack gets ported to target `jaxtap`'s A-shell/B-core equivalents, or
recorded as N/A with a reason (e.g. display-thread attacks — jax-tap ships no
display frontend per the ratified emission/read-out split). Do not edit these
files; port them.
