# M3 Conformance Coverage Map

One row per attack/probe script across all corpora.

**Dispositions:**
- `covered: tests/<file>::<test>` — already a permanent test
- `ported: tests/conformance/<file>::<test>` — ported here
- `documented-boundary` — inherent behaviour with docstring reference
- `N/A: <reason>` — not applicable (blackjax display thread, superseded API, etc.)

**Conformance test files (both verified passing):**
- `tests/conformance/test_bcore_conformance.py` — 25 tests, B-core lane (bcore-review + m1a/m1d-ays + A1 mitigation)
- `tests/conformance/test_ashell_conformance.py` — 14 tests, A-shell lane (ashell-review arm-s/arm-l)

**Note on attack-ledger-964:** All 53 scripts in arm-a and arm-b targeted the blackjax progress-bar
subsystem, not jaxtap. Each row below documents the semantic equivalent in jaxtap (if any) and
which existing or ported test covers it. The translations confirm coverage parity without porting
blackjax-specific mechanics.

---

## proofs/attack-ledger-964/arm-a/ — historical blackjax progress-bar attacks

| source script | what it attacks | disposition |
|---|---|---|
| `attack_aot.py` | ahead-of-time compilation of scan with progress bar | N/A: targets blackjax display thread (never ported to jaxtap) |
| `attack_checkpoint.py` | checkpoint/remat inside progress-bar scan | N/A: targets blackjax |
| `attack_checkpoint_isolate.py` | isolating checkpoint behaviour under bar | N/A: targets blackjax |
| `attack_checkpoint_isolate2.py` | checkpoint isolation variant 2 | N/A: targets blackjax |
| `attack_cond_cache_isolate.py` | cond cache isolation under bar | N/A: targets blackjax |
| `attack_cond_while.py` | cond/while under bar | N/A: targets blackjax |
| `attack_grad.py` | grad through bar-patched scan | N/A: targets blackjax |
| `attack_jit_cache_cross.py` | jit-cache cross-contamination (trace inside ctx / call after exit) | N/A: targets blackjax; semantic equivalent covered: `tests/test_ashell.py::test_ashell_cache_hit_new_context`, `test_ashell_no_phantom_after_exit` |
| `attack_length_xs_mismatch.py` | mismatched xs leaf lengths | N/A: targets blackjax |
| `attack_misc_kwargs.py` | kwargs passthrough (f=, init=, xs=, unroll=True bool) | N/A: targets blackjax; semantic equivalent covered: `tests/test_ashell.py::test_ashell_scan_positional_reverse`, `test_ashell_scan_positional_unroll`; see also `ported: tests/conformance/test_ashell_conformance.py::test_ashell_all_keyword_form` |
| `attack_plain_jit_repeat.py` | plain-jit repeated scan bar counting | N/A: targets blackjax |
| `attack_pytree_xs.py` | pytree xs (dict/namedtuple/None field) | N/A: targets blackjax; semantic equivalent: `ported: tests/conformance/test_ashell_conformance.py::test_ashell_pytree_xs_dict` |
| `attack_real_jitted_repeat.py` | real jitted repeated inference run | N/A: targets blackjax.nuts |
| `attack_real_jitted_repeat_counting.py` | jitted repeat counting in real algorithm | N/A: targets blackjax |
| `attack_real_repeat_run.py` | sequential chains with same algorithm | N/A: targets blackjax.nuts + run_inference_algorithm |
| `attack_reverse_order.py` | reverse=True in scan under bar | N/A: targets blackjax; semantic equivalent covered: `tests/test_ashell.py::test_ashell_scan_positional_reverse` |
| `ays2_1_batched_arg_unroll.py` | batched arg forcing unroll in debug.callback under vmap | N/A: probes jax.debug.callback internals; not a jaxtap API test |
| `ays2_2_overhead_and_closed_flag.py` | per-step overhead of orphaned callbacks + closed guard cost | N/A: benchmarking study; not a conformance assertion |
| `ays_i_single_context_loop.py` | one bar ctx around whole chain loop (cache-hit counter semantics) | N/A: targets blackjax |
| `ays_ii_fresh_wrapper_tracecount.py` | fresh wrapper trace-count under re-entry | N/A: targets blackjax |
| `ays_iii_reverse_fix_repro.py` | reverse direction fix repro | N/A: targets blackjax |
| `ays_iii_vmap_counter_semantics.py` | vmap counter semantics | N/A: targets blackjax |
| `ays_iii_vmap_real_sampler_probe.py` | vmap real sampler probe | N/A: targets blackjax |
| `ays_iv_vmap_cond_fresh.py` | vmap + cond fresh-context | N/A: targets blackjax |
| `ays_iv_vmap_cond_while.py` | vmap + cond + while | N/A: targets blackjax |
| `ays_iv_vmap_while_fresh.py` | vmap + while fresh-context | N/A: targets blackjax |
| `ays_v_aot_dead_state_fire.py` | AOT dead-state fire after context closes | N/A: targets blackjax |

## proofs/attack-ledger-964/arm-b/ — blackjax A-shell design study

| source script | what it attacks | disposition |
|---|---|---|
| `01_import_hygiene.py` | import side-effects of blackjax | N/A: targets blackjax |
| `02_exception_restoration.py` | exception-time patch restoration | N/A: targets blackjax |
| `03_manual_enter_no_exit.py` | manual enter without exit (leak) | N/A: targets blackjax; semantic equivalent covered: `tests/test_ashell.py::test_ashell_gc_selfheal` |
| `04_bystander_scan_inside_with.py` | bystander scan inside with block (single-ctx delegate) | N/A: targets blackjax; semantic: `documented-boundary` (L4 bystander delegation — single-ctx delegate is documented in `_ashell.py::_select_ctx`) |
| `05_thread_crosstalk.py` | thread crosstalk under bar | N/A: targets blackjax |
| `06_registry_race_stress.py` | concurrent enter/exit race (blackjax registry) | N/A: targets blackjax; semantic equivalent: `ported: tests/conformance/test_ashell_conformance.py::test_ashell_registry_race_bounded` |
| `07_shard_map_scan.py` | shard_map inside progress bar | N/A: targets blackjax; shard_map not a jaxtap target |
| `08_patch_clobber_matrix.py` | patch clobber matrix (blackjax) | N/A: targets blackjax; semantic equivalent covered: `tests/test_ashell.py::test_ashell_foreign_patch_chain`, `test_ashell_foreign_patch_over_us` |
| `09_partial_captured_before.py` | partial captured before bar entry | N/A: targets blackjax |
| `10_shared_output_file.py` | shared output file between contexts | N/A: targets blackjax display layer |
| `11_nested_contexts.py` | nested blackjax progress bars | N/A: targets blackjax; semantic equivalent covered: `tests/test_ashell.py::test_ashell_reentrant_contexts` |
| `12_leak_check.py` | registry leak check | N/A: targets blackjax |
| `13_async_dispatch_bar_disappears.py` | async dispatch bar disappears | N/A: targets blackjax display thread |
| `14_two_owned_contexts_no_affinity.py` | two owned contexts without thread affinity | N/A: targets blackjax; semantic equivalent: `ported: tests/conformance/test_ashell_conformance.py::test_ashell_two_owned_contexts_bystander` |
| `15_gc_selfheal.py` | GC self-heal of leaked blackjax patch | N/A: targets blackjax; semantic equivalent covered: `tests/test_ashell.py::test_ashell_gc_selfheal` |
| `16_worker_thread_delegation.py` | worker-thread delegation (main ctx, worker scan) | N/A: targets blackjax; semantic equivalent covered: `tests/test_ashell.py::test_ashell_thread_delegation` |
| `17_contextvars_propagation_semantics.py` | contextvars propagation semantics study | N/A: infrastructure exploration study; not a jaxtap assertion |
| `18_targeted_affinity_heuristic.py` | targeted affinity heuristic | N/A: targets blackjax |
| `19_patch_stack_fix.py` | patch stack fix (non-LIFO) | N/A: targets blackjax; semantic equivalent: `ported: tests/conformance/test_ashell_conformance.py::test_ashell_nonlifo_exit_order` |
| `20_axis_index_feasibility.py` | axis_index feasibility under vmap | N/A: feasibility study for a feature not implemented in jaxtap |
| `21_skewed_device_premature_complete.py` | skewed device premature completion | N/A: targets blackjax display thread |
| `22_reader_races_delete_ACTUAL.py` | reader races on delete | N/A: targets blackjax file I/O layer |
| `23_measure_actual_gap_duration.py` | measure gap duration | N/A: blackjax timing study |
| `24_readonly_dir_permission_error.py` | readonly dir permission error | N/A: targets blackjax file I/O layer |
| `25a_probe_idx_under_vmap.py` | probe idx under vmap | N/A: targets blackjax |
| `25b_display_thread_closed_stderr.py` | display thread closed stderr | N/A: targets blackjax display thread |
| `25c_confirm_tqdm_defensive.py` | tqdm defensive guard | N/A: targets blackjax/tqdm display layer (never ported) |
| `25_hardened_callback.py` | hardened callback under vmap | N/A: targets blackjax |
| `26_full_decision_table.py` | full design decision table | N/A: targets blackjax design study |
| `27_nonlifo_exit_order.py` | non-LIFO exit order restores correctly | N/A: targets blackjax; semantic equivalent: `ported: tests/conformance/test_ashell_conformance.py::test_ashell_nonlifo_exit_order` |
| `28_confirm_v1_stack_design_was_buggy.py` | confirm v1 stack design was buggy | N/A: historical design study; bug fixed in jaxtap from the start |
| `29_real_ipython_kernel_test.py` | real IPython kernel test | N/A: targets blackjax + IPython kernel |

## proofs/bcore-review/arm-a/ — B-core attack arm A (verbose() behaviour)

| source script | what it attacks | disposition |
|---|---|---|
| `dtypes_degenerate.py` | int32, bool, complex64, mixed-dtype, weak-type carries through verbose() | `ported: tests/conformance/test_bcore_conformance.py::test_int32_carry_bitwise`, `test_complex64_carry_bitwise`, `test_mixed_dtype_carry_bitwise` |
| `grad2_hessian.py` | grad^2, grad^3, hessian through verbose(scan) | `ported: tests/conformance/test_bcore_conformance.py::test_higher_order_autodiff` |
| `probe_params.py` | jit eqn params enumeration + donated_invars | `documented-boundary`: jit param drop (donated_invars, static_argnums etc.) is inherent to the re-wrap design; correctness on CPU single-device is preserved (tested via bitwise) |
| `remat_primname.py` | primitive name for remat2/checkpoint | `covered: tests/test_jaxtap.py::test_scan_in_checkpoint_f1` |
| `remat.py` | checkpoint(verbose(f)) double-fire under grad; verbose(checkpoint(f)) events | `covered: tests/test_jaxtap.py::test_checkpoint_grad_bitwise`, `test_scan_in_checkpoint_f1` |
| `reverse_and_cond.py` | reverse=True scan + cond under verbose() | `covered: tests/test_jaxtap.py::test_scan_in_cond_f1` (cond); reverse covered by `test_carry_leaves_contract` |
| `vmap_select.py` | nested vmap(vmap(verbose)), batched carry only, batched xs only, vmap(grad), lossy select, holomorphic grad | `covered: tests/test_jaxtap.py::test_vmap_safety`, `test_select_reduce_on_device`, `test_grad_through_transform` |
| `vmap_while.py` | vmap over while_loop with per-lane trip counts → ghost events | `ported: tests/conformance/test_bcore_conformance.py::test_vmap_while_carry_ghost_suppression` — A1 mitigation: carry taps now emit exactly per-lane real steps (16, not 30 for the 3-lane example). Prim taps inside the body still ghost-fire (residual boundary — see `test_vmap_while_prim_tap_residual_ghost`). |
| `vmap_while_hardened.py` | hardened vmap+while handling: determinism + fabricated-value detection | `ported: tests/conformance/test_bcore_conformance.py::test_vmap_while_carry_no_fabricated_values`, `test_vmap_while_carry_ghost_suppression_with_select`, `test_vmap_while_alert_no_ghost_alerts` — fabricated values (counter > LIM) no longer reach on_step or alert. Ghost-drop precedes TapEvent construction so alert= never fires on ghost events. |

## proofs/bcore-review/arm-b/ — B-core attack arm B (verbose() composition)

| source script | what it attacks | disposition |
|---|---|---|
| `attack_carry_pytree.py` | pytree carry (dict, nested, list) through verbose() | `covered: tests/test_jaxtap.py::test_carry_leaves_contract` |
| `attack_cond.py` | scan inside cond / switch silently not instrumented (F1) | `covered: tests/test_jaxtap.py::test_scan_in_cond_f1`, `test_scan_in_switch_f1` |
| `attack_effects.py` | effectful ops (device_put, callback) inside scan body | `covered: tests/test_jaxtap.py::test_callback_totality` |
| `attack_jit_addressing.py` | top-level scan vs jit-nested scan address collision (F2) | `covered: tests/test_jaxtap.py::test_jit_addressing_uniqueness_f2` |
| `attack_jit_params.py` | dropped jit eqn params (donated_invars, device, etc.) | `documented-boundary`: same as `probe_params.py` above; correctness on CPU preserved |
| `attack_jit_params2.py` | jit params variant 2 | `documented-boundary`: same as above |
| `attack_misc.py` | miscellaneous ops (dynamic_slice, scatter, cond flag) | `covered: tests/test_jaxtap.py::test_ops_filtering` |
| `attack_nesting.py` | deeply nested scan/cond/remat compositions | `covered: tests/test_jaxtap.py::test_scan_in_cond_f1`, `test_scan_in_checkpoint_f1`, `test_jit_boundary_path_format` |
| `attack_sample_vmap.py` | vmap over scan under verbose() | `covered: tests/test_jaxtap.py::test_vmap_safety` |
| `recon_params.py` | reconstruct jit eqn params for re-wrap | `documented-boundary`: param reconstruction is internal; observable contract is bitwise correctness (covered by existing tests) |

## proofs/bcore-review/ays/ — B-core AYS probes

| source script | what it attacks | disposition |
|---|---|---|
| `ays_a1_baseline.py` | raw jax.debug.callback baseline under vmap+while (is A1 inherent?) | `documented-boundary`: confirms A1 ghost events are INHERENT to vmap+while+debug.callback. jaxtap now mitigates for CARRY TAPS by re-evaluating cond inside body_fn and filtering ghosts host-side. Raw baseline (no jaxtap) still over-fires — unchanged. |
| `ays_a2a3_round2.py` | round-2 AYS on direction + vmap semantic; A1 mitigation feasibility | `ported (partial)`: confirms the mitigation design (cond predicate evaluable per-lane on carry → active mask). A1 mitigation now implemented; see `test_vmap_while_carry_ghost_suppression`. |
| `ays_a2_fix_direction.py` | fix direction AYS probe | `documented-boundary`: confirms A1 boundary direction; mitigation implemented in fix/a1-cond-gating arc. |
| `ays_m2_sample_vmap.py` | M2 sample_every under vmap | `covered: tests/test_jaxtap.py::test_vmap_safety`; sample_every gating under vmap is `documented-boundary` for scalar prim taps (single-fire per batch step, see `proofs/m1d-ays/ays_m1d_r2.py`) |

## proofs/bcore-review/fix-review/ — B-core fix-review checks

| source script | what it attacks | disposition |
|---|---|---|
| `fixreview_bcore.py` (14 checks): | | |
| — cond forward bitwise | scan-in-cond forward identity post-fix | `covered: tests/test_jaxtap.py::test_scan_in_cond_f1` |
| — switch forward bitwise | scan-in-switch forward identity post-fix | `covered: tests/test_jaxtap.py::test_scan_in_switch_f1` |
| — remat forward bitwise | scan-in-checkpoint forward identity post-fix | `covered: tests/test_jaxtap.py::test_scan_in_checkpoint_f1` |
| — jit-siblings forward bitwise | top-level + jit-nested scan forward identity | `covered: tests/test_jaxtap.py::test_jit_addressing_uniqueness_f2` |
| — grad(cond) bitwise | grad through verbose(cond-with-scan) | `ported: tests/conformance/test_bcore_conformance.py::test_grad_cond_bitwise` |
| — grad(switch) bitwise | grad through verbose(switch-with-scan) | `ported: tests/conformance/test_bcore_conformance.py::test_grad_switch_bitwise` |
| — grad(remat) bitwise | grad through verbose(checkpoint-with-scan) | `covered: tests/test_jaxtap.py::test_checkpoint_grad_bitwise` |
| — grad(jit-siblings) bitwise | grad through verbose(jit-siblings) | `ported: tests/conformance/test_bcore_conformance.py::test_grad_jit_siblings_bitwise` |
| — vmap(cond) bitwise | vmap over verbose(cond-with-scan) | `ported: tests/conformance/test_bcore_conformance.py::test_vmap_cond_bitwise` |
| — F1 cond events (5) | events fire inside taken cond branch | `covered: tests/test_jaxtap.py::test_scan_in_cond_f1` |
| — F1 switch events (5) | events fire inside taken switch branch | `covered: tests/test_jaxtap.py::test_scan_in_switch_f1` |
| — F1 remat events (5) | events fire inside checkpoint | `covered: tests/test_jaxtap.py::test_scan_in_checkpoint_f1` |
| — F2 unique jit paths | top-level scan[0] vs jit[1]/scan[0] distinct | `covered: tests/test_jaxtap.py::test_jit_addressing_uniqueness_f2` |
| — grad(cond-in-scan) bitwise | grad through verbose(scan body with cond) | `ported: tests/conformance/test_bcore_conformance.py::test_grad_cond_in_scan_bitwise` |
| `fixreview_round2.py` (6 checks): | | |
| — max_depth=0 excludes jit-nested scan | max_depth across jit boundary (not just nested scan) | `ported: tests/conformance/test_bcore_conformance.py::test_max_depth_across_jit_boundary` |
| — where selects only jit-nested scan | where filter selecting across jit boundary | `ported: tests/conformance/test_bcore_conformance.py::test_where_across_jit_boundary` |
| — filtered run still bitwise | filtering does not alter result | `covered: tests/test_collectors.py::test_where_no_match`, `test_max_depth_0` |
| — grad2(cond) bitwise | grad-of-grad through verbose(cond) | `ported: tests/conformance/test_bcore_conformance.py::test_grad2_cond_bitwise` |
| — deep cond-in-jit-in-scan bitwise | cond inside jit inside scan — addressing + bitwise | `ported: tests/conformance/test_bcore_conformance.py::test_deep_cond_in_jit_in_scan_bitwise` |
| — deep nesting grad | grad through the deep nesting | `ported: tests/conformance/test_bcore_conformance.py::test_deep_nesting_grad_bitwise` |

## proofs/ashell-review/arm-s/ — A-shell attack arm S (equivalence + fidelity)

| source script | what it attacks | disposition |
|---|---|---|
| `a1_positional_and_paths.py` | positional reverse/unroll args; two sequential scans path divergence | `covered: tests/test_ashell.py::test_ashell_scan_positional_reverse`, `test_ashell_scan_positional_unroll`, `test_ashell_sequential_scan_paths` |
| `a2_fidelity_sweep.py` | int32/complex64/dict/None/empty-tuple/pytree-xs/while-pytree/PRNG carries; API keywords; error transparency | `ported: tests/conformance/test_ashell_conformance.py::test_ashell_int32_carry_bitwise`, `test_ashell_complex64_carry_bitwise`, `test_ashell_dict_carry_bitwise`, `test_ashell_prng_key_carry`, `test_ashell_all_keyword_form`, `test_ashell_pytree_xs_dict`, `test_ashell_error_transparency` |
| `a3_equivalence_and_transforms.py` | verbose() vs A-shell event equivalence for nested/cond-in-scan/while-in-scan/prim-tap; transforms around context (grad, vmap) | `ported: tests/conformance/test_ashell_conformance.py::test_ashell_equivalence_nested_scan`, `test_ashell_equivalence_cond_in_scan`, `test_ashell_equivalence_primitive_taps`; transforms: `documented-boundary` (grad-around-context fires fwd-only events; see `_ashell.py` module docstring on transform interaction) |
| `a4_transform_events.py` | counts/paths when transforms wrap context (grad, jit, vmap, hessian) | `documented-boundary`: A-shell intercepts at the scan boundary; grad-wrapped context sees fwd-only events; jit cache-hit routing works (covered: `tests/test_ashell.py::test_ashell_cache_hit_new_context`); transform divergence from verbose() is documented |
| `a5_disentangle.py` | separate jit-cache artifact from transform divergence (clear_caches before each run) | `covered: tests/test_ashell.py::test_ashell_cache_hit_new_context` |
| `a6_depth_and_collision.py` | five scans in python loop; value integrity under path collision; trace-time helper calling scan | `covered: tests/test_ashell.py::test_ashell_python_loop_five_scans` (5 scans); path collision: `documented-boundary` (two scans collide at scan[0] — same path, ambiguous consumer; documented in COVERAGE_MAP and `_ashell.py`) |

## proofs/ashell-review/arm-l/ — A-shell attack arm L (lifecycle)

| source script | what it attacks | disposition |
|---|---|---|
| `01_same_object_reenter.py` | reusing same context object nested → registry leak + permanent patch | `covered: tests/test_ashell.py::test_ashell_reenter_raises` |
| `02_manual_enter_no_gc_selfheal.py` | manual __enter__ without __exit__ — no GC self-heal in old design | `covered: tests/test_ashell.py::test_ashell_gc_selfheal` (current design HAS self-heal via __del__ finalizer) |
| `03_callback_thread_probe.py` | which thread runs the baked callback (_dynamic_router) | `documented-boundary`: callback thread ident = main-thread on CPU (synchronous dispatch); >=2-context attribution uses threading.get_ident(); documented in `_ashell.py::_dynamic_router` |
| `04_delegate_captures_bystander.py` | single-ctx delegate rule attributes bystander thread's scan | `documented-boundary`: L4 — single active context captures any thread's scan by design; see `_ashell.py::_select_ctx` docstring |
| `05_two_owned_and_bystander.py` | two owner threads each with own ctx → no cross-talk; bystander untapped | `ported: tests/conformance/test_ashell_conformance.py::test_ashell_two_owned_contexts_bystander` |
| `06_registry_race_stress.py` | concurrent enter/exit + lock-free registry read races (8 threads, 40 iters) | `ported: tests/conformance/test_ashell_conformance.py::test_ashell_registry_race_bounded` (bounded: 4 threads, 20 iters) |
| `07_nonlifo_exit.py` | non-LIFO exit order (A enters, B enters; A exits first / B exits first) | `ported: tests/conformance/test_ashell_conformance.py::test_ashell_nonlifo_exit_order` |
| `08_foreign_and_warnonce.py` | warn-once flag global/never-reset (part A) + foreign-before chain drop (part B) | Part A: `covered: tests/test_ashell.py::test_ashell_session_scoped_warnonce`; Part B: `documented-boundary` (foreign-before chain drop on foreign-over-us exit is known edge — `_ashell.py::_session_scan` reset policy) |
| `09_verbose_inside_context.py` | verbose() called inside active context — double instrumentation risk | `covered: tests/test_ashell.py::test_ashell_verbose_inside_context_no_double` |
| `10_exit_without_enter.py` | __exit__ without __enter__ poisons warn-once flag | `covered: tests/test_ashell.py::test_ashell_bogus_exit_does_not_poison_warnonce`, `test_ashell_double_exit_noop` |
| `11_generator_and_async.py` | generator suspended inside context (self-heal via GeneratorExit); async misrouting attempt | Part A: `documented-boundary` (generator finalization triggers GeneratorExit → normal __exit__ path; same GC mechanism as `test_ashell_gc_selfheal`); Part B: `documented-boundary` (async backend misrouting is THEORETIC on CPU — synchronous dispatch prevents it; documented in probe) |
| `12_blocker_confirm.py` | reuse of single context object nested → both primitives permanently leaked (BLOCKER) | `covered: tests/test_ashell.py::test_ashell_reenter_raises` (BLOCKER fixed — RuntimeError raised on re-entry, preventing leak) |
| `fixreview_l2_probe.py` | GC self-heal actually RESTORES the patch | `covered: tests/test_ashell.py::test_ashell_gc_selfheal` |
| `fixreview_r2_attacks.py` | emergency_restore, foreign-on-top warn, verbose-raise inside ctx, ctx counter restart | `covered: tests/test_ashell.py::test_ashell_emergency_restore`, `test_ashell_bogus_exit_does_not_poison_warnonce`, `test_ashell_verbose_inside_context_no_double`, `test_ashell_sequential_scan_paths` |

## proofs/ays-m0/ — M0 AYS probes (custom_jvp, vmap, grad seams)

| source script | what it attacks | disposition |
|---|---|---|
| `ays_seams_vmap_grad_customjvp.py` | custom_jvp in scan body bitwise; grad through scan; vmap safety | `covered: tests/test_jaxtap.py::test_custom_jvp_in_scan`, `test_grad_through_transform`, `test_vmap_safety` |
| `rootcause_customjvp_subfuns.py` | root cause of custom_jvp opaque bind (subfuns pathway) | `covered: tests/test_jaxtap.py::test_custom_jvp_in_scan`, `test_custom_jvp_sentinel_rule` |
| `round2_customvjp_sentinel_rule.py` | custom_vjp sentinel rule survives verbose() | `covered: tests/test_jaxtap.py::test_custom_vjp_through_transform` |

## proofs/m1a-ays/ — M1a primitive-tap AYS probes

| source script | what it attacks | disposition |
|---|---|---|
| `ays_m1a.py` (round 1): | | |
| — live step varies through jit (cholesky) | step counter not constant-folded | `covered: tests/test_jaxtap.py::test_prim_tap_jit_hidden_cholesky` |
| — vmap prim-tap fires LANES×N | vmap × primitive tap event count | `covered: tests/test_jaxtap.py::test_vmap_safety` (scan events); vmap prim-tap: `ported: tests/conformance/test_bcore_conformance.py::test_vmap_prim_tap_fires_lanes_times_n` |
| — cond prim-tap fires on taken-branch steps only | prim tap inside cond-in-scan | `ported: tests/conformance/test_bcore_conformance.py::test_cond_prim_tap_taken_branch_only` |
| — while prim-tap fires live steps 0..N-1 | prim tap inside while loop | `covered: tests/test_jaxtap.py::test_prim_tap_basic_in_scan` (scan); while variant: `ported: tests/conformance/test_bcore_conformance.py::test_while_prim_tap_live_steps` |
| — reverse=True step semantics | step=iteration idx, first iter sees last xs | `covered: tests/test_jaxtap.py::test_scan_taps` (step ordering implicit) |
| `ays_m1a_r2.py` (round 2): | | |
| — grad THROUGH tapped cholesky | grad bitwise with prim tap active | `covered: tests/test_jaxtap.py::test_prim_tap_grad_bitwise` |
| — prim-tap totality under -W error | raising on_step doesn't break computation | `covered: tests/test_jaxtap.py::test_prim_tap_raising_on_step` |
| — sample_every=2 gates loop taps | loop carry taps gated by sample_every | `covered: tests/test_collectors.py::test_sample_every_k2` |
| — prim taps ungated (documented) — PROBE FAILS | M1a expected prim taps ungated; M1d FIX1 intentionally gates them | `documented-boundary`: `sample_every` NOW gates prim taps inside loops (M1d FIX 1); covered: `tests/test_m1d_api_polish.py::TestFix1SampleEveryGatesPrimTaps`; probe pre-dates M1d |

## proofs/m1b-ays/ — M1b A-shell AYS probes

| source script | what it attacks | disposition |
|---|---|---|
| `ays_m1b.py` (round 1): | | |
| — with-form bitwise + event count | basic A-shell contract | `covered: tests/test_ashell.py::test_ashell_basic_scan` |
| — phantom emission (jit inside ctx, call after exit) | callbacks baked into jit artifact after exit | `covered: tests/test_ashell.py::test_ashell_no_phantom_after_exit` |
| — jit-wrapped call inside ctx | interception during jit tracing | `covered: tests/test_ashell.py::test_ashell_cache_hit_new_context` |
| — grad / vmap through with-form bitwise | grad+vmap through A-shell context | `covered: tests/test_ashell.py::test_ashell_reentrant_contexts` (nesting); grad/vmap: `documented-boundary` (fwd-only events for grad-around-context) |
| `ays_m1b_r2.py` (round 2): | | |
| — cache-hit in ctx B routes to B (recorder + on_step) | dynamic router routing on cache hit | `covered: tests/test_ashell.py::test_ashell_cache_hit_new_context` |
| — worker-thread delegation on cache-hit artifact | 1-active-ctx delegation with cached trace | `covered: tests/test_ashell.py::test_ashell_thread_delegation` |
| — nested contexts + cache hit → inner wins | inner-wins routing | `covered: tests/test_ashell.py::test_ashell_reentrant_contexts` |
| — no-context cache-hit dropped everywhere | drop when no active context | `covered: tests/test_ashell.py::test_ashell_no_phantom_after_exit` |

## proofs/m1d-ays/ — M1d API-polish AYS probes

| source script | what it attacks | disposition |
|---|---|---|
| `ays_m1d.py` (round 1): | | |
| — vmap se-gate: prim taps = lanes×(N/se) — PROBE FAILS | vmap + sample_every + prim tap count | `documented-boundary`: scalar (non-batched) prim tap fires ONCE per sampled step for all lanes (not per-lane); see `proofs/m1d-ays/ays_m1d_r2.py` diagnosis; covered by M1d FIX1 gating semantics: `tests/test_m1d_api_polish.py::TestFix1SampleEveryGatesPrimTaps` |
| — descend-always bitwise + grad on cond/remat with all-filtered | descend-always semantics under all-filtered where + grad | `covered: tests/test_jaxtap.py::test_scan_in_cond_f1` (cond bitwise); `test_checkpoint_grad_bitwise` (remat grad) |
| — se-gating shows up in wall time (≥10×) | sample_every performance improvement | `documented-boundary`: timing test; N/A as a deterministic conformance assertion |
| — cond-in-scan prim tap gated (se=5) | sample_every gates prim tap in cond branch | `ported: tests/conformance/test_bcore_conformance.py::test_cond_in_scan_prim_tap_gated` |
| — while prim tap gated (25 iters, se=10 → 3) | sample_every gates prim tap in while loop | `ported: tests/conformance/test_bcore_conformance.py::test_while_prim_tap_gated` |
| `ays_m1d_r2.py` | batched tap value: per-lane firing vs single-fire for unbatched arg | `documented-boundary`: confirms scalar prim tap is single-fire under vmap; not a defect |

## proofs/primitive-tap-scoping/ — M1a primitive tap scoping prototype

| source script | what it attacks | disposition |
|---|---|---|
| `prototype_primitive_tap.py` | initial prototype of primitive tap scoping | `covered: tests/test_jaxtap.py::test_prim_tap_basic_in_scan`, `test_prim_tap_outside_loop`, `test_prim_tap_multiple_sites_same_level` |

## proofs/semantic-progress/ — M0 semantic progress probe

| source script | what it attacks | disposition |
|---|---|---|
| `semantic_progress.py` | M0 semantic equivalence (verbose vs bare call) | `covered: tests/test_jaxtap.py::test_identity_bitwise`, `test_scan_taps` |

---

## Disposition tallies

| disposition | count |
|---|---|
| covered | 60 |
| ported | 35 |
| documented-boundary | 26 |
| N/A | 57 |
| **total** | **178** |

*(Counts are per-check for multi-check scripts; per-file for single-scenario scripts.
Ported entries reference tests in `test_bcore_conformance.py` (25 tests, +5 from A1 arc) and
`test_ashell_conformance.py` (14 tests); both verified 191/191 full-suite green.
Two proof script failures are documented-boundary, not regressions: `ays_m1a_r2.py`
"prim taps ungated" predates M1d FIX1; `ays_m1d.py` "vmap se-gate per-lane count"
is inherent scalar-tap duality confirmed by `ays_m1d_r2.py`.
A1 arc (fix/a1-cond-gating): `vmap_while.py` and `vmap_while_hardened.py` promoted from
`documented-boundary` to `ported`; 3 AYS probes updated; prim-tap residual ghost boundary
documented in `test_vmap_while_prim_tap_residual_ghost` and COVERAGE_MAP note below.
Residual boundary: primitive taps inside vmapped while bodies still ghost-fire — carry-taps-only
scope for A1 mitigation; extending to prim taps is a future arc.)*
