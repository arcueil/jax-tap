# case_ledger/ — jaxtap re-run against our own bugs

One runnable `.py` per row of the jax-tap case ledger: each reproduces the
*essence* of a real bug we hit (in pure JAX, standalone — no blackjax), shows
the silent symptom a user actually saw, then shows jaxtap localizing it. Every
file self-reports `PASS`/`FAIL` and runs on the base install:

```
uv run python case_ledger/NN_name.py
```

## Honesty about tap classes

jaxtap v1 ships the **control-flow / carry tap** class (`tap.verbose` +
`select` at scan/while seams). The ledger spans more classes; each file states
which it uses and, where the ideal tap class is not yet built, says so.

| # | Bug (cost) | Ideal tap class | v1 status |
|---|-----------|-----------------|-----------|
| 01 | float32 Cholesky trap — silent NaN, DA freezes, *looks converged* (multi-day) | primitive NaN-watch | ✅ carry-tap + `select(isfinite)` localizes first-bad step |
| 02 | #949 low-rank wrong-sign — metric stuck ≈ identity (7 weeks parked) | carry-tap on adaptation state | ✅ carry-tap on metric eigenvalues per window |
| 03 | L-BFGS maxiter=30 silent curvature inflation (4 levels deep) | inner while-loop exit tap | ✅ carry-tap on `iters==maxiter` + grad-norm |
| 04 | IMM-diagonal — dense ran diagonal silently (5-PR, 3 days) | **trace-time shape tap** | ⚠️ approximated by a runtime ndim tap; zero-cost trace-time tap = roadmap |
| 05 | Multinomial traj_wt breaks DA — acceptance secretly bimodal | carry-tap on DA state | ✅ carry-tap → acceptance histogram bimodality |
| 06 | Treedepth saturation blind spot (found months later) | per-draw event tap | ✅ carry-tap on per-draw treedepth tripwire |
| 07 | Async-dispatch "83× compile blowup" — 427s hidden in "19s tracing" | **jit trace/execute event taps** | ⚠️ roadmap (jit-event class not in v1); file demos the concept + states the gap |
| 08 | probdiffeq `hypot(0,0)` VJP NaN — NaN only in the backward pass | backward-pass tap | 🚧 HONEST BOUNDARY: a forward tap does NOT catch it; the file proves the limit |

Legend: ✅ demonstrated with shipped v1 capability · ⚠️ approximated, ideal class
is roadmap · 🚧 documented boundary (out of scope by design).

## Why this directory exists

It is the empirical answer to "does the zero-code-change tap promise pay off on
real bugs?" — 5 clean wins, 2 honest roadmap markers, 1 honest boundary. The
files double as the acceptance corpus for the tap classes as they ship.
