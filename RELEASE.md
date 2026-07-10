# Releasing jax-tap

Versioning is **tag-driven** (hatch-vcs) — the git tag *is* the version. There
is no manual version bump; do not edit a `version =` field.

## Checklist

1. **Everything merged to `main`; GitHub CI green** — the 3.11–3.13 matrix plus
   `ruff check` / `ruff format --check`. Note CI is **CPU-only** (GitHub has no
   GPU runners), so it cannot catch GPU-specific behavior — step 3 exists for
   exactly that.
2. **CHANGELOG** — rename the top `## Unreleased — X.Y.Z` heading to
   `## X.Y.Z (YYYY-MM-DD)`. Commit + push.
3. **Pre-tag GPU gate (colossus)** — run the release candidate on real CUDA
   BEFORE tagging:

   ```bash
   rsync tools/gpu_rc_validate.sh gpu:gpu_rc_validate.sh
   ssh gpu 'tmux new-session -d -s rcval "bash ~/gpu_rc_validate.sh main"'
   # poll ~/arcueil/gpu-rc-validate/RESULTS.txt for the DONE_RC_GPU sentinel
   ```

   Must be green: full test suite + all demos + bench on GPU. This catches
   GPU-only failures (e.g. `jax.debug.callback(ordered=False)` cross-callback
   ordering, which differs CPU vs GPU — see
   `worklog/lessons/jax/ordered-false-no-cross-callback-order.md`). **Do not tag
   until this is green.**
4. **Tag + GitHub release `vX.Y.Z`** (target `main`; use `gh release create` or
   the UI). Publishing the release triggers PyPI via trusted publishing
   (`.github/workflows/publish.yml`): its `actions/checkout` is pinned to the
   tag ref with `fetch-depth: 0`, so hatch-vcs sees the tag and stamps the clean
   version (not a `.devN` suffix).
5. **Post-publish confirmation (colossus)** — validate the PUBLISHED wheel:

   ```bash
   ssh gpu 'tmux new-session -d -s pypival "bash ~/gpu_pypi_validate.sh vX.Y.Z"'
   ```

   `tools/gpu_pypi_validate.sh` installs `jax-tap[pandas]==X.Y.Z` from PyPI on
   GPU and runs suite + demos + bench — confirming exactly what users install.
6. **Verify + close out** — `pip install jax-tap==X.Y.Z` in a clean env; close
   the shipped GitHub issues (sign posts `— 🤖 Arcueil AI TL`); worklog closeout.

## Colossus (GPU box) notes

- `ssh gpu`; slow handshake (~60–120 s) — always use `ConnectTimeout=240` and
  widen command timeouts to 400 s+.
- Long runs go in a **tmux one-shot** writing `RESULTS.txt` + a `DONE_*`
  sentinel; poll the sentinel rather than holding an interactive session.
- After any `uv sync`/`uv venv`, `jax[cuda13]` must be (re)installed and the
  scripts **assert `'cuda' in jax.devices()`** — a silent CPU fallback would
  void the run.
- Runtime is ~15–20 min (env install + suite + 10 demos + bench). SSH slowness
  affects launch/poll latency, not the on-box compute.
