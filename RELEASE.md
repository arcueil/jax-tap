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
3. **Pre-tag GPU gate** — on a CUDA host, run the release candidate BEFORE
   tagging:

   ```bash
   bash tools/gpu_rc_validate.sh main   # writes ~/arcueil/gpu-rc-validate/RESULTS.txt
   ```

   Must be green: full test suite + the multi-device tests on two GPUs + all
   demos + bench, ending with the `DONE_RC_GPU` sentinel. This catches GPU-only failures (e.g.
   `jax.debug.callback(ordered=False)` cross-callback ordering, which differs
   CPU vs GPU — see the 0.3.1 test fix in `CHANGELOG.md`). **Do not tag until
   this is green.**
4. **Tag + GitHub release `vX.Y.Z`** (target `main`; use `gh release create` or
   the UI). Publishing the release triggers PyPI via trusted publishing
   (`.github/workflows/publish.yml`): its `actions/checkout` is pinned to the
   tag ref with `fetch-depth: 0`, so hatch-vcs sees the tag and stamps the clean
   version (not a `.devN` suffix).
5. **Post-publish confirmation** — validate the PUBLISHED wheel on the same
   CUDA host:

   ```bash
   bash tools/gpu_pypi_validate.sh vX.Y.Z   # ends with DONE_PYPI_GPU
   ```

   `tools/gpu_pypi_validate.sh` installs `jax-tap[pandas]==X.Y.Z` from PyPI on
   GPU and runs suite + demos + bench — confirming exactly what users install.
6. **Verify + close out** — `pip install jax-tap==X.Y.Z` in a clean env; close
   the shipped GitHub issues.

## GPU validation notes

- Both scripts are one-shot and autonomous: they write `RESULTS.txt` plus a
  `DONE_*` sentinel, so they can run detached (e.g. in tmux) and be polled.
- After any `uv sync`/`uv venv`, `jax[cuda13]` must be (re)installed; the
  scripts **assert `'cuda' in jax.devices()`** — a silent CPU fallback would
  void the run.
- The MULTI-GPU stage runs `tests/test_multidevice.py` with
  `CUDA_VISIBLE_DEVICES=0,1` and `JAXTAP_REQUIRE_MULTIDEVICE=1`, so it errors
  (rather than skips) on a host with fewer than two GPUs.
- Runtime is ~15–20 min (env install + suite + 10 demos + bench).
