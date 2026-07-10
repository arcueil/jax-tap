#!/bin/bash
# jax-tap RELEASE-CANDIDATE GPU validation (run on colossus, BEFORE tagging).
#
# Tests the SOURCE TREE at a git ref (default: current main) on real CUDA, so
# GPU-only issues are caught PRE-tag — e.g. jax.debug.callback ordering
# differences that the CPU-only GitHub CI cannot see. Complements
# tools/gpu_pypi_validate.sh, which validates the PUBLISHED wheel post-publish.
#
# One-shot, autonomous: writes RESULTS.txt and a DONE_RC_GPU sentinel.
# Usage (on the GPU box):  bash ~/gpu_rc_validate.sh [git-ref]   # default: main
set -uo pipefail

# tmux one-shot shells are non-login: uv lives outside the default PATH.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
command -v uv >/dev/null || { echo "FATAL: uv not found on PATH"; exit 1; }

REF="${1:-main}"
R="$HOME/arcueil/gpu-rc-validate"
OUT="$R/RESULTS.txt"
REPO_URL="https://github.com/arcueil/jax-tap"
mkdir -p "$R" && cd "$R"
: > "$OUT"
echo "=== jax-tap RC GPU validation | ref $REF | $(date -u) ===" >> "$OUT"

rm -rf env repo
git clone -q "$REPO_URL" repo >> "$OUT" 2>&1
git -C repo checkout -q "$REF" >> "$OUT" 2>&1
echo "RC commit: $(git -C repo rev-parse --short HEAD)" >> "$OUT"

uv venv -q env -p 3.13 >> "$OUT" 2>&1
# Editable install of the RC SOURCE (not PyPI). repo/.git is present so
# hatch-vcs can derive the version.
uv pip install -q --python env/bin/python -e "./repo[pandas]" >> "$OUT" 2>&1
uv pip install -q --python env/bin/python "jax[cuda13]" pytest blackjax >> "$OUT" 2>&1

# Hard assert: CUDA devices or bust (a CPU fallback must NOT look green).
env/bin/python - >> "$OUT" 2>&1 <<'PY'
import jax
import importlib.metadata as md
print("jax-tap", md.version("jax-tap"), "| jax", jax.__version__, "|", jax.devices())
assert "cuda" in str(jax.devices()).lower(), "NOT ON CUDA — validation void"
PY
if [ $? -ne 0 ]; then echo "FAILED_CUDA_ASSERT" >> "$OUT"; exit 1; fi

cd repo

echo "=== SUITE (GPU0, RC source) ===" >> "$OUT"
CUDA_VISIBLE_DEVICES=0 ../env/bin/python -m pytest tests/ -q >> "$OUT" 2>&1

echo "=== DEMOS (GPU0) ===" >> "$OUT"
for d in demo/*.py; do
  echo "--- $d" >> "$OUT"
  CUDA_VISIBLE_DEVICES=0 timeout 600 ../env/bin/python "$d" 2>&1 | tail -3 >> "$OUT"
done

echo "=== BENCH (GPU0, RC source) ===" >> "$OUT"
# report-only on GPU: machinery threshold is CPU-calibrated; GPU numbers are uncharacterized
CUDA_VISIBLE_DEVICES=0 ../env/bin/python bench/nightly_gate.py >> "$OUT" 2>&1 || true
CUDA_VISIBLE_DEVICES=0 ../env/bin/python bench/progress_bar.py >> "$OUT" 2>&1

echo "DONE_RC_GPU" >> "$OUT"
