#!/bin/bash
# jax-tap GPU validation of the PUBLISHED PyPI artifact (run on colossus).
# One-shot, autonomous: writes RESULTS.txt and a DONE_PYPI_GPU sentinel.
# Installs jax-tap[pandas] FROM PYPI (never the local tree); clones the repo
# at the release tag ONLY for tests/ and demo/ (src-layout means the clone's
# source is not importable — pytest exercises the installed wheel).
set -uo pipefail

# tmux one-shot shells are non-login: uv lives outside default PATH on colossus.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
command -v uv >/dev/null || { echo "FATAL: uv not found on PATH" ; exit 1; }

TAG="${1:-v0.1.0}"
R="$HOME/arcueil/gpu-pypi-validate"
OUT="$R/RESULTS.txt"
mkdir -p "$R" && cd "$R"
: > "$OUT"
echo "=== jax-tap PyPI GPU validation | tag $TAG | $(date -u) ===" >> "$OUT"

rm -rf env repo
uv venv -q env -p 3.13 >> "$OUT" 2>&1
uv pip install -q --python env/bin/python \
  "jax[cuda13]" "jax-tap[pandas]" pytest blackjax >> "$OUT" 2>&1

# Hard assert: CUDA devices or bust (a CPU fallback must NOT look green).
env/bin/python - >> "$OUT" 2>&1 <<'PY'
import jax, importlib.metadata as md
print("jax-tap", md.version("jax-tap"), "| jax", jax.__version__, "|", jax.devices())
assert "cuda" in str(jax.devices()).lower(), "NOT ON CUDA — validation void"
PY
if [ $? -ne 0 ]; then echo "FAILED_CUDA_ASSERT" >> "$OUT"; exit 1; fi

git clone -q --depth 1 --branch "$TAG" https://github.com/arcueil/jax-tap repo >> "$OUT" 2>&1
cd repo

echo "=== SUITE (GPU0, wheel from PyPI) ===" >> "$OUT"
CUDA_VISIBLE_DEVICES=0 ../env/bin/python -m pytest tests/ -q >> "$OUT" 2>&1

echo "=== DEMOS (GPU0) ===" >> "$OUT"
for d in demo/*.py; do
  echo "--- $d" >> "$OUT"
  CUDA_VISIBLE_DEVICES=0 timeout 600 ../env/bin/python "$d" 2>&1 | tail -3 >> "$OUT"
done

echo "=== BENCH (GPU0, wheel from PyPI) ===" >> "$OUT"
# report-only on GPU: machinery threshold is CPU-calibrated; GPU numbers are uncharacterized
CUDA_VISIBLE_DEVICES=0 ../env/bin/python bench/nightly_gate.py >> "$OUT" 2>&1 || true
CUDA_VISIBLE_DEVICES=0 ../env/bin/python bench/progress_bar.py >> "$OUT" 2>&1

echo "DONE_PYPI_GPU" >> "$OUT"
