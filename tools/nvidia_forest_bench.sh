#!/usr/bin/env bash
# The RF and ET pairs, us vs LightGBM, on a rented NVIDIA box.
#
#   tools/nvidia_forest_bench.sh <user@host> <dataset> [ntrees]
#
# WHAT CHANGES ON NVIDIA, AND IT IS THE WHOLE FRAMING
#
# On Apple, LightGBM ships no GPU arm -- its CUDA and OpenCL learners do
# not build for Metal -- so the M4 table's `pairs` suite runs our GPU
# against their CPU because that is their strongest legal arm there. On
# NVIDIA that is false: LightGBM ships a CUDA learner (`device_type=
# "cuda"`), and a portability claim that quietly benchmarks against the
# weaker arm on the vendor's own hardware is not a portability claim. So
# this script runs SIX arms interleaved in one process:
#
#   mojolearn-rf-gpu, lgbm-rf-cpu, lgbm-rf-gpu,
#   mojolearn-et-gpu, lgbm-et-cpu, lgbm-et-gpu
#
# Expect to lose the lgbm-*-gpu columns, possibly by a lot; recording how
# much is the point.
#
# WHAT HAS NEVER HAPPENED BEFORE THIS SCRIPT RUNS
#
# No line of ensemble/ or extratrees/ has ever been compiled for CUDA.
# The one-source rule (VENDOR_LIBS.md, ALWAYS GPU-agnostic) says the same
# Mojo should build through MAX's CUDA target; this script is the first
# time that claim meets a compiler. Treat the first run as a BUILD, not a
# benchmark. If it produces numbers on the first attempt, be suspicious
# rather than pleased.
#
# LightGBM's pip wheel has NO CUDA support; the script builds it from
# source with USE_CUDA=ON, which needs the CUDA toolkit and cmake on the
# box (true of every stock RunPod/Lambda CUDA image).
set -euo pipefail

HOST="${1:?usage: tools/nvidia_forest_bench.sh <user@host> <dataset> [ntrees]}"
DATASET="${2:?dataset required (year, covtype, higgs, ...)}"
NTREES="${3:-100}"

REMOTE_DIR="~/mojolearn"
ALGOS="mojolearn-rf-gpu,lgbm-rf-cpu,lgbm-rf-gpu,mojolearn-et-gpu,lgbm-et-cpu,lgbm-et-gpu"

echo "==> target $HOST, dataset $DATASET, ntrees $NTREES"
echo "==> syncing source only (no build products, no datasets, no worktrees)"
rsync -az --delete \
  --exclude '.git' --exclude '.claude' --exclude '*.trace' \
  --exclude '__pycache__' --exclude 'build' --exclude '.pixi' \
  --exclude 'bench/external/.gbm-bench' \
  ./ "$HOST:$REMOTE_DIR/"

ssh "$HOST" bash -s "$DATASET" "$NTREES" "$ALGOS" <<'REMOTE'
set -euo pipefail
DATASET="$1"; NTREES="$2"; ALGOS="$3"
cd ~/mojolearn

command -v pixi >/dev/null || {
  echo "==> installing pixi"
  curl -fsSL https://pixi.sh/install.sh | bash
}
export PATH="$HOME/.pixi/bin:$PATH"

echo "==> what this box is"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "==> gbmbench python env"
pixi install -e gbmbench

echo "==> LightGBM: replacing the CPU-only wheel with a CUDA build"
if ! pixi run -e gbmbench python -c \
    "import lightgbm; lightgbm.LGBMRegressor(device_type='cuda')" \
    >/dev/null 2>&1; then
  pixi run -e gbmbench python -m pip install --no-binary lightgbm \
    --config-settings=cmake.define.USE_CUDA=ON lightgbm \
    || echo "!!!! CUDA LightGBM build failed; lgbm-*-gpu arms will REFUSE"
fi

echo "==> building the python bindings for CUDA (the first-ever CUDA build)"
bash bindings/build_estimators.sh
bash bindings/build_rf.sh

echo "==> running the pairs"
GBM_BENCH_DATA="$HOME/datasets/gbm-bench" \
  pixi run -e gbmbench bash bench/external/run_gbm_bench.sh \
    "$DATASET" "$NTREES" "$ALGOS"
REMOTE

echo
echo "==> done. Pull bench/results/ from the box for the record:"
echo "    rsync -az '$HOST:$REMOTE_DIR/bench/results/' bench/results/nvidia/"
