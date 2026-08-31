#!/usr/bin/env bash
# The interleaved CatBoost comparison, on a rented NVIDIA box.
#
#   tools/nvidia_bench.sh <user@host> [rows] [feats]
#
# WHY THIS IS SEPARATE FROM `tools/remote_gpu.sh`
#
# `remote_gpu.sh` refuses to benchmark, and its reason is right for what it
# runs: it runs `probe_main`, a wall-clock-free correctness suite, and a
# rented box is shared, throttled and unknown, so any number it produced
# would be uncomparable to anything.
#
# The INTERLEAVED harness is the exception, and it is the exception for the
# same reason it exists at all. Both arms run inside ONE process on the SAME
# data and ALTERNATE per rep. A box that throttles mid-run throttles both
# arms; the ratio survives what an absolute number does not. That is the
# only comparison format this repository quotes, and it is the format that a
# rented box cannot invalidate.
#
# WHAT CHANGES ON NVIDIA, AND IT IS THE WHOLE FRAMING
#
# On Apple, CatBoost has no GPU arm -- `task_type="GPU"` raises -- so "our
# GPU against their CPU" was not a choice of opponent, it was the only
# opponent. On NVIDIA that is false. CatBoost ships a CUDA learner and it is
# what any NVIDIA user would run. So this script times BOTH of their arms:
#
#   cpu  -- the same comparison the Apple table records, for continuity
#   gpu  -- their CUDA learner, which is the honest NVIDIA opponent
#
# We expect to lose the gpu column, probably by a lot. Recording how much is
# the point. A portability claim that quietly benchmarks against the weaker
# arm on the vendor's own hardware is not a portability claim.
#
# WHAT HAS NEVER HAPPENED BEFORE THIS SCRIPT RUNS
#
# No line of this repository has ever been compiled for CUDA. `TARGET_COLUMN`
# has been `COLUMN_APPLE` for every build ever made here, the NVIDIA rows of
# the kernel matrix are ARITHMETIC rather than measurement, and the float
# atomic flush branch is unreachable on Apple and is the path NVIDIA takes.
# Treat the first run as a BUILD, not a benchmark. If it produces numbers on
# the first attempt, be suspicious rather than pleased.
set -euo pipefail

HOST="${1:?usage: tools/nvidia_bench.sh <user@host> [rows] [feats]}"
ROWS="${2:-800000}"
FEATS="${3:-100}"

REMOTE_DIR="~/mojolearn"

echo "==> target $HOST, ${ROWS} x ${FEATS} synthetic"

echo "==> syncing source only (no build products, no datasets, no worktrees)"
rsync -az --delete \
  --exclude '.git' --exclude '.claude' --exclude '*.trace' \
  --exclude '__pycache__' --exclude 'build' --exclude '.pixi' \
  ./ "$HOST:$REMOTE_DIR/"

ssh "$HOST" bash -s "$ROWS" "$FEATS" <<'REMOTE'
set -euo pipefail
ROWS="$1"; FEATS="$2"
cd ~/mojolearn

command -v pixi >/dev/null || {
  echo "==> installing pixi"
  curl -fsSL https://pixi.sh/install.sh | bash
  export PATH="$HOME/.pixi/bin:$PATH"
}
export PATH="$HOME/.pixi/bin:$PATH"

# TARGET_COLUMN is comptime, so the vendor is a BUILD and not a flag. That is
# the honest shape of the constraint: a threadgroup size cannot follow a
# runtime device query.
sed -i.bak "s/^comptime TARGET_COLUMN = .*/comptime TARGET_COLUMN = COLUMN_NVIDIA/" \
  checks/kernel_matrix.mojo
grep -n "^comptime TARGET_COLUMN" checks/kernel_matrix.mojo

echo "==> what this box is"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
pixi run mojo build --print-effective-target || true

SCRATCH="$(mktemp -d)"

echo "==> preparing the pool (CatBoost's own quantizer, outside both timers)"
pixi run -e bench python tools/interleaved_prep.py "$SCRATCH" synth "$ROWS" "$FEATS"

for ARM in CPU GPU; do
  echo
  echo "############ their arm: $ARM ############"
  # Their GPU learner refuses some settings their CPU learner accepts, and a
  # refusal must be visible as a refusal rather than as a missing row.
  MOJOLEARN_CATBOOST_TASK_TYPE="$ARM" \
    pixi run -e bench mojo run -I . \
      bench/interleaved/catboost_interleaved.mojo "$SCRATCH" synth "$ROWS" \
    || echo "!!!! their $ARM arm did not complete; see the error above"
done

rm -rf "$SCRATCH"
REMOTE

echo
echo "==> done. Ratios only. An absolute ms/tree from a rented box compares"
echo "    to nothing, including to the M4 table."
