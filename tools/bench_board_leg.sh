#!/bin/sh
# tools/bench_board_leg.sh -- the leg BODY that runs tools/bench_board.py on a
# rented Linux box (NVIDIA or AMD). Runs ON THE BOX from the shipped tree
# (git archive, cwd /root/mojolearn) after the runner staged taxi and
# Istella-S (and istella_rank.npz, the gbdt-rank-* lanes' query ids) from R2
# into /root/datasets/gbm-bench (tools/stage_from_r2.sh's default keys). Everything it writes lands under /root/gemm_leg_out, which
# the runner fetches home. docs/BENCH_BOARD.md has the exact leg commands.
#
# Settings arrive as MOJOLEARN_* variables (the runners pass NAME=value words
# with no spaces, so lists are commas):
#   MOJOLEARN_BOARD_VERSION   mojolearn version to pip install (required)
#   MOJOLEARN_BOARD_ROWS      a row cap for a smoke run (default: full size)
#   MOJOLEARN_BOARD_LANES     comma list of lanes (default: every lane)
#   MOJOLEARN_BOARD_FAMILIES  trees,classical,classical2,neural (default all four)
#   MOJOLEARN_BOARD_NEURAL_SHAPE  full (default) or small (a neural smoke)
#   MOJOLEARN_BOARD_DATASETS  taxi,istella (default both)
#   MOJOLEARN_BOARD_ROUNDS    timed rounds (default 5)
#   MOJOLEARN_BOARD_NO_INFER  1: time training only (no inference cells; default: timed)
#   MOJOLEARN_BOARD_NO_CPU_ARM 1: no ours-cpu arm (default: our CPU tier races too)
#   MOJOLEARN_BOARD_OUT       result dir (default /root/gemm_leg_out/bench-board)
#   MOJOLEARN_BOARD_CACHE     venv, wheel and classical blocks, NOT fetched (default /root/board-cache)
#
# THE INTERPRETER. AMD's pinned torch ROCm wheels are cp312, so on AMD the
# venv is Python 3.12: the image's python3 when it is 3.12 with venv, else a
# uv-managed 3.12 (the recipe of tools/classical_two_datasets_leg.sh). NVIDIA
# uses the image's python3 in a CLEAN venv (no --system-site-packages: the
# image's nvidia/__init__.py shadowed the venv's CUDA libraries and cuML could
# not load libcudf, 2026-09-29); the board installs torch==2.13.0+cu129 there.
# POSIX sh.
set -u
: "${MOJOLEARN_BOARD_VERSION:?set MOJOLEARN_BOARD_VERSION through the leg env words}"
OUT="${MOJOLEARN_BOARD_OUT:-/root/gemm_leg_out/bench-board}"
mkdir -p "$OUT"
PY=python3
SSP=
if [ -e /dev/kfd ] && ! command -v nvidia-smi > /dev/null 2>&1; then
    SSP=--system-site-packages
    if python3 -c 'import sys, venv, ensurepip; sys.exit(0 if sys.version_info[:2] == (3, 12) else 3)' > /dev/null 2>&1; then
        PY=python3
    else
        if [ ! -x /root/.local/uv/uv ]; then
            curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL=/root/.local/uv sh > "$OUT/uv_install.log" 2>&1
        fi
        /root/.local/uv/uv python install 3.12 >> "$OUT/uv_install.log" 2>&1
        PY=$(/root/.local/uv/uv python find 3.12)
    fi
elif ! python3 -c 'import ensurepip' > /dev/null 2>&1; then
    # Ubuntu images ship python3 without venv's ensurepip
    (apt-get update -qq && apt-get install -y -qq python3-venv) > "$OUT/apt_venv.log" 2>&1 || true
fi
set -- --mojolearn-version "$MOJOLEARN_BOARD_VERSION" --out "$OUT" \
    --base-python "$PY" $SSP --cache "${MOJOLEARN_BOARD_CACHE:-/root/board-cache}" \
    --data-root "${GBM_BENCH_DATA:-/root/datasets/gbm-bench}"
[ -n "${MOJOLEARN_BOARD_ROWS:-}" ] && set -- "$@" --rows "$MOJOLEARN_BOARD_ROWS"
[ -n "${MOJOLEARN_BOARD_LANES:-}" ] && set -- "$@" --lanes "$MOJOLEARN_BOARD_LANES"
[ -n "${MOJOLEARN_BOARD_FAMILIES:-}" ] && set -- "$@" --families "$MOJOLEARN_BOARD_FAMILIES"
[ -n "${MOJOLEARN_BOARD_DATASETS:-}" ] && set -- "$@" --datasets "$MOJOLEARN_BOARD_DATASETS"
[ -n "${MOJOLEARN_BOARD_ROUNDS:-}" ] && set -- "$@" --rounds "$MOJOLEARN_BOARD_ROUNDS"
[ -n "${MOJOLEARN_BOARD_NEURAL_SHAPE:-}" ] && set -- "$@" --neural-shape "$MOJOLEARN_BOARD_NEURAL_SHAPE"
[ "${MOJOLEARN_BOARD_NO_INFER:-0}" = 1 ] && set -- "$@" --no-infer
[ "${MOJOLEARN_BOARD_NO_CPU_ARM:-0}" = 1 ] && set -- "$@" --no-cpu-arm
echo "bench_board_leg: $PY tools/bench_board.py $*"
"$PY" tools/bench_board.py --dry-run "$@" > "$OUT/plan.txt" 2>&1
"$PY" tools/bench_board.py "$@"
