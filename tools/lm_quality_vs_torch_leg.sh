#!/bin/sh
# tools/lm_quality_vs_torch_leg.sh -- the body of one rented GPU leg that asks
# whether the PUBLISHED mojolearn wheel's IDENTICAL byte LM trainer learns as
# well as torch eager float32 (TF32 off) from the same init and the same
# batches (tools/lm_quality_vs_torch.py). VENDOR-AGNOSTIC (NVIDIA CUDA or AMD
# ROCm). No Mojo build, no pixi: the wheel goes into a throwaway venv.
#
# NVIDIA (RunPod H100, tools/gemm_remote_leg.sh hook):
#   MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key \
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/lm_quality_vs_torch_leg.sh \
#   MOJOLEARN_GEMM_LEG_OUT=<dir> \
#   sh tools/gemm_remote_leg.sh nvidia --rent --allow-concurrent \
#      --segment-lease 90 --dollar-cap 7 --gpu "NVIDIA H100 80GB HBM3" --local-card <apple.card>
# AMD (Hot Aisle MI300X):
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/lm_quality_vs_torch_leg.sh MOJOLEARN_GEMM_LEG_OUT=<dir> \
#   bash tools/hotaisle_leg.sh amd --rent --skip-gates --segment-lease 90 --dollar-cap 8
#
# Items, each with its exit code in status.tsv (a later item runs even when an
# earlier one fails):
#   corpus-enwik8   tools/fetch_corpus_enwik8.sh (sha256 checked)
#   wheel           venv + pip install mojolearn==$MOJOLEARN_LMQ_VERSION numpy
#   ours            the wheel, MOJOLEARN_NUMERIC_MODE=identical, $STEPS steps
#   torch-pin       tools/torch_lm_step_opponent_leg.sh, eager_fp32 on enwik8
#                   only: it installs the pinned torch (its own rules) and
#                   leaves one timing row as a by-product
#   torch           torch eager float32, $STEPS steps, and our final
#                   parameters evaluated by torch on the same held-out rows
#   compare         the table (tools/lm_quality_vs_torch.py compare)
set -u
ROOT=${MOJOLEARN_LMQ_ROOT:-/root/mojolearn}
OUT=${MOJOLEARN_LMQ_OUT:-/root/gemm_leg_out/lmq}
SCRATCH=${MOJOLEARN_LMQ_SCRATCH:-/root/lmq_scratch}
VERSION=${MOJOLEARN_LMQ_VERSION:-0.8.19}
STEPS=${MOJOLEARN_LMQ_STEPS:-300}
SHAPE=${MOJOLEARN_LMQ_SHAPE:-target}
VENV=${MOJOLEARN_LMQ_VENV:-/root/.venv-mojolearn-lmq}
mkdir -p "$OUT" "$SCRATCH"
cd "$ROOT" || exit 9
: > "$OUT/status.tsv"
rc=0
run() {
    _name=$1
    shift
    _t0=$(date +%s)
    "$@" > "$OUT/$_name.log" 2>&1
    _code=$?
    printf '%s\t%s\t%ss\n' "$_name" "$_code" "$(( $(date +%s) - _t0 ))" >> "$OUT/status.tsv"
    [ "$_code" = 0 ] || rc=1
    return "$_code"
}
{
    echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "commit=${MOJOLEARN_REPO_COMMIT:-$(sed -n 's/^commit=//p' /root/gemm_leg_out/leg.txt 2>/dev/null | head -1)}"
    echo "wheel=mojolearn==$VERSION steps=$STEPS shape=$SHAPE corpus=enwik8"
    command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi --query-gpu=name,driver_version,uuid --format=csv
    command -v rocm-smi > /dev/null 2>&1 && rocm-smi --showproductname --showdriverversion
    [ -f /opt/rocm/.info/version ] && echo "rocm $(cat /opt/rocm/.info/version)"
    echo "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-} ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
} > "$OUT/gate.txt" 2>&1
export MOJOLEARN_REPO_COMMIT="${MOJOLEARN_REPO_COMMIT:-$(sed -n 's/^commit=//p' "$OUT/gate.txt")}"

run corpus-enwik8 sh tools/fetch_corpus_enwik8.sh

make_venv() {
    if ! python3 -m venv "$VENV"; then
        DEBIAN_FRONTEND=noninteractive timeout -k 10 120 apt-get update -qq
        DEBIAN_FRONTEND=noninteractive timeout -k 10 300 apt-get install -y -qq python3-venv python3-pip
        python3 -m venv "$VENV" || return 1
    fi
    timeout -k 10 300 "$VENV/bin/pip" install -q --disable-pip-version-check --upgrade pip
    timeout -k 10 1200 "$VENV/bin/pip" install --disable-pip-version-check --no-input \
        "mojolearn==$VERSION" numpy || return 1
    "$VENV/bin/pip" freeze
    "$VENV/bin/python" -c 'import mojolearn, sys; print("mojolearn", mojolearn.__version__, mojolearn.__file__, sys.version)'
}
run wheel make_venv

rm -f "$OUT/ours.json" "$SCRATCH/ours_params.npy"
run ours env MOJOLEARN_NUMERIC_MODE=identical timeout -k 15 2400 "$VENV/bin/python" \
    tools/lm_quality_vs_torch.py ours --shape "$SHAPE" --steps "$STEPS" \
    --out "$OUT/ours.json" --params-out "$SCRATCH/ours_params.npy"

run torch-pin env MOJOLEARN_TORCH_LM_COLUMNS=eager_fp32 MOJOLEARN_TORCH_LM_CORPORA=enwik8 \
    MOJOLEARN_TORCH_LM_OUT="$OUT/torch-pin" sh tools/torch_lm_step_opponent_leg.sh
TPY=$(sed -n 's/^python_used=\([^ ]*\).*/\1/p' "$OUT/torch-pin/gate.txt" 2>/dev/null | head -1)
[ -n "$TPY" ] || TPY=$(sed -n 's/^python_found=\([^ ]*\).*/\1/p' "$OUT/torch-pin/gate.txt" 2>/dev/null | head -1)
echo "torch_python=${TPY:-NONE}" >> "$OUT/gate.txt"

rm -f "$OUT/torch.json"
if [ -n "$TPY" ]; then
    OURS_PARAMS=""
    [ -f "$SCRATCH/ours_params.npy" ] && OURS_PARAMS="--ours-params $SCRATCH/ours_params.npy"
    # shellcheck disable=SC2086
    run torch timeout -k 15 2400 "$TPY" tools/lm_quality_vs_torch.py torch --shape "$SHAPE" \
        --steps "$STEPS" --device cuda --out "$OUT/torch.json" $OURS_PARAMS
fi

if [ -f "$OUT/ours.json" ] && [ -f "$OUT/torch.json" ]; then
    run compare python3 tools/lm_quality_vs_torch.py compare --ours "$OUT/ours.json" --torch "$OUT/torch.json"
fi
echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/gate.txt"
cat "$OUT/status.tsv"
[ -f "$OUT/compare.log" ] && cat "$OUT/compare.log"
exit "$rc"
