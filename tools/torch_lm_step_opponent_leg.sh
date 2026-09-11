#!/bin/sh
# tools/torch_lm_step_opponent_leg.sh -- the OPPONENT row for our byte LM
# training step (bench/OPPONENT_REFERENCE.md "Rows that do not exist yet"
# item 5), measured once per GPU. VENDOR-AGNOSTIC: the same file runs on an
# AMD ROCm box and on an NVIDIA CUDA box; it detects which one it is on.
#
# AMD FIRST (Andrew, 2026-09-11: the deciding speed column is the Instinct
# MI325X on DigitalOcean). The orchestrator's DigitalOcean runner copies the
# source to /root/mojolearn and runs, on the droplet:
#
#   cd /root/mojolearn && MOJOLEARN_REPO_COMMIT=<sha> \
#     MOJOLEARN_TORCH_LM_OUT=<directory the runner fetches> \
#     sh tools/torch_lm_step_opponent_leg.sh
#
# NVIDIA H100 second, as tools/gemm_remote_leg.sh's MOJOLEARN_GEMM_LEG_EXTRA
# hook (after the leg's own device check and card, from /root/mojolearn;
# /root/gemm_leg_out/torch-lm-step/ comes home under remote/torch-lm-step/):
#
#   MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key \
#   MOJOLEARN_GPU_ARCHS=sm_90a \
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/torch_lm_step_opponent_leg.sh \
#   MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-nvidia-h100-torch-lm-step \
#   sh tools/gemm_remote_leg.sh nvidia --payload gemm --rent --minutes 60 \
#       --gpu "NVIDIA H100 80GB HBM3"
#
# ITEMS, each with its own exit code in status.tsv (name, exit, seconds); a
# later item runs even when an earlier one fails, because a red item is a
# finding. Exit 4 is NOT APPLICABLE: recorded, not a pass, not a failure.
#
#   torch-python   the first python on PATH that imports torch (never the
#                  pixi environment, whose pins every recorded timing uses).
#   torch-pin      the torch the row is labeled with. If the found torch is
#                  not the pin, the pin goes into a THROWAWAY venv and that
#                  venv runs the columns; the system python is never modified.
#                    amd     torch 2.6.0+rocm6.4.1.git1ded221d and
#                            pytorch-triton-rocm 3.2.0+rocm6.4.1.git6da9e660,
#                            cp312 wheels from repo.radeon.com rocm-rel-6.4.1
#                            with sha256 pinned, numpy 1.26.4: exactly what
#                            tools/do_byte_lm_setup.sh installed on the
#                            gpu-mi325x1-256gb droplet (image 188571990,
#                            Python 3.12) on 2026-09-07, where the preflight
#                            read hip 6.4.43483-a187df25c, "AMD Instinct
#                            Mi325X VF" (bench/results/resume/
#                            2026-09-07-root-byte-lm-do-amd/run6/remote/
#                            byte-lm-do-output/torch-hip-preflight.log).
#                    nvidia  torch 2.4.1+cu124 (the other torch rows in
#                            OPPONENT_REFERENCE.md), which
#                            runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
#                            already ships (e1g/2026-08-28_040316-nvidia-speed-
#                            gemmseq/remote/pip.log); else torch==2.4.1 from
#                            the cu124 index.
#   corpus-*       the two benchmark corpora (ENGINEERING_RULES section 9):
#                  English text (enwik8, tools/fetch_corpus_enwik8.sh) and
#                  source code (the Pile's GitHub component,
#                  tools/fetch_corpus_pile_github.sh), each verified against
#                  its manifest. The harness checks both sha256 again.
#   <column>-<corpus>  tools/torch_lm_step_opponent.py --shape target, 2
#                  warmups then 7 timed steps, one process each under
#                  `timeout 300` (124 is the deadline). eager_fp32 (THE ROW)
#                  on both corpora first, then eager_tf32 (NVIDIA only; on
#                  ROCm it writes a not_applicable record with the flag read
#                  back and exits 4), then compile_fp32 last so a compiler
#                  failure cannot cost the row.
#   summary.tsv    column, corpus, median seconds, tokens per second
#                  (NOT_APPLICABLE or NOT_RUN in both numeric fields).
#
# Everything the row is labeled with lands in the output directory:
# vendor.txt (nvidia-smi, or rocm-smi driver/product, amd-smi version and
# /opt/rocm/.info/version), versions.txt (torch, torch.version.cuda,
# torch.version.hip, device), and each column's JSON. No mojo build, no
# pixi. POSIX sh only (dash on the Ubuntu images).
set -u
ROOT=${MOJOLEARN_TORCH_LM_ROOT:-/root/mojolearn}
OUT=${MOJOLEARN_TORCH_LM_OUT:-/root/gemm_leg_out/torch-lm-step}
COLUMNS=${MOJOLEARN_TORCH_LM_COLUMNS:-eager_fp32,eager_tf32,compile_fp32}
CORPORA="enwik8 pile_github"
SHAPE=${MOJOLEARN_TORCH_LM_SHAPE:-target}
WARMUP=${MOJOLEARN_TORCH_LM_WARMUP:-2}
STEPS=${MOJOLEARN_TORCH_LM_STEPS:-7}
DEADLINE=${MOJOLEARN_TORCH_LM_DEADLINE:-300}
VENV=${MOJOLEARN_TORCH_LM_VENV:-/root/.venv-torch-lm-step}
mkdir -p "$OUT"
cd "$ROOT" || exit 9
: > "$OUT/status.tsv"

rc=0
record() {
    # record <name> <exit code> <seconds>; 4 = not applicable, not a failure.
    printf '%s\t%s\t%ss\n' "$1" "$2" "$3" >> "$OUT/status.tsv"
    case "$2" in
        0|4) ;;
        *) rc=1 ;;
    esac
}
run() {
    # run <name> <command...>: log to $OUT/<name>.log, record the exit code.
    _name=$1
    shift
    _t0=$(date +%s)
    "$@" > "$OUT/$_name.log" 2>&1
    _code=$?
    record "$_name" "$_code" "$(( $(date +%s) - _t0 ))"
    return "$_code"
}

if [ -z "${MOJOLEARN_REPO_COMMIT:-}" ]; then
    MOJOLEARN_REPO_COMMIT=${MOJOLEARN_COMMIT:-$(sed -n 's/^commit=//p' /root/gemm_leg_out/leg.txt 2>/dev/null | head -1)}
fi
export MOJOLEARN_REPO_COMMIT

# ---- vendor: /dev/kfd or an AMD tool is AMD evidence; /dev/dri is not -----------
VENDOR=unknown
if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi -L > /dev/null 2>&1; then
    VENDOR=nvidia
elif [ -e /dev/kfd ] || command -v rocm-smi > /dev/null 2>&1 || command -v amd-smi > /dev/null 2>&1; then
    VENDOR=amd
fi
vendor_snapshot() {
    echo "== date $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if [ "$VENDOR" = nvidia ]; then
        nvidia-smi --query-gpu=name,driver_version,compute_cap,uuid,clocks.sm,temperature.gpu --format=csv 2>&1
        nvidia-smi 2>&1
    fi
    if command -v rocm-smi > /dev/null 2>&1; then
        echo "== rocm-smi --showproductname --showdriverversion"
        rocm-smi --showproductname --showdriverversion 2>&1
        echo "== rocm-smi --showuse --showmemuse --showtemp"
        rocm-smi --showuse --showmemuse --showtemp 2>&1
    fi
    if command -v amd-smi > /dev/null 2>&1; then
        echo "== amd-smi version"
        amd-smi version 2>&1
    fi
    if [ -f /opt/rocm/.info/version ]; then
        echo "== /opt/rocm/.info/version"
        cat /opt/rocm/.info/version
    fi
    ls -d /opt/rocm-* 2> /dev/null
}
vendor_snapshot > "$OUT/vendor_before.txt" 2>&1

case "$VENDOR" in
    amd)
        PIN_VERSION="2.6.0+rocm6.4.1.git1ded221d"
        PIN_PYTHON_MINOR="3.12"
        PIN_ARGS="https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/pytorch_triton_rocm-3.2.0%2Brocm6.4.1.git6da9e660-cp312-cp312-linux_x86_64.whl#sha256=1d97c15798bf178299328032141a21d9777e7cdef59d5a7e3ac74e297c17198e https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/torch-2.6.0%2Brocm6.4.1.git1ded221d-cp312-cp312-linux_x86_64.whl#sha256=6b141e1a03148b007c6217519cd9947d760123ded5caebadffec22cba7358d2d numpy==1.26.4"
        PIN_INDEX_ARGS="" ;;
    nvidia)
        PIN_VERSION="2.4.1+cu124"
        PIN_PYTHON_MINOR=""
        PIN_ARGS="torch==2.4.1 numpy<2"
        PIN_INDEX_ARGS="--index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple" ;;
    *)
        PIN_VERSION=""
        PIN_PYTHON_MINOR=""
        PIN_ARGS=""
        PIN_INDEX_ARGS="" ;;
esac

{
    echo "lane=torch byte LM step opponent (OPPONENT_REFERENCE rows owed, item 5)"
    echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "vendor=$VENDOR"
    echo "root=$ROOT"
    echo "commit=$MOJOLEARN_REPO_COMMIT"
    echo "columns=$COLUMNS"
    echo "corpora=$CORPORA"
    echo "shape=$SHAPE warmup=$WARMUP steps=$STEPS deadline=$DEADLINE"
    echo "torch_pin=${PIN_VERSION:-none (vendor unknown)}"
} > "$OUT/gate.txt"
echo "---- vendor=$VENDOR"
cat "$OUT/vendor_before.txt"

# ---- torch-python ------------------------------------------------------------
_t0=$(date +%s)
PY=""
for cand in ${MOJOLEARN_TORCH_LM_PYTHON:-} python3 python3.12 python3.11 python; do
    if command -v "$cand" > /dev/null 2>&1 && "$cand" -c 'import torch' > /dev/null 2>&1; then
        PY=$(command -v "$cand")
        break
    fi
done
if [ -n "$PY" ]; then
    echo "python_found=$PY" >> "$OUT/gate.txt"
    record torch-python 0 "$(( $(date +%s) - _t0 ))"
else
    echo "python_found=NONE (no python on PATH imports torch)" >> "$OUT/gate.txt"
    record torch-python 1 "$(( $(date +%s) - _t0 ))"
fi

# ---- torch-pin ------------------------------------------------------------------
if [ -n "$PIN_VERSION" ]; then
    _t0=$(date +%s)
    found=""
    [ -n "$PY" ] && found=$("$PY" -c 'import torch; print(torch.__version__)' 2>/dev/null)
    echo "torch_found=$found" >> "$OUT/gate.txt"
    if [ "$found" = "$PIN_VERSION" ]; then
        echo "torch $PIN_VERSION already present in $PY" > "$OUT/torch-pin.log"
        record torch-pin 0 "$(( $(date +%s) - _t0 ))"
    else
        {
            echo "found torch '$found', pin $PIN_VERSION; installing into $VENV"
            base=""
            for cand in "python$PIN_PYTHON_MINOR" python3; do
                [ "$cand" = python ] && continue
                if command -v "$cand" > /dev/null 2>&1; then
                    base=$(command -v "$cand")
                    break
                fi
            done
            echo "venv base python: ${base:-NONE} ($("${base:-false}" --version 2>&1))"
            if [ -n "$base" ] && ! "$base" -m venv "$VENV" > /dev/null 2>&1; then
                echo "venv unavailable; apt-get install python3-venv"
                DEBIAN_FRONTEND=noninteractive timeout -k 10 120 apt-get update -qq
                DEBIAN_FRONTEND=noninteractive timeout -k 10 180 apt-get install -y -qq python3-venv python3-pip
                "$base" -m venv "$VENV"
            fi
            # shellcheck disable=SC2086
            "$VENV/bin/pip" install -q --disable-pip-version-check --upgrade pip \
                && "$VENV/bin/pip" install --disable-pip-version-check --no-input $PIN_INDEX_ARGS $PIN_ARGS
            echo "pip exit $?"
            "$VENV/bin/pip" freeze 2>&1
        } > "$OUT/torch-pin.log" 2>&1
        got=$("$VENV/bin/python" -c 'import torch; print(torch.__version__)' 2>> "$OUT/torch-pin.log")
        if [ "$got" = "$PIN_VERSION" ]; then
            PY="$VENV/bin/python"
            echo "python_used=$PY (pinned venv)" >> "$OUT/gate.txt"
            record torch-pin 0 "$(( $(date +%s) - _t0 ))"
        else
            echo "pin install gave '$got'; columns run on the found torch '$found', labeled by its JSON" \
                >> "$OUT/torch-pin.log"
            record torch-pin 1 "$(( $(date +%s) - _t0 ))"
        fi
    fi
else
    echo "torch_pin=not attempted (vendor unknown)" >> "$OUT/gate.txt"
fi

if [ -n "$PY" ]; then
    "$PY" - > "$OUT/versions.txt" 2>&1 <<'PY'
import platform, sys, torch
print("python", platform.python_version(), sys.executable)
print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
print("torch.version.hip", getattr(torch.version, "hip", None))
print("cuda api available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
    print("cudnn/miopen", torch.backends.cudnn.version())
try:
    import numpy
    print("numpy", numpy.__version__)
except ImportError:
    print("numpy absent (the harness falls back to a labeled torch init)")
PY
    echo "---- versions ($PY)"
    cat "$OUT/versions.txt"
fi

# ---- the two corpora -----------------------------------------------------------
run corpus-enwik8 sh tools/fetch_corpus_enwik8.sh
run corpus-pile-github sh tools/fetch_corpus_pile_github.sh

# ---- the columns, one process each ---------------------------------------------
if [ -n "$PY" ]; then
    for column in $(echo "$COLUMNS" | tr ',' ' '); do
        for corpus in $CORPORA; do
            rm -f "$OUT/$column-$corpus.json"
            run "$column-$corpus" timeout -k 15 "$DEADLINE" "$PY" tools/torch_lm_step_opponent.py \
                --shape "$SHAPE" --corpus "$corpus" --column "$column" --device cuda \
                --warmup "$WARMUP" --steps "$STEPS" --out "$OUT/$column-$corpus.json"
        done
    done
else
    echo "no torch python; columns NOT RUN" >> "$OUT/gate.txt"
    record columns 9 0
fi

# ---- summary --------------------------------------------------------------------
SUMMARY_PY=${PY:-python3}
"$SUMMARY_PY" - "$OUT" "$COLUMNS" "$CORPORA" > "$OUT/summary.tsv" 2> "$OUT/summary.err" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
print("column\tcorpus\tmedian_seconds\ttokens_per_second")
for column in sys.argv[2].split(","):
    for corpus in sys.argv[3].split():
        path = out / f"{column}-{corpus}.json"
        if not path.is_file():
            print(f"{column}\t{corpus}\tNOT_RUN\tNOT_RUN")
            continue
        j = json.loads(path.read_text())
        if j.get("status") != "measured":
            tag = "NOT_APPLICABLE" if j.get("status") == "not_applicable" else "NOT_RUN"
            print(f"{column}\t{corpus}\t{tag}\t{tag}")
            continue
        print(f"{column}\t{corpus}\t{j['median_seconds']:.6f}\t{j['tokens_per_second']:.1f}")
PY
record summary $? 0

vendor_snapshot > "$OUT/vendor_after.txt" 2>&1
echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/gate.txt"
echo "---- summary"
cat "$OUT/summary.tsv"
echo "---- status"
cat "$OUT/status.tsv"
exit "$rc"
