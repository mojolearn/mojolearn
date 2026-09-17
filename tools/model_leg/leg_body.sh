#!/bin/sh
# tools/model_leg/leg_body.sh: THE ON-BOX BODY OF THE MODEL LEG, one file for
# every column. One real open model (default HuggingFaceTB/SmolLM2-360M)
# through mojolearn.models on this box's device in every weight format, then
# the same prompts through transformers + PyTorch in the incumbent's fast
# default and under its own determinism switch, and every record written
# under $OUT, which comes home with the runner's fetch.
#
# Vendor-agnostic and runner-agnostic, exactly as tools/byte_lm_gpu_logits_leg.sh
# and tools/identity_three_columns_leg.sh: it runs as the
# MOJOLEARN_GEMM_LEG_EXTRA body of tools/gemm_remote_leg.sh (RunPod NVIDIA)
# and tools/do_extra_leg.sh (DigitalOcean AMD) from /root/mojolearn, as the
# --cmd-file of tools/runpod_cpu_leg.sh (the CPU column), and on the M4 from
# the worktree (tools/model_leg/run_local_m4.sh). Those wrappers bake the
# knobs into a generated wrapper body, because the RunPod runner passes no
# environment to the body and the DigitalOcean runner passes only
# MOJOLEARN_*/MODULAR_* names.
#
# KNOBS (all MOJOLEARN_MODEL_LEG_*):
#   ROOT            the checkout (default /root/mojolearn)
#   OUT             the record directory (default /root/gemm_leg_out/model-leg)
#   MODEL           the model's repo id (default HuggingFaceTB/SmolLM2-360M;
#                   TinyLlama/TinyLlama-1.1B-Chat-v1.0 and meta-llama/Llama-3.2-1B
#                   are the other two named in bench/model/README.md). Its last
#                   path component names the store group models/<name>.
#   MODEL_DIR       where the STAGED files are (default /root/models/<name>: the
#                   box path of the store's HOME/models/<name>/* keys, put there by
#                   tools/stage_from_r2.sh on the Mac BEFORE this body runs,
#                   DEVIATION 2704). Nothing is downloaded on the box.
#   ALLOW_HF_DOWNLOAD=1  the runners' --allow-hf-download: when the staged
#                   directory has no config.json, fetch from Hugging Face on the
#                   box with a warning naming DEVIATION 2704. Off by default: an
#                   unstaged model is REFUSED, which is the loud failure.
#   FORMATS         default float32,bfloat16,int8
#   MAX_NEW         default 64          RUNS  default 3 (timed runs per prompt)
#   LABEL           the column label; derived from the device when unset
#   BUILDS          GPU bindings built, space separated bindings/<name>.sh names
#                   (default "build build_linalg build_transformer build_training";
#                   lane B2 names what mojolearn.models needs and this default
#                   follows it)
#   HOST_BUILDS     CPU bindings (default "build_core_host build_linalg_host
#                   build_transformer_host build_tokenizer_host")
#   SKIP_BUILD=1    the bindings are already built (the CPU pod builds them
#                   through its cache; the M4 wrapper builds them itself)
#   CPU_COLUMN_ALSO=1  after the device column, run the harness again from a
#                   package copy that carries no GPU set, MOJOLEARN_HOST_DIR
#                   naming the host bindings: the CPU column of the same box
#                   (bench/results/identity_break/2026-09-17_lowbit-m4/README.md's recipe)
#   PYTHON          how to run OUR harness (default "pixi run python")
#   VENV            the incumbent's throwaway venv (default /root/.venv-model-leg)
#   TRANSFORMERS_PIN  e.g. 4.56.2; empty installs the newest and the record pins what ran
#   SKIP_TORCH=1    no incumbent arm
#
# No credential ever reaches this body: the store presigns on the Mac and the
# box verified every staged file against bench/results/dataset_store/manifest.tsv.
#
# PHASES, each with its exit code and seconds in status.tsv; a later phase
# runs even when an earlier one fails, because a red phase is a finding:
#   detect          vendor, arch, label, device snapshot
#   build-*         each GPU binding (skipped on a CPU box or with SKIP_BUILD)
#   build-*_host    each host binding
#   venv            the incumbent's python: torch found or pinned, transformers
#   model           the staged files are present and hashed (never inside a timing);
#                   REFUSED when not staged unless ALLOW_HF_DOWNLOAD=1
#   harness         bench/model/harness.py, ours, every format
#   harness-cpu     the CPU column of this box (CPU_COLUMN_ALSO=1)
#   torch-twin      bench/model/torch_twin.py, fast and deterministic arms
#   ratio           bench/model/diff.py --ratio, printed into ratio.txt
# POSIX sh only: RunPod's Ubuntu images link /bin/sh to dash.
set -u
ROOT=${MOJOLEARN_MODEL_LEG_ROOT:-/root/mojolearn}
OUT=${MOJOLEARN_MODEL_LEG_OUT:-/root/gemm_leg_out/model-leg}
MODEL=${MOJOLEARN_MODEL_LEG_MODEL:-HuggingFaceTB/SmolLM2-360M}
FORMATS=${MOJOLEARN_MODEL_LEG_FORMATS:-float32,bfloat16,int8}
MAX_NEW=${MOJOLEARN_MODEL_LEG_MAX_NEW:-64}
RUNS=${MOJOLEARN_MODEL_LEG_RUNS:-3}
LABEL=${MOJOLEARN_MODEL_LEG_LABEL:-}
BUILDS=${MOJOLEARN_MODEL_LEG_BUILDS:-build build_linalg build_transformer build_training}
HOST_BUILDS=${MOJOLEARN_MODEL_LEG_HOST_BUILDS:-build_core_host build_linalg_host build_transformer_host build_tokenizer_host}
SKIP_BUILD=${MOJOLEARN_MODEL_LEG_SKIP_BUILD:-0}
CPU_ALSO=${MOJOLEARN_MODEL_LEG_CPU_COLUMN_ALSO:-0}
PY_OURS=${MOJOLEARN_MODEL_LEG_PYTHON:-pixi run python}
VENV=${MOJOLEARN_MODEL_LEG_VENV:-/root/.venv-model-leg}
TRANSFORMERS_PIN=${MOJOLEARN_MODEL_LEG_TRANSFORMERS_PIN:-}
SKIP_TORCH=${MOJOLEARN_MODEL_LEG_SKIP_TORCH:-0}
ALLOW_HF=${MOJOLEARN_MODEL_LEG_ALLOW_HF_DOWNLOAD:-0}
# printf, not a bare pipe: `tr -c` would turn basename's trailing newline
# into an underscore, and the leg of 2026-09-17 17:59 looked for
# /root/models/SmolLM2-360M_ beside the files the store had staged.
NAME=$(printf '%s' "$(basename "$MODEL")" | tr -c 'A-Za-z0-9_.-' '_')
MODEL_DIR=${MOJOLEARN_MODEL_LEG_MODEL_DIR:-/root/models/$NAME}
COMPILE_JOBS=${MOJOLEARN_COMPILE_JOBS:-8}
# macOS ships no timeout(1); the network steps then run unbounded there
if ! command -v timeout >/dev/null 2>&1; then timeout() { shift; "$@"; }; fi

mkdir -p "$OUT/logs"
cd "$ROOT" || exit 9
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_NUMERIC_MODE=identical
export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
STATUS="$OUT/status.tsv"
: > "$STATUS"
G="$OUT/gate.txt"
say() { echo "$@" >> "$G"; }
phase() {
    _n=$1; shift
    _t0=$(date +%s)
    "$@" > "$OUT/logs/$_n.log" 2>&1
    _e=$?
    printf '%s\t%s\t%s\n' "$_n" "$_e" "$(( $(date +%s) - _t0 ))" >> "$STATUS"
    return "$_e"
}
say "lane=model-leg started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
say "model=$MODEL formats=$FORMATS max_new=$MAX_NEW runs=$RUNS"

# ---- the commit, as tools/identity_three_columns_leg.sh resolves it ------------
if [ -n "${MOJOLEARN_COMMIT:-}" ] && [ ! -s "$ROOT/commit.txt" ]; then echo "$MOJOLEARN_COMMIT" > "$ROOT/commit.txt"; fi
COMMIT=$(cat "$ROOT/commit.txt" 2>/dev/null | head -1)
[ -n "$COMMIT" ] || COMMIT=$(sed -n 's/^commit=//p' /root/gemm_leg_out/leg.txt 2>/dev/null | head -1)
[ -n "$COMMIT" ] || COMMIT=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)
export MOJOLEARN_COMMIT="$COMMIT"
say "commit=$COMMIT"

# ---- detect: /dev/kfd or an AMD tool is AMD evidence; /dev/dri is not ----------
detect() {
    VENDOR=cpu
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        VENDOR=nvidia
        nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader
        if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ]; then
            _cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
            case "$_cc" in 9.0) MOJOLEARN_GPU_ARCHS=sm_90a ;; 12.0) MOJOLEARN_GPU_ARCHS=sm_120a ;; *) MOJOLEARN_GPU_ARCHS="sm_$(echo "$_cc" | tr -d .)" ;; esac
        fi
        MOJOLEARN_TARGET_COLUMN=nvidia
    elif [ -e /dev/kfd ] || command -v rocm-smi >/dev/null 2>&1; then
        VENDOR=amd
        rocm-smi --showproductname 2>&1 || true
        [ -n "${MOJOLEARN_GPU_ARCHS:-}" ] || MOJOLEARN_GPU_ARCHS=$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-z]+')
        MOJOLEARN_TARGET_COLUMN=${MOJOLEARN_TARGET_COLUMN:-amd}
    elif [ "$(uname -s)" = Darwin ]; then
        VENDOR=apple
        sysctl -n machdep.cpu.brand_string 2>/dev/null || true
        MOJOLEARN_TARGET_COLUMN=${MOJOLEARN_TARGET_COLUMN:-apple}
    else
        lscpu 2>/dev/null | head -20 || true
        MOJOLEARN_TARGET_COLUMN=cpu
        unset MOJOLEARN_GPU_ARCHS
    fi
    echo "vendor=$VENDOR archs=${MOJOLEARN_GPU_ARCHS:-none} column=$MOJOLEARN_TARGET_COLUMN"
    (uname -a; lscpu 2>/dev/null || true) > "$OUT/host.txt"
    ($PY_OURS --version; pixi run mojo --version 2>&1 || true) > "$OUT/versions.txt" 2>&1
}
phase detect detect
# re-derive from the log what the rest of the body needs (one source, the log that comes home)
VENDOR=$(sed -n 's/^vendor=\([a-z]*\).*/\1/p' "$OUT/logs/detect.log" | tail -1)
[ -n "$VENDOR" ] || VENDOR=cpu
if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ]; then
    MOJOLEARN_GPU_ARCHS=$(sed -n 's/.*archs=\([A-Za-z0-9_]*\).*/\1/p' "$OUT/logs/detect.log" | tail -1)
fi
case "$MOJOLEARN_GPU_ARCHS" in none|'') unset MOJOLEARN_GPU_ARCHS ;; *) export MOJOLEARN_GPU_ARCHS ;; esac
MOJOLEARN_TARGET_COLUMN=$(sed -n 's/.*column=\([a-z]*\).*/\1/p' "$OUT/logs/detect.log" | tail -1)
export MOJOLEARN_TARGET_COLUMN="${MOJOLEARN_TARGET_COLUMN:-cpu}"
if [ -z "$LABEL" ]; then
    case "$VENDOR" in
        nvidia) LABEL="nvidia-$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr 'A-Z ' 'a-z-' | tr -cd 'a-z0-9-' | cut -c1-24)-${MOJOLEARN_GPU_ARCHS:-sm}" ;;
        amd) LABEL="amd-${MOJOLEARN_GPU_ARCHS:-gfx}" ;;
        apple) LABEL="apple-$(sysctl -n machdep.cpu.brand_string 2>/dev/null | tr 'A-Z ' 'a-z-' | tr -cd 'a-z0-9-')-metal" ;;
        *) LABEL="cpu-$(sed -n 's/^Model name:[ \t]*//p' "$OUT/host.txt" | head -1 | tr 'A-Z ' 'a-z-' | tr -cd 'a-z0-9-' | cut -c1-32)" ;;
    esac
fi
say "vendor=$VENDOR archs=${MOJOLEARN_GPU_ARCHS:-none} column=$MOJOLEARN_TARGET_COLUMN label=$LABEL"

# ---- the bindings ----------------------------------------------------------------
if [ "$SKIP_BUILD" != 1 ]; then
    if [ "$VENDOR" != cpu ]; then
        for b in $BUILDS; do
            phase "build-$b" env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS="$COMPILE_JOBS" sh "bindings/$b.sh" \
                || say "build-$b FAILED"
        done
    fi
    for b in $HOST_BUILDS; do
        phase "build-$b" env -u MOJOLEARN_GPU_ARCHS MOJOLEARN_TARGET_COLUMN=cpu MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS="$COMPILE_JOBS" sh "bindings/$b.sh" \
            || say "build-$b FAILED"
    done
fi
(find python/mojolearn -name '*.so' -o -name '*.dylib' 2>/dev/null | LC_ALL=C sort | while read -r f; do
    sha256sum "$f" 2>/dev/null || shasum -a 256 "$f"; done) > "$OUT/binaries.sha256"

# ---- the incumbent's python: the torch found, or the pin, plus transformers ------
venv() {
    PYT=""
    for cand in ${MOJOLEARN_MODEL_LEG_TORCH_PYTHON:-} python3 python3.12 python3.11 python; do
        if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import torch' >/dev/null 2>&1; then PYT=$(command -v "$cand"); break; fi
    done
    BASE=${PYT:-$(command -v python3 || command -v python)}
    [ -n "$BASE" ] || { echo 'no python3 on this box'; return 5; }
    echo "base python: $BASE (torch importable: ${PYT:+yes}${PYT:-no})"
    if [ ! -x "$VENV/bin/python" ]; then
        "$BASE" -m venv --system-site-packages "$VENV" || return 6
    fi
    "$VENV/bin/python" -m pip install --quiet --upgrade pip || true
    if ! "$VENV/bin/python" -c 'import torch' >/dev/null 2>&1; then
        case "$VENDOR" in
            amd)  # the ROCm pin tools/torch_lm_step_opponent_leg.sh installed on the MI325X droplet
                timeout 900 "$VENV/bin/python" -m pip install --quiet \
                    'https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/pytorch_triton_rocm-3.2.0%2Brocm6.4.1.git6da9e660-cp312-cp312-linux_x86_64.whl#sha256=1d97c15798bf178299328032141a21d9777e7cdef59d5a7e3ac74e297c17198e' \
                    'https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.1/torch-2.6.0%2Brocm6.4.1.git1ded221d-cp312-cp312-linux_x86_64.whl#sha256=6b141e1a03148b007c6217519cd9947d760123ded5caebadffec22cba7358d2d' \
                    'numpy==1.26.4' || return 7 ;;
            nvidia)
                timeout 900 "$VENV/bin/python" -m pip install --quiet 'torch==2.4.1' 'numpy<2' \
                    --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple || return 7 ;;
            apple)
                timeout 900 "$VENV/bin/python" -m pip install --quiet torch numpy || return 7 ;;
            *)
                timeout 900 "$VENV/bin/python" -m pip install --quiet torch numpy \
                    --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple || return 7 ;;
        esac
    fi
    if [ -n "$TRANSFORMERS_PIN" ]; then _tf="transformers==$TRANSFORMERS_PIN"; else _tf=transformers; fi
    _hf=""; [ "$ALLOW_HF" = 1 ] && _hf=huggingface_hub
    timeout 900 "$VENV/bin/python" -m pip install --quiet "$_tf" safetensors accelerate sentencepiece $_hf || return 8
    "$VENV/bin/python" -m pip freeze > "$OUT/pip_freeze.txt"
    "$VENV/bin/python" -c 'import torch, transformers; print("torch", torch.__version__, "cuda", torch.version.cuda, "hip", getattr(torch.version, "hip", None)); print("transformers", transformers.__version__)'
}
if [ "$SKIP_TORCH" != 1 ] || [ "$ALLOW_HF" = 1 ]; then
    phase venv venv || say "venv FAILED (exit $?): the incumbent arm needs it"
fi

# ---- the model: STAGED from the R2 dataset store before this body ran ----------
# DEVIATION 2704 (tools/stage_from_r2.sh): the runner staged the store group
# models/<name> onto this box at /root/models/<name>/, each file verified
# against bench/results/dataset_store/manifest.tsv, with no credential here.
# This phase only checks that it happened and hashes what it found; a
# download on the box is the explicit opt-in below and nothing else.
model() {
    if [ -f "$MODEL_DIR/config.json" ]; then
        echo "staged: $MODEL_DIR"
    elif [ "$ALLOW_HF" = 1 ]; then
        echo "WARNING: $MODEL_DIR carries no config.json and --allow-hf-download is set: fetching $MODEL"
        echo "WARNING: from Hugging Face ON THIS BOX, against DEVIATION 2704 (rented boxes stage from R2,"
        echo "WARNING: never download); this record is not a store-pinned record"
        mkdir -p "$MODEL_DIR"
        timeout 1800 "$VENV/bin/python" - "$MODEL" "$MODEL_DIR" <<'PY' || return 4
import os, sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2],
                  allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*"],
                  token=os.environ.get("HF_TOKEN") or None)
print("fetched", sys.argv[1], "into", sys.argv[2])
PY
        echo "model_source=huggingface-on-box (DEVIATION 2704 opt-in)"
    else
        echo "REFUSED: $MODEL_DIR carries no config.json. The model group models/$NAME was not staged"
        echo "REFUSED: from the R2 dataset store (tools/stage_from_r2.sh, DEVIATION 2704); read the runner's"
        echo "REFUSED: stage.log. A download on a rented box is --allow-hf-download only."
        return 2
    fi
    [ -f "$MODEL_DIR/config.json" ] || { echo "no config.json under $MODEL_DIR"; return 2; }
    (cd "$MODEL_DIR" && ls -l && { sha256sum ./* 2>/dev/null || shasum -a 256 ./*; }) > "$OUT/model_files.sha256" 2>&1
    echo "model_dir=$MODEL_DIR"
}
phase model model || say "model FAILED (exit $?): not staged"
if ! grep -q '^model_dir=' "$OUT/logs/model.log"; then
    say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ) refused=model-not-staged"
    cat "$OUT/logs/model.log"
    echo refused > "$OUT/done.txt"
    exit 2
fi
grep -q 'model_source=huggingface-on-box' "$OUT/logs/model.log" && say "model_source=huggingface-on-box (DEVIATION 2704 opt-in, not store-pinned)"
say "model_dir=$MODEL_DIR"

# ---- ours -----------------------------------------------------------------------
phase harness env MOJOLEARN_NUMERIC_MODE=identical $PY_OURS bench/model/harness.py \
    --model "$MODEL_DIR" --formats "$FORMATS" --prompts bench/model/prompts.txt \
    --max-new "$MAX_NEW" --runs "$RUNS" --column "$LABEL" --out "$OUT/ours.$LABEL.json"
say "harness_exit=$(awk -F'\t' '$1=="harness"{print $2}' "$STATUS")"
grep -E '^cells=' "$OUT/logs/harness.log" >> "$G" 2>/dev/null

if [ "$CPU_ALSO" = 1 ] && [ "$VENDOR" != cpu ]; then
    # a package copy that carries no GPU set, the host bindings named by directory
    CPU_PKG=$OUT/../cpu-package
    rm -rf "$CPU_PKG"; mkdir -p "$CPU_PKG"
    cp -R python/mojolearn "$CPU_PKG/mojolearn"
    rm -rf "$CPU_PKG/mojolearn/identical" "$CPU_PKG/mojolearn/fast" "$CPU_PKG/mojolearn/deterministic" "$CPU_PKG/mojolearn/host"
    find "$CPU_PKG/mojolearn" \( -name '*.so' -o -name '*.dylib' -o -name '__pycache__' \) -prune -exec rm -rf {} + 2>/dev/null
    phase harness-cpu env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH="$CPU_PKG" MOJOLEARN_HOST_DIR="$ROOT/python/mojolearn/host" \
        $PY_OURS bench/model/harness.py --model "$MODEL_DIR" --formats "$FORMATS" \
        --prompts bench/model/prompts.txt --max-new "$MAX_NEW" --runs "$RUNS" --device cpu \
        --column "$LABEL-cpu" --out "$OUT/ours.$LABEL-cpu.json"
    say "harness_cpu_exit=$(awk -F'\t' '$1=="harness-cpu"{print $2}' "$STATUS")"
fi

# ---- the incumbent ------------------------------------------------------------
if [ "$SKIP_TORCH" != 1 ]; then
    phase torch-twin "$VENV/bin/python" bench/model/torch_twin.py --model "$MODEL_DIR" \
        --prompts bench/model/prompts.txt --max-new "$MAX_NEW" --runs "$RUNS" \
        --column "torch-$LABEL" --out "$OUT/torch.$LABEL.json"
    say "torch_twin_exit=$(awk -F'\t' '$1=="torch-twin"{print $2}' "$STATUS")"
    grep -E '^cells=' "$OUT/logs/torch-twin.log" >> "$G" 2>/dev/null
    if [ -s "$OUT/ours.$LABEL.json" ] && [ -s "$OUT/torch.$LABEL.json" ]; then
        phase ratio $PY_OURS bench/model/diff.py --ratio "$OUT/ours.$LABEL.json" "$OUT/torch.$LABEL.json"
        cp "$OUT/logs/ratio.log" "$OUT/ratio.txt" 2>/dev/null
    fi
fi
say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo done > "$OUT/done.txt"
exit 0
