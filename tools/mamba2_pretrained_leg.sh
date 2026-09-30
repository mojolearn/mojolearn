#!/bin/sh
# tools/mamba2_pretrained_leg.sh -- state-spaces/mamba2-130m through
# mojolearn's GPU Mamba2Block in IDENTICAL mode on a DigitalOcean NVIDIA H100.
#
# ONE FILE, TWO ROLES.
#   sh tools/mamba2_pretrained_leg.sh dry-run   on the Mac: every local check,
#                                               nothing rented, no API call
#   sh tools/mamba2_pretrained_leg.sh rent      on the Mac: the real leg
#   (no argument, run as /root/gemm_leg_extra.sh) the body, ON THE DROPLET
#
# THE LAUNCHER RENTS THROUGH tools/do_extra_leg.sh nv, the proven DigitalOcean
# body runner, and adds nothing to its safety story; every guard below is
# that file's, cited by line so a reader can check it:
#   gpu-h100x1-80gb, nyc2, image 236925144, tag extra        do_extra_leg.sh:167
#   ssh key fingerprint df:f7:6b:...:f3:0b                   do_extra_leg.sh:129
#   token read by a builtin from a 0600 file outside the repo, written to a
#     0600 curl config, never exported, never in an argv       :463-480, :886-896
#   one GPU droplet at a time, shared GPU lock                 :929-983
#   LOCAL dead-man armed BEFORE the create, capped at --minutes :985-1007
#   EXIT trap: DELETE, then GET until 404, dead-man cancelled
#     only after the 404, loud banner if unconfirmed           :323-395
#   ON-DROPLET dead-man that DELETEs through the API at the
#     lease end, verified by process alive, id baked in and a
#     token GET returning 200, else the box is destroyed unused :1093-1121
#   the source: git archive of the commit minus bench/results,
#     mamba/corpus, oracles and *.bin, about 10 MB gzipped
#     (cap 15 MB), sha256 checked on the box                  :503-529, :1138-1173
#   fetch of /root/gemm_leg_out to <out>/remote/               :1253-1271
# The lease is 60 minutes, the hard cap, so both dead-men fire by then.
# After the runner returns, this launcher re-reads teardown.txt and
# deadman.txt and writes launcher_verdict.txt, so an unconfirmed destroy
# is a FAIL line and a nonzero exit, never silence.
#
# EVIDENCE lands OUTSIDE the repository, at
#   /Users/andrewhendel/CascadeProjects/mojolearn-evidence/mamba2-pretrained/<UTC stamp>-nvidia-h100/
# with the body's files under remote/mamba2-pretrained/.
#
# BEFORE `rent`: commit this file and tools/mamba2_pretrained_identity.py
# (the runner refuses a dirty tree and ships the COMMIT, not the worktree),
# fill MAMBA2_130M_PYTORCH_MODEL_SHA256 in the harness, and run `dry-run`.
#
# THE BODY'S PHASES, each with its exit code and seconds in status.tsv. A later
# phase runs even when an earlier one fails, because a red phase is a finding:
#   fetch-model          pytorch_model.bin + config.json (mamba2-130m) and
#                        tokenizer.json (gpt-neox-20b) from huggingface.co at
#                        the pinned revisions, each sha256-checked against
#                        the harness constants (the one place pins live)
#   build-mamba          bindings/build_mamba.sh for this GPU's architecture
#   identical            the harness, --mode identical --save-logits
#   build-base,          ONLY if `identical` failed for a reason other than a
#   identical-after-base refused input: the base binding, then the run again
#   identical-repeat     the same run again, fresh process
#   repeat-compare       --compare identical.json identical_repeat.json
#   fast-contrast        --mode fast; this lane is identical-only, so the
#                        EXPECTED result is exit 3 with the library's refusal
#                        recorded (a refusal is a record, never a pass)
#   torch-install,       optional (MOJOLEARN_PRETRAINED_TORCH_REF=1, default):
#   torch-reference      a CPU torch + transformers venv, then the harness's
#                        --torch-reference-only against identical.json, a
#                        SANITY check of the wiring, never an identity claim
#
# POSIX sh only: the droplet's /bin/sh is dash.
set -u

EVIDENCE_ROOT=/Users/andrewhendel/CascadeProjects/mojolearn-evidence/mamba2-pretrained
HARNESS=tools/mamba2_pretrained_identity.py

usage() {
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
}

pin() {  # <constant name>: its value in the harness
    sed -n "s/^$1 = \"\([0-9A-Za-z]*\)\"\$/\1/p" "$HARNESS" | head -n 1
}

# ============================================================== THE BODY
leg_body() {
    ROOT=/root/mojolearn
    OUT=/root/gemm_leg_out/mamba2-pretrained
    HF=/root/hf
    mkdir -p "$OUT" "$HF/mamba2-130m" "$HF/gpt-neox-20b"
    cd "$ROOT" || exit 9
    PATH="$HOME/.pixi/bin:$PATH"
    export PATH
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    export OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS
    MOJOLEARN_NUMERIC_MODE=identical
    export MOJOLEARN_NUMERIC_MODE
    # No sabotage, poison or alternate output directory may reach a build.
    unset MOJOLEARN_MAMBA_DEFINES MOJOLEARN_MAMBA_OUTDIR MOJOLEARN_BUILD_EXTRA_DEFINES MACOSX_DEPLOYMENT_TARGET
    MOJOLEARN_COMPILE_JOBS=${MOJOLEARN_COMPILE_JOBS:-8}
    export MOJOLEARN_COMPILE_JOBS
    NEW=${MOJOLEARN_PRETRAINED_MAX_NEW_TOKENS:-16}
    TORCH_REF=${MOJOLEARN_PRETRAINED_TORCH_REF:-1}
    STATUS="$OUT/status.tsv"
    : > "$STATUS"

    phase() {
        phase_name=$1
        shift
        phase_start=$(date +%s)
        "$@" > "$OUT/$phase_name.log" 2>&1
        phase_code=$?
        printf '%s\t%s\t%s\n' "$phase_name" "$phase_code" "$(( $(date +%s) - phase_start ))" >> "$STATUS"
        return $phase_code
    }

    # The GPU and its architecture, derived from the box as
    # tools/byte_lm_gpu_logits_leg.sh does (one mojo build is one GPU arch).
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        VENDOR=nvidia
        nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader > "$OUT/gpu.txt" 2>&1
        if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ]; then
            cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d ' ')
            case "$cap" in
                9.0) MOJOLEARN_GPU_ARCHS=sm_90a ;;
                *) MOJOLEARN_GPU_ARCHS=sm_$(printf '%s' "$cap" | tr -d '.') ;;
            esac
        fi
        MOJOLEARN_TARGET_COLUMN=nvidia
    elif [ -e /dev/kfd ] || command -v rocm-smi >/dev/null 2>&1; then
        VENDOR=amd
        (rocm-smi --showproductname 2>&1 || true) > "$OUT/gpu.txt"
        MOJOLEARN_TARGET_COLUMN=${MOJOLEARN_TARGET_COLUMN:-amd}
        if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ]; then
            echo 'amd: MOJOLEARN_GPU_ARCHS is required (gfx942 on an MI325X)' > "$OUT/refused.txt"
            exit 2
        fi
    else
        echo 'no NVIDIA or AMD GPU detected' > "$OUT/refused.txt"
        exit 2
    fi
    export MOJOLEARN_GPU_ARCHS MOJOLEARN_TARGET_COLUMN

    COMMIT=$(sed -n 's/^commit=//p' /root/gemm_leg_out/leg.txt 2>/dev/null | head -n 1)
    MOJOLEARN_GATE_COMMIT="$COMMIT"
    export MOJOLEARN_GATE_COMMIT
    printf 'vendor=%s\ngpu_archs=%s\ntarget_column=%s\ncommit=%s\nmax_new_tokens=%s\ntorch_ref=%s\n' \
        "$VENDOR" "$MOJOLEARN_GPU_ARCHS" "$MOJOLEARN_TARGET_COLUMN" "$COMMIT" "$NEW" "$TORCH_REF" > "$OUT/target.txt"
    (lscpu 2>/dev/null || true) > "$OUT/host.txt"
    (pixi run mojo --version 2>&1 || true) > "$OUT/mojo_version.txt"

    MODEL_REV=$(pin MAMBA2_130M_REVISION)
    MODEL_SHA=$(pin MAMBA2_130M_PYTORCH_MODEL_SHA256)
    CONFIG_SHA=$(pin MAMBA2_130M_CONFIG_SHA256)
    TOK_REV=$(pin GPT_NEOX_20B_REVISION)
    TOK_SHA=$(pin GPT_NEOX_20B_TOKENIZER_SHA256)
    for v in "$MODEL_SHA" "$CONFIG_SHA" "$TOK_SHA"; do
        if [ ${#v} -ne 64 ]; then
            echo "a sha256 pin in $HARNESS is not filled in (got '$v')" > "$OUT/refused.txt"
            exit 2
        fi
    done

    fetch_one() {  # <url> <dest> <sha256>
        curl -fsSL --retry 3 --max-time 900 -o "$2.part" "$1" || return 3
        got=$(sha256sum "$2.part" | cut -d' ' -f1)
        if [ "$got" != "$3" ]; then
            echo "sha256 mismatch for $2: got $got, pinned $3"
            rm -f "$2.part"
            return 4
        fi
        mv "$2.part" "$2"
        echo "verified $2 $got"
    }
    fetch_all() {
        fetch_one "https://huggingface.co/state-spaces/mamba2-130m/resolve/$MODEL_REV/pytorch_model.bin" \
            "$HF/mamba2-130m/pytorch_model.bin" "$MODEL_SHA" || return $?
        fetch_one "https://huggingface.co/state-spaces/mamba2-130m/resolve/$MODEL_REV/config.json" \
            "$HF/mamba2-130m/config.json" "$CONFIG_SHA" || return $?
        fetch_one "https://huggingface.co/EleutherAI/gpt-neox-20b/resolve/$TOK_REV/tokenizer.json" \
            "$HF/gpt-neox-20b/tokenizer.json" "$TOK_SHA" || return $?
        return 0
    }
    run_identity() {  # <out json> <harness args...>
        ri_out=$1
        shift
        pixi run python "$HARNESS" --model-dir "$HF/mamba2-130m" \
            --tokenizer "$HF/gpt-neox-20b/tokenizer.json" \
            --max-new-tokens "$NEW" --out "$ri_out" "$@"
    }
    install_torch() {
        if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
            curl -LsSf https://astral.sh/uv/install.sh | sh || return 3
        fi
        UV=$(command -v uv 2>/dev/null || echo "$HOME/.local/bin/uv")
        "$UV" venv --python 3.12 /root/torchref || return 4
        timeout 1200 "$UV" pip install --python /root/torchref/bin/python \
            --index-strategy unsafe-best-match \
            --index-url https://download.pytorch.org/whl/cpu \
            --extra-index-url https://pypi.org/simple \
            numpy torch transformers tokenizers || return 5
        /root/torchref/bin/python -c 'import torch, transformers; print(torch.__version__, transformers.__version__)'
    }

    if ! phase fetch-model fetch_all; then
        echo 'the model did not arrive verified; nothing else can run' > "$OUT/refused.txt"
        echo done > "$OUT/done.txt"
        exit 5
    fi
    phase build-mamba sh bindings/build_mamba.sh
    (find python/mojolearn -name '*.so' -exec sha256sum {} \; 2>/dev/null || true) > "$OUT/binaries.sha256"

    phase identical run_identity "$OUT/identical.json" --mode identical --save-logits
    rc=$?
    if [ "$rc" -ne 0 ] && [ "$rc" -ne 2 ]; then
        phase build-base env MOJOLEARN_SKIP_BUILD_GATE=1 sh bindings/build.sh
        (find python/mojolearn -name '*.so' -exec sha256sum {} \; 2>/dev/null || true) > "$OUT/binaries.sha256"
        phase identical-after-base run_identity "$OUT/identical.json" --mode identical --save-logits
    fi
    phase identical-repeat run_identity "$OUT/identical_repeat.json" --mode identical
    phase repeat-compare pixi run python "$HARNESS" --compare "$OUT/identical.json" "$OUT/identical_repeat.json"
    phase fast-contrast run_identity "$OUT/fast.json" --mode fast
    echo 'fast-contrast: exit 3 is the EXPECTED record (Mamba is an identical-only lane); it is not a pass' > "$OUT/fast_contrast_expectation.txt"

    if [ "$TORCH_REF" = 1 ]; then
        if phase torch-install install_torch; then
            phase torch-reference /root/torchref/bin/python "$HARNESS" \
                --torch-reference-only "$OUT/identical.json" \
                --model-dir "$HF/mamba2-130m" --tokenizer "$HF/gpt-neox-20b/tokenizer.json" \
                --out "$OUT/torch_reference.json"
        fi
    fi
    echo done > "$OUT/done.txt"
    exit 0
}

# ============================================================ THE LAUNCHER
launch() {
    how=$1
    if [ "$(uname)" != Darwin ]; then
        echo "the launcher runs on the Mac; the body runs only as /root/gemm_leg_extra.sh" >&2
        exit 2
    fi
    here=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd) || exit 2
    cd "$here" || exit 2
    sh -n tools/mamba2_pretrained_leg.sh || exit 1
    sha=$(pin MAMBA2_130M_PYTORCH_MODEL_SHA256)
    if [ ${#sha} -ne 64 ]; then
        echo "REFUSING: fill MAMBA2_130M_PYTORCH_MODEL_SHA256 in $HARNESS (and commit) first" >&2
        exit 2
    fi
    stamp=$(date -u +%Y-%m-%d_%H%M%S)
    out="$EVIDENCE_ROOT/$stamp-nvidia-h100"
    case "$out/" in "$here"/*)
        echo "REFUSING: the evidence directory $out is inside the repository" >&2
        exit 2 ;;
    esac
    mkdir -p "$EVIDENCE_ROOT" || exit 2
    set -- nv --minutes 60 --skip-gates
    if [ "$how" = dry ]; then
        set -- "$@" --dry-run
    fi
    MOJOLEARN_DO_TOKEN_FILE="${MOJOLEARN_DO_TOKEN_FILE:-$HOME/.mojolearn_do_token}" \
    MOJOLEARN_GEMM_LEG_EXTRA=tools/mamba2_pretrained_leg.sh \
    MOJOLEARN_GEMM_LEG_OUT="$out" \
    MOJOLEARN_DO_LOCK_LANE=extra:mamba2_pretrained \
    MOJOLEARN_DO_EXTRA_ENV="MOJOLEARN_PRETRAINED_MAX_NEW_TOKENS=16 MOJOLEARN_PRETRAINED_TORCH_REF=1 MOJOLEARN_COMPILE_JOBS=8" \
        bash tools/do_extra_leg.sh "$@"
    rc=$?
    if [ "$how" = dry ]; then
        exit $rc
    fi

    verdict="$out/launcher_verdict.txt"
    : > "$verdict"
    ok=1
    need() {  # <file> <grep pattern> <what>
        if grep -q -- "$2" "$1" 2>/dev/null; then
            echo "ok    $3" >> "$verdict"
        else
            echo "FAIL  $3 ($1 has no line matching '$2')" >> "$verdict"
            ok=0
        fi
    }
    echo "runner exit $rc" >> "$verdict"
    if ! grep -q '^create_http=' "$out/leg.txt" 2>/dev/null; then
        echo "no create was attempted; nothing was rented (read the runner output above)" >> "$verdict"
    elif ! grep -q '^droplet=' "$out/leg.txt" 2>/dev/null; then
        echo "a create was attempted and no droplet id was recorded" >> "$verdict"
        need "$out/teardown.txt" '^destroy_confirmed=1$' "the by-name sweep confirmed no droplet is left"
    else
        need "$out/teardown.txt" '^destroy_confirmed=1$' "destroy confirmed by the runner"
        need "$out/teardown.txt" '-> HTTP 404$' "GET on the droplet returned 404 after the DELETE"
        need "$out/deadman.txt" '^local_deadman_seconds=3600$' "local dead-man capped at 60 minutes"
        need "$out/deadman.txt" '^on_droplet_ON_DROPLET_DEADMAN_ARMED' "on-droplet dead-man process was alive"
        need "$out/deadman.txt" '^on_droplet_ID_BAKED_IN=[1-9]' "droplet id baked into the on-droplet dead-man"
        need "$out/deadman.txt" '^on_droplet_TOKEN_GET_HTTP=200$' "token GET from the droplet returned 200"
    fi
    body="$out/remote/mamba2-pretrained"
    if [ -f "$body/status.tsv" ]; then
        echo "body phases (name, exit, seconds):" >> "$verdict"
        sed 's/^/    /' "$body/status.tsv" >> "$verdict"
    else
        echo "no body status.tsv came home" >> "$verdict"
    fi
    cat "$verdict"
    if [ "$ok" != 1 ]; then
        echo "TEARDOWN NOT CONFIRMED. Check https://cloud.digitalocean.com/droplets now." >&2
        exit 1
    fi
    if [ -f "$body/identical.json" ]; then
        echo
        echo "compare with the Mac run:"
        echo "  pixi run python $HARNESS --compare <mac identical.json> $body/identical.json"
    fi
    exit $rc
}

case "${1:-}" in
    rent) launch rent ;;
    dry-run) launch dry ;;
    -h|--help) usage; exit 0 ;;
    "")
        if [ "$(basename "$0")" = gemm_leg_extra.sh ]; then
            leg_body
        fi
        usage >&2
        exit 2 ;;
    *) usage >&2; exit 2 ;;
esac
