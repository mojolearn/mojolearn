#!/bin/bash
# Guarded on-box body for the exact LM GEMM fold-stack specialization trial.
# R2 data is staged once; baseline/candidate run as alternating fresh processes.
set -euo pipefail

[ "${MOJOLEARN_LM_FOLD_RUN_GUARD:-}" = "R2_TAXI_ISTELLA" ] || {
    echo "refusing: set MOJOLEARN_LM_FOLD_RUN_GUARD=R2_TAXI_ISTELLA" >&2
    exit 64
}

R=${MOJOLEARN_TRIAL_ROOT:-/root/mojolearn}
B=${MOJOLEARN_LM_FOLD_BASELINE_ROOT:-/root/mojolearn-lmfs-baseline}
C=${MOJOLEARN_LM_FOLD_CANDIDATE_ROOT:-/root/mojolearn-lmfs-candidate}
S=${MOJOLEARN_LM_FOLD_SABOTAGE_ROOT:-/root/mojolearn-lmfs-sabotage}
OUT=${MOJOLEARN_LM_FOLD_OUT:-/root/lm_fold_specialize_out}
DATA=${GBM_BENCH_DATA:-/root/datasets/gbm-bench}
STEPS=${MOJOLEARN_LM_FOLD_STEPS:-6}
OUTERS=${MOJOLEARN_LM_FOLD_OUTERS:-3}
PY=${PYTHON:-$R/.pixi/envs/default/bin/python}
mkdir -p "$OUT/logs" "$OUT/json" "$OUT/runs" "$OUT/corpora"
export MOJOLEARN_NUMERIC_MODE=identical
export PYTHONUNBUFFERED=1
unset MOJOLEARN_GEMM_ARM MOJOLEARN_GEMM_PLAN_LABEL MOJOLEARN_TRANSFORMER_TIMING

say() { printf '[%s lmfs] %s\n' "$(date +%T)" "$*"; }
commit_of() {
    if [ -n "${MOJOLEARN_COMMIT:-}" ]; then printf '%s\n' "$MOJOLEARN_COMMIT"
    elif [ -f "$1/SHIPPED_COMMIT.txt" ]; then cat "$1/SHIPPED_COMMIT.txt"
    elif [ -f "$1/MOJOLEARN_COMMIT" ]; then cat "$1/MOJOLEARN_COMMIT"
    else git -C "$1" rev-parse HEAD
    fi
}
source_path() {
    case "$1" in
        taxi) printf '%s/taxi/taxi_speed.npz\n' "$DATA" ;;
        istella) printf '%s/istella/istella_speed.npz\n' "$DATA" ;;
        *) return 2 ;;
    esac
}
expected_sha() {
    case "$1" in
        taxi) echo 10d5d35f376a5b2ad5c66fabe624ee2caafb58bc5e7516824f801de9aab6cc15 ;;
        istella) echo 31f042376c0b819fe169cbbd998e840567c23dae25c14cb9054b460292f6ffef ;;
        *) return 2 ;;
    esac
}
prepare_corpora() {
    : > "$OUT/input_manifest.tsv"
    for ds in taxi istella; do
        src=$(source_path "$ds"); want=$(expected_sha "$ds")
        [ -f "$src" ] || { echo "missing R2-staged input $src" >&2; return 1; }
        got=$(sha256sum "$src" | awk '{print $1}')
        [ "$got" = "$want" ] || { echo "$ds sha256 $got, expected $want" >&2; return 1; }
        mkdir -p "$OUT/corpora/$ds"
        cp "$src" "$OUT/corpora/$ds/input.bin"
        bytes=$(stat -c %s "$src")
        "$PY" - "$OUT/corpora/$ds/manifest.json" "$want" "$bytes" "$ds" <<'PY'
import json,sys
path, sha, size, dataset = sys.argv[1:]
with open(path, "w") as stream:
    json.dump({"schema":"mojolearn.byte-lm.corpus.v1", "sha256":sha,
               "bytes":int(size), "source_url":"r2://gbm-bench/%s" % dataset}, stream,
              sort_keys=True)
    stream.write("\n")
PY
        printf '%s\t%s\t%s\t%s\n' "$ds" "$src" "$bytes" "$got" >> "$OUT/input_manifest.tsv"
    done
}
clone_tree() {
    src=$1; dst=$2
    rm -rf "$dst"; mkdir -p "$dst"
    (cd "$src" && tar --exclude=.git --exclude=.pixi -cf - .) | (cd "$dst" && tar -xf -)
    ln -s "$R/.pixi" "$dst/.pixi"
}
build_one() {
    tree=$1; arm=$2; defines=${3:-}
    rm -rf "$tree/python/mojolearn/identical/_mojolearn_byte_lm.so"
    (cd "$tree" && MOJOLEARN_BUILD_EXTRA_DEFINES="$defines" bash bindings/build_byte_lm.sh) \
        > "$OUT/logs/build_${arm}.log" 2>&1
    sha256sum "$tree/python/mojolearn/identical/_mojolearn_byte_lm.so" \
        > "$OUT/${arm}_binding.sha256"
}
phase_build() {
    case "${MOJOLEARN_TARGET_COLUMN:-}" in nvidia|amd) ;; *)
        echo "refusing: MOJOLEARN_TARGET_COLUMN must be nvidia or amd" >&2; return 2 ;; esac
    cd "$R"
    [ -x .pixi/envs/default/bin/python ] || pixi install
    [ -f python/mojolearn/identical/_mojolearn.so ] || bash bindings/build.sh
    clone_tree "$R" "$B"
    clone_tree "$R" "$C"
    clone_tree "$R" "$S"
    build_one "$B" baseline ""
    build_one "$C" candidate \
        "-D MOJOLEARN_GEMM_ARM_TRIAL=1 -D MOJOLEARN_GEMM_FOLD_SPECIALIZE_TRIAL=1"
    build_one "$S" sabotage \
        "-D MOJOLEARN_GEMM_ARM_TRIAL=1 -D MOJOLEARN_GEMM_FOLD_SPECIALIZE_TRIAL=1 -D MOJOLEARN_GEMM_FOLD_SPECIALIZE_SABOTAGE=1"
    {
        echo "source_commit=$(commit_of "$R")"
        echo "gpu_archs=${MOJOLEARN_GPU_ARCHS:-}"
        echo "target_column=${MOJOLEARN_TARGET_COLUMN:-}"
        "$R/.pixi/envs/default/bin/mojo" --version
        uname -a
        command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
        command -v rocm-smi >/dev/null && rocm-smi --showproductname --showuniqueid --showdriverversion
    } > "$OUT/build_provenance.txt" 2>&1
}
run_one() {
    tree=$1; arm=$2; ds=$3; outer=$4; position=$5; steps=$6
    tag="${ds}_${arm}_${outer}"
    run="$OUT/runs/$tag"
    rm -rf "$run"; mkdir -p "$run"
    (cd "$tree" && PYTHONPATH=python "$PY" tools/lm_step_memory_probe.py \
        --out "$run" --target --resident-lean --witness-every-step \
        --steps "$steps" --budget-seconds 900 \
        --corpus "$OUT/corpora/$ds/input.bin") > "$OUT/logs/$tag.log" 2>&1
    (cd "$tree" && PYTHONPATH=python "$PY" tools/lm_fold_specialize_probe.py record \
        --dataset "$ds" --arm "$arm" --outer "$outer" --launch-position "$position" \
        --commit "$(commit_of "$R")" \
        --binding "$tree/python/mojolearn/identical/_mojolearn_byte_lm.so" \
        --hardware "$OUT/build_provenance.txt" \
        --result "$run/result.json" --events "$run/events.jsonl" \
        --json "$OUT/json/$tag.json")
    say "$tag complete"
}
phase_run() {
    [ "$OUTERS" -eq 3 ] || { echo "gate requires exactly 3 fresh-process outers" >&2; return 2; }
    [ "$STEPS" -eq 6 ] || { echo "gate requires one warmup plus exactly 5 retained steps" >&2; return 2; }
    [ -s "$OUT/build_provenance.txt" ] || { echo "missing build/hardware provenance" >&2; return 2; }
    prepare_corpora
    : > "$OUT/run_order.tsv"
    for ds in taxi istella; do
        for outer in 1 2 3; do
            if [ $((outer % 2)) -eq 1 ]; then
                printf '%s\t%d\tbaseline,candidate\n' "$ds" "$outer" >> "$OUT/run_order.tsv"
                run_one "$B" baseline "$ds" "$outer" 0 "$STEPS"
                run_one "$C" candidate "$ds" "$outer" 1 "$STEPS"
            else
                printf '%s\t%d\tcandidate,baseline\n' "$ds" "$outer" >> "$OUT/run_order.tsv"
                run_one "$C" candidate "$ds" "$outer" 0 "$STEPS"
                run_one "$B" baseline "$ds" "$outer" 1 "$STEPS"
            fi
        done
        run_one "$S" sabotage "$ds" 1 0 1
    done
    (cd "$R" && PYTHONPATH=python "$PY" tools/lm_fold_specialize_probe.py summarize \
        "$OUT/json/*.json" --json "$OUT/summary.json") | tee "$OUT/summary.txt"
}

case "${1:-}" in
    build) phase_build ;;
    run) phase_run ;;
    all) phase_build && phase_run ;;
    *) echo "usage: $0 build|run|all" >&2; exit 64 ;;
esac
