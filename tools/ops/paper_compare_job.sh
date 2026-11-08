#!/bin/bash
# tools/ops/paper_compare_job.sh -- ONE paper-compare job on a rented GPU box (lane paper-compare-0866,
# 2026-11-08): the mlsys paper's 0.8.55 measurements, and ONLY those, rerun on a published release installed from
# PyPI (tools/bench_board.py --mojolearn-version: pip install mojolearn==<V> plus the vendor wheel), OUR ARMS ONLY.
# Opponents are never rerun (Andrew: opponents are scored once); the compare reuses the paper's stored opponent
# numbers from bench/results/paper_compare/recipes-<vendor>.json (tools/paper_compare.py recipes).
#
#   bash tools/ops/paper_compare_job.sh <nvidia|amd> [version, default 0.8.66]
#   bash tools/ops/paper_compare_job.sh --detect-vendor
#
# It runs under tools/box_persist_run.sh (results to R2 from the box), launched by the supported wrappers:
#   tools/dev_pod.sh run <lane> --results box-out/<tag> bash tools/ops/paper_compare_job.sh nvidia 0.8.66
#   tools/do_amd_steward.sh run <tag> box-out/<tag> bash tools/ops/paper_compare_job.sh amd 0.8.66
# and refuses to run unwrapped unless MOJOLEARN_ALLOW_UNPERSISTED=1.
#
# Phases (each logs its own seconds to job/phases.tsv; a phase failure is recorded and the job goes on where the
# next phase still has inputs -- fail soft; the exit status is the worst phase's):
#   plan     tools/paper_compare.py ids: the recipe races the board at this tree plans (plan/race-ids-*.txt,
#            plan/not-planned.json, plan/board-args.txt) and a --dry-run of them (plan/dry-run.log)
#   smoke    the board's measurement check over those races, SMOKE_SHARDS round-robin shards IN PARALLEL on the one
#            GPU (not timed; each shard its own cache/venv so no two pip installs share a venv)
#   board    the timed board over the races whose smoke passed: ONE process, serial BY NATURE: timed arms run one
#            at a time on one GPU with no competing job (AGENTS.md "Full-machine CPU race resources"); the drivers
#            themselves use every core (no thread caps are set; OMP/MKL/OPENBLAS caps are unset here)
#   compare  tools/paper_compare.py compare: COMPARE-<vendor>.md, compare-<vendor>.json, SUMMARY-<vendor>.txt
#
# Results ($MOJOLEARN_RESULTS_DIR): board/ (board.json, BOARD.md, logs/ per race, evidence/), smoke-s<i>-smoke/,
# plan/, logs/ (one per bench_board process), job/ (job.log, phases.tsv, cores.log = busy cores of total every
# 60 s, gpu.txt, box.txt, status.json), the COMPARE/SUMMARY files. Venvs and prepared data blocks live under
# $PC_CACHE_ROOT (default /root/pc-cache), outside the persisted dir.
#
# Inputs on the box (the orchestrator stages them first): the datasets under $PC_DATA_ROOT (default
# /root/datasets/gbm-bench: taxi/taxi_speed.npz, istella/istella_speed.npz; `sh tools/dataset_store.sh stage`
# from the Mac). Knobs: PC_SMOKE_SHARDS (default nproc/8, at most 4, at most MemAvailable / PC_SMOKE_MEM_GB
# (default 16) and at most the number of races; no measured per-shard memory yet: the job records free memory in
# job/box.txt and the cores log, read them to set it), PC_GPU_TOKEN (the paper's GPU, default L40S / MI300X),
# PC_REQUIRE_SAME_GPU=1 (stop on another GPU instead of recording HW-DIFF and going on).
# Tests: tools/test_paper_compare.py (syntax and the argument checks of this script).
set -uo pipefail

detect_vendor() {
    if command -v nvidia-smi > /dev/null 2>&1; then echo nvidia
    elif command -v rocm-smi > /dev/null 2>&1 || [ -x /opt/rocm/bin/rocm-smi ]; then echo amd
    else return 2; fi
}
usage() { echo "usage: $0 <nvidia|amd> [version] | --detect-vendor" >&2; exit 2; }
if [ "${1:-}" = --detect-vendor ]; then
    detect_vendor || { echo "paper-compare-job: neither nvidia-smi nor rocm-smi found" >&2; exit 2; }
    exit 0
fi
[ $# -ge 1 ] && [ $# -le 2 ] || usage
V=$1
VER=${2:-0.8.66}
case "$V" in nvidia|amd) ;; *) usage ;; esac
[[ "$VER" =~ ^[0-9]+(\.[0-9]+)+([.a-z0-9]*)$ ]] || { echo "paper-compare-job: bad version $VER" >&2; exit 2; }
if [ "${MOJOLEARN_BOX_PERSIST:-}" != 1 ] && [ "${MOJOLEARN_ALLOW_UNPERSISTED:-}" != 1 ]; then
    echo "paper-compare-job: runs under tools/box_persist_run.sh (tools/dev_pod.sh run / tools/do_amd_steward.sh run);" \
         "MOJOLEARN_ALLOW_UNPERSISTED=1 overrides" >&2
    exit 2
fi
TREE="$(cd "$(dirname "$0")/../.." && pwd)"
RES="${MOJOLEARN_RESULTS_DIR:?paper_compare_job.sh needs MOJOLEARN_RESULTS_DIR (set by tools/box_persist_run.sh)}"
mkdir -p "$RES" && RES="$(cd "$RES" && pwd)"
export MOJOLEARN_RESULTS_DIR="$RES"     # absolute: bench_board keeps evidence inside it (box_persisted_out)
RECIPES="$TREE/bench/results/paper_compare/recipes-$V.json"
[ -f "$RECIPES" ] || { echo "paper-compare-job: no $RECIPES" >&2; exit 2; }
DATA_ROOT=${PC_DATA_ROOT:-/root/datasets/gbm-bench}
CACHE_ROOT=${PC_CACHE_ROOT:-/root/pc-cache}
case "$V" in nvidia) TOKEN=${PC_GPU_TOKEN:-L40S} ;; amd) TOKEN=${PC_GPU_TOKEN:-MI300X} ;; esac
J="$RES/job"
mkdir -p "$J" "$RES/plan" "$RES/logs" "$CACHE_ROOT"
exec >> "$J/job.log" 2>&1
export PATH="/root/.pixi/bin:/usr/local/cuda/bin:/opt/rocm/bin:$PATH"
# the drivers use every core: no inherited thread caps (AGENTS.md "Full-machine CPU race resources")
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS
PY=${PC_PYTHON:-python3}
cd "$TREE" || exit 2
say() { echo "[$(date -u +%H:%M:%S) paper-compare-job] $*"; }
WORST=0
PHASE=start
P_T0=$(date +%s)
phase() { PHASE=$1; P_T0=$(date +%s); say "phase $PHASE start"; }
phase_end() {   # rc
    local rc=$1
    printf '%s\t%s\t%s\t%s\n' "$PHASE" "$rc" "$(( $(date +%s) - P_T0 ))" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$J/phases.tsv"
    say "phase $PHASE rc=$rc $(( $(date +%s) - P_T0 )) s"
    [ "$rc" -gt "$WORST" ] && WORST=$rc
    return 0
}
# busy cores of total, every 60 s, for the report (part of this job; it ends with the job)
cores_sampler() {
    local n prev cur
    n=$(nproc)
    prev=$(awk '/^cpu /{print $2+$3+$4+$5+$6+$7+$8+$9, $5+$6}' /proc/stat)
    while sleep 60; do
        cur=$(awk '/^cpu /{print $2+$3+$4+$5+$6+$7+$8+$9, $5+$6}' /proc/stat)
        echo "$prev $cur" | awk -v n="$n" -v t="$(date -u +%H:%M:%S)" -v ph="$(cat "$J/.phase" 2>/dev/null)" \
            '{dt=$3-$1; di=$4-$2; b=(dt>0)?(dt-di)/dt*n:0; printf "%s busy=%.1f total=%d phase=%s\n", t, b, n, ph}' \
            >> "$J/cores.log"
        prev=$cur
    done
}
SAMPLER=
if [ -r /proc/stat ]; then cores_sampler & SAMPLER=$!; fi
finish() {
    local code=$?
    trap - EXIT
    [ -n "$SAMPLER" ] && kill "$SAMPLER" 2> /dev/null
    [ "$WORST" -gt "$code" ] && code=$WORST
    printf '{"vendor":"%s","version":"%s","tree_head":"%s","phase":"%s","exit":%d,"job_script_sha256":"%s","recipes_sha256":"%s"}\n' \
        "$V" "$VER" "$(git -C "$TREE" rev-parse HEAD 2>/dev/null)" "$PHASE" "$code" \
        "$(sha256sum "$TREE/tools/ops/paper_compare_job.sh" | cut -d' ' -f1)" "$(sha256sum "$RECIPES" | cut -d' ' -f1)" \
        > "$J/status.json"
    say "exit $code (last phase $PHASE)"
    exit "$code"
}
trap finish EXIT
trap 'exit 124' TERM INT
mark() { echo "$1" > "$J/.phase"; }

say "paper compare $V $VER at $(git rev-parse HEAD 2>/dev/null) results $RES"
BOX_V=$(detect_vendor) || { say "neither nvidia-smi nor rocm-smi found"; exit 2; }
[ "$BOX_V" = "$V" ] || { say "asked for $V; this box is $BOX_V"; exit 3; }
{ uname -a; echo "nproc $(nproc)"; free -g 2> /dev/null || true; df -h /root 2> /dev/null || true; } > "$J/box.txt" 2>&1
if [ "$V" = nvidia ]; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > "$J/gpu.txt" 2>&1
else
    { rocm-smi --showproductname; rocm-smi --showmeminfo vram; } > "$J/gpu.txt" 2>&1
fi
NGPU=$(grep -c . "$J/gpu.txt" 2> /dev/null || echo 0)
[ "$V" = nvidia ] && [ "$NGPU" -gt 1 ] && say "NOTE: $NGPU GPUs; the timed board uses one (serial by nature)"
if ! grep -q "$TOKEN" "$J/gpu.txt"; then
    say "HW-DIFF: the paper ran on $TOKEN; this box: $(head -c 300 "$J/gpu.txt" | tr '\n' ' ')"
    [ "${PC_REQUIRE_SAME_GPU:-0}" = 1 ] && exit 3
fi
for f in taxi/taxi_speed.npz istella/istella_speed.npz; do
    [ -f "$DATA_ROOT/$f" ] || {
        say "missing dataset $DATA_ROOT/$f: stage it from the Mac first:" \
            "sh tools/dataset_store.sh stage \"<ssh flags+target>\" gbm-bench/taxi/taxi_speed.npz gbm-bench/istella/istella_speed.npz"
        exit 4; }
done

# ---- plan ------------------------------------------------------------------------------------------------------
phase plan; mark plan
$PY tools/paper_compare.py ids --recipes "$RECIPES" --planner tools/bench_board.py --out-dir "$RES/plan" \
    > "$RES/logs/ids.log" 2>&1
rc=$?; tail -n 3 "$RES/logs/ids.log"
if [ $rc -ne 0 ] || [ ! -s "$RES/plan/race-ids-full.txt" ]; then phase_end $(( rc ? rc : 5 )); exit; fi
mapfile -t BARGS < "$RES/plan/board-args.txt"
COMMON=("${BARGS[@]}" --mojolearn-version "$VER" --data-root "$DATA_ROOT" --rounds 1 --evidence-local-only)
$PY tools/bench_board.py "${COMMON[@]}" --cache "$CACHE_ROOT/c1" --race-ids @"$RES/plan/race-ids-full.txt" \
    --dry-run > "$RES/plan/dry-run.log" 2>&1
rc=$?; phase_end $rc
[ $rc -ne 0 ] && { say "dry run failed: $(tail -n 3 "$RES/plan/dry-run.log" | tr '\n' ' ')"; exit; }

# ---- direct ids (PC_IDS_FILE): time exactly these full race ids, no smoke --------------------------------------
# 0.8.66 NVIDIA run (2026-11-08): 15 of 62 paper races got "no smoke result" because their smoke id did not map
# back to the full id (taxi classical lanes, int8 gemm, mamba1/samba/transformer-decode, lossguide/pairlogit,
# iforest); the smoke itself PASSED every race it planned. A follow-up job times the missing ids directly.
if [ -n "${PC_IDS_FILE:-}" ]; then
    SJ=()
    cp "$PC_IDS_FILE" "$RES/plan/race-ids-pass.txt"
    say "direct ids: $(grep -c . "$RES/plan/race-ids-pass.txt") races from $PC_IDS_FILE (smoke skipped)"
    phase board; mark board
    $PY tools/bench_board.py "${COMMON[@]}" --cache "$CACHE_ROOT/c1" --race-ids @"$RES/plan/race-ids-pass.txt" \
        --out "$RES/board" > "$RES/logs/board.log" 2>&1
    rc=$?; tail -n 3 "$RES/logs/board.log"; phase_end $rc
    phase compare; mark compare
    $PY tools/paper_compare.py compare --recipes "$RECIPES" --board "$RES/board/board.json" \
        --out-md "$RES/COMPARE-$V.md" --out-json "$RES/compare-$V.json" --summary "$RES/SUMMARY-$V.txt" \
        > "$RES/logs/compare.log" 2>&1
    rc=$?; head -n 3 "$RES/logs/compare.log"; phase_end $rc
    mark finished
    exit "$WORST"
fi

# ---- smoke (parallel shards on the one GPU; not timed) ----------------------------------------------------------
phase smoke; mark smoke
NRACE=$(grep -c . "$RES/plan/race-ids-smoke.txt")
MEMG=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo 2> /dev/null || echo 16)
S=${PC_SMOKE_SHARDS:-$(( $(nproc) / 8 ))}
BYMEM=$(( MEMG / ${PC_SMOKE_MEM_GB:-16} ))
[ -z "${PC_SMOKE_SHARDS:-}" ] && [ "$S" -gt 4 ] && S=4
[ -z "${PC_SMOKE_SHARDS:-}" ] && [ "$S" -gt "$BYMEM" ] && S=$BYMEM
[ "$S" -gt "$NRACE" ] && S=$NRACE
[ "$S" -lt 1 ] && S=1
say "smoke: $NRACE races in $S parallel shards (nproc $(nproc), MemAvailable ${MEMG} GiB)"
pids=()
for i in $(seq 1 "$S"); do
    $PY tools/bench_board.py "${COMMON[@]}" --cache "$CACHE_ROOT/c$i" --race-ids @"$RES/plan/race-ids-smoke.txt" \
        --smoke --shard "$i/$S" --out "$RES/smoke-s$i" > "$RES/logs/smoke-s$i.log" 2>&1 &
    pids+=($!)
done
src=0
for p in "${pids[@]}"; do wait "$p"; r=$?; [ $r -gt $src ] && src=$r; done
grep -h -E '^SMOKE (PASS|FAIL)' "$RES"/logs/smoke-s*.log | head -n 8
# a SMOKE FAIL exits 1: recorded, the passed races still go on
phase_end $src

phase pass-ids; mark pass-ids
SMOKES=()
for f in "$RES"/smoke-s*-smoke/smoke.json; do [ -f "$f" ] && SMOKES+=("$f"); done
SJ=()
for f in "${SMOKES[@]}"; do SJ+=(--smoke-json "$f"); done
$PY tools/paper_compare.py pass-ids --full-ids "$RES/plan/race-ids-full.txt" "${SJ[@]}" \
    --out "$RES/plan/race-ids-pass.txt" > "$RES/logs/pass-ids.log" 2>&1
rc=$?; tail -n 1 "$RES/logs/pass-ids.log"; phase_end $rc

# ---- the timed board (one process: serial by nature on one GPU) -------------------------------------------------
phase board; mark board
if [ -s "$RES/plan/race-ids-pass.txt" ]; then
    $PY tools/bench_board.py "${COMMON[@]}" --cache "$CACHE_ROOT/c1" --race-ids @"$RES/plan/race-ids-pass.txt" \
        "${SJ[@]}" --out "$RES/board" > "$RES/logs/board.log" 2>&1
    rc=$?
    tail -n 3 "$RES/logs/board.log"
else
    say "no race passed its smoke; nothing timed"
    rc=6
fi
phase_end $rc

# ---- compare ----------------------------------------------------------------------------------------------------
phase compare; mark compare
NR=()
for f in "$RES/plan/not-planned.json" "$RES/plan/smoke-failed.json"; do [ -f "$f" ] && NR+=(--not-run "$f"); done
$PY tools/paper_compare.py compare --recipes "$RECIPES" --board "$RES/board/board.json" "${NR[@]}" \
    --out-md "$RES/COMPARE-$V.md" --out-json "$RES/compare-$V.json" --summary "$RES/SUMMARY-$V.txt" \
    > "$RES/logs/compare.log" 2>&1
rc=$?; head -n 3 "$RES/logs/compare.log"; phase_end $rc
mark finished
exit "$WORST"
