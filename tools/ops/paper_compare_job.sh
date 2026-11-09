#!/bin/bash
# tools/ops/paper_compare_job.sh -- ONE measurement job on a rented GPU box (lane paper-compare-0866, 2026-11-08;
# candidate mode and the owed rows: lane measure-prep, 2026-11-09). OUR ARMS ONLY: opponents are never rerun
# (Andrew: opponents are scored once); the compare reuses the paper's stored opponent numbers.
#
#   bash tools/ops/paper_compare_job.sh [<nvidia|amd> [<version>|candidate]]   (no arguments: vendor detected,
#                                                                              mode from the environment)
#   bash tools/ops/paper_compare_job.sh --detect-vendor
#
# MODES
#   published  (default; version default 0.8.66) the mlsys paper's 0.8.55 recipes on a release from PyPI
#              (tools/pypi_wheel_install.py through tools/bench_board.py --mojolearn-version).
#   candidate  (PC_WHEEL_DIR set, or the word `candidate`) the exact candidate wheels in PC_WHEEL_DIR on the box
#              (mojolearn_core-<v>-*.whl and mojolearn_<vendor>-<v>-*.whl, staged by tools/ops/measure_candidate.py
#              stage) of source commit PC_SOURCE_COMMIT (40 hex), installed by tools/candidate_wheel_install.py
#              through bench_board --candidate-wheel: the wheel RECORDs, metadata, tags and the frozen payload's
#              source commit are checked, every file's sha256 lands in job/wheel-manifest-candidate.json and in
#              every board's box.mojolearn.wheel, and the compare is labelled "candidate <sha12> (<version>)".
#              The compare has three columns: 0.8.55 (paper), the prior release (PC_PRIOR_VERSION, default 0.8.66:
#              bench/results/paper_compare/compare-<vendor>.json on main) and the candidate, with REGRESSION vs
#              each and RECOVERED. The owed rows (bench/results/paper_compare/owed-pre0867.json,
#              tools/measure_owed.py) run too (PC_OWED=0 turns them off): the candidate, the prior release
#              installed ours-only from PyPI, and each B wheel set listed in PC_B_ARMS (a file on the box, lines
#              `<name> <wheel dir> <source commit>`, names from the owed file's b_arms).
#
# ORDER (each phase logs its seconds to job/phases.tsv; fail soft: a failed phase is recorded and the job goes on
# where the next phase still has inputs; the exit status is the worst phase's)
#   plan     (CPU, seconds) paper_compare ids + measure_owed plan: race ids per board process
#   install  (CPU, ALL AT ONCE) one ours-only venv per arm (bench_board --install-only --ours-install-only: our
#            wheels + scikit-learn + pandas, no opponent stack: the 0.8.66 smoke made a 13 GB venv per shard), each
#            checked against its wheel manifest and our GPU set once; plus the board's dry run
#   smoke    the paper races + the owed full-row races in parallel shards on the one GPU (not timed), all on the
#            candidate venv (--python-env --skip-install: no per-shard install); shards = min(nproc/4, MemAvailable
#            / PC_SMOKE_MEM_GB (default 12), PC_SMOKE_MAX (default 8), races); PC_SMOKE_SHARDS overrides
#   board    TIMED, SERIAL BY NATURE: one board process at a time on the one GPU with nothing else running
#            (timing must not share the GPU; AGENTS.md "Full-machine CPU race resources"; the drivers themselves
#            use every core, no thread caps): candidate paper+owed full rows, then each owed unit
#            (tools/measure_owed.py units.tsv: neighbouring --rows groups, the prior arm, the B arms), then the
#            driver rows (tools/measure_owed_driver.py per arm interpreter)
#   compare  paper_compare compare (COMPARE/compare/SUMMARY-<vendor>) and measure_owed report (OWED*-<vendor>)
# PC_PARTS (default all: plan,install,smoke,board,owed,drivers,compare) runs a subset and PC_OWED_ARMS (comma list,
# default all) the owed units of some arms, so a job over the 240-minute nvidia_central.sh cap is split into
# submits on the same results dir (the boards resume from board.json; a later submit reruns install, which only
# re-checks an existing venv).
#
# It runs under tools/box_persist_run.sh (results to R2 from the box), launched by the supported wrappers
# (tools/ops/measure_candidate.py prints the exact lines; docs/plans/PRE_0867_MEASUREMENT.md):
#   tools/nvidia_central.sh submit <lane> --cap 240 --results box-out/<tag> --env PC_WHEEL_DIR=... \
#       --env PC_SOURCE_COMMIT=... tools/ops/paper_compare_job.sh
#   tools/do_amd_steward.sh run <tag> box-out/<tag> 'PC_WHEEL_DIR=... PC_SOURCE_COMMIT=... bash tools/ops/paper_compare_job.sh'
# and refuses to run unwrapped unless MOJOLEARN_ALLOW_UNPERSISTED=1.
#
# Results ($MOJOLEARN_RESULTS_DIR): board/ (the candidate's paper + owed full-row board), board-<arm>-<rows>/,
# smoke-s<i>-smoke/, install-<arm>/, drivers/drivers.jsonl, plan/ (paper), plan-owed/, logs/ (one per process),
# job/ (job.log with a BUSY line every 5 minutes, phases.tsv, cores.log = busy cores of total every 60 s, gpu.txt,
# box.txt, status.json, wheel-manifest-<arm>.json), COMPARE/SUMMARY/OWED files. Venvs and data blocks live under
# $PC_CACHE_ROOT (default /root/pc-cache), outside the persisted dir.
#
# Inputs on the box: the datasets under $PC_DATA_ROOT (default /root/datasets/gbm-bench: taxi/taxi_speed.npz,
# istella/istella_speed.npz; `sh tools/dataset_store.sh stage` from the Mac). Knobs: PC_GPU_TOKEN (the paper's GPU,
# default L40S / MI300X), PC_REQUIRE_SAME_GPU=1 (stop on another GPU instead of recording HW-DIFF and going on).
# Tests: tools/test_paper_compare.py (syntax and the argument checks of this script).
set -uo pipefail

detect_vendor() {
    if command -v nvidia-smi > /dev/null 2>&1; then echo nvidia
    elif command -v rocm-smi > /dev/null 2>&1 || [ -x /opt/rocm/bin/rocm-smi ]; then echo amd
    else return 2; fi
}
usage() { echo "usage: $0 [<nvidia|amd> [<version>|candidate]] | --detect-vendor" >&2; exit 2; }
if [ "${1:-}" = --detect-vendor ]; then
    detect_vendor || { echo "paper-compare-job: neither nvidia-smi nor rocm-smi found" >&2; exit 2; }
    exit 0
fi
[ $# -le 2 ] || usage
V=${1:-}
ARG2=${2:-}
[ -z "$V" ] || case "$V" in nvidia|amd) ;; *) usage ;; esac
MODE=published
VER=0.8.66
if [ "$ARG2" = candidate ] || { [ -z "$ARG2" ] && [ -n "${PC_WHEEL_DIR:-}" ]; }; then
    MODE=candidate
elif [ -n "$ARG2" ]; then
    VER=$ARG2
fi
[[ "$VER" =~ ^[0-9]+(\.[0-9]+)+([.a-z0-9]*)$ ]] || { echo "paper-compare-job: bad version $VER" >&2; exit 2; }
PRIOR=${PC_PRIOR_VERSION:-0.8.66}
[[ "$PRIOR" =~ ^[0-9]+(\.[0-9]+)+([.a-z0-9]*)$ ]] || { echo "paper-compare-job: bad PC_PRIOR_VERSION $PRIOR" >&2; exit 2; }
if [ "$MODE" = candidate ]; then
    [ -n "${PC_WHEEL_DIR:-}" ] || { echo "paper-compare-job: candidate mode needs PC_WHEEL_DIR" >&2; exit 2; }
    [[ "${PC_SOURCE_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] || {
        echo "paper-compare-job: candidate mode needs PC_SOURCE_COMMIT (the frozen source, 40 hex)" >&2; exit 2; }
fi
if [ "${MOJOLEARN_BOX_PERSIST:-}" != 1 ] && [ "${MOJOLEARN_ALLOW_UNPERSISTED:-}" != 1 ]; then
    echo "paper-compare-job: runs under tools/box_persist_run.sh (tools/nvidia_central.sh submit / tools/do_amd_steward.sh run);" \
         "MOJOLEARN_ALLOW_UNPERSISTED=1 overrides" >&2
    exit 2
fi
if [ -z "$V" ]; then
    V=$(detect_vendor) || { echo "paper-compare-job: neither nvidia-smi nor rocm-smi found" >&2; exit 2; }
fi
TREE="$(cd "$(dirname "$0")/../.." && pwd)"
RES="${MOJOLEARN_RESULTS_DIR:?paper_compare_job.sh needs MOJOLEARN_RESULTS_DIR (set by tools/box_persist_run.sh)}"
mkdir -p "$RES" && RES="$(cd "$RES" && pwd)"
export MOJOLEARN_RESULTS_DIR="$RES"     # absolute: bench_board keeps evidence inside it (box_persisted_out)
PCDIR="$TREE/bench/results/paper_compare"
RECIPES="$PCDIR/recipes-$V.json"
OWED_FILE=${PC_OWED_FILE:-$PCDIR/owed-pre0867.json}
PRIOR_COMPARE=${PC_PRIOR_COMPARE:-$PCDIR/compare-$V.json}
[ -f "$RECIPES" ] || { echo "paper-compare-job: no $RECIPES" >&2; exit 2; }
OWED=${PC_OWED:-$([ "$MODE" = candidate ] && echo 1 || echo 0)}
PARTS=",${PC_PARTS:-plan,install,smoke,board,owed,drivers,compare},"
part() { [[ "$PARTS" == *",$1,"* ]]; }
DATA_ROOT=${PC_DATA_ROOT:-/root/datasets/gbm-bench}
CACHE_ROOT=${PC_CACHE_ROOT:-/root/pc-cache}
case "$V" in nvidia) TOKEN=${PC_GPU_TOKEN:-L40S} ;; amd) TOKEN=${PC_GPU_TOKEN:-MI300X} ;; esac
J="$RES/job"
mkdir -p "$J" "$RES/plan" "$RES/plan-owed" "$RES/logs" "$CACHE_ROOT"
exec >> "$J/job.log" 2>&1
export PATH="/root/.pixi/bin:/usr/local/cuda/bin:/opt/rocm/bin:$PATH"
# the drivers use every core: no inherited thread caps (AGENTS.md "Full-machine CPU race resources")
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS
export MOJOLEARN_NUMERIC_MODE=identical
PY=${PC_PYTHON:-python3}
cd "$TREE" || exit 2
say() { echo "[$(date -u +%H:%M:%S) paper-compare-job] $*"; }
WORST=0
PHASE=start
P_T0=$(date +%s)
phase() { PHASE=$1; P_T0=$(date +%s); echo "$1" > "$J/.phase"; say "phase $PHASE start"; }
phase_end() {   # rc
    local rc=$1
    printf '%s\t%s\t%s\t%s\n' "$PHASE" "$rc" "$(( $(date +%s) - P_T0 ))" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$J/phases.tsv"
    say "phase $PHASE rc=$rc $(( $(date +%s) - P_T0 )) s"
    [ "$rc" -gt "$WORST" ] && WORST=$rc
    return 0
}
# busy cores of total every 60 s (job/cores.log) and a BUSY line in job.log every 5 minutes
cores_sampler() {
    local n prev cur i=0 line
    n=$(nproc)
    prev=$(awk '/^cpu /{print $2+$3+$4+$5+$6+$7+$8+$9, $5+$6}' /proc/stat)
    while sleep 60; do
        cur=$(awk '/^cpu /{print $2+$3+$4+$5+$6+$7+$8+$9, $5+$6}' /proc/stat)
        line=$(echo "$prev $cur" | awk -v n="$n" -v t="$(date -u +%H:%M:%S)" -v ph="$(cat "$J/.phase" 2>/dev/null)" \
            '{dt=$3-$1; di=$4-$2; b=(dt>0)?(dt-di)/dt*n:0; printf "%s busy=%.1f total=%d phase=%s", t, b, n, ph}')
        echo "$line" >> "$J/cores.log"
        i=$(( i + 1 )); [ $(( i % 5 )) -eq 0 ] && say "BUSY $line"
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
    printf '{"vendor":"%s","mode":"%s","version":"%s","source_commit":"%s","prior":"%s","tree_head":"%s","phase":"%s","exit":%d,"job_script_sha256":"%s","recipes_sha256":"%s"}\n' \
        "$V" "$MODE" "${CVER:-$VER}" "${PC_SOURCE_COMMIT:-}" "$PRIOR" "$(git -C "$TREE" rev-parse HEAD 2>/dev/null)" "$PHASE" \
        "$code" "$(sha256sum "$TREE/tools/ops/paper_compare_job.sh" | cut -d' ' -f1)" "$(sha256sum "$RECIPES" | cut -d' ' -f1)" \
        > "$J/status.json"
    say "exit $code (last phase $PHASE)"
    exit "$code"
}
trap finish EXIT
trap 'exit 124' TERM INT

say "paper compare $V $MODE ${PC_SOURCE_COMMIT:-$VER} at $(git rev-parse HEAD 2>/dev/null) results $RES parts $PARTS owed $OWED"
BOX_V=$(detect_vendor) || { say "neither nvidia-smi nor rocm-smi found"; exit 2; }
[ "$BOX_V" = "$V" ] || { say "asked for $V; this box is $BOX_V"; exit 3; }
{ uname -a; echo "nproc $(nproc)"; free -g 2> /dev/null || true; df -h /root 2> /dev/null || true; } > "$J/box.txt" 2>&1
say "BUSY at start: nproc $(nproc), $(awk '/MemAvailable/ {printf "%d GiB available", $2/1048576}' /proc/meminfo 2>/dev/null)"
if [ "$V" = nvidia ]; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > "$J/gpu.txt" 2>&1
else
    { rocm-smi --showproductname; rocm-smi --showmeminfo vram; } > "$J/gpu.txt" 2>&1
fi
NGPU=$(grep -c . "$J/gpu.txt" 2> /dev/null || echo 0)
[ "$V" = nvidia ] && [ "$NGPU" -gt 1 ] && say "NOTE: $NGPU GPUs; the timed boards use one (serial by nature)"
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

# ---- the candidate wheels: exactly the core and the vendor wheel of one version ---------------------------------
CAND=()
CVER=
if [ "$MODE" = candidate ]; then
    for w in "$PC_WHEEL_DIR"/mojolearn_core-*.whl "$PC_WHEEL_DIR"/mojolearn_"$V"-*.whl; do [ -f "$w" ] && CAND+=("$w"); done
    [ "${#CAND[@]}" -eq 2 ] || { say "PC_WHEEL_DIR $PC_WHEEL_DIR must hold one mojolearn_core and one mojolearn_$V wheel (found ${#CAND[@]})"; exit 4; }
    CVER=$(basename "${CAND[0]}" | cut -d- -f2)
    sha256sum "${CAND[@]}" > "$J/candidate-wheels.sha256"
    say "candidate $CVER of ${PC_SOURCE_COMMIT:0:12}: $(tr '\n' ' ' < "$J/candidate-wheels.sha256")"
fi

# ---- the box can make a venv (a fresh DO MI300X image lacks python3-venv: the AMD run of 2026-11-08 failed at once)
if ! "$PY" -m venv "$CACHE_ROOT/venv-probe" > /dev/null 2>&1; then
    say "python3 -m venv failed; installing python3-venv"
    v=$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
    # the image's package lists are stale (404s on security.ubuntu.com): update first
    DEBIAN_FRONTEND=noninteractive apt-get update -qq > "$RES/logs/apt-venv.log" 2>&1
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "python${v}-venv" >> "$RES/logs/apt-venv.log" 2>&1 || \
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv >> "$RES/logs/apt-venv.log" 2>&1
    rm -rf "$CACHE_ROOT"/venv-*/   # a venv made without ensurepip has no pip; never reuse it
    "$PY" -m venv "$CACHE_ROOT/venv-probe" > /dev/null 2>&1 || { say "no working venv module; see logs/apt-venv.log"; exit 4; }
fi
rm -rf "$CACHE_ROOT/venv-probe"
# the driver python itself needs numpy (tools/bench_board_evidence.py imports it in the shard process; the DO image's
# system python has none: AMD run c of 2026-11-08 ran every smoke race and then crashed at the evidence step)
if ! "$PY" -c 'import numpy' > /dev/null 2>&1; then
    say "driver python has no numpy; installing"
    "$PY" -m pip install -q --break-system-packages numpy > "$RES/logs/pip-numpy.log" 2>&1 || \
        "$PY" -m pip install -q numpy >> "$RES/logs/pip-numpy.log" 2>&1 || \
        { DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-numpy >> "$RES/logs/pip-numpy.log" 2>&1; }
    "$PY" -c 'import numpy' > /dev/null 2>&1 || { say "driver python still has no numpy; see logs/pip-numpy.log"; exit 4; }
fi

# ---- the arms: name -> install args (candidate or published) ---------------------------------------------------
ARMS=()                      # every arm with a venv on this box
declare -A ARM_ARGS ARM_PY ARM_MAN
add_arm() {   # name, install args...
    local name=$1; shift
    ARMS+=("$name"); ARM_ARGS[$name]="$*"
    ARM_PY[$name]="$CACHE_ROOT/venv-$name/bin/python"; ARM_MAN[$name]="$J/wheel-manifest-$name.json"
}
if [ "$MODE" = candidate ]; then
    a="--mojolearn-version $CVER --candidate-source-commit $PC_SOURCE_COMMIT"
    for w in "${CAND[@]}"; do a="$a --candidate-wheel $w"; done
    add_arm candidate $a
    [ "$OWED" = 1 ] && add_arm prior --mojolearn-version "$PRIOR"
else
    add_arm candidate --mojolearn-version "$VER"     # the published version under test (the "new" column)
fi
B_NAMES=()
if [ "$OWED" = 1 ] && [ -n "${PC_B_ARMS:-}" ]; then
    while read -r name dir commit; do
        [ -z "${name:-}" ] || [ "${name:0:1}" = "#" ] && continue
        [[ "$name" =~ ^[a-z0-9][a-z0-9.-]*$ ]] && [[ "${commit:-}" =~ ^[0-9a-f]{40}$ ]] || { say "bad PC_B_ARMS line: $name"; continue; }
        bw=(); for w in "$dir"/mojolearn_core-*.whl "$dir"/mojolearn_"$V"-*.whl; do [ -f "$w" ] && bw+=("$w"); done
        [ "${#bw[@]}" -eq 2 ] || { say "B set $name: $dir must hold one core and one $V wheel"; continue; }
        a="--mojolearn-version $(basename "${bw[0]}" | cut -d- -f2) --candidate-source-commit $commit"
        for w in "${bw[@]}"; do a="$a --candidate-wheel $w"; done
        add_arm "$name" $a; B_NAMES+=("$name")
    done < "$PC_B_ARMS"
fi
B_CSV=$(IFS=,; echo "${B_NAMES[*]:-}")

# ---- plan (CPU, seconds) -----------------------------------------------------------------------------------------
if part plan; then
    phase plan
    $PY tools/paper_compare.py ids --recipes "$RECIPES" --planner tools/bench_board.py --out-dir "$RES/plan" \
        > "$RES/logs/ids.log" 2>&1
    rc=$?; tail -n 3 "$RES/logs/ids.log"
    if [ "$OWED" = 1 ]; then
        $PY tools/measure_owed.py plan --owed "$OWED_FILE" --paper-recipes "$RECIPES" --planner tools/bench_board.py \
            --vendor "$V" --b-arms "$B_CSV" --out-dir "$RES/plan-owed" > "$RES/logs/owed-plan.log" 2>&1
        r=$?; tail -n 2 "$RES/logs/owed-plan.log"; [ $r -gt $rc ] && rc=$r
    fi
    phase_end $rc
fi
[ -s "$RES/plan/race-ids-full.txt" ] || { say "no paper plan in $RES/plan (run part plan)"; exit 5; }
# one board argument set for every process: the owed plan's union (paper + owed lanes), else the paper's
BA="$RES/plan/board-args.txt"; [ "$OWED" = 1 ] && [ -s "$RES/plan-owed/board-args.txt" ] && BA="$RES/plan-owed/board-args.txt"
mapfile -t BARGS < "$BA"
COMMON=("${BARGS[@]}" --data-root "$DATA_ROOT" --rounds 1 --evidence-local-only)
run_as() {   # arm: the board args that run on that arm's installed venv (no install)
    echo --python-env "${ARM_PY[$1]}" --skip-install --wheel-manifest "${ARM_MAN[$1]}" --mojolearn-version "$(arm_version "$1")"
}
arm_version() { set -- ${ARM_ARGS[$1]}; echo "$2"; }   # ARM_ARGS starts with --mojolearn-version <v>

# ---- install (CPU, every arm at once) + the dry run ------------------------------------------------------------
if part install; then
    phase install
    pids=(); names=()
    for arm in "${ARMS[@]}"; do
        # shellcheck disable=SC2086
        $PY tools/bench_board.py "${COMMON[@]}" ${ARM_ARGS[$arm]} --ours-install-only --install-only \
            --venv "$CACHE_ROOT/venv-$arm" --cache "$CACHE_ROOT/i-$arm" --wheel-manifest "${ARM_MAN[$arm]}" \
            --race-ids @"$RES/plan/race-ids-full.txt" --out "$RES/install-$arm" > "$RES/logs/install-$arm.log" 2>&1 &
        pids+=($!); names+=("$arm")
    done
    $PY tools/bench_board.py "${COMMON[@]}" --mojolearn-version "$(arm_version candidate)" --cache "$CACHE_ROOT/c1" \
        --race-ids @"$RES/plan/race-ids-full.txt" --dry-run > "$RES/plan/dry-run.log" 2>&1 &
    pids+=($!); names+=(dry-run)
    rc=0
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}"; r=$?
        say "install ${names[$i]} rc=$r $(grep -m 1 -h '^BOARD-INSTALL' "$RES/logs/install-${names[$i]}.log" 2>/dev/null)"
        [ $r -gt $rc ] && rc=$r
    done
    phase_end $rc
fi
[ -f "${ARM_MAN[candidate]}" ] || { say "the candidate arm has no wheel manifest (install failed: logs/install-candidate.log)"; exit 6; }

# ---- smoke (parallel shards on the one GPU; not timed) ----------------------------------------------------------
SMOKE_IDS="$RES/plan/race-ids-smoke-all.txt"
cat "$RES/plan/race-ids-smoke.txt" "$RES/plan-owed/smoke-ids.txt" 2> /dev/null | sort -u > "$SMOKE_IDS"
if part smoke; then
    phase smoke
    NRACE=$(grep -c . "$SMOKE_IDS")
    MEMG=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo 2> /dev/null || echo 16)
    S=${PC_SMOKE_SHARDS:-$(( $(nproc) / 4 ))}
    BYMEM=$(( MEMG / ${PC_SMOKE_MEM_GB:-12} ))
    if [ -z "${PC_SMOKE_SHARDS:-}" ]; then
        [ "$S" -gt "${PC_SMOKE_MAX:-8}" ] && S=${PC_SMOKE_MAX:-8}
        [ "$S" -gt "$BYMEM" ] && S=$BYMEM
    fi
    [ "$S" -gt "$NRACE" ] && S=$NRACE
    [ "$S" -lt 1 ] && S=1
    say "smoke: $NRACE races in $S parallel shards (nproc $(nproc), MemAvailable ${MEMG} GiB), one venv"
    pids=()
    # shellcheck disable=SC2046
    for i in $(seq 1 "$S"); do
        $PY tools/bench_board.py "${COMMON[@]}" $(run_as candidate) --cache "$CACHE_ROOT/c$i" --race-ids @"$SMOKE_IDS" \
            --smoke --shard "$i/$S" --out "$RES/smoke-s$i" > "$RES/logs/smoke-s$i.log" 2>&1 &
        pids+=($!)
    done
    src=0
    for p in "${pids[@]}"; do wait "$p"; r=$?; [ $r -gt $src ] && src=$r; done
    grep -h -E '^SMOKE (PASS|FAIL)' "$RES"/logs/smoke-s*.log | head -n 8
    phase_end $src     # a SMOKE FAIL exits 1: recorded, the passed races still go on
fi
SJ=()
for f in "$RES"/smoke-s*-smoke/smoke.json; do [ -f "$f" ] && SJ+=(--smoke-json "$f"); done
PASS_IDS="$RES/plan/race-ids-pass.txt"
FULL_ALL="$RES/plan/race-ids-full-all.txt"
cat "$RES/plan/race-ids-full.txt" "$RES/plan-owed/ids-candidate-full.txt" 2> /dev/null | sort -u > "$FULL_ALL"
$PY tools/paper_compare.py pass-ids --full-ids "$FULL_ALL" "${SJ[@]}" --out "$PASS_IDS" > "$RES/logs/pass-ids.log" 2>&1
tail -n 1 "$RES/logs/pass-ids.log"

# ---- the timed boards (one process at a time: serial by nature on one GPU) -------------------------------------
board_run() {   # arm, ids file, out dir, extra args...
    local arm=$1 ids=$2 out=$3; shift 3
    [ -s "$ids" ] || { say "board $arm $(basename "$ids"): no races"; return 0; }
    [ -f "${ARM_MAN[$arm]:-/nonexistent}" ] || { say "board $arm: no installed venv (logs/install-$arm.log)"; return 6; }
    # shellcheck disable=SC2046
    $PY tools/bench_board.py "${COMMON[@]}" $(run_as "$arm") --cache "$CACHE_ROOT/c1" --race-ids @"$ids" \
        "${SJ[@]}" "$@" --out "$out" > "$RES/logs/$(basename "$out").log" 2>&1
    local rc=$?
    say "board $arm $(basename "$ids") rc=$rc: $(tail -n 1 "$RES/logs/$(basename "$out").log" | head -c 200)"
    return $rc
}
if part board; then
    phase board
    if [ -s "$PASS_IDS" ]; then board_run candidate "$PASS_IDS" "$RES/board"; rc=$?; else say "no race passed its smoke"; rc=6; fi
    phase_end $rc
fi
if [ "$OWED" = 1 ] && part owed && [ -s "$RES/plan-owed/units.tsv" ]; then
    phase owed
    rc=0
    while IFS=$'\t' read -r arm rows ids; do
        [ "$arm" = candidate ] && [ "$rows" = full ] && continue   # timed with the paper races (part board)
        [ -n "${PC_OWED_ARMS:-}" ] && [[ ",$PC_OWED_ARMS," != *",$arm,"* ]] && continue   # a split submit
        if [ "$rows" = full ]; then
            # full rows: only the races whose smoke passed (the gate keys on family/lane/dataset, any venv)
            grep -Fx -f "$PASS_IDS" "$ids" > "$ids.pass" 2> /dev/null || true
            board_run "$arm" "$ids.pass" "$RES/board-$arm-$rows"; r=$?
        else
            board_run "$arm" "$ids" "$RES/board-$arm-$rows" --rows "$rows"; r=$?   # a neighbouring shape: no gate
        fi
        [ $r -gt $rc ] && rc=$r
    done < "$RES/plan-owed/units.tsv"
    phase_end $rc
fi
if [ "$OWED" = 1 ] && part drivers && [ -s "$RES/plan-owed/drivers.json" ]; then
    phase drivers
    mkdir -p "$RES/drivers"
    rc=0
    for row in $($PY -c 'import json,sys; print(" ".join(r["id"] for r in json.load(open(sys.argv[1]))))' "$RES/plan-owed/drivers.json"); do
        for arm in candidate prior; do
            [ -x "${ARM_PY[$arm]:-/nonexistent}" ] || continue
            "${ARM_PY[$arm]}" tools/measure_owed_driver.py --owed "$OWED_FILE" --row "$row" --arm "$arm" \
                --out "$RES/drivers/drivers.jsonl" >> "$RES/logs/drivers.log" 2>&1
            r=$?; [ $r -gt $rc ] && rc=$r
        done
    done
    grep -c '^OWED-DRIVER' "$RES/logs/drivers.log" 2> /dev/null | sed 's/^/driver cases: /'
    phase_end $rc
fi

# ---- compare ----------------------------------------------------------------------------------------------------
if part compare; then
    phase compare
    NR=()
    for f in "$RES/plan/not-planned.json" "$RES/plan/smoke-failed.json"; do [ -f "$f" ] && NR+=(--not-run "$f"); done
    PR=()
    [ "$MODE" = candidate ] && [ -f "$PRIOR_COMPARE" ] && PR=(--prior-compare "$PRIOR_COMPARE" --prior-label "$PRIOR")
    $PY tools/paper_compare.py compare --recipes "$RECIPES" --board "$RES/board/board.json" "${NR[@]}" "${PR[@]}" \
        --out-md "$RES/COMPARE-$V.md" --out-json "$RES/compare-$V.json" --summary "$RES/SUMMARY-$V.txt" \
        > "$RES/logs/compare.log" 2>&1
    rc=$?; head -n 3 "$RES/logs/compare.log"
    if [ "$OWED" = 1 ]; then
        $PY tools/measure_owed.py report --owed "$OWED_FILE" --plan-dir "$RES/plan-owed" --results "$RES" \
            --vendor "$V" ${PR:+--prior-compare "$PRIOR_COMPARE"} > "$RES/logs/owed-report.log" 2>&1
        r=$?; head -n 2 "$RES/logs/owed-report.log"; [ $r -gt $rc ] && rc=$r
    fi
    phase_end $rc
fi
echo finished > "$J/.phase"
exit "$WORST"
