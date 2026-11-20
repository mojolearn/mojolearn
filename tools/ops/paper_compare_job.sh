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
#   published  (no PC_WHEEL_DIR) the mlsys paper's 0.8.55 recipes on a release from PyPI, the version under test
#              given as the second argument (no default: the only prior is 0.8.55 and 0.8.66 is never timed,
#              Andrew 2026-11-16; harness-small-row) (tools/pypi_wheel_install.py through tools/bench_board.py
#              --mojolearn-version).
#   candidate  (PC_WHEEL_DIR set, or the word `candidate`) the exact candidate wheels in PC_WHEEL_DIR on the box
#              (mojolearn_core-<v>-*.whl and mojolearn_<vendor>-<v>-*.whl, staged by tools/ops/measure_candidate.py
#              stage) of source commit PC_SOURCE_COMMIT (40 hex), installed by tools/candidate_wheel_install.py
#              through bench_board --candidate-wheel: the wheel RECORDs, metadata, tags and the frozen payload's
#              source commit are checked, every file's sha256 lands in job/wheel-manifest-candidate.json and in
#              every board's box.mojolearn.wheel, and the compare is labelled "candidate <sha12> (<version>)".
#              The owed rows (bench/results/paper_compare/owed-pre0867.json, tools/measure_owed.py) run too
#              (PC_OWED=0 turns them off), ONLY those a report reads (tools/measure_owed.py plan): an A/B row when one
#              of its B wheel sets is listed in PC_B_ARMS (a file on the box, lines `<name> <wheel dir> <source
#              commit>`, names from the owed file's b_arms), a board row (candidate against the last release) only
#              when the job times a last release, a driver row when two of its arms run here.
# SAME BOX ONLY (AGENTS.md "Times are compared only on the same box", Andrew, 2026-11-14: two L40S hosts differed
# 1.2-2.2x on unchanged code and raised 56 false REGRESSION flags). In both modes the job installs OUR prior wheels
# from PyPI into their own ours-only venvs and TIMES every paper row for each, on this box in this job, serial on the
# GPU like the candidate (board-prior-<version>/). ONE PRIOR BY DEFAULT (Andrew, 2026-11-16: a measurement job times
# one prior, 0.8.55; 0.8.66 is never timed): PC_PRIOR_VERSIONS and PC_PRIOR_VERSION default to the paper baseline
# 0.8.55 (tools/paper_compare.py PAPER_VERSION). An explicit extra prior is still accepted: PC_PRIOR_VERSIONS lists
# more releases (each timed on the paper rows only), and a PC_PRIOR_VERSION other than 0.8.55 is the LAST RELEASE
# arm (put last, the compare's flagged prior column): only it also times the owed board rows' prior full rows in its
# board, and only it is the owed rows' "prior" arm. Each prior is planned and smoked only on the races it is compared
# on (the paper rows; the last release: plus the owed prior rows); a prior wheel that refuses a row is NOT-RUN in its
# column, never a regression. The compare's columns are those on-box
# boards (tools/paper_compare.py compare --prior-board <version>=<board.json>); the stored columns (the paper
# recipes' old_ms, bench/results/paper_compare/compare-<vendor>.json as --reference-compare, PC_REFERENCE_COMPARE)
# are reference only, with their host, and never make a flag. The job's first step is the host probe
# (tools/ops/host_probe.py: host id, CPU model, NUMA nodes, cgroup cores, memcpy GB/s, host-to-device GB/s pageable
# and pinned, PCIe generation and width during the copy, GPU and driver) in job/host-probe.json and job/box.txt;
# every board stores it (MOJOLEARN_HOST_PROBE -> bench_board box.host_probe and each race's host.host_id) and the
# compare refuses boards of two host ids. Budget: the prior boards double the board time (accepted, Andrew).
#
# ORDER (each phase logs its seconds to job/phases.tsv; fail soft: a failed phase is recorded and the job goes on
# where the next phase still has inputs; the exit status is the worst phase's)
#   probe    (every submit, seconds, before anything uses the GPU) tools/ops/host_probe.py -> job/host-probe.json
#   plan     (CPU, seconds) paper_compare ids + measure_owed plan: race ids per board process
#   install  (CPU, ALL AT ONCE: every arm's venv and the dry run in parallel) one ours-only venv per arm (bench_board --install-only --ours-install-only: our
#            wheels + scikit-learn + pandas, no opponent stack: the 0.8.66 smoke made a 13 GB venv per shard), each
#            checked against its wheel manifest and our GPU set once; plus the board's dry run
#   faults   (once per job, untimed, every flavour, right after the candidate's install) verify --self-test from the
#            candidate's wheel and the cheap sabotage checks on this GPU must each DETECT their fault
#            (box_job_lib.sh fault_gate, tools/ops/gpu_fault_gate.py; FAULTS-<vendor>.txt, status.json `faults`)
#   smoke    ALL ARMS AT ONCE (not timed): the candidate's paper + owed full-row races on the candidate venv and
#            every prior's own races on its venv (--python-env --skip-install: no per-shard install), in one set of
#            parallel shards on the one GPU split between the arms by race count; shards = min(alloc/2, MemAvailable
#            / PC_SMOKE_MEM_GB (default 12), NVIDIA GPU memory / PC_SMOKE_GPU_GB (default 3), PC_SMOKE_MAX (default
#            16), races); PC_SMOKE_SHARDS overrides. The shards are host-bound, not GPU-bound (2000-row races: worker
#            start, import, data read; see the smoke phase). After it each shard's work/ and evidence/ and the
#            caches other than c1 are deleted (BOARD.md, board.json, smoke.json and logs/ stay). A prior smoke runs
#            once per results dir (plan/prior-<v>-smoke.done)
#   board    TIMED, SERIAL BY NATURE: one board process at a time on the one GPU with nothing else running
#            (timing must not share the GPU; AGENTS.md "Full-machine CPU race resources"; the drivers themselves
#            use every core, no thread caps): candidate paper+owed full rows
#   prior    each prior wheel (PC_PRIOR_VERSIONS; its smoke ran in the smoke phase, or here first when that part was
#            not run), its TIMED board of the paper rows both it and the candidate passed (the last release: plus the
#            owed prior full rows), serial on the GPU, board-prior-<v>/
#   owed     each remaining owed unit (tools/measure_owed.py units.tsv: neighbouring --rows groups, the prior's
#            neighbouring shapes, the B arms), then the driver rows (tools/measure_owed_driver.py per arm interpreter)
#   compare  paper_compare compare --prior-board ... (COMPARE/compare/SUMMARY-<vendor>) and measure_owed report
#            (OWED*-<vendor>)
#   board-all (PC_PARTS=board-all; the full-board NVIDIA screen, Andrew 2026-11-13) instead of the above: every race
#            the board plans (tools/paper_compare.py ids-all), our arm only, smoke in the same parallel shards, one
#            timed board, each prior wheel smoked (with the candidate) and timed on the same races (all-prior), compare against the
#            on-box priors with the paper recipes and every row's best time ever (--best-ever) as reference; see the
#            board-all block below and docs/plans/PRE_0867_MEASUREMENT.md "Full-board NVIDIA screen"
#   targeted (PC_PARTS=targeted, PC_ROWS=<rows file>; Andrew 2026-11-17) instead of the above: only the listed rows
#            (tools/ops/targeted_rows.py), candidate and 0.8.55 interleaved by chunk, the staged B arms on their own
#            rows with their base back to back, the full board's compare for those rows; see the targeted block below
# PARALLEL PLAN (HARD RULE, 2026-11-07): what is independent runs at once across the cgroup's cores (alloc_cores:
# the cgroup quota, else nproc; the smoke shard count is taken from it, never from the host's visible CPUs): the installs
# of every arm and the dry run (install), the smokes of the candidate and of every prior (smoke, untimed, shards
# from smoke_shards), the plans (CPU, seconds). What is TIMED is serial by nature: one board process at a time on the
# one GPU with nothing else running (timing must not share the GPU, and the drivers themselves use every core).
# PC_PARTS (default all: plan,install,smoke,board,prior,owed,drivers,compare) runs a subset (the probe always runs)
# and PC_OWED_ARMS (comma list, default all) the owed units of some arms, so a job over the 240-minute
# nvidia_central.sh cap is split into submits on the same results dir (the boards resume from board.json; a later
# submit reruns install, which only re-checks an existing venv; a finished prior smoke is not rerun).
#
# It runs under tools/box_persist_run.sh (results to R2 from the box), launched by the supported wrappers
# (tools/ops/measure_candidate.py prints the exact lines; docs/plans/PRE_0867_MEASUREMENT.md):
#   tools/nvidia_central.sh submit <lane> --cap 240 --results box-out/<tag> --env PC_WHEEL_DIR=... \
#       --env PC_SOURCE_COMMIT=... tools/ops/paper_compare_job.sh
#   tools/do_amd_steward.sh run <tag> box-out/<tag> 'PC_WHEEL_DIR=... PC_SOURCE_COMMIT=... bash tools/ops/paper_compare_job.sh'
# and refuses to run unwrapped unless MOJOLEARN_ALLOW_UNPERSISTED=1.
#
# Results ($MOJOLEARN_RESULTS_DIR): board/ (the candidate's paper + owed full-row board), board-prior-<version>/
# (each prior wheel on this box), board-<arm>-<rows>/, smoke-s<i>-smoke/, smoke-prior-<version>-s<i>-smoke/, install-<arm>/, drivers/drivers.jsonl, plan/ (paper), plan-owed/, logs/ (one per process),
# job/ (job.log with a BUSY line every 5 minutes, phases.tsv, cores.log = busy cores of total every 60 s, gpu.txt,
# box.txt, host-probe.json, status.json, wheel-manifest-<arm>.json), COMPARE/SUMMARY/OWED files. Venvs and data blocks live under
# $PC_CACHE_ROOT (default /root/pc-cache), outside the persisted dir.
#
# Inputs on the box: the datasets under $PC_DATA_ROOT (default /root/datasets/gbm-bench), the list being
# tools/measure_owed.py data-keys (the board's data_needs over the paper and owed races; checked after the plan,
# exit 4 when one is missing), staged by tools/ops/measure_candidate.py stage through tools/dataset_store.sh stage.
# At start the job deletes the box_persist_run.sh upload leftovers of finished jobs. A phase that hits a full disk
# (ENOSPC in a log written during it, or under PC_DISK_MIN_FREE_MB left) stops the job with exit 28, and a smoke,
# board or owed phase refuses to start (exit 28) when the free disk is below its predicted evidence (disk_gate,
# printed as a DISK line; a timed board counts only its races not yet done in its board.json). Every timed board runs
# with --drop-race-scratch: each race's driver scratch (work/) goes once its record and evidence are written, the
# evidence stays (the 80 GB L40S pod filled during the d95f805cce full board, 2026-11-17); the prepared input blocks
# in the cache stay, because the prior boards later in this job read the same blocks.
# RESUME (2026-11-17): a job cut by the 240-minute cap ends its open phase with a `partial <done>/<total> <board>` line
# in job/phases.tsv (box_job_lib.sh phase_partial); the next submit says "resuming" and the boards resume from
# board.json. A reboot between submits (the boot id changed) is a `restart` line with both boot ids, drivers and
# kernels, and every race records its own (bench_board race_host): the compare gives no verdict across two. Knobs: PC_GPU_TOKEN (the paper's GPU,
# default L40S / MI300X). NVIDIA refuses another GPU (exit 3) unless PC_ALLOW_HW_DIFF=1; AMD records HW-DIFF and
# goes on unless PC_REQUIRE_SAME_GPU=1 (exit 3).
# Tests: tools/test_paper_compare.py (syntax and the argument checks of this script).
set -uo pipefail
# the shared box-job functions (detect_vendor, say, phase/phase_end, the disk gate, the busy-cores sampler, gpu_other,
# prune_persist_staging, host_probe_phase): tools/ops/box_job_lib.sh
JOB_NAME=paper-compare-job
# shellcheck source=tools/ops/box_job_lib.sh
source "$(dirname "$0")/box_job_lib.sh"

usage() { echo "usage: $0 [<nvidia|amd> [<version>|candidate]] | --detect-vendor" >&2; exit 2; }
if [ "${1:-}" = --detect-vendor ]; then
    detect_vendor || { echo "paper-compare-job: neither nvidia-smi nor rocm-smi found" >&2; exit 2; }
    exit 0
fi
# >>> pc-preflight (tools/test_job_preflight.py runs this block)
# job-preflight: `paper_compare_job.sh --preflight` checks the job's staged inputs at SUBMIT time, reading only (run on
# the box by tools/ops/job_preflight.sh from tools/nvidia_central.sh submit; lane fx-pod-job, 2026-11-18): the
# candidate wheel set (one core and one vendor wheel), its source commit, every B set of PC_B_ARMS, the rows file and
# the stored baseline board are where the job reads them. The job keeps its own run-time checks of the same inputs.
pc_preflight() {   # vendor: PREFLIGHT lines; 0 when the job would start, 4 otherwise
    local v=$1 fails=0 n name dir commit
    pf_fail() { echo "PREFLIGHT FAIL $*"; fails=$(( fails + 1 )); }
    wheels_in() { ls "$1"/mojolearn_core-*.whl "$1"/mojolearn_"$v"-*.whl 2> /dev/null | grep -c .; }
    if [ -n "${PC_WHEEL_DIR:-}" ]; then
        n=$(wheels_in "$PC_WHEEL_DIR")
        if [ "$n" -eq 2 ]; then echo "PREFLIGHT ok candidate wheels in $PC_WHEEL_DIR"
        else pf_fail "PC_WHEEL_DIR $PC_WHEEL_DIR holds $n of the 2 wheels (one mojolearn_core, one mojolearn_$v)"; fi
        [[ "${PC_SOURCE_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] || pf_fail "PC_SOURCE_COMMIT is not a 40-hex commit"
    fi
    if [[ ",${PC_PARTS:-}," == *",targeted,"* || ",${PC_PARTS:-}," == *",tg-"* ]]; then
        { [ -n "${PC_ROWS:-}" ] && [ -s "$PC_ROWS" ]; } && echo "PREFLIGHT ok rows file $PC_ROWS" \
            || pf_fail "PC_ROWS (the rows file on the box) is missing or empty: ${PC_ROWS:-unset}"
    fi
    if [ -n "${PC_BASELINE_BOARD:-}" ] && [ ! -s "$PC_BASELINE_BOARD" ]; then
        pf_fail "PC_BASELINE_BOARD $PC_BASELINE_BOARD: no such file on the box"
    fi
    if [ -n "${PC_B_ARMS:-}" ]; then
        if [ ! -s "$PC_B_ARMS" ]; then pf_fail "PC_B_ARMS $PC_B_ARMS: no such file on the box (or empty)"
        else
            while read -r name dir commit; do
                { [ -z "${name:-}" ] || [ "${name:0:1}" = "#" ]; } && continue
                [[ "${dir:-}" == env:* ]] && continue
                n=$(wheels_in "${dir:-/nonexistent}")
                [ "$n" -eq 2 ] || pf_fail "B set $name: ${dir:-?} holds $n of the 2 wheels"
            done < "$PC_B_ARMS"
            echo "PREFLIGHT ok B sets of $PC_B_ARMS checked"
        fi
    fi
    if [ -n "${PC_PRIOR_WHEELS:-}" ]; then
        if [ ! -s "$PC_PRIOR_WHEELS" ]; then pf_fail "PC_PRIOR_WHEELS $PC_PRIOR_WHEELS: no such file on the box (or empty)"
        else
            while read -r name dir commit; do
                { [ -z "${name:-}" ] || [ "${name:0:1}" = "#" ]; } && continue
                n=$(wheels_in "${dir:-/nonexistent}")
                [ "$n" -eq 2 ] || pf_fail "prior wheel $name: ${dir:-?} holds $n of the 2 wheels"
            done < "$PC_PRIOR_WHEELS"
            echo "PREFLIGHT ok prior wheels of $PC_PRIOR_WHEELS checked"
        fi
    fi
    return $(( fails > 0 ? 4 : 0 ))
}
# <<< pc-preflight
if [ "${1:-}" = --preflight ]; then
    pv=${2:-}
    [ -z "$pv" ] && { pv=$(detect_vendor) || { echo "PREFLIGHT FAIL neither nvidia-smi nor rocm-smi found"; exit 4; }; }
    pc_preflight "$pv"
    exit $?
fi
[ $# -le 2 ] || usage
V=${1:-}
ARG2=${2:-}
[ -z "$V" ] || case "$V" in nvidia|amd) ;; *) usage ;; esac
MODE=published
VER=
if [ "$ARG2" = candidate ] || { [ -z "$ARG2" ] && [ -n "${PC_WHEEL_DIR:-}" ]; }; then
    MODE=candidate
    VER=candidate
elif [ -n "$ARG2" ]; then
    VER=$ARG2
else
    # no default version (Andrew, 2026-11-16: the only prior is 0.8.55; 0.8.66 is never timed): published mode
    # measures the release named on the command line
    echo "paper-compare-job: published mode needs the version under test (<vendor> <version>), or PC_WHEEL_DIR" \
         "for a candidate" >&2
    exit 2
fi
[ "$MODE" = candidate ] || [[ "$VER" =~ ^[0-9]+(\.[0-9]+)+([.a-z0-9]*)$ ]] || {
    echo "paper-compare-job: bad version $VER" >&2; exit 2; }
# >>> prior-versions (tools/test_paper_compare_same_box.py runs this block)
# the paper's version (tools/paper_compare.py PAPER_VERSION; tools/test_paper_compare_same_box.py checks they agree)
PAPER_VERSION=0.8.55
PRIOR=${PC_PRIOR_VERSION:-$PAPER_VERSION}
[[ "$PRIOR" =~ ^[0-9]+(\.[0-9]+)+([.a-z0-9]*)$ ]] || { echo "paper-compare-job: bad PC_PRIOR_VERSION $PRIOR" >&2; exit 2; }
# OUR prior wheels re-timed on this box: by default the paper baseline only (Andrew, 2026-11-16). PRIOR is always in
# the list and always last. LAST_PRIOR is PRIOR when it is not the paper baseline (an explicit last release: the
# compare's flagged prior column and the owed rows' prior arm), else empty (no owed board row, no prior owed arm).
PRIOR_VERSIONS=
LAST_PRIOR=
[ "$PRIOR" != "$PAPER_VERSION" ] && LAST_PRIOR=$PRIOR
for pv in $(echo "${PC_PRIOR_VERSIONS:-$PAPER_VERSION,$PRIOR}" | tr ',' ' ') "$PRIOR"; do
    [[ "$pv" =~ ^[0-9]+(\.[0-9]+)+([.a-z0-9]*)$ ]] || { echo "paper-compare-job: bad PC_PRIOR_VERSIONS entry $pv" >&2; exit 2; }
    [ "$pv" = "$PRIOR" ] && continue
    [[ ",$PRIOR_VERSIONS," == *",$pv,"* ]] || PRIOR_VERSIONS="${PRIOR_VERSIONS:+$PRIOR_VERSIONS,}$pv"
done
PRIOR_VERSIONS="${PRIOR_VERSIONS:+$PRIOR_VERSIONS,}$PRIOR"
# <<< prior-versions
# THE STORED BASELINE (Andrew, 2026-11-17, AGENTS.md part 8 "0.8.55 is not timed again"): PC_BASELINE_BOARD = the
# 0.8.55 board of the 2026-11-17 L40S full board, staged on the box by tools/ops/measure_candidate.py stage --rows.
# The targeted job then times OUR new wheel only: no 0.8.55 venv, smoke or races; the compare reads the stored board
# (tools/paper_compare.py --stored-baseline-board: host and driver differences are WARNING lines, BOX-STATE rows).
BASELINE_BOARD=${PC_BASELINE_BOARD:-}
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
# a stored compare of an earlier run (another host): a reference column only, never a flag
REF_COMPARE=${PC_REFERENCE_COMPARE:-$PCDIR/compare-$V.json}
[ -f "$RECIPES" ] || { echo "paper-compare-job: no $RECIPES" >&2; exit 2; }
OWED=${PC_OWED:-$([ "$MODE" = candidate ] && echo 1 || echo 0)}
PARTS=",${PC_PARTS:-plan,install,smoke,board,prior,owed,drivers,compare},"
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
SAMPLER=
if [ -r /proc/stat ]; then cores_sampler & SAMPLER=$!; fi
# >>> cap-guard (tools/test_targeted_rows.py runs this block)
# ---- the CAP GUARD (lane fb15-ab-verdicts, 2026-11-18): the queue kills a job at its wall-clock cap (TERM, then
# KILL 40 s later), too soon for box_persist_run.sh's final upload of a multi-GB results dir and before the compare;
# the ab1 A/B job of d652a4b0a1 was cut that way and left no SUMMARY file. PC_CAP_GUARD_S (default 600) before the
# deadline (GQ_DEADLINE, exported by tools/gpu_queue_box.sh for a queued job, else the job's start + PC_CAP_MIN
# when that is set; a job with neither has no cap and no guard) the guard marks the job capped and stops the timed
# board in flight (its process group); no new timed board starts (run_timed), the compare and summary parts run on
# what is timed, the job exits, and the final upload has the rest of the time. The races already in board.json
# stay (each board resumes on the next submit).
CAP_DEADLINE=${GQ_DEADLINE:-}
[ -z "$CAP_DEADLINE" ] && [ -n "${PC_CAP_MIN:-}" ] && CAP_DEADLINE=$(( P_T0 + PC_CAP_MIN * 60 ))
rm -f "$J/.capped" "$J/.board.pgid"
capped() { [ -e "$J/.capped" ]; }
cap_guard() {
    local now pg CAP_AT=$(( CAP_DEADLINE - ${PC_CAP_GUARD_S:-600} ))
    now=$(date +%s)
    [ "$CAP_AT" -gt "$now" ] && sleep $(( CAP_AT - now ))
    : > "$J/.capped"
    say "CAP GUARD: ${PC_CAP_GUARD_S:-600} s before the cap: no new timed board; the board in flight stops; compare, summary, final upload"
    pg=$(cat "$J/.board.pgid" 2> /dev/null)
    if [ -n "$pg" ]; then
        kill -TERM -- "-$pg" 2> /dev/null
        for _ in $(seq 30); do kill -0 -- "-$pg" 2> /dev/null || return 0; sleep 1; done
        kill -KILL -- "-$pg" 2> /dev/null
    fi
    return 0
}
CAP_GUARD=
if [ -n "$CAP_DEADLINE" ]; then cap_guard & CAP_GUARD=$!; fi
# one TIMED bench_board process in its own process group (the cap guard stops the group); refused once capped.
# Usage: run_timed <log> <append 0|1> <command...>
run_timed() {
    local log=$1 app=$2 bp rc
    shift 2
    capped && { say "not run (cap guard): $(basename "$log" .log)"; return 124; }
    if [ "$app" = 1 ]; then setsid "$@" >> "$log" 2>&1 & else setsid "$@" > "$log" 2>&1 & fi
    bp=$!; echo "$bp" > "$J/.board.pgid"
    wait "$bp"; rc=$?
    rm -f "$J/.board.pgid"
    return $rc
}
# <<< cap-guard
finish() {
    local code=$?
    trap - EXIT
    [ -n "$SAMPLER" ] && kill "$SAMPLER" 2> /dev/null
    [ -n "${CAP_GUARD:-}" ] && { pkill -P "$CAP_GUARD" 2> /dev/null; kill "$CAP_GUARD" 2> /dev/null; }
    [ "$WORST" -gt "$code" ] && code=$WORST
    phase_partial "$code"     # a phase the cap or a crash cut: its `partial` line in phases.tsv (box_job_lib.sh)
    printf '{"vendor":"%s","mode":"%s","version":"%s","source_commit":"%s","prior":"%s","prior_versions":"%s","host_id":"%s","tree_head":"%s","phase":"%s","exit":%d,"job_script_sha256":"%s","recipes_sha256":"%s","faults":"%s"}\n' \
        "$V" "$MODE" "${CVER:-$VER}" "${PC_SOURCE_COMMIT:-}" "$PRIOR" "$PRIOR_VERSIONS" "${HOST_ID:-}" "$(git -C "$TREE" rev-parse HEAD 2>/dev/null)" "$PHASE" \
        "$code" "$(sha256sum "$TREE/tools/ops/paper_compare_job.sh" | cut -d' ' -f1)" "$(sha256sum "$RECIPES" | cut -d' ' -f1)" \
        "$(printf '%s' "${FAULTS_LINE:-not run}" | tr -d '"\\')" \
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
say "BUSY at start: nproc $(nproc), allocation $(alloc_cores) cores, $(awk '/MemAvailable/ {printf "%d GiB available", $2/1048576}' /proc/meminfo 2>/dev/null)"
if [ "$V" = nvidia ]; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > "$J/gpu.txt" 2>&1
else
    { rocm-smi --showproductname; rocm-smi --showmeminfo vram; } > "$J/gpu.txt" 2>&1
fi
NGPU=$(grep -c . "$J/gpu.txt" 2> /dev/null || echo 0)
[ "$V" = nvidia ] && [ "$NGPU" -gt 1 ] && say "NOTE: $NGPU GPUs; the timed boards use one (serial by nature)"
if gpu_other; then
    say "HW-DIFF: the paper ran on $TOKEN; this box: $(head -c 300 "$J/gpu.txt" | tr '\n' ' ')"
    if [ "$V" = nvidia ] && [ "${PC_ALLOW_HW_DIFF:-0}" != 1 ]; then
        say "refused: NVIDIA is measured on an $TOKEN only; PC_ALLOW_HW_DIFF=1 runs it anyway with every row HW-DIFF"
        exit 3
    fi
    [ "$V" = amd ] && [ "${PC_REQUIRE_SAME_GPU:-0}" = 1 ] && exit 3
fi
prune_persist_staging

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

# ---- numactl on a multi-node box (lane fx-harness, 2026-11-18): race workers start bound to the GPU's NUMA node,
# cores and memory (tools/numa_place.py); without numactl only the cores are bound (taskset) and the cells say so.
# Best effort: a failed install never stops the job.
if [ "$(ls -d /sys/devices/system/node/node[0-9]* 2> /dev/null | wc -l)" -gt 1 ] && ! command -v numactl > /dev/null 2>&1; then
    { DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 install -y -qq --no-install-recommends numactl \
        || { DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 update -qq \
             && DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 install -y -qq --no-install-recommends numactl; }
    } > "$RES/logs/apt-numactl.log" 2>&1
    say "numactl: $(command -v numactl || echo 'not installed (logs/apt-numactl.log): workers bound by taskset, memory unbound')"
fi

# ---- the host probe (every submit, before anything else uses the GPU): tools/ops/box_job_lib.sh host_probe_phase
host_probe_phase
arith_gate   # tools/ops/box_job_lib.sh: identical-path arithmetic on this GPU, before any timing

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
else
    add_arm candidate --mojolearn-version "$VER"     # the published version under test (the "new" column)
fi
# OUR prior wheels from PyPI, timed on this box in this job (the compare's columns): prior-<version>
PRIOR_ARMS=()
if [ -n "$BASELINE_BOARD" ]; then
    [[ "$PARTS" == *",targeted,"* || "$PARTS" == *",tg-"* ]] || { say "PC_BASELINE_BOARD is for the targeted job (PC_PARTS=targeted)"; exit 2; }
    [ -s "$BASELINE_BOARD" ] || { say "PC_BASELINE_BOARD $BASELINE_BOARD: no such file on the box (measure_candidate.py stage uploads it)"; exit 2; }
    say "baseline: STORED $BASELINE_BOARD ($(sha256sum "$BASELINE_BOARD" | cut -c1-16)); no prior venv, smoke or races in this job"
fi
for pv in ${PRIOR_VERSIONS//,/ }; do
    [ -n "$BASELINE_BOARD" ] && break                                 # the stored baseline: no prior arm at all
    [ "$MODE" = published ] && [ "$pv" = "$VER" ] && continue      # the version under test is the new column
    add_arm "prior-$pv" --mojolearn-version "$pv"; PRIOR_ARMS+=("prior-$pv")
done
# the owed rows name the last release "prior" (owed-pre0867.json, measure_owed units, measure_owed_driver --arm):
# LAST_PRIOR's venv, none when the job times no last release (the default: the paper baseline only)
venv_arm() { if [ "$1" = prior ]; then echo "prior-${LAST_PRIOR:-none}"; else echo "$1"; fi; }
# run_owed_drivers <drivers.json>: every planned driver row (tools/measure_owed_driver.py) on each arm present on this
# box, into $RES/drivers/drivers.jsonl; returns the worst rc. A driver row's B arms (its "arms" names a staged B set,
# e.g. restore-mamba-backward-glue's mamba-backward-generic) run on that set's venv; measure_owed_driver.py skips an
# arm the row does not name. Driver cases time one GPU each, so they run one after the other (cells serially per GPU).
# >>> owed-drivers (tools/test_targeted_rows.py runs this block with the tg-drivers one)
run_owed_drivers() {
    local plan=$1 row arm va r rc=0
    mkdir -p "$RES/drivers"
    for row in $($PY -c 'import json,sys; print(" ".join(r["id"] for r in json.load(open(sys.argv[1]))))' "$plan"); do
        for arm in candidate ${LAST_PRIOR:+prior} ${B_NAMES[@]+"${B_NAMES[@]}"}; do
            va=$(venv_arm "$arm")
            [ -x "${ARM_PY[$va]:-/nonexistent}" ] || continue
            local denv=(); [ -n "${ARM_XENV[$arm]:-}" ] && IFS=, read -r -a denv <<< "${ARM_XENV[$arm]}"   # a wheel arm's run variables
            env ${denv[@]+"${denv[@]}"} "${ARM_PY[$va]}" tools/measure_owed_driver.py --owed "$OWED_FILE" --row "$row" --arm "$arm" \
                --out "$RES/drivers/drivers.jsonl" >> "$RES/logs/drivers.log" 2>&1
            r=$?; [ $r -gt $rc ] && rc=$r
        done
    done
    grep -c '^OWED-DRIVER' "$RES/logs/drivers.log" 2> /dev/null | sed 's/^/driver cases: /'
    return $rc
}
# <<< owed-drivers
OWED_PRIOR=(); [ -n "$LAST_PRIOR" ] && OWED_PRIOR=(--prior)   # measure_owed plans the owed prior arm only then
# >>> b-arms (tools/test_measure_owed.py runs this block on a fake PC_B_ARMS file)
# A line is `<name> <wheel dir> <commit>` (a B wheel set built from another commit) or `<name> env:VAR=VALUE[,VAR=VALUE]
# <commit>` (an ENV-ONLY arm, no rebuild: the candidate's own venv run with those variables set, e.g.
# MOJOLEARN_CPU_THREADS=@cgroup_cores (the box's allocation) or MOJOLEARN_XPREP_R3_OFF=view; the targeted job only).
# ARM_ENV holds an env arm's variables.
# A wheel-set line may carry a 4th field `env:VAR=VALUE[,VAR=VALUE]`: variables that arm's own venv needs to RUN (lane
# fx-verdicts5, 2026-11-20: the gemm-leaf512 set reports GEMM profile version 4 and mojolearn.linalg admits an
# experimental profile only under MOJOLEARN_EXPERIMENT_GEMM_PROFILE=<its name>; without it A/B round 3 refused the
# arm's gemm row and its four matmul driver cases). ARM_XENV holds them; the board units and the driver rows of that
# arm run with them set. It is still a wheel arm (installed, its own venv), unlike an ARM_ENV arm.
B_NAMES=()
declare -A ARM_ENV
declare -A ARM_XENV
# wheel_set_args <name> <wheel dir> <commit>: the install args of a wheel set built from COMMIT (one core and one $V
# wheel in DIR), or a say line and rc 1. A B set and a prior wheel (PC_PRIOR_WHEELS, below) are installed the same way.
wheel_set_args() {
    local w bw=() a
    for w in "$2"/mojolearn_core-*.whl "$2"/mojolearn_"$V"-*.whl; do [ -f "$w" ] && bw+=("$w"); done
    # stderr: the caller captures stdout (the args); the job's log holds both
    [ "${#bw[@]}" -eq 2 ] || { say "wheel set $1: $2 must hold one core and one $V wheel" >&2; return 1; }
    a="--mojolearn-version $(basename "${bw[0]}" | cut -d- -f2) --candidate-source-commit $3"
    for w in "${bw[@]}"; do a="$a --candidate-wheel $w"; done
    echo "$a"
}
if [ "$OWED" = 1 ] && [ -n "${PC_B_ARMS:-}" ]; then
    [ -s "$PC_B_ARMS" ] || say "PC_B_ARMS=$PC_B_ARMS: no such file on the box (or empty); no B arm runs"
    while read -r name dir commit xenv; do
        [ -z "${name:-}" ] || [ "${name:0:1}" = "#" ] && continue
        [[ "$name" =~ ^[a-z0-9][a-z0-9.-]*$ ]] && [[ "${commit:-}" =~ ^[0-9a-f]{40}$ ]] || { say "bad PC_B_ARMS line: $name"; continue; }
        if [[ "$dir" == env:* ]]; then
            [[ "$PARTS" == *",targeted,"* || "$PARTS" == *",tg-"* ]] || { say "env arm $name: the targeted job only (PC_PARTS=targeted)"; continue; }
            [[ "${dir#env:}" =~ ^[A-Z_][A-Z0-9_]*=[^,[:space:]]*(,[A-Z_][A-Z0-9_]*=[^,[:space:]]*)*$ ]] || { say "bad env arm $name: $dir"; continue; }
            add_arm "$name" ${ARM_ARGS[candidate]}
            ARM_PY[$name]=${ARM_PY[candidate]}; ARM_MAN[$name]=${ARM_MAN[candidate]}; ARM_ENV[$name]=${dir#env:}
            # a value taken from the box (lane fb15-ab-verdicts, 2026-11-18: the cpu-threads-28 arm ran =28 on a
            # 14-core cgroup, 2x oversubscribed, and measured nothing): @cgroup_cores is this box's allocation
            if [[ "${ARM_ENV[$name]}" == *@cgroup_cores* ]]; then
                ARM_ENV[$name]=${ARM_ENV[$name]//@cgroup_cores/$(alloc_cores)}
                say "env arm $name: ${ARM_ENV[$name]} (@cgroup_cores = this box's allocation)"
            fi
            B_NAMES+=("$name")
            continue
        fi
        if [ -n "${xenv:-}" ]; then
            [[ "$xenv" == env:* ]] && [[ "${xenv#env:}" =~ ^[A-Z_][A-Z0-9_]*=[^,[:space:]]*(,[A-Z_][A-Z0-9_]*=[^,[:space:]]*)*$ ]] || { say "bad env field of wheel arm $name: $xenv"; continue; }
        fi
        a=$(wheel_set_args "$name" "$dir" "$commit") || continue
        add_arm "$name" $a; B_NAMES+=("$name")
        if [ -n "${xenv:-}" ]; then ARM_XENV[$name]=${xenv#env:}; say "wheel arm $name runs with ${ARM_XENV[$name]}"; fi
    done < "$PC_B_ARMS"
fi
B_CSV=$(IFS=,; echo "${B_NAMES[*]:-}")
# <<< b-arms
# >>> prior-wheels (tools/test_measure_owed.py runs this block with the b-arms one)
# OUR OWN EARLIER BUILT WHEELS beside the candidate (lane fx-same-pod-prior; Andrew, 2026-11-19: a speed verdict compares
# two wheels timed on the SAME host in the SAME job; the stored 0.8.55 board is a screen only). PC_PRIOR_WHEELS (a file
# on the box, lines `<name> <wheel dir> <commit>`, measure_candidate.py stage --prior-wheel) is installed exactly as a
# B set (wheel_set_args) and becomes the prior arm prior-<name>: the targeted job installs it in tg-install with the
# candidate, smokes it with the candidate, times every listed row on it chunk by chunk right after the candidate (our
# GPU arm only, one scored run, the same small-row median rule: the same board process), and the compare takes its
# board as the verdict baseline (paper_compare.py --verdict-prior <first name>), the stored board as a screen column.
PRIOR_WHEEL_NAMES=()
if [ -n "${PC_PRIOR_WHEELS:-}" ]; then
    if [[ "$PARTS" != *",targeted,"* && "$PARTS" != *",tg-"* ]]; then
        say "PC_PRIOR_WHEELS is for the targeted job (PC_PARTS=targeted)"; exit 2
    fi
    [ -s "$PC_PRIOR_WHEELS" ] || { say "PC_PRIOR_WHEELS=$PC_PRIOR_WHEELS: no such file on the box (or empty)"; exit 2; }
    while read -r name dir commit; do
        { [ -z "${name:-}" ] || [ "${name:0:1}" = "#" ]; } && continue
        [[ "$name" =~ ^[a-z0-9][a-z0-9.-]*$ ]] && [[ "${commit:-}" =~ ^[0-9a-f]{40}$ ]] || { say "bad PC_PRIOR_WHEELS line: $name"; exit 2; }
        a=$(wheel_set_args "$name" "$dir" "$commit") || exit 2
        add_arm "prior-$name" $a; PRIOR_ARMS+=("prior-$name"); PRIOR_WHEEL_NAMES+=("$name")
    done < "$PC_PRIOR_WHEELS"
    say "prior wheels (the verdict baseline: ${PRIOR_WHEEL_NAMES[0]:-none}): ${PRIOR_WHEEL_NAMES[*]:-none}"
fi
# <<< prior-wheels

run_as() {   # arm: the board args that run on that arm's installed venv (no install)
    echo --python-env "${ARM_PY[$1]}" --skip-install --wheel-manifest "${ARM_MAN[$1]}" --mojolearn-version "$(arm_version "$1")"
}
arm_version() { set -- ${ARM_ARGS[$1]}; echo "$2"; }   # ARM_ARGS starts with --mojolearn-version <v>
# the parallel smoke shard count for NRACE races (see the smoke phase for why it is host-bound): min(alloc_cores/2,
# MemAvailable / PC_SMOKE_MEM_GB (12), NVIDIA GPU GB / PC_SMOKE_GPU_GB (3), PC_SMOKE_MAX (16), races);
# PC_SMOKE_SHARDS overrides. Sets S, MEMG and GPUG.
smoke_shards() {
    local nrace=$1 bymem
    MEMG=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo 2> /dev/null || echo 16)
    S=${PC_SMOKE_SHARDS:-$(( $(alloc_cores) / 2 ))}     # the cgroup allocation, not the host's visible CPUs
    bymem=$(( MEMG / ${PC_SMOKE_MEM_GB:-12} ))
    GPUG=0
    if [ "$V" = nvidia ]; then
        GPUG=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2> /dev/null | head -1 | awk '{print int($1 / 1024)}')
    fi
    if [ -z "${PC_SMOKE_SHARDS:-}" ]; then
        [ "$S" -gt "${PC_SMOKE_MAX:-16}" ] && S=${PC_SMOKE_MAX:-16}
        [ "$S" -gt "$bymem" ] && S=$bymem
        [ "${GPUG:-0}" -gt 0 ] && [ "$S" -gt $(( GPUG / ${PC_SMOKE_GPU_GB:-3} )) ] && S=$(( GPUG / ${PC_SMOKE_GPU_GB:-3} ))
    fi
    [ "$S" -gt "$nrace" ] && S=$nrace
    [ "$S" -lt 1 ] && S=1
    return 0
}

# ---- the smokes (untimed, ALL ARMS AT ONCE) and the prior wheels' timed boards (AGENTS.md "Times are compared only
# on the same box", 2026-11-14) ---------------------------------------------------------------------------------------
# CARGS is the flow's board argument set (COMMON or COMMON_ALL), set before these run. PREFIX is "" (paper flow) or
# "all-" (board-all). PRIOR_SMOKE_IDS[<arm>] is the smoke ids file of each prior arm: the races it is compared on.
declare -A PRIOR_SMOKE_IDS
SMOKE_PIDS=(); SMOKE_PID_ARM=()
smoke_launch() {   # arm, ids, out base, shards, cache tag, extra smoke args...: the arm's shards in the background
    local arm=$1 ids=$2 base=$3 n=$4 tag=$5 i
    shift 5
    # shellcheck disable=SC2046
    for i in $(seq 1 "$n"); do
        $PY tools/bench_board.py "${CARGS[@]}" $(run_as "$arm") --cache "$CACHE_ROOT/c$tag$i" --race-ids @"$ids" \
            --smoke "$@" --shard "$i/$n" --out "$base-s$i" > "$RES/logs/$(basename "$base")-s$i.log" 2>&1 &
        SMOKE_PIDS+=($!); SMOKE_PID_ARM+=("$arm")
    done
}
# smoke_together <prefix> <candidate ids or ""> <candidate out base> [extra smoke args...]: the candidate's smoke and
# the smoke of every prior arm not yet smoked in this results dir (plan/<prefix><arm>-smoke.done), at once, in ONE set
# of smoke_shards shards on the one GPU split between the arms by race count (at least one each). Untimed, so they
# overlap; before 2026-11-16 each prior's smoke ran alone after the candidate's 3.6 h timed board. A row a prior
# refuses there is NOT-RUN in its column. Returns the worst shard rc; prints the arms' SMOKE lines.
smoke_together() {
    local pre=$1 cids=$2 cbase=$3 total=0 n k=0 i r src=0 arm d c
    local -a arms=() counts=()
    local -A arm_rc=()
    shift 3
    if [ -n "$cids" ]; then n=$(nlines "$cids"); [ "$n" -gt 0 ] && { arms+=(candidate); counts+=("$n"); total=$(( total + n )); }; fi
    for arm in ${PRIOR_ARMS[@]+"${PRIOR_ARMS[@]}"}; do
        [ -f "$RES/plan/${pre}${arm}-smoke.done" ] && { say "smoke $arm: done in an earlier submit ($(cat "$RES/plan/${pre}${arm}-smoke.done"))"; continue; }
        [ -f "${ARM_MAN[$arm]:-/nonexistent}" ] || { say "smoke $arm: no installed venv (logs/install-$arm.log); its column is NOT-RUN"; src=6; continue; }
        n=$(nlines "${PRIOR_SMOKE_IDS[$arm]:-/nonexistent}")
        [ "$n" -gt 0 ] || { say "smoke $arm: no races"; continue; }
        arms+=("$arm"); counts+=("$n"); total=$(( total + n ))
    done
    [ "$total" -gt 0 ] || return $src
    smoke_shards "$total"
    disk_gate "${pre}smoke" "$total" "$S" "${PC_SMOKE_EVIDENCE_MB_PER_RACE:-5}"
    SMOKE_PIDS=(); SMOKE_PID_ARM=()
    for i in "${!arms[@]}"; do
        arm=${arms[$i]}
        n=$(( S * counts[i] / total )); [ "$n" -lt 1 ] && n=1; [ "$n" -gt "${counts[$i]}" ] && n=${counts[$i]}
        say "smoke $arm: ${counts[$i]} races in $n of $S parallel shards (all arms at once; allocation $(alloc_cores) of nproc $(nproc), MemAvailable ${MEMG} GiB, GPU ${GPUG:-?} GiB)"
        if [ "$arm" = candidate ]; then
            smoke_launch candidate "$cids" "$cbase" "$n" "" "$@"
        else
            k=$(( k + 1 )); smoke_launch "$arm" "${PRIOR_SMOKE_IDS[$arm]}" "$RES/smoke-$pre$arm" "$n" "p$k-" "$@"
        fi
    done
    for i in "${!SMOKE_PIDS[@]}"; do
        wait "${SMOKE_PIDS[$i]}"; r=$?; arm=${SMOKE_PID_ARM[$i]}
        [ "$r" -gt "${arm_rc[$arm]:-0}" ] && arm_rc[$arm]=$r
        [ "$r" -gt "$src" ] && src=$r
    done
    for arm in "${arms[@]}"; do
        [ "$arm" = candidate ] && continue
        echo "rc=${arm_rc[$arm]:-0} together-with=${arms[*]} $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RES/plan/${pre}${arm}-smoke.done"
    done
    grep -h -E '^SMOKE (PASS|FAIL)' "$RES"/logs/smoke-*s[0-9]*.log 2> /dev/null | head -n 8
    # the smoke scratch is not needed past this point: each shard keeps BOARD.md, board.json, smoke.json and logs/
    # (what pass-ids and the persisted record read); its work/ and evidence/ go, and every shard cache but c1 (the
    # timed boards use c1). The 80 GB L40S pod filled with 12 GB of smoke scratch on 2026-11-13.
    for d in "$RES"/smoke-*s[0-9]*/; do [ -d "$d" ] && rm -rf "${d}work" "${d}evidence"; done
    for c in "$CACHE_ROOT"/c*/; do case "$(basename "$c")" in c1) ;; c[0-9]*|cp[0-9]*) rm -rf "$c" ;; esac; done
    say "smoke scratch pruned; $(df -h /root 2> /dev/null | awk 'NR==2 {print $4 " free on /root"}')"
    return $src
}
prior_smoke_args() {   # arm, prefix: the --smoke-json args of the arm's own smoke
    local f
    for f in "$RES"/smoke-"$2$1"-s*-smoke/smoke.json; do [ -f "$f" ] && printf -- '--smoke-json\n%s\n' "$f"; done
    return 0
}
prior_board_run() {   # arm, the ids the candidate passed (full), out dir, prefix: the arm's TIMED board, serial on
    # the GPU, of the races its own smoke passed too (plan/<prefix><arm>/race-ids-pass.txt); resumes from board.json
    local arm=$1 want=$2 out=$3 pre=$4 ids rc psj=()
    mapfile -t psj < <(prior_smoke_args "$arm" "$pre")
    ids="$RES/plan/$pre$arm/race-ids-pass.txt"
    mkdir -p "$(dirname "$ids")"
    $PY tools/paper_compare.py pass-ids --full-ids "$want" "${psj[@]}" --out "$ids" > "$RES/logs/pass-ids-$pre$arm.log" 2>&1
    say "$arm: $(tail -n 1 "$RES/logs/pass-ids-$pre$arm.log")"
    # each race the prior's own smoke failed, with its reason (the 23 0.8.55 rows of the d95f805cce full board reached
    # the Mac as a count only): job.log, and plan/<prefix><arm>/smoke-failed.json in full
    grep -m 40 '^PAPER-PASS-IDS-FAILED' "$RES/logs/pass-ids-$pre$arm.log" | cut -c1-300
    [ -s "$ids" ] || { say "board $arm: no race passed both smokes; its column is NOT-RUN"; return 0; }
    [ -f "${ARM_MAN[$arm]:-/nonexistent}" ] || { say "board $arm: no installed venv (logs/install-$arm.log)"; return 6; }
    disk_gate "board-$pre$arm" "$(races_left "$ids" "$out/board.json")" 1 "${PC_EVIDENCE_MB_PER_RACE:-40}"
    CUR_IDS=$ids; CUR_BOARD=$out
    # shellcheck disable=SC2046
    $PY tools/bench_board.py "${CARGS[@]}" $(run_as "$arm") --cache "$CACHE_ROOT/c1" --race-ids @"$ids" \
        "${psj[@]}" --drop-race-scratch --out "$out" > "$RES/logs/$(basename "$out").log" 2>&1
    rc=$?; CUR_IDS=
    say "board $arm rc=$rc: $(tail -n 1 "$RES/logs/$(basename "$out").log" | head -c 200)"
    return $rc
}
prior_board_args() {   # prefix ("" or "all-"): the compare's --prior-board args, one per line, PRIOR last
    local arm
    for arm in ${PRIOR_ARMS[@]+"${PRIOR_ARMS[@]}"}; do
        printf -- '--prior-board\n%s=%s\n' "${arm#prior-}" "$RES/board-${1}${arm}/board.json"
    done
    return 0
}

# ---- board-all: the full-board screen (Andrew, 2026-11-13) ------------------------------------------------------
# PC_PARTS=board-all (= all-plan,all-install,all-smoke,all-board,all-compare; each can be given alone to split a
# job over the 240-minute cap: the board resumes from board-all/board.json) runs EVERY race the board plans on this
# vendor (tools/paper_compare.py ids-all: every family and lane, identical, our arms only; opponents are the stored
# cells, never raced) instead of the paper flow, and compares the result against the paper recipes AND every row's
# best time ever (tools/paper_compare.py compare --best-ever: bench/results/board_best_ever/<vendor>.json, mined by
# tools/board_best_ever.py). The smoke runs in the smoke phase's parallel shards (smoke_shards); the timed board is
# one process on the one GPU, serial by nature (timing must not share the GPU). Results: plan-all/, install-all/,
# smoke-all-s<i>/, board-all/, COMPARE-ALL/compare-all/SUMMARY-ALL-<vendor>.
[[ "$PARTS" == *",board-all,"* ]] && PARTS="$PARTS,all-plan,all-install,all-smoke,all-board,all-prior,all-compare,"
if part all-plan || part all-install || part all-smoke || part all-board || part all-prior || part all-compare; then
    PA="$RES/plan-all"
    mkdir -p "$PA"
    if part all-plan; then
        phase all-plan
        $PY tools/paper_compare.py ids-all --vendor "$V" --planner tools/bench_board.py --out-dir "$PA" \
            > "$RES/logs/ids-all.log" 2>&1
        rc=$?; tail -n 2 "$RES/logs/ids-all.log"
        phase_end $rc
    fi
    [ -s "$PA/race-ids-full.txt" ] || { say "no full-board plan in $PA (run part all-plan)"; exit 5; }
    $PY tools/measure_owed.py data-keys --paper-recipes "$PA/races.json" --planner tools/bench_board.py \
        --vendor "$V" --data-root "$DATA_ROOT" > "$PA/data-keys.tsv" 2> "$RES/logs/data-keys-all.log" || {
        say "could not list the full board's datasets (logs/data-keys-all.log)"; exit 4; }
    MISSING_KEYS=$(while IFS=$'\t' read -r key path; do [ -f "$path" ] || printf '%s ' "$key"; done < "$PA/data-keys.tsv")
    if [ -n "$MISSING_KEYS" ]; then
        say "missing datasets under $DATA_ROOT: $MISSING_KEYS; stage them from the Mac:" \
            "sh tools/dataset_store.sh stage \"<ssh flags+target>\" $MISSING_KEYS"
        exit 4
    fi
    mapfile -t BARGS_ALL < "$PA/board-args.txt"
    COMMON_ALL=("${BARGS_ALL[@]}" --data-root "$DATA_ROOT" --rounds 1 --evidence-local-only --fixtures-venv "$CACHE_ROOT/venv-nf")
    if part all-install; then
        phase all-install
        # the candidate and every prior wheel at once (CPU)
        pids=(); names=()
        for arm in candidate ${PRIOR_ARMS[@]+"${PRIOR_ARMS[@]}"}; do
            lg="$RES/logs/install-all-$arm.log"; od="$RES/install-all-$arm"
            [ "$arm" = candidate ] && { lg="$RES/logs/install-all.log"; od="$RES/install-all"; }
            # shellcheck disable=SC2086
            $PY tools/bench_board.py "${COMMON_ALL[@]}" ${ARM_ARGS[$arm]} --ours-install-only --install-only \
                --venv "$CACHE_ROOT/venv-$arm" --cache "$CACHE_ROOT/i-$arm" --wheel-manifest "${ARM_MAN[$arm]}" \
                --race-ids @"$PA/race-ids-full.txt" --out "$od" > "$lg" 2>&1 &
            pids+=($!); names+=("$arm")
        done
        rc=0
        for i in "${!pids[@]}"; do
            wait "${pids[$i]}"; r=$?
            say "install ${names[$i]} rc=$r"
            [ $r -gt $rc ] && rc=$r
        done
        phase_end $rc
    fi
    [ -f "${ARM_MAN[candidate]}" ] || { say "the candidate arm has no wheel manifest (logs/install-all.log)"; exit 6; }
    fault_gate "${ARM_PY[candidate]}"   # box_job_lib.sh: verify --self-test and the cheap sabotage arms on this GPU, once per job
    CARGS=("${COMMON_ALL[@]}")
    for arm in ${PRIOR_ARMS[@]+"${PRIOR_ARMS[@]}"}; do PRIOR_SMOKE_IDS[$arm]="$PA/race-ids-smoke.txt"; done
    if part all-smoke; then
        # the candidate and every prior wheel at once (untimed; smoke_together)
        phase all-smoke
        smoke_together all- "$PA/race-ids-smoke.txt" "$RES/smoke-all" --drop-race-scratch; src=$?
        phase_end $src
    fi
    SJA=()
    for f in "$RES"/smoke-all-s*-smoke/smoke.json; do [ -f "$f" ] && SJA+=(--smoke-json "$f"); done
    $PY tools/paper_compare.py pass-ids --full-ids "$PA/race-ids-full.txt" "${SJA[@]}" --out "$PA/race-ids-pass.txt" \
        > "$RES/logs/pass-ids-all.log" 2>&1
    tail -n 1 "$RES/logs/pass-ids-all.log"
    if part all-board; then
        phase all-board
        disk_gate all-board "$(races_left "$PA/race-ids-pass.txt" "$RES/board-all/board.json")" 1 "${PC_EVIDENCE_MB_PER_RACE:-40}"
        if [ -s "$PA/race-ids-pass.txt" ]; then
            CUR_IDS="$PA/race-ids-pass.txt"; CUR_BOARD="$RES/board-all"
            # shellcheck disable=SC2046
            $PY tools/bench_board.py "${COMMON_ALL[@]}" $(run_as candidate) --cache "$CACHE_ROOT/c1" \
                --race-ids @"$PA/race-ids-pass.txt" "${SJA[@]}" --drop-race-scratch --out "$RES/board-all" \
                > "$RES/logs/board-all.log" 2>&1
            rc=$?; CUR_IDS=; say "board-all rc=$rc: $(tail -n 1 "$RES/logs/board-all.log" | head -c 200)"
        else
            say "no full-board race passed its smoke"; rc=6
        fi
        phase_end $rc
    fi
    if part all-prior && [ "${#PRIOR_ARMS[@]}" -gt 0 ]; then
        phase all-prior
        # a prior smoke the all-smoke part did not run (a split submit) runs here first, all priors at once
        smoke_together all- "" "" --drop-race-scratch; rc=$?
        for arm in "${PRIOR_ARMS[@]}"; do      # TIMED: one board at a time on the one GPU
            prior_board_run "$arm" "$PA/race-ids-pass.txt" "$RES/board-all-$arm" all-; r=$?; [ $r -gt $rc ] && rc=$r
        done
        phase_end $rc
    fi
    if part all-compare; then
        phase all-compare
        NR=()
        for f in "$PA/not-planned.json" "$PA/smoke-failed.json"; do [ -f "$f" ] && NR+=(--not-run "$f"); done
        mapfile -t PB < <(prior_board_args all-)
        RF=()
        [ -f "$REF_COMPARE" ] && RF=(--reference-compare "$REF_COMPARE")
        $PY tools/paper_compare.py compare --recipes "$RECIPES" --board "$RES/board-all/board.json" "${NR[@]}" "${PB[@]}" "${RF[@]}" \
            --best-ever "${PC_BEST_EVER:-$TREE/bench/results/board_best_ever/$V.json}" \
            --out-md "$RES/COMPARE-ALL-$V.md" --out-json "$RES/compare-all-$V.json" --summary "$RES/SUMMARY-ALL-$V.txt" \
            > "$RES/logs/compare-all.log" 2>&1
        rc=$?; head -n 4 "$RES/logs/compare-all.log"
        phase_end $rc
    fi
    echo finished > "$J/.phase"
    exit "$WORST"
fi

# ---- targeted: only the listed rows (Andrew, 2026-11-17: "can we only measure the things that have changed? we don't
# need a full board run") ------------------------------------------------------------------------------------------
# PC_PARTS=targeted (= tg-install,tg-prep,tg-smoke,tg-board,tg-compare; each can be given alone to split a job over the cap:
# the boards resume from board.json) with PC_ROWS=<rows file on the box> (tools/ops/targeted_rows.py writes it from a
# compare and the changed lanes; tools/ops/measure_candidate.py stage --rows uploads it and refuses one whose estimate
# is over the cap, naming the split). Our arms only, the same machinery as board-all on the listed races: the plan
# (every submit, CPU, seconds: targeted_rows.py plan = the board's current plan of the listed races, in rows-file order,
# and the recipes of those races only) -> the candidate's and the priors' venvs at once (the B arms' by window in
# tg-board, see arm-window) -> tg-prep: the timed races' blocks prepared at once into c1 -> every arm's smoke at once in one shard set (the
# disk gate counts only the listed races) -> the TIMED races, INTERLEAVED BY ARM in chunks: chunk 1 on the candidate,
# chunk 1 on 0.8.55, chunk 2 on the candidate, ..., so the two times of a row are minutes apart, not hours (a chunk is
# about PC_TG_CHUNK_S = 600 s of the rows' stored candidate time; 0 = one race per chunk). The harness times one
# bench_board process per board dir, which resumes from its board.json, so a chunk is one process per arm on the same
# out dir; each process start (plan, preflight audit, smoke gate) is paid per chunk and arm, which is why a chunk is
# not one race by default. Both arms run each chunk's races in the same order. -> the owed A/B arms staged in
# PC_B_ARMS (tools/measure_owed.py plan: each B set only on the rows its define affects), each unit chunk by chunk
# with its base (the candidate) back to back: board-tg-ab-<arm>-base-<rows>/ then board-tg-ab-<arm>-<rows>/ ->
# the same compare as the full board (paper recipes of the listed races, extra and internal rows, on-box 0.8.55
# column, best ever as reference) in COMPARE-TG/compare-tg/SUMMARY-TG-COMPARE-<vendor>, and SUMMARY-TG-<vendor>
# (targeted_rows.py summary: timed on both, FASTER/SAME/REGRESSION, rows still over 1.10 worst first, rows fixed,
# done/total per arm, the A/B ratios). GPU races are serial by nature (one process on the GPU); the installs, the
# smokes and the plans use every cgroup core. With PC_BASELINE_BOARD (the stored 0.8.55 board; Andrew, 2026-11-17:
# "0.8.55 is not timed again") the job has NO prior arm: the plan, the candidate's install (and the B sets'), its smoke
# of the listed races, its board, and the compare against the stored board (first line: the baseline is stored).
# Env-only A/B arms (PC_B_ARMS `<name> env:VAR=VALUE <commit>`) run the candidate's venv with those variables.
[[ "$PARTS" == *",targeted,"* ]] && PARTS="$PARTS,tg-install,tg-prep,tg-smoke,tg-board,tg-compare,tg-drivers,"
if part tg-install || part tg-prep || part tg-smoke || part tg-board || part tg-compare || part tg-drivers; then
    TG="$RES/plan-tg"; TGO="$RES/plan-tg-owed"
    mkdir -p "$TG"
    { [ -n "${PC_ROWS:-}" ] && [ -s "$PC_ROWS" ]; } || { say "targeted: PC_ROWS (the rows file on the box) is missing or empty"; exit 2; }
    cp "$PC_ROWS" "$TG/rows.tsv"
    phase tg-plan
    $PY tools/ops/targeted_rows.py plan --rows "$TG/rows.tsv" --vendor "$V" --planner tools/bench_board.py \
        --recipes "$RECIPES" --out-dir "$TG" --chunk-s "${PC_TG_CHUNK_S:-600}" --prior-wheels "${#PRIOR_WHEEL_NAMES[@]}" ${BASELINE_BOARD:+--one-arm} \
        > "$RES/logs/tg-plan.log" 2>&1
    rc=$?; tail -n 2 "$RES/logs/tg-plan.log"
    if [ $rc -eq 0 ] && [ -n "$B_CSV" ]; then
        # the owed A/B rows of the staged B sets; the empty paper doc leaves every unit its own base
        $PY tools/measure_owed.py plan --owed "$OWED_FILE" --paper-recipes "$TG/owed-paper.json" \
            --planner tools/bench_board.py --vendor "$V" --b-arms "$B_CSV" --out-dir "$TGO" > "$RES/logs/tg-owed-plan.log" 2>&1
        r=$?; tail -n 1 "$RES/logs/tg-owed-plan.log"; [ $r -gt $rc ] && rc=$r
    fi
    phase_end $rc
    [ -s "$TG/race-ids-full.txt" ] || { say "targeted: no listed race is planned by the board (logs/tg-plan.log)"; exit 5; }
    DK=(); [ -n "$B_CSV" ] && DK=(--owed "$OWED_FILE" --b-arms "$B_CSV")
    $PY tools/measure_owed.py data-keys "${DK[@]}" --paper-recipes "$TG/races.json" --planner tools/bench_board.py \
        --vendor "$V" --data-root "$DATA_ROOT" > "$TG/data-keys.tsv" 2> "$RES/logs/data-keys-tg.log" || {
        say "could not list the targeted rows' datasets (logs/data-keys-tg.log)"; exit 4; }
    MISSING_KEYS=$(while IFS=$'\t' read -r key path; do [ -f "$path" ] || printf '%s ' "$key"; done < "$TG/data-keys.tsv")
    if [ -n "$MISSING_KEYS" ]; then
        say "missing datasets under $DATA_ROOT: $MISSING_KEYS; stage them from the Mac (measure_candidate.py stage --rows)"
        exit 4
    fi
    mapfile -t BARGS_TG < "$TG/board-args.txt"
    COMMON_TG=("${BARGS_TG[@]}" --data-root "$DATA_ROOT" --rounds 1 --evidence-local-only --fixtures-venv "$CACHE_ROOT/venv-nf")
    CARGS=("${COMMON_TG[@]}")
    # >>> arm-window (tools/test_targeted_rows.py runs this block)
    # B-ARM VENVS BY WINDOW (lane fx-pod-job, 2026-11-18): every arm's venv used to be made at once in tg-install,
    # 1.3 GB each, and a 42-arm job did not fit the 80 GB pod (it was cut to 14 arms by hand). Now tg-install makes the
    # candidate's and the priors' venvs (all at once, as before); a B arm's venv is made in tg-board at the boundary
    # before its first unit, together with the next arms of a window (W = min(the cgroup's cores, the disk left after
    # the evidence still to be written over PC_VENV_MB), all at once), and its venv and install cache are deleted
    # after its last unit. Installs never run while a race is timed: this is the one pause between timed boards that
    # is kept, because an install would share the cores, the page cache and the disk with a millisecond row.
    # The staged wheels stay (a later submit that resumes the arm installs from them again).
    INSTALL_IDS="$TG/race-ids-full.txt"
    VENV_MB=${PC_VENV_MB:-1536}       # an ours-only venv with its install cache, measured 1.3 GB (2026-11-18)
    EV_LEFT_MB=0                      # the evidence the units not yet run will still write (tg-board)
    arm_ready() { [ -n "${ARM_ENV[$1]:-}" ] || [ -x "${ARM_PY[$1]}" ]; }   # an env arm runs in the candidate's venv
    install_arms() {   # arm...: their venvs at once (CPU), each checked against its wheel manifest; the worst rc
        local pids=() names=() i r rc=0 arm
        for arm in "$@"; do
            # shellcheck disable=SC2086
            $PY tools/bench_board.py "${COMMON_TG[@]}" ${ARM_ARGS[$arm]} --ours-install-only --install-only \
                --venv "$CACHE_ROOT/venv-$arm" --cache "$CACHE_ROOT/i-$arm" --wheel-manifest "${ARM_MAN[$arm]}" \
                --race-ids @"$INSTALL_IDS" --out "$RES/install-tg-$arm" > "$RES/logs/install-tg-$arm.log" 2>&1 &
            pids+=($!); names+=("$arm")
        done
        for i in "${!pids[@]}"; do
            wait "${pids[$i]}"; r=$?; say "install ${names[$i]} rc=$r"; [ $r -gt $rc ] && rc=$r
        done
        return $rc
    }
    window_size() {   # the arms one install window holds: min(cores, (free disk - evidence left - reserve) / venv)
        local free_mb w cores
        cores=$(alloc_cores)
        free_mb=$(df -Pk "$CACHE_ROOT" 2> /dev/null | awk 'NR==2 {print int($4 / 1024)}')
        w=$(( (${free_mb:-0} - EV_LEFT_MB - ${PC_DISK_RESERVE_GB:-5} * 1024) / VENV_MB ))
        [ "$w" -gt "$cores" ] && w=$cores
        [ "$w" -lt 1 ] && w=1
        echo "$w"
    }
    install_window() {   # arm: its venv and those of the next arms of WIN_ARMS not yet made, up to window_size
        local want=$1 w on=0 n=0 a pick=()
        w=$(window_size)
        for a in "${WIN_ARMS[@]}"; do
            [ "$a" = "$want" ] && on=1
            [ "$on" = 1 ] || continue
            arm_ready "$a" && continue
            pick+=("$a"); n=$(( n + 1 ))
            [ "$n" -ge "$w" ] && break
        done
        [ "$n" -gt 0 ] || return 0
        say "install window: ${pick[*]} ($n of window $w; $(df -Pk "$CACHE_ROOT" 2> /dev/null | awk 'NR==2 {printf "%.1f GB free", $4 / 1048576}'), evidence still to write $EV_LEFT_MB MB)"
        install_arms "${pick[@]}"
    }
    drop_arm_venv() {   # arm: its venv and install cache once its last unit has run (the staged wheels stay)
        rm -rf "$CACHE_ROOT/venv-$1" "$CACHE_ROOT/i-$1"
        say "venv $1 deleted after its last unit; $(df -Pk "$CACHE_ROOT" 2> /dev/null | awk 'NR==2 {printf "%.1f GB free", $4 / 1048576}')"
    }
    # <<< arm-window
    WIN_ARMS=()
    if part tg-install; then
        phase tg-install
        pids=(); names=(); now=()
        for arm in "${ARMS[@]}"; do      # the candidate and the priors at once (CPU); the B sets by window in tg-board
            [ -n "${ARM_ENV[$arm]:-}" ] && continue      # an env arm runs in the candidate's venv: nothing to install
            [[ " ${B_NAMES[*]:-} " == *" $arm "* ]] && continue
            now+=("$arm")
        done
        install_arms "${now[@]}"; rc=$?
        phase_end $rc
    fi
    [ -f "${ARM_MAN[candidate]}" ] || { say "the candidate arm has no wheel manifest (logs/install-tg-candidate.log)"; exit 6; }
    fault_gate "${ARM_PY[candidate]}"   # box_job_lib.sh: verify --self-test and the cheap sabotage arms on this GPU, once per job
    cat "$TG/race-ids-smoke.txt" "$TGO/smoke-ids.txt" 2> /dev/null | sort -u > "$TG/smoke-ids-all.txt"
    for arm in ${PRIOR_ARMS[@]+"${PRIOR_ARMS[@]}"}; do PRIOR_SMOKE_IDS[$arm]="$TG/race-ids-smoke.txt"; done
    if part tg-prep; then
        # ONE PARALLEL PREP (lane fx-pod-job, 2026-11-18): the timed board's first process prepared every full-row
        # block on one core while the GPU waited (graph-istella 229.5 s + graph-taxi 256.9 s of its 695 s in
        # nvc-tg-20261118T135756Z-7a2bff). Here, untimed and before the smoke, the races the timed boards run
        # (the listed rows and the owed full rows) are prepared into c1, one --preflight-only --prep-only process per
        # family and prep dataset (targeted_rows.py prep-groups: they write disjoint blocks), all at once up to
        # min(cgroup cores, MemAvailable / PC_PREP_MEM_GB (12)). No --fixtures-venv here: the neural fixtures are made
        # over every prepared mirror at once, by the timed board's first process, as before. The timed processes then
        # find every block prepared (the driver skips a block whose record exists); the prepared bytes are the same.
        phase tg-prep
        # the TIMED races: the listed rows and every owed unit at full rows (lane fx-pod-job2, 2026-11-18: the smoke
        # ids, rows=2000, were passed here; the full-row plan refused them, "--race-ids: not in this plan", and all 8
        # groups exited 1 in 0 s in nvc-ab-20261118T201200Z-d21ba0)
        PIDS=(--ids "$TG/race-ids-full.txt")
        if [ -s "$TGO/units.tsv" ]; then
            while IFS=$'\t' read -r _arm prow pids_f <&3; do
                [ "$prow" = full ] && PIDS+=(--ids "$pids_f")
            done 3< "$TGO/units.tsv"
        fi
        $PY tools/ops/targeted_rows.py prep-groups "${PIDS[@]}" \
            --out-dir "$TG/prep" > "$RES/logs/tg-prep-groups.log" 2>&1
        rc=$?; tail -n 1 "$RES/logs/tg-prep-groups.log"
        PREP_ARGS=("${BARGS_TG[@]}" --data-root "$DATA_ROOT" --rounds 1 --evidence-local-only)
        memg=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo 2> /dev/null || echo 16)
        pw=$(( memg / ${PC_PREP_MEM_GB:-12} )); [ "$pw" -gt "$(alloc_cores)" ] && pw=$(alloc_cores); [ "$pw" -lt 1 ] && pw=1
        pids=(); names=()
        for g in "$TG"/prep/group-*.txt; do
            [ -f "$g" ] || continue
            # at most pw at once (jobs -r would count the cores sampler and the cap guard too)
            while [ "$(n=0; for q in ${pids[@]+"${pids[@]}"}; do kill -0 "$q" 2> /dev/null && n=$(( n + 1 )); done; echo $n)" -ge "$pw" ]; do
                sleep 2
            done
            gn=$(basename "$g" .txt)
            # shellcheck disable=SC2046
            $PY tools/bench_board.py "${PREP_ARGS[@]}" $(run_as candidate) --cache "$CACHE_ROOT/c1" --race-ids @"$g" \
                --preflight-only --prep-only --out "$RES/prep-tg/$gn" > "$RES/logs/prep-tg-$gn.log" 2>&1 &
            pids+=($!); names+=("$gn")
        done
        say "tg-prep: ${#pids[@]} group(s), $pw at once (allocation $(alloc_cores) cores, MemAvailable $memg GiB)"
        for i in "${!pids[@]}"; do
            wait "${pids[$i]}"; r=$?; say "prep ${names[$i]} rc=$r"; [ $r -gt $rc ] && rc=$r
        done
        phase_end $rc     # a failed group is prepared again by its timed process, as before; the job goes on
    fi
    if part tg-smoke; then
        phase tg-smoke    # the candidate (listed + owed rows) and every prior (listed rows) at once, untimed
        smoke_together tg- "$TG/smoke-ids-all.txt" "$RES/smoke-tg" --drop-race-scratch; src=$?
        phase_end $src
    fi
    SJT=()
    for f in "$RES"/smoke-tg-s*-smoke/smoke.json; do [ -f "$f" ] && SJT+=(--smoke-json "$f"); done
    $PY tools/paper_compare.py pass-ids --full-ids "$TG/race-ids-full.txt" "${SJT[@]}" --out "$TG/race-ids-pass.txt" \
        > "$RES/logs/pass-ids-tg.log" 2>&1
    tail -n 1 "$RES/logs/pass-ids-tg.log"
    tg_board() {   # arm, ids file, out dir, rows ("" = full), smoke-json args...: one TIMED process, resumes board.json
        local arm=$1 ids=$2 out=$3 rows=$4 rc ra=()
        shift 4
        [ -s "$ids" ] || return 0
        [ -f "${ARM_MAN[$arm]:-/nonexistent}" ] || { say "board $arm: no installed venv (logs/install-tg-$arm.log)"; return 6; }
        [ -n "$rows" ] && ra=(--rows "$rows")
        # a placeholder board (targeted_rows.py empty-board, written by an earlier submit's tg-compare when nothing was
        # timed) is no board: bench_board starts the dir afresh
        grep -qs '"placeholder"' "$out/board.json" && rm -f "$out/board.json"
        CUR_IDS=$ids; CUR_BOARD=$out
        local envs=() ev=${ARM_ENV[$arm]:-${ARM_XENV[$arm]:-}}; [ -n "$ev" ] && IFS=, read -r -a envs <<< "$ev"
        # an env arm names the keys it sets on purpose, so tools/cpu_quota.py lets them through the race workers'
        # cap reset (lane fx-harness, 2026-11-18: cpu-threads-cgroup never took effect without this)
        if [ ${#envs[@]} -gt 0 ]; then
            local kv keys=""
            for kv in "${envs[@]}"; do keys="${keys:+$keys,}${kv%%=*}"; done
            envs+=("MOJOLEARN_BENCH_ARM_ENV_KEYS=$keys")
        fi
        # shellcheck disable=SC2046
        run_timed "$RES/logs/$(basename "$out").log" 1 env ${envs[@]+"${envs[@]}"} $PY tools/bench_board.py "${COMMON_TG[@]}" \
            $(run_as "$arm") --cache "$CACHE_ROOT/c1" --race-ids @"$ids" "$@" ${ra[@]+"${ra[@]}"} --drop-race-scratch --out "$out"
        rc=$?; CUR_IDS=
        say "board $arm $(basename "$ids") rc=$rc: $(tail -n 1 "$RES/logs/$(basename "$out").log" | head -c 160)"
        return $rc
    }
    if part tg-board; then
        phase tg-board
        smoke_together tg- "" ""; rc=$?     # a prior smoke the tg-smoke part did not run (a split submit), all at once
        left=$(races_left "$TG/race-ids-pass.txt" "$RES/board-tg/board.json"); nb=1
        for arm in ${PRIOR_ARMS[@]+"${PRIOR_ARMS[@]}"}; do
            # the races the candidate passed that this prior's own smoke passed too; its failures with their reasons
            mapfile -t psj < <(prior_smoke_args "$arm" tg-)
            mkdir -p "$TG/$arm"
            $PY tools/paper_compare.py pass-ids --full-ids "$TG/race-ids-pass.txt" "${psj[@]}" \
                --out "$TG/$arm/race-ids-pass.txt" > "$RES/logs/pass-ids-tg-$arm.log" 2>&1
            say "$arm: $(tail -n 1 "$RES/logs/pass-ids-tg-$arm.log")"
            grep -m 40 '^PAPER-PASS-IDS-FAILED' "$RES/logs/pass-ids-tg-$arm.log" | cut -c1-300
            left=$(( left + $(races_left "$TG/$arm/race-ids-pass.txt" "$RES/board-tg-$arm/board.json") )); nb=$(( nb + 1 ))
        done
        if [ -s "$TGO/units.tsv" ]; then
            while IFS=$'\t' read -r arm rows ids <&3; do
                { [ "$arm" = candidate ] || [ "$arm" = prior ]; } && continue
                left=$(( left + 2 * $(nlines "$ids") )); nb=$(( nb + 2 ))
            done 3< "$TGO/units.tsv"
        fi
        # a board dir's fixed part here is the measured one (lane fx-pod-job, 2026-11-18: the ab1 unit boards were
        # 98-166 MB each with their races, against the 400 MB default), so a 42-arm job's 85 unit boards are not
        # refused on a disk they fit; the B arms' venvs are counted per install window (window_size)
        PC_LOADED_MB=${PC_LOADED_MB:-200} disk_gate tg-board "$left" "$nb" "${PC_EVIDENCE_MB_PER_RACE:-40}"
        # the listed rows: chunk by chunk, the candidate then each prior on the same races in the same order
        $PY tools/ops/targeted_rows.py chunks --ids "$TG/race-ids-pass.txt" --rows "$TG/rows.tsv" \
            --chunk-s "${PC_TG_CHUNK_S:-600}" --out-dir "$TG/chunks" | tail -n 1
        for c in "$TG"/chunks/chunk-*.txt; do
            [ -f "$c" ] || continue
            tg_board candidate "$c" "$RES/board-tg" "" "${SJT[@]}"; r=$?; [ $r -gt $rc ] && rc=$r
            for arm in ${PRIOR_ARMS[@]+"${PRIOR_ARMS[@]}"}; do
                grep -Fx -f "$TG/$arm/race-ids-pass.txt" "$c" > "$c.$arm" 2> /dev/null
                mapfile -t psj < <(prior_smoke_args "$arm" tg-)
                tg_board "$arm" "$c.$arm" "$RES/board-tg-$arm" "" "${psj[@]}"; r=$?; [ $r -gt $rc ] && rc=$r
            done
        done
        # the owed A/B arms: each unit chunk by chunk, its base (the candidate) and the arm back to back. Two passes:
        # the units this submit runs (their pass ids, the arms in order, each arm's last unit, the evidence still to
        # write), then the units, each B arm's venv made by window before its first unit and deleted after its last
        if [ -s "$TGO/units.tsv" ]; then
            UNITS=(); declare -A LAST_UNIT=() SEEN_ARM=()
            while IFS=$'\t' read -r arm rows ids <&3; do
                { [ "$arm" = candidate ] || [ "$arm" = prior ]; } && continue
                [ -n "${PC_OWED_ARMS:-}" ] && [[ ",$PC_OWED_ARMS," != *",$arm,"* ]] && continue   # a split submit
                abrows=""; use=$ids
                if [ "$rows" = full ]; then      # full rows: only the races whose candidate smoke passed
                    $PY tools/paper_compare.py pass-ids --full-ids "$ids" "${SJT[@]}" --out "$ids.pass" \
                        > "$RES/logs/pass-ids-tg-ab-$arm.log" 2>&1
                    use="$ids.pass"
                else
                    abrows=$rows                 # a neighbouring shape: no smoke gate (as the owed part)
                fi
                # a resumed submit: a unit whose arm board holds every race and whose base board every base race
                # is done needs no process and no venv
                cp "$use" "$use.base"
                if [ "$(races_left "$use" "$RES/board-tg-ab-$arm-$rows/board.json")" -eq 0 ] 2> /dev/null \
                        && [ "$(races_left "$use.base" "$RES/board-tg-ab-$arm-base-$rows/board.json")" -eq 0 ] 2> /dev/null; then
                    if [ "$(nlines "$use")" -eq 0 ]; then say "owed $arm $rows: NOT TIMED, none of its races passed its smoke (logs/pass-ids-tg-ab-$arm.log; timed-check names it)"
                    else say "owed $arm $rows: every race already in its boards (an earlier submit)"; fi
                    continue
                fi
                LAST_UNIT[$arm]=${#UNITS[@]}
                UNITS+=("$arm"$'\t'"$rows"$'\t'"$ids"$'\t'"$use"$'\t'"$abrows")
                [ -n "${SEEN_ARM[$arm]:-}" ] || { SEEN_ARM[$arm]=1; WIN_ARMS+=("$arm"); }
                EV_LEFT_MB=$(( EV_LEFT_MB + 2 * $(nlines "$use") * ${PC_EVIDENCE_MB_PER_RACE:-40} ))
            done 3< "$TGO/units.tsv"
            for k in ${UNITS[@]+"${!UNITS[@]}"}; do
                IFS=$'\t' read -r arm rows ids use abrows <<< "${UNITS[$k]}"
                capped && { say "owed $arm $rows: not run (cap guard)"; continue; }
                if ! arm_ready "$arm"; then
                    install_window "$arm"; r=$?; [ $r -gt $rc ] && rc=$r
                fi
                $PY tools/ops/targeted_rows.py chunks --ids "$use" --chunk-s "${PC_TG_CHUNK_S:-600}" \
                    --out-dir "$TGO/chunks-$arm-$rows" | tail -n 1
                # IN-JOB BASE CAP (measure_owed.py; lane fx-same-pod-prior, 2026-11-19, replacing the 2026-11-17
                # STORED BASE from another host): base and arm back to back on every race; a race of the unit's
                # .cap.txt has the arm's rounds capped at the time the base just took for it in this job
                capf="${ids%.txt}.cap.txt"
                for c in "$TGO/chunks-$arm-$rows"/chunk-*.txt; do
                    [ -f "$c" ] || continue
                    cp "$c" "$c.base"
                    tg_board candidate "$c.base" "$RES/board-tg-ab-$arm-base-$rows" "$abrows" "${SJT[@]}"; r=$?; [ $r -gt $rc ] && rc=$r
                    cap=()
                    if [ -s "$capf" ]; then
                        $PY tools/measure_owed.py base-caps --board "$RES/board-tg-ab-$arm-base-$rows/board.json" \
                            --ids "$c" --cap-ids "$capf" --out "$c.caps.json" | tail -n 1
                        [ -s "$c.caps.json" ] && cap=(--race-round-seconds "$c.caps.json")
                    fi
                    tg_board "$arm" "$c" "$RES/board-tg-ab-$arm-$rows" "$abrows" "${SJT[@]}" ${cap[@]+"${cap[@]}"}; r=$?; [ $r -gt $rc ] && rc=$r
                done
                EV_LEFT_MB=$(( EV_LEFT_MB - 2 * $(nlines "$use") * ${PC_EVIDENCE_MB_PER_RACE:-40} ))
                if [ "${LAST_UNIT[$arm]}" = "$k" ] && [ -z "${ARM_ENV[$arm]:-}" ]; then
                    if capped; then say "venv $arm kept: the cap guard cut its units (the next submit resumes them)"
                    else drop_arm_venv "$arm"; fi
                fi
            done
        fi
        # EVERY STAGED ARM TIMED (lane fx-pod-job2, 2026-11-18): 12 of 14 B arms were skipped with one quiet line each
        # and the job's only failure was the compare's missing board. Each arm (the candidate on the listed rows, each
        # owed unit's arm and its base) with no timed race gets an UNTIMED line with why, and the phase fails
        $PY tools/ops/targeted_rows.py timed-check --results "$RES" --ids "$TG/race-ids-full.txt" \
            --units "$TGO/units.tsv" ${PC_OWED_ARMS:+--arms "$PC_OWED_ARMS"} \
            --prior-wheels "$(IFS=,; echo "${PRIOR_WHEEL_NAMES[*]:-}")" > "$RES/logs/tg-timed-check.log" 2>&1
        r=$?; grep -m 60 -e '^UNTIMED' -e '^TG-TIMED-CHECK' "$RES/logs/tg-timed-check.log" | cut -c1-400
        [ $r -gt $rc ] && rc=$r
        phase_end $rc
    fi
    if part tg-compare; then
        phase tg-compare
        # the candidate's targeted board exists whenever rows are given (lane fx-pod-job2, 2026-11-18): a job whose
        # listed rows all failed their smoke wrote none, and the compare printed "unreadable board" and "this job:
        # host not recorded"; the placeholder holds no race and the job's host probe, so every row is NOT-RUN with
        # its reason (smoke-failed.json) and the host is named
        $PY tools/ops/targeted_rows.py empty-board --out "$RES/board-tg" --host-probe "$J/host-probe.json" \
            --why "no listed race of the candidate was timed in this job (logs/tg-timed-check.log)" | tail -n 1
        NR=()
        for f in "$TG/not-planned.json" "$TG/smoke-failed.json"; do [ -f "$f" ] && NR+=(--not-run "$f"); done
        mapfile -t PB < <(prior_board_args tg-)     # the PyPI priors, or (with a stored baseline) the prior wheels only
        [ -n "$BASELINE_BOARD" ] && PB+=(--stored-baseline-board "$BASELINE_BOARD")
        # a prior wheel timed in this job is the verdict baseline; the stored board is then a screen column (no flag)
        [ ${#PRIOR_WHEEL_NAMES[@]} -gt 0 ] && PB+=(--verdict-prior "${PRIOR_WHEEL_NAMES[0]}")
        RF=()
        [ -f "$REF_COMPARE" ] && RF=(--reference-compare "$REF_COMPARE")
        $PY tools/paper_compare.py compare --recipes "$TG/recipes.json" --board "$RES/board-tg/board.json" "${NR[@]}" \
            "${PB[@]}" "${RF[@]}" --best-ever "${PC_BEST_EVER:-$TREE/bench/results/board_best_ever/$V.json}" \
            --out-md "$RES/COMPARE-TG-$V.md" --out-json "$RES/compare-tg-$V.json" \
            --summary "$RES/SUMMARY-TG-COMPARE-$V.txt" > "$RES/logs/compare-tg.log" 2>&1
        rc=$?; head -n 3 "$RES/logs/compare-tg.log"
        $PY tools/ops/targeted_rows.py summary --compare "$RES/compare-tg-$V.json" --rows "$TG/rows.tsv" \
            --results "$RES" --ids "$TG/race-ids-full.txt" --units "$TGO/units.tsv" \
            --prior-versions "$(IFS=,; echo "${PRIOR_ARMS[*]#prior-}")" \
            --out "$RES/SUMMARY-TG-$V.txt" > "$RES/logs/summary-tg.log" 2>&1
        r=$?; head -n 3 "$RES/logs/summary-tg.log"; [ $r -gt $rc ] && rc=$r
        phase_end $rc
    fi
    # The owed plan's driver rows. Before 2026-11-18 (lane fx-verdicts2) the targeted job exited here without them:
    # ab3 (nvc-ab-20261118T182558Z-cb6e02) installed the mamba-backward-generic set for restore-mamba-backward-glue
    # (plan-tg-owed/drivers.json, drivers=3) and never timed it, since the drivers phase sat only in the paper job.
    # >>> tg-drivers (tools/test_targeted_rows.py runs this block)
    if part tg-drivers && [ "$OWED" = 1 ] && [ -s "$TGO/drivers.json" ]; then
        phase tg-drivers
        # a B set only a driver row names has no board unit, so tg-board's install window never made its venv:
        # install those sets here (at once, CPU), time the rows, then drop them
        mapfile -t DRV_B < <($PY -c 'import json,sys; print("\n".join(sorted({a for r in json.load(open(sys.argv[1])) for a in r.get("b_arms") or []})))' "$TGO/drivers.json")
        drv_new=()
        for arm in ${DRV_B[@]+"${DRV_B[@]}"}; do
            [[ " ${B_NAMES[*]:-} " == *" $arm "* ]] || continue
            arm_ready "$arm" || drv_new+=("$arm")
        done
        rc=0
        if [ ${#drv_new[@]} -gt 0 ]; then install_arms "${drv_new[@]}"; rc=$?; fi
        run_owed_drivers "$TGO/drivers.json"; r=$?; [ $r -gt $rc ] && rc=$r
        for arm in ${drv_new[@]+"${drv_new[@]}"}; do drop_arm_venv "$arm"; done
        phase_end $rc
    fi
    # <<< tg-drivers
    echo finished > "$J/.phase"
    exit "$WORST"
fi

# ---- plan (CPU, seconds) -----------------------------------------------------------------------------------------
owed_plan() {   # the owed rows' plan for this job's B sets (plan-owed/plan.json records them)
    $PY tools/measure_owed.py plan --owed "$OWED_FILE" --paper-recipes "$RECIPES" --planner tools/bench_board.py \
        --vendor "$V" --b-arms "$B_CSV" ${OWED_PRIOR[@]+"${OWED_PRIOR[@]}"} --out-dir "$RES/plan-owed" \
        > "$RES/logs/owed-plan.log" 2>&1
    local r=$?; tail -n 2 "$RES/logs/owed-plan.log"; return $r
}
if part plan; then
    phase plan
    $PY tools/paper_compare.py ids --recipes "$RECIPES" --planner tools/bench_board.py --out-dir "$RES/plan" \
        > "$RES/logs/ids.log" 2>&1
    rc=$?; tail -n 3 "$RES/logs/ids.log"
    if [ "$OWED" = 1 ]; then
        owed_plan; r=$?; [ $r -gt $rc ] && rc=$r
    fi
    phase_end $rc
elif [ "$OWED" = 1 ]; then
    # OWED-PLAN STALE FIX (2026-11-13, job 0003 of ba08dac006 on the L40S): a split submit without the plan part
    # reused plan-owed/ from job 0001, made without PC_B_ARMS (b_arms=-): its units.tsv had no B unit, so
    # PC_OWED_ARMS=candidate,<b names> ran no B arm although all five B sets were installed. A plan made for other
    # B sets (or before plan.json existed) is re-made here (CPU, seconds); the candidate's ids do not change.
    chk=$($PY tools/measure_owed.py plan-check --plan-dir "$RES/plan-owed" --b-arms "$B_CSV" --vendor "$V" \
        ${OWED_PRIOR[@]+"${OWED_PRIOR[@]}"} 2>&1)
    if [ $? -ne 0 ]; then
        say "owed plan re-made: $chk"
        owed_plan || say "the owed plan failed (logs/owed-plan.log)"
    fi
fi
if [ "$OWED" = 1 ] && [ -n "${PC_OWED_ARMS:-}" ] && [ -s "$RES/plan-owed/units.tsv" ]; then
    for want in ${PC_OWED_ARMS//,/ }; do      # a named arm with no unit runs nothing: say so, never silently
        cut -f1 "$RES/plan-owed/units.tsv" | grep -qx -- "$want" \
            || say "PC_OWED_ARMS names $want: no owed unit for it in plan-owed/units.tsv (B sets planned: ${B_CSV:--})"
    done
fi
[ -s "$RES/plan/race-ids-full.txt" ] || { say "no paper plan in $RES/plan (run part plan)"; exit 5; }
# every dataset the races read, from the board's own data_needs over the paper and owed races (tools/measure_owed.py
# data-keys; the same list tools/ops/measure_candidate.py stage stages): a missing one stops the job here, before any
# install (2026-11-12: a missing higgs failed half the smoke shards)
DK_OWED=(); [ "$OWED" = 1 ] && DK_OWED=(--owed "$OWED_FILE" --b-arms "$B_CSV" ${OWED_PRIOR[@]+"${OWED_PRIOR[@]}"})
$PY tools/measure_owed.py data-keys "${DK_OWED[@]}" --paper-recipes "$RECIPES" --planner tools/bench_board.py \
    --vendor "$V" --data-root "$DATA_ROOT" > "$RES/plan/data-keys.tsv" 2> "$RES/logs/data-keys.log" || {
    say "could not list the datasets (logs/data-keys.log)"; exit 4; }
MISSING_KEYS=$(while IFS=$'\t' read -r key path; do [ -f "$path" ] || printf '%s ' "$key"; done < "$RES/plan/data-keys.tsv")
if [ -n "$MISSING_KEYS" ]; then
    say "missing datasets under $DATA_ROOT: $MISSING_KEYS; stage them from the Mac (tools/ops/measure_candidate.py stage" \
        "does it): sh tools/dataset_store.sh stage \"<ssh flags+target>\" $MISSING_KEYS"
    exit 4
fi
say "datasets present: $(cut -f1 "$RES/plan/data-keys.tsv" | tr '\n' ' ')"
# one board argument set for every process: the owed plan's union (paper + owed lanes), else the paper's
BA="$RES/plan/board-args.txt"; [ "$OWED" = 1 ] && [ -s "$RES/plan-owed/board-args.txt" ] && BA="$RES/plan-owed/board-args.txt"
mapfile -t BARGS < "$BA"
# --fixtures-venv: the algos layer lanes' own arm reads the neural-layer fixtures (tools/neural_fixtures.py); each
# board process makes them beside its algos blocks from one shared CPU-torch venv (candidate 4c2340386abc smoked
# conv2d, gcn, graphsage and moe as "not run": no fixture set was on the box)
COMMON=("${BARGS[@]}" --data-root "$DATA_ROOT" --rounds 1 --evidence-local-only --fixtures-venv "$CACHE_ROOT/venv-nf")

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
fault_gate "${ARM_PY[candidate]}"   # box_job_lib.sh: verify --self-test and the cheap sabotage arms on this GPU, once per job

# ---- smoke (parallel shards on the one GPU; not timed) ----------------------------------------------------------
SMOKE_IDS="$RES/plan/race-ids-smoke-all.txt"
cat "$RES/plan/race-ids-smoke.txt" "$RES/plan-owed/smoke-ids.txt" 2> /dev/null | sort -u > "$SMOKE_IDS"
# each prior is smoked only on the races it is compared on: the paper rows; the last release (LAST_PRIOR) also the
# owed rows' prior full rows (plan-owed/smoke-ids-prior.txt). 2026-11-16 (d95f805cce, L40S): 0.8.55 as PC_PRIOR_VERSION
# was planned on the last release's 159 races, of which 98 0.8.55 does not have.
CARGS=("${COMMON[@]}")
for arm in ${PRIOR_ARMS[@]+"${PRIOR_ARMS[@]}"}; do
    PRIOR_SMOKE_IDS[$arm]="$RES/plan/race-ids-smoke.txt"
    if [ "$OWED" = 1 ] && [ -n "$LAST_PRIOR" ] && [ "$arm" = "prior-$LAST_PRIOR" ]; then
        PRIOR_SMOKE_IDS[$arm]="$RES/plan/$arm-smoke-ids.txt"
        cat "$RES/plan/race-ids-smoke.txt" "$RES/plan-owed/smoke-ids-prior.txt" 2> /dev/null | sort -u > "${PRIOR_SMOKE_IDS[$arm]}"
    fi
done
if part smoke; then
    phase smoke
    # NOT GPU-bound (read from the code; not measured): a smoke race runs SMOKE_ROWS=2000 rows at the `small` shape,
    # so each shard's time is CPU and I/O on the host: a fresh worker process per arm (interpreter start, `import
    # mojolearn`, the GPU context), the dataset npz read and sliced, the evidence sealed. Each shard runs one race at a
    # time and mostly one core, which is why 5 shards (nproc/4) showed busy 2-4 of 20 cores (2026-11-12, AMD). So the
    # shard count is nproc/2, bounded by host memory (PC_SMOKE_MEM_GB per shard), by GPU memory (PC_SMOKE_GPU_GB per
    # shard's contexts and small buffers) and PC_SMOKE_MAX (16): smoke_shards. The candidate and every prior wheel
    # smoke at once in those shards (smoke_together), each on its own venv.
    smoke_together "" "$SMOKE_IDS" "$RES/smoke"; src=$?
    phase_end $src     # a SMOKE FAIL exits 1: recorded, the passed races still go on
fi
SJ=()
for f in "$RES"/smoke-s*-smoke/smoke.json; do [ -f "$f" ] && SJ+=(--smoke-json "$f"); done
PASS_IDS="$RES/plan/race-ids-pass.txt"
FULL_ALL="$RES/plan/race-ids-full-all.txt"
# paper rows first, in recipe order, then the owed full rows not already in it: the timed board runs the list in this
# order (bench_board select_race_ids keeps it), so a 240-min cap kill loses owed rows, never paper rows (2026-11-14: a
# sorted union ran 86 algos rows for 2.7 h before the first paper row).
cat "$RES/plan/race-ids-full.txt" "$RES/plan-owed/ids-candidate-full.txt" 2> /dev/null | awk 'NF && !seen[$0]++' > "$FULL_ALL"
$PY tools/paper_compare.py pass-ids --full-ids "$FULL_ALL" "${SJ[@]}" --out "$PASS_IDS" > "$RES/logs/pass-ids.log" 2>&1
tail -n 1 "$RES/logs/pass-ids.log"
# A wheel whose smoke fails most races is broken, not slow: stop before the timed boards and the priors instead of
# spending the cap on the survivors (2026-11-16: 132 of 199 races of 26709df246 failed its smoke on one device-plan
# defect and the job still ran 4 h of boards and priors for 6 comparable rows). PC_SMOKE_MIN_PASS_PCT (default 50).
SM_PLAN=$(sed -n 's/.*planned=\([0-9]*\).*/\1/p' "$RES/logs/pass-ids.log" | tail -n 1)
SM_PASS=$(sed -n 's/.*smoke_passed=\([0-9]*\).*/\1/p' "$RES/logs/pass-ids.log" | tail -n 1)
if [ -n "$SM_PLAN" ] && [ -n "$SM_PASS" ] && [ "$SM_PLAN" -gt 0 ] \
        && [ $(( SM_PASS * 100 )) -lt $(( SM_PLAN * ${PC_SMOKE_MIN_PASS_PCT:-50} )) ]; then
    say "STOP: only $SM_PASS of $SM_PLAN races passed the candidate smoke (under ${PC_SMOKE_MIN_PASS_PCT:-50}%); no board is timed. Read logs/smoke-s*.log"
    exit 7
fi

# ---- the timed boards (one process at a time: serial by nature on one GPU) -------------------------------------
board_run() {   # arm, ids file, out dir, extra args...
    local arm=$1 ids=$2 out=$3; shift 3
    [ -s "$ids" ] || { say "board $arm $(basename "$ids"): no races"; return 0; }
    [ -f "${ARM_MAN[$arm]:-/nonexistent}" ] || { say "board $arm: no installed venv (logs/install-$arm.log)"; return 6; }
    CUR_IDS=$ids; CUR_BOARD=$out
    # shellcheck disable=SC2046
    run_timed "$RES/logs/$(basename "$out").log" 0 $PY tools/bench_board.py "${COMMON[@]}" $(run_as "$arm") \
        --cache "$CACHE_ROOT/c1" --race-ids @"$ids" "${SJ[@]}" "$@" --drop-race-scratch --out "$out"
    local rc=$?; CUR_IDS=
    say "board $arm $(basename "$ids") rc=$rc: $(tail -n 1 "$RES/logs/$(basename "$out").log" | head -c 200)"
    return $rc
}
if part board; then
    phase board
    disk_gate board "$(races_left "$PASS_IDS" "$RES/board/board.json")" 1 "${PC_EVIDENCE_MB_PER_RACE:-40}"
    if [ -s "$PASS_IDS" ]; then board_run candidate "$PASS_IDS" "$RES/board"; rc=$?; else say "no race passed its smoke"; rc=6; fi
    phase_end $rc
fi
if part prior && [ "${#PRIOR_ARMS[@]}" -gt 0 ]; then
    # each prior wheel on this box: its timed board of the paper rows it and the candidate passed; the last release
    # (prior-$LAST_PRIOR) also times the owed rows' prior full rows here (no second timing of the prior). A prior smoke
    # the smoke part did not run (a split submit) runs here first, every such prior at once.
    phase prior
    smoke_together "" "" ""; rc=$?
    for arm in "${PRIOR_ARMS[@]}"; do      # TIMED: one board at a time on the one GPU
        want="$RES/plan/$arm-want.txt"
        cat "$RES/plan/race-ids-full.txt" > "$want.set"
        if [ "$OWED" = 1 ] && [ -n "$LAST_PRIOR" ] && [ "$arm" = "prior-$LAST_PRIOR" ]; then
            cat "$RES/plan-owed/ids-prior-full.txt" >> "$want.set" 2> /dev/null
        fi
        # in PASS_IDS order (paper rows first): a cap kill loses owed rows, never paper rows
        awk 'NR == FNR {w[$0] = 1; next} w[$0]' "$want.set" "$PASS_IDS" > "$want"
        prior_board_run "$arm" "$want" "$RES/board-$arm" ""; r=$?; [ $r -gt $rc ] && rc=$r
    done
    phase_end $rc
fi
if [ "$OWED" = 1 ] && part owed && [ -s "$RES/plan-owed/units.tsv" ]; then
    phase owed
    # every owed unit's races not yet done in its board (full rows may be fewer after the smoke gate: an upper
    # bound), one board dir per unit
    left=0
    while IFS=$'\t' read -r arm rows ids; do
        { [ "$arm" = candidate ] || [ "$arm" = prior ]; } && [ "$rows" = full ] && continue   # timed in board/, board-prior-*
        [ -n "${PC_OWED_ARMS:-}" ] && [[ ",$PC_OWED_ARMS," != *",$arm,"* ]] && continue
        left=$(( left + $(races_left "$ids" "$RES/board-$arm-$rows/board.json") ))
    done < "$RES/plan-owed/units.tsv"
    disk_gate owed "$left" "$(nlines "$RES/plan-owed/units.tsv")" "${PC_EVIDENCE_MB_PER_RACE:-40}"
    rc=0
    while IFS=$'\t' read -r arm rows ids; do
        [ "$arm" = candidate ] && [ "$rows" = full ] && continue   # timed with the paper races (part board)
        [ "$arm" = prior ] && [ "$rows" = full ] && continue       # timed in board-prior-$LAST_PRIOR (part prior)
        [ -n "${PC_OWED_ARMS:-}" ] && [[ ",$PC_OWED_ARMS," != *",$arm,"* ]] && continue   # a split submit
        # IN-JOB BASE CAP (measure_owed.py, 2026-11-19): a B unit's capped races stop at the time this job's base
        # (the candidate's board, timed before: part board for full rows, its own unit first in units.tsv otherwise)
        # took for them
        cap=()
        if [ -s "${ids%.txt}.cap.txt" ]; then
            bb="$RES/board-candidate-$rows/board.json"; [ "$rows" = full ] && bb="$RES/board/board.json"
            $PY tools/measure_owed.py base-caps --board "$bb" --ids "$ids" --cap-ids "${ids%.txt}.cap.txt" \
                --out "${ids%.txt}.caps.json" | tail -n 1
            [ -s "${ids%.txt}.caps.json" ] && cap=(--race-round-seconds "${ids%.txt}.caps.json")
        fi
        if [ "$rows" = full ]; then
            # full rows: only the races whose smoke passed (the gate keys on family/lane/dataset, any venv)
            grep -Fx -f "$PASS_IDS" "$ids" > "$ids.pass" 2> /dev/null || true
            board_run "$(venv_arm "$arm")" "$ids.pass" "$RES/board-$arm-$rows" ${cap[@]+"${cap[@]}"}; r=$?
        else
            board_run "$(venv_arm "$arm")" "$ids" "$RES/board-$arm-$rows" --rows "$rows" ${cap[@]+"${cap[@]}"}; r=$?   # a neighbouring shape: no gate
        fi
        [ $r -gt $rc ] && rc=$r
    done < "$RES/plan-owed/units.tsv"
    phase_end $rc
fi
if [ "$OWED" = 1 ] && part drivers && [ -s "$RES/plan-owed/drivers.json" ]; then
    phase drivers
    run_owed_drivers "$RES/plan-owed/drivers.json"
    phase_end $?
fi

# ---- compare ----------------------------------------------------------------------------------------------------
if part compare; then
    phase compare
    NR=()
    for f in "$RES/plan/not-planned.json" "$RES/plan/smoke-failed.json"; do [ -f "$f" ] && NR+=(--not-run "$f"); done
    # the columns are the prior wheels timed on this box in this job; the stored compare is a reference column only
    mapfile -t PB < <(prior_board_args "")
    RF=()
    [ -f "$REF_COMPARE" ] && RF=(--reference-compare "$REF_COMPARE")
    $PY tools/paper_compare.py compare --recipes "$RECIPES" --board "$RES/board/board.json" "${NR[@]}" "${PB[@]}" "${RF[@]}" \
        --out-md "$RES/COMPARE-$V.md" --out-json "$RES/compare-$V.json" --summary "$RES/SUMMARY-$V.txt" \
        > "$RES/logs/compare.log" 2>&1
    rc=$?; head -n 3 "$RES/logs/compare.log"
    if [ "$OWED" = 1 ]; then
        $PY tools/measure_owed.py report --owed "$OWED_FILE" --plan-dir "$RES/plan-owed" --results "$RES" \
            --vendor "$V" ${LAST_PRIOR:+--prior-version "$LAST_PRIOR"} > "$RES/logs/owed-report.log" 2>&1
        r=$?; head -n 2 "$RES/logs/owed-report.log"; [ $r -gt $rc ] && rc=$r
    fi
    phase_end $rc
fi
echo finished > "$J/.phase"
exit "$WORST"
