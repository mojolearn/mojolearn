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
finish() {
    local code=$?
    trap - EXIT
    [ -n "$SAMPLER" ] && kill "$SAMPLER" 2> /dev/null
    [ "$WORST" -gt "$code" ] && code=$WORST
    phase_partial "$code"     # a phase the cap or a crash cut: its `partial` line in phases.tsv (box_job_lib.sh)
    printf '{"vendor":"%s","mode":"%s","version":"%s","source_commit":"%s","prior":"%s","prior_versions":"%s","host_id":"%s","tree_head":"%s","phase":"%s","exit":%d,"job_script_sha256":"%s","recipes_sha256":"%s"}\n' \
        "$V" "$MODE" "${CVER:-$VER}" "${PC_SOURCE_COMMIT:-}" "$PRIOR" "$PRIOR_VERSIONS" "${HOST_ID:-}" "$(git -C "$TREE" rev-parse HEAD 2>/dev/null)" "$PHASE" \
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

# ---- the host probe (every submit, before anything else uses the GPU): tools/ops/box_job_lib.sh host_probe_phase
host_probe_phase

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
for pv in ${PRIOR_VERSIONS//,/ }; do
    [ "$MODE" = published ] && [ "$pv" = "$VER" ] && continue      # the version under test is the new column
    add_arm "prior-$pv" --mojolearn-version "$pv"; PRIOR_ARMS+=("prior-$pv")
done
# the owed rows name the last release "prior" (owed-pre0867.json, measure_owed units, measure_owed_driver --arm):
# LAST_PRIOR's venv, none when the job times no last release (the default: the paper baseline only)
venv_arm() { if [ "$1" = prior ]; then echo "prior-${LAST_PRIOR:-none}"; else echo "$1"; fi; }
OWED_PRIOR=(); [ -n "$LAST_PRIOR" ] && OWED_PRIOR=(--prior)   # measure_owed plans the owed prior arm only then
# >>> b-arms (tools/test_measure_owed.py runs this block on a fake PC_B_ARMS file)
B_NAMES=()
if [ "$OWED" = 1 ] && [ -n "${PC_B_ARMS:-}" ]; then
    [ -s "$PC_B_ARMS" ] || say "PC_B_ARMS=$PC_B_ARMS: no such file on the box (or empty); no B arm runs"
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
# <<< b-arms

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
# PC_PARTS=targeted (= tg-install,tg-smoke,tg-board,tg-compare; each can be given alone to split a job over the cap:
# the boards resume from board.json) with PC_ROWS=<rows file on the box> (tools/ops/targeted_rows.py writes it from a
# compare and the changed lanes; tools/ops/measure_candidate.py stage --rows uploads it and refuses one whose estimate
# is over the cap, naming the split). Our arms only, the same machinery as board-all on the listed races: the plan
# (every submit, CPU, seconds: targeted_rows.py plan = the board's current plan of the listed races, in rows-file order,
# and the recipes of those races only) -> every arm's venv at once -> every arm's smoke at once in one shard set (the
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
# smokes and the plans use every cgroup core.
[[ "$PARTS" == *",targeted,"* ]] && PARTS="$PARTS,tg-install,tg-smoke,tg-board,tg-compare,"
if part tg-install || part tg-smoke || part tg-board || part tg-compare; then
    TG="$RES/plan-tg"; TGO="$RES/plan-tg-owed"
    mkdir -p "$TG"
    { [ -n "${PC_ROWS:-}" ] && [ -s "$PC_ROWS" ]; } || { say "targeted: PC_ROWS (the rows file on the box) is missing or empty"; exit 2; }
    cp "$PC_ROWS" "$TG/rows.tsv"
    phase tg-plan
    $PY tools/ops/targeted_rows.py plan --rows "$TG/rows.tsv" --vendor "$V" --planner tools/bench_board.py \
        --recipes "$RECIPES" --out-dir "$TG" --chunk-s "${PC_TG_CHUNK_S:-600}" > "$RES/logs/tg-plan.log" 2>&1
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
    if part tg-install; then
        phase tg-install
        pids=(); names=()
        for arm in "${ARMS[@]}"; do      # every arm at once (CPU): the candidate, the priors, the B sets
            # shellcheck disable=SC2086
            $PY tools/bench_board.py "${COMMON_TG[@]}" ${ARM_ARGS[$arm]} --ours-install-only --install-only \
                --venv "$CACHE_ROOT/venv-$arm" --cache "$CACHE_ROOT/i-$arm" --wheel-manifest "${ARM_MAN[$arm]}" \
                --race-ids @"$TG/race-ids-full.txt" --out "$RES/install-tg-$arm" > "$RES/logs/install-tg-$arm.log" 2>&1 &
            pids+=($!); names+=("$arm")
        done
        rc=0
        for i in "${!pids[@]}"; do
            wait "${pids[$i]}"; r=$?; say "install ${names[$i]} rc=$r"; [ $r -gt $rc ] && rc=$r
        done
        phase_end $rc
    fi
    [ -f "${ARM_MAN[candidate]}" ] || { say "the candidate arm has no wheel manifest (logs/install-tg-candidate.log)"; exit 6; }
    cat "$TG/race-ids-smoke.txt" "$TGO/smoke-ids.txt" 2> /dev/null | sort -u > "$TG/smoke-ids-all.txt"
    for arm in ${PRIOR_ARMS[@]+"${PRIOR_ARMS[@]}"}; do PRIOR_SMOKE_IDS[$arm]="$TG/race-ids-smoke.txt"; done
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
        CUR_IDS=$ids; CUR_BOARD=$out
        # shellcheck disable=SC2046
        $PY tools/bench_board.py "${COMMON_TG[@]}" $(run_as "$arm") --cache "$CACHE_ROOT/c1" --race-ids @"$ids" \
            "$@" ${ra[@]+"${ra[@]}"} --drop-race-scratch --out "$out" >> "$RES/logs/$(basename "$out").log" 2>&1
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
        disk_gate tg-board "$left" "$nb" "${PC_EVIDENCE_MB_PER_RACE:-40}"
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
        # the owed A/B arms: each unit chunk by chunk, its base (the candidate) and the arm back to back
        if [ -s "$TGO/units.tsv" ]; then
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
                $PY tools/ops/targeted_rows.py chunks --ids "$use" --chunk-s "${PC_TG_CHUNK_S:-600}" \
                    --out-dir "$TGO/chunks-$arm-$rows" | tail -n 1
                for c in "$TGO/chunks-$arm-$rows"/chunk-*.txt; do
                    [ -f "$c" ] || continue
                    tg_board candidate "$c" "$RES/board-tg-ab-$arm-base-$rows" "$abrows" "${SJT[@]}"; r=$?; [ $r -gt $rc ] && rc=$r
                    tg_board "$arm" "$c" "$RES/board-tg-ab-$arm-$rows" "$abrows" "${SJT[@]}"; r=$?; [ $r -gt $rc ] && rc=$r
                done
            done 3< "$TGO/units.tsv"
        fi
        phase_end $rc
    fi
    if part tg-compare; then
        phase tg-compare
        NR=()
        for f in "$TG/not-planned.json" "$TG/smoke-failed.json"; do [ -f "$f" ] && NR+=(--not-run "$f"); done
        mapfile -t PB < <(prior_board_args tg-)
        RF=()
        [ -f "$REF_COMPARE" ] && RF=(--reference-compare "$REF_COMPARE")
        $PY tools/paper_compare.py compare --recipes "$TG/recipes.json" --board "$RES/board-tg/board.json" "${NR[@]}" \
            "${PB[@]}" "${RF[@]}" --best-ever "${PC_BEST_EVER:-$TREE/bench/results/board_best_ever/$V.json}" \
            --out-md "$RES/COMPARE-TG-$V.md" --out-json "$RES/compare-tg-$V.json" \
            --summary "$RES/SUMMARY-TG-COMPARE-$V.txt" > "$RES/logs/compare-tg.log" 2>&1
        rc=$?; head -n 3 "$RES/logs/compare-tg.log"
        $PY tools/ops/targeted_rows.py summary --compare "$RES/compare-tg-$V.json" --rows "$TG/rows.tsv" \
            --results "$RES" --ids "$TG/race-ids-full.txt" --units "$TGO/units.tsv" --prior-versions "$PRIOR_VERSIONS" \
            --out "$RES/SUMMARY-TG-$V.txt" > "$RES/logs/summary-tg.log" 2>&1
        r=$?; head -n 3 "$RES/logs/summary-tg.log"; [ $r -gt $rc ] && rc=$r
        phase_end $rc
    fi
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
    $PY tools/bench_board.py "${COMMON[@]}" $(run_as "$arm") --cache "$CACHE_ROOT/c1" --race-ids @"$ids" \
        "${SJ[@]}" "$@" --drop-race-scratch --out "$out" > "$RES/logs/$(basename "$out").log" 2>&1
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
        if [ "$rows" = full ]; then
            # full rows: only the races whose smoke passed (the gate keys on family/lane/dataset, any venv)
            grep -Fx -f "$PASS_IDS" "$ids" > "$ids.pass" 2> /dev/null || true
            board_run "$(venv_arm "$arm")" "$ids.pass" "$RES/board-$arm-$rows"; r=$?
        else
            board_run "$(venv_arm "$arm")" "$ids" "$RES/board-$arm-$rows" --rows "$rows"; r=$?   # a neighbouring shape: no gate
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
        # a driver row's B arms (its "arms" names a staged B set, e.g. restore-mamba-backward-glue's
        # mamba-backward-generic) run on that set's venv; measure_owed_driver.py skips an arm the row does not name
        for arm in candidate ${LAST_PRIOR:+prior} ${B_NAMES[@]+"${B_NAMES[@]}"}; do
            va=$(venv_arm "$arm")
            [ -x "${ARM_PY[$va]:-/nonexistent}" ] || continue
            "${ARM_PY[$va]}" tools/measure_owed_driver.py --owed "$OWED_FILE" --row "$row" --arm "$arm" \
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
