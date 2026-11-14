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
#            candidate venv (--python-env --skip-install: no per-shard install); shards = min(nproc/2, MemAvailable
#            / PC_SMOKE_MEM_GB (default 12), NVIDIA GPU memory / PC_SMOKE_GPU_GB (default 3), PC_SMOKE_MAX (default
#            16), races); PC_SMOKE_SHARDS overrides. The shards are host-bound, not GPU-bound (2000-row races: worker
#            start, import, data read; see the smoke phase). After it each shard's work/ and evidence/ and the
#            caches c2.. are deleted (BOARD.md, board.json, smoke.json and logs/ stay)
#   board    TIMED, SERIAL BY NATURE: one board process at a time on the one GPU with nothing else running
#            (timing must not share the GPU; AGENTS.md "Full-machine CPU race resources"; the drivers themselves
#            use every core, no thread caps): candidate paper+owed full rows, then each owed unit
#            (tools/measure_owed.py units.tsv: neighbouring --rows groups, the prior arm, the B arms), then the
#            driver rows (tools/measure_owed_driver.py per arm interpreter)
#   compare  paper_compare compare (COMPARE/compare/SUMMARY-<vendor>) and measure_owed report (OWED*-<vendor>)
#   board-all (PC_PARTS=board-all; the full-board NVIDIA screen, Andrew 2026-11-13) instead of the above: every race
#            the board plans (tools/paper_compare.py ids-all), our arm only, smoke in the same parallel shards, one
#            timed board, compare against the paper recipes and every row's best time ever (--best-ever); see the
#            board-all block below and docs/plans/PRE_0867_MEASUREMENT.md "Full-board NVIDIA screen"
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
# Inputs on the box: the datasets under $PC_DATA_ROOT (default /root/datasets/gbm-bench), the list being
# tools/measure_owed.py data-keys (the board's data_needs over the paper and owed races; checked after the plan,
# exit 4 when one is missing), staged by tools/ops/measure_candidate.py stage through tools/dataset_store.sh stage.
# At start the job deletes the box_persist_run.sh upload leftovers of finished jobs. A phase that hits a full disk
# (ENOSPC in a log written during it, or under PC_DISK_MIN_FREE_MB left) stops the job with exit 28, and a smoke,
# board or owed phase refuses to start (exit 28) when the free disk is below its predicted evidence (disk_gate,
# printed as a DISK line). Knobs: PC_GPU_TOKEN (the paper's GPU,
# default L40S / MI300X). NVIDIA refuses another GPU (exit 3) unless PC_ALLOW_HW_DIFF=1; AMD records HW-DIFF and
# goes on unless PC_REQUIRE_SAME_GPU=1 (exit 3).
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
    local rc=$1 full
    full=$(disk_full_in_phase)
    [ -n "$full" ] && rc=$EXIT_DISK
    printf '%s\t%s\t%s\t%s\n' "$PHASE" "$rc" "$(( $(date +%s) - P_T0 ))" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$J/phases.tsv"
    say "phase $PHASE rc=$rc $(( $(date +%s) - P_T0 )) s"
    [ "$rc" -gt "$WORST" ] && WORST=$rc
    if [ -n "$full" ]; then
        # fail soft stops here: the next phase would run into the same full disk (2026-11-14: the owed prior and all
        # 653 smoke races of the full-board screen failed one after the other on ENOSPC)
        say "DISK FULL in phase $PHASE: $full; $(free_gb) GB free on $(df -P "$RES" 2> /dev/null | awk 'NR==2 {print $6}'); stopping (exit $EXIT_DISK)"
        exit "$EXIT_DISK"
    fi
    return 0
}
# ---- disk (2026-11-14) ---------------------------------------------------------------------------------------------
EXIT_DISK=28    # ENOSPC: the job stopped on a full disk, or refused to start a phase whose evidence would not fit
nlines() { local c; c=$(grep -c . "$1" 2> /dev/null); echo "${c:-0}"; }
free_gb() { df -Pk "$RES" 2> /dev/null | awk 'NR==2 {printf "%.1f", $4 / 1048576}'; }
# non-empty (the evidence) when this phase hit a full disk: a log written during it that says ENOSPC, or less than
# PC_DISK_MIN_FREE_MB (512) left at its end
disk_full_in_phase() {
    local hit kb
    # job/.phase is written when the phase starts (phase()): a log modified after it was written during this phase.
    # job/, command.log and persist.log carry this job's own messages (a DISK FULL line of an earlier submit): not evidence
    hit=$(find "$RES" -maxdepth 3 -path "$J" -prune -o -type f -name '*.log' ! -name command.log ! -name persist.log \
        -newer "$J/.phase" -exec grep -l -m 1 -E 'No space left on device|Errno 28|ENOSPC' {} + 2> /dev/null | head -n 3 | tr '\n' ' ')
    if [ -n "$hit" ]; then echo "ENOSPC in ${hit% }"; return; fi
    kb=$(df -Pk "$RES" 2> /dev/null | awk 'NR==2 {print $4}')
    [ -n "$kb" ] && [ "$kb" -lt $(( ${PC_DISK_MIN_FREE_MB:-512} * 1024 )) ] && echo "only $(( kb / 1024 )) MB free at the phase end"
}
# refuse to START a phase that writes race evidence when the free disk is below the prediction:
# races x PC_EVIDENCE_MB_PER_RACE (smoke: PC_SMOKE_EVIDENCE_MB_PER_RACE) + boards x PC_LOADED_MB (the loaded native
# modules, retained once per board dir: tools/bench_board_evidence.py loaded_module_ref) + PC_DISK_RESERVE_GB.
# The per-race default is the retained receipts, logs and records (the .so bytes are no longer in each race,
# 2026-11-14); raise it for a board of large-output races.
disk_gate() {   # <what> <races> <board processes> <MB per race>
    local what=$1 races=$2 boards=$3 per=$4 need_mb free_kb
    need_mb=$(( races * per + boards * ${PC_LOADED_MB:-400} + ${PC_DISK_RESERVE_GB:-5} * 1024 ))
    free_kb=$(df -Pk "$RES" 2> /dev/null | awk 'NR==2 {print $4}')
    [ -n "$free_kb" ] || { say "DISK $what: cannot read the free space of $RES; not gated"; return 0; }
    say "DISK $what: $races races x $per MB + $boards boards x ${PC_LOADED_MB:-400} MB + reserve ${PC_DISK_RESERVE_GB:-5} GB = $need_mb MB needed; $(( free_kb / 1024 )) MB free"
    if [ $(( free_kb / 1024 )) -lt "$need_mb" ]; then
        say "DISK $what refused: free space is below the predicted evidence; free disk or lower the estimate (exit $EXIT_DISK)"
        printf '%s\t%s\t%s\t%s\n' "$what-disk-refused" "$EXIT_DISK" 0 "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$J/phases.tsv"
        WORST=$EXIT_DISK
        exit "$EXIT_DISK"
    fi
}
# busy cores of total every 60 s (job/cores.log) and a BUSY line in job.log every 5 minutes.
# BUSY-NEGATIVE FIX (2026-11-14, L40S pod nvc1: `busy=-18.3 total=128`): the old sampler took (all - idle - iowait)
# over the /proc/stat deltas and scaled by nproc. iowait is not monotonic (the kernel documents that it can go down),
# and a container's /proc/stat and nproc describe the host, not the pod. Now: the pod's own CPU time from its cgroup
# (cgroup v2 cpu.stat usage_usec, else v1 cpuacct.usage), busy = delta CPU time / delta wall time; total = the
# allocation (cgroup v2 cpu.max quota/period, else nproc, which honours affinity). Without a cgroup counter it falls
# back to /proc/stat with busy = delta(user+nice+system+irq+softirq+steal) over delta(those + idle + iowait), scaled
# by the cpu lines /proc/stat lists. A delta that goes backwards (counter reset) is not a sample: it is logged as
# `busy=NA` and the next sample starts from it; a value outside 0..total is clamped and marked `clamped`.
alloc_cores() {
    local q per n
    n=$(nproc 2> /dev/null || echo 1)
    if read -r q per < /sys/fs/cgroup/cpu.max 2> /dev/null && [ "$q" != max ] && [ "${per:-0}" -gt 0 ] 2> /dev/null; then
        q=$(( (q + per - 1) / per ))
        [ "$q" -ge 1 ] && [ "$q" -lt "$n" ] && n=$q
    elif read -r q < /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2> /dev/null && read -r per < /sys/fs/cgroup/cpu/cpu.cfs_period_us 2> /dev/null \
            && [ "$q" -gt 0 ] 2> /dev/null && [ "${per:-0}" -gt 0 ] 2> /dev/null; then
        q=$(( (q + per - 1) / per ))     # cgroup v1 (the RunPod L40S pods)
        [ "$q" -ge 1 ] && [ "$q" -lt "$n" ] && n=$q
    fi
    echo "$n"
}
cpu_sample() {   # "<wall_usec> <cpu_usec> cgroup" or "<total ticks> <busy ticks> <cpus> procstat"
    local u
    u=$(awk '$1 == "usage_usec" {print $2}' /sys/fs/cgroup/cpu.stat 2> /dev/null)
    # %.0f, not %d: mawk's %d is 32-bit, so cpuacct.usage (ns since the container started) / 1000 stuck at
    # 2147483647 after about 36 CPU-minutes and every later sample read busy=0.0 (L40S nvc1, 2026-11-14).
    [ -n "$u" ] || u=$(awk '{printf "%.0f", $1 / 1000}' /sys/fs/cgroup/cpuacct/cpuacct.usage 2> /dev/null)
    if [ -n "$u" ]; then echo "$(( $(date +%s%N) / 1000 )) $u cgroup"; return; fi
    awk '/^cpu / {b=$2+$3+$4+$7+$8+$9; t=b+$5+$6} /^cpu[0-9]/ {c++} END {printf "%.0f %.0f %d procstat\n", t, b, c}' /proc/stat
}
busy_line() {   # <allocated cores> <phase> <previous cpu_sample> <current cpu_sample>: one cores.log line
    echo "$3 $4" | awk -v n="$1" -v t="$(date -u +%H:%M:%S)" -v ph="$2" '
        { if ($3 == "cgroup" && $6 == "cgroup") { dw = $4 - $1; dc = $5 - $2; ok = (dw > 0 && dc >= 0); b = ok ? dc / dw : 0; src = "cgroup" }
          else { dt = $5 - $1; db = $6 - $2; ok = (dt > 0 && db >= 0 && db <= dt); b = ok ? db / dt * $7 : 0; src = "procstat" }
          if (!ok) { printf "%s busy=NA total=%d phase=%s source=%s (counter went backwards; next sample from here)", t, n, ph, src; exit }
          c = ""; if (b < 0) { b = 0; c = " clamped" } if (b > n) { b = n; c = " clamped" }
          printf "%s busy=%.1f total=%d phase=%s source=%s%s", t, b, n, ph, src, c }'
}
cores_sampler() {
    local n prev cur i=0 line
    n=$(alloc_cores)
    prev=$(cpu_sample)
    while sleep 60; do
        cur=$(cpu_sample)
        line=$(busy_line "$n" "$(cat "$J/.phase" 2>/dev/null)" "$prev" "$cur")
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
say "BUSY at start: nproc $(nproc), allocation $(alloc_cores) cores, $(awk '/MemAvailable/ {printf "%d GiB available", $2/1048576}' /proc/meminfo 2>/dev/null)"
if [ "$V" = nvidia ]; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > "$J/gpu.txt" 2>&1
else
    { rocm-smi --showproductname; rocm-smi --showmeminfo vram; } > "$J/gpu.txt" 2>&1
fi
NGPU=$(grep -c . "$J/gpu.txt" 2> /dev/null || echo 0)
[ "$V" = nvidia ] && [ "$NGPU" -gt 1 ] && say "NOTE: $NGPU GPUs; the timed boards use one (serial by nature)"
# NVIDIA: every GPU must be the paper's (L40S) unless PC_ALLOW_HW_DIFF=1 (2026-11-13: an out-of-stock L40S got a
# 2x RTX 6000 Ada rented; compare on the same device). AMD: HW-DIFF is recorded and the job goes on unless
# PC_REQUIRE_SAME_GPU=1 (the MI325X run of 2026-11-13). tools/paper_compare.py flags every row HW-DIFF either way.
# (nvidia-smi prints one name per GPU, so every line must carry the token; rocm-smi prints headers, so any line)
gpu_other() {
    if [ "$V" = nvidia ]; then [ "$NGPU" -eq 0 ] || grep -v -q "$TOKEN" "$J/gpu.txt"
    else ! grep -q "$TOKEN" "$J/gpu.txt"; fi
}
if gpu_other; then
    say "HW-DIFF: the paper ran on $TOKEN; this box: $(head -c 300 "$J/gpu.txt" | tr '\n' ' ')"
    if [ "$V" = nvidia ] && [ "${PC_ALLOW_HW_DIFF:-0}" != 1 ]; then
        say "refused: NVIDIA is measured on an $TOKEN only; PC_ALLOW_HW_DIFF=1 runs it anyway with every row HW-DIFF"
        exit 3
    fi
    [ "$V" = amd ] && [ "${PC_REQUIRE_SAME_GPU:-0}" = 1 ] && exit 3
fi
# upload leftovers of finished box-persisted jobs: a killed upload leaves its piece (at most max_bytes since
# 2026-11-14) or, from an older runner, its whole staging tarball (36 GB on the L40S pod, 2026-11-13); this job's
# own and running jobs' are kept
prune_persist_staging() {
    local root f tag pid
    root=$(dirname "$(dirname "${MOJOLEARN_PERSIST_RUNNER:-/root/box-persist/bin/x}")")
    for f in "$root"/state/*/upload.tgz "$root"/state/*/readback.tgz "$root"/state/*/piece "$root"/state/*/piece.back; do
        [ -f "$f" ] || continue
        tag=$(basename "$(dirname "$f")")
        [ "$tag" = "${MOJOLEARN_PERSIST_TAG:-}" ] && continue
        pid=$(sed -n 's/^PID=//p' "$root/active/$tag.env" 2> /dev/null | head -1)
        [ -n "$pid" ] && kill -0 "$pid" 2> /dev/null && continue
        say "pruning $(du -sh "$f" 2> /dev/null | cut -f1) upload leftover of finished job $tag: $f"
        rm -f "$f" "$(dirname "$f")/upload.sha256" "$f.sha256"
    done
}
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
# the parallel smoke shard count for NRACE races (see the smoke phase for why it is host-bound): min(nproc/2,
# MemAvailable / PC_SMOKE_MEM_GB (12), NVIDIA GPU GB / PC_SMOKE_GPU_GB (3), PC_SMOKE_MAX (16), races);
# PC_SMOKE_SHARDS overrides. Sets S, MEMG and GPUG.
smoke_shards() {
    local nrace=$1 bymem
    MEMG=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo 2> /dev/null || echo 16)
    S=${PC_SMOKE_SHARDS:-$(( $(nproc) / 2 ))}
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

# ---- board-all: the full-board screen (Andrew, 2026-11-13) ------------------------------------------------------
# PC_PARTS=board-all (= all-plan,all-install,all-smoke,all-board,all-compare; each can be given alone to split a
# job over the 240-minute cap: the board resumes from board-all/board.json) runs EVERY race the board plans on this
# vendor (tools/paper_compare.py ids-all: every family and lane, identical, our arms only; opponents are the stored
# cells, never raced) instead of the paper flow, and compares the result against the paper recipes AND every row's
# best time ever (tools/paper_compare.py compare --best-ever: bench/results/board_best_ever/<vendor>.json, mined by
# tools/board_best_ever.py). The smoke runs in the smoke phase's parallel shards (smoke_shards); the timed board is
# one process on the one GPU, serial by nature (timing must not share the GPU). Results: plan-all/, install-all/,
# smoke-all-s<i>/, board-all/, COMPARE-ALL/compare-all/SUMMARY-ALL-<vendor>.
[[ "$PARTS" == *",board-all,"* ]] && PARTS="$PARTS,all-plan,all-install,all-smoke,all-board,all-compare,"
if part all-plan || part all-install || part all-smoke || part all-board || part all-compare; then
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
        # shellcheck disable=SC2086
        $PY tools/bench_board.py "${COMMON_ALL[@]}" ${ARM_ARGS[candidate]} --ours-install-only --install-only \
            --venv "$CACHE_ROOT/venv-candidate" --cache "$CACHE_ROOT/i-candidate" --wheel-manifest "${ARM_MAN[candidate]}" \
            --race-ids @"$PA/race-ids-full.txt" --out "$RES/install-all" > "$RES/logs/install-all.log" 2>&1
        rc=$?; say "install candidate rc=$rc $(grep -m 1 -h '^BOARD-INSTALL' "$RES/logs/install-all.log" 2>/dev/null)"
        phase_end $rc
    fi
    [ -f "${ARM_MAN[candidate]}" ] || { say "the candidate arm has no wheel manifest (logs/install-all.log)"; exit 6; }
    if part all-smoke; then
        phase all-smoke
        NRACE=$(grep -c . "$PA/race-ids-smoke.txt")
        smoke_shards "$NRACE"
        disk_gate all-smoke "$NRACE" "$S" "${PC_SMOKE_EVIDENCE_MB_PER_RACE:-5}"
        say "full-board smoke: $NRACE races in $S parallel shards (nproc $(nproc), MemAvailable ${MEMG} GiB, GPU ${GPUG:-?} GiB)"
        pids=()
        # shellcheck disable=SC2046
        for i in $(seq 1 "$S"); do
            $PY tools/bench_board.py "${COMMON_ALL[@]}" $(run_as candidate) --cache "$CACHE_ROOT/c$i" \
                --race-ids @"$PA/race-ids-smoke.txt" --smoke --drop-race-scratch --shard "$i/$S" \
                --out "$RES/smoke-all-s$i" > "$RES/logs/smoke-all-s$i.log" 2>&1 &
            pids+=($!)
        done
        src=0
        for p in "${pids[@]}"; do wait "$p"; r=$?; [ $r -gt $src ] && src=$r; done
        grep -h -E '^SMOKE (PASS|FAIL)' "$RES"/logs/smoke-all-s*.log | head -n 8
        for d in "$RES"/smoke-all-s*/; do rm -rf "${d}work" "${d}evidence"; done
        for c in "$CACHE_ROOT"/c*/; do case "$(basename "$c")" in c1) ;; c[0-9]*) rm -rf "$c" ;; esac; done
        say "full-board smoke scratch pruned; $(df -h /root 2> /dev/null | awk 'NR==2 {print $4 " free on /root"}')"
        phase_end $src
    fi
    SJA=()
    for f in "$RES"/smoke-all-s*-smoke/smoke.json; do [ -f "$f" ] && SJA+=(--smoke-json "$f"); done
    $PY tools/paper_compare.py pass-ids --full-ids "$PA/race-ids-full.txt" "${SJA[@]}" --out "$PA/race-ids-pass.txt" \
        > "$RES/logs/pass-ids-all.log" 2>&1
    tail -n 1 "$RES/logs/pass-ids-all.log"
    if part all-board; then
        phase all-board
        disk_gate all-board "$(nlines "$PA/race-ids-pass.txt")" 1 "${PC_EVIDENCE_MB_PER_RACE:-40}"
        if [ -s "$PA/race-ids-pass.txt" ]; then
            # shellcheck disable=SC2046
            $PY tools/bench_board.py "${COMMON_ALL[@]}" $(run_as candidate) --cache "$CACHE_ROOT/c1" \
                --race-ids @"$PA/race-ids-pass.txt" "${SJA[@]}" --out "$RES/board-all" > "$RES/logs/board-all.log" 2>&1
            rc=$?; say "board-all rc=$rc: $(tail -n 1 "$RES/logs/board-all.log" | head -c 200)"
        else
            say "no full-board race passed its smoke"; rc=6
        fi
        phase_end $rc
    fi
    if part all-compare; then
        phase all-compare
        NR=()
        for f in "$PA/not-planned.json" "$PA/smoke-failed.json"; do [ -f "$f" ] && NR+=(--not-run "$f"); done
        PR=()
        [ "$MODE" = candidate ] && [ -f "$PRIOR_COMPARE" ] && PR=(--prior-compare "$PRIOR_COMPARE" --prior-label "$PRIOR")
        $PY tools/paper_compare.py compare --recipes "$RECIPES" --board "$RES/board-all/board.json" "${NR[@]}" "${PR[@]}" \
            --best-ever "${PC_BEST_EVER:-$TREE/bench/results/board_best_ever/$V.json}" \
            --out-md "$RES/COMPARE-ALL-$V.md" --out-json "$RES/compare-all-$V.json" --summary "$RES/SUMMARY-ALL-$V.txt" \
            > "$RES/logs/compare-all.log" 2>&1
        rc=$?; head -n 4 "$RES/logs/compare-all.log"
        phase_end $rc
    fi
    echo finished > "$J/.phase"
    exit "$WORST"
fi

# ---- plan (CPU, seconds) -----------------------------------------------------------------------------------------
owed_plan() {   # the owed rows' plan for this job's B sets (plan-owed/plan.json records them)
    $PY tools/measure_owed.py plan --owed "$OWED_FILE" --paper-recipes "$RECIPES" --planner tools/bench_board.py \
        --vendor "$V" --b-arms "$B_CSV" --out-dir "$RES/plan-owed" > "$RES/logs/owed-plan.log" 2>&1
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
    chk=$($PY tools/measure_owed.py plan-check --plan-dir "$RES/plan-owed" --b-arms "$B_CSV" --vendor "$V" 2>&1)
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
DK_OWED=(); [ "$OWED" = 1 ] && DK_OWED=(--owed "$OWED_FILE")
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
if part smoke; then
    phase smoke
    NRACE=$(grep -c . "$SMOKE_IDS")
    # NOT GPU-bound (read from the code; not measured): a smoke race runs SMOKE_ROWS=2000 rows at the `small` shape,
    # so each shard's time is CPU and I/O on the host: a fresh worker process per arm (interpreter start, `import
    # mojolearn`, the GPU context), the dataset npz read and sliced, the evidence sealed. Each shard runs one race at a
    # time and mostly one core, which is why 5 shards (nproc/4) showed busy 2-4 of 20 cores (2026-11-12, AMD). So the
    # shard count is nproc/2, bounded by host memory (PC_SMOKE_MEM_GB per shard), by GPU memory (PC_SMOKE_GPU_GB per
    # shard's contexts and small buffers) and PC_SMOKE_MAX (16): smoke_shards.
    smoke_shards "$NRACE"
    disk_gate smoke "$NRACE" "$S" "${PC_SMOKE_EVIDENCE_MB_PER_RACE:-5}"
    say "smoke: $NRACE races in $S parallel shards (nproc $(nproc), MemAvailable ${MEMG} GiB, GPU ${GPUG:-?} GiB), one venv"
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
    # the smoke scratch is not needed past this point: each shard keeps BOARD.md, board.json, smoke.json and logs/
    # (what pass-ids and the persisted record read); its work/ and evidence/ go, and the shards' caches c2.. (the timed
    # boards use c1). The 80 GB L40S pod filled with 12 GB of smoke scratch on 2026-11-13.
    for d in "$RES"/smoke-s*/; do rm -rf "${d}work" "${d}evidence"; done
    for c in "$CACHE_ROOT"/c*/; do case "$(basename "$c")" in c1) ;; c[0-9]*) rm -rf "$c" ;; esac; done
    say "smoke scratch pruned; $(df -h /root 2> /dev/null | awk 'NR==2 {print $4 " free on /root"}')"
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
    disk_gate board "$(nlines "$PASS_IDS")" 1 "${PC_EVIDENCE_MB_PER_RACE:-40}"
    if [ -s "$PASS_IDS" ]; then board_run candidate "$PASS_IDS" "$RES/board"; rc=$?; else say "no race passed its smoke"; rc=6; fi
    phase_end $rc
fi
if [ "$OWED" = 1 ] && part owed && [ -s "$RES/plan-owed/units.tsv" ]; then
    phase owed
    # every owed unit's races (full rows may be fewer after the smoke gate: an upper bound), one board dir per unit
    disk_gate owed "$(cut -f3 "$RES/plan-owed/units.tsv" | xargs cat 2> /dev/null | grep -c . || true)" \
        "$(nlines "$RES/plan-owed/units.tsv")" "${PC_EVIDENCE_MB_PER_RACE:-40}"
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
        # a driver row's B arms (its "arms" names a staged B set, e.g. restore-mamba-backward-glue's
        # mamba-backward-generic) run on that set's venv; measure_owed_driver.py skips an arm the row does not name
        for arm in candidate prior ${B_NAMES[@]+"${B_NAMES[@]}"}; do
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
