#!/bin/bash
# GPU native checks on the rented L40S pod: the device column's TestSuite check files built as group programs
# (tools/ops/native_testsuite.sh build, lane ts-build 2026-11-22; the registry and its one run_gpu program are gone),
# then each check file matching NC_FILTER (default every check) as its own process, with a per-check timeout. The CPU
# column runs on the free GitHub runners (.github/workflows/native-checks.yml) and the CPU box; both write the same
# results/<check>.txt through tools/ops/native_checks_report.py result, read by its summary and report.
#
# Launch (results persist to R2 from the pod through tools/box_persist_run.sh):
#   tools/nvidia_central.sh submit measure --note native-checks-gpu tools/ops/native_checks_gpu_job.sh
#   [--env NC_FILTER=spectral_%]   (`%` reads as `*`: submit --env refuses `*`; comma list of globs over names)
#   [--env NC_TIMEOUT=900] [--env NC_JOBS=K] [--env NC_EXPECT_COMMIT=<sha>]  (refuse a tree synced from another commit)
# Read:  tools/box_persist.py fetch --tag <tag>; python3 tools/ops/native_checks_report.py report <dir>
#
# Parallel plan (AGENTS.md hard rules on parallelism): the build is G group compiles at once (G = min(cgroup cores,
# memory / one group's peak), native_testsuite.sh build), timed with its peak RSS as the row build-gpu. The checks are
# independent processes, run K at once with xargs -P; each check splits its own host work across the cores, and the
# device parts share the one GPU. K = min(cgroup cores, memory available x 0.8 / the largest check's peak RSS
# (NC_CHECK_PEAK_MB, default 2048: the largest of 269 checks was 1.64 GB on 2026-11-18)), unless NC_JOBS. Each check
# runs under /usr/bin/time -v; SUMMARY.md states the peak RSS of the largest check, the memory available, the K memory
# alone would allow, and the busy cores of total of the build and of the checks (cgroup CPU time, else /proc/stat).
#
# Exit status: 0 when every check passed, 1 when any failed or timed out (rc 124) or the build failed, 2 on a setup
# failure.
set -u

PY=python3
python3 -c 'import sys' 2>/dev/null || PY="pixi run python3"

if [ "${1:-}" = "--one" ]; then
    # One check (run by xargs): its TestSuite binary, rc, seconds and the first error lines into results/<check>.txt
    # (tools/ops/native_testsuite.sh run, which also localises a crash to the test).
    name=$2; RES=$3; BIN_DIR=${4:?--one needs the build directory}
    IN_TIMEOUT="${NC_TIMEOUT:-900}" bash tools/ops/native_testsuite.sh run gpu "$BIN_DIR" "$RES" "$name"
    echo "[$(date +%H:%M:%S)] $name done"
    exit 0
fi

SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
RES=${MOJOLEARN_RESULTS_DIR:?native_checks_gpu_job.sh needs MOJOLEARN_RESULTS_DIR (set by tools/box_persist_run.sh)}
mkdir -p "$RES" && RES="$(cd "$RES" && pwd)"
mkdir -p "$RES/results" "$RES/logs" "$RES/time"
TREE=${NC_TREE:-/root/mojolearn-measure}
cd "$TREE" || { echo "no tree $TREE"; exit 2; }
# fixture cases are tracked as one archive each (tools/fixture_packs.py): recreate their directories
[ -f tools/fixture_packs.py ] && python3 tools/fixture_packs.py unpack > /dev/null
export PATH="/root/.pixi/bin:$PATH"
say() { echo "[$(date +%H:%M:%S) native-checks-gpu] $*"; }
[ -x /usr/local/cuda/bin/ptxas ] && export MODULAR_NVPTX_COMPILER_PATH=${MODULAR_NVPTX_COMPILER_PATH:-/usr/local/cuda/bin/ptxas}
if [ ! -x /usr/bin/time ]; then
    (apt-get install -y -q time > "$RES/apt-time.log" 2>&1) || say "/usr/bin/time unavailable: peak RSS not recorded"
fi

# The commit under test. tools/dev_pod.sh sync leaves the tree's git HEAD at the worktree's merge base with main and
# applies the worktree's diff on top, so `git rev-parse HEAD` names main, not the code that is built (nvc1-0004,
# 2026-11-22, printed main's 49f2b12e24). The synced commit is in .git/devpod_synced (commit=, worktree_dirty=,
# patch_sha256=); a tree without that record (a plain checkout) is its HEAD. NC_EXPECT_COMMIT, when given, must be it.
. "$(dirname "$SELF")/tree_commit.sh"
SYNC_REC="$(git rev-parse --absolute-git-dir 2>/dev/null)/devpod_synced"
SOURCE=$(tree_commit .)
if [ -f "$SYNC_REC" ] && [ "$SOURCE" != "$(git rev-parse HEAD 2>/dev/null)" ]; then
    SOURCE_NOTE="synced commit (box HEAD $(git rev-parse --short=10 HEAD 2>/dev/null) is its merge base; worktree_dirty=$(sed -n 's/^worktree_dirty=//p' "$SYNC_REC"), patch sha256 $(sed -n 's/^patch_sha256=//p' "$SYNC_REC" | cut -c1-16))"
else
    SOURCE=${SOURCE:-unknown}
    SOURCE_NOTE="tree HEAD (no current sync record)"
fi
if [ -n "${NC_EXPECT_COMMIT:-}" ] && [ "${SOURCE#"$NC_EXPECT_COMMIT"}" = "$SOURCE" ]; then
    say "the tree holds $SOURCE, not NC_EXPECT_COMMIT=$NC_EXPECT_COMMIT: sync the commit to test first"; exit 2
fi
say "source $SOURCE in $TREE: $SOURCE_NOTE"
echo "$SOURCE" > "$RES/source-commit.txt"
pixi install > "$RES/pixi-install.log" 2>&1 || { say "pixi install failed: $RES/pixi-install.log"; tail -n 20 "$RES/pixi-install.log"; exit 2; }
pixi run mojo --version > "$RES/compiler-version.log" 2>&1

FILTER=${NC_FILTER:-*}
$PY tools/ops/native_checks_report.py list --column gpu --filter "$FILTER" > "$RES/planned.txt" \
    || { say "check listing failed"; exit 2; }
n=$(grep -c . "$RES/planned.txt")
[ "$n" -gt 0 ] || { say "no check matches $FILTER"; exit 2; }

# The gitignored generated modules the checks import (archive Unicode table, tokenizer table), as the binding
# builds write them (bindings/merged_build.sh).
$PY tools/generated_sources.py prepare --repo "$PWD" < /dev/null > "$RES/logs/generated-sources.log" 2>&1 \
    || { say "generated sources failed (logs/generated-sources.log)"; exit 2; }

# Busy cores of total, sampled every 15 s while the build and the tasks run (part of this job, not a watcher), and
# the allocation's cores and memory: tools/ops/box_cores_lib.sh (busy_sampler, busy_of, alloc_cores, alloc_mem_kb).
. "$(dirname "$SELF")/box_cores_lib.sh"

# The build: the device column's TestSuite files as group programs. The binaries are built OUTSIDE the results dir
# (2026-11-21: the nvc-measure job persisted a 144 MB bin/run_gpu to R2 and the Mac; a job with only our own arms keeps
# no binaries, AGENTS.md "What a measurement keeps"): tools/box_persist_run.sh tars only $RES.
BIN_DIR=${NC_BIN_DIR:-$RES.work/bin}
mkdir -p "$BIN_DIR"
say "building the device column ($n checks planned)"
bsampler=$(busy_sampler "$RES/busy_cores_build.txt")
bstart=$(date +%s)
# NC_TS_CACHE: a group program whose inputs did not change since an earlier job on this pod is copied, not compiled
# (tools/ops/native_testsuite_cache.py; outside the results dir, so nothing of it is persisted). NC_TS_CACHE= (empty)
# compiles everything.
export NC_TS_CACHE="${NC_TS_CACHE-${PC_CACHE_ROOT:-/root/pc-cache}/ts-cache}"
IN_FILTER="$FILTER" bash tools/ops/native_testsuite.sh build gpu "$BIN_DIR" "$RES" > "$RES/logs/build-gpu.log" 2>&1
brc=$?
say "$(tail -n 1 "$RES/logs/build-gpu.log")"
kill "$bsampler" 2>/dev/null
# the largest single compile's peak RSS (a group program's), from the build's /usr/bin/time records
brss=$(cat "$RES"/logs/build-ts-*.log.time 2>/dev/null | awk -F': *' '/Maximum resident set size/ {if ($2>m) m=$2} END {if (m) print m}')
$PY tools/ops/native_checks_report.py result build-gpu --log "$RES/logs/build-gpu.log" --rc "$brc" \
    --seconds "$(( $(date +%s) - bstart ))" ${brss:+--maxrss-kb "$brss"} --out-dir "$RES/results"
say "build rc=$brc in $(( $(date +%s) - bstart ))s"

# Cores of the allocation (cgroup quota, affinity), not the host's visible CPUs.
cores=$(alloc_cores)
mem_kb=$(alloc_mem_kb)
# K = min(cores, MemAvailable x 0.8 / the largest check's peak RSS): every core of the allocation, unless memory
# measured per check says fewer (the build's RSS is not a check's and is not in this figure)
check_kb=$(( ${NC_CHECK_PEAK_MB:-2048} * 1024 ))
bymem=$(( mem_kb * 8 / 10 / check_kb ))
K=${NC_JOBS:-$(( bymem < cores ? bymem : cores ))}
[ "$K" -lt 1 ] && K=1
say "$n checks, filter $FILTER, cores $cores, K=$K (memory alone allows $bymem at ${NC_CHECK_PEAK_MB:-2048} MB a check), memory available $(( mem_kb / 1048576 )) GB, timeout ${NC_TIMEOUT:-900}s"
sampler=$(busy_sampler "$RES/busy_cores.txt")


started=$(date +%s)
if [ "$brc" -eq 0 ]; then
    xargs -a "$RES/planned.txt" -P "$K" -I{} bash "$SELF" --one {} "$RES" "$BIN_DIR" > "$RES/progress.log" 2>&1
else
    say "build could not start: no check runs (every planned check reads MISSING); $RES/results/build-gpu.txt, full log $RES/logs/build-gpu.log"
fi
kill "$sampler" 2>/dev/null
wall=$(( $(date +%s) - started ))

# the checks only: results/build-gpu.txt holds the build's RSS, not a check's
checkfiles=(); for f in "$RES"/results/*.txt; do [ "$(basename "$f")" = build-gpu.txt ] || checkfiles+=("$f"); done
peak_kb=0
[ "${#checkfiles[@]}" -gt 0 ] && peak_kb=$(awk -F': *' '/^maxrss_kb/ {if ($2>m) m=$2} END {print m+0}' "${checkfiles[@]}")
busy=$(busy_of "$RES/busy_cores.txt")
bbusy=$(busy_of "$RES/busy_cores_build.txt")
say "BUSY build: $bbusy of $cores cores; checks: $busy of $cores cores"
{
    echo "host: $(hostname) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -n 1)"
    echo "source: $SOURCE ($SOURCE_NOTE) compiler: $(head -n 1 "$RES/compiler-version.log")"
    echo "filter: $FILTER; checks: $n; build rc $brc; wall: ${wall}s; timeout per check: ${NC_TIMEOUT:-900}s"
    echo "parallel: K=$K concurrent checks on $cores cores (K = min(cores, memory / ${NC_CHECK_PEAK_MB:-2048} MB) unless NC_JOBS); busy cores of $cores: checks $busy, build $bbusy"
    if [ -n "${brss:-}" ]; then echo "build: largest compile's peak RSS $(( brss / 1024 )) MB (a group program; not a check's)"
    else echo "build: peak RSS not recorded"; fi
    if [ "$peak_kb" -gt 0 ]; then
        echo "peak RSS of the largest check: $(( peak_kb / 1024 )) MB; memory available $(( mem_kb / 1024 )) MB;" \
             "K memory alone allows: $(( mem_kb * 8 / 10 / peak_kb ))"
    else
        echo "peak RSS: not recorded (/usr/bin/time missing)"
    fi
} > "$RES/notes.txt"
$PY tools/ops/native_checks_report.py summary "$RES/results" --planned "$RES/planned.txt" --notes "$RES/notes.txt" \
    --out "$RES/SUMMARY.md"
rc=$?
$PY tools/ops/native_checks_report.py report "$RES/results" > "$RES/REPORT.txt"
cat "$RES/notes.txt"; head -n 40 "$RES/REPORT.txt"
exit "$rc"
