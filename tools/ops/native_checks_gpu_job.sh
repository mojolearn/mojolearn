#!/bin/bash
# GPU native checks on the rented L40S pod: ONE build of checks/native_tests/run_gpu.mojo (every registered check of
# the device column), then the binary once per check name matching NC_FILTER (default every check), with a per-check
# timeout. The CPU column runs on the free GitHub runners (.github/workflows/native-checks.yml); both write the same
# results/<check>.txt through tools/ops/native_checks_report.py result, read by its summary and report.
#
# Launch (results persist to R2 from the pod through tools/box_persist_run.sh):
#   tools/nvidia_central.sh submit measure --note native-checks-gpu tools/ops/native_checks_gpu_job.sh
#   [--env NC_FILTER=spectral_%]   (`%` reads as `*`: submit --env refuses `*`; comma list of globs over names)
#   [--env NC_TIMEOUT=900] [--env NC_JOBS=K]
# Read:  tools/box_persist.py fetch --tag <tag>; python3 tools/ops/native_checks_report.py report <dir>
#
# Parallel plan (AGENTS.md hard rules on parallelism): the build is one `mojo build` (the compiler's own threads use
# the cores; it is the one serial stage and is timed and its peak RSS recorded as the row build-gpu). The checks are
# independent processes of the prebuilt binary, run K at once with xargs -P; each check splits its own host work
# across the cores, and the device parts share the one GPU. K = cgroup cores / 4 unless NC_JOBS. Each check runs
# under /usr/bin/time -v; SUMMARY.md states the peak RSS of the largest check, the memory available, the K memory
# alone would allow, and the busy cores of total sampled from /proc/stat.
#
# Exit status: 0 when every check passed, 1 when any failed or timed out (rc 124) or the build failed, 2 on a setup
# failure.
set -u

PY=python3
python3 -c 'import sys' 2>/dev/null || PY="pixi run python3"

if [ "${1:-}" = "--one" ]; then
    # One check (run by xargs): rc, seconds, peak RSS and the first error lines into results/<check>.txt.
    name=$2; RES=$3
    log="$RES/logs/$name.log"; tm="$RES/time/$name.txt"; start=$(date +%s)
    if [ -x /usr/bin/time ]; then
        /usr/bin/time -v -o "$tm" timeout -k 30 "${NC_TIMEOUT:-900}" pixi run "$RES/bin/run_gpu" "$name" > "$log" 2>&1
    else
        timeout -k 30 "${NC_TIMEOUT:-900}" pixi run "$RES/bin/run_gpu" "$name" > "$log" 2>&1
    fi
    rc=$?
    rss=$(awk -F': *' '/Maximum resident set size/ {print $2}' "$tm" 2>/dev/null)
    $PY tools/ops/native_checks_report.py result "$name" --log "$log" --rc "$rc" \
        --seconds "$(( $(date +%s) - start ))" ${rss:+--maxrss-kb "$rss"} --out-dir "$RES/results"
    echo "[$(date +%H:%M:%S)] $name rc=$rc"
    exit 0
fi

SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
RES=${MOJOLEARN_RESULTS_DIR:?native_checks_gpu_job.sh needs MOJOLEARN_RESULTS_DIR (set by tools/box_persist_run.sh)}
mkdir -p "$RES" && RES="$(cd "$RES" && pwd)"
mkdir -p "$RES/results" "$RES/logs" "$RES/time"
TREE=${NC_TREE:-/root/mojolearn-measure}
cd "$TREE" || { echo "no tree $TREE"; exit 2; }
export PATH="/root/.pixi/bin:$PATH"
say() { echo "[$(date +%H:%M:%S) native-checks-gpu] $*"; }
[ -x /usr/local/cuda/bin/ptxas ] && export MODULAR_NVPTX_COMPILER_PATH=${MODULAR_NVPTX_COMPILER_PATH:-/usr/local/cuda/bin/ptxas}
if [ ! -x /usr/bin/time ]; then
    (apt-get install -y -q time > "$RES/apt-time.log" 2>&1) || say "/usr/bin/time unavailable: peak RSS not recorded"
fi

say "source $(git rev-parse --short=10 HEAD 2>/dev/null || echo unknown) in $TREE"
pixi install > "$RES/pixi-install.log" 2>&1 || { say "pixi install failed: $RES/pixi-install.log"; tail -n 20 "$RES/pixi-install.log"; exit 2; }
pixi run mojo --version > "$RES/compiler-version.log" 2>&1

FILTER=${NC_FILTER:-*}
$PY tools/ops/native_checks_report.py list --column gpu --filter "$FILTER" > "$RES/planned.txt" \
    || { say "check listing failed"; exit 2; }
n=$(grep -c . "$RES/planned.txt")
[ "$n" -gt 0 ] || { say "no check matches $FILTER"; exit 2; }

# The one compile: every registered device-column check in one binary.
mkdir -p "$RES/bin"
say "building run_gpu ($n checks planned)"
bstart=$(date +%s)
if [ -x /usr/bin/time ]; then
    /usr/bin/time -v -o "$RES/time/build-gpu.txt" pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . \
        -o "$RES/bin/run_gpu" checks/native_tests/run_gpu.mojo > "$RES/logs/build-gpu.log" 2>&1
else
    pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . -o "$RES/bin/run_gpu" checks/native_tests/run_gpu.mojo \
        > "$RES/logs/build-gpu.log" 2>&1
fi
brc=$?
brss=$(awk -F': *' '/Maximum resident set size/ {print $2}' "$RES/time/build-gpu.txt" 2>/dev/null)
$PY tools/ops/native_checks_report.py result build-gpu --log "$RES/logs/build-gpu.log" --rc "$brc" \
    --seconds "$(( $(date +%s) - bstart ))" ${brss:+--maxrss-kb "$brss"} --out-dir "$RES/results"
say "build rc=$brc in $(( $(date +%s) - bstart ))s"

# Cores of the allocation (cgroup quota, affinity), not the host's visible CPUs.
cores=$(nproc)
if [ -r /sys/fs/cgroup/cpu.max ]; then
    read -r q p < /sys/fs/cgroup/cpu.max
    [ "$q" != max ] && cores=$(( (q + p - 1) / p < cores ? (q + p - 1) / p : cores ))
elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then
    q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us); p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
    [ "$q" -gt 0 ] && cores=$(( (q + p - 1) / p < cores ? (q + p - 1) / p : cores ))
fi
K=${NC_JOBS:-$(( cores / 4 > 0 ? cores / 4 : 1 ))}
mem_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
if [ -r /sys/fs/cgroup/memory.max ] && [ "$(cat /sys/fs/cgroup/memory.max)" != max ]; then
    cg_kb=$(( $(cat /sys/fs/cgroup/memory.max) / 1024 )); [ "$cg_kb" -lt "$mem_kb" ] && mem_kb=$cg_kb
fi
say "$n checks, filter $FILTER, cores $cores, K=$K, memory available $(( mem_kb / 1048576 )) GB, timeout ${NC_TIMEOUT:-900}s"

# Busy cores of total, sampled every 15 s while the tasks run (part of this job, not a watcher): the cgroup's CPU
# time (cpu.stat usage_usec) per wall second, else the whole host's /proc/stat busy share times its CPUs.
( if [ -r /sys/fs/cgroup/cpu.stat ]; then
      prev=$(awk '/^usage_usec/ {print $2}' /sys/fs/cgroup/cpu.stat)
      while sleep 15; do
          cur=$(awk '/^usage_usec/ {print $2}' /sys/fs/cgroup/cpu.stat)
          awk -v a="$prev" -v b="$cur" 'BEGIN {printf "%.1f\n", (b - a) / 15e6}'; prev=$cur
      done
  else
      host=$(grep -c '^cpu[0-9]' /proc/stat); prev=$(awk '/^cpu / {print $2+$3+$4+$7+$8, $5+$6}' /proc/stat)
      while sleep 15; do
          cur=$(awk '/^cpu / {print $2+$3+$4+$7+$8, $5+$6}' /proc/stat)
          echo "$prev $cur" | awk -v h="$host" '{b=$3-$1; i=$4-$2; if (b+i>0) printf "%.1f\n", b/(b+i)*h}'; prev=$cur
      done
  fi ) > "$RES/busy_cores.txt" 2>/dev/null &
sampler=$!

started=$(date +%s)
if [ "$brc" -eq 0 ]; then
    xargs -a "$RES/planned.txt" -P "$K" -I{} bash "$SELF" --one {} "$RES" > "$RES/progress.log" 2>&1
else
    say "build failed: no check runs (every planned check reads MISSING); first error in $RES/results/build-gpu.txt"
fi
kill "$sampler" 2>/dev/null
wall=$(( $(date +%s) - started ))

peak_kb=$(awk -F': *' '/^maxrss_kb/ {if ($2>m) m=$2} END {print m+0}' "$RES"/results/*.txt)
busy=$(awk '{s+=$1; if ($1>m) m=$1; n++} END {if (n) printf "mean %.1f, max %.1f", s/n, m; else print "not sampled"}' "$RES/busy_cores.txt")
{
    echo "host: $(hostname) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -n 1)"
    echo "source: $(git rev-parse HEAD 2>/dev/null) compiler: $(head -n 1 "$RES/compiler-version.log")"
    echo "filter: $FILTER; checks: $n; build rc $brc; wall: ${wall}s; timeout per check: ${NC_TIMEOUT:-900}s"
    echo "parallel: K=$K concurrent checks on $cores cores (K = cores/4 unless NC_JOBS); busy cores of $cores: $busy"
    if [ "$peak_kb" -gt 0 ]; then
        echo "peak RSS of the largest check or the build: $(( peak_kb / 1024 )) MB; memory available $(( mem_kb / 1024 )) MB;" \
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
