#!/bin/bash
# GPU native checks on the rented L40S pod: every pixi task matching NC_FILTER (default check-native-*-gpu), each run
# exactly as pixi.toml writes it (`pixi run <task>`), with a per-task timeout. The CPU checks run on the free GitHub
# runners (.github/workflows/native-checks.yml); both write the same results/<task>.txt, read by
# tools/ops/native_checks_report.py (summary, report).
#
# Launch (results persist to R2 from the pod through tools/box_persist_run.sh):
#   tools/nvidia_central.sh submit measure --note native-checks-gpu tools/ops/native_checks_gpu_job.sh
#   [--env NC_FILTER=check-native-%-gpu]   (`%` reads as `*`: submit --env refuses `*`; comma list of globs)
#   [--env NC_TIMEOUT=900] [--env NC_JOBS=K]
# Read:  tools/box_persist.py fetch --tag <tag>; python3 tools/ops/native_checks_report.py report <dir>
#
# Parallel plan (AGENTS.md hard rules on parallelism): every task is independent, so the job runs K at once with
# xargs -P. `mojo run` compiles then runs, and the compile is the whole cost (the GPU parts are tiny and may overlap
# on the one GPU). K = cgroup cores / 4 because one Mojo compile of a check uses several threads and a few GB of
# memory; four cores per compile keeps every core busy without oversubscribing memory. The job measures it: each task
# runs under /usr/bin/time -v, and SUMMARY.md states the peak RSS of the largest task, the memory available, the K
# that memory alone would allow, and the busy cores of total sampled from /proc/stat. If memory allows a larger K,
# the next run passes NC_JOBS=<that K>; a smaller K is used only when the measured peak RSS forces it.
#
# Exit status: 0 when every task passed, 1 when any failed or timed out (rc 124), 2 on a setup failure.
set -u

if [ "${1:-}" = "--one" ]; then
    # One task (run by xargs): rc, seconds, peak RSS and the first error lines into results/<task>.txt.
    task=$2; RES=$3
    log="$RES/logs/$task.log"; tm="$RES/time/$task.txt"; start=$(date +%s)
    if [ -x /usr/bin/time ]; then
        /usr/bin/time -v -o "$tm" timeout -k 30 "${NC_TIMEOUT:-900}" pixi run "$task" > "$log" 2>&1
    else
        timeout -k 30 "${NC_TIMEOUT:-900}" pixi run "$task" > "$log" 2>&1
    fi
    rc=$?
    rss=$(awk -F': *' '/Maximum resident set size/ {print $2}' "$tm" 2>/dev/null)
    {
        echo "task: $task"
        echo "rc: $rc"
        echo "seconds: $(( $(date +%s) - start ))"
        [ -n "$rss" ] && echo "maxrss_kb: $rss"
        echo "--- first error ---"
        grep -m5 -B2 -A8 'error:' "$log" || tail -n 15 "$log"
    } > "$RES/results/$task.txt"
    echo "[$(date +%H:%M:%S)] $task rc=$rc"
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

PY=python3
python3 -c 'import tomllib' 2>/dev/null || PY="pixi run python3"
FILTER=${NC_FILTER:-check-native-*-gpu}
$PY tools/ops/native_checks_report.py list --filter "$FILTER" > "$RES/planned.txt" || { say "task listing failed"; exit 2; }
n=$(grep -c . "$RES/planned.txt")
[ "$n" -gt 0 ] || { say "no task matches $FILTER"; exit 2; }

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
say "$n tasks, filter $FILTER, cores $cores, K=$K, memory available $(( mem_kb / 1048576 )) GB, timeout ${NC_TIMEOUT:-900}s"

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
xargs -a "$RES/planned.txt" -P "$K" -I{} bash "$SELF" --one {} "$RES" > "$RES/progress.log" 2>&1
kill "$sampler" 2>/dev/null
wall=$(( $(date +%s) - started ))

peak_kb=$(awk -F': *' '/^maxrss_kb/ {if ($2>m) m=$2} END {print m+0}' "$RES"/results/*.txt)
busy=$(awk '{s+=$1; if ($1>m) m=$1; n++} END {if (n) printf "mean %.1f, max %.1f", s/n, m; else print "not sampled"}' "$RES/busy_cores.txt")
{
    echo "host: $(hostname) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -n 1)"
    echo "source: $(git rev-parse HEAD 2>/dev/null) compiler: $(head -n 1 "$RES/compiler-version.log")"
    echo "filter: $FILTER; tasks: $n; wall: ${wall}s; timeout per task: ${NC_TIMEOUT:-900}s"
    echo "parallel: K=$K concurrent tasks on $cores cores (K = cores/4 unless NC_JOBS); busy cores of $cores: $busy"
    if [ "$peak_kb" -gt 0 ]; then
        echo "peak RSS of the largest task: $(( peak_kb / 1024 )) MB; memory available $(( mem_kb / 1024 )) MB;" \
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
