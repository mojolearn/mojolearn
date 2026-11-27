# shellcheck shell=bash
# tools/ops/box_cores_lib.sh -- sourced (bash), never run. The allocation and busy-core helpers every rented-box job
# shares (AGENTS.md hard rules: use 100 percent of rented CPU; every report states busy cores of total):
#
#   alloc_cores            the cores of this allocation (python/mojolearn/_allocation.py: MOJOLEARN_ALLOC_CORES, else
#                          affinity capped by the cgroup v2/v1 quota), not the host's CPUs; alloc_reason says why
#   export_alloc           measures it once and exports MOJOLEARN_ALLOC_CORES for every later stage of the job
#   alloc_mem_kb           memory available to this allocation in kB (MemAvailable, capped by the cgroup's memory.max)
#   alloc_mem_total_kb     memory of this allocation in kB (MemTotal, capped by the cgroup's memory.max)
#   busy_sampler FILE [S]  starts a background sampler (prints its pid): one line every S seconds (default 15) with the
#                          busy cores of that interval (the cgroup's CPU time per wall second, else the host's /proc/stat
#                          busy share times its CPUs). Part of the job that starts it, not a watcher.
#   busy_of FILE           "mean X, max Y" of a sampler's file
#
# Lifted from tools/ops/native_checks_gpu_job.sh (lane fx-pod-job, 2026-11-18), which now sources this file, so
# tools/ops/cpu_box_checks_job.sh (lane fx-cpu-box-checks, 2026-11-19) reads cores, memory and busy cores the same way.
# ONE ALLOCATION FUNCTION (rel-fix-3, 2026-11-26): alloc_cores runs python/mojolearn/_allocation.py, the function
# verify, the recorder, the GPU smoke and the proving workers read, so every stage of a job sees one number (the
# copies differed: verify read cgroup v2 only and ran 128 workers on the L40S pod's 14-core v1 quota).
# export_alloc measures it once at the job's start and exports MOJOLEARN_ALLOC_CORES, which every later call and
# every stage the job starts takes as given.
_ALLOC_PY="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/python/mojolearn/_allocation.py"
alloc_cores() {
    local n
    n=$(python3 -I "$_ALLOC_PY" cores 2> /dev/null)
    if [ -n "$n" ] && [ "$n" -ge 1 ] 2> /dev/null; then echo "$n"; return; fi
    echo "alloc_cores: $_ALLOC_PY did not answer; the allocation is unknown, using nproc" >&2
    nproc
}
alloc_reason() { python3 -I "$_ALLOC_PY" describe 2> /dev/null | cut -f 2-; }
export_alloc() {   # measure once, pass down: MOJOLEARN_ALLOC_CORES for every later stage of this job
    [ -n "${MOJOLEARN_ALLOC_CORES:-}" ] || { MOJOLEARN_ALLOC_CORES=$(alloc_cores); export MOJOLEARN_ALLOC_CORES; }
}
_alloc_cap_kb() {  # <kB>: capped by the cgroup's memory.max
    local kb=$1 cg_kb
    if [ -r /sys/fs/cgroup/memory.max ] && [ "$(cat /sys/fs/cgroup/memory.max)" != max ]; then
        cg_kb=$(( $(cat /sys/fs/cgroup/memory.max) / 1024 )); [ "$cg_kb" -lt "$kb" ] && kb=$cg_kb
    elif [ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then   # cgroup v1; 2**63 rounded down is "no limit"
        cg_kb=$(awk '{printf "%.0f", ($1 >= 2^60) ? -1 : $1 / 1024}' /sys/fs/cgroup/memory/memory.limit_in_bytes)
        [ "$cg_kb" -ge 0 ] 2> /dev/null && [ "$cg_kb" -lt "$kb" ] && kb=$cg_kb
    fi
    echo "$kb"
}
alloc_mem_kb() { _alloc_cap_kb "$(awk '/MemAvailable/ {print $2}' /proc/meminfo)"; }
alloc_mem_total_kb() { _alloc_cap_kb "$(awk '/MemTotal/ {print $2}' /proc/meminfo)"; }
busy_sampler() {   # file [seconds]
    local every=${2:-15}
    # cgroup v1 (the RunPod L40S pods) has no cpu.stat: its own counter is cpuacct.usage (ns). Before rel-fix-3 the
    # sampler fell to the HOST's /proc/stat there and wrote the host's busy CPUs ("busy cores mean 40.7, max 65.0
    # of 14" for a record whose own CPU seconds were 9.8 cores).
    ( v1=; for d in cpuacct cpu,cpuacct cpuacct,cpu; do [ -r "/sys/fs/cgroup/$d/cpuacct.usage" ] && { v1=/sys/fs/cgroup/$d/cpuacct.usage; break; }; done
      if [ -r /sys/fs/cgroup/cpu.stat ]; then
          prev=$(awk '/^usage_usec/ {print $2}' /sys/fs/cgroup/cpu.stat)
          while sleep "$every"; do
              cur=$(awk '/^usage_usec/ {print $2}' /sys/fs/cgroup/cpu.stat)
              awk -v a="$prev" -v b="$cur" -v s="$every" 'BEGIN {printf "%.1f\n", (b - a) / (s * 1e6)}'; prev=$cur
          done
      elif [ -n "$v1" ]; then
          prev=$(cat "$v1")
          while sleep "$every"; do
              cur=$(cat "$v1")
              awk -v a="$prev" -v b="$cur" -v s="$every" 'BEGIN {printf "%.1f\n", (b - a) / (s * 1e9)}'; prev=$cur
          done
      else
          host=$(grep -c '^cpu[0-9]' /proc/stat); prev=$(awk '/^cpu / {print $2+$3+$4+$7+$8, $5+$6}' /proc/stat)
          while sleep "$every"; do
              cur=$(awk '/^cpu / {print $2+$3+$4+$7+$8, $5+$6}' /proc/stat)
              echo "$prev $cur" | awk -v h="$host" '{b=$3-$1; i=$4-$2; if (b+i>0) printf "%.1f\n", b/(b+i)*h}'; prev=$cur
          done
      fi ) > "$1" 2>/dev/null &
    echo $!
}
busy_of() { awk '{s+=$1; if ($1>m) m=$1; n++} END {if (n) printf "mean %.1f, max %.1f", s/n, m; else print "not sampled"}' "$1" 2>/dev/null; }
# Sourcing measures the allocation once for the job that sources it (at its start) and passes it to every stage.
export_alloc
