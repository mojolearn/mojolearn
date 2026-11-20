# shellcheck shell=bash
# tools/ops/box_cores_lib.sh -- sourced (bash), never run. The allocation and busy-core helpers every rented-box job
# shares (AGENTS.md hard rules: use 100 percent of rented CPU; every report states busy cores of total):
#
#   alloc_cores            the cores of this allocation (cgroup quota v2/v1, affinity via nproc), not the host's CPUs
#   alloc_mem_kb           memory available to this allocation in kB (MemAvailable, capped by the cgroup's memory.max)
#   alloc_mem_total_kb     memory of this allocation in kB (MemTotal, capped by the cgroup's memory.max)
#   busy_sampler FILE [S]  starts a background sampler (prints its pid): one line every S seconds (default 15) with the
#                          busy cores of that interval (the cgroup's CPU time per wall second, else the host's /proc/stat
#                          busy share times its CPUs). Part of the job that starts it, not a watcher.
#   busy_of FILE           "mean X, max Y" of a sampler's file
#
# Lifted from tools/ops/native_checks_gpu_job.sh (lane fx-pod-job, 2026-11-18), which now sources this file, so
# tools/ops/cpu_box_checks_job.sh (lane fx-cpu-box-checks, 2026-11-19) reads cores, memory and busy cores the same way.
alloc_cores() {
    local cores q p
    cores=$(nproc)
    if [ -r /sys/fs/cgroup/cpu.max ]; then
        read -r q p < /sys/fs/cgroup/cpu.max
        [ "$q" != max ] && cores=$(( (q + p - 1) / p < cores ? (q + p - 1) / p : cores ))
    elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then
        q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us); p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
        [ "$q" -gt 0 ] && cores=$(( (q + p - 1) / p < cores ? (q + p - 1) / p : cores ))
    fi
    echo "$cores"
}
_alloc_cap_kb() {  # <kB>: capped by the cgroup's memory.max
    local kb=$1 cg_kb
    if [ -r /sys/fs/cgroup/memory.max ] && [ "$(cat /sys/fs/cgroup/memory.max)" != max ]; then
        cg_kb=$(( $(cat /sys/fs/cgroup/memory.max) / 1024 )); [ "$cg_kb" -lt "$kb" ] && kb=$cg_kb
    fi
    echo "$kb"
}
alloc_mem_kb() { _alloc_cap_kb "$(awk '/MemAvailable/ {print $2}' /proc/meminfo)"; }
alloc_mem_total_kb() { _alloc_cap_kb "$(awk '/MemTotal/ {print $2}' /proc/meminfo)"; }
busy_sampler() {   # file [seconds]
    local every=${2:-15}
    ( if [ -r /sys/fs/cgroup/cpu.stat ]; then
          prev=$(awk '/^usage_usec/ {print $2}' /sys/fs/cgroup/cpu.stat)
          while sleep "$every"; do
              cur=$(awk '/^usage_usec/ {print $2}' /sys/fs/cgroup/cpu.stat)
              awk -v a="$prev" -v b="$cur" -v s="$every" 'BEGIN {printf "%.1f\n", (b - a) / (s * 1e6)}'; prev=$cur
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
