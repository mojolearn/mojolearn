# Sourced (POSIX sh) by every binding build script, right after build_defines.sh, and by bindings/merged_build.sh
# and tools/build_native_adapters.sh. Sets MOJOLEARN_COMPILE_JOBS_RESOLVED, the `-j` of the one compile line.
#
#   MOJOLEARN_COMPILE_JOBS set:   used as is (a positive integer, else the build stops).
#   unset or empty, Linux:        every core this process may use: `nproc` (CPU affinity / cpuset), capped by the
#                                 cgroup CPU quota when one is set (a RunPod or Docker container sees the whole
#                                 machine's cores in nproc but is billed and throttled at its quota).
#   unset or empty, elsewhere:    2 (macOS: the shared laptop and the Apple build Macs, as bindings/merged_build.sh
#                                 has done since its default moved to nproc on Linux).
#
# Why (Andrew 2026-11-05, HARD RULE "use 100 percent of rented CPU"): a rented build or GPU box is paid for whole,
# so a caller that forgets MOJOLEARN_COMPILE_JOBS must get the host's full core count, never a fixed 1 or 2. A
# measured memory limit is passed explicitly through MOJOLEARN_COMPILE_JOBS (and said in the report). The laptop's
# -j 1 comes from tools/compile_slot.sh, which exports MOJOLEARN_COMPILE_JOBS=1 on macOS when it is unset.
_mlcj_quota_cores() {
    # cgroup v2: "<quota> <period>" or "max <period>"; cgroup v1: cfs_quota_us (-1 = none) and cfs_period_us.
    # MLCJ_CGROUP_ROOT exists for tools/test_compile_jobs.py only (a fake cgroup tree).
    _mlcj_g=${MLCJ_CGROUP_ROOT:-/sys/fs/cgroup}
    _mlcj_q= _mlcj_p=
    if [ -r "$_mlcj_g/cpu.max" ]; then
        read -r _mlcj_q _mlcj_p < "$_mlcj_g/cpu.max" || true
    elif [ -r "$_mlcj_g/cpu/cpu.cfs_quota_us" ] && [ -r "$_mlcj_g/cpu/cpu.cfs_period_us" ]; then
        read -r _mlcj_q < "$_mlcj_g/cpu/cpu.cfs_quota_us" || true
        read -r _mlcj_p < "$_mlcj_g/cpu/cpu.cfs_period_us" || true
    fi
    case "$_mlcj_q:$_mlcj_p" in
        *[!0-9:]* | :* | *: ) return 0 ;;   # max, -1, empty or malformed: no quota
    esac
    [ "$_mlcj_p" -gt 0 ] || return 0
    echo $(( (_mlcj_q + _mlcj_p - 1) / _mlcj_p ))
}
mojolearn_compile_jobs() {
    if [ -n "${MOJOLEARN_COMPILE_JOBS:-}" ]; then
        case "$MOJOLEARN_COMPILE_JOBS" in
            *[!0-9]* | 0*) echo "error: MOJOLEARN_COMPILE_JOBS='$MOJOLEARN_COMPILE_JOBS' is not a positive integer" >&2
                           return 2 ;;
        esac
        echo "$MOJOLEARN_COMPILE_JOBS"
        return 0
    fi
    if [ "$(uname -s)" != Linux ]; then
        echo 2
        return 0
    fi
    _mlcj_n=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)
    case "$_mlcj_n" in '' | *[!0-9]* | 0) _mlcj_n=2 ;; esac
    _mlcj_c=$(_mlcj_quota_cores)
    if [ -n "$_mlcj_c" ] && [ "$_mlcj_c" -ge 1 ] && [ "$_mlcj_c" -lt "$_mlcj_n" ]; then
        _mlcj_n=$_mlcj_c
    fi
    echo "$_mlcj_n"
}
MOJOLEARN_COMPILE_JOBS_RESOLVED=$(mojolearn_compile_jobs) || exit 2
unset _mlcj_g _mlcj_q _mlcj_p _mlcj_n _mlcj_c
