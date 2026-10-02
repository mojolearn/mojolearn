"""Thread caps for board workers on a cgroup CPU quota. Stdlib only: the board controller imports it under a
system python that has no numpy."""
import os

THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def cpu_quota_threads():
    """The CPU count a cgroup quota grants when it is below the visible count (a RunPod pod shows 128 CPUs on a
    13.6-CPU quota): there every thread pool sized from os.cpu_count() is throttled. MOJOLEARN_BENCH_THREADS
    overrides. None when unset and no quota binds (the box's defaults stand)."""
    v = os.environ.get("MOJOLEARN_BENCH_THREADS", "").strip()
    if v:
        return int(v)
    try:
        q, p = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        n = int(int(q) / int(p)) if q != "max" else None
    except (OSError, ValueError):
        return None
    return n if n and n < (os.cpu_count() or n + 1) else None


def apply_cpu_quota(env):
    """After a worker env drops inherited thread caps: cap every pool at the cgroup quota (cpu_quota_threads)."""
    n = cpu_quota_threads()
    if n:
        for k in THREAD_ENV + ("MOJOLEARN_CPU_THREADS",):
            env[k] = str(n)
    return env
