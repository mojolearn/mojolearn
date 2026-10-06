"""Full-allocation CPU resources for race workers (stdlib only).

Race workers must not inherit laptop/check-only thread caps. A real Linux cgroup
quota is the allocation, not a benchmark handicap. Libraries choose their normal
parallelism on unrestricted machines; this does not make serial algorithms parallel.
"""
import os

THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
              "GOTO_NUM_THREADS", "NUMBA_NUM_THREADS")
INHERITED_CAPS = THREAD_ENV + ("OMP_THREAD_LIMIT", "NUMEXPR_MAX_THREADS",
                              "LOKY_MAX_CPU_COUNT", "MOJOLEARN_BENCH_THREADS",
                              "MOJOLEARN_CPU_THREADS")


def cpu_quota_threads():
    """Return a binding cgroup v2 CPU allocation; no user thread-count override.

    The former MOJOLEARN_BENCH_THREADS override could silently import a one-core
    diagnostic setting into races. Owner policy (2026-10-05): races always get
    the full allocation, including opponents and our own CPU workers.
    """
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            q, p = f.read().split()[:2]
        n = max(1, int(int(q) / int(p))) if q != "max" else None
    except (OSError, ValueError, ZeroDivisionError):
        return None
    return n if n and n < (os.cpu_count() or n + 1) else None


def apply_cpu_quota(env):
    """Remove diagnostic caps, then respect only the actual machine allocation.

    Apply after caller overrides too. Algorithm-specific nested-pool settings
    (implicit's single-thread BLAS inside its parallel solver) are applied by
    their own adapters; they are not a one-core allocation for the whole arm.
    """
    for key in INHERITED_CAPS:
        env.pop(key, None)
    n = cpu_quota_threads()
    if n:
        for key in THREAD_ENV + ("MOJOLEARN_CPU_THREADS",):
            env[key] = str(n)
    env["MOJOLEARN_BENCH_CPU_POLICY"] = "full-allocation"
    return env
