"""Full-allocation CPU resources for race workers (stdlib only).

Race workers must not inherit laptop/check-only thread caps. A real Linux cgroup
quota is the allocation, not a benchmark handicap. Libraries choose their normal
parallelism on unrestricted machines; this does not make serial algorithms parallel.
"""
import importlib.util
import os

THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
              "GOTO_NUM_THREADS", "NUMBA_NUM_THREADS")
INHERITED_CAPS = THREAD_ENV + ("OMP_THREAD_LIMIT", "NUMEXPR_MAX_THREADS",
                              "LOKY_MAX_CPU_COUNT", "MOJOLEARN_BENCH_THREADS",
                              "MOJOLEARN_CPU_THREADS")
#: the environment variable an A/B env arm sets to name the keys it sets on purpose (comma separated, set by
#: tools/ops/paper_compare_job.sh beside the arm's own assignments). Lane fx-harness (2026-11-18, fx-verdicts-RESULT:
#: the env arm cpu-threads-cgroup was INVALID because apply_cpu_quota popped MOJOLEARN_CPU_THREADS in every race
#: worker and set the quota, so both arms ran at 13): a key named here keeps the arm's explicit value. Without it the
#: behaviour is unchanged: every inherited cap is removed and the quota is applied.
ARM_ENV_KEYS = "MOJOLEARN_BENCH_ARM_ENV_KEYS"


def arm_env_keys(env):
    """The keys an A/B env arm declared it sets explicitly (ARM_ENV_KEYS), as a set; empty without one."""
    return {k.strip() for k in (env.get(ARM_ENV_KEYS) or "").split(",") if k.strip()}


_NUMA = []


def _numa():
    """tools/numa_place.py (same directory), loaded once by path: its cgroup quota read is the product's rule."""
    if not _NUMA:
        spec = importlib.util.spec_from_file_location(
            "cpu_quota_numa_place", os.path.join(os.path.dirname(os.path.abspath(__file__)), "numa_place.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _NUMA.append(mod)
    return _NUMA[0]


def _visible_cores():
    """The cores this process may run on: its affinity, else os.cpu_count()."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count()


def cpu_quota_threads():
    """Return the binding cgroup CPU allocation in whole cores; no user thread-count override.

    The former MOJOLEARN_BENCH_THREADS override could silently import a one-core
    diagnostic setting into races. Owner policy (2026-10-05): races always get
    the full allocation, including opponents and our own CPU workers.

    FAIRNESS FIX (2026-11-19, lane fx-numa-bind-smoke): this read cgroup v2's root `cpu.max` only. The RunPod L40S
    pod's quota is cgroup v1 (14 cores of 128 visible; tools/ops/host_probe.py cgroup_cores 14), so no thread count
    was set and every worker's OpenMP/BLAS pools and MOJOLEARN_CPU_THREADS followed the visible cores (128, or the
    64 of the GPU's node once the worker is bound) under a 14-core CFS quota. It now uses tools/numa_place.py
    cgroup_quota_cores (v2 and v1, along the process's own cgroup path, rounded up: the product's
    core/host_predict_threads.mojo rule). The bound node always holds the whole allocation (numa_place.plan binds
    only then), so the pools are every core of the allocation on that node: 14 on this pod.
    """
    try:
        n = _numa().cgroup_quota_cores()
    except Exception:  # noqa: BLE001 -- an unreadable cgroup is no quota
        return None
    visible = _visible_cores()
    return n if n and n < (visible or n + 1) else None


def apply_cpu_quota(env):
    """Remove diagnostic caps, then respect only the actual machine allocation.

    Apply after caller overrides too. Algorithm-specific nested-pool settings
    (implicit's single-thread BLAS inside its parallel solver) are applied by
    their own adapters; they are not a one-core allocation for the whole arm.
    An A/B env arm's explicit values (the keys it names in ARM_ENV_KEYS) are let through: neither removed nor
    replaced by the quota; that is the arm under test, recorded in MOJOLEARN_BENCH_ARM_ENV_KEPT.
    """
    keep = {k: env[k] for k in arm_env_keys(env) if k in INHERITED_CAPS and env.get(k)}
    for key in INHERITED_CAPS:
        env.pop(key, None)
    n = cpu_quota_threads()
    if n:
        for key in THREAD_ENV + ("MOJOLEARN_CPU_THREADS",):
            env[key] = str(n)
    if keep:
        env.update(keep)
        env["MOJOLEARN_BENCH_ARM_ENV_KEPT"] = ",".join("%s=%s" % kv for kv in sorted(keep.items()))
    env["MOJOLEARN_BENCH_CPU_POLICY"] = "full-allocation"
    return env
