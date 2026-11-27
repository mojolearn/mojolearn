"""The CPU, memory, GPU-memory and task allocation of this process, and the worker count it allows.

ONE ALLOCATION FUNCTION (rel-fix-3, 2026-11-26). The identity record job of 3ce09e341f on the L40S
(nvc-ab-20261126T235556Z-c90f67) printed "IDREC allocation: 14 core(s) of 128 visible" and then ran `verify --all`
with 128 cell workers: `_verify_all.verify_cell_workers` read the affinity (128) and only the cgroup v2 `cpu.max`,
which a RunPod pod does not have (its quota is cgroup v1 `cpu.cfs_quota_us` / `cpu.cfs_period_us`), and the
1313 GiB available at 4 GiB a worker allowed all 128. 128 interpreters on 14 cores ran 9 times oversubscribed, hit
the pids limit ("fork: retry: Resource temporarily unavailable") and cut 30 cells of 3 to 11 s at their 120 s limit.
Five copies of the core count (verify, the recorder, the GPU smoke, the multi-GPU checks, the proving workers) each
read a different subset of affinity, v2 and v1 and rounded differently. They all read this file now, and the job
scripts' `alloc_cores` (tools/ops/box_cores_lib.sh) runs it, so every stage of a job sees one number.

The core count, in order:
  1. MOJOLEARN_ALLOC_CORES, when a positive integer: the job measured the allocation once at its start and passes
     it down (tools/ops/box_cores_lib.sh `export_alloc`), so no stage guesses again;
  2. the scheduler's affinity set (`os.sched_getaffinity`; it follows a cpuset), else `os.cpu_count()`;
  3. capped by the cgroup CPU quota: v2 `cpu.max`, else v1 `cpu.cfs_quota_us` / `cpu.cfs_period_us` (under `cpu/`
     or `cpu,cpuacct/`), rounded UP (a 13.6-core quota is 14 workers: the whole allocation is used, as the shell
     helpers have always counted it).

Stdlib only and no package imports: the wheel's verifier imports it as `mojolearn._allocation`, and tools in the tree
load it by path (`load()` below) without importing the package. Python 3.8 or newer (the boxes' system python3).

    python3 -I python/mojolearn/_allocation.py cores       the core count
    python3 -I python/mojolearn/_allocation.py describe    the core count, a tab, and where it came from
    python3 -I python/mojolearn/_allocation.py usage       "<wall us> <cpu us> <source>": the allocation's CPU time
"""
import os
import shutil
import subprocess
import sys

#: the job's measured core count, passed to every stage it starts
ENV_CORES = "MOJOLEARN_ALLOC_CORES"
CGROUP = "/sys/fs/cgroup"
#: the cgroup v1 cpu controller's mount names
V1_CPU_DIRS = ("cpu", "cpu,cpuacct", "cpuacct,cpu")
#: a cgroup v1 memory limit at or above this is "no limit" (the kernel writes 2**63 rounded down to a page)
V1_NO_LIMIT = 1 << 60
#: share of the GPU's free memory workers may plan on (the rest: the driver and fragmentation); the GPU smoke's value
GPU_MEM_SHARE = 0.9
#: share of the cgroup's remaining tasks workers may plan on (the rest of the box keeps a quarter)
PIDS_SHARE = 0.75


def _words(path):
    try:
        with open(path, encoding="ascii") as stream:
            return stream.read().split()
    except (OSError, UnicodeDecodeError):
        return []


def _lines(path):
    try:
        with open(path, encoding="ascii") as stream:
            return stream.read().splitlines()
    except (OSError, UnicodeDecodeError):
        return []


def _ceil_div(a, b):
    return -(-a // b)


def affinity_cores():
    """(cores, source): the scheduler's affinity set, else os.cpu_count()."""
    try:
        return max(1, len(os.sched_getaffinity(0))), "affinity"
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1), "os.cpu_count"


def quota_cores(root=CGROUP):
    """(cores or None, source): the cgroup CPU quota in whole cores (rounded up), None when there is none."""
    w = _words(os.path.join(root, "cpu.max"))
    if len(w) >= 2:
        if w[0] == "max":
            return None, "cgroup v2 cpu.max: no quota"
        try:
            q, p = int(w[0]), int(w[1])
            if q > 0 and p > 0:
                return max(1, _ceil_div(q, p)), "cgroup v2 cpu.max %d/%d" % (q, p)
        except ValueError:
            pass
        return None, "cgroup v2 cpu.max unreadable (%s)" % " ".join(w[:2])
    for d in V1_CPU_DIRS:
        q, p = _words(os.path.join(root, d, "cpu.cfs_quota_us")), _words(os.path.join(root, d, "cpu.cfs_period_us"))
        if q and p:
            try:
                qi, pi = int(q[0]), int(p[0])
            except ValueError:
                return None, "cgroup v1 %s quota unreadable" % d
            if qi > 0 and pi > 0:
                return max(1, _ceil_div(qi, pi)), "cgroup v1 %s/cpu.cfs_quota_us %d/%d" % (d, qi, pi)
            return None, "cgroup v1 %s: no quota" % d
    return None, "no cgroup CPU quota found"


def cores(root=CGROUP, environ=None):
    """(cores, reason): the cores this process may use (module docstring for the order)."""
    env = os.environ if environ is None else environ
    pinned = (env.get(ENV_CORES) or "").strip()
    if pinned:
        try:
            n = int(pinned)
        except ValueError:
            n = 0
        if n >= 1:
            return n, "%s=%d (measured by the job)" % (ENV_CORES, n)
    n, src = affinity_cores()
    q, qsrc = quota_cores(root)
    if q is not None and q < n:
        return q, "%s (%s %d)" % (qsrc, src, n)
    return n, "%s %d (%s)" % (src, n, qsrc)


def mem_available_bytes(root=CGROUP):
    """MemAvailable, capped by the cgroup's remaining memory (v2 memory.max - memory.current, v1
    memory/memory.limit_in_bytes - memory.usage_in_bytes), or None where unreadable."""
    avail = None
    try:
        with open("/proc/meminfo", encoding="ascii") as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) * 1024
                    break
    except (OSError, ValueError, IndexError):
        pass
    if avail is None:
        try:
            avail = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
        except (AttributeError, OSError, ValueError):
            return None
    for limit_path, used_path, unlimited in (("memory.max", "memory.current", None),
                                             ("memory/memory.limit_in_bytes", "memory/memory.usage_in_bytes",
                                              V1_NO_LIMIT)):
        limit, used = _words(os.path.join(root, limit_path)), _words(os.path.join(root, used_path))
        if not limit or not used or limit[0] == "max":
            continue
        try:
            li, ui = int(limit[0]), int(used[0])
        except ValueError:
            continue
        if unlimited is None or li < unlimited:
            avail = min(avail, max(0, li - ui))
        break
    return avail


def pids_room(root=CGROUP):
    """Tasks (threads) this cgroup may still start, or None without a pids limit (v2 pids.max, v1 pids/pids.max)."""
    for limit_path, current_path in (("pids.max", "pids.current"), ("pids/pids.max", "pids/pids.current")):
        limit, current = _words(os.path.join(root, limit_path)), _words(os.path.join(root, current_path))
        if limit and current and limit[0] != "max":
            try:
                return int(limit[0]) - int(current[0])
            except ValueError:
                return None
    return None


def gpu_free_mb(timeout=30):
    """The smallest free memory of the box's GPUs in MB (nvidia-smi, else rocm-smi), or None without a GPU tool."""
    try:
        if shutil.which("nvidia-smi"):
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=timeout)
            vals = [float(x) for x in out.stdout.split()] if out.returncode == 0 else []
            return int(min(vals)) if vals else None
        if shutil.which("rocm-smi"):
            import json
            out = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--json"], capture_output=True, text=True,
                                 timeout=timeout)
            doc = json.loads(out.stdout or "{}")
            vals = [(float(v.get("VRAM Total Memory (B)", 0)) - float(v.get("VRAM Total Used Memory (B)", 0)))
                    / 2 ** 20 for v in doc.values() if isinstance(v, dict) and "VRAM Total Memory (B)" in v]
            return int(min(vals)) if vals else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return None


#: the cgroup v1 cpuacct controller's mount names
V1_CPUACCT_DIRS = ("cpuacct", "cpu,cpuacct", "cpuacct,cpu")


def cpu_usage_seconds(root=CGROUP):
    """(CPU seconds, source): the CPU time this allocation has used, the counter every busy-core sampler differences
    over wall time (rel-fix-3). The pod's own cgroup counter: v2 cpu.stat usage_usec, else v1 cpuacct.usage (ns).
    Only without a cgroup counter (a bare host, whose allocation is the host) the host's /proc/stat busy jiffies
    (user, nice, system, irq, softirq, steal). Before, the samplers read /proc/stat on every box, so a RunPod pod
    (cgroup v1) reported its host's busy CPUs: "busy cores mean 40.7, max 65.0 of 14" for a record whose own CPU
    seconds were 9.8 cores. (None, "none") where nothing is readable (macOS)."""
    for line in _lines(os.path.join(root, "cpu.stat")):
        parts = line.split()
        if len(parts) == 2 and parts[0] == "usage_usec":
            try:
                return int(parts[1]) / 1e6, "cgroup v2 cpu.stat"
            except ValueError:
                break
    for d in V1_CPUACCT_DIRS:
        w = _words(os.path.join(root, d, "cpuacct.usage"))
        if w:
            try:
                return int(w[0]) / 1e9, "cgroup v1 %s/cpuacct.usage" % d
            except ValueError:
                break
    first = (_lines("/proc/stat") or [""])[0].split()
    if first[:1] == ["cpu"] and len(first) >= 9:
        try:
            f = [int(x) for x in first[1:9]]
            tick = os.sysconf("SC_CLK_TCK")
            return (f[0] + f[1] + f[2] + f[5] + f[6] + f[7]) / float(tick), "host /proc/stat"
        except (ValueError, OSError, AttributeError):
            pass
    return None, "none"


def busy_cores(before, after):
    """Busy cores between two (wall seconds, CPU seconds) samples: CPU seconds over wall seconds, or None."""
    if not before or not after or before[1] is None or after[1] is None:
        return None
    wall, cpu = after[0] - before[0], after[1] - before[1]
    if wall <= 0 or cpu < 0:
        return None
    return round(cpu / wall, 1)


def usage_sample(root=CGROUP):
    """(wall seconds, CPU seconds or None, source) now, for `busy_cores`."""
    import time
    cpu, source = cpu_usage_seconds(root)
    return time.time(), cpu, source


def interpreter_threads():
    """Threads one worker interpreter with Mojolearn imported starts: its runtime starts one per VISIBLE CPU, 193
    on the 128-CPU L40S pod (tests/proving/_proving.py, 2026-11-22), so the visible count plus 72."""
    return (os.cpu_count() or 1) + 72


def workers(requested=None, host_gib_per_worker=None, gpu_mb_per_worker=None, threads_per_worker=None,
            cores_per_worker=1, units=None, gpu_free=None, root=CGROUP, environ=None):
    """(workers, reason): concurrent worker processes. `requested` pins the count (the reason says so). Otherwise
    the allocation's cores (`cores`) over `cores_per_worker`, bounded by host memory at `host_gib_per_worker`, by
    the GPU's free memory (`gpu_free` MB, else `gpu_free_mb()` when `gpu_mb_per_worker` is given) at
    GPU_MEM_SHARE, by the cgroup's remaining tasks at `threads_per_worker` each, and by `units`."""
    if requested:
        n = max(1, int(requested))
        return n, "%d requested" % n
    n, why = cores(root, environ)
    parts = ["cores %d (%s)" % (n, why)]
    n = max(1, n // max(1, cores_per_worker))
    if cores_per_worker > 1:
        parts.append("%d core(s) a worker" % cores_per_worker)
    if host_gib_per_worker:
        avail = mem_available_bytes(root)
        if avail is not None:
            cap = max(1, int(avail // int(host_gib_per_worker * (1 << 30))))
            parts.append("host memory %d at %g GiB of %.0f GiB" % (cap, host_gib_per_worker, avail / 2 ** 30))
            n = min(n, cap)
    if gpu_mb_per_worker:
        free = gpu_free if gpu_free is not None else gpu_free_mb()
        if free is not None:
            cap = max(1, int(free * GPU_MEM_SHARE // gpu_mb_per_worker))
            parts.append("gpu memory %d at %g MB of %d MB free" % (cap, gpu_mb_per_worker, free))
            n = min(n, cap)
    if threads_per_worker:
        room = pids_room(root)
        if room is not None:
            cap = max(1, int(room * PIDS_SHARE) // threads_per_worker)
            parts.append("tasks %d at %d a worker of %d left" % (cap, threads_per_worker, room))
            n = min(n, cap)
    if units is not None:
        n = min(n, max(1, units))
        parts.append("units %d" % units)
    return max(1, n), "; ".join(parts)


# THE COST OF A LANE WITH NO MEASURED COST (rel-fix-3). The shipped verify_reference/cell_seconds.json comes from one
# L40S run and holds 470 of the 512 lanes of the reference table; the other 42 (gbdt-multiclass, gbdt-symmetric,
# trees-stacking-clf, x-decomp-umap-options, ...) planned at the median of every lane (0.41 s) and limited at the
# blanket floor. A lane without its own figure now takes its family's: the slowest measured lane sharing its
# longest name prefix (gbdt-multiclass -> gbdt-*, x-decomp-eigh-wide-fastica -> x-decomp-eigh-wide-*), else the
# 90th percentile of every measured lane. High on purpose: it orders the lane early and never cuts it short.
def lane_cost(hints, lane):
    """(seconds, source) of one cell of `lane` from {lane: mean cell seconds} (keys starting '@' are not lanes)."""
    value = hints.get(lane)
    if isinstance(value, (int, float)) and value > 0:
        return float(value), "measured"
    known = {k: float(v) for k, v in hints.items()
             if not k.startswith("@") and isinstance(v, (int, float)) and v > 0}
    if not known:
        return 1.0, "no hints"
    segments = lane.split("-")
    for i in range(len(segments) - 1, 0, -1):
        prefix = "-".join(segments[:i]) + "-"
        family = [v for k, v in known.items() if k.startswith(prefix)]
        if family:
            return max(family), "family %s*" % prefix
    ranked = sorted(known.values())
    return ranked[min(len(ranked) - 1, int(0.9 * len(ranked)))], "p90 of all lanes"


def load(root=None):
    """This file as a module, for tools in the tree that must not import the package (path-loaded once)."""
    import importlib.util
    path = os.path.abspath(__file__) if root is None else os.path.join(root, "python", "mojolearn", "_allocation.py")
    name = "mojolearn_allocation_by_path"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main(argv):
    what = argv[0] if argv else "cores"
    n, why = cores()
    if what == "cores":
        print(n)
    elif what == "describe":
        print("%d\t%s" % (n, why))
    elif what == "usage":   # "<wall us> <cpu us> <source>" for the shell samplers, or "none"
        wall, cpu, source = usage_sample()
        print("none" if cpu is None else "%d %d %s" % (wall * 1e6, cpu * 1e6, source.replace(" ", "_")))
    else:
        print("usage: _allocation.py cores|describe|usage", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
