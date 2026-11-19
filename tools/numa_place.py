"""NUMA placement of race workers, the facts behind it, and per-call host usage (stdlib only).

Lane fx-harness (2026-11-18; ~/mojolearn-evidence/briefs-20261118/ideas-unexplained-rows-RESULT.md section 1): on
L40S host c6dd9ebea3e3 every worker that started on NUMA node 1 and first-touched 2.5 GB or more lost 13-27 s, in the
clock (train-test-split/istella 13021 ms against 752-791 ms on the same wheel and settings) or outside it, while node-0
workers did not. The node a worker ran on, the GPU's node, the per-node free memory and the THP mode were not recorded,
so the cell could not say which it was. This module:

  * box_facts()      the GPU's NUMA node, each node's CPUs, MemFree / FilePages, and the THP enabled/defrag modes
                     (tools/ops/host_probe.py stores them in job/host-probe.json and job/box.txt);
  * wrap(cmd, env)   starts a race worker bound to the GPU's node, cores and memory, when the allocation allows it:
                     the node's CPUs inside this process's affinity must hold every core of the allocation (the cgroup
                     quota, else the affinity), so binding never takes a core away from the race (AGENTS.md part 1:
                     use 100 percent of rented CPU). Memory is bound (numactl --membind) when the node has
                     MOJOLEARN_BENCH_NUMA_MIN_FREE_GB free (default 32), else preferred (--preferred: falls back to
                     the other node instead of reclaiming on a full one). Without numactl the cores are bound with
                     taskset and memory is left to the kernel, and the placement says so. Each form is tried once on
                     `true` before use (a container may refuse set_mempolicy): the first that starts is used.
                     MOJOLEARN_BENCH_NUMA_BIND=0 turns it off (the placement records "off").
  * worker_node()    inside the worker: the node(s) it may run on and allocate from, and the node of the CPU it is on;
                     the cell records it (worker_numa).
  * usage() / delta  getrusage around one scored call: ru_stime, ru_utime, minor and major faults. A call whose system
                     time is more than half its wall is a host stall (tools/paper_compare.py HOST-STALL: shown, never a
                     REGRESSION).
No box is touched by importing this module; every read is a sysfs or procfs file and fails soft to None.
"""
import json
import os
import re
import shutil
import subprocess

PLACEMENT_ENV = "MOJOLEARN_BENCH_NUMA_PLACEMENT"
#: the job's plan (with the command prefix), made once by the first process that asks and inherited by every child
PLAN_ENV = "MOJOLEARN_BENCH_NUMA_PLAN"
GPU_NODE_ENV = "MOJOLEARN_BENCH_GPU_NUMA_NODE"
BIND_ENV = "MOJOLEARN_BENCH_NUMA_BIND"
MIN_FREE_ENV = "MOJOLEARN_BENCH_NUMA_MIN_FREE_GB"
MIN_FREE_GB = 32.0
#: a scored call whose system time is above this fraction of its wall is a host stall (page faults through direct
#: reclaim or THP compaction run in the faulting thread's system time; our own work is user time or GPU wait)
STALL_FRACTION = 0.5
NODE_ROOT = "/sys/devices/system/node"


def _read(path):
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return None


def parse_cpulist(text):
    """'0-3,8,10-11' -> {0, 1, 2, 3, 8, 10, 11}; '' or None -> empty set."""
    out = set()
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        a, _, b = part.partition("-")
        try:
            out.update(range(int(a), int(b or a) + 1))
        except ValueError:
            continue
    return out


def cpulist(cpus):
    """{0, 1, 2, 3, 8} -> '0-3,8' (the form taskset -c and numactl take)."""
    cpus = sorted(cpus)
    out, i = [], 0
    while i < len(cpus):
        j = i
        while j + 1 < len(cpus) and cpus[j + 1] == cpus[j] + 1:
            j += 1
        out.append(str(cpus[i]) if i == j else "%d-%d" % (cpus[i], cpus[j]))
        i = j + 1
    return ",".join(out)


def nodes():
    """The NUMA node ids of this host, sorted ([] on a host without the sysfs tree)."""
    try:
        names = os.listdir(NODE_ROOT)
    except OSError:
        return []
    return sorted(int(n[4:]) for n in names if re.fullmatch(r"node\d+", n))


def node_cpus(n):
    return parse_cpulist(_read(os.path.join(NODE_ROOT, "node%d" % n, "cpulist")))


def node_meminfo(n):
    """{mem_total_gb, mem_free_gb, file_pages_gb} of node n from its meminfo, or {} when unreadable."""
    text = _read(os.path.join(NODE_ROOT, "node%d" % n, "meminfo")) or ""
    out = {}
    for key, name in (("MemTotal", "mem_total_gb"), ("MemFree", "mem_free_gb"), ("FilePages", "file_pages_gb")):
        m = re.search(r"\b%s:\s*(\d+)\s*kB" % key, text)
        if m:
            out[name] = round(int(m.group(1)) / 1048576.0, 2)
    return out


def thp_mode():
    """{"enabled": "madvise", "defrag": "madvise"}: the bracketed choice of each THP control, None when unreadable."""
    out = {}
    for key in ("enabled", "defrag"):
        text = _read("/sys/kernel/mm/transparent_hugepage/%s" % key)
        m = re.search(r"\[(\w[\w+-]*)\]", text or "")
        out[key] = m.group(1) if m else text
    return out


def _run(argv, timeout=20):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return p.stdout.strip() if p.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _visible_index(var):
    v = (os.environ.get(var) or "").split(",")[0].strip()
    return v if v.isdigit() else None


def _gpu_numa_node_uncached():
    """The NUMA node of the first visible GPU: NVIDIA through nvidia-smi's PCI bus id and sysfs, AMD through the
    first AMD display device in /sys/class/drm. None when unknown (a one-node host reports -1 there)."""
    if shutil.which("nvidia-smi"):
        argv = ["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader"]
        idx = _visible_index("CUDA_VISIBLE_DEVICES")
        if idx is not None:
            argv += ["-i", idx]
        s = _run(argv)
        if s:
            bus = s.splitlines()[0].strip().lower()
            m = re.fullmatch(r"([0-9a-f]+):([0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f])", bus)
            if m:
                node = _read("/sys/bus/pci/devices/%04x:%s/numa_node" % (int(m.group(1), 16), m.group(2)))
                if node is not None and node.lstrip("-").isdigit():
                    return int(node) if int(node) >= 0 else None
    try:
        cards = sorted(os.listdir("/sys/class/drm"))
    except OSError:
        cards = []
    amd = [c for c in cards if re.fullmatch(r"card\d+", c)
           and (_read("/sys/class/drm/%s/device/vendor" % c) or "").lower() == "0x1002"]
    idx = _visible_index("HIP_VISIBLE_DEVICES") or _visible_index("ROCR_VISIBLE_DEVICES")
    pick = amd[int(idx)] if idx is not None and int(idx) < len(amd) else (amd[0] if amd else None)
    if pick:
        node = _read("/sys/class/drm/%s/device/numa_node" % pick)
        if node is not None and node.lstrip("-").isdigit():
            return int(node) if int(node) >= 0 else None
    return None


def gpu_numa_node():
    """The GPU's NUMA node, read once per job: the value is kept in MOJOLEARN_BENCH_GPU_NUMA_NODE ('none' when
    unknown) so every child process of the conductor inherits it instead of asking nvidia-smi again."""
    v = os.environ.get(GPU_NODE_ENV)
    if v is not None:
        return int(v) if v.lstrip("-").isdigit() and int(v) >= 0 else None
    n = _gpu_numa_node_uncached()
    os.environ[GPU_NODE_ENV] = "none" if n is None else str(n)
    return n


CGROUP_ROOT = "/sys/fs/cgroup"
PROC_CGROUP = "/proc/self/cgroup"


def _quota_cores(quota, period):
    """Whole cores (rounded up) of one quota/period pair, or None ('max', -1, empty or unreadable)."""
    try:
        q, p = int(quota), int(period)
    except (TypeError, ValueError):
        return None
    return -(-q // p) if q > 0 and p > 0 else None


def _cgroup_dirs(root, rel):
    """ROOT and every directory of REL below it ('' or '/' is the root alone)."""
    out, at = [root], root
    for part in (rel or "").split("/"):
        if part and part != "..":
            at = os.path.join(at, part)
            out.append(at)
    return out


def cgroup_quota_cores():
    """The CPU quota of this process's cgroup in whole cores (rounded up), or None when there is none: cgroup v2
    `cpu.max`, and cgroup v1 `cpu.cfs_quota_us` / `cpu.cfs_period_us`, each read at the mount root and along the
    process's own cgroup path (/proc/self/cgroup), taking the tightest. The same rule as the product's
    core/host_predict_threads.mojo `_cgroup_cpu_quota` and tools/ops/host_probe.py `cgroup_cores`.

    FAIRNESS FIX (2026-11-19, lane fx-cagra-taxi): this read cgroup v2's root `cpu.max` only. On the RunPod L40S pod
    bd9c355c19c0 the quota is cgroup v1 (host_probe: cgroup_cores 14), so the allocation read as the affinity (128),
    plan() answered "node 1 holds 64 of the allocation's 128 cores" and bound nothing: every worker started on
    whichever node the scheduler chose. algos/cagra fit loses 5-7 s when its worker is on the node that is not the
    GPU's (7 of 7 recorded cells: node 0 11,083 / 11,382 ms taxi and 18,709 ms istella; node 1 4,480 ms taxi and
    13,748 / 13,838 ms istella, across wheels d95f805cce, d3549c756b and 444367af36 whose cagra code objects are
    byte-identical), which the same-pod pair job nvc-tg-20261119T140227Z-feb7a5 read as a 2.54x regression of the
    newer wheel."""
    v2_rel, v1_rel = "", ""
    for line in (_read(PROC_CGROUP) or "").splitlines():
        parts = line.split(":", 2)
        if len(parts) < 3:
            continue
        if parts[0] == "0" and parts[1] == "":
            v2_rel = parts[2]
        elif "cpu" in parts[1].split(","):
            v1_rel = parts[2]
    found = []
    for d in _cgroup_dirs(CGROUP_ROOT, v2_rel):
        words = (_read(os.path.join(d, "cpu.max")) or "").split()
        if len(words) >= 2:
            found.append(_quota_cores(words[0], words[1]))
    for d in _cgroup_dirs(os.path.join(CGROUP_ROOT, "cpu"), v1_rel):
        found.append(_quota_cores(_read(os.path.join(d, "cpu.cfs_quota_us")),
                                  _read(os.path.join(d, "cpu.cfs_period_us"))))
    found = [n for n in found if n]
    return min(found) if found else None


def allocation_cores():
    """The cores this process may use: its cgroup CPU quota (`cgroup_quota_cores`, v2 or v1), capped by its
    affinity; its affinity when there is no quota; None if unknown."""
    try:
        aff = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        aff = None
    n = cgroup_quota_cores()
    if n:
        return min(n, aff) if aff else n
    return aff


def mems_allowed():
    m = re.search(r"^Mems_allowed_list:\s*(\S+)", _read("/proc/self/status") or "", re.M)
    return parse_cpulist(m.group(1)) if m else None


def box_facts():
    """The GPU's node, every node's CPUs and memory, the THP modes, the allocation: for job/box.txt."""
    return {"gpu_numa_node": gpu_numa_node(),
            "numa": {str(n): dict(node_meminfo(n), cpus=cpulist(node_cpus(n))) for n in nodes()},
            "thp": thp_mode(), "allocation_cores": allocation_cores()}


def plan():
    """{"node", "cpus", "mem", "how", "why"}: where race workers go. node None = not bound (why says so)."""
    if (os.environ.get(BIND_ENV) or "1").strip() == "0":
        return {"node": None, "why": "off (%s=0)" % BIND_ENV}
    ns = nodes()
    if len(ns) < 2:
        return {"node": None, "why": "one NUMA node"}
    g = gpu_numa_node()
    if g is None or g not in ns:
        return {"node": None, "why": "the GPU's NUMA node is unknown"}
    try:
        aff = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        return {"node": None, "why": "no affinity on this platform"}
    cpus = node_cpus(g) & aff
    need = allocation_cores() or len(aff)
    if len(cpus) < need:
        # binding would leave cores of the allocation idle (AGENTS.md part 1): the workers stay unbound
        return {"node": None, "why": "node %d holds %d of the allocation's %d cores" % (g, len(cpus), need)}
    allowed = mems_allowed()
    if allowed is not None and g not in allowed:
        return {"node": None, "why": "the cgroup does not allow memory on node %d" % g}
    try:
        min_free = float(os.environ.get(MIN_FREE_ENV) or MIN_FREE_GB)
    except ValueError:
        min_free = MIN_FREE_GB
    free = node_meminfo(g).get("mem_free_gb")
    mem = "bind" if free is not None and free >= min_free else "preferred"
    # TRIED BEFORE USE (lane fx-pod-job2, 2026-11-18): on the RunPod L40S container set_mempolicy is refused
    # ("set_mempolicy: Operation not permitted", no CAP_SYS_NICE), so `numactl --membind` exited 1 before the worker
    # started and every worker-protocol race failed its smoke (42 of 46 in nvc-ab-20261118T201200Z-d21ba0; 12 of 14
    # owed arms were never timed). Each form is now run once on `true` and the first that starts is used: cores and
    # memory, then cores only with numactl, then taskset; the placement records why memory is unbound.
    numactl, taskset = shutil.which("numactl"), shutil.which("taskset")
    tries = []
    if numactl:
        tries.append(([numactl, "--physcpubind=%s" % cpulist(cpus),
                       ("--membind=%d" if mem == "bind" else "--preferred=%d") % g], mem))
    refused = None
    for how, m in tries:
        err = _try_prefix(how)
        if err is None:
            return {"node": g, "cpus": cpulist(cpus), "n_cpus": len(cpus), "allocation_cores": need, "mem": m,
                    "threads": min(need, len(cpus)), "state": "bound", "node_free_gb": free, "how": how,
                    "why": "the GPU's node"}
        refused = err
    why_mem = ("unbound (numactl memory policy refused: %s)" % refused) if refused else "unbound (numactl missing)"
    cores_only = []
    if numactl:
        cores_only.append([numactl, "--physcpubind=%s" % cpulist(cpus)])
    if taskset:
        cores_only.append([taskset, "-c", cpulist(cpus)])
    last = None
    for how in cores_only:
        err = _try_prefix(how)
        if err is None:
            # cores bound by sched_setaffinity (numactl --physcpubind or taskset; a container that refuses
            # set_mempolicy allows it): first touch then places the worker's memory on this node
            return {"node": g, "cpus": cpulist(cpus), "n_cpus": len(cpus), "allocation_cores": need,
                    "threads": min(need, len(cpus)), "state": "cores", "mem": why_mem, "node_free_gb": free,
                    "how": how, "why": "the GPU's node"}
        last = err
    if not numactl and not taskset:
        return {"node": None, "why": "neither numactl nor taskset on this host"}
    return {"node": None, "why": "no placement command starts a process here: %s" % (last or refused)}


def _try_prefix(how, timeout=20):
    """None when HOW followed by `true` starts and exits 0 on this host, else a short reason (its stderr). Run once
    per process by plan() (wrap caches the plan), never per race."""
    try:
        p = subprocess.run(list(how) + ["true"], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return str(exc)[:160]
    if p.returncode == 0:
        return None
    return ((p.stderr or p.stdout or "").strip().splitlines() or ["exit %d" % p.returncode])[0][:160]


_PLAN = []


def job_plan():
    """The placement of every race worker of this job. ONE PLAN PER JOB (lane fx-numa-bind-smoke, 2026-11-19): the
    first job with binding live on the RunPod L40S pod (nvc-tg-20261119T191651Z-c11a4c, wheel a7c89e7558) failed all
    21 smokes "incomplete worker evidence; ... worker runtime settings changed or missing: ours": the parameter probe
    and the timed worker are started by two driver processes, each made its own plan, and the placement in each
    worker's environment carried node_free_gb read at that moment (259.76 GB in the probe, 266.39 GB in the worker of
    algos/bpe-train/enwik8), so tools/bench_board_evidence.py saw the runtime settings change between them. The plan
    is now made once, kept in PLAN_ENV and inherited by every child (tools/bench_board.py child_env seeds it), so the
    probe and the timed worker carry the same placement. Memoized per process as well."""
    if _PLAN:
        return _PLAN[0]
    p = None
    v = os.environ.get(PLAN_ENV)
    if v:
        try:
            p = json.loads(v)
        except ValueError:
            p = None
    if not isinstance(p, dict):
        try:
            p = plan()
        except Exception as exc:  # noqa: BLE001 -- placement is never a reason for a race to fail
            p = {"node": None, "why": "placement failed: %s" % str(exc)[:200]}
        if p.get("node") is None:
            p.setdefault("state", "unbound")
        os.environ[PLAN_ENV] = json.dumps(p, sort_keys=True)
    _PLAN.append(p)
    return p


def wrap(cmd, env):
    """(cmd, placement): the worker command prefixed by the job's placement, and the placement recorded in the
    worker's environment (PLACEMENT_ENV) for its cell. A worker environment that sets MOJOLEARN_BENCH_NUMA_BIND=0
    is not bound."""
    p = job_plan()
    if (env.get(BIND_ENV) or "1").strip() == "0" and p.get("node") is not None:
        p = {"node": None, "state": "unbound", "why": "off (%s=0)" % BIND_ENV}
    rec = {k: v for k, v in p.items() if k != "how"}
    env[PLACEMENT_ENV] = json.dumps(rec, sort_keys=True)
    if p.get("node") is None:
        return list(cmd), rec
    return list(p["how"]) + list(cmd), rec


def worker_node():
    """Inside a worker: {"placement" (what the conductor asked), "cpu_nodes" (nodes of its allowed CPUs),
    "mems_allowed", "node_now" (the node of the CPU it is on now)}."""
    try:
        placement = json.loads(os.environ.get(PLACEMENT_ENV) or "null")
    except ValueError:
        placement = None
    out = {"placement": placement}
    aff = None
    try:
        aff = os.sched_getaffinity(0)
        out["cpu_nodes"] = sorted(n for n in nodes() if node_cpus(n) & aff)
    except (AttributeError, OSError):
        out["cpu_nodes"] = None
    # whether the placement asked was applied: the worker's own affinity lies inside the planned cores. A bind that
    # did not apply is recorded as unbound (tools/paper_compare.py unbound_node: NUMA-MIXED sees it)
    want = parse_cpulist((placement or {}).get("cpus")) if isinstance(placement, dict) else set()
    if not isinstance(placement, dict) or placement.get("node") is None:
        out["bound"] = False
    elif aff is not None and want:
        out["bound"] = bool(aff) and set(aff) <= want
    else:
        out["bound"] = None
    ma = mems_allowed()
    out["mems_allowed"] = cpulist(ma) if ma is not None else None
    stat = _read("/proc/self/stat") or ""
    try:
        cpu = int(stat.rsplit(")", 1)[1].split()[36])     # field 39, processor
        out["node_now"] = next((n for n in nodes() if cpu in node_cpus(n)), None)
    except (IndexError, ValueError):
        out["node_now"] = None
    return out


def usage():
    """A getrusage snapshot of this process (None where resource is unavailable)."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF)
    except (ImportError, OSError):
        return None


def delta(a, b, wall_ms):
    """{stime_ms, utime_ms, minflt, majflt, wall_ms} between two snapshots, or None."""
    if a is None or b is None:
        return None
    return {"stime_ms": round((b.ru_stime - a.ru_stime) * 1000.0, 3),
            "utime_ms": round((b.ru_utime - a.ru_utime) * 1000.0, 3),
            "minflt": int(b.ru_minflt - a.ru_minflt), "majflt": int(b.ru_majflt - a.ru_majflt),
            "wall_ms": None if wall_ms is None else round(wall_ms, 3)}


def summarize(calls):
    """The scored calls' usage of one phase over the scored rounds -> {stime_ms, wall_ms, minflt, majflt,
    stime_fraction, host_stall}: sums over every call; host_stall when system time is above STALL_FRACTION of the
    wall. None when no call carried usage."""
    calls = [c for c in calls or [] if c and c.get("wall_ms")]
    if not calls:
        return None
    s = {k: sum(c.get(k) or 0 for c in calls) for k in ("stime_ms", "utime_ms", "wall_ms", "minflt", "majflt")}
    s["calls"] = len(calls)
    s["per_call"] = [{k: c.get(k) for k in ("wall_ms", "stime_ms", "minflt", "majflt")} for c in calls]
    s["stime_fraction"] = round(s["stime_ms"] / s["wall_ms"], 4) if s["wall_ms"] else None
    s["host_stall"] = bool(s["stime_fraction"] is not None and s["stime_fraction"] > STALL_FRACTION)
    return s


def phase_summary(host_usage, phase):
    """summarize() of one phase ('fit' or 'infer') over a worker's scored rounds (the conductor's host_usage list of
    {"fit": [call...], "infer": [call...] or None, "round": r}); None when none was recorded (an older worker)."""
    calls = []
    for rnd in host_usage or []:
        calls.extend((rnd or {}).get(phase) or [])
    return summarize(calls)
