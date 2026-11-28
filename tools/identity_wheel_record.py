#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE ONE IDENTITY RUN, ON THE BUILT RELEASE WHEELS (Andrew 2026-11-05; tools/release.py --identity-on-wheels).

    python3 tools/identity_wheel_record.py --columns nvidia,ptx --label-device l40s \\
        --wheel CORE.whl --wheel NVIDIA.whl --sha256 <name>=<hex> ... \\
        --out DIR [--gpus N] [--workers-per-gpu auto|K] [--venv DIR] [--python python3]

Runs in a clean tree at the wheels' source commit (tools/ and the harness are that commit's), on the device the
columns name: the NVIDIA box (nvidia, ptx), the AMD box (amd), the Mac (apple) or a GitHub runner (cpu). In order:

1. install  a fresh venv, the exact wheel files, each checked against its --sha256 BEFORE install; the checked
            "<file> <sha256>" lines go to DIR/installed-wheels.txt (record_identity_column.sh INSTALLED_WHEELS);
2. select   per column, the lanes of the families that record (--record-families, which tools/release.py decides
            from its family plan: AGENTS.md part 4, tools/release_families.py; `all` by default): no family
            records -> the column records nothing (REUSED n / RUN 0);
3. record   the owed lanes in chunks, ALL AT ONCE: one worker slot per (device, k) for k < workers per device,
            each chunk `RECORD_FROM_INSTALLED=1 LANES=<chunk> LANE_SELECTION=family-chunk|full-chunk
            record_identity_column.sh <label> DIR/<label>/w<i>` pinned to its device (CUDA_VISIBLE_DEVICES /
            HIP_VISIBLE_DEVICES). With more devices than columns each column gets its own device(s); with fewer,
            columns share them. Workers per device (--workers-per-gpu auto): one per core of the allocation,
            bounded by the GPU's free memory at GPU_GIB_PER_WORKER and the host's at HOST_GIB_PER_WORKER; the
            lanes are packed into chunks of equal estimated cost, longest lane first (`chunks`). It prints
            "busy plan: W worker(s) of T" before it starts and "timing: record <s>s wall, busy cores mean B of N;
            chunks <min>-<max>s; slowest lanes: ..." and "gpu memory: peak ..." after.
4. summary  DIR/identity-wheels.json (schema mojolearn.identity-wheels.v1): per column the selection, the
            records written (with their admissible= line) and the rc; exit 0 only when every chunk recorded
            admissibly. The records go to admission (tools/admit_identity_columns.sh) with every other column.
5. extended the single-device verify commands' reference captures, from the installed wheel, all at once, one
            process each on device 0: `verify-graph --vendor V --require-installed --out`, `verify-distributed-program
            --output`, `verify-partitioned-program --output`, into DIR/extended-references/<column>/<profile>.json
            (tools/admit_extended_references.py admits them beside the lane tables; skipped on a --shard > 0 job,
            the shard-0 job of the CPU matrix captures them once); a failed capture fails the record;
6. evidence DIR/evidence/<column>/record[-shard-i-of-N]/ (tools/verification_evidence.py stage): the summary, every
            record JSON, the chunk logs (capped) and a MANIFEST (wheels, commit, chunk times, busy plan); the fetch
            on the Mac writes it into verification/latest/<column>/ (always overwritten; git history is the history).

The box: IDENTITY_RECORD_BOX=<provider>:<id> (runpod:<pod id>, digitalocean:<droplet id>, github:<run id>/<job>) is
passed through to every recorder unchanged, so each record's run block names the machine (tools/identity_break.py
box_identity derives it when unset), and the summary carries it as `box`. The wheel list reaches each record as
IDENTITY_RECORD_WHEELS through record_identity_column.sh.

Nothing here builds, rents or publishes. Tests: tools/test_identity_wheel_record.py (fake recorder, no device).
"""
import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

SCHEMA = "mojolearn.identity-wheels.v1"
ROOT = Path(__file__).resolve().parent.parent
RECORDER = ROOT / "tools" / "record_identity_column.sh"
#: column -> (label pattern, device environment variable). The labels are the ones record_identity_column.sh
#: derives the backend from (nvidia-*, nvidia-ptx-*-sm80, amd-*, apple-*, cpu).
COLUMNS = {
    "nvidia": ("nvidia-{device}", "CUDA_VISIBLE_DEVICES"),
    # ONE NVIDIA WHEEL (lane nvidia-one-wheel-simple): the ptx column installs the same core + nvidia wheels;
    # record_identity_column.sh loads the PTX set for an nvidia-ptx-* label (MOJOLEARN_GPU_ARCH=sm_80)
    "ptx": ("nvidia-ptx-{device}-sm80", "CUDA_VISIBLE_DEVICES"),
    "amd": ("amd-{device}", "HIP_VISIBLE_DEVICES"),
    "apple": ("apple-{device}", None),
    "cpu": ("cpu", None),
}
#: CPU threads per record worker on the CPU column: record_identity_column.sh's own default (CPU_THREADS=3). The CPU
#: column's fits ARE the host thread pool, and a 3-thread pool is the parallel host path that column records.
CPU_THREADS = 3
#: CPU threads per record worker on a GPU column (nvidia, ptx, amd, apple) (lane/verify-speed-1, 2026-11-28; Andrew:
#: more workers on Apple). Its fits run on the device; the host pool runs the lane's glue only. The 0.9.0 Apple record
#: (0f0842675b, apple-m4, 10 cores) planned 10 // 3 = 3 workers of 3 threads; its 4,197 cells averaged 4.1 s, and the
#: same cells took 996 s in all on the CPU column, so the Metal side waits on the device, not on its pool. One thread
#: a worker lets the cores carry workers instead. Identical bits never depend on the pool size (the CPU column, the
#: verify of every user and the admission's tamper test run other counts), so the record at 1 thread is the same bits.
GPU_COLUMN_THREADS = 1
#: evidence column -> the graph verifier's --vendor (its loaded x_sequence binding's own vendor read-back)
GRAPH_VENDOR = {"cpu": "cpu", "nvidia": "cuda", "amd": "hip", "apple": "metal"}
#: the reference captures of the single-device verify commands: profile -> argv after `-m mojolearn`
EXTENDED_COMMANDS = {
    "verify-graph": lambda column, out: ["verify-graph", "--vendor", GRAPH_VENDOR[column], "--require-installed",
                                         "--out", out],
    "verify-distributed-program": lambda column, out: ["verify-distributed-program", "--output", out],
    "verify-partitioned-program": lambda column, out: ["verify-partitioned-program", "--output", out],
}
#: GPU memory per concurrent record worker (GiB) for --workers-per-gpu auto (rel-fix-2, 2026-11-26). Was a 12 GiB
#: guess, which held the L40S record of nvc-ab-20261126T214026Z-7116a8 (acdf0ad303) to 3 workers on 44 GiB free:
#: 1030 s at busy cores mean 2.5 of 14. In the same job `verify --all` ran 13 cell workers at once on that GPU (the
#: same harness lanes and fixtures, both sides of every cell) with no out-of-memory, so a worker takes at most
#: 44 / 13 = 3.4 GiB there. The record now samples the GPU's used memory while it runs and prints the measured peak
#: per worker ("gpu memory:" line), the number to put here after the next run.
GPU_GIB_PER_WORKER = 3
#: Cores one GPU record worker keeps busy. The same job's record.busy: 3 workers of CPU_THREADS=3 at mean 2.5 and
#: max 3.6 busy cores, so a worker averages 0.83 of a core (the GPU and the lane's serial host glue bound it, not its
#: thread pool). One worker per core of the allocation, CPU_THREADS unchanged.
GPU_WORKER_CORES = 1
#: Host memory per record worker (GiB): the verify cell workers' bound (mojolearn._verify_all.VERIFY_WORKER_GIB). The
#: planned figure until a record of the column has measured its own (`PEAKS`).
HOST_GIB_PER_WORKER = 4
#: THE MEASURED PEAK RSS OF ONE RECORD WORKER, per column (lane/verify-speed-1, 2026-11-28): the largest resident set
#: of any process a record's chunks ran (getrusage RUSAGE_CHILDREN ru_maxrss after every chunk is reaped), which the
#: record prints as "host memory: peak RSS ..." and writes to identity-wheels.json timing.host_memory. A column with
#: no figure plans at its ESTIMATES entry, else HOST_GIB_PER_WORKER. Put the printed figure here after a record; the
#: plan reads it with PEAK_HEADROOM on top. No record has printed it yet.
PEAKS = {}
#: UNMEASURED, until PEAKS has the column: the CPU smoke's largest lane process is 0.34 GiB (mamba2,
#: tools/ops/wheel_verify_smoke_plan/lanes.tsv, every lane in its own process); a record chunk runs a few hundred
#: lanes in one interpreter, so the Apple plan takes 1.5 GiB, four times that, not the verify sweep's 4 GiB guess
#: (which on a 16 GiB Mac with 10 GiB available would plan 2 workers, fewer than the 3 of the 0.9.0 record).
ESTIMATES = {"apple": 1.5}
#: the share of a measured peak added before dividing the available memory by it
PEAK_HEADROOM = 1.25
#: APPLE GPU MEMORY IS THE HOST'S (unified memory). Metal admits a working set of about two thirds of the RAM
#: (MTLDevice.recommendedMaxWorkingSetSize on a 16 GiB M-series Mac: 10.7 GiB); the record's Metal buffers are small
#: fixtures (the largest cell's arrays are under 0.1 GiB), so a worker plans METAL_GIB_PER_WORKER of that set.
METAL_WORKING_SET_SHARE = 2.0 / 3.0
METAL_GIB_PER_WORKER = 1.0
#: a column's cells are SMALL when its measured mean cell is under this many seconds; a small-cell GPU column records
#: at GPU_COLUMN_THREADS, a column of large cells keeps CPU_THREADS (a cell of minutes on one host thread would set
#: the wall)
SMALL_CELL_S = 10.0
#: The per-lane cost hints that pack the chunks: the verify sweep's measured mean seconds per cell of each lane
#: (the same harness lanes), shipped in the tree; a lane's record runs its fixtures one after another.
COST_HINTS = ROOT / "python" / "mojolearn" / "verify_reference" / "cell_seconds.json"
#: A COLUMN'S OWN MEASURED SECONDS (lane/verify-speed-1, 2026-11-28): tools/identity_record_seconds/<column>.json,
#: the same format (mean seconds of one cell per lane), written by a record as DIR/record-seconds-<column>.json and
#: committed after it. The shipped hints are the L40S's: x-decomp-robust-cov is 1.2 s a cell there and 49 s on the
#: apple-m4, so Apple lanes packed by them start in the wrong order. A column with its own file packs by it.
COLUMN_HINTS = ROOT / "tools" / "identity_record_seconds"
#: seconds a lane costs the recorder beyond its fits (start, hashing, the record write): the L40S chunk w1 spent
#: 1013 s wall on 330 lanes whose fits summed 724 s
LANE_OVERHEAD_S = 1.0


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def label(column, device):
    if column not in COLUMNS:
        raise SystemExit("identity_wheel_record: unknown column %s (%s)" % (column, ", ".join(COLUMNS)))
    pattern = COLUMNS[column][0]
    if "{device}" in pattern and not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", device or ""):
        raise SystemExit("identity_wheel_record: --label-device must be a lowercase slug (e.g. l40s, mi325x, m2pro)")
    return pattern.format(device=device)


def cost_hints_path(column):
    """The cost hints a column packs by: its own measured file when committed, else the shipped verify hints."""
    own = COLUMN_HINTS / ("%s.json" % column)
    return own if own.is_file() else COST_HINTS


def column_hints(column, shipped=None, own=None):
    """{lane: mean cell seconds} a column packs by: its own measured seconds (`cost_hints_path`) where it has them;
    a lane it has not measured takes the shipped hint times the median ratio own / shipped over the lanes both
    measured (the column's speed against the L40S), so a new or unmeasured lane still packs in its place."""
    shipped = read_hints(COST_HINTS) if shipped is None else shipped
    own = (read_hints(cost_hints_path(column)) if cost_hints_path(column) != COST_HINTS else {}) if own is None else own
    if not own:
        return dict(shipped)
    ratios = sorted(own[k] / shipped[k] for k in own if shipped.get(k, 0) > 0)
    scale = ratios[len(ratios) // 2] if ratios else 1.0
    merged = {k: v * scale for k, v in shipped.items()}
    merged.update(own)
    return merged


def read_hints(path=None):
    """{lane: mean seconds of one cell} of a hints file ('@' keys are not lanes); {} when unreadable."""
    try:
        doc = json.loads(Path(path or COST_HINTS).read_text())
        return {k: float(v) for k, v in (doc.get("seconds") or {}).items()
                if not k.startswith("@") and isinstance(v, (int, float)) and v >= 0}
    except (OSError, ValueError, AttributeError):
        return {}


def record_threads(column, hints=None):
    """CPU threads per record worker: CPU_THREADS on the CPU column; on a GPU column GPU_COLUMN_THREADS when its
    cells are small (the hints' mean cell under SMALL_CELL_S, or no hints), else CPU_THREADS. RECORD_CPU_THREADS in
    the environment pins it (a positive integer)."""
    pinned = (os.environ.get("RECORD_CPU_THREADS") or "").strip()
    if pinned.isdigit() and int(pinned) > 0:
        return int(pinned)
    if column == "cpu":
        return CPU_THREADS
    values = list((hints or {}).values())
    mean = sum(values) / len(values) if values else 0.0
    return GPU_COLUMN_THREADS if mean < SMALL_CELL_S else CPU_THREADS


def lane_costs(lanes, path=None, hints=None):
    """{lane: estimated record seconds}: the cost hint (mean seconds of one cell; `hints`, else the file `path`) plus
    LANE_OVERHEAD_S; a lane with no hint gets its family's (`_allocation.lane_cost`). Only packs the chunks; nothing recorded reads it."""
    hints = read_hints(path) if hints is None else hints
    # a lane with no hint packs at its family's cost (`_allocation.lane_cost`, rel-fix-3), not the median: 42 lanes
    # (most of them gbdt-*) have no row in the shipped table and packed at 0.41 s
    return {x: ALLOCATION.lane_cost(hints, x)[0] + LANE_OVERHEAD_S for x in lanes}


def chunks(lanes, n, cost=None):
    """N chunks of LANES of about equal estimated cost (rel-fix-2): longest lane first, each to the chunk with the
    least cost so far, so every worker ends at about the same time and the heaviest lanes start first. Equal costs
    deal the lanes round robin. The L40S record of acdf0ad303 dealt round robin: one chunk ran 1013 s, two 780 s."""
    n = max(1, min(n, len(lanes)))
    if not lanes:
        return []
    cost = cost or {}
    out, load = [[] for _ in range(n)], [0.0] * n
    for lane in sorted(lanes, key=lambda x: -cost.get(x, 1.0)):   # sorted is stable: equal costs keep lane order
        i = min(range(n), key=lambda k: (load[k], k))
        out[i].append(lane)
        load[i] += cost.get(lane, 1.0)
    return [c for c in out if c]


def plan_slots(columns, devices, workers_per_device):
    """{column: [(device or None, k), ...]}: the worker slots each column records on. Devices are split between
    the columns (a column on its own device(s) when there are enough); with no devices (cpu, apple) every column
    shares the host's worker slots."""
    if not devices:
        slots = [(None, k) for k in range(workers_per_device)]
        return {c: slots for c in columns}
    out = {c: [] for c in columns}
    if len(devices) >= len(columns):
        for i, d in enumerate(devices):
            out[columns[i % len(columns)]].extend((d, k) for k in range(workers_per_device))
    else:
        for i, c in enumerate(columns):
            d = devices[i % len(devices)]
            out[c].extend((d, k) for k in range(workers_per_device))
    return out


def worker_gib(column):
    """Host memory one record worker plans on: its column's measured peak RSS (`PEAKS`) with PEAK_HEADROOM, else its
    `ESTIMATES` entry, else HOST_GIB_PER_WORKER."""
    peak = PEAKS.get(column)
    return peak * PEAK_HEADROOM if peak else ESTIMATES.get(column, HOST_GIB_PER_WORKER)


def auto_workers(gpus, nproc, free_gib, host_gib=None, column="cpu", threads=None, metal_gib=None):
    """Record workers per device: every core busy, bounded by the GPU's free memory and the host's available memory.
    A GPU worker keeps GPU_WORKER_CORES busy (measured). With no devices (the cpu and apple columns) a worker runs
    `threads` host threads (`record_threads`), so the cores allow nproc // threads of them; the apple column is also
    bounded by its Metal working set (`metal_gib`, the unified memory Metal admits) at METAL_GIB_PER_WORKER. Host
    memory: the available GiB over `worker_gib(column)`."""
    threads = threads or (CPU_THREADS if column == "cpu" else GPU_COLUMN_THREADS)
    per = worker_gib(column)
    by_host = max(1, int(host_gib // per)) if host_gib else None
    if gpus <= 0:
        n = max(1, nproc // threads)
        if column == "apple" and metal_gib:
            n = min(n, max(1, int(metal_gib // METAL_GIB_PER_WORKER)))
        return max(1, min(n, by_host)) if by_host else n
    by_cores = max(1, nproc // (GPU_WORKER_CORES * gpus))
    by_memory = max(1, int(free_gib // GPU_GIB_PER_WORKER)) if free_gib else 1
    n = min(by_cores, by_memory)
    if by_host:
        n = min(n, max(1, by_host // gpus))
    return max(1, n)


def host_available_gib(run=subprocess.run):
    """Available host memory in GiB (python/mojolearn/_allocation.py `mem_available_bytes`: MemAvailable capped by
    the cgroup v2 or v1 memory limit less its use); on macOS, where that reads nothing, `darwin_available_bytes`;
    None where unreadable."""
    avail = ALLOCATION.mem_available_bytes()
    if avail is None and sys.platform == "darwin":
        avail = darwin_available_bytes(run)
    return None if avail is None else avail / 2 ** 30


VM_STAT_PAGE = re.compile(r"page size of (\d+) bytes")
VM_STAT_LINE = re.compile(r"^Pages (free|inactive|speculative|purgeable):\s+(\d+)\.", re.M)


def darwin_available_bytes(run=subprocess.run):
    """macOS available memory: the free, inactive, speculative and purgeable pages of `vm_stat` (the pages the
    kernel hands out without swapping), or None. Kept in this tool, not in _allocation.mem_available_bytes, so the
    verify sweep's worker plan on a Mac (which reads no memory bound today) does not change with it."""
    try:
        text = run(["vm_stat"], capture_output=True, text=True, timeout=30).stdout or ""
    except (OSError, subprocess.SubprocessError):
        return None
    page = VM_STAT_PAGE.search(text)
    pages = VM_STAT_LINE.findall(text)
    if not page or not pages:
        return None
    return int(page.group(1)) * sum(int(n) for _, n in pages)


def darwin_memsize_gib(run=subprocess.run):
    """The Mac's RAM in GiB (`sysctl -n hw.memsize`), or None: the base of the Metal working set (unified memory)."""
    try:
        out = run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=30).stdout
        return int(out.strip()) / 2 ** 30
    except (OSError, ValueError, AttributeError, subprocess.SubprocessError):
        return None


def peak_rss_gib():
    """The largest resident set of any reaped child process of this run, in GiB (ru_maxrss: bytes on macOS, KiB on
    Linux), or None."""
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    except (ImportError, OSError, AttributeError):
        return None
    return rss / (2 ** 30 if sys.platform == "darwin" else 2 ** 20) if rss else None


def children_cpu_seconds():
    """CPU seconds of every reaped child of this process and their descendants (the recorder chunks)."""
    try:
        import resource
        u = resource.getrusage(resource.RUSAGE_CHILDREN)
        return u.ru_utime + u.ru_stime
    except (ImportError, OSError, AttributeError):
        return 0.0


DONE_LINE = re.compile(r"^# DONE (\S+?)/\S+ \S+ ([0-9.]+)s$", re.M)


CELL_LINE = re.compile(r"^# CELL (\S+?)/\S+ \S+ ([0-9.]+)s$", re.M)


def cell_means(paths):
    """{lane: mean seconds of one cell} over the harness's `# CELL <lane>/<fixture> <verdict> <s>s` lines."""
    total, count = {}, {}
    for path in paths:
        try:
            text = Path(path).read_text(errors="replace")
        except OSError:
            continue
        for lane, sec in CELL_LINE.findall(text):
            total[lane] = total.get(lane, 0.0) + float(sec)
            count[lane] = count.get(lane, 0) + 1
    return {lane: total[lane] / count[lane] for lane in total}


def lane_seconds(paths):
    """{lane: seconds} summed over the harness's `# DONE <lane>/<fixture> repeat=../<part> <s>s` lines."""
    out = {}
    for path in paths:
        try:
            text = Path(path).read_text(errors="replace")
        except OSError:
            continue
        for lane, sec in DONE_LINE.findall(text):
            out[lane] = out.get(lane, 0.0) + float(sec)
    return out


def _allocation():
    """python/mojolearn/_allocation.py of this tree, loaded by path (the package is not imported here)."""
    spec = importlib.util.spec_from_file_location("mojolearn_allocation_by_path",
                                                  ROOT / "python" / "mojolearn" / "_allocation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ALLOCATION = _allocation()


def alloc_cores():
    """The process's core allocation from the one allocation function (python/mojolearn/_allocation.py `cores`): the
    job's MOJOLEARN_ALLOC_CORES, else the affinity capped by the cgroup v2 or v1 CPU quota (AGENTS.md part 1).
    rel-fix-3, 2026-11-26: the copy here read only v2 cpu.max and planned the L40S pod's v1 14-core quota as 128
    ("busy plan: 14 worker(s) of 14 (1 device(s), 128 core(s)", the 14 from GPU memory, not the cores)."""
    return ALLOCATION.cores()[0]


def gpu_used_gib(vendor):
    """The largest used memory of the box's GPUs in GiB, or None."""
    try:
        if vendor == "nvidia":
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=60).stdout
            vals = [float(x) / 1024 for x in out.split()]
        else:
            out = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--json"], capture_output=True, text=True,
                                 timeout=60).stdout
            doc = json.loads(out or "{}")
            vals = [float(v.get("VRAM Total Used Memory (B)", 0)) / 2 ** 30 for v in doc.values() if isinstance(v, dict)]
        return max(vals) if vals else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def gpu_free_gib(vendor):
    """The smallest free memory of the box's GPUs in GiB (nvidia-smi / rocm-smi), or None."""
    try:
        if vendor == "nvidia":
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=60).stdout
            vals = [float(x) / 1024 for x in out.split()]
        else:
            out = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--json"], capture_output=True, text=True,
                                 timeout=60).stdout
            doc = json.loads(out or "{}")
            vals = [(float(v.get("VRAM Total Memory (B)", 0)) - float(v.get("VRAM Total Used Memory (B)", 0))) / 2 ** 30
                    for v in doc.values() if isinstance(v, dict)]
        return min(vals) if vals else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


# What the identity harness itself needs in the venv beyond the release wheels (python/pyproject.toml `verify` extra).
HARNESS_REQUIREMENTS = ("numpy>=1.26.4",)


class Run:
    def __init__(self, args, popen=subprocess.Popen, run=subprocess.run):
        self.a, self.popen, self.run = args, popen, run
        self.out = Path(args.out).resolve()
        self.lock = threading.Lock()
        self.scratch = None  # the throwaway venv's temp dir (install), removed by main
        self.threads = {}    # column -> CPU threads per record worker (`record_threads`), set by _main

    def say(self, msg):
        with self.lock:
            print("identity-wheels: " + msg, flush=True)

    def install(self):
        """A fresh venv with the exact wheels; every file checked against its expected sha256 first."""
        expected = dict(x.split("=", 1) for x in self.a.sha256)
        lines = []
        for w in self.a.wheel:
            name = Path(w).name
            got = sha256(w)
            if expected.get(name) != got:
                raise SystemExit("identity_wheel_record: %s is %s, not the release's %s" % (name, got, expected.get(name)))
            lines.append("%s %s" % (name, got))
        # A throwaway venv lives under a temp dir and goes when the run ends (Run.main); the evidence folder holds
        # logs and records only (2026-11-09: the release's identity-wheels folders held 6.4 GB of venvs). --venv
        # names a caller-owned venv instead, which the caller removes.
        if self.a.venv:
            venv = Path(self.a.venv)
            if venv.exists():
                shutil.rmtree(venv)
        else:
            self.scratch = Path(tempfile.mkdtemp(prefix="identity-wheels-venv-"))
            venv = self.scratch / "venv"
        self.run([self.a.python, "-m", "venv", str(venv)], check=True)
        py = venv / "bin" / "python"
        self.run([str(py), "-m", "pip", "install", "-q", "--no-cache-dir", *[str(w) for w in self.a.wheel]], check=True)
        # The wheels carry no NumPy (python/pyproject.toml: dependencies = []), but the harness the recorder runs in this
        # venv (tools/identity_break.py) imports numpy at module level, so a wheels-only venv fails every lane at import.
        # Install the harness requirement after the wheels, as tools/ops/release_gate_job.sh does for its shard venv;
        # the spec is the wheel's own `verify` extra, so the recorded bytes stay the release's.
        self.run([str(py), "-m", "pip", "install", "-q", "--no-cache-dir", *HARNESS_REQUIREMENTS], check=True)
        listed = self.out / "installed-wheels.txt"
        listed.write_text("\n".join(lines) + "\n")
        return py, listed

    def env(self, py, listed, device, column, extra=None):
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "LANES", "SHARD", "ONLY_CHANGED", "FULL", "REUSE")}
        env.update(RECORD_FROM_INSTALLED="1", PYTHON=str(py), INSTALLED_WHEELS=str(listed), REPEATS="1",
                   SCOPE="all", CPU_THREADS=str(self.threads.get(column, CPU_THREADS)))
        var = COLUMNS[column][1]
        if var and device is not None:
            env[var] = str(device)
            if var == "HIP_VISIBLE_DEVICES":
                env["ROCR_VISIBLE_DEVICES"] = str(device)
        env.update(extra or {})
        return env

    def all_lanes(self, py, column, lab):
        """Every lane of the full scope (identity_columns.py lanes --scope all), on every column. A lane with no side
        on this column (the harness's `_one_sided`, registry ONE_SIDED, 2026-11-24) is recorded too: its cell reads
        `n/a:one-sided: ...` by name, so the record says which side is owed instead of dropping the lane. No per-lane
        selector decides (Andrew 2026-11-05: the selector cannot be trusted to narrow); --record-families narrows
        to whole families (family_lanes)."""
        out = self.out / lab / "select"
        out.mkdir(parents=True, exist_ok=True)
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        p = self.run([str(py), "tools/identity_columns.py", "lanes", "--scope", "all"], cwd=str(ROOT),
                     capture_output=True, text=True, env=env)
        (out / "lanes.log").write_text((p.stdout or "") + (p.stderr or ""))
        text = (p.stdout or "").strip() if p.returncode == 0 else ""
        lanes = [x for x in text.splitlines()[-1].split(",") if x] if text else []
        if not lanes:
            raise RuntimeError("could not list the lanes for %s (%s)" % (lab, out / "lanes.log"))
        return lanes, 0

    def family_lanes(self, py, column, lab):
        """--record-families (tools/release.py, AGENTS.md part 4): the lanes of the families that record, in the
        full scope's order, by this tree's bindings/release_families.json (tools/release_families.py). `all`: every
        lane; `none` or empty: no lane (every family keeps its columns). Returns (lanes, kept count)."""
        lanes, _ = self.all_lanes(py, column, lab)
        want = (self.a.record_families or "").strip()
        if want == "all":
            return lanes, 0
        families = [] if want in ("", "none") else [f for f in want.split(",") if f]
        import release_families as RF
        fam = RF.load()
        unknown = [f for f in families if f not in fam.families]
        if unknown:
            raise SystemExit("identity_wheel_record: --record-families names unknown famil(y/ies): " + ", ".join(unknown))
        picked = RF.lanes_to_record(dict(families=[dict(family=f, record=f in families) for f in fam.names()]),
                                    lanes, fam)
        return picked, len(lanes) - len(picked)

    def main(self):
        try:
            return self._main()
        finally:
            if self.scratch is not None:
                shutil.rmtree(self.scratch, ignore_errors=True)

    def _main(self):
        self.out.mkdir(parents=True, exist_ok=True)
        columns = [c for c in self.a.columns.split(",") if c]
        if not columns or len(set(columns)) != len(columns):
            raise SystemExit("identity_wheel_record: --columns is a comma list of distinct columns")
        labels = {c: label(c, self.a.label_device) for c in columns}
        py, listed = self.install()
        nproc = alloc_cores()
        devices = list(range(self.a.gpus))
        vendor = "nvidia" if columns[0] in ("nvidia", "ptx") else columns[0]
        free = gpu_free_gib(vendor) if devices else None
        host = host_available_gib()
        hints = {c: column_hints(c) for c in columns}
        self.threads = {c: record_threads(c, hints[c]) for c in columns}
        pool = max(self.threads.values())
        metal = (darwin_memsize_gib() or 0.0) * METAL_WORKING_SET_SHARE if "apple" in columns else None
        workers = (auto_workers(len(devices), nproc, free, host, column=columns[0], threads=pool,
                                metal_gib=metal) if self.a.workers_per_gpu == "auto"
                   else int(self.a.workers_per_gpu))
        slots = plan_slots(columns, devices, workers)
        selected, report = {}, dict(schema=SCHEMA, columns={}, nproc=nproc, gpus=len(devices), shard=self.a.shard,
                                    workers_per_device=workers, gpu_free_gib=free,
                                    box=os.environ.get("IDENTITY_RECORD_BOX") or None,
                                    wheels=listed.read_text().split("\n")[:-1])
        shard_i, shard_n = (int(x) for x in self.a.shard.split("/"))
        for c in columns:
            lanes, reused = self.family_lanes(py, c, labels[c])
            # --shard i/N (the GitHub CPU matrix): every job selects by the same rule and records lanes[i::N]
            lanes = lanes[shard_i::shard_n]
            selected[c] = lanes
            report["columns"][c] = dict(label=labels[c], owed=len(lanes), reused=reused, chunks=[], rc=0)
            self.say("%s: REUSED %s / RUN %s (families: %s)" % (
                labels[c], reused, len(lanes) if self.a.record_families != "all" else "all (%d)" % len(lanes),
                self.a.record_families or "none"))
        tasks = []
        for c in columns:
            cost = lane_costs(selected[c], hints=hints[c])
            for i, (chunk, slot) in enumerate(zip(chunks(selected[c], len(slots[c]), cost), slots[c])):
                tasks.append((c, i, chunk, slot))
        total = sum(len(s) for s in slots.values()) if devices else len(next(iter(slots.values())))
        busy = len(tasks)
        self.say("busy plan: %d worker(s) of %d (%d device(s), %d core(s) from %s, %d CPU thread(s) per worker%s%s%s;"
                 " %.2f GiB a worker planned%s; lanes packed by %s)"
                 % (busy, total, len(devices), nproc, ALLOCATION.cores()[1], pool,
            "" if free is None else ", %.0f GiB free per device" % free,
            "" if host is None else ", %.0f GiB host memory available" % host,
            "" if not metal else ", %.1f GiB Metal working set" % metal,
            worker_gib(columns[0]), " (measured peak)" if PEAKS.get(columns[0]) else " (unmeasured)",
            ", ".join(sorted({str(cost_hints_path(c).relative_to(ROOT)) for c in columns}))))
        threads, results = [], {}
        # WHERE THE TIME GOES (rel-fix-2): the record's wall, busy cores (CPU seconds of the reaped chunks over the
        # wall), the chunks' spread, the 5 slowest lanes, and the GPU's peak used memory per worker.
        t_rec, cpu0 = time.monotonic(), children_cpu_seconds()
        peak, sampling = [None, gpu_used_gib(vendor) if devices else None], threading.Event()

        def sample():
            while not sampling.wait(5):
                used = gpu_used_gib(vendor)
                if used is not None and (peak[0] is None or used > peak[0]):
                    peak[0] = used
        sampler = threading.Thread(target=sample, daemon=True) if devices else None
        if sampler:
            sampler.start()

        def work(c, i, chunk, slot):
            # A FAILURE NEVER STOPS THE OTHERS: every chunk runs; failures are collected and reported once below
            try:
                record(c, i, chunk, slot)
            except Exception as exc:   # a crashed launch is this chunk's failure, never the run's
                results[(c, i)] = dict(worker=i, device=slot[0], lanes=len(chunk), rc=1, records=[],
                                       error="%s: %s" % (type(exc).__name__, exc))
                self.say("%s w%d (device %s): CRASHED %s" % (labels[c], i, slot[0], exc))

        def record(c, i, chunk, slot):
            out = self.out / labels[c] / ("w%d" % i)
            out.mkdir(parents=True, exist_ok=True)
            t0, started = time.monotonic(), utc()
            env = self.env(py, listed, slot[0], c, dict(LANES=",".join(chunk), LANE_SELECTION=(
                "full-chunk" if self.a.record_families == "all" else "family-chunk")))
            with open(out / "record.log", "w") as log:
                rc = self.popen(["bash", str(RECORDER), labels[c], str(out)], cwd=str(ROOT), env=env,
                                stdout=log, stderr=subprocess.STDOUT).wait()
            text = (out / "record.log").read_text(errors="replace")
            m = re.findall(r"^RECORDED (\S+) rc=(\d+) admissible=(yes|NO) json=(\S+)$", text, re.M)
            results[(c, i)] = dict(worker=i, device=slot[0], lanes=len(chunk), rc=rc,
                                   records=[dict(name=n, harness_rc=int(r), admissible=adm == "yes", json=j)
                                            for n, r, adm, j in m],
                                   seconds=round(time.monotonic() - t0), start=started, end=utc())
            self.say("%s w%d (device %s): %d lane(s), rc=%d" % (labels[c], i, slot[0], len(chunk), rc))
        # every slot of a device runs one chunk at a time; chunks of columns sharing a device queue on it
        by_slot = {}
        for t in tasks:
            by_slot.setdefault(t[3], []).append(t)

        def drain(queue):
            for t in queue:
                work(*t)
        for queue in by_slot.values():
            th = threading.Thread(target=drain, args=(queue,))
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
        if sampler:
            sampling.set()
            sampler.join()
        report["timing"] = self.timing(time.monotonic() - t_rec, children_cpu_seconds() - cpu0, nproc, results,
                                       labels, peak, workers)
        report["threads_per_worker"] = dict(self.threads)
        ok = True
        for (c, i), r in sorted(results.items()):
            row = report["columns"][c]
            row["chunks"].append(r)
            good = r["rc"] == 0 and r["records"] and all(x["admissible"] for x in r["records"])
            if not good:
                row["rc"] = r["rc"] or 1
                ok = False
        failed = ["%s w%d (%d lane(s), rc %s)" % (labels[c], i, r["lanes"], r["rc"]) for (c, i), r in sorted(results.items())
                  if not (r["rc"] == 0 and r["records"] and all(x["admissible"] for x in r["records"]))]
        if shard_i == 0:
            report["extended_references"] = self.capture_extended(py, columns, devices)
            bad = ["%s %s rc %s" % (r["column"], r["profile"], r["rc"]) for r in report["extended_references"]
                   if r["rc"] != 0]
            if bad:
                ok = False
                failed.extend("extended reference " + b for b in bad)
        report["ok"] = ok
        report["failures"] = failed
        if failed:
            self.say("FAILURES (%d chunk(s); every other chunk ran): %s" % (len(failed), "; ".join(failed)))
        (self.out / "identity-wheels.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
        self.say("%s: %s" % ("DONE" if ok else "FAILED", self.out / "identity-wheels.json"))
        self.stage_evidence(columns, results, labels, shard_i, shard_n)
        return 0 if ok else 1

    def timing(self, wall, cpu, nproc, results, labels, peak, per_device):
        """Print and return the record step's wall seconds, busy cores, chunk spread, 5 slowest lanes and the GPU's
        measured peak memory per worker."""
        secs = [r.get("seconds") for r in results.values() if r.get("seconds") is not None]
        logs = [p for c, i in results for p in (self.out / labels[c] / ("w%d" % i)).glob("*.log")
                if p.name != "record.log"]
        slow = sorted(lane_seconds(logs).items(), key=lambda kv: -kv[1])[:5]
        busy = cpu / wall if wall > 0 else 0.0
        self.say("timing: record %.0fs wall, busy cores mean %.1f of %d; chunks %s; slowest lanes: %s" % (
            wall, busy, nproc, "%d-%ds" % (min(secs), max(secs)) if secs else "none",
            ", ".join("%s %.0fs" % kv for kv in slow) or "none"))
        gpu = None
        if peak[0] is not None:
            base = peak[1] or 0.0
            gpu = dict(peak_used_gib=round(peak[0], 2), before_gib=round(base, 2), workers_per_device=per_device,
                       per_worker_gib=round(max(0.0, peak[0] - base) / per_device, 2))
            self.say("gpu memory: peak %.1f GiB used (%.1f before) with %d worker(s) on a device: %.2f GiB a worker "
                     "(GPU_GIB_PER_WORKER plans %d)" % (peak[0], base, per_device, gpu["per_worker_gib"],
                                                        GPU_GIB_PER_WORKER))
        rss = peak_rss_gib()
        host_memory = None
        if rss is not None:
            host_memory = dict(peak_rss_gib=round(rss, 2), planned_gib=round(worker_gib(next(iter(labels))), 2))
            self.say("host memory: peak RSS %.2f GiB of one worker process (plans %.2f GiB a worker; put the peak in "
                     "identity_wheel_record.py PEAKS)" % (rss, host_memory["planned_gib"]))
        # the column's own measured seconds, for tools/identity_record_seconds/<column>.json (COLUMN_HINTS)
        for column in labels:
            col_logs = [p for p in logs if p.parent.parent.name == labels[column]]
            means = cell_means(col_logs)
            if means:
                (self.out / ("record-seconds-%s.json" % column)).write_text(json.dumps(dict(
                    format="mojolearn.record-seconds.v1", column=column, label=labels[column],
                    note="mean seconds of one cell per lane, from this record's # CELL lines",
                    seconds={k: round(v, 3) for k, v in sorted(means.items())}), indent=1, sort_keys=True) + "\n")
        return dict(wall_s=round(wall), busy_cores_mean=round(busy, 1), cores=nproc,
                    chunk_seconds=sorted(secs), slowest_lanes=[dict(lane=k, seconds=round(v, 1)) for k, v in slow],
                    gpu_memory=gpu, host_memory=host_memory)

    def capture_extended(self, py, columns, devices):
        """The single-device verify commands' reference captures (tools/admit_extended_references.py), from the
        installed wheel in an empty directory, every capture at once (one process each, device 0: they are small and
        independent). One set per evidence column (the nvidia and ptx columns share the nvidia one)."""
        rows, procs = [], []
        empty = Path(tempfile.mkdtemp(prefix="identity-wheels-extended-"))
        try:
            for column in sorted({"nvidia" if c in ("nvidia", "ptx") else c for c in columns}):
                base = self.out / "extended-references" / column
                base.mkdir(parents=True, exist_ok=True)
                env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
                env["MOJOLEARN_NUMERIC_MODE"] = "identical"
                if devices:
                    var = COLUMNS["nvidia" if column == "nvidia" else column][1]
                    if var:
                        env[var] = str(devices[0])
                for profile, argv in EXTENDED_COMMANDS.items():
                    out = base / (profile + ".json")
                    if out.exists():
                        out.unlink()
                    log = open(base / (profile + ".log"), "w")
                    t0, started = time.monotonic(), utc()
                    procs.append((column, profile, out, log, t0, started, self.popen(
                        [str(py), "-I", "-m", "mojolearn", *argv(column, str(out))], cwd=str(empty), env=env,
                        stdout=log, stderr=subprocess.STDOUT)))
            for column, profile, out, log, t0, started, proc in procs:
                rc = proc.wait()
                log.close()
                rows.append(dict(column=column, profile=profile, rc=rc if out.is_file() else (rc or 1),
                                 report=str(out), seconds=round(time.monotonic() - t0), start=started, end=utc()))
                self.say("extended reference %s %s: rc=%s" % (column, profile, rows[-1]["rc"]))
        finally:
            shutil.rmtree(empty, ignore_errors=True)
        return rows

    def stage_evidence(self, columns, results, labels, shard_i, shard_n):
        """The record part of the in-repo evidence (tools/verification_evidence.py): one per evidence column (the
        nvidia and ptx columns are one nvidia part). Staging never changes the run's exit status."""
        import verification_evidence as E
        rows = ["%s-w%d\t%s\t%s\t%s\t%s\t" % (labels[c], i, r.get("rc"), r.get("seconds", ""), r.get("start", ""),
                                                 r.get("end", "")) for (c, i), r in sorted(results.items())]
        (self.out / "steps.tsv").write_text("".join(row + "\n" for row in rows))
        part = "record" if shard_n == 1 else "record-shard-%d-of-%d" % (shard_i, shard_n)
        for column in sorted({"nvidia" if c in ("nvidia", "ptx") else c for c in columns}):
            try:
                E.stage(self.out, self.out / "evidence" / column / part, column, part)
                self.say("evidence staged: %s" % (self.out / "evidence" / column / part))
            except (Exception, SystemExit) as exc:   # evidence never fails the record
                self.say("evidence NOT staged for %s: %s" % (column, exc))


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--columns", required=True, help="comma list of nvidia, ptx, amd, apple, cpu")
    ap.add_argument("--label-device", default="", help="the device slug in the labels (l40s, mi325x, m2pro)")
    ap.add_argument("--wheel", action="append", required=True)
    ap.add_argument("--sha256", action="append", required=True, help="<wheel file name>=<sha256>, one per wheel")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", type=int, default=0, help="devices on this box (0: cpu or apple)")
    ap.add_argument("--workers-per-gpu", default="auto", help="auto (cores and measured memory) or a number")
    ap.add_argument("--venv", default="")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--shard", default="0/1", help="i/N: record every N-th owed lane from i (a CI job matrix)")
    ap.add_argument("--record-families", default="all",
                    help="the release families whose lanes record (tools/release_families.py; tools/release.py passes "
                         "the families its plan records): a comma list, `all` (default: every lane) or `none`")
    a = ap.parse_args(argv)
    m = re.fullmatch(r"(\d+)/(\d+)", a.shard)
    if not m or not 0 <= int(m.group(1)) < int(m.group(2)):
        ap.error("--shard is i/N with 0 <= i < N")
    if a.workers_per_gpu != "auto" and not a.workers_per_gpu.isdigit():
        ap.error("--workers-per-gpu is auto or a positive number")
    return a


if __name__ == "__main__":
    sys.exit(Run(parse()).main())
