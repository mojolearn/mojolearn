# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""The box's host-memory state beside a timed run, and a timed run under a chosen memory placement.

Why (lane fb3-drift, 2026-11-17): the same wheel timed twice on the same boot of the L40S pod was slower the
second time on exactly the rows scored by ONE call (the first call after the warmup in a fresh process: fresh
pages, pageable host-to-device copies), by the same milliseconds as the harness's own separate upload probe
(wheel independent: istella X 85 ms in three runs, 113..126 ms in the slow one), in two time windows; rows
scored as the median of 20 calls did not move. That is the host's memory path (huge pages available or not,
which NUMA node the process landed on, memory bandwidth), which no job recorded. This tool records it and
reproduces it on purpose:

    mem_state.py snapshot
        one JSON line: transparent-huge-page mode, free memory and free 2 MB-or-larger blocks per NUMA node
        (/proc/buddyinfo), the kernel's huge-page fault counters (thp_fault_alloc / thp_fault_fallback: the
        difference of two snapshots around a run says whether its fresh pages came as huge pages), NUMA hit and
        miss counters, the GPU's NUMA node and this process's allowed CPUs per node.
    mem_state.py exec [--node near|far|any|N] [--thp on|off] -- command...
        run the command (and every child) with its CPUs limited to one NUMA node (`near`: the GPU's node; `far`:
        another node; memory follows by first touch, so no mempolicy call a container may refuse) and/or with
        transparent huge pages off for the process tree (prctl PR_SET_THP_DISABLE, inherited over fork and exec).

tools/ops/attribution_job.sh logs a snapshot before and after each timing run and takes `exec` as AT_TIME_PREFIX.
Diagnostic tooling, never product runtime. Nothing here is a workload: `snapshot` reads a few /proc and /sys
files once and `exec` replaces itself with the command, so there is no unit to spread over cores.
"""
import argparse
import ctypes
import glob
import json
import os
import subprocess
import sys

PR_SET_THP_DISABLE = 41
VMSTAT_KEYS = ("thp_fault_alloc", "thp_fault_fallback", "thp_collapse_alloc", "compact_stall", "pgfault",
               "numa_hit", "numa_miss", "numa_foreign", "numa_local", "numa_other")
MEMINFO_KEYS = ("MemFree", "MemAvailable", "Cached", "AnonHugePages")
HUGE_ORDER = 9      # a 2 MB block is 2**9 pages of 4 KiB


def _read(path):
    try:
        with open(path) as fh:
            return fh.read()
    except OSError:
        return None


def _bracketed(text):
    """The selected word of a sysfs choice file (`always [madvise] never` -> `madvise`)."""
    if not text:
        return None
    for word in text.split():
        if word.startswith("[") and word.endswith("]"):
            return word[1:-1]
    return text.strip()


def parse_cpulist(text):
    """{cpu} of a kernel cpulist (`0-63,128-191`)."""
    cpus = set()
    for part in (text or "").strip().split(","):
        if not part:
            continue
        lo, _, hi = part.partition("-")
        cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def node_cpus():
    """{node: {cpu}} from sysfs."""
    out = {}
    for d in glob.glob("/sys/devices/system/node/node[0-9]*"):
        text = _read(os.path.join(d, "cpulist"))
        if text is not None:
            out[int(os.path.basename(d)[4:])] = parse_cpulist(text)
    return out


def gpu_node():
    """The NUMA node of GPU 0 (its PCI device's numa_node), or None when the box does not say (-1 or no sysfs)."""
    try:
        bus = subprocess.run(["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader", "-i", "0"],
                             capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[0].strip()
    except (OSError, IndexError, subprocess.SubprocessError):
        return None
    # nvidia-smi prints an 8-digit PCI domain (00000000:81:00.0); sysfs names the device with 4 (0000:81:00.0)
    dom, _, rest = bus.lower().partition(":")
    text = _read("/sys/bus/pci/devices/%s:%s/numa_node" % (dom[-4:], rest))
    try:
        node = int(text)
    except (TypeError, ValueError):
        return None
    return node if node >= 0 else None


def huge_free_gb():
    """{node: GiB free in blocks of 2 MB or larger} from /proc/buddyinfo (every zone of the node)."""
    out = {}
    for line in (_read("/proc/buddyinfo") or "").splitlines():
        f = line.replace(",", " ").split()
        if len(f) < 5 or f[0] != "Node":
            continue
        counts = [int(x) for x in f[4:]]
        pages = sum(c << order for order, c in enumerate(counts) if order >= HUGE_ORDER)
        out[int(f[1])] = out.get(int(f[1]), 0.0) + pages * 4096 / 2.0 ** 30
    return {k: round(v, 1) for k, v in out.items()}


def node_free_gb():
    out = {}
    for d in glob.glob("/sys/devices/system/node/node[0-9]*"):
        for line in (_read(os.path.join(d, "meminfo")) or "").splitlines():
            f = line.split()
            if len(f) >= 4 and f[2] == "MemFree:":
                out[int(f[1])] = round(int(f[3]) / 2.0 ** 20, 1)
    return out


def snapshot():
    vm = {}
    for line in (_read("/proc/vmstat") or "").splitlines():
        k, _, v = line.partition(" ")
        if k in VMSTAT_KEYS:
            vm[k] = int(v)
    mem = {}
    for line in (_read("/proc/meminfo") or "").splitlines():
        k, _, v = line.partition(":")
        if k in MEMINFO_KEYS:
            mem[k + "_gb"] = round(int(v.split()[0]) / 2.0 ** 20, 1)
    allowed = os.sched_getaffinity(0)
    return {
        "thp": _bracketed(_read("/sys/kernel/mm/transparent_hugepage/enabled")),
        "thp_defrag": _bracketed(_read("/sys/kernel/mm/transparent_hugepage/defrag")),
        "meminfo": mem,
        "vmstat": vm,
        "node_free_gb": node_free_gb(),
        "node_huge_free_gb": huge_free_gb(),
        "gpu_node": gpu_node(),
        "allowed_cpus_per_node": {n: len(c & allowed) for n, c in sorted(node_cpus().items())},
        "boot_id": (_read("/proc/sys/kernel/random/boot_id") or "").strip() or None,
    }


def pick_node(want, gpu, nodes):
    """The node number for `near`, `far` or a number; refuses with the reason when the box cannot say."""
    if want.lstrip("-").isdigit():
        node = int(want)
        if node not in nodes:
            raise SystemExit("mem_state: no NUMA node %d (nodes %s)" % (node, sorted(nodes)))
        return node
    if gpu is None:
        raise SystemExit("mem_state: the box does not say the GPU's NUMA node; name a node number (--node N)")
    if want == "near":
        return gpu
    others = sorted(n for n in nodes if n != gpu)
    if not others:
        raise SystemExit("mem_state: one NUMA node only; there is no far node")
    return others[0]


def run_exec(node, thp, command):
    if not command:
        raise SystemExit("mem_state exec: no command after --")
    note = []
    if node != "any":
        nodes = node_cpus()
        chosen = pick_node(node, gpu_node(), nodes)
        cpus = nodes[chosen] & os.sched_getaffinity(0)
        if not cpus:
            raise SystemExit("mem_state: none of node %d's CPUs is allowed to this process" % chosen)
        os.sched_setaffinity(0, cpus)
        note.append("cpus of node %d (%d)" % (chosen, len(cpus)))
    if thp == "off":
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0) != 0:
            raise SystemExit("mem_state: prctl(PR_SET_THP_DISABLE) failed: errno %d" % ctypes.get_errno())
        note.append("huge pages off")
    print("mem_state exec: %s: %s" % (", ".join(note) or "unchanged", " ".join(command[:3])), file=sys.stderr,
          flush=True)
    os.execvp(command[0], command)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("snapshot")
    e = sub.add_parser("exec")
    e.add_argument("--node", default="any", help="near (the GPU's NUMA node), far (another node), any, or a number")
    e.add_argument("--thp", default="on", choices=("on", "off"))
    e.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args(argv)
    if a.cmd == "snapshot":
        print(json.dumps(snapshot(), sort_keys=True, separators=(",", ":")))
        return 0
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    run_exec(a.node, a.thp, command)
    return 0


if __name__ == "__main__":
    sys.exit(main())
