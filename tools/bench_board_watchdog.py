# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE HOST-MEMORY WATCHDOG of the bench board (tools/bench_board.py run_logged).

2026-09-29, do-amd (157 GB): scikit-learn's DBSCAN arm on 1M taxi rows at
eps=3 grew to 157 GB, the kernel OOM-killed it, and systemd stopped the whole
steward service with it (the board, the steward and every other job). On a
Mac the same arm swaps instead of dying. So every driver the board starts is
watched: its process tree's memory (every descendant, also one that started
its own session, found by parent pid) is sampled twice a second, and when it
passes `limit_frac` of the box's RAM (default 0.9, MOJOLEARN_BOARD_MEM_LIMIT
overrides), or on Linux the box's MemAvailable falls under 3% of RAM, the
heaviest process of the tree (the arm that is growing) is killed with SIGKILL
and recorded by name. The driver sees its arm die and goes on; the board
marks the race failed with HOST MEMORY and the arm's cell HOST-MEMORY(...).

Per-process memory: Linux, VmRSS from /proc/<pid>/status; macOS, the
process's physical footprint (proc_pid_rusage ri_phys_footprint, which counts
compressed and swapped pages, so a swapping arm is still seen), else ps rss.
Standard library only.
"""
import os
import signal
import subprocess
import sys
import threading
import time

LIMIT_ENV = "MOJOLEARN_BOARD_MEM_LIMIT"
_MB = 1024.0 * 1024.0


def total_ram_bytes():
    if sys.platform == "darwin":
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout
        return int(out.strip())
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    raise OSError("no MemTotal in /proc/meminfo")


def mem_available_bytes():
    """Linux MemAvailable, None elsewhere."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


class _Footprint(object):
    """macOS proc_pid_rusage(RUSAGE_INFO_V4).ri_phys_footprint of any pid."""
    FOOTPRINT = 7                    # uint64 index after the 16-byte uuid (sys/resource.h)

    def __init__(self):
        import ctypes
        import ctypes.util
        self.ct = ctypes
        self.lib = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libSystem.B.dylib")
        self.lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        self.lib.proc_pid_rusage.restype = ctypes.c_int

        class Info(ctypes.Structure):
            _fields_ = [("uuid", ctypes.c_uint8 * 16), ("v", ctypes.c_uint64 * 40)]
        self.Info = Info

    def bytes(self, pid):
        info = self.Info()
        if self.lib.proc_pid_rusage(int(pid), 4, self.ct.byref(info)) != 0:
            return None
        return int(info.v[self.FOOTPRINT])


def processes():
    """{pid: (ppid, bytes, command)} of every live process this user can see."""
    out = {}
    if os.path.isdir("/proc") and sys.platform != "darwin":
        for d in os.listdir("/proc"):
            if not d.isdigit():
                continue
            try:
                with open("/proc/%s/status" % d) as fh:
                    st = fh.read()
                with open("/proc/%s/cmdline" % d, "rb") as fh:
                    cmd = fh.read().replace(b"\0", b" ").decode(errors="replace").strip()
            except OSError:
                continue
            ppid, rss = None, 0
            for line in st.splitlines():
                if line.startswith("PPid:"):
                    ppid = int(line.split()[1])
                elif line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) * 1024
                elif line.startswith("State:") and " Z" in line[6:9]:
                    ppid = None
                    break
            if ppid is not None:
                out[int(d)] = (ppid, rss, cmd)
        return out
    ps = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,rss=,stat=,command="],
                        capture_output=True, text=True).stdout
    fp = _FOOTPRINT
    for line in ps.splitlines():
        f = line.split(None, 4)
        if len(f) < 5 or f[3].startswith("Z"):
            continue
        pid = int(f[0])
        b = fp.bytes(pid) if fp is not None else None
        out[pid] = (int(f[1]), b if b is not None else int(f[2]) * 1024, f[4])
    return out


try:
    _FOOTPRINT = _Footprint() if sys.platform == "darwin" else None
except Exception:                                  # noqa: BLE001
    _FOOTPRINT = None


def tree(root, procs):
    """root and every descendant of it in `procs` (by parent pid)."""
    kids = {}
    for pid, (ppid, _, _) in procs.items():
        kids.setdefault(ppid, []).append(pid)
    seen, todo = [], [root]
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        if p in procs:
            seen.append(p)
        todo.extend(kids.get(p, []))
    return seen


class HostMemoryWatchdog(object):
    """Watch the process tree under `root_pid` until stop(); `kills` lists
    what it killed: {pid, command, mb, tree_mb, limit_mb, ram_mb, why}."""

    def __init__(self, root_pid, log=None, limit_frac=None, interval=0.5, ram=None,
                 sampler=processes, available=mem_available_bytes):
        frac = limit_frac if limit_frac is not None else float(os.environ.get(LIMIT_ENV) or 0.9)
        self.root = root_pid
        self.log = log
        self.ram = ram if ram is not None else total_ram_bytes()
        self.limit = frac * self.ram
        self.interval = interval
        self.sampler = sampler
        self.available = available
        self.kills = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._t.start()
        return self

    def stop(self):
        self._stop.set()
        self._t.join(timeout=10)
        return self.kills

    def check_once(self):
        procs = self.sampler()
        pids = tree(self.root, procs)
        if not pids:
            return None
        total = sum(procs[p][1] for p in pids)
        why = None
        if total > self.limit:
            why = "the driver's process tree held %.1f GB, over %.0f%% of the box's %.1f GB" % (
                total / 1e9, 100.0 * self.limit / self.ram, self.ram / 1e9)
        else:
            avail = self.available() if self.available else None
            if avail is not None and avail < 0.03 * self.ram:
                why = "the box had %.1f GB available, under 3%% of its %.1f GB" % (
                    avail / 1e9, self.ram / 1e9)
        if why is None:
            return None
        # the heaviest process of the tree, never the driver's root while a child is heavier
        victim = max(pids, key=lambda p: procs[p][1])
        rec = {"pid": victim, "command": procs[victim][2][:400], "mb": round(procs[victim][1] / _MB, 1),
               "tree_mb": round(total / _MB, 1), "limit_mb": round(self.limit / _MB, 1),
               "ram_mb": round(self.ram / _MB, 1), "why": why,
               "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        try:
            os.kill(victim, signal.SIGKILL)
        except OSError as exc:
            rec["kill_error"] = repr(exc)
        self.kills.append(rec)
        if self.log is not None:
            try:
                self.log.write("\n=== bench_board: HOST MEMORY: killed pid %d (%.1f GB): %s; %s\n"
                               % (victim, rec["mb"] / 1024.0, why, rec["command"][:200]))
                self.log.flush()
            except (OSError, ValueError):
                pass
        return rec

    def _run(self):
        while not self._stop.wait(self.interval):
            try:
                if self.check_once() is not None:
                    self._stop.wait(2.0)      # let the kernel reclaim before sampling again
            except Exception:                 # noqa: BLE001 - a failed sample never stops the race
                pass


def arm_of(command, arms):
    """The race arm a killed command belongs to: the drivers' `--arm <name>`,
    else the longest arm name that appears in it as a word, else None (the
    trees driver runs every arm in one process: the race is failed whole)."""
    import re
    m = re.search(r"--arm[ =](\S+)", command)        # the drivers' worker processes: --arm <name>
    if m and m.group(1) in (arms or ()):
        return m.group(1)
    hit = None
    for a in arms or ():
        if re.search(r"(^|[\s=/,])%s($|[\s,.\-])" % re.escape(a), command) and (hit is None or len(a) > len(hit)):
            hit = a
    return hit
