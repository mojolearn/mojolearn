#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""tools/ops/call_profile.py -- the plan, the parsers and the report of tools/ops/call_profile_job.sh: WHERE THE WALL
TIME OF ONE SMALL PUBLIC CALL GOES, at function level, host side and GPU side, on OUR wheel alone (diagnostic tooling,
not a board and not a measurement).

Per row the job runs the row's race once per pass, each a forced scored run whose extra calls (2..n) sit inside the
harness's profile window (tools/bench_board_small_rows.py range_open/range_close) and whose n fills the pass's
seconds at the first scored call's time:
    time       nothing attached: wall per call (the window's wall clock over its calls)
    pyspy      `py-spy record --native --threads --pid <race>` attached by the harness for the window only
               (permission-free: the race lets its child trace it, prctl PR_SET_PTRACER_ANY)
    perf       `perf record -e cpu-clock --call-graph dwarf -D -1 --control fifo:...` (only where the container allows
               perf events; RunPod's perf_event_paranoid=4 does not)
    nsys       `nsys profile -t cuda,nvtx --capture-range=nvtx`: launches, syncs, allocations, copies, API and kernel ms
    shim       LD_PRELOAD tools/ops/call_profile_shim.c: count, ns and bytes per call of the CUDA driver entries,
               libc allocations, mmap/munmap and memcpy/memset of 64 KiB or more (no ptrace, no perf)
    callgrind  `valgrind --tool=callgrind --instr-atstart=no --collect-atstart=no`, instrumentation and collection on
               for the window only (tools/ops/call_profile_cg.c): exact instructions per call by function
    strace     `strace -f -ttt -T` cut to the window by the marker stats: syscalls per call (opt-in)
LONG ROWS (lane fx-profile-big-rows, 2026-11-20): a rows-file line may carry `ms=<stored board ms>` and `cap=<s>`. A
row of LONG_ROW_MS or more is profiled in ONE call per pass (no warmup; the window around the first scored call;
tools/bench_board_small_rows.py one_call), with no time pass and no callgrind, each pass under the row's cap; a pass the
job stops at its cap leaves run.capped and the row is reported CAPPED with what its tools collected
(exit 0: a capped pass is a reported state, never a report failure; nvc1-0005, 2026-11-20, lost its job to exit 8). The
report adds, per long row: stored ms beside the one call's wall, GPU kernel / CUDA API (of it sync) / host ms, launches,
syncs, H2D/D2H MB, the top GPU kernels, and one LARGEST COST line (tools/ops/call_profile_rows_big.tsv).
Subcommands (each prints one line; the job keeps their files):
    rows       --rows-file F --out-dir D      D/runs.tsv (row, race id, phase, cap s or 0, long 1/0) and D/ids.txt
    estimate   --rows-file F [--passes ...]   pod minutes per row and in all (printed by the job's plan phase); for
                                              long rows also stored ms x passes and the sum of the caps
    window     --window W                     calls and wall of a pass's window (JSON)
    perf       --script S --window W --out J --folded F      folded stacks and the top self/inclusive frames
    pyspy      --raw C --window W --out J [--rate R]         the same from py-spy's raw stacks (main thread)
    callgrind  --dir D --window W --out J                    instructions per call by function, self and inclusive
    strace     --trace T --window W --out J                  syscalls per call in the window
    report     --res R [--rows-file F] [--top N]   CALL-PROFILE-<vendor>.md and .json; exit 0 every row has its wall,
                                              nsys counts and one function-level source (py-spy, perf, callgrind or the
                                              shim), 6 a row lacks one of those, 7 a row lacks a wall time; a row that
                                              hit its cap is CAPPED in the report, exit 0
"""
import argparse
import collections
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROWS_FILE = os.path.join(HERE, "call_profile_rows.tsv")
PASSES = ("time", "pyspy", "perf", "nsys", "shim", "callgrind", "strace")
#: the job's default passes: perf needs perf events (refused in RunPod's containers) and strace is opt-in (the shim
#: carries the allocation and mmap counts; strace only adds the other syscalls)
DEFAULT_PASSES = ("time", "pyspy", "nsys", "shim", "callgrind")
#: seconds each pass's window runs (CP_<PASS>_SECONDS in the job); py-spy and perf want >= 10 s of samples, nsys
#: counts, the shim's counts and the strace counts are per call and settle in a few seconds; callgrind's calls are
#: sized on the first call under valgrind's core (uninstrumented) and capped by CP_CALLGRIND_CALLS: its window runs
#: about CALLGRIND_WINDOW_SLOWDOWN times longer once instrumented
PASS_SECONDS = {"time": 3.0, "pyspy": 10.0, "perf": 10.0, "nsys": 3.0, "shim": 3.0, "callgrind": 10.0, "strace": 3.0}
CALLGRIND_WINDOW_SLOWDOWN = 6.0
#: the wall of one race process outside its window (start-up, data, fit before an inference, the first calls), and
#: each tool's start-up slowdown on it: the TG board of d652a4b0a1 ran the default rows' races in 8-30 s (mean 14 s)
#: with 21 calls of each phase; bench_board's own start-up added. callgrind: every process of the race under
#: valgrind's core without instrumentation (about 7x for Python start-up and imports)
OVERHEAD_S = 18.0
PASS_SLOWDOWN = {"time": 1.0, "pyspy": 1.0, "perf": 1.4, "nsys": 1.8, "shim": 1.1, "callgrind": 7.0, "strace": 2.5}
#: callgrind passes of different rows run at once (instruction counts do not depend on what else runs), this many
#: assumed for the estimate (the job takes min(rows, cores, memory / CP_CALLGRIND_GB))
CALLGRIND_PARALLEL = 8
SETUP_S = 240.0         # tool installs (in parallel) and the post-processing (in parallel over the cores)
TOP = 40
EXIT_OK, EXIT_NO_PROFILE, EXIT_NO_WALL = 0, 6, 7
STRACE_MARK_OPEN = "/callprof-window-open"
STRACE_MARK_CLOSE = "/callprof-window-close"
#: frames that only carry the call down to the work (interpreter entry, thread start): never named as a host cost
GENERIC_FRAME = re.compile(r"^(_start|__libc_start|main|Py_|_Py|PyEval|PyObject_(Vectorcall|Call)|pymain|"
                           r"start_thread|clone3?|__clone|\[unknown\]|_PyFunction|method_vectorcall|cfunction_|"
                           r"thread \(|<module>|0x[0-9a-f]+$)")
#: the shim's CUDA driver entries and its libc entries (tools/ops/call_profile_shim.c, in its order)
SHIM_LIBC = ("malloc", "calloc", "realloc", "free", "aligned", "mmap", "munmap", "memcpy_64k", "memset_64k")
#: where a frame's time goes in the report's split, by the object (shared library or Python file) it runs in
CATEGORIES = ("bindings", "cpython", "mojo_runtime", "cuda_driver", "libc", "other", "unresolved")
_CAT_RULES = (
    ("bindings", re.compile(r"(^|/)_mojolearn[^/]*\.so")),
    ("cuda_driver", re.compile(r"(^|/)lib(cuda|nvidia-[a-z]+|cudart)[^/]*\.so")),
    ("mojo_runtime", re.compile(r"(^|/)lib(KGENCompilerRT|AsyncRT|MSupport|MGP|Mojo|modular|M[A-Z][A-Za-z]*Globals)"
                                r"[^/]*\.so")),
    ("cpython", re.compile(r"(^|/)(libpython[0-9.]*\.so|python[0-9.]*$)|\.py$|^<[a-z_]+")),
    ("libc", re.compile(r"(^|/)(libc|libm|libpthread|libdl|librt|ld-linux[^/]*|libstdc\+\+|libgcc_s)[.-][^/]*")),
)
#: libc leaf functions the shim already times (left out of the sampled split so they are not counted twice)
_SHIM_LEAF = re.compile(r"^(__libc_)?(malloc|calloc|realloc|free|memalign|aligned_alloc|posix_memalign|mmap(64)?|"
                        r"munmap|__memcpy|memcpy|memmove|memset|__memset)[_a-z0-9]*$")


def category(obj):
    """The report category of a frame's object: a shared library path or name, or a Python source file."""
    o = (obj or "").strip()
    if not o or o in ("?", "???"):
        return "unresolved"
    for name, rx in _CAT_RULES:
        if rx.search(o):
            return name
    return "other"


#: LONG ROWS (lane fx-profile-big-rows, 2026-11-20): a row given `ms=<stored board ms>` whose call takes at least
#: this long is profiled in ONE call per pass (the job's one-call mode, tools/bench_board_small_rows.py one_call): no
#: warmup, the window around the first scored call, no window seconds to fill
LONG_ROW_MS = 10000.0
#: a long row's own cap when its line gives no `cap=`: its stored seconds times CAP_FACTOR plus CAP_PAD_S, per pass
CAP_FACTOR, CAP_PAD_S = 2.0, 300.0
#: each tool's slowdown of ONE long call (the estimate): py-spy at a low rate and the shim cost little per second of
#: a call that runs kernels for minutes; nsys traces every CUDA API call
LONG_CALL_SLOWDOWN = {"time": 1.0, "pyspy": 1.05, "perf": 1.2, "nsys": 1.3, "shim": 1.05, "callgrind": 50.0,
                      "strace": 1.5}


class Row(object):
    def __init__(self, name, race_id, phase, ms=None, cap_s=None):
        self.name, self.race_id, self.phase = name, race_id, phase
        self.ms, self.cap_s = ms, cap_s

    @property
    def long(self):
        """True for a row profiled in one call per pass (its stored ms at or above LONG_ROW_MS)."""
        return self.ms is not None and self.ms >= LONG_ROW_MS

    def cap(self):
        """The row's per-pass cap in seconds (its `cap=`, else from its stored ms for a long row), or None."""
        if self.cap_s:
            return self.cap_s
        if self.long:
            return int(math.ceil(self.ms / 1000.0 * CAP_FACTOR + CAP_PAD_S))
        return None


def parse_rows(text, where="rows file"):
    """[Row] from a rows file (race id, phase, optional name=, ms=<stored board ms>, cap=<seconds a pass may run>);
    a malformed line or a repeated name is refused."""
    rows, names, seen = [], set(), collections.Counter()
    for i, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        f = line.split()
        # fit+infer (lane fx-harness, 2026-11-18): board order, each fit then the first predict right after it
        if len(f) < 2 or f[0].count("/") < 2 or not (f[1] in ("fit", "fit+infer") or f[1].startswith("infer")):
            raise ValueError("%s line %d: want <race id> <fit|infer[/<batch>]|fit+infer> [name=]: %r" % (where, i, raw))
        name, ms, cap = None, None, None
        for opt in f[2:]:
            if opt.startswith("name="):
                name = opt[5:]
            elif opt.startswith("ms=") or opt.startswith("cap="):
                k, _, v = opt.partition("=")
                try:
                    x = float(v)
                except ValueError:
                    x = -1.0
                if not (x > 0 and math.isfinite(x)):
                    raise ValueError("%s line %d: %s= wants a positive number: %r" % (where, i, k, raw))
                if k == "ms":
                    ms = x
                else:
                    cap = int(math.ceil(x))
            else:
                raise ValueError("%s line %d: unknown field %r" % (where, i, opt))
        seen[f[0]] += 1
        if not name:
            name = f[0].split("/")[1] + ("-" + f[1].replace("/", "-").replace("+", "-") if seen[f[0]] > 1 else "")
        if not re.match(r"^[A-Za-z0-9._-]+$", name) or name in names:
            raise ValueError("%s line %d: row name %r is repeated or not a plain name (give name=)" % (where, i, name))
        names.add(name)
        if ms is not None and ms >= LONG_ROW_MS and f[1] == "fit+infer":
            raise ValueError("%s line %d: a long row (ms >= %d) is profiled in one call of fit or infer, not fit+infer"
                             % (where, i, LONG_ROW_MS))
        rows.append(Row(name, f[0], f[1], ms, cap))
    if not rows:
        raise ValueError("%s: no rows" % where)
    return rows


def load_rows(path=None):
    path = path or DEFAULT_ROWS_FILE
    with open(path) as fh:
        return parse_rows(fh.read(), path)


def estimate(rows, passes=DEFAULT_PASSES, seconds=None, overhead_s=OVERHEAD_S, setup_s=SETUP_S,
             callgrind_parallel=CALLGRIND_PARALLEL):
    """{rows: [{row, seconds, pod_min}], total_min, assumptions}: one race process per row and pass; the passes on
    the GPU run one at a time, the callgrind passes of CALLGRIND_PARALLEL rows at once."""
    secs = dict(PASS_SECONDS, **(seconds or {}))
    serial = [p for p in passes if p != "callgrind"]
    per = []
    for r in rows:
        if r.long:
            # one call per pass, no warmup: the stored call under each tool's slowdown, beside the process overhead;
            # the cap bounds a pass (a capped pass ends there)
            one = [min(overhead_s * PASS_SLOWDOWN[p] + r.ms / 1000.0 * LONG_CALL_SLOWDOWN[p], float(r.cap() or 1e18))
                   for p in serial if p != "time"]          # no time pass on a long row (the shim's wall stands)
            s = sum(one)
            per.append({"row": r.name, "seconds": round(s, 1), "pod_min": round(s / 60.0, 1), "long": True,
                        "stored_ms": r.ms, "cap_s": r.cap()})
        else:
            s = sum(overhead_s * PASS_SLOWDOWN[p] + secs[p] for p in serial)
            per.append({"row": r.name, "seconds": round(s, 1), "pod_min": round(s / 60.0, 1)})
    cg = 0.0
    short = [r for r in rows if not r.long]          # no callgrind on a long row (the job skips it)
    if "callgrind" in passes and short:
        cg = math.ceil(len(short) / float(max(callgrind_parallel, 1))) * (
            overhead_s * PASS_SLOWDOWN["callgrind"] + secs["callgrind"] * CALLGRIND_WINDOW_SLOWDOWN)
    total = sum(p["seconds"] for p in per) + cg + setup_s
    stored = sum(r.ms for r in rows if r.long) / 1000.0
    return {"rows": per, "callgrind_s": round(cg, 1), "total_min": round(total / 60.0, 1), "passes": list(passes),
            "long_rows": sum(1 for r in rows if r.long), "long_stored_s": round(stored, 1),
            "long_passes": [p for p in serial if p != "time"],
            "long_stored_x_passes_min": round(stored * len([p for p in serial if p != "time"]) / 60.0, 1),
            "cap_total_min": round(sum(float(r.cap() or 0) for r in rows if r.long) *
                                   len([p for p in serial if p != "time"]) / 60.0, 1),
            "assumptions": {"overhead_s_per_race_process": overhead_s, "slowdown": PASS_SLOWDOWN,
                            "long_call_slowdown": LONG_CALL_SLOWDOWN,
                            "window_seconds": secs, "setup_and_postprocess_s": setup_s,
                            "callgrind_parallel": callgrind_parallel}}


# ---- the window of one pass --------------------------------------------------------------------------------------

def shim_per_call(o, c, calls):
    """{entry: {calls, us, bytes}} per call of the window's shim counters (open and close), entries with calls."""
    if not (calls > 0 and isinstance(o, dict) and isinstance(c, dict)):
        return None
    out = {}
    for k, v in c.items():
        v0 = o.get(k) or [0, 0, 0]
        n, ns, b = (float(v[i]) - float(v0[i]) for i in range(3))
        if n > 0:
            out[k] = {"calls": n / calls, "us": ns / 1e3 / calls, "bytes": b / calls}
    return out


def read_window(path):
    """{calls, wall_ms_per_call, wall_s, shim_per_call, perf_enabled, pyspy, callgrind, windows} of a pass's window
    file (the first open/close pair; `windows` counts the pairs). Missing file or no close: calls 0."""
    out = {"calls": 0, "wall_ms_per_call": None, "wall_s": None, "shim_per_call": None, "perf_enabled": None,
           "pyspy": None, "pyspy_error": None, "callgrind": None, "pid": None, "windows": 0, "opened_unix": None}
    if not path or not os.path.isfile(path):
        return out
    opened = None
    pairs = []
    with open(path, errors="replace") as fh:
        for ln in fh:
            try:
                rec = json.loads(ln)
            except ValueError:
                continue
            if rec.get("event") == "open-mark":       # a one-call run writes its open at the open (a capped run)
                if out["opened_unix"] is None:
                    out["opened_unix"] = rec.get("t_unix")
                    out["pid"] = rec.get("pid")
            elif rec.get("event") == "open":
                opened = rec
            elif rec.get("event") == "close" and opened is not None:
                pairs.append((opened, rec))
                opened = None
    out["windows"] = len(pairs)
    if not pairs:
        return out
    o, c = pairs[0]
    calls = int(c.get("calls") or 0)
    out["calls"] = calls
    out["pid"] = o.get("pid")
    if calls > 0 and c.get("wall_ns") is not None:
        out["wall_s"] = c["wall_ns"] / 1e9
        out["wall_ms_per_call"] = c["wall_ns"] / 1e6 / calls
    out["perf_enabled"] = bool(o.get("perf")) and bool(c.get("perf"))
    out["pyspy"] = bool(o.get("pyspy")) if "pyspy" in o else None
    out["pyspy_error"] = o.get("pyspy_error") or (None if c.get("pyspy_exit") in (None, 0) else
                                                  "py-spy exit %s" % c.get("pyspy_exit"))
    out["callgrind"] = c.get("callgrind")
    out["shim_per_call"] = shim_per_call(o.get("shim"), c.get("shim"), calls)
    return out


# ---- perf script -> folded stacks and top frames -----------------------------------------------------------------

_FRAME = re.compile(r"^\s+([0-9a-fA-F]+)\s+(.*?)\s+\(([^()]*)\)\s*$")


def frame_name(sym, dso):
    sym = (sym or "").strip() or "[unknown]"
    sym = re.sub(r"\+0x[0-9a-f]+$", "", sym)
    base = os.path.basename(dso or "") or "?"
    return "%s [%s]" % (sym, base)


def perf_samples(lines, skip_comm=("perf", "py-spy")):
    """Yield (comm, period_ns, [frames root..leaf]) from `perf script -F comm,tid,period,ip,sym,dso` text."""
    head, frames = None, []
    for raw in lines:
        line = raw.rstrip("\n")
        if not line.strip():
            if head is not None:
                yield head[0], head[1], list(reversed(frames))
            head, frames = None, []
            continue
        if line[0] not in " \t":
            if head is not None:
                yield head[0], head[1], list(reversed(frames))
            toks = line.replace(":", " ").split()
            nums = [t for t in toks if t.isdigit()]
            period = int(nums[-1]) if len(nums) >= 2 else 1
            comm = toks[0] if toks else "?"
            head, frames = (comm, period), []
            continue
        m = _FRAME.match(line)
        if m and head is not None:
            frames.append(frame_name(m.group(2), m.group(3)))
    if head is not None:
        yield head[0], head[1], list(reversed(frames))


def top_frames(stacks, calls, top=TOP, unit_ns=1.0):
    """{samples, cpu_ms_per_call, self: [...], inclusive: [...]} from [(weight, [frames root..leaf])]; weight is
    nanoseconds when unit_ns == 1 (perf periods) or samples times unit_ns (py-spy at a fixed rate)."""
    total = 0.0
    nsamp = 0
    selfc, incl = collections.Counter(), collections.Counter()
    for w, frames in stacks:
        if not frames:
            continue
        nsamp += 1
        total += w
        selfc[frames[-1]] += w
        for f in dict.fromkeys(frames):
            incl[f] += w
    calls = max(int(calls or 0), 1)

    def rows(counter):
        return [{"frame": f, "pct": round(100.0 * w / total, 2) if total else 0.0,
                 "us_per_call": round(w * unit_ns / 1e3 / calls, 2)} for f, w in counter.most_common(top)]
    return {"samples": nsamp, "cpu_ms_per_call": round(total * unit_ns / 1e6 / calls, 4) if total else 0.0,
            "self": rows(selfc), "inclusive": rows(incl)}


def largest_host_costs(prof, n=6):
    """The named functions with the largest self time (generic interpreter and thread entry frames left out)."""
    out = []
    for r in (prof or {}).get("self") or []:
        sym = re.split(r" [\[(]", r["frame"], 1)[0]
        if GENERIC_FRAME.search(sym):
            continue
        out.append(r)
        if len(out) >= n:
            break
    return out


def cmd_perf(a):
    win = read_window(a.window)
    stacks = []
    skipped = collections.Counter()
    with open(a.script, errors="replace") as fh:
        for comm, period, frames in perf_samples(fh):
            if comm in ("perf", "py-spy", "perf-exec"):
                skipped[comm] += 1
                continue
            stacks.append((period, frames))
    weights = collections.Counter()
    for w, f in stacks:
        weights[";".join(f)] += w
    with open(a.folded, "w") as fh:
        for k, w in weights.most_common():
            fh.write("%s %d\n" % (k, w))
    prof = top_frames(stacks, win["calls"])
    prof.update({"window": win, "skipped_comm": dict(skipped), "stacks": len(weights),
                 "weight": "cpu-clock nanoseconds (perf period)"})
    with open(a.out, "w") as fh:
        json.dump(prof, fh, indent=1, sort_keys=True)
    print("PERF samples=%d calls=%d cpu_ms_per_call=%s top_self=%s" % (
        prof["samples"], win["calls"], prof["cpu_ms_per_call"], (prof["self"] or [{}])[0].get("frame")))
    return 0 if prof["samples"] else EXIT_NO_PROFILE


# ---- py-spy raw stacks (the window only) -> top frames and the category shares of the main thread ---------------

_PYSPY_FRAME = re.compile(r"^(.*) \(([^()]*)\)$")


def pyspy_frame(f):
    """(display name, object) of a py-spy frame: `func (file.py:12)` -> (`func (file.py)`, `file.py`);
    `sym (lib.so)` -> (`sym (lib.so)`, `lib.so`); an unresolved address keeps its library."""
    m = _PYSPY_FRAME.match(f.strip())
    if not m:
        return f.strip(), ""
    name, obj = m.group(1), m.group(2)
    obj = re.sub(r":\d+$", "", obj)
    return "%s (%s)" % (name, obj), obj


def pyspy_threads(lines):
    """{thread label: [(samples, [frames root..leaf])]} of `py-spy record --format raw --threads` output (a stack
    without a thread frame goes under '?'; `process N:...` frames of --subprocesses are dropped)."""
    out = collections.defaultdict(list)
    for raw in lines:
        raw = raw.rstrip("\n")
        if " " not in raw:
            continue
        stack, n = raw.rsplit(" ", 1)
        try:
            n = int(n)
        except ValueError:
            continue
        frames = [f for f in stack.split(";") if f and not f.startswith("process ")]
        label = "?"
        if frames and frames[0].startswith("thread ("):
            label = frames.pop(0)
        out[label].append((n, frames))
    return out


def main_thread(threads):
    """The label of the thread that runs the calls: py-spy's `MainThread`, else the one with the most samples."""
    for label in threads:
        if "MainThread" in label:
            return label
    return max(threads, key=lambda k: sum(n for n, _ in threads[k])) if threads else None


def leaf_split(stacks, frame_fn):
    """{category: weight}, {binding frame: weight}, weight left out as the shim's: the leaf of every stack by its
    category. A stack inside a CUDA driver entry or a libc allocation/copy function (at any depth: the shim times
    the whole call, its callees included) is the shim's, timed exactly there."""
    cats, binds, shim = collections.Counter(), collections.Counter(), 0.0
    for w, frames in stacks:
        if not frames:
            continue
        named = [frame_fn(f) for f in frames]
        if any(category(obj) == "cuda_driver" or
               (category(obj) == "libc" and _SHIM_LEAF.match(name.split(" (", 1)[0].split(" [", 1)[0]))
               for name, obj in named):
            shim += w
            continue
        name, obj = named[-1]
        cat = category(obj)
        cats[cat] += w
        if cat == "bindings":
            binds[name] += w
    return cats, binds, shim


def cmd_pyspy(a):
    win = read_window(a.window)
    threads = {}
    if os.path.isfile(a.raw):
        with open(a.raw, errors="replace") as fh:
            threads = pyspy_threads(fh)
    main = main_thread(threads)
    stacks = [(n, [pyspy_frame(f)[0] for f in fr]) for n, fr in threads.get(main, [])]
    prof = top_frames(stacks, win["calls"], unit_ns=1e9 / float(a.rate))
    prof["sample_count"] = sum(n for n, _ in stacks)
    cats, binds, shim = leaf_split(threads.get(main, []), pyspy_frame)
    tot = float(sum(cats.values()) + shim) or 1.0
    prof.update({"window": win, "thread": main, "rate": a.rate,
                 "other_threads": {k: sum(n for n, _ in v) for k, v in threads.items() if k != main},
                 "shares": {k: round(v / tot, 4) for k, v in cats.items()}, "shim_share": round(shim / tot, 4),
                 "bindings": [{"frame": k, "share": round(v / tot, 4)} for k, v in binds.most_common(12)],
                 "weight": "samples of the main thread at %s Hz, window only (py-spy's own pauses stretch the "
                           "pass's wall; shares hold)" % a.rate})
    with open(a.out, "w") as fh:
        json.dump(prof, fh, indent=1, sort_keys=True)
    print("PYSPY samples=%d stacks=%d thread=%s calls=%d top_self=%s" % (
        prof["sample_count"], prof["samples"], main, win["calls"], (prof["self"] or [{}])[0].get("frame")))
    return 0 if prof["samples"] else EXIT_NO_PROFILE


# ---- callgrind out files -> instructions per call by function ----------------------------------------------------

_CG_NAME = re.compile(r"^\((\d+)\)(?: (.*))?$")
_CG_KIND = {"ob": "ob", "cob": "ob", "fl": "fl", "fi": "fl", "fe": "fl", "cfi": "fl", "cfl": "fl", "fn": "fn",
            "cfn": "fn"}


def callgrind_parse(lines):
    """{self: {(fn, ob): Ir}, inclusive: {(fn, ob): Ir}, summary, trigger, pid} of one callgrind out file (the first
    event, Ir). A cost line after `calls=` is the call's inclusive cost: the caller's inclusive, not its self.
    Inclusive costs come from a function's outermost recursion level only."""
    names = {"ob": {}, "fl": {}, "fn": {}}
    selfc, incl = collections.Counter(), collections.Counter()
    out = {"summary": None, "trigger": None, "pid": None, "events": None}
    npos, cur_ob, cur, pending = 1, "?", None, False

    def resolve(kind, val):
        m = _CG_NAME.match(val)
        if not m:
            return val
        if m.group(2) is not None:
            names[kind][m.group(1)] = m.group(2)
        return names[kind].get(m.group(1), "(%s)" % m.group(1))

    for raw in lines:
        line = raw.rstrip("\n")
        if not line or line[0] == "#":
            continue
        c0 = line[0]
        if c0.isdigit() or c0 in "+-*":
            if cur is None:
                continue
            parts = line.split()
            cost = int(parts[npos]) if len(parts) > npos and parts[npos].isdigit() else 0
            if pending:
                pending = False
            else:
                selfc[cur[0]] += cost
            if cur[1]:
                incl[cur[0]] += cost
            continue
        key, sep, val = line.partition("=")
        if sep and key in _CG_KIND:
            name = resolve(_CG_KIND[key], val.strip())
            if key == "ob":
                cur_ob = name
            elif key == "fn":
                # a recursion level past the first (`f'2`, --separate-recs): its self cost is f's, its inclusive cost
                # is already inside the outermost level's (counting it again would multiply nested calls)
                base = re.sub(r"'\d+$", "", name)
                cur = ((base, cur_ob), base == name)
            continue
        if sep and key == "calls":
            pending = True
            continue
        hk, _, hv = line.partition(":")
        hv = hv.strip()
        if hk == "positions":
            npos = len(hv.split()) or 1
        elif hk == "events":
            out["events"] = hv.split()
        elif hk in ("summary", "totals") and hv.split() and hv.split()[0].isdigit():
            out["summary"] = int(hv.split()[0])
        elif hk == "pid" and hv.isdigit():
            out["pid"] = int(hv)
        elif line.startswith("desc: Trigger:"):
            out["trigger"] = line.split(":", 2)[2].strip()
    out["self"], out["inclusive"] = selfc, incl
    return out


def cg_frame(key):
    fn, ob = key
    return "%s [%s]" % (fn, os.path.basename(ob) or "?")


def cmd_callgrind(a):
    win = read_window(a.window)
    calls = max(int(win["calls"] or 0), 1)
    picked, seen = None, []
    for f in sorted(os.listdir(a.dir)) if os.path.isdir(a.dir) else []:
        if not f.startswith("callgrind.out"):
            continue
        path = os.path.join(a.dir, f)
        if f.endswith(".gz"):          # a re-run of the post phase reads the kept (compressed) files
            import gzip
            fh = gzip.open(path, "rt", errors="replace")
        else:
            fh = open(path, errors="replace")
        with fh:
            got = callgrind_parse(fh)
        seen.append({"file": f, "pid": got["pid"], "trigger": got["trigger"], "summary": got["summary"]})
        if not (got["trigger"] or "").endswith("callprof-window") or not got["summary"]:
            continue
        if picked is None or got["pid"] == win["pid"] or (picked["pid"] != win["pid"] and
                                                          got["summary"] > picked["summary"]):
            picked = got
    out = {"window": win, "files": seen, "ir_per_call": None, "self": [], "inclusive": [], "categories": {},
           "bindings": [], "weight": "instructions (Ir) per call, window only (the inclusive cost of frames already on "
                     "the stack at the window's open is approximate)"}
    if picked is not None:
        tot = float(sum(picked["self"].values())) or 1.0
        out["ir_per_call"] = round(tot / calls, 1)
        out["pid"] = picked["pid"]

        def rows(counter):
            return [{"frame": cg_frame(k), "category": category(k[1]), "ir_per_call": round(v / calls, 1),
                     "pct": round(100.0 * v / tot, 2)} for k, v in counter.most_common(TOP)]
        out["self"], out["inclusive"] = rows(picked["self"]), rows(picked["inclusive"])
        cats, binds, shim = leaf_split([(v, [k]) for k, v in picked["self"].items()],
                                       lambda k: (cg_frame(k), k[1]))
        out["categories"] = {k: round(v / calls, 1) for k, v in cats.items()}
        out["shim_ir_per_call"] = round(shim / calls, 1)
        den = float(sum(cats.values()) + shim) or 1.0
        out["shares"] = {k: round(v / den, 4) for k, v in cats.items()}
        out["shim_share"] = round(shim / den, 4)
        out["bindings"] = [{"frame": k, "share": round(v / den, 4)} for k, v in binds.most_common(12)]
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print("CALLGRIND files=%d ir_per_call=%s calls=%d top_self=%s" % (
        len(seen), out["ir_per_call"], win["calls"], (out["self"] or [{}])[0].get("frame")))
    return 0 if out["ir_per_call"] else EXIT_NO_PROFILE


# ---- strace -f -ttt -T -> syscalls per call in the window ---------------------------------------------------------

_ST = re.compile(r"^(?:\[pid\s+)?(\d+)\]?\s+(\d+\.\d+)\s+(.*)$")
_CALL = re.compile(r"^([a-z_0-9]+)\(")
_RESUMED = re.compile(r"^<\.\.\. ([a-z_0-9]+) resumed>")
_DUR = re.compile(r"<(\d+\.\d+)>\s*$")


def strace_counts(lines):
    """{calls: {name: n}, seconds: {name: s}, found_open, found_close} of the syscalls between the window's marker
    stats (by timestamp, every thread and process of the trace); the markers themselves are not counted."""
    recs = []
    t_open = t_close = None
    for raw in lines:
        m = _ST.match(raw.rstrip("\n"))
        if not m:
            continue
        ts, body = float(m.group(2)), m.group(3)
        if STRACE_MARK_OPEN in body and t_open is None:
            t_open = ts
            continue
        if STRACE_MARK_CLOSE in body and t_open is not None and t_close is None:
            t_close = ts
            continue
        r = _RESUMED.match(body)
        d = _DUR.search(body)
        if r:
            recs.append((ts, r.group(1), False, float(d.group(1)) if d else 0.0))
            continue
        c = _CALL.match(body)
        if c:
            recs.append((ts, c.group(1), True, float(d.group(1)) if d and "<unfinished" not in body else 0.0))
    calls, secs = collections.Counter(), collections.Counter()
    if t_open is not None and t_close is not None:
        for ts, name, counted, dur in recs:
            if t_open <= ts <= t_close:
                if counted:
                    calls[name] += 1
                secs[name] += dur
    return {"calls": dict(calls), "seconds": dict(secs), "found_open": t_open is not None,
            "found_close": t_close is not None}


def cmd_strace(a):
    win = read_window(a.window)
    with open(a.trace, errors="replace") as fh:
        st = strace_counts(fh)
    n = max(win["calls"], 1)
    per = sorted(({"syscall": k, "per_call": round(v / n, 3), "us_per_call": round(st["seconds"].get(k, 0.0) * 1e6 / n, 2)}
                  for k, v in st["calls"].items()), key=lambda r: -r["per_call"])
    out = {"window": win, "per_call": per, "total_per_call": round(sum(st["calls"].values()) / n, 3),
           "markers": {"open": st["found_open"], "close": st["found_close"]},
           "note": "strace -f slows every syscall; counts per call hold, times do not"}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print("STRACE calls=%d syscalls_per_call=%s markers=%s/%s" % (win["calls"], out["total_per_call"],
                                                                 st["found_open"], st["found_close"]))
    return 0 if st["found_open"] and st["found_close"] else EXIT_NO_PROFILE


# ---- the report --------------------------------------------------------------------------------------------------

def _load(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _f(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, float):
        return ("%." + str(nd) + "f") % x
    return str(x)


def host_split(wall_ms, shim, sampled, shim_wall_ms=None):
    """Host microseconds per call of one row split into the report's parts: the CUDA driver entries and libc
    allocations/copies the shim timed exactly, and the rest of the wall (wall - CUDA - libc) shared out by the
    sampled source's leaf categories (py-spy's main-thread samples, else callgrind's instructions); `unexplained` is
    the rest's share on frames with no object, or the whole rest when nothing was sampled. With the shim, the wall
    split is the shim pass's own (the same run as its times: two runs' walls differ by more than a small entry);
    the time pass's untraced wall stands beside it (`wall_us`; the difference is the shim's cost and run noise)."""
    if wall_ms is None:
        return None
    W = (shim_wall_ms if shim and shim_wall_ms is not None else wall_ms) * 1e3
    out = {"wall_us": round(wall_ms * 1e3, 2), "split_wall_us": round(W, 2), "cuda_us": None, "libc_us": None, "cuda": {}, "libc": {}, "source": None,
           "parts_us": {}, "bindings": [], "unexplained_us": None}
    C = L = 0.0
    if shim:
        for k, v in shim.items():
            (out["libc"] if k in SHIM_LIBC else out["cuda"])[k] = {
                "calls": round(v["calls"], 2), "us": round(v["us"], 2), "bytes": round(v["bytes"], 1)}
        C = sum(v["us"] for v in out["cuda"].values())
        L = sum(v["us"] for v in out["libc"].values())
        out["cuda_us"], out["libc_us"] = round(C, 2), round(L, 2)
    R = max(W - C - L, 0.0)
    out["rest_us"] = round(R, 2)
    src = None
    for name, prof in sampled:
        if prof and prof.get("shares"):
            src = (name, prof)
            break
    if src is None:
        out["unexplained_us"] = round(R, 2)
        return out
    name, prof = src
    out["source"] = name
    shares = dict(prof["shares"])
    if shim:
        den = max(1.0 - float(prof.get("shim_share") or 0.0), 1e-9)      # the shim's leaves are timed above
    else:
        den = 1.0
        shares["cuda_driver+libc_alloc (sampled)"] = float(prof.get("shim_share") or 0.0)
    for k, v in shares.items():
        if k != "unresolved":
            out["parts_us"][k] = round(R * v / den, 2)
    out["unexplained_us"] = round(R * float(shares.get("unresolved", 0.0)) / den, 2)
    out["bindings"] = [{"frame": b["frame"], "us": round(R * b["share"] / den, 2)} for b in prof.get("bindings") or []]
    return out


def read_capped(path):
    """{cap_s, elapsed_s, ...} of a pass the job stopped at its cap (run.capped, one JSON object), or None."""
    if not os.path.isfile(path):
        return None
    got = _load(path)
    return got if isinstance(got, dict) else {"cap_s": None, "elapsed_s": None}


def largest_cost(rec):
    """One line naming the largest cost of a row's call: GPU kernels (with the top kernel), the host outside CUDA
    (with the top sampled host frame) or the CUDA API outside its waits (with the top entry), whichever is largest;
    None without a wall or an nsys capture. Kernel time and the host's waits overlap: the API's sync time is the host
    waiting for those kernels, so it is not a cost of its own."""
    ns = rec.get("nsys") or {}
    sp = rec.get("split") or {}
    wall = rec.get("wall_ms_per_call")
    if wall is None and rec.get("capped"):
        cp = rec["capped"]
        wall = float(cp.get("in_call_s") or cp.get("elapsed_s") or 0.0) * 1e3 or None
    if not wall or not ns.get("captured"):
        return None
    api, sync = ns.get("api_ms"), ns.get("sync_ms") or 0.0
    parts = {"GPU kernels": ns.get("kernel_ms"),
             "host outside CUDA": sp.get("host_outside_cuda_ms") if rec.get("wall_ms_per_call") is not None else
             (max(wall - api, 0.0) if api is not None else None),
             "CUDA API outside its waits (launches, copies, allocations)": (max(api - sync, 0.0)
                                                                          if api is not None else None)}
    parts = {k: v for k, v in parts.items() if v is not None}
    if not parts:
        return None
    k, v = max(parts.items(), key=lambda kv: kv[1])
    if k == "GPU kernels":
        top = sorted(ns.get("kernels") or [], key=lambda x: -x.get("total_ms_per_call", 0.0))
        det = ("top kernel `%s` %s ms in %s launches" % (top[0]["name"][:90], _f(top[0]["total_ms_per_call"], 1),
                                                         _f(top[0]["instances_per_call"], 0))) if top else "-"
    elif k == "host outside CUDA":
        top = rec.get("largest_host_costs") or []
        det = ("top host frame `%s` %s (%s)" % (top[0]["frame"][:90], _cost(top[0]), rec.get("largest_host_costs_source"))
               if top else "no sampled host frame")
    else:
        top = sorted(((n, x) for n, x in (ns.get("api") or {}).items() if x.get("kind") != "sync"),
                     key=lambda kv: -kv[1].get("ms", 0.0))
        det = ("top entry %s %s ms in %s calls" % (top[0][0], _f(top[0][1]["ms"], 1), _f(top[0][1]["calls"], 0))
               if top else "-")
    return "%s %s ms of %s ms (%s%%): %s" % (k, _f(v, 1), _f(wall, 1), _f(100.0 * v / wall, 1), det)


def collect_row(res, row):
    d = os.path.join(res, "rows", row.name)
    out = {"row": row.name, "race_id": row.race_id, "phase": row.phase, "passes": {}, "stored_ms": row.ms,
           "long": row.long, "cap_s": row.cap(), "capped": None}
    for p in PASSES:
        pd = os.path.join(d, p)
        err = None
        ef = os.path.join(pd, "run.error")
        if os.path.isfile(ef):
            with open(ef, errors="replace") as fh:
                err = fh.read().strip()[:400]
        cap = read_capped(os.path.join(pd, "run.capped"))
        out["passes"][p] = {"window": read_window(os.path.join(pd, "window.jsonl")), "error": err,
                            "ran": os.path.isdir(pd), "capped": cap}
        if cap is not None:
            # seconds inside the call when the cap stopped it: from the window's open-mark to the stop
            w = out["passes"][p]["window"]
            if w.get("opened_unix") and cap.get("t_capped_unix"):
                cap["in_call_s"] = round(float(cap["t_capped_unix"]) - float(w["opened_unix"]), 1)
        if cap is not None and out["capped"] is None:
            out["capped"] = dict(cap, passes=[])
        if cap is not None:
            out["capped"]["passes"].append(p)
    for src, f in (("perf", "perf.json"), ("pyspy", "pyspy.json"), ("nsys", "nsys.json"), ("strace", "strace.json"),
                   ("callgrind", "callgrind.json")):
        out[src] = _load(os.path.join(d, src, f))
    out["wall_ms_per_call"], out["wall_source"] = None, None
    for p in ("time", "shim", "pyspy", "perf"):         # the time pass; else the least perturbed pass that ran
        w = out["passes"][p]["window"]["wall_ms_per_call"]
        if w is not None:
            out["wall_ms_per_call"] = w
            out["wall_source"] = "time pass" if p == "time" else "%s pass (perturbed by the tool)" % p
            break
    ns = out["nsys"] or {}
    wall = out["wall_ms_per_call"]
    split = {"kernel_ms": ns.get("kernel_ms"), "cuda_api_ms": ns.get("api_ms"), "sync_ms": ns.get("sync_ms"),
             "nsys_window_ms": ns.get("window_ms"), "host_outside_cuda_ms": None}
    if wall is not None and ns.get("api_ms") is not None:
        # host outside CUDA at the clean wall: the nsys window carries CUPTI's own cost, so its API time is
        # subtracted from the untraced wall (never below zero)
        split["host_outside_cuda_ms"] = max(wall - ns["api_ms"], 0.0)
    out["split"] = split
    out["shim"] = out["passes"]["shim"]["window"]["shim_per_call"]
    out["host"] = host_split(wall, out["shim"], (("py-spy", out["pyspy"]), ("callgrind", out["callgrind"])),
                             out["passes"]["shim"]["window"]["wall_ms_per_call"])
    out["largest_host_costs"], out["largest_host_costs_source"] = [], None
    for src in ("perf", "pyspy", "callgrind"):
        got = largest_host_costs(out[src])
        if got:
            out["largest_host_costs"], out["largest_host_costs_source"] = got, src
            break
    out["largest_cost"] = largest_cost(out)
    return out


def _cost(h):
    return ("%s us" % h["us_per_call"]) if "us_per_call" in h else ("%s Ir" % h.get("ir_per_call"))


def _h2d(m):
    """The table's H2D cell: host-to-device copies per call and their MB per call (lane fx-harness, 2026-11-18: the
    fit+infer rows are read by the bytes each call copies up)."""
    if not m or m.get("count") is None:
        return "-"
    return "%s (%s MB)" % (_f(m.get("count"), 1), _f(m.get("mb"), 2)) if m.get("mb") is not None else _f(m["count"], 1)


def render(doc):
    L = ["# Call profile (%s): where the wall time of one small call goes" % doc.get("vendor", "?"), "",
         "Wheel: %s. Host: %s. Diagnostic only: not a measurement and not a board cell. %s" % (
             doc.get("wheel") or "?", doc.get("host") or "?", doc.get("tools") or ""), "",
         "Host split, microseconds per call of the shim pass's wall (the wall ms column is the untraced time pass): "
         "CUDA driver entries and libc allocations/copies timed by the LD_PRELOAD shim; the rest shared by the leaf categories of "
         "py-spy's main-thread samples (else callgrind's instructions); unexplained = frames with no object, or the "
         "whole rest when nothing was sampled.", ""]
    L.append("| row | phase | wall ms/call | GPU kernel ms | CUDA API ms (nsys) | launches | syncs | allocs | H2D (MB) | D2H "
             "| bindings us | CPython us | Mojo runtime us | CUDA driver us (shim) | libc alloc+copy us (shim) "
             "| other us | unexplained us | split from |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in doc["rows"]:
        ns = r.get("nsys") or {}
        c = ns.get("counts") or {}
        mem = ns.get("mem") or {}
        h = r.get("host") or {}
        pu = h.get("parts_us") or {}
        other = float(sum(v for k, v in pu.items() if k not in ("bindings", "cpython", "mojo_runtime"))) if pu else None
        L.append("| %s |" % " | ".join(str(x) for x in (
            r["row"], r["phase"], _f(r["wall_ms_per_call"]), _f(r["split"]["kernel_ms"]), _f(r["split"]["cuda_api_ms"]),
            _f(c.get("launch"), 1), _f(c.get("sync"), 1), _f(c.get("alloc"), 1),
            _h2d(mem.get("h2d")), _f((mem.get("d2h") or {}).get("count"), 1),
            _f(pu.get("bindings"), 1), _f(pu.get("cpython"), 1), _f(pu.get("mojo_runtime"), 1),
            _f(h.get("cuda_us"), 1), _f(h.get("libc_us"), 1), _f(other, 1), _f(h.get("unexplained_us"), 1),
            h.get("source") or "-")))
    L.append("")
    longs = [r for r in doc["rows"] if r.get("long")]
    if longs:
        L.append("Long rows: ONE call per pass (no warmup; the call carries what a first call carries), ms per call. "
                 "host = the wall less the CUDA API time (nsys); the API's sync time is the host waiting on kernels. "
                 "CAPPED: a pass the job stopped at the row's cap, reported with what it collected.")
        L.append("")
        L.append("| row | stored ms | wall ms | status | GPU kernel ms | CUDA API ms | of it sync ms | host ms | "
                 "launches | syncs | H2D MB | D2H MB | largest cost |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in longs:
            ns = r.get("nsys") or {}
            c = ns.get("counts") or {}
            mem = ns.get("mem") or {}
            L.append("| %s |" % " | ".join(str(x) for x in (
                r["row"], _f(r.get("stored_ms"), 0), _f(r["wall_ms_per_call"], 0), r.get("status") or "-",
                _f(ns.get("kernel_ms"), 0), _f(ns.get("api_ms"), 0), _f(ns.get("sync_ms"), 0),
                _f(r["split"].get("host_outside_cuda_ms"), 0), _f(c.get("launch"), 0), _f(c.get("sync"), 0),
                _f((mem.get("h2d") or {}).get("mb"), 1), _f((mem.get("d2h") or {}).get("mb"), 1),
                (r.get("largest_cost") or "-").replace("|", "/"))))
        L.append("")
    top_n = int(doc.get("top") or TOP)
    for r in doc["rows"]:
        L.append("## %s (%s, %s)" % (r["row"], r["race_id"], r["phase"]))
        L.append("LARGEST COST: %s" % (r.get("largest_cost") or "- (no wall or no nsys capture)"))
        if r.get("capped"):
            cp = r["capped"]
            L.append("CAPPED: %s stopped at the cap of %s s (%s s inside the call; the call ran at least that long); "
                     "what they collected is below" % (", ".join(cp.get("passes") or []), _f(cp.get("cap_s"), 0),
                                                        _f(cp.get("in_call_s"), 0)))
        names = ", ".join("%s %s" % (x["frame"], _cost(x)) for x in r["largest_host_costs"]) or "-"
        L.append("LARGEST HOST COSTS (%s, self per call): %s" % (r.get("largest_host_costs_source") or "none", names))
        L.append("")
        L.append("wall %s ms/call (%s); passes: %s" % (
            _f(r["wall_ms_per_call"]), r.get("wall_source") or "no window",
            "; ".join("%s calls=%d%s" % (p, v["window"]["calls"], (" ERROR " + v["error"][:160]) if v["error"] else "")
                      for p, v in r["passes"].items() if v["ran"])))
        h = r.get("host") or {}
        if h.get("cuda"):
            L.append("CUDA driver per call (shim): %s" % ", ".join(
                "%s %s x %s us%s" % (k, _f(v["calls"], 2), _f(v["us"], 2),
                                     (" %s B" % _f(v["bytes"], 0)) if v["bytes"] else "")
                for k, v in sorted(h["cuda"].items(), key=lambda kv: -kv[1]["us"])))
        if h.get("libc"):
            L.append("libc per call (shim): %s" % ", ".join(
                "%s %s x %s us %s B" % (k, _f(v["calls"], 2), _f(v["us"], 2), _f(v["bytes"], 0))
                for k, v in sorted(h["libc"].items(), key=lambda kv: -kv[1]["us"])))
        if h.get("parts_us"):
            L.append("rest of the shim pass's wall (%s us/call; untraced %s) %s us/call by %s: %s; unexplained %s" % (
                _f(h.get("split_wall_us"), 1), _f(h.get("wall_us"), 1), _f(h.get("rest_us"), 1), h.get("source"),
                ", ".join("%s %s" % (k, _f(v, 1)) for k, v in sorted(h["parts_us"].items(), key=lambda kv: -kv[1])),
                _f(h.get("unexplained_us"), 1)))
        if h.get("bindings"):
            L.append("our binding code by function (us/call): %s" % ", ".join(
                "%s %s" % (b["frame"], _f(b["us"], 1)) for b in h["bindings"]))
        ns = r.get("nsys") or {}
        if ns:
            mem = ns.get("mem") or {}
            L.append("nsys per call: %s; memory: %s" % (
                ", ".join("%s %s" % (k, _f(v, 2)) for k, v in sorted((ns.get("counts") or {}).items())),
                ", ".join("%s %s x %s MB" % (k, _f(v.get("count"), 2), _f(v.get("mb"), 4)) for k, v in sorted(mem.items()))))
            api = sorted((ns.get("api") or {}).items(), key=lambda kv: -kv[1].get("ms", 0))[:8]
            L.append("CUDA API per call (nsys): %s" % (", ".join("%s %s x %s ms" % (k, _f(v["calls"], 1), _f(v["ms"], 4))
                                                                for k, v in api) or "-"))
            kern = sorted(ns.get("kernels") or [], key=lambda x: -x.get("total_ms_per_call", 0.0))[:8]
            if kern:
                L.append("GPU kernels per call (nsys, top %d of %d): %s" % (len(kern), len(ns.get("kernels") or []), ", ".join(
                    "`%s` %s ms in %s launches" % (k["name"][:70], _f(k["total_ms_per_call"], 2),
                                                   _f(k["instances_per_call"], 1)) for k in kern)))
        st = r.get("strace") or {}
        if st:
            L.append("syscalls per call (%s in all): %s" % (_f(st.get("total_per_call"), 1), ", ".join(
                "%s %s" % (x["syscall"], _f(x["per_call"], 2)) for x in (st.get("per_call") or [])[:14])))
        for src in ("perf", "pyspy", "callgrind"):
            prof = r.get(src)
            if not prof or not (prof.get("self") or prof.get("inclusive")):
                continue
            L.append("")
            if src == "callgrind":
                L.append("callgrind: %s instructions per call (%s); by category: %s" % (
                    _f(prof.get("ir_per_call"), 1), prof.get("weight"),
                    ", ".join("%s %s" % (k, _f(v, 1)) for k, v in sorted((prof.get("categories") or {}).items(),
                                                                         key=lambda kv: -kv[1]))))
            else:
                L.append("%s: %d samples, %s ms per call (%s)" % (src, prof.get("sample_count", prof.get("samples", 0)),
                                                                 _f(prof.get("cpu_ms_per_call"), 4), prof.get("weight")))
            unit = "Ir/call" if src == "callgrind" else "us/call"
            for kind in ("self", "inclusive"):
                L.append("")
                L.append("| %s top %s | %% | %s |" % (src, kind, unit))
                L.append("|---|---|---|")
                for x in (prof.get(kind) or [])[:top_n]:
                    L.append("| `%s` | %s | %s |" % (x["frame"][:150].replace("|", "/"), x["pct"],
                                                    x.get("us_per_call", x.get("ir_per_call"))))
        L.append("")
    return "\n".join(L) + "\n"


def function_sources(rec):
    """The function-level sources a row has: py-spy or perf samples, callgrind instructions, the shim's counts."""
    got = []
    if (rec.get("pyspy") or {}).get("samples"):
        got.append("pyspy")
    if (rec.get("perf") or {}).get("samples"):
        got.append("perf")
    if (rec.get("callgrind") or {}).get("ir_per_call"):
        got.append("callgrind")
    if rec.get("shim"):
        got.append("shim")
    return got


def cmd_report(a):
    rows = load_rows(a.rows_file)
    doc = {"vendor": a.vendor, "wheel": a.wheel, "host": a.host, "tools": a.tools, "rows": [], "top": a.top}
    code = EXIT_OK
    for r in rows:
        rec = collect_row(a.res, r)
        rec["function_sources"] = function_sources(rec)
        doc["rows"].append(rec)
        if rec.get("capped"):
            # a pass stopped at the row's cap: reported with what it collected (never a silent timeout). Not an
            # exit code (lane fx-small, 2026-11-21): nvc1-0005's exit 8 made the job a failure though the report
            # was whole; the CAPPED status and SUMMARY line carry it.
            why = "CAPPED (%s at %s s)" % (",".join(rec["capped"].get("passes") or []),
                                           _f(rec["capped"].get("cap_s"), 0))
        elif rec["wall_ms_per_call"] is None:
            code = max(code, EXIT_NO_WALL)
            why = "no wall time"
        elif not rec["function_sources"]:
            code = max(code, EXIT_NO_PROFILE)
            why = "no function-level source (py-spy, perf, callgrind, shim)"
        elif not (rec["nsys"] or {}).get("captured"):
            code = max(code, EXIT_NO_PROFILE)
            why = "no nsys capture"
        else:
            why = "ok"
        rec["status"] = "CAPPED" if rec.get("capped") else ("ok" if why == "ok" else "incomplete")
        h = rec.get("host") or {}
        if rec.get("long"):
            print("LARGEST row=%s %s" % (r.name, rec.get("largest_cost") or "-"))
        print("SUMMARY row=%s wall_ms=%s kernel_ms=%s api_ms=%s cuda_us=%s libc_us=%s unexplained_us=%s sources=%s "
              "top=%s status=%s" % (
                  r.name, _f(rec["wall_ms_per_call"]), _f(rec["split"]["kernel_ms"]), _f(rec["split"]["cuda_api_ms"]),
                  _f(h.get("cuda_us"), 1), _f(h.get("libc_us"), 1), _f(h.get("unexplained_us"), 1),
                  ",".join(rec["function_sources"]) or "-",
                  (rec["largest_host_costs"] or [{}])[0].get("frame", "-")[:80], why))
    with open(os.path.join(a.res, "CALL-PROFILE-%s.json" % a.vendor), "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
    with open(os.path.join(a.res, "CALL-PROFILE-%s.md" % a.vendor), "w") as fh:
        fh.write(render(doc))
    print("CALL-PROFILE rows=%d exit=%d" % (len(rows), code))
    return code


def cmd_rows(a):
    rows = load_rows(a.rows_file)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "runs.tsv"), "w") as fh:
        for r in rows:
            # cap (s, 0 = none) and long (1: one call per pass): tools/ops/call_profile_job.sh run_pass
            fh.write("%s\t%s\t%s\t%d\t%d\n" % (r.name, r.race_id, r.phase, r.cap() or 0, 1 if r.long else 0))
    with open(os.path.join(a.out_dir, "ids.txt"), "w") as fh:
        for rid in dict.fromkeys(r.race_id for r in rows):
            fh.write(rid + "\n")
    print("ROWS rows=%d races=%d" % (len(rows), len(set(r.race_id for r in rows))))
    return 0


def cmd_estimate(a):
    rows = load_rows(a.rows_file)
    passes = [p for p in (a.passes or ",".join(DEFAULT_PASSES)).split(",") if p]
    bad = [p for p in passes if p not in PASSES]
    if bad:
        print("unknown passes %s (known: %s)" % (bad, ",".join(PASSES)), file=sys.stderr)
        return 2
    secs = {}
    for kv in a.seconds or []:
        k, _, v = kv.partition("=")
        secs[k] = float(v)
    e = estimate(rows, passes, secs, overhead_s=a.overhead_s, callgrind_parallel=a.callgrind_parallel)
    for r in e["rows"]:
        print("ESTIMATE row=%s pod_min=%s%s" % (r["row"], r["pod_min"], (" one-call stored_s=%s cap_s=%s" % (
            _f(r["stored_ms"] / 1000.0, 1), r["cap_s"])) if r.get("long") else ""))
    if e["long_rows"]:
        print("ESTIMATE long rows=%d stored %s s x %d passes = %s min (one call per pass, no warmup); at every cap %s min"
              % (e["long_rows"], e["long_stored_s"], len(e["long_passes"]),
                 e["long_stored_x_passes_min"], e["cap_total_min"]))
    print("ESTIMATE total pod_min=%s passes=%s (%s s per race process outside its window; callgrind %s s with %d "
          "rows at once; setup %s s)" % (e["total_min"], ",".join(passes), a.overhead_s, e["callgrind_s"],
                                         a.callgrind_parallel, SETUP_S))
    return 0


def cmd_window(a):
    print(json.dumps(read_window(a.window), sort_keys=True))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rows")
    r.add_argument("--rows-file", default=None)
    r.add_argument("--out-dir", required=True)
    e = sub.add_parser("estimate")
    e.add_argument("--rows-file", default=None)
    e.add_argument("--passes", default=None)
    e.add_argument("--seconds", action="append", help="<pass>=<window seconds>")
    e.add_argument("--overhead-s", type=float, default=OVERHEAD_S)
    e.add_argument("--callgrind-parallel", type=int, default=CALLGRIND_PARALLEL)
    w = sub.add_parser("window")
    w.add_argument("--window", required=True)
    p = sub.add_parser("perf")
    p.add_argument("--script", required=True)
    p.add_argument("--window", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--folded", required=True)
    y = sub.add_parser("pyspy")
    y.add_argument("--raw", required=True)
    y.add_argument("--window", required=True)
    y.add_argument("--rate", type=float, default=250.0)
    y.add_argument("--out", required=True)
    g = sub.add_parser("callgrind")
    g.add_argument("--dir", required=True)
    g.add_argument("--window", required=True)
    g.add_argument("--out", required=True)
    s = sub.add_parser("strace")
    s.add_argument("--trace", required=True)
    s.add_argument("--window", required=True)
    s.add_argument("--out", required=True)
    rp = sub.add_parser("report")
    rp.add_argument("--res", required=True)
    rp.add_argument("--rows-file", default=None)
    rp.add_argument("--vendor", default="nvidia")
    rp.add_argument("--wheel", default=None)
    rp.add_argument("--host", default=None)
    rp.add_argument("--tools", default=None)
    rp.add_argument("--top", type=int, default=TOP, help="frames per top table (the job: CP_REPORT_TOP)")
    a = ap.parse_args(argv)
    try:
        return {"rows": cmd_rows, "estimate": cmd_estimate, "window": cmd_window, "perf": cmd_perf,
                "pyspy": cmd_pyspy, "callgrind": cmd_callgrind, "strace": cmd_strace, "report": cmd_report}[a.cmd](a)
    except ValueError as exc:
        print("refused: %s" % exc, file=sys.stderr)
        return 5


if __name__ == "__main__":
    sys.exit(main())
