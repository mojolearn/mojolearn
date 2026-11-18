#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""tools/ops/call_profile.py -- the plan, the parsers and the report of tools/ops/call_profile_job.sh: WHERE THE WALL
TIME OF ONE SMALL PUBLIC CALL GOES, at function level, host side and GPU side, on OUR wheel alone (diagnostic tooling,
not a board and not a measurement).

Per row the job runs the row's race once per pass, each a forced scored run whose extra calls (2..n) sit inside the
harness's profile window (tools/bench_board_small_rows.py range_open/range_close) and whose n fills the pass's
seconds at the first scored call's time:
    time    nothing attached: wall per call (the window's wall clock over its calls)
    perf    `perf record -e cpu-clock --call-graph dwarf -D -1 --control fifo:...`: samples only inside the window;
            in the same process `py-spy record --native` (cross-check: stacks under the forced phase's runner method;
            py-spy's pauses are off-CPU, so they do not move perf's on-CPU shares). Without usable perf: py-spy alone
    nsys    `nsys profile -t cuda,nvtx --capture-range=nvtx`: launches, syncs, allocations, copies, API and kernel ms
    strace  `strace -f -ttt -T` cut to the window by the marker stats, plus the LD_PRELOAD allocation counter
Subcommands (each prints one line; the job keeps their files):
    rows      --rows-file F --out-dir D      D/runs.tsv (row, race id, phase) and D/ids.txt (the races to plan)
    estimate  --rows-file F [--passes ...]   pod minutes per row and in all (printed by the job's plan phase)
    window    --window W                     calls and wall of a pass's window (JSON)
    perf      --script S --window W --out J --folded F      folded stacks and the top self/inclusive frames
    pyspy     --collapsed C --phase P --window W --out J    the same from py-spy's raw stacks
    strace    --trace T --window W --out J                  syscalls per call in the window
    report    --res R [--rows-file F]        CALL-PROFILE-<vendor>.md and .json; exit 0 every row has its wall, frames
                                             and nsys counts, 6 a row lacks a profile part, 7 a row lacks a wall time
"""
import argparse
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROWS_FILE = os.path.join(HERE, "call_profile_rows.tsv")
PASSES = ("time", "perf", "nsys", "strace")
#: seconds each pass's window runs (CP_<PASS>_SECONDS in the job); perf and py-spy want >= 10 s of samples, nsys
#: counts and the strace counts are per call and settle in a few seconds (strace slows every syscall)
PASS_SECONDS = {"time": 3.0, "perf": 10.0, "nsys": 3.0, "strace": 3.0}
#: the wall of one race process outside its window (start-up, data, fit before an inference, the first calls), and
#: each tool's start-up slowdown on it: the TG board of d652a4b0a1 ran the default rows' races in 8-30 s (mean 14 s)
#: with 21 calls of each phase; bench_board's own start-up added
OVERHEAD_S = 18.0
PASS_SLOWDOWN = {"time": 1.0, "perf": 1.4, "nsys": 1.8, "strace": 2.5}
SETUP_S = 240.0         # tool installs (in parallel) and the post-processing (in parallel over the cores)
TOP = 40
EXIT_OK, EXIT_NO_PROFILE, EXIT_NO_WALL = 0, 6, 7
STRACE_MARK_OPEN = "/callprof-window-open"
STRACE_MARK_CLOSE = "/callprof-window-close"
#: frames that only carry the call down to the work (interpreter entry, thread start): never named as a host cost
GENERIC_FRAME = re.compile(r"^(_start|__libc_start|main|Py_|_Py|PyEval|PyObject_(Vectorcall|Call)|pymain|"
                           r"start_thread|clone3?|__clone|\[unknown\]|_PyFunction|method_vectorcall|cfunction_)")


class Row(object):
    def __init__(self, name, race_id, phase):
        self.name, self.race_id, self.phase = name, race_id, phase


def parse_rows(text, where="rows file"):
    """[Row] from a rows file (race id, phase, optional name=); a malformed line or a repeated name is refused."""
    rows, names, seen = [], set(), collections.Counter()
    for i, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        f = line.split()
        if len(f) < 2 or f[0].count("/") < 2 or not (f[1] == "fit" or f[1].startswith("infer")):
            raise ValueError("%s line %d: want <race id> <fit|infer[/<batch>]> [name=]: %r" % (where, i, raw))
        name = None
        for opt in f[2:]:
            if opt.startswith("name="):
                name = opt[5:]
            else:
                raise ValueError("%s line %d: unknown field %r" % (where, i, opt))
        seen[f[0]] += 1
        if not name:
            name = f[0].split("/")[1] + ("-" + f[1].replace("/", "-") if seen[f[0]] > 1 else "")
        if not re.match(r"^[A-Za-z0-9._-]+$", name) or name in names:
            raise ValueError("%s line %d: row name %r is repeated or not a plain name (give name=)" % (where, i, name))
        names.add(name)
        rows.append(Row(name, f[0], f[1]))
    if not rows:
        raise ValueError("%s: no rows" % where)
    return rows


def load_rows(path=None):
    path = path or DEFAULT_ROWS_FILE
    with open(path) as fh:
        return parse_rows(fh.read(), path)


def estimate(rows, passes=PASSES, seconds=None, overhead_s=OVERHEAD_S, setup_s=SETUP_S):
    """{rows: [{row, seconds, pod_min}], total_min, assumptions}: one race process per row and pass."""
    secs = dict(PASS_SECONDS, **(seconds or {}))
    per = []
    for r in rows:
        s = sum(overhead_s * PASS_SLOWDOWN[p] + secs[p] for p in passes)
        per.append({"row": r.name, "seconds": round(s, 1), "pod_min": round(s / 60.0, 1)})
    total = sum(p["seconds"] for p in per) + setup_s
    return {"rows": per, "total_min": round(total / 60.0, 1), "passes": list(passes),
            "assumptions": {"overhead_s_per_race_process": overhead_s, "slowdown": PASS_SLOWDOWN,
                            "window_seconds": secs, "setup_and_postprocess_s": setup_s}}


# ---- the window of one pass --------------------------------------------------------------------------------------

def read_window(path):
    """{calls, wall_ms_per_call, wall_s, malloc_per_call, perf_enabled, windows} of a pass's window file (the first
    open/close pair; `windows` counts the pairs, more than one is reported). Missing file or no close: calls 0."""
    out = {"calls": 0, "wall_ms_per_call": None, "wall_s": None, "malloc_per_call": None, "perf_enabled": None,
           "windows": 0}
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
            if rec.get("event") == "open":
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
    if calls > 0 and c.get("wall_ns") is not None:
        out["wall_s"] = c["wall_ns"] / 1e9
        out["wall_ms_per_call"] = c["wall_ns"] / 1e6 / calls
    out["perf_enabled"] = bool(o.get("perf")) and bool(c.get("perf"))
    if calls > 0 and isinstance(o.get("malloc"), dict) and isinstance(c.get("malloc"), dict):
        out["malloc_per_call"] = {k: (c["malloc"].get(k, 0) - o["malloc"].get(k, 0)) / float(calls)
                                  for k in c["malloc"]}
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
        sym = r["frame"].split(" [", 1)[0]
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


def under_regex(phase):
    """The py-spy frame of the forced phase's call in a race driver: the runner method (`fit`, `infer`, the tree
    drivers' `predict`) defined in a tools/bench_board_*.py or bench/speed driver file."""
    meth = "fit" if phase == "fit" else "(infer|predict|predict_proba|transform)"
    return r"^%s \((?:.*/)?(bench_board_[a-z0-9_]+|classical_two_datasets|speed_[a-z0-9_]+|forest_speed_arm)\.py:" % meth


def cmd_pyspy(a):
    win = read_window(a.window)
    rx = re.compile(a.under or under_regex(a.phase))
    stacks, outside = [], 0
    if os.path.isfile(a.collapsed):
        with open(a.collapsed, errors="replace") as fh:
            for raw in fh:
                raw = raw.rstrip("\n")
                if " " not in raw:
                    continue
                stack, n = raw.rsplit(" ", 1)
                try:
                    n = int(n)
                except ValueError:
                    continue
                frames = stack.split(";")
                start = next((i for i, f in enumerate(frames) if rx.search(f)), None)
                if start is None:
                    outside += n
                    continue
                stacks.append((n, frames[start:]))
    # py-spy samples every active thread `rate` times a second: a sample stands for 1/rate seconds of that thread
    prof = top_frames(stacks, win["calls"], unit_ns=1e9 / float(a.rate))
    prof.update({"window": win, "outside_samples": outside, "under": rx.pattern,
                 "weight": "samples at %s Hz (wall time of each active thread, warmup and first call included)"
                           % a.rate})
    with open(a.out, "w") as fh:
        json.dump(prof, fh, indent=1, sort_keys=True)
    print("PYSPY samples_under_call=%d outside=%d top_self=%s" % (
        prof["samples"], outside, (prof["self"] or [{}])[0].get("frame")))
    return 0 if prof["samples"] else EXIT_NO_PROFILE


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


def collect_row(res, row):
    d = os.path.join(res, "rows", row.name)
    out = {"row": row.name, "race_id": row.race_id, "phase": row.phase, "passes": {}}
    for p in PASSES:
        pd = os.path.join(d, p)
        err = None
        ef = os.path.join(pd, "run.error")
        if os.path.isfile(ef):
            with open(ef, errors="replace") as fh:
                err = fh.read().strip()[:400]
        out["passes"][p] = {"window": read_window(os.path.join(pd, "window.jsonl")), "error": err,
                            "ran": os.path.isdir(pd)}
    out["perf"] = _load(os.path.join(d, "perf", "perf.json"))
    out["pyspy"] = _load(os.path.join(d, "perf", "pyspy.json"))
    out["nsys"] = _load(os.path.join(d, "nsys", "nsys.json"))
    out["strace"] = _load(os.path.join(d, "strace", "strace.json"))
    tw = out["passes"]["time"]["window"]
    pw = out["passes"]["perf"]["window"]
    out["wall_ms_per_call"] = tw["wall_ms_per_call"]
    out["wall_source"] = "time pass" if tw["wall_ms_per_call"] is not None else None
    if out["wall_ms_per_call"] is None and pw["wall_ms_per_call"] is not None:
        out["wall_ms_per_call"], out["wall_source"] = pw["wall_ms_per_call"], "perf pass (sampling at 999 Hz)"
    ns = out["nsys"] or {}
    wall = out["wall_ms_per_call"]
    split = {"kernel_ms": ns.get("kernel_ms"), "cuda_api_ms": ns.get("api_ms"), "sync_ms": ns.get("sync_ms"),
             "nsys_window_ms": ns.get("window_ms"), "host_outside_cuda_ms": None}
    if wall is not None and ns.get("api_ms") is not None:
        # host outside CUDA at the clean wall: the nsys window carries CUPTI's own cost, so its API time is
        # subtracted from the untraced wall (never below zero)
        split["host_outside_cuda_ms"] = max(wall - ns["api_ms"], 0.0)
    out["split"] = split
    out["malloc_per_call"] = out["passes"]["strace"]["window"].get("malloc_per_call")
    out["largest_host_costs"] = largest_host_costs(out["perf"]) or largest_host_costs(out["pyspy"])
    out["largest_host_costs_source"] = "perf" if largest_host_costs(out["perf"]) else (
        "py-spy" if out["largest_host_costs"] else None)
    return out


def render(doc):
    L = ["# Call profile (%s): where the wall time of one small call goes" % doc.get("vendor", "?"), "",
         "Wheel: %s. Host: %s. Diagnostic only: not a measurement and not a board cell. %s" % (
             doc.get("wheel") or "?", doc.get("host") or "?", doc.get("tools") or ""), ""]
    L.append("| row | phase | wall ms/call | GPU kernel ms | CUDA API ms | sync ms | host outside CUDA ms | "
             "launches | syncs | allocs | H2D | D2H | syscalls | mallocs |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in doc["rows"]:
        ns = r.get("nsys") or {}
        c = ns.get("counts") or {}
        mem = ns.get("mem") or {}
        st = r.get("strace") or {}
        mp = r.get("malloc_per_call") or {}
        mall = sum(mp.get(k, 0) for k in ("malloc", "calloc", "realloc", "aligned")) if mp else None
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["row"], r["phase"], _f(r["wall_ms_per_call"]), _f(r["split"]["kernel_ms"]),
            _f(r["split"]["cuda_api_ms"]), _f(r["split"]["sync_ms"]), _f(r["split"]["host_outside_cuda_ms"]),
            _f(c.get("launch"), 1), _f(c.get("sync"), 1), _f(c.get("alloc"), 1),
            _f((mem.get("h2d") or {}).get("count"), 1), _f((mem.get("d2h") or {}).get("count"), 1),
            _f(st.get("total_per_call"), 1), _f(mall, 1)))
    L.append("")
    for r in doc["rows"]:
        L.append("## %s (%s, %s)" % (r["row"], r["race_id"], r["phase"]))
        names = ", ".join("%s %s us" % (h["frame"], h["us_per_call"]) for h in r["largest_host_costs"]) or "-"
        L.append("LARGEST HOST COSTS (%s, self time per call): %s" % (r.get("largest_host_costs_source") or "none",
                                                                       names))
        L.append("")
        L.append("wall %s ms/call (%s); passes: %s" % (
            _f(r["wall_ms_per_call"]), r.get("wall_source") or "no window",
            "; ".join("%s calls=%d%s" % (p, v["window"]["calls"], (" ERROR " + v["error"][:160]) if v["error"] else "")
                      for p, v in r["passes"].items() if v["ran"])))
        ns = r.get("nsys") or {}
        if ns:
            mem = ns.get("mem") or {}
            L.append("nsys per call: %s; memory: %s" % (
                ", ".join("%s %s" % (k, _f(v, 2)) for k, v in sorted((ns.get("counts") or {}).items())),
                ", ".join("%s %s x %s MB" % (k, _f(v.get("count"), 2), _f(v.get("mb"), 4)) for k, v in sorted(mem.items()))))
            api = sorted((ns.get("api") or {}).items(), key=lambda kv: -kv[1].get("ms", 0))[:8]
            L.append("CUDA API per call: %s" % (", ".join("%s %s x %s ms" % (k, _f(v["calls"], 1), _f(v["ms"], 4))
                                                         for k, v in api) or "-"))
        st = r.get("strace") or {}
        if st:
            L.append("syscalls per call (%s in all): %s" % (_f(st.get("total_per_call"), 1), ", ".join(
                "%s %s" % (x["syscall"], _f(x["per_call"], 2)) for x in (st.get("per_call") or [])[:14])))
        mp = r.get("malloc_per_call")
        if mp:
            L.append("libc allocations per call: %s" % ", ".join("%s %s" % (k, _f(v, 1)) for k, v in sorted(mp.items())))
        for src in ("perf", "pyspy"):
            prof = r.get(src)
            if not prof:
                continue
            L.append("")
            L.append("%s: %d samples, %s ms CPU per call (%s)" % (src, prof.get("samples", 0),
                                                                _f(prof.get("cpu_ms_per_call"), 4), prof.get("weight")))
            for kind in ("self", "inclusive"):
                L.append("")
                L.append("| %s top %s | %% | us/call |" % (src, kind))
                L.append("|---|---|---|")
                for x in prof.get(kind) or []:
                    L.append("| `%s` | %s | %s |" % (x["frame"][:150].replace("|", "/"), x["pct"], x["us_per_call"]))
        L.append("")
    return "\n".join(L) + "\n"


def cmd_report(a):
    rows = load_rows(a.rows_file)
    doc = {"vendor": a.vendor, "wheel": a.wheel, "host": a.host, "tools": a.tools, "rows": []}
    code = EXIT_OK
    for r in rows:
        rec = collect_row(a.res, r)
        doc["rows"].append(rec)
        if rec["wall_ms_per_call"] is None:
            code = max(code, EXIT_NO_WALL)
            why = "no wall time"
        elif not (rec["perf"] or {}).get("samples") and not (rec["pyspy"] or {}).get("samples"):
            code = max(code, EXIT_NO_PROFILE)
            why = "no frames (perf and py-spy)"
        elif not (rec["nsys"] or {}).get("captured"):
            code = max(code, EXIT_NO_PROFILE)
            why = "no nsys capture"
        else:
            why = "ok"
        print("SUMMARY row=%s wall_ms=%s kernel_ms=%s api_ms=%s host_ms=%s top=%s status=%s" % (
            r.name, _f(rec["wall_ms_per_call"]), _f(rec["split"]["kernel_ms"]), _f(rec["split"]["cuda_api_ms"]),
            _f(rec["split"]["host_outside_cuda_ms"]),
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
            fh.write("%s\t%s\t%s\n" % (r.name, r.race_id, r.phase))
    with open(os.path.join(a.out_dir, "ids.txt"), "w") as fh:
        for rid in dict.fromkeys(r.race_id for r in rows):
            fh.write(rid + "\n")
    print("ROWS rows=%d races=%d" % (len(rows), len(set(r.race_id for r in rows))))
    return 0


def cmd_estimate(a):
    rows = load_rows(a.rows_file)
    passes = [p for p in (a.passes or ",".join(PASSES)).split(",") if p]
    bad = [p for p in passes if p not in PASSES]
    if bad:
        print("unknown passes %s (known: %s)" % (bad, ",".join(PASSES)), file=sys.stderr)
        return 2
    secs = {}
    for kv in a.seconds or []:
        k, _, v = kv.partition("=")
        secs[k] = float(v)
    e = estimate(rows, passes, secs, overhead_s=a.overhead_s)
    for r in e["rows"]:
        print("ESTIMATE row=%s pod_min=%s" % (r["row"], r["pod_min"]))
    print("ESTIMATE total pod_min=%s passes=%s (%s s per race process outside its window; setup %s s)" % (
        e["total_min"], ",".join(passes), a.overhead_s, SETUP_S))
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
    w = sub.add_parser("window")
    w.add_argument("--window", required=True)
    p = sub.add_parser("perf")
    p.add_argument("--script", required=True)
    p.add_argument("--window", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--folded", required=True)
    y = sub.add_parser("pyspy")
    y.add_argument("--collapsed", required=True)
    y.add_argument("--window", required=True)
    y.add_argument("--phase", required=True)
    y.add_argument("--rate", type=float, default=250.0)
    y.add_argument("--under", default=None)
    y.add_argument("--out", required=True)
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
    a = ap.parse_args(argv)
    try:
        return {"rows": cmd_rows, "estimate": cmd_estimate, "window": cmd_window, "perf": cmd_perf,
                "pyspy": cmd_pyspy, "strace": cmd_strace, "report": cmd_report}[a.cmd](a)
    except ValueError as exc:
        print("refused: %s" % exc, file=sys.stderr)
        return 5


if __name__ == "__main__":
    sys.exit(main())
