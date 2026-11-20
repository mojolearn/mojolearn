#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""Attribution of the regressed paper rows: where each row's time goes, on both wheels, same box, same job.

The driver of tools/ops/attribution_job.sh (lane attribution-job, 2026-11-16). Measurement tooling, never product
runtime. The rows (race id, forced phase, profiled arm, timing runs) come from a rows file, tools/ops/attribution_rows.tsv
by default (the nine regressed paper rows); `--rows-file` (the job's AT_ROWS_FILE) profiles any board row.

    attribution_report.py plan [--rows-file F]      TSV the job runs: kind, row, race id, forced phase, own arms
    attribution_report.py estimate --recipes R [--rows-file F] [--wheels N]
                                                    pod minutes per row and in all, from the stored 0.8.55 row times
    attribution_report.py error --out DIR --log L   the first error line of one failed run (job.log)
    attribution_report.py nsys --prefix P --calls N --out J [--kernel-regex RE]
                                                    per-call counts from `nsys stats` CSVs of one capture
    attribution_report.py ncu --csv C --out J       registers/occupancy/duration of the matched kernels
    attribution_report.py report --res RES --vendor V --arms A,B [--rows-file F]
                                                    ATTRIBUTION-<vendor>.md + .json in RES; SUMMARY lines per row and
                                                    the job's exit code (EXIT_* below)

What each row records, per wheel:
  A. the scored run of 20 consecutive calls after one excluded warmup (MOJOLEARN_BENCH_FORCE_CALLS=20 on the row's
     phase): median, min, max, from the board cell (fit) or the driver's FSPEED/FSPEED-INFER line;
  B. per call, from one `nsys profile -t cuda,nvtx --capture-range=nvtx` run that captures exactly calls 2..20 (the
     harness opens the NVTX range around them, tools/bench_board_small_rows.py range_open): kernel launches,
     synchronizations, device allocations and frees, host-to-device and device-to-host copies and bytes, GPU kernel
     time, CUDA API time, and the host time outside CUDA API calls (the range's wall time minus the API time);
  C. for the distance kernels (pinned_distance_register_tile_kernel, smem_distance*): per-kernel duration from the
     CUDA trace, registers per thread, block size and shared memory from the trace, the theoretical occupancy on the
     L40S (sm_89 limits) from those, and ncu's achieved occupancy when ncu has counter permission on the pod.
"""
import argparse
import csv
import json
import math
import os
import re
import statistics
import sys

#: the scored calls of an attribution run (one excluded warmup first, as on the board)
CALLS = 20
#: the distance kernels of part C
KERNEL_REGEX = r"(?i)distan"  # kernel names are cut to 30 characters: `neighbors_checks_pinned_distan...` (the register tile) never matched "distance"


class Row:
    """One row: its race, the phase whose scored run is forced to CALLS calls (`fit`, `infer/<batch>`), the own arm
    the profile times, the timing runs (each an own-arm list in order; None = the race's planned arms), and whether
    part C (distance kernels) applies."""

    def __init__(self, name, race_id, phase, arm, timing=(None,), kernels=False):
        self.name, self.race_id, self.phase, self.arm = name, race_id, phase, arm
        self.timing, self.kernels = list(timing), kernels

    @property
    def batch(self):
        return self.phase.split("/", 1)[1] if "/" in self.phase else None


#: the default row list (attribution_job.sh AT_ROWS_FILE names another file of the same form)
DEFAULT_ROWS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attribution_rows.tsv")


def parse_rows(text, where="rows file"):
    """[Row] from a rows file: per line race id, forced phase, profiled arm, then `name=`, `timing=` (runs split by
    `|`, each `planned` or an own-arm list `a,b`) and `kernels`; `#` comments and blank lines skipped. A malformed
    line, an unknown field or a duplicate name is refused by line number, never dropped."""
    rows, names = [], set()
    for i, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        f = line.split()
        if len(f) < 3 or f[0].count("/") < 2 or not (f[1] == "fit" or f[1].startswith("infer")):
            raise ValueError("%s line %d: want <race id> <fit|infer/<batch>> <arm> [name=] [timing=] [kernels]: %r"
                             % (where, i, raw))
        rid, phase, arm = f[0], f[1], f[2]
        name, timing, kernels = rid.split("/")[1], [None], False
        for opt in f[3:]:
            if opt == "kernels":
                kernels = True
            elif opt.startswith("name="):
                name = opt[5:]
            elif opt.startswith("timing="):
                timing = [None if run == "planned" else [a for a in run.split(",") if a]
                          for run in opt[7:].split("|") if run]
            else:
                raise ValueError("%s line %d: unknown field %r" % (where, i, opt))
        if not name or name in names:
            raise ValueError("%s line %d: row name %r empty or repeated (give name=)" % (where, i, name))
        names.add(name)
        rows.append(Row(name, rid, phase, arm, timing=timing or [None], kernels=kernels))
    if not rows:
        raise ValueError("%s: no rows" % where)
    return rows


def load_rows(path=None):
    path = path or DEFAULT_ROWS_FILE
    with open(path) as fh:
        return parse_rows(fh.read(), path)


ROWS = load_rows()
ROW_BY_NAME = {r.name: r for r in ROWS}


def variant_tag(arms):
    """The directory tag of one timing run: `planned` or the own arms in order joined by `+`."""
    return "planned" if not arms else "+".join(arms)


def plan_lines(rows=ROWS):
    """The job's runs, one TSV line each: kind (time|profile|kernels), row, race id, forced phase, own arms (comma
    list, `-` for the race's planned arms), tag. GPU runs are serial by nature (the job runs them one at a time)."""
    out = []
    for r in rows:
        for arms in r.timing:
            out.append(("time", r.name, r.race_id, r.phase, ",".join(arms) if arms else "-", variant_tag(arms)))
        own = [r.arm] if r.race_id.startswith("trees/") else None   # only tree races take MOJOLEARN_BENCH_OWN_ARMS
        out.append(("profile", r.name, r.race_id, r.phase, ",".join(own) if own else "-", "profile"))
        if r.kernels:
            out.append(("kernels", r.name, r.race_id, r.phase, ",".join(own) if own else "-", "ncu"))
    return ["\t".join(x) for x in out]


# ---- pod minutes from the stored row times ------------------------------------------------------------------------

def _stored(recipes):
    """{(race_id, phase, arm, batch): ms} of the recipes' stored 0.8.55 cells."""
    got = {}
    for rec in recipes.get("recipes", []):
        v = rec.get("v0855") or {}
        ms = v.get("median_ms")
        if isinstance(ms, (int, float)) and not isinstance(ms, bool) and math.isfinite(ms):
            got[(rec["race_id"], rec.get("phase"), rec.get("arm"), rec.get("batch"))] = float(ms)
    return got


def run_seconds(stored, race_id, arms, phase, calls=CALLS, overhead_s=90.0, slowdown=1.0):
    """One race process: start-up, data and evidence (overhead_s) plus, per own arm it times, its fit and each
    inference batch: a warmup and the scored run (`calls` calls on the forced phase, else the small-row rule), and
    no fit warmup when the forced phase is an inference (tools/bench_board_small_rows.py fit_warmup_skipped).
    `slowdown` scales the forced phase's calls (the profiled range); the rest runs untraced."""
    total = 0.0
    for (rid, ph, arm, batch), ms in stored.items():
        if rid != race_id or arm not in arms:
            continue
        key = ph if batch is None or ph == "fit" else "%s/%s" % (ph, batch)
        if key == phase:
            total += ms * (1 + (calls - 1) * slowdown) / 1000.0 + ms / 1000.0
        elif ph == "fit" and phase.startswith("infer"):
            total += ms / 1000.0
        else:
            total += ms * (1 + (CALLS if ms < 10.0 else 1)) / 1000.0
    return overhead_s + total


def estimate(recipes, arms_per_wheel=2, overhead_s=90.0, profile_slowdown=1.5, ncu_s=180.0, setup_s=600.0,
             rows=None):
    """{rows: [{row, runs, seconds, pod_min}], wheel_s, total_min}: the job's pod minutes from the stored times (0.8.55,
    one wheel), per row (`pod_min`: every wheel) and in all, plus tool setup and the nsys stats (setup_s)."""
    stored = _stored(recipes)
    per = []
    for r in (ROWS if rows is None else rows):
        planned = sorted({a for (rid, _, a, _) in stored if rid == r.race_id})
        secs, runs = 0.0, 0
        for arms in r.timing:
            secs += run_seconds(stored, r.race_id, arms or planned or [r.arm], r.phase, overhead_s=overhead_s)
            runs += 1
        prof = [r.arm] if r.race_id.startswith("trees/") else (planned or [r.arm])
        secs += run_seconds(stored, r.race_id, prof, r.phase, overhead_s=overhead_s, slowdown=profile_slowdown)
        runs += 1
        if r.kernels:
            secs += ncu_s
            runs += 1
        per.append({"row": r.name, "runs": runs, "seconds": round(secs, 1),
                    "pod_min": round(secs * arms_per_wheel / 60.0, 1)})
    wheel = sum(p["seconds"] for p in per)
    return {"rows": per, "wheel_s": round(wheel, 1), "wheels": arms_per_wheel,
            "total_min": round((wheel * arms_per_wheel + setup_s) / 60.0, 1),
            "assumptions": {"overhead_s_per_race_process": overhead_s, "profile_slowdown": profile_slowdown,
                            "ncu_s_per_kernel_row": ncu_s, "setup_and_stats_s": setup_s,
                            "times": "stored 0.8.55 L40S cells (recipes v0855); a row without one counts overhead only"}}


# ---- nsys stats CSVs ---------------------------------------------------------------------------------------------

API_KINDS = (
    ("launch", re.compile(r"^(cuLaunch(Kernel|KernelEx|CooperativeKernel)|cudaLaunch(Kernel|KernelExC|CooperativeKernel)"
                          r"|cuGraphLaunch|cudaGraphLaunch)")),
    ("sync", re.compile(r"Synchronize")),
    ("alloc", re.compile(r"^(cuMemAlloc|cuMemHostAlloc|cuMemCreate|cudaMalloc|cudaHostAlloc|cuMemAllocFromPool)")),
    ("free", re.compile(r"^(cuMemFree|cuMemRelease|cudaFree)")),
    ("copy_api", re.compile(r"^(cuMemcpy|cudaMemcpy)")),
    ("memset_api", re.compile(r"^(cuMemset|cudaMemset)")),
    ("host_register", re.compile(r"^(cuMemHost(Un)?[Rr]egister|cudaHost(Un)?[Rr]egister)")),
)
MEM_OPS = (
    ("h2d", re.compile(r"(?i)host-to-device|htod")),
    ("d2h", re.compile(r"(?i)device-to-host|dtoh")),
    ("d2d", re.compile(r"(?i)device-to-device|dtod")),
    ("memset", re.compile(r"(?i)memset")),
)


def api_kind(name):
    """The attribution kind of one CUDA API name (`_v2`/`_ptsz`/`_ptds` suffixes ignored), or `other`."""
    base = re.sub(r"(_v\d+)?(_pt(sz|ds))?$", "", (name or "").strip())
    for kind, rx in API_KINDS:
        if rx.search(base):
            return kind
    return "other"


def _num(row, *names):
    """The first of `names` present in a CSV row, as a float (thousands separators dropped); None when absent."""
    for n in names:
        v = row.get(n)
        if v is None or str(v).strip() == "":
            continue
        try:
            return float(str(v).replace(",", ""))
        except ValueError:
            continue
    return None


def read_csv(path):
    """The rows of one `nsys stats --format csv --output` file; [] when it is missing or empty. Lines before the
    header (some nsys versions print a notice) are skipped: the header is the first line with a comma."""
    if not path or not os.path.isfile(path):
        return []
    with open(path, newline="", errors="replace") as fh:
        lines = fh.read().splitlines()
    start = next((i for i, ln in enumerate(lines) if "," in ln), None)
    if start is None:
        return []
    return list(csv.DictReader(lines[start:]))


def report_file(prefix, report):
    """`<prefix>_<report>.csv`, the name `nsys stats --output <prefix>` gives each report."""
    return "%s_%s.csv" % (prefix, report)


def parse_nsys(prefix, calls, kernel_regex=KERNEL_REGEX):
    """Per-call counts of one capture of `calls` calls (the job captures calls 2..N, so calls = N - 1). Missing
    reports leave their fields None; `captured` is False when the capture holds no CUDA API call (no range came)."""
    calls = max(int(calls), 1)
    out = {"calls_captured": calls, "api": {}, "counts": {}, "kernels": [], "mem": {}}
    api = read_csv(report_file(prefix, "cuda_api_sum"))
    tot = {k: 0.0 for k, _ in API_KINDS}
    tot_ns = {k: 0.0 for k, _ in API_KINDS}
    api_ns = 0.0
    other = 0.0
    for r in api:
        name = r.get("Name") or ""
        n = _num(r, "Num Calls", "Calls", "Count") or 0.0
        ns = _num(r, "Total Time (ns)", "Total Time") or 0.0
        k = api_kind(name)
        out["api"][name] = {"calls": n / calls, "ms": ns / calls / 1e6, "kind": k}
        api_ns += ns
        if k == "other":
            other += n
        else:
            tot[k] += n
            tot_ns[k] += ns
    out["captured"] = bool(api)
    for k in tot:
        out["counts"][k] = tot[k] / calls if api else None
    out["counts"]["other_api"] = other / calls if api else None
    out["api_ms"] = api_ns / calls / 1e6 if api else None
    out["sync_ms"] = tot_ns["sync"] / calls / 1e6 if api else None
    kern = read_csv(report_file(prefix, "cuda_gpu_kern_sum"))
    kns = 0.0
    kinst = 0.0
    rx = re.compile(kernel_regex) if kernel_regex else None
    for r in kern:
        ns = _num(r, "Total Time (ns)", "Total Time") or 0.0
        inst = _num(r, "Instances", "Count") or 0.0
        kns += ns
        kinst += inst
        name = r.get("Name") or ""
        if rx is not None and rx.search(name):
            out["kernels"].append({"name": name, "instances_per_call": inst / calls,
                                   "total_ms_per_call": ns / calls / 1e6,
                                   "avg_us": (_num(r, "Avg (ns)") or 0.0) / 1e3,
                                   "med_us": (_num(r, "Med (ns)") or 0.0) / 1e3,
                                   "min_us": (_num(r, "Min (ns)") or 0.0) / 1e3,
                                   "max_us": (_num(r, "Max (ns)") or 0.0) / 1e3})
    out["kernel_ms"] = kns / calls / 1e6 if kern else None
    out["kernel_instances"] = kinst / calls if kern else None
    size = read_csv(report_file(prefix, "cuda_gpu_mem_size_sum"))
    mtime = read_csv(report_file(prefix, "cuda_gpu_mem_time_sum"))
    for kind, _ in MEM_OPS:
        out["mem"][kind] = {"count": 0.0, "mb": 0.0, "ms": 0.0}
    for r in size:
        op = r.get("Operation") or r.get("Name") or ""
        for kind, rx2 in MEM_OPS:
            if rx2.search(op):
                out["mem"][kind]["count"] += (_num(r, "Count", "Num Calls") or 0.0) / calls
                out["mem"][kind]["mb"] += (_num(r, "Total (MB)", "Total") or 0.0) / calls
                break
    for r in mtime:
        op = r.get("Operation") or r.get("Name") or ""
        for kind, rx2 in MEM_OPS:
            if rx2.search(op):
                out["mem"][kind]["ms"] += (_num(r, "Total Time (ns)", "Total Time") or 0.0) / calls / 1e6
                if not size:
                    out["mem"][kind]["count"] += (_num(r, "Count", "Instances") or 0.0) / calls
                break
    if not size and not mtime:
        out["mem"] = {}
    nv = read_csv(report_file(prefix, "nvtx_sum"))
    window = None
    for r in nv:
        rng = (r.get("Range") or r.get("Name") or "").lstrip(":")
        if rng.startswith("attr-calls"):
            window = (_num(r, "Total Time (ns)", "Total Time") or 0.0) / calls / 1e6
    out["window_ms"] = window
    out["host_ms"] = max(window - out["api_ms"], 0.0) if window is not None and out["api_ms"] is not None else None
    trace = read_csv(report_file(prefix, "cuda_gpu_trace"))
    if trace and rx is not None:
        out["kernel_launch_shapes"] = kernel_shapes(trace, rx)
    return out


def _int(row, *names):
    v = _num(row, *names)
    return int(v) if v is not None else None


def kernel_shapes(trace, rx):
    """{kernel: [{block, grid, regs, smem_bytes, launches, occupancy}]} of the trace rows whose name matches: one
    entry per distinct launch shape (block, registers, shared memory)."""
    seen = {}
    for r in trace:
        name = r.get("Name") or ""
        if not rx.search(name):
            continue
        block = (_int(r, "BlkX") or 1) * (_int(r, "BlkY") or 1) * (_int(r, "BlkZ") or 1)
        grid = (_int(r, "GrdX") or 1) * (_int(r, "GrdY") or 1) * (_int(r, "GrdZ") or 1)
        regs = _int(r, "Reg/Trd")
        smem = ((_num(r, "StcSMem (MB)") or 0.0) + (_num(r, "DymSMem (MB)") or 0.0)) * 1024 * 1024
        key = (name, block, regs, int(round(smem)))
        e = seen.setdefault(key, {"block": block, "grid": grid, "regs": regs, "smem_bytes": int(round(smem)),
                                  "launches": 0, "durations_ns": []})
        e["launches"] += 1
        d = _num(r, "Duration (ns)", "Duration")
        if d is not None:
            e["durations_ns"].append(d)
    out = {}
    for (name, _, _, _), e in seen.items():
        durs = e.pop("durations_ns")
        e["med_us"] = statistics.median(durs) / 1e3 if durs else None
        e["occupancy"] = occupancy_sm89(e["block"], e["regs"], e["smem_bytes"])
        out.setdefault(name, []).append(e)
    return out


#: L40S (sm_89) per-SM limits, CUDA C Programming Guide "Compute capability 8.9"
SM89 = {"max_warps": 48, "max_blocks": 24, "regs": 65536, "reg_unit": 256, "smem": 100 * 1024,
        "smem_reserved_per_block": 1024, "warp_alloc": 4}


def occupancy_sm89(block_threads, regs_per_thread, smem_bytes, lim=SM89):
    """Theoretical occupancy (resident warps / max warps) of one launch shape on sm_89, and the limiting resource.
    None when the block size is unknown."""
    if not block_threads or block_threads <= 0:
        return None
    warps = -(-int(block_threads) // 32)
    by = {"warps": lim["max_warps"] // warps, "blocks": lim["max_blocks"]}
    if regs_per_thread:
        per_warp = -(-int(regs_per_thread) * 32 // lim["reg_unit"]) * lim["reg_unit"]
        warps_by_regs = (lim["regs"] // per_warp) // lim["warp_alloc"] * lim["warp_alloc"]
        by["registers"] = warps_by_regs // warps
    if smem_bytes:
        by["shared_memory"] = lim["smem"] // (int(smem_bytes) + lim["smem_reserved_per_block"])
    blocks = min(by.values())
    limiter = min(by, key=lambda k: by[k])
    return {"blocks_per_sm": blocks, "occupancy": blocks * warps / lim["max_warps"], "limiter": limiter}


# ---- ncu ---------------------------------------------------------------------------------------------------------

NCU_METRICS = {"Registers Per Thread": "regs", "Theoretical Occupancy": "theoretical_occupancy_pct",
               "Achieved Occupancy": "achieved_occupancy_pct", "Duration": "duration",
               "Block Size": "block", "Grid Size": "grid", "Block Limit Registers": "block_limit_registers",
               "Block Limit Shared Mem": "block_limit_smem", "Waves Per SM": "waves_per_sm"}


def parse_ncu(path):
    """{status, kernels: {name: {metric: value}}} from `ncu --csv --page details` output. status is `ok`,
    `no-permission` (ERR_NVGPUCTRPERM: the pod does not expose GPU performance counters to a non-admin), `no-kernels`
    or `missing`."""
    if not path or not os.path.isfile(path):
        return {"status": "missing", "kernels": {}}
    with open(path, errors="replace") as fh:
        text = fh.read()
    if "ERR_NVGPUCTRPERM" in text:
        return {"status": "no-permission", "kernels": {}}
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith('"ID"') or "Kernel Name" in ln), None)
    kernels = {}
    if start is not None:
        for r in csv.DictReader(lines[start:]):
            m = NCU_METRICS.get((r.get("Metric Name") or "").strip())
            if not m:
                continue
            name = r.get("Kernel Name") or "?"
            unit = (r.get("Metric Unit") or "").strip()
            raw = (r.get("Metric Value") or "").replace(",", "")
            try:
                val = float(raw)
            except ValueError:
                val = raw
            if m == "duration" and isinstance(val, float):
                val = val / {"nsecond": 1e3, "usecond": 1.0, "msecond": 1e-3, "second": 1e-6}.get(unit, 1.0)
                m = "duration_us"
            kernels.setdefault(name, {}).setdefault(m, []).append(val)
    out = {}
    for name, ms in kernels.items():
        out[name] = {k: (statistics.median(v) if v and all(isinstance(x, float) for x in v) else v[0])
                     for k, v in ms.items()}
    return {"status": "ok" if out else "no-kernels", "kernels": out}


# ---- timing (part A) ---------------------------------------------------------------------------------------------

_KV = re.compile(r"(\S+?)=(\S*)")


def parse_driver_lines(path, phase):
    """{arm: [{ms, calls, min_ms, max_ms, order}]} from a tree driver log: FSPEED lines for `fit`, FSPEED-INFER lines
    of the batch for `infer/<batch>`. `order` is the arm's first-appearance index (the arm-order test)."""
    head = "FSPEED" if phase == "fit" else "FSPEED-INFER"
    batch = phase.split("/", 1)[1] if "/" in phase else None
    out, order = {}, []
    if not path or not os.path.isfile(path):
        return out
    with open(path, errors="replace") as fh:
        for line in fh:
            first, _, rest = line.rstrip("\n").partition(" ")
            if first != head:
                continue
            f = dict(_KV.findall(rest))
            if batch is not None and f.get("batch") != batch:
                continue
            arm = f.get("arm")
            try:
                ms = float(f["ms"])
            except (KeyError, ValueError):
                continue
            if arm not in order:
                order.append(arm)
            rec = {"ms": ms, "calls": int(f.get("calls", 1)), "order": order.index(arm)}
            for k in ("min_ms", "max_ms"):
                rec[k] = float(f[k]) if k in f else ms
            out.setdefault(arm, []).append(rec)
    return out


def board_cell(board, race_id, phase, arm):
    """{ms, calls, min_ms, max_ms, status} of the board cell of `arm` (fit: `cells`; infer: `infer_cells` of the
    batch), or None."""
    race = (board.get("races") or {}).get(race_id) or {}
    batch = phase.split("/", 1)[1] if "/" in phase else None
    cells = race.get("cells") if phase == "fit" else race.get("infer_cells")
    for c in cells or []:
        if c.get("arm") != arm or (batch is not None and c.get("batch") != batch):
            continue
        ms = c.get("median_ms")
        if ms is None:
            return {"status": c.get("status") or "no-time"}
        return {"ms": ms, "calls": c.get("calls"), "min_ms": c.get("min_ms", ms), "max_ms": c.get("max_ms", ms),
                "status": c.get("status")}
    return None


def run_timing(run_dir, row, arms):
    """{arm: timing} of one timing run directory (bench_board --out): the board cell, else the driver log's line."""
    out = {}
    board = {}
    bj = os.path.join(run_dir, "board.json")
    if os.path.isfile(bj):
        with open(bj) as fh:
            board = json.load(fh)
    logs = []
    race = (board.get("races") or {}).get(row.race_id) or {}
    if race.get("log"):
        logs.append(os.path.join(run_dir, race["log"]))
    logdir = os.path.join(run_dir, "logs")
    if os.path.isdir(logdir):
        logs += sorted(os.path.join(logdir, f) for f in os.listdir(logdir) if f.endswith(".log"))
    lines = {}
    for lg in logs:
        for a, recs in parse_driver_lines(lg, row.phase).items():
            lines.setdefault(a, recs)
    for arm in arms:
        cell = board_cell(board, row.race_id, row.phase, arm) if board else None
        if (cell is None or "ms" not in cell) and lines.get(arm):
            rec = dict(lines[arm][-1])
            rec["source"] = "driver log"
            cell = rec
        elif cell is not None:
            cell["source"] = "board cell"
            if lines.get(arm):
                cell["order"] = lines[arm][-1]["order"]
        out[arm] = cell
    return out


# ---- the report --------------------------------------------------------------------------------------------------

def _load(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _f(x, nd=2):
    if x is None:
        return "-"
    if isinstance(x, int) and not isinstance(x, bool):
        x = float(x)
    if isinstance(x, float):
        if abs(x) >= 100:
            return "%.0f" % x
        return ("%." + str(nd) + "f") % x
    return str(x)


def components(t, p):
    """The per-call components of one wheel: median ms (A), window, kernel, sync wait, other API, host (B)."""
    c = {"median_ms": (t or {}).get("ms")}
    if p and p.get("captured"):
        c.update(window_ms=p.get("window_ms"), kernel_ms=p.get("kernel_ms"), sync_ms=p.get("sync_ms"),
                 api_other_ms=(p["api_ms"] - (p.get("sync_ms") or 0.0)) if p.get("api_ms") is not None else None,
                 host_ms=p.get("host_ms"))
    return c


def reading(row, arms, timing, prof):
    """One line: where the time goes (one wheel) or what changed (candidate against the prior): kernel time vs
    launches vs syncs vs allocations vs host."""
    def comp(a):
        return components(timing.get(a), prof.get(a))

    def counts(a):
        return ((prof.get(a) or {}).get("counts") or {})
    if len(arms) >= 2 and all(comp(a).get("median_ms") is not None for a in arms[:2]):
        new, old = arms[0], arms[1]
        cn, co = comp(new), comp(old)
        d = cn["median_ms"] - co["median_ms"]
        parts = []
        for k, label in (("kernel_ms", "kernel time"), ("sync_ms", "sync wait"), ("api_other_ms", "other CUDA API"),
                         ("host_ms", "host outside CUDA")):
            if cn.get(k) is not None and co.get(k) is not None:
                parts.append((cn[k] - co[k], label))
        parts.sort(key=lambda x: -abs(x[0]))
        txt = "%s %+.2f ms (%+.0f%%) vs %s" % (new, d, 100.0 * d / co["median_ms"] if co["median_ms"] else 0.0, old)
        if parts:
            txt += "; largest change: %s %+.2f ms" % (parts[0][1], parts[0][0])
        cnt = []
        for k, label in (("launch", "launches"), ("sync", "syncs"), ("alloc", "allocs"), ("free", "frees")):
            a, b = counts(new).get(k), counts(old).get(k)
            if a is not None and b is not None and abs(a - b) >= 0.5:
                cnt.append("%s %.0f->%.0f" % (label, b, a))
        hn, ho = (prof.get(new) or {}).get("mem") or {}, (prof.get(old) or {}).get("mem") or {}
        for k in ("h2d", "d2h"):
            if k in hn and k in ho and abs(hn[k]["count"] - ho[k]["count"]) >= 0.5:
                cnt.append("%s copies %.0f->%.0f" % (k.upper(), ho[k]["count"], hn[k]["count"]))
        if cnt:
            txt += "; per call " + ", ".join(cnt)
        if not parts:
            txt += "; no profile on both wheels (counts unavailable)"
        return txt
    for a in arms:
        c = comp(a)
        if c.get("median_ms") is None:
            continue
        parts = [(c.get(k), label) for k, label in (("kernel_ms", "kernel time"), ("sync_ms", "sync wait"),
                                                    ("api_other_ms", "other CUDA API"), ("host_ms", "host outside CUDA"))
                 if c.get(k) is not None]
        if not parts:
            return "%s %.2f ms; no profile (counts unavailable)" % (a, c["median_ms"])
        parts.sort(key=lambda x: -x[0])
        return "%s %.2f ms; most of it %s (%.2f ms)" % (a, c["median_ms"], parts[0][1], parts[0][0])
    return "no timing"


def _spread(t):
    if not t or t.get("ms") is None:
        return "-"
    return "%s..%s (%sx)" % (_f(t.get("min_ms")), _f(t.get("max_ms")),
                             _f(t["max_ms"] / t["min_ms"]) if t.get("min_ms") else "-")


def collect(res, arms, rows_in=None):
    """{rows: {row: {...}}} from a results dir: time-<arm>/<row>/<tag>/ (bench_board out), profile-<arm>/<row>.json
    (parse_nsys output), ncu-<arm>/<row>.json (parse_ncu output)."""
    rows = {}
    for row in (ROWS if rows_in is None else rows_in):
        rec = {"race_id": row.race_id, "phase": row.phase, "arm": row.arm, "timing": {}, "order_test": {},
               "profile": {}, "ncu": {}, "runs": {}}
        for wheel in arms:
            for arms_list in row.timing:
                tag = variant_tag(arms_list)
                d = os.path.join(res, "time-%s" % wheel, row.name, tag)
                if not os.path.isdir(d):
                    rec["runs"].setdefault(wheel, {})[tag] = {"error": _error_note(d) or "not run"}
                    continue
                got = run_timing(d, row, arms_list or [row.arm])
                missing = [a for a in (arms_list or [row.arm]) if (got.get(a) or {}).get("ms") is None]
                rec["runs"].setdefault(wheel, {})[tag] = (
                    {"error": "no time for %s: %s" % (",".join(missing), _error_note(d) or first_error(d, d + ".log"))}
                    if missing else {"ok": True})
                if arms_list is None or arms_list == [row.arm]:
                    rec["timing"][wheel] = got.get(row.arm)
                if arms_list and len(arms_list) > 1:
                    rec["order_test"].setdefault(tag, {})[wheel] = got
            p = _load(os.path.join(res, "profile-%s" % wheel, "%s.json" % row.name))
            if p is not None:
                rec["profile"][wheel] = p
            if not (p or {}).get("captured"):
                note = _error_note(os.path.join(res, "profile-%s" % wheel, "%s-board" % row.name))
                rec.setdefault("profile_error", {})[wheel] = (
                    note or ("capture empty (no NVTX range)" if p is not None else "no nsys capture"))
            n = _load(os.path.join(res, "ncu-%s" % wheel, "%s.json" % row.name))
            if n is not None:
                rec["ncu"][wheel] = n
        rec["reading"] = reading(row, arms, rec["timing"], rec["profile"])
        rows[row.name] = rec
    return {"rows": rows}


def render(doc, arms, header_lines, rows=None):
    """ATTRIBUTION-<vendor>.md text."""
    out = ["# Attribution of board rows", ""] + header_lines + [""]
    if doc.get("summary"):
        out += ["## Summary", ""] + ["- " + ln for ln in doc["summary"]["lines"]] + [""]
    for row in (ROWS if rows is None else rows):
        rec = doc["rows"][row.name]
        out += ["## %s (%s, %s, %s arm)" % (row.name, row.race_id, row.phase, row.arm), "",
                "Reading: " + rec["reading"], ""]
        out.append("| per call | " + " | ".join(arms) + " |")
        out.append("|---|" + "---|" * len(arms))

        def line(label, fn):
            out.append("| %s | %s |" % (label, " | ".join(fn(a) for a in arms)))
        tim, prof = rec["timing"], rec["profile"]
        line("median ms (%d calls after 1 warmup)" % CALLS, lambda a: _f((tim.get(a) or {}).get("ms")))
        line("min..max ms (max/min)", lambda a: _spread(tim.get(a)))
        line("scored calls / source", lambda a: "%s / %s" % ((tim.get(a) or {}).get("calls", "-"),
                                                             (tim.get(a) or {}).get("source", "-")))

        def pv(a, k):
            p = prof.get(a) or {}
            return p.get(k) if p.get("captured") else None

        def cnt(a, k):
            p = prof.get(a) or {}
            return (p.get("counts") or {}).get(k) if p.get("captured") else None

        def mem(a, k):
            p = prof.get(a) or {}
            m = (p.get("mem") or {}).get(k)
            return "%s (%s MB, %s ms)" % (_f(m["count"], 1), _f(m["mb"]), _f(m["ms"], 3)) if m and p.get("captured") \
                else "-"
        line("profiled window ms (nsys range / calls)", lambda a: _f(pv(a, "window_ms")))
        line("GPU kernel ms", lambda a: _f(pv(a, "kernel_ms"), 3))
        line("kernel launches", lambda a: _f(cnt(a, "launch"), 1))
        line("kernels executed", lambda a: _f(pv(a, "kernel_instances"), 1))
        line("synchronizations (wait ms)", lambda a: "%s (%s)" % (_f(cnt(a, "sync"), 1), _f(pv(a, "sync_ms"), 3)))
        line("device allocations / frees", lambda a: "%s / %s" % (_f(cnt(a, "alloc"), 1), _f(cnt(a, "free"), 1)))
        line("host register calls", lambda a: _f(cnt(a, "host_register"), 1))
        line("H2D copies (MB, GPU ms)", lambda a: mem(a, "h2d"))
        line("D2H copies (MB, GPU ms)", lambda a: mem(a, "d2h"))
        line("D2D copies / memsets", lambda a: "%s / %s" % (mem(a, "d2d"), mem(a, "memset")))
        line("CUDA API ms (all calls)", lambda a: _f(pv(a, "api_ms"), 3))
        line("host ms outside CUDA API", lambda a: _f(pv(a, "host_ms"), 3))
        out.append("")
        if rec["order_test"]:
            out += ["Arm-order test (median ms of each arm, %d calls; the arm timed first is marked *):" % CALLS, "",
                    "| run | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
            for tag, per in rec["order_test"].items():
                cells = []
                for a in arms:
                    got = per.get(a) or {}
                    cells.append(", ".join("%s%s %s" % (arm, "*" if (c or {}).get("order") == 0 else "",
                                                        _f((c or {}).get("ms")))
                                           for arm, c in got.items()) or "-")
                out.append("| %s | %s |" % (tag, " | ".join(cells)))
            out.append("")
        kern = {}
        for a in arms:
            for k in ((prof.get(a) or {}).get("kernels") or []):
                kern.setdefault(k["name"], {})[a] = k
        if row.kernels:
            out += ["Distance kernels (per call; duration from the CUDA trace; registers, block and shared memory "
                    "from the trace; occupancy theoretical on sm_89 unless ncu achieved is shown):", ""]
            if not kern:
                out += ["- no kernel matching %s in the capture" % KERNEL_REGEX, ""]
            for name, per in sorted(kern.items()):
                out.append("- `%s`" % name[:160])
                for a in arms:
                    k = per.get(a)
                    if not k:
                        out.append("  - %s: not launched" % a)
                        continue
                    shapes = ((prof.get(a) or {}).get("kernel_launch_shapes") or {}).get(name) or []
                    sh = "; ".join("block %s regs %s smem %s B occ %s (%s)" % (
                        s["block"], s["regs"], s["smem_bytes"],
                        _f((s.get("occupancy") or {}).get("occupancy")), (s.get("occupancy") or {}).get("limiter"))
                        for s in shapes) or "no trace shape"
                    n = (((rec["ncu"].get(a) or {}).get("kernels")) or {})
                    nk = next((v for kname, v in n.items() if kname and (kname in name or name in kname)), None)
                    ncu_txt = (" ncu: regs %s, theoretical %s%%, achieved %s%%, %s us" % (
                        _f(nk.get("regs")), _f(nk.get("theoretical_occupancy_pct")),
                        _f(nk.get("achieved_occupancy_pct")), _f(nk.get("duration_us"))) if nk else
                        " ncu: %s" % ((rec["ncu"].get(a) or {}).get("status") or "not run"))
                    out.append("  - %s: %s launches, %s ms total; avg %s us, med %s, min %s, max %s; %s;%s" % (
                        a, _f(k["instances_per_call"], 1), _f(k["total_ms_per_call"], 3), _f(k["avg_us"]),
                        _f(k["med_us"]), _f(k["min_us"]), _f(k["max_us"]), sh, ncu_txt))
            out.append("")
    return "\n".join(out) + "\n"


#: the job's exit codes for a finished report (attribution_job.sh exits with the report's code; its own earlier exits:
#: 2 refused to start, 3 not an NVIDIA box / other GPU, 4 no venv, 5 no plan, 124 terminated, EXIT_DISK disk full)
EXIT_OK = 0                 # every row has its timing runs and a captured profile on every wheel
EXIT_NO_PROFILE = 6         # every row timed; one or more rows without a captured profile on some wheel
EXIT_NO_TIMING = 7          # one or more rows without a timing run's time on some wheel
ERROR_RX = re.compile(r"Traceback|\b\w*(error|exception)\b|\b(refused|failed)\b|No reports were generated", re.I)


def _error_note(out_dir):
    """The first error line attribution_job.sh recorded for one run (`<out>.error`), or None."""
    try:
        with open(out_dir + ".error", errors="replace") as fh:
            return fh.readline().strip() or None
    except OSError:
        return None


def first_error(out_dir, log=None):
    """The first error line of one failed bench_board run: the board's preflight refusal (its problems), a failed
    race's failure, else the first error-looking line of the run's own log and then of the driver logs under the
    run directory, else the log's last line. fb3-attr-tool (2026-11-17): the job printed only the last log line
    (`bench_board: board .../BOARD.md`), which hid the preflight refusal of the tree infer rows."""
    board = _load(os.path.join(out_dir, "board.json")) or {}
    for rid, rec in sorted((board.get("failed_preflight_recipes") or {}).items()):
        probs = (rec or {}).get("problems") or []
        if probs:
            return "preflight refused %s: %s" % (rid, "; ".join(str(p) for p in probs))[:400]
    for rid, rec in sorted((board.get("races") or {}).items()):
        if isinstance(rec, dict) and rec.get("failure"):
            return "race %s failed: %s" % (rid, rec["failure"])[:400]
    logs = [log] if log else []
    for root, _, files in os.walk(out_dir):
        logs += sorted(os.path.join(root, f) for f in files if f.endswith(".log"))
    last = None
    for lg in logs:
        try:
            with open(lg, errors="replace") as fh:
                for ln in fh:
                    ln = ln.strip()
                    if not ln:
                        continue
                    if lg == log:
                        last = ln
                    if ERROR_RX.search(ln) and "tcmalloc" not in ln:
                        return ("%s: %s" % (os.path.basename(lg), ln))[:400]
        except OSError:
            continue
    return ("no error line; last line: %s" % last)[:400] if last else "no error line and no log"


def summarize(doc, arms):
    """{lines, code, failed}: one SUMMARY line per row (timing and profile on each wheel, ncu shown, never failing)
    and the job's exit code (EXIT_*)."""
    lines, no_time, no_prof = [], [], []
    for name, rec in doc["rows"].items():
        bad_t = ["%s %s: %s" % (w, tag, r["error"]) for w in arms for tag, r in (rec["runs"].get(w) or {}).items()
                 if r.get("error")]
        bad_p = ["%s: %s" % (w, rec.get("profile_error", {})[w]) for w in arms if w in rec.get("profile_error", {})]
        ncu = ["%s %s" % (w, (rec["ncu"].get(w) or {}).get("status", "?")) for w in arms if rec["ncu"].get(w)]
        if bad_t:
            no_time.append(name)
        if bad_p:
            no_prof.append(name)
        lines.append("SUMMARY row=%s timing=%s profile=%s%s" % (
            name, "ok" if not bad_t else "FAILED(" + " | ".join(bad_t) + ")",
            "ok" if not bad_p else "FAILED(" + " | ".join(bad_p) + ")",
            (" ncu=" + ",".join(ncu)) if ncu else ""))
    code = EXIT_NO_TIMING if no_time else EXIT_NO_PROFILE if no_prof else EXIT_OK
    lines.append("SUMMARY rows=%d complete=%d no_timing=%s no_profile=%s exit=%d" % (
        len(doc["rows"]), sum(1 for n in doc["rows"] if n not in no_time and n not in no_prof),
        ",".join(no_time) or "-", ",".join(no_prof) or "-", code))
    return {"lines": lines, "code": code, "no_timing": no_time, "no_profile": no_prof}


def cores_summary(path):
    """`busy max B of N (mean M) over K samples` from job/cores.log, or None."""
    vals, total = [], None
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        for ln in fh:
            m = re.search(r"busy=([0-9.]+) total=(\d+)", ln)
            if m:
                vals.append(float(m.group(1)))
                total = int(m.group(2))
    if not vals:
        return None
    return "busy cores: max %.1f of %d, mean %.1f over %d samples (job/cores.log)" % (
        max(vals), total, statistics.mean(vals), len(vals))


def cmd_report(a):
    arms = [x for x in a.arms.split(",") if x]
    rows = load_rows(a.rows_file)
    doc = collect(a.res, arms, rows)
    doc["summary"] = summarize(doc, arms)
    j = os.path.join(a.res, "job")
    head = ["- GPU work is SERIAL: one race process on the GPU at a time, nothing else running (timing must not "
            "share the GPU); the tool installs, nsys stats and this report use every allocated core."]
    probe = _load(os.path.join(j, "host-probe.json")) or {}
    head.append("- host %s, %s, GPU %s" % (probe.get("host_id", "?"), probe.get("cpu_model", "?"),
                                          (open(os.path.join(j, "gpu.txt")).read().strip().replace("\n", "; ")
                                           if os.path.isfile(os.path.join(j, "gpu.txt")) else "?")))
    for arm in arms:
        v = os.path.join(j, "version-%s.txt" % arm)
        head.append("- wheel %s: %s" % (arm, open(v).read().strip() if os.path.isfile(v) else "?"))
    tools = os.path.join(j, "tools.txt")
    if os.path.isfile(tools):
        head += ["- tools: " + ln.strip() for ln in open(tools) if ln.strip()]
    cs = cores_summary(os.path.join(j, "cores.log"))
    if cs:
        head.append("- " + cs)
    head.append("- B counts are per call over calls 2..%d of one `nsys profile -t cuda,nvtx --capture-range=nvtx` "
                "run (profiled; its times are not the A times); the reading compares %s against %s." %
                (CALLS, arms[0] if arms else "?", arms[1] if len(arms) > 1 else "-"))
    doc.update(arms=arms, header=head)
    md = render(doc, arms, head, rows)
    with open(os.path.join(a.res, "ATTRIBUTION-%s.md" % a.vendor), "w") as fh:
        fh.write(md)
    with open(os.path.join(a.res, "ATTRIBUTION-%s.json" % a.vendor), "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True, default=str)
    print("ATTRIBUTION rows=%d with_timing=%d with_profile=%d" % (
        len(doc["rows"]), sum(1 for r in doc["rows"].values() if r["timing"]),
        sum(1 for r in doc["rows"].values() if any((p or {}).get("captured") for p in r["profile"].values()))))
    print("\n".join(doc["summary"]["lines"]))
    return doc["summary"]["code"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("plan")
    pl.add_argument("--rows-file")
    e = sub.add_parser("estimate")
    e.add_argument("--recipes", required=True)
    e.add_argument("--overhead-s", type=float, default=90.0)
    e.add_argument("--rows-file")
    e.add_argument("--wheels", type=int, default=2)
    er = sub.add_parser("error")
    er.add_argument("--out", required=True)
    er.add_argument("--log")
    n = sub.add_parser("nsys")
    n.add_argument("--prefix", required=True)
    n.add_argument("--calls", type=int, required=True)
    n.add_argument("--out", required=True)
    n.add_argument("--kernel-regex", default=KERNEL_REGEX)
    c = sub.add_parser("ncu")
    c.add_argument("--csv", required=True)
    c.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("--res", required=True)
    r.add_argument("--vendor", default="nvidia")
    r.add_argument("--arms", required=True)
    r.add_argument("--rows-file")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        print("\n".join(plan_lines(load_rows(a.rows_file))))
        return 0
    if a.cmd == "error":
        print(first_error(a.out, a.log))
        return 0
    if a.cmd == "estimate":
        with open(a.recipes) as fh:
            est = estimate(json.load(fh), arms_per_wheel=a.wheels, overhead_s=a.overhead_s,
                           rows=load_rows(a.rows_file))
        for p in est["rows"]:
            print("ESTIMATE row=%s runs_per_wheel=%d seconds_per_wheel=%.0f pod_min=%.1f" % (
                p["row"], p["runs"], p["seconds"], p["pod_min"]))
        print("ESTIMATE total_min=%.0f wheels=%d (%s)" % (est["total_min"], est["wheels"],
                                                          json.dumps(est["assumptions"], sort_keys=True)))
        return 0
    if a.cmd == "nsys":
        got = parse_nsys(a.prefix, a.calls, a.kernel_regex)
        with open(a.out, "w") as fh:
            json.dump(got, fh, indent=1, sort_keys=True)
        print("NSYS captured=%s launches=%s syncs=%s kernel_ms=%s" % (
            got["captured"], _f(got["counts"].get("launch"), 1), _f(got["counts"].get("sync"), 1),
            _f(got.get("kernel_ms"), 3)))
        return 0 if got["captured"] else 3
    if a.cmd == "ncu":
        got = parse_ncu(a.csv)
        with open(a.out, "w") as fh:
            json.dump(got, fh, indent=1, sort_keys=True)
        print("NCU status=%s kernels=%d" % (got["status"], len(got["kernels"])))
        return 0
    return cmd_report(a)


if __name__ == "__main__":
    sys.exit(main())
