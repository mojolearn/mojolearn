#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE MEASUREMENT SMOKE OF THE RELEASE (Andrew 2026-11-06: "make the smoke on the measurements a test as part of
our pre release process"): every recipe of the frozen measurement plan, once, at its smallest declared shape, OUR GPU
arm only, against the INSTALLED release wheel bytes, so that no release ships with the measurement tooling broken
against its public fit / predict / transform doors.

WHAT RUNS. The plan is the board's own: `bench_board.plan_races(vendor, ["identical"])` over every family, at the
board's smoke shape (`bench_board.SMOKE_ROWS` rows, the `small` neural shape) -- the 653 recipes the full
measurement campaign times on NVIDIA and AMD (tools/gpu_gap_campaign.py `selected_plan(..., cpu_fallback=True)`,
tools/test_measurement_coverage.py keeps the two equal). The runner is the board's own measurement check,
`tools/bench_board.py --smoke` (one round, the smoke rows, the small neural shape, its own per-arm budget and per-race
deadline), never a second runner: this tool only splits the plan into batches (`--race-ids`), runs them as
independent processes, and judges. OUR ARM ONLY: no --with-opponents, no --retime-opponents, no opponent store, no
opponent installs (bench_board's default is our GPU arm(s) only; an unstored opponent is recorded under
skipped_opponents and never run). Identical mode only. Nothing here times a CPU or host route.

HOW IT IS JUDGED. Never on speed: no timing from this run is compared, kept in the summary or written to any board
(the per-batch smoke boards stay in the box's work directory; only the summary comes home). A recipe PASSES when
  * bench_board's own smoke verdict (`bench_board.smoke_verdict`) has no failure for it, and
  * every own arm cell it ran holds the fields the board ingests: status ok, at least one quality value and every
    quality value a finite number (bench_board.QUALITY_EXEMPT lanes excepted, as on the board), the output digest
    (`hash`), mode `identical`, the library version and the device.
A recipe whose own arm the binding declares unsupported on this vendor through the public capability record
(`bench_board.unsupported_our_arms`, the same record the board shows) is UNSUPPORTED: listed by name, not a failure,
exactly as the board's own smoke treats it. FAIL CLOSED: a recipe that raises, is refused, is skipped, times out,
was never reached (its batch died, or the phase ceiling expired) or yields no record is a FAIL with a one-line reason.
The release verdict (tools/release_identity_wheels.py `single_gate_ok`, `summary_errors` below) requires every
planned recipe to be PASS or UNSUPPORTED.

UNCONDITIONAL. tools/ops/release_gate_job.sh phase `measuresmoke` runs it in every release gate job on NVIDIA and AMD,
whatever changed: no change detection, no selector, no reuse. Apple has no measurement smoke in its gate (Apple timing
belongs to the M3 Ultra's own tools), so none is added there.

PARALLEL. Batches are independent bench_board processes sharing the one GPU. Workers =
min(cores, host GB available / HOST_GB_PER_WORKER, device GB / DEVICE_GB_PER_WORKER, batches); the summary records the
bound that limited it and the measured peak device memory, so the next release can set the per-worker figures from
this run. The plan runs in STAGING GROUPS (one per board dataset, the neural recipes in one): a group's data is staged,
its untimed block prep runs once into the group's own cache (bench_board `--prep-only`, so no two batches write the
same block), its batches run, then its cache, its passing batches' scratch and the data no later group reads are
deleted. Groups overlap while the footprint (data this run fetched + its work dir) stays under --max-work-gb; each
race's own scratch goes as soon as its record is written (bench_board --drop-race-scratch). Disk lines:
`MEASURESMOKE disk free=<GB> footprint=<GB> after <what>`.

Usage:
  release_measurement_smoke.py plan --vendor nvidia                      # the recipe count and plan digest
  release_measurement_smoke.py data-plan --vendor nvidia --out data.json # on the Mac: presigned GETs (no secret leaves)
  release_measurement_smoke.py stage --data-plan data.json               # on the box: fetch missing, verify pins
  release_measurement_smoke.py run --vendor nvidia --python VENV/bin/python --out DIR --work DIR \
      --data-plan data.json [--max-work-gb 2.1]                          # stages per group, cleans after it
  release_measurement_smoke.py budget --vendor nvidia                    # the work-dir bound (0 bytes staged)

THE INPUTS (Andrew 2026-11-08: "the smoke should be light"; "maybe the smoke fixtures should be the same as the
verify, for those algos"; then: the smoke stages NO data at all). Over a tree whose board takes `--inputs`
(bench_board.py, tools/smoke_inputs.py), every batch runs `--inputs synthetic --inputs-map <work>/smoke_inputs.json`:
a recipe the map puts on the verify fixture preps from the identity verification's fixture `base`; a loader or corpus
recipe, and a NO VERIFY LANE recipe that reads a file, opens a synthetic file in that file's own schema at the
loader-side smoke size, generated on the box and deleted with the race's scratch. No data plan entry is read, no
stager is called, nothing is downloaded: the header prints SMOKE_INPUTS_LINE, and the work dir stays under
WORK_MAX_BYTES (budget(), printed too). Over an older tree the run keeps the datasets, the data plan and the 8 GB cap
(the staging code stays for that and for the real measurement runs).
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

SCHEMA = "mojolearn.release-measurement-smoke.v1"
SUMMARY = "measuresmoke.json"
MODE = "identical"
VENDORS = ("nvidia", "amd")
#: per batch process, until this run's own measured figure replaces them (the summary records peak device memory).
#: Basis: the stored full-size AMD MI325X campaign (full-measurement-amd-20261027-r3, 162 own cells): own-cell peak
#: GPU memory median 1.9 GB (max 8.6 GB, the full-shape LM; the smoke runs the small shape), peak host memory median
#: 2.1 GB (max 7.4 GB). One batch runs one race at a time, so twice the full-size median covers a smoke-size batch.
DEVICE_GB_PER_WORKER = 4.0
HOST_GB_PER_WORKER = 4.0
#: the worker bound when the device memory cannot be read (a 24 GB card / DEVICE_GB_PER_WORKER, rounded down)
UNKNOWN_DEVICE_WORKERS = 6
#: batches per worker (load balance: a slow batch does not hold the phase while the other workers idle)
BATCHES_PER_WORKER = 3
#: a batch process's own setup (venv checks, the GPU-set probe, the evidence source snapshot) before its first race
BATCH_SETUP_S = 600
#: the whole phase's ceiling; a batch not finished by then is killed and its unrun recipes FAIL
PHASE_CEILING_S = 5400
#: recipe id prefixes that run in one batch of their own (see split)
SERIAL_PREFIXES = ("neural/",)
#: the own-arm cell fields the board ingests besides status / quality (provenance)
OWN_FIELDS = ("hash", "mode", "library_version", "device")
PASS, FAIL, UNSUPPORTED = "PASS", "FAIL", "UNSUPPORTED"
#: the refusal when no summary came home. It comes only from a box: --reuse-unchanged (a vendor composed on the Mac
#: with no box job) never has one, by design (the smoke is unconditional).
NO_SUMMARY = ("no measurement-smoke summary: it is produced only by the release gate box job, which runs it last "
              "(tools/ops/release_gate_job.sh phase measuresmoke, launched by the gate-<vendor> step; its final upload "
              "brings it home); a vendor composed on the Mac under --reuse-unchanged has none. To pass, rerun the "
              "gate-<vendor> step so that box job runs (with or without --reuse-unchanged); nothing on the Mac can "
              "stand in for it")


def _bb():
    sys.path.insert(0, str(HERE))
    import bench_board as B
    return B


# --------------------------------------------------------------------------------------------- the plan
def plan(vendor, B=None):
    """The frozen measurement plan at its smoke shape: every family, identical, our arms (opponents never run)."""
    if vendor not in VENDORS:
        raise SystemExit("measuresmoke: vendor is nvidia or amd (Apple has no measurement smoke in its gate)")
    B = B or _bb()
    return B.plan_races(vendor, [MODE], B.FAMILIES, None, B.DATASETS, B.SMOKE_ROWS, "small")


def plan_sha256(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def worker_count(cores, host_gb, device_gb, batches):
    """(workers, what limited it)."""
    bounds = {"cores": max(1, int(cores))}
    if host_gb is not None:
        bounds["host memory"] = max(1, int(host_gb // HOST_GB_PER_WORKER))
    if device_gb is not None:
        bounds["device memory"] = max(1, int(device_gb // DEVICE_GB_PER_WORKER))
    else:
        # no device figure (the smi tool did not answer): a smallest-card bound, never the host's cores
        bounds["device memory unknown"] = UNKNOWN_DEVICE_WORKERS
    bounds["batches"] = max(1, batches)
    limit = min(bounds, key=lambda k: bounds[k])
    return bounds[limit], "%s (%s)" % (limit, ", ".join("%s %d" % kv for kv in sorted(bounds.items())))


# --------------------------------------------------------------------------------------------- the verdict
def _finite_number(v):
    return type(v) in (int, float) and math.isfinite(v)


def own_cell_errors(cell, quality_exempt):
    """[reason] keeping one own-arm cell from what the board ingests."""
    errors = []
    status = str(cell.get("status"))
    if status != "ok":
        return ["status %s" % status[:160]]
    quality = cell.get("quality")
    if not quality_exempt:
        if not isinstance(quality, dict):
            errors.append("no quality record")
        else:
            if "error" in quality:
                errors.append("quality error: %s" % str(quality["error"])[:160])
            # at least one finite number (every ok own cell of the stored AMD campaign has one); no NaN / inf anywhere
            if not any(_finite_number(v) for k, v in quality.items() if k != "error"):
                errors.append("no finite quality value")
            bad = sorted(k for k, v in quality.items() if isinstance(v, float) and not math.isfinite(v))
            if bad:
                errors.append("quality not finite: %s" % ", ".join(bad))
    for field in OWN_FIELDS:
        if cell.get(field) in (None, "", {}, []):
            errors.append("no %s" % field)
    if cell.get("mode") not in (None, MODE):
        errors.append("mode %s, not %s" % (cell.get("mode"), MODE))
    return errors


def recipe_verdict(race, rec, smoke_fails, unsupported, quality_exempt, missing_reason=None):
    """(verdict, one-line reason) of one planned recipe.

    race         the plan's race (id, our_arms, ...)
    rec          its board record from the batch (None: never recorded)
    smoke_fails  bench_board.smoke_verdict(race, rec, vendor): [(arm, reason)]
    unsupported  bench_board.unsupported_our_arms(race): {arm: status} the public capability record declares
    missing_reason  why the batch left no record (timeout, ceiling, crash), when it did not
    """
    if rec is None:
        return FAIL, missing_reason or "not run: no record from its batch"
    if smoke_fails:
        return FAIL, "; ".join("%s: %s" % (a, w) for a, w in smoke_fails)[:400]
    ours = list((race.get("our_arms") or {}).keys())
    if not ours:
        return FAIL, "the plan holds no own arm for this recipe"
    if set(ours) <= set(unsupported or {}):
        return UNSUPPORTED, "own arm(s) declared unsupported: %s" % "; ".join(
            "%s %s" % (a, str(unsupported[a])[:120]) for a in ours)
    cells = {c.get("arm"): c for c in rec.get("cells") or []}
    errors = []
    for arm in ours:
        if arm in (unsupported or {}):
            continue
        cell = cells.get(arm)
        if cell is None:
            errors.append("%s: no cell" % arm)
            continue
        errors += ["%s: %s" % (arm, e) for e in own_cell_errors(cell, quality_exempt)]
    return (FAIL, "; ".join(errors)[:400]) if errors else (PASS, "")


def summarize(vendor, ids, verdicts, meta):
    """The summary document the release judges (no timings)."""
    counts = {PASS: 0, FAIL: 0, UNSUPPORTED: 0}
    for v in verdicts.values():
        counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1
    failures = [dict(id=rid, reason=verdicts[rid]["reason"]) for rid in sorted(verdicts)
                if verdicts[rid]["verdict"] == FAIL]
    missing = sorted(set(ids) - set(verdicts))
    failures += [dict(id=rid, reason="no verdict recorded") for rid in missing]
    ok = bool(ids) and not failures and counts[PASS] + counts[UNSUPPORTED] == len(ids)
    return dict(schema=SCHEMA, vendor=vendor, mode=MODE, ours_only=True, opponents_run=False,
                plan_count=len(ids), plan_sha256=plan_sha256(ids), counts=counts, ok=ok,
                failures=failures, unsupported=sorted(r for r, v in verdicts.items() if v["verdict"] == UNSUPPORTED),
                recipes={rid: verdicts[rid] for rid in sorted(verdicts)}, **meta)


def summary_errors(doc):
    """[] when the summary proves every planned recipe passed (or is declared unsupported), else why not."""
    if not isinstance(doc, dict):
        return [NO_SUMMARY]
    errors = []
    if doc.get("schema") != SCHEMA:
        return ["measurement-smoke summary schema %r, not %r" % (doc.get("schema"), SCHEMA)]
    if doc.get("mode") != MODE or doc.get("ours_only") is not True or doc.get("opponents_run") is not False:
        errors.append("not an identical, ours-only run")
    recipes = doc.get("recipes") if isinstance(doc.get("recipes"), dict) else {}
    if not recipes or doc.get("plan_count") != len(recipes):
        errors.append("%d recipe verdicts for a plan of %s" % (len(recipes), doc.get("plan_count")))
    elif doc.get("plan_sha256") != plan_sha256(recipes):
        errors.append("recipe verdicts are not the recorded plan")
    bad = sorted(r for r, v in recipes.items() if not isinstance(v, dict) or v.get("verdict") not in (PASS, UNSUPPORTED))
    if bad:
        errors.append("%d of %d recipes failed: %s" % (len(bad), len(recipes), "; ".join(
            "%s (%s)" % (r, str((recipes[r] or {}).get("reason") if isinstance(recipes[r], dict) else "")[:120])
            for r in bad[:8]) + (" ..." if len(bad) > 8 else "")))
    if doc.get("ok") is not True and not errors:
        errors.append("the summary does not say ok")
    if doc.get("ok") is True and (bad or doc.get("failures")):
        errors.append("the summary says ok over failures")
    return errors


def line(rid, verdict, reason):
    return "MEASURESMOKE %s %s%s" % (rid, verdict, (" " + " ".join(str(reason).split())[:300]) if reason else "")


# --------------------------------------------------------------------------------------------- the box
def box_cores():
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 1
    try:  # a container's CPU quota (cgroup v2), when one is set
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            cores = min(cores, max(1, int(int(quota) // int(period))))
    except (OSError, ValueError):
        pass
    return cores


def host_available_gb():
    try:
        for row in Path("/proc/meminfo").read_text().splitlines():
            if row.startswith("MemAvailable:"):
                return int(row.split()[1]) / (1024 * 1024)
    except (OSError, ValueError, IndexError):
        pass
    return None


def device_memory_mb(vendor, field):
    """(total|used) MB of GPU 0, or None (nvidia-smi / rocm-smi)."""
    try:
        if vendor == "nvidia":
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.%s" % field, "--format=csv,noheader,nounits",
                                  "-i", "0"], capture_output=True, text=True, timeout=30).stdout
            return float(out.split()[0])
        out = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--json"], capture_output=True, text=True,
                             timeout=30).stdout
        card = next(iter(json.loads(out).values()))
        key = next(k for k in card if k.lower().startswith("vram total %s" % ("used memory" if field == "used" else "memory")))
        return float(card[key]) / (1024 * 1024)
    except (OSError, ValueError, IndexError, StopIteration, subprocess.SubprocessError, AttributeError):
        return None


class DevicePeak(threading.Thread):
    """Samples device memory in use every `every` seconds; `.peak` is the measured maximum (MB) or None."""

    def __init__(self, vendor, every=10):
        super().__init__(daemon=True)
        self.vendor, self.every, self.peak, self.stopped = vendor, every, None, threading.Event()

    def run(self):
        while not self.stopped.is_set():
            used = device_memory_mb(self.vendor, "used")
            if used is not None:
                self.peak = used if self.peak is None else max(self.peak, used)
            self.stopped.wait(self.every)


def cpu_jiffies():
    """(busy, total) jiffies of all CPUs from /proc/stat, or None (not Linux)."""
    try:
        f = [int(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
    except (OSError, ValueError, IndexError):
        return None
    idle = f[3] + (f[4] if len(f) > 4 else 0)
    return sum(f) - idle, sum(f)


def busy_cores(before, after, cores):
    """Cores busy between two cpu_jiffies() samples (busy fraction of the whole host times its cores)."""
    if not before or not after or after[1] <= before[1]:
        return None
    return round((after[0] - before[0]) / float(after[1] - before[1]) * cores, 1)


class BusyCores(threading.Thread):
    """The rented-box rule: busy cores of total, printed (MEASURESMOKE busy ...) first at `first` seconds into the
    batches, then every `every` seconds, and kept for the summary."""

    def __init__(self, cores, first=120, every=600, window=10):
        super().__init__(daemon=True)
        self.cores, self.first, self.every, self.window = cores, first, every, window
        self.samples, self.stopped, self.t0 = [], threading.Event(), time.time()

    def run(self):
        wait = self.first
        while not self.stopped.wait(wait):
            a = cpu_jiffies()
            if self.stopped.wait(self.window):
                return
            busy = busy_cores(a, cpu_jiffies(), self.cores)
            self.samples.append(dict(at_seconds=round(time.time() - self.t0), busy=busy, of=self.cores))
            print("MEASURESMOKE busy cores=%s of %d at %ds" % (busy, self.cores, time.time() - self.t0), flush=True)
            wait = self.every


def board_command(args, out, extra, cache=None):
    inputs = (["--inputs", "synthetic", "--inputs-map", str(args.inputs_map)]
              if getattr(args, "inputs_map", None) else [])
    return [args.python, str(HERE / "bench_board.py"), "--smoke", "--vendor", args.vendor, "--modes", MODE,
            "--python-env", args.python, "--skip-install", "--cache", str(cache or Path(args.work) / "cache"),
            "--data-root", args.data_root, "--evidence-local-only", "--out", str(out)] + inputs + extra


def run_batch(args, k, ids, deadline, recipe_timeout, group=None):
    """One bench_board --smoke process over `ids`, bounded by its own budget and the phase deadline. With a staging
    group, the batch reads that group's prep cache and writes under that group's directory, and every race's scratch
    is deleted as soon as its record is written (bench_board --drop-race-scratch)."""
    import release_gate_profiles as P
    gdir = Path(group["dir"]) if group else Path(args.work)
    bdir = gdir / "batches"
    bdir.mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "logs").mkdir(parents=True, exist_ok=True)
    idfile = bdir / ("b%03d.ids.json" % k)
    idfile.write_text(json.dumps(ids))
    out = bdir / ("b%03d" % k)
    log = Path(args.out) / "logs" / ("b%03d.log" % k)   # the logs come home; the smoke boards stay on the box
    budget = BATCH_SETUP_S + recipe_timeout * len(ids)
    left = deadline - time.time()
    started = time.time()
    extra = ["--race-ids", "@" + str(idfile)]
    if not getattr(args, "keep_scratch", False) and board_drops_scratch():
        extra.append("--drop-race-scratch")
    result = P.bounded_run(board_command(args, out, extra, group_cache(args, group)), log, min(budget, left),
                           env=batch_env(args, group), cwd=str(ROOT))
    result.update(batch=k, recipes=len(ids), dir=str(out) + "-smoke", log=str(log),
                  ceiling_hit=left <= budget and result.get("timed_out"), started_late=left <= 0,
                  started_at=started, group=group["name"] if group else None)
    return result


def board_takes_inputs(here=HERE):
    """Whether this tree's board and drivers take the smoke's synthetic inputs (this tool is a gate overlay file and
    may run over a frozen tree that predates them; then the run keeps the datasets)."""
    try:
        return ("--inputs-map" in (Path(here) / "bench_board.py").read_text()
                and "def synthetic_path" in (Path(here) / "smoke_inputs.py").read_text())
    except OSError:
        return False


def board_drops_scratch():
    """Whether this tree's bench_board.py has --drop-race-scratch (this tool is a gate overlay file and may run over a
    frozen tree whose board predates it; then the group cleanup alone bounds the scratch)."""
    try:
        return "--drop-race-scratch" in (HERE / "bench_board.py").read_text()
    except OSError:
        return False


def group_cache(args, group):
    """The prep cache a batch reads: its staging group's own (deleted with the group), else the run's one cache."""
    return Path(group["dir"]) / "cache" if group else Path(args.work) / "cache"


def batch_env(args, group=None):
    """The installed wheel's shipped defaults: every MOJOLEARN_* variable of the box is dropped (a job-level knob, a
    vendor override or a fault switch would change a route: a FAIL that is not a product defect, as in
    tools/logical_shard_identity.py worker_env), then only what the harness needs is set."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "PYTHONHOME") and not k.startswith("MOJOLEARN_")}
    env["PYTHONUNBUFFERED"] = "1"
    env["MOJOLEARN_NUMERIC_MODE"] = MODE
    base = Path(group["dir"]) if group else Path(args.work)
    # the two causal-LM recipes' public checkpoints (bench_board_neural.lm_checkpoint fetches them once); per staging
    # group, so they are deleted with the neural group
    env["MOJOLEARN_BOARD_LM_CHECKPOINTS"] = str(base / "hf-checkpoints")
    env.setdefault("GBM_BENCH_DATA", args.data_root)
    # the layer lanes' fixtures, generated before the group's batches (fixtures() below)
    env["MOJOLEARN_NEURAL_FIXTURES"] = str(fixture_dir(args, group))
    # no run summary into the gate tree: retain_run_summary.retain_board (smoke=True) would write a timing summary
    # under docs/measurements/latest/auto/ of the source tree, from every batch at once. An absent repository makes
    # it print RETENTION FAILED and keep the run's own exit code (tools/retain_run_summary.py try_retain).
    env["MOJOLEARN_SUMMARY_REPO"] = str(Path(args.work) / "no-run-summary")
    return env


def fixture_dir(args, group=None):
    return group_cache(args, group) / "algos-data" / "neural-fixtures"


def fixtures(args, races, out, deadline, group=None):
    """The algos layer lanes' input fixtures (tools/neural_fixtures.py generate over the smoke-size blocks the prep
    wrote, then verify), with --fixture-python: a separate venv holding the CPU torch / torch-geometric the fixture
    generator pins (tools/neural_fixtures_box.sh), never the measured venv; torch only writes inputs here, it is not
    raced. Without it, or when it fails, the layer recipes refuse on their own ("neural fixtures missing") and FAIL
    with that reason. Returns the record kept in the summary."""
    import release_gate_profiles as P
    B = _bb()
    lanes = sorted({r["lane"] for r in races if r["family"] == "algos" and B.ALGOS.LANES[r["lane"]]["kind"] == "layer"})
    if not lanes:
        return dict(lanes=0)
    if not args.fixture_python:
        return dict(lanes=len(lanes), exit_code=None, why="no --fixture-python: the layer recipes refuse")
    # a layer lane on the verify fixture (the graph layers on taxi/istella) preps into the fixture data dir
    imap = getattr(args, "input_map", None) or {}
    tag = "-verify-fixtures" if any(_on_fixture(imap.get(r["id"])) for r in races if r["lane"] in lanes
                                    and r["family"] == "algos") else ""
    data = group_cache(args, group) / "algos-data" / ("rows-%d%s" % (B.SMOKE_ROWS, tag))
    log = out / "logs" / ("fixtures%s.log" % ("-" + group["name"] if group else ""))
    env = batch_env(args, group)
    env["MOJOLEARN_REPO_COMMIT"] = B.repo_commit()
    rec = P.bounded_run([args.fixture_python, str(HERE / "neural_fixtures.py"), "generate", "--out",
                         str(fixture_dir(args, group)), "--data", str(data), "--lanes", ",".join(lanes)], log,
                        max(0.0, deadline - time.time()), env=env, cwd=str(ROOT))
    if rec["exit_code"] == 0:
        rec = P.bounded_run([args.python, str(HERE / "neural_fixtures.py"), "verify", "--dir",
                             str(fixture_dir(args, group))],
                            log, max(0.0, deadline - time.time()), env=env, cwd=str(ROOT))
    return dict(lanes=len(lanes), exit_code=rec["exit_code"], timed_out=rec["timed_out"])


# --------------------------------------------------------------------------------------------- the disk
# 0.8.66 run t (1x RTX 4090 pod, 80 GB overlay disk): one --prep-only pass over the whole 653-recipe plan, then every
# batch's scratch kept to the end, grew the work dir to 35 GB; with the gate's venvs and reports the disk hit 100 %,
# the smoke died, its final upload and the pod's lease writes failed. The run is therefore split into STAGING GROUPS
# (one per board dataset; the neural recipes are one group, as they are one batch): a group's data is staged, its
# block prep runs into the group's own cache, its batches run (each race's scratch is dropped once its record is
# written), its recipes are judged, then its cache, its batch dirs and the data files no later group reads are
# deleted. Groups run side by side while the footprint (the data this run fetched + the work dir) stays under the
# cap; a group that alone would exceed it is refused by name, never staged.
#: the default cap on the smoke's footprint (the data it fetched + its work dir), GB (--max-work-gb)
MAX_WORK_GB = 8.0


def dir_bytes(path):
    """Bytes of the regular files under `path` (links not followed); 0 when it does not exist."""
    total = 0
    for dirpath, _, files in os.walk(str(path)):
        for f in files:
            try:
                st = os.lstat(os.path.join(dirpath, f))
            except OSError:
                continue
            total += st.st_size
    return total


def disk_free_gb(path):
    import shutil
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        return shutil.disk_usage(str(p)).free / 1e9
    except OSError:
        return None


def _on_fixture(rec):
    """A recipe whose prep takes fixture `base` (bench_board.fixture_race's rule): on the verify fixture and reading
    a data file."""
    return bool(rec and rec.get("input") == INPUT_VERIFY and rec.get("files"))


def race_keys(race, B=None, rec=None):
    """{R2 key: box path relative to $HOME} ONE recipe reads (data_keys is their union over the plan). With its
    smoke input record `rec` (the synthetic inputs): nothing at all, every recipe's inputs are generated on the box."""
    if rec is not None:
        return {}
    return _race_keys(race, B)


def _si():
    sys.path.insert(0, str(HERE))
    import smoke_inputs
    return smoke_inputs


def _race_keys(race, B=None):
    B = B or _bb()
    needed, _ = B.data_needs([race], B.DATASETS)
    keys = {B.R2_KEYS[d]: "datasets/gbm-bench/" + B.DATA_FILES[d] for d in needed}
    if race["family"] == "algos":
        for k in B.ALGOS.r2_keys(race["lane"], race["dataset"]):
            if k.startswith("corpus/"):
                keys[k] = "r2-stage/" + k
            elif k not in keys:
                keys[k] = "datasets/" + k
    return keys


def group_name(race):
    return "neural" if race["id"].startswith(SERIAL_PREFIXES) else "ds-" + "".join(
        c if c.isalnum() or c in "-_." else "_" for c in str(race["dataset"]))


#: recipes per staging group at most (the neural group excepted): a dataset with more recipes is split into parts of
#: consecutive sorted ids (family/lane/dataset order, so the lanes sharing a prep block stay together), each with its
#: own prep cache, so one part's blocks, not the whole dataset's 250-odd lanes', are on disk at once. Its data file is
#: staged once and kept until the last part that reads it closes (DataStager counts the readers).
GROUP_MAX_RECIPES = 64


def staging_groups(races, B=None, max_recipes=GROUP_MAX_RECIPES, imap=None):
    """[{name, ids, keys}] in name order: one group per board dataset (split into parts of at most `max_recipes`),
    the neural recipes in one. With the smoke input map `imap`, each recipe's keys are its smoke inputs' (race_keys)."""
    B = B or _bb()
    by_name = {}
    for r in races:
        by_name.setdefault(group_name(r), []).append(r)
    out = []
    for name, rs in sorted(by_name.items()):
        rs = sorted(rs, key=lambda r: r["id"])
        n = 1 if name == "neural" else max(1, -(-len(rs) // max(1, max_recipes)))
        step = -(-len(rs) // n)
        for part in range(n):
            chunk = rs[part * step:(part + 1) * step]
            keys = set()
            for r in chunk:
                keys |= set(race_keys(r, B, imap.get(r["id"]) if imap is not None else None))
            out.append(dict(name=name if n == 1 else "%s-%d" % (name, part + 1), ids=[r["id"] for r in chunk],
                            keys=sorted(keys)))
    return out


def group_batches(group, size):
    """A group's batches: the neural group whole (shared inputs, lazy checkpoint fetch), every other group in
    round-robin batches of at most `size` recipes (lanes interleave, so no batch holds only the slow ones)."""
    ids = list(group["ids"])
    if not ids:
        return []
    if group["name"] == "neural":
        return [ids]
    n = max(1, -(-len(ids) // max(1, size)))
    return [ids[k::n] for k in range(n)]


class DataStager:
    """The data plan's files, staged per group and deleted when no remaining group reads them. Only files this run
    fetched are counted and deleted; a file already at its pin on the box (another job's) is used and left."""

    def __init__(self, entries, home, groups):
        self.entries = {e["key"]: e for e in entries}
        self.home = Path(home)
        self.owned = {}                     # key -> bytes this run fetched and still holds
        self.readers = {}                   # key -> groups not yet closed that read it
        for g in groups:
            for k in g["keys"]:
                self.readers[k] = self.readers.get(k, 0) + 1

    def owned_bytes(self):
        return sum(self.owned.values())

    def need_bytes(self, group):
        """Bytes staging `group` would add: its files neither held by this run nor already at their size."""
        need = 0
        for k in group["keys"]:
            e = self.entries.get(k)
            if e is None or k in self.owned:
                continue
            dest = self.home / e["path"]
            if not (dest.is_file() and dest.stat().st_size == e["bytes"]):
                need += int(e["bytes"])
        return need

    def stage(self, group, stage_fn=None):
        """[] or ["<key>: <why>"] for the group's files that did not stage."""
        stage_fn = stage_fn or stage_one
        bad = []
        for k in group["keys"]:
            e = self.entries.get(k)
            if e is None:
                bad.append("%s: not in the data plan" % k)
                continue
            if k in self.owned:
                continue
            got = stage_fn(e, self.home)
            print("MEASURESMOKE data %s %s (group %s)" % (k, got, group["name"]), flush=True)
            if got == "fetched":
                self.owned[k] = int(e["bytes"])
            elif got != "present":
                bad.append("%s: %s" % (k, got))
        return bad

    def release(self, group):
        """The group is done: delete the files this run fetched that no open or pending group reads. Returns bytes."""
        freed = 0
        for k in group["keys"]:
            self.readers[k] = self.readers.get(k, 1) - 1
            if self.readers[k] <= 0 and k in self.owned:
                (self.home / self.entries[k]["path"]).unlink(missing_ok=True)
                freed += self.owned.pop(k)
        return freed


def cap_refusal(group, need, in_use, cap):
    return ("measuresmoke: staging group %s (%d recipes) needs %.2f GB on top of %.2f GB in use; the smoke's work "
            "cap is %.2f GB (--max-work-gb): refused, never staged. Raise --max-work-gb (the pod's disk must hold it) "
            "or split the group's data" % (group["name"], len(group["ids"]), need / 1e9, in_use / 1e9, cap / 1e9))


def clean_group(args, group, verdicts):
    """After a group is judged: its prep cache, fixtures and checkpoints are deleted, and every batch directory whose
    recipes all passed (or are declared unsupported); a batch with a failure keeps board.json and logs/ only (the
    reason is in the summary). --keep-scratch keeps everything (diagnosis on a box with room)."""
    import shutil
    if getattr(args, "keep_scratch", False):
        return
    gdir = Path(group["dir"])
    for sub in ("cache", "hf-checkpoints", "prep", "prep-smoke"):
        shutil.rmtree(gdir / sub, ignore_errors=True)
    for idfile in (gdir / "batches").glob("b*.ids.json"):
        idfile.unlink(missing_ok=True)
    for b in group.get("batches") or []:
        ok = all((verdicts.get(rid) or {}).get("verdict") in (PASS, UNSUPPORTED) for rid in b["ids"])
        smoke_dir = Path(b["dir"]) if b.get("dir") else None
        dirs = [d for d in (smoke_dir, Path(str(smoke_dir)[:-len("-smoke")]) if smoke_dir else None) if d]
        for d in dirs:
            if not d.is_dir():
                continue
            if ok:
                shutil.rmtree(d, ignore_errors=True)
                continue
            for child in d.iterdir():
                if child.name in ("board.json", "logs"):
                    continue
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)


def _tail(path, n=1):
    if not path:
        return ""
    try:
        rows = [r for r in Path(path).read_text(errors="replace").splitlines() if r.strip()]
        return " | ".join(" ".join(r.split())[:200] for r in rows[-n:])
    except OSError:
        return ""


def missing_reason(batch):
    if batch.get("started_late"):
        return "phase ceiling reached before its batch started"
    if batch.get("ceiling_hit"):
        return "phase ceiling reached while its batch ran (killed)"
    if batch.get("timed_out"):
        return "its batch exceeded %d recipes x the per-recipe timeout (killed)" % batch.get("recipes", 0)
    return "its batch exited %s without a record for it: %s" % (batch.get("exit_code"), _tail(batch.get("log")))


def judge(args, races, batches):
    B = _bb()
    by_batch = {rid: b for b in batches for rid in b["ids"]}
    boards = {}
    verdicts = {}
    for race in races:
        rid = race["id"]
        b = by_batch.get(rid)
        rec = None
        if b is not None:
            if b["dir"] not in boards:
                try:
                    boards[b["dir"]] = B.load_result(os.path.join(b["dir"], "board.json")) or {}
                except (OSError, ValueError):
                    boards[b["dir"]] = {}
            rec = (boards[b["dir"]].get("races") or {}).get(rid)
        why = missing_reason(b) if b is not None else "not in any batch"
        if rec is None and b is not None:
            # the board's preflight kept it from running (bench_board.select_after_preflight): refused or failed
            board = boards.get(b["dir"]) or {}
            for kind in ("skipped_recipes", "failed_preflight_recipes"):
                held = (board.get(kind) or {}).get(rid)
                if isinstance(held, dict):
                    why = "preflight %s: %s" % (held.get("status"), "; ".join(map(str, held.get("problems") or []))[:300])
        try:
            import fnmatch
            exempt = any(fnmatch.fnmatch(B.smoke_key(race), g) for g in B.QUALITY_EXEMPT)
            fails = B.smoke_verdict(race, rec, args.vendor) if rec is not None else []
            unsupported = B.unsupported_our_arms(dict(race, capability_probes=(rec or {}).get("capability_probes")
                                                      or race.get("capability_probes") or {}))
            verdict, reason = recipe_verdict(race, rec, fails, unsupported, exempt, why)
        except Exception as exc:  # noqa: BLE001  (a judge error is a failure, never a pass)
            verdict, reason = FAIL, "judging raised %r" % (exc,)
        verdicts[rid] = dict(verdict=verdict, reason=reason, batch=b["batch"] if b else None)
    return verdicts


def disk_line(work, footprint, what):
    free = disk_free_gb(work)
    print("MEASURESMOKE disk free=%s footprint=%.2fGB after %s" % ("%.1fGB" % free if free is not None else "?",
                                                                  footprint / 1e9, what), flush=True)


def cmd_run(args):
    out, work = Path(args.out), Path(args.work)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    started = time.time()
    deadline = started + args.ceiling
    B = _bb()
    races = plan(args.vendor, B)
    ids = [r["id"] for r in races]
    by_id = {r["id"]: r for r in races}
    cores, host_gb = box_cores(), host_available_gb()
    total_mb = device_memory_mb(args.vendor, "total")
    workers, limit = worker_count(cores, host_gb, total_mb / 1024 if total_mb else None, len(ids))
    if args.jobs:
        workers, limit = args.jobs, "--jobs %d" % args.jobs
    # the inputs: the verify fixture / synthetic file / no verify lane map, once for the run (every batch reads it)
    imap = None
    args.inputs_map = args.input_map = None
    if board_takes_inputs():
        imap = smoke_inputs(races, B)
        args.input_map = imap
        args.inputs_map = work / "smoke_inputs.json"
        args.inputs_map.write_text(json.dumps(imap, indent=1, sort_keys=True) + "\n")
    # the synthetic inputs read no data plan and never call the stager: nothing is staged
    plan_doc = json.loads(Path(args.data_plan).read_text()) if args.data_plan and not imap else {}
    if args.max_work_gb is None:
        args.max_work_gb = WORK_MAX_BYTES / 1e9 if imap else MAX_WORK_GB
    groups = staging_groups(races, B, imap=imap)
    rest = sum(len(g["ids"]) for g in groups if g["name"] != "neural")
    size = max(1, -(-rest // max(1, workers * BATCHES_PER_WORKER)))
    for g in groups:
        g.update(dir=str(work / "groups" / g["name"]), batches=[])
    entries = plan_doc.get("files", [])
    stager = DataStager(entries, args.home, groups)
    cap = int(args.max_work_gb * 1e9)
    recipe_timeout = B.SMOKE_STEP_S * 4    # bench_board --smoke's own per-race deadline (4 x SMOKE_STEP_S)
    print("MEASURESMOKE plan vendor=%s recipes=%d groups=%d batch_size=%d workers=%d limit=%s ceiling=%ds cap=%.1fGB"
          % (args.vendor, len(ids), len(groups), size, workers, limit, args.ceiling, args.max_work_gb), flush=True)
    if imap:
        bud = budget(races, imap, B, workers)
        print(SMOKE_INPUTS_LINE, flush=True)
        print("MEASURESMOKE inputs %s staged=0 work_bound=%.2fGB (budget %.1fGB)%s"
              % (" ".join("%s=%d" % (k.replace(" ", "-"), n) for k, n in bud["counts"].items()),
                 bud["work_bound_bytes"] / 1e9, WORK_MAX_BYTES / 1e9,
                 " (the data plan given is not read)" if args.data_plan else ""), flush=True)
    else:
        bud = None
        print("MEASURESMOKE inputs datasets (this tree's board predates --inputs verify-fixtures)", flush=True)

    def footprint():
        return stager.owned_bytes() + dir_bytes(work)

    disk_line(work, footprint(), "start")
    peak = DevicePeak(args.vendor)
    peak.start()
    busy = BusyCores(cores)
    import release_gate_profiles as P
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
    verdicts, batches, group_recs, fixture_recs = {}, [], [], {}
    peak_footprint = [0]

    def fail_group(g, reason):
        for rid in g["ids"]:
            verdicts[rid] = dict(verdict=FAIL, reason=reason, batch=None)

    def open_group(g):
        """Stage, prep and generate fixtures for one group: None, or the reason its recipes FAIL."""
        gdir = Path(g["dir"])
        (gdir / "batches").mkdir(parents=True, exist_ok=True)
        bad = stager.stage(g) if entries else []
        if bad:
            return "staging the group's data failed: " + "; ".join(bad)[:300]
        idfile = gdir / "batches" / "prep.ids.json"
        idfile.write_text(json.dumps(g["ids"]))
        log = out / "logs" / ("prep-%s.log" % g["name"])
        # the untimed block prep, once per group, before its batches (bench_board --prep-only), into its own cache
        prep = P.bounded_run(board_command(args, gdir / "prep", ["--preflight-only", "--prep-only", "--race-ids",
                                                                 "@" + str(idfile)], group_cache(args, g)),
                             log, max(0.0, deadline - time.time()), env=batch_env(args, g), cwd=str(ROOT))
        g["prep"] = dict(exit_code=prep["exit_code"], timed_out=prep["timed_out"])
        if prep["exit_code"] != 0:
            return "the untimed data prep exited %s: %s" % (prep["exit_code"], _tail(log))
        fixture_recs[g["name"]] = fixtures(args, [by_id[i] for i in g["ids"]], out, deadline, g)
        return None

    def close_group(g):
        got = judge(args, [by_id[i] for i in g["ids"]], g["batches"])
        verdicts.update(got)
        held = footprint()
        peak_footprint[0] = max(peak_footprint[0], held)
        clean_group(args, g, verdicts)
        freed = stager.release(g)
        group_recs.append(dict(name=g["name"], recipes=len(g["ids"]), data_keys=g["keys"],
                               batches=len(g["batches"]), prep=g.get("prep"), footprint_before_clean=held,
                               data_freed=freed))
        disk_line(work, footprint(), "group %s" % g["name"])

    pending, queue, futures, live = list(groups), [], {}, {}
    k = 0
    busy.start()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while pending or queue or futures:
            # open the next group only when no batch waits for a worker, so the footprint holds only what runs
            while pending and not queue and len(futures) < workers:
                g = pending.pop(0)
                if time.time() >= deadline:
                    fail_group(g, "phase ceiling reached before its group was staged")
                    stager.release(g)
                    continue
                in_use, need = footprint(), stager.need_bytes(g)
                if in_use + need > cap:
                    if live:
                        pending.insert(0, g)     # wait for a running group to close and free its share
                        break
                    reason = cap_refusal(g, need, in_use, cap)
                    print("MEASURESMOKE " + reason, flush=True)
                    fail_group(g, reason)
                    stager.release(g)
                    group_recs.append(dict(name=g["name"], recipes=len(g["ids"]), data_keys=g["keys"],
                                           refused=reason))
                    continue
                why = open_group(g)
                held = footprint()
                peak_footprint[0] = max(peak_footprint[0], held)
                disk_line(work, held, "staging %s" % g["name"])
                if why:
                    fail_group(g, why)
                    clean_group(args, g, verdicts)
                    stager.release(g)
                    group_recs.append(dict(name=g["name"], recipes=len(g["ids"]), data_keys=g["keys"],
                                           prep=g.get("prep"), failed=why))
                    continue
                todo = group_batches(g, size)
                live[g["name"]] = len(todo)
                for bids in todo:
                    queue.append((k, bids, g))
                    k += 1
            while queue and len(futures) < workers:
                kk, bids, g = queue.pop(0)
                futures[pool.submit(run_batch, args, kk, bids, deadline, recipe_timeout, g)] = (bids, g)
            if not futures:
                continue
            done, _ = wait(list(futures), return_when=FIRST_COMPLETED)
            for f in done:
                bids, g = futures.pop(f)
                rec = f.result()
                rec["ids"] = bids
                g["batches"].append(rec)
                batches.append(rec)
                peak_footprint[0] = max(peak_footprint[0], footprint())
                live[g["name"]] -= 1
                if live[g["name"]] == 0:
                    del live[g["name"]]
                    close_group(g)
    peak.stopped.set()
    busy.stopped.set()
    meta = dict(rows=B.SMOKE_ROWS, neural_shape="small", runner="tools/bench_board.py --smoke",
                cores=cores, host_available_gb=host_gb, device_total_mb=total_mb, workers=workers, limit=limit,
                device_gb_per_worker=DEVICE_GB_PER_WORKER, host_gb_per_worker=HOST_GB_PER_WORKER,
                measured_peak_device_mb=peak.peak, busy_cores=busy.samples, ceiling_seconds=args.ceiling,
                recipe_timeout_seconds=recipe_timeout, elapsed_seconds=round(time.time() - started, 1),
                prep={g["name"]: g.get("prep") for g in groups}, fixtures=fixture_recs,
                inputs=dict(mode=SYNTHETIC_INPUTS if imap else DATASETS_INPUTS,
                            **({k: v for k, v in bud.items() if k != "parts"} if bud else {})),
                disk=dict(cap_bytes=cap, peak_footprint_bytes=peak_footprint[0], groups=group_recs),
                batches=[{k: b.get(k) for k in ("batch", "group", "recipes", "exit_code", "timed_out", "ceiling_hit",
                                                 "started_late", "elapsed_seconds", "log")} for b in batches])
    doc = summarize(args.vendor, ids, verdicts, meta)
    (out / SUMMARY).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    text = [line(f["id"], FAIL, f["reason"]) for f in doc["failures"]]
    text += [line(rid, UNSUPPORTED, verdicts[rid]["reason"]) for rid in doc["unsupported"]]
    (out / "measuresmoke.txt").write_text("\n".join(text) + "\n")
    for row in text[:40]:
        print(row, flush=True)
    c = doc["counts"]
    print("MEASURESMOKE vendor=%s recipes=%d PASS=%d UNSUPPORTED=%d FAIL=%d workers=%d elapsed=%.0fs peak_device_mb=%s "
          "peak_footprint=%.2fGB %s"
          % (args.vendor, len(ids), c[PASS], c[UNSUPPORTED], c[FAIL] + len(set(ids) - set(verdicts)), workers,
             doc["elapsed_seconds"], peak.peak, peak_footprint[0] / 1e9, "ok" if doc["ok"] else "FAILED"), flush=True)
    return 0 if not summary_errors(doc) else 1


# --------------------------------------------------------------------------------------------- the inputs
# ONE FIXTURE SET FOR VERIFY AND SMOKE (Andrew 2026-11-08: "maybe the smoke fixtures should be the same as the
# verify"): a recipe whose driver takes arrays is to run on the identity verification's own fixture set
# (tools/identity_break.py `fixture("base")`, deterministic, shipped with the tooling, nothing staged); a recipe whose
# purpose is a dataset FILE loader (the gbm-bench .npz loaders of the trees family) or a corpus reader keeps a file.
# smoke_inputs() is that mapping, recipe by recipe; `inputs` writes it as docs/measurements/SMOKE_INPUTS.md. The verify
# lane is found from names only, never invented: the board lane's own name in identity_break.LANES, else the identity
# lanes named "<board lane>-*", else (algos) the identity lanes whose own code touches the public class the board
# races (bench_board_algos LANES[lane]["ours"]; tools/verify_coverage_audit.py lane_names). None found: NO VERIFY LANE.
VERIFY_FIXTURE = "base"
#: families whose recipes exercise a dataset FILE loader (tools/speed_gbdt_arm.py load_dataset reads the npz itself)
FILE_LOADER_FAMILIES = ("trees",)
INPUT_VERIFY, INPUT_FILE, INPUT_NONE = "verify fixture", "synthetic file", "NO VERIFY LANE"


def verify_lane_index():
    """(identity lane names, {name a lane's own code touches: {lane}}, {lane: number of names})."""
    import lane_select as ls
    import verify_coverage_audit as V
    ib = ls.identity_break()
    names = V.lane_names(ls, ib)
    by_name = {}
    for lane, ns in names.items():
        for n in ns:
            by_name.setdefault(n, set()).add(lane)
    return sorted(ib.LANES), by_name, {lane: len(ns) for lane, ns in names.items()}


def verify_lanes_for(race, B, index):
    """[identity lane], best first, for one board recipe ([] when the names give none)."""
    lanes, by_name, width = index
    lane = race["lane"]
    if lane in lanes:
        return [lane]
    pre = [l for l in lanes if l.startswith(lane + "-")]
    if pre:
        return pre
    if race["family"] != "algos":
        return []
    classes = [str(c).split(".")[-1] for c in (B.ALGOS.LANES.get(lane, {}).get("ours") or ())]
    cand = set()
    for c in classes:
        cand |= by_name.get(c, set())
    tokens = set(lane.split("-"))
    return sorted(cand, key=lambda l: (-len(tokens & set(l.split("-"))), width.get(l, 0), l))


def smoke_inputs(races, B=None, index=None):
    """{recipe id: {input, verify_lanes, files}}: the input each recipe's smoke runs on."""
    B = B or _bb()
    index = index or verify_lane_index()
    out = {}
    for r in races:
        files = sorted(race_keys(r, B))
        verify = verify_lanes_for(r, B, index)
        loader = r["family"] in FILE_LOADER_FAMILIES or any(k.startswith("corpus/") for k in files)
        kind = INPUT_FILE if loader and files else (INPUT_VERIFY if verify else INPUT_NONE)
        out[r["id"]] = dict(input=kind, verify_lanes=verify, files=files)
    return out


def inputs_table(vendors=VENDORS, B=None):
    """The SMOKE_INPUTS.md text: counts, then one row per board recipe (every vendor's plan; a recipe planned on one
    vendor only says which)."""
    B = B or _bb()
    index = verify_lane_index()
    rows, on = {}, {}
    for v in vendors:
        races = plan(v, B)
        for rid, rec in smoke_inputs(races, B, index).items():
            rows.setdefault(rid, rec)
            on.setdefault(rid, []).append(v)
    counts = {k: sum(1 for r in rows.values() if r["input"] == k) for k in (INPUT_VERIFY, INPUT_FILE, INPUT_NONE)}
    text = ["# Measurement smoke inputs", "",
            "Generated by `python3 tools/release_measurement_smoke.py inputs` from the board plan "
            "(`bench_board.plan_races`, identical, smoke shape) and the identity lanes (`tools/identity_break.py` "
            "LANES). Do not edit by hand.", "",
            "Each board recipe's smoke input, all of it generated on the box (the smoke stages no data): the "
            "identity verification's own fixture (`%s`) of the verify lane named; a synthetic file in the loader's "
            "own schema for a recipe whose purpose is a dataset file loader or a corpus reader; or NO VERIFY LANE "
            "when the identity table verifies no lane for what the board measures (that list is Andrew's to "
            "decide; those recipes run on synthetic inputs of the smoke size too)." % VERIFY_FIXTURE, "",
            "| input | recipes |", "|---|---:|"]
    text += ["| %s | %d |" % (k, n) for k, n in counts.items()]
    text += ["| total | %d |" % len(rows), ""]
    # what a smoke run writes under these inputs (budget(), from the plan; the run prints the same)
    races = plan(vendors[0], B)
    bud = budget(races, smoke_inputs(races, B, index), B)
    SI = _si()
    text += ["Under `--inputs synthetic` (every smoke run over a tree whose board takes it): 0 bytes staged. A "
             "synthetic file holds the full file's keys and dtypes with %d head rows and %d tail rows "
             "(`tools/smoke_inputs.py`; a corpus: %d bytes of generated text) and is deleted with the race's "
             "scratch. From the %s plan, the work dir at most %.2f GB at %d workers (budget %.1f GB): %d distinct "
             "prep blocks %.1f MB, race scratch %.1f MB, synthetic files %.1f MB, LM checkpoints %.1f MB, layer "
             "fixtures %.1f MB."
             % (SI.SMOKE_TRAIN_ROWS, SI.SMOKE_TEST_ROWS, SI.CORPUS_SYNTH_BYTES, vendors[0],
                bud["work_bound_bytes"] / 1e9, bud["parts"]["workers"], WORK_MAX_BYTES / 1e9,
                bud["parts"]["distinct_blocks"], bud["parts"]["blocks"] / 1e6, bud["parts"]["scratch"] / 1e6,
                bud["parts"]["synthetic"] / 1e6, bud["parts"]["checkpoints"] / 1e6,
                bud["parts"]["layer_fixtures"] / 1e6), ""]
    text += ["| recipe | input | smoke input | full-size files | vendors |", "|---|---|---|---|---|"]
    for rid in sorted(rows):
        r = rows[rid]
        lanes = r["verify_lanes"]
        schemas = sorted({SI.schema_of_key(f) for f in r["files"]} - {None})
        if r["input"] == INPUT_VERIFY:
            what = "verify fixture `%s`%s" % (lanes[0], " (+%d)" % (len(lanes) - 1) if len(lanes) > 1 else "")
        elif schemas:
            what = "synthetic " + ", ".join("`%s`" % x for x in schemas)
        else:
            what = "synthetic (the driver's seeded inputs)"
        text.append("| `%s` | %s | %s | %s | %s |" % (rid, r["input"], what, ", ".join(r["files"]) or "-",
                                                        ",".join(on[rid])))
    return "\n".join(text) + "\n", counts


def cmd_inputs(args):
    text, counts = inputs_table()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text)
    print("MEASURESMOKE inputs %s" % " ".join("%s=%d" % (k.replace(" ", "-"), n) for k, n in counts.items()))
    return 0


# --------------------------------------------------------------------------------------------- the budgets
#: Andrew (2026-11-08): the smoke stages NO data; its work dir stays under 2 GB
WORK_MAX_BYTES = 2 * 10 ** 9
DATASETS_INPUTS, SYNTHETIC_INPUTS = "datasets", "synthetic"
#: the header line of a smoke on the synthetic inputs (a test asserts it)
SMOKE_INPUTS_LINE = "SMOKE inputs: generated on the box, 0 bytes staged"
#: the arrays one prep block holds at most (X, Xq, y, yq and their derived copies: init rows, the graph's two CSR
#: pairs); each the block's rows x its width x 4 bytes
BLOCK_ARRAYS = 8
#: the width of a block no file feeds (the drivers' seeded series blocks: TS_SERIES = 64 series)
SYNTH_WIDTH = 64
#: a race's own scratch (work/ and its evidence archive, deleted once its record is written): the scored arrays on
#: the held-out rows, at most two arrays of its block's rows x width
SCRATCH_ARRAYS = 2
#: the two causal-LM recipes' public checkpoints (bench_board_neural.LM_CHECKPOINTS), fetched into the neural group's
#: work dir: parameters x bytes per parameter as published (SmolLM2-135M ships bfloat16; mojolearn-162m float32)
CHECKPOINT_BYTES = {"smollm2-135m": 135 * 10 ** 6 * 2, "mojolearn-162m": 162 * 10 ** 6 * 4}
#: one layer lane's fixture (tools/neural_fixtures.py generate at the smoke size: inputs, weights, outputs), declared
#: until a run records it (the summary's disk.peak_footprint_bytes sets the next figure)
LAYER_FIXTURE_BYTES = 8 * 10 ** 6
#: the largest worker count a gate box reaches (an MI325X's 256 GB / DEVICE_GB_PER_WORKER): the plan-time bound
MAX_BOX_WORKERS = 64


def prep_block(race, B):
    """(driver, block name) of the prep block a recipe reads, or None (trees and neural read in their race)."""
    fam, lane, ds = race["family"], race["lane"], race["dataset"]
    if fam == "classical":
        return ("classical", "%s-%s" % (B.CLASSICAL_BLOCK[lane], ds))
    if fam == "classical2":
        return ("classical2", "%s-%s" % (B.MORE.block_of(lane), ds))
    if fam == "algos":
        name = B.ALGOS.block_file(lane, ds)
        return ("algos", name) if name else None
    return None


def block_bound_bytes(race, rec, B):
    """Bytes one recipe's prep block holds at most, from the plan: rows (the smoke rows) x width x 4 x BLOCK_ARRAYS;
    width 16 on the verify fixture, the schema's width on a synthetic file, SYNTH_WIDTH with no file; a corpus block
    is the byte-bigram counts of the synthetic text's documents (bench_board_algos.text_counts: docs x TEXT_BINS)."""
    SI = _si()
    files = rec.get("files") or []
    if _on_fixture(rec):
        width = len(SI.FIXTURE_COLUMNS)
    elif any(f.startswith("corpus/") for f in files):
        docs = SI.CORPUS_SYNTH_BYTES // B.ALGOS.TEXT_DOC_BYTES
        return sum(docs * B.ALGOS.TEXT_BINS * 4 for f in files if f.startswith("corpus/"))
    elif files:
        width = max(SI.SCHEMA_WIDTH.get(SI.SCHEMAS.get(SI.schema_of_key(f) or ""), 16) for f in files)
    else:
        width = SYNTH_WIDTH
    return BLOCK_ARRAYS * B.SMOKE_ROWS * width * 4


_SYNTH_BYTES = {}


def synthetic_bytes(rel):
    """A synthetic file's bytes (its arrays generated once here, never written)."""
    if rel not in _SYNTH_BYTES:
        SI = _si()
        if rel.endswith(".txt"):
            _SYNTH_BYTES[rel] = SI.CORPUS_SYNTH_BYTES
        else:
            _SYNTH_BYTES[rel] = sum(a.nbytes for a in SI.synthetic_arrays(rel).values()) + 4096
    return _SYNTH_BYTES[rel]


def budget(races, imap, B=None, workers=MAX_BOX_WORKERS):
    """The smoke's disk budget from the plan (synthetic inputs; 0 bytes staged):
      blocks      every distinct prep block at its bound, as if every group were open at once (the run closes them);
      scratch     `workers` races' scratch at once (the largest);
      synthetic   `workers` races' synthetic files at once (the largest sets; deleted with each race's scratch) + one
                  group's prep (the groups open one at a time; a prep's files go when it ends);
      checkpoints the LM checkpoints; layer_fixtures the layer lanes' fixtures.
    Returns {counts, staged_bytes (0), work_bound_bytes, parts}."""
    B = B or _bb()
    SI = _si()
    blocks, scratch, synth, by_group = {}, [], [], {}
    for r in races:
        rec = imap[r["id"]]
        b = block_bound_bytes(r, rec, B)
        pb = prep_block(r, B)
        if pb:
            blocks[pb + (_on_fixture(rec),)] = b
        scratch.append(b * SCRATCH_ARRAYS // BLOCK_ARRAYS)
        rels = set() if _on_fixture(rec) else {SI.schema_of_key(f) for f in rec.get("files") or []} - {None}
        synth.append(sum(synthetic_bytes(x) for x in rels))
        by_group.setdefault(group_name(r), set()).update(rels)
    w = max(1, int(workers))
    lm = sum(v for k, v in CHECKPOINT_BYTES.items() if any(r["family"] == "neural" and r["dataset"] == k
                                                            for r in races))
    layers = len({r["lane"] for r in races if r["family"] == "algos" and B.ALGOS.LANES[r["lane"]]["kind"] == "layer"})
    parts = dict(blocks=sum(blocks.values()), distinct_blocks=len(blocks),
                 scratch=sum(sorted(scratch, reverse=True)[:w]), workers=w,
                 synthetic=sum(sorted(synth, reverse=True)[:w])
                 + max([sum(synthetic_bytes(x) for x in g) for g in by_group.values()] or [0]),
                 checkpoints=lm, layer_fixtures=layers * LAYER_FIXTURE_BYTES)
    work = sum(v for k, v in parts.items() if k not in ("workers", "distinct_blocks"))
    counts = {k: sum(1 for r in races if imap[r["id"]]["input"] == k) for k in (INPUT_VERIFY, INPUT_FILE, INPUT_NONE)}
    return dict(counts=counts, staged_bytes=0, work_bound_bytes=work, parts=parts)


def cmd_budget(args):
    B = _bb()
    races = plan(args.vendor, B)
    bud = budget(races, smoke_inputs(races, B), B, args.workers)
    print("MEASURESMOKE budget vendor=%s %s" % (args.vendor, " ".join(
        "%s=%d" % (k.replace(" ", "-"), n) for k, n in bud["counts"].items())))
    print("MEASURESMOKE budget staged=0 work_bound=%.2fGB budget=%.1fGB %s" % (
        bud["work_bound_bytes"] / 1e9, WORK_MAX_BYTES / 1e9,
        " ".join("%s=%s" % (k, ("%.1fMB" % (v / 1e6)) if k not in ("workers", "distinct_blocks") else v)
                 for k, v in sorted(bud["parts"].items()))))
    return 0 if bud["work_bound_bytes"] < WORK_MAX_BYTES else 1


# --------------------------------------------------------------------------------------------- data
def data_keys(vendor, B=None, inputs=None, imap=None):
    """{R2 key: box path relative to $HOME} the smoke plan reads: the board's datasets (bench_board.data_needs) under
    datasets/gbm-bench, and the algos corpus keys under r2-stage/<key> (bench_board_algos.corpus_path looks there).
    The union of race_keys over the plan: the run stages them group by group (DataStager). With the synthetic inputs
    (the default over a tree whose board takes them): none, the smoke stages no data."""
    B = B or _bb()
    inputs = inputs or (SYNTHETIC_INPUTS if board_takes_inputs() else DATASETS_INPUTS)
    if inputs == SYNTHETIC_INPUTS:
        return {}
    keys = {}
    for r in plan(vendor, B):
        keys.update(race_keys(r, B))
    return dict(sorted(keys.items()))


def pins(root=ROOT):
    out = {}
    for row in (root / "bench/results/dataset_store/manifest.tsv").read_text().splitlines():
        f = row.split("\t")
        if len(f) >= 3:
            out[f[0]] = (int(f[1]), f[2])
    return out


def presign_get(key, seconds):
    """One presigned GET minted here (tools/dataset_store.sh presign; the R2 secret never leaves this machine)."""
    got = subprocess.run(["sh", str(HERE / "dataset_store.sh"), "presign", key, str(seconds)],
                         capture_output=True, text=True)
    return got.stdout.strip() if got.returncode == 0 else ""


def write_data_plan(vendor, out, presign=presign_get, seconds=86400, inputs=None):
    """The data plan the box stages: each file the smoke plan reads, its pin and one presigned GET (presign=None: keys
    and pins only), and the inputs it was written for (the run refuses a plan of the other kind). Written 0600: the
    URLs are short-lived read grants. Returns the entries."""
    pinned = pins()
    entries = []
    inputs = inputs or (SYNTHETIC_INPUTS if board_takes_inputs() else DATASETS_INPUTS)
    for key, rel in data_keys(vendor, None, inputs).items():
        if key not in pinned:
            raise SystemExit("measuresmoke: %s has no pin in bench/results/dataset_store/manifest.tsv" % key)
        url = None
        if presign is not None:
            url = presign(key, seconds)
            if not str(url).startswith("https://"):
                raise SystemExit("measuresmoke: presigning %s failed" % key)
        entries.append(dict(key=key, path=rel, bytes=pinned[key][0], sha256=pinned[key][1], url=url))
    out = Path(out)
    out.write_text(json.dumps(dict(schema=SCHEMA + "/data", vendor=vendor, inputs=inputs, files=entries), indent=1)
                   + "\n")
    os.chmod(out, 0o600)
    return entries


def cmd_data_plan(args):
    entries = write_data_plan(args.vendor, args.out, None if args.no_urls else presign_get, args.seconds)
    print("MEASURESMOKE data-plan %d files, %.2f GB" % (len(entries), sum(e["bytes"] for e in entries) / 1e9))
    return 0


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def stage_one(entry, home):
    """'present' | 'fetched' | an error line. A present file is kept only when its size and sha256 are the pin's."""
    dest = Path(home) / entry["path"]
    if dest.is_file() and dest.stat().st_size == entry["bytes"] and file_sha256(dest) == entry["sha256"]:
        return "present"
    if not entry.get("url"):
        return "missing and no URL"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    cfg = dest.with_name(dest.name + ".curl")
    cfg.write_text('url = "%s"\n' % entry["url"])   # the URL never reaches the process list
    os.chmod(cfg, 0o600)
    try:
        rc = subprocess.run(["curl", "-sS", "-C", "-", "--retry", "8", "--retry-delay", "5", "--retry-all-errors",
                             "-o", str(part), "-K", str(cfg)]).returncode
    finally:
        cfg.unlink(missing_ok=True)
    if rc:
        return "curl exited %d" % rc
    if part.stat().st_size != entry["bytes"] or file_sha256(part) != entry["sha256"]:
        part.unlink(missing_ok=True)
        return "size/sha256 differ from the pin"
    part.replace(dest)
    return "fetched"


def cmd_stage(args):
    """On the box: every data-plan file at its pinned bytes, in parallel (I/O bound; one thread per file)."""
    doc = json.loads(Path(args.data_plan).read_text())
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max(1, len(doc["files"]))) as pool:
        got = list(pool.map(lambda e: stage_one(e, args.home), doc["files"]))
    bad = 0
    for e, g in zip(doc["files"], got):
        print("MEASURESMOKE data %s %s" % (e["key"], g), flush=True)
        bad += g not in ("present", "fetched")
    return 1 if bad else 0


def cmd_plan(args):
    ids = [r["id"] for r in plan(args.vendor)]
    print("MEASURESMOKE plan vendor=%s recipes=%d sha256=%s" % (args.vendor, len(ids), plan_sha256(ids)))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--vendor", required=True, choices=VENDORS)
    p = sub.add_parser("data-plan")
    p.add_argument("--vendor", required=True, choices=VENDORS)
    p.add_argument("--out", required=True)
    p.add_argument("--seconds", type=int, default=86400, help="presigned GET lifetime (the job may queue)")
    p.add_argument("--no-urls", action="store_true", help="keys and pins only (tests, dry runs)")
    p = sub.add_parser("budget")
    p.add_argument("--vendor", required=True, choices=VENDORS)
    p.add_argument("--workers", type=int, default=MAX_BOX_WORKERS)
    p = sub.add_parser("inputs")
    p.add_argument("--out", default=str(ROOT / "docs/measurements/SMOKE_INPUTS.md"))
    p = sub.add_parser("stage")
    p.add_argument("--data-plan", required=True)
    p.add_argument("--home", default=os.path.expanduser("~"))
    p = sub.add_parser("run")
    p.add_argument("--vendor", required=True, choices=VENDORS)
    p.add_argument("--python", required=True, help="the venv the exact release wheels are installed into")
    p.add_argument("--out", required=True, help="the summary and logs (fetched home)")
    p.add_argument("--work", required=True, help="the batch smoke boards, prep and caches (stay on the box)")
    p.add_argument("--data-root", default=os.path.join(os.path.expanduser("~"), "datasets", "gbm-bench"))
    p.add_argument("--fixture-python", default=None,
                   help="a venv with the fixture generator's CPU torch + torch-geometric (the layer lanes' inputs)")
    p.add_argument("--jobs", type=int, default=0, help="override the worker count (default: cores and memory)")
    p.add_argument("--ceiling", type=int, default=PHASE_CEILING_S, help="the phase ceiling, seconds")
    p.add_argument("--data-plan", default=None,
                   help="the data plan (data-plan): its files are staged per group and deleted after the last group "
                        "that reads them; without it the plan's data must already be at its pins")
    p.add_argument("--home", default=os.path.expanduser("~"), help="where the data plan's paths are rooted")
    p.add_argument("--max-work-gb", type=float, default=None,
                   help="cap on the smoke's footprint (data it fetched + its work dir), GB; a group that would pass "
                        "it waits for a running group to close, or is refused by name when nothing else runs "
                        "(default: the work budget, %.1f GB, with the synthetic inputs; %.0f GB over a tree whose "
                        "board reads the datasets)" % (WORK_MAX_BYTES / 1e9, MAX_WORK_GB))
    p.add_argument("--keep-scratch", action="store_true",
                   help="keep every group's prep cache and batch directories (diagnosis on a box with room)")
    args = ap.parse_args(argv)
    return {"plan": cmd_plan, "data-plan": cmd_data_plan, "stage": cmd_stage, "run": cmd_run, "inputs": cmd_inputs,
            "budget": cmd_budget}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
