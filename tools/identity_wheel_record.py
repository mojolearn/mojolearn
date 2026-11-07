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
2. select   per column, the recorder's own default selection (record_identity_column.sh SELECT_ONLY=1, the
            REUSE GOOD OLD RUNS rule: only lanes whose closure changed since this class's newest good evidence);
            nothing owed -> the column records nothing (REUSED n / RUN 0);
3. record   the owed lanes in chunks, ALL AT ONCE: one worker slot per (device, k) for k < workers per device,
            each chunk `RECORD_FROM_INSTALLED=1 LANES=<chunk> LANE_SELECTION=auto-reuse-chunk
            record_identity_column.sh <label> DIR/<label>/w<i>` pinned to its device (CUDA_VISIBLE_DEVICES /
            HIP_VISIBLE_DEVICES). With more devices than columns each column gets its own device(s); with fewer,
            columns share them. It prints "busy plan: W worker(s) of T" before it starts.
4. summary  DIR/identity-wheels.json (schema mojolearn.identity-wheels.v1): per column the selection, the
            records written (with their admissible= line) and the rc; exit 0 only when every chunk recorded
            admissibly. The records go to admission (tools/admit_identity_columns.sh) with every other column.

Timing ledger: every chunk and the whole run are lines of kind job (the run with busy cores of total, sampled
while it runs); each record's lanes are lines of kind lane (record_identity_column.sh). See tools/timing_ledger.py.

Nothing here builds, rents or publishes. Tests: tools/test_identity_wheel_record.py (fake recorder, no device).
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import timing_ledger  # noqa: E402

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
#: CPU threads per record worker: record_identity_column.sh's own default (CPU_THREADS=3), unchanged, so a
#: parallel run records under exactly the thread pools a serial one did.
CPU_THREADS = 3
#: assumed GPU memory per concurrent record worker (GiB) for --workers-per-gpu auto, until a first run measures
#: the peak (tools/identity_wheel_record.py prints the free memory it planned with)
GPU_GIB_PER_WORKER = 12


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def label(column, device):
    if column not in COLUMNS:
        raise SystemExit("identity_wheel_record: unknown column %s (%s)" % (column, ", ".join(COLUMNS)))
    pattern = COLUMNS[column][0]
    if "{device}" in pattern and not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", device or ""):
        raise SystemExit("identity_wheel_record: --label-device must be a lowercase slug (e.g. l40s, mi325x, m2pro)")
    return pattern.format(device=device)


def chunks(lanes, n):
    """N chunks of LANES, round robin (neighbouring lanes, often of one family, spread over the workers)."""
    n = max(1, min(n, len(lanes)))
    return [lanes[i::n] for i in range(n)] if lanes else []


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


def auto_workers(gpus, nproc, free_gib):
    """Record workers per device: every core busy (CPU_THREADS each), bounded by the measured free memory."""
    if gpus <= 0:
        return max(1, nproc // CPU_THREADS)
    by_cores = max(1, nproc // (CPU_THREADS * gpus))
    by_memory = max(1, int(free_gib // GPU_GIB_PER_WORKER)) if free_gib else 1
    return max(1, min(by_cores, by_memory))


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


class Run:
    def __init__(self, args, popen=subprocess.Popen, run=subprocess.run):
        self.a, self.popen, self.run = args, popen, run
        self.out = Path(args.out).resolve()
        self.lock = threading.Lock()

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
        venv = Path(self.a.venv or self.out / "venv")
        if venv.exists():
            shutil.rmtree(venv)
        self.run([self.a.python, "-m", "venv", str(venv)], check=True)
        py = venv / "bin" / "python"
        self.run([str(py), "-m", "pip", "install", "-q", "--no-cache-dir", *[str(w) for w in self.a.wheel]], check=True)
        listed = self.out / "installed-wheels.txt"
        listed.write_text("\n".join(lines) + "\n")
        return py, listed

    def env(self, py, listed, device, column, extra=None):
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "LANES", "SHARD", "ONLY_CHANGED", "FULL")}
        env.update(RECORD_FROM_INSTALLED="1", PYTHON=str(py), INSTALLED_WHEELS=str(listed), REPEATS="1",
                   SCOPE="all", CPU_THREADS=str(CPU_THREADS))
        var = COLUMNS[column][1]
        if var and device is not None:
            env[var] = str(device)
            if var == "HIP_VISIBLE_DEVICES":
                env["ROCR_VISIBLE_DEVICES"] = str(device)
        env.update(extra or {})
        return env

    def all_lanes(self, py, column, lab):
        """RUN EVERYTHING (Andrew 2026-11-05: the selector cannot be trusted to narrow): every lane of the full scope
        (identity_columns.py lanes --scope all), the harness's declared GPU-only lanes left out of the cpu column
        (identity_break.GPU_ONLY_LANES: no CPU route; a full CPU column drops them by name). No selector decides."""
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
        if column == "cpu":
            q = self.run([str(py), "-c", "import sys; sys.path.insert(0, 'tools'); import identity_columns as c; "
                          "print(','.join(sorted(getattr(c.harness(), 'GPU_ONLY_LANES', {}) or {})))"],
                         cwd=str(ROOT), capture_output=True, text=True, env=env)
            if q.returncode != 0:
                raise RuntimeError("could not read the harness's GPU-only lanes for %s" % lab)
            gpu_only = set(x for x in (q.stdout or "").strip().split(",") if x)
            lanes = [x for x in lanes if x not in gpu_only]
        return lanes, 0

    def select(self, py, listed, column, lab):
        """--reuse-unchanged only: the recorder's own reuse selection for this column (SELECT_ONLY=1)."""
        out = self.out / lab / "select"
        out.mkdir(parents=True, exist_ok=True)
        p = self.run(["bash", str(RECORDER), lab, str(out)], cwd=str(ROOT), capture_output=True, text=True,
                     env=self.env(py, listed, 0 if COLUMNS[column][1] else None, column, dict(SELECT_ONLY="1")))
        (out / "select.log").write_text((p.stdout or "") + (p.stderr or ""))
        if p.returncode != 0:
            raise RuntimeError("selection for %s exited %s (%s)" % (lab, p.returncode, out / "select.log"))
        text = p.stdout or ""
        reused = re.search(r"REUSED (\d+) / RECORD (\d+)", text)
        if "NOTHING-CHANGED" in text:
            return [], int(reused.group(1)) if reused else None
        m = re.search(r"^LANES-SELECTED (\S+)\s*$", text, re.M)
        if not m:
            raise RuntimeError("selection for %s printed no LANES-SELECTED line (%s)" % (lab, out / "select.log"))
        return m.group(1).split(","), int(reused.group(1)) if reused else None

    def main(self):
        self.out.mkdir(parents=True, exist_ok=True)
        columns = [c for c in self.a.columns.split(",") if c]
        if not columns or len(set(columns)) != len(columns):
            raise SystemExit("identity_wheel_record: --columns is a comma list of distinct columns")
        labels = {c: label(c, self.a.label_device) for c in columns}
        py, listed = self.install()
        nproc = os.cpu_count() or 1
        devices = list(range(self.a.gpus))
        vendor = "nvidia" if columns[0] in ("nvidia", "ptx") else columns[0]
        free = gpu_free_gib(vendor) if devices else None
        workers = (auto_workers(len(devices), nproc, free) if self.a.workers_per_gpu == "auto"
                   else int(self.a.workers_per_gpu))
        slots = plan_slots(columns, devices, workers)
        selected, report = {}, dict(schema=SCHEMA, columns={}, nproc=nproc, gpus=len(devices), shard=self.a.shard,
                                    workers_per_device=workers, gpu_free_gib=free,
                                    wheels=listed.read_text().split("\n")[:-1])
        shard_i, shard_n = (int(x) for x in self.a.shard.split("/"))
        for c in columns:
            lanes, reused = (self.select(py, listed, c, labels[c]) if self.a.reuse_unchanged
                             else self.all_lanes(py, c, labels[c]))
            # --shard i/N (the GitHub CPU matrix): every job selects by the same rule and records lanes[i::N]
            lanes = lanes[shard_i::shard_n]
            selected[c] = lanes
            report["columns"][c] = dict(label=labels[c], owed=len(lanes), reused=reused, chunks=[], rc=0)
            self.say("%s: REUSED %s / RUN %s" % (labels[c], "?" if reused is None else reused,
                                                 len(lanes) if self.a.reuse_unchanged else "all (%d)" % len(lanes)))
        tasks = []
        for c in columns:
            for i, (chunk, slot) in enumerate(zip(chunks(selected[c], len(slots[c])), slots[c])):
                tasks.append((c, i, chunk, slot))
        total = sum(len(s) for s in slots.values()) if devices else len(next(iter(slots.values())))
        busy = len(tasks)
        self.say("busy plan: %d worker(s) of %d (%d device(s), %d core(s), %d CPU thread(s) per worker%s)" % (
            busy, total, len(devices), nproc, CPU_THREADS,
            "" if free is None else ", %.0f GiB free per device" % free))
        threads, results = [], {}

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
            env = self.env(py, listed, slot[0], c, dict(LANES=",".join(chunk), LANE_SELECTION=(
                "auto-reuse-chunk" if self.a.reuse_unchanged else "full-chunk")))
            t0 = time.time()
            with open(out / "record.log", "w") as log:
                rc = self.popen(["bash", str(RECORDER), labels[c], str(out)], cwd=str(ROOT), env=env,
                                stdout=log, stderr=subprocess.STDOUT).wait()
            timing_ledger.safe(timing_ledger.record, "job", "identity %s w%d" % (labels[c], i), t0, time.time(),
                               rc=rc, extra=dict(device=slot[0], lanes=len(chunk), out=str(out)))
            text = (out / "record.log").read_text(errors="replace")
            m = re.findall(r"^RECORDED (\S+) rc=(\d+) admissible=(yes|NO) json=(\S+)$", text, re.M)
            results[(c, i)] = dict(worker=i, device=slot[0], lanes=len(chunk), rc=rc,
                                   records=[dict(name=n, harness_rc=int(r), admissible=adm == "yes", json=j)
                                            for n, r, adm, j in m])
            self.say("%s w%d (device %s): %d lane(s), rc=%d" % (labels[c], i, slot[0], len(chunk), rc))
        # every slot of a device runs one chunk at a time; chunks of columns sharing a device queue on it
        by_slot = {}
        for t in tasks:
            by_slot.setdefault(t[3], []).append(t)

        def drain(queue):
            for t in queue:
                work(*t)
        run_t0 = time.time()
        with timing_ledger.BusySampler() as sampler:
            for queue in by_slot.values():
                th = threading.Thread(target=drain, args=(queue,))
                th.start()
                threads.append(th)
            for th in threads:
                th.join()
        busy_summary = sampler.summary()
        report["busy_cores"] = busy_summary
        self.say("busy cores: %s of %d" % ("not sampled (run under 30 s)" if busy_summary is None else
                                            "mean %(mean)s, max %(max)s over %(samples)d sample(s)" % busy_summary,
                                            nproc))
        ok = True
        for (c, i), r in sorted(results.items()):
            row = report["columns"][c]
            row["chunks"].append(r)
            good = r["rc"] == 0 and r["records"] and all(x["admissible"] for x in r["records"])
            if not good:
                row["rc"] = r["rc"] or 1
                ok = False
        report["ok"] = ok
        failed = ["%s w%d (%d lane(s), rc %s)" % (labels[c], i, r["lanes"], r["rc"]) for (c, i), r in sorted(results.items())
                  if not (r["rc"] == 0 and r["records"] and all(x["admissible"] for x in r["records"]))]
        report["failures"] = failed
        if failed:
            self.say("FAILURES (%d chunk(s); every other chunk ran): %s" % (len(failed), "; ".join(failed)))
        (self.out / "identity-wheels.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
        timing_ledger.safe(timing_ledger.record, "job", "identity-wheels " + ",".join(labels[c] for c in columns),
                           run_t0, time.time(), cores=nproc, busy=busy_summary, rc=0 if ok else 1,
                           extra=dict(out=str(self.out), workers=busy, gpus=len(devices)))
        self.say("%s: %s" % ("DONE" if ok else "FAILED", self.out / "identity-wheels.json"))
        return 0 if ok else 1


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
    ap.add_argument("--reuse-unchanged", action="store_true",
                    help="opt in to the recorder's reuse selection (unchanged lanes are not run); default: every lane")
    a = ap.parse_args(argv)
    m = re.fullmatch(r"(\d+)/(\d+)", a.shard)
    if not m or not 0 <= int(m.group(1)) < int(m.group(2)):
        ap.error("--shard is i/N with 0 <= i < N")
    if a.workers_per_gpu != "auto" and not a.workers_per_gpu.isdigit():
        ap.error("--workers-per-gpu is auto or a positive number")
    return a


if __name__ == "__main__":
    sys.exit(Run(parse()).main())
