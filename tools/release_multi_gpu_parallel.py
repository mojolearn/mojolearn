#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""multi-gpu-<vendor> producer: parallel CV on two physical GPUs from INSTALLED candidate wheels.

Test controller only: never compiles or edits product code. On a box with two
or more GPUs of one vendor it

  1. installs the exact candidate wheels into a fresh venv (records their
     sha256 and the packaged source commit);
  2. records the driver tool's GPU inventory (nvidia-smi / amd-smi or rocm-smi:
     index, UUID, PCI bus id) and samples the driver's per-process GPU table for
     the whole run (the physical execution trace);
  3. runs the shipped verifier ``python -m mojolearn._verify_parallel_cv
     --require-installed --devices D0,D1`` (five uneven folds, one/two/reversed
     schedules, saved-model replay, eight comparator controls);
  4. runs a probe in the installed venv: two-device scores equal the serial
     native CV bit for bit; native fold-to-device assignments and worker pids;
     Python callback refusals (estimator, scorer, splitter, protocol, devices)
     with no worker started; wrong-device refusals (repeated and nonexistent
     index); a native fold fault (an estimator whose native fit fails) on two
     devices fails with no partial result and the next run recovers; and a
     hidden-device control (the parent sees one GPU but asks for two) refuses;
  5. judges everything from the retained files (``judge``), joining each worker
     pid and the PCI device its driver inventory reported with the driver
     tool's inventory and process samples.

    python3 tools/release_multi_gpu_parallel.py run --vendor nvidia --devices 0,1 \\
        --wheel CORE.whl --wheel PLUGIN.whl --out NEW_DIR [--python python3.12]
    python3 tools/release_multi_gpu_parallel.py judge --out DIR

The ledger adapter (tools/release_installed_evidence.py, adapter ``multi-gpu``)
re-runs ``judge`` over the retained files. Coverage IDs:
  native device placement                      fold records on the requested physical GPUs,
                                               probe assignments, one driver-table sighting per GPU
  consumed outputs                             six validated capture runs, parallel == serial,
                                               recovery after the fault == serial
  wrong-device and callback negative controls  8 comparator + 5 callback + 2 wrong-device
                                               + 1 hidden-device + 1 native fold fault
A native arithmetic fault (perturbing a kernel result inside a fold) is not
injected: that needs a product test hook and is reported as such.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile

HERE = Path(__file__).resolve().parent
REPORT = "multi-gpu-report.json"
REPORT_SCHEMA = "mojolearn.release-multi-gpu.v1"
BACKEND = {"nvidia": "cuda", "amd": "hip"}
VISIBLE = {"nvidia": "CUDA_VISIBLE_DEVICES", "amd": "HIP_VISIBLE_DEVICES"}
COVERAGE_IDS = ("native device placement", "consumed outputs", "wrong-device and callback negative controls")
CALLBACK_KINDS = ("estimator", "scorer", "splitter", "protocol", "devices")
CAPTURE_CONTROLS = 8
CAPTURE_RUNS = 6
NATIVE_ARITHMETIC_FAULT = ("not injected: perturbing a fold's native arithmetic needs a product test hook "
                           "(a native fault-injection switch in the parallel CV worker); the native fold "
                           "fault control here is a real native fit failure")
SAMPLE_SECONDS = 0.05


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normal_pci(value):
    """'00000000:3B:00.0', '0000:3b:00.0' and '3b:00.0' -> '0:3b:00.0'."""
    parts = str(value).strip().lower().split(":")
    require(len(parts) in (2, 3) and "." in parts[-1], "malformed PCI bus id: " + str(value))
    domain = int(parts[0], 16) if len(parts) == 3 else 0
    return "%x:%02x:%s" % (domain, int(parts[-2], 16), parts[-1])


# ---------------------------------------------------------------- driver tools
def _parse_nvidia_inventory(text):
    rows = []
    for line in text.strip().splitlines():
        index, uuid, bus, *name = [part.strip() for part in line.split(",")]
        rows.append(dict(index=int(index), uuid=uuid, pci_bus_id=normal_pci(bus), name=",".join(name).strip()))
    return rows


def _parse_nvidia_processes(text):
    rows = []
    for line in text.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit():
            rows.append(dict(pid=int(parts[0]), pci_bus_id=normal_pci(parts[1])))
    return rows


def _parse_amd_inventory(text):
    """amd-smi list --json: [{gpu, bdf, uuid}, ...]."""
    rows = []
    for item in json.loads(text):
        rows.append(dict(index=int(item["gpu"]), uuid=str(item.get("uuid", "")),
                         pci_bus_id=normal_pci(item["bdf"]), name=str(item.get("name", ""))))
    return rows


def _parse_amd_processes(text, inventory):
    """rocm-smi --showpidgpus: 'PID 1234 is using 1 DRM device(s):' followed by the GPU index line."""
    by_index = {row["index"]: row["pci_bus_id"] for row in inventory}
    rows = []
    for pid, indices in re.findall(r"PID (\d+) is using \d+ DRM device\(s\):\s*\n\s*([\d ]+)", text):
        for index in indices.split():
            if int(index) in by_index:
                rows.append(dict(pid=int(pid), pci_bus_id=by_index[int(index)]))
    return rows


def driver_inventory(vendor):
    if vendor == "nvidia":
        command = ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,name", "--format=csv,noheader"]
        text = subprocess.check_output(command, text=True, timeout=60)
        return command, text, _parse_nvidia_inventory(text)
    command = ["amd-smi", "list", "--json"]
    text = subprocess.check_output(command, text=True, timeout=60)
    return command, text, _parse_amd_inventory(text)


class Sampler(threading.Thread):
    """The driver's per-process GPU table, sampled for the whole run."""
    def __init__(self, vendor, inventory, path):
        super().__init__(daemon=True)
        self.vendor, self.inventory, self.path = vendor, inventory, Path(path)
        self.stop_event = threading.Event()
        self.samples = self.errors = 0

    def query(self):
        if self.vendor == "nvidia":
            text = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_bus_id",
                                            "--format=csv,noheader"], text=True, timeout=30)
            return _parse_nvidia_processes(text)
        text = subprocess.check_output(["rocm-smi", "--showpidgpus"], text=True, timeout=30)
        return _parse_amd_processes(text, self.inventory)

    def run(self):
        with self.path.open("a") as out:
            while not self.stop_event.is_set():
                try:
                    rows = self.query()
                    self.samples += 1
                    out.write(json.dumps(dict(t=time.time(), rows=rows)) + "\n")
                    out.flush()
                except (subprocess.SubprocessError, OSError, ValueError) as error:
                    self.errors += 1
                    out.write(json.dumps(dict(t=time.time(), error=str(error))) + "\n")
                self.stop_event.wait(SAMPLE_SECONDS)

    def stop(self):
        self.stop_event.set()
        self.join(timeout=60)


# ---------------------------------------------------------------- probe (runs in the installed venv)
def probe(request_path):
    request = json.loads(Path(request_path).read_text())
    out = Path(request["result"])
    devices = tuple(request["devices"])
    checks, workers = [], []
    result = dict(mode=request["mode"], checks=checks, workers=workers)

    def check(name, passed, **detail):
        checks.append(dict(name=name, passed=bool(passed), **detail))

    try:
        import numpy as np
        import mojolearn as ml
        from mojolearn import GradientBoostingRegressor, LinearRegression, _backend
        from mojolearn import model_selection as serial
        from mojolearn import parallel_model_selection as parallel
        result.update(package=str(Path(ml.__file__).resolve()), vendor=ml.vendor(),
                      numeric_mode=_backend.default_mode())

        def trace():
            return serial._NATIVE(None, "family_msel_parallel_last_run", (), serial._NATIVE_CONTEXT)

        rng = np.random.default_rng(37)
        x = rng.normal(size=(41, 4)).astype("f4")
        y = (2 * x[:, 0] - x[:, 1] + .2 * x[:, 2] ** 2).astype("f4")

        def model(**options):
            return GradientBoostingRegressor(n_estimators=3, max_depth=options.get("max_depth", 2),
                                             random_state=7, numeric_mode="identical")

        if request["mode"] == "hidden":
            # The parent process sees one GPU and asks for two: a refusal, never scores.
            try:
                scores = parallel.cross_val_score(model(), x, y, devices=devices, cv=5)
                check("hidden-device", False, scores=np.asarray(scores).tobytes().hex())
            except Exception as error:  # any refusal; scores are never returned
                check("hidden-device", True, error=type(error).__name__ + ": " + str(error))
            return result

        expected = serial.cross_val_score(model(), x, y, cv=5)
        actual = parallel.cross_val_score(model(), x, y, devices=devices, cv=5)
        record = trace()
        workers.extend(int(pid) for pid in record.get("workers", []))
        width = len(devices)
        check("parallel-equals-serial", np.asarray(actual).tobytes() == np.asarray(expected).tobytes(),
              parallel=np.asarray(actual).tobytes().hex(), serial=np.asarray(expected).tobytes().hex())
        assignments = [list(row) for row in record.get("assignments", [])]
        result["assignments"] = assignments
        check("assignments", record.get("status") == "COMPLETE"
              and [row[0] for row in assignments] == list(range(5))
              and [row[1] for row in assignments] == [devices[i % width] for i in range(5)]
              and len(set(record.get("workers", []))) == width, record=record)

        for kind in CALLBACK_KINDS:
            calls = []

            class Foreign:
                def __getattribute__(self, name):
                    calls.append(name)
                    raise AssertionError("unsupported protocol invoked")

                def __call__(self, *args):
                    calls.append("call")
                    return 1.

                def __iter__(self):
                    calls.append("iter")
                    return iter(())
            kwargs, estimator, features, value = dict(devices=devices, cv=3), LinearRegression(), x, Foreign()
            if kind == "estimator":
                estimator = value
            elif kind == "scorer":
                kwargs["scoring"] = value
            elif kind == "splitter":
                kwargs["cv"] = value
            elif kind == "protocol":
                features = value
            else:
                kwargs["devices"] = value
            try:
                parallel.cross_val_score(estimator, features, y, **kwargs)
                check("callback:" + kind, False, error=None)
            except TypeError as error:
                check("callback:" + kind, calls == [] and trace().get("workers") == [],
                      error=str(error), calls=list(calls))
            except Exception as error:
                check("callback:" + kind, False, error=type(error).__name__ + ": " + str(error))

        for name, wrong in (("repeated", (devices[0], devices[0])), ("nonexistent", (devices[0], request["absent"]))):
            try:
                parallel.cross_val_score(model(), x, y, devices=wrong, cv=5)
                check("wrong-device:" + name, False, devices=list(wrong))
            except Exception as error:
                check("wrong-device:" + name, not trace().get("status") == "COMPLETE",
                      devices=list(wrong), error=type(error).__name__ + ": " + str(error))

        try:
            parallel.cross_val_score(model(max_depth=-1), x, y, devices=devices, cv=5)
            check("native-fold-fault", False)
        except RuntimeError as error:
            failed = trace()
            workers.extend(int(pid) for pid in failed.get("workers", []))
            check("native-fold-fault", "native model worker failed" in str(error)
                  and failed.get("status") == "FAILED" and failed.get("closed") is True,
                  error=str(error), record=failed)
        except Exception as error:
            check("native-fold-fault", False, error=type(error).__name__ + ": " + str(error))
        recovered = parallel.cross_val_score(model(), x, y, devices=devices, cv=5)
        workers.extend(int(pid) for pid in trace().get("workers", []))
        check("recovery-equals-serial", np.asarray(recovered).tobytes() == np.asarray(expected).tobytes())
    except Exception as error:
        result["error"] = type(error).__name__ + ": " + str(error)
    finally:
        out.write_text(json.dumps(result, indent=2, default=str) + "\n")
    return 0


# ---------------------------------------------------------------- judge (pure, over retained files)
def _load(path):
    return json.loads(Path(path).read_text())


def _samples(path):
    rows, errors = [], 0
    for line in Path(path).read_text().splitlines():
        if line.strip():
            item = json.loads(line)
            if "error" in item:
                errors += 1
            else:
                rows.append(item)
    return rows, errors


def judge(out):
    out = Path(out).resolve()
    request = _load(out / "request.json")
    vendor, devices = request["vendor"], request["devices"]
    require(vendor in BACKEND and len(devices) == 2 and len(set(devices)) == 2, "two distinct devices on nvidia or amd")
    inventory = _load(out / "driver-inventory.json")["parsed"]
    require(len(inventory) >= 2, "the box has fewer than two GPUs of this vendor")
    by_index = {row["index"]: row["pci_bus_id"] for row in inventory}
    require(all(d in by_index for d in devices), "requested device index absent from the driver inventory")
    physical = {d: by_index[d] for d in devices}
    rows = {name: [] for name in COVERAGE_IDS}

    def add(coverage, name, passed, **detail):
        rows[coverage].append(dict(name=name, passed=bool(passed), **detail))

    # 1. the shipped verifier capture
    capture = _load(out / "capture/report.json")
    capture_ok = (capture.get("status") == "NUMERICS_AND_PLACEMENT_PASS" and capture.get("vendor") == BACKEND[vendor]
                  and capture.get("devices") == devices and len(capture.get("runs", [])) == CAPTURE_RUNS
                  and (capture.get("distribution") or {}).get("import_matches_distribution") is True
                  and (capture.get("distribution") or {}).get("editable") is False)
    worker_gpu = {}
    for run in capture.get("runs", []):
        order = run.get("devices", [])
        wanted = {physical.get(d) for d in order}
        seen = set()
        for fixture, record in sorted((run.get("fold_records") or {}).items()):
            inv = record.get("inventory") or {}
            gpus = [normal_pci(dev["pci_bus_id"]) for dev in inv.get("devices", [])]
            ok = len(gpus) == 1 and gpus[0] in wanted and type(inv.get("pid")) is int
            if ok:
                if worker_gpu.setdefault(inv["pid"], gpus[0]) != gpus[0]:
                    ok = False
                seen.add(gpus[0])
            add("native device placement", "capture:%s:%s:%s" % (run.get("model"), "-".join(map(str, order)), fixture[:12]),
                ok, pid=inv.get("pid"), gpu=gpus)
        add("native device placement", "capture-schedule:%s:%s" % (run.get("model"), "-".join(map(str, order))),
            seen == wanted, used=sorted(seen), wanted=sorted(x for x in wanted if x))
        add("consumed outputs", "capture-run:%s:%s" % (run.get("model"), "-".join(map(str, order))),
            capture_ok and len(run.get("fold_records") or {}) == 5)
    controls = capture.get("controls", [])
    for name in sorted(controls):
        add("wrong-device and callback negative controls", "comparator:" + name, capture_ok)
    if len(controls) != CAPTURE_CONTROLS:
        add("wrong-device and callback negative controls", "comparator:complete", False, found=len(controls))

    # 2. the installed-venv probe
    result = _load(out / "probe.json")
    checks = {row["name"]: row for row in result.get("checks", [])}
    probe_ok = result.get("vendor") == BACKEND[vendor] and result.get("numeric_mode") == "identical" and "error" not in result
    for row in result.get("assignments", []):
        add("native device placement", "probe-fold:%s" % row[0], probe_ok and row[1] in devices, device=row[1])
    add("native device placement", "probe-assignments", probe_ok and checks.get("assignments", {}).get("passed"))
    for name in ("parallel-equals-serial", "recovery-equals-serial"):
        add("consumed outputs", name, probe_ok and checks.get(name, {}).get("passed"))
    names = ["callback:" + k for k in CALLBACK_KINDS] + ["wrong-device:repeated", "wrong-device:nonexistent",
                                                        "native-fold-fault"]
    for name in names:
        add("wrong-device and callback negative controls", name, probe_ok and checks.get(name, {}).get("passed"))
    hidden = {row["name"]: row for row in _load(out / "hidden.json").get("checks", [])}
    add("wrong-device and callback negative controls", "hidden-device", hidden.get("hidden-device", {}).get("passed"))

    # 3. the physical execution trace: driver process table x worker pids
    samples, sample_errors = _samples(out / "driver-processes.jsonl")
    workers = set(worker_gpu) | {int(pid) for pid in result.get("workers", [])}
    sighted = {}
    stray = []
    for sample in samples:
        for row in sample["rows"]:
            if row["pid"] in workers:
                sighted.setdefault(row["pci_bus_id"], set()).add(row["pid"])
                expected_gpu = worker_gpu.get(row["pid"])
                if row["pci_bus_id"] not in physical.values() or (expected_gpu and expected_gpu != row["pci_bus_id"]):
                    stray.append(row)
    for d in devices:
        add("native device placement", "driver-table:%d" % d, bool(sighted.get(physical[d])) and not stray,
            gpu=physical[d], pids=sorted(sighted.get(physical[d], ())))
    trace = dict(source="driver process table x worker driver inventory", samples=len(samples),
                 sample_errors=sample_errors, requested=physical,
                 worker_gpu={str(pid): gpu for pid, gpu in sorted(worker_gpu.items())},
                 sighted={gpu: sorted(pids) for gpu, pids in sorted(sighted.items())}, stray=stray)
    coverage = []
    for name in COVERAGE_IDS:
        items = rows[name]
        passed = sum(1 for item in items if item["passed"])
        ok = bool(items) and passed == len(items)
        coverage.append(dict(id=name, status="passed" if ok else "failed", exit_code=0 if ok else 1,
                             expected=len(items), completed=passed, failed=len(items) - passed, skipped=0))
    passed = all(row["status"] == "passed" for row in coverage) and request.get("capture_exit") == 0
    evidence = [dict(path=str(p), sha256=sha256(p)) for p in
                (out / "request.json", out / "driver-inventory.json", out / "driver-processes.jsonl",
                 out / "capture/report.json", out / "probe.json", out / "hidden.json", out / "install.log")]
    return dict(schema=REPORT_SCHEMA, tool="tools/release_multi_gpu_parallel.py", vendor=vendor,
                backend=BACKEND[vendor], numeric_mode="identical", devices=devices,
                source_revision=request.get("source_commit"), wheels=request.get("wheels"),
                coverage=coverage, checks=rows, physical_execution_trace=trace,
                native_fault_controls=dict(fold_fault=checks.get("native-fold-fault", {}).get("passed") is True,
                                           hidden_device=hidden.get("hidden-device", {}).get("passed") is True,
                                           arithmetic=NATIVE_ARITHMETIC_FAULT),
                status="passed" if passed else "failed", exit_code=0 if passed else 1, evidence=evidence)


# ---------------------------------------------------------------- controller
def wheel_commit(wheels):
    found = set()
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            if "mojolearn/identity_columns/COMMIT" in archive.namelist():
                found.add(archive.read("mojolearn/identity_columns/COMMIT").decode().strip())
    require(len(found) == 1 and re.fullmatch("[0-9a-f]{40}", next(iter(found))),
            "the candidate wheels name no single packaged source commit")
    return found.pop()


def environment(vendor):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME") and not k.startswith("MOJOLEARN_")}
    env.update(PYTHONNOUSERSITE="1", MOJOLEARN_NUMERIC_MODE="identical", CUDA_DEVICE_ORDER="PCI_BUS_ID")
    for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        env.pop(key, None)
    return env


def run(args):
    out = args.out.resolve()
    require(not out.exists(), "output must be new")
    devices = [int(part) for part in args.devices.split(",")]
    require(len(devices) == 2 and len(set(devices)) == 2 and min(devices) >= 0, "two distinct device indices")
    wheels = [Path(w).resolve() for w in args.wheel]
    out.mkdir(parents=True)
    (out / "work").mkdir()
    env = environment(args.vendor)
    request = dict(vendor=args.vendor, devices=devices, wheels={w.name: sha256(w) for w in wheels},
                   source_commit=wheel_commit(wheels), controller_sha256=sha256(__file__))
    command, text, parsed = driver_inventory(args.vendor)
    (out / "driver-inventory.json").write_text(json.dumps(dict(command=command, raw=text, parsed=parsed), indent=2) + "\n")
    require(len(parsed) >= 2, "multi-gpu-%s needs a box with two or more GPUs; this one has %d"
            % (args.vendor, len(parsed)))
    log = out / "install.log"
    venv = out / "venv"
    python = str(venv / "bin/python")
    for argv in ([args.python, "-m", "venv", str(venv)],
                 [python, "-m", "pip", "install", "--no-input", "--disable-pip-version-check",
                  *map(str, wheels), "numpy"]):
        with log.open("ab") as stream:
            require(subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, env=env).returncode == 0,
                    "venv/install failed; see install.log")
    sampler = Sampler(args.vendor, parsed, out / "driver-processes.jsonl")
    sampler.start()
    try:
        with (out / "capture.log").open("wb") as stream:
            request["capture_exit"] = subprocess.run(
                [python, "-m", "mojolearn._verify_parallel_cv", "--require-backend", BACKEND[args.vendor],
                 "--require-installed", "--devices", args.devices, "--out", str(out / "capture")],
                stdout=stream, stderr=subprocess.STDOUT, env=env, cwd=out / "work", timeout=args.timeout).returncode
        shutil.copyfile(__file__, out / "controller.py")
        for mode, visible in (("probe", None), ("hidden", str(devices[0]))):
            spec = dict(mode=mode, devices=devices if mode == "probe" else [0, 1],
                        absent=len(parsed), result=str(out / (mode + ".json")))
            (out / (mode + "-request.json")).write_text(json.dumps(spec) + "\n")
            child = dict(env)
            if visible is not None:
                child[VISIBLE[args.vendor]] = visible
            with (out / (mode + ".log")).open("wb") as stream:
                request[mode + "_exit"] = subprocess.run(
                    [python, str(out / "controller.py"), "probe", "--request", str(out / (mode + "-request.json"))],
                    stdout=stream, stderr=subprocess.STDOUT, env=child, cwd=out / "work", timeout=args.timeout).returncode
    finally:
        sampler.stop()
    (out / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    try:
        report = judge(out)
    except (OSError, ValueError, KeyError, TypeError) as error:
        report = dict(schema=REPORT_SCHEMA, tool="tools/release_multi_gpu_parallel.py", vendor=args.vendor,
                      status="failed", exit_code=1, error=type(error).__name__ + ": " + str(error))
    (out / REPORT).write_text(json.dumps(report, indent=2) + "\n")
    print("MULTI_GPU vendor=%s status=%s %s" % (args.vendor, report["status"], " ".join(
        "%s=%d/%d" % (row["id"].split()[0], row["completed"], row["expected"]) for row in report.get("coverage", []))))
    return report["exit_code"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    go = sub.add_parser("run")
    go.add_argument("--vendor", choices=sorted(BACKEND), required=True)
    go.add_argument("--devices", default="0,1")
    go.add_argument("--wheel", action="append", required=True)
    go.add_argument("--python", default=sys.executable)
    go.add_argument("--timeout", type=int, default=900)
    go.add_argument("--out", type=Path, required=True)
    again = sub.add_parser("judge")
    again.add_argument("--out", type=Path, required=True)
    child = sub.add_parser("probe")
    child.add_argument("--request", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        return run(args)
    if args.command == "probe":
        return probe(args.request)
    report = judge(args.out)
    print(json.dumps({k: report[k] for k in ("status", "coverage")}))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
