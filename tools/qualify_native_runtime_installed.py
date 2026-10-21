#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Test selected public native-runtime operations from actual installed wheels.

Test/packaging controller only: never compiles, patches a loader, or changes a
wheel's source witness. Linux assembly delegates to qualification_wheels.py;
prebuilt Mac/Linux wheels can use the same fresh-install checks. This produces
partial installed-test evidence, never a release or native source-reuse proof.
"""
import argparse
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import types
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_TESTS = {
    "test_samba_step_native.py": "SambaStack.train_step, optimizer owners, rollback and native schedule refusal",
    "test_samba_inference_native.py": "SambaStack.forward/loss/allocate_state/step and consumed full states",
    "test_samba_checkpoint_native.py": "SambaStack save/load and continuation/corruption",
    "test_samba_native_admission.py": "Public Samba argument admission before Python hooks",
    "test_sequential_native_execution.py": "Sequential.fit/predict/score and consumed training state",
    "test_sequential_native_checkpoint.py": "Sequential checkpoint state and continuation",
    "test_sequential_native_admission.py": "Public Sequential argument admission before Python hooks",
    "test_generator_draw_native.py": "Generator public draws and stream/error semantics",
    "test_generation_native.py": "CausalLM generation/stream/stopping/resume and consumed outputs",
}


def require(condition, why):
    if not condition:
        raise ValueError(why)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def wheel_inventory(wheels):
    """Verify every wheel RECORD row independently before installing anything."""
    payload, records = {}, []
    for path in wheels:
        path = Path(path).resolve()
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            require(len(names) == len(set(names)), "duplicate ZIP member")
            files = {name for name in names if not name.endswith("/")}
            for name in names:
                parts = PurePosixPath(name).parts
                require(not name.startswith("/") and "\\" not in name
                        and ".." not in parts, "unsafe ZIP path")
            record_names = [name for name in files if name.endswith(".dist-info/RECORD")]
            require(len(record_names) == 1, "wheel must have one RECORD")
            record_name = record_names[0]
            rows = list(csv.reader(io.StringIO(archive.read(record_name).decode())))
            require(all(len(row) == 3 for row in rows), "invalid RECORD row")
            require(len(rows) == len({row[0] for row in rows}), "duplicate RECORD row")
            require({row[0] for row in rows} == files, "RECORD does not cover every wheel file")
            for name, recorded, size in rows:
                raw = archive.read(name)
                sha = hashlib.sha256(raw).hexdigest()
                if name == record_name:
                    require(not recorded and not size, "RECORD self hash must be blank")
                else:
                    encoded = base64.urlsafe_b64encode(bytes.fromhex(sha)).decode().rstrip("=")
                    require(recorded == "sha256=" + encoded and size == str(len(raw)),
                            "wheel RECORD mismatch: " + name)
                require(name not in payload, "two wheels own " + name)
                payload[name] = dict(sha256=sha, bytes=len(raw), wheel=str(path))
            records.append(dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size))
    require("mojolearn/__init__.py" in payload, "core wheel missing")
    return payload, records


def source_match(source, commit, wheels, payload):
    require(subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
                                    text=True).strip() == commit, "source checkout is not requested freeze")
    witness = "mojolearn/identity_columns/COMMIT"
    require(witness in payload, "wheel source witness missing")
    with zipfile.ZipFile(payload[witness]["wheel"]) as archive:
        require(archive.read(witness).decode().strip() == commit, "wheel source witness differs")
    matched = {}
    for member, row in payload.items():
        path = source / "python" / member
        if member.startswith("mojolearn/") and member.endswith(".py") and path.is_file():
            require(digest(path) == row["sha256"], "wheel/source Python mismatch: " + member)
            blob = subprocess.check_output(["git", "-C", str(source), "show", commit + ":python/" + member])
            require(hashlib.sha256(blob).hexdigest() == row["sha256"], "uncommitted package source: " + member)
            matched[member] = row["sha256"]
    return matched


def installed_module_inventory(modules, prefix, payload, tests):
    """Tests may live externally; every product Python/extension byte must ship."""
    prefix, tests = Path(prefix).resolve(), Path(tests).resolve()
    found = {}
    candidates = dict(modules)
    for name, module in list(modules.items()):
        if name.startswith("mojolearn") and isinstance(module, types.ModuleType):
            for key, value in vars(module).items():
                if isinstance(value, types.ModuleType):
                    candidates.setdefault(name + ":" + key, value)
    for name, module in candidates.items():
        if not isinstance(module, types.ModuleType):
            continue
        path = vars(module).get("__file__")
        if not path:
            continue
        path = Path(path).resolve()
        actual_name = vars(module).get("__name__", name)
        if actual_name.startswith("mojolearn.tests"):
            require(path.is_relative_to(tests), "test namespace escaped frozen test copy")
            continue
        if not (actual_name.startswith(("mojolearn", "_mojolearn")) or path.name.startswith("_mojolearn")):
            continue
        require(path.is_relative_to(prefix) and "site-packages" in path.parts,
                "product imported outside installed environment: " + str(path))
        at = path.parts.index("site-packages")
        member = PurePosixPath(*path.parts[at + 1:]).as_posix()
        require(member in payload, "loaded product file absent from wheel: " + member)
        require(digest(path) == payload[member]["sha256"], "installed product bytes differ: " + member)
        found[member] = dict(path=str(path), sha256=payload[member]["sha256"],
                             native=path.suffix in (".so", ".dylib", ".pyd"))
    require("mojolearn/__init__.py" in found, "installed package was not observed")
    return found


def junit_summary(path):
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    failed = [dict(name=c.get("name"), classname=c.get("classname"),
                   reason="\n".join((x.text or x.get("message", "")) for x in list(c)
                                    if x.tag in ("failure", "error")))
              for c in cases if c.find("failure") is not None or c.find("error") is not None]
    skipped = [dict(name=c.get("name"), reason=c.find("skipped").get("message", ""))
               for c in cases if c.find("skipped") is not None]
    return dict(total=len(cases), passed=len(cases)-len(failed)-len(skipped),
                failures=failed, skipped=skipped)


def worker(request_path):
    request = json.loads(Path(request_path).read_text())
    out, tests = Path(request["out"]), Path(request["tests"])
    require(Path.cwd() == out / "work", "test worker must run outside the source checkout")
    import mojolearn
    import pytest
    require(mojolearn.vendor() == request["vendor"], "installed backend silently changed")
    from mojolearn import _backend
    require(_backend.default_mode() == request["numeric_mode"], "installed numeric mode differs")
    namespace = types.ModuleType("mojolearn.tests")
    namespace.__path__ = [str(tests)]
    namespace.__package__ = "mojolearn.tests"
    sys.modules["mojolearn.tests"] = namespace
    mojolearn.tests = namespace
    loaded = {}

    class InstalledGuard:
        def pytest_runtest_teardown(self, item, nextitem):
            loaded.update(installed_module_inventory(sys.modules, sys.prefix, request["payload"], tests))

    code = 1
    try:
        loaded.update(installed_module_inventory(sys.modules, sys.prefix, request["payload"], tests))
        code = int(pytest.main(["--import-mode=importlib", "-q", "-ra", "--confcutdir=" + str(tests),
                               "--junitxml=" + str(out / "tests.xml"),
                               "--basetemp=" + str(out / "test-tmp")]
                              + [str(tests / name) for name in request["test_files"]],
                              plugins=[InstalledGuard()]))
    finally:
        loaded.update(installed_module_inventory(sys.modules, sys.prefix, request["payload"], tests))
        (out / "installed-modules.json").write_text(json.dumps(dict(
            package=str(Path(mojolearn.__file__).resolve()), prefix=sys.prefix,
            vendor=mojolearn.vendor(), numeric_mode=_backend.default_mode(), modules=loaded,
            pytest_exit=code), indent=2) + "\n")
    return code


def run_command(name, command, out, *, cwd, env=None, timeout=900):
    with (out / (name + ".log")).open("wb") as log:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=timeout)
    require(result.returncode == 0, name + " failed; see retained log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--wheel", type=Path, action="append", default=[])
    parser.add_argument("--installed-site", type=Path)
    parser.add_argument("--native-overlay", type=Path)
    parser.add_argument("--native-proof", type=Path, action="append", default=[])
    parser.add_argument("--vendor", choices=("cpu", "metal", "cuda", "hip"), required=True)
    parser.add_argument("--numeric-mode", choices=("identical", "fast"), default="identical")
    parser.add_argument("--test-file", choices=tuple(PUBLIC_TESTS), action="append")
    parser.add_argument("--dependency", action="append", default=[])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    source, out = args.source_root.resolve(), args.out.resolve()
    require(not out.exists() and not out.is_relative_to(source), "output must be new and outside source")
    require(bool(args.wheel) != bool(args.installed_site), "supply wheels OR installed-site for Linux assembly")
    require(args.numeric_mode != "fast" or args.vendor == "metal", "fast is Apple only")
    out.mkdir(parents=True)
    report = dict(schema="mojolearn.installed-public-native-audit.v1", source_commit=args.source_commit,
                  vendor=args.vendor, numeric_mode=args.numeric_mode, release_qualified=False,
                  coverage_complete=False, native_source_reuse_qualified=False,
                  scope="Selected public operations only; no private-owner or whole-library qualification",
                  source_root=str(source), controller_sha256=digest(__file__), status="running")
    try:
        wheels = args.wheel
        if wheels:
            retained = out / "input-wheels"; retained.mkdir()
            require(len({p.name for p in wheels}) == len(wheels), "duplicate supplied wheel filename")
            for path in wheels:
                shutil.copyfile(path, retained / path.name)
            wheels = [retained / path.name for path in wheels]
        if args.installed_site:
            require(sys.platform.startswith("linux"), "qualification assembly is Linux only; supply existing Mac wheels")
            command = [args.python, str(source / "packaging/linux/qualification_wheels.py"),
                       "--source-root", str(source), "--source-commit", args.source_commit,
                       "--installed-site", str(args.installed_site.resolve()), "--out", str(out / "wheels")]
            if args.native_overlay:
                command += ["--native-overlay", str(args.native_overlay.resolve())]
            run_command("assemble", command, out, cwd=out)
            manifest = json.loads((out / "wheels/qualification-inputs.json").read_text())
            report["assembly_manifest"] = dict(path=str(out / "wheels/qualification-inputs.json"),
                                               sha256=digest(out / "wheels/qualification-inputs.json"))
            wheels = [Path(row["path"]) for row in manifest["wheels"]]
        payload, report["wheels"] = wheel_inventory(wheels)
        report["package_source_hashes"] = source_match(source, args.source_commit, wheels, payload)
        report["native_assets"] = {name:row for name,row in payload.items()
                                    if name.startswith("mojolearn/") and name.endswith((".so", ".dylib"))}
        report["supplied_native_proofs"] = [dict(path=str(p.resolve()), sha256=digest(p)) for p in args.native_proof]
        report["native_proof_limit"] = "Proof files are retained, not interpreted as successful build or source equivalence. Review them alongside loaded native hashes."
        proofdir = out / "native-proofs"; proofdir.mkdir()
        for i, path in enumerate(args.native_proof):
            shutil.copyfile(path, proofdir / (str(i) + "-" + path.name))
        tests = out / "frozen-tests"
        shutil.copytree(source / "python/mojolearn/tests", tests,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        tree = subprocess.check_output(["git", "-C", str(source), "ls-tree", "-r", "-z", args.source_commit])
        committed = {}
        for row in tree.split(b"\0"):
            if row:
                metadata, name = row.split(b"\t", 1)
                committed[name.decode()] = metadata.split()[2].decode()
        report["requested_native_source_hashes"] = {}
        for rel, expected_blob in committed.items():
            if rel.endswith((".mojo", ".mojopkg")) or rel in ("pixi.toml", "pixi.lock"):
                raw = (source / rel).read_bytes()
                require(hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest() == expected_blob,
                        "uncommitted requested native source: " + rel)
                report["requested_native_source_hashes"][rel] = hashlib.sha256(raw).hexdigest()
        report["test_hashes"] = {}
        for path in tests.rglob("*"):
            if path.is_file():
                rel = path.relative_to(tests).as_posix(); raw = path.read_bytes()
                blob = hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest()
                require(committed.get("python/mojolearn/tests/" + rel) == blob,
                        "uncommitted frozen test: " + rel)
                report["test_hashes"][rel] = hashlib.sha256(raw).hexdigest()
        selected = args.test_file or list(PUBLIC_TESTS)
        report["selected_public_scope"] = {name:PUBLIC_TESTS[name] for name in selected}
        venv = out / "venv"
        run_command("venv", [args.python, "-m", "venv", str(venv)], out, cwd=out)
        python = venv / "bin/python"
        env = {k:v for k,v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")
               and not k.startswith("MOJOLEARN_")}
        env.update(PYTHONNOUSERSITE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
                   MOJOLEARN_NUMERIC_MODE=args.numeric_mode, MOJOLEARN_VENDOR=args.vendor)
        run_command("install-wheels", [str(python), "-m", "pip", "install", "--no-deps", "--no-cache-dir"]
                    + [str(Path(p).resolve()) for p in wheels], out, cwd=out, env=env)
        run_command("install-test-dependencies", [str(python), "-m", "pip", "install", "numpy", "pytest"]
                    + args.dependency, out, cwd=out, env=env)
        run_command("pip-freeze", [str(python), "-m", "pip", "freeze", "--all"], out, cwd=out, env=env)
        shutil.copyfile(__file__, out / "controller.py")
        request = dict(out=str(out), tests=str(tests), payload=payload, test_files=selected,
                       vendor=args.vendor, numeric_mode=args.numeric_mode)
        (out / "request.json").write_text(json.dumps(request, indent=2) + "\n")
        (out / "work").mkdir()
        with (out / "tests.log").open("wb") as log:
            result = subprocess.run([str(python), str(out / "controller.py"), "--worker", str(out / "request.json")],
                                    cwd=out / "work", env=env, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=args.timeout)
        report["pytest_exit"] = result.returncode
        if (out / "tests.xml").exists():
            report["tests"] = junit_summary(out / "tests.xml")
        report["installed_guard_receipt"] = str(out / "installed-modules.json")
        tests_summary = report.get("tests", {})
        passed = (result.returncode == 0 and tests_summary.get("total", 0) > 0
                  and not tests_summary.get("failures") and not tests_summary.get("skipped")
                  and (out / "installed-modules.json").is_file())
        report["status"] = "selected-tests-passed" if passed else "failed-or-incomplete"
    except Exception as error:
        report["status"] = "failed-or-incomplete"
        report["error"] = type(error).__name__ + ": " + str(error)
    finally:
        (out / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k:report.get(k) for k in ("status", "pytest_exit", "release_qualified")}))
    return 0 if report["status"] == "selected-tests-passed" else 1


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        raise SystemExit(worker(sys.argv[2]))
    raise SystemExit(main())
