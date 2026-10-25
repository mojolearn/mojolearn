#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""published-<vendor> producer (post stage): the PUBLISHED release installed from the index.

Test controller only; it never builds, uploads or edits anything. In a fresh venv:

  index-install              tools/pypi_wheel_install.py install: pip download from
                             https://pypi.org/simple, every file's sha256 equal to PyPI's
                             JSON digest, then installed (and verified loaded from site-packages)
  published-artifact-hashes  every downloaded file's sha256 equal to the qualified candidate
                             wheel of the same name, and every installed payload byte (per the
                             candidate wheel's verified RECORD) equal to the candidate's
  installed-smoke            tools/qualify_verifier_wheel.py --installed-python --scope expanded:
                             the light gate's eleven installed correctness jobs
  installed-adapters         tools/adapter_gate.py over the downloaded files: torch-compile,
                             torch-training and max-training on the vendor device

    python3 tools/release_published_install.py run --vendor nvidia --version 0.8.57 \\
        --candidate-wheel CORE.whl --candidate-wheel PLUGIN.whl [--candidate-wheel PTX.whl] \\
        --out NEW_DIR [--python python3.12]
    python3 tools/release_published_install.py judge --out DIR

``judge`` re-derives every count from the retained files; the ledger adapter
(tools/release_installed_evidence.py, adapter ``published-install``) runs it and
also compares the published hashes with the ledger's artifacts:<vendor>.
"""
import argparse
import glob
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
REPORT = "published-report.json"
REPORT_SCHEMA = "mojolearn.release-published-install.v1"
VENDORS = ("nvidia", "amd", "apple")
BACKEND = {"nvidia": "cuda", "amd": "hip", "apple": "metal"}
COVERAGE_IDS = ("index-install", "published-artifact-hashes", "installed-smoke", "installed-adapters")
PYPI_INDEX = "https://pypi.org/simple"
#: files pip writes or relocates itself; never compared with the wheel's bytes
PIP_OWNED = ("RECORD", "INSTALLER", "REQUESTED", "direct_url.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    import hashlib
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _load(path):
    return json.loads(Path(path).read_text())


def installed_bytes(candidates, site):
    """Every candidate payload member installed in SITE with the candidate's bytes."""
    import qualify_native_runtime_installed as q
    payload, _ = q.wheel_inventory(candidates)
    rows, differ, missing = 0, [], []
    for member, row in sorted(payload.items()):
        parts = PurePosixPath(member).parts
        if parts[0].endswith(".data") or (parts[0].endswith(".dist-info") and parts[-1] in PIP_OWNED):
            continue
        path = Path(site) / member
        if not path.is_file():
            missing.append(member)
        elif sha256(path) != row["sha256"]:
            differ.append(member)
        else:
            rows += 1
    return dict(site=str(site), compared=rows, differ=differ, missing=missing)


def judge(out):
    import adapter_gate
    import release_gate_profiles as profiles
    from release_adapter_evidence import _smoke
    out = Path(out).resolve()
    request = _load(out / "request.json")
    vendor, version = request["vendor"], request["version"]
    require(vendor in VENDORS, "vendor is nvidia, amd or apple")
    rows = {name: [] for name in COVERAGE_IDS}

    def add(coverage, name, passed, **detail):
        rows[coverage].append(dict(name=name, passed=bool(passed), **detail))

    manifest = _load(out / "wheel-manifest.json")
    files = manifest.get("files", [])
    add("index-install", "install-exit", request.get("install_exit") == 0)
    add("index-install", "index", manifest.get("index_url") == PYPI_INDEX and manifest.get("version") == version
        and manifest.get("vendor") == vendor, index=manifest.get("index_url"))
    for row in files:
        add("index-install", "pypi-digest:" + row["file"], row.get("pypi_digest_match") is True
            and row.get("pypi_sha256") == row.get("sha256") and sha256(row["path"]) == row["sha256"])
    import pypi_wheel_install
    why = pypi_wheel_install.installed_refusal(manifest, manifest.get("loaded")) if files else "no files"
    add("index-install", "loaded-from-site-packages", why is None, why=why)
    published = {row["file"]: row["sha256"] for row in files}
    candidates = request.get("candidates", {})
    for name, digest in sorted(published.items()):
        add("published-artifact-hashes", "candidate:" + name, candidates.get(name) == digest)
    names = [row["distribution"] for row in files]
    unpublished = sorted(name for name in candidates if name not in published
                         and any(name.lower().startswith(re.sub("-", "_", d).lower() + "-" + version + "-") for d in names))
    add("published-artifact-hashes", "no-unpublished-candidate", not unpublished, unpublished=unpublished)
    installed = _load(out / "installed-bytes.json")
    add("published-artifact-hashes", "installed-payload", installed.get("compared", 0) > 0
        and not installed.get("differ") and not installed.get("missing"), compared=installed.get("compared"))
    correctness = _load(out / "correctness/results.json") if (out / "correctness/results.json").is_file() else None
    commit = request.get("source_commit")
    errors = profiles.correctness_errors(correctness, commit, version, vendor if vendor != "apple" else None)
    if vendor == "apple" and isinstance(correctness, dict) and correctness.get("installed", {}).get("vendor") != "metal":
        errors.append("installed correctness ran another vendor backend")
    jobs = (correctness or {}).get("jobs") or []
    for job in jobs:
        add("installed-smoke", "job:" + str(job.get("name")), not errors and job.get("exit_code") in ((0, 5) if job.get("name") == "extended" else (0,)))
    if len(jobs) != len(profiles.CORRECTNESS_JOBS) or errors:
        add("installed-smoke", "complete", False, errors=errors)
    try:
        smoke = _smoke(out / "adapter-smoke", vendor, version)
        for name, count in sorted(smoke["counts"].items()):
            add("installed-adapters", name, count > 0, checks=count)
        status, ok, why = adapter_gate.judge(out / "adapter-smoke" / adapter_gate.RECEIPT)
        add("installed-adapters", "judge", ok, why=why)
        adapter_refs = smoke["evidence"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        add("installed-adapters", "smoke", False, error=str(error))
        adapter_refs = []
    coverage = []
    for name in COVERAGE_IDS:
        items = rows[name]
        passed = sum(1 for item in items if item["passed"])
        ok = bool(items) and passed == len(items)
        coverage.append(dict(id=name, status="passed" if ok else "failed", exit_code=0 if ok else 1,
                             expected=len(items), completed=passed, failed=len(items) - passed, skipped=0))
    passed = all(row["status"] == "passed" for row in coverage)
    paths = [out / "request.json", out / "wheel-manifest.json", out / "installed-bytes.json"]
    paths += [out / "correctness/results.json"] if correctness is not None else []
    paths += [Path(row["path"]) for row in files]
    evidence = [dict(path=str(p.resolve()), sha256=sha256(p)) for p in paths]
    evidence += [ref for ref in adapter_refs if ref not in evidence]
    return dict(schema=REPORT_SCHEMA, tool="tools/release_published_install.py", vendor=vendor, version=version,
                backend=BACKEND[vendor], index_url=manifest.get("index_url"), source_revision=commit,
                published=published, coverage=coverage, checks=rows,
                status="passed" if passed else "failed", exit_code=0 if passed else 1, evidence=evidence)


def published_commit(files):
    found = set()
    for row in files:
        with zipfile.ZipFile(row["path"]) as archive:
            if "mojolearn/identity_columns/COMMIT" in archive.namelist():
                found.add(archive.read("mojolearn/identity_columns/COMMIT").decode().strip())
    require(len(found) == 1 and re.fullmatch("[0-9a-f]{40}", next(iter(found))),
            "the published wheels name no single packaged source commit")
    return found.pop()


def run(args):
    import adapter_gate
    out = args.out.resolve()
    require(not out.exists(), "output must be new")
    require(os.environ.get("MOJOLEARN_WHEEL_INDEX", "pypi") == "pypi", "published-<vendor> installs from PyPI only")
    out.mkdir(parents=True)
    candidates = {Path(w).name: sha256(w) for w in args.candidate_wheel}
    request = dict(vendor=args.vendor, version=args.version, candidates=candidates, controller_sha256=sha256(__file__))
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    venv = out / "venv"
    python = str(venv / "bin/python")
    log = out / "install.log"
    with log.open("ab") as stream:
        code = subprocess.run([args.python, "-m", "venv", str(venv)], stdout=stream, stderr=subprocess.STDOUT).returncode
    if code == 0:
        with log.open("ab") as stream:
            code = subprocess.run([sys.executable, str(HERE / "pypi_wheel_install.py"), "install", "--python", python,
                                   "--version", args.version, "--vendor", args.vendor, "--dir", str(out / "wheels"),
                                   "--manifest", str(out / "wheel-manifest.json"), "--log", str(log)],
                                  stdout=stream, stderr=subprocess.STDOUT, env=env).returncode
    request["install_exit"] = code
    try:
        manifest = _load(out / "wheel-manifest.json")
        request["source_commit"] = published_commit(manifest["files"])
        site = subprocess.check_output([python, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                                       text=True, env=env).strip()
        (out / "installed-bytes.json").write_text(json.dumps(
            installed_bytes([Path(w).resolve() for w in args.candidate_wheel], site), indent=2) + "\n")
        if args.vendor == "amd":
            env["LD_LIBRARY_PATH"] = ":".join(glob.glob(str(venv / "lib/python*/site-packages/*/lib64"))
                                              + ["/opt/rocm-7.2.3/lib", "/opt/rocm-6.4.0/lib"])
        with (out / "correctness.log").open("ab") as stream:
            request["correctness_exit"] = subprocess.run(
                [sys.executable, str(HERE / "qualify_verifier_wheel.py"), "--installed-python", python,
                 "--expected-version", args.version, "--expected-source-commit", request["source_commit"],
                 "--scope", "expanded", "--output", str(out / "correctness")],
                stdout=stream, stderr=subprocess.STDOUT, env=env, timeout=args.timeout).returncode
        if args.vendor == "amd":
            env["LD_LIBRARY_PATH"] = ":".join(glob.glob(str(out / "adapter-smoke/venv/lib/python*/site-packages/*/lib64"))
                                              + ["/opt/rocm-7.2.3/lib", "/opt/rocm-6.4.0/lib"])
        record = adapter_gate.run(args.vendor, sys.executable, out / "adapter-smoke",
                                  [row["path"] for row in manifest["files"]], env=env)
        request["adapters"] = {k: record.get(k) for k in ("status", "ok", "rc", "why")}
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        request["error"] = type(error).__name__ + ": " + str(error)
    (out / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    try:
        report = judge(out)
        if "error" in request:
            report.update(status="failed", exit_code=1, error=request["error"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        report = dict(schema=REPORT_SCHEMA, tool="tools/release_published_install.py", vendor=args.vendor,
                      status="failed", exit_code=1, error=type(error).__name__ + ": " + str(error))
    (out / REPORT).write_text(json.dumps(report, indent=2) + "\n")
    print("PUBLISHED vendor=%s version=%s status=%s %s" % (args.vendor, args.version, report["status"], " ".join(
        "%s=%d/%d" % (row["id"], row["completed"], row["expected"]) for row in report.get("coverage", []))))
    return report["exit_code"]


def main(argv=None):
    sys.path.insert(0, str(HERE))
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    go = sub.add_parser("run")
    go.add_argument("--vendor", choices=VENDORS, required=True)
    go.add_argument("--version", required=True)
    go.add_argument("--candidate-wheel", action="append", required=True)
    go.add_argument("--python", default=sys.executable)
    go.add_argument("--timeout", type=int, default=1800)
    go.add_argument("--out", type=Path, required=True)
    again = sub.add_parser("judge")
    again.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        return run(args)
    report = judge(args.out)
    print(json.dumps({k: report[k] for k in ("status", "coverage")}))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
