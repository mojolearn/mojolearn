#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""published-<vendor> producer (post stage): the PUBLISHED bytes are the gated candidate bytes.

Andrew, 2026-11-02: the gates verified the exact wheel bytes on every vendor; nothing
reinstalls them from PyPI to verify again. So the default producer runs no device job,
installs nothing and needs no box. For each published package of the vendor:

  index-resolution           `pip download --no-deps --only-binary=:all:` with the candidate
                             wheel's own platform tags (--platform/--abi/--python-version)
                             resolves the file a user on that platform gets from the index; its
                             sha256 equals the digest the index's simple page advertises
  published-artifact-hashes  the resolved file's sha256 equals the gated candidate wheel's;
                             when they differ the report FAILS and names both hashes (never a
                             silent rerun)
  wheel-integrity            the archive's CRCs, every member's RECORD hash and size, no
                             unlisted or missing member, METADATA Name/Version and the WHEEL
                             tags equal to the file name's
  gate-evidence              the gate's own runtime/correctness/adapter receipts, copied into
                             gate-evidence/ under their sha256 (content keyed) and judged by
                             reference: every published sha256 is named by them, and the
                             release's gate verdict is recorded

    python3 tools/release_published_install.py check --vendor nvidia --version 0.8.66 \\
        --candidate-wheel CORE.whl --candidate-wheel PLUGIN.whl [--candidate-wheel PTX.whl] \\
        --gate-evidence GATE/gate-smoke.json [--gate-evidence ...] --gate-verdict "PASS: ..." \\
        --out NEW_DIR [--python PY_WITH_PIP] [--index-url URL]
    python3 tools/release_published_install.py judge --out DIR

``judge`` re-derives every count from the retained files; the ledger adapter
(tools/release_installed_evidence.py, adapter ``published-install``) runs it against
https://pypi.org/simple and also compares the published hashes with the ledger's
artifacts:<vendor>. A report made against any other index re-derives differently and is
refused there.

``run`` is the old installed smoke (a fresh venv from PyPI, the eleven installed correctness
jobs and the adapter phases on the vendor device). tools/release.py calls it only for an
explicit ``--published-smoke <vendor>`` with a stated reason (recorded in the release state);
its report (schema v1) is never published-<vendor> ledger evidence.
"""
import argparse
import glob
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import urllib.parse
import urllib.request
import zipfile
from html.parser import HTMLParser

HERE = Path(__file__).resolve().parent
REPORT = "published-report.json"
#: the published-<vendor> report: resolution, bytes, integrity and the gate's evidence by reference
REPORT_SCHEMA = "mojolearn.release-published-install.v2"
#: the explicit --published-smoke report (``run``); never published-<vendor> ledger evidence
SMOKE_SCHEMA = "mojolearn.release-published-install.v1"
VENDORS = ("nvidia", "amd", "apple")
BACKEND = {"nvidia": "cuda", "amd": "hip", "apple": "metal"}
COVERAGE_IDS = ("index-resolution", "published-artifact-hashes", "wheel-integrity", "gate-evidence")
SMOKE_COVERAGE_IDS = ("index-install", "published-artifact-hashes", "installed-smoke", "installed-adapters")
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


def judge(out, index_url=PYPI_INDEX):
    """The report of a retained producer directory: the bytes check (``check``), else the smoke (``run``)."""
    request = _load(Path(out).resolve() / "request.json")
    if request.get("mode") == "bytes":
        return judge_bytes(out, index_url)
    return judge_smoke(out)


def judge_smoke(out):
    import adapter_gate
    import release_gate_profiles as profiles
    from release_adapter_evidence import _smoke
    out = Path(out).resolve()
    request = _load(out / "request.json")
    vendor, version = request["vendor"], request["version"]
    require(vendor in VENDORS, "vendor is nvidia, amd or apple")
    rows = {name: [] for name in SMOKE_COVERAGE_IDS}

    def add(coverage, name, passed, **detail):
        rows[coverage].append(dict(name=name, passed=bool(passed), **detail))

    manifest = _load(out / "wheel-manifest.json")
    files = manifest.get("files", [])
    add("index-install", "install-exit", request.get("install_exit") == 0)
    add("index-install", "index", manifest.get("index_url") == PYPI_INDEX and manifest.get("version") == version
        and manifest.get("vendor") == vendor, index=manifest.get("index_url"))
    for row in files:
        add("index-install", "pypi-digest:" + row["file"], row.get("pypi_digest_match") is True
            and row.get("pypi_sha256") == row.get("sha256") and sha256(out / "wheels" / row["file"]) == row["sha256"])
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
    for name in SMOKE_COVERAGE_IDS:
        items = rows[name]
        passed = sum(1 for item in items if item["passed"])
        ok = bool(items) and passed == len(items)
        coverage.append(dict(id=name, status="passed" if ok else "failed", exit_code=0 if ok else 1,
                             expected=len(items), completed=passed, failed=len(items) - passed, skipped=0))
    passed = all(row["status"] == "passed" for row in coverage)
    paths = [out / "request.json", out / "wheel-manifest.json", out / "installed-bytes.json"]
    paths += [out / "correctness/results.json"] if correctness is not None else []
    paths += [out / "wheels" / row["file"] for row in files]
    paths += [Path(ref["path"]) for ref in adapter_refs]
    # Paths relative to the producer directory: the report is judged again after it moves home.
    evidence = []
    for path in paths:
        ref = dict(path=Path(path).resolve().relative_to(out).as_posix(), sha256=sha256(path))
        if ref not in evidence:
            evidence.append(ref)
    return dict(schema=SMOKE_SCHEMA, tool="tools/release_published_install.py", vendor=vendor, version=version,
                backend=BACKEND[vendor], index_url=manifest.get("index_url"), source_revision=commit,
                published=published, coverage=coverage, checks=rows,
                status="passed" if passed else "failed", exit_code=0 if passed else 1, evidence=evidence)


# ---------------------------------------------------------------- the bytes check (default; no device)
#: the packages each vendor's published-<vendor> covers (tools/pypi_wheel_install.py distributions)
WHEEL_NAME = re.compile(r"^(?P<name>[^-]+)-(?P<version>[^-]+)(?:-(?P<build>\d[^-]*))?-(?P<py>[^-]+)-(?P<abi>[^-]+)"
                        r"-(?P<plat>[^-]+)\.whl$")
#: dist-info members a wheel may carry outside its own RECORD (PEP 376/427 signatures)
RECORD_EXEMPT = ("RECORD", "RECORD.jws", "RECORD.p7s")


def normalize(name):
    """PEP 503 project name normalization."""
    return re.sub(r"[-_.]+", "-", name).lower()


def wheel_tags(filename):
    """{distribution, version, python, abi, platform} of a wheel file name (PEP 427)."""
    m = WHEEL_NAME.match(Path(filename).name)
    require(m is not None, "not a wheel file name: %s" % filename)
    return dict(distribution=normalize(m["name"]), version=m["version"], python=m["py"].split("."),
                abi=m["abi"].split("."), platform=m["plat"].split("."))


def tag_set(tags):
    return sorted("%s-%s-%s" % (p, a, f) for p in tags["python"] for a in tags["abi"] for f in tags["platform"])


def pip_download_command(python, index_url, distribution, version, tags, dest):
    """`pip download` of DISTRIBUTION==VERSION as an installer on the candidate's platform resolves it.

    The candidate's own tags select the target: every platform tag (`any` is no constraint), every
    ABI other than `none` (pip always accepts `none`), and a CPython tag's version and implementation.
    `--no-deps --only-binary=:all:` downloads exactly that one wheel; nothing is installed."""
    cmd = [python, "-m", "pip", "download", "--no-input", "--disable-pip-version-check", "--no-cache-dir",
           "--no-deps", "--only-binary=:all:", "--index-url", index_url, "--dest", str(dest)]
    for plat in tags["platform"]:
        if plat != "any":
            cmd += ["--platform", plat]
    for abi in tags["abi"]:
        if abi != "none":
            cmd += ["--abi", abi]
    cpython = [t for t in tags["python"] if re.fullmatch(r"cp3\d+", t)]
    if cpython:
        cmd += ["--python-version", "3." + cpython[0][3:], "--implementation", "cp"]
    return cmd + ["%s==%s" % (distribution, version)]


def pip_python(python, out, log):
    """An interpreter that has pip: PYTHON itself, else a venv of it under OUT (ensurepip, offline)."""
    probe = subprocess.run([python, "-m", "pip", "--version"], capture_output=True, text=True)
    if probe.returncode == 0:
        return python
    venv = Path(out) / "pip-venv"
    with Path(log).open("ab") as stream:
        code = subprocess.run([python, "-m", "venv", str(venv)], stdout=stream, stderr=subprocess.STDOUT).returncode
    require(code == 0, "no pip in %s and no venv could be made for it (%s)" % (python, log))
    return str(venv / "bin" / "python")


class _Anchors(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.hrefs.extend(v for k, v in attrs if k == "href" and v)


def simple_page(index_url, distribution, timeout=60):
    """The PEP 503 simple page of DISTRIBUTION on INDEX_URL (a file:// index reads its index.html)."""
    url = index_url.rstrip("/") + "/" + normalize(distribution) + "/"
    if url.startswith("file://"):
        path = Path(urllib.request.url2pathname(urllib.parse.urlparse(url).path))
        return (path / "index.html").read_text()
    request = urllib.request.Request(url, headers={"Accept": "text/html"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode()


def index_digest(index_url, distribution, filename, page=simple_page):
    """The sha256 the index's simple page advertises for FILENAME (its `#sha256=` fragment), or None."""
    parser = _Anchors()
    parser.feed(page(index_url, distribution))
    for href in parser.hrefs:
        url, _, fragment = href.partition("#")
        if urllib.parse.unquote(url.rsplit("/", 1)[-1]) == filename:
            algo, _, digest = fragment.partition("=")
            return digest.lower() if algo == "sha256" and re.fullmatch(r"[0-9a-fA-F]{64}", digest) else None
    return None


def wheel_integrity(path):
    """{file, members, problems}: CRCs, RECORD hashes and sizes, METADATA and WHEEL against the file name."""
    import base64
    import csv
    import email.parser
    import hashlib
    path = Path(path)
    problems, members = [], 0
    try:
        tags = wheel_tags(path.name)
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            if bad:
                problems.append("bad CRC: " + bad)
            names = [n for n in archive.namelist() if not n.endswith("/")]
            records = [n for n in names if re.fullmatch(r"[^/]+\.dist-info/RECORD", n)]
            if len(records) != 1:
                return dict(file=path.name, members=0, problems=["expected one .dist-info/RECORD, found %d" % len(records)])
            info = records[0].split("/")[0]
            listed = {}
            for row in csv.reader(archive.read(records[0]).decode().splitlines()):
                if row:
                    listed[row[0]] = row[1:]
            for name in names:
                if name.startswith(info + "/") and name[len(info) + 1:] in RECORD_EXEMPT:
                    continue
                if name not in listed:
                    problems.append("not in RECORD: " + name)
                    continue
                digest, size = (listed[name] + ["", ""])[:2]
                algo, _, b64 = digest.partition("=")
                data = archive.read(name)
                actual = base64.urlsafe_b64encode(hashlib.new(algo or "sha256", data).digest()).rstrip(b"=").decode() \
                    if algo in ("sha256", "sha384", "sha512") else None
                if actual is None or actual != b64:
                    problems.append("RECORD hash differs: " + name)
                elif size and int(size) != len(data):
                    problems.append("RECORD size differs: " + name)
                else:
                    members += 1
            for name in sorted(set(listed) - set(names)):
                if not (name.startswith(info + "/") and name[len(info) + 1:] in RECORD_EXEMPT):
                    problems.append("in RECORD, not in the archive: " + name)
            meta = email.parser.Parser().parsestr(archive.read(info + "/METADATA").decode())
            if normalize(meta.get("Name", "")) != tags["distribution"] or meta.get("Version") != tags["version"]:
                problems.append("METADATA names %s %s" % (meta.get("Name"), meta.get("Version")))
            wheel = email.parser.Parser().parsestr(archive.read(info + "/WHEEL").decode())
            if sorted(wheel.get_all("Tag") or []) != tag_set(tags):
                problems.append("WHEEL tags %s differ from the file name's" % sorted(wheel.get_all("Tag") or []))
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as error:
        problems.append(type(error).__name__ + ": " + str(error))
    return dict(file=path.name, members=members, problems=problems)


def judge_bytes(out, index_url=PYPI_INDEX):
    """The published-<vendor> report re-derived from the retained directory (request, wheels, gate copies)."""
    import pypi_wheel_install
    out = Path(out).resolve()
    request = _load(out / "request.json")
    vendor, version = request["vendor"], request["version"]
    require(vendor in VENDORS, "vendor is nvidia, amd or apple")
    rows = {name: [] for name in COVERAGE_IDS}

    def add(coverage, name, passed, **detail):
        rows[coverage].append(dict(name=name, passed=bool(passed), **detail))

    resolved = request.get("resolved") or []
    candidates = request.get("candidates") or {}
    wanted = pypi_wheel_install.distributions(vendor)
    add("index-resolution", "index", request.get("index_url") == index_url, index=request.get("index_url"))
    add("index-resolution", "distributions", sorted(r.get("distribution") for r in resolved) == sorted(wanted)
        and len(resolved) == len(wanted), resolved=sorted(r.get("distribution") for r in resolved), wanted=wanted)
    published, mismatch = {}, []
    for row in resolved:
        dist, name = row.get("distribution"), row.get("file")
        local = out / "wheels" / str(dist) / str(name)
        digest = sha256(local) if name and local.is_file() else None
        one = row.get("pip_exit") == 0 and name is not None and len(row.get("files") or []) == 1 and digest is not None
        add("index-resolution", "resolved:%s" % dist, one and wheel_tags(name)["distribution"] == dist
            and wheel_tags(name)["version"] == version, file=name, files=row.get("files"), pip_exit=row.get("pip_exit"))
        if not one:
            continue
        published[name] = digest
        add("index-resolution", "index-digest:" + name, row.get("index_sha256") == digest == row.get("sha256"),
            index_sha256=row.get("index_sha256"), sha256=digest)
        candidate = row.get("candidate")
        same = candidate == name and candidates.get(candidate) == digest
        add("published-artifact-hashes", "candidate:" + name, same, published_sha256=digest,
            candidate=candidate, candidate_sha256=candidates.get(candidate))
        if not same:
            mismatch.append(dict(file=name, published_sha256=digest, candidate=candidate,
                                 candidate_sha256=candidates.get(candidate)))
        check = wheel_integrity(local)
        add("wheel-integrity", "record:" + name, not check["problems"] and check["members"] > 0,
            members=check["members"], problems=check["problems"][:20])
    if not published:
        add("published-artifact-hashes", "resolved-files", False, why="nothing was resolved")
        add("wheel-integrity", "resolved-files", False, why="nothing was resolved")
    gate = request.get("gate") or {}
    add("gate-evidence", "gate-verdict", isinstance(gate.get("verdict"), str) and gate["verdict"].strip() != ""
        and not gate.get("refused"), verdict=gate.get("verdict"), refused=gate.get("refused"))
    texts = []
    for ref in gate.get("evidence") or []:
        relative = Path(str(ref.get("path", "")))
        local = out / relative
        ok = (not relative.is_absolute() and ".." not in relative.parts and relative.parts[:1] == ("gate-evidence",)
              and local.is_file() and sha256(local) == ref.get("sha256"))
        add("gate-evidence", "evidence:" + relative.as_posix(), ok, source=ref.get("source"), sha256=ref.get("sha256"))
        if ok:
            texts.append(local.read_bytes())
    if not texts:
        add("gate-evidence", "evidence", False, why="no gate evidence retained")
    for name, digest in sorted(published.items()):
        add("gate-evidence", "gated-bytes:" + name, any(digest.encode() in text for text in texts), sha256=digest)
    coverage = []
    for name in COVERAGE_IDS:
        items = rows[name]
        passed = sum(1 for item in items if item["passed"])
        ok = bool(items) and passed == len(items)
        coverage.append(dict(id=name, status="passed" if ok else "failed", exit_code=0 if ok else 1,
                             expected=len(items), completed=passed, failed=len(items) - passed, skipped=0))
    passed = all(row["status"] == "passed" for row in coverage)
    paths = [out / "request.json"] + [out / "wheels" / r["distribution"] / r["file"] for r in resolved
                                      if r.get("file") and (out / "wheels" / r["distribution"] / r["file"]).is_file()]
    paths += [out / ref["path"] for ref in gate.get("evidence") or [] if (out / str(ref.get("path", ""))).is_file()]
    evidence = []
    for path in paths:
        ref = dict(path=Path(path).resolve().relative_to(out).as_posix(), sha256=sha256(path))
        if ref not in evidence:
            evidence.append(ref)
    return dict(schema=REPORT_SCHEMA, tool="tools/release_published_install.py", vendor=vendor, version=version,
                backend=BACKEND[vendor], index_url=request.get("index_url"), source_revision=request.get("source_commit"),
                published=published, mismatch=mismatch, gate_verdict=gate.get("verdict"), coverage=coverage,
                checks=rows, status="passed" if passed else "failed", exit_code=0 if passed else 1, evidence=evidence)


def check(args):
    """The published-<vendor> bytes check: resolve, download, compare, reference the gate. No device, no install."""
    import shutil
    out = args.out.resolve()
    require(not out.exists(), "output must be new")
    out.mkdir(parents=True)
    log = out / "resolve.log"
    candidates = {Path(w).name: sha256(w) for w in args.candidate_wheel}
    request = dict(mode="bytes", vendor=args.vendor, version=args.version, index_url=args.index_url,
                   candidates=candidates, controller_sha256=sha256(__file__),
                   source_commit=args.source_commit, gate=dict(verdict=args.gate_verdict, evidence=[]))
    resolved = []
    try:
        python = pip_python(args.python, out, log)
        for wheel in args.candidate_wheel:
            tags = wheel_tags(wheel)
            require(tags["version"] == args.version, "candidate %s is not version %s" % (Path(wheel).name, args.version))
            dest = out / "wheels" / tags["distribution"]
            dest.mkdir(parents=True, exist_ok=True)
            cmd = pip_download_command(python, args.index_url, tags["distribution"], args.version, tags, dest)
            with log.open("ab") as stream:
                stream.write(("\n$ " + " ".join(cmd) + "\n").encode())
                stream.flush()
                code = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT,
                                      timeout=args.timeout).returncode
            files = sorted(p.name for p in dest.glob("*.whl"))
            row = dict(distribution=tags["distribution"], requested="%s==%s" % (tags["distribution"], args.version),
                       tags=tag_set(tags), candidate=Path(wheel).name, pip_exit=code, files=files,
                       file=files[0] if code == 0 and len(files) == 1 else None)
            if row["file"]:
                row["sha256"] = sha256(dest / row["file"])
                try:
                    row["index_sha256"] = index_digest(args.index_url, tags["distribution"], row["file"])
                except (OSError, ValueError) as error:
                    row["index_error"] = type(error).__name__ + ": " + str(error)
            resolved.append(row)
        for source in args.gate_evidence:
            source = Path(source).resolve()
            digest = sha256(source)
            target = out / "gate-evidence" / ("%s-%s" % (digest[:16], source.name))
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            request["gate"]["evidence"].append(dict(source=str(source), path=target.relative_to(out).as_posix(),
                                                    sha256=digest))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        request["error"] = type(error).__name__ + ": " + str(error)
    request["resolved"] = resolved
    (out / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    report = judge_bytes(out, args.index_url)
    if "error" in request:
        report.update(status="failed", exit_code=1, error=request["error"])
    (out / REPORT).write_text(json.dumps(report, indent=2) + "\n")
    print("PUBLISHED vendor=%s version=%s status=%s %s" % (args.vendor, args.version, report["status"], " ".join(
        "%s=%d/%d" % (row["id"], row["completed"], row["expected"]) for row in report["coverage"])))
    for row in report["mismatch"]:
        print("PUBLISHED-DIFFERS %s published=%s candidate=%s" % (row["file"], row["published_sha256"],
                                                                   row["candidate_sha256"]))
    return report["exit_code"]


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
        report = dict(schema=SMOKE_SCHEMA, tool="tools/release_published_install.py", vendor=args.vendor,
                      status="failed", exit_code=1, error=type(error).__name__ + ": " + str(error))
    (out / REPORT).write_text(json.dumps(report, indent=2) + "\n")
    print("PUBLISHED-SMOKE vendor=%s version=%s status=%s %s" % (args.vendor, args.version, report["status"], " ".join(
        "%s=%d/%d" % (row["id"], row["completed"], row["expected"]) for row in report.get("coverage", []))))
    return report["exit_code"]


def main(argv=None):
    sys.path.insert(0, str(HERE))
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    bytes_check = sub.add_parser("check", help="the published-<vendor> producer: resolution, bytes, integrity, "
                                               "the gate's evidence by reference (no device, no install)")
    bytes_check.add_argument("--vendor", choices=VENDORS, required=True)
    bytes_check.add_argument("--version", required=True)
    bytes_check.add_argument("--candidate-wheel", action="append", required=True)
    bytes_check.add_argument("--gate-evidence", action="append", default=[],
                             help="a receipt of the gate that passed these candidate bytes (copied, content keyed)")
    bytes_check.add_argument("--gate-verdict", required=True, help="the release's verdict on that gate")
    bytes_check.add_argument("--source-commit", default=None)
    bytes_check.add_argument("--index-url", default=PYPI_INDEX)
    bytes_check.add_argument("--python", default=sys.executable, help="runs pip (a venv of it when it has none)")
    bytes_check.add_argument("--timeout", type=int, default=900, help="seconds per pip download")
    bytes_check.add_argument("--out", type=Path, required=True)
    go = sub.add_parser("run", help="the installed smoke on the vendor device: only for release.py "
                                    "--published-smoke <vendor> with a stated reason")
    go.add_argument("--vendor", choices=VENDORS, required=True)
    go.add_argument("--version", required=True)
    go.add_argument("--candidate-wheel", action="append", required=True)
    go.add_argument("--python", default=sys.executable)
    go.add_argument("--timeout", type=int, default=1800)
    go.add_argument("--out", type=Path, required=True)
    again = sub.add_parser("judge")
    again.add_argument("--out", type=Path, required=True)
    again.add_argument("--index-url", default=PYPI_INDEX)
    args = parser.parse_args(argv)
    if args.command == "check":
        return check(args)
    if args.command == "run":
        return run(args)
    report = judge(args.out, args.index_url)
    print(json.dumps({k: report[k] for k in ("status", "coverage")}))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
