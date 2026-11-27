#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""published (post stage): ONE CPU check, on a free GitHub runner, that the published release is the tested one.

Andrew, 2026-11-27: after publication no release step rents or runs a GPU. The published files are hash-identical
to the candidate wheels the gates tested, so a GPU rerun proves nothing new (AGENTS.md "No re-verification of green
work"). .github/workflows/published-check.yml runs ``check`` on an ubuntu runner (no GPU, nothing rented), and
tools/release.py step_published dispatches it after publication and judges its report. In order:

  index-resolution           the index lists every candidate file (the PyPI JSON API and the simple page, polled
                             at most --index-wait seconds, a WAIT line every --index-wait-step seconds); then
                             `pip download --no-deps --only-binary=:all:` with each candidate's own platform tags
                             fetches the file a user on that platform gets (the vendor wheels and the macOS wheel
                             as files); the sha256 the simple page advertises equals the downloaded file's
  published-artifact-hashes  the sha256 of every downloaded file equals the candidate's (both named when not)
  pypi-install               `pip install mojolearn==<version>` from the index into a fresh venv; every mojolearn
                             distribution it installed is at <version>
  import                     `import mojolearn` in that venv reports <version>
  verify-quick               `python -m mojolearn verify --quick` in that venv exits 0

    python3 tools/release_published_install.py check --version 0.9.1 --candidates candidates.json --out NEW_DIR
    python3 tools/release_published_install.py judge --out DIR

candidates.json is {file name: sha256} of every candidate wheel the release published (tools/release.py writes it
into the workflow's inputs). ``judge`` re-derives the report from the retained request.json and the step exit codes
it records; the downloaded wheels are not kept (their sha256 is in request.json).

A FAILED report names its kind (``failure``): ``index-lag`` when the index never listed an expected file within the
wait or nothing was resolved (no verdict on the bytes: the record step refuses to record it and `published` runs
again once the index lists the files), ``hash-mismatch`` when a published file's sha256 differs from the candidate's
(both hashes named), ``error`` when the check itself broke, and ``failed`` for any other check (install, import,
verify).

REMOVED 2026-11-27 (Andrew): the per-vendor published-apple / published-nvidia / published-amd producers (a bytes
check on the Mac with the gate's receipts by reference) and the ``run`` installed smoke on a vendor GPU box; this one
CPU check replaces them. Recoverable at eec67ac1204b (this file, tools/ops/published_install.py and
tools/ops/published_install_job.sh).
"""
import argparse
import atexit
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

HERE = Path(__file__).resolve().parent
REPORT = "published-report.json"
REPORT_SCHEMA = "mojolearn.release-published-check.v3"
TOOL = "tools/release_published_install.py"
COVERAGE_IDS = ("index-resolution", "published-artifact-hashes", "pypi-install", "import", "verify-quick")
PYPI_INDEX = "https://pypi.org/simple"
#: the project `pip install` resolves on Linux (the file-free meta wheel: the core and both vendors)
PROJECT = "mojolearn"
# THE INDEX WAIT (0.8.66 run v, 2026-11-08): a check that ran a minute after an upload found the file not listed
# yet, resolved nothing and recorded a FAILED check. The check waits for the index to list every expected file
# before it resolves; a listing that never appears within the cap is the failure (index lag).
INDEX_WAIT_S = 900
INDEX_WAIT_STEP_S = 20
#: the kinds of a failed report (report["failure"]); only INDEX_LAG is refused by the record step
INDEX_LAG, HASH_MISMATCH, CHECK_ERROR, CHECK_FAILED = "index-lag", "hash-mismatch", "error", "failed"
NOTHING_RESOLVED = "nothing was resolved"
WHEEL_NAME = re.compile(r"^(?P<name>[^-]+)-(?P<version>[^-]+)(?:-(?P<build>\d[^-]*))?-(?P<py>[^-]+)-(?P<abi>[^-]+)"
                        r"-(?P<plat>[^-]+)\.whl$")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    import hashlib
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _load(path):
    return json.loads(Path(path).read_text())


def normalize(name):
    """PEP 503 project name normalization."""
    return re.sub(r"[-_.]+", "-", name).lower()


def wheel_tags(filename):
    """{distribution, version, python, abi, platform} of a wheel file name (PEP 427)."""
    m = WHEEL_NAME.match(Path(filename).name)
    require(m is not None, "not a wheel file name: %s" % filename)
    return dict(distribution=normalize(m["name"]), version=m["version"], python=m["py"].split("."),
                abi=m["abi"].split("."), platform=m["plat"].split("."))


def pip_download_command(python, index_url, distribution, version, tags, dest):
    """`pip download` of DISTRIBUTION==VERSION as an installer on the candidate's platform resolves it.

    The candidate's own tags select the target: every platform tag (`any` is no constraint), every ABI other than
    `none` (pip always accepts `none`), and a CPython tag's version and implementation. `--no-deps
    --only-binary=:all:` downloads exactly that one wheel; nothing is installed."""
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


def scratch_dir(prefix):
    """A temp dir removed when this process exits: downloads and the venv live here, never under OUT (the
    evidence folder holds logs and records only)."""
    path = Path(tempfile.mkdtemp(prefix=prefix))
    atexit.register(shutil.rmtree, str(path), True)
    return path


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


def json_api_url(index_url, distribution, version):
    """PyPI's JSON API of DISTRIBUTION==VERSION beside a PyPI-shaped simple index (``<host>/simple``), else None
    (a file:// or other index is judged by its simple page alone)."""
    parts = urllib.parse.urlparse(index_url)
    path = parts.path.rstrip("/")
    if parts.scheme not in ("http", "https") or not path.endswith("/simple"):
        return None
    api = "%s/pypi/%s/%s/json" % (path[:-len("/simple")], normalize(distribution), version)
    return urllib.parse.urlunparse(parts._replace(path=api, params="", query="", fragment=""))


def json_api_files(url, timeout=60):
    """The file names the JSON API lists for one release; [] while the release is not there yet (404)."""
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            doc = json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return []
        raise
    return sorted(str(row["filename"]) for row in doc.get("urls") or [] if isinstance(row, dict) and row.get("filename"))


def simple_files(index_url, distribution, page=None):
    """The file names the simple page of DISTRIBUTION links (what pip resolves from)."""
    parser = _Anchors()
    parser.feed((page or simple_page)(index_url, distribution))
    return sorted({urllib.parse.unquote(h.partition("#")[0].rsplit("/", 1)[-1]) for h in parser.hrefs})


def index_listing(index_url, version, expected, api=None, page=None):
    """(missing, errors): the EXPECTED files ({distribution: [file name]}) the index does not list yet. A file is
    listed when the simple page pip reads names it and, on a PyPI-shaped index, the JSON API does too. A page or
    API that cannot be read yet (404 before the first upload, a network error) lists nothing this time."""
    missing, errors = [], []
    for dist, files in sorted(expected.items()):
        try:
            listed = set(simple_files(index_url, dist, page))
            url = json_api_url(index_url, dist, version)
            if url:
                listed &= set((api or json_api_files)(url))
        except (OSError, ValueError) as error:       # URLError and HTTPError are OSError
            errors.append("%s: %s: %s" % (dist, type(error).__name__, str(error)[:200]))
            listed = set()
        missing += [f for f in files if f not in listed]
    return sorted(missing), errors


def wait_for_index(index_url, version, expected, cap=INDEX_WAIT_S, step=INDEX_WAIT_STEP_S, listing=None,
                   sleep=None, clock=None, say=None):
    """Poll the index until it lists every EXPECTED file ({distribution: [file name]}), at most CAP seconds, every
    STEP seconds, saying one WAIT line per step. Returns the record kept in request.json (``index_wait``):
    listed (bool), waited_s, polls, missing (still unlisted at the end), errors (the last few), cap_s, step_s."""
    listing, sleep, clock = listing or index_listing, sleep or time.sleep, clock or time.monotonic
    say = say or (lambda line: print(line, flush=True))
    start, polls = clock(), 0
    want = sorted(f for files in expected.values() for f in files)
    while True:
        missing, errors = listing(index_url, version, expected)
        polls += 1
        waited = clock() - start
        done = not missing
        if done or waited >= cap:
            return dict(listed=done, waited_s=round(waited, 1), polls=polls, expected=want, missing=missing,
                        errors=errors[-5:], cap_s=cap, step_s=step)
        say("WAIT published-index version=%s waited=%ds/%ds listed=%d/%d missing=%s%s" % (
            version, waited, cap, len(want) - len(missing), len(want), ",".join(missing),
            (" error=" + errors[-1]) if errors else ""))
        sleep(max(0, min(step, cap - waited)))


def failure_kind(report):
    """The kind of a published report's failure (None when it passed): INDEX_LAG, HASH_MISMATCH, CHECK_ERROR or
    CHECK_FAILED."""
    if not isinstance(report, dict) or report.get("status") != "failed":
        return None
    if report.get("failure"):
        return report["failure"]
    if report.get("mismatch"):
        return HASH_MISMATCH
    return CHECK_ERROR if report.get("error") else CHECK_FAILED


def judge(out, index_url=PYPI_INDEX):
    """The report re-derived from the retained request.json (the run's recorded facts and exit codes)."""
    out = Path(out).resolve()
    request = _load(out / "request.json")
    version = request["version"]
    candidates = request.get("candidates") or {}
    rows = {name: [] for name in COVERAGE_IDS}

    def add(coverage, name, passed, **detail):
        rows[coverage].append(dict(name=name, passed=bool(passed), **detail))

    add("index-resolution", "index", request.get("index_url") == index_url, index=request.get("index_url"))
    wait = request.get("index_wait")
    add("index-resolution", "index-listing", isinstance(wait, dict) and wait.get("listed") is True,
        waited_s=(wait or {}).get("waited_s"), polls=(wait or {}).get("polls"), missing=(wait or {}).get("missing"))
    resolved = {row.get("candidate"): row for row in request.get("resolved") or []}
    published, mismatch = {}, []
    for name, want in sorted(candidates.items()):
        row = resolved.get(name) or {}
        got = row.get("file")
        one = row.get("pip_exit") == 0 and got == name and len(row.get("files") or []) == 1 and bool(row.get("sha256"))
        add("index-resolution", "resolved:" + name, one, files=row.get("files"), pip_exit=row.get("pip_exit"))
        if not one:
            continue
        digest = row["sha256"]
        published[name] = digest
        add("index-resolution", "index-digest:" + name, row.get("index_sha256") == digest,
            index_sha256=row.get("index_sha256"), sha256=digest)
        add("published-artifact-hashes", "candidate:" + name, digest == want, published_sha256=digest,
            candidate_sha256=want)
        if digest != want:
            mismatch.append(dict(file=name, published_sha256=digest, candidate_sha256=want))
    if not candidates:
        add("index-resolution", "candidates", False, why="no candidate wheels named")
    if not published:
        add("published-artifact-hashes", "resolved-files", False, why=NOTHING_RESOLVED)
    install = request.get("install") or {}
    add("pypi-install", "pip-install", install.get("exit") == 0, requested="%s==%s" % (PROJECT, version),
        exit=install.get("exit"))
    dists = install.get("installed") or {}
    add("pypi-install", "installed-versions", bool(dists) and PROJECT in dists
        and all(v == version for v in dists.values()), installed=dists)
    imported = request.get("import") or {}
    add("import", "import-mojolearn", imported.get("exit") == 0 and imported.get("version") == version,
        exit=imported.get("exit"), version=imported.get("version"))
    quick = request.get("verify_quick") or {}
    add("verify-quick", "verify-quick", quick.get("exit") == 0, exit=quick.get("exit"))
    coverage = []
    for name in COVERAGE_IDS:
        items = rows[name]
        passed = sum(1 for item in items if item["passed"])
        ok = bool(items) and passed == len(items)
        coverage.append(dict(id=name, status="passed" if ok else "failed", exit_code=0 if ok else 1,
                             expected=len(items), completed=passed, failed=len(items) - passed, skipped=0))
    passed = all(row["status"] == "passed" for row in coverage) and not request.get("error")
    unlisted = not (isinstance(wait, dict) and wait.get("listed") is True)
    failure, failure_why = None, None
    if not passed:
        if mismatch:
            failure, failure_why = HASH_MISMATCH, "; ".join("%s published %s, candidate %s" % (
                m["file"], m["published_sha256"], m["candidate_sha256"]) for m in mismatch)
        elif request.get("error"):
            failure, failure_why = CHECK_ERROR, str(request["error"])[:300]
        elif unlisted or not published:
            failure = INDEX_LAG
            failure_why = ("the index did not list %s after %ss (%s polls)" % (
                ", ".join((wait or {}).get("missing") or []) or "the expected files", (wait or {}).get("waited_s"),
                (wait or {}).get("polls")) if unlisted else NOTHING_RESOLVED)
        else:
            failure, failure_why = CHECK_FAILED, ", ".join(r["id"] for r in coverage if r["status"] != "passed")
    return dict(schema=REPORT_SCHEMA, tool=TOOL, version=version, index_url=request.get("index_url"),
                runner=request.get("runner"), published=published, mismatch=mismatch, coverage=coverage,
                checks=rows, status="passed" if passed else "failed", exit_code=0 if passed else 1,
                failure=failure, failure_why=failure_why, error=request.get("error"))


def _run(cmd, log, timeout, **kw):
    """Exit code of CMD with its output appended to LOG (a timeout is exit 124)."""
    with Path(log).open("ab") as stream:
        stream.write(("\n$ " + " ".join(str(c) for c in cmd) + "\n").encode())
        stream.flush()
        try:
            return subprocess.run([str(c) for c in cmd], stdout=stream, stderr=subprocess.STDOUT, timeout=timeout,
                                  **kw).returncode
        except subprocess.TimeoutExpired:
            stream.write(("TIMEOUT after %ss\n" % timeout).encode())
            return 124


def resolve(args, candidates, request, log):
    """index-resolution and the hashes: wait for the listing, then download every candidate file by its own tags.
    The downloads are independent, one per file: they run at once, one worker per file."""
    from concurrent.futures import ThreadPoolExecutor
    expected = {}
    for name in candidates:
        tags = wheel_tags(name)
        require(tags["version"] == args.version, "candidate %s is not version %s" % (name, args.version))
        expected.setdefault(tags["distribution"], []).append(name)
    wait = wait_for_index(args.index_url, args.version, expected, args.index_wait, args.index_wait_step)
    request["index_wait"] = wait
    if not wait["listed"]:
        # pip still runs once: its own answer is kept beside the listing; the report fails as index lag
        print("WAIT-EXPIRED published-index version=%s after %ss: not listed %s" % (
            args.version, wait["waited_s"], ",".join(wait["missing"])), flush=True)
    scratch = scratch_dir("published-files-")

    def one(item):
        i, name = item
        tags = wheel_tags(name)
        dest = scratch / str(i)
        dest.mkdir(parents=True)
        code = _run(pip_download_command(args.python, args.index_url, tags["distribution"], args.version, tags,
                                         dest), Path(log).with_name("download-%d.log" % i), args.timeout)
        files = sorted(p.name for p in dest.glob("*.whl"))
        row = dict(candidate=name, distribution=tags["distribution"], pip_exit=code, files=files,
                   file=files[0] if code == 0 and len(files) == 1 else None)
        if row["file"]:
            row["sha256"] = sha256(dest / row["file"])
            try:
                row["index_sha256"] = index_digest(args.index_url, tags["distribution"], row["file"])
            except (OSError, ValueError) as error:
                row["index_error"] = type(error).__name__ + ": " + str(error)
        return row

    with ThreadPoolExecutor(max_workers=max(1, len(candidates))) as pool:
        return list(pool.map(one, enumerate(sorted(candidates))))


def install_and_verify(args, request, out):
    """pypi-install, import and verify-quick in a fresh venv. Serial by nature: each step needs the one before."""
    venv = scratch_dir("published-venv-") / "venv"
    python = str(venv / "bin" / "python")
    log = out / "install.log"
    code = _run([args.python, "-m", "venv", str(venv)], log, 600)
    if code == 0:
        code = _run([python, "-m", "pip", "install", "--no-input", "--disable-pip-version-check", "--no-cache-dir",
                     "--index-url", args.index_url, "%s==%s" % (PROJECT, args.version)], log, args.timeout)
    installed = {}
    if code == 0:
        listing = subprocess.run([python, "-m", "pip", "list", "--format=json", "--disable-pip-version-check"],
                                 capture_output=True, text=True)
        for row in json.loads(listing.stdout or "[]") if listing.returncode == 0 else []:
            if normalize(row["name"]) == PROJECT or normalize(row["name"]).startswith(PROJECT + "-"):
                installed[normalize(row["name"])] = row["version"]
    request["install"] = dict(exit=code, installed=installed)
    if code != 0:
        return
    probe = subprocess.run([python, "-c", "import mojolearn; print(mojolearn.__version__)"], capture_output=True,
                           text=True, timeout=600, cwd=str(scratch_dir("published-cwd-")))
    (out / "import.log").write_text(probe.stdout + probe.stderr)
    lines = probe.stdout.strip().splitlines()
    request["import"] = dict(exit=probe.returncode, version=lines[-1].strip() if lines else None)
    if probe.returncode != 0:
        return
    request["verify_quick"] = dict(exit=_run([python, "-m", "mojolearn", "verify", "--quick"], out / "verify-quick.log",
                                             args.verify_timeout, cwd=str(scratch_dir("published-cwd-"))))


def check(args):
    """The published check: resolve, download, compare, install, import, verify --quick. CPU only."""
    out = args.out.resolve()
    require(not out.exists(), "output must be new")
    out.mkdir(parents=True)
    candidates = _load(args.candidates)
    request = dict(version=args.version, index_url=args.index_url, candidates=candidates,
                   controller_sha256=sha256(__file__), runner=args.runner)
    try:
        require(isinstance(candidates, dict) and candidates
                and all(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) for v in candidates.values()),
                "candidates must be a {file name: sha256} map")
        request["resolved"] = resolve(args, candidates, request, out / "resolve.log")
        install_and_verify(args, request, out)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        request["error"] = type(error).__name__ + ": " + str(error)
    (out / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    report = judge(out, args.index_url)
    (out / REPORT).write_text(json.dumps(report, indent=2) + "\n")
    print("PUBLISHED version=%s status=%s %s%s" % (args.version, report["status"], " ".join(
        "%s=%d/%d" % (row["id"], row["completed"], row["expected"]) for row in report["coverage"]),
        (" failure=" + report["failure"]) if report.get("failure") else ""))
    if report.get("failure") == INDEX_LAG:
        print("PUBLISHED-INDEX-LAG version=%s %s (no verdict on the bytes; not recorded)" % (
            args.version, report.get("failure_why")))
    for row in report["mismatch"]:
        print("PUBLISHED-DIFFERS %s published=%s candidate=%s" % (row["file"], row["published_sha256"],
                                                                   row["candidate_sha256"]))
    return report["exit_code"]


def main(argv=None):
    sys.path.insert(0, str(HERE))
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    go = sub.add_parser("check", help="the published check on a CPU runner: files, hashes, install, import, "
                                      "verify --quick")
    go.add_argument("--version", required=True)
    go.add_argument("--candidates", type=Path, required=True, help="JSON {wheel file name: sha256}")
    go.add_argument("--index-url", default=PYPI_INDEX)
    go.add_argument("--python", default=sys.executable, help="runs pip and makes the venv")
    go.add_argument("--runner", default="", help="where this ran (the workflow run URL)")
    go.add_argument("--timeout", type=int, default=1800, help="seconds per pip download or install")
    go.add_argument("--verify-timeout", type=int, default=3600, help="seconds for verify --quick")
    go.add_argument("--index-wait", type=float, default=INDEX_WAIT_S,
                    help="seconds to wait for the index to list every expected file (0: one look)")
    go.add_argument("--index-wait-step", type=float, default=INDEX_WAIT_STEP_S,
                    help="seconds between two looks at the index")
    go.add_argument("--out", type=Path, required=True)
    again = sub.add_parser("judge")
    again.add_argument("--out", type=Path, required=True)
    again.add_argument("--index-url", default=PYPI_INDEX)
    args = parser.parse_args(argv)
    if args.command == "check":
        return check(args)
    report = judge(args.out, args.index_url)
    print(json.dumps({k: report[k] for k in ("status", "coverage")}))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
