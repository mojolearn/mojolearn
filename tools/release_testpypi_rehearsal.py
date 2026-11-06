#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE TESTPYPI REHEARSAL (Andrew, 2026-11-06: everything that can be checked before the real publish is
checked before it). tools/release.py's `testpypi-rehearsal` step uses these pieces, inside `--publish pypi`,
after every gate and the one verification round have passed and before the first real publish step:

1. UPLOAD. The exact gated wheel files (mojolearn-core, mojolearn-nvidia, mojolearn-amd, the Linux
   `mojolearn`, the macOS `mojolearn`) go to TestPyPI through the same publisher as the real release
   (tools/release_linux_publish.sh -> .github/workflows/release-provenance.yml, Trusted Publishing into the
   `testpypi[-core|-nvidia|-amd]` environments: no API token exists or is read). Before each upload the
   TestPyPI JSON API is read (`testpypi_file_state`): the same file name with the same sha256 is already
   rehearsed and is not uploaded again; the same file name with other bytes FAILS (TestPyPI, like PyPI,
   never takes a file name twice, even after a delete: https://pypi.org/help/#file-name-reuse), so a
   re-freeze of a version whose files are on TestPyPI needs a new version number. A file over TestPyPI's
   upload limit (`testpypi_file_limit`) is NOT uploaded and is reported as OWED, never silently: every check
   that does not need the upload still runs, with the gated file served locally in its place.

2. RESOLUTION, our names from TestPyPI ONLY. pip runs against `RoutingIndex`, a PEP 691 simple index on
   127.0.0.1 that routes per project: every project named `mojolearn` or `mojolearn-*` is fetched from
   TestPyPI and from nowhere else; every other project (numpy, torch, max, ...) from the real PyPI. A name
   of ours can never reach the real PyPI, whatever pip's resolver, extras or a stray pip config do (pip runs
   with --isolated, so PIP_INDEX_URL / PIP_EXTRA_INDEX_URL and user config are ignored; `--index-url` is the
   router alone). This is per-project index pinning without uv: pip has no per-project index option, and
   `--extra-index-url` mixes both indexes per name (dependency confusion), which is exactly what must not
   happen. For every root (`mojolearn`, each `mojolearn[<extra>]` the wheel declares, `mojolearn-core`,
   `mojolearn-nvidia`, `mojolearn-amd`), every platform tag the gated wheels carry and every Python version
   their classifiers list within Requires-Python, `pip install --dry-run --ignore-installed --report`
   resolves the set; `judge_resolution` requires our resolved projects to be exactly the expected ones at
   the version, each from a TestPyPI file URL the router served, each advertised sha256 equal to the gated
   wheel's, and no third-party project served by TestPyPI.

3. BYTES. `pip download --no-deps` of each rehearsed file through the router; its sha256 must equal the
   gated wheel's.

4. INSTALL. The macOS `mojolearn` installs into a fresh venv on this Mac from TestPyPI, pinned to the gated
   sha256 (`--require-hashes`), and imports (`import mojolearn`, version checked): no device, no lane.

Nothing in this module uploads, rents or touches a device; the step in tools/release.py drives the upload.
"""
import email.parser
import http.server
import json
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

TESTPYPI_SIMPLE = "https://test.pypi.org/simple/"
TESTPYPI_JSON = "https://test.pypi.org/pypi"
PYPI_SIMPLE = "https://pypi.org/simple/"
SIMPLE_JSON = "application/vnd.pypi.simple.v1+json"

#: TestPyPI upload limits (bytes per file). TestPyPI is its own index with its own project settings: a limit
#: PyPI granted (packaging/pypi_limits.json: mojolearn-nvidia and mojolearn-amd 250 MiB on pypi.org) does
#: not carry over. Every project takes TestPyPI's default, 100 MiB per file, until TestPyPI grants more;
#: record a TestPyPI grant here (project -> bytes) with its pypi/support issue in a comment.
TESTPYPI_DEFAULT_FILE_BYTES = 100 * 1024 ** 2
TESTPYPI_FILE_BYTES = {}


def normalize(name):
    """PEP 503 project name normalization."""
    return re.sub(r"[-_.]+", "-", str(name)).lower()


def is_ours(name):
    """Every project named `mojolearn` or `mojolearn-*` is ours: the five shipped projects, the folded
    mojolearn-nvidia-ptx and any future one. During the rehearsal ours come from TestPyPI only."""
    n = normalize(name)
    return n == "mojolearn" or n.startswith("mojolearn-")


def testpypi_file_limit(project):
    return TESTPYPI_FILE_BYTES.get(normalize(project), TESTPYPI_DEFAULT_FILE_BYTES)


def wheel_project(wheel):
    """The PyPI project a wheel file belongs to (mojolearn_nvidia-... -> mojolearn-nvidia)."""
    return normalize(Path(wheel).name.split("-")[0])


def wheel_platform(wheel):
    """The platform tag of a wheel file name (PEP 427: the last tag)."""
    return Path(wheel).name[:-4].split("-")[-1]


def wheel_metadata(wheel):
    """METADATA of a wheel: name, version, extras, requires_python, the Python classifiers."""
    with zipfile.ZipFile(wheel) as z:
        member = next(n for n in z.namelist() if n.endswith(".dist-info/METADATA") and n.count("/") == 1)
        msg = email.parser.Parser().parsestr(z.read(member).decode("utf-8"))
    pythons = []
    for c in msg.get_all("Classifier") or []:
        m = re.fullmatch(r"Programming Language :: Python :: (3\.\d+)", c.strip())
        if m:
            pythons.append(m.group(1))
    return dict(name=msg.get("Name"), version=msg.get("Version"), extras=sorted(msg.get_all("Provides-Extra") or []),
                requires_python=msg.get("Requires-Python") or "", pythons=pythons)


def python_versions(meta, fallback=("3.10", "3.11", "3.12", "3.13", "3.14")):
    """The Python versions a wheel declares (classifiers) within its Requires-Python lower bound."""
    floor = re.search(r">=\s*3\.(\d+)", meta.get("requires_python") or "")
    lo = int(floor.group(1)) if floor else 0
    found = meta.get("pythons") or list(fallback)
    return sorted({v for v in found if int(v.split(".")[1]) >= lo}, key=lambda v: int(v.split(".")[1]))


# ------------------------------------------------------------ TestPyPI file state (resume, re-freeze)
def fetch_json(url, timeout=30):
    """(status, doc): 404 -> (404, None); any other HTTP or network error raises."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200, json.load(r)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return 404, None
        raise


def testpypi_file_state(project, version, filename, sha, fetch=fetch_json):
    """('absent'|'same'|'different', served_sha): what TestPyPI holds under this exact file name.
    `same`: rehearsed earlier, never uploaded again. `different`: TestPyPI never takes a file name twice."""
    status, doc = fetch(f"{TESTPYPI_JSON}/{normalize(project)}/{version}/json")
    if status == 404 or not doc:
        return "absent", None
    for f in doc.get("urls") or []:
        if f.get("filename") == filename:
            served = (f.get("digests") or {}).get("sha256")
            return ("same" if served == sha else "different"), served
    return "absent", None


def different_bytes_reason(filename, served, gated, version):
    return (f"TestPyPI already holds {filename} with sha256 {served}, the gated wheel is {gated}: TestPyPI never "
            f"takes a file name twice (https://pypi.org/help/#file-name-reuse), so this re-freeze of {version} cannot "
            f"be rehearsed. Fix: release the re-frozen source as the next version (bump the version and refreeze); "
            f"nothing went to the real PyPI")


# ------------------------------------------------------------ the routing index
def _absolute(doc, base):
    """A PEP 691 project page with every file URL made absolute (pip then downloads from the upstream host)."""
    for f in doc.get("files") or []:
        if "url" in f:
            f["url"] = urllib.parse.urljoin(base, f["url"])
    return doc


def upstream_page(url, timeout=60):
    """(status, PEP 691 JSON doc or None) of one upstream simple project page."""
    req = urllib.request.Request(url, headers={"Accept": SIMPLE_JSON})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return 200, json.load(r)
    except urllib.error.HTTPError as exc:
        return exc.code, None


class RoutingIndex:
    """A local PEP 691 simple index: `/simple/<project>/` comes from TestPyPI when the project is ours
    (`is_ours`), else from the real PyPI. `served` records every file URL handed out per side, so a judge can
    prove each resolved file came from the side it must. `page` is the upstream fetch (tests pass a fake)."""

    def __init__(self, testpypi=TESTPYPI_SIMPLE, pypi=PYPI_SIMPLE, page=upstream_page):
        self.upstream = dict(testpypi=testpypi, pypi=pypi)
        self.page = page
        self.served = dict(testpypi=set(), pypi=set())
        self.requests = []          # (project, side) in order
        self._lock = threading.Lock()
        self._server = None

    def route(self, project):
        """'testpypi' for ours, 'pypi' for everything else. The only place the side is chosen."""
        return "testpypi" if is_ours(project) else "pypi"

    def project_page(self, project):
        """(status, body bytes) of the routed project page."""
        project = normalize(project)
        side = self.route(project)
        url = self.upstream[side].rstrip("/") + "/" + project + "/"
        status, doc = self.page(url)
        with self._lock:
            self.requests.append((project, side))
        if status != 200 or doc is None:
            return status or 502, b""
        doc = _absolute(doc, url)
        with self._lock:
            self.served[side].update(f["url"].split("#")[0] for f in doc.get("files") or [] if f.get("url"))
        return 200, json.dumps(doc).encode()

    # the HTTP side
    def start(self):
        index = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                parts = [p for p in urllib.parse.urlparse(self.path).path.split("/") if p]
                if len(parts) != 2 or parts[0] != "simple":
                    self.send_response(404)
                    self.end_headers()
                    return
                status, body = index.project_page(parts[1])
                self.send_response(status)
                self.send_header("Content-Type", SIMPLE_JSON)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self.url

    @property
    def url(self):
        return f"http://127.0.0.1:{self._server.server_address[1]}/simple/" if self._server else None

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()


# ------------------------------------------------------------ pip commands
def pip_base(pip_python):
    """pip with no user config and no PIP_* environment (--isolated): the router is the only index."""
    return [str(pip_python), "-m", "pip", "--isolated", "--disable-pip-version-check"]


def resolve_command(pip_python, index_url, root, platform, python, report, find_links=None):
    cmd = pip_base(pip_python) + ["install", "--dry-run", "--ignore-installed", "--no-cache-dir",
                                  "--only-binary=:all:", "--platform", platform, "--python-version", python,
                                  "--implementation", "cp", "--index-url", index_url, "--report", str(report)]
    if find_links:
        cmd += ["--find-links", str(find_links)]
    return cmd + [root]


def download_command(pip_python, index_url, project, version, platform, dest):
    return pip_base(pip_python) + ["download", "--no-deps", "--no-cache-dir", "--only-binary=:all:",
                                   "--platform", platform, "--implementation", "cp", "--index-url", index_url,
                                   "--dest", str(dest), f"{project}=={version}"]


def install_command(venv, index_url, requirements):
    return pip_base(Path(venv) / "bin" / "python") + [
        "install", "--no-cache-dir", "--only-binary=:all:", "--require-hashes", "--index-url", index_url,
        "-r", str(requirements)]


def import_command(venv, version):
    return [str(Path(venv) / "bin" / "python"), "-I", "-c",
            f"import mojolearn; assert mojolearn.__version__ == {version!r}, mojolearn.__version__; "
            "print('import mojolearn', mojolearn.__version__)"]


# ------------------------------------------------------------ the judge
def judge_resolution(report, expected, version, served, local=()):
    """Problems (empty: passed) of one `pip install --dry-run --report` against the expected set.

    expected: {project: (filename, sha256)} of OUR projects this root must install, nothing more.
    served:   RoutingIndex.served ({'testpypi': urls, 'pypi': urls}).
    local:    projects whose upload was skipped (served from the local gated file instead of TestPyPI)."""
    problems = []
    got = {}
    for item in (report or {}).get("install") or []:
        meta = item.get("metadata") or {}
        name = normalize(meta.get("name", ""))
        info = item.get("download_info") or {}
        url = (info.get("url") or "").split("#")[0]
        hashes = (info.get("archive_info") or {}).get("hashes") or {}
        sha = hashes.get("sha256") or ((info.get("archive_info") or {}).get("hash") or "").partition("sha256=")[2]
        if is_ours(name):
            got[name] = dict(version=meta.get("version"), url=url, sha=sha, file=url.rsplit("/", 1)[-1])
        elif url in served.get("testpypi", ()):
            problems.append(f"{name} (not ours) came from TestPyPI ({url}): third-party projects come from PyPI only")
    want = {normalize(k): v for k, v in expected.items()}
    if set(got) != set(want):
        problems.append(f"resolved {sorted(got)}, want exactly {sorted(want)}")
    for name, row in sorted(got.items()):
        if name not in want:
            continue
        filename, sha = want[name]
        if row["version"] != version:
            problems.append(f"{name} resolved version {row['version']}, want {version}")
        if name in local:
            if not row["url"].startswith("file:"):
                problems.append(f"{name}: its upload was skipped, but pip took {row['url']}")
        elif row["url"] not in served.get("testpypi", ()):
            problems.append(f"{name} came from {row['url'] or 'nowhere'}, not from a TestPyPI file the router served")
        if row["file"] != filename:
            problems.append(f"{name} resolved {row['file']}, want {filename}")
        if row["sha"] != sha:
            problems.append(f"{name} {row['file']}: index sha256 {row['sha'] or '(none)'}, gated {sha}")
    return problems


def requirements_text(project, version, sha):
    return f"{project}=={version} --hash=sha256:{sha}\n"
