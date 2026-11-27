# SPDX-License-Identifier: Apache-2.0
"""The published check (Andrew, 2026-11-27): ONE CPU job on a GitHub runner, no GPU, nothing rented.

tools/release_published_install.py check downloads every candidate file from a FAKE file:// index by the candidate's
own platform tags (nothing reaches PyPI), compares the sha256 with the candidate wheel, then installs, imports and
runs `verify --quick` (faked here: the fixtures are not real Mojolearn wheels). Different bytes fail with both
hashes; an index that never lists the files fails as index lag."""
import base64
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import release_published_install as P  # noqa: E402

VERSION = "0.8.66"
LINUX = "py3-none-manylinux_2_35_x86_64"
MAC = "py3-none-macosx_26_0_arm64"


def make_wheel(directory, distribution, tag=LINUX, payload=b"x = 1\n", break_record=False):
    """A real wheel: payload, METADATA, WHEEL and a RECORD with urlsafe sha256 and sizes."""
    stem = distribution.replace("-", "_")
    path = Path(directory) / f"{stem}-{VERSION}-{tag}.whl"
    path.parent.mkdir(parents=True, exist_ok=True)
    info = f"{stem}-{VERSION}.dist-info"
    members = {f"{stem}/__init__.py": payload,
               f"{info}/METADATA": f"Metadata-Version: 2.1\nName: {distribution}\nVersion: {VERSION}\n".encode(),
               f"{info}/WHEEL": f"Wheel-Version: 1.0\nGenerator: fixture\nRoot-Is-Purelib: false\nTag: {tag}\n".encode()}
    rows = []
    for name, data in members.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        rows.append(f"{name},sha256={digest},{len(data)}")
    if break_record:
        rows[0] = rows[0].split(",")[0] + ",sha256=AAAA,6"
    rows.append(f"{info}/RECORD,,")
    with zipfile.ZipFile(path, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
        z.writestr(f"{info}/RECORD", "\n".join(rows) + "\n")
    return path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fake_index(root, wheels, advertise=None):
    """A PEP 503 simple index on disk: simple/<project>/index.html with #sha256= links to files/."""
    files = Path(root) / "files"
    files.mkdir(parents=True, exist_ok=True)
    for w in wheels:
        shutil.copyfile(w, files / Path(w).name)
        project = P.wheel_tags(w)["distribution"]
        page = Path(root) / "simple" / project / "index.html"
        page.parent.mkdir(parents=True, exist_ok=True)
        digest = (advertise or {}).get(Path(w).name, sha(w))
        old = page.read_text() if page.is_file() else ""
        page.write_text(old + f'<a href="../../files/{Path(w).name}#sha256={digest}">{Path(w).name}</a>\n')
    return (Path(root) / "simple").as_uri()


def fake_install(monkeypatch, install=0, imported=0, quick=0, version=VERSION):
    """install_and_verify without a real pip install: the exit codes and versions a runner would record."""
    def run(args, request, out):
        request["install"] = dict(exit=install, installed={"mojolearn": version, "mojolearn-core": version}
                                  if install == 0 else {})
        if install == 0:
            request["import"] = dict(exit=imported, version=version if imported == 0 else None)
            if imported == 0:
                request["verify_quick"] = dict(exit=quick)
    monkeypatch.setattr(P, "install_and_verify", run)


def check(tmp_path, candidates, index, out="out", wait=("0", "20")):
    # --index-wait 0: one look at the index (the fixtures list their files up front unless a test says otherwise)
    cands = tmp_path / (out + "-candidates.json")
    cands.write_text(json.dumps({Path(w).name: sha(w) for w in candidates}))
    argv = ["check", "--version", VERSION, "--candidates", str(cands), "--index-url", index, "--python",
            sys.executable, "--out", str(tmp_path / out), "--index-wait", wait[0], "--index-wait-step", wait[1]]
    rc = P.main(argv)
    return rc, json.loads((tmp_path / out / P.REPORT).read_text())


def release_set(directory, **kw):
    """Every file a release publishes: the core, both vendors, the Linux meta wheel and the macOS wheel."""
    return [make_wheel(directory, d, **kw) for d in ("mojolearn-core", "mojolearn-nvidia", "mojolearn-amd")] + [
        make_wheel(directory, "mojolearn", payload=b""), make_wheel(directory, "mojolearn", tag=MAC)]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("the published check reached the network")
    monkeypatch.setattr(P.urllib.request, "urlopen", refuse)


def test_published_files_equal_to_the_candidates_pass_install_import_and_verify(tmp_path, monkeypatch):
    fake_install(monkeypatch)
    cands = release_set(tmp_path / "cand")
    rc, report = check(tmp_path, cands, fake_index(tmp_path / "index", cands))
    assert rc == 0 and report["status"] == "passed" and report["failure"] is None, report["checks"]
    assert report["schema"] == P.REPORT_SCHEMA and [r["id"] for r in report["coverage"]] == list(P.COVERAGE_IDS)
    assert report["published"] == {w.name: sha(w) for w in cands} and report["mismatch"] == []
    # the Linux meta wheel and the macOS wheel share the project name: each is fetched by its own tags
    assert {r["candidate"] for r in json.loads((tmp_path / "out" / "request.json").read_text())["resolved"]} == \
        {w.name for w in cands}
    assert P.judge(tmp_path / "out", fake_index(tmp_path / "index", cands)) == report    # re-derived the same
    # nothing downloaded is kept in the evidence folder (the hashes are in request.json)
    assert not list((tmp_path / "out").rglob("*.whl"))


def test_published_bytes_that_differ_fail_with_both_hashes_and_nothing_reruns(tmp_path, monkeypatch):
    fake_install(monkeypatch)
    cands = release_set(tmp_path / "cand")
    other = make_wheel(tmp_path / "pub", "mojolearn-nvidia", payload=b"x = 2\n")
    rc, report = check(tmp_path, cands, fake_index(tmp_path / "index", [c for c in cands if c.name != other.name]
                                                   + [other]))
    assert rc == 1 and report["failure"] == P.HASH_MISMATCH == P.failure_kind(report)
    assert report["mismatch"] == [dict(file=other.name, published_sha256=sha(other),
                                       candidate_sha256=sha(cands[1]))]
    cov = {r["id"]: r["status"] for r in report["coverage"]}
    assert cov["published-artifact-hashes"] == "failed" and cov["index-resolution"] == "passed"
    assert sha(other) in report["failure_why"] and sha(cands[1]) in report["failure_why"]


def test_an_advertised_digest_that_differs_from_the_file_fails_resolution(tmp_path, monkeypatch):
    fake_install(monkeypatch)
    cands = release_set(tmp_path / "cand")
    rc, report = check(tmp_path, cands, fake_index(tmp_path / "index", cands, advertise={cands[0].name: "0" * 64}))
    assert rc == 1 and {r["id"]: r["status"] for r in report["coverage"]}["index-resolution"] == "failed"


@pytest.mark.parametrize("step,kw", [("pypi-install", dict(install=1)), ("import", dict(imported=1)),
                                     ("verify-quick", dict(quick=1))])
def test_a_failed_install_import_or_verify_fails_that_check(tmp_path, monkeypatch, step, kw):
    fake_install(monkeypatch, **kw)
    cands = release_set(tmp_path / "cand")
    rc, report = check(tmp_path, cands, fake_index(tmp_path / "index", cands))
    cov = {r["id"]: r["status"] for r in report["coverage"]}
    assert rc == 1 and cov[step] == "failed" and cov["published-artifact-hashes"] == "passed"
    assert report["failure"] == P.CHECK_FAILED and step in report["failure_why"]


def test_an_installed_distribution_at_another_version_fails_the_install_check(tmp_path, monkeypatch):
    fake_install(monkeypatch, version="0.0.1")
    cands = release_set(tmp_path / "cand")
    rc, report = check(tmp_path, cands, fake_index(tmp_path / "index", cands))
    assert rc == 1 and {r["id"]: r["status"] for r in report["coverage"]}["pypi-install"] == "failed"


def test_the_runner_check_names_no_gpu_box_or_vendor(tmp_path):
    text = (TOOLS / "release_published_install.py").read_text()
    for word in ("nvidia_central", "do_amd_steward", "box_persist", "qualify_verifier_wheel", "adapter_gate"):
        assert word not in text
    wf = (TOOLS.parent / ".github" / "workflows" / "published-check.yml").read_text()
    assert "runs-on: ubuntu-24.04" in wf and "release_published_install.py check" in wf
    assert not any(word in wf for word in ("nvidia_central", "do_amd_steward", "runpod", "self-hosted"))


# ---------------------------------------------------------------- the index wait (0.8.66 run v, 2026-11-08)
class FakeTime:
    """time for the check: sleep() advances a fake clock and may run a hook (the index publishing late)."""
    def __init__(self, on_sleep=None):
        self.now, self.sleeps, self.on_sleep = 0.0, [], on_sleep

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds
        if self.on_sleep:
            self.on_sleep(len(self.sleeps))


def test_the_check_waits_for_an_index_that_lists_the_files_late(tmp_path, monkeypatch, capsys):
    fake_install(monkeypatch)
    cands = [make_wheel(tmp_path / "cand", d) for d in ("mojolearn-core", "mojolearn-nvidia")]
    index = (tmp_path / "index" / "simple").as_uri()       # nothing there yet: the upload is not listed
    clock = FakeTime(on_sleep=lambda n: fake_index(tmp_path / "index", cands) if n == 2 else None)
    monkeypatch.setattr(P, "time", clock)
    rc, report = check(tmp_path, cands, index, wait=("900", "20"))
    assert rc == 0 and report["status"] == "passed" and report["failure"] is None, report["checks"]
    wait = json.loads((tmp_path / "out" / "request.json").read_text())["index_wait"]
    assert wait["listed"] is True and wait["polls"] == 3 and wait["waited_s"] == 40 and clock.sleeps == [20, 20]
    assert any(c["name"] == "index-listing" and c["passed"] for c in report["checks"]["index-resolution"])
    waits = [line for line in capsys.readouterr().out.splitlines() if line.startswith("WAIT published-index")]
    assert len(waits) == 2 and "listed=0/2" in waits[0] and cands[0].name in waits[0]
    # pip ran after the listing appeared, once per file
    assert sorted(p.name for p in (tmp_path / "out").glob("download-*.log")) == ["download-0.log", "download-1.log"]


def test_a_listing_that_never_appears_fails_as_index_lag_not_as_a_hash_mismatch(tmp_path, monkeypatch, capsys):
    fake_install(monkeypatch)
    mac = make_wheel(tmp_path / "cand", "mojolearn", tag=MAC)
    index = (tmp_path / "index" / "simple").as_uri()
    clock = FakeTime()
    monkeypatch.setattr(P, "time", clock)
    rc, report = check(tmp_path, [mac], index, wait=("60", "20"))
    assert rc == 1 and report["status"] == "failed"
    assert clock.sleeps == [20, 20, 20]                      # looks at 0, 20, 40 and 60 s, then the cap
    assert report["failure"] == P.INDEX_LAG == P.failure_kind(report) and mac.name in report["failure_why"]
    assert report["mismatch"] == []
    assert any(c["name"] == "index-listing" and not c["passed"] for c in report["checks"]["index-resolution"])
    out = capsys.readouterr().out
    assert out.count("WAIT published-index") == 3 and "WAIT-EXPIRED" in out and "PUBLISHED-INDEX-LAG" in out
    assert "failure=index-lag" in out
    assert P.judge(tmp_path / "out", index)["failure"] == P.INDEX_LAG     # re-derived the same way


def test_the_listing_reads_pypis_json_api_and_the_simple_page():
    assert P.json_api_url("https://pypi.org/simple", "mojolearn_core", VERSION) == \
        "https://pypi.org/pypi/mojolearn-core/%s/json" % VERSION
    assert P.json_api_url("https://test.pypi.org/simple/", "mojolearn", VERSION) == \
        "https://test.pypi.org/pypi/mojolearn/%s/json" % VERSION
    assert P.json_api_url("file:///x/simple", "mojolearn", VERSION) is None
    name = "mojolearn-%s-%s.whl" % (VERSION, MAC)
    page = lambda index, dist: '<a href="https://files.example/%s#sha256=%s">x</a>' % (name, "0" * 64)
    api_lists = {"now": []}
    api = lambda url: api_lists["now"]
    expected = {"mojolearn": [name]}
    # on the simple page but not yet in the JSON API: not listed
    assert P.index_listing("https://pypi.org/simple", VERSION, expected, api=api, page=page)[0] == [name]
    api_lists["now"] = [name]
    assert P.index_listing("https://pypi.org/simple", VERSION, expected, api=api, page=page) == ([], [])
    # a page that cannot be read yet lists nothing and says why
    def missing(index, dist):
        raise OSError("404")
    missing_files, errors = P.index_listing("https://pypi.org/simple", VERSION, expected, api=api, page=missing)
    assert missing_files == [name] and "OSError" in errors[0]


def test_the_json_api_answers_404_with_no_files(monkeypatch):
    def not_found(request, timeout=None):
        raise P.urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)
    monkeypatch.setattr(P.urllib.request, "urlopen", not_found)
    assert P.json_api_files("https://pypi.org/pypi/mojolearn/%s/json" % VERSION) == []


def test_failure_kinds():
    assert P.failure_kind(dict(status="passed")) is None
    assert P.failure_kind(dict(status="failed", mismatch=[dict(file="f")])) == P.HASH_MISMATCH
    assert P.failure_kind(dict(status="failed", failure=P.INDEX_LAG)) == P.INDEX_LAG
    assert P.failure_kind(dict(status="failed", error="OSError: x")) == P.CHECK_ERROR
    assert P.failure_kind(dict(status="failed")) == P.CHECK_FAILED
