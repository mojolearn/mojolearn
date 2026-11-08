# SPDX-License-Identifier: Apache-2.0
"""published-<vendor> is a bytes check, never a reinstall (Andrew, 2026-11-02; lane published-no-reverify).

tools/release_published_install.py check resolves each published package with pip for the candidate's
own platform tags from a FAKE file:// index (nothing reaches PyPI, nothing is installed, no device runs),
compares the sha256 with the gated candidate wheel, checks the archive's RECORD, and takes the gate's
receipts by reference. Different bytes fail with both hashes."""
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


def gate_receipt(directory, wheels, name="gate-smoke.json"):
    path = Path(directory) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(verdict="PASS", candidate=dict(wheels={
        P.wheel_tags(w)["distribution"]: sha(w) for w in wheels}))))
    return path


def check(tmp_path, vendor, candidates, index, gate, out="out", verdict="PASS: 11 records", wait=("0", "20")):
    # --index-wait 0: one look at the index (the fixtures list their files up front unless a test says otherwise)
    argv = ["check", "--vendor", vendor, "--version", VERSION, "--index-url", index, "--python", sys.executable,
            "--gate-verdict", verdict, "--out", str(tmp_path / out), "--index-wait", wait[0], "--index-wait-step", wait[1]]
    argv += [a for w in candidates for a in ("--candidate-wheel", str(w))]
    argv += [a for g in gate for a in ("--gate-evidence", str(g))]
    rc = P.main(argv)
    return rc, json.loads((tmp_path / out / P.REPORT).read_text())


def nvidia_set(directory, **kw):
    # the NVIDIA wheel carries its PTX set (2026-11-05): two distributions
    return [make_wheel(directory, d, **kw) for d in ("mojolearn-core", "mojolearn-nvidia")]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("the bytes check reached the network")
    monkeypatch.setattr(P.urllib.request, "urlopen", refuse)


def test_published_bytes_equal_to_the_gated_candidates_pass_without_any_install(tmp_path):
    cands = nvidia_set(tmp_path / "cand")
    index = fake_index(tmp_path / "index", cands)
    gate = gate_receipt(tmp_path / "gate", cands)
    rc, report = check(tmp_path, "nvidia", cands, index, [gate])
    assert rc == 0 and report["status"] == "passed", report["checks"]
    assert report["schema"] == P.REPORT_SCHEMA and [r["id"] for r in report["coverage"]] == list(P.COVERAGE_IDS)
    assert report["published"] == {w.name: sha(w) for w in cands} and report["mismatch"] == []
    out = tmp_path / "out"
    assert not (out / "venv").exists()                     # nothing was installed
    log = (out / "resolve.log").read_text()
    assert log.count("pip download") == 2 and "--platform manylinux_2_35_x86_64" in log and "--no-deps" in log
    # the gate's receipt is kept by content, and judge re-derives the same report from the retained files
    copied = json.loads((out / "request.json").read_text())["gate"]["evidence"]
    assert copied[0]["path"] == f"gate-evidence/{sha(gate)[:16]}-gate-smoke.json" and copied[0]["sha256"] == sha(gate)
    assert P.judge(out, index) == report
    # against PyPI (the ledger's re-derivation) the same directory does not pass: a fake index is no evidence
    assert P.judge(out)["status"] == "failed"


def test_published_bytes_that_differ_fail_with_both_hashes_and_nothing_reruns(tmp_path):
    cands = nvidia_set(tmp_path / "cand")
    other = make_wheel(tmp_path / "pub", "mojolearn-nvidia", payload=b"x = 2\n")
    index = fake_index(tmp_path / "index", [cands[0], other])
    rc, report = check(tmp_path, "nvidia", cands, index, [gate_receipt(tmp_path / "gate", cands)])
    assert rc == 1 and report["status"] == "failed"
    assert report["mismatch"] == [dict(file=other.name, published_sha256=sha(other), candidate=cands[1].name,
                                       candidate_sha256=sha(cands[1]))]
    cov = {r["id"]: r["status"] for r in report["coverage"]}
    assert cov["published-artifact-hashes"] == "failed" and cov["index-resolution"] == "passed"
    assert report["failure"] == P.HASH_MISMATCH == P.failure_kind(report) and sha(other) in report["failure_why"]
    assert (tmp_path / "out" / "resolve.log").read_text().count("pip download") == 2   # one resolution each


def test_a_package_missing_from_the_index_or_a_wrong_index_digest_fails_resolution(tmp_path):
    cands = nvidia_set(tmp_path / "cand")
    index = fake_index(tmp_path / "index", cands[:1], advertise={cands[0].name: "0" * 64})
    rc, report = check(tmp_path, "nvidia", cands, index, [gate_receipt(tmp_path / "gate", cands)])
    assert rc == 1
    failed = [c["name"] for c in report["checks"]["index-resolution"] if not c["passed"]]
    assert "resolved:mojolearn-nvidia" in failed
    # a file whose bytes are not what the index advertises: pip refuses it, or the digest check does
    assert "resolved:mojolearn-core" in failed or "index-digest:" + cands[0].name in failed


def test_a_wheel_for_another_platform_is_not_resolved(tmp_path):
    mac = make_wheel(tmp_path / "cand", "mojolearn", tag=MAC)
    linux = make_wheel(tmp_path / "linux", "mojolearn", tag=LINUX)
    index = fake_index(tmp_path / "index", [linux])        # only the Linux file is on the index
    rc, report = check(tmp_path, "apple", [mac], index, [gate_receipt(tmp_path / "gate", [mac])])
    assert rc == 1 and any(c["name"] == "resolved:mojolearn" and not c["passed"]
                           for c in report["checks"]["index-resolution"])
    assert "--platform macosx_26_0_arm64" in (tmp_path / "out" / "resolve.log").read_text()


def test_apple_passes_for_its_one_wheel(tmp_path):
    mac = make_wheel(tmp_path / "cand", "mojolearn", tag=MAC)
    index = fake_index(tmp_path / "index", [mac])
    receipt = tmp_path / "self-test" / "results.json"
    receipt.parent.mkdir()
    receipt.write_text(json.dumps(dict(status="PASSED", wheel_sha256=sha(mac))))
    rc, report = check(tmp_path, "apple", [mac], index, [receipt])
    assert rc == 0, report["checks"]


def test_a_broken_record_fails_integrity_even_with_equal_bytes(tmp_path):
    cands = nvidia_set(tmp_path / "cand", break_record=True)
    index = fake_index(tmp_path / "index", cands)
    rc, report = check(tmp_path, "nvidia", cands, index, [gate_receipt(tmp_path / "gate", cands)])
    assert rc == 1
    assert {r["id"]: r["status"] for r in report["coverage"]}["wheel-integrity"] == "failed"
    assert any("RECORD hash differs" in p for c in report["checks"]["wheel-integrity"] for p in c["problems"])


def test_gate_evidence_must_name_the_published_bytes_and_stay_unchanged(tmp_path):
    cands = nvidia_set(tmp_path / "cand")
    index = fake_index(tmp_path / "index", cands)
    stale = gate_receipt(tmp_path / "gate", cands[:1])       # the gate never saw the NVIDIA wheel's bytes
    rc, report = check(tmp_path, "nvidia", cands, index, [stale])
    assert rc == 1 and any(c["name"] == "gated-bytes:" + cands[1].name and not c["passed"]
                           for c in report["checks"]["gate-evidence"])
    rc, report = check(tmp_path, "nvidia", cands, index, [], out="none")
    assert rc == 1 and {r["id"]: r["status"] for r in report["coverage"]}["gate-evidence"] == "failed"
    good = gate_receipt(tmp_path / "gate2", cands)
    rc, _ = check(tmp_path, "nvidia", cands, index, [good], out="edited")
    assert rc == 0
    copy = next((tmp_path / "edited" / "gate-evidence").iterdir())
    copy.write_text(copy.read_text() + " ")
    assert P.judge(tmp_path / "edited", index)["status"] == "failed"


def test_pip_command_carries_the_candidate_tags():
    tags = P.wheel_tags("mojolearn_core-0.8.66-cp312-cp312-manylinux_2_28_x86_64.manylinux_2_35_x86_64.whl")
    cmd = P.pip_download_command("py", "file:///i", "mojolearn-core", "0.8.66", tags, "/d")
    assert cmd[-1] == "mojolearn-core==0.8.66"
    assert cmd.count("--platform") == 2 and "--abi" in cmd and cmd[cmd.index("--python-version") + 1] == "3.12"
    assert "install" not in cmd and "--no-deps" in cmd and "--only-binary=:all:" in cmd


def test_judge_still_reads_an_old_smoke_directory_as_a_smoke(tmp_path):
    (tmp_path / "request.json").write_text(json.dumps(dict(vendor="amd", version=VERSION)))
    with pytest.raises((OSError, ValueError, KeyError)):
        P.judge(tmp_path)              # the smoke judge wants its manifest; the bytes judge is never used


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
    cands = nvidia_set(tmp_path / "cand")
    index = (tmp_path / "index" / "simple").as_uri()       # nothing there yet: the upload is not listed
    clock = FakeTime(on_sleep=lambda n: fake_index(tmp_path / "index", cands) if n == 2 else None)
    monkeypatch.setattr(P, "time", clock)
    rc, report = check(tmp_path, "nvidia", cands, index, [gate_receipt(tmp_path / "gate", cands)],
                       wait=("900", "20"))
    assert rc == 0 and report["status"] == "passed" and report["failure"] is None, report["checks"]
    wait = json.loads((tmp_path / "out" / "request.json").read_text())["index_wait"]
    assert wait["listed"] is True and wait["polls"] == 3 and wait["waited_s"] == 40 and clock.sleeps == [20, 20]
    assert any(c["name"] == "index-listing" and c["passed"] for c in report["checks"]["index-resolution"])
    waits = [line for line in capsys.readouterr().out.splitlines() if line.startswith("WAIT published-index")]
    assert len(waits) == 2 and "listed=0/2" in waits[0] and cands[0].name in waits[0]
    # pip ran after the listing appeared, once per package
    assert (tmp_path / "out" / "resolve.log").read_text().count("pip download") == 2


def test_a_listing_that_never_appears_fails_as_index_lag_not_as_a_hash_mismatch(tmp_path, monkeypatch, capsys):
    mac = make_wheel(tmp_path / "cand", "mojolearn", tag=MAC)
    index = (tmp_path / "index" / "simple").as_uri()
    clock = FakeTime()
    monkeypatch.setattr(P, "time", clock)
    rc, report = check(tmp_path, "apple", [mac], index, [gate_receipt(tmp_path / "gate", [mac])], wait=("60", "20"))
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


def test_an_old_report_with_nothing_resolved_is_index_lag():
    old = dict(status="failed", mismatch=[], checks={"published-artifact-hashes": [
        dict(name="resolved-files", passed=False, why="nothing was resolved")]})
    assert P.failure_kind(old) == P.INDEX_LAG
    assert P.failure_kind(dict(status="passed")) is None
    assert P.failure_kind(dict(status="failed", mismatch=[dict(file="f")])) == P.HASH_MISMATCH
