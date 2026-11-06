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


def check(tmp_path, vendor, candidates, index, gate, out="out", verdict="PASS: 11 records"):
    argv = ["check", "--vendor", vendor, "--version", VERSION, "--index-url", index, "--python", sys.executable,
            "--gate-verdict", verdict, "--out", str(tmp_path / out)]
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
