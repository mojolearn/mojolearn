"""tools/verification_evidence.py: staging on the box, installing on the Mac. No box, no network."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import verification_evidence as E  # noqa: E402

COMMIT = "a" * 40


def job(tmp_path, name):
    res = tmp_path / name
    (res / "logs").mkdir(parents=True)
    return res


def verify_results(tmp_path):
    res = job(tmp_path, "idrec")
    report = dict(verdict="VERIFIED", exit=0, detail="d", device=dict(mojolearn_version="0.8.67", commit=COMMIT),
                  extended={"verify-graph": dict(verdict="OWED")}, extended_summary=dict(OWED=1),
                  self_test=dict(passed=True), what_ran=dict(sides={}), paths_sample=[dict(lane="ols")])
    (res / "verify-all.json").write_text(json.dumps(report))
    (res / "logs" / "verify.log").write_text("x" * (E.LOG_CAP_BYTES + 100))
    (res / "logs" / "verify.busy").write_text("10\n12\n")
    (res / "steps.tsv").write_text("verify\t0\t42\t2026-11-26T00:00:00Z\t2026-11-26T00:00:42Z\tbusy cores mean 11 of 14\n")
    return res


def test_stage_verify_part_keeps_the_report_whole_and_caps_logs(tmp_path, monkeypatch):
    monkeypatch.setenv("MOJOLEARN_PERSIST_TAG", "idrec-x")
    res = verify_results(tmp_path)
    m = E.stage(res, res / "evidence/nvidia/verify", "nvidia", "verify", commit=COMMIT)
    out = res / "evidence/nvidia/verify"
    assert json.loads((out / "verify-all.json").read_text())["verdict"] == "VERIFIED"
    assert json.loads((out / "extended/verify-graph.json").read_text())["verdict"] == "OWED"
    log = m["logs"][0]
    assert log["file"] == "logs/verify.log.txt" and log["kept"] == E.LOG_CAP_BYTES
    assert log["full"] == "r2:idrec-x/logs/verify.log"
    assert m["steps"][0]["seconds"] == 42 and m["busy_cores"]["verify"]["max"] == 12
    assert m["version"] == "0.8.67" and not list(out.rglob("*.log"))
    assert json.loads((out / "what-ran.json").read_text())["paths_sample"] == [dict(lane="ols")]


def test_stage_proving_and_smoke_summaries(tmp_path):
    res = job(tmp_path, "proving")
    (res / "logs" / "proving.log").write_text(
        "FAILED tests/proving/test_a.py::test_x - boom\nERROR tests/proving/test_b.py::test_y\n"
        "== 2 failed, 40 passed, 1 error in 9.0s ==\n")
    m = E.stage(res, res / "evidence/nvidia/proving", "nvidia", "proving", commit=COMMIT)
    p = m["proving"]["proving"]
    assert p["counts"] == dict(failed=2, passed=40, error=1)
    assert p["failed"] == ["tests/proving/test_a.py::test_x", "tests/proving/test_b.py::test_y"]
    res = job(tmp_path, "smoke")
    (res / "smoke" / "lanes").mkdir(parents=True)
    (res / "smoke" / "SUMMARY.txt").write_text("lanes rc=1 1 failed\noverall rc=1 failed: lanes\n")
    (res / "smoke" / "lanes" / "results.tsv").write_text(
        "unit\trc\tstatus\tseconds\tpeak_rss_mb\tlog\nols\t0\tok\t1\t1\tl\nrf\t-11\tcrash\t1\t1\tl\n")
    m = E.stage(res, res / "evidence/nvidia/smoke", "nvidia", "smoke", commit=COMMIT)
    assert m["smoke"]["failed"] == {"lanes": [dict(unit="rf", rc="-11", status="crash")]}
    assert (res / "evidence/nvidia/smoke/SUMMARY.txt").is_file()


def git_tree(tmp_path):
    tree = tmp_path / "repo"
    tree.mkdir()
    for args in (["init", "-q"], ["config", "user.email", "t@t"], ["config", "user.name", "t"],
                 ["commit", "-q", "--allow-empty", "-m", "root"]):
        subprocess.run(["git", "-C", str(tree), *args], check=True)
    return tree


def test_install_overwrites_per_column_and_commits(tmp_path):
    tree = git_tree(tmp_path)
    res = verify_results(tmp_path)
    E.stage(res, res / "evidence/nvidia/verify", "nvidia", "verify", commit=COMMIT)
    cpu = job(tmp_path, "cpu")
    E.stage(cpu, cpu / "evidence/cpu/proving", "cpu", "proving", commit=COMMIT)
    assert E.install([res, cpu], tree=tree, commit=True, say=lambda m: None) == ["cpu", "nvidia"]
    latest = tree / "verification/latest"
    assert json.loads((latest / "nvidia/MANIFEST.json").read_text())["commit"] == COMMIT
    # the same commit's smoke joins the column beside the verify part
    sm = job(tmp_path, "smoke")
    E.stage(sm, sm / "evidence/nvidia/smoke", "nvidia", "smoke", commit=COMMIT)
    E.install([sm], tree=tree, say=lambda m: None)
    assert (latest / "nvidia/verify").is_dir() and (latest / "nvidia/smoke").is_dir()
    # a new commit replaces the nvidia column whole and leaves cpu alone
    E.stage(sm, sm / "evidence/nvidia/smoke", "nvidia", "smoke", commit="b" * 40)
    E.install([sm], tree=tree, say=lambda m: None)
    assert not (latest / "nvidia/verify").exists() and (latest / "nvidia/smoke").is_dir()
    assert (latest / "cpu/proving").is_dir()
    log = subprocess.run(["git", "-C", str(tree), "log", "--oneline"], capture_output=True, text=True).stdout
    assert "Verification evidence of aaaaaaaaaaaa" in log


def test_parts_of_two_commits_for_one_column_are_refused(tmp_path):
    a, b = job(tmp_path, "a"), job(tmp_path, "b")
    E.stage(a, a / "evidence/amd/record", "amd", "record", commit=COMMIT)
    E.stage(b, b / "evidence/amd/proving", "amd", "proving", commit="c" * 40)
    with pytest.raises(SystemExit, match="different commits"):
        E.install([a, b], tree=git_tree(tmp_path), say=lambda m: None)
