"""Tooling test for tools/ops/native_checks_report.py (sharding, summary, report) and the native-checks workflow."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("native_checks_report", ROOT / "tools/ops/native_checks_report.py")
ncr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ncr)


def _write(d: Path, task: str, rc: int, error: str = "", rss: int | None = None) -> None:
    d.mkdir(parents=True, exist_ok=True)
    head = f"task: {task}\nrc: {rc}\nseconds: 10\n" + (f"maxrss_kb: {rss}\n" if rss else "")
    (d / f"{task}.txt").write_text(head + "--- first error ---\n" + error)


def test_select_and_list_from_pixi_toml(tmp_path):
    toml = tmp_path / "pixi.toml"
    toml.write_text(
        '[tasks]\n"check-native-a-cpu" = "mojo run a.mojo"\n"check-native-a-gpu" = "mojo run b.mojo"\n'
        'other = "x"\n[feature.test.tasks]\n"check-native-z-cpu" = { cmd = "mojo run z.mojo" }\n'
    )
    tasks = ncr.load_tasks(toml)
    assert ncr.select(tasks, "check-native-*-cpu") == ["check-native-a-cpu", "check-native-z-cpu"]
    assert ncr.select(tasks, "check-native-%-gpu,other") == ["check-native-a-gpu", "other"]


def test_repo_has_native_check_tasks():
    tasks = ncr.load_tasks(ROOT / "pixi.toml")
    assert len(ncr.select(tasks, "check-native-*-cpu")) > 0
    assert len(ncr.select(tasks, "check-native-*-gpu")) > 0


def test_shard_sizes_cover_and_stable():
    tasks = [f"check-native-t{i}-cpu" for i in range(106)]
    shards = ncr.shard(tasks)
    assert len(shards) == 9  # ceil(106 / 12)
    flat = [t for s in shards for t in s]
    assert sorted(flat) == sorted(tasks) and len(flat) == len(set(flat))
    sizes = [len(s) for s in shards]
    assert max(sizes) - min(sizes) <= 1
    assert ncr.shard(list(reversed(tasks))) == shards  # independent of input order


def test_shard_cap_and_empty():
    assert len(ncr.shard([f"t{i}" for i in range(1000)])) == 20
    assert ncr.shard([]) == []
    assert ncr.shard_count(1) == 1 and ncr.shard_count(12) == 1 and ncr.shard_count(13) == 2


def test_plan_matrix_json(tmp_path, capsys):
    toml = tmp_path / "pixi.toml"
    toml.write_text("[tasks]\n" + "".join(f'"check-native-t{i}-cpu" = "x"\n' for i in range(25)))
    planned = tmp_path / "planned.txt"
    assert ncr.main(["plan", "--pixi-toml", str(toml), "--tasks-out", str(planned)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["count"] == 25 and plan["shards"] == 3
    names = [t for row in plan["matrix"]["include"] for t in row["tasks"].split()]
    assert sorted(names) == planned.read_text().split()


def test_summary_counts_failures_and_missing(tmp_path):
    res = tmp_path / "results"
    _write(res, "check-native-ok-cpu", 0)
    _write(res, "check-native-bad-cpu", 1, "x.mojo:3:5: error: use of unknown declaration 'foo'\n")
    _write(res, "check-native-slow-cpu", 124, "")
    planned = ["check-native-ok-cpu", "check-native-bad-cpu", "check-native-slow-cpu", "check-native-gone-cpu"]
    text, n_bad = ncr.summarize(ncr.read_results(tmp_path), planned, ["host: test"])
    assert n_bad == 3
    assert "passed: 1  failed: 2  missing: 1" in text
    assert "| check-native-gone-cpu | MISSING |" in text
    assert "error: use of unknown declaration 'foo'" in text
    assert "TIMEOUT" in text and "- host: test" in text
    # failures come first in the table
    rows = [ln for ln in text.splitlines() if ln.startswith("| check-native-")]
    assert rows[-2].startswith("| check-native-ok-cpu") and rows[-1].startswith("| check-native-gone-cpu")


def test_summary_command_exit_status(tmp_path):
    res = tmp_path / "results"
    _write(res, "a", 0, rss=2048)
    out = tmp_path / "SUMMARY.md"
    assert ncr.main(["summary", str(res), "--out", str(out)]) == 0
    _write(res, "b", 1, "error: boom\n")
    assert ncr.main(["summary", str(res), "--out", str(out)]) == 1
    assert ncr.parse_result(res / "a.txt")["maxrss_kb"] == 2048


def test_report_groups_by_normalized_first_error(tmp_path):
    res = tmp_path / "results"
    _write(res, "t1", 1, "ctx\na/x.mojo:10:3: error: no matching function in call to 'f'\nmore\n")
    _write(res, "t2", 1, "b/y.mojo:99:1: error: no matching function in call to 'f'\n")
    _write(res, "t3", 1, "z.mojo:1:1: error: cannot implicitly convert 'Int' value\n")
    _write(res, "t4", 0)
    text = ncr.report(ncr.read_results(tmp_path))
    assert text.startswith("3 of 4 tasks failed, 2 distinct first errors")
    first_group = text.split("## ")[1]
    assert first_group.startswith("2 x <loc>: error: no matching function in call to 'f'")
    assert "t1" in first_group and "t2" in first_group and "t3" not in first_group
    assert ncr.report(ncr.read_results(tmp_path / "missing")) == "all 0 tasks passed\n"


def test_workflow_shape():
    wf = yaml.safe_load((ROOT / ".github/workflows/native-checks.yml").read_text())
    on = wf.get("on", wf.get(True))
    assert set(on) == {"workflow_dispatch"}  # manual only (light-checks.yml)
    inputs = on["workflow_dispatch"]["inputs"]
    assert inputs["filter"]["default"] == "check-native-*-cpu" and "ref" in inputs
    assert set(wf["jobs"]) == {"plan", "check", "summary"}
    run = "\n".join(s.get("run", "") for s in wf["jobs"]["check"]["steps"])
    assert "set +e" in run and 'pixi run "$task"' in run and "grep -m5 -B2 -A8 'error:'" in run
    assert wf["jobs"]["check"]["strategy"]["fail-fast"] is False
