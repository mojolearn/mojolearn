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


def _registry(root: Path, checks: dict[str, tuple[bool, bool]]) -> None:
    nt = root / "checks/native_tests"
    nt.mkdir(parents=True, exist_ok=True)
    lines = [f"from checks.native_tests.{n} import register as register_{n}" for n in checks]
    (nt / "registry.mojo").write_text("\n".join(lines) + "\nfrom checks.native_tests.runner import Check\n")
    for n, (cpu, gpu) in checks.items():
        (nt / f"{n}.mojo").write_text(
            f'def register() -> Check:\n    return Check("{n}", cpu={cpu}, gpu={gpu}, serial="", run=run_cases)\n')


def test_select_and_list_from_registry(tmp_path, capsys):
    _registry(tmp_path, {"a_both": (True, True), "b_host": (True, False), "c_device": (False, True)})
    checks = ncr.load_checks(tmp_path)
    assert checks == {"a_both": (True, True), "b_host": (True, False), "c_device": (False, True)}
    assert ncr.column_names(checks, "cpu") == ["a_both", "b_host"]
    assert ncr.column_names(checks, "gpu") == ["a_both", "c_device"]
    assert ncr.select(ncr.column_names(checks, "gpu"), "c_%,zzz") == ["c_device"]
    assert ncr.main(["list", "--root", str(tmp_path), "--column", "cpu", "--filter", "b_*"]) == 0
    assert capsys.readouterr().out.split() == ["b_host"]


def test_repo_registry_matches_check_files():
    checks = ncr.load_checks(ROOT)
    assert len(ncr.column_names(checks, "cpu")) > 0
    assert len(ncr.column_names(checks, "gpu")) > 0
    nt = ROOT / "checks/native_tests"
    registered = set(ncr._REGISTERED.findall((nt / "registry.mojo").read_text()))
    # every module with a register() is in the registry, and each check is one file named after itself
    with_register = {p.stem for p in nt.glob("*.mojo") if "def register() -> Check:" in p.read_text()}
    assert with_register == registered
    assert set(checks) == registered
    # the five-file scaffolding is gone for every registered check
    for name in registered:
        for suffix in ("_cases", "_cpu_suite", "_gpu_suite", "_cpu", "_gpu"):
            assert not (nt / f"{name}{suffix}.mojo").exists(), f"{name}{suffix}.mojo"
    # the two mains import only the registry and the runner
    for col in ("cpu", "gpu"):
        imports = [ln for ln in (nt / f"run_{col}.mojo").read_text().splitlines() if ln.startswith(("from ", "import "))]
        assert all("native_tests.registry" in ln or "native_tests.runner" in ln for ln in imports), imports


def test_pixi_has_two_column_tasks_and_no_per_check_tasks():
    import tomllib

    tasks = tomllib.loads((ROOT / "pixi.toml").read_text())["tasks"]
    assert "run_cpu.mojo" in tasks["check-native-cpu"] and "-D MOJOLEARN_COLUMN_CPU=1" in tasks["check-native-cpu"]
    assert "run_gpu.mojo" in tasks["check-native-gpu"] and "MOJOLEARN_COLUMN_CPU" not in tasks["check-native-gpu"]
    names = ncr.load_checks(ROOT)
    per_check = [t for t in tasks if t.startswith("check-native-") and t.endswith(("-cpu", "-gpu"))
                 and t.removeprefix("check-native-")[:-4].replace("-", "_") in names]
    assert per_check == []


def test_shard_sizes_cover_and_stable():
    tasks = [f"t{i}" for i in range(106)]
    shards = ncr.shard(tasks)
    assert len(shards) == 5  # ceil(106 / 25)
    flat = [t for s in shards for t in s]
    assert sorted(flat) == sorted(tasks) and len(flat) == len(set(flat))
    sizes = [len(s) for s in shards]
    assert max(sizes) - min(sizes) <= 1
    assert ncr.shard(list(reversed(tasks))) == shards  # independent of input order


def test_shard_cap_and_empty():
    assert len(ncr.shard([f"t{i}" for i in range(1000)])) == 20
    assert ncr.shard([]) == []
    assert ncr.shard_count(1) == 1 and ncr.shard_count(25) == 1 and ncr.shard_count(26) == 2


def test_plan_matrix_json(tmp_path, capsys):
    _registry(tmp_path, {f"t{i}": (True, i % 2 == 0) for i in range(60)})
    planned = tmp_path / "planned.txt"
    assert ncr.main(["plan", "--root", str(tmp_path), "--tasks-out", str(planned)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["count"] == 60 and plan["shards"] == 3  # ceil(60 / 25)
    names = [t for row in plan["matrix"]["include"] for t in row["tasks"].split()]
    assert sorted(names) == planned.read_text().split()
    assert ncr.main(["plan", "--root", str(tmp_path), "--column", "gpu"]) == 0
    assert json.loads(capsys.readouterr().out)["count"] == 30


def test_result_writer_reads_the_check_line(tmp_path):
    log = tmp_path / "x.log"
    log.write_text("RUN spectral column=cpu\nFAIL case 3 boom\nCHECK spectral column=cpu cases=6 failed=1 skipped=0 41ms\n"
                   "CHECKS 1 column=cpu failed=1\n")
    out = tmp_path / "results"
    assert ncr.main(["result", "spectral", "--log", str(log), "--rc", "1", "--seconds", "3", "--out-dir", str(out)]) == 0
    r = ncr.parse_result(out / "spectral.txt")
    assert r["task"] == "spectral" and r["rc"] == 1 and r["cases"] == 6 and r["failed_cases"] == 1
    assert r["skipped"] == 0 and r["first_error"] == "FAIL case 3 boom"
    blog = tmp_path / "b.log"
    blog.write_text("x\ncheck.mojo:4:5: error: 'case' cannot be used as a function name\n")
    assert ncr.main(["result", "build-cpu", "--log", str(blog), "--rc", "1", "--seconds", "9", "--out-dir", str(out),
                     "--maxrss-kb", "2048"]) == 0
    b = ncr.parse_result(out / "build-cpu.txt")
    assert b["cases"] is None and b["maxrss_kb"] == 2048 and "error: 'case'" in b["first_error"]


def test_summary_counts_failures_and_missing(tmp_path):
    res = tmp_path / "results"
    _write(res, "check-native-ok-cpu", 0)
    _write(res, "check-native-bad-cpu", 1, "x.mojo:3:5: error: use of unknown declaration 'foo'\n")
    _write(res, "check-native-slow-cpu", 124, "")
    planned = ["check-native-ok-cpu", "check-native-bad-cpu", "check-native-slow-cpu", "check-native-gone-cpu"]
    text, n_bad = ncr.summarize(ncr.read_results(tmp_path), planned, ["host: test"])
    assert n_bad == 3
    assert "checks: 4  passed: 1  failed: 2  missing: 1" in text
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
    assert text.startswith("3 of 4 checks failed, 2 distinct first errors")
    first_group = text.split("## ")[1]
    assert first_group.startswith("2 x <loc>: error: no matching function in call to 'f'")
    assert "t1" in first_group and "t2" in first_group and "t3" not in first_group
    assert ncr.report(ncr.read_results(tmp_path / "missing")) == "all 0 checks passed\n"


def test_workflow_shape():
    wf = yaml.safe_load((ROOT / ".github/workflows/native-checks.yml").read_text())
    on = wf.get("on", wf.get(True))
    assert set(on) == {"workflow_dispatch"}  # manual only (light-checks.yml)
    inputs = on["workflow_dispatch"]["inputs"]
    assert inputs["filter"]["default"] == "*" and inputs["column"]["default"] == "cpu" and "ref" in inputs
    assert set(wf["jobs"]) == {"plan", "build", "check", "summary"}
    build = "\n".join(s.get("run", "") for s in wf["jobs"]["build"]["steps"])
    assert "mojo build" in build and "run_$IN_COLUMN.mojo" in build and "MOJOLEARN_COLUMN_CPU=1" in build
    run = "\n".join(s.get("run", "") for s in wf["jobs"]["check"]["steps"])
    assert "set +e" in run and 'pixi run "$bin" "$name"' in run and "native_checks_report.py result" in run
    assert "mojo build" not in run and "mojo run" not in run  # the shards never compile
    assert wf["jobs"]["check"]["needs"] == ["plan", "build"]
    assert wf["jobs"]["check"]["strategy"]["fail-fast"] is False
