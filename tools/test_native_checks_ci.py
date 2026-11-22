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


def _suites(root: Path, checks: dict[str, tuple[bool, bool]]) -> None:
    nt = root / "checks/native_tests"
    nt.mkdir(parents=True, exist_ok=True)
    for n, (cpu, gpu) in checks.items():
        cols = " ".join(c for c, on in (("cpu", cpu), ("gpu", gpu)) if on)
        (nt / f"{n}.mojo").write_text(f"# SPDX-License-Identifier: Apache-2.0\n# native-check: {cols}\n")


def test_select_and_list(tmp_path, capsys):
    _suites(tmp_path, {"a_both": (True, True), "b_host": (True, False), "c_device": (False, True)})
    checks = ncr.load_checks(tmp_path)
    assert checks == {"a_both": (True, True), "b_host": (True, False), "c_device": (False, True)}
    assert ncr.column_names(checks, "cpu") == ["a_both", "b_host"]
    assert ncr.column_names(checks, "gpu") == ["a_both", "c_device"]
    assert ncr.select(ncr.column_names(checks, "gpu"), "c_%,zzz") == ["c_device"]
    assert ncr.main(["list", "--root", str(tmp_path), "--column", "cpu", "--filter", "b_*"]) == 0
    assert capsys.readouterr().out.split() == ["b_host"]


def test_repo_checks_are_testsuite_files():
    # lane ts-build (2026-11-22): every check is a TestSuite file; the registry, its runner and the two mains are gone
    checks = ncr.load_checks(ROOT)
    assert len(ncr.column_names(checks, "cpu")) > 0
    assert len(ncr.column_names(checks, "gpu")) > 0
    assert checks == ncr.testsuite_checks(ROOT)
    nt = ROOT / "checks/native_tests"
    for gone in ("registry", "runner", "run_cpu", "run_gpu", "testsuite_serial"):
        assert not (nt / f"{gone}.mojo").exists(), f"{gone}.mojo"
    with_register = [p.name for d in (nt, ROOT / "checks") for p in d.glob("*.mojo")
                     if "def register() -> Check:" in p.read_text()]
    assert with_register == []


def test_pixi_has_two_column_tasks_and_no_per_check_tasks():
    import tomllib

    tasks = tomllib.loads((ROOT / "pixi.toml").read_text())["tasks"]
    assert tasks["check-native-cpu"] == "bash tools/ops/native_testsuite.sh one cpu"
    assert tasks["check-native-gpu"] == "bash tools/ops/native_testsuite.sh one gpu"
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
    _suites(tmp_path, {f"t{i}": (True, i % 2 == 0) for i in range(60)})
    planned = tmp_path / "planned.txt"
    assert ncr.main(["plan", "--root", str(tmp_path), "--tasks-out", str(planned)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["count"] == 60 and plan["shards"] == 3  # ceil(60 / 25)
    names = [t for row in plan["matrix"]["include"] for t in row["tasks"].split()]
    assert sorted(names) == planned.read_text().split()
    assert ncr.main(["plan", "--root", str(tmp_path), "--column", "gpu"]) == 0
    assert json.loads(capsys.readouterr().out)["count"] == 30


def test_testsuite_files_are_checks_by_their_marker(tmp_path, capsys):
    nt = tmp_path / "checks/native_tests"
    nt.mkdir(parents=True)
    (nt / "ts_both.mojo").write_text("# SPDX-License-Identifier: Apache-2.0\n# native-check: cpu gpu\ndef main(): pass\n")
    (nt / "ts_host.mojo").write_text("# SPDX-License-Identifier: Apache-2.0\n# native-check: cpu\n")
    (nt / "helper.mojo").write_text("# SPDX-License-Identifier: Apache-2.0\n# native-check is not on this line\n")
    (nt / "late.mojo").write_text("# SPDX\n\n# native-check: cpu\n")  # only the second line marks a check
    checks = ncr.load_checks(tmp_path)
    assert checks == {"ts_both": (True, True), "ts_host": (True, False)}
    assert ncr.main(["testsuite", "--root", str(tmp_path), "--column", "gpu"]) == 0
    assert capsys.readouterr().out.split() == ["ts_both", "checks/native_tests/ts_both.mojo"]


def test_testsuite_files_at_the_top_of_checks(tmp_path, capsys):
    # lane ts-b9 (2026-11-22): the standalone checks at the top of checks/ are TestSuite files in place
    (tmp_path / "checks/native_tests").mkdir(parents=True)
    (tmp_path / "checks/top_gpu.mojo").write_text("# SPDX-License-Identifier: Apache-2.0\n# native-check: gpu\n")
    assert ncr.load_checks(tmp_path) == {"top_gpu": (False, True)}
    assert ncr.main(["testsuite", "--root", str(tmp_path), "--column", "gpu"]) == 0
    assert capsys.readouterr().out.split() == ["top_gpu", "checks/top_gpu.mojo"]
    (tmp_path / "checks/native_tests/top_gpu.mojo").write_text("# SPDX\n# native-check: cpu\n")
    try:
        ncr.load_checks(tmp_path)
        raise AssertionError("one name as a TestSuite file in two directories was accepted")
    except ValueError as e:
        assert "both" in str(e)


def test_result_writer_reads_the_testsuite_summary_and_a_crash(tmp_path):
    log = tmp_path / "t.log"
    log.write_text("Running 3 tests for x.mojo\n    PASS [ 0.001 ] test_a\n    FAIL [ 0.010 ] test_b\n"
                   "      At x.mojo:13:17: AssertionError: `left == right` comparison failed:\n"
                   "--------\nSummary [ 0.010 ] 3 tests run: 1 passed , 1 failed , 1 skipped \n")
    out = tmp_path / "results"
    assert ncr.main(["result", "ts", "--log", str(log), "--rc", "1", "--seconds", "2", "--out-dir", str(out)]) == 0
    r = ncr.parse_result(out / "ts.txt")
    assert (r["cases"], r["failed_cases"], r["skipped"]) == (3, 1, 1)
    assert r["first_error"].startswith("FAIL [ 0.010 ] test_b")
    clog = tmp_path / "c.log"
    clog.write_text("Stack dump without symbol names\n--- crash localisation ---\nalone test_a rc=0: PASS\n"
                    "CRASHED test_segv rc=139 (alone, no summary)\n")
    assert ncr.main(["result", "tc", "--log", str(clog), "--rc", "139", "--seconds", "2", "--out-dir", str(out)]) == 0
    c = ncr.parse_result(out / "tc.txt")
    assert c["cases"] is None and c["rc"] == 139 and c["first_error"].startswith("CRASHED test_segv")


def test_result_writer_reads_a_compile_error(tmp_path):
    out = tmp_path / "results"
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
    assert set(wf["jobs"]) == {"plan", "host-parse", "device-parse", "apple-parse", "device-parse-triage",
                               "device-parse-retry", "device-parse-verdict", "build", "check", "summary"}
    assert inputs["suite"]["default"] == "all"
    # every job body is a step of tools/ops/native_checks_steps.sh (lane fx-cpu-box-checks, 2026-11-19), checked out from
    # the workflow's own commit at .wf/ so the workflow on main runs on any older ref; the CPU box runs the same steps
    steps_sh = (ROOT / "tools/ops/native_checks_steps.sh").read_text()
    body = lambda name: steps_sh[steps_sh.index(f"step_{name}() {{"):].split("\n}\n", 1)[0]

    def runs(job, step):
        st = [s for s in wf["jobs"][job]["steps"] if s.get("run", "").strip() == f"bash .wf/tools/ops/native_checks_steps.sh {step}"]
        assert len(st) == 1, (job, step)
        wf_co = [s for s in wf["jobs"][job]["steps"] if s.get("with", {}).get("path") == ".wf"]
        assert wf_co and wf_co[0]["with"]["ref"] == "${{ github.workflow_sha }}", job
        return st[0]
    runs("plan", "plan")
    plan = body("plan")
    assert "merged_bindings.py plan --dir host --platform linux --byte-lm 0 --layout release" in plan
    # one parse script for every parse job (lane fx-trees-device-parse, 2026-11-19)
    assert "PARSE_GROUP_SH" not in wf.get("env", {})
    script = body("parse_group")
    assert "--emit llvm --no-optimization" in script and "MOJOLEARN_COLUMN_CPU" in script  # parse only, no codegen
    assert "merged_bindings.py root" in script and wf["jobs"]["host-parse"]["strategy"]["fail-fast"] is False
    # a limit, never a lowered -j (`limited` is `timeout -k 60`, or the same rule in Python on macOS, which has none)
    assert 'limited "$limit" pixi run mojo build -j "$cores"' in script and 'timeout -k 60 "$@"' in script
    assert "cores=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN)" in script
    assert "memory.peak" in script and 'result-$KEY.txt' in script
    for job in ("host-parse", "device-parse", "device-parse-retry"):
        step = runs(job, "parse-group")
        assert step["timeout-minutes"] <= 30 and step["env"]["KIND"] == ("host" if job == "host-parse" else "device")
        upload = wf["jobs"][job]["steps"][-1]
        assert upload["if"] == "always()" and "out/*.txt" in upload["with"]["path"]
    assert wf["jobs"]["device-parse"]["continue-on-error"] is True  # the verdict decides from the last attempt
    assert "PARSE_SPLIT" in plan and "_mojolearn_native_trees" in plan
    assert wf["jobs"]["host-parse"]["name"].startswith("host-parse")  # tools/prebuild_check.py reads job names
    # device-parse (lane fx-build-tools, 2026-11-18): the identical device groups with the sm_89 column's flags
    assert "merged_bindings.py plan --dir identical --platform linux --byte-lm 1 --layout release" in plan
    assert "--target-accelerator sm_89" in script
    assert "MOJOLEARN_COLUMN_NVIDIA" in script and "MOJOLEARN_NUMERIC_IDENTICAL=1" in script
    assert wf["jobs"]["device-parse"]["name"].startswith("device-parse")  # tools/prebuild_check.py reads job names
    assert wf["jobs"]["device-parse-retry"]["name"].startswith("device-parse retry ")
    assert not wf["jobs"]["device-parse-verdict"]["name"].startswith(("host-parse", "device-parse"))
    # apple-parse (lane fx-apple-fast-ci, 2026-11-20): the Apple sets cross-elaborated on Linux; its own suite, never `all`
    step = runs("apple-parse", "parse-group")
    assert step["env"]["KIND"] == "${{ matrix.kind }}" and step["env"]["CLS"] == "${{ matrix.cls }}"
    assert wf["jobs"]["apple-parse"]["runs-on"] == "ubuntu-latest"  # no job of this workflow runs on a Mac
    assert all("macos" not in str(job.get("runs-on")) for job in wf["jobs"].values())
    assert wf["jobs"]["apple-parse"]["strategy"]["fail-fast"] is False
    assert 'if [ "$IN_SUITE" = apple-parse ]; then' in plan and "apple_fast_check.py\" plan" in plan
    assert "--target-triple arm64-apple-macosx11.0 --target-cpu apple-m1" in script
    assert 'apple-fast) flags="$accel_flags" ;;' in script  # the legacy fast set: no MOJOLEARN_NUMERIC_IDENTICAL
    assert 'metal1) accel_flags="--target-accelerator metal:1" ;;' in script
    runs("build", "build"), runs("check", "shard"), runs("summary", "summary"), runs("device-parse-verdict", "verdict")
    build = body("build")
    # the column's TestSuite group programs (lane ts-build): their defines and target come from native_testsuite.sh
    assert 'native_testsuite.sh build "$IN_COLUMN"' in build and "native_checks_report.py result" in build
    ts = (ROOT / "tools/ops/native_testsuite.sh").read_text()
    assert "MOJOLEARN_COLUMN_CPU=1" in ts and "--target-cpu x86-64-v3" in ts and "fixture_packs.py unpack" in ts
    run = body("shard")
    assert "set +e" in run and 'native_testsuite.sh run "$IN_COLUMN"' in run
    assert "mojo build" not in run and "mojo run" not in run  # the shards never compile
    assert wf["jobs"]["check"]["needs"] == ["plan", "build"]
    assert wf["jobs"]["check"]["strategy"]["fail-fast"] is False


def test_gpu_job_uses_every_core_unless_memory_says_fewer(tmp_path):
    # lane fx-pod-job (2026-11-18): K was cores/4 with no memory reason (3.6 of 14 cores busy), and the summary's
    # "K memory alone allows" divided by the build's 43 GB peak RSS (results/build-gpu.txt), not a check's 1.64 GB
    import re
    import subprocess
    text = (ROOT / "tools/ops/native_checks_gpu_job.sh").read_text()
    assert "cores / 4" not in text
    m = re.search(r"^check_kb=.*\n^bymem=.*\n^K=.*\n", text, re.M)
    assert m, "the K block"
    for mem_kb, cores, want in ((179 * 1048576, 14, 14), (8 * 1048576, 14, 3), (1048576, 14, 1)):
        script = f"mem_kb={mem_kb}; cores={cores}\n{m.group(0)}[ \"$K\" -lt 1 ] && K=1\necho K=$K\n"
        got = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)
        assert got.stdout.strip() == f"K={want}", (mem_kb, cores, got.stdout, got.stderr)
    peak = text[text.index("checkfiles=()"):text.index("busy=$(busy_of")]
    res = tmp_path / "res"
    _write(res / "results", "build-gpu", 0, rss=44141068)
    _write(res / "results", "hist2_check", 0, rss=1600000)
    _write(res / "results", "gbdt_resident_ns_predict", 0, rss=1719664)
    got = subprocess.run(["bash", "-c", f"RES={res}\n{peak}\necho $peak_kb"], capture_output=True, text=True, timeout=30)
    assert got.stdout.strip() == "1719664", got.stdout + got.stderr     # the largest check, never the build
    assert 'busy_sampler "$RES/busy_cores_build.txt"' in text and "BUSY build:" in text


def test_gpu_job_keeps_results_not_binaries():
    # 2026-11-21: the nvc-measure job persisted a 144 MB bin/run_gpu to R2. The binaries are built outside the results
    # dir (box_persist_run.sh tars only $RES); since lane ts-build they are the TestSuite group programs
    text = (ROOT / "tools/ops/native_checks_gpu_job.sh").read_text()
    assert "$RES/bin" not in text
    assert 'BIN_DIR=${NC_BIN_DIR:-$RES.work/bin}' in text and "run_gpu.mojo" not in text
    assert 'native_testsuite.sh build gpu "$BIN_DIR" "$RES"' in text
    assert '--one {} "$RES" "$BIN_DIR"' in text and 'native_testsuite.sh run gpu "$BIN_DIR" "$RES" "$name"' in text
    row = text.index('native_checks_report.py result build-gpu --log "$RES/logs/build-gpu.log"')
    assert text.index('native_testsuite.sh build gpu') < row

def test_prebuild_counts_a_parse_group_by_its_last_attempt():
    spec = importlib.util.spec_from_file_location("prebuild_check", ROOT / "tools/prebuild_check.py")
    pc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pc)
    jobs = [{"name": "host-parse _h", "conclusion": "success"},
            {"name": "device-parse _t_p1of2 sm_89", "conclusion": "failure"},
            {"name": "device-parse _t_p2of2 sm_89", "conclusion": "success"},
            {"name": "device-parse retry _t_p1of2 sm_89", "conclusion": "success"},
            {"name": "device-parse _n sm_89", "conclusion": "failure"},
            {"name": "parse-verdict device", "conclusion": "success"}]
    last = pc.parse_group_jobs(pc.host_parse_jobs({"jobs": jobs}))
    assert sorted(j["name"] for j in last) == ["device-parse _n sm_89", "device-parse _t_p2of2 sm_89",
                                               "device-parse retry _t_p1of2 sm_89", "host-parse _h"]
    assert [j["name"] for j in last if j["conclusion"] != "success"] == ["device-parse _n sm_89"]


def test_prebuild_ignores_the_skipped_retry_job_when_no_group_needed_one():
    # 2026-11-19: with no retry needed GitHub lists the unexpanded matrix job as skipped; counted as a group, it failed
    # the pre-build check of a run whose every parse job had passed
    spec = importlib.util.spec_from_file_location("prebuild_check", ROOT / "tools/prebuild_check.py")
    pc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pc)
    jobs = [{"name": "device-parse a sm_89", "conclusion": "success"},
            {"name": "device-parse retry ${{ matrix.key }} sm_89", "conclusion": "skipped"}]
    last = pc.parse_group_jobs(jobs)
    assert [j["name"] for j in last] == ["device-parse a sm_89"] and all(j["conclusion"] == "success" for j in last)


def _groups_module():
    spec = importlib.util.spec_from_file_location("native_testsuite_groups", ROOT / "tools/ops/native_testsuite_groups.py")
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    return g


def test_grouped_build_copy_lists_tests_by_name_and_keeps_listed_mains():
    # lane ts-build, 2026-11-22: discovery from an imported module's main() saw 0 tests, so a member copy lists them
    g = _groups_module()
    src = ("# x\nfrom std.testing import TestSuite\nfrom m.h import helper, test_shared\n\ndef test_b() raises:\n"
           "    pass\n\ndef _x():\n    pass\n\ndef test_a() raises:\n    pass\n\ndef main() raises:\n"
           "    TestSuite.discover_tests[__functions_in_module()]().run()\n")
    out, why = g.copy_source(src)
    assert why == "" and "__functions_in_module" not in out
    assert out.index("suite.test[test_shared]()") < out.index("suite.test[test_b]()") < out.index("suite.test[test_a]()")
    listed = "def test_a() raises:\n    pass\n\ndef main() raises:\n    var suite = TestSuite()\n    suite.test[test_a]()\n    suite^.run()\n"
    assert g.copy_source(listed) == (listed, "")
    assert g.copy_source("def test_a() raises:\n    pass\n")[0] is None


def test_grouped_build_deal_is_deterministic_balanced_and_keeps_families():
    g = _groups_module()
    names = [f"fam{f}_{i}" for f in range(6) for i in range(5)] + ["big_0", "big_1", "big_2", "big_3"]
    cost = {n: 1.0 for n in names} | {"big_0": 9.0, "big_1": 9.0, "big_2": 9.0, "big_3": 9.0}
    a = g.deal(names, cost, 4)
    assert a == g.deal(list(reversed(names)), cost, 4)
    assert sorted(n for grp in a for n in grp) == sorted(names)
    loads = [sum(cost[n] for n in grp) for grp in a]
    assert max(loads) - min(loads) <= 9.0
    for f in range(6):  # a small family never splits
        assert sum(1 for grp in a if any(n.startswith(f"fam{f}_") for n in grp)) == 1
    assert len(g.deal(["one"], {"one": 1.0}, 8)) == 1
