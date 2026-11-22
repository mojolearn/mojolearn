# SPDX-License-Identifier: Apache-2.0
"""tools/dev/multi_gpu_checks.py without a device: that it is OPTIONAL (no release, gate, admission, workflow, hook
or wheel file names it), that the retired par-* names left the shipped verify table for the suite's own table, the
unit list, and the fail-closed verdicts including the fault arm. The drivers themselves run only on a box with two
or more GPUs (tools/ops/multi_gpu_checks_job.sh, by hand)."""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "dev"))
import multi_gpu_checks as MG  # noqa: E402

#: the names no required module may carry: the suite, its job script, its summary and its retired-cells table
TOKENS = ("multi_gpu_checks", "multi_gpu_retired", "MULTI_GPU_CHECKS")
#: the only files that may name the suite
SUITE_FILES = {
    "tools/dev/multi_gpu_checks.py",
    "tools/dev/multi_gpu_retired_par_cells.json",
    "tools/ops/multi_gpu_checks_job.sh",
    "tools/test_multi_gpu_checks_optional.py",
    "docs/dev/MULTI_GPU_CHECKS.md",
    # a docstring that says where the retired cells went; it reads no suite file
    "tools/lane_accounting.py",
}
#: scanned whole: everything a release, a gate, an admission, a merge, a push or a wheel is made of
SCANNED = (("tools", ("*.py", "*.sh", "*.json", "*.toml", "*.yml", "*.yaml", "*.tsv", "lq", "pre-commit", "pre-push")),
           (".github", ("*.yml", "*.yaml", "*.sh", "*.py")),
           ("python", ("*.py", "*.toml", "*.cfg", "*.in")),
           ("packaging", ("*.py", "*.sh", "*.toml", "*.json")),
           ("bindings", ("*.sh", "*.json")))
#: the rule files (AGENTS.md, CLAUDE.md) are not scanned: a rule sentence may name the suite to say it is optional
ROOT_FILES = ("pixi.toml", "pyproject.toml", "Makefile")


def naming_files():
    """Every scanned file whose text names the suite, as repo-relative paths. One pass, one process."""
    found = set()
    paths = [ROOT / name for name in ROOT_FILES]
    for top, patterns in SCANNED:
        base = ROOT / top
        if not base.is_dir():
            continue
        for pattern in patterns:
            paths.extend(base.rglob(pattern))
    for path in paths:
        if not path.is_file() or "__pycache__" in path.parts or ".pixi" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if any(token in text for token in TOKENS):
            found.add(path.relative_to(ROOT).as_posix())
    return found


def test_no_release_gate_admission_workflow_hook_or_wheel_file_names_the_suite():
    """NOT REQUIRED ANYWHERE: the only files that name the suite are the suite's own. A release or gate module
    (tools/release*.py, tools/ops/release_gate*), an admission tool, a CI workflow, a push hook, the tool test
    runner or anything under python/ that imports or names it fails here."""
    extra = naming_files() - SUITE_FILES
    assert not extra, "the optional multi-GPU suite is named by: %s" % sorted(extra)


def test_release_py_and_the_gate_job_exist_and_were_scanned():
    """The scan above is only a guard if the release tool and the gate job are among the scanned files."""
    for path in ("tools/release.py", "tools/ops/release_gate_job.sh", "tools/admit_identity_columns.sh",
                 "tools/hooks/pre-push", "tools/run_tool_tests.sh"):
        text = (ROOT / path).read_text(encoding="utf-8", errors="ignore")
        assert not any(token in text for token in TOKENS), path
    workflows = sorted((ROOT / ".github" / "workflows").glob("*.y*ml"))
    assert workflows, "no workflow was scanned"


def test_nothing_under_the_package_and_no_verify_option():
    """Not in the wheel, not in the user command: the suite lives under tools/ (never packaged: python/pyproject.toml
    packages `mojolearn*` from python/), and `python -m mojolearn verify` has no multi-GPU option."""
    for name in ("multi_gpu_checks.py", "multi_gpu_retired_par_cells.json"):
        assert not list((ROOT / "python").rglob(name))
        assert (ROOT / "tools" / "dev" / name).is_file()
    main = (ROOT / "python/mojolearn/__main__.py").read_text()
    for option in ('"--par"', "'--par'", '"--par-devices"', "'--par-devices'", "--multi-gpu", "--multigpu"):
        assert option not in main, option


def test_the_shipped_verify_table_carries_no_par_lane_and_the_suite_table_holds_them():
    """The 59 retired par-* names (531 cells, 5 lane revisions) moved from the shipped reference table to the
    suite's own repo-only table; the shipped table's other cells are not this test's business."""
    shipped = json.loads((ROOT / "python/mojolearn/verify_reference/table.json").read_text())
    for field in ("cells", "lane_revisions", "batch_revisions", "lane_admission"):
        assert not [k for k in shipped.get(field, {}) if k.startswith("par-")], field
    assert len(shipped["cells"]) >= 4608
    retired = json.loads(MG.RETIRED_TABLE.read_text())
    assert retired["format"] == "mojolearn.multi-gpu-retired-par-cells.v1"
    assert len(retired["cells"]) == 531 and len(retired["lanes"]) == 59
    assert sorted({k.split("/")[0] for k in retired["cells"]}) == retired["lanes"]
    assert all(k.startswith("par-") for k in list(retired["cells"]) + list(retired["lane_revisions"]))
    assert sorted(MG.RETIRED_PAR_LANES) == retired["lanes"]


def test_the_unit_list_is_well_formed_and_every_retired_lane_has_a_unit_or_a_reason():
    assert MG.unit_errors() == []
    names = MG.unit_names()
    assert len(names) == len(set(names)) >= 40
    covered = {t for t in MG.RETIRED_PAR_LANES.values() if not t.startswith("not covered: ")}
    assert covered <= set(names)
    # every driver module of the multi-GPU capability has at least one unit
    modules = {row[1] for row in MG.UNITS}
    for module in ("parallel_classical", "parallel_ensemble", "parallel_forecasting", "parallel_gaussian_process",
                   "parallel_graph", "parallel_ivf", "parallel_model_selection", "parallel_neighbors",
                   "parallel_neighbors_reference", "parallel_preprocessing", "parallel_training",
                   "_distributed_program"):
        assert module in modules, module
        assert (ROOT / "python/mojolearn" / (module + ".py")).is_file(), module
    # a malformed line is refused
    bad = MG.UNITS[:1] + (("x", "m", None, "plan", "shift", 10),)
    assert len(MG.unit_errors(bad)) >= 2


def part(name, sha):
    return dict(name=name, shape=[2], dtype="<f4", sha256=sha * 64)


def arm(name, which, devices, parts, ranks=None, error=None):
    return dict(unit=name, arm=which, devices=list(devices), parts=parts, ranks=ranks, error=error)


def arms(name, one="a", multi="a", fault="b", ranks=2, n=2):
    return dict(one=arm(name, "one", (1,), [part("p", one)]),
                multi=arm(name, "multi", range(n), [part("p", multi)], ranks=ranks),
                fault=arm(name, "fault", range(n), [part("p", fault)], ranks=ranks))


def test_verdicts_ok_differ_and_fail_closed():
    plan, none = "kmeans", "scaler-standard"
    assert MG.unit_row(plan)[3] == "plan" and MG.unit_row(none)[3] == "none"
    assert MG.unit_verdict(plan, arms(plan), 2)[0] == "OK"
    verdict, reason, digests = MG.unit_verdict(plan, arms(plan, multi="c"), 2)
    assert verdict == "DIFFER" and "'p'" in reason and digests["one"] != digests["multi"]
    # a missing arm, an arm that raised and an arm without parts are FAIL, never a skip
    missing = arms(plan)
    del missing["multi"]
    assert MG.unit_verdict(plan, missing, 2)[0] == "FAIL"
    raised = arms(plan)
    raised["multi"] = arm(plan, "multi", range(2), [], error="RuntimeError: device 1 refused")
    assert MG.unit_verdict(plan, raised, 2)[:2] == ("FAIL", "arm multi: RuntimeError: device 1 refused")
    empty = arms(plan)
    empty["one"]["parts"] = []
    assert MG.unit_verdict(plan, empty, 2)[0] == "FAIL"
    # the arms must have run on one and on N devices
    assert MG.unit_verdict(plan, arms(plan, n=2), 3)[0] == "FAIL"


def test_a_multi_arm_that_stayed_on_one_rank_is_a_fail_where_the_unit_has_a_witness():
    plan, seen, none = "kmeans", "hdbscan", "scaler-standard"
    assert MG.unit_row(seen)[3] == "seen"
    assert MG.unit_verdict(plan, arms(plan, ranks=1), 2)[0] == "FAIL"
    assert MG.unit_verdict(plan, arms(plan, ranks=None), 2)[0] == "FAIL"
    assert MG.unit_verdict(plan, arms(plan, ranks=3), 2)[0] == "FAIL"
    assert MG.unit_verdict(seen, arms(seen, ranks=1), 2)[0] == "OK"
    assert MG.unit_verdict(seen, arms(seen, ranks=1), 2, strict_spread=True)[0] == "FAIL"
    assert MG.unit_verdict(none, arms(none, ranks=None), 2)[0] == "OK"


def test_the_fault_arm_must_be_seen_so_every_check_is_shown_able_to_fail():
    for name in ("kmeans", "scaler-standard"):
        verdict, reason, _ = MG.unit_verdict(name, arms(name, fault="a"), 2)
        assert verdict == "FAIL" and "cannot fail" in reason
        assert MG.unit_verdict(name, arms(name, fault="a"), 2, want_fault=False)[0] == "OK"
        failed = arms(name)
        failed["fault"] = arm(name, "fault", range(2), [], error="timed out after 5s")
        assert MG.unit_verdict(name, failed, 2)[0] == "FAIL"
    # the native negative control is switched on for a shift unit's fault arm only
    assert MG.unit_row("scaler-standard")[4] == "shift" and MG.unit_row("kmeans")[4] == "input"
    env = MG.worker_env("fault", "shift")
    assert all(env[k] == v for k, v in MG.FAULT_ENV.items())
    for which, fault in (("one", "shift"), ("multi", "shift"), ("fault", "input")):
        env = MG.worker_env(which, fault)
        assert not any(k in env for k in MG.FAULT_ENV), (which, fault)
        assert env["MOJOLEARN_NUMERIC_MODE"] == "identical"
    assert sorted(MG.FAULT_ENV) == ["MOJOLEARN_HOST_ALLOW_SABOTAGE", "MOJOLEARN_PAR_DRIVER_SABOTAGE"]
    door = (ROOT / "bindings/parallel_shard_common_run.mojo").read_text()
    assert all(name in door for name in MG.FAULT_ENV), "the native negative control's switches moved"


def test_the_box_environment_never_reaches_a_worker(monkeypatch):
    monkeypatch.setenv("MOJOLEARN_GBDT_SHARD_ONE_DEVICE", "1")
    monkeypatch.setenv("MOJOLEARN_GP_DEVICE_COUNT", "1")
    monkeypatch.setenv("PYTHONPATH", "/somewhere")
    env = MG.worker_env("multi", "input")
    assert "MOJOLEARN_GBDT_SHARD_ONE_DEVICE" not in env and "MOJOLEARN_GP_DEVICE_COUNT" not in env
    assert "PYTHONPATH" not in env


def test_the_parallel_plan_uses_the_cores_within_the_memory_bounds():
    names = MG.unit_names()
    jobs = MG.jobs_for(names, (0, 1, 2))
    assert len(jobs) == 3 * len(names)
    ones = [devs for _, which, devs in jobs if which == "one"]
    assert all(len(d) == 1 for d in ones) and {d[0] for d in ones} == {0, 1, 2}
    assert all(devs == (0, 1, 2) for _, which, devs in jobs if which != "one")
    assert len(MG.jobs_for(names, (0, 1), want_fault=False)) == 2 * len(names)
    workers, bounds = MG.plan_workers(162, 64, 256.0, [45000, 45000], 2.0, 1.5)
    assert workers == 29 and bounds["gpu_memory"] == 29 and bounds["cores"] == 64
    assert MG.plan_workers(162, 64, 256.0, [], 2.0, 1.5)[0] == 64
    assert MG.plan_workers(162, 64, 16.0, [], 2.0, 1.5)[0] == 8
    assert MG.plan_workers(4, 64, None, [], 2.0, 1.5)[0] == 4
    assert MG.plan_workers(162, 64, 0.5, [100], 2.0, 1.5)[0] == 1
    assert MG.alloc_cores() >= 1


def test_main_refuses_one_gpu_and_unknown_units(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(MG, "detect_gpus", lambda: ("nvidia", [40000]))
    assert MG.main(["--out", str(tmp_path / "o")]) == 2
    assert "not applicable" in capsys.readouterr().out
    assert not (tmp_path / "o").exists()
    monkeypatch.setattr(MG, "detect_gpus", lambda: ("nvidia", [40000, 40000]))
    assert MG.main(["--out", str(tmp_path / "o"), "--units", "no-such-unit"]) == 2
    assert MG.main(["--list"]) == 0
    assert "kmeans" in capsys.readouterr().out


def test_main_runs_every_arm_at_once_and_judges_them(tmp_path, monkeypatch, capsys):
    """The controller with the arm runner replaced: a unit that matches, one that differs and one whose fault is
    not seen; the summary is not ok and names them."""
    monkeypatch.setattr(MG, "detect_gpus", lambda: ("nvidia", [40000, 40000]))
    calls = []

    def fake(python, name, which, devices, out, timeout):
        calls.append((name, which, tuple(devices)))
        sha = "a"
        if which == "fault" and name != "gmm":
            sha = "b"
        if which == "multi" and name == "dbscan":
            sha = "c"
        return arm(name, which, devices, [part("p", sha)], ranks=len(devices))

    monkeypatch.setattr(MG, "run_arm", fake)
    out = tmp_path / "o"
    assert MG.main(["--out", str(out), "--units", "kmeans,dbscan,gmm", "--jobs", "4"]) == 1
    assert len(calls) == 9 and {c[2] for c in calls if c[1] != "one"} == {(0, 1)}
    doc = json.loads((out / MG.SUMMARY).read_text())
    assert doc["schema"] == MG.SCHEMA and doc["optional"] is True and doc["required_by"] == [] and doc["ok"] is False
    assert {n: r["verdict"] for n, r in doc["units"].items()} == dict(kmeans="OK", dbscan="DIFFER", gmm="FAIL")
    assert len(MG.summary_errors(doc)) == 2
    printed = capsys.readouterr().out
    assert "MULTIGPU kmeans" in printed and "MULTIGPU summary 1/3 OK, not: dbscan gmm" in printed
    monkeypatch.setattr(MG, "run_arm", lambda python, name, which, devices, out, timeout:
                        arm(name, which, devices, [part("p", "b" if which == "fault" else "a")], ranks=len(devices)))
    assert MG.main(["--out", str(out), "--units", "kmeans,dbscan"]) == 0
    assert json.loads((out / MG.SUMMARY).read_text())["ok"] is True


def test_digests_cover_nested_and_ragged_outputs():
    np = __import__("pytest").importorskip("numpy")
    leaves = MG.flatten(np, "radius", ([np.zeros(2, np.float32), np.zeros(0, np.float32)],
                                       [np.arange(2), np.arange(0)]))
    assert [name for name, _ in leaves][:3] == ["radius (length)", "radius[0] (length)", "radius[0][0]"]
    assert len(leaves) == 7
    assert [n for n, _ in MG.flatten(np, "w", {"b": np.ones(1), "a": np.ones(1)})] == ["w.a", "w.b"]
    assert MG.flatten(np, "s", [1.0, 2.0])[0][1].shape == (2,)
    a = MG.part_digest(np, np.arange(6, dtype=np.float32).reshape(2, 3))
    b = MG.part_digest(np, np.arange(6, dtype=np.float32).reshape(3, 2))
    assert a["sha256"] == b["sha256"] and a["shape"] != b["shape"]
    one = [dict(name="p", **a)]
    assert MG.arm_digest(one) != MG.arm_digest([dict(name="p", **b)]) and MG.arm_digest([]) is None
    # the fault fixture differs from the clean one in every array a recipe reads, and only when asked
    clean, again, fault = MG.fixture(np, 7), MG.fixture(np, 7), MG.fixture(np, 7, fault=True)
    for key in ("X", "yc", "yr", "Xb", "yb"):
        assert clean[key].tobytes() == again[key].tobytes() and clean[key].tobytes() != fault[key].tobytes(), key
        assert clean[key].flags["C_CONTIGUOUS"]
    assert clean["X"].shape == (MG.ROWS, MG.FEATURES) and clean["X"].dtype == np.float32


def test_the_doc_carries_the_sentence_and_the_job_is_started_by_hand_only():
    doc = (ROOT / "docs/dev/MULTI_GPU_CHECKS.md").read_text()
    assert "Optional. Not required for any merge or release. Not shipped to users." in doc
    assert "Optional. Not required for any merge or release. Not shipped to users." in MG.__doc__
    job = (ROOT / "tools/ops/multi_gpu_checks_job.sh").read_text()
    assert "tools/dev/multi_gpu_checks.py" in job and "busy cores" in job and "alloc_cores" in job
    assert "MOJOLEARN_RESULTS_DIR" in job
    assert not (ROOT / "docs-site").joinpath("MULTI_GPU_CHECKS.md").exists()
