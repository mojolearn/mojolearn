# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Algorithm-scoped tests must neither widen silently nor drop repeat checks."""
from types import SimpleNamespace
import pytest
import identity_break as ib
import identity_iterate as iterate
import lane_select
import verify_lanes


def test_named_algorithm_never_builds_whole_dependency_graph(monkeypatch):
    monkeypatch.setattr(lane_select, "lane_sources", lambda: pytest.fail("unneeded full dependency graph"))
    plan = iterate.plan([], "HEAD", ["base"], ["kmeans"])
    assert plan["lanes"] == ["kmeans"]
    assert not plan["fallback"]
    args = SimpleNamespace(all=False, lanes="", lane=["kmeans"])
    lanes, selection, _ = verify_lanes._selection(args)
    assert lanes == ["kmeans"]
    assert selection["mode"] == "named"


def test_named_algorithms_are_deduplicated_and_unknowns_refuse():
    assert iterate.plan([], "HEAD", ["base"], ["kmeans", "ridge", "kmeans"])["lanes"] == ["kmeans", "ridge"]
    with pytest.raises(ValueError, match="unknown lanes"):
        iterate.plan([], "HEAD", ["base"], ["typo"])
    with pytest.raises(ValueError, match="not both"):
        iterate.plan(["core/gemm.mojo"], "HEAD", ["base"], ["kmeans"])


def test_single_algorithm_plan_is_two_fits_not_eighteen(capsys):
    assert iterate.main(["--lane", "kmeans", "--plan"]) == 0
    import json
    plan = json.loads(capsys.readouterr().out)
    assert plan["lanes"] == ["kmeans"]
    assert plan["fixtures"] == ["base"]
    assert plan["cell_count"] == 1 and plan["fit_count"] == 2


def test_exhaustive_is_explicit_and_still_only_selected_algorithm(capsys):
    assert iterate.main(["--lane", "kmeans", "--exhaustive", "--plan"]) == 0
    import json
    plan = json.loads(capsys.readouterr().out)
    assert plan["lanes"] == ["kmeans"]
    assert plan["fixtures"] == ib.FIXTURES
    assert plan["fit_count"] == 2 * len(ib.FIXTURES)


def test_execution_runs_only_named_cells(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(iterate, "run_job", lambda cmd, env: calls.append(cmd) or 0)
    assert iterate.main(["--lane", "ridge", "--mode", "run", "--out", str(tmp_path)]) == 0
    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[cmd.index("--lanes") + 1] == "ridge"
    assert cmd[cmd.index("--fixtures") + 1] == "base"
    assert cmd[cmd.index("--repeats") + 1] == "1"


@pytest.mark.parametrize("extra,expected", [([], "base"), (["--exhaustive"], ",".join(ib.FIXTURES))])
def test_existing_runner_uses_same_explicit_fixture_policy(monkeypatch, capsys, extra, expected):
    monkeypatch.setattr(lane_select, "shard", lambda lanes, shards: ([lanes], [1]))
    assert verify_lanes.main(["--lane", "ridge", "--plan", *extra]) == 0
    out = capsys.readouterr().out
    assert f"--fixtures {expected}" in out


def test_invalid_scope_combinations_and_zero_repeats_refuse():
    with pytest.raises(SystemExit):
        verify_lanes.main(["--all", "--lane", "ridge", "--plan"])
    with pytest.raises(SystemExit):
        verify_lanes.main(["--lane", "ridge", "--repeats", "0", "--plan"])
    with pytest.raises(SystemExit):
        iterate.main(["--lane", "ridge", "--exhaustive", "--fixtures", "base", "--plan"])


def test_default_is_bounded_cpu_core_and_all_splits_groups(capsys):
    import json
    iterate.main(["--lane", "transformer", "--plan"])
    p = json.loads(capsys.readouterr().out)
    assert p["mode"] == "cpu"
    assert p["timeout"] == p["wait_timeout"] == 60
    assert [j["group"] for j in p["jobs"]] == ["core"]
    iterate.main(["--lane", "transformer", "--probe-group", "all", "--plan"])
    p = json.loads(capsys.readouterr().out)
    assert [j["group"] for j in p["jobs"]] == ["core", "batch", "rlpair"]
    assert p["job_count"] == 3 and p["fit_count"] == 6


def test_group_commands_keep_cpu_guard_and_timeouts(tmp_path):
    for group, skip_batch, skip_rl in (("core", True, True), ("batch", False, True), ("rlpair", True, False)):
        cmd = iterate.command("python", "transformer", "base", tmp_path / (group + ".json"),
                              60, "cpu", False, group)
        assert "--require-cpu" in cmd
        assert ("--no-batch" in cmd) == skip_batch
        assert ("--no-rlpair" in cmd) == skip_rl
        assert cmd[cmd.index("--timeout") + 1] == "60"
        assert cmd[cmd.index("--wait-timeout") + 1] == "60"
        assert "metal" not in cmd


def test_cpu_staging_excludes_gpu_binaries_and_rejects_contamination(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "__init__.py").write_text("# source")
    (source / "gpu.so").write_bytes(b"not loaded")
    host = tmp_path / "host"
    host.mkdir()
    (host / "_mojolearn_core_host.so").write_bytes(b"host")
    root, chosen = iterate.cpu_package(tmp_path / "out", host, source)
    assert chosen == host
    assert (root / "mojolearn" / "__init__.py").read_text() == "# source"
    assert not list(root.rglob("*.so"))
    assert (source / "gpu.so").read_bytes() == b"not loaded"
    (root / "mojolearn" / "rogue.so").write_bytes(b"bad")
    with pytest.raises(ValueError, match="native library"):
        iterate.cpu_package(tmp_path / "out", host, source)
    with pytest.raises(ValueError, match="no CPU host bindings"):
        iterate.cpu_package(tmp_path / "other", tmp_path / "missing", source)


def test_inapplicable_cpu_lane_refuses_before_staging_or_running(monkeypatch, tmp_path, capsys):
    import json
    monkeypatch.setattr(iterate, "cpu_package", lambda *a: pytest.fail("staged inapplicable job"))
    monkeypatch.setattr(iterate, "run_job", lambda *a: pytest.fail("ran inapplicable job"))
    iterate.main(["--lane", "par-forest", "--plan"])
    assert "par-forest" in json.loads(capsys.readouterr().out)["inapplicable"]
    with pytest.raises(SystemExit):
        iterate.main(["--lane", "par-forest", "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_all_does_not_schedule_inapplicable_batch_job(capsys):
    import json
    name = next(n for n in ib.LANES if isinstance(ib.BATCH.get(n), str) and n not in ib.RLPAIR)
    iterate.main(["--lane", name, "--probe-group", "all", "--plan"])
    assert [j["group"] for j in json.loads(capsys.readouterr().out)["jobs"]] == ["core"]
    with pytest.raises(SystemExit):
        iterate.main(["--lane", name, "--probe-group", "batch", "--plan"])


@pytest.mark.parametrize("flag", ["--timeout", "--wait-timeout", "--budget"])
@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_nonfinite_iteration_limits_refuse(flag, value):
    with pytest.raises(SystemExit):
        iterate.main(["--lane", "ridge", f"{flag}={value}", "--plan"])


def test_budget_stops_later_jobs_and_reports_unfinished(monkeypatch, tmp_path):
    import json
    clock = [100.0]
    monkeypatch.setattr(iterate.time, "monotonic", lambda: clock[0])
    calls = []
    def run(cmd, env):
        calls.append(cmd)
        assert float(cmd[cmd.index("--deadline") + 1]) == 101.0
        clock[0] = 102.0
        return 0
    monkeypatch.setattr(iterate, "run_job", run)
    assert iterate.main(["--lanes", "ridge,kmeans", "--mode", "run", "--budget", "1", "--out", str(tmp_path)]) == 124
    result = json.loads((tmp_path / "run-summary.json").read_text())
    assert len(calls) == 1
    assert not result["complete"] and result["status"] == "budget-exhausted"
    assert result["completed"][0]["lane"] == "ridge"
    assert result["pending"][0]["lane"] == "kmeans"


def test_failed_job_is_not_reported_as_completed(monkeypatch, tmp_path):
    import json
    monkeypatch.setattr(iterate, "run_job", lambda *a: 7)
    assert iterate.main(["--lanes", "ridge,kmeans", "--mode", "run", "--out", str(tmp_path)]) == 7
    result = json.loads((tmp_path / "run-summary.json").read_text())
    assert result["status"] == "failed" and not result["complete"]
    assert result["completed"] == []
    assert result["failed"]["exit_code"] == 7
    assert result["pending"][0]["lane"] == "kmeans"


def test_metal_plan_has_one_minute_total_budget(capsys):
    import json
    iterate.main(["--lane", "transformer", "--mode", "metal", "--plan"])
    plan = json.loads(capsys.readouterr().out)
    assert plan["budget"] == 60 and plan["job_count"] == 1
    iterate.main(["--lane", "transformer", "--mode", "metal", "--budget", "90", "--plan"])
    assert json.loads(capsys.readouterr().out)["budget"] == 90


@pytest.mark.parametrize("extra", [["--exhaustive"], ["--probe-group", "all"]])
def test_metal_runs_a_multi_job_round_without_any_marker(monkeypatch, tmp_path, extra):
    monkeypatch.delenv(ib.APPLE_RELEASE_RECORD_ENV, raising=False)
    calls = []
    monkeypatch.setattr(iterate, "run_job", lambda cmd, env: calls.append(cmd) or 0)
    assert iterate.main(["--lane", "transformer", "--mode", "metal", "--out", str(tmp_path), *extra]) == 0
    assert len(calls) > 1


def test_expanded_metal_diagnostic_is_explicit_and_still_budgeted(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(iterate, "run_job", lambda cmd, env: calls.append(cmd) or 0)
    assert iterate.main(["--lane", "transformer", "--mode", "metal", "--metal-diagnostic",
                         "--metal-expanded", "--probe-group", "all", "--out", str(tmp_path)]) == 0
    assert len(calls) == 3
    deadlines = {cmd[cmd.index("--deadline") + 1] for cmd in calls}
    assert len(deadlines) == 1


def test_apple_pass_is_metal_core_one_fit_three_fixtures(monkeypatch, capsys):
    monkeypatch.setattr(lane_select, "shard", lambda lanes, shards: ([lanes], [1]))
    assert verify_lanes.main(["--lanes", "ridge,ols", "--apple-pass", "--plan"]) == 0
    out = capsys.readouterr().out
    assert "backend=metal" in out and f"budget={verify_lanes.APPLE_PASS_BUDGET}" in out
    assert f"--fixtures {verify_lanes.APPLE_PASS_FIXTURES}" in out
    assert "--repeats 1" in out and "--no-batch" in out and "--no-rlpair" in out


@pytest.mark.parametrize("extra", [["--repeats", "2"], ["--probe-group", "all"], ["--exhaustive"],
                                   ["--backend", "cuda"]])
def test_apple_pass_refuses_anything_that_widens_it(extra):
    with pytest.raises(SystemExit):
        verify_lanes.main(["--lane", "ridge", "--apple-pass", "--plan", *extra])


def test_cpu_pass_is_local_cpu_with_the_apple_pass_cells(monkeypatch, capsys):
    monkeypatch.delenv("MAC_SLOTS", raising=False)
    assert verify_lanes.main(["--lanes", "ridge,ols,kmeans,knn,lasso", "--cpu-pass", "--plan"]) == 0
    out = capsys.readouterr().out
    assert "backend=cpu jobs=5" in out and "runpod" not in out
    assert f"--fixtures {verify_lanes.APPLE_PASS_FIXTURES}" in out
    assert "--repeats 1" in out and "--no-batch" in out and "--no-rlpair" in out


def test_a_pass_selects_what_changed_since_the_last_release_tag(monkeypatch, capsys, tmp_path):
    # No finished pass on record, so the tag is the anchor (a real pass
    # record on this machine would otherwise anchor it: test_release_pass_anchor).
    monkeypatch.setenv("MOJOLEARN_RELEASE_CHECK_DIR", str(tmp_path))
    monkeypatch.setattr(verify_lanes, "last_release_tag", lambda: "v9.9.9")
    seen = {}
    def selection(args):
        seen.update(changed_since=args.changed_since, all=args.all, full=args.full_selection)
        return ["ridge"], dict(mode="derived", fallback=False), None
    monkeypatch.setattr(verify_lanes, "_selection", selection)
    assert verify_lanes.main(["--cpu-pass", "--plan"]) == 0
    assert seen == dict(changed_since="v9.9.9", all=False, full=True)
    assert "lanes changed since v9.9.9" in capsys.readouterr().out


def test_a_pass_with_no_release_tag_runs_every_lane_and_an_untouched_release_passes(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("MOJOLEARN_RELEASE_CHECK_DIR", str(tmp_path))
    monkeypatch.setattr(verify_lanes, "last_release_tag", lambda: "")
    monkeypatch.setattr(verify_lanes, "_selection", lambda args: ([], dict(mode="all", fallback=False), None)
                        if args.all else pytest.fail("no tag must mean --all"))
    assert verify_lanes.main(["--apple-pass", "--plan"]) == 0
    assert "touched no lane" in capsys.readouterr().out


def test_a_pass_keeps_its_records_by_commit_outside_the_checkout_and_resumes(monkeypatch, tmp_path):
    monkeypatch.setenv("MOJOLEARN_RELEASE_CHECK_DIR", str(tmp_path))
    monkeypatch.setattr(verify_lanes, "_commit", lambda: "0123456789abcdef")
    assert verify_lanes.pass_out_dir("metal") == str(tmp_path / "0123456789ab" / "metal")
