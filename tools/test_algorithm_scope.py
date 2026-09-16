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
    assert cmd[cmd.index("--repeats") + 1] == "2"


@pytest.mark.parametrize("extra,expected", [([], "base"), (["--exhaustive"], ",".join(ib.FIXTURES))])
def test_existing_runner_uses_same_explicit_fixture_policy(monkeypatch, capsys, extra, expected):
    monkeypatch.setattr(lane_select, "shard", lambda lanes, shards: ([lanes], [1]))
    assert verify_lanes.main(["--lane", "ridge", "--plan", *extra]) == 0
    out = capsys.readouterr().out
    assert f"--fixtures {expected}" in out


def test_invalid_scope_combinations_and_single_repeat_refuse():
    with pytest.raises(SystemExit):
        verify_lanes.main(["--all", "--lane", "ridge", "--plan"])
    with pytest.raises(SystemExit):
        verify_lanes.main(["--lane", "ridge", "--repeats", "1", "--plan"])
    with pytest.raises(SystemExit):
        iterate.main(["--lane", "ridge", "--exhaustive", "--fixtures", "base", "--plan"])
