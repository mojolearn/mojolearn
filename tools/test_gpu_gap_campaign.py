from types import SimpleNamespace

import pytest
import gpu_gap_campaign as C


def test_manifest_only_selects_requested_supported_gpu_recipes():
    rid = "algos/adam/synthetic/rows=full"
    plan = C.selected_plan("amd", {"amd": [rid]})
    assert [r["id"] for r in plan] == [rid]
    assert plan[0]["our_arms"] == {"ours": "identical"}
    assert all(a.startswith("torch-") for a in plan[0]["opponents"])
    for vendor, manifest in (("apple", {"apple": [rid]}), ("amd", {"amd": [rid, rid]}),
                             ("amd", {"amd": ["not-a-recipe"]})):
        with pytest.raises(ValueError):
            C.selected_plan(vendor, manifest)


def test_synthetic_recipe_is_not_passed_to_tabular_dataset_selector(tmp_path):
    args = SimpleNamespace(python_env="python", vendor="amd", cache=str(tmp_path))
    for dataset in ("synthetic", "gaussian", "taxi", "istella"):
        cmd = C.command(args, dict(family="algos", lane="adam", dataset=dataset), tmp_path)
        assert ("--datasets" in cmd) == (dataset in ("taxi", "istella"))
        assert cmd[cmd.index("--rows") + 1] == "full"
        assert cmd[cmd.index("--rounds") + 1] == "1"
        assert cmd[cmd.index("--round-seconds") + 1] == "0"
        assert "--arm-budget-s" not in cmd
        assert "--race-deadline-s" not in cmd


def test_explicit_wall_time_ceiling_does_not_change_workload(tmp_path):
    args = SimpleNamespace(python_env="python", vendor="nvidia", cache=str(tmp_path), round_seconds=1800)
    cmd = C.command(args, dict(family="classical", lane="dbscan", dataset="taxi"), tmp_path)
    assert cmd[cmd.index("--round-seconds") + 1] == "1800"
    assert cmd[cmd.index("--rows") + 1] == "full"
    assert cmd[cmd.index("--rounds") + 1] == "1"


def test_repair_can_reuse_a_verified_opponent_without_forcing_a_retime(tmp_path):
    args = SimpleNamespace(python_env="python", vendor="nvidia", cache=str(tmp_path),
                           opponent_store=str(tmp_path / "retained-opponents.jsonl"))
    cmd = C.command(args, dict(family="classical", lane="dbscan", dataset="taxi"), tmp_path)
    assert "--retime-opponents" not in cmd
    assert "--with-opponents" in cmd
    assert cmd[cmd.index("--opponent-store") + 1] == args.opponent_store
