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
