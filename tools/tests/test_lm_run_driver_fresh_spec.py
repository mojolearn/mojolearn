"""The driver re-reads the spec's rental parameters when a segment starts,
and refuses a spec whose plan changed under the run."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lm_run_driver as drv  # noqa: E402


def _spec(tmp_path, wheel="0.8.15", steps=10):
    p = tmp_path / "spec.json"
    p.write_text(json.dumps({"run": "runs/x", "recipe": "r.json", "recipe_key": "runs/x/recipe.json",
                             "tokens_stage": "corpus/t", "wheel": wheel, "amd_size": "gpu-mi325x1-256gb",
                             "routes": {"A": [{"segment": "1", "vendor": "nvidia", "steps": steps}],
                                        "B": [{"segment": "1", "vendor": "amd", "steps": steps}]}}))
    return p


def test_a_changed_wheel_and_size_are_read_at_segment_start(tmp_path):
    p = _spec(tmp_path)
    plan = drv.segment_plan(drv.load_spec(p))
    p.write_text(p.read_text().replace('"0.8.15"', '"0.8.17"').replace("gpu-mi325x1-256gb", "gpu-mi325x8-2048gb"))
    now = drv.fresh_spec(p, plan)
    assert now["wheel"] == "0.8.17" and now["amd_size"] == "gpu-mi325x8-2048gb"


def test_a_changed_plan_is_refused_by_name(tmp_path):
    p = _spec(tmp_path)
    plan = drv.segment_plan(drv.load_spec(p))
    p.write_text(_spec(tmp_path, steps=12).read_text())
    with pytest.raises(RuntimeError) as err:
        drv.fresh_spec(p, plan)
    assert str(p) in str(err.value) and "step counts changed" in str(err.value)


def test_an_unchanged_spec_reads_back_equal(tmp_path):
    p = _spec(tmp_path)
    spec = drv.load_spec(p)
    assert drv.fresh_spec(p, drv.segment_plan(spec)) == spec
