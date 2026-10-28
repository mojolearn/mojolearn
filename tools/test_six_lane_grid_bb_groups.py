"""tools/six_lane_grid_bb.py groups(): each neural lane gets its own bench_board call.

    .pixi/envs/test/bin/python -m pytest tools/test_six_lane_grid_bb_groups.py
"""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("six_lane_grid_bb", os.path.join(HERE, "six_lane_grid_bb.py"))
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)


def test_every_neural_lane_in_a_line_gets_its_own_call():
    races = [("neural", lane, "taxi") for lane in ("gemm", "gemm-bf16", "lm-forward", "lm-train-step")]
    calls = G.groups(races)
    assert sorted(lanes[0] for fam, lanes, _ in calls) == ["gemm", "gemm-bf16", "lm-forward", "lm-train-step"]
    assert all(fam == "neural" and len(lanes) == 1 for fam, lanes, _ in calls)


def test_board_families_still_share_a_call_per_dataset_set():
    ds = G.BOARD_DATASETS[0]
    calls = G.groups([("trees", "gbdt", ds), ("trees", "rf", ds), ("neural", "gemm", ds)])
    assert ("trees", ["gbdt", "rf"], [ds]) in calls
    assert ("neural", ["gemm"], [ds]) in calls
    assert len(calls) == 2
