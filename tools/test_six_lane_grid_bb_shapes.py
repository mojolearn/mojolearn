"""tools/six_lane_grid_bb.py: a neural race may name its own shape (neural:lm-train-step:bytes@target), which gets
its own bench_board call with that --neural-shape; every other race stays at full.

    .pixi/envs/test/bin/python -m pytest tools/test_six_lane_grid_bb_shapes.py
"""
import argparse
import importlib.util
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("six_lane_grid_bb", os.path.join(HERE, "six_lane_grid_bb.py"))
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)
_nspec = importlib.util.spec_from_file_location("gridbb_shapes_neural", os.path.join(HERE, "bench_board_neural.py"))
N = importlib.util.module_from_spec(_nspec)
_nspec.loader.exec_module(N)


def test_parse_race_shape_suffix():
    assert G.parse_race("neural:lm-train-step:bytes") == ("neural", "lm-train-step", "bytes")
    assert G.parse_race("neural:lm-train-step:bytes@target") == ("neural", "lm-train-step", "bytes", "target")
    assert G.parse_race("neural:lm-train-step:bytes@full") == ("neural", "lm-train-step", "bytes")
    with pytest.raises(argparse.ArgumentTypeError):
        G.parse_race("neural:lm-train-step:bytes@huge")
    with pytest.raises(argparse.ArgumentTypeError):
        G.parse_race("trees:gbdt:taxi@target")


def test_target_race_gets_its_own_call_and_others_stay_full():
    races = [G.parse_race(t) for t in ("neural:gemm:gaussian", "neural:lm-train-step:bytes",
                                       "neural:lm-train-step:bytes@target", "neural:lm-forward:bytes")]
    calls = G.shaped_groups(races)
    assert ("neural", ["lm-train-step"], ["taxi"], "target") in calls
    assert ("neural", ["lm-train-step"], ["taxi"], "full") in calls
    assert sorted(lanes[0] for fam, lanes, _, shape in calls if shape == "full") == ["gemm", "lm-forward", "lm-train-step"]
    assert len(calls) == 4
    # groups() keeps its three-field form
    assert all(len(c) == 3 for c in G.groups(races))


def test_target_is_lm_only():
    models = {"lm-train-step": "lm", "gemm": "gemm"}
    assert G.unplanned_reason(_BB, "neural", "lm-train-step", "bytes", "target", models) is None
    assert "not_an_LM_lane" in G.unplanned_reason(_BB, "neural", "gemm", "gaussian", "target", models)


def test_target_shape_matches_the_torch_opponent_target():
    assert N.LM_SHAPES["target"] == [1, 2048, 768, 12, 12, 64, 2048, 12, 50257]
    assert "target" in N.LM_ONLY_SHAPES


def test_summary_line_names_the_shape(capsys):
    board = {"races": {
        "neural/lm-train-step/bytes/shape=full": {"family": "neural", "lane": "lm-train-step", "shape": "full",
                                                  "cells": [{"arm": "ours", "status": "ok", "median_ms": 30.0, "hash": "a" * 16}]},
        "neural/lm-train-step/bytes/shape=target": {"family": "neural", "lane": "lm-train-step", "shape": "target",
                                                    "cells": [{"arm": "ours", "status": "ok", "median_ms": 300.0, "hash": "b" * 16}]}}}
    races = [("neural", "lm-train-step", "bytes"), ("neural", "lm-train-step", "bytes", "target")]
    assert G.summarize(board, races, "t", "nvidia", "abc") == 2
    out = capsys.readouterr().out.splitlines()
    assert "dataset=bytes status=ok median_ms=30.0 hash=" + "a" * 16 in out[0]
    assert "dataset=bytes shape=target status=ok median_ms=300.0 hash=" + "b" * 16 in out[1]


class _BB:
    FAMILIES = ("neural",)

    @staticmethod
    def family_lanes(fam):
        return ("lm-train-step", "gemm")
