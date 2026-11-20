# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FAIRNESS FIX (2026-10-30): gbdt-lossguide races LightGBM against `ours-gain0` on the pre-binned input.

The parameter check (tools/bench_board_params.py) is run on a REAL LightGBM read-back: the driver's own
arm (tools/speed_gbdt_arm.py lightgbm_arms) builds the LGBMClassifier, whose get_params is compared with
the `ours-gain0`, CatBoost GPU and XGBoost GPU records the 2026-10-30 NVIDIA smoke read back
(~/mojolearn-evidence/v0855-paper-20261029/final/lgbm-attempt/nvlg.tgz, board-smoke/board.json). Every
mapping is value-conditional: each test that changes one LightGBM value sees the race refused by name.
A tiny CPU LightGBM fit checks the two LightGBM facts the mapping rests on (every pre-binned id its own bin;
min_child_samples 0 with min_child_weight 1e-12 trains). The catboost border oracle is replaced by numpy
quantile borders (the binning rule itself is not under test here).

    .pixi/envs/test/bin/python -m pytest tools/test_lgbm_lossguide_mapping.py
"""
import copy
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

lgb = pytest.importorskip("lightgbm")

import bench_board_params as BP        # noqa: E402
import board_pairing                   # noqa: E402
import speed_gbdt_arm as spec          # noqa: E402

LANE = "gbdt-lossguide"
SMOKE_RECORD = ("pre-binned input on both arms (GreedyLogSum, 254 borders); bins sha256 "
                "03b59849cf8ff59cfea9cabd98df62afbccc042c0df5460b214a70362ff55108")

#: the 2026-10-30 NVIDIA smoke read-backs (board.json preflight.races[...].audit.arms[...].params), verbatim
OURS_GAIN0 = {
    "bagging_temperature": 1.0, "best_model_min_trees": 1, "boost_from_average": False, "boosting_type": "Plain",
    "bootstrap_type": "No", "border_build_max_samples": 0, "cat_features": None,
    "ctr_estimation_permutation_id": None, "feature_border_type": "GreedyLogSum", "feature_fraction": 1.0,
    "fold_len_multiplier": 2.0, "fold_permutation_block": None, "grow_policy": "Lossguide",
    "input_binning": SMOKE_RECORD, "leaf_estimation_iterations": 1, "leaf_estimation_method": "Newton",
    "learning_rate": 0.1, "loss": "Logloss", "loss_alpha": None, "loss_border": None, "loss_delta": None,
    "loss_q": None, "loss_variance_power": None, "max_bin": 255, "max_depth": 16, "max_leaves": 256,
    "min_child_weight": 0.0, "min_samples_leaf": 1, "min_split_gain": 0.0, "n_estimators": 500,
    "nan_mode": "Min", "od_pvalue": None, "od_type": None, "od_wait": None, "one_hot_features": None,
    "permutation_count": None, "random_strength": 0.0, "reg_lambda": 1.0,
    "scale_pos_weight": 1.2987012987012987, "score_function": "NewtonL2", "seed": 7, "subsample": None,
    "use_best_model": None, "use_pointwise_searcher": False}
OURS = dict(OURS_GAIN0, border_build_max_samples=200000, min_split_gain=None)
del OURS["input_binning"]
CATBOOST_GPU_RAW = {
    "allow_writing_files": False, "boost_from_average": False, "boosting_type": "Plain", "bootstrap_type": "No",
    "border_count": 254, "depth": 16, "devices": "0", "feature_border_type": "GreedyLogSum",
    "grow_policy": "Lossguide", "iterations": 500, "l2_leaf_reg": 1.0, "leaf_estimation_iterations": 1,
    "leaf_estimation_method": "Newton", "learning_rate": 0.1, "loss_function": "Logloss", "max_leaves": 256,
    "min_data_in_leaf": 1, "nan_mode": "Min", "random_seed": 7, "random_strength": 0.0,
    "scale_pos_weight": 1.2987012987012987, "score_function": "NewtonL2", "subsample": None,
    "task_type": "GPU", "verbose": False}
XGBOOST_GPU_RAW = {
    "base_score": 0.5, "booster": "gbtree", "callbacks": None, "colsample_bylevel": 1.0, "colsample_bynode": 1.0,
    "colsample_bytree": 1.0, "device": "cuda", "early_stopping_rounds": None, "enable_categorical": False,
    "eval_metric": None, "feature_types": None, "feature_weights": None, "gamma": 0.0,
    "grow_policy": "lossguide", "importance_type": None, "input_binning": SMOKE_RECORD,
    "interaction_constraints": None, "learning_rate": 0.1, "max_bin": 255, "max_cat_threshold": None,
    "max_cat_to_onehot": None, "max_delta_step": 0.0, "max_depth": 16, "max_leaves": 256,
    "min_child_weight": 0.0, "missing": float("nan"), "monotone_constraints": None, "multi_strategy": None,
    "n_estimators": 500, "n_jobs": None, "num_parallel_tree": None, "objective": "binary:logistic",
    "random_state": 7, "reg_alpha": 0.0, "reg_lambda": 1.0, "sampling_method": None,
    "scale_pos_weight": 1.2987012987012987, "subsample": 1.0, "tree_method": "hist",
    "validate_parameters": None, "verbosity": 0}


def _record(library, raw):
    return {"__record__": True, "library": library, "source": "smoke 2026-10-30 read-back", "raw_params": raw,
            "params": raw}


def _quantile_borders(x_train, cfg):
    """Stand-in for CatBoost's Pool.quantize: at most cfg['borders'] sorted unique float32 borders a feature."""
    out = []
    for f in range(x_train.shape[1]):
        col = np.asarray(x_train[:, f], dtype=np.float32)
        values = np.unique(col)
        if len(values) <= int(cfg["borders"]) + 1:      # few values: each its own bin, as GreedyLogSum
            out.append(((values[:-1] + values[1:]) / 2).astype(np.float32))
            continue
        qs = np.unique(np.quantile(col, np.linspace(0, 1, int(cfg["borders"]) + 2)[1:-1]).astype(np.float32))
        out.append(qs[qs > col.min()])
    return out, "numpy-quantile"


@pytest.fixture
def toy(monkeypatch):
    monkeypatch.setattr(spec, "greedylogsum_borders", _quantile_borders)
    monkeypatch.setattr(spec, "_smoke_staging", lambda: None)
    monkeypatch.delenv("MOJOLEARN_SPEED_LGBM_PARAMS", raising=False)
    rng = np.random.default_rng(7)
    n, f = 3000, 4
    x = rng.normal(size=(n, f)).astype(np.float32) * 10
    x[:, 3] = np.round(x[:, 3])                       # a few distinct values, the tails rare (1-2 rows)
    xt = rng.normal(size=(400, f)).astype(np.float32) * 10
    y = (x[:, 0] + rng.normal(size=n) > 0).astype(np.float32)
    yt = (xt[:, 0] > 0).astype(np.float32)
    cfg = spec.lane_config(LANE, "shipped")
    return spec.Data("toy", x, xt, y, yt, "binary"), cfg


def _lgbm_arm(data, cfg, dev="cuda"):
    arms = spec.lightgbm_arms(LANE, cfg, data, [dev])
    assert len(arms) == 1
    return arms[0]


def _race(model, ours_gain0):
    """The preflight's own call (tools/bench_board.py params check): every arm, the lane's pairing. The
    smoke records carry taxi's positive-class weight; the toy data's (scale_pos_weight_for) replaces it."""
    spw = dict(scale_pos_weight=model.get_params()["scale_pos_weight"])
    arms = {"ours": _record("mojolearn", dict(OURS, **spw)),
            "ours-gain0": _record("mojolearn", dict(ours_gain0, **spw)),
            "catboost-gpu": _record("catboost", dict(CATBOOST_GPU_RAW, **spw)),
            "xgboost-gpu": _record("xgboost", dict(XGBOOST_GPU_RAW, input_binning=ours_gain0["input_binning"],
                                                   **spw)),
            "lightgbm-cuda": model}
    return BP.check(LANE, arms, family="trees", pairing=board_pairing.lane_pairing(LANE))


def _lgbm_problems(rep):
    return [p for p in rep["problems"] if p.startswith("lightgbm-")]


def test_real_lightgbm_read_back_matches_ours_gain0(toy):
    data, cfg = toy
    arm = _lgbm_arm(data, cfg)
    model = arm.make()
    assert type(model).__module__.startswith("lightgbm") and BP.library_of(model) == "lightgbm"
    raw = model.get_params()
    # the explicit LightGBM-only knobs and the bin settings, as the driver set them
    assert (raw["min_child_samples"], raw["min_child_weight"]) == (0, 1e-12)
    assert (raw["min_data_in_bin"], raw["subsample_for_bin"]) == (1, data.X_train.shape[0])
    assert (raw["extra_trees"], raw["max_delta_step"], raw["path_smooth"], raw["use_quantized_grad"]) \
        == (False, 0.0, 0.0, False)
    assert arm.data.input_binning == model._board_input_binning
    assert spec.PREBIN_ROWS % data.X_train.shape[0] in arm.data.input_binning
    rep = _race(model, dict(OURS_GAIN0, input_binning=arm.data.input_binning))
    assert rep["references"]["lightgbm-cuda"] == "ours-gain0"
    assert rep["verdict"] == "MATCHED", rep["problems"]
    mapped = {r["param"] for r in rep["exceptions"] if r["arm"] == "lightgbm-cuda"}
    # every field the 2026-10-30 smoke refused on is now mapped (or compared equal), by name
    assert {"bagging_temperature", "best_model_min_trees", "bootstrap_type", "border_build_max_samples",
            "cat_features", "class_weight", "ctr_estimation_permutation_id", "feature_border_type",
            "fold_len_multiplier", "fold_permutation_block", "grow_policy", "leaf_estimation_iterations",
            "leaf_estimation_method", "loss", "loss_alpha", "loss_border", "loss_delta", "loss_q",
            "loss_variance_power", "nan_mode", "objective", "od_pvalue", "od_type", "od_wait",
            "one_hot_features", "permutation_count", "random_strength", "reg_alpha", "score_function",
            "subsample_for_bin", "subsample_freq", "use_best_model", "use_pointwise_searcher",
            "min_child_weight", "min_samples_leaf", "boosting_type", "min_data_in_bin", "extra_trees",
            "max_delta_step", "path_smooth", "use_quantized_grad"} <= mapped, mapped
    same = {r["param"] for r in rep["compared"] if r["arm"] == "lightgbm-cuda" and r["same"]}
    assert {"min_split_gain", "input_binning", "max_bin", "max_leaves", "max_depth", "learning_rate",
            "reg_lambda", "n_estimators", "scale_pos_weight", "boost_from_average", "feature_fraction"} <= same


@pytest.mark.parametrize("change,needle", [
    (dict(min_data_in_bin=3), "min_data_in_bin"),
    (dict(subsample_for_bin=200), "subsample_for_bin"),
    (dict(extra_trees=True), "extra_trees"),
    (dict(max_delta_step=0.7), "max_delta_step"),
    (dict(path_smooth=1.0), "path_smooth"),
    (dict(use_quantized_grad=True), "use_quantized_grad"),
    (dict(reg_alpha=0.5), "reg_alpha"),
    (dict(subsample_freq=1, subsample=0.5), "subsample"),
    (dict(min_child_weight=1e-3), "min_child_weight"),
    (dict(min_child_samples=20), "min_samples_leaf"),
    (dict(boosting_type="dart"), "boosting_type"),
    (dict(class_weight="balanced"), "class_weight"),
    (dict(num_leaves=31), "max_leaves"),
])
def test_each_mapping_is_value_conditional(toy, change, needle):
    data, cfg = toy
    arm = _lgbm_arm(data, cfg)
    model = arm.make()
    model.set_params(**change)
    rep = _race(model, dict(OURS_GAIN0, input_binning=arm.data.input_binning))
    probs = _lgbm_problems(rep)
    assert rep["verdict"] == "REFUSED" and any(needle in p for p in probs), probs


def test_lightgbm_is_never_compared_with_the_catboost_configuration(toy):
    data, cfg = toy
    model = _lgbm_arm(data, cfg).make()
    # without the pairing LightGBM meets `ours` (min_split_gain None, raw input): refused
    arms = {"ours": _record("mojolearn", OURS), "lightgbm-cuda": model}
    rep = BP.check(LANE, arms, family="trees")
    assert [p for p in rep["problems"] if p.startswith("lightgbm-cuda: min_split_gain")], rep["problems"]
    assert [p for p in rep["problems"] if "border_build_max_samples" in p], rep["problems"]


def test_other_bins_or_raw_input_refuse(toy):
    data, cfg = toy
    arm = _lgbm_arm(data, cfg)
    model = arm.make()
    # ours-gain0 on another block (another sha256): the record differs and the border fields stay unmapped
    rep = _race(model, dict(OURS_GAIN0, input_binning=arm.data.input_binning.replace(
        arm.data.bins_sha256, "cd" * 32)))
    assert any(p.startswith("lightgbm-cuda: input_binning") for p in rep["problems"]), rep["problems"]
    assert any("feature_border_type missing" in p for p in _lgbm_problems(rep))
    # a record without the row count (the 2026-10-30 smoke's): the bin sample cannot be checked
    model._board_input_binning = SMOKE_RECORD
    rep = _race(model, OURS_GAIN0)
    assert any("subsample_for_bin missing" in p for p in _lgbm_problems(rep)), rep["problems"]


def test_other_boosted_lanes_keep_the_refusal(toy):
    data, _ = toy
    for lane in ("gbdt-depthwise", "gbdt-multiclass", "gbdt-symmetric"):
        with pytest.raises(RuntimeError, match="MIN-CHILD|DIFFERENT-GROWER|symmetric"):
            spec.lightgbm_arms(lane, spec.lane_config(lane, "shipped"), data, ["cuda"])
    assert not board_pairing.prebinned_arm("gbdt-depthwise", "lightgbm-cuda")


def test_override_back_to_the_estimated_count_rule_is_refused_by_name(toy, monkeypatch):
    data, cfg = toy
    monkeypatch.setenv("MOJOLEARN_SPEED_LGBM_PARAMS", "min_child_samples=1")
    with pytest.raises(RuntimeError, match="MIN-CHILD-ESTIMATE"):
        spec.lightgbm_arms(LANE, cfg, data, ["cuda"])


def test_lightgbm_facts_the_mapping_rests_on(toy):
    """CPU LightGBM, tiny: (1) at min_data_in_bin 1 and a bin sample of every row each pre-binned id is its
    own bin, at LightGBM's defaults (3, a smaller sample) rare ids merge; (2) min_child_samples 0 with
    min_child_weight 1e-12 trains, every leaf non-empty."""
    data, cfg = toy
    arm = _lgbm_arm(data, cfg, dev="cpu")
    pb = arm.data
    distinct = [len(np.unique(pb.X_train[:, f])) for f in range(pb.X_train.shape[1])]
    assert max(distinct) <= 255

    def bins(**kw):
        ds = lgb.Dataset(pb.X_train, label=pb.y_train, params=dict(verbose=-1, max_bin=255, **kw),
                         free_raw_data=False).construct()
        return [ds.feature_num_bin(f) for f in range(pb.X_train.shape[1])]

    assert bins(min_data_in_bin=1, bin_construct_sample_cnt=pb.X_train.shape[0]) == distinct
    assert bins(min_data_in_bin=3, bin_construct_sample_cnt=pb.X_train.shape[0]) != distinct
    model = arm.make()
    model.set_params(n_estimators=5, n_jobs=1, device_type="cpu")
    arm.fit(model, pb)
    tree = model.booster_.dump_model()["tree_info"]

    def leaf_counts(node):
        if "leaf_index" in node:
            return [node["leaf_count"]]
        return leaf_counts(node["left_child"]) + leaf_counts(node["right_child"])

    counts = [c for t in tree for c in leaf_counts(t["tree_structure"])]
    assert len(tree) == 5 and min(counts) >= 1 and len(counts) > 5


def test_planned_refusal_and_note_text_follow_the_lane(toy):
    import bench_board as bb
    for vendor in ("nvidia", "amd"):
        assert bb._planned_refusal(vendor, "trees/gbdt-lossguide/taxi", "lightgbm-cuda") is None
        assert bb._planned_refusal(vendor, "trees/gbdt-depthwise/taxi", "lightgbm-cuda")
    assert copy.deepcopy(board_pairing.lane_pairing(LANE)) == {
        "ours": ["catboost-gpu"], "ours-gain0": ["xgboost-*", "lightgbm-*"]}
