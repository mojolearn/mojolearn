# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The formerly refused GPU lanes: their real on-box parameter read-backs, replayed.

    .pixi/envs/test/bin/python -m pytest tools/test_refused_param_mappings.py

tools/param_readbacks/smoke0845_refused_lanes.json holds what every arm of these races
read back on the L40S and the MI325X in the 0.8.45 smoke (smoke-0845, f4c365b4c; AMD
CAGRA from its hipVS recheck): the lanes the 2026-10-15 quality review listed as
"blocked until the parameter mapping is fixed" (decision-tree, qn-reg, ivf-sq, CAGRA,
ALS, UMAP, spectral, agglomerative, ARIMA, ETS, the CatBoost GPU lanes, the torch
references), plus sgd-reg (the same loss spelling as qn-reg). Each race is replayed
through bench_board_params.check, and every value-conditional mapping is shown to
refuse again once its condition stops holding: a mapping explains one reviewed
difference and never silences another.
"""
import copy
import importlib.util
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BP = _load("bench_board_params")
FIXTURE = os.path.join(HERE, "param_readbacks", "smoke0845_refused_lanes.json")
with open(FIXTURE) as _f:
    RACES = json.load(_f)

#: The two querywise ranking lanes: 0.8.45 ran its greedy-subsets searcher against CatBoost's
#: DocParallel pointwise searcher, a different algorithm. The harness on main asks for the native
#: pointwise searcher (f63ef2235, first released in 0.8.46), so this 0.8.45 read-back must refuse.
OLD_RANKING_ROUTE = {
    "nvidia|trees/gbdt-rank-pairlogit/istella/rows=full",
    "nvidia|trees/gbdt-rank-yetirank/istella/rows=full",
}


def _records(race):
    return {n: dict(a, __record__=True) for n, a in race["arms"].items()}


def _check(race, records=None):
    try:
        rep = BP.check(race["lane"], records or _records(race), family=race["family"], pairing=race["pairing"])
        return rep["verdict"], rep["problems"], rep
    except BP.ParamsRefused as exc:
        return "REFUSED", [str(exc)], None


@pytest.mark.parametrize("key", sorted(RACES))
def test_box_readback_replays_to_the_box_verdict(key):
    race = RACES[key]
    verdict, problems, _ = _check(race)
    if key in OLD_RANKING_ROUTE:
        assert verdict == "REFUSED"
        assert problems == ["catboost-gpu: use_pointwise_searcher missing from catboost-gpu read-back; "
                            "explicit semantic mapping required"]
    else:
        assert race["box_verdict"] == "MATCHED"
        assert verdict == "MATCHED", problems


@pytest.mark.parametrize("key", sorted(RACES))
def test_every_opponent_read_back_names_its_real_library_class(key):
    bb = _load("bench_board")
    race = RACES[key]
    for name, arm in race["arms"].items():
        if name.startswith("ours") or name in race["our_arms"]:
            continue
        assert bb.real_implementation(arm), (name, arm.get("implementation"))
    assert not bb.real_implementation({"implementation": {"module": "bench_board_algos", "name": "Wrapper"}})
    assert not bb.real_implementation({"implementation": None})


def _other(v):
    if isinstance(v, bool):
        return not v
    if isinstance(v, (int, float)):
        return v + 7
    if isinstance(v, str):
        return v + "-other"
    return 3


#: The 0.8.45 MLP read-backs compared nothing: ours declared no setting and the torch twin only its
#: seed. bench_board_neural.py now reads back the network on both arms (config_readback, ARCH_FIELDS
#: 'mlp'); test_mlp_lanes_compare_the_network_both_arms_run covers it until a box read-back replaces these.
NOTHING_COMPARED_IN_0845 = {
    "amd|neural/mlp-predict/gaussian/shape=full",
    "nvidia|neural/mlp-predict/gaussian/shape=full",
}


@pytest.mark.parametrize("key", sorted(k for k in RACES if k not in OLD_RANKING_ROUTE))
def test_every_compared_setting_still_refuses_when_the_opponent_changes_it(key):
    race = RACES[key]
    _, _, rep = _check(race)
    same = [(c["arm"], c["param"]) for c in rep["compared"] if c["same"]]
    if key in NOTHING_COMPARED_IN_0845:
        assert not same
        return
    assert same
    for arm, param in same:
        records = _records(race)
        raw = records[arm]["params"]
        names = [k for k, (v, src) in BP.canonical(records[arm]["library"], raw).items() if k == param]
        source = BP.canonical(records[arm]["library"], raw)[names[0]][1]
        records[arm] = copy.deepcopy(records[arm])
        records[arm]["params"][source] = _other(raw[source])
        verdict, problems, _ = _check(race, records)
        assert verdict == "REFUSED", (arm, param, source)


#: (race, arm, raw parameter, value, the parameter the refusal must name): each changes the one
#: value a value-conditional mapping was reviewed for, everything else as read back on the box.
BROKEN_CONDITIONS = [
    # cuML spells squared error 'l2' (QN) / 'squared_loss' (MBSGD); only that pair maps
    ("nvidia|algos/qn-reg/taxi/rows=full", "ours", "loss", "huber", "loss"),
    ("nvidia|algos/sgd-reg/taxi/rows=full", "ours", "loss", "huber", "loss"),
    # no leaf cap on either arm: ours None <-> cuML -1 only
    ("nvidia|algos/decision-tree-clf/taxi/rows=full", "cuml-gpu", "max_leaves", 64, "max_leaves"),
    ("nvidia|algos/decision-tree-reg/istella/rows=full", "ours", "max_leaf_nodes", 64, "max_leaves"),
    ("nvidia|algos/decision-tree-clf/istella/rows=full", "ours", "ccp_alpha", 0.01, "ccp_alpha"),
    # implicit-gpu must read back its CUDA class
    ("nvidia|algos/als/text/rows=full", "implicit-gpu", "use_gpu", False, "use_gpu"),
    # UMAP a/b None maps only when both fit the curve from the same spread/min_dist
    ("nvidia|classical2/umap/taxi/rows=full", "ours", "a", 1.58, " a "),
    ("nvidia|classical2/agglomerative/taxi/rows=full", "ours", "compute_distances", True, "compute_distances"),
    ("nvidia|classical2/arima/synthetic/rows=full", "ours", "trend", "n", "trend"),
    ("nvidia|classical2/ets/synthetic/rows=full", "ours", "initialization_method", "estimated",
     "initialization_method"),
    ("nvidia|classical2/spectral-embedding/taxi/rows=full", "ours", "eigen_tol", 1e-3, "eigen_tol"),
    ("nvidia|classical2/spectral/istella/rows=full", "ours", "assign_labels", "discretize", "assign_labels"),
    # CatBoost GPU: bagging_temperature is inactive only with bootstrap 'No' on both
    ("nvidia|trees/gbdt-symmetric-1000/taxi/rows=full", "ours", "bootstrap_type", "Bayesian", "bootstrap"),
]


@pytest.mark.parametrize("key,arm,param,value,named", BROKEN_CONDITIONS)
def test_a_mapping_refuses_once_its_reviewed_condition_breaks(key, arm, param, value, named):
    race = RACES[key]
    records = _records(race)
    assert param in records[arm]["params"], param
    records[arm] = copy.deepcopy(records[arm])
    records[arm]["params"][param] = value
    verdict, problems, _ = _check(race, records)
    assert verdict == "REFUSED"
    assert any(named in " %s " % p for p in problems), problems


def test_row_sampling_mapping_needs_bootstrap_no_on_both():
    race = RACES["nvidia|trees/gbdt-symmetric-1000/istella/rows=full"]
    records = copy.deepcopy(_records(race))
    for name in ("ours", "catboost-gpu"):
        records[name]["params"]["bootstrap_type"] = "Bernoulli"
    records["ours"]["params"]["subsample"] = 0.5
    records["catboost-gpu"]["params"]["subsample"] = 0.8
    verdict, problems, _ = _check(race, records)
    assert verdict == "REFUSED" and any("subsample" in p for p in problems), problems


def test_exception_rows_apply_only_to_their_reviewed_values():
    o = {"boosting_type": "Plain", "bootstrap_type": "No", "subsample": None, "class_weight": None}
    assert BP.exception_for("gbdt-depthwise", "boosting_type", "xgboost-gpu", o, {"boosting_type": "gbtree"})
    assert BP.exception_for("gbdt-depthwise", "boosting_type", "xgboost-gpu", dict(o, boosting_type="Ordered"),
                            {"boosting_type": "gbtree"}) is None
    assert BP.exception_for("gbdt-depthwise", "boosting_type", "xgboost-gpu", o, {"boosting_type": "dart"}) is None
    assert BP.exception_for("gbdt-depthwise", "subsample", "xgboost-gpu", o, {"subsample": 1.0})
    assert BP.exception_for("gbdt-depthwise", "subsample", "xgboost-gpu", o, {"subsample": 0.8}) is None
    assert BP.exception_for("logreg", "class_weight", "cuml-gpu", o, {"class_weight": None})
    assert BP.exception_for("logreg", "class_weight", "cuml-gpu", dict(o, class_weight="balanced"),
                            {"class_weight": None}) is None
    # a pinned row never applies without the two arms' values
    assert BP.exception_for("logreg", "class_weight", "cuml-gpu") is None
    # every pinned row is a real EXCEPTIONS row
    rows = {r[:3] for r in BP.EXCEPTIONS}
    assert set(BP.EXCEPTION_VALUES) <= rows


def test_mlp_lanes_compare_the_network_both_arms_run():
    neural = _load("bench_board_neural")
    assert {k for k, v in neural.MODEL_OF.items() if v == "mlp"} >= {"mlp-predict", "mlp-train-step"}
    ours = {"__library__": "mojolearn", "config_readback": neural.mlp_arch(8, 16, 3, "relu", True)}

    def twin(**kw):
        arch = dict(d_in=8, hidden=16, d_out=3, activation="relu", bias=True)
        arch.update(kw)
        return {"__library__": "torch", "seed": 7, "config_readback": neural.mlp_arch(**arch)}

    rep = BP.check("neural/mlp-predict", {"ours": ours, "torch-eager-fp32": twin()}, family="neural")
    assert rep["verdict"] == "MATCHED", rep["problems"]
    assert [c["param"] for c in rep["compared"]] == ["config_readback"]
    for change in (dict(hidden=32), dict(activation="gelu"), dict(bias=False), dict(d_out=4)):
        rep = BP.check("neural/mlp-predict", {"ours": ours, "torch-eager-fp32": twin(**change)}, family="neural")
        assert rep["verdict"] == "REFUSED" and any("config_readback" in p for p in rep["problems"]), change
