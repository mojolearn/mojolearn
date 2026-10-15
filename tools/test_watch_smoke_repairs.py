"""Regression checks for the retained 0.8.45 smoke failures; no GPU work."""
import pytest
import bench_board as B
import bench_board_params as P
import board_pairing as PAIR


def test_amd_worker_sets_rocm_home_and_preserves_explicit_path(monkeypatch):
    ctx=dict(vendor="amd", python="/tmp/venv/bin/python", data_root="/tmp/data")
    monkeypatch.delenv("ROCM_HOME", raising=False)
    assert B.child_env(ctx)["ROCM_HOME"] == "/opt/rocm"
    monkeypatch.setenv("ROCM_HOME", "/opt/rocm-custom")
    assert B.child_env(ctx)["ROCM_HOME"] == "/opt/rocm-custom"


def test_categorical_gpu_plan_uses_only_the_no_gain_floor_pairing():
    race=B.plan_races("nvidia", ["identical"], ["trees"], ["gbdt-categorical"], ["taxi"])[0]
    selected=B.gpu_opponents_only([race], "nvidia")[0]
    assert selected["our_arms"] == {"ours-catboost-gpu": "identical"}
    assert "ours" not in selected["arms"]
    assert PAIR.OUR_PAIRINGS["gbdt-categorical"]["ours-catboost-gpu"]["settings"]["min_split_gain"] is None
    assert PAIR.reference_for("catboost-cpu", race["our_pairing"]) == "ours"
    assert PAIR.reference_for("catboost-gpu", race["our_pairing"]) == "ours-catboost-gpu"
    apple=B.plan_races("apple", ["fast", "identical"], ["trees"], ["gbdt-categorical"], ["taxi"])[0]
    assert apple["our_arms"]["ours-ab"] == "fast"
    assert PAIR.reference_for("catboost-cpu",apple["our_pairing"]) == "ours"


def categorical():
    ours=dict(boosting_type="Plain", loss="Logloss", grow_policy="Lossguide", cat_features=[0, 2],
              one_hot_features=None, permutation_count=1, ctr_estimation_permutation_id=0, min_split_gain=None)
    theirs=dict(boosting_type="Plain", loss="Logloss", grow_policy="Lossguide", cat_features=[0, 2],
                has_time=True, one_hot_max_size=2)
    return ours,theirs


def test_categorical_mappings_require_matching_explicit_policy():
    ours,theirs=categorical()
    for param in ("one_hot_features", "one_hot_max_size", "min_split_gain"):
        assert P.semantic_exception("gbdt-categorical", param, "catboost-gpu", ours, theirs, "catboost")
    for key,value in (("one_hot_max_size",3),("cat_features",[0]),("has_time",False)):
        changed=dict(theirs, **{key:value})
        assert not P.semantic_exception("gbdt-categorical","one_hot_features","catboost-gpu",ours,changed,"catboost")
    for key,value in (("one_hot_features",[0]),("min_split_gain",0.0)):
        changed=dict(ours, **{key:value})
        assert not P.semantic_exception("gbdt-categorical",key,"catboost-gpu",changed,theirs,"catboost")


def test_parameter_audit_accepts_gpu_pair_without_the_cpu_reference():
    ours,theirs=categorical()
    arms={"ours-catboost-gpu":dict(ours,__library__="mojolearn"),
          "catboost-gpu":dict(theirs,__library__="catboost")}
    pairing=PAIR.lane_pairing("gbdt-categorical")
    result=P.check("gbdt-categorical",arms,pairing=pairing)
    assert result["verdict"] == "MATCHED", result["problems"]
    assert result["references"]["catboost-gpu"] == "ours-catboost-gpu"
    arms["ours-catboost-gpu"]["min_split_gain"]=0.0
    assert P.check("gbdt-categorical",arms,pairing=pairing)["verdict"] == "REFUSED"


@pytest.mark.parametrize("lane,loss",[("gbdt-rank-pairlogit","PairLogit"),("gbdt-rank-yetirank","YetiRank")])
def test_rank_unit_weights_do_not_waive_the_unsupported_searcher(lane,loss):
    ours=dict(loss=loss,boosting_type="Plain",grow_policy="SymmetricTree",scale_pos_weight=1.0,use_pointwise_searcher=False)
    theirs=dict(loss=loss,boosting_type="Plain",grow_policy="SymmetricTree")
    assert P.semantic_exception(lane,"scale_pos_weight","catboost-gpu",ours,theirs,"catboost")
    assert not P.semantic_exception(lane,"scale_pos_weight","catboost-gpu",dict(ours,scale_pos_weight=2.0),theirs,"catboost")
    assert not P.semantic_exception(lane,"use_pointwise_searcher","catboost-gpu",ours,theirs,"catboost")
