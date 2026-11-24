#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""tools/test_alm_drivers.py -- A/B LOOP MODE in the board drivers (lane ab-loop-mode / alm-drivers, 2026-11-24).

Pure Python, laptop: never imports mojolearn (the estimators here are fakes). For each driver: the function that
applies the plan (tools/ab_loop.py capped_params) touches OUR arms only, and only when the plan's environment key
(ab_loop.ENV) is set; the loop reads happen outside the call; the cell record is ab_loop.cell_record's.

    python -m pytest tools/test_alm_drivers.py
"""
import importlib.util
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def _mod(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AB = _mod("tools/ab_loop.py", "alm_ab_loop")
CAPS = {
    ("trees", "rf"): dict(valid=True, param="n_estimators", cap=20, pin={}, unit="tree", reason="r"),
    ("classical", "kmeans"): dict(valid=True, param="max_iter", cap=10, pin={"tol": 1e-30}, unit="it", reason="r"),
    ("classical", "pca"): dict(valid=False, param=None, cap=0, pin={}, unit="-", reason="no loop"),
    ("classical2", "umap"): dict(valid=True, param="n_epochs", cap=5, pin={}, unit="epoch", reason="r"),
    ("algos", "sgd-clf"): dict(valid=True, param="max_iter", cap=5, pin={"tol": None}, unit="epoch", reason="r"),
    ("neural", "mlp-fit"): dict(valid=True, param="max_iter", cap=2, pin={"tol": 0.0}, unit="epoch", reason="r"),
    ("neural", "sequential-fit"): dict(valid=True, param="fit:epochs", cap=2, pin={}, unit="epoch", reason="r"),
}


def _env(family, lane):
    return {AB.ENV: json.dumps(AB.race_plan(family, lane, caps=CAPS))}


def _plan(family, lane):
    return AB.race_plan(family, lane, caps=CAPS)


class FakeModel(object):
    """A fitted estimator: fit_loop_ and get_params, nothing else."""

    def __init__(self, params, loop=(10, 4.0)):
        self.params, self.fit_loop_ = dict(params), loop

    def get_params(self):
        return dict(self.params)


# ------------------------------------------------------------------ the shared helpers (tools/bench_board_more.py)

@pytest.fixture(scope="module")
def more():
    return _mod("tools/bench_board_more.py", "alm_bench_board_more")


def test_more_plan_only_when_env_set(more):
    assert more.loop_plan("umap", environ={}) is None
    plan = more.loop_plan("kmeans", environ=_env("classical", "kmeans"), family="classical")
    assert plan["mode"] == "capped" and plan["param"] == "max_iter" and plan["k"] == 10
    with pytest.raises(ValueError):          # a plan for another race is refused, never applied
        more.loop_plan("pca", environ=_env("classical", "kmeans"), family="classical")


def test_more_params_ours_only(more):
    plan, params = _plan("classical", "kmeans"), {"max_iter": 300, "tol": 1e-7, "n_clusters": 8}
    assert more.loop_params(None, params, True) is params                # no plan: the same object
    assert more.loop_params(plan, params, False) is params               # an opponent: the same object
    capped = more.loop_params(plan, params, True)
    assert capped == {"max_iter": 10, "tol": 1e-30, "n_clusters": 8}
    assert params == {"max_iter": 300, "tol": 1e-7, "n_clusters": 8}     # never PARAMS itself
    assert more.loop_params(_plan("classical", "pca"), params, True) == params   # 'whole': unchanged values


def test_more_round_and_cell(more):
    plan = _plan("classical", "kmeans")
    model = FakeModel({"max_iter": 10, "tol": 1e-30, "n_clusters": 8}, loop=(10, 5.0))
    rec = more.loop_round(plan, model)
    assert rec["call"] == {"status": "recorded", "iterations": 10, "loop_ms": 5.0}
    assert rec["effective"] == {"max_iter": 10, "tol": 1e-30}
    cell = more.loop_cell(plan, [(0, rec), (1, rec), (2, more.loop_round(plan, FakeModel(model.params, (10, 7.0))))])
    assert cell["mode"] == "capped" and cell["iterations"] == 10 and cell["loop_ms"] == 6.0
    assert cell["warmup"]["loop_ms"] == 5.0 and len(cell["calls"]) == 2
    # a `fit:` cap has no get_params read-back; it enters the fit() keywords, never the constructor's
    seq = _plan("neural", "sequential-fit")
    rec = more.loop_round(seq, object())
    assert rec["effective"] == {} and rec["call"]["status"] == "unavailable"
    kw = {"epochs": 1, "batch_size": 64}
    assert more.loop_fit_kwargs(seq, kw, True) == {"epochs": 2, "batch_size": 64}
    assert more.loop_fit_kwargs(seq, kw, False) is kw and more.loop_fit_kwargs(None, kw, True) is kw
    assert more.loop_params(seq, {"hidden": 8}, True) == {"hidden": 8}


def test_more_build_refuses_capped_ours(more):
    pytest.importorskip("numpy")              # build() imports numpy before it looks at the arm
    more._LOOP_PLANS[("classical2", "umap")] = _plan("classical2", "umap")
    try:
        with pytest.raises(RuntimeError, match="REFUSED: loop mode"):
            more.build("umap", "ours", {}, {})
    finally:
        more._LOOP_PLANS.pop(("classical2", "umap"), None)


# ------------------------------------------------------------------ classical (tools/classical_two_datasets.py)

def test_classical_kmeans_cap(monkeypatch):
    pytest.importorskip("numpy")
    ctd = _mod("tools/classical_two_datasets.py", "alm_classical_two_datasets")
    built = []

    class FakeML(object):
        @staticmethod
        def KMeans(**kw):
            built.append(kw)
            return FakeModel(kw)

    monkeypatch.setattr(ctd, "_ours_module", lambda: FakeML)
    monkeypatch.setattr(ctd, "_ours_info", lambda ml, obj: {"library": "mojolearn"})
    data = {"X": None, "init": None}
    monkeypatch.setattr(ctd, "loop_plan", lambda lane, environ=None: None)
    off = ctd.OursKMeans(data, {})
    assert built[-1]["max_iter"] == ctd.KMEANS_ITER and built[-1]["tol"] == ctd.KMEANS_TOL
    assert not off.loop_capped
    monkeypatch.setattr(ctd, "loop_plan", lambda lane, environ=None: _plan("classical", "kmeans"))
    on = ctd.OursKMeans(data, {})
    assert built[-1]["max_iter"] == 10 and built[-1]["tol"] == 1e-30 and on.loop_capped
    on.make()                                 # every construction (the warm-up's too) is the capped one
    assert built[-1]["max_iter"] == 10
    assert ctd.loop_params(_plan("classical", "kmeans"), {"max_iter": 300}, False) == {"max_iter": 300}


# ------------------------------------------------------------------ algos (tools/bench_board_algos.py)

def test_algos_ours_params(monkeypatch):
    pytest.importorskip("numpy")
    bba = _mod("tools/bench_board_algos.py", "alm_bench_board_algos")
    params = {"max_iter": 1000, "tol": 1e-3, "alpha": 1e-4}
    monkeypatch.setattr(bba, "loop_plan", lambda lane, environ=None: None)
    assert bba._loop_ours_params("sgd-clf", params) is params and "sgd-clf" not in bba._LOOP_APPLIED
    monkeypatch.setattr(bba, "loop_plan", lambda lane, environ=None: _plan("algos", "sgd-clf"))
    assert bba._loop_ours_params("sgd-clf", params) == {"max_iter": 5, "tol": None, "alpha": 1e-4}
    assert "sgd-clf" in bba._LOOP_APPLIED and params["max_iter"] == 1000
    assert bba.loop_params(_plan("algos", "sgd-clf"), params, False) is params   # opponents untouched


# ------------------------------------------------------------------ neural (tools/bench_board_neural.py)

def test_neural_params():
    pytest.importorskip("numpy")
    bbn = _mod("tools/bench_board_neural.py", "alm_bench_board_neural")
    assert bbn.loop_plan("mlp-fit", environ={}) is None
    kw = {"max_iter": 1, "tol": 0.0, "batch_size": 512}
    assert bbn.loop_params(None, kw, True) is kw
    assert bbn.loop_params(_plan("neural", "mlp-fit"), kw, False) is kw
    assert bbn.loop_params(_plan("neural", "mlp-fit"), kw, True)["max_iter"] == 2
    seq = _plan("neural", "sequential-fit")
    assert bbn._loop().loop_fit_kwargs(seq, {"epochs": 1, "batch_size": 512}, True) == {"epochs": 2, "batch_size": 512}
    assert bbn.loop_params(seq, {"lr": 1e-3}, True) == {"lr": 1e-3}       # a fit: cap never enters the constructor


# ------------------------------------------------------------------ trees (bench/speed/forest_speed_arm.py)

@pytest.fixture(scope="module")
def fsa():
    pytest.importorskip("numpy")
    return _mod("bench/speed/forest_speed_arm.py", "alm_forest_speed_arm")


def test_trees_extra(fsa):
    settings = {"numeric_mode": "identical"}
    assert fsa.loop_extra(None, settings) is settings and fsa.loop_extra(None, None) is None
    assert fsa.loop_extra(_plan("trees", "rf"), None) == {"n_estimators": 20}
    assert fsa.loop_extra(_plan("trees", "rf"), settings) == {"numeric_mode": "identical", "n_estimators": 20}
    assert settings == {"numeric_mode": "identical"}
    whole = AB.race_plan("trees", "gbdt-lossguide", caps={})
    assert whole["mode"] == "whole" and fsa.loop_extra(whole, None) is None


def test_trees_loop_reads_after_the_clock_own_arms_only(fsa):
    plan = _plan("trees", "rf")
    reads = fsa.LoopReads(plan, ["ours"])
    fitted = FakeModel({"n_estimators": 20}, loop=(20, 9.0))

    def run_like(fit_context):
        # speed_gbdt_arm.run's shape: the clock and `model` inside the with block, the read on its exit
        for name, r in (("ours", 0), ("xgboost-gpu", 0), ("ours", 1), ("xgboost-gpu", 1)):
            with fit_context(name, r):
                model = fitted
        return model

    run_like(reads)
    assert [r for r, _ in reads.rounds["ours"]] == [0, 1] and "xgboost-gpu" not in reads.rounds
    cell = reads.cell("ours")
    assert cell["mode"] == "capped" and cell["iterations"] == 20 and cell["warmup"]["loop_ms"] == 9.0
    assert reads.cell("xgboost-gpu") is None
