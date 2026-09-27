# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The GBDT task lanes of the benchmark board (ranking, multiclass,
categorical): planning per vendor, the lane configs, the quality functions and
the loaders on tiny data. numpy only; no library is fitted here.

    .pixi/envs/test/bin/python -m pytest tools/test_speed_gbdt_tasks.py
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bench_board as bb                 # noqa: E402
import speed_gbdt_arm as spec            # noqa: E402
import speed_gbdt_rank as rank           # noqa: E402

TASKS = ("gbdt-rank-yetirank", "gbdt-rank-pairlogit", "gbdt-multiclass", "gbdt-categorical")


# --- planning ----------------------------------------------------------------

def test_task_lanes_known_to_driver_and_board():
    assert bb.TREE_TASK_LANES == TASKS
    for lane in TASKS:
        assert lane in spec.LANE_NAMES and lane in spec.TASK_LANES
        assert spec.LANE_DEFAULT_DATASET[lane] in spec.TASK_LANES[lane]["datasets"]
        for driver_ds in bb.TREE_TASK_DATASETS[lane].values():
            assert driver_ds in spec.TASK_LANES[lane]["datasets"], (lane, driver_ds)
            assert driver_ds in spec.DATASET_STORE_KEYS, driver_ds


@pytest.mark.parametrize("vendor", bb.VENDORS)
def test_task_rosters_per_vendor(vendor):
    races = bb.plan_races(vendor, bb.modes_for(vendor), ["trees"], list(TASKS))
    ids = sorted(r["id"] for r in races)
    assert ids == sorted([
        "trees/gbdt-categorical/taxi/rows=full",
        "trees/gbdt-multiclass/istella/rows=full",
        "trees/gbdt-multiclass/taxi/rows=full",
        "trees/gbdt-rank-pairlogit/istella/rows=full",
        "trees/gbdt-rank-yetirank/istella/rows=full"])
    for r in races:
        opp = r["opponents"]
        libs = {o.split("-")[0] for o in opp}
        want = {"catboost", "xgboost"} | (set() if r["lane"] == "gbdt-rank-pairlogit"
                                          else {"lightgbm"})
        assert libs == want, (vendor, r["id"], opp)
        if vendor == "apple":
            assert r["our_arms"] == {"ours": "identical", "ours-ab": "fast"}
            assert all(o.endswith("-cpu") for o in opp)
        else:
            assert r["our_arms"] == {"ours": "identical"}
        if vendor == "nvidia":
            assert all(not o.endswith("-cpu") for o in opp)
        if vendor == "amd":
            assert "xgboost-gpu" in opp and "catboost-cpu" in opp
    # every arm the board plans is one the driver builds for that lane
    for r in races:
        cfg = spec.lane_config(r["lane"], "shipped")
        built = [n for names, _ in spec.opponent_builders(r["lane"], cfg, None, ["cpu", "gpu"])
                 for n in names]
        assert set(r["opponents"]) <= set(built), (r["id"], r["opponents"], built)


def test_task_lanes_follow_requested_datasets_and_map_driver_names():
    races = bb.plan_races("apple", ["fast", "identical"], ["trees"], list(TASKS), ["taxi"])
    assert sorted(r["lane"] for r in races) == ["gbdt-categorical", "gbdt-multiclass"]
    ctx = {"python": "py", "tree_driver": "drv", "vendor": "apple", "rounds": 2,
           "arm_budget_s": 1, "race_deadline_s": 1}
    want = {("gbdt-categorical", "taxi"): "taxicat", ("gbdt-multiclass", "taxi"): "taximc",
            ("gbdt-multiclass", "istella"): "istellamc",
            ("gbdt-rank-yetirank", "istella"): "istellarank"}
    for (lane, ds), drv in want.items():
        race = bb.plan_races("apple", ["fast", "identical"], ["trees"], [lane], [ds])[0]
        cmd, env = bb.tree_cmd(ctx, race)
        assert cmd[cmd.index("--dataset") + 1] == drv
        assert "--ours-ab" in cmd and env["MOJOLEARN_NUMERIC_MODE"] == "identical"
    # the original lanes keep the board dataset as the driver's
    race = bb.plan_races("apple", ["fast", "identical"], ["trees"], ["rf"], ["istella"])[0]
    assert bb.tree_cmd(ctx, race)[0][5:7] == ["--dataset", "istella"]


def test_ranking_needs_the_rank_side_file():
    races = bb.plan_races("nvidia", ["identical"], ["trees"], ["gbdt-rank-yetirank"])
    assert bb.race_data_keys(races[0]) == ["istella", "istella-rank"]
    assert bb.R2_KEYS["istella-rank"] == "gbm-bench/istella/istella_rank.npz"
    pins = open(os.path.join(os.path.dirname(HERE), "bench", "results", "dataset_store",
                             "manifest.tsv")).read()
    assert "gbm-bench/istella/istella_rank.npz\t" in pins
    races = bb.plan_races("nvidia", ["identical"], ["trees"], ["gbdt-multiclass"])
    assert all(bb.race_data_keys(r) == [r["dataset"]] for r in races)


def test_dry_run_lists_task_races_and_rank_data(capsys):
    assert bb.main(["--dry-run", "--vendor", "amd", "--families", "trees",
                    "--lanes", ",".join(TASKS)]) == 0
    text = capsys.readouterr().out
    assert "data istella-rank" in text and "gbm-bench/istella/istella_rank.npz" in text
    assert "TOTAL races=5 cells=24" in text
    assert "INFER cells=48" in text


# --- configs -----------------------------------------------------------------

@pytest.mark.parametrize("lane", TASKS)
def test_task_config_shares_the_gbdt_knobs_and_names_mismatches(lane):
    base = spec.lane_config("gbdt-symmetric", "shipped")
    cfg = spec.lane_config(lane, "shipped")
    for k in ("n_estimators", "max_depth", "learning_rate", "l2", "borders", "seed", "max_leaves"):
        assert cfg[k] == base[k], k
    task = spec.TASK_LANES[lane]
    assert cfg["loss"] == task["loss"] and cfg["task"] == task["task"]
    assert cfg["objectives"]["mojolearn"] == task["loss"]
    assert cfg["mismatches"], lane
    for line in cfg["mismatches"]:
        # one line each, and short enough that FSPEED-NOTE (240) carries it whole
        assert "\n" not in line and "=" not in line and len(line) <= 240, line
    if task["grow_policy"] == "SymmetricTree":
        assert cfg["xgboost_grow_policy"] == "depthwise" and cfg["lightgbm_leafwise"]
    else:
        assert "xgboost_grow_policy" not in cfg


def test_objectives_per_library():
    o = {l: spec.TASK_LANES[l]["objectives"] for l in TASKS}
    assert o["gbdt-rank-yetirank"] == {"mojolearn": "YetiRank", "catboost": "YetiRank",
                                       "xgboost": "rank:ndcg", "lightgbm": "lambdarank"}
    assert o["gbdt-rank-pairlogit"] == {"mojolearn": "PairLogit", "catboost": "PairLogit",
                                        "xgboost": "rank:pairwise"}
    assert o["gbdt-multiclass"]["xgboost"] == "multi:softprob"
    assert o["gbdt-multiclass"]["lightgbm"] == "multiclass"
    assert spec.TASK_LANES["gbdt-categorical"]["grow_policy"] == "Lossguide"
    # the original lanes are untouched
    assert "loss" not in spec.lane_config("gbdt-lossguide", "shipped")


# --- quality functions -------------------------------------------------------

def test_ndcg_matches_the_rank_driver():
    rng = np.random.default_rng(7)
    qid = np.repeat(np.arange(40), rng.integers(1, 30, size=40))
    grades = rng.integers(0, 5, size=qid.size).astype(np.float32)
    grades[qid == 3] = 0                                  # a query with nothing relevant
    scores = np.round(rng.normal(size=qid.size), 1)       # ties on purpose
    b = spec.query_bounds(qid)
    assert np.array_equal(b, rank.query_starts(qid))
    for k in (5, 10):
        assert spec.ndcg_at(k, b, grades, scores) == pytest.approx(
            rank.ndcg_at(k, b, grades, scores), abs=1e-15)
    # rank's own hand-worked example
    y = np.array([3, 0, 1, 0, 0], dtype=np.float32)
    bb_ = spec.query_bounds(np.array([1, 1, 1, 2, 2]))
    assert spec.ndcg_at(10, bb_, y, np.array([3.0, 1.0, 2.0, 0.0, 0.0])) == 1.0


def test_map_by_hand():
    b = spec.query_bounds(np.array([0, 0, 0, 0, 1, 1, 2]))
    grades = np.array([2, 0, 1, 0, 0, 0, 3], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7, 0.1, 0.5, 0.4, 1.0])
    # q0: relevant at ranks 1 and 3 -> (1 + 2/3) / 2; q1: none relevant -> 1; q2: 1
    want = ((1.0 + 2.0 / 3.0) / 2.0 + 1.0 + 1.0) / 3.0
    assert spec.mean_average_precision(b, grades, scores) == pytest.approx(want)
    # a tie is broken pessimistically: the relevant document goes last
    tie = spec.mean_average_precision(spec.query_bounds(np.array([0, 0])),
                                      np.array([1, 0], dtype=np.float32), np.array([0.5, 0.5]))
    assert tie == pytest.approx(0.5)
    names = [m for m, _ in spec.ranking_metrics(b, grades, scores)]
    assert names == ["ndcg10", "ndcg5", "map"]


def test_mlogloss_and_multiclass_scoring():
    y = np.array([0, 2, 1, 2], dtype=np.float32)
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8], [0.3, 0.4, 0.3], [0.0, 1.0, 0.0]])
    want = -(np.log(0.7) + np.log(0.8) + np.log(0.4) + np.log(1e-15)) / 4
    assert spec.mlogloss(y, p) == pytest.approx(want)

    class M(object):
        def predict_proba(self, X):
            return p

    d = spec.Data("t", np.zeros((4, 2)), np.zeros((4, 2)), y, y, "multiclass", 3)
    out = spec.score_sklearn_like(M(), d)
    assert [m for m, _, _ in out] == ["mlogloss", "accuracy"]
    assert out[0][2] is p or np.array_equal(out[0][2], p)       # the hash covers the matrix
    assert out[1][1] == pytest.approx(0.75)


def test_score_ranking_hashes_the_scores():
    d = spec.Data("r", np.zeros((4, 1)), np.zeros((4, 1)), np.zeros(4),
                  np.array([1, 0, 0, 2], dtype=np.float32), "ranking", 0)
    d.bounds_test = spec.query_bounds(np.array([5, 5, 9, 9]))
    out = spec.score_ranking([0.2, 0.1, 0.3, 0.4], d)
    assert [m for m, _, _ in out] == ["ndcg10", "ndcg5", "map"]
    assert out[0][1] == 1.0 and out[0][2].dtype == np.float64
    assert out[1][2] is None and out[2][2] is None


# --- fit equivalence on multiclass --------------------------------------------

def test_fit_verdict_divides_per_class_trees(capsys, monkeypatch):
    shapes = {"ours": dict(library="mojolearn", trees=100, leaves=6400, source="x"),
              "catboost-cpu": dict(library="catboost", trees=100, leaves=6400, source="x"),
              "xgboost-cpu": dict(library="xgboost", trees=400, leaves=25000, source="x")}

    class A(object):
        def __init__(self, name):
            self.name, self.library = name, shapes[name]["library"]

    monkeypatch.setattr(spec, "model_shape", lambda arm, model: dict(shapes[arm.name]))
    arms = [A(n) for n in shapes]
    cfg = dict(n_estimators=100, task="multiclass", n_classes=4)
    out = spec.check_fit_equivalence("gbdt-multiclass", arms, {n: object() for n in shapes}, cfg)
    text = capsys.readouterr().out
    assert out["xgboost-cpu"]["trees"] == 100 and out["xgboost-cpu"]["trees_raw"] == 400
    assert out["xgboost-cpu"]["leaves"] == 6250
    assert "FSPEED-REFUSED" not in text and "verdict=COMPARABLE" in text
    assert "per_class=trees and leaves divided by n_classes 4" in text


# --- loaders on tiny data -----------------------------------------------------

@pytest.fixture
def tiny_data(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    n = 400
    x = rng.normal(size=(n, 16)).astype(np.float32)
    x[:, 0] = rng.choice([1.0, 2.0, 6.0], size=n)
    x[:, 3] = rng.choice([-1.0, 1.0, 2.0, 99.0], size=n)
    x[:, 4] = rng.choice([-1.0, 0.0, 1.0], size=n)
    x[:, 5] = rng.integers(1, 266, size=n)
    x[:, 6] = rng.integers(1, 266, size=n)
    fare = rng.uniform(5, 50, size=n).astype(np.float32)
    tip = (fare * rng.uniform(0, 0.4, size=n)).astype(np.float32)
    card = rng.random(n) < 0.8
    (tmp_path / "taxi").mkdir()
    np.savez(tmp_path / "taxi" / "taxi_speed.npz", x=x, fare=fare, tip=tip, card=card)
    qtr = np.repeat(np.array([30, 10, 20, 11, 50]), [60, 40, 50, 30, 20])
    qte = np.repeat(np.array([7, 3, 9]), [25, 25, 30])
    (tmp_path / "istella").mkdir()
    np.savez(tmp_path / "istella" / "istella_speed.npz",
             x_train=rng.normal(size=(qtr.size, 5)).astype(np.float32),
             r_train=rng.integers(0, 5, size=qtr.size).astype(np.float32),
             x_test=rng.normal(size=(60, 5)).astype(np.float32),
             r_test=rng.integers(0, 5, size=60).astype(np.float32))
    np.savez(tmp_path / "istella" / "istella_rank.npz", qid_train=qtr, qid_test=qte,
             x_test=rng.normal(size=(qte.size, 5)).astype(np.float32),
             r_test=rng.integers(0, 5, size=qte.size).astype(np.float32))
    monkeypatch.setenv("GBM_BENCH_DATA", str(tmp_path))
    monkeypatch.setattr(spec, "TAXI_N_TEST", 50)
    return tmp_path


def test_taxi_multiclass_refines_the_binary_label(tiny_data):
    b = spec.load_dataset("taxi", "shipped")
    m = spec.load_dataset("taximc", "shipped")
    assert m.task == "multiclass" and m.n_classes == 4 and m.name == "taximc"
    assert np.array_equal((m.y_train >= 1).astype(np.float32), b.y_train)
    assert np.array_equal(m.X_train, b.X_train)
    assert set(np.unique(m.y_train)) <= {0.0, 1.0, 2.0, 3.0}


def test_taxi_categorical_codes_are_dense_in_train(tiny_data):
    b = spec.load_dataset("taxi", "shipped")
    c = spec.load_dataset("taxicat", "shipped")
    assert c.cat_idx == spec.TAXI_CAT_COLUMNS and c.task == "binary"
    assert np.array_equal(c.y_train, b.y_train)
    for j in range(16):
        if j in c.cat_idx:
            k = np.unique(c.X_train[:, j])
            assert np.array_equal(k, np.arange(k.size, dtype=np.float32)), j
            assert c.X_test[:, j].max() <= k.size          # k is the unknown bucket
        else:
            assert np.array_equal(c.X_train[:, j], b.X_train[:, j])


def test_istella_multiclass_and_ranking(tiny_data):
    mc = spec.load_dataset("istellamc", "shipped")
    assert mc.task == "multiclass" and mc.n_classes == 5
    r = spec.load_dataset("istellarank", "shipped")
    assert r.task == "ranking" and r.name == "istellarank"
    assert list(r.group_sizes) == [60, 40, 50, 30, 20]
    assert list(np.unique(r.qid_train_seq)) == [0, 1, 2, 3, 4]
    assert np.all(np.diff(r.qid_train_seq.astype(np.int64)) >= 0)   # XGBoost's order
    assert list(r.bounds_test) == [0, 25, 50, 80]
    capped = spec.load_dataset("istellarank", "shipped", rows_cap=120)
    assert capped.X_train.shape[0] == 100                 # cut at a query boundary
