# SPDX-License-Identifier: Apache-2.0
"""tools/smoke_inputs.py and the drivers' smoke input mode without a device: the smoke's arrays are the identity
verification's fixture `base`, the drivers' prep builds its blocks from it, the loaders read a slice through their
own code, and a timed run never sees any of it."""
import argparse
import importlib.util
import json
import os
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import smoke_inputs as SI  # noqa: E402


def _mod(name):
    spec = importlib.util.spec_from_file_location("t_si_" + name, str(ROOT / "tools" / (name + ".py")))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def smoke_mode(monkeypatch):
    monkeypatch.setenv(SI.MODE_ENV, SI.FIXTURES_MODE)


# ------------------------------------------------------------------ the fixture is the verification's
def test_a_recipe_takes_the_verify_fixture_and_its_held_out_draw():
    ib = SI._identity_break()
    X, yc, yr = ib.fixture("base")
    Xh = ib.heldout("base")
    for task, want in (("clf", yc), ("reg", yr)):
        got, y, meta = SI.for_recipe("algos", "ridge", "taxi", 2000, task)
        assert np.array_equal(got, X[:2000]) and np.array_equal(meta["Xq"], Xh[:2000])
        assert np.array_equal(y, np.asarray(want[:2000], dtype=np.float32)) and meta["fixture"] == "base"
    X1, y3, _ = SI.for_recipe("algos", "knn", "taxi", None, "mc")
    assert X1.shape == X.shape and set(np.unique(y3)) == {0.0, 1.0, 2.0}
    Xc, _, meta = SI.for_recipe("algos", "cat", "taxi", 2000, "cat")
    assert all(set(np.unique(Xc[:, j])) == set(range(8)) for j in range(Xc.shape[1]))    # dense codes on the fit rows
    with pytest.raises(ValueError):
        SI.for_recipe("algos", "x", "taxi", 10, "rank")


def test_the_stand_ins_answer_what_the_drivers_read():
    ctd = _mod("classical_two_datasets")
    h, knn = SI.prep_sources(SI.FIXTURES_MODE, "real-harness", "real-knn")
    d = h.load_taxi("shipped", regression=True)
    assert ctd._taxi_numeric(h, d.X_train).shape == (20000, 16) and d.X_test.shape == (20000, 16)
    assert np.array_equal(d.y_train, SI.base()[2])
    assert set(np.unique(h.load_taxi("shipped", multiclass=True).y_train)) == {0.0, 1.0, 2.0}
    b = knn.real_block("istella", n_index=3000, n_queries=64)
    assert b["index"].shape == (3000, 16) and b["queries"].shape == (64, 16) and b["d"] == 16
    assert np.array_equal(b["queries"], SI.base()[3][:64])                      # never an index row
    assert SI.prep_sources("datasets", "real-harness", "real-knn") == ("real-harness", "real-knn")
    with pytest.raises(SystemExit):
        SI.prep_sources("other", None, None)
    Y = SI.series("intermittent", 64, 1440)
    assert Y.shape == (64, 1440) and (Y >= 0).all() and (Y == 0).any()
    assert np.abs(SI.series("garch", 8, 100)).max() < 1.4 and (SI.zone_matrix(500) >= 0).all()


# ------------------------------------------------------------------ the drivers' prep on the fixture
def test_the_classical_and_classical2_prep_build_their_blocks_from_the_fixture(tmp_path, monkeypatch):
    monkeypatch.setenv("GBM_BENCH_DATA", str(tmp_path / "no-datasets-here"))
    ctd = _mod("classical_two_datasets")
    ctd.prep(argparse.Namespace(data=str(tmp_path / "ctd"), lanes=",".join(ctd.LANES), datasets="taxi,istella",
                                max_rows=2000, inputs=SI.FIXTURES_MODE))
    more = _mod("bench_board_more")
    more.prep(argparse.Namespace(data=str(tmp_path / "more"), lanes=",".join(more.LANE_ORDER),
                                 datasets="taxi,istella", max_rows=2000, inputs=SI.FIXTURES_MODE))
    X = SI.base()[0]
    for d in ("ctd", "more"):
        recs = [json.loads(p.read_text()) for p in (tmp_path / d).glob("*.json")]
        assert recs and all(r["data_root"] == "verify-fixture:base" for r in recs if "data_root" in r)
    with np.load(tmp_path / "ctd" / "big-taxi.npz") as z:
        assert np.array_equal(z["X"], X[:2000])                                 # the fixture's rows, as they are
    with np.load(tmp_path / "more" / "ivf-istella.npz") as z:
        assert z["index"].shape[1] == 16


def test_the_algos_prep_builds_its_blocks_from_the_fixture_and_only_its_races_datasets(tmp_path, monkeypatch):
    monkeypatch.setenv("GBM_BENCH_DATA", str(tmp_path / "no-datasets-here"))
    monkeypatch.setenv("MOJOLEARN_CORPUS_ROOT", str(tmp_path / "no-corpora-here"))
    A = _mod("bench_board_algos")
    lanes = sorted(l for l in A.LANES if set(A.datasets_of(l)) & {"taxi", "istella", "taxi-hourly", "taxi-zones"}
                   and A.LANES[l]["block"] not in ("dense", "sym", "tensor", "optim", "images", "corpus"))
    # the races' datasets only: these lanes also run on the text corpora, which this group never staged
    A.prep(argparse.Namespace(data=str(tmp_path / "a"), lanes=",".join(lanes), datasets="taxi,istella",
                              max_rows=2000, inputs=SI.FIXTURES_MODE,
                              race_datasets="taxi,istella,taxi-hourly,taxi-zones"))
    names = {p.stem for p in (tmp_path / "a").glob("*.npz")}
    assert {"cls-taxi", "reg-istella", "ts-taxi-hourly", "zones"} <= names and "text" not in names
    with np.load(tmp_path / "a" / "ts-taxi-hourly.npz") as z:
        assert z["Y"].shape[0] == A.TS_SERIES


# ------------------------------------------------------------------ the slices: the same loader code
def _full_files(root):
    rng = np.random.default_rng(3)
    n = 40000
    (root / "higgs").mkdir(parents=True)
    np.savez(root / "higgs" / "higgs_speed.npz", x=rng.standard_normal((n, 28)).astype(np.float32),
             y=(rng.random(n) > 0.5).astype(np.float32))
    (root / "taxi").mkdir()
    card = (np.arange(n) % 3) != 0
    np.savez(root / "taxi" / "taxi_speed.npz", x=rng.standard_normal((n, 16)).astype(np.float32),
             fare=(rng.random(n) * 30 + 1).astype(np.float32), tip=(rng.random(n) * 8).astype(np.float32), card=card)
    (root / "istella").mkdir()
    np.savez(root / "istella" / "istella_speed.npz", x_train=rng.standard_normal((n, 220)).astype(np.float32),
             r_train=rng.integers(0, 5, n).astype(np.float32), x_test=rng.standard_normal((9000, 220)).astype(np.float32),
             r_test=rng.integers(0, 5, 9000).astype(np.float32))
    np.savez(root / "istella" / "istella_rank.npz", qid_train=np.repeat(np.arange(n // 40), 40),
             qid_test=np.repeat(np.arange(9000 // 30), 30), x_test=rng.standard_normal((9000, 220)).astype(np.float32),
             r_test=rng.integers(0, 5, 9000).astype(np.float32))
    return card


def test_the_loaders_read_the_slice_through_their_own_code(tmp_path, monkeypatch, smoke_mode):
    root = tmp_path / "gbm-bench"
    card = _full_files(root)
    for key, form in SI.SLICE_FORMS.items():
        src = root / key.split("/", 1)[1]
        rec = SI.make_slice(key, str(src), str(root / SI.slice_name(key.split("/", 1)[1])))
        assert rec["key"] == SI.slice_key(key) and os.path.getsize(SI.slice_name(str(src))) <= SI.slice_bound_bytes(key)
    monkeypatch.setenv("GBM_BENCH_DATA", str(root))
    H = _mod("speed_gbdt_arm")
    with np.load(root / "taxi" / "taxi_speed.npz") as z:
        fx, cardx = z["x"], z["card"]
    d = H.load_taxi("shipped", rows_cap=2000)
    assert d.X_train.shape == (2000, 16) and np.array_equal(d.X_train, fx[cardx][:2000])
    assert d.X_test.shape[0] == SI.SLICE_TEST and np.array_equal(d.X_test, fx[cardx][-SI.SLICE_TEST:])
    r = H.load_taxi("shipped", rows_cap=2000, regression=True)
    assert np.array_equal(r.X_test, fx[-r.X_test.shape[0]:]) and r.X_train.shape[0] == 2000
    with np.load(root / "higgs" / "higgs_speed.npz") as z:
        hx = z["x"]
    g = H.load_higgs("shipped", rows_cap=2000)
    assert np.array_equal(g.X_train, hx[:2000]) and np.array_equal(g.X_test, hx[-SI.SLICE_TEST:])
    i = H.load_istella("shipped", rows_cap=2000, regression=True)
    assert i.X_train.shape == (2000, 220) and i.X_test.shape == (SI.SLICE_TEST, 220)
    k = H.load_istella_rank_data("shipped", rows_cap=2000)
    assert k.X_train.shape[0] <= 2000 and k.X_test.shape[0] <= SI.SLICE_TEST and k.X_test.shape[0] % 30 == 0
    N = _mod("bench_board_neural")
    assert N.higgs_path() == str(root / "higgs" / "higgs_speed.smoke.npz")


def test_a_corpus_slice_is_its_head_and_corpus_path_finds_it(tmp_path, monkeypatch, smoke_mode):
    corpus = tmp_path / "corpus" / "enwik8"
    corpus.mkdir(parents=True)
    (corpus / "input.txt").write_bytes(b"abc" * 1_000_000)
    SI.make_slice("corpus/enwik8/input.txt", str(corpus / "input.txt"), str(corpus / "input.smoke.txt"))
    assert (corpus / "input.smoke.txt").read_bytes() == (b"abc" * 1_000_000)[:SI.CORPUS_HEAD_BYTES]
    monkeypatch.setenv("MOJOLEARN_CORPUS_ROOT", str(tmp_path))
    A = _mod("bench_board_algos")
    assert A.corpus_path("corpus/enwik8/input.txt") == str(corpus / "input.smoke.txt")


def test_a_timed_run_never_sees_the_slices_or_the_fixture(tmp_path, monkeypatch):
    monkeypatch.delenv(SI.MODE_ENV, raising=False)
    assert SI.loader_path("/d/taxi/taxi_speed.npz") == "/d/taxi/taxi_speed.npz" and not SI.slices_on()
    H = _mod("speed_gbdt_arm")
    assert H._smoke_slice("/d/x.npz") == ("/d/x.npz", False)
    monkeypatch.setenv(SI.MODE_ENV, "bogus")
    with pytest.raises(SystemExit):
        SI.mode()
    monkeypatch.setenv(SI.MODE_ENV, SI.FIXTURES_MODE)
    with pytest.raises(RuntimeError):                      # the smoke never falls back to the full file
        H._smoke_slice(str(tmp_path / "taxi_speed.npz"))
    assert SI.slice_name("a/b.smoke.npz") == "a/b.smoke.npz"


# ------------------------------------------------------------------ the board routes each race
def test_the_board_routes_fixture_races_to_their_own_prep_and_refuses_the_mode_outside_the_smoke(tmp_path,
                                                                                                  monkeypatch):
    B = _mod("bench_board")
    imap = {"algos/ridge/taxi/rows=2000": dict(input="verify fixture", files=["gbm-bench/taxi/taxi_speed.npz"]),
            "algos/adam/synthetic/rows=2000": dict(input="verify fixture", files=[]),
            "classical2/logreg/taxi/rows=2000": dict(input="NO VERIFY LANE", files=["gbm-bench/taxi/taxi_speed.npz"])}
    path = tmp_path / "map.json"
    path.write_text(json.dumps(imap))
    with pytest.raises(SystemExit):
        B.set_inputs(argparse.Namespace(inputs="verify-fixtures", inputs_map=str(path), smoke=False))
    with pytest.raises(SystemExit):
        B.set_inputs(argparse.Namespace(inputs="verify-fixtures", inputs_map=None, smoke=True))
    monkeypatch.setenv(SI.MODE_ENV, "datasets")
    assert B.set_inputs(argparse.Namespace(inputs="verify-fixtures", inputs_map=str(path), smoke=True)) \
        == "verify-fixtures" and os.environ[SI.MODE_ENV] == "verify-fixtures"
    ridge = dict(id="algos/ridge/taxi/rows=2000", family="algos", lane="ridge", dataset="taxi", rows=2000)
    adam = dict(id="algos/adam/synthetic/rows=2000", family="algos", lane="adam", dataset="synthetic", rows=2000)
    ctx = {"algos_data": str(tmp_path / "a"), "out": str(tmp_path), "data_root": str(tmp_path / "d")}
    assert B.fixture_race(ridge) and not B.fixture_race(adam)
    assert B.algos_data_dir(ctx, 2000, ridge).endswith("rows-2000-verify-fixtures")
    assert B.algos_data_dir(ctx, 2000, adam).endswith("rows-2000") and B.race_data_files(ctx, ridge) == []
    assert "identity fixture" in B.race_data_sha(ctx, ridge, {})
    st = B.data_status(str(tmp_path / "d"), ["taxi"])
    assert st["taxi"]["r2_key"] == "gbm-bench/taxi/taxi_speed.smoke.npz" and st["taxi"]["path"].endswith(".smoke.npz")
    cmds = []
    monkeypatch.setattr(B, "run_logged", lambda cmd, env, log, t, nice=0: cmds.append(cmd) or 0)
    ctx.update(python="py", algos_driver="algos.py", nice=0, vendor="nvidia")
    B.ensure_algos_prep(ctx, [ridge, adam])
    fx = [c for c in cmds if "--inputs" in c]
    assert len(cmds) == 2 and len(fx) == 1 and fx[0][fx[0].index("--data") + 1].endswith("-verify-fixtures")
    assert all(c[c.index("--race-datasets") + 1] in ("taxi", "synthetic") for c in cmds)
    B.INPUT_MAP.clear()
