# SPDX-License-Identifier: Apache-2.0
"""tools/smoke_inputs.py and the drivers' smoke input mode without a device: the smoke's arrays are the identity
verification's fixture `base`, the drivers' prep builds its blocks from it, the loaders read a synthetic file in
their own schema generated on the box, and a timed run never sees any of it."""
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
def smoke_mode(monkeypatch, tmp_path):
    monkeypatch.setenv(SI.MODE_ENV, SI.SYNTHETIC_MODE)
    monkeypatch.setenv(SI.SYNTH_DIR_ENV, str(tmp_path / "synthetic"))
    return tmp_path / "synthetic"


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


# ------------------------------------------------------------------ synthetic files: the same loader code
def test_the_loaders_read_a_synthetic_file_in_their_own_schema_through_their_own_code(tmp_path, monkeypatch,
                                                                                         smoke_mode):
    monkeypatch.setenv("GBM_BENCH_DATA", str(tmp_path / "no-datasets-here"))     # nothing staged, nothing read
    H = _mod("speed_gbdt_arm")
    d = H.load_taxi("shipped", rows_cap=2000)
    assert d.X_train.shape == (2000, 16) and d.X_test.shape == (SI.SMOKE_TEST_ROWS, 16)
    assert set(np.unique(d.y_train)) == {0.0, 1.0}
    with np.load(smoke_mode / "taxi" / "taxi_speed.npz") as z:
        assert sorted(z.files) == sorted(["x", "fare", "tip", "card", SI.TAIL]) and z["card"].dtype == bool
        assert np.array_equal(d.X_test, z["x"][z["card"]][-SI.SMOKE_TEST_ROWS:])
    assert len(set(np.unique(H.load_taxi("shipped", rows_cap=2000, multiclass=True).y_train))) == 4
    c = H.load_taxi("shipped", rows_cap=2000, categorical=True)
    assert c.cat_idx and c.X_train.shape[0] == 2000
    g = H.load_higgs("shipped", rows_cap=2000)
    assert g.X_train.shape == (2000, 28) and g.X_test.shape == (SI.SMOKE_TEST_ROWS, 28)
    i = H.load_istella("shipped", rows_cap=2000, multiclass=True)
    assert i.X_train.shape == (2000, 220) and i.X_test.shape == (SI.SMOKE_TEST_ROWS, 220)
    k = H.load_istella_rank_data("shipped", rows_cap=2000)
    assert 0 < k.X_train.shape[0] <= 2000 and k.X_test.shape[0] % SI.RANK_QUERY_ROWS == 0
    assert all(os.path.getsize(p) <= SI.synthetic_bound_bytes(SI._schema_of(p)) for p in smoke_mode.rglob("*.npz"))
    N = _mod("bench_board_neural")
    assert N.higgs_path() == str(smoke_mode / "higgs" / "higgs_speed.npz")
    # deterministic: the same bytes on every box
    first = (smoke_mode / "istella" / "istella_speed.npz").read_bytes()
    (smoke_mode / "istella" / "istella_speed.npz").unlink()
    H.load_istella("shipped", rows_cap=2000)
    assert (smoke_mode / "istella" / "istella_speed.npz").read_bytes() == first


def test_the_corpus_readers_get_generated_text_of_their_kind(monkeypatch, smoke_mode):
    A = _mod("bench_board_algos")
    p = A.corpus_path("corpus/enwik8/input.txt")
    q = A.corpus_path("corpus/pile_github/input.txt")
    assert p == str(smoke_mode / "enwik8" / "input.txt") and os.path.getsize(p) == SI.CORPUS_SYNTH_BYTES
    assert open(p, "rb").read() != open(q, "rb").read() and b"def" in open(q, "rb").read()
    X = A.text_counts(p)
    assert X.shape[0] == SI.CORPUS_SYNTH_BYTES // A.TEXT_DOC_BYTES >= 20       # both labels reach the held-out rows


def test_a_timed_run_never_sees_the_synthetic_files_or_the_fixture(tmp_path, monkeypatch):
    monkeypatch.delenv(SI.MODE_ENV, raising=False)
    assert SI.loader_path("/d/taxi/taxi_speed.npz") == "/d/taxi/taxi_speed.npz" and not SI.synthetic_on()
    H = _mod("speed_gbdt_arm")
    assert H._smoke_input("/d/x.npz") == ("/d/x.npz", False)
    monkeypatch.setenv(SI.MODE_ENV, "bogus")
    with pytest.raises(SystemExit):
        SI.mode()
    monkeypatch.setenv(SI.MODE_ENV, SI.SYNTHETIC_MODE)
    monkeypatch.delenv(SI.SYNTH_DIR_ENV, raising=False)
    with pytest.raises(SystemExit):                      # never a default place, never the full file
        H._smoke_input(str(tmp_path / "taxi" / "taxi_speed.npz"))
    with pytest.raises(SystemExit):
        SI.synthetic_path("/d/criteo/criteo.npz", {SI.SYNTH_DIR_ENV: str(tmp_path)})


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
        B.set_inputs(argparse.Namespace(inputs="synthetic", inputs_map=str(path), smoke=False, out=str(tmp_path)))
    with pytest.raises(SystemExit):
        B.set_inputs(argparse.Namespace(inputs="synthetic", inputs_map=None, smoke=True, out=str(tmp_path)))
    monkeypatch.setenv(SI.MODE_ENV, "datasets")
    monkeypatch.setenv(SI.SYNTH_DIR_ENV, "")
    out = tmp_path / "b000-smoke"
    assert B.set_inputs(argparse.Namespace(inputs="synthetic", inputs_map=str(path), smoke=True, out=str(out))) \
        == "synthetic" and os.environ[SI.MODE_ENV] == "synthetic"
    assert os.environ[SI.SYNTH_DIR_ENV] == str(out / "work" / "smoke-synthetic")   # under the race scratch
    ridge = dict(id="algos/ridge/taxi/rows=2000", family="algos", lane="ridge", dataset="taxi", rows=2000)
    adam = dict(id="algos/adam/synthetic/rows=2000", family="algos", lane="adam", dataset="synthetic", rows=2000)
    ctx = {"algos_data": str(tmp_path / "a"), "out": str(tmp_path), "data_root": str(tmp_path / "d")}
    assert B.fixture_race(ridge) and not B.fixture_race(adam)
    assert B.algos_data_dir(ctx, 2000, ridge).endswith("rows-2000-verify-fixtures")
    assert B.algos_data_dir(ctx, 2000, adam).endswith("rows-2000") and B.race_data_files(ctx, ridge) == []
    assert "identity fixture" in B.race_data_sha(ctx, ridge, {})
    gru = dict(id="neural/gru-fit/higgs/shape=small", family="neural", lane="gru-fit", dataset="higgs")
    assert B.race_data_files(ctx, gru) == [("higgs", str(out / "work" / "smoke-synthetic" / "higgs" / "higgs_speed.npz"))]
    cmds = []
    monkeypatch.setattr(B, "run_logged", lambda cmd, env, log, t, nice=0: cmds.append(cmd) or 0)
    ctx.update(python="py", algos_driver="algos.py", nice=0, vendor="nvidia")
    B.ensure_algos_prep(ctx, [ridge, adam])
    fx = [c for c in cmds if "--inputs" in c]
    assert len(cmds) == 2 and len(fx) == 1 and fx[0][fx[0].index("--data") + 1].endswith("-verify-fixtures")
    assert all(c[c.index("--race-datasets") + 1] in ("taxi", "synthetic") for c in cmds)
    B.INPUT_MAP.clear()
