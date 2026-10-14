"""smoke-fixes r12: the in-process smoke loads each input once (tools/smoke_inproc.py staged/sha256_array, the
algos driver's params-only staging, the smoke worker's stage-data op). The staged values are the builders' own
(same functions, same seeds, same bytes), so the parameter records equal the timed path's."""
import hashlib
import importlib.util
import json
import os

import numpy as np
import pytest

import smoke_inproc as SI
import smoke_worker as SW

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location("t_staging_" + name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A = _load("bench_board_algos")


@pytest.fixture
def in_worker(monkeypatch):
    """This test process plays the smoke worker, with an empty staged-input cache and hash memo."""
    monkeypatch.setenv(SI.ENV_FLAG, str(os.getpid()))
    monkeypatch.setattr(SI, "STAGED", SI.StagedInputs())
    monkeypatch.setattr(SI, "HASHES", SI._HashMemo())
    return SI


@pytest.fixture
def timed(monkeypatch):
    """A timed worker: the flag is absent (or another process's)."""
    monkeypatch.setenv(SI.ENV_FLAG, str(os.getpid() + 1))
    return SI


def _counting(monkeypatch, mod, name):
    calls = []
    real = getattr(mod, name)

    def wrapped(*a, **k):
        calls.append(a)
        return real(*a, **k)
    monkeypatch.setattr(mod, name, wrapped)
    return calls


# ---------------------------------------------------------------------------
# the cache itself
# ---------------------------------------------------------------------------

def test_staged_inputs_keep_the_budget_and_count_hits():
    st = SI.StagedInputs(budget_bytes=3000)
    a = st.get(("a",), lambda: {"x": np.zeros(250, np.float32)})          # 1000 bytes
    assert st.get(("a",), lambda: pytest.fail("built twice")) is a
    st.get(("b",), lambda: [np.zeros(250, np.float32)] * 2)              # the same array once: 1000 bytes
    st.get(("c",), lambda: (np.zeros(500, np.float32),))                 # 2000 bytes: "a" goes first
    assert list(st.entries) == [("b",), ("c",)] and st.bytes == 3000
    rep = st.report()
    assert rep["hits"] == 1 and rep["misses"] == 3 and rep["evictions"] == 1
    big = st.get(("d",), lambda: np.zeros(10_000, np.float32))           # over budget alone: still kept
    assert big.shape == (10_000,) and list(st.entries) == [("d",)]


def test_staged_is_a_plain_build_outside_the_worker(timed):
    calls = []
    for _ in range(2):
        SI.staged(("k",), lambda: calls.append(1) or 7)
    assert calls == [1, 1]
    assert not SI.STAGED.entries or ("k",) not in SI.STAGED.entries


def test_sha256_array_is_hashlibs_digest_once_per_live_array(in_worker):
    a = np.arange(1000, dtype=np.float32)
    want = hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
    assert SI.sha256_array(a) == want and SI.sha256_array(a) == want
    assert SI.HASHES.hits == 1 and SI.HASHES.misses == 1
    v = a[::2]                                                           # another object, its own bytes
    assert SI.sha256_array(v) == hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
    del a, v
    assert SI.HASHES.report()["live"] == 0                               # entries go with their arrays


# ---------------------------------------------------------------------------
# the algos driver's params-only staging
# ---------------------------------------------------------------------------

def _block(tmp_path, monkeypatch):
    rng = np.random.default_rng(3)
    arrays = dict(X=rng.standard_normal((60, 4)).astype(np.float32), y=rng.standard_normal(60).astype(np.float32),
                  Xq=rng.standard_normal((12, 4)).astype(np.float32), yq=rng.standard_normal(12).astype(np.float32))
    np.savez(tmp_path / "blk.npz", **arrays)
    (tmp_path / "blk.json").write_text(json.dumps({"block": "reg", "dataset": "istella"}))
    monkeypatch.setattr(A, "block_file", lambda lane, ds: "blk")
    return arrays


def test_lane_inputs_read_the_block_once_and_equal_the_timed_build(tmp_path, monkeypatch):
    _block(tmp_path, monkeypatch)
    monkeypatch.setenv(SI.ENV_FLAG, str(os.getpid() + 1))
    B0, rec0 = A._load_block("lstsq", "istella", str(tmp_path))
    D0 = A.lane_arrays("lstsq", B0)
    monkeypatch.setenv(SI.ENV_FLAG, str(os.getpid()))
    monkeypatch.setattr(SI, "STAGED", SI.StagedInputs())
    reads = _counting(monkeypatch, A, "_read_block")
    got = [A._lane_inputs("lstsq", "istella", str(tmp_path)) for _ in range(3)]
    assert len(reads) == 1                                              # conductor + two arms: one read
    for B, rec, D in got:
        assert rec == rec0 and sorted(D) == sorted(D0)
        for k in D0:
            assert D[k].dtype == D0[k].dtype and D[k].tobytes() == D0[k].tobytes()
    assert got[0][2] is not got[1][2] and got[0][2]["X"] is got[1][2]["X"]   # new dicts, shared arrays
    got[0][1]["block"] = "changed"                                      # a record is each caller's own
    assert A._lane_inputs("lstsq", "istella", str(tmp_path))[1] == rec0
    assert A._DATASET == "istella"


def test_optim_inputs_once_with_the_timed_records(in_worker, monkeypatch):
    P, STEPS = A._optim_size({"_cap": 64})
    monkeypatch.setenv(SI.ENV_FLAG, str(os.getpid() + 1))
    p_t, g_t, rec_t = A._optim_inputs(P, STEPS)                         # the timed path's build
    monkeypatch.setenv(SI.ENV_FLAG, str(os.getpid()))
    made = _counting(monkeypatch, A, "_make_optim_inputs")
    first, second = A._optim_inputs(P, STEPS), A._optim_inputs(P, STEPS)
    assert len(made) == 1
    assert first[2] == rec_t == second[2]                               # byte-identical input-array records
    assert first[0].tobytes() == p_t.tobytes() and [g.tobytes() for g in first[1]] == [g.tobytes() for g in g_t]
    assert first[0] is second[0] and first[1] is not second[1]          # shared arrays, own lists
    assert rec_t[0]["sha256"] == hashlib.sha256(p_t.tobytes()).hexdigest()


def test_linalg_systems_once_per_process(in_worker, monkeypatch):
    built = _counting(monkeypatch, A, "sym_system")
    a1, = A._linalg_args("cholesky", {"_cap": 256})
    a2, = A._linalg_args("cholesky", {"_cap": 256})
    assert len(built) == 1 and a1 is a2
    monkeypatch.setenv(SI.ENV_FLAG, str(os.getpid() + 1))
    assert A._linalg_args("cholesky", {"_cap": 256})[0].tobytes() == a1.tobytes()


def test_shared_ridge_once_per_staged_pair(in_worker, monkeypatch):
    rng = np.random.default_rng(5)
    X, y = rng.standard_normal((40, 3)).astype(np.float32), rng.standard_normal(40).astype(np.float32)
    solved = _counting(monkeypatch, A, "_shared_ridge_f32")
    w1, b1 = A._staged_shared_ridge(X, y)
    w2, b2 = A._staged_shared_ridge(X, y)
    assert len(solved) == 1 and w1.tobytes() == w2.tobytes() and b1 == b2 and w1 is not w2
    w0, b0 = A._shared_ridge_f32(X, y)
    assert w0.tobytes() == w1.tobytes() and b0 == b1


def test_stage_inputs_is_a_no_op_outside_the_worker(timed, tmp_path):
    assert A.stage_inputs("adam", "synthetic", str(tmp_path)) == {"status": "not in the smoke worker"}


def test_stage_inputs_stages_what_the_arms_then_read(in_worker, tmp_path, monkeypatch):
    _block(tmp_path, monkeypatch)
    got = A.stage_inputs("lstsq", "istella", str(tmp_path))
    assert got["status"] == "ok" and "linalg inputs + digests" in got["staged"]
    reads = _counting(monkeypatch, A, "_read_block")
    _, _, D = A._lane_inputs("lstsq", "istella", str(tmp_path))
    assert reads == []
    hits = SI.HASHES.hits
    A._array_sha256(D["X"])
    assert SI.HASHES.hits == hits + 1                                   # hashed during staging, not again


def test_params_probe_only_names_the_worker_only(monkeypatch):
    monkeypatch.setenv(SI.ENV_FLAG, str(os.getpid()))
    assert A._params_probe_only()
    monkeypatch.setenv(SI.ENV_FLAG, str(os.getpid() + 1))
    assert not A._params_probe_only()
    monkeypatch.delenv(SI.ENV_FLAG)
    assert not A._params_probe_only() and A._staging() is None


# ---------------------------------------------------------------------------
# the worker's stage-data op
# ---------------------------------------------------------------------------

def test_stage_items_records_each_item(in_worker, tmp_path):
    good = tmp_path / "good_driver.py"
    good.write_text("def stage_inputs(lane, dataset, data):\n    return {'status': 'ok', 'lane': lane}\n")
    bad = tmp_path / "bad_driver.py"
    bad.write_text("def stage_inputs(lane, dataset, data):\n    raise ValueError('no block')\n")
    none = tmp_path / "none_driver.py"
    none.write_text("X = 1\n")
    got = SI.stage_items([dict(driver=str(p), lane="adam", dataset="synthetic", data=str(tmp_path))
                          for p in (good, bad, none)])
    assert [i["status"] for i in got["items"]] == ["ok", "failed", "absent"]
    assert got["items"][0]["staged"] == {"status": "ok", "lane": "adam"}
    assert "no block" in got["items"][1]["error"]
    assert set(got["report"]) == {"staged", "hashes"}


def test_stage_data_op_without_arm_servers(in_worker, tmp_path):
    drv = tmp_path / "d_driver.py"
    drv.write_text("def stage_inputs(lane, dataset, data):\n    return {'status': 'ok'}\n")
    got = SW.stage_data([dict(driver=str(drv), lane="sgd", dataset="synthetic", data=str(tmp_path))], [])
    assert got["items"][0]["status"] == "ok" and got["arm_servers"] == {}


# ---------------------------------------------------------------------------
# smoke-fixes r14: pinned blocks, a budget sized from free memory, every family staged
# ---------------------------------------------------------------------------

CTD = _load("classical_two_datasets")
MORE = _load("bench_board_more")
SPEC = _load("speed_gbdt_arm")


def test_pinned_entries_are_never_evicted_and_shrink_the_lru():
    st = SI.StagedInputs(budget_bytes=4000)
    blk = st.get(("blk",), lambda: np.zeros(500, np.float32), pin=True)          # 2000 bytes, pinned
    st.get(("a",), lambda: np.zeros(250, np.float32))                             # 1000
    st.get(("b",), lambda: np.zeros(300, np.float32))                             # 1200: over 4000 - 2000
    assert list(st.entries) == [("b",)] and st.evictions == 1
    assert st.get(("blk",), lambda: pytest.fail("pinned block rebuilt")) is blk
    st.get(("big",), lambda: np.zeros(1000, np.float32), pin=True)               # 4000: the pins would exceed
    rep = st.report()
    assert rep["pinned"] == 1 and rep["pinned_bytes"] == 2000 and rep["pin_declined"] == 1
    assert rep["budget"] == 4000 and rep["budget_source"] == "explicit" and rep["lru_budget"] == 2000


def test_budget_is_sized_from_free_memory_unless_named(monkeypatch):
    monkeypatch.delenv(SI.STAGE_BUDGET_ENV, raising=False)
    monkeypatch.setattr(SI, "free_memory_bytes", lambda: (10_000, "MemAvailable"))
    st = SI.StagedInputs()
    assert st.budget is None                                                      # sized at first use
    st.get(("a",), lambda: np.zeros(10, np.float32))
    rep = st.report()
    assert rep["budget"] == 5000 and rep["free_bytes_at_sizing"] == 10_000 and "MemAvailable" in rep["budget_source"]
    monkeypatch.setenv(SI.STAGE_BUDGET_ENV, "2")
    st = SI.StagedInputs()
    assert st.budget == 2 << 30 and st.budget_source == SI.STAGE_BUDGET_ENV


def test_free_memory_bytes_reads_a_number_here():
    free, source = SI.free_memory_bytes()
    assert (free is None) == (source == "free memory unreadable")
    assert free is None or free > 0


def _npz(path, **arrays):
    np.savez(path, **arrays)
    return {k: np.ascontiguousarray(v) for k, v in arrays.items()}


def test_staged_npz_reads_once_and_hands_new_dicts(in_worker, tmp_path, monkeypatch):
    want = _npz(tmp_path / "b.npz", X=np.arange(12, dtype=np.float32).reshape(3, 4), y=np.arange(3.0))
    loads = _counting(monkeypatch, np, "load")
    one, two = SI.staged_npz(str(tmp_path / "b.npz")), SI.staged_npz(str(tmp_path / "b.npz"))
    assert len(loads) == 1 and one is not two and one["X"] is two["X"]
    assert {k: v.tobytes() for k, v in one.items()} == {k: v.tobytes() for k, v in want.items()}
    assert SI.STAGED.report()["pinned"] == 1


def test_staged_npz_reads_every_time_outside_the_worker(timed, tmp_path, monkeypatch):
    _npz(tmp_path / "b.npz", X=np.zeros(3))
    loads = _counting(monkeypatch, np, "load")
    SI.staged_npz(str(tmp_path / "b.npz"))
    SI.staged_npz(str(tmp_path / "b.npz"))
    assert len(loads) == 2


def test_memo_on_builds_once_per_live_input(in_worker):
    a, b = np.arange(10.0), np.arange(10.0)
    calls = []
    f = lambda x: SI.memo_on("t", (x,), lambda: calls.append(1) or x * 2)       # noqa: E731
    r1, r2, r3 = f(a), f(a), f(b)
    assert len(calls) == 2 and r1 is r2 and r3 is not r1
    assert SI.STAGED.bytes == r1.nbytes + r3.nbytes                             # the derived bytes only


def test_memo_on_is_a_plain_build_outside_the_worker(timed):
    calls = []
    a = np.arange(3.0)
    SI.memo_on("t", (a,), lambda: calls.append(1))
    SI.memo_on("t", (a,), lambda: calls.append(1))
    assert calls == [1, 1]


def test_sha256_is_memoized_per_owner_across_view_objects(in_worker):
    a = np.arange(1000, dtype=np.float32)
    want = hashlib.sha256(a.tobytes()).hexdigest()
    assert SI.sha256_array(a[:]) == want and SI.sha256_array(a[:]) == want      # two view objects, one hash
    assert SI.HASHES.hits == 1 and SI.HASHES.misses == 1
    assert SI.sha256_array(a[:10]) == hashlib.sha256(a[:10].tobytes()).hexdigest()  # another view: its own
    assert SI.HASHES.misses == 2


def _ctd_block(tmp_path, lane="kmeans", ds="istella"):
    rng = np.random.default_rng(4)
    name = "%s-%s" % (CTD.BLOCK_OF[lane], ds)
    want = _npz(tmp_path / (name + ".npz"), X=rng.standard_normal((50, 3)).astype(np.float32))
    (tmp_path / (name + ".json")).write_text(json.dumps({"block": CTD.BLOCK_OF[lane], "dataset": ds}))
    return want


def test_classical_block_is_read_once_in_the_worker(in_worker, tmp_path, monkeypatch):
    want = _ctd_block(tmp_path)
    got = CTD.stage_inputs("kmeans", "istella", str(tmp_path))
    assert got["status"] == "ok" and got["arrays"] == ["X"]
    loads = _counting(monkeypatch, CTD.np, "load")
    data = CTD.read_block_arrays(str(tmp_path / "big-istella.npz"))
    assert loads == [] and data["X"].tobytes() == want["X"].tobytes()


def test_classical_block_outside_the_worker(timed, tmp_path):
    _ctd_block(tmp_path)
    assert CTD.stage_inputs("kmeans", "istella", str(tmp_path)) == {"status": "not in the smoke worker"}
    a = CTD.read_block_arrays(str(tmp_path / "big-istella.npz"))
    b = CTD.read_block_arrays(str(tmp_path / "big-istella.npz"))
    assert a["X"] is not b["X"] and a["X"].tobytes() == b["X"].tobytes()


def test_classical2_block_and_lane_arrays_once_and_equal_the_timed_build(tmp_path, monkeypatch):
    lane, ds = "knn-clf", "istella"
    rng = np.random.default_rng(6)
    name = "%s-%s" % (MORE.block_of(lane), ds)
    _npz(tmp_path / (name + ".npz"), X=rng.standard_normal((40, 3)).astype(np.float32),
         y=rng.integers(0, 2, 40).astype(np.float32), Xq=rng.standard_normal((9, 3)).astype(np.float32),
         yq=rng.integers(0, 2, 9).astype(np.float32))
    (tmp_path / (name + ".json")).write_text(json.dumps({"block": MORE.block_of(lane), "dataset": ds}))
    with np.load(tmp_path / (name + ".npz")) as z:
        D0 = MORE.lane_arrays(lane, {k: np.ascontiguousarray(z[k]) for k in z.files}, {})
    monkeypatch.setenv(SI.ENV_FLAG, str(os.getpid()))
    monkeypatch.setattr(SI, "STAGED", SI.StagedInputs())
    assert MORE.stage_inputs(lane, ds, str(tmp_path))["status"] == "ok"
    subs = _counting(monkeypatch, MORE, "lane_arrays")
    loads = _counting(monkeypatch, np, "load")
    B, rec, D = MORE._staged_block_inputs(SI, lane, str(tmp_path / name))
    _, _, D2 = MORE._staged_block_inputs(SI, lane, str(tmp_path / name))
    assert subs == [] and loads == [] and D is not D2 and D["X"] is D2["X"]
    assert {k: v.tobytes() for k, v in D.items()} == {k: v.tobytes() for k, v in D0.items()}


def _istella(tmp_path, monkeypatch):
    folder = tmp_path / "istella"
    folder.mkdir()
    rng = np.random.default_rng(8)
    _npz(folder / "istella_speed.npz", x_train=rng.standard_normal((30, 4)).astype(np.float32),
         r_train=rng.integers(0, 5, 30).astype(np.float32), x_test=rng.standard_normal((7, 4)).astype(np.float32),
         r_test=rng.integers(0, 5, 7).astype(np.float32))
    monkeypatch.setenv("GBM_BENCH_DATA", str(tmp_path))


def _data_bytes(d):
    return {k: v.tobytes() for k, v in vars(d).items() if hasattr(v, "tobytes")}


def test_trees_dataset_once_per_worker_shared_by_its_forms(tmp_path, monkeypatch):
    _istella(tmp_path, monkeypatch)
    monkeypatch.setenv(SI.ENV_FLAG, str(os.getpid() + 1))
    timed_bin = SPEC.load_with_fallback("istella", "shipped", None)
    timed_reg = SPEC.load_with_fallback("istellareg", "shipped", None)
    monkeypatch.setenv(SI.ENV_FLAG, str(os.getpid()))
    monkeypatch.setattr(SI, "STAGED", SI.StagedInputs())
    assert SPEC.stage_inputs("rf", "istella", dict(size="shipped", rows=None, arms=["xgboost-gpu"]))["status"] == "ok"
    loads = _counting(monkeypatch, SPEC.np, "load")
    one = SPEC.load_with_fallback("istella", "shipped", None)
    two = SPEC.load_with_fallback("istella", "shipped", None)
    reg = SPEC.load_with_fallback("istellareg", "shipped", None)               # another form: the pinned npz
    assert loads == []
    assert one is not two and one.X_train is two.X_train
    assert _data_bytes(one) == _data_bytes(timed_bin) and _data_bytes(reg) == _data_bytes(timed_reg)
    one._catboost_X_train = "this run's own"
    assert not hasattr(SPEC.load_with_fallback("istella", "shipped", None), "_catboost_X_train")
    assert SI.STAGED.report()["pinned"] >= 2                                    # the npz and the Data


def test_trees_stage_is_a_no_op_outside_the_worker(timed):
    assert SPEC.stage_inputs("rf", "istella", {}) == {"status": "not in the smoke worker"}


def test_catboost_frame_once_per_staged_matrix(in_worker, monkeypatch):
    pytest.importorskip("pandas")
    x = np.array([[0, 1.5], [2, 2.5]], dtype=np.float32)
    built = _counting(monkeypatch, SPEC, "catboost_cat_frame")
    f1, f2 = SPEC.catboost_cat_frame_once(x, (0,)), SPEC.catboost_cat_frame_once(x, (0,))
    assert len(built) == 1 and f1 is f2


def test_count_csr_and_als_split_once_per_staged_array(in_worker):
    sp = pytest.importorskip("scipy.sparse")
    X = np.array([[0, 2, 1], [3, 0, 0], [1, 1, 4]], dtype=np.float32)
    c1, c2 = A._count_csr(X), A._count_csr(X)
    assert c1 is c2 and (c1 != sp.csr_matrix(X)).nnz == 0
    s1, s2 = A._staged_als_split(X), A._staged_als_split(X)
    C0, held0 = A._als_split(X)
    assert s1 is s2 and (s1[0] != C0).nnz == 0 and s1[1].tobytes() == held0.tobytes()


def test_cross_entropy_inputs_once_with_the_timed_bytes(in_worker, monkeypatch):
    E = A._EXTRA
    monkeypatch.setattr(E, "CE_ROWS", 16)
    monkeypatch.setattr(E, "CE_CLASSES", 8)
    drawn = _counting(monkeypatch, E, "_ce_inputs")
    x1, t1 = E._ce_inputs_once(np, 7, vars(A))
    x2, t2 = E._ce_inputs_once(np, 7, vars(A))
    assert len(drawn) == 1 and x1 is x2
    rng = np.random.default_rng(7)
    assert x1.tobytes() == rng.standard_normal((16, 8)).astype(np.float32).tobytes()
    assert t1.tobytes() == rng.integers(0, 8, 16).astype(np.int64).tobytes()


# ---------------------------------------------------------------------------
# smoke-fixes r17: the pre-binned block of the XGBoost pairing (gbdt-depthwise / gbdt-lossguide)
# ---------------------------------------------------------------------------

def _fake_borders(x_train, cfg):
    """Stands in for CatBoost's Pool.quantize (not installed here): per-feature quantile borders."""
    x = np.asarray(x_train, dtype=np.float32)
    return [np.unique(np.quantile(x[:, f], [0.25, 0.5, 0.75]).astype(np.float32)) for f in range(x.shape[1])], "fake"


@pytest.mark.parametrize("lane,arm", [("gbdt-depthwise", "xgboost-gpu"), ("gbdt-lossguide", "ours-gain0")])
def test_prebinned_block_staged_once_and_equal_to_the_timed_record(tmp_path, monkeypatch, lane, arm):
    _istella(tmp_path, monkeypatch)
    monkeypatch.setattr(SPEC, "greedylogsum_borders", _fake_borders)
    cfg = SPEC.lane_config(lane, "shipped")
    monkeypatch.setenv(SI.ENV_FLAG, str(os.getpid() + 1))                     # the timed path
    timed_pb = SPEC.prebinned_data(lane, SPEC.load_with_fallback("istella", "shipped", None), cfg)
    monkeypatch.setenv(SI.ENV_FLAG, str(os.getpid()))
    monkeypatch.setattr(SI, "STAGED", SI.StagedInputs())
    calls = _counting(monkeypatch, SPEC, "greedylogsum_borders")
    got = SPEC.stage_inputs(lane, "istella", dict(size="shipped", rows=None, arms=[arm]))
    assert got["status"] == "ok" and any(timed_pb.bins_sha256 in s for s in got["staged"])
    assert len(calls) == 1
    one = SPEC.prebinned_data(lane, SPEC.load_with_fallback("istella", "shipped", None), cfg)
    two = SPEC.prebinned_data(lane, SPEC.load_with_fallback("istella", "shipped", None), cfg)
    assert len(calls) == 1                                                       # the recipes only read it
    assert one is not two and one.X_train is two.X_train
    for pb in (one, two):
        assert pb.input_binning == timed_pb.input_binning and pb.bins_sha256 == timed_pb.bins_sha256
        assert pb.X_train.tobytes() == timed_pb.X_train.tobytes()
        assert pb.X_test.tobytes() == timed_pb.X_test.tobytes()
    assert SI.STAGED.report()["pinned"] >= 3                                    # the npz, the Data, the block


def test_prebinned_block_not_staged_without_a_prebinned_arm(in_worker, tmp_path, monkeypatch):
    _istella(tmp_path, monkeypatch)
    calls = _counting(monkeypatch, SPEC, "greedylogsum_borders")
    got = SPEC.stage_inputs("gbdt-depthwise", "istella", dict(size="shipped", rows=None, arms=["catboost-gpu"]))
    assert got["status"] == "ok" and calls == []
    SPEC.stage_inputs("rf", "istella", dict(size="shipped", rows=None, arms=["xgboost-gpu"]))
    assert calls == []


def test_prebinned_block_is_built_every_time_outside_the_worker(timed, tmp_path, monkeypatch):
    _istella(tmp_path, monkeypatch)
    monkeypatch.setattr(SPEC, "greedylogsum_borders", _fake_borders)
    calls = _counting(monkeypatch, SPEC, "greedylogsum_borders")
    cfg = SPEC.lane_config("gbdt-depthwise", "shipped")
    a = SPEC.prebinned_data("gbdt-depthwise", SPEC.load_with_fallback("istella", "shipped", None), cfg)
    b = SPEC.prebinned_data("gbdt-depthwise", SPEC.load_with_fallback("istella", "shipped", None), cfg)
    assert len(calls) == 2 and a.bins_sha256 == b.bins_sha256 and a.X_train is not b.X_train
