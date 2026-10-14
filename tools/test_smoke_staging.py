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
