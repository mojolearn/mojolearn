"""smoke-fixes r16 (2026-10-14; Andrew: "slow" = broken until proven otherwise): the parameter probe skips the
work its read-back never uses (weights built, loaded or compared, device copies, cache-filling forwards, the
own-tier arm server started inside a recipe's timer). Each test builds the arm the timed way and the probe way
and checks the parameter records are identical, and that the skipped work is skipped only in the probe.

The torch tests run where torch (and transformers / torch_geometric) import, on the CPU only (the arms' device
is forced to the CPU; nothing touches a GPU); elsewhere they skip."""
import importlib.util
import os
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import smoke_inproc as SI
import smoke_worker as SW

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location("t_r16_" + name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load("bench_board_params")


@pytest.fixture
def in_worker(monkeypatch):
    """This test process plays the smoke worker (the parameter probe), with empty staging caches."""
    monkeypatch.setenv(SI.ENV_FLAG, str(os.getpid()))
    monkeypatch.setattr(SI, "STAGED", SI.StagedInputs())
    monkeypatch.setattr(SI, "HASHES", SI._HashMemo())
    return SI


def _timed(monkeypatch):
    monkeypatch.setenv(SI.ENV_FLAG, str(os.getpid() + 1))


def _probe(monkeypatch):
    monkeypatch.setenv(SI.ENV_FLAG, str(os.getpid()))
    monkeypatch.setattr(SI, "STAGED", SI.StagedInputs())
    monkeypatch.setattr(SI, "HASHES", SI._HashMemo())


# ---------------------------------------------------------------------------
# the own-tier arm server starts before the per-recipe loop
# ---------------------------------------------------------------------------

def _fake_bench_board(python):
    import argparse
    fake = ModuleType("bench_board")

    def build_parser():
        ap = argparse.ArgumentParser()
        ap.add_argument("--vendor", default="apple")
        ap.add_argument("--python-env", default=None)
        ap.add_argument("--mojolearn-version", default=None)
        return ap
    fake.build_parser = build_parser
    fake.repo_commit = lambda: "c0ffee"
    fake.smoke_files_sha256 = lambda: "h"
    fake.box_fingerprint = lambda ctx: {}
    fake.sha256_file = lambda p: "x"
    return fake


def test_warm_box_starts_each_other_modes_arm_server_under_the_key_the_conflicting_arm_looks_up(monkeypatch):
    python = sys.executable
    monkeypatch.setitem(sys.modules, "bench_board", _fake_bench_board(python))
    monkeypatch.setattr(SI, "ARM_SERVERS", SI._ArmServers())
    monkeypatch.delenv("MOJOLEARN_NUMERIC_MODE", raising=False)
    started = []

    def fake_start(self):
        self.proc = SimpleNamespace(poll=lambda: None, pid=4242)
        rec = dict(interpreter=" ".join(self.interp), wall_seconds=0.01, mode=self.env.get("MOJOLEARN_NUMERIC_MODE"))
        self.starts.append(rec)
        SI.ARM_SERVERS.starts.append(rec)
        started.append(rec)
        return True
    monkeypatch.setattr(SI.ArmServer, "start", fake_start)
    got = SW.warm_box(["--vendor", "apple", "--python-env", python], [], (), ["fast", "identical"])
    assert got["arm-server numeric_mode=fast"]["status"] == "ok"
    assert "arm-server numeric_mode=identical" not in got          # the worker's own tier needs none
    assert [s["mode"] for s in started] == ["fast"]
    # the ours-fast arm's worker environment (bench_board_neural/_classical _worker_env: the thread settings
    # dropped, the mode set) finds that very server: no second start inside a recipe
    arm_env = {k: v for k, v in os.environ.items() if k not in SI.LOAD_TIME_ENV}
    arm_env["MOJOLEARN_NUMERIC_MODE"] = "fast"
    server = SI.ARM_SERVERS.server([python], arm_env)
    assert server.alive() and len(server.starts) == 1 and len(started) == 1
    # with no other mode planned (NVIDIA, AMD: identical only) nothing is started
    monkeypatch.setattr(SI, "ARM_SERVERS", SI._ArmServers())
    started.clear()
    SW.warm_box(["--vendor", "nvidia", "--python-env", python], [], (), ["identical"])
    assert started == []


def test_the_campaign_asks_warm_box_for_every_planned_mode():
    src = open(os.path.join(HERE, "gpu_gap_campaign.py")).read()
    assert "numeric_modes=sorted(plans)" in src


# ---------------------------------------------------------------------------
# classical: the uploads the probe never reads (kmeans / pca / ols on istella)
# ---------------------------------------------------------------------------

class _FakeEst:
    def __init__(self, **kw):
        self.kw = dict(kw)

    def get_params(self, deep=False):
        return dict(self.kw)


def _fake_cuml(monkeypatch, calls):
    cp = ModuleType("cupy")
    cp.__version__ = "13.0"
    cp.asarray = lambda a: calls.append(np.shape(a)) or np.array(a)
    cp.cuda = SimpleNamespace(runtime=SimpleNamespace(
        deviceSynchronize=lambda: None, getDeviceProperties=lambda i: {"name": b"Fake GPU"},
        runtimeGetVersion=lambda: 12000, driverGetVersion=lambda: 12000))
    cuml = ModuleType("cuml")
    cuml.__version__ = "25.08"
    for sub, names in (("cluster", ("KMeans",)), ("decomposition", ("PCA",)), ("linear_model", ("LinearRegression",))):
        m = ModuleType("cuml." + sub)
        for n in names:
            setattr(m, n, type(n, (_FakeEst,), {"__module__": "cuml." + sub}))
        setattr(cuml, sub, m)
        monkeypatch.setitem(sys.modules, "cuml." + sub, m)
    monkeypatch.setitem(sys.modules, "cupy", cp)
    monkeypatch.setitem(sys.modules, "cuml", cuml)


@pytest.mark.parametrize("lane", ["kmeans", "pca", "ols"])
def test_classical_cuml_arms_skip_the_upload_only_in_the_probe_and_record_the_same(monkeypatch, lane):
    C = _load("classical_two_datasets")
    calls = []
    _fake_cuml(monkeypatch, calls)
    rng = np.random.default_rng(0)
    data = {"X": rng.standard_normal((64, 5)).astype(np.float32), "y": rng.standard_normal(64).astype(np.float32),
            "init": rng.standard_normal((C.KMEANS_K, 5)).astype(np.float32)}
    recs = {}
    for how in ("timed", "probe"):
        (_timed if how == "timed" else _probe)(monkeypatch)
        calls.clear()
        arm = C.BUILDERS[(lane, "cuml-gpu")](data, {"dataset": "istella"})
        recs[how] = C.params_record(arm.params_obj)
        if how == "timed":
            assert calls and arm.info["upload_ms_untimed"] is not None
        else:
            assert calls == [] and arm.info["upload_ms_untimed"] is None
            assert isinstance(arm.x, np.ndarray) and arm.x.shape == data["X"].shape
    assert recs["timed"] == recs["probe"]


def test_classical_setups_upload_for_every_arm_that_did_not_opt_in(monkeypatch):
    C = _load("classical_two_datasets")
    calls = []
    _fake_cuml(monkeypatch, calls)
    _probe(monkeypatch)
    dev, info = C._cuml_setup({"X": np.zeros((3, 2), np.float32)})          # e.g. DBSCAN, kNN: unchanged
    assert calls == [(3, 2)] and info["upload_ms_untimed"] is not None


# ---------------------------------------------------------------------------
# algos layer lanes: the torch arms (conv1d, maxpool2d, gcn, moe) and ours' moe weight load
# ---------------------------------------------------------------------------

def _algos_cpu(monkeypatch, tmp_path):
    A = _load("bench_board_algos")
    monkeypatch.setattr(A, "_torch_device", lambda torch: "cpu")
    monkeypatch.setattr(A, "_torch_sync", lambda torch, dev: None)
    NF = A._tool("neural_fixtures")
    monkeypatch.setenv(NF.ENV_DIR, str(tmp_path / "no-fixtures"))   # the drift check says "absent"
    monkeypatch.setattr(A, "_NEURAL_FIXTURES", None)
    return A


def _layer_D(lane):
    rng = np.random.default_rng(1)
    D = {"_cap": 1}
    if lane == "gcn":
        n = 40
        D["X"] = rng.standard_normal((n, 6)).astype(np.float32)
        D["indptr"] = np.arange(0, 3 * n + 1, 3, dtype=np.int64)
        D["indices"] = rng.integers(0, n, 3 * n).astype(np.int64)
    return D


@pytest.mark.parametrize("lane", ["conv1d", "maxpool2d", "gcn", "moe"])
def test_torch_layer_arm_probe_record_equals_the_timed_record(monkeypatch, tmp_path, lane):
    torch = pytest.importorskip("torch")
    if lane == "gcn":
        pytest.importorskip("torch_geometric")
    if lane == "moe":
        pytest.importorskip("transformers")
    A = _algos_cpu(monkeypatch, tmp_path)
    if lane == "moe":   # the read-back is shape-independent: a small block of the same class
        monkeypatch.setitem(A.LANES["moe"], "params", dict(hidden_size=32, intermediate_size=24, num_experts=4,
                                                           top_k=2, norm_topk_prob=True))
    out = {}
    for how in ("timed", "probe"):
        (_timed if how == "timed" else _probe)(monkeypatch)
        uploads = []
        real = A._torch_timed_upload

        def upload(t, d, arrays, real=real):
            # a device copy is a new tensor; on the CPU .to() would hand back the staged tensor itself
            uploads.append(len(arrays))
            return real(t, d, [a.clone() if t.is_tensor(a) else np.array(a) for a in arrays])
        monkeypatch.setattr(A, "_torch_timed_upload", upload)
        runner = A._build_layer(lane, "torch-eager-fp32", _layer_D(lane))
        out[how] = (P.arm_record(runner.record), runner.info, uploads)
    (rt, it, ut), (rp, ip, up) = out["timed"], out["probe"]
    assert rt == rp
    assert ut == [1 + (1 if lane == "gcn" else 0)] and up == []
    assert it["upload_ms_untimed"] is not None and ip["upload_ms_untimed"] is None
    same = {k for k in it if k not in ("upload_ms_untimed", "predict_upload_ms", "gcn_cache")}
    assert {k: it[k] for k in same} == {k: ip.get(k) for k in same}


def test_moe_block_is_built_on_meta_and_never_loaded_in_the_probe(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    A = _algos_cpu(monkeypatch, tmp_path)
    p = dict(hidden_size=32, intermediate_size=24, num_experts=4, top_k=2, norm_topk_prob=True)
    torch.manual_seed(A.SEED)
    ref = A._MoE(torch, **p)
    state = {k: v.detach().numpy() for k, v in ref.state_dict().items()}
    _timed(monkeypatch)
    full = A._hf_moe_block(torch, p, state, "torch-eager-fp32")
    _probe(monkeypatch)
    meta = A._hf_moe_block(torch, p, state, "torch-eager-fp32")
    assert type(full.blk) is type(meta.blk) and full.library == meta.library
    assert {k: tuple(v.shape) for k, v in full.blk.state_dict().items()} == \
        {k: tuple(v.shape) for k, v in meta.blk.state_dict().items()}
    assert all(v.device.type == "meta" for v in meta.blk.state_dict().values())
    assert all(v.device.type == "cpu" for v in full.blk.state_dict().values())
    # the probe still refuses a fixture that does not map by shape
    bad = dict(state, **{"gate.weight": np.zeros((p["num_experts"], p["hidden_size"] + 1), np.float32)})
    with pytest.raises(RuntimeError, match="REFUSED"):
        A._hf_moe_block(torch, p, bad, "torch-eager-fp32")


def test_ours_moe_skips_the_weight_load_only_in_the_probe_and_records_the_same(monkeypatch, tmp_path):
    A = _load("bench_board_algos")
    NF = A._tool("neural_fixtures")
    E, D_, F = 4, 8, 6
    rng = np.random.default_rng(2)
    state = {"gate.weight": rng.standard_normal((E, D_)).astype(np.float32)}
    for e in range(E):
        state["experts.%d.w1.weight" % e] = rng.standard_normal((F, D_)).astype(np.float32)
        state["experts.%d.w3.weight" % e] = rng.standard_normal((F, D_)).astype(np.float32)
        state["experts.%d.w2.weight" % e] = rng.standard_normal((D_, F)).astype(np.float32)
    fx = {"state": state, "x": rng.standard_normal((16, D_)).astype(np.float32), "dy": None,
          "entry": {"lane": "moe", "dataset": "synthetic", "variant": "full"}}
    loads = []

    class MoEBlock:
        def __init__(self, **kw):
            self.kw = kw

        def load_state_dict(self, sd):
            loads.append(sorted(sd))
            return self

        def forward(self, x):
            return x
    monkeypatch.setattr(NF, "load_for_lane", lambda *a, **k: fx)
    monkeypatch.setattr(NF, "provenance", lambda fx: {"fake": True})
    monkeypatch.setattr(A, "_ours_class", lambda lane: ("MoEBlock", MoEBlock))
    monkeypatch.setattr(A, "_ours_info", lambda lane, est=None: {"library": "mojolearn", "input_home": "host"})
    monkeypatch.setitem(A.LANES["moe"], "params", dict(hidden_size=D_, intermediate_size=F, num_experts=E,
                                                       top_k=2, norm_topk_prob=True))
    recs = {}
    for how in ("timed", "probe"):
        (_timed if how == "timed" else _probe)(monkeypatch)
        loads.clear()
        runner = A._build_layer("moe", "ours", {"_cap": 1})
        recs[how] = P.arm_record(runner.record)
        assert runner.info["weights_loaded"] is True and runner.info["output_comparable"] is True
        assert loads == ([["down_proj", "gate_up_proj", "router"]] if how == "timed" else [])
    assert recs["timed"] == recs["probe"]


# ---------------------------------------------------------------------------
# neural: the LM torch arms' LlamaForCausalLM
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lane", ["lm-forward", "lm-train-step"])
def test_lm_torch_arm_probe_builds_on_meta_and_records_the_same(monkeypatch, lane):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    N = _load("bench_board_neural")
    monkeypatch.setattr(N, "_torch_device", lambda: (torch, torch.device("cpu"), "cpu", "cpu", lambda: None))
    dims = N.lm_dims(lane, "full")
    n = sum(int(np.prod(s)) for _, s in N.lm_registry(dims))
    data = {"init": np.random.default_rng(3).normal(0, .02, n).astype(np.float32),
            "batches": np.zeros((2, dims[0], dims[1] + 1), np.int32)}
    recs = {}
    for probe in (False, True):
        monkeypatch.setattr(N, "PARAMS_PROBE", probe)
        arm = N.TorchArm(lane, "full", data, "torch-eager-fp32")
        recs[probe] = P.arm_record(arm.record)
        devices = {p.device.type for p in arm.module.parameters()}
        assert devices == ({"meta"} if probe else {"cpu"})
    assert recs[False] == recs[True]
