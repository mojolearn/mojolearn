"""torch.compile arms of the bench board's neural family (2026-09-29, do-amd):

* a tools/ module loaded under an alias must be in sys.modules, or dynamo
  cannot import a traced function's module ("No module named
  'bbn_torch_lm_step_opponent'": every lm torch-compile arm refused);
* SDPA's inputs share one dtype (under a bf16 autocast RoPE promoted q and k
  to float32 while v stayed bf16; compiled SDPA refused "self and mat2 must
  have the same dtype").
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bench_board_neural as bbn        # noqa: E402


def test_load_registers_the_alias(tmp_path, monkeypatch):
    (tmp_path / "bbn_probe_mod.py").write_text("X = 7\n")
    monkeypatch.setattr(bbn, "HERE", str(tmp_path))
    sys.modules.pop("bbn_bbn_probe_mod", None)
    mod = bbn._load("bbn_probe_mod")
    try:
        assert sys.modules["bbn_bbn_probe_mod"] is mod and mod.X == 7
        import importlib
        assert importlib.import_module("bbn_bbn_probe_mod") is mod
    finally:
        sys.modules.pop("bbn_bbn_probe_mod", None)


def test_a_failed_load_leaves_no_half_module(tmp_path, monkeypatch):
    (tmp_path / "broken_mod.py").write_text("raise RuntimeError('no')\n")
    monkeypatch.setattr(bbn, "HERE", str(tmp_path))
    with pytest.raises(RuntimeError):
        bbn._load("broken_mod")
    assert "bbn_broken_mod" not in sys.modules


def test_sdpa_takes_mixed_input_dtypes():
    torch = pytest.importorskip("torch")
    import speed_torch_seq as sts
    cfg = {"n_heads": 2, "n_kv": 2, "head_dim": 8, "ctx": 0, "l": 4}
    m = sts.LlamaEager(torch, torch.device("cpu"), cfg, {}, torch.float32)
    q = torch.randn(1, 2, 4, 8)                        # float32, as RoPE leaves it
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8).to(torch.bfloat16)     # a linear's bf16 output
    out = m.attention_sdpa(q, k, v, 1, 4)
    assert out.dtype == torch.bfloat16 and out.shape == (4, 16)


class _FakeTorch(object):
    def __init__(self, root):
        self.__file__ = os.path.join(root, "__init__.py")


def test_cpu_compile_pins_torchs_openmp_on_macos(tmp_path, monkeypatch):
    # the board venv's base Python lib dir holds a second libomp; inductor
    # must bind to torch's own (OMP Error #15 on the M3 Ultra, 2026-09-29)
    (tmp_path / "include").mkdir()
    (tmp_path / "include" / "omp.h").write_text("")
    monkeypatch.setattr(bbn.sys, "platform", "darwin")
    monkeypatch.delenv("OMP_PREFIX", raising=False)
    what = bbn._pin_inductor_openmp(_FakeTorch(str(tmp_path)))
    assert os.environ["OMP_PREFIX"] == str(tmp_path)
    assert "torch's own libomp" in what


def test_cpu_compile_refuses_without_torchs_omp_header(tmp_path, monkeypatch):
    monkeypatch.setattr(bbn.sys, "platform", "darwin")
    monkeypatch.delenv("OMP_PREFIX", raising=False)
    with pytest.raises(RuntimeError, match="REFUSED"):
        bbn._pin_inductor_openmp(_FakeTorch(str(tmp_path)))
    assert "OMP_PREFIX" not in os.environ


def test_cpu_compile_leaves_other_platforms_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(bbn.sys, "platform", "linux")
    monkeypatch.delenv("OMP_PREFIX", raising=False)
    bbn._pin_inductor_openmp(_FakeTorch(str(tmp_path)))
    assert "OMP_PREFIX" not in os.environ
