# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Public CPU Cholesky inference (lane/inference-embedding-ivf-cholesky,
2026-09-15): a factor of a given matrix, or a factor from a saved model,
then solve, on a CPU-only install through the linalg host binding, which
ships in the inference wheel.

Source checks run on a box with nothing built. The runtime checks run where
`_mojolearn_linalg_host.so` is built (MOJOLEARN_HOST_DIR or mojolearn/host)
and require bits: a solve from a loaded factor equals the solve from the
fitted object, `host_model` answers a `HostCholesky`, and on a CPU-only
install the plain class runs. The bit claim against the GPU columns is the
identity harness's cholesky lane (train, infer and model cells).

    cd python && python -m pytest mojolearn/tests/test_cholesky_cpu_inference.py
"""
import os
import re
from pathlib import Path

import numpy as np
import pytest

from mojolearn import _backend, host_surface
from mojolearn._cholesky_impl import _CHOLESKY_FORMAT, Cholesky, HostCholesky

ROOT = Path(__file__).resolve().parents[3]


def _registers(text, name):
    return re.search(r'def_function\[\w+\]\(\s*"' + re.escape(name) + r'"\s*\)', text) is not None


def _spd(n, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n, n))
    return np.ascontiguousarray((m @ m.T + n * np.eye(n)).astype(np.float32))


def _host_built():
    return os.path.exists(_backend.host_module_path("_mojolearn_linalg_host"))


# -- source ------------------------------------------------------------------

def test_the_door_ships_in_the_linalg_family():
    linalg = host_surface.family("linalg")
    assert linalg["ships_in_wheel"] is True
    assert "cholesky" in linalg["training_lanes"] and "Cholesky" in linalg["classes"]
    assert "cholesky" not in host_surface.family("gp")["training_lanes"]
    assert "cholesky" in host_surface.public_reference_lanes()
    src = (ROOT / host_surface.binding_source("linalg")).read_text(encoding="utf-8")
    gpu = (ROOT / "bindings/_mojolearn_gp.mojo").read_text(encoding="utf-8")
    for name in ("cholesky_profile_jitter", "cholesky_factor", "cholesky_solve"):
        assert _registers(src, name), f"the linalg host binding does not register {name}"
        assert _registers(gpu, name), f"{name} is not a GPU gp binding name"
        assert name in linalg["exports"]


def test_cpu_fit_is_public(monkeypatch):
    from mojolearn import KernelDensity
    monkeypatch.setattr(_backend, "_CPU_ONLY", "test CPU")
    calls = []

    class FakeDoor:
        def linalg_numeric_mode(self):
            return 1

        def cholesky_profile_jitter(self):
            return 2.0 ** -20

        def cholesky_factor(self, addrs, params):
            calls.append(("factor", params))
            return 0

    monkeypatch.setattr(Cholesky, "_door", lambda self: (FakeDoor(), "linalg_numeric_mode"))
    c = Cholesky().fit(np.eye(3, dtype=np.float32))
    assert calls and calls[0][1][0] == 3 and c.info_ == 0


def test_the_cpu_route_is_the_shipped_binding(monkeypatch):
    seen = []
    monkeypatch.setattr(Cholesky, "_bind", lambda self, name=None: seen.append(name) or object())
    monkeypatch.setattr(_backend, "_CPU_ONLY", "test CPU")
    assert Cholesky()._door()[1] == "linalg_numeric_mode" and seen[-1] == "_mojolearn_linalg"
    monkeypatch.setattr(_backend, "_CPU_ONLY", None)
    assert Cholesky()._door()[1] == "gp_numeric_mode" and seen[-1] is None


def _fitted_without_binary(n=4, info=0):
    from mojolearn._buffer import as_f32_c
    c = Cholesky(jitter=0.0)
    c.numeric_mode = "identical"
    c.n_, c.info_, c.nb_ = n, info, 32
    c.L_ = as_f32_c(np.tril(_spd(n)), ndim=2, name="L")[0]
    c._logdet, c.jitter_ = 1.25, 0.0
    return c


def test_save_load_round_trips_every_field_without_a_binary(tmp_path):
    c = _fitted_without_binary(info=2)
    path = str(tmp_path / "c.npz")
    c.save(path)
    d = Cholesky.load(path)
    assert (d.n_, d.info_, d.nb_, d.jitter_, d._logdet, d.jitter, d.numeric_mode) == (4, 2, 32, 0.0, 1.25, 0.0, "identical")
    assert np.asarray(d.L_).tobytes() == np.asarray(c.L_).tobytes()
    with pytest.raises(ValueError, match="info=2"):
        d.logdet_
    e = Cholesky()
    e.__dict__.update({k: v for k, v in c.__dict__.items() if k != "jitter"})
    e.save(path)
    assert Cholesky.load(path).jitter is None


def test_load_refuses_a_wrong_format_and_a_cast(tmp_path):
    from mojolearn import _serialize
    from mojolearn._array import Array
    c = _fitted_without_binary()
    path = str(tmp_path / "c.npz")
    c.save(path)
    arrays = _serialize.read_npz(path, _CHOLESKY_FORMAT)
    arrays["meta"] = Array.from_list([4, 0, 32], "<i4")
    _serialize.write_npz(path, arrays)
    with pytest.raises(ValueError, match="refusing to cast"):
        Cholesky.load(path)
    arrays["meta"] = Array.from_list([5, 0, 32], "<i8")
    _serialize.write_npz(path, arrays)
    with pytest.raises(ValueError, match="the factor is 5 x 5"):
        Cholesky.load(path)
    arrays["format"] = "mojolearn-linear-1"
    _serialize.write_npz(path, arrays)
    with pytest.raises(ValueError, match="model format"):
        Cholesky.load(path)


def test_host_cholesky_refuses_a_non_identical_factor():
    c = HostCholesky()
    c.numeric_mode = "fast"
    with pytest.raises(ValueError, match="IDENTICAL only"):
        c._door()


# -- runtime (needs the linalg host binding) ----------------------------------

needs_host = pytest.mark.skipif(not _host_built(), reason="_mojolearn_linalg_host.so is not built here")


@needs_host
def test_a_loaded_factor_solves_bit_for_bit(tmp_path):
    import mojolearn
    a = _spd(64, seed=3)
    b = np.random.default_rng(4).standard_normal((64, 3)).astype(np.float32)
    c = HostCholesky().fit(a)
    assert c.info_ == 0
    x = np.asarray(c.solve(b))
    path = str(tmp_path / "c.npz")
    c.save(path)
    for loaded in (HostCholesky.load(path), mojolearn.host_model(path)):
        assert isinstance(loaded, HostCholesky)
        assert np.asarray(loaded.solve(b)).tobytes() == x.tobytes()
        assert loaded.logdet_ == c.logdet_
    col = np.asarray(c.solve(np.ascontiguousarray(b[:, 1])))
    assert col.tobytes() == np.ascontiguousarray(x[:, 1]).tobytes()


@needs_host
def test_a_failed_factor_refuses_to_solve_on_the_host():
    c = HostCholesky(jitter=0.0).fit(-np.eye(8, dtype=np.float32))
    assert c.info_ != 0
    with pytest.raises(Exception, match="FAILED|info"):
        c.solve(np.ones((8,), np.float32))


@needs_host
@pytest.mark.skipif(_backend._CPU_ONLY is None, reason="a GPU set loaded; the plain class binds the GPU")
def test_the_plain_class_runs_on_a_cpu_only_install():
    a = _spd(16, seed=5)
    c = Cholesky().fit(a)
    h = HostCholesky().fit(a)
    assert np.asarray(c.L_).tobytes() == np.asarray(h.L_).tobytes()
    assert c.vendor_used() == "cpu"
