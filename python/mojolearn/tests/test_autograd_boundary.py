# SPDX-License-Identifier: Apache-2.0
"""The composed-model surface (Sequential, ops.Session and its distributed programs) at the Python boundary: the
refusals of damaged, foreign and state-less files and of bad distribution arguments."""
import numpy as np
import pytest

from mojolearn import _backend


def _binding_has(*entries, mode=None):
    try:
        b = _backend.binding("_mojolearn_x_sequence") if mode is None else _backend.binding("_mojolearn_x_sequence", mode)
        return all(hasattr(b, name) for name in entries)
    except Exception:
        return False


# ---------------------------------------------------------------- moved from test_autograd_save_resume.py

needs_checkpoint = pytest.mark.skipif(not _binding_has("model_save_checkpoint"),
                                      reason="the sequence binding has no checkpoint entries (rebuild it)")


def _resume_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((120, 5)).astype(np.float32)
    yr = (X[:, 0] * 2 - X[:, 1] + 0.5 * X[:, 2]).astype(np.float32)
    y3 = (X[:, 0] > 0).astype(np.int64) + (X[:, 1] > 0.5)
    return X, yr, y3


@needs_checkpoint
def test_export_refuses_training_state(tmp_path):
    import mojolearn as ml
    # The export's predictions are the native check training_autograd_save_resume;
    # this keeps the public surface: the classes' dtype, the refusals and a
    # fresh fit of a served model.
    X, _, y3 = _resume_data()
    c = (ml.Sequential().linear(8).relu().linear(3)
         .compile(loss="cross_entropy", optimizer="momentum", lr=0.05, nesterov=True))
    c.fit(X, y3, epochs=3, seed=0, batch_size=25)
    path = tmp_path / "clf.ckpt"
    c.export(path)
    served = ml.Sequential.load(path)
    assert served.classes_.dtype == c.classes_.dtype
    with pytest.raises(ValueError):
        served.fit(X, y3, epochs=1, resume=True)
    with pytest.raises(ValueError):
        served.save_checkpoint(tmp_path / "no_state.ckpt")
    # a fresh fit of an exported model is allowed
    served.fit(X, y3, epochs=1, seed=0)


@needs_checkpoint
def test_refuses_damaged_and_foreign_files(tmp_path):
    import mojolearn as ml
    X, yr, _ = _resume_data()
    m = (ml.Sequential().linear(8).tanh().linear(1)
         .compile(loss="mse", optimizer="adam", lr=0.02, schedule="step", step_size=1, gamma=0.7))
    m.fit(X, yr, epochs=1, seed=0, batch_size=32)
    good = tmp_path / "good.ckpt"
    m.save_checkpoint(good)
    raw = bytearray(good.read_bytes())
    raw[len(raw) // 2] ^= 0x01
    bad = tmp_path / "bad.ckpt"
    bad.write_bytes(bytes(raw))
    with pytest.raises(Exception, match="CKPT_REFUSE"):
        ml.Sequential.from_checkpoint(bad)
    short = tmp_path / "short.ckpt"
    short.write_bytes(good.read_bytes()[:-5])
    with pytest.raises(Exception, match="CKPT_REFUSE"):
        ml.Sequential.from_checkpoint(short)
    # a resume under another optimizer is refused
    other = ml.Sequential.from_checkpoint(good).compile(loss="mse", optimizer="sgd", lr=0.01)
    with pytest.raises(Exception, match="resume"):
        other.fit(X, yr, epochs=1, resume=True)


# ---------------------------------------------------------------- moved from test_distributed_program.py

def _distributed_available():
    return _binding_has(*(f"ops_distributed_{name}" for name in
                          ("open", "load", "fit", "run", "state", "residency", "save", "close")), mode="identical")


@pytest.fixture
def require_native_distributed():
    import os
    if not _distributed_available():
        if os.environ.get("MOJOLEARN_REQUIRE_DISTRIBUTED_PROGRAM") == "1":
            pytest.fail("installed sequence binding lacks native distributed-program entries")
        pytest.skip("native distributed-program binding has not been installed")


def test_distributed_refusals_and_damaged_checkpoint(tmp_path, require_native_distributed):
    from mojolearn import ops
    with ops.Session(numeric_mode="identical") as s:
        x = s.tensor(np.zeros((3, 1), np.float32))
        w = s.tensor(np.zeros((1, 1), np.float32))
        p = s.capture([x], [s.matmul(x, w)])
        o = s.optimizer("sgd", lr=0.1)
        with pytest.raises(Exception, match="duplicate"):
            p.distribute((0, 0), optimizer=o)
        with pytest.raises(Exception, match="logical_shards"):
            p.distribute((0,), logical_shards=0, optimizer=o)
        with p.distribute((0,), optimizer=o) as d:
            with pytest.raises(Exception, match="sample count"):
                d.fit(np.ones((3, 1), np.float32), targets=np.ones((2, 1), np.float32))
            # Admission errors happen before mutation and do not poison state.
            assert d.state()["step"] == 0
            path = tmp_path / "damaged.ckpt"
            d.save(str(path))
        data = bytearray(path.read_bytes())
        data[len(data) // 2] ^= 1
        path.write_bytes(data)
        with pytest.raises(Exception):
            s.load_distributed(str(path))
