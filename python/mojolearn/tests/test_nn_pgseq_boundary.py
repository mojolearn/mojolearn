# SPDX-License-Identifier: Apache-2.0
"""The Python boundary of `Sequential.compile(param_groups=..., no_decay=...)` (lane nn-blocks-pgseq, 2026-11-25):
admission, refusals in words, and the checkpoint round trip (save, load, resume: the same bits as no save). The
grouped update itself is checked natively (checks/native_tests/nn_pgseq.mojo) and by the verify lane
sequential-param-groups."""
import inspect

import numpy as np
import pytest

import mojolearn


def _data():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((32, 5)).astype(np.float32)
    y = (X @ np.arange(1, 6, dtype=np.float32) * np.float32(0.1)).astype(np.float32)
    return X, y


def _model():
    return mojolearn.Sequential().linear(8).tanh().linear(1)


def test_signature():
    params = inspect.signature(mojolearn.Sequential.compile).parameters
    assert params["param_groups"].default is None
    assert params["no_decay"].default is None


def test_equal_groups_are_the_ungrouped_fit():
    X, y = _data()
    a = _model().compile(optimizer="adamw", lr=0.01, weight_decay=0.01).fit(X, y, epochs=3, batch_size=8, shuffle=False)
    b = _model().compile(optimizer="adamw", lr=0.01, weight_decay=0.01,
                         param_groups=[{"layers": [0]}, {"params": ["2.weight", "2.bias"], "lr": 0.01}])
    b.fit(X, y, epochs=3, batch_size=8, shuffle=False)
    assert a.params_.tobytes() == b.params_.tobytes()


def test_head_rate_moves_the_fit():
    X, y = _data()
    a = _model().compile(optimizer="sgd", lr=0.01).fit(X, y, epochs=2, batch_size=8, shuffle=False)
    b = _model().compile(optimizer="sgd", lr=0.01, param_groups=[{"layers": [2], "lr": 0.1}])
    b.fit(X, y, epochs=2, batch_size=8, shuffle=False)
    assert a.params_.tobytes() != b.params_.tobytes()
    assert np.isfinite(b.params_).all()


@pytest.mark.parametrize("groups, words", [
    ([{"lr": 0.1}], "takes 'layers'"),
    ([{"layers": [0], "params": ["0.weight"]}], "takes 'layers'"),
    ([{"layers": []}], "has no parameters"),
    (["0.weight"], "list of dicts"),
    ([{"layers": [0], "schedule": "step"}], "optimizer-wide"),
    ([{"layers": [0], "max_grad_norm": 1.0}], "optimizer-wide"),
    ([{"layers": [0], "momentum": 0.5}], "the adamw optimizer does not take 'momentum'"),
    ([{"layers": [0], "colour": 1}], "unknown parameter group key 'colour'"),
    ([{"layers": [0], "lr": -1.0}], "lr must be positive"),
    ([{"layers": [0], "lr": 1e-100}], "lr must be positive"),
    ([{"layers": [0], "lr": 1e100}], "lr must be positive"),
])
def test_compile_refusals(groups, words):
    with pytest.raises((ValueError, TypeError, RuntimeError, Exception), match=words):
        _model().compile(optimizer="adamw", param_groups=groups)


def test_sgd_group_momentum_refused():
    with pytest.raises(Exception, match="compile with optimizer='momentum'"):
        _model().compile(optimizer="sgd", param_groups=[{"layers": [0], "momentum": 0.9}])


def test_plateau_refused():
    with pytest.raises(Exception, match="reduce_on_plateau"):
        _model().compile(schedule="reduce_on_plateau", param_groups=[{"layers": [0], "lr": 0.1}])


@pytest.mark.parametrize("groups, words", [
    ([{"params": ["7.weight"]}], "the model has no parameter '7.weight'"),
    ([{"layers": [1]}], "layer 1 has no parameters"),
    ([{"layers": [0]}, {"params": ["0.bias"]}], "a parameter belongs to one group"),
])
def test_fit_refusals(groups, words):
    X, y = _data()
    model = _model().compile(optimizer="adam", param_groups=groups)
    with pytest.raises(Exception, match=words):
        model.fit(X, y, epochs=1)


def test_no_decay_alone():
    X, y = _data()
    a = _model().compile(optimizer="adamw", lr=0.01, weight_decay=0.1, no_decay="bias")
    a.fit(X, y, epochs=2, batch_size=8, shuffle=False)
    b = _model().compile(optimizer="adamw", lr=0.01, weight_decay=0.1,
                         param_groups=[{"params": ["0.bias", "2.bias"], "weight_decay": 0.0}])
    b.fit(X, y, epochs=2, batch_size=8, shuffle=False)
    assert a.params_.tobytes() == b.params_.tobytes()


@pytest.mark.parametrize("optimizer", ["adamw", "lamb", "momentum"])
def test_checkpoint_round_trip(tmp_path, optimizer):
    X, y = _data()
    groups = [{"layers": [2], "lr": 0.05}]
    straight = _model().compile(optimizer=optimizer, lr=0.005, no_decay="bias", param_groups=groups)
    straight.fit(X, y, epochs=4, batch_size=8, seed=1)
    first = _model().compile(optimizer=optimizer, lr=0.005, no_decay="bias", param_groups=groups)
    first.fit(X, y, epochs=2, batch_size=8, seed=1)
    path = tmp_path / "grouped.ckpt"
    first.save_checkpoint(str(path))
    loaded = mojolearn.Sequential.from_checkpoint(str(path))
    loaded.fit(X, y, epochs=2, resume=True)
    assert loaded.params_.tobytes() == straight.params_.tobytes()
    assert loaded.optimizer_m_.tobytes() == straight.optimizer_m_.tobytes()
