# SPDX-License-Identifier: Apache-2.0
"""Real-path and refusal tests for neural endpoints changed since 0.8.55.

Sequence optimizer state (`RMSprop.square_avg`, `Adagrad.sum`, `Lion.exp_avg`),
`Adafactor`, the `StepLR` / `ExponentialLR` / `OneCycleLR` constructors,
`SmallMLPTrainer` (`weights_`, `loss_and_grads`, `predict_logits_sessions`,
`train_steps`), `training.mlp_inference_sessions` and the `Mamba3Block` owned
weight snapshot. No binding is replaced: every call is the public path.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("MOJOLEARN_NUMERIC_MODE", "identical")

import mojolearn as ml  # noqa: E402
from mojolearn import training  # noqa: E402

G = np.full((2, 3), 0.5, np.float32)


def _stepped(optimizer, **kw):
    w = np.zeros((2, 3), np.float32)
    opt = optimizer([w], **kw)
    opt.step([G])
    return opt, w


# ---------------------------------------------------------------- optimizer state
def test_rmsprop_square_avg_after_one_step():
    opt, _ = _stepped(ml.RMSprop, alpha=0.99)
    np.testing.assert_allclose(np.asarray(opt.square_avg).ravel(), np.full(6, 0.01 * 0.25), rtol=1e-6)


def test_adagrad_sum_after_one_step():
    opt, _ = _stepped(ml.Adagrad, initial_accumulator_value=0.1)
    np.testing.assert_allclose(np.asarray(opt.sum).ravel(), np.full(6, 0.1 + 0.25), rtol=1e-6)


def test_lion_exp_avg_after_one_step():
    opt, w = _stepped(ml.Lion, lr=0.1, betas=(0.9, 0.99))
    np.testing.assert_allclose(np.asarray(opt.exp_avg).ravel(), np.full(6, 0.01 * 0.5), rtol=1e-5)
    np.testing.assert_allclose(w, np.full((2, 3), -0.1), rtol=1e-6)


@pytest.mark.parametrize("optimizer,state", [(ml.RMSprop, "square_avg"), (ml.Adagrad, "sum"), (ml.Lion, "exp_avg")],
                         ids=["RMSprop.square_avg", "Adagrad.sum", "Lion.exp_avg"])
def test_optimizer_state_unchanged_by_refused_step(optimizer, state):
    opt, w = _stepped(optimizer)
    before, weights, t = np.asarray(getattr(opt, state)).copy(), w.copy(), opt.t
    with pytest.raises(TypeError, match="float64"):
        opt.step([G.astype(np.float64)])
    with pytest.raises(ValueError, match="shape"):
        opt.step([np.zeros((3, 2), np.float32)])
    assert opt.t == t
    np.testing.assert_array_equal(np.asarray(getattr(opt, state)), before)
    np.testing.assert_array_equal(w, weights)


def test_adafactor_constructs_and_steps():
    opt, w = _stepped(ml.Adafactor, lr=1e-2)
    assert opt.t == 1
    assert np.isfinite(w).all() and (w < 0).all()


@pytest.mark.parametrize("params,kw,err", [
    pytest.param([np.zeros(3)], {}, TypeError, id="float64-params"),
    pytest.param([np.zeros(3, np.float32)], dict(lr=-1.0), ValueError, id="negative-lr"),
    pytest.param([np.zeros(3, np.float32)], dict(beta2_decay=0.5), ValueError, id="positive-beta2-decay"),
    pytest.param([np.zeros(3, np.float32)], dict(eps=(None,)), ValueError, id="one-eps"),
    pytest.param([np.zeros(3, np.float32)], dict(numeric_mode="nope"), ValueError, id="numeric-mode"),
])
def test_adafactor_init_refusals(params, kw, err):
    with pytest.raises(err):
        ml.Adafactor(params, **kw)


# ---------------------------------------------------------------- schedules
def test_steplr_values():
    sched = ml.StepLR(0.1, step_size=2, gamma=0.5)
    assert [sched.lr_at(t) for t in (1, 2, 3, 5)] == pytest.approx([0.1, 0.1, 0.05, 0.025], rel=1e-6)


def test_exponentiallr_values():
    sched = ml.ExponentialLR(0.1, gamma=0.5)
    assert [sched.lr_at(t) for t in (1, 2, 3)] == pytest.approx([0.1, 0.05, 0.025], rel=1e-6)


def test_onecyclelr_values():
    sched = ml.OneCycleLR(1.0, total_steps=10)
    lrs = [sched.lr_at(t) for t in range(1, 11)]
    assert lrs[0] == pytest.approx(1.0 / 25, rel=1e-6)
    assert max(lrs) == pytest.approx(1.0, rel=1e-6)
    assert lrs[-1] < lrs[0]


@pytest.mark.parametrize("build,err,match", [
    pytest.param(lambda: ml.StepLR(0.1, step_size=0), ValueError, "step_size", id="StepLR-step_size"),
    pytest.param(lambda: ml.StepLR(float("nan"), step_size=2).lr_at(1), ValueError, "finite", id="StepLR-nan"),
    pytest.param(lambda: ml.ExponentialLR(0.1, gamma=float("nan")).lr_at(1), ValueError, "finite", id="ExponentialLR-nan"),
    pytest.param(lambda: ml.OneCycleLR(1.0, total_steps=0), ValueError, "total_steps", id="OneCycleLR-total"),
    pytest.param(lambda: ml.OneCycleLR(1.0, total_steps=10, pct_start=1.5), ValueError, "pct_start",
                 id="OneCycleLR-pct"),
    pytest.param(lambda: ml.OneCycleLR(1.0, total_steps=10, anneal_strategy="nope"), ValueError, "anneal",
                 id="OneCycleLR-anneal"),
    pytest.param(lambda: ml.OneCycleLR(float("nan"), total_steps=5), ValueError, "finite", id="OneCycleLR-nan"),
    pytest.param(lambda: ml.OneCycleLR(1.0, total_steps=5, div_factor=0.0), ZeroDivisionError, "nonzero",
                 id="OneCycleLR-div"),
])
def test_schedule_init_refusals(build, err, match):
    with pytest.raises(err, match=match):
        build()


# ---------------------------------------------------------------- small MLP
_rng = np.random.default_rng(0)
W1 = (_rng.standard_normal((16, 8)) * 0.1).astype(np.float32)
B1 = np.zeros(16, np.float32)
W2 = (_rng.standard_normal((3, 16)) * 0.1).astype(np.float32)
B2 = np.zeros(3, np.float32)
XM = _rng.random((12, 8), dtype=np.float32)
YM = _rng.integers(0, 3, 12)


def _logits(X):
    return np.maximum(X @ W1.T + B1, 0) @ W2.T + B2


def _trainer():
    return ml.SmallMLPTrainer(W1, B1, W2, B2, data_schedule={"batch_size": 12})


def test_small_mlp_weights_are_the_initial_parameters():
    weights = _trainer().weights_
    assert sorted(weights) == ["bias1", "bias2", "weight1", "weight2"]
    for name, want in (("weight1", W1), ("bias1", B1), ("weight2", W2), ("bias2", B2)):
        np.testing.assert_array_equal(np.asarray(weights[name]), want)


def test_small_mlp_loss_and_grads_is_cross_entropy_without_a_step():
    trainer = _trainer()
    loss, grads = trainer.loss_and_grads(XM, YM)
    z = _logits(XM)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    assert loss == pytest.approx(float(-np.log(p[np.arange(12), YM]).mean()), rel=1e-5)
    assert [np.asarray(g).shape for g in grads] == [(16, 8), (16,), (3, 16), (3,)]
    assert trainer.step_ == 0


@pytest.mark.parametrize("args,match", [
    pytest.param((XM, np.full(12, 3)), "classes 0..2", id="label-3"),
    pytest.param((XM[:, :7], YM), r"shape \(batch, 8\)", id="width"),
    pytest.param((XM, YM[:-1]), "targets", id="short-targets"),
    pytest.param((np.zeros((300, 8), np.float32), np.zeros(300, np.int64)), "batch", id="batch-300"),
])
def test_small_mlp_loss_and_grads_refusals(args, match):
    with pytest.raises(ValueError, match=match):
        _trainer().loss_and_grads(*args)


def test_small_mlp_predict_logits_sessions_concatenates():
    trainer = _trainer()
    got = np.asarray(trainer.predict_logits_sessions([XM[:5], XM[5:]]))
    np.testing.assert_allclose(got, _logits(XM), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(got, np.asarray(trainer.predict_logits(XM)), atol=1e-6)


@pytest.mark.parametrize("inputs,err", [
    pytest.param([], ValueError, id="empty"),
    pytest.param([XM[:, :7]], ValueError, id="width"),
    pytest.param([XM.astype(np.float64)], TypeError, id="float64"),
])
def test_small_mlp_predict_logits_sessions_refusals(inputs, err):
    with pytest.raises(err):
        _trainer().predict_logits_sessions(inputs)


def test_small_mlp_train_steps_runs_or_refuses_by_name():
    trainer = _trainer()
    try:
        out = trainer.train_steps(np.concatenate([XM, XM]), np.concatenate([YM, YM]), 2)
    except RuntimeError as exc:
        # Builds without the multistep route say so by name and leave the state alone.
        assert "train_steps" in str(exc)
        assert trainer.step_ == 0
        return
    assert out["step"] == 2 and len(out["losses"]) == 2 and trainer.step_ == 2


def test_small_mlp_train_steps_refuses_nonpositive_k():
    trainer = _trainer()
    with pytest.raises(ValueError, match="positive"):
        trainer.train_steps(XM, YM, 0)
    assert trainer.step_ == 0


def test_mlp_inference_sessions_matches_reference():
    got = np.asarray(training.mlp_inference_sessions([XM[:5], XM[5:]], W1, B1, W2, B2))
    np.testing.assert_allclose(got, _logits(XM), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("inputs,kw,err", [
    pytest.param([XM[:5, :7]], {}, ValueError, id="width"),
    pytest.param([XM[:0]], {}, ValueError, id="empty-session"),
    pytest.param([XM.astype(np.float64)], {}, TypeError, id="float64"),
    pytest.param([XM], dict(numeric_mode="nope"), ValueError, id="numeric-mode"),
])
def test_mlp_inference_sessions_refusals(inputs, kw, err):
    with pytest.raises(err):
        training.mlp_inference_sessions(inputs, W1, B1, W2, B2, **kw)


# ---------------------------------------------------------------- Mamba3 owned weights
def _mamba3():
    dm, di, nh = 32, 64, 1
    dip = 2 * di + 256 + 3 * nh + 32
    shapes = {"block_norm.weight": (dm,), "in_proj.weight": (dip, dm), "dt_bias": (nh,),
              "B_norm.weight": (128,), "C_norm.weight": (128,), "B_bias": (nh, 128), "C_bias": (nh, 128),
              "D": (nh,), "out_proj.weight": (dm, di)}
    weights = {}
    for i, (name, shape) in enumerate(sorted(shapes.items())):
        n = int(np.prod(shape))
        a = (np.ones(n, np.float32) if name == "block_norm.weight"
             else ((np.arange(n, dtype=np.float32) + 7 * i) % 11 - 5) / 32.0)
        weights[name] = np.ascontiguousarray(a.reshape(shape))
    try:
        block = ml.Mamba3Block(weights)
    except ImportError as exc:
        pytest.skip(f"no mamba binding on this install: {exc}")
    x = np.ascontiguousarray((((np.arange(2 * 4 * dm, dtype=np.float32)) % 13 - 6) / 64.0).reshape(2, 4, dm))
    return block, x


def test_mamba3_session_info_counts_weight_uploads():
    block, x = _mamba3()
    assert block.session_info() is None
    block.forward(x)
    info = block.session_info()
    assert set(info) == {"weight_uploads", "weight_recopies", "weight_reuses",
                         "backward_reuses", "backward_recomputes", "stages_held"}
    assert info["weight_uploads"] == 1
    block.forward(x)
    assert block.session_info()["weight_reuses"] == info["weight_reuses"] + 1


def test_mamba3_owned_weights_round_trip():
    block, x = _mamba3()
    want = np.asarray(block.forward(x)).copy()
    try:
        generation = block.install_owned_weights()
    except RuntimeError as exc:
        # Builds without the owned-weight arm refuse by name; the block keeps working.
        assert "owned weights" in str(exc)
        assert np.asarray(block.forward(x)).tobytes() == want.tobytes()
        return
    assert generation is not None
    assert np.asarray(block.forward(x)).tobytes() == want.tobytes()
    exported = block.export_owned_weights()
    for name, w in zip(block._W_NAMES, block._w):
        assert np.asarray(exported[name]).tobytes() == np.asarray(w).tobytes(), name
    block.release_owned_weights()
    assert block._owned_generation is None
    assert np.asarray(block.forward(x)).tobytes() == want.tobytes()


def test_mamba3_export_without_snapshot_refuses():
    block, x = _mamba3()
    with pytest.raises(RuntimeError, match="no installed owned snapshot"):
        block.export_owned_weights()
    block.release_owned_weights()  # releasing nothing is a no-op
    block.forward(x)
    with pytest.raises(RuntimeError, match="no installed owned snapshot"):
        block.export_owned_weights()
