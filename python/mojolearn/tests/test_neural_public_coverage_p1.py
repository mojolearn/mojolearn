# SPDX-License-Identifier: Apache-2.0
"""Real-path and refusal tests for neural endpoints changed since 0.8.55.

Sequence optimizer refusals (`RMSprop.square_avg`, `Adagrad.sum`, `Lion.exp_avg`),
`Adafactor`, the `StepLR` / `ExponentialLR` / `OneCycleLR` constructor refusals,
`SmallMLPTrainer` (`weights_`, refusals, `train_steps`),
`training.mlp_inference_sessions` refusals and the `Mamba3Block` owned
weight snapshot, and the keyword-to-option mapping of the optimizer and
schedule constructors. No binding is replaced: every call is the public path.

Moved to native checks (lane tmig-training, 2026-11-14): the one-step state
values of RMSprop, Adagrad and Lion (checks/native_tests/sequence_optimizer_*),
the StepLR / ExponentialLR / OneCycleLR values (lr_schedule_check.mojo), and the
SmallMLP cross-entropy, sessions-forward and inference-sessions references
(training_small_mlp_{cpu,gpu}.mojo).
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


# ---------------------------------------------------------------- keyword mapping
# The public classes hand their keywords to the native constructor as one
# positional tuple; these check only that every keyword lands under its own
# name (the update values are native checks).
def test_rmsprop_keywords_reach_their_options():
    opt = ml.RMSprop([np.zeros(3, np.float32)], lr=0.03, alpha=0.5, eps=1e-6, weight_decay=0.25,
                     momentum=0.125, centered=True)
    assert opt.lr == pytest.approx(0.03)
    assert opt.options == dict(alpha=0.5, eps=1e-6, weight_decay=0.25, momentum=0.125, centered=True)


def test_adagrad_keywords_reach_their_options_and_the_accumulator():
    opt = ml.Adagrad([np.zeros(3, np.float32)], lr=0.03, lr_decay=0.5, weight_decay=0.25,
                     initial_accumulator_value=0.1, eps=1e-6)
    assert opt.lr == pytest.approx(0.03)
    assert opt.options == dict(lr_decay=0.5, weight_decay=0.25, initial_accumulator_value=0.1, eps=1e-6)
    np.testing.assert_array_equal(np.asarray(opt.sum).ravel(), np.full(3, 0.1, np.float32))


def test_lion_keywords_reach_their_options():
    opt = ml.Lion([np.zeros(3, np.float32)], lr=0.1, betas=(0.8, 0.95), weight_decay=0.25)
    assert opt.lr == pytest.approx(0.1)
    assert opt.options == dict(betas=(0.8, 0.95), weight_decay=0.25)


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


def test_schedule_keywords_reach_their_fields():
    step = ml.StepLR(0.1, step_size=2, gamma=0.5)
    assert (step.base_lr, step.step_size, step.gamma) == (pytest.approx(0.1), 2, pytest.approx(0.5))
    exp = ml.ExponentialLR(0.2, gamma=0.25)
    assert (exp.base_lr, exp.gamma) == (pytest.approx(0.2), pytest.approx(0.25))
    cyc = ml.OneCycleLR(1.5, total_steps=10, pct_start=0.25, anneal_strategy="linear", div_factor=10.0,
                        final_div_factor=100.0, three_phase=True)
    assert (cyc.max_lr, cyc.total_steps, cyc.pct_start) == (pytest.approx(1.5), 10, pytest.approx(0.25))
    assert (cyc.anneal_strategy, cyc.div_factor, cyc.final_div_factor, cyc.three_phase) == (
        "linear", pytest.approx(10.0), pytest.approx(100.0), True)


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


def _trainer():
    return ml.SmallMLPTrainer(W1, B1, W2, B2, data_schedule={"batch_size": 12})


def test_small_mlp_weights_are_the_initial_parameters():
    weights = _trainer().weights_
    assert sorted(weights) == ["bias1", "bias2", "weight1", "weight2"]
    for name, want in (("weight1", W1), ("bias1", B1), ("weight2", W2), ("bias2", B2)):
        np.testing.assert_array_equal(np.asarray(weights[name]), want)


@pytest.mark.parametrize("args,match", [
    pytest.param((XM, np.full(12, 3)), "classes 0..2", id="label-3"),
    pytest.param((XM[:, :7], YM), r"shape \(batch, 8\)", id="width"),
    pytest.param((XM, YM[:-1]), "targets", id="short-targets"),
    pytest.param((np.zeros((300, 8), np.float32), np.zeros(300, np.int64)), "batch", id="batch-300"),
])
def test_small_mlp_loss_and_grads_refusals(args, match):
    with pytest.raises(ValueError, match=match):
        _trainer().loss_and_grads(*args)


@pytest.mark.parametrize("inputs,err", [
    pytest.param([], ValueError, id="empty"),
    pytest.param([XM[:, :7]], ValueError, id="width"),
    pytest.param([XM.astype(np.float64)], TypeError, id="float64"),
])
def test_small_mlp_predict_logits_sessions_refusals(inputs, err):
    with pytest.raises(err):
        _trainer().predict_logits_sessions(inputs)


def test_small_mlp_train_steps_runs():
    trainer = _trainer()
    out = trainer.train_steps(np.concatenate([XM, XM]), np.concatenate([YM, YM]), 2)
    assert out["step"] == 2 and len(out["losses"]) == 2 and trainer.step_ == 2


def test_small_mlp_train_steps_refuses_nonpositive_k():
    trainer = _trainer()
    with pytest.raises(ValueError, match="positive"):
        trainer.train_steps(XM, YM, 0)
    assert trainer.step_ == 0


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
    from mojolearn import _backend
    if _backend.vendor() == "cpu":
        # The CPU host binding (bindings/_mojolearn_mamba_host.mojo) has no
        # mamba3_prefill_session_forward: a plain forward runs the fresh host
        # prefill without a session, so there is no weight upload to count.
        pytest.skip("Mamba3 prefill-session upload counters need a GPU binding; the CPU host forward holds no session")
    block, x = _mamba3()
    # A GPU identical block opens its prefill session handle when it is built or restored
    # (bindings/mamba_public_native_door.mojo `mamba3_prefill_session_open`); the weights go up at the first
    # forward. Before it: no session, or a session that has uploaded nothing (was: `is None` only).
    before = block.session_info()
    assert before is None or (before["weight_uploads"] == 0 and before["weight_reuses"] == 0)
    block.forward(x)
    info = block.session_info()
    assert set(info) == {"weight_uploads", "weight_recopies", "weight_reuses",
                         "backward_reuses", "backward_recomputes", "stages_held"}
    assert info["weight_uploads"] == 1
    block.forward(x)
    assert block.session_info()["weight_reuses"] == info["weight_reuses"] + 1


def test_mamba3_forward_holds_one_session_across_calls():
    # Lane fix-mamba-forward-doors (2026-11-14): a block's stateless forwards
    # run on ONE prefill session held by the block: the weights are uploaded
    # once and reused, the stages stay held, the session object is the same
    # after every call, and the output bytes do not move between calls.
    from mojolearn import _backend
    if _backend.vendor() == "cpu":
        pytest.skip("Mamba3 prefill-session counters need a GPU binding; the CPU host forward holds no session")
    block, x = _mamba3()
    first = np.asarray(block.forward(x)).copy()
    session = vars(block).get("_prefill_session")
    assert session is not None
    for _ in range(3):
        y = np.asarray(block.forward(x))
        assert y.tobytes() == first.tobytes()
        assert vars(block).get("_prefill_session") is session
    info = block.session_info()
    assert info["weight_uploads"] == 1
    assert info["weight_recopies"] == 0
    assert info["weight_reuses"] >= 3
    assert info["stages_held"] == 1


def test_mamba3_has_no_owned_weight_methods():
    # The NN40 owned-weight snapshot was deleted (2026-11-25): its three
    # methods left the public surface, with no alias or stub.
    block, x = _mamba3()
    for name in ("install_owned_weights", "export_owned_weights", "release_owned_weights"):
        assert not hasattr(block, name), name
    block.forward(x)


def test_pattern_name_is_total():
    # No failure path: any other spelling or a non-str is simply not a known pattern.
    from mojolearn.models.tokenizer import PATTERNS, pattern_name
    assert pattern_name(PATTERNS["gpt2"][0]) == "gpt2"
    for other in ("abc", b"abc", 5, None):
        assert pattern_name(other) is None


# ---------------------------------------------------------------- refusal classes (lane fix-p1-defects)
@pytest.mark.parametrize("build,err,match", [
    pytest.param(lambda: ml.StepLR(0.1, step_size=2).lr_at(0), ValueError, "ONE-BASED", id="StepLR-lr_at0"),
    pytest.param(lambda: ml.ExponentialLR(0.1, gamma=0.5).lr_at(0), ValueError, "ONE-BASED", id="ExponentialLR-lr_at0"),
    pytest.param(lambda: ml.StepLR(0.1, step_size="2"), TypeError, "step_size", id="StepLR-str-step"),
    pytest.param(lambda: ml.StepLR(0.1, step_size=2.5), TypeError, "step_size", id="StepLR-float-step"),
    pytest.param(lambda: ml.StepLR("0.1", step_size=2), TypeError, "base_lr", id="StepLR-str-lr"),
    pytest.param(lambda: ml.StepLR(-0.1, step_size=2), ValueError, "base_lr", id="StepLR-negative"),
    pytest.param(lambda: ml.StepLR(float("nan"), step_size=2), ValueError, "finite", id="StepLR-nan-init"),
    pytest.param(lambda: ml.ExponentialLR(0.1, gamma=float("nan")), ValueError, "finite", id="ExponentialLR-nan-init"),
    pytest.param(lambda: ml.ExponentialLR(0.1, gamma=-0.5), ValueError, "gamma", id="ExponentialLR-negative"),
    pytest.param(lambda: ml.ExponentialLR(0.1, gamma="x"), TypeError, "gamma", id="ExponentialLR-str"),
    pytest.param(lambda: ml.OneCycleLR("1", total_steps=5), TypeError, "max_lr", id="OneCycleLR-str"),
    pytest.param(lambda: ml.OneCycleLR(1.0, total_steps=5.5), TypeError, "total_steps", id="OneCycleLR-float-total"),
    pytest.param(lambda: ml.Adafactor([np.zeros(3, np.float32)], lr="1"), TypeError, "lr", id="Adafactor-str-lr"),
    pytest.param(lambda: ml.Adafactor([np.zeros(3, np.float32)], d="2"), TypeError, "d must", id="Adafactor-str-d"),
])
def test_typed_refusals(build, err, match):
    with pytest.raises(err, match=match):
        build()
