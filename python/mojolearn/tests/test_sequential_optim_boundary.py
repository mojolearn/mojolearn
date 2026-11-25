# SPDX-License-Identifier: Apache-2.0
"""Public boundary of Sequential.compile's rmsprop, adagrad and lamb optimizers and its learning-rate schedules
(lane nn-optim, 2026-11-23): admission and defaults, refusals in clear words, and the checkpoint round trip
(save_checkpoint, from_checkpoint, resume = one continuous fit). The numerics (each step against a float64
reference, every schedule value, the fits, host versus device words) are the native check
checks/native_tests/nn_optim.mojo."""
import os

import numpy as np
import pytest

import mojolearn


def _data(n=64, d=4, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype(np.float32)
    y = (0.6 * x[:, 0] - 0.4 * x[:, 1] * x[:, 2]).astype(np.float32)
    return x, y


def _model():
    return mojolearn.Sequential(numeric_mode="identical").linear(8).tanh().linear(1)


CONFIGS = [
    dict(optimizer="rmsprop", lr=0.01),
    dict(optimizer="rmsprop", lr=0.01, centered=True, schedule="cosine", T_max=3),
    dict(optimizer="rmsprop", lr=0.005, momentum=0.5, schedule="multi_step", milestones=[1, 2]),
    dict(optimizer="adagrad", lr=0.1, lr_decay=0.01, initial_accumulator_value=0.1),
    dict(optimizer="adagrad", lr=0.1, schedule="cosine_warm_restarts", T_0=1, T_mult=2),
    dict(optimizer="lamb", lr=0.02, schedule="one_cycle", total_steps=8, cycle_momentum=False),
    dict(optimizer="lamb", lr=0.02, trust_clip=True, max_grad_norm=None, schedule="linear_warmup",
         warmup_steps=2, total_steps=8),
    dict(optimizer="adam", lr=0.01, schedule="reduce_on_plateau", factor=0.5, patience=0),
]


@pytest.mark.parametrize("kw", CONFIGS)
def test_each_optimizer_and_schedule_fits(kw):
    x, y = _data()
    m = _model().compile(loss="mse", **kw).fit(x, y, epochs=4, batch_size=32, seed=0)
    assert np.isfinite(m.params_).all() and np.isfinite(m.loss_curve_).all()
    assert np.isfinite(m.predict(x)).all()
    assert m.optimizer_m_.shape == m.params_.shape and m.optimizer_v_.shape == m.params_.shape


def test_defaults_follow_pytorch():
    o = _model().compile(optimizer="adagrad")._options
    assert o["eps"] == 1e-10 and o["weight_decay"] == 0.0
    o = _model().compile(optimizer="lamb")._options
    assert o["eps"] == 1e-6 and o["weight_decay"] == 0.01 and o["f7"] == 1.0
    o = _model().compile(optimizer="rmsprop")._options
    assert o["f2"] == 0.99 and o["f7"] == 0.0 and o["eps"] == 1e-8
    o = _model().compile(optimizer="momentum")._options
    assert o["f1"] == 0.9
    o = _model().compile(schedule="one_cycle", total_steps=10, cycle_momentum=False)._options
    assert tuple(o["sched_args"][:6]) == (10.0, 0.3, 25.0, 1e4, 0.0, 0.0)


@pytest.mark.parametrize("kw, message", [
    (dict(optimizer="rmsprop", alpha=1.0), "alpha must be in [0, 1)"),
    (dict(optimizer="adagrad", lr_decay=-1.0), "lr_decay must be >= 0"),
    (dict(optimizer="lamb", beta1=1.0), "beta1 and beta2 must be in [0, 1)"),
    (dict(optimizer="nope"), "optimizer must name a supported Sequential optimizer"),
    (dict(schedule="cosine"), "cosine needs T_max"),
    (dict(schedule="cosine", T_max=2.5), "T_max, a whole number"),
    (dict(schedule="cosine_warm_restarts"), "needs T_0"),
    (dict(schedule="multi_step"), "multi_step needs milestones"),
    (dict(schedule="multi_step", milestones=[3, 2]), "milestones must be rising"),
    (dict(schedule="linear_warmup"), "linear_warmup needs total_steps"),
    (dict(schedule="linear_warmup", warmup_steps=5, total_steps=5), "warmup_steps must be"),
    (dict(schedule="one_cycle", cycle_momentum=False), "one_cycle needs total_steps"),
    (dict(schedule="one_cycle", total_steps=4), "cycle_momentum=True"),
    (dict(schedule="one_cycle", total_steps=4, anneal_strategy="exp", cycle_momentum=False), "anneal_strategy must be"),
    (dict(schedule="reduce_on_plateau", factor=1.5), "factor must be in (0, 1)"),
    (dict(schedule="sometimes"), "schedule must name a supported Sequential schedule"),
])
def test_refusals_in_words(kw, message):
    with pytest.raises(Exception, match=__import__("re").escape(message)):
        _model().compile(**kw)


def test_one_cycle_past_total_steps_is_refused_at_fit():
    x, y = _data()
    m = _model().compile(optimizer="lamb", schedule="one_cycle", total_steps=3, cycle_momentum=False)
    with pytest.raises(Exception, match="total_steps is 3"):
        m.fit(x, y, epochs=4, batch_size=32)


@pytest.mark.parametrize("kw", [CONFIGS[1], CONFIGS[5], CONFIGS[7]])
def test_checkpoint_round_trip_resumes_as_one_fit(kw, tmp_path):
    x, y = _data()
    whole = _model().compile(loss="mse", **kw).fit(x, y, epochs=4, batch_size=32, seed=0)
    first = _model().compile(loss="mse", **kw).fit(x, y, epochs=2, batch_size=32, seed=0)
    path = os.path.join(tmp_path, "m.ckpt")
    first.save_checkpoint(path)
    back = mojolearn.Sequential.from_checkpoint(path)
    assert back._options.get("sched_args") == first._options.get("sched_args")
    assert back._options.get("f7") == first._options.get("f7")
    back.fit(x, y, epochs=2, resume=True)
    assert np.array_equal(back.params_, whole.params_)
    assert np.array_equal(back.optimizer_m_, whole.optimizer_m_)
    assert np.array_equal(back.optimizer_v_, whole.optimizer_v_)


def test_centered_rmsprop_with_momentum_keeps_its_third_buffer(tmp_path):
    # lane stream-fits (2026-11-25): refused at compile before; its momentum buffer is optimizer_w_, saved and
    # resumed with the two others
    x, y = _data()
    kw = dict(optimizer="rmsprop", lr=0.01, centered=True, momentum=0.9)
    whole = _model().compile(loss="mse", **kw).fit(x, y, epochs=4, batch_size=32, seed=0)
    assert whole.optimizer_w_.shape == whole.params_.shape and np.any(whole.optimizer_w_ != 0)
    part = _model().compile(loss="mse", **kw).fit(x, y, epochs=2, batch_size=32, seed=0)
    path = os.path.join(tmp_path, "m.ckpt")
    part.save_checkpoint(path)
    back = mojolearn.Sequential.from_checkpoint(path)
    assert np.array_equal(back.optimizer_w_, part.optimizer_w_)
    back.fit(x, y, epochs=2, resume=True)
    for name in ("params_", "optimizer_m_", "optimizer_v_", "optimizer_w_"):
        assert np.array_equal(getattr(back, name), getattr(whole, name)), name
    assert _model().compile(loss="mse", optimizer="adam").fit(x, y, epochs=1, batch_size=32).optimizer_w_ is None


def test_resume_continues_the_schedule_and_the_optimizer():
    x, y = _data()
    kw = dict(optimizer="adagrad", lr=0.1, initial_accumulator_value=0.1, schedule="reduce_on_plateau",
              factor=0.5, patience=0)
    whole = _model().compile(**kw).fit(x, y, epochs=4, batch_size=32, seed=0)
    part = _model().compile(**kw).fit(x, y, epochs=2, batch_size=32, seed=0).fit(x, y, epochs=2, resume=True)
    assert np.array_equal(part.params_, whole.params_)
