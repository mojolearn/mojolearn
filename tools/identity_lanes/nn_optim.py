# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# THE nn-optim LANE'S IDENTITY LANES (lane nn-optim, 2026-11-23).
#
# Sequential models trained with the optimizers and learning-rate schedules of
# autograd/nn_optim.mojo: a small seeded fit each, the loss curve, both
# optimizer buffers, the parameters and the predictions hashed. Loaded by
# tools/identity_break.py in its own namespace (`_load_lane_fragments`): np,
# _h, _fit and `lane` are the harness's; helpers here are prefixed `_nn_optim_`.


def _nn_optim_data(X, yr, Xh):
    x = np.ascontiguousarray(X[:96, :6], dtype=np.float32)
    xh = np.ascontiguousarray((X if Xh is None else Xh)[:96, :6], dtype=np.float32)
    y = np.ascontiguousarray(yr[:96], dtype=np.float32)
    return x, y, xh


def _nn_optim_fit(ml, x, y, xh, epochs, validation=False, **compile_kw):
    m = (ml.Sequential(numeric_mode="identical").linear(16).tanh().linear(1)
         .compile(loss="mse", **compile_kw))
    v = (x[:32], y[:32]) if validation else None
    m = m.fit(x, y, epochs=epochs, batch_size=32, shuffle=True, seed=0, validation_data=v)
    return dict(loss=np.asarray(m.loss_curve_, dtype=np.float32), m=m.optimizer_m_, v=m.optimizer_v_,
                params=m.params_, pred=m.predict(x), held=m.predict(xh))


def _nn_optim_hash(parts):
    return {k: _h(v) for k, v in parts.items()}


@lane("sequence-optim-rmsprop")
def _(ml, X, yc, yr, Xh=None):
    """rmsprop: centered with cosine_warm_restarts, and with momentum and multi_step: four epochs each."""
    x, y, xh = _nn_optim_data(X, yr, Xh)
    a = _nn_optim_fit(ml, x, y, xh, 4, optimizer="rmsprop", lr=0.01, centered=True,
                      schedule="cosine_warm_restarts", T_0=2, T_mult=2, eta_min=1e-4)
    b = _nn_optim_fit(ml, x, y, xh, 4, optimizer="rmsprop", lr=0.005, momentum=0.5, weight_decay=0.01,
                      schedule="multi_step", milestones=[2, 3], gamma=0.5)
    parts = {"centered_" + k: v for k, v in _nn_optim_hash(a).items()}
    parts.update({"momentum_" + k: v for k, v in _nn_optim_hash(b).items()})
    return _fit(parts)


@lane("sequence-optim-adagrad")
def _(ml, X, yc, yr, Xh=None):
    """adagrad (lr_decay, initial_accumulator_value) with reduce_on_plateau on validation data, and with cosine."""
    x, y, xh = _nn_optim_data(X, yr, Xh)
    a = _nn_optim_fit(ml, x, y, xh, 5, validation=True, optimizer="adagrad", lr=0.1, lr_decay=0.01,
                      initial_accumulator_value=0.1, schedule="reduce_on_plateau", factor=0.5, patience=0)
    b = _nn_optim_fit(ml, x, y, xh, 4, optimizer="adagrad", lr=0.05, schedule="cosine", T_max=4)
    parts = {"plateau_" + k: v for k, v in _nn_optim_hash(a).items()}
    parts.update({"cosine_" + k: v for k, v in _nn_optim_hash(b).items()})
    return _fit(parts)


@lane("sequence-optim-lamb")
def _(ml, X, yc, yr, Xh=None):
    """lamb (global clip, weight decay; trust_clip) with one_cycle and with linear_warmup: twelve steps each."""
    x, y, xh = _nn_optim_data(X, yr, Xh)
    a = _nn_optim_fit(ml, x, y, xh, 4, optimizer="lamb", lr=0.02, schedule="one_cycle", total_steps=12,
                      cycle_momentum=False)
    b = _nn_optim_fit(ml, x, y, xh, 4, optimizer="lamb", lr=0.02, trust_clip=True, max_grad_norm=None,
                      schedule="linear_warmup", warmup_steps=3, total_steps=12)
    parts = {"onecycle_" + k: v for k, v in _nn_optim_hash(a).items()}
    parts.update({"warmup_" + k: v for k, v in _nn_optim_hash(b).items()})
    return _fit(parts)
