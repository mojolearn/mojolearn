# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# LANE nn-losses' IDENTITY LANE (2026-11-23): the losses of Sequential.compile(loss=...) added by the lane
# (l1, smooth_l1, nll, kl_div, bce_with_logits with pos_weight, reduction 'sum' and 'batchmean'), each one
# seeded full-batch training step of a small model, digested whole (loss curve, parameters, predictions).
# Loaded by tools/identity_break.py `_load_lane_fragments` like tools/identity_lanes/sequence.py, whose tape
# helpers (_sequence_tape_binding, _sequence_tape_seeded, _sequence_tape_digest) it calls at run time.

_NN_LOSSES_ROWS = 64


def _nn_losses_step(ml, B, x, y, loss, stream, out=1, **kw):
    m = ml.Sequential(numeric_mode="identical", input_shape=(8,)).linear(6).tanh().linear(out)
    m = m.compile(loss=loss, optimizer="sgd", lr=0.05, **kw)
    m = m.fit(x, y, epochs=1, batch_size=_NN_LOSSES_ROWS, shuffle=False, seed=stream)
    d = lambda a: _sequence_tape_digest(B, a)
    return dict(loss=d(np.asarray(m.loss_curve_, dtype=np.float32)), params=d(m.params_), pred=d(m.predict(x)))


@lane("sequence-composed-losses")
def _(ml, X, yc, yr, Xh=None):
    """One seeded full-batch step of linear(6), tanh, linear(k) under each loss of lane nn-losses: l1,
    smooth_l1 (beta 0.5, reduction 'sum'), bce_with_logits (pos_weight 2.5), nll (3 classes) and kl_div
    (reduction 'batchmean', 3 outputs). Inputs and targets are seeded in Mojo; the labels and the target
    distributions are cut from them."""
    B = _sequence_tape_binding(ml)
    x = _sequence_tape_seeded(B, (_NN_LOSSES_ROWS, 8), 51, 0)
    y = _sequence_tape_seeded(B, (_NN_LOSSES_ROWS, 1), 53, 0)
    z = _sequence_tape_seeded(B, (_NN_LOSSES_ROWS, 3), 57, 0)
    # cpu-route: verification targets cut from the seeded arrays before the run
    labels2 = np.ascontiguousarray((y[:, 0] > 0).astype(np.int32))
    labels3 = np.ascontiguousarray(np.argmax(z, axis=1).astype(np.int32))
    dist = np.abs(z) + np.float32(0.05)
    dist = np.ascontiguousarray((dist / dist.sum(axis=1, keepdims=True)).astype(np.float32))
    parts = {}
    for key, value in _nn_losses_step(ml, B, x, y, "l1", 1).items():
        parts["l1_" + key] = value
    for key, value in _nn_losses_step(ml, B, x, y, "smooth_l1", 2, beta=0.5, reduction="sum").items():
        parts["smooth_l1_" + key] = value
    for key, value in _nn_losses_step(ml, B, x, labels2, "bce_with_logits", 3, pos_weight=2.5).items():
        parts["bce_pos_" + key] = value
    for key, value in _nn_losses_step(ml, B, x, labels3, "nll", 4, out=3).items():
        parts["nll_" + key] = value
    for key, value in _nn_losses_step(ml, B, x, dist, "kl_div", 5, out=3, reduction="batchmean").items():
        parts["kl_div_" + key] = value
    return _fit(parts)


_batch_decl("n/a:training-step (one seeded training step of a model per loss; its forward outputs are train "
            "parts, digested whole)", "sequence-composed-losses")
