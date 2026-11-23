# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# LANE nn-losses-2's IDENTITY LANE (2026-11-23): the log_softmax layer with the nll and kl_div losses, and the ctc
# loss on a GRU over frame sequences (with lengths and target lengths), each one seeded training step of a small
# Sequential, digested whole (loss curve, parameters, predictions). Loaded by tools/identity_break.py
# `_load_lane_fragments` like tools/identity_lanes/sequence.py, whose tape helpers (_sequence_tape_binding,
# _sequence_tape_seeded, _sequence_tape_digest) it calls at run time.

_NN_LOSSES_2_ROWS = 48


def _nn_losses_2_parts(B, m, x, y, prefix, parts, **fit):
    m = m.fit(x, y, epochs=1, batch_size=_NN_LOSSES_2_ROWS, shuffle=False, seed=7, **fit)
    d = lambda a: _sequence_tape_digest(B, a)
    parts[prefix + "loss"] = d(np.asarray(m.loss_curve_, dtype=np.float32))
    parts[prefix + "params"] = d(m.params_)
    parts[prefix + "pred"] = d(m.predict(x) if "lengths" not in fit else m.predict(x, lengths=fit["lengths"]))


@lane("sequence-composed-log-softmax-ctc")
def _(ml, X, yc, yr, Xh=None):
    """One seeded full-batch step of: linear(6), tanh, linear(3), log_softmax under nll (3 classes) and under
    kl_div (reduction 'batchmean'); and gru(8), linear(4), log_softmax over (6, 4) frame samples under ctc (blank 0,
    sequence lengths 3..6, target lengths 1..2). Inputs are seeded in Mojo; labels, distributions and lengths are
    cut from them."""
    B = _sequence_tape_binding(ml)
    x = _sequence_tape_seeded(B, (_NN_LOSSES_2_ROWS, 8), 61, 0)
    z = _sequence_tape_seeded(B, (_NN_LOSSES_2_ROWS, 3), 63, 0)
    f = _sequence_tape_seeded(B, (_NN_LOSSES_2_ROWS, 6 * 4), 67, 0)
    # cpu-route: verification targets cut from the seeded arrays before the run
    labels3 = np.ascontiguousarray(np.argmax(z, axis=1).astype(np.int32))
    dist = np.abs(z) + np.float32(0.05)
    dist = np.ascontiguousarray((dist / dist.sum(axis=1, keepdims=True)).astype(np.float32))
    rows = np.arange(_NN_LOSSES_2_ROWS)
    ctc_y = np.ascontiguousarray((1 + (np.abs(f[:, :2]) * 7).astype(np.int64) % 3).astype(np.float32))
    target_lengths = (1 + rows % 2).astype(np.int64)
    lengths = (3 + rows % 4).astype(np.int64)
    parts = {}
    S = ml.Sequential
    head = lambda: S(numeric_mode="identical", input_shape=(8,)).linear(6).tanh().linear(3).log_softmax()
    _nn_losses_2_parts(B, head().compile(loss="nll", optimizer="sgd", lr=0.05), x, labels3, "nll_", parts)
    _nn_losses_2_parts(B, head().compile(loss="kl_div", reduction="batchmean", optimizer="sgd", lr=0.05), x, dist,
                       "kl_div_", parts)
    ctc = S(numeric_mode="identical", input_shape=(6, 4)).gru(8).linear(4).log_softmax()
    _nn_losses_2_parts(B, ctc.compile(loss="ctc", blank=0, optimizer="sgd", lr=0.05), f, ctc_y, "ctc_", parts,
                       lengths=lengths, target_lengths=target_lengths)
    return _fit(parts)


_batch_decl("n/a:training-step (one seeded training step of a model per loss; its forward outputs are train "
            "parts, digested whole)", "sequence-composed-log-softmax-ctc")
