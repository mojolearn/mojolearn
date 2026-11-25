# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# THE RECURRENT ESTIMATORS' OPTIONS: IDENTITY LANES (lane rnn-parity-D, 2026-11-25).
#
# Loaded by tools/identity_break.py `_load_lane_fragments` like tools/identity_lanes/sequence.py, into the same
# namespace; it reuses that fragment's `_sequence_seq` / `_sequence_targets` (called at run time, once every
# fragment is loaded). Each lane runs through the device view (`device_view`): once with device="cpu" and once
# with device="gpu" (docs/plans/VERIFY_THROUGH_DEVICE.md), so the two digests are the host column and the
# device column of GRUClassifier / LSTMRegressor with the options of lane rnn-parity
# (python/mojolearn/_x_sequence_rnn.py; sequence/recurrent.mojo rnn_fit's options loop, autograd/rules_recurrent.mojo
# rnn_forward_options / rnn_weighted_loss / rnn_predict_options). Registered in python/mojolearn/_surface_sequence.py
# (training_lanes, names, no reference). Sabotage arms (tools/identity_lanes/coverage_sabotage.tsv):
# coverage-sequence-rnn_parity_weight_gpu / _cpu (MOJOLEARN_RNN_PARITY_SABOTAGE_WEIGHT_DEVICE / _HOST forced on:
# the fit's loss weights drop their sample-weight factor on one side) must move sequence-rnn-weighted-steps.


def _rnn_parity_steps_y(n, t, tag):
    """(n, t) int64 labels, about one step in five positive, from the hashed stream (both classes present on
    every fixture: the fixtures' own labels can be constant)."""
    return (_hw((n, t), tag, 0.0, 1.0) > np.float32(0.8)).astype(np.int64)


def _rnn_parity_names(state):
    """The state_dict's keys, sorted and joined, as uint8 bytes (hashable by `_h`)."""
    return np.frombuffer(" ".join(sorted(state)).encode(), dtype=np.uint8)


@lane("sequence-rnn-weighted-steps")
def _(ml, X, yc, yr, Xh=None):
    """The user's whole-flight model on the estimator (lane rnn-parity, Andrew 2026-11-25): a GRUClassifier with
    output='sequence' (a score and a loss at every step) over 48 sequences of 8 steps of 6 fixture columns,
    lengths 1 .. 8 (each sequence's padded steps weigh 0), a label at every step (about one in five positive),
    sample_weight per step with the positives weighted 30, class_weight {0: 1, 1: 2}, clip_grad_norm 0.5,
    Adam, batches of 16 shuffled, two epochs. Train column: the loss curve, the trained parameters, the per-step
    probabilities and labels, and the score (accuracy over every valid step, weighted). The sabotage arms
    coverage-sequence-rnn_parity_weight_gpu / _cpu drop the sample weights from the loss weights on one side:
    the loss curve, parameters and probabilities move on that side only."""
    n, t, d = 48, 8, 6
    Xs = _sequence_seq(X, rows=n * t, t=t, d=d)
    lengths = (np.arange(n) % t + 1).astype(np.int64)
    y = _rnn_parity_steps_y(n, t, "rnn-ws:y")
    sw = np.where(y > 0, np.float32(30.0), np.float32(1.0)).astype(np.float32)
    c = ml.GRUClassifier(hidden_size=6, output="sequence", class_weight={0: 1.0, 1: 2.0}, clip_grad_norm=0.5,
                         learning_rate=1e-2, batch_size=16, max_epochs=2, random_state=7)
    c.fit(Xs, y, lengths=lengths, sample_weight=sw)
    Xhs = _sequence_seq(Xh, rows=n * t, t=t, d=d)
    return _fit(dict(loss=_h(c.loss_curve_), params=_h(c.params_),
                     proba=_h(c.predict_proba(Xs, lengths=lengths)),
                     labels=_h(c.predict(Xs, lengths=lengths)),
                     score=_h(np.float64(c.score(Xs, y, sample_weight=sw, lengths=lengths)))),
                c, lambda e: (e.predict_proba(Xhs, lengths=lengths),))


@lane("sequence-rnn-bidi-options")
def _(ml, X, yc, yr, Xh=None):
    """The recurrent options on an LSTMRegressor (lane rnn-parity): two bidirectional layers (hidden 5,
    dropout 0.25 between them, bias=False) over 48 sequences of 8 steps of 6 fixture columns, fitted with
    validation_data (the next 48 sequences) and early_stopping (n_iter_no_change 2, tol 0, up to 4 epochs, the
    best epoch restored), then partial_fit once from given initial states h_0 / c_0 (hashed stream, shape
    (4, 48, 5)). Parts: the validation curve and best epoch; the partial_fit loss curve and parameters; predict
    and hidden_sequence from h_0 / c_0; step() over the whole sequence from (h_0, c_0), which must be predict's
    words (one body), and its final states; step() in two pieces of 4 steps carrying the state (bidirectional:
    the reverse direction sees each piece alone, so the pieces are digested, not compared); the state_dict's
    keys (`_reverse` names, no recurrent biases) and arrays. Then the streaming promise on a unidirectional twin
    (two layers, output='sequence', bias=False, one epoch, per-step targets): step() in two pieces from
    (h_0, c_0) must be predict's every-step words, and its final state the one-call state (6 inputs <= 64: the
    input projection keeps one fold order whatever the number of steps)."""
    n, t, d, H = 48, 8, 6, 5
    Xs = _sequence_seq(X, rows=n * t, t=t, d=d)
    _, yrs = _sequence_targets(yc, yr, rows=n * t, t=t)
    Xv = _sequence_seq(X[n * t:], rows=n * t, t=t, d=d)
    _, yv = _sequence_targets(yc[n * t:], yr[n * t:], rows=n * t, t=t)
    h0 = _hw((4, n, H), "rnn-bo:h0", -0.5, 0.5)
    c0 = _hw((4, n, H), "rnn-bo:c0", -0.5, 0.5)
    m = ml.LSTMRegressor(hidden_size=H, num_layers=2, bidirectional=True, dropout=0.25, bias=False,
                         early_stopping=True, n_iter_no_change=2, tol=0.0, learning_rate=1e-2, batch_size=16,
                         max_epochs=4, random_state=9)
    m.fit(Xs, yrs, validation_data=(Xv, yv))
    parts = dict(validation=_h(np.asarray(m.validation_loss_), np.int64(m.best_epoch_)),
                 fit_params=_h(m.params_))
    m.partial_fit(Xs, yrs, h_0=h0, c_0=c0)
    parts["partial"] = _h(m.loss_curve_, m.params_, np.int64(m.n_iter_))
    pred = m.predict(Xs, h_0=h0, c_0=c0)
    parts["predict"] = _h(pred, m.hidden_sequence(Xs, h_0=h0, c_0=c0))
    whole, (hn, cn) = m.step(Xs, (h0, c0))
    _same_bytes("LSTMRegressor.step over the whole sequence", np.asarray(whole), "LSTMRegressor.predict from h_0",
                np.asarray(pred))
    parts["step_whole"] = _h(whole, hn, cn)
    a, sa = m.step(Xs[:, :4], (h0, c0))
    b, sb = m.step(Xs[:, 4:], sa)
    parts["step_pieces"] = _h(a, sa[0], sa[1], b, sb[0], sb[1])
    state = m.state_dict()
    parts["state_dict"] = _h(_rnn_parity_names(state), *[np.asarray(state[k]) for k in sorted(state)])
    # the unidirectional twin: streaming in pieces equals the whole sequence
    ys = np.ascontiguousarray(yr[:n * t], dtype=np.float32).reshape(n, t)
    hu = _hw((2, n, H), "rnn-bo:hu", -0.5, 0.5)
    cu = _hw((2, n, H), "rnn-bo:cu", -0.5, 0.5)
    u = ml.LSTMRegressor(hidden_size=H, num_layers=2, output="sequence", bias=False, learning_rate=1e-2,
                         batch_size=16, max_epochs=1, random_state=10).fit(Xs, ys)
    pu = np.asarray(u.predict(Xs, h_0=hu, c_0=cu))
    ua, usa = u.step(Xs[:, :4], (hu, cu))
    ub, usb = u.step(Xs[:, 4:], usa)
    _, (uhn, ucn) = u.step(Xs, (hu, cu))
    _same_bytes("LSTMRegressor.step in two pieces", np.concatenate([np.asarray(ua), np.asarray(ub)], axis=1),
                "LSTMRegressor.predict every step", pu)
    _same_bytes("LSTMRegressor.step pieces h_n", np.asarray(usb[0]), "LSTMRegressor.step whole h_n", np.asarray(uhn))
    _same_bytes("LSTMRegressor.step pieces c_n", np.asarray(usb[1]), "LSTMRegressor.step whole c_n", np.asarray(ucn))
    parts["stream"] = _h(u.params_, pu, usb[0], usb[1])
    Xhs = _sequence_seq(Xh, rows=n * t, t=t, d=d)
    return _fit(parts, m, lambda e: (e.predict(Xhs),))


_batch_decl("n/a:per-row-arguments (the endpoints under test take per-sequence lengths, per-step sample weights "
            "and initial states whose rows are the call's; the plain predict batch axis of these estimators is "
            "sequence-gru's and sequence-lstm's)", "sequence-rnn-weighted-steps", "sequence-rnn-bidi-options")
