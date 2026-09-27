# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# THE SEQUENCE LANE'S IDENTITY LANES (docs/lanes/ALGORITHM_EXPANSION_BRIEFS.md).
#
# Owned by the `sequence` expansion lane. tools/identity_break.py executes this
# file in ITS OWN namespace after every helper and registry exists
# (`_load_lane_fragments`), so a lane here is written exactly like a lane there:
#
#     @lane("sequence-example")
#     def _(ml, X, yc, yr, Xh=None):
#         m = ml.Example().fit(X)
#         return _fit(dict(out=_h(m.transform(X[:256]))), m, lambda e: (e.transform(Xh[:256]),))
#
#     _batch_decl(_rows_calls("transform", sl=slice(0, 256)), "sequence-example")
#
# Rules the loader enforces: spell every lane name literally; add only your
# own lanes (to LANES and the per-lane registries); rebind no existing name;
# prefix your own helpers with `_sequence_`. No imports are needed: np, _h,
# _fit, _rows_calls and the rest are this module's.


def _sequence_seq(X, rows=384, t=4, d=8):
    """`rows` fixture rows cut into rows // t sequences of t steps of the first
    d columns, with the target of each sequence's LAST row."""
    n = rows // t
    return np.ascontiguousarray(X[:rows, :d], dtype=np.float32).reshape(n, t, d)


def _sequence_targets(yc, yr, rows=384, t=4):
    return (np.ascontiguousarray(yc[t - 1:rows:t]), np.ascontiguousarray(yr[t - 1:rows:t], dtype=np.float32))


@lane("sequence-lstm")
def _(ml, X, yc, yr, Xh=None):
    """A two-layer LSTM regressor (Adam) and a one-layer LSTM classifier
    (SGD with Nesterov momentum): 96 sequences of 4 steps of 8 columns,
    batches of 32 shuffled, two epochs. Train column: both loss curves,
    both trained parameter vectors, the predictions and probabilities."""
    Xs = _sequence_seq(X)
    ycs, yrs = _sequence_targets(yc, yr)
    r = ml.LSTMRegressor(hidden_size=12, num_layers=2, learning_rate=1e-2, batch_size=32,
                         max_epochs=2, random_state=3).fit(Xs, yrs)
    c = ml.LSTMClassifier(hidden_size=10, optimizer="sgd", learning_rate=5e-2, batch_size=32,
                          max_epochs=2, random_state=4,
                          optimizer_options=dict(momentum=0.9, nesterov=True)).fit(Xs, ycs)
    Xhs = _sequence_seq(Xh)
    return _fit(dict(r_loss=_h(r.loss_curve_), r_params=_h(r.params_), r_pred=_h(r.predict(Xs)),
                     r_seq=_h(r.hidden_sequence(Xs[:16])),
                     c_loss=_h(c.loss_curve_), c_params=_h(c.params_), c_proba=_h(c.predict_proba(Xs))),
                r, lambda e: (e.predict(Xhs),))


@lane("sequence-gru")
def _(ml, X, yc, yr, Xh=None):
    """A two-layer GRU regressor (RMSprop, centered, momentum) and a
    one-layer GRU classifier (Adagrad), the LSTM lane's data and batches."""
    Xs = _sequence_seq(X)
    ycs, yrs = _sequence_targets(yc, yr)
    r = ml.GRURegressor(hidden_size=12, num_layers=2, optimizer="rmsprop", learning_rate=1e-2,
                        batch_size=32, max_epochs=2, random_state=5,
                        optimizer_options=dict(momentum=0.5, centered=True)).fit(Xs, yrs)
    c = ml.GRUClassifier(hidden_size=10, optimizer="adagrad", learning_rate=5e-2, batch_size=32,
                         max_epochs=2, random_state=6).fit(Xs, ycs)
    Xhs = _sequence_seq(Xh)
    return _fit(dict(r_loss=_h(r.loss_curve_), r_params=_h(r.params_), r_pred=_h(r.predict(Xs)),
                     r_seq=_h(r.hidden_sequence(Xs[:16])),
                     c_loss=_h(c.loss_curve_), c_params=_h(c.params_), c_proba=_h(c.predict_proba(Xs))),
                r, lambda e: (e.predict(Xhs),))


def _sequence_opt_run(ml, cls, X, steps=6, **kw):
    """Two float32 tensors cut from the fixture, `steps` updates with
    gradients cut from further fixture rows; the params and the state after."""
    p1 = np.ascontiguousarray(X[:32, :8], dtype=np.float32).copy()
    p2 = np.ascontiguousarray(X[32:40, 0], dtype=np.float32).copy()
    opt = cls([p1, p2], **kw)
    for k in range(steps):
        base = 40 + 48 * k
        g1 = np.ascontiguousarray(X[base:base + 32, 8:16], dtype=np.float32) * np.float32(0.5)
        g2 = np.ascontiguousarray(X[base + 32:base + 40, 1], dtype=np.float32)
        opt.step([g1, g2])
    return dict(params=_h(p1, p2), state=_h(*opt.state))


@lane("sequence-rmsprop")
def _(ml, X, yc, yr, Xh=None):
    """RMSprop at torch's defaults and centered with momentum and weight decay."""
    a = _sequence_opt_run(ml, ml.RMSprop, X, lr=1e-2)
    b = _sequence_opt_run(ml, ml.RMSprop, X, lr=3e-3, centered=True, momentum=0.7, weight_decay=1e-2)
    return _fit(dict(plain=a["params"], plain_state=a["state"], centered=b["params"], centered_state=b["state"]))


@lane("sequence-adagrad")
def _(ml, X, yc, yr, Xh=None):
    """Adagrad at torch's defaults and with lr decay, weight decay and an
    initial accumulator."""
    a = _sequence_opt_run(ml, ml.Adagrad, X, lr=1e-2)
    b = _sequence_opt_run(ml, ml.Adagrad, X, lr=5e-2, lr_decay=0.1, weight_decay=1e-2,
                          initial_accumulator_value=0.1)
    return _fit(dict(plain=a["params"], plain_state=a["state"], decayed=b["params"], decayed_state=b["state"]))


@lane("sequence-autoarima")
def _(ml, X, yc, yr, Xh=None):
    """Four series of 200 observations: two raw fixture columns and two
    running sums of them (so KPSS splits the batch between d = 0 and d = 1),
    the search over p, q in {0, 1} (the orders the CPU ARIMA carries) by
    aicc, the refit and a 12-step forecast."""
    raw = np.ascontiguousarray(X[:200, 3:5].T, dtype=np.float32)
    walk = np.cumsum(raw, axis=1, dtype=np.float32)
    y = np.ascontiguousarray(np.concatenate([raw, walk]), dtype=np.float32)
    m = ml.AutoARIMA(y).search(d=range(2), p=range(2), q=range(2))
    m.fit()
    return _fit(dict(d=_h(m.d_), order=_h(m.order_), ic=_h(m.ic_), forecast=_h(m.forecast(12))))


@lane("sequence-stl")
def _(ml, X, yc, yr, Xh=None):
    """Three series of 120 observations (a fixture column plus a period-12
    wave, a running sum, a column alone): the default STL, and a robust one
    with jumps of 2 and degree-0 seasonal smoothing, both batched."""
    t = np.arange(120, dtype=np.float32)
    wave = np.sin(t * np.float32(2 * np.pi / 12)).astype(np.float32) * np.float32(3.0)
    c = np.ascontiguousarray(X[:120, 5], dtype=np.float32)
    y = np.ascontiguousarray(np.stack([c + wave, np.cumsum(c, dtype=np.float32) + wave, c]), dtype=np.float32)
    a = ml.STL(y, period=12).fit()
    b = ml.STL(y, period=12, robust=True, seasonal_deg=0, seasonal_jump=2, trend_jump=2,
               low_pass_jump=2).fit()
    return _fit(dict(seasonal=_h(a.seasonal), trend=_h(a.trend), resid=_h(a.resid),
                     r_seasonal=_h(b.seasonal), r_trend=_h(b.trend), r_weights=_h(b.weights)))
