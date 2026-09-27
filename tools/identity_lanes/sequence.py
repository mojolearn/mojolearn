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
    if isinstance(opt.state, list) and opt.state and isinstance(opt.state[0], dict):
        return dict(params=_h(p1, p2), opt=opt)
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


@lane("sequence-var")
def _(ml, X, yc, yr, Xh=None):
    """A three-variable VAR(2) with a constant on 300 fixture rows (the
    columns turned into a stable autoregression by a running filter), and a
    VAR(1) without one; params, sigma_u, residuals and a 10-step forecast."""
    e = np.ascontiguousarray(X[:300, 6:9], dtype=np.float32)
    y = np.zeros_like(e)
    for t in range(1, 300):
        y[t] = (np.float32(0.5) * y[t - 1] + e[t]).astype(np.float32)
    a = ml.VAR(y).fit(maxlags=2)
    b = ml.VAR(y).fit(maxlags=1, trend="n")
    return _fit(dict(params=_h(a.params), sigma_u=_h(a.sigma_u), resid=_h(a.resid),
                     forecast=_h(a.forecast(y, 10)), n_params=_h(b.params), n_forecast=_h(b.forecast(y, 10))))


@lane("sequence-mlp")
def _(ml, X, yc, yr, Xh=None):
    """sklearn-shaped MLPs on 384 rows of the first 10 columns: a regressor
    (two tanh layers, Adam, L2) and a three-class classifier (relu, SGD with
    Nesterov momentum and the adaptive rate), shuffled batches of 64,
    six epochs each, plus a binary logistic-output classifier."""
    Xm = np.ascontiguousarray(X[:384, :10], dtype=np.float32)
    y3 = (np.asarray(yc[:384]) + (Xm[:, 5] > 0).astype(np.int64)).astype(np.int64)
    r = ml.MLPRegressor(hidden_layer_sizes=(12, 8), activation="tanh", alpha=1e-3, batch_size=64,
                        max_iter=6, random_state=0, learning_rate_init=1e-2).fit(Xm, yr[:384])
    c = ml.MLPClassifier(hidden_layer_sizes=(10,), solver="sgd", learning_rate="adaptive", batch_size=64,
                         max_iter=6, random_state=1, learning_rate_init=5e-2).fit(Xm, y3)
    b = ml.MLPClassifier(hidden_layer_sizes=(6,), activation="logistic", batch_size=64, max_iter=6,
                         random_state=2).fit(Xm, yc[:384])
    Xhm = np.ascontiguousarray(Xh[:256, :10], dtype=np.float32)
    return _fit(dict(r_curve=_h(np.asarray(r.loss_curve_)), r_coefs=_h(*r.coefs_, *r.intercepts_),
                     r_pred=_h(r.predict(Xm)), c_curve=_h(np.asarray(c.loss_curve_)),
                     c_coefs=_h(*c.coefs_, *c.intercepts_), c_proba=_h(c.predict_proba(Xm)),
                     b_proba=_h(b.predict_proba(Xm))),
                r, lambda e: (e.predict(Xhm),))


@lane("sequence-rnn")
def _(ml, X, yc, yr, Xh=None):
    """A two-layer tanh RNN regressor (Adam) and a one-layer relu RNN
    classifier (AdamW), the LSTM lane's data and batches."""
    Xs = _sequence_seq(X)
    ycs, yrs = _sequence_targets(yc, yr)
    r = ml.RNNRegressor(hidden_size=12, num_layers=2, learning_rate=1e-2, batch_size=32, max_epochs=2,
                        random_state=7).fit(Xs, yrs)
    c = ml.RNNClassifier(hidden_size=10, nonlinearity="relu", optimizer="adamw", learning_rate=1e-2,
                         batch_size=32, max_epochs=2, random_state=8).fit(Xs, ycs)
    Xhs = _sequence_seq(Xh)
    return _fit(dict(r_loss=_h(r.loss_curve_), r_params=_h(r.params_), r_pred=_h(r.predict(Xs)),
                     r_seq=_h(r.hidden_sequence(Xs[:16])),
                     c_loss=_h(c.loss_curve_), c_params=_h(c.params_), c_proba=_h(c.predict_proba(Xs))),
                r, lambda e: (e.predict(Xhs),))


@lane("sequence-lion")
def _(ml, X, yc, yr, Xh=None):
    """Lion at the paper's defaults and with weight decay and other betas,
    and an LSTM regressor trained by it."""
    a = _sequence_opt_run(ml, ml.Lion, X, lr=1e-3)
    b = _sequence_opt_run(ml, ml.Lion, X, lr=3e-3, betas=(0.95, 0.98), weight_decay=0.1)
    Xs = _sequence_seq(X)
    ycs, yrs = _sequence_targets(yc, yr)
    r = ml.LSTMRegressor(hidden_size=8, optimizer="lion", learning_rate=1e-3, batch_size=32, max_epochs=1,
                         random_state=9).fit(Xs, yrs)
    return _fit(dict(plain=a["params"], plain_state=a["state"], wd=b["params"], wd_state=b["state"],
                     lstm=_h(r.params_, r.loss_curve_)))


@lane("sequence-adafactor")
def _(ml, X, yc, yr, Xh=None):
    """Adafactor at torch's defaults and with weight decay, d and beta2_decay
    moved: the factored arm (a 32 x 8 matrix) and the vector arm."""
    a = _sequence_opt_run(ml, ml.Adafactor, X)
    b = _sequence_opt_run(ml, ml.Adafactor, X, lr=3e-2, beta2_decay=-0.6, d=2.0, weight_decay=0.1)
    return _fit(dict(plain=a["params"], plain_state=_h(*[v for s in a["opt"].state for v in s.values()]),
                     moved=b["params"], moved_state=_h(*[v for s in b["opt"].state for v in s.values()])))


@lane("sequence-lamb")
def _(ml, X, yc, yr, Xh=None):
    """LAMB at timm's defaults (global clip 1.0, weight decay 0.01) and with
    trust_clip, always_adapt, no decay and no clip."""
    a = _sequence_opt_run(ml, ml.LAMB, X, lr=1e-2)
    b = _sequence_opt_run(ml, ml.LAMB, X, lr=1e-2, weight_decay=0.0, always_adapt=True, trust_clip=True,
                          max_grad_norm=None)
    return _fit(dict(plain=a["params"], plain_state=a["state"], adapt=b["params"], adapt_state=b["state"]))


@lane("sequence-adamax")
def _(ml, X, yc, yr, Xh=None):
    """Adamax at torch's defaults and with other betas and weight decay, and
    a GRU regressor trained by it."""
    a = _sequence_opt_run(ml, ml.Adamax, X)
    b = _sequence_opt_run(ml, ml.Adamax, X, lr=1e-2, betas=(0.8, 0.99), weight_decay=0.05)
    Xs = _sequence_seq(X)
    ycs, yrs = _sequence_targets(yc, yr)
    r = ml.GRURegressor(hidden_size=8, optimizer="adamax", learning_rate=2e-3, batch_size=32, max_epochs=1,
                        random_state=10).fit(Xs, yrs)
    return _fit(dict(plain=a["params"], plain_state=a["state"], moved=b["params"], moved_state=b["state"],
                     gru=_h(r.params_, r.loss_curve_)))


@lane("sequence-nadam")
def _(ml, X, yc, yr, Xh=None):
    """NAdam at torch's defaults and with decoupled decay and a faster
    momentum schedule, and an RNN classifier trained by it."""
    a = _sequence_opt_run(ml, ml.NAdam, X)
    b = _sequence_opt_run(ml, ml.NAdam, X, lr=1e-2, weight_decay=0.05, decoupled_weight_decay=True,
                          momentum_decay=0.01)
    Xs = _sequence_seq(X)
    ycs, _ = _sequence_targets(yc, yr)
    c = ml.RNNClassifier(hidden_size=8, optimizer="nadam", learning_rate=2e-3, batch_size=32, max_epochs=1,
                         random_state=11).fit(Xs, ycs)
    return _fit(dict(plain=a["params"], plain_state=a["state"], moved=b["params"], moved_state=b["state"],
                     rnn=_h(c.params_, c.loss_curve_)))
