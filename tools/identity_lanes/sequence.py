# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# THE SEQUENCE LANE'S IDENTITY LANES.
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
    return _fit(dict(d=_h(m.d_), order=_h(m.order_), ic=_h(m.ic_), forecast=_h(m.forecast(12))), est=[("forecast", 12, lambda h: (m.forecast(h),), -1)])


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
                     r_seasonal=_h(b.seasonal), r_trend=_h(b.trend), r_weights=_h(b.weights)), est=y)


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
                     forecast=_h(a.forecast(y, 10)), n_params=_h(b.params), n_forecast=_h(b.forecast(y, 10))), est=[("VAR", 10, lambda h: (a.forecast(y, h), b.forecast(y, h)), 0)])


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


@lane("sequence-lr-schedulers")
def _(ml, X, yc, yr, Xh=None):
    """StepLR, ExponentialLR and OneCycleLR (cosine two-phase, linear
    three-phase) over 40 steps as float32 bits, and an LSTM regressor and a
    NAdam run driven by schedules."""
    sched = [ml.StepLR(0.1, step_size=7, gamma=0.5), ml.ExponentialLR(0.05, gamma=0.9),
             ml.OneCycleLR(0.2, total_steps=40), ml.OneCycleLR(0.2, total_steps=40, anneal_strategy="linear",
                                                               three_phase=True, pct_start=0.25)]
    bits = np.asarray([[s.bits_at(t) for t in range(1, 41)] for s in sched], dtype=np.uint32)
    Xs = _sequence_seq(X)
    ycs, yrs = _sequence_targets(yc, yr)
    r = ml.LSTMRegressor(hidden_size=8, batch_size=32, max_epochs=2, random_state=12,
                         lr_schedule=ml.OneCycleLR(0.02, total_steps=6)).fit(Xs, yrs)
    p1 = np.ascontiguousarray(X[:16, :4], dtype=np.float32).copy()
    opt = ml.NAdam([p1])
    opt.lr_schedule = ml.StepLR(1e-2, step_size=2)
    for k in range(5):
        opt.step([np.ascontiguousarray(X[16 + 16 * k:32 + 16 * k, 4:8], dtype=np.float32)])
    return _fit(dict(bits=_h(bits), lstm=_h(r.params_, r.loss_curve_), nadam=_h(p1)), est=(sched, r))


@lane("sequence-layernorm")
def _(ml, X, yc, yr, Xh=None):
    """LayerNorm over the 16 columns of 512 rows with a fixture-derived
    weight and bias, forward and backward (dy from further rows), and the
    functional form without affine over a (2, 8) normalized shape."""
    x = np.ascontiguousarray(X[:512], dtype=np.float32)
    ln = ml.LayerNorm(x.shape[1])
    ln.weight[:] = np.float32(1.0) + np.float32(0.125) * np.ascontiguousarray(X[512, :x.shape[1]], dtype=np.float32)
    ln.bias[:] = np.ascontiguousarray(X[513, :x.shape[1]], dtype=np.float32)
    y = ln(x)
    dx = ln.backward(np.ascontiguousarray(X[1024:1536], dtype=np.float32))
    x3 = np.ascontiguousarray(X[:256, :16], dtype=np.float32).reshape(256, 2, 8)
    y3 = ml.layer_norm_forward(x3, (2, 8), eps=1e-3)
    Xh3 = np.ascontiguousarray(Xh[:256, :x.shape[1]], dtype=np.float32)
    return _fit(dict(y=_h(y), dx=_h(dx), dw=_h(ln.weight_grad), db=_h(ln.bias_grad), y3=_h(y3)),
                ln, lambda e: (e.forward(Xh3),))


@lane("sequence-theta")
def _(ml, X, yc, yr, Xh=None):
    """Four series of 96 observations: a positive seasonal series (period 12,
    multiplicative after the ACF test), the same minus its minimum (additive),
    a random walk and a fixture column. AutoTheta over the batch and a
    DynamicOptimizedTheta with a fixed alpha; 18-step forecasts."""
    t = np.arange(96, dtype=np.float32)
    wave = np.sin(t * np.float32(2 * np.pi / 12)).astype(np.float32)
    c = np.ascontiguousarray(X[:96, 7], dtype=np.float32)
    y = np.stack([np.float32(20.0) + np.float32(4.0) * wave + np.float32(0.3) * c + np.float32(0.05) * t,
                  np.float32(4.0) * wave + np.float32(0.3) * c,
                  np.cumsum(c, dtype=np.float32), c]).astype(np.float32)
    a = ml.AutoTheta(season_length=12).fit(y)
    fa = a.predict(18)["mean"]
    d = ml.DynamicOptimizedTheta(season_length=12, alpha=0.3).fit(y)
    fd = d.predict(18)["mean"]
    return _fit(dict(auto=_h(fa), auto_info=_h(a.info_), dotm=_h(fd), dotm_info=_h(d.info_)), est=[("Theta", 18, lambda h: (a.predict(h)["mean"], d.predict(h)["mean"]), -1)])


@lane("sequence-croston")
def _(ml, X, yc, yr, Xh=None):
    """Six intermittent series of 120 observations (fixture columns kept
    where they exceed a threshold, one all-zero, one with negative events):
    the classic, optimized and SBA forecasts."""
    c = np.ascontiguousarray(X[:120, :6].T, dtype=np.float32)
    y = np.where(c > np.float32(0.8), c + np.float32(1.0), np.float32(0.0)).astype(np.float32)
    y[4] = np.float32(0.0)
    y[5] = np.where(c[5] < np.float32(-1.0), c[5], y[5]).astype(np.float32)
    out, models = {}, []
    for k, cls in (("classic", ml.CrostonClassic), ("optimized", ml.CrostonOptimized), ("sba", ml.CrostonSBA)):
        model = cls().fit(y)
        out[k] = _h(model.predict(4)["mean"])
        models.append((k, 4, lambda h, model=model: (model.predict(h)["mean"],), -1))
    return _fit(out, est=models)


@lane("sequence-ets")
def _(ml, X, yc, yr, Xh=None):
    """Four series of 80 observations (a trend plus a fixture column, a
    random walk, a positive level series, a fixture column): damped
    ETS(A,Ad,N), ETS(M,Ad,N) on the positive rows, undamped AAN and simple
    ANN; 12-step forecasts. Seasonal (period 12, a fixed wave added): AAA
    and damped MAM on 80 observations (decomposition initial states), ANA
    on the first 30 (the Fourier initial states), MNM with period 5 on the
    positive rows."""
    t = np.arange(80, dtype=np.float32)
    c = np.ascontiguousarray(X[:80, 9], dtype=np.float32)
    y = np.stack([np.float32(5.0) + np.float32(0.3) * t + c, np.cumsum(c, dtype=np.float32),
                  np.float32(40.0) + c, c]).astype(np.float32)
    pos = np.ascontiguousarray(y[:1] + np.float32(10.0) - np.minimum(y[:1].min(), np.float32(0.0)))
    out = dict(damped=_h(ml.DampedETS().fit(y).predict(12)["mean"]),
               mult=_h(ml.DampedETS(error="M").fit(pos).predict(12)["mean"]),
               aan=_h(ml.ETS(model="AAN", damped=False).fit(y).predict(12)["mean"]),
               ann=_h(ml.ETS(model="ANN", damped=False).fit(y).predict(12)["mean"]))
    wave = (np.float32(3.0) * np.sin(np.float32(2.0 * np.pi / 12.0) * t)).astype(np.float32)
    ys = (y + wave).astype(np.float32)
    ps = (np.float32(20.0) + np.abs(y) + wave).astype(np.float32)
    p5 = (np.float32(20.0) + np.abs(y) + np.float32(2.0) * np.cos(np.float32(2.0 * np.pi / 5.0) * t)).astype(np.float32)
    aaa = ml.ETS(season_length=12, model="AAA", damped=False).fit(ys)
    out.update(aaa=_h(aaa.predict(12)["mean"], aaa.info_, aaa.seasonal_states_),
               mamd=_h(ml.ETS(season_length=12, model="MAM", damped=True).fit(ps).predict(12)["mean"]),
               ana30=_h(ml.ETS(season_length=12, model="ANA", damped=False).fit(
                   np.ascontiguousarray(ys[:, :30])).predict(12)["mean"]),
               mnm5=_h(ml.ETS(season_length=5, model="MNM", damped=False).fit(p5).predict(12)["mean"]))
    return _fit(out, est=[("ETS AAA", 12, lambda h: (aaa.predict(h)["mean"],), -1)])


@lane("sequence-garch")
def _(ml, X, yc, yr, Xh=None):
    """Three return-like series of 300 observations built from fixture
    columns by a GARCH(1,1) filter: GARCH(1,1) with a constant mean,
    GJR-GARCH(1,1,1) with a zero mean; parameters, log-likelihoods,
    conditional volatility and 5-step variance forecasts."""
    z = np.ascontiguousarray(X[:300, 10:13].T, dtype=np.float32)
    # bounded shocks, |u| < 1.4, so the filter is stationary on every fixture
    zz = (np.float32(1.4) * z / (np.float32(1.0) + np.abs(z))).astype(np.float32)
    r = np.zeros_like(zz)
    s2 = np.full(3, np.float32(1.0), dtype=np.float32)
    for t in range(300):
        r[:, t] = (np.sqrt(s2) * zz[:, t]).astype(np.float32)
        s2 = (np.float32(0.1) + np.float32(0.1) * r[:, t] * r[:, t] + np.float32(0.8) * s2).astype(np.float32)
    g = ml.GARCH().fit(r, horizon=5)
    j = ml.GARCH(p=1, o=1, q=1, mean="Zero").fit(r, horizon=5)
    return _fit(dict(params=_h(g.params_), ll=_h(g.loglikelihood_), vol=_h(g.conditional_volatility_),
                     fc=_h(g.forecast(5)), gjr=_h(j.params_, j.loglikelihood_, j.forecast(5))), est=[("GARCH", 5, lambda h: (g.forecast(h), j.forecast(h)), -1)])


@lane("sequence-prophet")
def _(ml, X, yc, yr, Xh=None):
    """Two daily series over 120 days with a trend break, a weekly cycle and
    one holiday column (the auto weekly seasonality and 25 changepoints),
    additive, and the same with multiplicative seasonality; 21-day
    forecasts with the holiday continued."""
    t = np.arange(120, dtype=np.float64) + 19000.0
    c = np.ascontiguousarray(X[:120, 13:15].T, dtype=np.float32)
    k = np.where(np.arange(120) < 60, np.float32(0.05), np.float32(-0.02)).astype(np.float32)
    trend = np.cumsum(k, dtype=np.float32) + np.float32(10.0)
    week = np.sin(np.arange(120, dtype=np.float32) * np.float32(2 * np.pi / 7)).astype(np.float32)
    hol = (np.arange(141) % 30 == 5).astype(np.float32)
    y = (trend + week + np.float32(3.0) * hol[:120] + np.float32(0.2) * c).astype(np.float32)
    tf = np.arange(120, 141, dtype=np.float64) + 19000.0
    a = ml.ProphetForecaster().fit(t, y, holidays=hol[:120, None])
    fa = a.predict(tf, holidays=hol[120:, None])
    m = ml.ProphetForecaster(seasonality_mode="multiplicative").fit(t, y, holidays=hol[:120, None])
    fm = m.predict(tf, holidays=hol[120:, None])
    return _fit(dict(a_params=_h(a.params_), a_fc=_h(fa), m_params=_h(m.params_), m_fc=_h(fm)), est=[("Prophet", 21, lambda h: (a.predict(tf[:h], holidays=hol[120:120+h, None]), m.predict(tf[:h], holidays=hol[120:120+h, None])), -1)])


@lane("sequence-moe")
def _(ml, X, yc, yr, Xh=None):
    """A Mixtral block (16 -> 24, 6 experts, top 2) over 256 tokens of the
    fixture rows, and a top-3 block without renormalisation whose router has
    two identical rows (every probability tie, exercising the lower-index
    rule)."""
    x = np.ascontiguousarray(X[:256, :16], dtype=np.float32)
    a = ml.MoEBlock(16, 24, num_experts=6, top_k=2, random_state=1)
    ya = a(x)
    b = ml.MoEBlock(16, 12, num_experts=5, top_k=3, norm_topk_prob=False, random_state=2)
    router = b.router.copy()
    router[3] = router[1]
    b.router = router
    yb = b(x)
    return _fit(dict(ya=_h(ya), la=_h(a.router_logits_), sa=_h(a.selected_experts_), wa=_h(a.routing_weights_),
                     yb=_h(yb), sb=_h(b.selected_experts_), wb=_h(b.routing_weights_)),
                a, lambda e: (e(np.ascontiguousarray(Xh[:256, :16], dtype=np.float32)),))

# Batch declarations describe the returned estimator's public inference axis.
# Training minibatches are a separate property and are not inferred from these.
_batch_decl(_rows_calls("predict", "hidden_sequence", prep=_sequence_seq),
            "sequence-lstm", "sequence-gru", "sequence-rnn", revision="expansion-batch-2026-09-28-v1")
_batch_decl(_rows_calls("predict", sl=np.s_[:256, :10]), "sequence-mlp", revision="expansion-batch-2026-09-28-v1")
# The fitted LayerNorm uses every fixture column (odd has 17, base has 16).
_batch_decl(_rows_calls("forward", sl=slice(0, 256)), "sequence-layernorm", revision="layernorm-batch-width-2026-09-28-v2")
_batch_decl(_rows_calls("__call__", sl=np.s_[:256, :16]), "sequence-moe", revision="expansion-batch-2026-09-28-v1")
_batch_decl("n/a:parameter-coupled optimizer (Adafactor factors second moments across tensor rows and "
            "columns and scales updates by tensor RMS; splitting parameter rows changes the algorithm)",
            "sequence-adafactor", revision="expansion-batch-2026-09-28-v1")
_batch_decl("n/a:parameter-coupled optimizer (LAMB uses tensor norms for its trust ratio and global "
            "gradient clipping; a sliced parameter registry is a different optimization problem)",
            "sequence-lamb", revision="expansion-batch-2026-09-28-v1")


def _sequence_batch_forecasts(ml, entries, Xh):
    return [_BatchPrefix(label, horizon, call, axis) for label, horizon, call, axis in entries]


_batch_decl(_sequence_batch_forecasts, "sequence-autoarima", "sequence-var", "sequence-theta",
            "sequence-croston", "sequence-ets", "sequence-garch", "sequence-prophet", revision="expansion-batch-2026-09-28-v1")


def _sequence_batch_stl(ml, series, Xh):
    def fit_rows(rows, options):
        model = ml.STL(rows, period=12, **options).fit()
        return tuple(np.asarray(getattr(model, name)) for name in ("seasonal", "trend", "resid"))
    return [_BatchRows("STL " + label, series, lambda r, options=options: fit_rows(r, options))
            for label, options in (("default", {}), ("robust", dict(robust=True, seasonal_deg=0,
                                      seasonal_jump=2, trend_jump=2, low_pass_jump=2)))]


_batch_decl(_sequence_batch_stl, "sequence-stl", revision="expansion-batch-2026-09-28-v1")


def _sequence_batch_schedulers(ml, fitted, Xh):
    schedules, rnn = fitted
    steps = np.arange(1, 41, dtype=np.int64).reshape(-1, 1)
    return [_BatchRows("scheduler bits at fixed absolute step", steps,
                       lambda rows: (np.asarray([[s.bits_at(int(t)) for s in schedules]
                                                  for t in rows[:, 0]], dtype=np.uint32),)),
            _BatchRows("scheduled LSTM predict", _sequence_seq(Xh), lambda rows: (rnn.predict(rows),))]


_batch_decl(_sequence_batch_schedulers, "sequence-lr-schedulers", revision="expansion-batch-2026-09-28-v1")


def _sequence_batch_optimizer(name, options):
    """Only coordinate-local optimizers; Adafactor/LAMB are declared separately."""
    def spec(ml, e, Xh):
        cls = getattr(ml, name)
        params = np.ascontiguousarray(Xh[:64, :8])
        gradients = [np.ascontiguousarray(Xh[64 + 64*j:128 + 64*j, 8:16]) for j in range(2)]
        indices = np.arange(64, dtype=np.int64).reshape(-1, 1)
        def evaluate(rows, kwargs):
            ids = rows[:, 0]
            p = np.ascontiguousarray(params[ids]).copy()
            optimizer = cls([p], **kwargs)
            for g in gradients:
                optimizer.step([np.ascontiguousarray(g[ids])])
            return (p,) + tuple(np.asarray(s).reshape(len(ids), 8) for s in optimizer.state)
        return [_BatchRows(name + " " + label, indices, lambda r, kw=kw: evaluate(r, kw))
                for label, kw in options]
    return spec


_batch_decl(_sequence_batch_optimizer("RMSprop", (("default", {}), ("centered", dict(
    lr=3e-3, centered=True, momentum=0.7, weight_decay=1e-2)))), "sequence-rmsprop", revision="expansion-batch-2026-09-28-v1")
_batch_decl(_sequence_batch_optimizer("Adagrad", (("default", {}), ("decay", dict(
    lr=5e-2, lr_decay=0.1, weight_decay=1e-2, initial_accumulator_value=0.1)))), "sequence-adagrad", revision="expansion-batch-2026-09-28-v1")
_batch_decl(_sequence_batch_optimizer("Lion", (("default", dict(lr=1e-3)), ("decay", dict(
    lr=3e-3, betas=(0.95, 0.98), weight_decay=0.1)))), "sequence-lion", revision="expansion-batch-2026-09-28-v1")
_batch_decl(_sequence_batch_optimizer("Adamax", (("default", {}), ("decay", dict(
    lr=1e-2, betas=(0.8, 0.99), weight_decay=0.05)))), "sequence-adamax", revision="expansion-batch-2026-09-28-v1")
_batch_decl(_sequence_batch_optimizer("NAdam", (("default", {}), ("decay", dict(
    lr=1e-2, weight_decay=0.05, decoupled_weight_decay=True, momentum_decay=0.01)))), "sequence-nadam", revision="expansion-batch-2026-09-28-v1")
