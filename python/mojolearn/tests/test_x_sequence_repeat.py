# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Every sequence-family entry point called TWICE in one process, on the GPU
binding and on the CPU host binding, with the second pass's output equal to
the first's byte for byte (CURRENT DIRECTIVES 2026-09-27: x_cluster and
x_neighbors hung on the second GPU call of a process because each call built
a DeviceContext whose buffers outlived it).

Covers `_mojolearn_x_sequence` (one process-lifetime context,
sequence/exec_device.mojo `sequence_ctx`) and the family's older bindings
`_mojolearn_arima` (through AutoARIMA) and `_mojolearn_tsa` (kpss_test,
ExponentialSmoothing). Each backend runs in a child process with a timeout,
so a hang is a failure, not a stuck test run."""
import os
import subprocess
import sys
import textwrap

import pytest

_CHILD = textwrap.dedent('''
    import numpy as np
    import mojolearn as ml

    rng = np.random.default_rng(3)
    t = np.arange(96)
    Y = np.stack([20 + 0.3 * t + 3 * np.sin(2 * np.pi * t / 12) + rng.standard_normal(96),
                  50 + 2 * np.cos(2 * np.pi * t / 12) + rng.standard_normal(96)]).astype(np.float32)
    Xs = rng.standard_normal((32, 5, 4)).astype(np.float32)
    ys = (Xs[:, -1, 0] + 0.5 * Xs[:, -2, 1]).astype(np.float32)
    yc = (ys > 0).astype(np.int64)
    Xm = rng.standard_normal((40, 4)).astype(np.float32)
    ym = (Xm[:, 0] - Xm[:, 1]).astype(np.float32)
    inter = np.where(rng.random((2, 60)) < 0.3, rng.integers(1, 5, (2, 60)), 0).astype(np.float32)
    V = rng.standard_normal((80, 3)).astype(np.float32).cumsum(0) * np.float32(0.1)
    G = (rng.standard_normal(400) * 0.5).astype(np.float32)
    days = np.arange(120, dtype=np.float64) + 18000.0
    yp = (10 + 0.05 * np.arange(120) + np.sin(2 * np.pi * np.arange(120) / 7)).astype(np.float32)
    xn = rng.standard_normal((6, 5)).astype(np.float32)

    def run():
        out = []
        for cls in (ml.LSTMRegressor, ml.GRURegressor, ml.RNNRegressor):              # rnn_fit, rnn_predict
            m = cls(hidden_size=4, batch_size=16, max_epochs=2).fit(Xs, ys)
            out += [m.predict(Xs)]
        for cls in (ml.LSTMClassifier, ml.GRUClassifier, ml.RNNClassifier):
            out += [cls(hidden_size=4, batch_size=16, max_epochs=2).fit(Xs, yc).predict_proba(Xs)]
        for cls in (ml.RMSprop, ml.Adagrad, ml.Lion, ml.Adamax, ml.NAdam, ml.LAMB):  # opt_step, lamb_step
            p = np.ones((4, 3), dtype=np.float32)
            o = cls([p])
            for k in range(3):
                o.step([np.full((4, 3), 0.1 * (k + 1), dtype=np.float32)])
            out += [p]
        W = np.ones((4, 3), dtype=np.float32)
        a = ml.Adafactor([W])                                                         # adafactor_step
        for k in range(3):
            a.step([np.full((4, 3), 0.1 * (k + 1), dtype=np.float32)])
        out += [W]
        r = ml.STL(Y, period=12).fit()                                                # stl
        out += [r.seasonal, r.trend]
        vr = ml.VAR(V).fit(maxlags=2)                                                 # var_fit, var_forecast
        out += [vr.params, vr.forecast(V, 3)]
        out += [ml.MLPRegressor(hidden_layer_sizes=(5,), max_iter=3, random_state=0).fit(Xm, ym).predict(Xm)]
        out += [ml.layer_norm_forward(xn), *ml.layer_norm_backward(xn, xn)]           # layer_norm
        out += [ml.AutoTheta(season_length=12).fit(Y).predict(4)["mean"]]             # theta
        out += [ml.CrostonOptimized().fit(inter).predict(3)["mean"]]                  # croston
        out += [ml.DampedETS().fit(Y).predict(5)["mean"]]                             # ets
        g = ml.GARCH().fit(G, horizon=2)                                              # garch
        out += [np.asarray(g.params_), np.asarray(g.forecast(2))]
        pf = ml.ProphetForecaster().fit(days, yp)                                     # prophet fit / predict
        out += [pf.predict(days[:10])]
        mo = ml.MoEBlock(4, 6, num_experts=3, top_k=2, random_state=1)                # moe_forward
        out += [mo(xn[:, :4])]
        am = ml.AutoARIMA(Y).search(d=range(2), p=range(2), q=range(2), ic="aic")     # _mojolearn_arima
        am.fit()
        out += [am.forecast(3)]
        out += [np.asarray(ml.kpss_test(Y), dtype=np.float32)]                        # _mojolearn_tsa
        hw = ml.ExponentialSmoothing(Y.copy(), seasonal_periods=12, ts_num=2).fit()
        out += [np.asarray(hw.forecast(4))]
        return [np.ascontiguousarray(v).tobytes() for v in out]

    first = run()
    second = run()
    assert len(first) == len(second)
    for i, (u, v) in enumerate(zip(first, second)):
        assert u == v, f"output {i} of the second pass differs from the first"
    print("TWICE OK", ml.vendor(), len(first))
''')


def _run(env_extra):
    env = dict(os.environ)
    env.update(env_extra)
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env["PYTHONPATH"] = here + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, "-c", _CHILD], env=env, capture_output=True, text=True, timeout=1800)


def _binding_present(name):
    try:
        from mojolearn import _backend
        _backend.binding(name, "identical")
        return True
    except Exception:
        return False


@pytest.mark.parametrize("backend", ["gpu", "cpu"])
def test_every_sequence_entry_twice_in_one_process(backend):
    if backend == "gpu":
        if not _binding_present("_mojolearn_x_sequence"):
            pytest.skip("no GPU x_sequence binding in this install")
        r = _run({})
    else:
        r = _run({"MOJOLEARN_VENDOR": "cpu"})
    if r.returncode != 0 and "No module named" in r.stderr and backend == "cpu":
        pytest.skip("no CPU host binding in this install")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-4000:]
    assert "TWICE OK" in r.stdout
