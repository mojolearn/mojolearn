# SPDX-License-Identifier: Apache-2.0
"""Inputs and reference values of the batch-8 Mojo checks that replaced numeric Python tests.

Lane py2mojo-A (2026-11-22, docs/plans/PY2MOJO_CONVERSION.md, batch 8). Raw little-endian words as in
`tools/py2mojo_native_fixtures.py` (`_put`): `.f32`, `.i32`, `.f64`, row-major.

    checks/native_tests/fixtures/xcnn_graph_numeric/  xcnn_graph_numeric.mojo  (test_x_cnn_gnn.py test_gcn, test_sage,
        test_sage_max_normalize, test_sage_project): the original NumPy draws of x, the edges and the upstream
        gradient; the layer weights (the original read the layer's own seeded init, which is not under test: here
        they are drawn by NumPy and handed to the same RUN calls); the float64 references of the original's
        ref_gcn / ref_sage / ref_sage_max_norm and its project composition.
    checks/native_tests/fixtures/tsa_ets_statsforecast/  tsa_sequence_ets.mojo  (test_x_sequence_ets.py
        test_against_statsforecast): statsforecast AutoETS(model="AAN", damped=True) 8-step means of the two
        stored series `fixtures/tsa/ets-y.f32`.
    checks/native_tests/fixtures/sequence_garch_arch/  sequence_garch.mojo  (test_x_sequence_garch.py
        test_matches_arch): arch_model(mean="Zero", p=1, q=1, rescale=False) params (omega, alpha[1], beta[1]) and
        log-likelihood on the stored series `checks/fixtures/sequence/garch_sim_n1500_seed0.f32`.
    checks/native_tests/fixtures/xseq_mlp_sklearn/  xseq_mlp_fit.mojo  (test_x_sequence_mlp.py
        test_against_sklearn): X and the two targets of `_data()`, the initial weights (Glorot-uniform bounds, w
        then b of each layer, drawn by NumPy default_rng(11) and handed to both fits: the original handed
        scikit-learn our seeded draws; either way both fits start from the same weights and the curves compare the
        optimisation alone), scikit-learn's loss_curve_ of each fit.

    checks/native_tests/fixtures/sequence_prophet_structure/  prophet_state.mojo  (test_x_sequence_prophet.py
        test_recovers_structure): the 200-point series y (trend break, weekly cycle, holiday effect and the
        default_rng(0) noise) as float32 words; t and the holiday column are integer formulas rebuilt in Mojo.

Generated with numpy 2.5.5, scikit-learn 1.9.1, statsforecast 2.1.1, arch 9.0.0 (2026-11-22), after
`python3 tools/fixture_packs.py unpack` (it reads the two stored series). Regenerate and repack:

    python tools/py2mojo_native_fixtures_b8.py && python3 tools/fixture_packs.py pack checks/native_tests/fixtures

The sets are independent and are written one per process across the host cores.
"""
from concurrent.futures import ProcessPoolExecutor
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from py2mojo_native_fixtures import _put  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
GNN = "xcnn_graph_numeric"


def _graph(n, seed):
    # test_x_cnn_gnn.py::_graph
    rng = np.random.default_rng(seed)
    e = rng.integers(0, n, size=(2, 3 * n))
    e = np.concatenate([e, [[1, 2, 2], [1, 3, 3]]], axis=1)
    return e[:, e[1] != 0]


def ref_gcn(x, ei, w, W, b, improved, g):
    n = x.shape[0]
    src, dst = ei
    w = np.ones(len(src)) if w is None else w.astype(np.float64)
    loop = src == dst
    lw = np.full(n, 2.0 if improved else 1.0)
    lw[src[loop]] = w[loop]
    src = np.concatenate([src[~loop], np.arange(n)])
    dst = np.concatenate([dst[~loop], np.arange(n)])
    w = np.concatenate([w[~loop], lw])
    deg = np.zeros(n)
    np.add.at(deg, dst, w)
    dis = np.where(deg > 0, deg ** -0.5, 0)
    norm = dis[src] * w * dis[dst]
    A = np.zeros((n, n))
    np.add.at(A, (dst, src), norm)
    h = x.astype(np.float64) @ W.T.astype(np.float64)
    out = A @ h + b
    dh = A.T @ g
    return out, dh @ W.astype(np.float64), dh.T @ x.astype(np.float64), g.sum(0)


def ref_sage(x, ei, Wl, bl, Wr, aggr, g):
    n = x.shape[0]
    src, dst = ei
    A = np.zeros((n, n))
    np.add.at(A, (dst, src), 1.0)
    if aggr == "mean":
        deg = A.sum(1, keepdims=True)
        A = np.where(deg > 0, A / np.maximum(deg, 1), 0)
    x64 = x.astype(np.float64)
    agg = A @ x64
    out = agg @ Wl.T + bl + x64 @ Wr.T
    dx = (A.T @ (g @ Wl)) + g @ Wr
    return out, dx, g.T @ agg, g.sum(0), g.T @ x64


def ref_sage_max_norm(x, ei, Wl, bl, Wr, normalize, g):
    n, F = x.shape
    src, dst = ei
    x64 = x.astype(np.float64)
    agg = np.zeros((n, F))
    cnt = np.zeros((n, F))
    for t in range(n):
        m = src[dst == t]
        if len(m):
            vals = x64[m]
            agg[t] = vals.max(0)
            cnt[t] = (vals == agg[t]).sum(0)
    pre = agg @ Wl.T + bl + x64 @ Wr.T
    if normalize:
        den = np.maximum(np.linalg.norm(pre, axis=1, keepdims=True), 1e-12)
        y = pre / den
        G = (g - y * (g * y).sum(1, keepdims=True)) / den
    else:
        y, G = pre, g
    dagg = G @ Wl
    dx = G @ Wr
    for e in range(len(src)):
        s_, t = src[e], dst[e]
        hit = x64[s_] == agg[t]
        dx[s_] += np.where(hit, dagg[t] / np.maximum(cnt[t], 1), 0)
    return y, dx


def _weights(rng, f, d):
    bound = 1.0 / np.sqrt(d)
    return (rng.uniform(-bound, bound, (f, d)).astype(np.float32), rng.uniform(-bound, bound, f).astype(np.float32))


def _edges(prefix, ei):
    return [_put(GNN, f"{prefix}-src.i32", ei[0], np.int32), _put(GNN, f"{prefix}-dst.i32", ei[1], np.int32)]


def gcn():
    out = []
    for unit, (improved, weighted) in enumerate(((False, False), (True, True))):
        # test_gcn: the original draw order (x, edges, edge weights, bias, gradient)
        rng = np.random.default_rng(0)
        n = 40
        x = rng.standard_normal((n, 6)).astype(np.float32)
        ei = _graph(n, 1)
        w = (rng.random(ei.shape[1]) + 0.5).astype(np.float32) if weighted else None
        bias = rng.standard_normal(4).astype(np.float32)
        g = rng.standard_normal((n, 4)).astype(np.float32)
        W = _weights(np.random.default_rng(100 + unit), 4, 6)[0]
        ry, rdx, rdw, rdb = ref_gcn(x, ei, w, W, bias, improved, g.astype(np.float64))
        p = f"gcn{unit}"
        out += _edges(p, ei)
        out += [_put(GNN, f"{p}-x.f32", x, np.float32), _put(GNN, f"{p}-w.f32", W, np.float32),
                _put(GNN, f"{p}-b.f32", bias, np.float32), _put(GNN, f"{p}-g.f32", g, np.float32),
                _put(GNN, f"{p}-y.f64", ry, np.float64), _put(GNN, f"{p}-dx.f64", rdx, np.float64),
                _put(GNN, f"{p}-dw.f64", rdw, np.float64), _put(GNN, f"{p}-db.f64", rdb, np.float64)]
        if weighted:
            out.append(_put(GNN, f"{p}-ew.f32", w, np.float32))
    return out


def sage():
    out = []
    for unit, aggr in enumerate(("mean", "sum")):
        # test_sage: x, edges, gradient
        rng = np.random.default_rng(3)
        n = 40
        x = rng.standard_normal((n, 6)).astype(np.float32)
        ei = _graph(n, 4)
        g = rng.standard_normal((n, 5)).astype(np.float32)
        wrng = np.random.default_rng(200 + unit)
        Wl, bl = _weights(wrng, 5, 6)
        Wr = _weights(wrng, 5, 6)[0]
        ry, rdx, rdwl, rdbl, rdwr = ref_sage(x, ei, Wl, bl, Wr, aggr, g.astype(np.float64))
        p = f"sage{unit}"
        out += _edges(p, ei)
        out += [_put(GNN, f"{p}-x.f32", x, np.float32), _put(GNN, f"{p}-wl.f32", Wl, np.float32),
                _put(GNN, f"{p}-bl.f32", bl, np.float32), _put(GNN, f"{p}-wr.f32", Wr, np.float32),
                _put(GNN, f"{p}-g.f32", g, np.float32), _put(GNN, f"{p}-y.f64", ry, np.float64),
                _put(GNN, f"{p}-dx.f64", rdx, np.float64), _put(GNN, f"{p}-dwl.f64", rdwl, np.float64),
                _put(GNN, f"{p}-dbl.f64", rdbl, np.float64), _put(GNN, f"{p}-dwr.f64", rdwr, np.float64)]
    return out


def sage_max():
    out = []
    for unit, normalize in enumerate((False, True)):
        # test_sage_max_normalize: integer x (ties), edges, gradient
        rng = np.random.default_rng(6)
        n = 30
        x = rng.integers(-2, 3, size=(n, 5)).astype(np.float32)
        ei = _graph(n, 7)
        g = rng.standard_normal((n, 4)).astype(np.float32)
        wrng = np.random.default_rng(300 + unit)
        Wl, bl = _weights(wrng, 4, 5)
        Wr = _weights(wrng, 4, 5)[0]
        ry, rdx = ref_sage_max_norm(x, ei, Wl, bl, Wr, normalize, g.astype(np.float64))
        p = f"smax{unit}"
        out += _edges(p, ei)
        out += [_put(GNN, f"{p}-x.f32", x, np.float32), _put(GNN, f"{p}-wl.f32", Wl, np.float32),
                _put(GNN, f"{p}-bl.f32", bl, np.float32), _put(GNN, f"{p}-wr.f32", Wr, np.float32),
                _put(GNN, f"{p}-g.f32", g, np.float32), _put(GNN, f"{p}-y.f64", ry, np.float64),
                _put(GNN, f"{p}-dx.f64", rdx, np.float64)]
    return out


def sage_project():
    out = []
    for unit, aggr in enumerate(("mean", "max")):
        # test_sage_project: x, edges, gradient
        rng = np.random.default_rng(9)
        n = 30
        x = rng.standard_normal((n, 5)).astype(np.float32)
        ei = _graph(n, 10)
        g = rng.standard_normal((n, 4)).astype(np.float32)
        wrng = np.random.default_rng(400 + unit)
        Wp, bp = _weights(wrng, 5, 5)
        Wl, bl = _weights(wrng, 4, 5)
        Wr = _weights(wrng, 4, 5)[0]
        x64, g64 = x.astype(np.float64), g.astype(np.float64)
        Wp64, bp64 = Wp.astype(np.float64), bp.astype(np.float64)
        h = x64 @ Wp64.T + bp64
        pr = np.maximum(h, 0)
        src, dst = ei
        if aggr == "mean":
            A = np.zeros((n, n))
            np.add.at(A, (dst, src), 1.0)
            deg = A.sum(1, keepdims=True)
            A = np.where(deg > 0, A / np.maximum(deg, 1), 0)
            agg = A @ pr
            dp = A.T @ (g64 @ Wl)
        else:
            agg, _ = ref_sage_max_norm(pr, ei, np.eye(5), np.zeros(5), np.zeros((5, 5)), False, np.zeros((n, 5)))
            _, dp = ref_sage_max_norm(pr, ei, np.eye(5), np.zeros(5), np.zeros((5, 5)), False, g64 @ Wl)
        ry = agg @ Wl.T + bl + x64 @ Wr.T
        rdx = (dp * (h > 0)) @ Wp64 + g64 @ Wr
        rdwp = (dp * (h > 0)).T @ x64
        p = f"sproj{unit}"
        out += _edges(p, ei)
        out += [_put(GNN, f"{p}-x.f32", x, np.float32), _put(GNN, f"{p}-wp.f32", Wp, np.float32),
                _put(GNN, f"{p}-bp.f32", bp, np.float32), _put(GNN, f"{p}-wl.f32", Wl, np.float32),
                _put(GNN, f"{p}-bl.f32", bl, np.float32), _put(GNN, f"{p}-wr.f32", Wr, np.float32),
                _put(GNN, f"{p}-g.f32", g, np.float32), _put(GNN, f"{p}-y.f64", ry, np.float64),
                _put(GNN, f"{p}-dx.f64", rdx, np.float64), _put(GNN, f"{p}-dwp.f64", rdwp, np.float64)]
    return out


def ets():
    from statsforecast.models import AutoETS
    Y = np.fromfile(REPO / "checks/native_tests/fixtures/tsa/ets-y.f32", dtype="<f4").reshape(2, 80)
    want = [AutoETS(model="AAN", damped=True).forecast(Y[i].astype(np.float64), 8)["mean"] for i in range(2)]
    return [_put("tsa_ets_statsforecast", "ets-sf-mean.f64", np.stack(want), np.float64)]


def garch():
    import arch
    y = np.fromfile(REPO / "checks/fixtures/sequence/garch_sim_n1500_seed0.f32", dtype="<f4")
    a = arch.arch_model(y.astype(np.float64), mean="Zero", p=1, q=1, rescale=False).fit(disp="off")
    return [_put("sequence_garch_arch", "arch-params.f64", a.params.values, np.float64),
            _put("sequence_garch_arch", "arch-ll.f64", [a.loglikelihood], np.float64)]


def mlp():
    from sklearn import neural_network as sk
    # test_x_sequence_mlp.py::_data
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 5)).astype(np.float32)
    yr = (X[:, 0] * 2 - X[:, 1]).astype(np.float32)
    y3 = (X[:, 0] > 0).astype(int) + (X[:, 1] > 0.5)
    fam = "xseq_mlp_sklearn"
    out = [_put(fam, "x.f32", X, np.float32), _put(fam, "yr.f32", yr, np.float32), _put(fam, "y3.i32", y3, np.int32)]
    cases = ((sk.MLPRegressor, yr, dict(hidden_layer_sizes=(8, 4), activation="tanh", max_iter=15), 1),
             (sk.MLPClassifier, y3, dict(hidden_layer_sizes=(6,), solver="sgd", max_iter=15), 3))
    init = np.random.default_rng(11)
    for unit, (theirs, y, kw, n_out) in enumerate(cases):
        sizes = [X.shape[1]] + list(kw["hidden_layer_sizes"]) + [n_out]
        coefs, flat = [], []
        for fi, fo in zip(sizes[:-1], sizes[1:]):
            bound = float(np.sqrt((2.0 if kw.get("activation") == "logistic" else 6.0) / (fi + fo)))
            w = init.uniform(-bound, bound, (fi, fo)).astype(np.float32)
            c = init.uniform(-bound, bound, fo).astype(np.float32)
            coefs.append((w, c))
            flat += [w.ravel(), c]
        b = theirs(shuffle=False, random_state=0, batch_size=50, **kw)
        layers = iter(coefs)

        def init_coef(fan_in, fan_out, dtype, layers=layers):
            w, c = next(layers)
            assert w.shape == (fan_in, fan_out)
            return w.astype(dtype), c.astype(dtype)
        b._init_coef = init_coef
        b.fit(X, y)
        out += [_put(fam, f"init{unit}.f32", np.concatenate(flat), np.float32),
                _put(fam, f"curve{unit}.f64", b.loss_curve_, np.float64)]
    return out


def prophet():
    # test_x_sequence_prophet.py::test_recovers_structure
    rng = np.random.default_rng(0)
    N = 200
    i = np.arange(N)
    hol = (np.arange(N + 14) % 50 == 7).astype(np.float32)
    y = (10 + np.where(i < 100, 0.05 * i, 5 - 0.02 * (i - 100)) + np.sin(2 * np.pi * i / 7)
         + 3 * hol[:N] + 0.1 * rng.standard_normal(N)).astype(np.float32)
    return [_put("sequence_prophet_structure", "prophet-y.f32", y, np.float32)]


SETS = (prophet, gcn, sage, sage_max, sage_project, ets, garch, mlp)


def _run(index):
    return SETS[index]()


def main():
    with ProcessPoolExecutor(max_workers=min(len(SETS), os.cpu_count() or 1)) as pool:
        for lines in pool.map(_run, range(len(SETS))):
            for line in lines:
                print(line)


if __name__ == "__main__":
    main()
