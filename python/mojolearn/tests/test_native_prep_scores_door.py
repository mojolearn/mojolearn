# SPDX-License-Identifier: Apache-2.0
"""The univariate score functions and TargetEncoder.fit_transform are one
native call (bindings/prep_owner_native.mojo, bindings/prep_encode_native.mojo).
The former Python routines are kept below as the oracle: the door must return
the same bytes. The shell checks run without bindings; the rest need the
compiled package."""
import ast
import pathlib

import pytest

_SRC = pathlib.Path(__file__).resolve().parents[1] / "_expansion_prep.py"
_TREE = ast.parse(_SRC.read_text())
_ORACLE_NAMES = ("_x2d", "encode_labels", "_Prog", "_cls", "_cs", "_mode", "_blocked", "_idn_fam",
                 "_IDN_SELECT_BLOCKED", "_XB", "_NONE", "as_f32_c")


def _def(path):
    # the function (or Class.method) definition in the module source, without importing it
    nodes = _TREE.body
    node = None
    for part in path.split("."):  # glue: test helper walking the module source
        node = next(n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part)
        nodes = node.body
    return node


def _one_call(path):
    body = [s for s in _def(path).body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    assert len(body) == 1 and isinstance(body[0], ast.Return), path
    call = body[0].value
    assert isinstance(call, ast.Call) and call.func.id == "_ENCODE_CALL", path


@pytest.mark.parametrize("name", ["f_classif", "chi2", "f_regression", "r_regression",
                                  "TargetEncoder.fit_transform"])
def test_shell_is_one_native_call(name):
    _one_call(name)


def test_classifier_fit_transform_raises_only():
    body = [s for s in _def("_Classifier.fit_transform").body
            if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    assert len(body) == 1 and isinstance(body[0], ast.Raise)


@pytest.fixture
def P():
    # the compiled package; the oracle below reads the module's helpers as globals
    try:
        from mojolearn import _expansion_prep as module
    except Exception as error:  # noqa: BLE001  (no bindings in a lane worktree)
        pytest.skip(f"compiled mojolearn unavailable: {error}")
    for name in _ORACLE_NAMES:  # glue: test oracle setup
        globals()[name] = getattr(module, name)
    return module


def test_classifier_fit_transform_unavailable(P):
    for cls in (P.GaussianNB, P.MultinomialNB, P.BernoulliNB, P.ComplementNB, P.CategoricalNB,
                P.QuadraticDiscriminantAnalysis):
        assert not hasattr(cls(), "fit_transform"), cls


# ---------------------------------------------------------------- the former Python routines (oracle)
def _scores_classif(X, y, kind):
    arr = _x2d(X)
    n, d = arr.shape
    classes, codes = encode_labels(y)
    if codes.size != n:
        raise ValueError("mojolearn: X and y have different numbers of rows")
    K = len(classes)
    pr = _Prog()
    xo, yo = pr.put(arr), pr.put_codes(codes)
    cnt, mean, sums = pr.alloc(K), pr.alloc(K * d), pr.alloc(K * d)
    sc, pv, st = pr.alloc(d), pr.alloc(d), pr.alloc(6 * d)
    mode = _mode()
    _cls(pr, mode, xo, n, d, yo, K, cnt, mean, _NONE, sums)
    if kind == "chi2":
        _cs(pr, mode, xo, n, d, st, var=False)
        pr.stage("chi2", d, sums, K, d, cnt, n, sc, pv)
    elif _blocked() and _idn_fam(mode) & _IDN_SELECT_BLOCKED:
        # x_prep/select_blocked.mojo: the within-class squares by row blocks, then the score
        nb = (n + _XB - 1) // _XB
        ps = pr.work(nb * d)
        pr.stage("fcb_part", nb * d, xo, n, d, yo, mean, ps, nb)
        pr.stage("fcb_fin", d, ps, n, d, nb, K, cnt, mean, sc, pv)
    else:
        pr.stage("f_classif", d, xo, n, d, yo, K, cnt, mean, sc, pv)
    pr.run(mode)
    if kind == "chi2" and any(v < 0 for v in pr.values(st + 3 * d, d)):  # glue: raises on a negative per-column minimum (d-sized: feature count)
        raise ValueError("mojolearn: Input X must be non-negative.")
    return pr.get(sc, d), pr.get(pv, d)


def _pearson(X, y, center, force_finite):
    arr = _x2d(X)
    n, d = arr.shape
    yv = as_f32_c(y, ndim=1, name="y")[0]
    if yv.size != n:
        raise ValueError("mojolearn: X and y have different numbers of rows")
    pr = _Prog()
    xo, yo = pr.put(arr), pr.put(yv)
    sc, pv, co = pr.alloc(d), pr.alloc(d), pr.alloc(d)
    mode = _mode()
    if _blocked() and _idn_fam(mode) & _IDN_SELECT_BLOCKED:
        # x_prep/select_blocked.mojo: the sums and the centred products by row blocks
        nb = (n + _XB - 1) // _XB
        mx = my = _NONE
        if center:
            psx, psy, mx, my = pr.work(nb * d), pr.work(nb), pr.work(d), pr.work(1)
            pr.stage("frb_part1", nb * d, xo, n, d, yo, psx, psy, nb)
            pr.stage("frb_mean", d, psx, psy, nb, d, n, mx, my)
        pxy, pxx, pyy = pr.work(nb * d), pr.work(nb * d), pr.work(nb)
        pr.stage("frb_part2", nb * d, xo, n, d, yo, mx, my, pxy, pxx, pyy, nb)
        pr.stage("frb_fin", d, pxy, pxx, pyy, nb, d, n, 1 if center else 0, sc, pv, co, 1 if force_finite else 0)
    else:
        pr.stage("f_regression", d, xo, n, d, yo, 1 if center else 0, sc, pv, co, 1 if force_finite else 0)
    pr.run(mode)
    return pr.get(sc, d), pr.get(pv, d), pr.get(co, d)



def _bytes(a):
    import numpy as np
    return np.ascontiguousarray(np.asarray(a)).tobytes()


def _fixtures():
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(3)
    X = rng.random((300, 7), dtype=np.float32)
    X[:, 2] = 1.0                       # constant: NaN scores
    Xb = rng.integers(0, 6, (300, 7)).astype(np.float32)
    y3 = rng.integers(0, 3, 300)
    yr = (2 * X[:, 0] + rng.random(300)).astype(np.float32)
    return np, X, Xb, y3, yr


@pytest.mark.parametrize("kind", ["f", "chi2"])
def test_classif_scores_equal_former_path(P, kind):
    np, X, Xb, y3, _ = _fixtures()
    fn = P.f_classif if kind == "f" else P.chi2
    for x in (X, Xb):
        got = fn(x, y3)
        want = _scores_classif(x, y3, kind)
        assert [_bytes(g) for g in got] == [_bytes(w) for w in want]


@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("force_finite", [True, False])
def test_pearson_scores_equal_former_path(P, center, force_finite):
    np, X, _, _, yr = _fixtures()
    sc, pv, co = _pearson(X, yr, center, force_finite)
    got = P.f_regression(X, yr, center=center, force_finite=force_finite)
    assert [_bytes(g) for g in got] == [_bytes(sc), _bytes(pv)]
    assert _bytes(P.r_regression(X, yr, center=center, force_finite=force_finite)) == _bytes(co)


def test_chi2_negative_refusal(P):
    np, X, _, y3, _ = _fixtures()
    with pytest.raises(ValueError, match="Input X must be non-negative"):
        P.chi2(X - 0.5, y3)


def test_target_encoder_splitters(P):
    np, _, Xb, y3, yr = _fixtures()
    import mojolearn as ml
    pairs = list(ml.KFold(4).split(Xb, yr))
    a = ml.TargetEncoder(cv=ml.KFold(4)).fit_transform(Xb, yr)
    b = ml.TargetEncoder(cv=pairs).fit_transform(Xb, yr)
    c = ml.TargetEncoder(cv=tuple((list(tr), list(te)) for tr, te in pairs)).fit_transform(Xb, yr)
    assert _bytes(a) == _bytes(b) == _bytes(c)

    class Splits:
        def split(self, X, y=None):
            yield from pairs
    with pytest.raises(TypeError, match="got Splits"):
        ml.TargetEncoder(cv=Splits()).fit_transform(Xb, yr)
    with pytest.raises(ValueError, match="cover each sample index exactly once"):
        ml.TargetEncoder(cv=[(np.arange(300), np.arange(300))]).fit_transform(Xb, yr)
    bad = [(np.arange(10), te) for _, te in pairs]
    with pytest.raises(NotImplementedError, match="training rows are not every"):
        ml.TargetEncoder(cv=bad).fit_transform(Xb, yr)
