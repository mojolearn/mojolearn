# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/py-bugs regressions.

1. No platform libm and no interpreter-dependent sum in the Python front
   door's float math (DEVIATIONS 6900-6902): the pinned helpers are checked
   against exact or high-precision references, and the touched modules are
   scanned for a float `**`.
2. A search and a validation curve draw their folds once (scikit-learn).
3. learning_curve permutes each fold once and takes nested prefixes.
4. The RNN's row order is int32: its length is no longer capped at 2^24.
"""
import ast
import decimal
import random
import struct
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from mojolearn import _portable_math as pm

PKG = Path(__file__).resolve().parents[1]


def _ulps(a, b):
    ia = struct.unpack("<q", struct.pack("<d", a))[0]
    ib = struct.unpack("<q", struct.pack("<d", b))[0]
    return abs(ia - ib)


# ------------------------------------------------------------- 1. the helpers

@pytest.mark.skipif(sys.version_info < (3, 12), reason="the builtin is a plain fold before 3.12")
def test_nsum_is_the_builtin_sum_of_312():
    rng = random.Random(1)
    cases = [[-0.0], [-0.0, -0.0], [1e100, 1.0, -1e100], [float("inf"), 1.0], [3.0], [0.1] * 10]
    for _ in range(5000):
        cases.append([rng.choice([rng.uniform(-1, 1), rng.gauss(0, 1) * 10 ** rng.randint(-30, 30), -0.0, 0.0])
                      for _ in range(rng.randint(1, 30))])
    for c in cases:
        a, b = sum(c), pm.nsum(c)
        assert struct.pack("<d", a) == struct.pack("<d", b), c


def test_nsum_is_not_a_plain_fold():
    # 3.10/3.11's builtin gives 0.0 here; the twin gives 3.12+'s 1.0 everywhere
    assert pm.nsum([1e100, 1.0, -1e100]) == 1.0


def test_powi_is_correctly_rounded():
    rng = random.Random(2)
    for _ in range(2000):
        x = rng.choice([rng.random(), rng.uniform(0.9, 0.99999), 0.9, 0.999, rng.uniform(1, 2), -rng.random()])
        n = rng.randint(0, 60)
        assert pm.powi(x, n) == float(Fraction(x) ** n), (x, n)
    # large steps (the Adam bias correction) against an exact reference
    for x, n in [(0.9, 1000), (0.999, 5000), (0.99, 777)]:
        assert pm.powi(x, n) == float(Fraction(x) ** n)
    assert pm.powi(10.0, 6) == 1e6 and pm.powi(2.0, 0) == 1.0 and pm.powi(0.0, 3) == 0.0
    with pytest.raises(OverflowError):
        pm.powi(10.0, 400)


def test_powr_rounds_the_logspace_grid_once():
    ctx = decimal.Context(prec=90)
    for m in range(2, 40):
        for i in range(m):
            y = -4 + 8 * i / (m - 1)
            ref = float(ctx.power(decimal.Decimal(10), decimal.Decimal(y)))
            assert pm.powr(10.0, y) == ref, (m, i)
    # the case a Mac's pow misrounds (one ulp low): LogisticRegressionCV(Cs=16)
    assert pm.powr(10.0, -2.9333333333333336) == 0.0011659144011798312


ERFC = [  # mpmath.erfc at 200 bits, rounded to binary64
    (-5.5, 1.9999999999999927), (-3.0, 1.9999779095030015), (-1.1, 1.8802050695740817),
    (-0.5, 1.5204998778130465), (-0.001, 1.0011283787909693), (0.0, 1.0), (1e-20, 1.0),
    (0.2, 0.7772974107895215), (0.3, 0.6713732405408726), (0.6, 0.3961439091520741),
    (0.9, 0.20309178757716786), (1.0, 0.15729920705028513), (1.2, 0.08968602177036464),
    (1.5, 0.033894853524689274), (2.0, 0.004677734981047266), (2.8, 7.501319466545911e-05),
    (2.9, 4.109787809945886e-05), (3.5, 7.430983723414128e-07), (5.0, 1.537459794428035e-12),
    (5.9, 7.190409783550478e-17), (6.5, 3.8421483271206475e-20), (10.0, 2.088487583762545e-45),
    (20.0, 5.395865611607901e-176), (27.5, 0.0),
]


def test_erfc_is_fdlibm_accurate():
    for x, ref in ERFC:
        assert _ulps(pm.erfc(x), ref) <= 2, x
    assert pm.erfc(float("inf")) == 0.0 and pm.erfc(float("-inf")) == 2.0


def test_normal_inv_cdf_is_as241():
    # statistics' pure-Python AS241, run on the same pinned log and sqrt
    import statistics
    src = Path(statistics.__file__).read_text()
    i = src.index("def _normal_dist_inv_cdf")
    j = src.index("return mu + (x * sigma)", src.index("r = p if q <= 0.0", i))
    ns = {"fabs": abs, "sqrt": pm.sqrt, "log": pm.log}
    exec(src[i:j] + "return mu + (x * sigma)\n", ns)
    rng = random.Random(3)
    for _ in range(3000):
        p = rng.choice([rng.random(), rng.random() * 1e-10, 1 - rng.random() * 1e-10])
        if 0.0 < p < 1.0:
            assert pm.normal_inv_cdf(p) == ns["_normal_dist_inv_cdf"](p, 0.0, 1.0), p


def test_exp_array_is_the_scalar_exp():
    xs = [-800.0, -700.0, -1.5, -0.0, 0.0, 1e-300, 0.5, 3.0, 700.0, float("-inf"), float("nan")]
    out = pm.exp_array(xs)
    for x, y in zip(xs, out):
        e = pm.exp(x)
        assert (y != y and e != e) or struct.pack("<d", y) == struct.pack("<d", e), x
    with pytest.raises(OverflowError):
        pm.exp_array([1.0, 710.0])


# The modules whose float math the lane pinned: a `**` there must be an
# integer power (exact: an int base), an exact constant power of two, or one
# of the integer expressions below, never the platform pow.
_POW_OK = {"max(pp, 1) ** 2", "max(ka, kb) ** 2", "b_n ** 2",
           "x ** n", "num ** n"}  # the last two: powi's exact special cases and exact integer power


def _exact_pow(node):
    left, right = node.left, node.right
    if isinstance(left, ast.Constant) and type(left.value) is int:
        return True
    return (isinstance(left, ast.Constant) and left.value == 2.0 and isinstance(right, (ast.Constant, ast.UnaryOp)))


_POW_MODULES = ["_expansion_metrics.py", "model_selection.py", "_expansion_cluster.py", "_expansion_linear.py",
                "_x_sequence_autoarima.py", "_expansion_cnn.py", "_expansion_prep.py", "_expansion_decomp.py",
                "_portable_math.py", "_x_sequence_rnn.py"]


@pytest.mark.parametrize("name", _POW_MODULES)
def test_no_platform_pow_in_pinned_modules(name):
    src = (PKG / name).read_text()
    bad = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            seg = ast.get_source_segment(src, node)
            if not _exact_pow(node) and seg not in _POW_OK:
                bad.append(f"{name}:{node.lineno}: {seg}")
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id in ("math", "_math", "_m")
                and node.func.attr in ("exp", "log", "log2", "log10", "log1p", "expm1", "pow", "erf", "erfc")):
            imported = f"import {node.func.value.id}" in src or f"import math as {node.func.value.id}" in src
            if imported and "_portable_math as " + node.func.value.id not in src:
                bad.append(f"{name}:{node.lineno}: stdlib math.{node.func.attr}")
    assert not bad, bad


# ------------------------------------------------ 2, 3. search folds, curves

class _RowSpy:
    """Scores a fold by the rows it trained on."""
    _estimator_type = "regressor"

    def __init__(self, c=0):
        self.c = c

    def get_params(self, deep=False):
        return {"c": self.c}

    def set_params(self, **p):
        self.c = p.get("c", self.c)
        return self

    def fit(self, X, y=None):
        rows = X.tolist() if hasattr(X, "tolist") else X
        self.key_ = float(sum(r[0] for r in rows))
        _SEEN.append([r[0] for r in rows])
        return self

    def score(self, X, y=None):
        return self.key_


_SEEN = []


def _xy(n=90):
    rng = random.Random(5)
    X = [[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(n)]
    return X, [r[0] + r[1] for r in X]


def test_search_draws_its_folds_once():
    from mojolearn import model_selection as ms
    X, y = _xy()
    gs = ms.GridSearchCV(_RowSpy(), {"c": [0, 1, 2, 3, 4]}, cv=ms.KFold(3, shuffle=True), refit=False).fit(X, y)
    for i in range(3):
        assert len(set(gs.cv_results_[f"split{i}_test_score"].tolist())) == 1


def test_validation_curve_draws_its_folds_once():
    from mojolearn import model_selection as ms
    X, y = _xy()
    tr, te = ms.validation_curve(_RowSpy(), X, y, param_name="c", param_range=[0, 1, 2],
                                 cv=ms.KFold(3, shuffle=True))
    for col in zip(*tr.tolist()):
        assert len(set(col)) == 1


def test_learning_curve_takes_nested_prefixes_of_one_order():
    from mojolearn import model_selection as ms
    X, y = _xy()
    _SEEN.clear()
    ms.learning_curve(_RowSpy(), X, y, cv=3, train_sizes=[0.25, 0.5, 1.0], shuffle=True, random_state=0)
    k = 3
    for s in range(3):
        for f in range(k):
            assert _SEEN[s * k + f][:len(_SEEN[f])] == _SEEN[f]


# ------------------------------------------------------------- 4. the RNN

def test_rnn_order_is_int32_and_uncapped_at_2_24():
    np = pytest.importorskip("numpy")
    from mojolearn._x_sequence_rnn import RNNRegressor
    m = RNNRegressor(batch_size=3, max_epochs=2, shuffle=True)
    order, steps = m._schedule(7, np.random.default_rng(0))
    assert order.dtype == np.int32 and steps.dtype == np.int32
    assert steps.tolist() == [0, 3, 3, 3, 6, 1, 7, 3, 10, 3, 13, 1]
    assert sorted(order[:7].tolist()) == list(range(7)) and sorted(order[7:].tolist()) == list(range(7))
    big = RNNRegressor(batch_size=1_000_000, max_epochs=17, shuffle=False)
    order, steps = big._schedule(1_000_000, np.random.default_rng(0))
    assert len(order) == 17_000_000 > 2 ** 24 and int(order[-1]) == 999_999
    with pytest.raises(ValueError):
        RNNRegressor(max_epochs=2200)._schedule(1_000_000, np.random.default_rng(0))
