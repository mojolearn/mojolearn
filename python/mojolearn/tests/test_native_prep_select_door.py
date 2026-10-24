# SPDX-License-Identifier: Apache-2.0
"""RFE and the mutual-information functions are one native call each
(bindings/prep_select_native.mojo, through the family door). The former
Python routines are kept here as the oracle: the native door must return the
same bits and set the same attributes.

The structural checks run without bindings; the oracle checks need the built
x_prep and runtime-classical bindings."""
import array
import ast
import numbers
import operator

import numpy as np
import pytest

import os

_ORACLE_NAMES = ("Array", "_Prog", "_Scratch", "_cs", "_codes", "_class_counts", "_fit_categories",
                 "_label_buffer", "_mode", "_x2d", "as_f32_c", "encode_labels", "full", "_MI_WORK")


def _load():
    """mojolearn and the names the oracle below uses (needs the bindings)."""
    import mojolearn as ml
    from mojolearn import _expansion_prep as ep
    g = globals()
    for name in _ORACLE_NAMES:  # the oracle reads these module globals
        g[name] = getattr(ep, name)
    return ml, ep


# ---------------------------------------------------------------- the former Python routines (oracle)
def _key_words(values, d):
    """(int32 words, c2_key64 KIND) of d values: a float32 / float64 / int32 /
    int64 buffer as its own words; anything else (a user score function's
    list) converted once to float64 (the explicit input-prep step)."""
    lb = _label_buffer(values)
    if lb is not None and lb.kind in (0, 1, 3, 4) and lb.n == d:
        return lb.words, lb.kind
    vals = values.tolist() if hasattr(values, "tolist") else values
    flat = array.array("d", [float(v) for v in vals])  # glue: a user callable's scores, converted once
    if len(flat) != d:
        raise ValueError(f"mojolearn: expected {d} scores, got {len(flat)}")
    w = array.array("i")
    w.frombytes(flat.tobytes())
    return Array._owned(w, (2 * d,), "<i4", "C"), 4


def _mask_list(pr, off, d):
    """A device 0.0 / 1.0 mask read back as the fitted bool list."""
    return [v != 0.0 for v in pr.values(off, d)]  # glue: the device's mask as the fitted list




def _mi_discrete_mask(discrete_features, d):
    """The reference's discrete_features: 'auto' (dense X: none), a bool for
    every column, a bool mask of length d, or column indices (negative ones
    count from the end)."""
    if isinstance(discrete_features, str):
        if discrete_features != "auto":
            raise ValueError("mojolearn: Invalid string value for discrete_features.")
        return [False] * d
    if isinstance(discrete_features, bool) or type(discrete_features).__name__ == "bool_":
        return [bool(discrete_features)] * d
    vals = discrete_features.tolist() if hasattr(discrete_features, "tolist") else list(discrete_features)
    if isinstance(vals, bool):
        return [vals] * d
    if vals and all(isinstance(v, bool) for v in vals):  # glue: validates the discrete_features argument (vals-sized: user discrete_features argument)
        if len(vals) != d:
            raise ValueError(f"mojolearn: discrete_features mask has {len(vals)} entries; X has {d} features")
        return list(vals)
    mask = [False] * d
    for v in vals:  # glue: converts the discrete_features argument (vals-sized: user discrete_features argument)
        if isinstance(v, bool) or not isinstance(v, numbers.Integral):
            raise ValueError("mojolearn: discrete_features must be 'auto', a bool, a bool mask or indices")
        j = int(v)
        if not -d <= j < d:
            raise IndexError(f"mojolearn: discrete_features index {j} is out of bounds for {d} features")
        mask[j % d] = True
    return mask


#: lane/apple-fast-mi (2026-10-03): the A/B switch MOJOLEARN_MI_WORK=1 (opt-in,
#: FAST only, read once at import): `_mutual_info` keeps the noised columns, their
#: noise words and the per-point terms (3 n d words) as device-only scratch
#: (`_Prog.work`) instead of arena words that the host zeroes and the device
#: copies back (265 MB on Istella). Where a word lives moves no bit.

def _plus1(off):
    """off + 1 for an arena offset or a `_Scratch` (the units' offset + 1 noise words)."""
    return _Scratch(off.off + 1, off.kind) if isinstance(off, _Scratch) else off + 1


def _mutual_info(X, y, discrete_target, discrete_features, n_neighbors, random_state):
    """The reference's `_estimate_mi`: continuous columns are scaled and
    noised (row-major over the continuous columns only, as the reference's
    draw of shape (n, n_continuous)), the 1e-10-scaled noise kept as a
    second word that breaks exact ties as the reference's float64 sum does
    (DEVIATION 5407); each column then takes the estimator
    its kinds name: Kraskov (continuous x, continuous y), Ross (one side
    discrete: the classes, or a discrete feature's categories against the
    noised target) or the contingency table (both discrete)."""
    k = int(n_neighbors)
    if not 1 <= k <= 32:
        raise NotImplementedError("mojolearn: mutual_info supports 1 <= n_neighbors <= 32")
    arr = _x2d(X)
    n, d = arr.shape
    mask = _mi_discrete_mask(discrete_features, d)
    cont = [j for j in range(d) if not mask[j]]  # glue: continuous column index list (d-sized: feature count)
    disc = [j for j in range(d) if mask[j]]  # glue: discrete column index list (d-sized: feature count)
    mode = _mode()
    seed = 0 if random_state is None else int(random_state) & 0x3FFFFFFF
    pr = _Prog()
    work = pr.work if (_MI_WORK and mode == "fast") else pr.alloc
    if discrete_target:
        classes, codes = encode_labels(y)
        if codes.size != n:
            raise ValueError("mojolearn: X and y have different numbers of rows")
        counts = _class_counts(codes, len(classes))
        if cont and max(counts) < 2:  # glue: raises when every class has one sample (counts-sized: class counts from the binding)
            raise ValueError("mojolearn: mutual_info: every class has one sample (the reference's "
                             "neighbour search over the classes with more than one finds 0 samples)")
        yo, lc = pr.put_codes(codes), pr.put_list(counts)
    else:
        yv = as_f32_c(y, ndim=1, name="y")[0]
        if yv.size != n:
            raise ValueError("mojolearn: X and y have different numbers of rows")
    outc = outd = None
    if cont:
        dc = len(cont)
        xo = pr.put(arr if not disc else _gather(arr, cont, mode))
        st, sc, ma, z, zs = pr.alloc(6 * dc), pr.alloc(dc), pr.alloc(dc), work(n * dc), work(n * dc)
        _cs(pr, mode, xo, n, dc, st)
        pr.stage("mi_colscale", dc, xo, n, dc, st, sc, ma)
        pr.stage("mi_noise", n * dc, xo, n, dc, sc, ma, 2 * seed, z, _plus1(zs))
    if not discrete_target:
        yo = pr.put(yv)
        sty, scy, may, zy, zys = pr.alloc(6), pr.alloc(1), pr.alloc(1), work(n), work(n)
        _cs(pr, mode, yo, n, 1, sty)
        pr.stage("mi_colscale", 1, yo, n, 1, sty, scy, may)
        pr.stage("mi_noise", n, yo, n, 1, scy, may, 2 * seed + 1, zy, _plus1(zys))
    if cont:
        term, outc = work(n * dc), pr.alloc(dc)
        if discrete_target:
            pr.stage("mi_cd", n * dc, z, n, dc, yo, lc, k, term, _plus1(zs))
            # the rows of the classes with more than one (KIND 3: summed on the device)
            pr.stage("mi_reduce", dc, term, n, dc, 3, k, 0, outc, lc, len(counts))
        else:
            pr.stage("mi_cc", n * dc, z, n, dc, zy, k, term, _plus1(zs), _plus1(zys))
            pr.stage("mi_reduce", dc, term, n, dc, 0, k, n, outc)
    if disc:
        dd = len(disc)
        xd = _gather(arr, disc, mode)
        cats = _fit_categories(mode, xd)
        kx = [c.size for c in cats]  # glue: per-column category counts (cats-sized: per-column category arrays)
        kmax = max(kx)  # glue: largest category count for the program layout (kx-sized: per-column category counts)
        if not discrete_target and n in kx:
            raise ValueError(f"mojolearn: mutual_info: discrete feature {disc[kx.index(n)]} has one sample per "
                             "value (the reference's neighbour search finds 0 samples)")
        xc, _neg = _codes(pr, xd, cats)
        outd = pr.alloc(dd)
        if discrete_target:
            ky = len(classes)
            stride = kmax * ky + kmax + ky
            tb = pr.alloc(dd * stride)
            pr.stage("mi_dd", dd, xc, n, dd, yo, ky, pr.put_list(kx), tb, stride, outd)
        else:
            cnti, cntf, term = pr.alloc(dd * kmax), pr.alloc(dd * kmax), work(n * dd)
            pr.stage("code_counts", dd, xc, n, dd, kmax, cnti)
            pr.stage("i2f", dd * kmax, cnti, cntf)
            pr.stage("mi_dc", n * dd, zy, n, dd, xc, cntf, kmax, k, term, _plus1(zys))
            pr.stage("mi_reduce", dd, term, n, dd, 2, k, 0, outd, cntf, kmax)
    pr.run(mode)
    if not disc:
        return pr.get(outc, d)
    vals = [0.0] * d
    for j, v in zip(cont, pr.values(outc, len(cont)) if cont else []):  # glue: scatters device MI values to their columns (cont-sized: continuous column indices)
        vals[j] = v
    for j, v in zip(disc, pr.values(outd, len(disc))):  # glue: scatters device MI values to their columns (disc-sized: discrete column indices)
        vals[j] = v
    return Array.from_list(vals, "<f4")


def _gather(arr, cols, mode):
    n, d = arr.shape
    pr = _Prog()
    xo, ko = pr.put(arr), pr.put_list(cols)
    out = pr.alloc(n * len(cols))
    pr.stage("gather_cols", n * len(cols), xo, n, d, ko, len(cols), out)
    pr.run(mode)
    return pr.get(out, (n, len(cols)))


def _stage_importance_keys(pr, est, m, getter="auto"):
    """Stages the ordered keys (c2_key64) of a fitted estimator's m column
    importances: coef_ squared (summed over rows when 2-D, sqsum_cols) on the
    device, else feature_importances_ as given (a monotone stand-in for its
    square). A str getter (a dotted attribute path, as operator.attrgetter)
    or a callable picks the importances instead; they are squared (summed
    over rows when 2-D) on the device, as the reference's
    transform_func='square'. Returns the keys' offset."""
    if getter != "auto":
        coef = operator.attrgetter(getter)(est) if isinstance(getter, str) else getter(est)
    else:
        coef = getattr(est, "coef_", None)
    key = pr.work(2 * m)
    if coef is None and getter == "auto":
        imp = getattr(est, "feature_importances_", None)
        if imp is None:
            raise ValueError("mojolearn: RFE needs an estimator with coef_ or feature_importances_")
        words, kind = _key_words(imp, m)
        pr.stage("c2_key64", m, pr.put_words(words), kind, key)
        return key
    c = as_f32_c(coef, ndim=None, name="coef_")[0]
    rows, d = (1, c.shape[0]) if c.ndim == 1 else c.shape
    if d != m:
        raise ValueError(f"mojolearn: RFE importances have {d} columns, expected {m}")
    out = pr.work(d)
    pr.stage("sqsum_cols", d, pr.put(c), rows, d, out)
    pr.stage("c2_key64", d, out, 0, key)
    return key




def _former_rfe_fit(self, X, y, **fit_params):
    g = self.importance_getter
    if not (callable(g) or isinstance(g, str)):
        raise ValueError("mojolearn: RFE importance_getter must be 'auto', a str or a callable")
    arr = _x2d(X)
    n, d = arr.shape
    mode = _mode()
    nsel = self.n_features_to_select
    if nsel is None:
        nsel = d // 2
    elif isinstance(nsel, float) and 0 < nsel < 1:
        nsel = int(nsel * d)
    nsel = max(1, int(nsel))
    step = int(max(1, self.step * d)) if isinstance(self.step, float) and self.step < 1 else int(self.step)
    if step <= 0:
        raise ValueError("mojolearn: step must be > 0")
    support = [True] * d
    ranking = full((d,), 1, "<f4")
    nsup = d
    while nsup > nsel:
        features = [j for j in range(d) if support[j]]  # glue: the supported column list (control)
        est = _former_clone(self).fit(_gather(arr, features, mode), y, **fit_params)
        # lane cpu2-l3-prep: the ranking and elimination on the device (c2_rfe_step: the
        # `step` weakest by a stable ascending sort of the importance keys leave the
        # support; c2_rfe_rank: every unsupported column's ranking + 1)
        pr = _Prog()
        key = _stage_importance_keys(pr, est, len(features), self.importance_getter)
        sup = pr.put_list([1.0 if v else 0.0 for v in support], inout=True)  # glue: the support mask up
        rk = pr.put(ranking, inout=True)
        pr.stage("c2_rfe_step", len(features), key, pr.put_ints(features), len(features),
                 min(step, nsup - nsel), sup)
        pr.stage("c2_rfe_rank", d, sup, rk)
        pr.run(mode)
        support = _mask_list(pr, sup, d)
        ranking = pr.get(rk, d)
        nsup = len([1 for v in support if v])  # glue: the support size (control)
    features = [j for j in range(d) if support[j]]  # glue: support index list (d-sized: feature count)
    self.estimator_ = _former_clone(self).fit(_gather(arr, features, mode), y, **fit_params)
    self._mask, self.support_ = support, list(support)
    self.ranking_ = ranking.astype("<i8")
    self.n_features_ = sum(support)  # glue: count of selected features (support-sized: feature support mask)
    self.numeric_mode_, self.n_features_in_ = mode, d
    return self



def _former_clone(self):
    est = self.estimator
    return type(est)(**est.get_params()) if hasattr(est, "get_params") else est


# ---------------------------------------------------------------- structure (no bindings)
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_expansion_prep.py")


def _tree():
    with open(_SRC) as fh:
        return ast.parse(fh.read())


def _defs(tree):
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            out[node.name] = node
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    out[node.name + "." + item.name] = item
    return out


@pytest.mark.parametrize("name", ["RFE.fit", "RFE.fit_transform", "RFE.predict", "RFE.predict_proba",
                                  "RFE.decision_function", "RFE.score", "mutual_info_classif",
                                  "mutual_info_regression"])
def test_pure_one_native_call(name):
    node = _defs(_tree())[name]
    body = [s for s in node.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    assert len(body) == 1 and isinstance(body[0], ast.Return)
    call = body[0].value
    assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "_ENCODE_CALL"


def test_pure_former_helpers_gone():
    defs = _defs(_tree())
    for name in ("_mutual_info", "_mi_discrete_mask", "_plus1", "_gather", "_stage_importance_keys",
                 "_key_words", "_mask_list", "RFE._clone"):
        assert name not in defs, name


# ---------------------------------------------------------------- oracle (bindings; RUN OWED)
def _data(n=300, d=8, seed=0):
    _load()
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    X = (rng.standard_normal((n, d)) + y[:, None] * np.linspace(0, 1.2, d)).astype(np.float32)
    yr = (X[:, 0] * 2 + rng.standard_normal(n)).astype(np.float32)
    return X, y, yr


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype and a.shape == b.shape
    assert a.tobytes() == b.tobytes()


@pytest.mark.parametrize("kw", [dict(n_features_to_select=3, step=2), dict(n_features_to_select=0.4, step=0.25),
                                dict(), dict(importance_getter="coef_", n_features_to_select=2)])
def test_rfe_matches_former(kw):
    X, y, _ = _data()
    ml, _ = _load()
    new = ml.RFE(ml.LinearDiscriminantAnalysis(), **kw).fit(X, y)
    old = _former_rfe_fit(ml.RFE(ml.LinearDiscriminantAnalysis(), **kw), X, y)
    assert new.support_ == old.support_ and new._mask == old._mask
    _same(new.ranking_, old.ranking_)
    assert new.n_features_ == old.n_features_ and new.n_features_in_ == old.n_features_in_
    assert new.numeric_mode_ == old.numeric_mode_
    _same(new.predict(X), old.estimator_.predict(old.transform(X)))
    _same(new.predict_proba(X), old.estimator_.predict_proba(old.transform(X)))
    assert new.score(X, y) == old.estimator_.score(old.transform(X), y)
    _same(ml.RFE(ml.LinearDiscriminantAnalysis(), **kw).fit_transform(X, y), old.transform(X))


def test_rfe_coef_estimator_matches_former():
    X, _, yr = _data()
    ml, _ = _load()
    new = ml.RFE(ml.Ridge(), n_features_to_select=3).fit(X, yr)
    old = _former_rfe_fit(ml.RFE(ml.Ridge(), n_features_to_select=3), X, yr)
    assert new.support_ == old.support_
    _same(new.ranking_, old.ranking_)


def test_rfe_refusals():
    X, y, _ = _data()
    ml, _ = _load()
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SkLDA
    with pytest.raises(TypeError, match="RFE requires a mojolearn estimator; got LinearDiscriminantAnalysis"):
        ml.RFE(SkLDA()).fit(X, y)
    with pytest.raises(TypeError, match="a callable is not supported"):
        ml.RFE(ml.LinearDiscriminantAnalysis(), importance_getter=lambda e: e.coef_).fit(X, y)
    with pytest.raises(ValueError, match="importance_getter must be"):
        ml.RFE(ml.LinearDiscriminantAnalysis(), importance_getter=5).fit(X, y)
    with pytest.raises(ValueError, match="step must be > 0"):
        ml.RFE(ml.LinearDiscriminantAnalysis(), step=0).fit(X, y)
    with pytest.raises(AttributeError):
        ml.RFE(ml.LinearDiscriminantAnalysis()).predict(X)


@pytest.mark.parametrize("df", ["auto", True, [0, 3], [-1], [True, False] * 4])
def test_mutual_info_matches_former(df):
    X, y, yr = _data()
    ml, _ = _load()
    Xd = X.copy()
    Xd[:, 0] = np.floor(Xd[:, 0] * 2)
    Xd[:, 3] = np.floor(Xd[:, 3])
    Xd[:, 7] = np.floor(Xd[:, 7])
    _same(ml.mutual_info_classif(Xd, y, discrete_features=df, random_state=3),
          _mutual_info(Xd, y, True, df, 3, 3))
    _same(ml.mutual_info_regression(Xd, yr, discrete_features=df, n_neighbors=4, random_state=None),
          _mutual_info(Xd, yr, False, df, 4, None))


def test_mutual_info_refusals():
    X, y, _ = _data()
    ml, _ = _load()
    with pytest.raises(NotImplementedError, match="1 <= n_neighbors <= 32"):
        ml.mutual_info_classif(X, y, n_neighbors=33)
    with pytest.raises(ValueError, match="Invalid string value"):
        ml.mutual_info_classif(X, y, discrete_features="some")
    with pytest.raises(ValueError, match="mask has 3 entries"):
        ml.mutual_info_classif(X, y, discrete_features=[True, False, True])
    with pytest.raises(IndexError, match="out of bounds for 8 features"):
        ml.mutual_info_classif(X, y, discrete_features=[8])
    with pytest.raises(ValueError, match="a bool mask or indices"):
        ml.mutual_info_classif(X, y, discrete_features=[1.5])
    with pytest.raises(ValueError, match="different numbers of rows"):
        ml.mutual_info_regression(X, y[:10].astype(np.float32))
