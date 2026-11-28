# SPDX-License-Identifier: Apache-2.0
"""mojolearn.rank_genes_groups at the Python API boundary (lane sc-stats-1, 2026-11-28): the input forms (dense,
CSR buffers on a plain object; no SciPy anywhere), the refusals, the result layout, and the statistics against an independent float64 NumPy oracle of
Scanpy's formulas (Wilcoxon with average ranks and optional tie correction, Welch's t-test, Benjamini-Hochberg,
the log fold change). The kernels' exact checks are native (checks/native_tests/x_prep_markers_check.mojo)."""
import math

import numpy as np
import pytest

import mojolearn


def _a(res, key):
    """A result block back in the input's gene order (the result rows are sorted by score, `names` beside)."""
    names = np.asarray(res["names"])
    vals = np.asarray(res[key])
    out = np.empty_like(vals)
    for j in range(vals.shape[0]):
        out[j, names[j]] = vals[j]
    return out


def _data(n=240, d=7, k=3, seed=0):
    rng = np.random.default_rng(seed)
    g = np.arange(n) % k
    lam = 0.5 + 0.8 * (g[:, None] == (np.arange(d)[None, :] % k))
    x = np.log1p(rng.poisson(lam * 2.0)).astype(np.float32)
    return x, g


def _avg_ranks(v):
    order = np.argsort(v, kind="stable")
    ranks = np.empty(len(v), np.float64)
    sv = v[order]
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _normal_two_sided(z):
    return math.erfc(abs(z) / math.sqrt(2.0))


def _oracle_wilcoxon(x, g, group, ref=None, tie=False):
    pool = np.ones(len(g), bool) if ref is None else (g == group) | (g == ref)
    xs = x[pool].astype(np.float64)
    gs = g[pool]
    ng = int((gs == group).sum())
    nr = len(gs) - ng
    big = ng + nr
    z = np.zeros(x.shape[1])
    p = np.ones(x.shape[1])
    for c in range(x.shape[1]):
        r = _avg_ranks(xs[:, c])
        rs = r[gs == group].sum()
        var = ng * nr * (big + 1) / 12.0
        if tie:
            _, counts = np.unique(xs[:, c], return_counts=True)
            var *= 1.0 - (counts ** 3 - counts).sum() / (big ** 3 - big)
        if var > 0:
            z[c] = (rs - ng * (big + 1) / 2.0) / math.sqrt(var)
            p[c] = _normal_two_sided(z[c])
    return z, p


def _bh(p):
    m = len(p)
    order = np.argsort(p, kind="stable")
    q = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out


def test_result_layout_and_groups():
    x, g = _data()
    res = mojolearn.rank_genes_groups(x, g)
    assert set(res) == {"groups", "names", "scores", "pvals", "pvals_adj", "logfoldchanges"}
    names = np.asarray(res["names"])
    assert names.dtype == np.int32 and names.shape == (3, 7)
    scores = np.asarray(res["scores"])
    for j in range(3):
        assert sorted(names[j].tolist()) == list(range(7))
        # descending, a tie by the lower gene index (Scanpy's stable sort)
        expected = np.lexsort((np.arange(7), -_a(res, "scores")[j]))
        assert names[j].tolist() == expected.tolist()
        assert np.all(np.diff(scores[j]) <= 0)
    assert list(res["groups"]) == [0, 1, 2]
    for key in ("scores", "pvals", "pvals_adj", "logfoldchanges"):
        assert np.asarray(res[key]).shape == (3, 7)
        assert np.asarray(res[key]).dtype == np.float32
    ref = mojolearn.rank_genes_groups(x, g, reference=1)
    assert list(ref["groups"]) == [0, 2]
    assert np.asarray(ref["scores"]).shape == (2, 7)


def test_string_labels_are_sorted_groups():
    x, g = _data()
    names = np.array(["b", "c", "a"])[g]
    res = mojolearn.rank_genes_groups(x, names)
    assert list(res["groups"]) == ["a", "b", "c"]


@pytest.mark.parametrize("tie", [False, True])
def test_wilcoxon_rest_matches_oracle(tie):
    x, g = _data()
    res = mojolearn.rank_genes_groups(x, g, tie_correct=tie)
    for j in range(3):
        z, p = _oracle_wilcoxon(x, g, j, tie=tie)
        np.testing.assert_allclose(_a(res, "scores")[j], z, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(_a(res, "pvals")[j], p, rtol=1e-3, atol=1e-6)
        np.testing.assert_allclose(_a(res, "pvals_adj")[j], _bh(_a(res, "pvals")[j].astype(np.float64)),
                                   rtol=1e-5, atol=1e-7)


def test_wilcoxon_reference_matches_oracle():
    x, g = _data()
    res = mojolearn.rank_genes_groups(x, g, reference=2)
    for row, group in enumerate((0, 1)):
        z, _ = _oracle_wilcoxon(x, g, group, ref=2)
        np.testing.assert_allclose(_a(res, "scores")[row], z, rtol=2e-5, atol=2e-5)


def _betainc(a, b, x):
    """The regularized incomplete beta I_x(a, b) in float64 (Lentz's continued fraction), the test's own oracle."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _betainc(b, a, 1.0 - x)
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x)
    tiny = 1e-300
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        for num in (m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
                    -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))):
            d = 1.0 + num * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + num / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < 1e-15:
            break
    return math.exp(lbt) * h / a


def test_t_test_matches_welch_oracle():
    x, g = _data()
    res = mojolearn.rank_genes_groups(x, g, method="t-test", corr_method="bonferroni")
    for j in range(3):
        a = x[g == j].astype(np.float64)
        b = x[g != j].astype(np.float64)
        va, vb = a.var(0, ddof=1) / len(a), b.var(0, ddof=1) / len(b)
        t = (a.mean(0) - b.mean(0)) / np.sqrt(va + vb)
        dof = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
        p = np.array([_betainc(df / 2.0, 0.5, df / (df + tt * tt)) for tt, df in zip(t, dof)])
        np.testing.assert_allclose(_a(res, "scores")[j], t, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(_a(res, "pvals")[j], p, rtol=2e-3, atol=1e-6)
        np.testing.assert_allclose(_a(res, "pvals_adj")[j], np.minimum(_a(res, "pvals")[j] * 7, 1), rtol=1e-6)


def test_log_fold_change_is_scanpys():
    x, g = _data()
    res = mojolearn.rank_genes_groups(x, g)
    for j in range(3):
        mg = x[g == j].astype(np.float64).mean(0)
        mr = x[g != j].astype(np.float64).mean(0)
        lfc = np.log2((np.expm1(mg) + 1e-9) / (np.expm1(mr) + 1e-9))
        np.testing.assert_allclose(_a(res, "logfoldchanges")[j], lfc, rtol=1e-4, atol=1e-4)


class _Csr:
    """CSR buffers as plain attributes (what the door reads), built with NumPy only."""

    def __init__(self, x):
        nz = x != 0
        self.indptr = np.concatenate([[0], np.cumsum(nz.sum(axis=1))]).astype(np.int32)
        self.indices = np.nonzero(nz)[1].astype(np.int32)
        self.data = x[nz].astype(np.float32)
        self.shape = x.shape
        self.format = "csr"


def test_csr_equals_dense():
    x, g = _data()
    x[x < 0.8] = 0
    dense = mojolearn.rank_genes_groups(x, g)
    csr = mojolearn.rank_genes_groups(_Csr(x), g)
    for key in ("names", "scores", "pvals", "pvals_adj", "logfoldchanges"):
        assert np.asarray(dense[key]).tobytes() == np.asarray(csr[key]).tobytes()


def test_other_sparse_format_refused():
    x, g = _data()
    other = _Csr(x)
    other.format = "csc"
    with pytest.raises(ValueError, match="CSR"):
        mojolearn.rank_genes_groups(other, g)


def test_refusals():
    x, g = _data()
    with pytest.raises(ValueError):
        mojolearn.rank_genes_groups(x, g, method="logreg")
    with pytest.raises(ValueError):
        mojolearn.rank_genes_groups(x, g, corr_method="holm")
    with pytest.raises(ValueError):
        mojolearn.rank_genes_groups(x, g, reference=7)
    with pytest.raises(ValueError):
        mojolearn.rank_genes_groups(x, g[:-1])
    with pytest.raises(ValueError):
        mojolearn.rank_genes_groups(x, np.zeros(len(g), int))
    lone = g.copy()
    lone[0] = 9
    with pytest.raises(ValueError, match="fewer than two cells"):
        mojolearn.rank_genes_groups(x, lone)

