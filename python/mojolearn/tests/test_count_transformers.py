# SPDX-License-Identifier: Apache-2.0
"""The public boundary of the count-data transformers (lane sc-prep-1, 2026-11-28): TotalCountNormalizer,
Log1pTransformer, ClippedStandardScaler, HighlyVariableGenes, RegressOut. Inputs, refusals, returned shapes and
types, the CSR form kept. Expected values are computed by hand (or by NumPy on float64 as an independent oracle);
no SciPy anywhere: the sparse input is a plain object exposing the CSR form. The numerical cases live in
checks/native_tests/prep_count_data.mojo. NOT RUN by the lane (lanes do not run tests)."""
import numpy as np
import pytest

import mojolearn as ml


class _Csr:
    """The CSR form of a dense block (format, shape, indptr, indices, data): what the transformers read."""
    format = "csr"

    def __init__(self, arg, shape=None):
        if isinstance(arg, tuple):
            self.data, self.indices, self.indptr = (np.asarray(a) for a in arg)
            self.shape = tuple(shape)
            return
        A = np.asarray(arg, dtype=np.float32)
        nz = A != 0
        self.indptr = np.concatenate([[0], np.cumsum(nz.sum(axis=1))]).astype(np.int32)
        self.indices = np.nonzero(nz)[1].astype(np.int32)
        self.data = A[nz]
        self.shape = A.shape

    def dense(self):
        out = np.zeros(self.shape, np.float32)
        for r in range(self.shape[0]):
            for e in range(self.indptr[r], self.indptr[r + 1]):
                out[r, self.indices[e]] += self.data[e]
        return out


X = np.array([[1, 3, 0], [2, 2, 4]], dtype=np.float32)


def test_total_count_target_and_median():
    np.testing.assert_allclose(np.asarray(ml.TotalCountNormalizer(target_sum=10).fit_transform(X)),
                               [[2.5, 7.5, 0], [2.5, 2.5, 5]], rtol=1e-6)
    m = ml.TotalCountNormalizer().fit(X)
    assert m.target_sum_ == pytest.approx(6.0)
    assert m.n_features_in_ == 3


def test_total_count_sparse_keeps_structure():
    S = _Csr(X)
    R = ml.TotalCountNormalizer(target_sum=10).fit_transform(S)
    assert type(R) is _Csr and R.shape == (2, 3)
    np.testing.assert_array_equal(np.asarray(R.indices), S.indices)
    np.testing.assert_allclose(R.dense(), [[2.5, 7.5, 0], [2.5, 2.5, 5]], rtol=1e-6)


def test_total_count_rule_equals_explicit_columns():
    D = np.array([[1, 1, 8], [1, 1, 0]], dtype=np.float32)
    rule = ml.TotalCountNormalizer(target_sum=4, exclude_highly_expressed=True, max_fraction=0.5).fit(D)
    np.testing.assert_array_equal(np.asarray(rule.excluded_), [0, 0, 1])
    out = np.asarray(rule.transform(D))
    np.testing.assert_allclose(out, [[2, 2, 16], [2, 2, 0]], rtol=1e-6)
    for columns in ([0, 1], np.array([True, True, False])):
        explicit = np.asarray(ml.TotalCountNormalizer(target_sum=4, columns=columns).fit_transform(D))
        assert explicit.tobytes() == out.tobytes()


@pytest.mark.parametrize("kwargs, message", [
    (dict(target_sum=0), "target_sum"), (dict(exclude_highly_expressed=True, max_fraction=2), "max_fraction"),
    (dict(columns=[5]), "out of range"), (dict(columns=np.array([True, False])), "one entry per feature")])
def test_total_count_refusals(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ml.TotalCountNormalizer(**kwargs).fit(X)


def test_log1p_dense_sparse_base():
    v = np.array([[0, 1, 3]], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(ml.Log1pTransformer().fit_transform(v)), np.log1p(v.astype(np.float64)),
                               rtol=1e-6)
    np.testing.assert_allclose(ml.Log1pTransformer(base=2).fit_transform(_Csr(v)).dense(), [[0, 1, 2]], rtol=1e-6)
    with pytest.raises(ValueError, match="base"):
        ml.Log1pTransformer(base=1).fit(v)


def test_clipped_scaler_matches_scanpy_definition():
    A = np.array([[1], [2], [3], [4]], dtype=np.float32)
    m = ml.ClippedStandardScaler(max_value=1.0).fit(A)
    np.testing.assert_allclose(np.asarray(m.var_), [5 / 3], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(m.std_), [np.sqrt(5 / 3)], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(m.transform(A)).ravel(), [-1, -0.3872983, 0.3872983, 1], rtol=1e-5)
    S = _Csr(X)
    u = ml.ClippedStandardScaler(zero_center=False).fit(S)
    assert type(u.transform(S)) is _Csr
    with pytest.raises(ValueError, match="zero_center"):
        ml.ClippedStandardScaler().fit(S).transform(S)


@pytest.mark.parametrize("flavor", ["seurat", "cell_ranger", "seurat_v3"])
def test_hvg_selects_n_top(flavor):
    rng = np.random.default_rng(0)
    C = rng.poisson(rng.gamma(1.0, 1.0, 60), (300, 60)).astype(np.float32)
    data = C if flavor == "seurat_v3" else np.log1p(C)
    m = ml.HighlyVariableGenes(flavor=flavor, n_top_genes=10).fit(data)
    hv = np.asarray(m.highly_variable_)
    assert hv.shape == (60,) and m.n_highly_variable_ == int(hv.sum()) >= 10
    assert np.asarray(m.transform(data)).shape == (300, m.n_highly_variable_)
    batches = np.arange(300) % 2
    b = ml.HighlyVariableGenes(flavor=flavor, n_top_genes=10).fit(_Csr(data), batches)
    assert type(b.transform(_Csr(data))) is _Csr
    assert np.asarray(b.highly_variable_nbatches_).max() <= 2


def test_hvg_refusals():
    with pytest.raises(ValueError, match="n_top_genes"):
        ml.HighlyVariableGenes(flavor="seurat_v3").fit(X)
    with pytest.raises(ValueError, match="flavor"):
        ml.HighlyVariableGenes(flavor="pearson").fit(X)


def test_regress_out_residuals():
    rng = np.random.default_rng(1)
    cov = rng.random((50, 2)).astype(np.float32)
    beta = np.array([[2.0, -1.0, 0.5], [0.0, 3.0, 1.0]])
    Y = (cov @ beta + 1.0 + 0.01 * rng.standard_normal((50, 3))).astype(np.float32)
    R = np.asarray(ml.RegressOut().fit_transform(Y, cov))
    design = np.column_stack([np.ones(50), cov.astype(np.float64)])
    coef, *_ = np.linalg.lstsq(design, Y.astype(np.float64), rcond=None)
    np.testing.assert_allclose(R, Y - design @ coef, atol=2e-4)
    Rs = np.asarray(ml.RegressOut().fit_transform(_Csr(Y), cov))
    np.testing.assert_allclose(Rs, R, atol=1e-5)
    m = ml.RegressOut().fit(Y, cov)
    with pytest.raises(ValueError, match="covariates"):
        m.transform(Y)
    with pytest.raises(ValueError, match="rows"):
        ml.RegressOut().fit(Y, cov[:10])


def test_pipeline_and_params():
    pipe = ml.Pipeline([("norm", ml.TotalCountNormalizer(target_sum=1e4)), ("log", ml.Log1pTransformer())]) \
        if hasattr(ml, "Pipeline") else None
    if pipe is not None:
        assert np.asarray(pipe.fit_transform(X)).shape == X.shape
    assert ml.HighlyVariableGenes(n_top_genes=5).get_params()["n_top_genes"] == 5
