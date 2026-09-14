# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""identity_break lane bodies for the kernel methods door (workstream D,
2026-09-14), for the harness's owner to merge into tools/identity_break.py.
Three lanes, one per class, sized like the gp lane (the kernel matrix and
the Cholesky are n^2): 256 rows of four columns for KernelRidge and
Nystroem, the full 16 columns for RBFSampler whose fit never reads X.
`gamma` is spelled explicitly so no `1 / n_features` division sits on the
Python side of the record.
"""


@lane("kernel-ridge")
def _(ml, X, yc, yr, Xh=None):
    """KernelRidge (python/mojolearn/kernel_methods.py) at the rbf kernel:
    the kernel matrix, the ridge, potrf and potrs, and the identical GEMM
    at OP_NN in predict. Train hashes dual_coef_ and the predictions on
    the training rows; infer predicts 64 held-out rows; the model column
    is n/a:no-save."""
    m = ml.KernelRidge(alpha=0.1, kernel="rbf", gamma=0.5).fit(X[:256, :4], yr[:256])
    return _fit(dict(dual=_h(m.dual_coef_), info=_h(np.int64(m.info_)), predict=_h(m.predict(X[256:320, :4]))),
                m, lambda e: (e.predict(Xh[:64, :4]),))


@lane("nystroem")
def _(ml, X, yc, yr, Xh=None):
    """Nystroem at the rbf kernel with 32 components of 256 rows: the
    device permutation (km_basis_indices), the basis kernel, the Jacobi
    eigensolver with its sweep count, the sign flip, the clip and the
    transposed normalization (DEVIATION 1674). Train hashes every model
    array the estimator carries; infer transforms 64 held-out rows."""
    m = ml.Nystroem(kernel="rbf", gamma=0.5, n_components=32, random_state=7).fit(X[:256, :4])
    return _fit(dict(components=_h(m.components_), indices=_h(m.component_indices_), normalization=_h(m.normalization_),
                     eigenvalues=_h(m.eigenvalues_), eigenvectors=_h(m.eigenvectors_), sweeps=_h(np.int64(m.sweeps_)),
                     transform=_h(m.transform(X[256:320, :4]))),
                m, lambda e: (e.transform(Xh[:64, :4]),))


@lane("rbf-sampler")
def _(ml, X, yc, yr, Xh=None):
    """RBFSampler with 64 random Fourier features over the 16 fixture
    columns: the Philox draws (weights and offsets), sigma and scale, and
    the identical GEMM at OP_NN plus cos in transform. The fit reads only
    n_features, so the train column's draws are the same on every fixture
    and only the transform moves with the data."""
    m = ml.RBFSampler(gamma=0.5, n_components=64, random_state=1).fit(X)
    return _fit(dict(weights=_h(m.random_weights_), offset=_h(m.random_offset_),
                     sigma=_h(np.float32(m.sigma_)), scale=_h(np.float32(m.scale_)), transform=_h(m.transform(X[:256]))),
                m, lambda e: (e.transform(Xh[:256]),))
