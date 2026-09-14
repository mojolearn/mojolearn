# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""identity_break lane bodies for the Gaussian mixture door (workstream D,
2026-09-14), for the harness's owner to merge into tools/identity_break.py.
Sized like the dbscan lane (6000 rows of four columns): the E-step is
n x K x d^2 and the M-step folds over n. Two lanes, one per implemented
init. A collapsed component RAISES (DEVIATION 1723) and the harness records
the refusal; on the `dupes` fixture that is a legal cell, not a bug.
"""


@lane("gmm")
def _(ml, X, yc, yr, Xh=None):
    """GaussianMixture (python/mojolearn/mixture.py), full covariance,
    four components, init through the identity-certified k-means. Train
    hashes every model array plus n_iter_, converged_ and lower_bound_
    (the estimator's header: part of the card); infer scores and labels
    64 held-out rows; the model column is n/a:no-save."""
    m = ml.GaussianMixture(n_components=4, max_iter=30, random_state=3).fit(X[:6000, :4])
    return _fit(dict(weights=_h(m.weights_), means=_h(m.means_), covariances=_h(m.covariances_),
                     precisions=_h(m.precisions_cholesky_), logdet=_h(m.log_det_chol_),
                     n_iter=_h(np.int64(m.n_iter_)), converged=_h(np.int64(int(m.converged_))),
                     lower_bound=_h(np.float32(m.lower_bound_)),
                     labels=_h(m.predict(X[:6000, :4])), proba=_h(m.predict_proba(X[:256, :4]))),
                m, lambda e: (e.score_samples(Xh[:64, :4]), e.predict(Xh[:64, :4]), e.predict_proba(Xh[:64, :4])))


@lane("gmm-random-init")
def _(ml, X, yc, yr, Xh=None):
    """The same fit under init_params='random' (position-mapped Philox
    responsibilities, DEVIATION 1733), the other implemented init."""
    m = ml.GaussianMixture(n_components=4, max_iter=30, random_state=3, init_params="random").fit(X[:6000, :4])
    return _fit(dict(weights=_h(m.weights_), means=_h(m.means_), covariances=_h(m.covariances_),
                     n_iter=_h(np.int64(m.n_iter_)), lower_bound=_h(np.float32(m.lower_bound_)),
                     labels=_h(m.predict(X[:6000, :4]))),
                m, lambda e: (e.score_samples(Xh[:64, :4]),))
