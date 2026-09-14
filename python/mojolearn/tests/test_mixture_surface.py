# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate for the Python surface of `mojolearn.GaussianMixture`
(workstream D, 2026-09-14): a gate on the WIRING (the params list, the
strings decoded on the Mojo side, the model arrays round-tripping from fit
to the four scoring entries, every refusal by name). The arithmetic is
gated by `pixi run check-mixture`.

    cd python && python3 -m mojolearn.tests.test_mixture_surface

Exit 2 naming `bindings/build_mixture.sh` when unbuilt. Written on one
Apple M4 with no built binary in the worktree; the first run is owed.
"""
import sys

import numpy as np

import mojolearn
from mojolearn import GaussianMixture
from mojolearn.tests._expose_d_harness import Report, bind_or_exit, mode, run


def _blobs(seed=0):
    rng = np.random.default_rng(seed)
    a = rng.random((48, 2), dtype=np.float32) * 0.5
    b = rng.random((48, 2), dtype=np.float32) * 0.5 + np.float32(4.0)
    x = np.ascontiguousarray(np.concatenate([a, b]).astype(np.float32))
    planted = np.concatenate([np.zeros(48, np.int32), np.ones(48, np.int32)])
    return x, planted


def _bits_same(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))


def arm_fit(rep):
    x, planted = _blobs()
    m = GaussianMixture(n_components=2, max_iter=50, random_state=0).fit(x)
    rep.check("FIT", np.asarray(m.weights_).shape == (2,) and np.asarray(m.means_).shape == (2, 2), "weights_ (k,), means_ (k, d)")
    rep.check("FIT", np.asarray(m.covariances_).shape == (2, 2, 2) and np.asarray(m.precisions_cholesky_).shape == (2, 2, 2), "covariances_ and precisions_cholesky_ (k, d, d)")
    rep.check("FIT", abs(float(np.sum(np.asarray(m.weights_))) - 1.0) < 1e-5, "weights_ sum to one", float(np.sum(np.asarray(m.weights_))))
    rep.check("FIT", 1 <= m.n_iter_ <= 50 and isinstance(m.converged_, bool) and np.isfinite(m.lower_bound_), "n_iter_, converged_ and lower_bound_ carried", (m.n_iter_, m.converged_, m.lower_bound_))
    lab = np.asarray(m.predict(x))
    agree = max(np.mean(lab == planted), np.mean(lab == 1 - planted))
    rep.check("FIT", agree == 1.0, "predict separates two planted blobs (up to relabeling)", agree)
    pr = np.asarray(m.predict_proba(x))
    rep.check("FIT", pr.shape == (96, 2) and np.max(np.abs(pr.sum(1) - 1.0)) < 1e-4, "predict_proba rows sum to one", float(np.max(np.abs(pr.sum(1) - 1.0))))
    ll = np.asarray(m.score_samples(x))
    sc = m.score(x)
    rep.check("FIT", ll.shape == (96,) and abs(sc - float(np.mean(ll.astype(np.float64)))) < 1e-3 * max(1.0, abs(sc)), "score is the mean of score_samples at 1e-3", (sc, float(np.mean(ll))))
    rep.check("FIT", np.isfinite(m.bic(x)) and np.isfinite(m.aic(x)) and m.bic(x) != m.aic(x), "bic and aic are finite and differ", (m.bic(x), m.aic(x)))
    ll1 = np.asarray(m.score_samples(x[:1]))
    if mode() == "identical":
        rep.check("FIT", _bits_same(ll1, ll[:1]), "one row scored alone equals that row of the batch, bit for bit (DEVIATION 1739)")
    else:
        rep.report_only("FIT", _bits_same(ll1, ll[:1]), "row alone vs in batch")
    m2 = GaussianMixture(n_components=2, max_iter=50, random_state=0, init_params="random").fit(x)
    lab2 = np.asarray(m2.predict(x))
    rep.check("FIT", max(np.mean(lab2 == planted), np.mean(lab2 == 1 - planted)) == 1.0, "init_params='random' also separates the blobs")
    rep.check("FIT", np.asarray(m.fit_predict(x)).shape == (96,), "fit_predict")


def arm_refusals(rep):
    x, _ = _blobs()
    for ct in ("tied", "diag", "spherical"):
        rep.raises("REFUSE", Exception, ct, "covariance_type=%r refused by name on the Mojo host" % ct, GaussianMixture(n_components=2, covariance_type=ct).fit, x)
    for ip in ("k-means++", "random_from_data"):
        rep.raises("REFUSE", Exception, ip, "init_params=%r refused by name on the Mojo host" % ip, GaussianMixture(n_components=2, init_params=ip).fit, x)
    rep.raises("REFUSE", ValueError, "n_init", "n_init=2 refused by name (DEVIATION 1734)", GaussianMixture(n_components=2, n_init=2).fit, x)
    rep.raises("REFUSE", ValueError, "warm_start", "warm_start refused by name", GaussianMixture(n_components=2, warm_start=True).fit, x)
    rep.raises("REFUSE", ValueError, "means_init", "means_init refused by name", GaussianMixture(n_components=2, means_init=x[:2]).fit, x)
    rep.raises("REFUSE", ValueError, "positive", "n_components=0", GaussianMixture(n_components=0).fit, x)
    rep.raises("REFUSE", Exception, "", "n_components > n_samples, refused on the Mojo host", GaussianMixture(n_components=97).fit, x)
    bad = x.copy(); bad[3, 1] = np.float32("nan")
    rep.raises("REFUSE", Exception, "row 3", "a NaN cell named by row on the Mojo host", GaussianMixture(n_components=2).fit, bad)
    rep.raises("REFUSE", ValueError, "fit before", "predict before fit", GaussianMixture().predict, x)
    m = GaussianMixture(n_components=2, max_iter=10).fit(x)
    rep.raises("REFUSE", ValueError, "features", "predict with the wrong feature count", m.predict, x[:, :1])


def arm_provenance(rep):
    rep.check("PROVENANCE", "GaussianMixture" in mojolearn.__all__ and "mixture" in mojolearn.__all__, "GaussianMixture and mojolearn.mixture exported")
    rep.check("PROVENANCE", GaussianMixture().numeric_mode_used() == mode(), "numeric_mode_used() is the process default")


def main(out=sys.stdout):
    bind_or_exit("_mojolearn_mixture", "build_mixture.sh")
    rep = Report("test_mixture_surface")
    return run("test_mixture_surface", [("FIT", arm_fit), ("REFUSE", arm_refusals), ("PROVENANCE", arm_provenance)], rep, out)


if __name__ == "__main__":
    sys.exit(main())
