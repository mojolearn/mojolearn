# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate for the Python surface of `mojolearn.kernel_methods`
(`KernelRidge`, `Nystroem`, `RBFSampler`; workstream D, 2026-09-14). A gate
on the WIRING: the params lists, the model arrays round-tripping from fit
to predict/transform, and every refusal by name. The arithmetic is gated by
`pixi run check-kernel-methods`.

    cd python && python3 -m mojolearn.tests.test_kernel_methods_surface

Exit 2 naming `bindings/build_kernel_methods.sh` when unbuilt. Written on
one Apple M4 with no built binary in the worktree; the first run is owed.
"""
import sys

import numpy as np

import mojolearn
from mojolearn import kernel_methods as km
from mojolearn.tests._expose_d_harness import Report, bind_or_exit, mode, run


def _xy(n=48, d=3, seed=0):
    rng = np.random.default_rng(seed)
    x = (rng.random((n, d), dtype=np.float32) * 2.0 - 1.0).astype(np.float32)
    y = (x[:, 0] - 0.5 * x[:, 1] + 0.25 * x[:, 2]).astype(np.float32)
    return x, y


def _bits_same(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))


def arm_kernel_ridge(rep):
    x, y = _xy()
    m = km.KernelRidge(alpha=1e-3, kernel="linear").fit(x, y)
    rep.check("KRR", m.info_ == 0 and np.asarray(m.dual_coef_).shape == (48,), "linear fit: info_ 0, dual_coef_ (n,)")
    p = np.asarray(m.predict(x))
    rep.check("KRR", p.dtype == np.float32 and np.max(np.abs(p - y)) < 5e-2, "a linear kernel with a small ridge recovers a linear target at 5e-2",
              float(np.max(np.abs(p - y))))
    p1 = np.asarray(m.predict(x[:1]))
    if mode() == "identical":
        rep.check("KRR", _bits_same(p1, p[:1]), "one row predicted alone equals that row of the batch, bit for bit")
    else:
        rep.report_only("KRR", _bits_same(p1, p[:1]), "row alone vs in batch")
    m2 = km.KernelRidge(alpha=0.1, kernel="rbf", gamma=0.7).fit(x, np.stack([y, -y], 1))
    p2 = np.asarray(m2.predict(x[:5]))
    rep.check("KRR", p2.shape == (5, 2) and np.isfinite(p2).all(), "rbf fit with two targets predicts (q, 2)", p2.shape)
    rep.check("KRR", np.max(np.abs(p2[:, 0] + p2[:, 1])) < 1e-4, "the two targets are negatives of each other", float(np.max(np.abs(p2[:, 0] + p2[:, 1]))))
    for k in ("poly", "sigmoid", "laplacian"):
        mk = km.KernelRidge(alpha=0.5, kernel=k, gamma=0.3, degree=2, coef0=1.0).fit(x, y)
        rep.check("KRR", mk.info_ == 0 and np.isfinite(np.asarray(mk.predict(x[:3]))).all(), "kernel %r fits and predicts finite values" % k)


def arm_nystroem(rep):
    x, _ = _xy(n=32)
    ny = km.Nystroem(kernel="linear", n_components=32, random_state=7).fit(x)
    rep.check("NYS", np.asarray(ny.components_).shape == (32, 3) and np.asarray(ny.component_indices_).shape == (32,), "components_ and component_indices_ shaped")
    rep.check("NYS", sorted(np.asarray(ny.component_indices_).tolist()) == list(range(32)), "with n_components == n the indices are a permutation of the rows")
    ev = np.asarray(ny.eigenvalues_)
    rep.check("NYS", np.all(np.diff(ev) <= 0) and np.all(ev >= 1e-12), "eigenvalues_ descending and clipped at 1e-12 (DEVIATION 1670)")
    rep.check("NYS", ny.sweeps_ >= 1, "sweeps_ is carried", ny.sweeps_)
    phi = np.asarray(ny.transform(x))
    k_exact = x.astype(np.float64) @ x.astype(np.float64).T
    gram = phi.astype(np.float64) @ phi.astype(np.float64).T
    rep.check("NYS", np.max(np.abs(gram - k_exact)) < 1e-2 * np.max(np.abs(k_exact)), "with every row a component, phi phi^T reproduces the linear kernel at 1e-2",
              float(np.max(np.abs(gram - k_exact))))
    phi1 = np.asarray(ny.transform(x[:1]))
    if mode() == "identical":
        rep.check("NYS", _bits_same(phi1, phi[:1]), "one row transformed alone equals that row of the batch, bit for bit")
    else:
        rep.report_only("NYS", _bits_same(phi1, phi[:1]), "row alone vs in batch")
    ny2 = km.Nystroem(kernel="rbf", gamma=0.5, n_components=8, random_state=3).fit(x)
    t = np.asarray(ny2.transform(x[:6]))
    rep.check("NYS", t.shape == (6, 8) and np.isfinite(t).all(), "rbf Nystroem with 8 components transforms (m, 8)")
    t2 = np.asarray(km.Nystroem(kernel="rbf", gamma=0.5, n_components=8, random_state=3).fit_transform(x))
    rep.check("NYS", _bits_same(t2[:6], t), "fit_transform equals fit then transform, bit for bit (same seed, same rows)")


def arm_rbf_sampler(rep):
    x, _ = _xy(n=20, d=4)
    rf = km.RBFSampler(gamma=0.5, n_components=16, random_state=1).fit(x)
    w = np.asarray(rf.random_weights_)
    rep.check("RFF", w.shape == (4, 16) and np.asarray(rf.random_offset_).shape == (16,), "random_weights_ (d, q) and random_offset_ (q,)")
    rep.check("RFF", abs(rf.sigma_ - np.sqrt(1.0)) < 1e-6 and abs(rf.scale_ - np.sqrt(2.0 / 16)) < 1e-6, "sigma_ = sqrt(2 gamma), scale_ = sqrt(2 / q)", (rf.sigma_, rf.scale_))
    f = np.asarray(rf.transform(x))
    rep.check("RFF", f.shape == (20, 16) and np.all(np.abs(f) <= rf.scale_ + 1e-6), "transform is (m, q) and bounded by scale_")
    rf2 = km.RBFSampler(gamma=0.5, n_components=16, random_state=1).fit(x)
    rep.check("RFF", _bits_same(np.asarray(rf2.random_weights_), w), "the same seed draws the same weights, bit for bit")
    rf3 = km.RBFSampler(gamma=0.5, n_components=16, random_state=2).fit(x)
    rep.check("RFF", not _bits_same(np.asarray(rf3.random_weights_), w), "a different seed draws different weights")
    f1 = np.asarray(rf.transform(x[:1]))
    if mode() == "identical":
        rep.check("RFF", _bits_same(f1, f[:1]), "one row transformed alone equals that row of the batch, bit for bit")
    else:
        rep.report_only("RFF", _bits_same(f1, f[:1]), "row alone vs in batch")


def arm_refusals(rep):
    x, y = _xy(n=16)
    rep.raises("REFUSE", ValueError, "precomputed", "kernel='precomputed' by name", km.KernelRidge(kernel="precomputed").fit, x, y)
    rep.raises("REFUSE", ValueError, "kernel must be", "an unknown kernel name", km.KernelRidge(kernel="cosine").fit, x, y)
    rep.raises("REFUSE", Exception, "alpha", "a negative alpha, refused on the Mojo host by name", km.KernelRidge(alpha=-1.0).fit, x, y)
    rep.raises("REFUSE", Exception, "gamma", "gamma <= 0 under rbf, refused on the Mojo host by name", km.KernelRidge(kernel="rbf", gamma=0.0).fit, x, y)
    rep.raises("REFUSE", ValueError, "rows", "y with the wrong row count", km.KernelRidge().fit, x, y[:5])
    rep.raises("REFUSE", ValueError, "fit before predict", "predict before fit", km.KernelRidge().predict, x)
    m = km.KernelRidge().fit(x, y)
    rep.raises("REFUSE", ValueError, "features", "predict with the wrong feature count", m.predict, x[:, :2])
    rep.raises("REFUSE", ValueError, "positive", "Nystroem n_components=0", km.Nystroem(n_components=0).fit, x)
    rep.raises("REFUSE", Exception, "", "Nystroem n_components > n, refused on the Mojo host", km.Nystroem(kernel="linear", n_components=17).fit, x)
    rep.raises("REFUSE", ValueError, "scale", "RBFSampler gamma='scale' by name", km.RBFSampler(gamma="scale").fit, x)
    rep.raises("REFUSE", Exception, "gamma", "RBFSampler gamma=0, refused on the Mojo host by name", km.RBFSampler(gamma=0.0, n_components=4).fit, x)
    rep.raises("REFUSE", ValueError, "n_components", "RBFSampler n_components=0, refused by name before any buffer is made", km.RBFSampler(gamma=0.5, n_components=0).fit, x)
    rep.raises("REFUSE", TypeError, "degree", "a float degree", km.KernelRidge(kernel="poly", gamma=0.5, degree=2.5).fit, x, y)


def arm_provenance(rep):
    for name in ("KernelRidge", "Nystroem", "RBFSampler"):
        rep.check("PROVENANCE", name in mojolearn.__all__ and name in km.__all__, "%s exported from mojolearn and mojolearn.kernel_methods" % name)
    rep.check("PROVENANCE", km.KernelRidge().numeric_mode_used() == mode(), "numeric_mode_used() is the process default")


def main(out=sys.stdout):
    bind_or_exit("_mojolearn_kernel_methods", "build_kernel_methods.sh")
    rep = Report("test_kernel_methods_surface")
    return run("test_kernel_methods_surface", [("KRR", arm_kernel_ridge), ("NYS", arm_nystroem), ("RFF", arm_rbf_sampler),
                                               ("REFUSE", arm_refusals), ("PROVENANCE", arm_provenance)], rep, out)


if __name__ == "__main__":
    sys.exit(main())
