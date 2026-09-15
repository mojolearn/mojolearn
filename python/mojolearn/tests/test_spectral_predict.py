# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate for `SpectralClustering.predict` (lane/spectral-predict,
2026-09-15; DEVIATION 2860, NEW capability that neither cuML nor
scikit-learn has): the Nystrom out-of-sample extension (Bengio et al., NIPS
2003) and the fit's own k-means assignment.

    cd python && python3 -m mojolearn.tests.test_spectral_predict

It gates the WIRING and the RULE on small fixtures: `prediction_data=True`
moves no fit byte, a query equal to a training row, a far-away query,
duplicated training rows (k-NN distance ties), float64 queries, rows alone
and in reversed order against the batch, the precomputed affinity arm, save
and load, the eigenvalue threshold and every refusal. It REPORTS how often
`predict(X_train)` equals `labels_`, which the rule does not promise.

When `MOJOLEARN_HOST_DIR` holds `_mojolearn_metrics_host.so` the HOST arm
loads each saved model through `_classical_host.host_model` and holds the
CPU host binding's answers to the GPU binding's bytes.

Exit 2 naming the build script when the GPU binding is unbuilt.
"""
import os
import sys
import tempfile

import numpy as np

import mojolearn
from mojolearn import SpectralClustering
from mojolearn._array import Array
from mojolearn.tests._expose_d_harness import Report, bind_or_exit, mode, run


def _identical(rep, arm, cond, what):
    if mode() == "identical":
        return rep.check(arm, cond, what)
    return rep.report_only(arm, cond, what)


def _blobs():
    """Three separated 2-D blobs of 40 rows each, blob order interleaved so
    no cluster owns the low indices."""
    rng = np.random.default_rng(20260915)
    centers = np.asarray([[0.0, 0.0], [8.0, 0.0], [0.0, 8.0]])
    rows = [centers[i % 3] + rng.normal(0.0, 0.7, 2) for i in range(120)]
    return np.ascontiguousarray(np.asarray(rows, dtype=np.float32))


def _affinity(Q, P):
    """1 / (1 + d2) under the median of P's own squared distances, else 0."""
    Pd, Qd = P.astype(np.float64), Q.astype(np.float64)
    d2 = ((Pd[:, None, :] - Pd[None, :, :]) ** 2).sum(axis=2)
    c2 = ((Qd[:, None, :] - Pd[None, :, :]) ** 2).sum(axis=2)
    t = float(np.median(d2))
    return np.ascontiguousarray(np.where(c2 < t, 1.0 / (1.0 + c2), 0.0).astype(np.float32))


def arm_nearest_neighbors(rep):
    x = _blobs()
    kw = dict(n_clusters=3, n_neighbors=8, random_state=3)
    plain = SpectralClustering(**kw).fit(x)
    m = SpectralClustering(prediction_data=True, **kw).fit(x)
    rep.check("NN", np.array_equal(np.asarray(plain.labels_), np.asarray(m.labels_))
              and np.asarray(plain.embedding_).tobytes() == np.asarray(m.embedding_).tobytes(),
              "prediction_data=True fits the same labels_ and embedding_ bytes as the default")
    rep.check("NN", getattr(plain, "_pd_eigenvalues", None) is None, "the default fit stores no prediction data")
    ev = np.asarray(m._pd_eigenvalues)
    rep.check("NN", ev.shape == (3,) and np.asarray(m._pd_eigenvectors).shape == (120, 3)
              and np.asarray(m._pd_diag).shape == (120,) and np.asarray(m._pd_centroids).shape == (3, 3),
              "prediction data shapes: eigenvalues (k,), eigenvectors (n, k), diag (n,), centroids (n_clusters, k)")
    rep.check("NN", bool(np.all(np.abs(1.0 + ev) > 1e-3)), "every 1 + theta clears the DEVIATION 2860 threshold", ev.tolist())
    rep.check("NN", len(set(np.asarray(m.labels_).tolist())) == 3, "the fixture fits three clusters")

    p, emb = (np.asarray(a) for a in m._predict_embedding(x))
    rep.check("NN", p.dtype == np.int32 and p.shape == (120,) and emb.dtype == np.float32 and emb.shape == (120, 3),
              "predict returns int32 (n,), the extended embedding float32 (n, k)")
    rep.check("NN", bool(np.all(np.isfinite(emb))) and set(p.tolist()) <= {0, 1, 2}, "labels in range, embedding finite")
    agree = float(np.mean(p == np.asarray(m.labels_)))
    rep.report_only("NN", agree == 1.0, f"predict(X_train) == labels_ on {agree:.4f} of the training rows (not promised)")
    gap = float(np.max(np.abs(emb - np.asarray(m.embedding_))))
    rep.report_only("NN", gap == 0.0, f"training-row embedding max |extension - embedding_| = {gap:.3e} (not promised)")
    rep.check("NN", np.array_equal(np.asarray(m.predict(x)), p), "predict equals _predict_embedding's labels")

    one = np.asarray(m.predict(x[5:6]))
    _identical(rep, "NN", int(one[0]) == int(p[5]), "a query equal to a training row alone gives the batch's label")
    rep.report_only("NN", int(one[0]) == int(m.labels_[5]), "a query equal to training row 5 predicts its fitted label")
    far = np.asarray([[1.0e4, -1.0e4], [-3.0e4, 2.5e4]], dtype=np.float32)
    fl, fe = (np.asarray(a) for a in m._predict_embedding(far))
    rep.check("NN", set(fl.tolist()) <= {0, 1, 2} and bool(np.all(np.isfinite(fe))),
              "far-away queries get a label in range and a finite embedding", (fl.tolist(), fe.tolist()))
    q64 = x[:9].astype(np.float64)
    rep.check("NN", np.array_equal(np.asarray(m.predict(q64)), p[:9]), "a float64 query returns the float32 query's labels")
    alone = [int(np.asarray(m.predict(x[i:i + 1]))[0]) for i in range(0, 120, 7)]
    rev = np.asarray(m.predict(x[::-1].copy()))
    _identical(rep, "NN", alone == p[::7].tolist() and np.array_equal(rev[::-1], p),
               "rows alone and reversed query order give the batch's bytes")

    # Duplicated training rows: every query equal to a training row has two
    # neighbors at distance 0, a k-NN tie the search breaks by index.
    dup = np.ascontiguousarray(np.concatenate([x, x]).astype(np.float32))
    dm = SpectralClustering(prediction_data=True, **kw).fit(dup)
    dp = np.asarray(dm.predict(dup))
    dalone = [int(np.asarray(dm.predict(dup[i:i + 1]))[0]) for i in (0, 120, 7, 127)]
    _identical(rep, "NN", dalone == [int(dp[0]), int(dp[120]), int(dp[7]), int(dp[127])],
               "duplicated rows (distance-0 k-NN ties): rows alone give the batch's labels")
    rep.check("NN", np.array_equal(dp[:120], dp[120:]), "a row and its duplicate predict the same label")
    rep.report_only("NN", np.array_equal(dp, np.asarray(dm.labels_)),
                    f"duplicated rows: predict(X_train) == labels_ on {float(np.mean(dp == np.asarray(dm.labels_))):.4f}")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "spectral.npz")
        m.save(path)
        back = SpectralClustering.load(path)
        rep.check("NN", np.array_equal(np.asarray(back.predict(x)), p) and not hasattr(back, "embedding_"),
                  "save and load predict the same bytes; the file holds no embedding_")
        bad = SpectralClustering.load(path)
        vals = np.asarray(bad._pd_eigenvalues).copy()
        vals[1] = np.float32(-1.0)
        bad._pd_eigenvalues = Array.from_list([float(v) for v in vals], "<f4")
        rep.raises("NN", (Exception,), "DEVIATION 2860", "a used column with 1 + theta = 0 is refused by name",
                   bad.predict, x[:3])

    rep.raises("NN", ValueError, "prediction_data=True", "predict without prediction data refused by name", plain.predict, x)
    rep.raises("NN", ValueError, "prediction_data=True", "save without prediction data refused by name",
               plain.save, os.path.join(tempfile.gettempdir(), "never_spectral.npz"))
    rep.raises("NN", ValueError, "not fitted", "predict before fit refused by name", SpectralClustering().predict, x)
    rep.raises("NN", ValueError, "features", "a query with the wrong feature count", m.predict, x[:, :1])
    nanq = x[:3].copy(); nanq[1, 0] = np.float32("nan")
    rep.raises("NN", ValueError, "NaN", "a NaN query refused by name", m.predict, nanq)
    rep.raises("NN", TypeError, "prediction_data", "prediction_data must be a bool",
               SpectralClustering(prediction_data=1).fit, x)
    rep.raises("NN", ValueError, "refused", "affinity='rbf' is refused at construction", SpectralClustering, affinity="rbf")
    odd = SpectralClustering(prediction_data=True, **kw).fit(x)
    odd.affinity = "rbf"
    rep.raises("NN", ValueError, "no out-of-sample rule", "an unsupported affinity refused by name at predict", odd.predict, x)
    return m, x


def arm_precomputed(rep):
    x = _blobs()
    A = _affinity(x, x)
    kw = dict(n_clusters=3, affinity="precomputed", random_state=3)
    plain = SpectralClustering(**kw).fit(A)
    m = SpectralClustering(prediction_data=True, **kw).fit(A)
    rep.check("PRE", np.array_equal(np.asarray(plain.labels_), np.asarray(m.labels_))
              and np.asarray(plain.embedding_).tobytes() == np.asarray(m.embedding_).tobytes(),
              "prediction_data=True fits the same labels_ and embedding_ bytes")
    p = np.asarray(m.predict(A))
    agree = float(np.mean(p == np.asarray(m.labels_)))
    rep.check("PRE", p.dtype == np.int32 and p.shape == (120,) and set(p.tolist()) <= {0, 1, 2}, "predict on (n_new, n_train)")
    rep.report_only("PRE", agree == 1.0, f"predict(A_train) == labels_ on {agree:.4f} of the training rows (not promised)")
    rows = np.asarray([[1.0e4, 1.0e4], [4.0, 4.0]], dtype=np.float32)
    Ah = _affinity(rows, x)
    hl = np.asarray(m.predict(Ah))
    rep.check("PRE", set(hl.tolist()) <= {0, 1, 2} and float(Ah[0].sum()) == 0.0,
              "a row with no affinity to any training row (degree 0, the zero-to-one rule) gets a label in range", hl.tolist())
    alone = [int(np.asarray(m.predict(A[i:i + 1]))[0]) for i in range(0, 120, 11)]
    _identical(rep, "PRE", alone == p[::11].tolist(), "rows alone give the batch's labels")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "spectral_pre.npz")
        m.save(path)
        back = SpectralClustering.load(path)
        rep.check("PRE", np.array_equal(np.asarray(back.predict(A)), p), "save and load predict the same bytes")
    rep.raises("PRE", ValueError, "shape (n_new, 120)", "an affinity with the wrong training width refused by name", m.predict, A[:, :50])
    neg = A[:2].copy(); neg[0, 3] = np.float32(-0.5)
    rep.raises("PRE", ValueError, "negative", "a negative affinity refused by name", m.predict, neg)
    inf = A[:2].copy(); inf[1, 0] = np.float32("inf")
    rep.raises("PRE", ValueError, "non-finite", "a non-finite affinity refused by name", m.predict, inf)
    return m, A


def arm_host(rep, nn_model, x, pre_model, A):
    try:
        from mojolearn import _backend, _classical_host
        _backend.load_host_module("_mojolearn_metrics_host")
    except Exception as exc:  # noqa: BLE001
        rep.report_only("HOST", False, f"host binding not loadable ({exc}); set MOJOLEARN_HOST_DIR")
        return
    far = np.asarray([[1.0e4, -1.0e4]], dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        for name, model, queries in (("nearest_neighbors", nn_model, np.concatenate([x, far]).astype(np.float32)),
                                     ("precomputed", pre_model, A)):
            path = os.path.join(tmp, name + ".npz")
            model.save(path)
            host = _classical_host.host_model(path)
            rep.check("HOST", host.vendor_used() == "cpu", f"{name}: host_model binds the CPU host binding")
            gl, ge = (np.asarray(a) for a in model._predict_embedding(queries))
            cl, ce = (np.asarray(a) for a in host._predict_embedding(queries))
            rep.check("HOST", gl.tobytes() == cl.tobytes() and ge.tobytes() == ce.tobytes(),
                      f"{name}: CPU host labels and embedding equal the GPU bytes")


def main(out=sys.stdout):
    bind_or_exit("_mojolearn_metrics", "build_metrics.sh")
    rep = Report("test_spectral_predict")
    state = {}

    def nn(r):
        state["nn"] = arm_nearest_neighbors(r)

    def pre(r):
        state["pre"] = arm_precomputed(r)

    def host(r):
        if "nn" in state and "pre" in state:
            arm_host(r, state["nn"][0], state["nn"][1], state["pre"][0], state["pre"][1])

    def provenance(r):
        r.check("PROVENANCE", "SpectralClustering" in mojolearn.__all__, "SpectralClustering exported")

    return run("test_spectral_predict", [("NN", nn), ("PRE", pre), ("HOST", host), ("PROVENANCE", provenance)], rep, out)


if __name__ == "__main__":
    sys.exit(main())
