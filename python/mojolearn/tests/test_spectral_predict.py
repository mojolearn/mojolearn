# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate for `SpectralClustering.predict` (lane/spectral-predict,
2026-09-15; DEVIATION 2860, NEW capability that neither cuML nor
scikit-learn has): the Nystrom out-of-sample extension (Bengio et al., NIPS
2003) and the fit's own k-means assignment.

    cd python && python3 -m mojolearn.tests.test_spectral_predict

It gates the WIRING on small fixtures: the prediction-data arrays, float64
queries, save and load, the CPU host binding and every refusal. It REPORTS
how often `predict(X_train)` equals `labels_`, which the rule does not
promise. The RULE (`prediction_data=True` moves no fit byte, the eigenvalue
threshold, a query equal to a training row, far-away queries, duplicated
training rows, rows alone and in reversed order, the zero-degree precomputed
row, labels in range and finite embeddings) is the Mojo check
`checks/native_tests/spectral_prediction_*` (`pixi run
check-native-spectral-prediction-cpu` / `-gpu`).

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
from mojolearn.tests._expose_d_harness import Report, bind_or_exit, run


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
    rep.check("NN", getattr(plain, "_pd_eigenvalues", None) is None, "the default fit stores no prediction data")
    ev = np.asarray(m._pd_eigenvalues)
    rep.check("NN", ev.shape == (3,) and np.asarray(m._pd_eigenvectors).shape == (120, 3)
              and np.asarray(m._pd_diag).shape == (120,) and np.asarray(m._pd_centroids).shape == (3, 3),
              "prediction data shapes: eigenvalues (k,), eigenvectors (n, k), diag (n,), centroids (n_clusters, k)")

    p, emb = (np.asarray(a) for a in m._predict_embedding(x))
    rep.check("NN", p.dtype == np.int32 and p.shape == (120,) and emb.dtype == np.float32 and emb.shape == (120, 3),
              "predict returns int32 (n,), the extended embedding float32 (n, k)")
    agree = float(np.mean(p == np.asarray(m.labels_)))
    rep.report_only("NN", agree == 1.0, f"predict(X_train) == labels_ on {agree:.4f} of the training rows (not promised)")
    gap = float(np.max(np.abs(emb - np.asarray(m.embedding_))))
    rep.report_only("NN", gap == 0.0, f"training-row embedding max |extension - embedding_| = {gap:.3e} (not promised)")
    rep.check("NN", np.array_equal(np.asarray(m.predict(x)), p), "predict equals _predict_embedding's labels")

    one = np.asarray(m.predict(x[5:6]))
    rep.report_only("NN", int(one[0]) == int(m.labels_[5]), "a query equal to training row 5 predicts its fitted label")
    q64 = x[:9].astype(np.float64)
    rep.check("NN", np.array_equal(np.asarray(m.predict(q64)), p[:9]), "a float64 query returns the float32 query's labels")

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
    rep.check("NN", SpectralClustering(affinity="rbf").affinity == "rbf",
              "RBF affinity is accepted for fitting")
    rep.raises("NN", ValueError, "refused", "unknown affinity is refused",
               SpectralClustering(affinity="unknown").fit, x)
    odd = SpectralClustering(prediction_data=True, **kw).fit(x)
    odd.affinity = "rbf"
    rep.raises("NN", ValueError, "no out-of-sample rule", "an unsupported affinity refused by name at predict", odd.predict, x)
    return m, x


def arm_precomputed(rep):
    x = _blobs()
    A = _affinity(x, x)
    kw = dict(n_clusters=3, affinity="precomputed", random_state=3)
    m = SpectralClustering(prediction_data=True, **kw).fit(A)
    p = np.asarray(m.predict(A))
    agree = float(np.mean(p == np.asarray(m.labels_)))
    rep.check("PRE", p.dtype == np.int32 and p.shape == (120,), "predict on (n_new, n_train)")
    rep.report_only("PRE", agree == 1.0, f"predict(A_train) == labels_ on {agree:.4f} of the training rows (not promised)")
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
