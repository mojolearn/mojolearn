# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate for `DBSCAN.predict` and `AgglomerativeClustering.predict`
(lane/inference-transductive-predict, 2026-09-15; DEVIATION 2740, NEW
capability that neither cuML nor scikit-learn has).

    cd python && python3 -m mojolearn.tests.test_transductive_predict

This controller keeps both models' public array packing, prediction-data
flags, archive handling, the CPU host binding against the GPU bytes and the
refusals. The DBSCAN fit/predict corner oracles execute in
checks/native_tests/dbscan_native_cases.mojo and the agglomerative tree and
query corner oracles in checks/native_tests/agglo_native_cases.mojo.

When `MOJOLEARN_HOST_DIR` holds `_mojolearn_estimators_host.so` the HOST
arm loads every saved model through `_classical_host.host_model` and holds
the CPU host binding's answers to the GPU binding's bytes on every fixture.

Exit 2 naming the build script when the GPU binding is unbuilt.
"""
import os
import sys
import tempfile

import numpy as np

import mojolearn
from mojolearn import AgglomerativeClustering, DBSCAN
from mojolearn.tests._expose_d_harness import Report, bind_or_exit, run


def _grid_clusters():
    """Two integer-grid clusters with cluster B's rows FIRST, so the lowest
    core index belongs to the cluster with the higher x (the tie rule is
    then visible), and a few isolated points that are noise at eps=1."""
    b = [(4.0 + i, float(j)) for i in range(3) for j in range(3)]
    a = [(-4.0 + i, float(j)) for i in range(3) for j in range(3)]
    noise = [(0.0, 10.0), (20.0, -20.0)]
    return np.ascontiguousarray(np.asarray(b + a + noise, dtype=np.float32))


def arm_dbscan(rep):
    # The numerical corners (fit labels and core/noise membership, every
    # training row's prediction, a query at eps and one ulp past it, the
    # exact tie to the lowest (distance, core index), far points, rows alone
    # and reversed, duplicated rows, algorithm='rbc', metric='manhattan',
    # uniform weights and an empty core) run natively on this same grid in
    # checks/native_tests/dbscan_native_cases.mojo (profiles grid_*). This
    # arm keeps the public flags, attributes, packing, archive and refusals.
    x = _grid_clusters()
    plain = DBSCAN(eps=1.0, min_samples=4, algorithm="brute").fit(x)
    m = DBSCAN(eps=1.0, min_samples=4, algorithm="brute", prediction_data=True).fit(x)
    fitted = np.asarray(m.labels_)
    rep.check("DBSCAN", np.array_equal(np.asarray(plain.labels_), fitted),
              "prediction_data=True fits the same labels_ bytes as the default")
    rep.check("DBSCAN", np.array_equal(np.asarray(plain.core_sample_indices_), np.asarray(m.core_sample_indices_)),
              "the default fit keeps core_sample_indices_ too (scikit-learn's attribute)")
    try:
        plain.predict(x)
        rep.check("DBSCAN", False, "the default fit refuses predict by name")
    except ValueError as e:
        rep.check("DBSCAN", "prediction_data=True" in str(e), "the default fit refuses predict by name", str(e))
    idx = np.asarray(m.core_sample_indices_)
    comps = np.asarray(m.components_)
    rep.check("DBSCAN", idx.dtype == np.int32 and comps.dtype == np.float32 and comps.shape == (idx.size, 2),
              "core_sample_indices_ int32 and components_ float32 (n_core, d)", (idx.dtype, comps.shape))
    rep.check("DBSCAN", np.array_equal(comps, x[idx]), "components_ are the training rows at core_sample_indices_")

    p = np.asarray(m.predict(x))
    rep.check("DBSCAN", p.dtype == np.int32 and p.shape == (x.shape[0],), "predict returns int32 (n,)")
    q64 = x[:7].astype(np.float64)
    rep.check("DBSCAN", np.array_equal(np.asarray(m.predict(q64)), p[:7]), "a float64 query returns the float32 query's labels")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "dbscan.npz")
        m.save(path)
        back = DBSCAN.load(path)
        rep.check("DBSCAN", np.array_equal(np.asarray(back.predict(x)), p), "save and load predict the same bytes")

    rep.raises("DBSCAN", ValueError, "prediction_data=True", "predict without prediction data refused by name", plain.predict, x)
    rep.raises("DBSCAN", ValueError, "prediction_data=True", "save without prediction data refused by name",
               plain.save, os.path.join(tempfile.gettempdir(), "never.npz"))
    rep.raises("DBSCAN", ValueError, "not fitted", "predict before fit refused by name", DBSCAN().predict, x)
    rep.raises("DBSCAN", ValueError, "features", "a query with the wrong feature count", m.predict, x[:, :1])
    bad = x[:3].copy(); bad[1, 1] = np.float32("nan")
    rep.raises("DBSCAN", ValueError, "NaN", "a NaN query refused by name", m.predict, bad)
    rep.raises("DBSCAN", ValueError, "empty", "an empty query refused by name (as_f32_c)", m.predict, x[:0])
    rep.raises("DBSCAN", TypeError, "prediction_data", "prediction_data must be a bool", DBSCAN(prediction_data=1).fit, x)
    return m, x


def arm_agglomerative(rep):
    x = _grid_clusters()[:18]
    plain = AgglomerativeClustering(n_clusters=2).fit(x)
    m = AgglomerativeClustering(n_clusters=2, prediction_data=True).fit(x)
    fitted = np.asarray(m.labels_)
    rep.check("AGGLOM", np.array_equal(np.asarray(plain.labels_), fitted)
              and np.array_equal(np.asarray(plain.children_), np.asarray(m.children_)),
              "prediction_data=True fits the same labels_ and children_ bytes")
    p = np.asarray(m.predict(x))
    rep.check("AGGLOM", p.dtype == np.int32 and np.array_equal(p, fitted), "every training row predicts its fitted label")

    # Tree/query corner oracles execute in native agglo_native_cases.
    # Keep public packed attributes, prediction-data flags and archive seams.
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "agg.npz")
        m.save(path)
        back = AgglomerativeClustering.load(path)
        rep.check("AGGLOM", np.array_equal(np.asarray(back.predict(x)), p)
                  and not hasattr(back, "children_") and not hasattr(back, "n_boruvka_rounds_"),
                  "save and load predict the same bytes; the file holds no fit-algorithm state (children_, n_boruvka_rounds_)")
    rep.raises("AGGLOM", ValueError, "prediction_data=True", "predict without prediction data refused by name", plain.predict, x)
    rep.raises("AGGLOM", ValueError, "not fitted", "predict before fit refused by name", AgglomerativeClustering().predict, x)
    rep.raises("AGGLOM", ValueError, "features", "a query with the wrong feature count", m.predict, x[:, :1])
    rep.raises("AGGLOM", TypeError, "prediction_data", "prediction_data must be a bool",
               AgglomerativeClustering(prediction_data="yes").fit, x)
    return m, x


def arm_host(rep, dbscan_model, agg_model, x):
    """The CPU host binding against the GPU binding, bytes, on the saved
    models and on the corner queries."""
    try:
        from mojolearn import _backend, _classical_host
        _backend.load_host_module("_mojolearn_estimators_host")
    except Exception as exc:  # noqa: BLE001
        rep.report_only("HOST", False, f"host binding not loadable ({exc}); set MOJOLEARN_HOST_DIR")
        return
    queries = np.concatenate([
        x, np.asarray([[7.0, 1.0], [-5.0, 1.0], [1.0, 1.0], [1000.0, 0.0]], dtype=np.float32),
        np.asarray([[np.nextafter(np.float32(7.0), np.float32(8.0)), 1.0]], dtype=np.float32),
    ]).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        for name, model in (("dbscan", dbscan_model), ("agglomerative", agg_model)):
            path = os.path.join(tmp, name + ".npz")
            model.save(path)
            host = _classical_host.host_model(path)
            rep.check("HOST", host.vendor_used() == "cpu", f"{name}: host_model binds the CPU host binding")
            gpu = np.asarray(model.predict(queries))
            cpu = np.asarray(host.predict(queries))
            rep.check("HOST", gpu.tobytes() == cpu.tobytes(), f"{name}: CPU host predict equals the GPU bytes", (gpu.tolist(), cpu.tolist()))


def main(out=sys.stdout):
    bind_or_exit("_mojolearn_estimators", "build_estimators.sh")
    rep = Report("test_transductive_predict")
    state = {}

    def dbscan(r):
        state["dbscan"] = arm_dbscan(r)

    def agg(r):
        state["agg"] = arm_agglomerative(r)

    def host(r):
        if "dbscan" in state and "agg" in state:
            arm_host(r, state["dbscan"][0], state["agg"][0], state["dbscan"][1][:18])

    def provenance(r):
        r.check("PROVENANCE", "DBSCAN" in mojolearn.__all__ and "AgglomerativeClustering" in mojolearn.__all__,
                "DBSCAN and AgglomerativeClustering exported")

    return run("test_transductive_predict", [("DBSCAN", dbscan), ("AGGLOM", agg), ("HOST", host),
                                              ("PROVENANCE", provenance)], rep, out)


if __name__ == "__main__":
    sys.exit(main())
