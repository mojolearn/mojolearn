# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The gate for `DBSCAN.predict` and `AgglomerativeClustering.predict`
(lane/inference-transductive-predict, 2026-09-15; DEVIATION 2740, NEW
capability that neither cuML nor scikit-learn has).

    cd python && python3 -m mojolearn.tests.test_transductive_predict

It gates the WIRING and the RULE on small fixtures built so the rule's
corners are hit on purpose: a query exactly at eps, a query one ulp past
eps, a query equidistant from two clusters (the tie rule), duplicated rows,
far points, float64 queries, rows alone against the batch, reversed query
order, save and load, and every refusal. It asserts the training-row
properties the rule guarantees (DBSCAN core rows, DBSCAN noise rows on
algorithm='brute', every agglomerative row without a distance-0 twin in
another cluster) and REPORTS the border rows where the rule does not
promise the fitted label.

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
from mojolearn.tests._expose_d_harness import Report, bind_or_exit, mode, run


def _identical(rep, arm, cond, what):
    if mode() == "identical":
        return rep.check(arm, cond, what)
    return rep.report_only(arm, cond, what)


def _grid_clusters():
    """Two integer-grid clusters with cluster B's rows FIRST, so the lowest
    core index belongs to the cluster with the higher x (the tie rule is
    then visible), and a few isolated points that are noise at eps=1."""
    b = [(4.0 + i, float(j)) for i in range(3) for j in range(3)]
    a = [(-4.0 + i, float(j)) for i in range(3) for j in range(3)]
    noise = [(0.0, 10.0), (20.0, -20.0)]
    return np.ascontiguousarray(np.asarray(b + a + noise, dtype=np.float32))


def arm_dbscan(rep):
    x = _grid_clusters()
    plain = DBSCAN(eps=1.0, min_samples=4, algorithm="brute").fit(x)
    m = DBSCAN(eps=1.0, min_samples=4, algorithm="brute", prediction_data=True).fit(x)
    fitted = np.asarray(m.labels_)
    rep.check("DBSCAN", np.array_equal(np.asarray(plain.labels_), fitted),
              "prediction_data=True fits the same labels_ bytes as the default")
    rep.check("DBSCAN", not hasattr(plain, "components_") and not hasattr(plain, "core_sample_indices_"),
              "the default fit stores no prediction data")
    idx = np.asarray(m.core_sample_indices_)
    comps = np.asarray(m.components_)
    rep.check("DBSCAN", idx.dtype == np.int32 and comps.dtype == np.float32 and comps.shape == (idx.size, 2),
              "core_sample_indices_ int32 and components_ float32 (n_core, d)", (idx.dtype, comps.shape))
    rep.check("DBSCAN", np.array_equal(comps, x[idx]), "components_ are the training rows at core_sample_indices_")
    rep.check("DBSCAN", bool(np.all(np.diff(idx) > 0)), "core_sample_indices_ ascending")
    rep.check("DBSCAN", len(set(fitted[fitted >= 0].tolist())) == 2 and int(np.sum(fitted == -1)) == 2,
              "the fixture fits two clusters and two noise points", fitted.tolist())

    p = np.asarray(m.predict(x))
    rep.check("DBSCAN", p.dtype == np.int32 and p.shape == (x.shape[0],), "predict returns int32 (n,)")
    rep.check("DBSCAN", np.array_equal(p[idx], fitted[idx]), "every core training row predicts its fitted label")
    noise = fitted == -1
    rep.check("DBSCAN", np.array_equal(p[noise], fitted[noise]), "every noise training row predicts -1 (algorithm='brute')")
    border = ~noise & ~np.isin(np.arange(x.shape[0]), idx)
    rep.report_only("DBSCAN", np.array_equal(p[border], fitted[border]),
                    f"{int(border.sum())} border rows predict their fitted label (not promised by the rule)")

    label_b = int(fitted[0])
    label_a = int(fitted[9])
    at_eps = np.asarray([[7.0, 1.0], [-5.0, 1.0]], dtype=np.float32)
    past = np.asarray([[np.nextafter(np.float32(7.0), np.float32(8.0)), 1.0]], dtype=np.float32)
    rep.check("DBSCAN", np.array_equal(np.asarray(m.predict(at_eps)), [label_b, label_a]),
              "a query exactly at eps from a core row takes that row's cluster (acc <= thresh)")
    rep.check("DBSCAN", int(np.asarray(m.predict(past))[0]) == -1, "a query one ulp past eps is noise")
    # The tie by construction: A spans x in [-4, -2], B x in [4, 6]; at
    # eps=3 every grid row is core and (1, 1) is at squared distance 9 from
    # B's (4, 1) (training index 1) and A's (-2, 1) (index 16), and at least
    # 10 from every other row.
    mid = np.asarray([[1.0, 1.0]], dtype=np.float32)
    wide = DBSCAN(eps=3.0, min_samples=4, algorithm="brute", prediction_data=True).fit(x)
    wl = np.asarray(wide.labels_)
    wcore = set(np.asarray(wide.core_sample_indices_).tolist())
    got = int(np.asarray(wide.predict(mid))[0])
    rep.check("DBSCAN", {1, 16} <= wcore and wl[1] != wl[16] and got == int(wl[1]),
              "an exact tie between two clusters goes to the lowest (distance, core index)", (got, int(wl[1]), int(wl[16])))

    far = np.asarray([[1000.0, -1000.0], [-500.0, 700.0]], dtype=np.float32)
    rep.check("DBSCAN", np.array_equal(np.asarray(m.predict(far)), [-1, -1]), "far points are noise")
    q64 = x[:7].astype(np.float64)
    rep.check("DBSCAN", np.array_equal(np.asarray(m.predict(q64)), p[:7]), "a float64 query returns the float32 query's labels")
    alone = [int(np.asarray(m.predict(x[i:i + 1]))[0]) for i in range(x.shape[0])]
    rev = np.asarray(m.predict(x[::-1].copy()))
    _identical(rep, "DBSCAN", alone == p.tolist() and np.array_equal(rev[::-1], p),
               "rows alone and reversed query order give the batch's bytes")

    dup = np.concatenate([x, x]).astype(np.float32)
    dm = DBSCAN(eps=1.0, min_samples=4, algorithm="brute", prediction_data=True).fit(dup)
    dp = np.asarray(dm.predict(dup))
    didx = np.asarray(dm.core_sample_indices_)
    rep.check("DBSCAN", np.array_equal(dp[didx], np.asarray(dm.labels_)[didx]),
              "duplicated rows (distance-0 ties): core rows still predict their fitted label")

    rbc = DBSCAN(eps=1.0, min_samples=4, prediction_data=True).fit(x)
    ridx = np.asarray(rbc.core_sample_indices_)
    rp = np.asarray(rbc.predict(x))
    rep.check("DBSCAN", np.array_equal(rp[ridx], np.asarray(rbc.labels_)[ridx]), "algorithm='rbc': core rows predict their fitted label")

    l1 = DBSCAN(eps=1.0, min_samples=4, metric="manhattan", algorithm="brute", prediction_data=True).fit(x)
    # (6.5, 1.5) is at L1 distance exactly 1 from the core row (6, 1);
    # (6.75, 1.75) is 1.5 from it and 1 only from the BORDER row (6, 2).
    lp = np.asarray(l1.predict(np.asarray([[7.0, 1.0], [6.5, 1.5], [6.75, 1.75]], dtype=np.float32)))
    rep.check("DBSCAN", lp[0] == l1.labels_[0] and lp[1] == l1.labels_[0] and lp[2] == -1,
              "metric='manhattan': the L1 sum against Float32(eps), at eps inside and past it outside", lp.tolist())

    w = np.ones(x.shape[0], dtype=np.float32)
    wm = DBSCAN(eps=1.0, min_samples=4, algorithm="brute", prediction_data=True).fit(x, sample_weight=w)
    rep.check("DBSCAN", np.array_equal(np.asarray(wm.predict(x)), p), "uniform sample_weight predicts the unweighted labels")

    empty_core = DBSCAN(eps=0.1, min_samples=50, prediction_data=True).fit(x)
    rep.check("DBSCAN", np.asarray(empty_core.core_sample_indices_).size == 0
              and np.array_equal(np.asarray(empty_core.predict(x[:3])), [-1, -1, -1]),
              "a fit with no core sample predicts noise")

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
    rep.raises("DBSCAN", ValueError, "no rows", "an empty query refused by name", m.predict, x[:0])
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
    mid = np.asarray([[1.0, 1.0]], dtype=np.float32)
    got = int(np.asarray(m.predict(mid))[0])
    rep.check("AGGLOM", fitted[1] != fitted[16] and got == int(min(fitted[1], fitted[16])),
              "a query equidistant from two clusters goes to the lowest cluster label", (got, int(fitted[1]), int(fitted[16])))
    near_a = np.asarray([[-1.0, 1.0], [3.5, 0.0]], dtype=np.float32)
    rep.check("AGGLOM", np.array_equal(np.asarray(m.predict(near_a)), [fitted[16], fitted[0]]),
              "a query takes the cluster of its nearest training row")

    dup = np.concatenate([x, x]).astype(np.float32)
    dm = AgglomerativeClustering(n_clusters=2, prediction_data=True).fit(dup)
    rep.check("AGGLOM", np.array_equal(np.asarray(dm.predict(dup)), np.asarray(dm.labels_)),
              "duplicated rows: every row still predicts its fitted label")
    many = AgglomerativeClustering(n_clusters=17, prediction_data=True).fit(x)
    rep.check("AGGLOM", np.array_equal(np.asarray(many.predict(x)), np.asarray(many.labels_)),
              "n_clusters=17 of 18 distinct rows (singletons): every row predicts its fitted label")

    alone = [int(np.asarray(m.predict(x[i:i + 1]))[0]) for i in range(x.shape[0])]
    _identical(rep, "AGGLOM", alone == p.tolist(), "rows alone give the batch's bytes")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "agg.npz")
        m.save(path)
        back = AgglomerativeClustering.load(path)
        rep.check("AGGLOM", np.array_equal(np.asarray(back.predict(x)), p)
                  and np.array_equal(np.asarray(back.children_), np.asarray(m.children_)),
                  "save and load predict the same bytes and keep children_")
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
