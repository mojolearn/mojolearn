#!/usr/bin/env python3
"""Unscored real-estimator captures. Run only on M3 with verified static arms.

No build, install, fit timing, threshold tuning, or promotion. Pair orchestration
is deliberately left to the serial manager. See shared-gemm-downstream.md.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

FAMILIES = {"kmeans": "_mojolearn", "knn": "_mojolearn",
            "ols": "_mojolearn_estimators", "ridge": "_mojolearn_estimators",
            "pca": "_mojolearn_estimators", "kde": "_mojolearn_estimators",
            "svc": "_mojolearn_svm", "rbf": "_mojolearn_kernel_methods"}
POLICY = "shared-downstream-f64-independent-errors-v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def fixture(d):
    import numpy as np
    rng = np.random.default_rng(724190)
    x = rng.normal(size=(513, d)).astype("float32")
    q = rng.normal(size=(73, d)).astype("float32")
    w = rng.normal(size=d)
    y = (x.astype("float64") @ w + .2 * rng.normal(size=len(x))).astype("float32")
    return x, q, y


def dump(args):
    import numpy as np
    assert os.environ.get("MOJOLEARN_NUMERIC_MODE") == "fast"
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert source == args.source, "source HEAD mismatch"
    assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip(), "dirty tracked source"
    from mojolearn import _backend
    b = _backend.binding(FAMILIES[args.case], "fast")
    binary = str(Path(b.__file__).resolve())
    assert sha(binary) == args.binary_sha, "actual loaded estimator binding hash mismatch"
    assert b.shared_gemm_mode() == 0 and b.shared_gemm_vendor() == "metal"
    assert b.shared_gemm_variant() == args.variant
    x, q, y = fixture(args.features)
    arrays = {"x": x, "q": q, "y": y}
    phases = {}

    def phase(name, operation):
        b.shared_gemm_reset()
        value = operation()
        phases[name] = [[int(b.shared_gemm_count(r, c)) for c in range(4)] for r in range(4)]
        return value

    def save(name, value):
        arrays[name] = np.array(value, copy=True, order="C")

    if args.case == "kmeans":
        from mojolearn.cluster import KMeans
        model = KMeans(n_clusters=7, n_init=1, random_state=7, max_iter=30)
        phase("fit", lambda: model.fit(x))
        for key in ("cluster_centers_", "labels_", "inertia_", "n_iter_"):
            save(key, getattr(model, key))
        save("prediction", phase("predict", lambda: model.predict(q)))
        save("transform", phase("transform", lambda: model.transform(q)))
    elif args.case == "knn":
        from mojolearn.neighbors import NearestNeighbors
        model = NearestNeighbors(n_neighbors=7, algorithm="brute")
        phase("fit", lambda: model.fit(x))
        distances, indices = phase("kneighbors", lambda: model.kneighbors(q))
        save("distances", distances)
        save("indices", indices)
    elif args.case in ("ols", "ridge"):
        from mojolearn.linear_model import LinearRegression, Ridge
        model = LinearRegression() if args.case == "ols" else Ridge(alpha=1.0)
        phase("fit", lambda: model.fit(x, y))
        save("coef_", model.coef_)
        save("intercept_", model.intercept_)
        save("prediction", phase("predict", lambda: model.predict(q)))
    elif args.case == "pca":
        from mojolearn.decomposition import PCA
        model = PCA(n_components=7, svd_solver="full")
        phase("fit", lambda: model.fit(x))
        for key in ("components_", "mean_", "explained_variance_", "singular_values_"):
            save(key, getattr(model, key))
        z = phase("transform", lambda: model.transform(q))
        save("transform", z)
        save("inverse", phase("inverse", lambda: model.inverse_transform(z)))
    elif args.case == "kde":
        from mojolearn.density import KernelDensity
        model = KernelDensity(bandwidth=2.0, metric="euclidean")
        phase("fit", lambda: model.fit(x))
        save("scores", phase("score_samples", lambda: model.score_samples(q)))
    elif args.case == "svc":
        from mojolearn.svm import SVC
        labels = (y >= 0).astype("int32")
        model = SVC(kernel="rbf", gamma=1.0 / args.features, C=1.0)
        phase("fit", lambda: model.fit(x, labels))
        for key in ("support_", "support_vectors_", "dual_coef_", "intercept_", "classes_", "n_support_"):
            save(key, getattr(model, key))
        save("decision", phase("decision", lambda: model.decision_function(q)))
        save("prediction", phase("predict", lambda: model.predict(q)))
    else:
        from mojolearn.kernel_methods import RBFSampler
        model = RBFSampler(gamma=1.0 / args.features, n_components=67, random_state=7)
        save("fit_transform", phase("fit_transform", lambda: model.fit_transform(x)))
        for key in ("random_weights_", "random_offset_", "scale_"):
            save(key, getattr(model, key))
        save("transform", phase("transform", lambda: model.transform(q)))
    for key, value in arrays.items():
        assert np.all(np.isfinite(value)), "nonfinite " + key
    output = Path(args.output)
    assert not output.exists() and not output.with_suffix(".json").exists()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **arrays)
    write_json(output.with_suffix(".json"), dict(policy=POLICY, source=source,
        case=args.case, features=args.features, variant=args.variant,
        binding=FAMILIES[args.case], binary=binary, binary_sha=sha(binary),
        capture_sha=sha(output), fixture_sha=hashlib.sha256(x.tobytes()+q.tobytes()+y.tobytes()).hexdigest(),
        phases=phases, mode="fast", vendor="metal", scored=False))


def compare(args):
    import numpy as np
    from scipy.special import logsumexp
    a, b = np.load(args.a, allow_pickle=False), np.load(args.b, allow_pickle=False)
    ma, mb = [json.loads(Path(p).with_suffix(".json").read_text()) for p in (args.a, args.b)]
    for field in ("policy", "source", "case", "features", "fixture_sha", "binding", "mode", "vendor"):
        assert ma[field] == mb[field], field
    assert ma["policy"] == POLICY and ma["variant"] == 0 and mb["variant"] in (1, 5)
    assert ma["capture_sha"] == sha(args.a) and mb["capture_sha"] == sha(args.b)
    assert set(a.files) == set(b.files)
    assert all(np.array_equal(a[k], b[k]) for k in ("x", "q", "y"))
    case = ma["case"]
    x, q, y = [a[k].astype("float64") for k in ("x", "q", "y")]
    exact = {}
    metrics = {}

    def same(key):
        exact[key] = a[key].dtype == b[key].dtype and a[key].shape == b[key].shape and a[key].tobytes() == b[key].tobytes()

    def metric(name, va, vb, oracle):
        oracle = np.asarray(oracle, dtype="float64")
        ea, eb = np.asarray(va, dtype="float64")-oracle, np.asarray(vb, dtype="float64")-oracle
        denom = float(np.linalg.norm(oracle.ravel())) or 1.0  # zero-oracle errors use absolute L2
        av = dict(relative_l2=float(np.linalg.norm(ea.ravel())/denom), max_absolute=float(np.max(np.abs(ea))))
        bv = dict(relative_l2=float(np.linalg.norm(eb.ravel())/denom), max_absolute=float(np.max(np.abs(eb))))
        metrics[name] = dict(A=av, B=bv, pass_no_worse=all(np.isfinite(bv[k]) and bv[k] <= av[k] for k in av))

    if case in ("ols", "ridge"):
        xc, yc = x-x.mean(0), y-y.mean()
        coef = np.linalg.lstsq(xc, yc, rcond=None)[0] if case == "ols" else np.linalg.solve(xc.T@xc+np.eye(x.shape[1]), xc.T@yc)
        intercept = y.mean()-x.mean(0)@coef
        for key, ref in (("coef_", coef), ("intercept_", intercept), ("prediction", q@coef+intercept)):
            metric(key, a[key], b[key], ref)
    elif case == "knn":
        same("indices")
        dd = np.sum((q[:, None, :]-x[None, :, :])**2, axis=2)
        indices = np.argsort(dd, axis=1, kind="stable")[:, :7]
        exact["oracle_indices_A"] = bool(np.array_equal(a["indices"], indices))
        exact["oracle_indices_B"] = bool(np.array_equal(b["indices"], indices))
        metric("distances", a["distances"], b["distances"], np.sqrt(np.take_along_axis(dd, indices, axis=1)))
    elif case == "kmeans":
        for key in ("labels_", "prediction", "n_iter_"): same(key)
        labels = a["labels_"].astype(int)
        assert len(np.unique(labels)) == 7, "empty cluster oracle unsupported"
        centers = np.stack([x[labels == k].mean(0) for k in range(7)])
        metric("centers", a["cluster_centers_"], b["cluster_centers_"], centers)
        metric("inertia", a["inertia_"], b["inertia_"], np.sum((x-centers[labels])**2))
        dist = np.sqrt(np.sum((q[:, None, :]-centers[None, :, :])**2, axis=2))
        metric("transform", a["transform"], b["transform"], dist)
        exact["oracle_prediction_B"] = bool(np.array_equal(b["prediction"], dist.argmin(1)))
    elif case == "pca":
        mean = x.mean(0)
        _, singular, vt = np.linalg.svd(x-mean, full_matrices=False)
        projection = vt[:7].T @ vt[:7]
        metric("mean", a["mean_"], b["mean_"], mean)
        metric("projector", a["components_"].astype(float).T@a["components_"], b["components_"].astype(float).T@b["components_"], projection)
        metric("variance", a["explained_variance_"], b["explained_variance_"], singular[:7]**2/(len(x)-1))
        metric("singular_values", a["singular_values_"], b["singular_values_"], singular[:7])
        metric("reconstruction", a["inverse"], b["inverse"], (q-mean)@projection+mean)
        # Algebra oracle uses each arm's saved components, avoiding arbitrary SVD signs.
        metric("transform_arithmetic", a["transform"]-(q-a["mean_"])@a["components_"].T,
               b["transform"]-(q-b["mean_"])@b["components_"].T, np.ones_like(a["transform"])*0)
    elif case == "kde":
        dd = np.sum((q[:, None, :]-x[None, :, :])**2, axis=2)
        oracle = logsumexp(-dd/8.0, axis=1)-np.log(len(x))-x.shape[1]*np.log(2*np.sqrt(2*np.pi))
        metric("scores", a["scores"], b["scores"], oracle)
    elif case == "svc":
        for key in ("support_", "support_vectors_", "dual_coef_", "intercept_", "classes_", "n_support_", "prediction"): same(key)
        sv = a["support_vectors_"].astype(float)
        kernel = np.exp(-np.sum((q[:, None, :]-sv[None, :, :])**2, axis=2)/x.shape[1])
        ref = kernel@a["dual_coef_"].reshape(-1).astype(float)+float(a["intercept_"].reshape(-1)[0])
        metric("decision", a["decision"].reshape(-1), b["decision"].reshape(-1), ref)
    else:
        for key in ("random_weights_", "random_offset_", "scale_"): same(key)
        w, offset, scale = a["random_weights_"].astype(float), a["random_offset_"].astype(float), float(a["scale_"])
        for key, inp in (("fit_transform", x), ("transform", q)):
            metric(key, a[key], b[key], scale*np.cos(inp@w+offset))
    def totals(meta, column):
        return sum(row[column] for phase in meta["phases"].values() for row in phase)
    assert totals(ma, 3) == 0, "A unexpectedly launched candidate"
    selected = 1 if mb["variant"] == 1 else 2
    assert totals(mb, 3) == totals(mb, selected), "mixed candidate counters"
    reached = totals(mb, 3) > 0
    passed = reached and all(exact.values()) and bool(metrics) and all(v["pass_no_worse"] for v in metrics.values())
    status = "PASS" if passed else ("NO_REACH" if not reached else "HOLD")
    result = dict(policy=POLICY, source=ma["source"], case=case, status=status,
                  capture_sha=[sha(args.a), sha(args.b)], binary_sha=[ma["binary_sha"], mb["binary_sha"]],
                  phases_A=ma["phases"], phases_B=mb["phases"], exact=exact, metrics=metrics, scored=False)
    assert not Path(args.report).exists()
    write_json(args.report, result)
    print(json.dumps(dict(status=status, case=case, source=ma["source"])))
    return 0 if passed else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    d = sub.add_parser("dump")
    d.add_argument("source"); d.add_argument("binary_sha")
    d.add_argument("variant", type=int, choices=(0, 1, 5)); d.add_argument("case", choices=tuple(FAMILIES))
    d.add_argument("output", help="new .npz path")
    d.add_argument("--features", type=int, choices=(11, 65, 220), default=65)
    c = sub.add_parser("compare")
    c.add_argument("a"); c.add_argument("b"); c.add_argument("report")
    args = p.parse_args()
    if args.action == "dump":
        assert args.output.endswith(".npz")
        dump(args)
        return 0
    return compare(args)

if __name__ == "__main__":
    sys.exit(main())
