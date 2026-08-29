#!/usr/bin/env python3
"""identity_break: TRY TO BREAK cross-vendor bit-identity, on every estimator.

`tools/repeat_run_stability.py` asks one lane one question on one benign
fixture. This asks every public estimator EIGHT hostile questions, under the
`identical` tier, and writes a per-cell fingerprint so three vendors' runs
can be diffed cell by cell:

    ties      integer-grid data, every distance and split value tied many ways
    hashed    values derived from a hash, no structure, no ties, no symmetry
    wide      column magnitudes spanning 1e-4 .. 1e4 (accumulation order bites)
    denormal  a slice of the data sits below float32 normal (flush-to-zero)
    denormal_ftz  the same with subnormals flushed: the diagnostic twin
    dupes     duplicated rows, a constant column, an all-zero column
    odd       n = 12345, d = 17 -- nothing a block, warp or tile divides evenly
    negative  every value negative, shifted off the origin

    plus `base`, the stability harness's own fixture, as the control.

Every cell is fitted TWICE in one process so a run-to-run mover on this box
is separated from a cross-vendor divergence. A lane that raises reports
REFUSED with the message; nothing is swallowed.

    MOJOLEARN_NUMERIC_MODE=identical python3 tools/identity_break.py --json apple.json
    python3 tools/identity_break.py --diff apple.json nvidia.json amd.json

A `--diff` prints, per lane, per fixture: IDENTICAL (all columns agree),
DIVERGENT (they do not), MOVED (a column disagreed with itself), or the
refusal. It exits non-zero on any DIVERGENT or MOVED cell. FAST is refused
on purpose: a bitwise question to a FAST arm is a category error
(fast-is-not-identical).
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback

import numpy as np


def _h(*arrays):
    m = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a))
        m.update(str(a.dtype).encode())
        m.update(str(a.shape).encode())
        m.update(a.tobytes())
    return m.hexdigest()[:16]


# ---------------------------------------------------------------- fixtures
# Small on purpose: the Mac side of this runs under the no-heavy-local-compute
# rule and the point is the ARITHMETIC PATH, which n=20000 already exercises
# multi-block. Sizes are fixed so a hash means the same thing on every box.

N, D = 20000, 16


def _hashed_uniform(n, d, seed):
    """Values from sha256 of the index, so no RNG-family assumption and no
    repeated structure; per-cell distinct with overwhelming probability."""
    out = np.empty(n * d, dtype=np.float32)
    ctr = np.arange(n * d, dtype=np.uint64)
    # vectorised: hash 8-byte blocks in chunks
    raw = ctr.tobytes()
    dig = hashlib.sha256(raw + str(seed).encode()).digest()
    # expand deterministically with a counter-mode stream
    stream = bytearray()
    blk = 0
    while len(stream) < n * d * 4:
        stream += hashlib.sha256(dig + blk.to_bytes(8, "little")).digest()
        blk += 1
    u32 = np.frombuffer(bytes(stream[: n * d * 4]), dtype=np.uint32)
    out[:] = (u32.astype(np.float64) / 2**32).astype(np.float32)
    return out.reshape(n, d)


def fixture(kind, n=N, d=D, seed=0):
    rng = np.random.default_rng(seed)
    if kind == "base":
        X = rng.standard_normal((n, d)).astype(np.float32)
    elif kind == "ties":
        X = rng.integers(0, 6, size=(n, d)).astype(np.float32)
    elif kind == "hashed":
        X = (_hashed_uniform(n, d, seed) * 4.0 - 2.0).astype(np.float32)
    elif kind == "wide":
        X = rng.standard_normal((n, d)).astype(np.float32)
        scale = np.logspace(-4, 4, d).astype(np.float32)
        X = (X * scale).astype(np.float32)
    elif kind == "denormal":
        X = rng.standard_normal((n, d)).astype(np.float32)
        # a quarter of the rows, three columns, pushed into the subnormal range
        X[: n // 4, :3] = (X[: n // 4, :3] * np.float32(1e-40)).astype(np.float32)
        assert np.any((X != 0) & (np.abs(X) < np.finfo(np.float32).tiny))
    elif kind == "denormal_ftz":
        # THE DIAGNOSTIC TWIN of `denormal`: the same bytes with every
        # subnormal replaced by a signed zero, which is what a flush-to-zero
        # backend sees. A vendor whose `denormal` cell equals another vendor's
        # `denormal_ftz` cell is flushing where the other is not.
        X, _, _ = fixture("denormal", n, d, seed)
        sub = (X != 0) & (np.abs(X) < np.finfo(np.float32).tiny)
        X = X.copy()
        X[sub] = np.copysign(np.float32(0.0), X[sub])
    elif kind == "dupes":
        X = rng.standard_normal((n, d)).astype(np.float32)
        X[n // 2:] = X[: n - n // 2]          # second half duplicates the first
        X[:, d - 2] = np.float32(3.5)         # a constant column
        X[:, d - 1] = np.float32(0.0)         # an all-zero column
    elif kind == "odd":
        n, d = 12345, 17
        X = rng.standard_normal((n, d)).astype(np.float32)
    elif kind == "negative":
        X = (-np.abs(rng.standard_normal((n, d))) - 2.0).astype(np.float32)
    else:
        raise ValueError(kind)
    # targets: a signed rule on two columns, and a linear regression target
    # Labels come from columns 3 and 4, which NO fixture perturbs (denormal
    # rewrites columns 0-2), so `denormal` and `denormal_ftz` hand every lane
    # the same labels and their twin comparison is about the features alone.
    # Until 2026-08-29 this read columns 0 and 1 and the twins carried 2493
    # different labels, which made the classifier twins uncomparable.
    s01 = X[:, 3] + 0.5 * X[:, 4]
    y_clf = (s01 > np.median(s01)).astype(np.int32)
    w = np.random.default_rng(seed + 1).standard_normal(X.shape[1]).astype(np.float32)
    # FIXED-ORDER, ELEMENTWISE, NO BLAS. `X @ w` in float32 goes through the
    # host BLAS (Accelerate on the Mac, OpenBLAS on a Linux box) and the
    # accumulation order differs between them: measured 2026-08-29, 13876 of
    # 20000 targets differed on ONE Mac between `X @ w` and this loop, and
    # every regression lane read DIVERGENT Apple-vs-AMD for that reason alone.
    # A cross-vendor probe must hand every vendor the same bytes.
    y_reg = np.zeros(X.shape[0], dtype=np.float32)
    for j in range(X.shape[1]):
        y_reg = (y_reg + X[:, j] * w[j]).astype(np.float32)
    return X, y_clf, y_reg


FIXTURES = ["base", "ties", "hashed", "wide", "denormal", "denormal_ftz", "dupes", "odd", "negative"]

LANES = {}


def lane(name):
    def deco(fn):
        LANES[name] = fn
        return fn
    return deco


# ---------------------------------------------------------------- lanes
# One per PUBLIC estimator, plus linalg and metrics. Each returns a hash of
# the things a user would read back.

@lane("rf-clf")
def _(ml, X, yc, yr):
    m = ml.RandomForestClassifier(n_estimators=16, max_depth=8, random_state=7).fit(X, yc)
    return dict(predict=_h(m.predict(X)), proba=_h(m.predict_proba(X)))


@lane("rf-reg")
def _(ml, X, yc, yr):
    m = ml.RandomForestRegressor(n_estimators=16, max_depth=8, random_state=7).fit(X, yr)
    return dict(predict=_h(m.predict(X)))


@lane("et-clf")
def _(ml, X, yc, yr):
    m = ml.ExtraTreesClassifier(n_estimators=16, max_depth=8, random_state=7).fit(X, yc)
    return dict(predict=_h(m.predict(X)), proba=_h(m.predict_proba(X)))


@lane("et-reg")
def _(ml, X, yc, yr):
    m = ml.ExtraTreesRegressor(n_estimators=16, max_depth=8, random_state=7).fit(X, yr)
    return dict(predict=_h(m.predict(X)))


@lane("gbdt-symmetric")
def _(ml, X, yc, yr):
    m = ml.GradientBoosting(n_estimators=20, max_depth=6, loss="Logloss").fit(X, yc)
    return dict(predict=_h(m.predict(X)), proba=_h(m.predict_proba(X)))


@lane("gbdt-depthwise")
def _(ml, X, yc, yr):
    m = ml.GradientBoosting(n_estimators=20, max_depth=6, grow_policy="Depthwise",
                            loss="Logloss").fit(X, yc)
    return dict(predict=_h(m.predict(X)))


@lane("gbdt-lossguide")
def _(ml, X, yc, yr):
    m = ml.GradientBoosting(n_estimators=20, max_leaves=32, grow_policy="Lossguide",
                            loss="Logloss").fit(X, yc)
    return dict(predict=_h(m.predict(X)))


@lane("gbdt-rmse")
def _(ml, X, yc, yr):
    m = ml.GradientBoosting(n_estimators=20, max_depth=6, loss="RMSE").fit(X, yr)
    return dict(predict=_h(m.predict(X)))


@lane("kmeans")
def _(ml, X, yc, yr):
    m = ml.KMeans(n_clusters=8, random_state=3).fit(X)
    return dict(centers=_h(m.cluster_centers_), labels=_h(m.labels_))


@lane("knn")
def _(ml, X, yc, yr):
    m = ml.NearestNeighbors(n_neighbors=8).fit(X[:4096])
    d, i = m.kneighbors(X[4096:4160])
    return dict(dist=_h(d), idx=_h(i))


@lane("knn-clf")
def _(ml, X, yc, yr):
    m = ml.KNeighborsClassifier(n_neighbors=8).fit(X[:4096], yc[:4096])
    return dict(predict=_h(m.predict(X[4096:4160])), proba=_h(m.predict_proba(X[4096:4160])))


@lane("knn-reg")
def _(ml, X, yc, yr):
    m = ml.KNeighborsRegressor(n_neighbors=8).fit(X[:4096], yr[:4096])
    return dict(predict=_h(m.predict(X[4096:4160])))


@lane("dbscan")
def _(ml, X, yc, yr):
    m = ml.DBSCAN(eps=0.9, min_samples=5).fit(X[:6000, :4])
    return dict(labels=_h(m.labels_))


@lane("pca")
def _(ml, X, yc, yr):
    m = ml.PCA(n_components=4).fit(X)
    return dict(components=_h(m.components_), variance=_h(m.explained_variance_), transform=_h(m.transform(X[:256])))


@lane("tsvd")
def _(ml, X, yc, yr):
    m = ml.TruncatedSVD(n_components=4).fit(X)
    return dict(components=_h(m.components_), transform=_h(m.transform(X[:256])))


@lane("ols")
def _(ml, X, yc, yr):
    m = ml.LinearRegression().fit(X, yr)
    return dict(coef=_h(m.coef_), predict=_h(m.predict(X[:256])))


@lane("ridge")
def _(ml, X, yc, yr):
    m = ml.Ridge(alpha=1.0).fit(X, yr)
    return dict(coef=_h(m.coef_), predict=_h(m.predict(X[:256])))


@lane("logistic")
def _(ml, X, yc, yr):
    m = ml.LogisticRegression(max_iter=50).fit(X, yc)
    return dict(coef=_h(m.coef_), proba=_h(m.predict_proba(X[:256])))


@lane("lasso")
def _(ml, X, yc, yr):
    m = ml.Lasso(alpha=0.01, max_iter=200).fit(X, yr)
    return dict(coef=_h(m.coef_), predict=_h(m.predict(X[:256])))


@lane("elasticnet")
def _(ml, X, yc, yr):
    m = ml.ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=200).fit(X, yr)
    return dict(coef=_h(m.coef_), predict=_h(m.predict(X[:256])))


@lane("svc")
def _(ml, X, yc, yr):
    m = ml.SVC(C=1.0, kernel="rbf", max_iter=200).fit(X[:2000], yc[:2000])
    return dict(decision=_h(m.decision_function(X[2000:2256])), predict=_h(m.predict(X[2000:2256])))


@lane("kde")
def _(ml, X, yc, yr):
    m = ml.KernelDensity(bandwidth=0.7).fit(X[:4096, :4])
    return dict(scores=_h(m.score_samples(X[4096:4352, :4])))


@lane("agglomerative")
def _(ml, X, yc, yr):
    m = ml.AgglomerativeClustering(n_clusters=4).fit(X[:2000, :4])
    return dict(labels=_h(m.labels_))


@lane("spectral")
def _(ml, X, yc, yr):
    m = ml.SpectralClustering(n_clusters=4, random_state=3).fit(X[:2000, :4])
    return dict(labels=_h(m.labels_))


@lane("holtwinters")
def _(ml, X, yc, yr):
    series = (np.cumsum(X[:512, 0]) + 50.0).astype(np.float32)
    series = series - series.min() + 1.0     # positive, for the multiplicative path
    m = ml.ExponentialSmoothing(series, seasonal="additive", seasonal_periods=12).fit()
    return dict(forecast=_h(m.forecast(24)))


@lane("gemm-pinned")
def _(ml, X, yc, yr):
    a = X[:256].T.copy()                     # 16 x 256 or 17 x 256
    b = X[256:256 + 128].copy()              # 128 x d
    wide = np.ascontiguousarray(X[:4096, :].reshape(-1)[: 256 * 4096].reshape(256, 4096)) \
        if X.size >= 256 * 4096 else X[:256, :]
    c1 = ml.linalg.matmul(a.astype(np.float32), X[:256].astype(np.float32), identical=True)
    c2 = ml.linalg.matmul(wide.astype(np.float32),
                          np.ascontiguousarray(wide[:128].T).astype(np.float32),
                          identical=True) if wide.shape[1] == 4096 else c1
    return dict(small=_h(c1), wide=_h(c2))


@lane("metrics")
def _(ml, X, yc, yr):
    labels = ml.KMeans(n_clusters=4, random_state=3).fit(X[:3000, :4]).labels_
    mt = ml.metrics
    return dict(
        accuracy=_h(np.float64(mt.accuracy_score(yc[:3000], (labels % 2).astype(np.int32)))),
        ari=_h(np.float64(mt.adjusted_rand_score(yc[:3000], labels))),
        vmeasure=_h(np.float64(mt.v_measure_score(yc[:3000], labels))),
        r2=_h(np.float64(mt.r2_score(yr[:3000], yr[:3000] * np.float32(0.9)))),
        silhouette=_h(np.float64(mt.silhouette_score(X[:3000, :4], labels))),
    )


# LAST ON PURPOSE (2026-08-29): the isolation forest binding HANGS on a RunPod
# RTX 4090 (driver 570 + the CUDA 12.4 ptxas escape) in every tier, fit never
# returns, GPU at 0%. A hang cannot be caught from inside this process, so the
# lane runs last, the JSON is written after every lane, and the caller wraps
# the run in `timeout`; a killed run still leaves every earlier lane on disk.
@lane("iforest")
def _(ml, X, yc, yr):
    m = ml.IsolationForest(n_estimators=16, random_state=5).fit(X)
    return dict(scores=_h(m.score_samples(X)), predict=_h(m.predict(X[:512])))


# ---------------------------------------------------------------- run / diff

def run(args):
    import mojolearn as ml
    mode = ml.numeric_mode()
    want = os.environ.get("MOJOLEARN_NUMERIC_MODE", "fast").strip().lower() or "fast"
    if mode != want:
        raise SystemExit(f"REFUSING TO MEASURE: asked for {want!r}, loaded {mode!r}")
    if mode == "fast" and not args.allow_fast:
        raise SystemExit("REFUSING: this is a bitwise question and FAST makes no "
                         "bitwise promise; use MOJOLEARN_NUMERIC_MODE=identical")

    lanes = [n for n in LANES if not args.lanes or n in args.lanes.split(",")]
    skip = set(x for x in args.skip.split(",") if x)
    lanes = [n for n in lanes if n not in skip]
    if skip:
        print(f"# SKIPPED on request: {sorted(skip)}")
    fixtures = [f for f in FIXTURES if not args.fixtures or f in args.fixtures.split(",")]
    data = {f: fixture(f) for f in fixtures}
    # THE FIXTURE IS PART OF THE RESULT. Every vendor must be handed the same
    # bytes, and --diff refuses to compare columns whose fixtures differ.
    fixture_hashes = {f: dict(X=_h(X), y_clf=_h(yc), y_reg=_h(yr))
                      for f, (X, yc, yr) in data.items()}

    print(f"# identity_break  mode={mode}  {time.strftime('%Y-%m-%d %H:%M:%S')}  "
          f"{platform.platform()}")
    print(f"| {'lane':<16} | " + " | ".join(f"{f:<16}" for f in fixtures) + " |")
    print(f"|{'-'*18}|" + "|".join("-" * 18 for _ in fixtures) + "|")
    cells = {}
    for name in lanes:
        row = []
        for f in fixtures:
            X, yc, yr = data[f]
            hs, parts, err = [], [], None
            for _ in range(args.repeats):
                try:
                    p = LANES[name](ml, X, yc, yr)
                    parts.append(p)
                    hs.append(_h(np.frombuffer("|".join(f"{k}={v}" for k, v in sorted(p.items())).encode(), dtype=np.uint8)))
                except Exception as exc:
                    err = f"{type(exc).__name__}: {exc}"
                    if args.verbose:
                        traceback.print_exc()
                    break
            if err:
                cell = dict(verdict="REFUSED", error=err[:300], hashes=hs, parts=parts)
                shown = "REFUSED"
            elif len(set(hs)) == 1:
                cell = dict(verdict="STABLE", hashes=hs, parts=parts)
                shown = hs[0]
            else:
                cell = dict(verdict="MOVED", hashes=hs, parts=parts)
                shown = "MOVED " + hs[0][:8]
            cells[f"{name}/{f}"] = cell
            row.append(f"{shown:<16}")
        print(f"| {name:<16} | " + " | ".join(row) + " |", flush=True)
        if args.json:
            # written after EVERY lane so a hang that gets killed by the
            # caller's timeout still leaves the finished lanes on disk
            with open(args.json, "w") as fh:
                json.dump(dict(mode=mode, repeats=args.repeats, platform=platform.platform(),
                               vendor=args.vendor, commit=os.environ.get("MOJOLEARN_COMMIT", ""),
                               fixtures=fixture_hashes, cells=cells, complete=False,
                               skipped=sorted(skip)), fh, indent=1)

    refused = {k: v["error"] for k, v in cells.items() if v["verdict"] == "REFUSED"}
    moved = [k for k, v in cells.items() if v["verdict"] == "MOVED"]
    print()
    print(f"cells={len(cells)} stable={len(cells)-len(refused)-len(moved)} "
          f"moved={len(moved)} refused={len(refused)}")
    for k in moved:
        print(f"MOVED   {k}: {cells[k]['hashes']}")
    for k, e in refused.items():
        print(f"REFUSED {k}: {e}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(dict(mode=mode, repeats=args.repeats, platform=platform.platform(),
                           vendor=args.vendor, commit=os.environ.get("MOJOLEARN_COMMIT", ""),
                           fixtures=fixture_hashes, cells=cells, complete=True,
                           skipped=sorted(skip)), fh, indent=1)
        print(f"wrote {args.json}")
    return 1 if moved else 0


def diff(paths):
    cols = []
    for p in paths:
        with open(p) as fh:
            j = json.load(fh)
        cols.append((j.get("vendor") or os.path.basename(p), j))
    keys = sorted(set(k for _, j in cols for k in j["cells"]))
    names = [c for c, _ in cols]
    for n, (_, j) in zip(names, cols):
        if not j.get("complete", True):
            print(f"NOTE: column {n} is INCOMPLETE (the run was killed); lanes after the last one written are absent, not clean")
        if j.get("skipped"):
            print(f"NOTE: column {n} SKIPPED {j['skipped']} on request")
    # Refuse to blame the library for a fixture the vendors did not share.
    fx = [j.get("fixtures") for _, j in cols]
    if all(fx):
        for f in sorted(set(k for x in fx for k in x)):
            vals = [json.dumps(x.get(f), sort_keys=True) for x in fx]
            if len(set(vals)) != 1:
                print(f"FIXTURE MISMATCH on {f!r}: the columns were not handed the same bytes; "
                      f"cells on it are the fixture's divergence, not the library's")
                for n, x in zip(names, fx):
                    print(f"    {n}: {x.get(f)}")
    else:
        print("NOTE: a column carries no fixture hashes (older harness); fixture equality unchecked")
    print(f"| {'lane/fixture':<28} | {'verdict':<10} | " + " | ".join(f"{n:<16}" for n in names) + " |")
    print(f"|{'-'*30}|{'-'*12}|" + "|".join("-" * 18 for _ in names) + "|")
    bad, counts = 0, {}
    for k in keys:
        vals, shown = [], []
        for _, j in cols:
            c = j["cells"].get(k)
            if c is None:
                shown.append("(not run)"); continue
            if c["verdict"] == "REFUSED":
                shown.append("REFUSED"); continue
            if c["verdict"] == "MOVED":
                shown.append("MOVED"); vals.append("MOVED"); continue
            shown.append(c["hashes"][0]); vals.append(c["hashes"][0])
        ran = [v for v in vals]
        if "MOVED" in ran:
            verdict = "MOVED"
        elif len(ran) < 2:
            verdict = "ONE-COLUMN" if len(ran) == 1 else "REFUSED"
        elif len(set(ran)) == 1:
            verdict = f"IDENTICAL x{len(ran)}"
        else:
            verdict = "DIVERGENT"
        if verdict in ("MOVED", "DIVERGENT"):
            bad += 1
        if verdict == "DIVERGENT":
            # localise: which named part disagrees
            per = {}
            for n, (_, j) in zip(names, cols):
                c = j["cells"].get(k)
                if c and c.get("parts"):
                    for pk, pv in c["parts"][0].items():
                        per.setdefault(pk, []).append(pv)
            diverging = [pk for pk, pv in per.items() if len(set(pv)) > 1]
            agreeing = [pk for pk, pv in per.items() if len(set(pv)) == 1]
            if per:
                shown[0] = f"parts differ: {','.join(diverging) or '?'}; agree: {','.join(agreeing) or '-'}"
        counts[verdict.split(" ")[0]] = counts.get(verdict.split(" ")[0], 0) + 1
        print(f"| {k:<28} | {verdict:<10} | " + " | ".join(f"{s:<16}" for s in shown) + " |")
    print()
    print("summary: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--lanes", default="")
    ap.add_argument("--skip", default="", help="lanes to leave out, comma separated; each is reported as SKIPPED")
    ap.add_argument("--fixtures", default="")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--vendor", default=platform.machine())
    ap.add_argument("--allow-fast", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--diff", nargs="+", default=None, metavar="JSON")
    args = ap.parse_args()
    if args.diff:
        return diff(args.diff)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
