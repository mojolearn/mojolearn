"""Does the same fit, twice, on the same GPU, return the same bits?

THE MEASUREMENT THE MIDDLE TIER IS MADE OF, and until 2026-08-29 nothing in
this repository took it. Every existing harness asks a CROSS-VENDOR question
(E2/E2U diff two machines' cards) or a SPEED question. `deterministic`
promises something neither of them measures: that ONE box, running ONE build,
on ONE input, returns one answer however many times you ask.

WHY IT HAS TO BE A SEPARATE TOOL. A cross-vendor card that matches proves
run-to-run stability as a side effect -- two runs that both equal a third
thing equal each other. But that inference is only available where a card
exists, it needs two machines, and it says nothing at all about FAST, which
is precisely the arm whose instability we want to price. This asks the
question directly, on one box, in every mode.

WHAT "SAME BITS" MEANS HERE. Outputs are hashed as RAW BYTES
(`ndarray.tobytes()`), not compared with `allclose`. A tolerance would hide
exactly the one-ulp movement the tier exists to remove. NaN payloads and
signed zeros therefore count as differences, which is correct: two runs that
disagree about the sign of a zero disagree.

REPEATS RUN IN ONE PROCESS, BACK TO BACK. A fresh process per repeat would
also re-roll allocator addresses and library load order, which is a stronger
test but a different one -- it would fold environment variation into a number
meant to isolate scheduling. One process, same buffers, same seeds: any
movement seen here is the device's own order.

**SEQUENTIAL REPEATS UNDER-REPORT, AND THAT IS MEASURED, NOT SUSPECTED.**
This harness's first run called `knn-tied` STABLE under FAST at 4 repeats.
It is not. The same fit, run INTERLEAVED with other GPU work in the same
process, returned three different neighbour sets in eight calls, while the
deterministic and identical tiers returned one answer all eight times. Four
quiet back-to-back repeats do not perturb the schedule: the queue is empty,
the blocks launch in the same order, and an arrival-order defect reproduces
its own previous answer. `--interleave` therefore runs a DIFFERENT lane
between every pair of repeats, which is what made the k-NN mutex merge
(ledger row 23) move. A stability number taken without it is a lower bound
on the instability, never an upper one.

    pixi run python tools/repeat_run_stability.py --repeats 5
    MOJOLEARN_NUMERIC_MODE=deterministic pixi run python tools/repeat_run_stability.py
    MOJOLEARN_NUMERIC_MODE=identical     pixi run python tools/repeat_run_stability.py

A lane that RAISES is reported REFUSED with its message, not skipped: an
upper tier narrowing what it accepts (k-NN's `k > SELECT_BLOCK` refusal now
binds `deterministic` too) is a result, and a harness that silently drops it
would report a tier as clean by not asking.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import traceback

import numpy as np


def _h(*arrays):
    """One hash over several outputs, order-significant, raw bytes."""
    m = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        m.update(str(a.dtype).encode())
        m.update(str(a.shape).encode())
        m.update(a.tobytes())
    return m.hexdigest()[:16]


# ---------------------------------------------------------------- fixtures
# SMALL AND FIXED. This is the Mac, and the standing rule is that heavy
# compute goes to rented GPUs. These sizes are large enough to engage the
# multi-block paths whose order is the thing under test (that is the point of
# n_index = 4096 for k-NN and 200k rows for the tree families) and no larger.

def _clf(n=20000, d=16, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.int32)
    return X, y


def _reg(n=20000, d=16, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (X @ rng.standard_normal(d)).astype(np.float32)
    return X, y


def _tied_knn(n_index=4096, d=8, seed=2):
    """Deliberately TIE-RICH. Ties are the whole mechanism of rows 11 and 23:
    a selector that never sees two equal distances cannot expose an
    arrival-order defect, and a uniform fixture is how that gets missed."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, 8, size=(n_index, d)).astype(np.float32)
    q = rng.integers(0, 8, size=(64, d)).astype(np.float32)
    return q, idx


LANES = {}


def lane(name):
    def deco(fn):
        LANES[name] = fn
        return fn
    return deco


@lane("rf-clf")
def _rf_clf(ml):
    X, y = _clf()
    m = ml.RandomForestClassifier(n_estimators=16, max_depth=8, random_state=7)
    m.fit(X, y)
    return _h(m.predict(X))


@lane("et-clf")
def _et_clf(ml):
    X, y = _clf()
    m = ml.ExtraTreesClassifier(n_estimators=16, max_depth=8, random_state=7)
    m.fit(X, y)
    return _h(m.predict(X))


@lane("gbdt-depthwise")
def _gbdt_dw(ml):
    X, y = _clf()
    m = ml.GradientBoosting(
        n_estimators=20, max_depth=6, grow_policy="Depthwise", loss="Logloss"
    )
    m.fit(X, y)
    return _h(m.predict(X))


@lane("gbdt-lossguide")
def _gbdt_lg(ml):
    X, y = _clf()
    m = ml.GradientBoosting(
        n_estimators=20, max_leaves=32, grow_policy="Lossguide", loss="Logloss"
    )
    m.fit(X, y)
    return _h(m.predict(X))


@lane("gbdt-symmetric")
def _gbdt_sym(ml):
    X, y = _clf()
    m = ml.GradientBoosting(n_estimators=20, max_depth=6, loss="Logloss")
    m.fit(X, y)
    return _h(m.predict(X))


@lane("kmeans")
def _kmeans(ml):
    X, _ = _clf()
    m = ml.KMeans(n_clusters=8, random_state=3)
    m.fit(X)
    return _h(m.cluster_centers_, m.labels_)


@lane("knn-tied")
def _knn(ml):
    q, idx = _tied_knn()
    m = ml.NearestNeighbors(n_neighbors=8)
    m.fit(idx)
    d, i = m.kneighbors(q)
    return _h(d, i)


@lane("dbscan")
def _dbscan(ml):
    X, _ = _clf(n=6000, d=4, seed=11)
    m = ml.DBSCAN(eps=0.9, min_samples=5)
    m.fit(X)
    return _h(m.labels_)


@lane("pca")
def _pca(ml):
    X, _ = _clf()
    m = ml.PCA(n_components=4)
    m.fit(X)
    return _h(m.components_, m.explained_variance_)


@lane("tsvd")
def _tsvd(ml):
    X, _ = _clf()
    m = ml.TruncatedSVD(n_components=4)
    m.fit(X)
    return _h(m.components_)


@lane("ols")
def _ols(ml):
    X, y = _reg()
    m = ml.LinearRegression()
    m.fit(X, y)
    return _h(np.asarray(m.coef_))


@lane("ridge")
def _ridge(ml):
    X, y = _reg()
    m = ml.Ridge(alpha=1.0)
    m.fit(X, y)
    return _h(np.asarray(m.coef_))


@lane("logistic")
def _logistic(ml):
    X, y = _clf()
    m = ml.LogisticRegression(max_iter=50)
    m.fit(X, y)
    return _h(np.asarray(m.coef_))


@lane("iforest")
def _iforest(ml):
    X, _ = _clf()
    m = ml.IsolationForest(n_estimators=16, random_state=5)
    m.fit(X)
    return _h(np.asarray(m.score_samples(X)))


# THE TWELVE LANES THAT WERE NOT HERE (2026-08-29). The deterministic tier had
# been measured on 16 lanes and the other 12 public surfaces had never been
# asked the one-box question under the middle tier's own binary. Every public
# estimator is a lane now; fixtures match tools/identity_break.py's base.

@lane("rf-reg")
def _rf_reg(ml):
    X, y = _reg()
    m = ml.RandomForestRegressor(n_estimators=16, max_depth=8, random_state=7)
    m.fit(X, y)
    return _h(m.predict(X))


@lane("et-reg")
def _et_reg(ml):
    X, y = _reg()
    m = ml.ExtraTreesRegressor(n_estimators=16, max_depth=8, random_state=7)
    m.fit(X, y)
    return _h(m.predict(X))


@lane("gbdt-rmse")
def _gbdt_rmse(ml):
    X, y = _reg()
    m = ml.GradientBoosting(n_estimators=20, max_depth=6, loss="RMSE")
    m.fit(X, y)
    return _h(m.predict(X))


@lane("knn-clf")
def _knn_clf(ml):
    X, y = _clf()
    m = ml.KNeighborsClassifier(n_neighbors=8)
    m.fit(X[:4096], y[:4096])
    return _h(m.predict(X[4096:4160]), m.predict_proba(X[4096:4160]))


@lane("knn-reg")
def _knn_reg(ml):
    X, y = _reg()
    m = ml.KNeighborsRegressor(n_neighbors=8)
    m.fit(X[:4096], y[:4096])
    return _h(m.predict(X[4096:4160]))


@lane("lasso")
def _lasso(ml):
    X, y = _reg()
    m = ml.Lasso(alpha=0.01, max_iter=200)
    m.fit(X, y)
    return _h(np.asarray(m.coef_))


@lane("elasticnet")
def _elasticnet(ml):
    X, y = _reg()
    m = ml.ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=200)
    m.fit(X, y)
    return _h(np.asarray(m.coef_))


@lane("svc")
def _svc(ml):
    X, y = _clf()
    m = ml.SVC(C=1.0, kernel="rbf", max_iter=200)
    m.fit(X[:2000], y[:2000])
    return _h(m.decision_function(X[2000:2256]))


@lane("kde")
def _kde(ml):
    X, _ = _clf()
    m = ml.KernelDensity(bandwidth=0.7)
    m.fit(X[:4096, :4])
    return _h(m.score_samples(X[4096:4352, :4]))


@lane("agglomerative")
def _agglomerative(ml):
    X, _ = _clf()
    m = ml.AgglomerativeClustering(n_clusters=4)
    m.fit(X[:2000, :4])
    return _h(m.labels_)


@lane("spectral")
def _spectral(ml):
    X, _ = _clf()
    m = ml.SpectralClustering(n_clusters=4, random_state=3)
    m.fit(X[:2000, :4])
    return _h(m.labels_)


@lane("holtwinters")
def _holtwinters(ml):
    X, _ = _clf()
    series = (np.cumsum(X[:512, 0]) + 50.0).astype(np.float32)
    series = series - series.min() + 1.0
    m = ml.ExponentialSmoothing(series, seasonal="additive", seasonal_periods=12).fit()
    return _h(m.forecast(24))


@lane("metrics")
def _metrics(ml):
    X, y = _clf()
    labels = ml.KMeans(n_clusters=4, random_state=3).fit(X[:3000, :4]).labels_
    mt = ml.metrics
    return _h(
        np.float64(mt.adjusted_rand_score(y[:3000], labels)),
        np.float64(mt.v_measure_score(y[:3000], labels)),
        np.float64(mt.silhouette_score(X[:3000, :4], labels)),
    )


@lane("gemm-vendor")
def _gemm_vendor(ml):
    """THE OPEN QUESTION, ASKED DIRECTLY.

    Ledger rows 24/27/28/40/41 all pin a CLOSED VENDOR LIBRARY (cuBLAS,
    rocBLAS, MAX `linalg.matmul`) and classify it CROSS_VENDOR, on the
    ground that a k-split is chosen per vendor and per shape but is the SAME
    on two runs of one build. That is what keeps the 4.7x and 2.85x pins out
    of the middle tier. But `hierarchy/mojo_only/linkage_check.mojo:583`
    speculates the other way -- "a split-K or atomic epilogue MAY land
    differently" between two launches of one shape.

    If that speculation is right, four more sites become BOTH and the
    deterministic tier gets much more expensive. It is a measurement, so
    measure it: one shape, run twice, bits compared. A wide k is chosen
    deliberately -- a split-K epilogue is what a wide k provokes."""
    rng = np.random.default_rng(17)
    a = rng.standard_normal((256, 4096)).astype(np.float32)
    b = rng.standard_normal((4096, 128)).astype(np.float32)
    return _h(ml.linalg.matmul(a, b, identical=False))


@lane("gemm-pinned")
def _gemm_pinned(ml):
    """The same shape through the pinned profile. REFUSED under fast, which
    is the correct answer and is reported rather than skipped."""
    rng = np.random.default_rng(17)
    a = rng.standard_normal((256, 4096)).astype(np.float32)
    b = rng.standard_normal((4096, 128)).astype(np.float32)
    return _h(ml.linalg.matmul(a, b, identical=True))


def concurrent_probe(repeats=12):
    """THE CHECK THAT ACTUALLY FOUND FAST MOVING, and the reason it is a
    separate mode rather than a flag on the loop above.

    `--interleave` runs a different LANE between repeats and did NOT move the
    k-NN tie set: one process, one device context, an empty queue between
    launches, and the blocks come back in the same order. What moved it was
    THREE TIERS LIVE AT ONCE -- three estimators, each bound to a different
    binary, each holding its own device context, called round-robin. That is
    contention between contexts, not between lanes, and it is a realistic
    condition rather than a contrived one: it is what a serving process with
    two models loaded looks like.

    Measured on an Apple M4, 2026-08-29, 12 rounds, TEN attempts:

        fast           MOVED in 8 of 10 attempts (2 or 3 different answers
                       out of 24 calls); STABLE in the other 2
        deterministic  STABLE in 10 of 10 (one answer in 12 calls)
        identical      STABLE in 10 of 10 (one answer in 12 calls)

    **THE 8-IN-10 IS THE HONEST NUMBER AND THE 2 MATTER.** A race does not
    fire on demand; it fires when the schedule happens to interleave the
    right way, and how busy the box is changes that. So a single STABLE
    reading of the fast arm is not evidence that fast is stable -- it is one
    sample from a distribution that is 80% MOVED here. Any report of this
    probe that quotes one attempt is quoting noise.

    Same fit, same input, same GPU. This is ledger row 23's mutex merge:
    which of several equidistant neighbours survives is decided by the order
    the blocks won the mutex, and under FAST nothing pins it.

    **THE ROUND COUNT MATTERS AND IS NOT A DETAIL.** At 3 rounds this probe
    reported fast STABLE and would have cleared an arm that is not. At 12 it
    caught the movement on every attempt. A race that needs N samples to
    surface will report "clean" at N-1, so a PASS from this probe is only
    ever "not caught at this sample size" -- which is why the default is 12
    and why lowering it is a decision, not a convenience.

    NEEDS ALL THREE BINARY SETS BUILT. Reports what is missing rather than
    quietly comparing two.
    """
    import mojolearn as ml
    rng = np.random.default_rng(2)
    idx = rng.integers(0, 8, size=(4096, 8)).astype(np.float32)   # tie-rich
    q = rng.integers(0, 8, size=(64, 8)).astype(np.float32)

    tiers = ("fast", "deterministic", "identical")
    ests, unavailable = {}, []
    for m in tiers:
        try:
            e = ml.NearestNeighbors(n_neighbors=8, numeric_mode=m)
            e.fit(idx)
            ests[m] = e
        except Exception as exc:
            unavailable.append(f"{m}: {type(exc).__name__}: {exc}")
    if unavailable:
        print("# tiers unavailable, NOT compared:")
        for u in unavailable:
            print("#   " + u)
    if len(ests) < 2:
        print("# fewer than two tiers available; nothing to compare")
        return 1

    print("# concurrent-context probe: one process, one GPU, one fit,")
    print("# every available tier live at the same time")
    print()
    seen = {m: [] for m in ests}
    order = list(ests) + [next(iter(ests))]
    for rnd in range(repeats):
        for m in order:
            _, i = ests[m].kneighbors(q)
            h = _h(i)
            seen[m].append(h)
            print(f"  round {rnd}  {m:14} -> {ests[m].numeric_mode_used():14} {h}")
    print()
    bad = []
    for m in ests:
        d = len(set(seen[m]))
        n = len(seen[m])
        if d == 1:
            print(f"  {m:14} STABLE  all {n} calls returned THE SAME answer")
        else:
            print(f"  {m:14} MOVED   {n} calls returned {d} DIFFERENT answers")
        if d > 1 and m != "fast":
            bad.append(m)
    print()
    if bad:
        print("FAILED: " + ", ".join(bad) + " promise run-to-run identity and moved")
        return 1
    if len(set(seen.get("fast", ["x"]))) > 1:
        print("fast MOVED, which is fast working as specified: it promises "
              "speed and makes no bitwise claim.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument(
        "--interleave", action="store_true",
        help="run a different lane between repeats, so the schedule is not "
             "identical each time; see the module docstring -- without it "
             "this harness UNDER-REPORTS non-determinism",
    )
    ap.add_argument("--lanes", default="")
    ap.add_argument("--json", default="")
    ap.add_argument(
        "--concurrent", action="store_true",
        help="hold every tier live at once and call them round-robin; this "
             "is the probe that actually caught FAST moving (see "
             "concurrent_probe)",
    )
    args = ap.parse_args()

    if args.concurrent:
        # `--repeats` defaults to 3 for the sequential loop, which is far too
        # few here (see concurrent_probe: 3 reported a false STABLE). Honour
        # an EXPLICIT --repeats and otherwise take the probe's own default.
        explicit = any(a.startswith("--repeats") for a in sys.argv[1:])
        return concurrent_probe(args.repeats if explicit else 12)

    import mojolearn as ml

    mode = ml.numeric_mode()
    want = os.environ.get("MOJOLEARN_NUMERIC_MODE", "fast").strip().lower() or "fast"
    if mode != want:
        # The failure this whole tree is disciplined against: a number
        # measured on one arm and filed under another's name.
        raise SystemExit(
            f"REFUSING TO MEASURE: asked for {want!r}, the process loaded "
            f"{mode!r}. Build that set before running this."
        )

    names = [n for n in LANES if not args.lanes or n in args.lanes.split(",")]
    print(f"# repeat-run stability, mode={mode}, repeats={args.repeats}"
          f", interleave={args.interleave}")
    print(f"# {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    # THE COLUMN IS "SAME ANSWER EVERY TIME", NOT A SCORE. An earlier version
    # printed a bare "distinct" count and it was read as a pass rate -- "1 in
    # 12" looks like 1 success out of 12 when it means the opposite: twelve
    # calls, ONE answer between them, which is a perfect result. Spell it.
    print(f"| {'lane':<16} | {'verdict':<8} | same answer every time? | hashes |")
    print(f"|{'-'*18}|{'-'*10}|-------------------------|--------|")

    rows = []
    for name in sorted(names):
        hashes, err = [], None
        # The disturber is a DIFFERENT lane, chosen once and cheap, whose job
        # is only to leave the device queue non-empty between repeats.
        others = [n for n in sorted(LANES) if n != name]
        for r in range(args.repeats):
            try:
                hashes.append(LANES[name](ml))
                if args.interleave and others:
                    try:
                        LANES[others[r % len(others)]](ml)
                    except Exception:
                        # A disturber that refuses still disturbed nothing we
                        # need; it must never mark THIS lane as refused.
                        pass
            except Exception as exc:            # REPORTED, never swallowed
                err = f"{type(exc).__name__}: {exc}"
                break
        if err is not None:
            verdict, distinct, shown = "REFUSED", 0, err[:90]
        else:
            uniq = sorted(set(hashes))
            distinct = len(uniq)
            verdict = "STABLE" if distinct == 1 else "MOVED"
            shown = " ".join(uniq[:3]) + (" ..." if distinct > 3 else "")
        if verdict == "REFUSED":
            same = "n/a (refused)"
        elif distinct == 1:
            same = f"YES  {args.repeats}/{args.repeats} calls agreed"
        else:
            same = f"NO   {distinct} different answers in {args.repeats} calls"
        print(f"| {name:<16} | {verdict:<8} | {same:<23} | {shown} |")
        rows.append(
            dict(lane=name, verdict=verdict, distinct=distinct,
                 hashes=hashes, error=err)
        )

    moved = [r["lane"] for r in rows if r["verdict"] == "MOVED"]
    refused = [r["lane"] for r in rows if r["verdict"] == "REFUSED"]
    print()
    print(f"mode={mode}  stable={sum(1 for r in rows if r['verdict']=='STABLE')}"
          f"  moved={len(moved)}  refused={len(refused)}")
    if moved:
        print("MOVED: " + ", ".join(moved))
    if refused:
        print("REFUSED: " + ", ".join(refused))
        for r in rows:
            if r["verdict"] == "REFUSED":
                print(f"  {r['lane']}: {r['error']}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(dict(mode=mode, repeats=args.repeats, rows=rows), fh, indent=1)
        print(f"wrote {args.json}")

    # A MOVED lane is NOT a failure under fast -- fast promises nothing and a
    # moving lane is the finding. It IS a failure under either upper tier.
    if mode != "fast" and moved:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
