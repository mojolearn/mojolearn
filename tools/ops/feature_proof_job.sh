#!/bin/bash
# tools/ops/feature_proof_job.sh -- FIRST REAL RUN, from the installed candidate wheel on a GPU box, of the user
# features that so far only compiled (2026-11-21): the `device=` parameter and `model.check(X)` on both devices in one
# process (Ridge, LinearRegression), `model.digest_` per estimator family, and the verify options
# `--self-test`, `--check`, `--fit-both`, `--cross-check quick`. Diagnostic tooling: it times nothing and decides no
# release. It starts with two load steps (every family binary; every GPU/host image pair) and the `device=` boundary
# test. Each step prints one line `PROOF <name> PASS|FAIL rc=<n> <seconds>s` and keeps its full output in
# logs/<name>.log; the job exits 1 when any step failed. Steps are independent and short, the GPU is one device, so
# they run one after another (a GPU step owns the device); nothing here is a loop over heavy independent units.
#
#   tools/nvidia_central.sh submit ab --cap 30 --results box-out/proof-<tag> tools/ops/feature_proof_job.sh
#
# FP_VENV (default /root/pc-cache/venv-candidate): the venv holding the candidate wheel (a paper-compare or
# call-profile job installs it). FP_OUT_RECEIPTS: where the digests and their data are left for a second machine
# (default $RES/receipts): copy that folder to another box (a CPU box, another vendor) and run
# `FP_REPLAY=<folder> feature_proof_job.sh` there: it fits again and every digest must be the recorded one. With
# FP_REPLAY set only the digest step runs. FP_RELEASE=1 also runs the verify options (section 3), which are release
# only; a round leaves it unset.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
RES="${MOJOLEARN_RESULTS_DIR:?feature_proof_job.sh needs MOJOLEARN_RESULTS_DIR (set by tools/box_persist_run.sh)}"
mkdir -p "$RES/logs" && RES="$(cd "$RES" && pwd)"
VENV="${FP_VENV:-/root/pc-cache/venv-candidate}"
PY="$VENV/bin/python"
[ -x "$PY" ] || { echo "PROOF setup FAIL no python at $PY (FP_VENV)"; exit 4; }
W="$RES/work"; mkdir -p "$W"; cd "$W" || exit 4
REC="${FP_OUT_RECEIPTS:-$RES/receipts}"; mkdir -p "$REC"
fails=0
step() { # name, command...
    local n=$1; shift; local t0=$SECONDS
    "$@" > "$RES/logs/$n.log" 2>&1; local rc=$?
    local want=${WANT_RC:-0}
    if [ $rc = "$want" ]; then echo "PROOF $n PASS rc=$rc $((SECONDS - t0))s"; else echo "PROOF $n FAIL rc=$rc (want $want) $((SECONDS - t0))s: $(grep -v '^\s*$' "$RES/logs/$n.log" | tail -n 2 | tr '\n' ' ' | cut -c1-300)"; fails=$((fails + 1)); fi
}
"$PY" -I -c "import mojolearn as ml; print('PROOF wheel', ml.__version__)" 2>&1 | tail -n 1

# D. digest_: per estimator family, fit on the GPU and with device="cpu" in ONE process; the two `digest_` strings are
#    equal. Also per family: a second read (hashed again: nothing is kept) and a second fit give the same string, a fit on changed data and a refit
#    change it, an unfitted model has no digest_, and the copy loaded from the model's saved file (by its own class and
#    by `host_model`, the CPU twin) has the digest of the model that wrote it. A family that has no digest yet prints NOT-YET with its reason
#    and does not fail the step. The digests are left in $REC/digests.json for a second machine (FP_REPLAY), with
#    the data. The families run one after another: the GPU is one device and each fit is a few milliseconds.
cat > "$W/digest.py" <<'EOF'
import json, sys, numpy as np, mojolearn as ml
d, replay = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
src = replay or d
if replay is None:
    r = np.random.default_rng(0); X = r.random((1000, 8), dtype=np.float32); y = X @ np.arange(1, 9, dtype=np.float32)
    np.save(d + "/X.npy", X); np.save(d + "/y.npy", y)
X, y = np.load(src + "/X.npy"), np.load(src + "/y.npy")
c = (y > np.median(y)).astype(np.int32)
X2 = X * np.float32(1.5) + np.float32(0.25)             # every training value changed
# (class, constructor parameters, target): one or two estimators per family; r = regression target, c = classes,
# None = no y
FAMILIES = [
    ("Ridge", dict(alpha=1.0), "r"), ("LinearRegression", {}, "r"), ("Lasso", dict(alpha=0.01), "r"),
    ("ElasticNet", dict(alpha=0.01), "r"), ("BayesianRidge", {}, "r"), ("LogisticRegression", {}, "c"),
    ("SGDClassifier", dict(random_state=0), "c"), ("LinearSVC", {}, "c"),
    ("SVC", {}, "c"), ("SVR", {}, "r"), ("KernelRidge", {}, "r"),
    ("GaussianNB", {}, "c"), ("LinearDiscriminantAnalysis", {}, "c"),
    ("KNeighborsClassifier", {}, "c"), ("KNeighborsRegressor", {}, "r"), ("NearestNeighbors", {}, None),
    ("KMeans", dict(n_clusters=4, random_state=0), None), ("MiniBatchKMeans", dict(n_clusters=4, random_state=0), None),
    ("GaussianMixture", dict(n_components=3, random_state=0), None), ("DBSCAN", {}, None),
    ("PCA", dict(n_components=3), None), ("TruncatedSVD", dict(n_components=3, random_state=0), None),
    ("StandardScaler", {}, None), ("MinMaxScaler", {}, None), ("RobustScaler", {}, None),
    ("Nystroem", dict(n_components=16, random_state=0), None), ("RBFSampler", dict(n_components=16, random_state=0), None),
    ("HDBSCAN", {}, None), ("AgglomerativeClustering", dict(n_clusters=3), None),
    ("DecisionTreeRegressor", dict(random_state=0), "r"), ("DecisionTreeClassifier", dict(random_state=0), "c"),
    ("RandomForestRegressor", dict(n_estimators=20, random_state=0), "r"),
    ("RandomForestClassifier", dict(n_estimators=20, random_state=0), "c"),
    ("ExtraTreesRegressor", dict(n_estimators=20, random_state=0), "r"),
    ("IsolationForest", dict(n_estimators=20, random_state=0), None),
    ("GradientBoostingRegressor", dict(random_state=0), "r"), ("GradientBoostingClassifier", dict(random_state=0), "c"),
    ("AdaBoostClassifier", dict(n_estimators=10, random_state=0), "c"),
    ("BaggingRegressor", dict(n_estimators=10, random_state=0), "r"),
    ("MLPRegressor", dict(hidden_layer_sizes=(16,), max_iter=20, random_state=0), "r"),
    ("MLPClassifier", dict(hidden_layer_sizes=(16,), max_iter=20, random_state=0), "c"),
    ("GaussianProcessRegressor", {}, "r"),
]


# the three estimators whose random draws continue from call to call: their digest changes with use, by decision
ADVANCES = ("IterativeImputer", "LatentDirichletAllocation", "MinCovDet")
# fitted state drawn from random_state and the input's width only, never from its values (RBFSampler: "`fit` draws random
# weights and offsets from `random_state`", python/mojolearn/kernel_methods.py): changed data of the same width fits the same model, so
# the changed fit and the refit change the seed instead
DATA_FREE = {"RBFSampler": dict(random_state=1)}


def fitted(cls, params, target, data, device=None):
    kw = dict(params)
    if device is not None:
        kw["device"] = device
    est = cls(**kw)
    args = (data,) if target is None else (data, y if target == "r" else c)
    return est.fit(*args)


recorded = json.load(open(replay + "/digests.json")) if replay else {}
out, bad = {}, []
for name, params, target in FAMILIES:
    cls = getattr(ml, name, None)
    if cls is None:
        print("DIGEST", name, "ABSENT"); continue
    try:
        try:
            cls(**params).digest_
            bad.append(name); print("DIGEST", name, "FAIL an unfitted model has a digest_"); continue
        except AttributeError:
            pass
        m = fitted(cls, params, target, X)
        try:
            g = m.digest_
        except AttributeError as e:                      # a family with no digest yet says so in words
            print("DIGEST", name, "NOT-YET", str(e)[:200]); continue
        assert isinstance(g, str) and len(g) == 64 and g == g.lower(), g
        problems = []
        if m.digest_ != g:
            problems.append("a second read differs")
        if fitted(cls, params, target, X).digest_ != g:
            problems.append("a second fit on the same data differs")
        changed = {**params, **DATA_FREE.get(name, {})}
        if fitted(cls, changed, target, X2).digest_ == g:
            problems.append("a fit on changed data has the same digest (the digest does not see the fitted state)")
        for method in ("predict", "transform"):          # using the model does not change what it is
            if hasattr(m, method):
                try:
                    getattr(m, method)(X)
                except Exception:
                    pass
        if name not in ADVANCES and m.digest_ != g:
            problems.append("a predict / transform call changed the digest (a cache or a call record is hashed)")
        if name in DATA_FREE:
            m.set_params(**DATA_FREE[name])
        m.fit(*((X2,) if target is None else (X2, y if target == "r" else c)))
        if m.digest_ == g:
            problems.append("the digest was not dropped by a refit")
        side = "gpu-only"
        if getattr(cls, "_mojolearn_device", False) and ml.vendor() != "cpu":
            side = "gpu=cpu"
            if fitted(cls, params, target, X, "cpu").digest_ != g:
                problems.append('device="cpu" gives another digest than the GPU')
        if hasattr(cls, "save") and hasattr(cls, "load"):   # one model, one digest: the saved file's copies
            path = d + "/digest-" + name + ".npz"
            try:
                fitted(cls, params, target, X).save(path)
            except Exception as e:                           # this model has no saved form (said in words)
                path = None; side += " no-file(" + type(e).__name__ + ")"
            if path:
                if cls.load(path).digest_ != g:
                    problems.append("the model loaded from its file has another digest")
                try:
                    twin = ml.host_model(path)
                except Exception as e:                       # no CPU twin for this format
                    twin = None
                if twin is not None and type(twin) is not cls:
                    try:
                        if twin.digest_ != g:
                            problems.append("the CPU twin " + type(twin).__name__ + " of the file has another digest")
                        side += " file=twin"
                    except AttributeError:
                        side += " twin-without-digest(" + type(twin).__name__ + ")"
                else:
                    side += " file"
        if replay and name in recorded and recorded[name] != g:
            problems.append(f"another machine recorded {recorded[name][:16]}, this one has {g[:16]}")
        out[name] = g
        if problems:
            bad.append(name); print("DIGEST", name, "FAIL", "; ".join(problems))
        else:
            print("DIGEST", name, "SAME", side, g[:16], "(matches the other machine)" if replay and name in recorded else "")
    except Exception as e:
        bad.append(name); print("DIGEST", name, "ERROR", type(e).__name__, str(e)[:200])
if replay is None:
    json.dump(dict(version=str(ml.__version__), vendor=str(ml.vendor())), open(d + "/digests-info.json", "w"), sort_keys=True)
    json.dump(out, open(d + "/digests.json", "w"), indent=1, sort_keys=True)
print("DIGEST SUMMARY", len(out), "with a digest;", len(bad), "failed:", " ".join(bad))
assert not bad
EOF
if [ -n "${FP_REPLAY:-}" ]; then
    # second machine (another vendor, or a CPU box): fit again here and compare with the digests the first box left
    step digest-replay "$PY" -I "$W/digest.py" "$REC" "$FP_REPLAY"
    grep -E '^DIGEST' "$RES/logs/digest-replay.log" | cut -c1-300
    # the user-algorithm recipes: fitted again here, each compared with the first box's record (--against)
    for r in "$FP_REPLAY"/recipes/*.record.json; do
        [ -e "$r" ] || continue; n=$(basename "$r" .record.json); n=${n#example-}; mkdir -p "$REC/recipes"
        step "recipe-against-$n" "$PY" -I -m mojolearn verify --recipe "example:$n" --against "$r" \
            --record-out "$REC/recipes/$n.replay.record.json"
    done
    echo "PROOF SUMMARY failed=$fails (replay)"; [ $fails = 0 ]; exit $?
fi

# 0. FIRST: does the installed build load at all? Every family binary in one process, then every (GPU image, host
# image) pair of the CPU training routes, each pair in its own child across all cores of the allocation (the smoke
# sizes its own pool; it prints one line per pair). A failure here explains every later step, so its lines are shown.
step load-all "$PY" -I -c "import mojolearn; mojolearn._families.load_all(); print('LOADED every family binary of', mojolearn.__version__)"
step device-pair-load "$PY" -I "$HERE/device_pair_load_smoke.py"
grep -E '^FAIL' "$RES/logs/device-pair-load.log" | head -n 12 | cut -c1-300
tail -n 1 "$RES/logs/device-pair-load.log" | cut -c1-300

# 0b. the Python boundary test of `device=` for every class of the generated table, against the INSTALLED wheel:
# the one test file is copied out of the checkout so that `import mojolearn` is the venv's package, never the source
# tree. Runs only when the venv has pytest; nothing is installed here.
if "$PY" -I -c "import pytest" > /dev/null 2>&1; then
    mkdir -p "$W/device_test" && cp "$REPO/python/mojolearn/tests/test_device_parameter.py" "$W/device_test/"
    step device-parameter-test "$PY" -I -m pytest -q -p no:cacheprovider --import-mode=importlib --rootdir "$W/device_test" "$W/device_test/test_device_parameter.py"
    tail -n 1 "$RES/logs/device-parameter-test.log" | cut -c1-300
else
    echo "PROOF device-parameter-test SKIPPED rc=- 0s: pytest is not importable in $VENV (python/mojolearn/tests/test_device_parameter.py not run; install nothing here, run it where the wheel tests run)"
fi

# 1. device= and check on both devices in one process (the pilot's completion test)
cat > "$W/device.py" <<'EOF'
import numpy as np, mojolearn as ml
from mojolearn.linear_model import Ridge, LinearRegression
r = np.random.default_rng(0)
X = r.random((4000, 16), dtype=np.float32); y = (X @ np.arange(1, 17, dtype=np.float32)).astype(np.float32)
for cls, kw in ((Ridge, {"alpha": 1.0}), (LinearRegression, {})):
    g = cls(**kw).fit(X, y); c = cls(device="cpu", **kw).fit(X, y)
    pg = np.asarray(g.predict(X)); pc = np.asarray(c.predict(X))
    same = pg.tobytes() == pc.tobytes()
    print("DEVICE", cls.__name__, "default params", g.get_params().get("device"), "cpu params", c.get_params().get("device"),
          "predict bytes equal", same, "max abs diff", float(np.max(np.abs(pg - pc))))
    rep = g.check(X)
    print("CHECK", cls.__name__, rep)
    assert same, "GPU and CPU predictions differ"
try:
    Ridge(device="tpu")
    raise SystemExit("unknown device accepted")
except (ValueError, TypeError) as e:
    print("REFUSAL", type(e).__name__, str(e)[:160])
EOF
step device-and-check "$PY" -I "$W/device.py"
grep -E '^(DEVICE|CHECK|REFUSAL)' "$RES/logs/device-and-check.log" | cut -c1-400

# 1b. the same for LogisticRegression (lane dev2-logistic-device's completion test): two classes and four, with
# sample weights and class weights; labels, scores, probabilities and coefficients compared; check on both devices
cat > "$W/device_logistic.py" <<'EOF'
import numpy as np, mojolearn as ml
from mojolearn.linear_model import LogisticRegression
r = np.random.default_rng(0)
X = r.random((4000, 16), dtype=np.float32)
s = X @ np.linspace(-1.0, 1.5, 16, dtype=np.float32) + r.normal(0.0, 0.2, 4000).astype(np.float32)
w = np.linspace(0.5, 2.0, 4000, dtype=np.float32)
bad = []
for name, y in (("binary", (s > np.median(s)).astype(np.int64)),
                ("multiclass", np.digitize(s, np.quantile(s, [0.25, 0.5, 0.75])).astype(np.int64))):
    for kw, sw in (({}, None), ({"class_weight": "balanced"}, w)):
        g = LogisticRegression(max_iter=60, **kw).fit(X, y, sw); c = LogisticRegression(max_iter=60, device="cpu", **kw).fit(X, y, sw)
        same = {"coef": np.asarray(g.coef_).tobytes() == np.asarray(c.coef_).tobytes(),
                "n_iter": int(np.asarray(g.n_iter_)[0]) == int(np.asarray(c.n_iter_)[0])}
        for m in ("predict", "decision_function", "predict_proba", "predict_log_proba"):
            same[m] = np.asarray(getattr(g, m)(X)).tobytes() == np.asarray(getattr(c, m)(X)).tobytes()
        print("DEVICE LogisticRegression", name, sorted(kw), "weighted" if sw is not None else "unweighted",
              "vendors", g.vendor_used(), c.vendor_used(), "params", g.get_params().get("device"), c.get_params().get("device"), same)
        bad += [(name, k) for k, v in same.items() if not v]
        if c.vendor_used() != "cpu" or g.vendor_used() == "cpu":
            bad.append((name, "one device"))
    for m in ("predict", "predict_proba"):
        print("CHECK LogisticRegression", name, m, g.check(X, method=m))
try:
    LogisticRegression(device="tpu")
    raise SystemExit("unknown device accepted")
except (ValueError, TypeError) as e:
    print("REFUSAL", type(e).__name__, str(e)[:160])
assert not bad, f"GPU and CPU differ: {bad}"
EOF
step device-and-check-logistic "$PY" -I "$W/device_logistic.py"
grep -E '^(DEVICE|CHECK|REFUSAL)' "$RES/logs/device-and-check-logistic.log" | cut -c1-400

# 2. digest_ per estimator family (the script and its rules are at the top, section D)
step digest-families "$PY" -I "$W/digest.py" "$REC"
grep -E '^DIGEST' "$RES/logs/digest-families.log" | cut -c1-300

# 3. verify options: RELEASE ONLY (AGENTS.md hard rule "RELEASE-ONLY WORK", 2026-11-26). The self-test compares
#    with the shipped references, the cross-check compares the devices' bits and the recipes record columns, so a
#    round (proving_job.sh) skips them; the release passes FP_RELEASE=1.
if [ "${FP_RELEASE:-0}" = 1 ]; then
step verify-self-test "$PY" -I -m mojolearn verify --self-test
step verify-cross-check-quick "$PY" -I -m mojolearn verify --cross-check quick
cat > "$W/save.py" <<'EOF'
import sys, numpy as np, mojolearn as ml
d = sys.argv[1]
r = np.random.default_rng(1); X = r.random((2000, 8), dtype=np.float32); y = (X[:, 0] > 0.5).astype(np.int32)
np.save(d + "/cx.npy", X); np.save(d + "/cy.npy", y)
ml.RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y).save(d + "/rfc.model")
print("saved")
EOF
step verify-check-save "$PY" -I "$W/save.py" "$W"
step verify-check "$PY" -I -m mojolearn verify --check "$W/rfc.model" --estimator RandomForestClassifier --inputs "$W/cx.npy"
step verify-fit-both "$PY" -I -m mojolearn verify --fit-both --estimator Ridge --fit-params '{"alpha": 1.0}' --fit-inputs "$REC/X.npy" --fit-targets "$REC/y.npy" --inputs "$REC/X.npy"
# a user's ALGORITHM (verify --recipe): every shipped recipe fitted from scratch in a fresh process per device,
# compared, recorded for a second machine (FP_REPLAY compares with --against), with the recipe self-test
mkdir -p "$REC/recipes"
for n in ridge logistic kmeans gbdt-categorical seq-transformer seq-mamba2 seq-recurrent seq-conv1d-residual seq-optim-lamb seq-optim-rmsprop; do
    step "verify-recipe-$n" "$PY" -I -m mojolearn verify --recipe "example:$n" --self-test \
        --record-out "$REC/recipes/example-$n.record.json"
done

else
    echo "PROOF verify-options SKIP release only (FP_RELEASE=1 runs --self-test, --cross-check, --check, --fit-both, --recipe)"
fi

echo "PROOF SUMMARY failed=$fails digests=$REC"
[ $fails = 0 ]
