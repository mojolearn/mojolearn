#!/bin/bash
# tools/ops/feature_proof_job.sh -- FIRST REAL RUN, from the installed candidate wheel on a GPU box, of the user
# features that so far only compiled (2026-11-21): the `device=` parameter and `model.check(X)` on both devices in one
# process (Ridge, LinearRegression), `model.receipt` + `python -m mojolearn reproduce`, and the verify options
# `--self-test`, `--check`, `--fit-both`, `--cross-check quick`. Diagnostic tooling: it times nothing and decides no
# release. Each step prints one line `PROOF <name> PASS|FAIL rc=<n> <seconds>s` and keeps its full output in
# logs/<name>.log; the job exits 1 when any step failed. Steps are independent and short, the GPU is one device, so
# they run one after another (a GPU step owns the device); nothing here is a loop over heavy independent units.
#
#   tools/nvidia_central.sh submit ab --cap 30 --results box-out/proof-<tag> tools/ops/feature_proof_job.sh
#
# FP_VENV (default /root/pc-cache/venv-candidate): the venv holding the candidate wheel (a paper-compare or
# call-profile job installs it). FP_OUT_RECEIPTS: where the receipts and their data are left for a second machine
# (default $RES/receipts): copy that folder to a CPU box and run `FP_REPLAY=<folder> feature_proof_job.sh` there to
# prove the same receipt reproduces on another machine.
set -u
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

if [ -n "${FP_REPLAY:-}" ]; then
    # second machine: replay the receipts another box wrote
    for r in ridge rf; do
        step "replay-$r-same" "$PY" -I -m mojolearn reproduce "$FP_REPLAY/$r.json" --X "$FP_REPLAY/X.npy" --y "$FP_REPLAY/y.npy"
        WANT_RC=1 step "replay-$r-flipped-bit" "$PY" -I -m mojolearn reproduce "$FP_REPLAY/$r.json" --X "$FP_REPLAY/Xbad.npy" --y "$FP_REPLAY/y.npy" --json
    done
    echo "PROOF SUMMARY failed=$fails (replay)"; [ $fails = 0 ]; exit $?
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

# 2. receipts and reproduce (same machine); the folder is left for a second machine
cat > "$W/receipt.py" <<'EOF'
import sys, numpy as np, mojolearn as ml
d = sys.argv[1]
r = np.random.default_rng(0); X = r.random((1000, 8), dtype=np.float32); y = X @ np.arange(1, 9, dtype=np.float32)
np.save(d + "/X.npy", X); np.save(d + "/y.npy", y); B = X.copy(); B.view(np.uint8)[0, 0] ^= 1; np.save(d + "/Xbad.npy", B)
ml.Ridge(alpha=1.0).fit(X, y).receipt(X, y, path=d + "/ridge.json")
ml.RandomForestRegressor(n_estimators=20, random_state=0).fit(X, y).receipt(X, y, path=d + "/rf.json")
print("receipts written")
EOF
step receipt-write "$PY" -I "$W/receipt.py" "$REC"
for r in ridge rf; do
    step "reproduce-$r-same" "$PY" -I -m mojolearn reproduce "$REC/$r.json" --X "$REC/X.npy" --y "$REC/y.npy"
    WANT_RC=1 step "reproduce-$r-flipped-bit" "$PY" -I -m mojolearn reproduce "$REC/$r.json" --X "$REC/Xbad.npy" --y "$REC/y.npy" --json
done

# 3. verify options
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

echo "PROOF SUMMARY failed=$fails receipts=$REC"
[ $fails = 0 ]
