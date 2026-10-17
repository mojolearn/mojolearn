#!/bin/bash
# correctness_round.sh <vendor-label> <outdir>
#
# One correctness round on a GPU box, run as ONE lq CMD job so the bindings are
# built once (box_job.sh builds BUILDS= before calling this) and every check
# runs against that build. Correctness only: nothing here is timed or compared
# against an opponent (Andrew 2026-10-17: fix everything before measuring).
#
#   1. the pixi Mojo checks for the GBDT multi-dim device walker, the oracle
#      and the device CTR expansion
#   2. the identity re-record of the lanes whose sources changed
#      (ONLY_CHANGED=table, REPEATS=1)
#   3. the SGD-family quality and scikit-learn estimator checks
#   4. GBDT outputs on the default build (tools/apple_slow_paths_check.py gbdt)
#   5. non-board GBDT MultiClass / MultiRMSE digests at class counts around the
#      walker's shared-to-global solve switch, for the NVIDIA == AMD compare
#   6. the _OFF arms of the new switches rebuilt together, then step 4 again:
#      every switch is bit-neutral, so the two outputs must be equal
#
# Each step writes <outdir>/<step>.log and one "STEP <name> rc=<rc>" line to
# <outdir>/summary.txt; the script never stops at a failed step.
set -u
label=$1; out=$2; mkdir -p "$out"; S=$out/summary.txt; : > "$S"
PY="pixi run -e default python"
step() { local name=$1; shift; ( "$@" ) > "$out/$name.log" 2>&1; echo "STEP $name rc=$?" >> "$S"; }

step mojo-checks pixi run check-multiclass-oracle check-newton-walker check-logloss-estimator check-loss-oracle \
  check-logloss-leaf-oracle check-exact-estimation check-fit-pointwise check-ordered-boosting check-multiclass-train \
  check-ctr-apply check-ctr-device
step idrec env ONLY_CHANGED=table REPEATS=1 bash tools/record_identity_column.sh "$label" "$out/idrec"
step sgd-quality $PY tools/sgd_quality_check.py "$out/sgd"
step gbdt-default $PY tools/apple_slow_paths_check.py gbdt --n-estimators 50 --mode identical
cat > "$out/mcdig.py" <<'EOF'
import hashlib, numpy as np, mojolearn as ml
def h(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
rng = np.random.default_rng(7)
for k in (3, 17, 33, 62, 63, 64, 65):
    X = rng.standard_normal((4000, 12)).astype(np.float32)
    y = (np.abs(X[:, 0] * 7 + X[:, 1] * 3).astype(np.int64) + np.arange(4000)) % k
    m = ml.GradientBoostingClassifier(n_estimators=5).fit(X, y)
    print(f"MCDIG multiclass k={k} {h(m.predict_proba(X))}", flush=True)
X = rng.standard_normal((4000, 12)).astype(np.float32)
y = (np.arange(4000) % 5)
m = ml.GradientBoostingClassifier(n_estimators=5, leaf_estimation_iterations=3).fit(X, y)
print(f"MCDIG multiclass k=5 leaf_iters=3 {h(m.predict_proba(X))}", flush=True)
Y = np.stack([X[:, 0] * 2 + X[:, 1], X[:, 2] - X[:, 3], X[:, 4] * X[:, 5]], axis=1).astype(np.float32)
m = ml.GradientBoostingRegressor(n_estimators=5, loss="MultiRMSE").fit(X, Y)
print(f"MCDIG multirmse dim=3 {h(m.predict(X))}", flush=True)
EOF
step multiclass-digests $PY "$out/mcdig.py"
export MOJOLEARN_BUILD_DEFINES=MOJOLEARN_GBDT_WALK_DEV_STEP_OFF=1,MOJOLEARN_GBDT_ORD_TREE_BUFS_OFF=1,MOJOLEARN_SGD_AUTO_STEPS_256_OFF=1
step build-off bash -c 'bash bindings/build_gbdt.sh && bash bindings/build_x_linear.sh'
unset MOJOLEARN_BUILD_DEFINES
step gbdt-off $PY tools/apple_slow_paths_check.py gbdt --n-estimators 50 --mode identical
step multiclass-digests-off $PY "$out/mcdig.py"
for a in gbdt multiclass-digests; do
  if cmp -s <(grep -E '^(GBDT|MCDIG) ' "$out/$a-default.log" "$out/$a.log" 2>/dev/null | sed 's/^[^:]*://') \
            <(grep -E '^(GBDT|MCDIG) ' "$out/$a-off.log" | sed 's/^[^:]*://'); then
    echo "STEP $a-off-equal rc=0" >> "$S"; else echo "STEP $a-off-equal rc=1" >> "$S"; fi
done
echo "ROUND $label $(grep -c 'rc=0' "$S")/$(wc -l < "$S") steps rc=0"
