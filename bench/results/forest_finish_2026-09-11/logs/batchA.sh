#!/bin/sh
# Lane forest-finish, H100-class leg 2026-09-11 night, batch A.
# RUNS ON THE POD. /root/mojolearn = lane/forest-finish (DEVIATIONS 2637, 2638
# merged with main); /root/mojolearn_main = the main commit that merge took.
# Sets: baseline = main source for the rf, trees and svm extensions (base and
# gbdt are the same source on both sides and come from the setup build);
# rowmajor = this lane's source for all five. IDENTICAL tier, 1M rows.
cd /root/mojolearn || exit 9
AB="sh tools/trees_identical_ab.sh"
OUT=/root/trees_out
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical
mark() { echo "$1 $(date -u +%T)" | tee -a $OUT/ab.txt; : > "$OUT/phase.$1"; }
LANES=rf-clf,rf-reg,et-clf,et-reg,iforest
echo "batchA start $(date -u +%T)"

# ---- A1: the main-source extensions, built while setup builds this lane's.
while ! grep -q '^pixi_install=' $OUT/setup.txt 2>/dev/null; do sleep 10; done
mkdir -p /root/bins/baseline /root/bins/rowmajor $OUT/logs
( cd /root/mojolearn_main && timeout -k 30 1500 pixi install > $OUT/logs/main_pixi_install.log 2>&1
  echo "main_pixi_install=$? $(date -u +%T)" | tee -a $OUT/ab.txt
  for b in rf trees svm; do
    MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=4 \
      timeout -k 30 1500 bash bindings/build_$b.sh > $OUT/logs/build.baseline.$b.log 2>&1
    echo "build_exit baseline.$b=$? $(date -u +%T)" | tee -a $OUT/ab.txt
    cp python/mojolearn/identical/_mojolearn_$b.so /root/bins/baseline/ || echo "MISSING baseline $b" | tee -a $OUT/ab.txt
  done )
while [ ! -f $OUT/track_mojo.done ]; do sleep 10; done
cat $OUT/setup.txt
for b in "" _gbdt; do cp python/mojolearn/identical/_mojolearn$b.so /root/bins/baseline/; done
for b in "" _gbdt _rf _trees; do cp python/mojolearn/identical/_mojolearn$b.so /root/bins/rowmajor/; done
if $AB build rowmajor svm; then :; else rm -f /root/bins/rowmajor/_mojolearn_svm.so; echo "ROWMAJOR SVM BUILD FAILED" | tee -a $OUT/ab.txt; fi
sha256sum /root/bins/baseline/*.so /root/bins/rowmajor/*.so | tee $OUT/sets_so_sha256.txt
grep -c 'build_exit.*=0' $OUT/ab.txt
mark A1_BUILD_DONE

# ---- A2: identity. Fingerprints of both sets, the diff, the iforest gate,
# and the non-finite refusal message through the Python surface (the new
# threaded scan must re-raise the serial scan's exact text).
$AB ib baseline $LANES
$AB ib rowmajor $LANES
$AB diff baseline rowmajor
timeout -k 30 1500 pixi run mojo run -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 isolation_forest/checks/if_check.mojo \
  > $OUT/logs/check_if.rowmajor.log 2>&1
echo "check_if rowmajor=$? $(date -u +%T)" | tee -a $OUT/ab.txt
tail -5 $OUT/logs/check_if.rowmajor.log
for set in baseline rowmajor; do
  $AB use $set
  PYTHONPATH=/root/mojolearn/python timeout -k 30 600 python3 - > $OUT/logs/refusal.$set.log 2>&1 <<'PY'
import numpy as np, mojolearn
rng = np.random.default_rng(7)
for layout in ("C", "F"):
    X = np.ascontiguousarray(rng.normal(size=(20000, 64)).astype(np.float32))
    X[12345, 17] = np.nan
    X[15000, 3] = np.inf
    if layout == "F":
        X = np.asfortranarray(X)
    for name, make in (("iforest", lambda: mojolearn.IsolationForest(n_estimators=10, random_state=1)),
                       ("rf", lambda: mojolearn.RandomForestClassifier(n_estimators=4, random_state=1, device="gpu")),
                       ("et", lambda: mojolearn.ExtraTreesClassifier(n_estimators=4, random_state=1, device="gpu"))):
        try:
            m = make()
            if name == "iforest":
                m.fit(X)
            else:
                m.fit(X, (np.arange(20000) % 2).astype(np.int64))
            print("REFUSAL", layout, name, "ACCEPTED")
        except Exception as exc:
            print("REFUSAL", layout, name, type(exc).__name__, str(exc)[:400])
PY
  echo "refusal_probe $set=$? $(date -u +%T)" | tee -a $OUT/ab.txt
done
diff $OUT/logs/refusal.baseline.log $OUT/logs/refusal.rowmajor.log > $OUT/logs/refusal.diff.txt 2>&1
echo "refusal_diff_exit=$?" | tee -a $OUT/ab.txt
mark A2_IDENTITY_DONE

# ---- A3: same-pod speed. Pair 1 is full (opponent interleaved), baseline
# then rowmajor; pair 2 is ours-only, rowmajor then baseline (ABBA).
cells() {
  ds=$1
  $AB speed baseline rf $ds 1000000 3 full
  $AB speed rowmajor rf $ds 1000000 3 full
  $AB speed baseline iforest $ds 1000000 3 full
  $AB speed rowmajor iforest $ds 1000000 3 full
  MOJOLEARN_SPEED_DEVICES=cpu MOJOLEARN_SPEED_ARMS=sklearn-et-cpu $AB speed baseline et $ds 1000000 3 full
  MOJOLEARN_SPEED_DEVICES=cpu MOJOLEARN_SPEED_ARMS=sklearn-et-cpu $AB speed rowmajor et $ds 1000000 3 full
  for lane in rf iforest et; do
    $AB speed rowmajor $lane $ds 1000000 3 ours
    $AB speed baseline $lane $ds 1000000 3 ours
  done
}
cells taxi
mark A3_TAXI_DONE
while [ ! -f $OUT/setup.done ]; do sleep 10; done
cells istella
mark A3_ISTELLA_DONE

# ---- A4: one untimed stage replicate per forest cell on the lane's set.
for lane in et rf; do
  for ds in istella taxi; do
    $AB speed rowmajor $lane $ds 1000000 1 stage
  done
done
mark A4_STAGE_DONE
echo "batchA end $(date -u +%T)"
