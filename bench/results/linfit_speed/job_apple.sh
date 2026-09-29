#!/bin/bash
# lane/linfit-speed on Apple (the m2pro steward): the linear identity lanes
# (--pass 2, the e2e sabotage seen failing), then the wide GLM's bits against
# the one-block fit and against tiny bounded launches on synthetic data.
set -uo pipefail
export MOJOLEARN_NUMERIC_MODE=identical
# the lanes the e2e sabotage reaches (the wide GLM, the speculative SGD row
# dot); SGDOneClassSVM and PassiveAggressive fold norms or |x|^2 with the dot
# (the exact path, which the sabotage does not touch) and Perceptron's
# fixture did not move under it (m2pro, ee967e460f): they run clean, --pass 2
L=x-glm-poisson,x-glm-gamma,x-glm-tweedie,x-glm-poisson-sw,x-sgd-clf,x-sgd-reg,x-sgd-clf-w,x-sgd-reg-sw
sh tools/algos_lane_check.sh "$L" --pass 2 --sabotage x_linear/checks/sabotage/e2e_linfit_wide.patch; rc=$?
echo "LANE-CHECK sabotaged set rc=$rc"
sh tools/algos_lane_check.sh x-sgd-ocsvm,x-perceptron,x-pa-clf,x-pa-reg --pass 2; rc2=$?
echo "LANE-CHECK clean set rc=$rc2"
[ $rc2 -eq 0 ] || rc=$rc2
sh bindings/build_x_linear.sh >/dev/null 2>&1 || true
PY="pixi run -e default python"
O=${TMPDIR:-/tmp}/linfit
mkdir -p "$O"
S="--synthetic 20000,30 --datasets s --lanes poisson --max-iter 20"
$PY tools/linfit_speed.py $S --out "$O/a.json" --env MOJOLEARN_X_LINEAR_GW_TRACE=1
$PY tools/linfit_speed.py $S --out "$O/b.json" --env MOJOLEARN_X_LINEAR_GW_STEPS=4096
$PY tools/linfit_speed.py $S --out "$O/c.json" --env MOJOLEARN_X_LINEAR_GLM_TEAM=1
python3 - "$O" <<'PY'
import json, sys
shas = [json.load(open("%s/%s.json" % (sys.argv[1], k)))[0]["sha"] for k in "abc"]
print("GLM-WIDE-BITS", "EQUAL" if len(set(shas)) == 1 else "DIFFER", shas)
PY
exit $rc
