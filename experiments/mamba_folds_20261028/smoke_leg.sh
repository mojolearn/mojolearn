#!/bin/bash
# Smoke-only pod leg for lane mamba-folds (the first merge leg's lease ended inside the smoke: 44/46 host families built rc=0).
set -u
rm -f python/mojolearn/host/*.so
LEG_OUT="$LEG_OUT/smoke" bash experiments/mamba_folds_20261028/pod_smoke.sh > "$LEG_OUT/smoke.out" 2>&1; s1=$?
echo "LEG smoke=$s1" | tee -a "$LEG_OUT/host.txt"
[ $s1 = 0 ]
