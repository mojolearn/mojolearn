#!/bin/sh
# XGBoost was absent from the pod (setup pip line lacked it); depthwise and
# lossguide re-run with the full roster after pip install xgboost.
cd /root/mojolearn || exit 9
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical
sh tools/trees_identical_ab.sh speed baseline gbdt-depthwise istella 1000000 7 full
sh tools/trees_identical_ab.sh speed baseline gbdt-lossguide istella 1000000 7 full
echo "PHASE_F_DONE $(date -u +%T)" | tee -a /root/trees_out/ab.txt
