#!/bin/sh
# gbdt-speed lane, same-pod baseline: our three IDENTICAL grow policies
# interleaved with the GPU opponents (symmetric: CatBoost GPU only;
# depthwise and lossguide: CatBoost GPU and XGBoost GPU), taxi and Istella-S
# at 1M rows, 5 timed rounds each; then one stage-timed replicate per cell.
# Taxi starts once the bindings are built; Istella-S waits for its download
# (the Istella-S decode may overlap the last taxi cell; noted in the summary).
set -u
cd /root/mojolearn || exit 9
OUT=/root/trees_out; AB="sh tools/trees_identical_ab.sh"
export PATH="$HOME/.pixi/bin:$PATH" MOJOLEARN_NUMERIC_MODE=identical
mark() { echo "$1 $(date -u +%T)" | tee -a $OUT/progress.txt; }
cells() {  # dataset
  MOJOLEARN_SPEED_ARMS=catboost-gpu $AB speed baseline gbdt-symmetric $1 1000000 5 full
  MOJOLEARN_SPEED_ARMS=catboost-gpu,xgboost-gpu $AB speed baseline gbdt-depthwise $1 1000000 5 full
  MOJOLEARN_SPEED_ARMS=catboost-gpu,xgboost-gpu $AB speed baseline gbdt-lossguide $1 1000000 5 full
  for lane in gbdt-symmetric gbdt-depthwise gbdt-lossguide; do
    $AB speed baseline $lane $1 1000000 1 stage
  done
}
while [ ! -f $OUT/build.done ]; do sleep 10; done
while ! grep -q "^download_taxi=" $OUT/setup.txt; do sleep 10; done
mark baseline_taxi_start
cells taxi
mark taxi_done
while [ ! -f $OUT/data.done ]; do sleep 10; done
mark baseline_istella_start
cells istella
mark istella_done
mark full_done
