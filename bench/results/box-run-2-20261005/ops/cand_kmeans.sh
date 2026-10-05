#!/bin/bash
# cand_kmeans.sh <build|identity|timing> <vendor> <arch> <cuda|hip>  (box-run-2)
# KMeans acc256 opt-in at the consolidated head: master ON + MOJOLEARN_IDN_KMEANS_ACC_ROWS_256, recipe builders only.
# Baseline = this box's wave ON tree (same flags, no define). timing: caller holds /root/lq/TIMING_LOCK.
OP=$1; V=$2; A=$3; GV=$4; R=/root/mojolearn-main; PY=/root/mojolearn/.pixi/envs/default/bin/python; SHA=$(git -C $R rev-parse HEAD)
C=/root/lq/br2-cand-kmeans-acc256-${SHA:0:9}; B=/root/lq/br2-wave-${SHA:0:9}/on/source; G=$R/tools/identical_candidate_gate.py
DATA=/root/board-0833/cache/algos-data/rows-small
gate() { # label source vendor operation outdir
  env -u MOJOLEARN_IDN_ALL_OFF PYTHONPATH=$2/python MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_VENDOR=$3 MOJOLEARN_TARGET_COLUMN=$V MOJOLEARN_GPU_ARCHS=$A MOJOLEARN_SKIP_BUILD_GATE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    $PY $G --source $2 --lane kmeans --data $DATA --vendor $3 --operation $4 --out $5/$1-kmeans-$3 > $5/$1-kmeans-$3.log 2>&1
  echo "$1 kmeans $3 $4 rc=$?" >> $5/summary.txt; }
case $OP in
build) $PY $R/tools/identical_wave_native_build.py --sha $SHA --repo $R --out $C --python $PY --vendor $V --gpu-arch $A --arm on \
         --define MOJOLEARN_IDN_KMEANS_ACC_ROWS_256=1 --builders base,base_host,estimators,estimators_host > $C.log 2>&1; echo $? > $C.rc ;;
identity) O=/root/lq/br2-cand-identity-${SHA:0:9}; mkdir -p $O
  for v in $GV cpu; do gate base $B $v identity $O; gate cand $C/source $v identity $O; done; echo DONE >> $O/summary.txt ;;
timing) O=/root/lq/br2-cand-timing-${SHA:0:9}; mkdir $O || exit 3
  gate base $B $GV timing $O; gate cand $C/source $GV timing $O; echo DONE >> $O/summary.txt ;;
esac
