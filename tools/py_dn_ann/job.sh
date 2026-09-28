#!/bin/bash
# py-dn-ann: lane checks head vs base, one sabotage run, then timing base vs head.
set -u
EV=/root/ev-py-dn-ann; OUT=$EV/out; mkdir -p $OUT
HEAD=/root/mojolearn-py-dn-ann; BASE=/root/mojolearn-py-dn-base
L=ivf,ivf-euclidean,ivf-extend,ivf-filter,par-ivf,x-ann-ivf-pq,x-ann-ivf-sq,x-ann-ivf-rabitq,x-ann-cagra,x-ann-cagra-filter,x-ann-filter,x-ann-refine,x-ann-refine-euclidean
export PATH=$HOME/.pixi/bin:$PATH
echo "$(date -u +%FT%TZ) waiting for the base columns of the ann lanes"
for i in $(seq 1 360); do
  c=0; for l in ${L//,/ }; do [ -f /root/ev-py-decomp-nbrs/base/clean.$l.cpu.json ] && c=$((c+1)); done
  [ $c -ge 13 ] && break; sleep 30
done
echo "base ann columns present: $c/13"
/root/ev-py-decomp-nbrs/head_job.sh $HEAD $L $OUT/head
echo "== SABOTAGE (resident path) on ivf,x-ann-ivf-pq,x-ann-cagra"
cd $HEAD && PIXI=pixi sh tools/algos_lane_check.sh ivf,x-ann-ivf-pq,x-ann-cagra --sabotage x_ann/checks/sabotage/resident_index_py_dn_ann.patch --out $OUT/sab 2>&1 | grep -E 'RESULT|CLEAN:|SABOTAGED:|RESTORED:'
echo "== TIMING"
for T in $BASE $HEAD; do
  tag=$(basename $T)
  cd $T
  echo "-- $tag gpu"; PYTHONPATH=python MOJOLEARN_NUMERIC_MODE=identical SIZE=1000000 DIM=128 BATCHES=20 BQ=500 pixi run -e default python -u $EV/bench_ann.py 2>&1 | grep -E 'BENCH|FIT|Error|error' | sed "s/^/$tag /"
  echo "-- $tag cpu"; PYTHONPATH=python MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_VENDOR=cpu MOJOLEARN_HOST_DIR=$T/python/mojolearn/.host SIZE=200000 DIM=128 BATCHES=10 BQ=500 WHAT=flat,dist DEVICES=0 pixi run -e default python -u $EV/bench_ann.py 2>&1 | grep -E 'BENCH|FIT|Error|error' | sed "s/^/$tag /"
done
echo "JOB END $(date -u +%FT%TZ)"
