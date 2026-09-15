#!/bin/bash
# Metal evidence for lane/inference-gbdt-ctr-tables. One Metal job per step,
# each through the exclusive Metal slot, short chunks (the 0.8.6 release lane
# has priority on the slot).
#   step new   the two CTR table lanes, fixtures base,ties,odd, two repeats,
#              models saved into the evidence directory (repeat 0)
#   step spot  existing GBDT lanes on the base fixture, one repeat: the
#              GPU predict of gbdt-feature-freq runs the moved tensor apply body
set -u
SP=/private/tmp/claude-501/-Users-andrewhendel-CascadeProjects-mojolearn/e52730fc-0ee7-4bf3-8741-5fa0e0f1876f/scratchpad
WT=$SP/wt-gbdt-ctr
EV=$WT/bench/results/identity_break/2026-09-15_gbdt-ctr-tables
PY=/Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/test/bin/python
L=$SP/ctr/logs/metal-evidence.log
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MOJOLEARN_CPU_THREADS=1 MODULAR_THREAD_BUSY_WAIT_US=0
export MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=$WT/python
mkdir -p $EV/models
cd $WT || exit 2
NEW=gbdt-categorical-ctr-tables,gbdt-tensor-ctr-tables
SPOT=gbdt-feature-freq,gbdt-categorical-ctr,gbdt-symmetric,gbdt-rmse
R=bench/results/identity_break/2026-09-14_166-lanes
step=${1:-all}
if [ "$step" = all ] || [ "$step" = new ]; then
  # one part per lane, all three fixtures in each: --merge refuses parts
  # whose fixture hashes differ
  for ln in gbdt-categorical-ctr-tables gbdt-tensor-ctr-tables; do
    s=$(date +%s)
    MOJOLEARN_IDENTITY_GBDT_CTR_MODELS=$EV/models bash $SP/mac_slot.sh metal nice -n 19 $PY tools/identity_break.py \
      --lanes $ln --fixtures base,ties,odd --repeats 2 --vendor apple-m4 --json $SP/ctr/metal-$ln.json > $SP/ctr/logs/metal-$ln.log 2>&1
    echo "new $ln rc=$? $(( $(date +%s)-s ))s" >> $L
  done
  $PY tools/identity_break.py --merge $SP/ctr/metal-gbdt-categorical-ctr-tables.json $SP/ctr/metal-gbdt-tensor-ctr-tables.json \
    --json $EV/apple-m4.json > $SP/ctr/logs/metal-merge.log 2>&1
  echo "merge rc=$?" >> $L
fi
if [ "$step" = all ] || [ "$step" = spot ]; then
  s=$(date +%s)
  bash $SP/mac_slot.sh metal nice -n 19 $PY tools/identity_break.py --lanes $SPOT --fixtures base --repeats 1 \
    --vendor apple-m4 --json $EV/apple-m4.spot-base.json > $SP/ctr/logs/metal-spot.log 2>&1
  echo "spot rc=$? $(( $(date +%s)-s ))s" >> $L
  $PY tools/identity_break.py --diff $R/apple-m4.json $R/nvidia-h100-sm_90a.json $R/amd-mi325x-gfx942.json \
    $EV/apple-m4.spot-base.json --lanes $SPOT > $EV/diff-metal-spot-base.txt 2>&1
  echo "spot diff rc=$?" >> $L
fi
echo DONE >> $L
