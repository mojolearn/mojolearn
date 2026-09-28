#!/bin/bash
# lane/py-bugs: the before/after bit record on one shared NVIDIA pod (x86 CPU
# column and NVIDIA column), LIGHT: each tree built once, each lane's GPU and
# CPU arm once, the probe once per column, then base vs new part by part.
# BASE is lane py-bugs-base's tree (lane/apple2-merged at the merge), NEW this lane's.
set -u
NEW=/root/mojolearn-py-bugs
BASE=${BASE:-/root/mojolearn-py-bugs-base}
EV=${EV:-/root/ev-py-bugs/$(date -u +%m%d-%H%M)}
mkdir -p "$EV"
export MOJOLEARN_LANE_CHECK_STORE=/root/py-bugs-store MOJOLEARN_BUILD_LOCK_HELD=1
export MOJOLEARN_COMPILE_JOBS=${MOJOLEARN_COMPILE_JOBS:-4}
LANES=${LANES:-x-cluster-bgmm,x-cluster-bgmm-inits,x-cluster-bgmm-covtypes,x-cluster-gmm-options,x-logistic-cv,x-logistic-cv-w,x-huber,x-ridge-clf,x-metrics-classification,x-metrics-cluster,x-metrics-search,x-metrics-splitters,x-cnn-trainer,x-cnn-trainer-options,x-prep-iterative-options,x-prep-user-objects,x-neighbors-svgp,x-neighbors-svc-probability,x-decomp-grp,x-decomp-srp,sequence-autoarima,sequence-rnn,sequence-lstm,sequence-gru}
cd "$NEW"
echo "$(date -u +%FT%TZ) $(hostname) new $(git rev-parse --short HEAD) base $(git -C "$BASE" rev-parse --short HEAD) out $EV"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1; lscpu | grep -m1 'Model name'
PIXI=$(command -v pixi || echo ~/.pixi/bin/pixi)
$PIXI install -e default > "$EV/pixi.log" 2>&1 || { echo "PIXI INSTALL FAIL"; tail -5 "$EV/pixi.log"; exit 1; }
P="$PIXI run -e default python -u tools/py_bugs/check.py"
if [ "${SKIP_BASE:-0}" != 1 ]; then
    $P arms --tree "$BASE" --out "$EV/base" --lanes "$LANES" 2>&1 | tee "$EV/base.out" | grep -v '^\s*$' | tail -80
fi
$P arms --tree "$NEW" --out "$EV/new" --lanes "$LANES" 2>&1 | tee "$EV/new.out" | grep -v '^\s*$' | tail -80
$P cross --base "${BASE_EV:-$EV/base}" --new "$EV/new" --lanes "$LANES" 2>&1 | tee "$EV/cross.txt"
$PIXI install -e test > "$EV/pixi_test.log" 2>&1 || echo "PIXI TEST ENV INSTALL FAIL"
for col in gpu cpu; do
    echo "== pytest test_py_bugs ($col)"
    ( cd python && if [ $col = cpu ]; then export MOJOLEARN_VENDOR=cpu; fi
      MOJOLEARN_NUMERIC_MODE=identical $PIXI run -e test python -m pytest -q -p no:cacheprovider \
          mojolearn/tests/test_py_bugs.py 2>&1 | tail -15 )
done
echo "JOB END $(date -u +%FT%TZ)"
