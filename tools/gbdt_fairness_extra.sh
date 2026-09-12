#!/bin/sh
# lane gbdt-fairness, the follow-up cells. RUNS ON THE POD from
# /root/mojolearn, AFTER tools/gbdt_fairness_body.sh has written body.done:
# one GPU, and a timed cell here must never overlap one there.
#
#   nohup sh tools/gbdt_fairness_extra.sh > /root/fair_out/extra_console.log 2>&1 &
#
# These answer the three questions the main body leaves open.
#
# `gpu_witness`  Does CatBoost's CUDA learner actually run? `task_type='GPU'`
#                is taken on trust everywhere in the harness. This asks
#                CatBoost's own device count and fits one model verbosely, and
#                the `.smi` sample of `flags_taxi` (a CatBoost-only cell) is
#                the independent witness beside it.
# `stage`        WHERE our 325 ms goes, host-stamped by the library itself
#                (DEVIATION 2510, MOJOLEARN_STAGE_TIMES=1): copy-in, train,
#                model text. A fit whose stamped total matches the harness's
#                wall clock has nothing hiding between the two.
# `fortran`      What the in-timer transpose costs us. `GradientBoosting.fit`
#                calls `np.asfortranarray` INSIDE the timer (DEVIATION 1840)
#                and MOJOLEARN_SPEED_FORTRAN=1 hands it an already-Fortran
#                copy prepared outside every timer. The difference is a real
#                host cost we are currently CHARGED for, so pricing it can
#                only make our number look worse, never better.
set -u
R=/root/mojolearn
OUT=/root/fair_out
C="$OUT/cells"
mkdir -p "$C"
cd "$R" || exit 9
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONPATH="$R/python"
export MOJOLEARN_NUMERIC_MODE=identical
export MOJOLEARN_SPEED_EXPECTED_VENDOR=cuda
export MOJOLEARN_SPEED_SIZE=shipped
export MOJOLEARN_SPEED_BUDGET_S=1800
export MOJOLEARN_SPEED_DEADLINE_S=3600
export GBM_BENCH_DATA=/root/datasets/gbm-bench
PY=python3
mark() { echo "$* $(date -u +%T)" | tee -a "$OUT/progress.txt"; }

smi_start() {
    ( while :; do
        printf 't=%s gpu=%s apps=%s\n' "$(date -u +%T)" \
          "$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr '\n' ' ')" \
          "$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')"
        sleep 0.2
      done ) > "$1" 2>&1 &
    echo $!
}

cell() {
    _n="$1"; shift
    _s=$(smi_start "$C/$_n.smi")
    mark "$_n start"
    timeout -k 30 1800 "$@" > "$C/$_n.log" 2>&1
    _rc=$?
    kill "$_s" 2>/dev/null
    mark "$_n=$_rc"
}

cell gpu_witness $PY -u - <<'PY'
import numpy as np, catboost
from catboost.utils import get_gpu_device_count
print("FAIR catboost version", catboost.__version__)
print("FAIR catboost get_gpu_device_count", get_gpu_device_count())
rng = np.random.default_rng(0)
x = rng.normal(size=(200000, 16)).astype(np.float32)
y = (x[:, 0] > 0).astype(np.float32)
m = catboost.CatBoostClassifier(loss_function="Logloss", iterations=20, depth=6,
                                learning_rate=0.1, l2_leaf_reg=1.0,
                                border_count=254, random_seed=7,
                                bootstrap_type="No", boosting_type="Plain",
                                task_type="GPU", devices="0", verbose=5,
                                allow_writing_files=False)
m.fit(x, y)
print("FAIR catboost fitted trees", m.tree_count_)
PY

# ONE untimed replicate, stage-stamped. Rounds is 1 because the stamps are the
# point, not the median.
MOJOLEARN_STAGE_TIMES=1 MOJOLEARN_SPEED_ROUNDS=1 \
    cell stage_taxi $PY -u bench/speed/forest_speed_arm.py \
    --lane gbdt-symmetric --dataset taxi --rows 1000000 --ours-only

MOJOLEARN_SPEED_FORTRAN=1 MOJOLEARN_SPEED_ROUNDS=5 \
    cell fortran_taxi $PY -u bench/speed/forest_speed_arm.py \
    --lane gbdt-symmetric --dataset taxi --rows 1000000 --ours-only

mark extra_done
: > "$OUT/extra.done"
