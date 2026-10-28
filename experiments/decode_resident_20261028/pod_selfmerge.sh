#!/bin/bash
# Self-merge gate leg for lane decode-resident-20261028 (runs on the RunPod CPU pod, in the repo root).
# 1) the lane's compile roots, both arms, CPU column and sm_89, all allocated cores; 2) the two scripted
# gates; 3) the pod smoke (copy of integration-provider-queue-20261028/scripts/pod_smoke_5b.sh) plus a tiny
# classical fit+predict on the CPU column. Exit codes go to $LEG_OUT/selfmerge.txt.
set -u
OUT=${LEG_OUT:?}; mkdir -p "$OUT"
S=$OUT/selfmerge.txt
CORES=$(python3 -c 'import os;print(len(os.sched_getaffinity(0)))')
echo "cores_affinity=$CORES cgroup=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null)" > "$S"
MOJO=$(pixi run -e default which mojo 2>/dev/null | tail -n 1)
export NO_PIXI=1 MOJO="$MOJO" JOBS=$CORES MAIN=$PWD
bash experiments/decode_resident_20261028/compile.sh "$OUT/cpu" > "$OUT/compile_cpu.out" 2>&1; echo "compile_cpu rc=$?" >> "$S"
COLUMN_DEFS="-D MOJOLEARN_COLUMN_NVIDIA" bash experiments/decode_resident_20261028/compile.sh "$OUT/sm89" --target-accelerator sm_89 > "$OUT/compile_sm89.out" 2>&1; echo "compile_sm89 rc=$?" >> "$S"
pixi run -e default python tools/tests/test_no_python_runtime.py > "$OUT/gate_no_python_runtime.log" 2>&1; echo "gate_no_python_runtime rc=$?" >> "$S"
pixi run -e default python tools/tests/test_tombstones_preserved.py > "$OUT/gate_tombstones.log" 2>&1; echo "gate_tombstones rc=$?" >> "$S"
LEG_OUT=$OUT/smoke bash experiments/decode_resident_20261028/pod_smoke.sh > "$OUT/pod_smoke.out" 2>&1; echo "pod_smoke rc=$?" >> "$S"
MOJOLEARN_VENDOR=cpu pixi run -e default python -c "
import numpy as np
from mojolearn.linear_model import LinearRegression
X = np.arange(40, dtype=np.float32).reshape(20, 2); y = X[:, 0] * 2 + 1
m = LinearRegression().fit(X, y); p = m.predict(X); print('classical_fit_predict', p.shape)
" > "$OUT/classical_fit_predict.log" 2>&1; echo "classical_fit_predict rc=$?" >> "$S"
cat "$S"
! grep -v '^cores' "$S" | grep -qv 'rc=0'
