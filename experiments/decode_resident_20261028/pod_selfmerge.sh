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
export POD_PIXI=1 JOBS=$CORES MAIN=$PWD
# The shipped tree has no .git; the tombstone gate reads markers with `git grep` over tracked files,
# so index exactly the shipped (tracked) files.
if [ ! -d .git ]; then
  git init -q . && git -c user.email=pod@local -c user.name=pod add -A > /dev/null 2>&1 \
    && git -c user.email=pod@local -c user.name=pod commit -q -m shipped-tree > /dev/null 2>&1
  echo "git_index_rc=$?" >> "$S"
fi
# No CPU-column leg here: these roots import the device bindings, and on a GPU-less Linux pod a
# CPU-column build of them stops in std's GPU info ("Unknown GPU architecture detected", at main's own
# mamba2_block_forward launch); the CPU column of these roots is the Mac receipt.
COLUMN_DEFS="-D MOJOLEARN_COLUMN_NVIDIA" bash experiments/decode_resident_20261028/compile.sh "$OUT/sm89" --target-accelerator sm_89 > "$OUT/compile_sm89.out" 2>&1; echo "compile_sm89 rc=$?" >> "$S"
pixi run -e default python tools/tests/test_no_python_runtime.py > "$OUT/gate_no_python_runtime.log" 2>&1; echo "gate_no_python_runtime rc=$?" >> "$S"
pixi run -e default python tools/tests/test_tombstones_preserved.py > "$OUT/gate_tombstones.log" 2>&1; echo "gate_tombstones rc=$?" >> "$S"
# --release-bincache stages host binaries before the command; the smoke builds every family through
# the cache into python/mojolearn/host and refuses an existing output, so start it from an empty dir.
rm -f python/mojolearn/host/_mojolearn_*.so
LEG_OUT=$OUT/smoke bash experiments/decode_resident_20261028/pod_smoke.sh > "$OUT/pod_smoke.out" 2>&1; echo "pod_smoke rc=$?" >> "$S"
MOJOLEARN_VENDOR=cpu pixi run -e default python -c "
import numpy as np
from mojolearn.linear_model import LinearRegression
X = np.arange(40, dtype=np.float32).reshape(20, 2); y = X[:, 0] * 2 + 1
m = LinearRegression().fit(X, y); p = m.predict(X); print('classical_fit_predict', p.shape)
" > "$OUT/classical_fit_predict.log" 2>&1; echo "classical_fit_predict rc=$?" >> "$S"
cat "$S"
! grep -v '^cores' "$S" | grep -qv 'rc=0'
