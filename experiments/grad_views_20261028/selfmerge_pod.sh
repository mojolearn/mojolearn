#!/bin/bash
# grad-views self-merge leg: gates, affected byte LM roots (both arms, CPU + sm_89), the pod smoke, a tiny classical fit.
set -u
cd /root/mojolearn || exit 2
OUT=${LEG_OUT:?}; mkdir -p "$OUT/gates" "$OUT/roots"
CORES=$(python3 -c 'import os;print(len(os.sched_getaffinity(0)))'); J=$(( CORES / 4 )); [ $J -lt 1 ] && J=1
pixi run -e default mojo --version > "$OUT/compiler.txt" 2>&1
echo "cores=$CORES jobs_per_root=$J" > "$OUT/resources.txt"
pixi run -e default python3 tools/tests/test_no_python_runtime.py > "$OUT/gates/no_python_runtime.log" 2>&1; echo "gate_no_python_runtime rc=$?" >> "$OUT/summary.txt"
pixi run -e default python3 tools/tests/test_tombstones_preserved.py > "$OUT/gates/tombstones.log" 2>&1; echo "gate_tombstones rc=$?" >> "$OUT/summary.txt"
r() { local tgt=$1 arm=$2 extra="" tf; [ "$arm" = on ] && extra="-D MOJOLEARN_IDN_PARAM_GRAD_VIEWS=1"
  if [ "$tgt" = cpu ]; then tf="-D MOJOLEARN_COLUMN_CPU"; else tf="--target-accelerator sm_89 -D MOJOLEARN_COLUMN_NVIDIA"; fi
  local t0=$(date +%s)
  pixi run -e default mojo build --optimization-level 0 -j $J --emit shared-lib --target-cpu x86-64-v3 $tf -D MOJOLEARN_NUMERIC_IDENTICAL=1 $extra -I . -I bindings bindings/entry/entry__mojolearn_byte_lm.mojo -o "$OUT/roots/$tgt-$arm.so" > "$OUT/roots/$tgt-$arm.log" 2>&1
  local rc=$?; echo "root_${tgt}_${arm} rc=$rc s=$(( $(date +%s)-t0 ))" >> "$OUT/summary.txt"
  [ $rc = 0 ] && sha256sum "$OUT/roots/$tgt-$arm.so" >> "$OUT/roots/sha256.txt"; }
r cpu off & r cpu on & r sm89 off & r sm89 on & wait
mkdir -p "$OUT/smoke"; LEG_OUT="$OUT/smoke" bash experiments/grad_views_20261028/pod_smoke_5b.sh; echo "pod_smoke rc=$?" >> "$OUT/summary.txt"
cat > "$OUT/classical_fit.py" <<'PY'
import numpy as np, mojolearn
from mojolearn.linear_model import LinearRegression
rng = np.random.default_rng(0)
X = rng.standard_normal((64, 4)).astype(np.float32); y = X @ np.arange(1, 5, dtype=np.float32)
m = LinearRegression().fit(X, y); p = m.predict(X)
print("classical_fit ok", p.shape)
PY
MOJOLEARN_VENDOR=cpu pixi run -e default python "$OUT/classical_fit.py" > "$OUT/classical_fit.log" 2>&1; echo "classical_fit rc=$?" >> "$OUT/summary.txt"
! grep -qv 'rc=0' <(grep -o 'rc=[0-9]*' "$OUT/summary.txt")
