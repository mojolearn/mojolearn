#!/bin/bash
# lane/py-misc: ONE light job for the CNN epoch entry on this box.
# Builds x_cnn (GPU) and x_cnn_host when stale, then
#  1. tools/py_misc/cnn_epoch.py on the GPU column and on the CPU host twin:
#     Python step loop vs x_cnn_fit_epoch_r, digests + timing in one process;
#  2. the identity lanes x-cnn-trainer,x-cnn-trainer-options, GPU and CPU
#     arms, before (MOJOLEARN_XCNN_PY_STEPS=1) and after; before == after per
#     column (identity_break --diff) and GPU == CPU after.
# usage: cnn_job.sh <out dir> [gpu timing rows] [cpu timing rows]
cd "$(dirname "$0")/../.."
OUT=${1:?out}; GROWS=${2:-200000}; CROWS=${3:-20000}; mkdir -p "$OUT"
export MOJOLEARN_BUILD_LOCK_HELD=1 MOJOLEARN_COMPILE_JOBS=${MOJOLEARN_COMPILE_JOBS:-4}
PIXI=$(command -v pixi || echo ~/.pixi/bin/pixi)
echo "$(date -u +%FT%TZ) $(hostname) $(git rev-parse --short HEAD 2>/dev/null) out $OUT"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1
lscpu 2>/dev/null | grep -E 'Model name|^CPU\(s\)'
$PIXI install -e default > "$OUT/pixi_install.log" 2>&1 || { echo "PIXI INSTALL FAIL"; tail -5 "$OUT/pixi_install.log"; exit 1; }
PY="$PIXI run -e default python -u"
$PY - "$OUT" <<'PY' || { echo BUILD FAIL; exit 1; }
import sys
sys.path.insert(0, "tools")
import algos_lane_check as alc
out = sys.argv[1]
alc.ensure_portable_math(out + "/build_portable_math.log")
for b in ("_mojolearn_x_cnn", "_mojolearn_x_cnn_host"):
    why = alc.stale(b)
    print(b, "stale:", why, flush=True)
    if why:
        alc.build(b, out + "/build.log")
PY
HOST="$PWD/python/mojolearn/host"
export PYTHONPATH="$PWD/python" MOJOLEARN_NUMERIC_MODE=identical OMP_NUM_THREADS=1
echo "== cnn_epoch GPU"
$PY tools/py_misc/cnn_epoch.py "$PWD" "$GROWS" 3 2>&1 | tee "$OUT/cnn_epoch_gpu.txt" | grep PYMISC
echo "== cnn_epoch CPU"
MOJOLEARN_VENDOR=cpu MOJOLEARN_HOST_DIR="$HOST" $PY tools/py_misc/cnn_epoch.py "$PWD" "$CROWS" 3 2>&1 \
  | tee "$OUT/cnn_epoch_cpu.txt" | grep PYMISC
LANES=x-cnn-trainer,x-cnn-trainer-options
H="$PY tools/identity_break.py --lanes $LANES --repeats 1 --fail-on-refused"
for arm in before after; do
  if [ $arm = before ]; then export MOJOLEARN_XCNN_PY_STEPS=1; else unset MOJOLEARN_XCNN_PY_STEPS; fi
  $H --json "$OUT/$arm.gpu.json" --require-backend cuda > "$OUT/$arm.gpu.log" 2>&1; echo "arm $arm gpu exit $?"
  MOJOLEARN_VENDOR=cpu MOJOLEARN_HOST_DIR="$HOST" $H --json "$OUT/$arm.cpu.json" --require-cpu --require-backend cpu \
    > "$OUT/$arm.cpu.log" 2>&1; echo "arm $arm cpu exit $?"
done
unset MOJOLEARN_XCNN_PY_STEPS
for col in gpu cpu; do
  $PY tools/identity_break.py --diff "$OUT/before.$col.json" "$OUT/after.$col.json" --lanes $LANES --require-columns 2 \
    > "$OUT/diff_before_after_$col.txt" 2>&1; echo "DIFF before==after $col exit $?"; tail -4 "$OUT/diff_before_after_$col.txt"
done
$PY tools/identity_break.py --diff "$OUT/after.gpu.json" "$OUT/after.cpu.json" --lanes $LANES --require-columns 2 \
  > "$OUT/diff_gpu_cpu_after.txt" 2>&1; echo "DIFF gpu==cpu after exit $?"; tail -4 "$OUT/diff_gpu_cpu_after.txt"
echo "JOB END $(date -u +%FT%TZ)"
