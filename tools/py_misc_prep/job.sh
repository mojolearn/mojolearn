#!/bin/bash
# lane py-misc-prep: ONE light job. Build the touched bindings once, then
# (1) the lanes' GPU == CPU check on the new route (algos_lane_check),
# (2) every covering lane on the old route (MOJOLEARN_HOTPATH=python) and the
#     new route, GPU and CPU, compared cell for cell,
# (3) the targeted A/B bits (ab.py bits) GPU and CPU,
# (4) the timings (ab.py time) GPU and x86 CPU.
cd "$(dirname "$0")/../.."
OUT=${1:-/root/ev-py-misc-prep}; mkdir -p "$OUT"
LANES=x-prep-iterative-imputer,x-prep-iterative-options,x-prep-user-objects,trees-calibrated,trees-oob-cv-link
PIXI=$(command -v pixi || echo ~/.pixi/bin/pixi)
PY="$PIXI run -e default python -u"
echo "$(date -u +%FT%TZ) $(hostname) $(git rev-parse --short HEAD) $(nproc) cpus; $(lscpu | grep 'Model name' | head -1)"
$PIXI install -e default > "$OUT/pixi.log" 2>&1 || { tail -5 "$OUT/pixi.log"; exit 1; }
echo "== 1 lane check (new route, GPU vs CPU)"
tools/algos_lane_check.sh $LANES --out "$OUT/lc" > "$OUT/lc.log" 2>&1; echo "lane check exit $?"; grep -E 'RESULT|AGREE|DISAGREE|FAIL' "$OUT/lc.log" | tail -12
HOST="$PWD/python/mojolearn/host"
export PYTHONPATH="$PWD/python" MOJOLEARN_NUMERIC_MODE=identical OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
CPUENV="MOJOLEARN_VENDOR=cpu MOJOLEARN_HOST_DIR=$HOST MOJOLEARN_FOREST_HOST_BINARY=$HOST/_mojolearn_forest_host.so MOJOLEARN_BYTE_LM_HOST_BINARY=$HOST/_mojolearn_byte_lm_host.so"
echo "== 2 lanes, old route vs new route"
for col in gpu cpu; do
  if [ $col = gpu ]; then E=""; R="--require-backend cuda"; else E="$CPUENV"; R="--require-cpu --require-backend cpu"; fi
  env $E MOJOLEARN_HOTPATH=python $PY tools/identity_break.py --lanes $LANES --repeats 1 --fail-on-refused $R --json "$OUT/$col.before.json" > "$OUT/$col.before.log" 2>&1; echo "$col before exit $?"
  env $E $PY tools/identity_break.py --lanes $LANES --repeats 1 --fail-on-refused $R --json "$OUT/$col.after.json" > "$OUT/$col.after.log" 2>&1; echo "$col after exit $?"
  $PY tools/py_misc_prep/cmp_cells.py "$OUT/$col.before.json" "$OUT/$col.after.json" $LANES | tail -4 | sed "s/^/$col: /"
done
echo "== 3 targeted A/B bits"
$PY tools/py_misc_prep/ab.py bits > "$OUT/bits.gpu.log" 2>&1; echo "gpu bits exit $?"; tail -2 "$OUT/bits.gpu.log"
env $CPUENV $PY tools/py_misc_prep/ab.py bits > "$OUT/bits.cpu.log" 2>&1; echo "cpu bits exit $?"; tail -2 "$OUT/bits.cpu.log"
echo "== 4 timings (1M rows)"
$PY tools/py_misc_prep/ab.py time --rows ${ROWS:-1000000} > "$OUT/time.gpu.log" 2>&1; echo "gpu time exit $?"; grep TIME "$OUT/time.gpu.log"
env $CPUENV $PY tools/py_misc_prep/ab.py time --rows ${ROWS:-1000000} > "$OUT/time.cpu.log" 2>&1; echo "cpu time exit $?"; grep TIME "$OUT/time.cpu.log"
echo "JOB END $(date -u +%FT%TZ)"
