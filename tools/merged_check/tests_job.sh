#!/bin/bash
# lane/merged tests: test_host_surface, test_lane_select, every test_x_*_repeat
# (the repeat tests at MOJOLEARN_CPU_THREADS 1, 3 and unset). $1 = out dir.
OUT=${1:-/root/ev-merged}; mkdir -p "$OUT"
cd "$(dirname "$0")/../.."
until [ -f "$OUT/build.done" ]; do sleep 30; done
grep -q "^exit 0" "$OUT/build.done" || { echo "BUILD NOT OK"; exit 1; }
PIXI=/root/.pixi/bin/pixi
$PIXI install -e test > "$OUT/pixi_test_install.log" 2>&1 || { echo "TEST ENV FAIL"; exit 1; }
export PYTHONPATH=$PWD/python MOJOLEARN_NUMERIC_MODE=identical
rc=0
.pixi/envs/test/bin/python -m pytest python/mojolearn/tests/test_host_surface.py -q > "$OUT/t_host_surface.log" 2>&1; r=$?; echo "test_host_surface exit $r: $(tail -1 "$OUT/t_host_surface.log")"; rc=$((rc|r))
$PIXI run -e default python tools/test_lane_select.py > "$OUT/t_lane_select.log" 2>&1; r=$?; echo "test_lane_select exit $r: $(tail -1 "$OUT/t_lane_select.log")"; rc=$((rc|r))
for t in python/mojolearn/tests/test_x_*_repeat.py; do
  for th in 1 3 default; do
    if [ $th = default ]; then unset MOJOLEARN_CPU_THREADS; else export MOJOLEARN_CPU_THREADS=$th; fi
    n=$(basename $t .py)_t$th
    .pixi/envs/test/bin/python -m pytest $t -q > "$OUT/$n.log" 2>&1; r=$?
    echo "$n exit $r: $(tail -1 "$OUT/$n.log")"; rc=$((rc|r))
  done
done
unset MOJOLEARN_CPU_THREADS
echo "TESTS RESULT: rc=$rc"
exit $rc
