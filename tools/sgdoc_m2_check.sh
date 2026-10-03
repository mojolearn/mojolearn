#!/bin/bash
# sgdoc_m2_check.sh: lane/apple-fast-sgdoc-parallel's M2 verification (no timing): scikit-learn's
# SGDOneClassSVM objective and seed spread on rows-small, then the FAST x_linear binding with
# -D MOJOLEARN_SGDOC_FAST_PAR against it and a float64 Frank-Wolfe reference (tools/sgdoc_check.py).
set -u
for b in board-0834 board-0833; do [ -f $HOME/$b/cache/algos-data/rows-small/SMALL_ROWS ] && { B=$HOME/$b; break; }; done
VP=$B/cache/venv/bin/python; DATA=$B/cache/algos-data/rows-small
for ds in istella taxi; do $VP tools/sgdoc_objective.py $DATA $ds 7,0,1,2,3 2>&1 | grep -E 'SGDOC|Error|error' | head -8; done
MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_MOJO_BUILD_FLAGS="-D MOJOLEARN_SGDOC_FAST_PAR" MOJOLEARN_SKIP_BUILD_GATE=1 \
  pixi run bash bindings/build_x_linear.sh > /tmp/sgdoc_build.log 2>&1; echo "SGDOC-BUILD rc=$?"
grep -m 5 -B 2 -A 6 -i ' error' /tmp/sgdoc_build.log | cut -c1-300
for ds in istella taxi; do
  MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_BENCH_INSTALLED=0 PYTHONPATH=$PWD/python $VP tools/sgdoc_check.py $DATA $ds 2>&1 \
    | grep -E 'SGDOC|Error|error|Traceback' | head -8
done
