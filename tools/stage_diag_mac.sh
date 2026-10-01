#!/bin/bash
# Same per-stage timing on the M3 Ultra (Metal), main source tree; no `timeout` on macOS.
set -u
O=$HOME/stage-diag-metal; rm -rf $O; mkdir -p $O
export PATH=$HOME/.pixi/bin:$PATH MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=apple PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE; unset PYTHONPATH
rc() { echo "$(date -u +%T) $1 rc=$2" >> $O/rc.txt; }
cd $HOME/mojolearn; git fetch -q origin lane/stage-diag-20261001; rm -rf $HOME/mojolearn-sdiag; git worktree prune; git worktree add -f $HOME/mojolearn-sdiag FETCH_HEAD >/dev/null 2>&1
cd $HOME/mojolearn-sdiag; git rev-parse HEAD > $O/head.txt; pixi install > $O/pixi.log 2>&1
mkdir -p python/mojolearn/.dylibs; PYTHONPATH=$PWD/packaging/portable_math pixi run python -c "import pathlib, stage; stage.build(pathlib.Path('$PWD/python/mojolearn/.dylibs/libMojolearnMath.dylib'))" > $O/math.log 2>&1
for b in build build_transformer build_byte_lm build_training build_mamba; do pixi run bash bindings/$b.sh > $O/$b.log 2>&1; rc $b $?; done
PYTHONPATH=$PWD/python $HOME/torchvenv/bin/python tools/neural_stage_timing.py --lane lm-forward --lane lm-train-step --lane transformer-forward --calls 6 > $O/stage-timing.log 2>&1; rc stage-timing $?
echo done > $O/done
