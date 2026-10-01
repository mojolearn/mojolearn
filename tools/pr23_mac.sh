#!/bin/bash
# PR #23 merged with main, on the M3 Ultra (Metal): svd main vs main+#23 (new U) vs MOJOLEARN_LINALG_SVD_U=householder. No `timeout`.
set -u
while [ ! -f $HOME/pr40c-metal/done ]; do sleep 60; done
O=$HOME/pr23-metal; rm -rf $O; mkdir -p $O
export PATH=$HOME/.pixi/bin:$PATH MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=apple PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE; unset PYTHONPATH
rc() { echo "$(date -u +%T) $1 rc=$2" >> $O/rc.txt; }
TV=$HOME/torchvenv
cd $HOME/mojolearn; git fetch -q origin lane/neural-pass17-measure; rm -rf $HOME/mojolearn-p17; git worktree prune; git worktree add -f $HOME/mojolearn-p17 FETCH_HEAD >/dev/null 2>&1
for d in mainnow p17; do (cd $HOME/mojolearn-$d && git rev-parse HEAD > $O/head-$d.txt); done
cd $HOME/mojolearn-p17; pixi install > $O/pixi.log 2>&1
mkdir -p python/mojolearn/.dylibs; PYTHONPATH=$PWD/packaging/portable_math pixi run python -c "import pathlib, stage; stage.build(pathlib.Path('$PWD/python/mojolearn/.dylibs/libMojolearnMath.dylib'))" > $O/math.log 2>&1
for b in build build_x_decomp build_linalg; do pixi run bash bindings/$b.sh > $O/$b.log 2>&1; rc $b $?; done
PYTHONPATH=python $TV/bin/python tools/bench_board_algos.py prep --data $O/data --lanes svd --datasets taxi,istella > $O/prep.log 2>&1; rc prep $?
race() { cd $HOME/mojolearn-$1; env $5 MOJOLEARN_BENCH_INSTALLED=0 PYTHONPATH=$PWD/python $TV/bin/python tools/bench_board_algos.py race --lane $2 --dataset $3 --data $O/data --arms ours --rounds 3 --out $O/race-$4 --work $O/work --ours-python $TV/bin/python > $O/race-$4.log 2>&1; rc race-$4 $?
  echo "$4 $(grep -oE "ALGOS-ROUND.*" $O/race-$4.log | tail -1 | grep -oE " ms=[0-9.]+|digest=[0-9a-f]+" | tr "\n" " ") $(grep -o "ALGOS-REFUSED.\{0,140\}" $O/race-$4.log | head -1)" >> $O/races.txt; }
for ds in taxi istella; do race mainnow svd $ds svd-$ds-main X=1; race p17 svd $ds svd-$ds-branch X=1; race p17 svd $ds svd-$ds-householder MOJOLEARN_LINALG_SVD_U=householder; done
echo done > $O/done
