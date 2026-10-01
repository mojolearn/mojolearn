#!/bin/bash
# PR #40 merged with main, on the M3 Ultra (Metal): lu-factor/lu-solve main vs main+#40 default vs MOJOLEARN_XD_LU_FUSED=1. No `timeout`.
set -u
O=$HOME/pr40c-metal; rm -rf $O; mkdir -p $O
export PATH=$HOME/.pixi/bin:$PATH MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=apple PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE; unset PYTHONPATH
rc() { echo "$(date -u +%T) $1 rc=$2" >> $O/rc.txt; }
TV=$HOME/torchvenv
cd $HOME/mojolearn; git fetch -q origin lane/neural-pass36-measure; rm -rf $HOME/mojolearn-p36main; git worktree prune; git worktree add -f $HOME/mojolearn-p36main FETCH_HEAD >/dev/null 2>&1
for d in mainnow p36main; do (cd $HOME/mojolearn-$d && git rev-parse HEAD > $O/head-$d.txt); done
cd $HOME/mojolearn-p36main; pixi install > $O/pixi.log 2>&1
mkdir -p python/mojolearn/.dylibs; PYTHONPATH=$PWD/packaging/portable_math pixi run python -c "import pathlib, stage; stage.build(pathlib.Path('$PWD/python/mojolearn/.dylibs/libMojolearnMath.dylib'))" > $O/math.log 2>&1
for b in build build_x_decomp build_linalg; do pixi run bash bindings/$b.sh > $O/$b.log 2>&1; rc $b $?; done
PYTHONPATH=python $TV/bin/python tools/bench_board_algos.py prep --data $O/data --lanes lu-factor,lu-solve --datasets synthetic > $O/prep.log 2>&1; rc prep $?
race() { cd $HOME/mojolearn-$1; env $5 MOJOLEARN_BENCH_INSTALLED=0 PYTHONPATH=$PWD/python $TV/bin/python tools/bench_board_algos.py race --lane $2 --dataset $3 --data $O/data --arms ours --rounds 3 --out $O/race-$4 --work $O/work --ours-python $TV/bin/python > $O/race-$4.log 2>&1; rc race-$4 $?
  echo "$4 $(grep -oE "ALGOS-ROUND.*" $O/race-$4.log | tail -1 | grep -oE " ms=[0-9.]+|digest=[0-9a-f]+" | tr "\n" " ") $(grep -o "ALGOS-REFUSED.\{0,140\}" $O/race-$4.log | head -1)" >> $O/races.txt; }
for l in lu-factor lu-solve; do race mainnow $l synthetic $l-main X=1; race p36main $l synthetic $l-branch X=1; race p36main $l synthetic $l-fused MOJOLEARN_XD_LU_FUSED=1; done
echo done > $O/done
