#!/bin/bash
# M3 Ultra (Metal), main vs branch source trees, sequential. PR #44: per S16 arm the Mamba3Block fixture
# digests (2x512x384), the backward stage walls, and neural_experiments s16_apple. PR #40: lu-factor/lu-solve
# main vs branch default vs MOJOLEARN_XD_LU_FUSED=1. No `timeout` on macOS.
set -u
O=$HOME/pr44-40-metal; rm -rf $O; mkdir -p $O
export PATH=$HOME/.pixi/bin:$PATH MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=apple PYTHONUNBUFFERED=1 KMP_DUPLICATE_LIB_OK=TRUE; unset PYTHONPATH
rc() { echo "$(date -u +%T) $1 rc=$2" >> $O/rc.txt; }
TV=$HOME/torchvenv
cd $HOME/mojolearn; for r in main-now:mainnow lane/neural-apple3-measure:apple3 lane/neural-pass36:pass36; do b=${r%%:*}; d=${r##*:}; git fetch -q origin $b; rm -rf $HOME/mojolearn-$d; git worktree prune; git worktree add -f $HOME/mojolearn-$d FETCH_HEAD >/dev/null 2>&1; (cd $HOME/mojolearn-$d && git rev-parse HEAD > $O/head-$d.txt); done
bld() { cd $HOME/mojolearn-$1; pixi install > $O/pixi-$1.log 2>&1
  mkdir -p python/mojolearn/.dylibs; PYTHONPATH=$PWD/packaging/portable_math pixi run python -c "import pathlib, stage; stage.build(pathlib.Path('$PWD/python/mojolearn/.dylibs/libMojolearnMath.dylib'))" > $O/math-$1.log 2>&1
  shift; for b in "$@"; do pixi run bash bindings/$b.sh > $O/$b-$T.log 2>&1; rc $b-$T $?; done; }
T=mainnow; bld mainnow build build_mamba build_x_decomp build_linalg
T=apple3; bld apple3 build build_mamba
T=pass36; bld pass36 build build_x_decomp build_linalg
# PR #44
m44() { cd $HOME/mojolearn-$1; ex=""; [ $2 != default ] && ex="MOJOLEARN_MAMBA3_S16_QK_ARM=$2"
  env $ex PYTHONPATH=$PWD/python $TV/bin/python $HOME/mojolearn-apple3/tools/strides_digest.py 2 512 384 > $O/digest-$1-$2.json 2> $O/digest-$1-$2.err; rc digest-$1-$2 $?
  env $ex MOJOLEARN_MAMBA_TIMING=1 PYTHONPATH=$PWD/python $TV/bin/python $HOME/mojolearn-apple3/tools/mamba3_backward_timing.py --batch 2 --length 512 --d-model 384 --calls 4 > $O/timing-$1-$2.log 2>&1; rc timing-$1-$2 $?; }
m44 mainnow default
for a in default naive regs2h regsh; do m44 apple3 $a; done
cd $HOME/mojolearn-apple3; PYTHONPATH=$PWD/python $TV/bin/python tools/neural_experiments.py --set s16_apple --lane mamba3-forward --lane samba-train-step --calls 10 --json $O/set-s16_apple.json > $O/set-s16_apple.log 2>&1; rc set-s16_apple $?
cd $HOME/mojolearn-mainnow; PYTHONPATH=$PWD/python $TV/bin/python $HOME/mojolearn-apple3/tools/neural_experiments.py --set s16_apple --only baseline --lane mamba3-forward --lane samba-train-step --calls 10 --json $O/set-main-baseline.json > $O/set-main-baseline.log 2>&1; rc set-main-baseline $?
# PR #40
cd $HOME/mojolearn-pass36; PYTHONPATH=python $TV/bin/python tools/bench_board_algos.py prep --data $O/data --lanes lu-factor,lu-solve --datasets synthetic > $O/prep.log 2>&1; rc prep $?
race() { cd $HOME/mojolearn-$1; env $5 MOJOLEARN_BENCH_INSTALLED=0 PYTHONPATH=$PWD/python $TV/bin/python tools/bench_board_algos.py race --lane $2 --dataset $3 --data $O/data --arms ours --rounds 3 --out $O/race-$4 --work $O/work --ours-python $TV/bin/python > $O/race-$4.log 2>&1; rc race-$4 $?
  echo "$4 $(grep -oE "ALGOS-ROUND.*" $O/race-$4.log | tail -1 | grep -oE " ms=[0-9.]+|digest=[0-9a-f]+" | tr "\n" " ") $(grep -o "ALGOS-REFUSED.\{0,140\}" $O/race-$4.log | head -1)" >> $O/races.txt; }
for l in lu-factor lu-solve; do race mainnow $l synthetic $l-main X=1; race pass36 $l synthetic $l-branch X=1; race pass36 $l synthetic $l-fused MOJOLEARN_XD_LU_FUSED=1; done
echo done > $O/done
