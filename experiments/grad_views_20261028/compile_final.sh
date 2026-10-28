#!/bin/bash
# usage: compile_final.sh cpu|sm89 off|on
tgt=$1; arm=$2; extra=""; [ "$arm" = on ] && extra="-D MOJOLEARN_IDN_PARAM_GRAD_VIEWS=1"
if [ "$tgt" = cpu ]; then tflags="--target-cpu apple-m1 -D MOJOLEARN_COLUMN_CPU"; else tflags="--target-accelerator sm_89 -D MOJOLEARN_COLUMN_NVIDIA"; fi
d=/Users/andrewhendel/mojolearn-evidence/grad-views-20261028/final-$tgt-$arm; mkdir -p $d
git -C /Users/andrewhendel/mojolearn-wt/grad-views-20261028 rev-parse HEAD > $d/source-commit.txt
start=$(date +%s)
pixi run --manifest-path /Users/andrewhendel/CascadeProjects/mojolearn/pixi.toml --environment default --locked --no-install --executable /Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/default/bin/mojo build --optimization-level 0 -j 10 --emit shared-lib $tflags -D MOJOLEARN_NUMERIC_IDENTICAL=1 $extra -I /Users/andrewhendel/mojolearn-wt/grad-views-20261028 -I /Users/andrewhendel/mojolearn-wt/grad-views-20261028/bindings /Users/andrewhendel/mojolearn-wt/grad-views-20261028/bindings/entry/entry__mojolearn_byte_lm.mojo -o $d/_mojolearn_byte_lm.so > $d/compile.log 2>&1
rc=$?
echo $rc > $d/exit; echo $(( $(date +%s) - start )) > $d/seconds
[ $rc = 0 ] && shasum -a 256 $d/_mojolearn_byte_lm.so > $d/sha256
exit $rc
