#!/bin/bash
# usage: b.sh <tag> <mode fast|identical> [defines...]
cd /Users/andrewhendel/mojolearn-wt/af-sym-ctr
tag=$1; mode=$2; shift 2
export TMPDIR=/Users/andrewhendel/mojolearn-wt/af-sym-ctr/.buildtmp
export MOJOLEARN_COMPILE_JOBS=1 MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_NUMERIC_MODE=$mode
export MOJOLEARN_EXTRA_DEFINES="$*"
bash ~/mojolearn-evidence/compile_slot.sh bash bindings/build_gbdt.sh > .buildtmp/build-$tag.log 2>&1
echo "rc=$? tag=$tag defines=$*" >> .buildtmp/summary.txt
