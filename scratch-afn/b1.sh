#!/bin/bash
# one build: tag binding mode defines
cd ~/mojolearn-wt/afn-samba || exit 2
tag=$1 b=$2 mode=$3 defs=$4
MOJOLEARN_COMPILE_JOBS=1 MOJOLEARN_NUMERIC_MODE=$mode MOJOLEARN_MOJO_BUILD_FLAGS="$defs" \
  bash ~/mojolearn-evidence/compile_slot.sh bash bindings/build_$b.sh > scratch-afn/build-$tag.log 2>&1
rc=$?
echo "$tag rc=$rc" >> scratch-afn/SUMMARY.txt
echo "$tag rc=$rc"
