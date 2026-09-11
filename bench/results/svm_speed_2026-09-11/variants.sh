#!/bin/bash
# svm-speed lane: build the SVM binding once per block-solve schedule into its own root,
# then launch probe + hashes + timing per variant. Runs ON THE POD.
# usage: bash variants.sh name:DEFINES [name:DEFINES ...]   (DEFINES comma separated, may be empty)
set -u
export PATH="$HOME/.pixi/bin:$PATH" MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_GPU_ARCHS=sm_90a
export MOJOLEARN_COMPILE_JOBS="$(nproc)"
R=/root/mojolearn
L=/root/lane
mkdir -p $L/var
for spec in "$@"; do
    name=${spec%%:*}
    defs=${spec#*:}
    extra=""
    for d in $(echo "$defs" | tr ',' ' '); do extra="$extra -D $d"; done
    V=$L/var/$name
    rm -rf $V; mkdir -p $V
    echo "== $name defines=[$extra] $(date -u +%T)" | tee -a $L/variants.progress
    ( cd $R && MOJOLEARN_BUILD_EXTRA_DEFINES="$extra" timeout -k 30 1500 sh bindings/build_svm.sh > $V/build.log 2>&1 )
    rc=$?
    echo "   build rc=$rc $(date -u +%T)" | tee -a $L/variants.progress
    [ $rc -eq 0 ] || { grep -m5 "error" $V/build.log | tee -a $L/variants.progress; continue; }
    mkdir -p $V/root
    cp -r $R/python $V/root/python
    cd /tmp
    timeout -k 30 600 python3 $L/svm_probe.py hash $V/root /root/ctd-data taxi istella > $V/hash.log 2>&1
    grep -E "^(HASH|ROOT)" $V/hash.log | tee -a $L/variants.progress
    grep -E "Error|error|Exception" $V/hash.log | grep -v mbind | head -3 | tee -a $L/variants.progress
    timeout -k 30 600 python3 $L/svm_probe.py stage $V/root /root/ctd-data taxi istella > $V/stage.log 2>&1
    grep -E "STAGE-BEGIN|block_solve|fit_total|outer iter" $V/stage.log | tee -a $L/variants.progress
done
echo "VARIANTS-DONE $(date -u +%T)" | tee -a $L/variants.progress
