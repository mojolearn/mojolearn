#!/bin/bash
# Route A probe on the AMD box: Mojo-emitted IR -> ROCm's own LLVM (llc +
# ld.lld) -> code object -> hipModuleLoadData -> bits vs Mojo's native launch.
# Also cross-compiles the same gfx942 IR for other targets (compile only).
# Output: /root/lq/amd-portable/route_a/{summary.txt,...}
set -u
H=/root/lq/amd-portable; L=/opt/rocm/llvm/bin; R=$H/route_a
mkdir -p $R; cd $H
/opt/rocm/bin/hipcc -O2 -o hip_load hip_load.cpp > $R/hipcc.log 2>&1 || { echo "hipcc rc=$?" > $R/summary.txt; exit 1; }
: > $R/summary.txt
for kind in llvm llvm-opt; do
  for k in k_muladd k_pinned k_math k_dot; do
    ll=emit/$k.gfx942.$kind.ll
    sym=$(grep -m1 -oE 'define [^@]*amdgpu_kernel void @[A-Za-z0-9_$.]+' $ll | sed 's/.*@//')
    python3 downgrade_ll.py $ll $R/$k.$kind.down.ll 2> $R/$k.$kind.rules
    for t in gfx942 gfx90a gfx1100; do
      feat=""; [ $t = gfx1100 ] && feat="-mattr=+wavefrontsize64"
      $L/llc -O3 -mtriple=amdgcn-amd-amdhsa -mcpu=$t $feat -filetype=obj -o $R/$k.$kind.$t.o $R/$k.$kind.down.ll > $R/$k.$kind.$t.llc.log 2>&1; rc=$?
      [ $rc = 0 ] && { $L/ld.lld -shared -o $R/$k.$kind.$t.hsaco $R/$k.$kind.$t.o > $R/$k.$kind.$t.lld.log 2>&1; rc=$?; }
      echo "BUILD $k $kind $t rc=$rc rules=$(tr '\n' ';' < $R/$k.$kind.rules)" >> $R/summary.txt
    done
    ./hip_load $k $R/$k.$kind.gfx942.hsaco "$sym" > $R/$k.$kind.run.out 2> $R/$k.$kind.run.err; rc=$?
    awk -v k="=== $k" '$0==k{f=1;print;next} /^===/{f=0} f' native.out > $R/$k.native.out
    if cmp -s $R/$k.native.out $R/$k.$kind.run.out; then v=MATCH; else v="DIFFER($(diff $R/$k.native.out $R/$k.$kind.run.out | grep -c '^<') words)"; fi
    echo "RUN $k $kind gfx942 rc=$rc native-vs-rocm19=$v" >> $R/summary.txt
  done
done
cat $R/summary.txt
