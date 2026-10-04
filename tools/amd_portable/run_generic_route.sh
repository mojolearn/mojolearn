#!/bin/bash
# Route C probe: the same Mojo IR compiled for LLVM's GENERIC AMD targets
# (code object v6 "family" ISAs). gfx9-4-generic covers gfx940/941/942/950, so
# the MI325X can load and run it; the others are compile-only here.
set -u
H=/root/lq/amd-portable; R=$H/route_c
mkdir -p $R; cd $H; : > $R/summary.txt
for k in k_muladd k_pinned k_math k_dot; do
  ll=route_a/$k.llvm-opt.down.ll
  sym=$(grep -m1 -oE 'define [^@]*amdgpu_kernel void @[A-Za-z0-9_$.]+' $ll | sed 's/.*@//')
  for t in gfx9-4-generic gfx9-generic gfx10-3-generic gfx11-generic gfx12-generic; do
    ./comgr_build bc route_a_comgr/$k.bc $t $R/$k.$t.hsaco -O3 > $R/$k.$t.comgr.log 2>&1
    echo "COMGR $k $t rc=$?" >> $R/summary.txt
  done
  ./hip_load $k $R/$k.gfx9-4-generic.hsaco "$sym" > $R/$k.run.out 2> $R/$k.run.err; rc=$?
  if cmp -s route_a/$k.native.out $R/$k.run.out; then m=MATCH; else m="DIFFER($(diff route_a/$k.native.out $R/$k.run.out | grep -c '^<'))"; fi
  echo "RUN $k gfx9-4-generic-on-gfx942 rc=$rc native-vs-generic=$m $(head -c 200 $R/$k.run.err | tr '\n' ' ')" >> $R/summary.txt
  /opt/rocm/llvm/bin/llvm-readelf -h $R/$k.gfx9-4-generic.hsaco 2>/dev/null | grep -m2 -E "Flags|ABI Version" | tr -s ' ' | tr '\n' ' ' >> $R/summary.txt; echo >> $R/summary.txt
done
cat $R/summary.txt
