#!/bin/bash
# Route A via COMGR only: downgraded Mojo IR -> LLVM 19 bitcode (opt) ->
# COMGR codegen+link for each target -> (gfx942) hipModuleLoadData -> bits.
# Also the textual-IR-as-BC variant (does COMGR parse text IR?).
set -u
H=/root/lq/amd-portable; L=/opt/rocm/llvm/bin; R=$H/route_a_comgr
mkdir -p $R; cd $H
gcc -O2 -I/opt/rocm/include -o comgr_build comgr_build.c -L/opt/rocm/lib -lamd_comgr -Wl,-rpath,/opt/rocm/lib > $R/gcc.log 2>&1 || { echo "gcc rc=$?" > $R/summary.txt; exit 1; }
: > $R/summary.txt
for k in k_muladd k_pinned k_math k_dot; do
  ll=route_a/$k.llvm-opt.down.ll
  sym=$(grep -m1 -oE 'define [^@]*amdgpu_kernel void @[A-Za-z0-9_$.]+' $ll | sed 's/.*@//')
  $L/opt -o $R/$k.bc $ll > $R/$k.opt.log 2>&1; echo "BC $k opt rc=$?" >> $R/summary.txt
  for t in gfx942 gfx90a gfx1100 gfx1201; do
    ./comgr_build bc $R/$k.bc $t $R/$k.$t.hsaco -O3 > $R/$k.$t.comgr.log 2>&1
    echo "COMGR-BC $k $t rc=$?" >> $R/summary.txt
  done
  ./comgr_build bc $ll gfx942 $R/$k.text.gfx942.hsaco -O3 > $R/$k.text.comgr.log 2>&1
  echo "COMGR-TEXTIR $k gfx942 rc=$?" >> $R/summary.txt
  for v in $k.gfx942 $k.text.gfx942; do
    [ -f $R/$v.hsaco ] || continue
    ./hip_load $k $R/$v.hsaco "$sym" > $R/$v.run.out 2> $R/$v.run.err; rc=$?
    if cmp -s route_a/$k.native.out $R/$v.run.out; then m=MATCH; else m="DIFFER($(diff route_a/$k.native.out $R/$v.run.out | grep -c '^<'))"; fi
    echo "RUN $v rc=$rc native-vs-comgr=$m" >> $R/summary.txt
  done
done
grep -m1 'comgr version' $R/k_muladd.gfx942.comgr.log >> $R/summary.txt
cat $R/summary.txt
