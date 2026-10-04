#!/bin/bash
# Route B feasibility on the AMD box: Mojo IR -> spirv64-amd-amdhsa IR ->
# AMDGCN-flavoured SPIR-V (ROCm opt + amd-llvm-spirv, the flags clang uses) -> HIP offload bundle ->
# hipModuleLoadData (runtime JIT) -> bits vs native.
set -u
H=/root/lq/amd-portable; R=$H/route_b; C=/opt/rocm/llvm/bin
cd $H; : > $R/mojo_summary.txt
for k in k_muladd k_pinned k_math k_dot; do
  ll=route_a/$k.llvm-opt.down.ll
  sym=$(grep -m1 -oE 'define [^@]*amdgpu_kernel void @[A-Za-z0-9_$.]+' $ll | sed 's/.*@//')
  python3 mojo_ll_to_spirv_ir.py $ll $R/$k.spirv.ll > $R/$k.rewrite.log 2>&1; rc=$?
  echo "REWRITE $k rc=$rc $(head -c 150 $R/$k.rewrite.log)" >> $R/mojo_summary.txt
  [ $rc = 0 ] || continue
  { $C/opt -o $R/$k.spirv.bc $R/$k.spirv.ll && $C/amd-llvm-spirv --spirv-max-version=1.6 --spirv-ext=+all --spirv-allow-unknown-intrinsics --spirv-lower-const-expr --spirv-preserve-auxdata $R/$k.spirv.bc -o $R/$k.spv; } > $R/$k.spv.log 2>&1; rc=$?
  echo "SPIRV $k rc=$rc $(grep -m1 -i error $R/$k.spv.log | cut -c1-200)" >> $R/mojo_summary.txt
  [ $rc = 0 ] || continue
  $C/clang-offload-bundler --type=o --targets=host-x86_64-unknown-linux--,hip-spirv64-amd-amdhsa--amdgcnspirv \
    --input=/dev/null --input=$R/$k.spv --output=$R/$k.spv.co > $R/$k.bundle.log 2>&1
  ./hip_load $k $R/$k.spv.co "$sym" > $R/$k.spv.run.out 2> $R/$k.spv.run.err; rc=$?
  if cmp -s route_a/$k.native.out $R/$k.spv.run.out; then m=MATCH; else m="DIFFER($(diff route_a/$k.native.out $R/$k.spv.run.out | grep -c '^<'))"; fi
  echo "RUN $k spirv-jit rc=$rc native-vs-spirv=$m $(head -c 200 $R/$k.spv.run.err | tr '\n' ' ')" >> $R/mojo_summary.txt
done
cat $R/mojo_summary.txt
