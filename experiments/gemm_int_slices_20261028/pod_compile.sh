#!/bin/bash
# lane/gemm-int-slices compile closure on one RunPod CPU box (no GPU): NVIDIA sm_89 and AMD gfx942 targets.
# Run by tools/runpod_cpu_leg.sh inside the repo; $LEG_OUT comes back to the Mac. Compile only, no run.
set -u
J=$(python3 -c 'import os; print(len(os.sched_getaffinity(0)))' 2>/dev/null || nproc)
pixi run mojo --version > "$LEG_OUT/compiler.txt" 2>&1
git rev-parse HEAD > "$LEG_OUT/commit.txt" 2>/dev/null || true
echo "affinity_jobs=$J nproc=$(nproc)" > "$LEG_OUT/alloc.txt"
cat /sys/fs/cgroup/cpu.max >> "$LEG_OUT/alloc.txt" 2>/dev/null || true
build() {
  tag=$1; arch=$2; col=$3; emit=$4; entry=$5; shift 5
  s=$(date +%s)
  out="$LEG_OUT/$tag.bin"
  pixi run mojo build --optimization-level 0 -j "$J" $emit --target-cpu x86-64-v3 --target-accelerator "$arch" \
    -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D "$col" "$@" -I . -I bindings "$entry" -o "$out" > "$LEG_OUT/$tag.log" 2>&1
  rc=$?
  sha=none; [ -f "$out" ] && sha=$(sha256sum "$out" | cut -d' ' -f1)
  echo "tag=$tag arch=$arch entry=$entry defines=$* rc=$rc seconds=$(( $(date +%s) - s )) sha256=$sha" >> "$LEG_OUT/receipts.txt"
}
ON=-DMOJOLEARN_IDN_GEMM_INT_SLICES=1
build check_sm89      sm_89  MOJOLEARN_COLUMN_NVIDIA "" gemm/checks/gemm_int_slices_check.mojo
build check_gfx942    gfx942 MOJOLEARN_COLUMN_AMD    "" gemm/checks/gemm_int_slices_check.mojo
build linalg_sm89_on  sm_89  MOJOLEARN_COLUMN_NVIDIA "--emit shared-lib" bindings/entry/entry__mojolearn_linalg.mojo $ON
build linalg_gfx942_on gfx942 MOJOLEARN_COLUMN_AMD   "--emit shared-lib" bindings/entry/entry__mojolearn_linalg.mojo $ON
build linalg_sm89_off sm_89  MOJOLEARN_COLUMN_NVIDIA "--emit shared-lib" bindings/entry/entry__mojolearn_linalg.mojo
build linalg_gfx942_off gfx942 MOJOLEARN_COLUMN_AMD  "--emit shared-lib" bindings/entry/entry__mojolearn_linalg.mojo
cat "$LEG_OUT/receipts.txt"
