#!/bin/bash
# grad-views lane: compile the byte LM root for sm_89 (NVIDIA column), arm off and arm on.
set -u
cd /root/mojolearn || exit 2
jobs=$(python3 -c 'import os;print(len(os.sched_getaffinity(0)))')
git rev-parse HEAD > "$LEG_OUT/source-commit.txt" 2>/dev/null || true
.pixi/envs/default/bin/mojo --version > "$LEG_OUT/compiler.txt" 2>&1
sha256sum .pixi/envs/default/bin/mojo > "$LEG_OUT/compiler.sha256"
echo "$jobs" > "$LEG_OUT/jobs.txt"
final=0
for arm in off on; do
  extra=""; [ "$arm" = on ] && extra="-D MOJOLEARN_IDN_PARAM_GRAD_VIEWS=1"
  mkdir -p "$LEG_OUT/sm89-$arm"
  s=$(date +%s)
  pixi run --locked --no-install mojo build --optimization-level 0 -j "$jobs" --emit shared-lib \
    --target-accelerator sm_89 --target-cpu x86-64-v3 -D MOJOLEARN_COLUMN_NVIDIA \
    -D MOJOLEARN_NUMERIC_IDENTICAL=1 $extra -I . -I bindings \
    bindings/entry/entry__mojolearn_byte_lm.mojo -o "$LEG_OUT/sm89-$arm/_mojolearn_byte_lm.so" \
    > "$LEG_OUT/sm89-$arm/compile.log" 2>&1
  rc=$?
  echo $rc > "$LEG_OUT/sm89-$arm/exit"; echo $(( $(date +%s) - s )) > "$LEG_OUT/sm89-$arm/seconds"
  [ $rc = 0 ] && sha256sum "$LEG_OUT/sm89-$arm/_mojolearn_byte_lm.so" > "$LEG_OUT/sm89-$arm/sha256"
  [ $rc = 0 ] || final=$rc
done
exit $final
