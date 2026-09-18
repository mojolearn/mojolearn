#!/bin/sh
set -eu
cd /Users/andrewhendel/mojolearn-wt/cpu-proof-followup
out=/Users/andrewhendel/mojolearn-evidence/next-wheel-coverage/cpu-proof-followup
release=/Users/andrewhendel/mojolearn-wt/release-087-final
mkdir -p "$out/host-sab"
unset MACOSX_DEPLOYMENT_TARGET
sdk=$(xcrun --sdk macosx --show-sdk-version)
for family in core byte_lm; do
  test ! -e "$out/host-sab/_mojolearn_${family}_host.so"
  if [ "$family" = byte_lm ]; then set -- -D MOJOLEARN_BYTE_LM_HOST_SABOTAGE=1; else set --; fi
  mojo build -j 1 --emit shared-lib --target-cpu apple-m1 -Xlinker -platform_version -Xlinker macos -Xlinker 11.0 -Xlinker "$sdk" "$@" -D MOJOLEARN_HOST_SABOTAGE=1 -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_CPU -I . -I bindings "bindings/_mojolearn_${family}_host.mojo" -o "$out/host-sab/_mojolearn_${family}_host.so" > "$out/build-$family.log" 2>&1
  echo "BUILT $family"
done
for binding in "$release"/python/mojolearn/host/*.so; do
  name=$(basename "$binding")
  test -e "$out/host-sab/$name" || ln -s "$binding" "$out/host-sab/$name"
done
export PYTHONPATH="$PWD/python:$PWD/tools" MOJOLEARN_NUMERIC_MODE=identical
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MOJOLEARN_CPU_THREADS=1
export MOJOLEARN_HOST_DIR="$release/python/mojolearn/host"
lanes=par-reference-knn,par-reference-knn-reg,byte-lm-host-infer
"$release/.pixi/envs/test/bin/python" tools/identity_break.py --require-backend cpu --vendor cpu-apple-m4 --lanes "$lanes" --fixtures ties --repeats 2 --json "$out/clean.json" > "$out/clean.log" 2>&1
export MOJOLEARN_HOST_DIR="$out/host-sab" MOJOLEARN_HOST_ALLOW_SABOTAGE=1
export MOJOLEARN_BYTE_LM_HOST_BINARY="$out/host-sab/_mojolearn_byte_lm_host.so" MOJOLEARN_BYTE_LM_HOST_ALLOW_SABOTAGE=1
code=0
"$release/.pixi/envs/test/bin/python" tools/identity_break.py --require-backend cpu --vendor cpu-apple-m4 --lanes "$lanes" --fixtures ties --repeats 2 --json "$out/sabotage.json" > "$out/sabotage.log" 2>&1 || code=$?
test "$code" = 1
"$release/.pixi/envs/test/bin/python" tools/cpu_identity_gate_check.py column "$out/sabotage.json" --sabotage --covered "$lanes" > "$out/control-check.log" 2>&1
cat "$out/control-check.log"
