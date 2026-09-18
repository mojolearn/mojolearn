#!/bin/sh
set -eu
cd /Users/andrewhendel/mojolearn-wt/cpu-kernel-identity
out=/Users/andrewhendel/mojolearn-evidence/next-wheel-coverage/cpu-kernel-identity
unset MACOSX_DEPLOYMENT_TARGET
sdk=$(xcrun --sdk macosx --show-sdk-version)
for arm in clean sabotage; do
  target="$out/$arm/_mojolearn_estimators_host.so"
  test -L "$target"
  if [ "$arm" = sabotage ]; then set -- -D MOJOLEARN_HOST_SABOTAGE=1; else set --; fi
  mojo build -j 1 --emit shared-lib --target-cpu apple-m1 -Xlinker -platform_version -Xlinker macos -Xlinker 11.0 -Xlinker "$sdk" "$@" -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_CPU -I . -I bindings bindings/_mojolearn_estimators_host.mojo -o "$out/$arm/estimators-fresh.so" > "$out/$arm/estimators-build.log" 2>&1
  mv "$out/$arm/estimators-fresh.so" "$target"
  echo "BUILT estimators $arm"
done
export MOJOLEARN_HOST_DIR="$out/clean"
export PYTHONPATH="$PWD/python:$PWD/tools"
export MOJOLEARN_NUMERIC_MODE=identical
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MOJOLEARN_CPU_THREADS=1
/Users/andrewhendel/mojolearn-wt/release-087-final/.pixi/envs/test/bin/python -m pytest python/mojolearn/tests/test_cpu_kernel_variants.py python/mojolearn/tests/test_cpu_training_d.py -q -ra > "$out/runtime-tests-fresh-estimators.log" 2>&1
cat "$out/runtime-tests-fresh-estimators.log"
