#!/bin/sh
set -eu
cd /Users/andrewhendel/mojolearn-wt/cpu-kernel-identity
out=/Users/andrewhendel/mojolearn-evidence/next-wheel-coverage/cpu-kernel-identity
mojo=/Users/andrewhendel/mojolearn-wt/release-087-final/.pixi/envs/default/bin/mojo
sdk=$(xcrun --sdk macosx --show-sdk-version)
unset MACOSX_DEPLOYMENT_TARGET
for arm in clean sabotage; do
  mkdir -p "$out/$arm"
  test ! -e "$out/$arm/_mojolearn_kernel_methods_host.so"
  if [ "$arm" = sabotage ]; then set -- -D MOJOLEARN_HOST_SABOTAGE=1; else set --; fi
  "$mojo" build -j 1 --emit shared-lib --target-cpu apple-m1 -Xlinker -platform_version -Xlinker macos -Xlinker 11.0 -Xlinker "$sdk" "$@" -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_CPU -I . -I bindings bindings/_mojolearn_kernel_methods_host.mojo -o "$out/$arm/_mojolearn_kernel_methods_host.so" > "$out/$arm/build.log" 2>&1
  echo "BUILT $arm"
done
