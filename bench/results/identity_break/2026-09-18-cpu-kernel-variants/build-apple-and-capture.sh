#!/bin/sh
set -eu
cd /Users/andrewhendel/mojolearn-wt/cpu-kernel-identity
out=/Users/andrewhendel/mojolearn-evidence/next-wheel-coverage/cpu-kernel-identity
mkdir -p "$out/apple"
test ! -e "$out/apple/_mojolearn_kernel_methods.so"
unset MACOSX_DEPLOYMENT_TARGET
sdk=$(xcrun --sdk macosx --show-sdk-version)
mojo build -j 1 --emit shared-lib --target-cpu apple-m1 -Xlinker -platform_version -Xlinker macos -Xlinker 11.0 -Xlinker "$sdk" -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_APPLE -I . -I bindings bindings/_mojolearn_kernel_methods.mojo -o "$out/apple/_mojolearn_kernel_methods.so" > "$out/apple/build.log" 2>&1
python3 -u "$out/capture.py"
