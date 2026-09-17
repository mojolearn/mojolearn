#!/bin/bash
set -euo pipefail
unset MOJOLEARN_GPU_ARCHS MACOSX_DEPLOYMENT_TARGET
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd /Users/andrewhendel/mojolearn-wt/cpu-verification-completion
sdk=$(xcrun --sdk macosx --show-sdk-version)
base=/Users/andrewhendel/mojolearn-evidence/cpu-verification-completion
for family in forest gbdt; do
    arm=clean
    mkdir -p "$base/mac-$family-$arm"
    flags=()
    if [ "$arm" = sabotage ]; then flags=(-D MOJOLEARN_HOST_SABOTAGE=1); fi
    /Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/default/bin/mojo build -j 1 --emit shared-lib --target-cpu apple-m1 \
        -Xlinker -platform_version -Xlinker macos -Xlinker 11.0 -Xlinker "$sdk" \
        -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_CPU "${flags[@]}" -I . -I bindings \
        bindings/_mojolearn_${family}_host.mojo -o "$base/mac-$family-$arm/_mojolearn_${family}_host.so"
done
