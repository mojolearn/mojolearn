#!/bin/bash
set -uo pipefail
cd /Users/andrewhendel/mojolearn-wt/cpu-verification-completion
base=/Users/andrewhendel/mojolearn-evidence/cpu-verification-completion
export MOJOLEARN_NUMERIC_MODE=identical OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH="$PWD/python"
for arm in clean sabotage; do
    export MOJOLEARN_HOST_DIR="$base/mac-training-$arm"
    unset MOJOLEARN_HOST_ALLOW_SABOTAGE
    if [ "$arm" = sabotage ]; then export MOJOLEARN_HOST_ALLOW_SABOTAGE=1; fi
    /Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/test/bin/python tools/identity_break.py --require-cpu --lanes ordered-gradient-sum --repeats 2 --json "$base/ordered-sum-$arm.json" > "$base/ordered-sum-$arm.log" 2>&1
    echo "$arm exit=$?"
done
