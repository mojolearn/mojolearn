#!/bin/bash
set -u
PY=/Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/test/bin/python
export PYTHONPATH=/Users/andrewhendel/mojolearn-wt/gpt3-tooling/python
cd /Users/andrewhendel/mojolearn-wt/gpt3-tooling
O=<scratchpad>/live; mkdir -p $O
COL=bench/results/par_lm_xvendor/2026-09-21/apple-m4-1gpu.json
$PY - <<PY
import sys; sys.path.insert(0, "tools")
import mojolearn as ml
from mojolearn.parallel_training import ParallelByteLanguageModelTrainer as Par
import par_lm_xvendor as recipe, argparse
args = argparse.Namespace(seed=20260921, batch=2, length=32, d_model=32, heads=4, kv=2, ff=64, layers=2, vocab=512, steps=2, shards=4)
shape, state0, ids = recipe.build_problem(ml, args)
t = Par(state0, devices=(0,), logical_shards=1, pool_optimizer=False)
print("has_device_fold:", t.has_device_fold)
t.close()
PY
echo "=== device fold: chained 2 workers (first block folds on the device) vs the recorded column"
$PY tools/live_xvendor.py local --workers 2 --port 7821 --chained --out $O/devfold2.json --expect $COL 2>&1 | tail -1
echo "=== device fold: chained 3 workers, unequal blocks"
$PY tools/live_xvendor.py local --workers 3 --port 7822 --chained --out $O/devfold3.json --expect $COL 2>&1 | tail -1
