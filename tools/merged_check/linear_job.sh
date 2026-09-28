#!/bin/bash
# the 32 x_linear lanes at the Metal-fix commit: CUDA vs CPU at threads 1/3/default
cd "$(dirname "$0")/../.."
OUT=/root/ev-merged-lin; mkdir -p $OUT
export MOJOLEARN_BUILD_LOCK_HELD=1 MOJOLEARN_COMPILE_JOBS=2 MOJOLEARN_GPU_ARCHS=sm_89
P="/root/.pixi/bin/pixi run -e default python -u tools/merged_check/merged_check.py"
/root/.pixi/bin/pixi install -e default > $OUT/pixi_install.log 2>&1
$P plan --out $OUT > $OUT/plan.log 2>&1
python3 - $OUT <<'PY'
import json, sys
out = sys.argv[1]; lanes = open("tools/merged_check/linear_lanes.txt").read().split()
p = json.load(open(out + "/plan.json")); p["lanes"] = lanes
p["bindings"] = sorted(set().union(*[p["needed"][l] for l in lanes]))
json.dump(p, open(out + "/plan.json", "w"))
PY
$P build --out $OUT --jobs 6 || { echo "BUILD NOT OK"; exit 1; }
$P clean --out $OUT --shard 0/1 --cpu-threads 1,3,default
