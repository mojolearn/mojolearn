#!/bin/bash
# lane/apple2-merged, m2pro: every selected lane (mac_job.sh 0/1), then the prep
# mutual_info lanes' GPU arm again under the Metal validation layer (the M2 drops a
# dispatch above its pipeline max silently; dmi _tile_kernel is 512 threads + 16 KB).
cd "$(dirname "$0")/../.."
OUT=${1:?out}
bash tools/merged_check/mac_job.sh 0/1 "$OUT"
D="$OUT/mtl_debug"; mkdir -p "$D"
PIXI=$(command -v pixi || echo ~/.pixi/bin/pixi)
for lane in x-prep-mutual-info x-prep-mi-discrete; do
  PYTHONPATH=$PWD/python MOJOLEARN_NUMERIC_MODE=identical OMP_NUM_THREADS=1 MTL_DEBUG_LAYER=1 MTL_DEBUG_LAYER_ERROR_MODE=nslog $PIXI run -e default python -u tools/identity_break.py \
    --lanes $lane --repeats 1 --fail-on-refused --require-backend metal --json "$D/$lane.json" > "$D/$lane.log" 2>&1
  echo "MTL_DEBUG $lane rc=$? validation_lines=$(grep -c -i -E 'validation|MTLDebug|exceeds|maxTotalThreadsPerThreadgroup' "$D/$lane.log")"
  grep -i -E 'validation|MTLDebug|exceeds|maxTotalThreads' "$D/$lane.log" | head -5
done
echo "M2PRO JOB END $(date -u +%FT%TZ)"
