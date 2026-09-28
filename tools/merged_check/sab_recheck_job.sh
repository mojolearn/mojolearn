#!/bin/bash
# after the recheck job (0007) is done with this tree: the sabotage lines whose
# lanes the old tree could not run (decomp from_input, kmeans CPU fixes)
cd "$(dirname "$0")/../.."
until grep -q "TESTS RESULT" /root/gpu-queue/0007/log 2>/dev/null || grep -qE "^(done|failed|cancelled)" /root/gpu-queue/0007/status 2>/dev/null; do sleep 60; done
export MOJOLEARN_BUILD_LOCK_HELD=1 MOJOLEARN_COMPILE_JOBS=2 MOJOLEARN_GPU_ARCHS=sm_89
OUT=/root/ev-merged-re
while IFS=$'\t' read -r fam patch lanes; do
  [ -z "$fam" ] && continue
  /root/.pixi/bin/pixi run -e default python -u tools/merged_check/merged_check.py sabotage --out $OUT --patch "$patch" --lanes "$lanes"
done < tools/merged_check/sab_recheck.tsv
echo "SAB RECHECK DONE"
