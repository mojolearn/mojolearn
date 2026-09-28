#!/bin/bash
# every family's e2e sabotage, one after another (each edits the tree); OUT=$1
cd "$(dirname "$0")/../.."
OUT=${1:?out}
PY=${PY:-"/root/.pixi/bin/pixi run -e default python -u"}
while IFS=$'\t' read -r fam patch lanes; do
  [ -z "$fam" ] && continue
  echo "=== $fam $patch"
  $PY tools/merged_check/merged_check.py sabotage --out "$OUT" --patch "$patch" --lanes "$lanes"
done < tools/merged_check/sab_plan.tsv
echo "SABOTAGE ALL DONE"
