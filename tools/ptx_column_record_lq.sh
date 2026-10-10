#!/bin/bash
# ptx_column_record_lq.sh <ptx-set-dir> <outdir>
# The PTX identity column, run ONCE through `lq add <nv box> CMD` (2026-10-10).
# Copies the cuda-sm_80 leg's set (scp'd to the box, outside the tree) into
# python/mojolearn/cuda/sm_80 of this lq branch tree, then records the column
# with REPEATS=1. The record lands in <outdir> (outside the tree).
set -u
SET=${1:?ptx set dir (holding the sm_80 bindings)}; OUT=${2:?outdir}
[ -f "$SET/PTX_BASELINE.json" ] || { echo "PTXREC no PTX_BASELINE.json in $SET"; exit 2; }
rm -rf python/mojolearn/cuda/sm_80
mkdir -p python/mojolearn/cuda
cp -a "$SET" python/mojolearn/cuda/sm_80
echo "PTXREC set=$(ls python/mojolearn/cuda/sm_80 | wc -l) files"
REPEATS=1 bash tools/record_identity_column.sh nvidia-ptx-l40s-sm80 "$OUT"
rc=$?
echo "PTXREC rc=$rc out=$OUT $(ls "$OUT" 2>/dev/null | grep -m 3 identical | tr '\n' ' ')"
exit $rc
