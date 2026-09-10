#!/bin/bash
set -euo pipefail
cd /Users/andrewhendel/CascadeProjects/mojolearn/.claude/worktrees/knn-width-pass
outdir=/tmp/mojolearn-knn-width-qualified
mkdir -p "$outdir"
mojo=/Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/default/bin/mojo
for column in SPEC_BASELINE INTEL QUALCOMM AMD_RDNA; do
 "$mojo" run -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D "MOJOLEARN_COLUMN_${column}=1" neighbors/checks/lane_minimum_check.mojo > "$outdir/$column.log" 2>&1
done
"$mojo" run -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 neighbors/checks/knn_selector_long_rows_check.mojo > "$outdir/long-rows.log" 2>&1
printf 'PASS\n' > "$outdir/status.txt"
