#!/bin/bash
set -euo pipefail
cd /Users/andrewhendel/CascadeProjects/mojolearn/.claude/worktrees/knn-ivf-width-pass
outdir=/tmp/mojolearn-knn-ivf-refusal-gates
mkdir -p "$outdir"
mojo=/Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/default/bin/mojo
"$mojo" run -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 ivf/checks/ivf_large_k_check.mojo > "$outdir/ivf-large.log" 2>&1
"$mojo" run -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 ivf/checks/ivf_check.mojo > "$outdir/ivf-full.log" 2>&1
for arm in native simulated64; do
 flags=()
 if [[ $arm == simulated64 ]]; then flags=(-D MOJOLEARN_COLUMN_AMD=1); fi
 "$mojo" run -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 "${flags[@]}" neighbors/checks/fused_logical32_check.mojo > "$outdir/fused-$arm.log" 2>&1
 "$mojo" run -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 "${flags[@]}" neighbors/checks/knn_identity_check.mojo > "$outdir/identity-$arm.log" 2>&1
done
echo PASS > "$outdir/status.txt"
