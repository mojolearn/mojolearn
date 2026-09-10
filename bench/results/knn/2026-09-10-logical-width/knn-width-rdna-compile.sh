#!/bin/bash
set -u
cd /root/knn-width
outdir=/root/knn-width-results
mkdir -p "$outdir/bin"
for target in gfx1100 gfx1201; do
 mojo build -j4 -I . --target-accelerator="$target" -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_REQUIRE_RDNA_TARGET=1 neighbors/checks/lane_minimum_check.mojo -o "$outdir/bin/lane-$target" > "$outdir/compile-$target.log" 2>&1
 printf '%s %s\n' "$target" "$?" >> "$outdir/rdna-compile-status.txt"
done
mojo build -j4 -I . --target-accelerator=gfx1100 -D MOJOLEARN_NUMERIC_IDENTICAL=1 neighbors/checks/knn_selector_long_rows_check.mojo -o "$outdir/bin/selector-gfx1100" > "$outdir/compile-selector-gfx1100.log" 2>&1
printf 'selector-gfx1100 %s\n' "$?" >> "$outdir/rdna-compile-status.txt"
