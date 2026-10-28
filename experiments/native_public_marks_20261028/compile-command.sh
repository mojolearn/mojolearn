#!/bin/bash
# native-public-marks-20261028: compile the touched .mojo roots on the Mac (CPU column, identical, O0).
# Usage: bash experiments/native_public_marks_20261028/compile-command.sh <evidence-dir>
set -u
WT=$(cd "$(dirname "$0")/../.." && pwd); cd "$WT" || exit 2
OUT=${1:?evidence dir}; mkdir -p "$OUT/bin"
git rev-parse HEAD > "$OUT/source-commit.txt"
pixi run -e default mojo --version > "$OUT/compiler.txt" 2>&1
rc_all=0
for root in checks/expansion_linear_public_lifecycle_compile.mojo checks/sequential_weights_admission_compile.mojo; do
  name=$(basename "$root" .mojo)
  start=$(date +%s)
  bash tools/compile_slot.sh pixi run -e default mojo build --optimization-level 0 -j 10 --emit shared-lib \
    -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_CPU -I . -I bindings "$root" -o "$OUT/bin/$name.so" > "$OUT/$name.log" 2>&1
  rc=$?
  end=$(date +%s)
  sha=$( [ -f "$OUT/bin/$name.so" ] && shasum -a 256 "$OUT/bin/$name.so" | cut -d' ' -f1 )
  echo "$name rc=$rc seconds=$((end-start)) sha256=${sha:-none}" | tee -a "$OUT/results.txt"
  [ $rc -ne 0 ] && rc_all=$rc
done
exit $rc_all
