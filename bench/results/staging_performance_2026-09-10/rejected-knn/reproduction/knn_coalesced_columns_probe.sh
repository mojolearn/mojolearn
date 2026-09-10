#!/usr/bin/env bash
# Run only under the owner's granted device/build slot and activated Pixi env.
# Usage: bash tools/knn_coalesced_columns_probe.sh /fresh/absolute/output [checks]
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
output=${1:?fresh absolute output directory required}
[[ "$output" = /* && ! -e "$output" ]]
mkdir -p "$output"
mojo --version > "$output/compiler.txt" 2>&1
if [[ -f commit.txt ]]; then cat commit.txt > "$output/source.txt"; else git rev-parse HEAD > "$output/source.txt"; fi

for arm in baseline coalesced; do
  flags=(-D MOJOLEARN_NUMERIC_IDENTICAL=1)
  if [[ "$arm" == coalesced ]]; then
    flags+=(-D MOJOLEARN_KNN_IDENTICAL_COALESCED_COLUMNS=1)
  fi
  mojo build -I "$root" "${flags[@]}" neighbors/checks/coalesced_distance_check.mojo -o "$output/$arm-gate" > "$output/$arm-build-gate.log" 2>&1
  "$output/$arm-gate" > "$output/$arm-gate.log" 2>&1
  mojo build -I "$root" "${flags[@]}" bench/knn_layout_dispatch_check.mojo -o "$output/$arm-public-gate" > "$output/$arm-build-public-gate.log" 2>&1
  "$output/$arm-public-gate" > "$output/$arm-public-gate.log" 2>&1
  awk '/^(DISPATCH_CELL|LAYOUT_CELL)/' "$output/$arm-public-gate.log" > "$output/$arm-public.cells"
  if [[ ${2:-} != checks ]]; then
    mojo build -I "$root" "${flags[@]}" bench/knn_reference_price_main.mojo -o "$output/$arm-price" > "$output/$arm-build-price.log" 2>&1
  fi
done
cmp "$output/baseline-public.cells" "$output/coalesced-public.cells"
[[ ${2:-} != checks ]] || exit 0
# Forward and reversed order; major request plus low-feature/small/tail controls.
for pass in 0 1; do
  arms=(baseline coalesced)
  [[ $pass == 0 ]] || arms=(coalesced baseline)
  for shape in '400000 4000 32 10' '400000 4000 32 15' '400000 1000 8 10' '10000 32 32 15' '65537 129 17 10'; do
    read -r n q d k <<< "$shape"
    tag="n${n}-q${q}-d${d}-k${k}-p${pass}"
    for arm in "${arms[@]}"; do
      MOJOLEARN_KNN_REF_INDEX=$n MOJOLEARN_KNN_REF_QUERIES=$q \
      MOJOLEARN_KNN_REF_FEATURES=$d MOJOLEARN_KNN_REF_K=$k \
      MOJOLEARN_KNN_REF_ROUNDS=5 MOJOLEARN_KNN_REF_DUMP_FULL="$output/$tag-$arm.bin" \
        "$output/$arm-price" > "$output/$tag-$arm.log" 2>&1
    done
    cmp "$output/$tag-baseline.bin" "$output/$tag-coalesced.bin"
  done
done
