#!/usr/bin/env bash
# Diagnostic only: explicit synchronizations perturb request/device timings.
# Run only under root's granted build/device slot in an activated Pixi env.
# Usage: bash tools/knn_residual_phase_probe.sh apple|nvidia /fresh/absolute/output
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
if [[ ${MOJOLEARN_BUILD_LOCK_HELD:-0} != 1 ]]; then
  exec nice -n 19 tools/with_build_lock.sh bash "$0" "$@"
fi
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
column=${1:?apple or nvidia required}
output=${2:?fresh absolute output directory required}
case "$column" in
  apple) [[ $(uname -s) == Darwin ]]; control=MOJOLEARN_KNN_IDENTICAL_NO_PREFLIGHT ;;
  nvidia) [[ $(uname -s) == Linux ]]; control=MOJOLEARN_KNN_LEGACY_QUERY_TILE ;;
  *) exit 2 ;;
esac
[[ "$output" = /* && ! -e "$output" ]]
mkdir -p "$output"
mojo --version > "$output/compiler.txt" 2>&1
uname -a > "$output/host.txt"
if [[ "$column" == nvidia ]]; then
  nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > "$output/gpu.csv"
fi
if git rev-parse HEAD > "$output/source.txt" 2>/dev/null; then
  git diff --binary -- neighbors core checks/kernel_matrix.mojo bench/knn_reference_price_main.mojo tools/knn_residual_phase_probe.sh > "$output/working-tree.patch"
else
  cat commit.txt > "$output/source.txt"
fi
printf '%s\n' "column=$column" "control=$control" 'phase_synchronization=enabled; prices are diagnostic only' > "$output/experiment.txt"
for arm in default control; do
  flags=(-D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_KNN_PHASE_TIMERS=1)
  [[ "$arm" != control ]] || flags+=(-D "$control=1")
  mojo build -j 2 -I "$root" "${flags[@]}" bench/knn_reference_price_main.mojo \
    -o "$output/$arm" > "$output/$arm-build.log" 2>&1
done
# Full output equality is mandatory; neither control disables exact repair.
# Reverse arm order on the second pass to expose clock/order drift.
shapes=('400000 4000 32 10' '400000 4000 32 15' '400000 1000 8 15' '65537 129 17 10')
if [[ ${MOJOLEARN_KNN_PHASE_SMOKE:-0} == 1 ]]; then
  shapes=('65537 129 17 10')
  echo 'scope=small phase smoke; not the 400k target price' >> "$output/experiment.txt"
fi
pairs=0
for pass in 0 1; do
  arms=(default control)
  [[ $pass == 0 ]] || arms=(control default)
  for shape in "${shapes[@]}"; do
    read -r n q d k <<< "$shape"
    tag="n${n}-q${q}-d${d}-k${k}-p${pass}"
    for arm in "${arms[@]}"; do
      MOJOLEARN_KNN_REF_INDEX=$n MOJOLEARN_KNN_REF_QUERIES=$q \
      MOJOLEARN_KNN_REF_FEATURES=$d MOJOLEARN_KNN_REF_K=$k \
      MOJOLEARN_KNN_REF_ROUNDS=3 MOJOLEARN_KNN_REF_DUMP_FULL="$output/$tag-$arm.bin" \
        "$output/$arm" > "$output/$tag-$arm.log" 2>&1
      # Missing phase records must not produce an apparently useful result.
      awk '/^KNN_PHASE_TIMERS/ {print; n++} END {if (!n) exit 1}' \
        "$output/$tag-$arm.log" > "$output/$tag-$arm.phases"
    done
    cmp "$output/$tag-default.bin" "$output/$tag-control.bin"
    pairs=$((pairs + 1))
  done
done
printf '%s\n' "PASS: $pairs paired full-output comparisons; phase times are diagnostic only" > "$output/status.txt"
