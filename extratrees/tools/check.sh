#!/bin/sh
# Run THIS LANE'S checks, and nothing else.
#
# Deliberately not a pixi task: `pixi.toml` is a shared root file that a
# parallel lane is also editing, and rule 12 says file convergence is what
# predicts integration pain. Deliberately not the repository's full suite
# either -- rule 12 again: own checks only, one merge-time run.
#
# NO TIMING NUMBER is taken by any of them. Some of them DO enqueue kernels --
# range_kernel_check, split_reduce_check, score_kernel_check and
# device_tree_check run on the GPU
# and check device output per cell against a host oracle. (This header said
# "every check here is host-side, no kernel is enqueued" until 2026-08-21; it
# was false from the moment the range kernel landed.)
set -eu
cd "$(dirname "$0")/../.."

checks="
extratrees/checks/split_check.mojo
extratrees/checks/params_check.mojo
extratrees/checks/flatnode_check.mojo
extratrees/checks/pcg_rng_check.mojo
extratrees/checks/fixtures_check.mojo
extratrees/checks/objectives_check.mojo
extratrees/checks/fixed_point_check.mojo
extratrees/checks/partition_check.mojo
extratrees/checks/range_draw_check.mojo
extratrees/checks/builder_check.mojo
extratrees/checks/leaf_check.mojo
extratrees/checks/feature_sampler_check.mojo
extratrees/checks/host_splitter_check.mojo
extratrees/checks/tree_check.mojo
extratrees/checks/forest_check.mojo
extratrees/checks/range_kernel_check.mojo
extratrees/checks/quality_band_check.mojo
extratrees/checks/estimator_check.mojo
extratrees/checks/split_reduce_check.mojo
extratrees/checks/score_kernel_check.mojo
extratrees/checks/partition_leaf_kernel_check.mojo
extratrees/checks/device_tree_check.mojo
extratrees/checks/regression_score_check.mojo
extratrees/checks/device_forest_check.mojo
extratrees/checks/sampler_kernel_check.mojo
extratrees/checks/device_regression_check.mojo
extratrees/checks/partition_multiblock_check.mojo
extratrees/checks/rescue_check.mojo
extratrees/checks/device_batched_check.mojo
extratrees/checks/bestfirst_check.mojo
"

failed=0
cpu_names=""
gpu_names=""
for c in $checks; do
  if [ ! -f "$c" ]; then
    printf 'SKIP  %s (not present)\n' "$c"
    continue
  fi
  # A registered check has no main(): its file is the suite (`run_checks`) and
  # checks/native_tests/extratrees_<stem>.mojo is its entry in the one native
  # runner (lane fx-checks-orphans, 2026-11-18). It runs below, by name.
  name="extratrees_$(basename "$c" .mojo)"
  if [ -f "checks/native_tests/$name.mojo" ]; then
    if grep -q 'cpu=True' "checks/native_tests/$name.mojo"; then
      cpu_names="$cpu_names $name"
    else
      gpu_names="$gpu_names $name"
    fi
    continue
  fi
  if pixi run mojo run ${MOJOLEARN_MOJO_DEFINES:-} -I . "$c" >/tmp/et_check.$$ 2>&1; then
    printf 'ok    %s\n' "$c"
  else
    printf 'FAIL  %s\n' "$c"
    tail -20 /tmp/et_check.$$
    failed=1
  fi
done
for column in cpu gpu; do
  if [ "$column" = cpu ]; then names="$cpu_names"; else names="$gpu_names"; fi
  [ -n "$names" ] || continue
  # $names is a word list of registered check names.
  if pixi run check-native-$column $names >/tmp/et_check.$$ 2>&1; then
    printf 'ok    native %s:%s\n' "$column" "$names"
  else
    printf 'FAIL  native %s\n' "$column"
    grep -E '^(CHECK|FAIL)' /tmp/et_check.$$ | grep -v 'failed=0' | tail -20
    failed=1
  fi
done
rm -f /tmp/et_check.$$

# The one check that crosses the PYTHON boundary (DEVIATION 458): every
# Mojo check above takes an `ExtraTreesConfig` directly, so none of them
# could see `bindings/_mojolearn_trees.mojo` overwrite the regressor's
# max_features after reading it. Runs against the SHIPPED extension under the
# python it was built for (pkg/gbmbench, 3.14); skipped with a SKIP line, not
# silently, when that environment is absent.
reach=extratrees/tools/wrapper_reach_check.py
if [ -f python/mojolearn/_mojolearn_trees.so ]; then
  if PYTHONPATH=python pixi run -e gbmbench python "$reach" >/tmp/et_check.$$ 2>&1; then
    printf 'ok    %s\n' "$reach"
  else
    printf 'FAIL  %s\n' "$reach"
    tail -20 /tmp/et_check.$$
    failed=1
  fi
  rm -f /tmp/et_check.$$
else
  printf 'SKIP  %s (python/mojolearn/_mojolearn_trees.so not built)\n' "$reach"
fi

if [ "$failed" -ne 0 ]; then
  echo "extratrees: SOME CHECKS FAILED"
  exit 1
fi
echo "extratrees: all lane checks pass"
