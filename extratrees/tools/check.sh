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
extratrees/original/split_check.mojo
extratrees/original/params_check.mojo
extratrees/original/flatnode_check.mojo
extratrees/original/pcg_rng_check.mojo
extratrees/original/fixtures_check.mojo
extratrees/original/objectives_check.mojo
extratrees/original/fixed_point_check.mojo
extratrees/original/partition_check.mojo
extratrees/original/range_draw_check.mojo
extratrees/original/builder_check.mojo
extratrees/original/leaf_check.mojo
extratrees/original/feature_sampler_check.mojo
extratrees/original/host_splitter_check.mojo
extratrees/original/tree_check.mojo
extratrees/original/forest_check.mojo
extratrees/original/range_kernel_check.mojo
extratrees/original/quality_band_check.mojo
extratrees/original/estimator_check.mojo
extratrees/original/split_reduce_check.mojo
extratrees/original/score_kernel_check.mojo
extratrees/original/partition_leaf_kernel_check.mojo
extratrees/original/device_tree_check.mojo
extratrees/original/regression_score_check.mojo
extratrees/original/device_forest_check.mojo
extratrees/original/sampler_kernel_check.mojo
extratrees/original/device_regression_check.mojo
extratrees/original/partition_multiblock_check.mojo
extratrees/original/rescue_check.mojo
extratrees/original/device_batched_check.mojo
"

failed=0
for c in $checks; do
  if [ ! -f "$c" ]; then
    printf 'SKIP  %s (not present)\n' "$c"
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
