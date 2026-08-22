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
extratrees/mojo_only/split_check.mojo
extratrees/mojo_only/params_check.mojo
extratrees/mojo_only/flatnode_check.mojo
extratrees/mojo_only/pcg_rng_check.mojo
extratrees/mojo_only/fixtures_check.mojo
extratrees/mojo_only/objectives_check.mojo
extratrees/mojo_only/fixed_point_check.mojo
extratrees/mojo_only/partition_check.mojo
extratrees/mojo_only/range_draw_check.mojo
extratrees/mojo_only/builder_check.mojo
extratrees/mojo_only/leaf_check.mojo
extratrees/mojo_only/feature_sampler_check.mojo
extratrees/mojo_only/host_splitter_check.mojo
extratrees/mojo_only/tree_check.mojo
extratrees/mojo_only/forest_check.mojo
extratrees/mojo_only/range_kernel_check.mojo
extratrees/mojo_only/quality_band_check.mojo
extratrees/mojo_only/estimator_check.mojo
extratrees/mojo_only/split_reduce_check.mojo
extratrees/mojo_only/score_kernel_check.mojo
extratrees/mojo_only/partition_leaf_kernel_check.mojo
extratrees/mojo_only/device_tree_check.mojo
extratrees/mojo_only/regression_score_check.mojo
extratrees/mojo_only/device_forest_check.mojo
extratrees/mojo_only/sampler_kernel_check.mojo
extratrees/mojo_only/device_regression_check.mojo
extratrees/mojo_only/partition_multiblock_check.mojo
extratrees/mojo_only/rescue_check.mojo
extratrees/mojo_only/device_batched_check.mojo
"

failed=0
for c in $checks; do
  if [ ! -f "$c" ]; then
    printf 'SKIP  %s (not present)\n' "$c"
    continue
  fi
  if pixi run mojo run -I . "$c" >/tmp/et_check.$$ 2>&1; then
    printf 'ok    %s\n' "$c"
  else
    printf 'FAIL  %s\n' "$c"
    tail -20 /tmp/et_check.$$
    failed=1
  fi
done
rm -f /tmp/et_check.$$
if [ "$failed" -ne 0 ]; then
  echo "extratrees: SOME CHECKS FAILED"
  exit 1
fi
echo "extratrees: all lane checks pass"
