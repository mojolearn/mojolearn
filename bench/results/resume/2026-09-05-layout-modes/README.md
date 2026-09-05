# Local experiment isolation and remote controller rehearsal

At `fbce6103` plus the new `bench/knn_layout_mode_check.mojo` driver, all four
Apple M4 runs passed: FAST and DETERMINISTIC, each with neither and both
experimental compiler defines. Both effective experiment flags remained zero.
The existing estimator checks passed the query-tile planner, 320 exact host
reference neighbors and agreement between TILED, FUSED and AUTO. Baseline and
armed logs matched exactly within each mode. An IDENTICAL build correctly
refused this gate with its named mode error.

Build serially under `tools/with_build_lock.sh` using `pixi run mojo build
-j 2 -I . bench/knn_layout_mode_check.mojo -o /tmp/knn-layout-mode`.
DETERMINISTIC adds `-D MOJOLEARN_NUMERIC_DETERMINISTIC=1`; armed builds add
`-D MOJOLEARN_EXPERIMENTAL_SMALLK_IDENTICAL=1` and
`-D MOJOLEARN_EXPERIMENTAL_KNN_TRANSPOSE_IDENTICAL=1`. Run each resulting
binary under the same build lock. For the refusal control, use
`-D MOJOLEARN_NUMERIC_IDENTICAL=1` and require a nonzero exit with
`layout mode isolation gate requires FAST or DETERMINISTIC`.

The controller's displayed rental command omitted the layout-only environment
flag. The fix retains that flag, source commit and work timeout. This command
passed its full rehearsal after the fix:

```sh
MOJOLEARN_KNN_LAYOUT_ONLY=1 tools/gemm_remote_leg.sh nvidia \
  --payload mamba --dry-run --minutes 30 --work-timeout 1200
```

`results.json` retains source hashes; compressed raw logs retain each build,
execution and the controller rehearsal. No rental or performance measurement
was performed. These checks do not extend certification to another vendor.

The tested parent `fbce6103` was amended as `75319203` to store the prior
correctness logs as plain text for Git delta compression. Tested source
files were unchanged; the recorded source hashes remain authoritative.
