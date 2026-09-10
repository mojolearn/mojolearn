# Extended k-NN source status

September 6, 2026: source edits and authored tests only. No tests, builds,
models, benchmarks or measurements were run by the author. Qualification
and performance remain root-only remote NVIDIA/AMD work.

## Already implemented before this change

`python/mojolearn/neighbors.py` exposes `weights='distance'` on both classifier
and regressor. Native `estimator.mojo` calls
`impl/selection/distance_weights.mojo`, then the weighted vote kernels in
`impl/selection/knn.mojo`. No new weighted arithmetic or ABI was needed.

The existing pinned rule first computes FTZ inverse distances. If any inverse
is positive infinity (including an exact zero distance), the entire row is
replaced by its positive-infinity mask: those neighbors get weight one and
every other neighbor gets zero. Thus duplicate exact matches exclude distant
neighbors. Votes and regression sums accumulate in neighbor-slot order;
classification uses ascending unique labels and first maximum, giving the
smallest label for equal class mass. Regression divides its weighted target
sum by the row weight sum, separately for every output. The profile refuses
invalid/negative weights and a normalizer that underflows to zero. This is
the project's pinned FP32 schedule, not an external-library bitwise claim.

Brute-force Manhattan/L1 and cosine are also already public/native, alongside
Chebyshev and Minkowski. `impl/distance/detail/distance_ops.mojo` documents
their reduction and epilogue rules; `checks/metric_check.mojo` contains the
existing oracle and sabotage checks. Cosine explicitly refuses zero-norm
rows, uses pinned square roots, and preserves the upstream unclamped
`1-dot/(norm_x*norm_y)` epilogue. Near-identical rows may produce slightly
negative values from rounding; changing that to clamping or defining a
zero-vector answer requires a new explicit profile and independent fixtures.
Distance-weighted cosine therefore needs its own adversarial validation
before claiming support for all nearly collinear inputs. No cosine arithmetic
or zero-vector policy changed here. Manhattan needs vendor evidence expansion,
not another duplicate metric kernel.

## Newly authored bounded extension

The shared pinned radix selector now admits k through 1024 for IDENTICAL and
DETERMINISTIC. Selection uses the same distance/index composite key and
radix passes. The final rank phase stages all winners, synchronizes once,
then each of the 256 threads processes slots in strides of 256. Every rank
compares against the immutable staged winners, preserving ascending distance
then original-index order, including ties across stride boundaries.

The maximum staging is 1024 Float32/UInt32 pairs (8 KiB); ranking remains
O(k²) per query. Inputs above 1024 remain explicitly refused by this pinned
launcher. FAST keeps its existing selector. The kernel is specialized as
`radix_topk_identical_kernel[RANK_CAPACITY: Int]`: the launcher uses capacity
256 for k<=256 (the historical 2 KiB pair staging), and capacity 1024 only
for the extended range. Other shared allocations and launch width are
unchanged. Occupancy/performance has not been measured. This does not relax
dataset-size, workspace, or index-size checks.

Unchanged refusals include kd_tree/ball_tree, approximate index algorithms,
callable weights and unimplemented metrics. The existing exact ball-cover index
and its triangle-inequality metric restrictions remain unchanged.

## Authored gate and root command

`python/mojolearn/tests/test_neighbors_weighted_largek.py` independently checks
duplicate-zero row replacement, classification ties, multi-output signed
regression, a weighted winner that differs from uniform, and inverse-distance
normalization. For pinned modes it compares k=256/257/513/1024 against an
independent FP64 Manhattan plus lexicographic-index oracle, including exact
ties, repeated raw FP32 bytes and query-tile invariance; k=1025 and kd_tree
must still refuse. These are bounded two-query, 1056-index fixtures, not a
performance benchmark or full feature certificate.

After root rebuilds the main binding from this source under its usual serial
guard, root can run on NVIDIA:

```sh
MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_NEIGHBORS_EXTENDED_GATE=1 \
MOJOLEARN_EXPECT_VENDOR=cuda PYTHONPATH=python \
python3 tools/nvidia_serial_guard.py --seconds 600 --rss-gib 8 -- \
pixi run python python/mojolearn/tests/test_neighbors_weighted_largek.py
```

For the separate AMD job use `MOJOLEARN_EXPECT_VENDOR=hip` and
`tools/amd_serial_guard.py`. Repeat separately in DETERMINISTIC and FAST;
the FAST run intentionally skips the pinned selector test. Guards enforce
two CPU threads/cores. Retain source/binary/vendor/mode witnesses, commands,
guard zero-exit evidence and raw comparison artifacts before promoting a
cross-vendor identity claim; this unittest alone does not capture a paired
vendor certificate.
