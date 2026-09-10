# Numeric per-tree GBDT feature sampling

This slice adds `feature_fraction=1.0` to legacy `GradientBoosting`, the
bounded classifier/regressor adapters, native training and prepared numeric
fits. It is a learning control, not a promise of faster training. The GPU-only
scope and all three numeric modes remain unchanged.

```python
from mojolearn import GradientBoostingRegressor
model = GradientBoostingRegressor(
    grow_policy='Lossguide', max_leaves=16,
    feature_fraction=0.75, numeric_mode='identical',
)
model.fit(X, y)
```

The default 1.0 preserves existing training and ABI behavior. An enabled
fraction must be finite and in (0, 1]; boolean/string values are not accepted
by the public constructor. Eligible numeric features have positive fold counts.
Each tree selects `max(1, floor(eligible_count * fraction + 0.5))` features,
capped by the eligible count. Empty eligibility stays empty. Selection is
without replacement; original feature IDs and their existing border grids
are retained. This is per-tree sampling, not per-node/per-level sampling.

Values below one require numeric features: nonempty public `cat_features`
or `one_hot_features`, and native categorical/one-hot flags or dynamic CTR
paths, are refused. Fraction one retains existing categorical behavior.
Experimental ordered/FeatureFreq APIs do not expose this parameter. The
bounded adapters add it to their existing fixed-loss parameter subset and
reuse the base learner validation; other objective/search restrictions remain.

Sampling reuses the existing `gbdt.data.permutation.TRandom` integer RNG in
an independent fit-local stream seeded with `random_seed XOR
0x4645415455524553`. Partial Fisher–Yates calls its existing rejection-sampled
`uniform` method; it is not LightGBM's LCG/density-dependent sampler. Selecting all
eligible features consumes no draws. This declared RNG difference means
same-seed cross-library masks/models are not promised. Each selected feature requests one bounded integer; rejection
may consume additional raw draws. Feature order, draw schedule and selected count are shared across numeric modes.

The shared implementation is `gbdt/gpu_data/feature_sampling.mojo`, called
from the existing boosting driver before the policy searcher. It constructs
a tree-local fold-count vector with excluded features zeroed and GPU-repacks
the selected bins from the full compressed index into the existing packed
layout. This runs once per sampled boosting tree, not once per depth or leaf
split and not merely once when a growth policy is selected. Existing searchers then operate on the smaller layout; no new
SymmetricTree, Depthwise or Lossguide learner is introduced. Stored model
feature IDs and the original full training/prediction schema remain unchanged.

Packing is needed because the existing histogram kernels infer feature bit
slots from their layout. Merely hiding candidates cannot safely reinterpret
the original packed words. This path removes excluded histogram/candidate
work, but adds per-tree buffer allocation, metadata upload, an integer packing
kernel and synchronization. Existing search workspace descriptors are keyed
by shape rather than sampled IDs, so those caches are cleared between sampled
trees to prevent stale feature mappings. No net throughput improvement has been established;
reusable packing buffers are a later profiling-driven optimization.

The Python adapter forwards through the same legacy parameter validator and
packer. The binding retains all previous parameter layouts. After counted
class weights, optional tails are `[min_split_gain, min_child_hessian,
feature_fraction]`; missing guards use -1 and missing fraction uses 1.
An enabled fraction includes preceding placeholders; default callers omit it.
Old extensions must be rebuilt to accept the new tail.

Source reference: LightGBM pin
`3d1cf3011adfed7209ba54bfeb05e8b2309040e4`,
[`ColSampler`](https://github.com/microsoft/LightGBM/blob/3d1cf3011adfed7209ba54bfeb05e8b2309040e4/src/treelearner/col_sampler.hpp)
and its actual
[CUDA consumer](https://github.com/microsoft/LightGBM/blob/3d1cf3011adfed7209ba54bfeb05e8b2309040e4/src/treelearner/cuda/cuda_single_gpu_tree_learner.cpp).
FEATURE-SAMPLE-1 declares the reused Mojo RNG/sampling mapping;
FEATURE-SAMPLE-2 declares packed-index projection. Host sampling metadata is
permitted GPU orchestration, not a dedicated CPU training arm.

All three public bindings build and the focused public checks pass.
[Local evidence](../../bench/results/feature_fraction_2026-09-10/RESULTS.md)
records default fingerprints, sampled fits and the native packing oracle. Cross-vendor identity, large-input scaling
and net training speed remain separate qualification work.


## Avoiding projection is possible, but not a dispatch-only change

Existing histogram families derive a packed source word from their feature-group
position (`hist_2_one_byte_base.mojo:630–639,916–923` and the `first_column`
launch setup in `greedy_search_helper.mojo`). Their zero-fold mask currently
suppresses output flushing, not all accumulation. A retained original layout
would need an explicit source-word map and per-lane active descriptors in both
direct and gathered histogram paths. That is a candidate optimization, not a
mathematical requirement to copy data and not something a one-time switch fixes.
Preserve the sampling contract and compare whole-fit timing before choosing it.
