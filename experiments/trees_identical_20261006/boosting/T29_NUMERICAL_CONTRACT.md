# T29 generated-PairLogit V1 source contract

NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.

`MOJOLEARN_TREES_T29_VERSIONED=1` selects this default-OFF arithmetic version under IDENTICAL. Omission preserves the incumbent. `T29_GROUP_VERSIONED` in the per-ID record selects it together with short/long query scheduling; scheduling does not change the logical graph.

The supported path is the existing generated-pair group layout. Query offsets, original-row IDs, grades, per-group first-row weights, per-row pair weights and the setup refusal/count rules are unchanged. Every unequal-grade partner participates; equal grades contribute nothing. Winner/loser choice uses the original grade comparison. No query or pair is truncated. Explicit supplied pair lists retain their incumbent path. This objective consumes no new RNG draws and changes no estimator setting.

For each row, partners remain in ascending original query-row order. Consecutive 32-partner chunks fold serially from +0. Adjacent chunks merge through a binary-carry stack; the final incomplete set of subtrees folds oldest first. Empty chunks contribute +0. The stack has 32 levels because existing row indices are bounded by UInt32/Int32 representation; it imposes no new dataset-specific cap. Derivative, curvature and winner-only objective value all use this graph.

Each group has 256 logical lanes. Original row `(row - group_begin) % 256` owns a lane; rows fold in ascending order within each lane. Group value and magnitudes then fold at strides 128, 64, 32, 16, 8, 4, 2, 1. This width is part of the arithmetic version on the host and all GPU vendors, independent of hardware warp width. The GPU kernel uses 256 threads to realize that graph. The S+V configuration separates groups at one such logical tile but preserves group IDs and all arithmetic.

Each add, subtract, multiply and divide converts FTZ Float32 operands to software binary64, computes with the existing software operation, rounds to Float32, applies FTZ and normalizes signed zero to +0. Portable `identical_exp` and `identical_log` supply the transcendental operations. The logistic finite-denominator fallback is retained; the clamp endpoints follow this version's FTZ operations. Products never fuse into sums. Software overflow/NaN behavior is shared; existing input checks and downstream loss refusals remain in place. Bits may differ from B. Cross-vendor bits and quality remain unverified.

The GPU path is `pair_logit._launch_pair_logit_group_layout` → `pair_logit_group.launch_pair_logit_group` → `tree_t29_pair.pair_logit_group_versioned_kernel`. Search stores V1 derivatives/curvature/value in the existing reuse accumulator. Leaf estimation either reuses those same words at the unchanged search point or calls the same objective at its new point. The final learn-loss call uses the same dispatch.

The host path is `gbdt_oracle_losses` → `gbdt_oracle_pair` search/evaluation/final-value functions → `_group_values_t29`. Host and device call the same `gbdt/targets/tree_t29_units.mojo` row and group-fold units. Host group magnitudes use those same software additions and logical lanes. The public host ranking fitter retains its existing SymmetricTree/Cosine scope; no missing host policy is claimed implemented.

YetiRank permutation sampling and position-derived pair weights form a distinct objective graph in its device and host modules. They are not replaced by PairLogit's graph. Legacy group-layout OFF or global old master-OFF keeps the explicit-pair incumbent route. These cases must not be reported as demonstrated V1 reach. Full saved PairLogit recipe facts, all-column identity, predictive quality and end-to-end measurement remain pending.
