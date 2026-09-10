# Symmetric GBDT: CatBoost comparison contract

This audit covers the dense numeric HIGGS binary Logloss benchmark in
`bench/speed/nvidia_identical_trees.py`, its MojoLearn arm in
`bench/speed/forest_speed_arm.py`, and shared CatBoost/timing code in
`tools/speed_gbdt_arm.py`. It does not establish equal models or a speed advantage.
NVIDIA comparisons use MojoLearn IDENTICAL; this is not a claim that CatBoost
uses the same arithmetic or produces identical models.

## Controls and actual differences

The recorded workload uses 1 million rows, 28 columns, 100 trees, depth 6,
learning rate 0.1, L2 regularization 1, 254 borders and seed 7. Both arms use
SymmetricTree, Logloss and no bootstrap; CatBoost explicitly uses Plain boosting.
The timing includes constructor, host input preparation, quantization, fit and
synchronization. Prediction and quality scoring are outside the fit timer.

| Item | Audited behavior and implication |
| --- | --- |
| Split noise | MojoLearn defaults to `random_strength=0`; CatBoost defaults to 1. `bootstrap_type='No'` does not disable score noise. The focused runner now defaults to `--symmetric-profile matched-no-noise`, setting CatBoost's strength to 0. `native-defaults` preserves the historical comparison. |
| Structure search | MojoLearn defaults to greedy subsets (`use_pointwise_searcher=False`). CatBoost's scalar Plain numeric GPU path selects DocParallel and its pointwise oblivious searcher. Equal growth policy does not imply equal search implementation. |
| Leaf estimation | Binary Logloss resolves to Newton with 10 iterations; MojoLearn implements AnyImprovement backtracking. Its greedy path still estimates these leaves. Skipped leaf estimation is not an established explanation for these timings. |
| Split score | Symmetric defaults resolve to Cosine. Equal score names do not establish equal histogram rounding, random perturbation, tie handling or accumulation order. |
| Quantization | Both raw fits include their own quantizer. MojoLearn defaults to a 200,000-row border sample; pinned CatBoost quantization also exposes a 200,000 default subset cap. Border count alone does not prove identical sampled rows or border grids. Do not label either arm prequantized or assume CatBoost sorts every row without evidence. |
| Sampling and bias | No bootstrap is explicit; numeric feature fraction remains 1. Binary Logloss has no initial average bias by default. Capture resolved settings rather than infer all defaults from the few shared constructor arguments. |

The focused runner rotates the first arm each measured round and records
CatBoost `get_all_params()` after fitting. These corrections improve the
comparison; they do not make the libraries algorithmically identical.

## Source provenance

Pinned CatBoost source at `54a8143a` supplies the dispatch evidence:

- [`pointwise.cpp`](https://github.com/catboost/catboost/blob/54a8143a/catboost/cuda/train_lib/pointwise.cpp) registers scalar RMSE/Logloss with the pointwise trainer.
- [`train.cpp`](https://github.com/catboost/catboost/blob/54a8143a/catboost/cuda/train_lib/train.cpp), `UpdateDataPartitionType`, chooses DocParallel for Plain numeric data when partitioning is not explicit.
- [`train_template_pointwise.h`](https://github.com/catboost/catboost/blob/54a8143a/catboost/cuda/train_lib/train_template_pointwise.h) dispatches DocParallel to `TDocParallelObliviousTree`; [`doc_parallel_pointwise_oblivious_tree.h`](https://github.com/catboost/catboost/blob/54a8143a/catboost/cuda/methods/doc_parallel_pointwise_oblivious_tree.h) constructs `TDocParallelObliviousTreeSearcher`.
- [`oblivious_tree_options.cpp`](https://github.com/catboost/catboost/blob/54a8143a/catboost/private/libs/options/oblivious_tree_options.cpp) initializes random strength 1, Cosine and AnyImprovement. Objective-specific defaults are resolved in `catboost_options.cpp`.
- [`binarization_options.h`](https://github.com/catboost/catboost/blob/54a8143a/catboost/private/libs/options/binarization_options.h) and [`quantization.cpp`](https://github.com/catboost/catboost/blob/54a8143a/catboost/libs/data/quantization.cpp) define the border subset policy.

Local implementations are `gbdt/train.mojo` (preparation/defaults),
`gbdt/methods/doc_parallel_boosting.mojo` (searcher dispatch and shared leaf
estimation), and `python/mojolearn/ensemble.py` (public defaults). The historical
H100 package is CatBoost 1.2.10; its fitted settings should be retained alongside
this source audit, rather than treating a different source pin as runtime proof.

## Evidence and next acceptance gate

The historical [H100 results](../../bench/results/nvidia_identical_trees_2026-09-10/gbdt_higgs_1m.json)
are the native-default profile. Both five-fit arms failed the runner's stability
criterion: max/min was about 1.345 for MojoLearn and 1.125 for CatBoost. Quality
was close, not equal; MojoLearn's repeated model/prediction hashes matched.
These observations support no accepted speed ratio or causal performance claim.
Keep that artifact immutable and identify corrected profiles separately.

The next production comparison uses matched-no-noise with the existing shared
workload, records resolved score/leaf method/iterations/backtracking/bias and
partition settings, and retains rotated per-round times, dataset hashes,
model fingerprints, held-out quality, package versions and device provenance.
Accept timing only after both arms pass the declared stability rule; a completed
run alone is insufficient. Match control changes explicitly rather than silently
changing the workload to obtain a favorable result.

A separate diagnostic may enable MojoLearn `use_pointwise_searcher=True` while
keeping the same no-noise controls. Name it as a different learner profile and
verify model/quality behavior; it must not replace the default production arm.
For a quantization-controlled experiment, establish the same border grid and
row assignments and exclude preparation consistently on both sides. Report it
separately from raw end-to-end fit.

Searcher bookkeeping, histogram arithmetic/layout, host/device synchronization,
quantization and leaf-estimation work are profiling candidates. Source inspection
alone cannot assign the observed timing difference to any of them. Collect
stage timings or device traces before proposing a causal explanation or a
performance-driven default change.

## Runtime option compatibility audit

No concrete IDENTICAL-only conflict was found for `random_strength` or
`use_pointwise_searcher`. The score-noise reduction in
`gbdt/methods/random_score_helper.mojo::std_dev_blocks` pins its virtual SM count
to 32 under IDENTICAL, so physical device size does not change that reduction's
partitioning. Pointwise histogram dispatch also reads the numeric mode. This
source audit does not qualify every option combination across devices.

The shared Python constructor/fit parameter guard rejects nonfinite, negative
or non-Float32-representable random strength, nonzero noise with L2/NewtonL2,
and pointwise search with a non-symmetric growth policy. These restrictions
apply in every mode and are checked again after direct attribute mutation.
Portable seeded randomness remains supported; different seeds are not an
identity failure. Defaults remain noise zero and greedy subsets search.

Focused host guard tests plus existing feature-fraction/Hessian API regressions
passed 152 checks, with two optional sklearn checks skipped in the test
environment. No native build, GPU fit or quality experiment was performed for
this validation change.
