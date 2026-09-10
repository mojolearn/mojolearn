# GPU decision-tree roadmap

Updated 2026-09-10. Working branch:
`lane/trees-gpu-growth-rf-hist-2026-09-09`.
This is the consolidated execution plan for single trees, Random Forest (RF),
Extra Trees (ET), and symmetric/Depthwise/Lossguide GBDT. Status describes
the source implementation, not a published wheel. The prior implementation
was integrated into main at `8f14f107`.

Pipeline expansion: [GPU_PIPELINE_PLAN.md](GPU_PIPELINE_PLAN.md) covers metrics,
preprocessing, sklearn compatibility, model selection and end-to-end identity.
DETERMINISTIC stays supported pending measured retirement criteria.

## Contract and priorities

Build GPU learners in Mojo. CPU input preparation and orchestration remain
part of the pipeline; a separate CPU training implementation is out of scope.
Preserve existing default behavior. Performance-only changes must retain the
reference model under each mode's existing contract. New learning features
are explicit options with disabled-default equivalence tests.

Support FAST, DETERMINISTIC and IDENTICAL where the feature has a defined
numeric contract. IDENTICAL means MojoLearn cross-device identity, not model
equality with another library. Local same-device checks are necessary but do
not qualify a new path across vendors. Never relax reductions or tie rules to
obtain a timing win. Integer counts can be reorganized only while preserving
width, overflow behavior and downstream arithmetic.

Use upstream algorithms as design references, with pinned commits and entries
in the relevant `DERIVATION_MAP.tsv`. Preserve required license/attribution
notices for adapted code. Translate the data flow into Mojo GPU kernels and
explicit ownership; copying an option name is not implementing its semantics.
Backend support in competitors must be checked separately from general APIs.

## Performance priorities — user steering, 2026-09-10

Large-data training is the optimization target. Accept training-speed claims
and performance-driven defaults only with representative large real datasets,
including held-out quality, stable whole-fit timing and memory-pressure evidence.
HIGGS 1M is an example, not a universal size threshold: feature/class counts,
bins, depth and device capacity can dominate rows. Small correctness/smoke tests
and kernel diagnostics remain useful; they cannot establish a large-data gain.

Success is IDENTICAL versus the relevant GPU competitor on NVIDIA, across
all learners, including RF, ET and symmetric/oblivious GBDT. FAST performance
work is requested only for decision trees on the local MacBook. This latest
user clarification supersedes earlier FAST-versus-cuML/CatBoost NVIDIA plans.
An internal FAST/DETERMINISTIC/IDENTICAL speed contest is not a deliverable.
Keep supported modes correct through shared source, but spend NVIDIA tuning
and benchmark effort on IDENTICAL. Do not describe identity as free or as an
acceleration mechanism; similar observed timings do not isolate identity cost.
DETERMINISTIC is not an integer-only algorithm. Integer counts may be shared
across modes while floating scoring, regression and reductions still differ.

Use cuML RF as the NVIDIA RF competitor and CatBoost GPU as the symmetric
GBDT competitor. ET needs an actual equivalent learner before claiming ET
parity; cuML RF is only a labeled contextual comparison. Qualify HIP correctness
and identity separately: cuML is not an AMD backend. Preserve competitor
semantics, quality measurements and honest workload boundaries.

Current order:

1. Dedicated NVIDIA IDENTICAL-versus-competitor baselines and profiling, starting
   with RF/cuML and symmetric GBDT/CatBoost. Include fit and prediction
   separately, transfer-inclusive timings, memory, quality and repeated runs.
2. Optimize IDENTICAL RF/ET bottlenecks: existing shared histogram/count
   candidates, scratch/input reuse, independent-tree overlap, and shared GPU
   forest inference. RF already has cuML-derived shared histograms.
3. Qualify CUDA/HIP correctness and identity for changed paths. MacBook FAST
   decision-tree tuning is a separate target, not evidence for NVIDIA defaults.
4. Continue useful feature gaps and prepared-data ownership in bounded slices;
   retain the full backlog without displacing competitor performance work.

Independent-tree overlap is not implemented in production yet. Introduce it
behind an explicit opt-in concurrency control, defaulting to the existing serial
schedule. Use private per-slot scratch, fixed tree IDs/RNG and output slots.
Shared integer counts can be scheduled freely only within their overflow
contract; floating reductions within trees and final forest aggregation must
retain their pinned association in IDENTICAL. Fixed output order does not
require serial tree construction. Measure bandwidth contention, occupancy and
peak scratch memory: overlap can be slower. Promote only measured workload and
device configurations; keep a serial fallback. An opt-in flag alone is not a
performance improvement.

The user permits RunPod performance tests when warranted and has instructed
us to proceed. Start with a bounded single-GPU measurement session and automatic
teardown, not an open-ended campaign; never consume the protected training pod. Parallelize source work and review; serialize local
heavy builds through `tools/with_build_lock.sh`. Commit completed slices and
push to main after integration/conflict review, as explicitly requested.

Serial pipeline CV now has a bounded public API; see the
[cross-validation contract](GPU_CROSS_VALIDATION.md). Native GPU splitting,
encoders and device-resident pipeline ownership remain queued.

## Competitor feature gaps and implementation sequence

The [feature-by-feature comparison](TREE_COMPETITOR_FEATURE_MATRIX.md) lists
LightGBM, XGBoost, CatBoost and MojoLearn side by side, with GPU restrictions,
implementation dependencies and source references. Core growth is present;
broad product parity remains incomplete. F2a numeric
[per-tree feature sampling](GBDT_FEATURE_FRACTION.md) is implemented; focus next
on its allocation costs and AMD/NVIDIA evidence before proceeding to interaction
masks and coherent leaf
regularization/monotonic bounds. P6 reusable Python datasets remains the
parallel performance priority. All three numeric modes stay supported.

## What is already present

| Area | Implemented scope | Remaining qualification or limitation |
| --- | --- | --- |
| GBDT growth | Symmetric, Depthwise, best-leaf-first Lossguide; depth and leaf limits | Policy names do not establish competitor equivalence |
| GPU non-symmetric objectives | Eleven objectives matching the audited CatBoost GPU registry | Broad CatBoost parity is incomplete |
| Split controls | CatBoost-style terminal row-count limit; optional `min_split_gain` for non-symmetric trees | Row count is not a guaranteed minimum in both children; gain units are our selected score |
| IDENTICAL partition statistics | Unchanged leaves cached; both new children recomputed with pinned schedule | M4 Lossguide fit improvement about 2.8%; new path needs cross-vendor qualification |
| Prepared numeric GBDT | Native reusable borders/input/target/weight buffers; three losses and all policies/modes | Python handles, eval/categorical pools and general workspace reuse remain |
| Child Hessian eligibility | Optional Newton-score control for RMSE/Logloss/CrossEntropy, including prepared/native/public APIs | Broader objective/score support and cross-vendor qualification remain; see [feature contract](GBDT_MIN_CHILD_HESSIAN.md) |
| RF column tiles | Opt-in two/four-column shared histograms; same row and label loaded for multiple features | Sep9 full-forest gates pass; no established speedup; defaults off |
| ET shared counts | Apple default for 5–32 classes; exact shared integer class counts | Strong local multiclass gains; binary/regression and other vendors keep old default |
| Symmetric row-index splits | Existing optional FAST experiment | Workload-dependent regression prevents global default change |
| NVIDIA stable partition | Existing single-pass large-leaf candidate | Large-leaf NVIDIA branch unqualified |
| CUDA streams | H100 probe confirms created streams expose selectable context views | No production concurrent-tree scheduler; buffer ownership and timing still need validation |

Evidence and bounded feature coverage:
[growth controls](NON_SYMMETRIC_GPU_PARITY.md),
[prepared data](PREPARED_GBDT.md),
[partition cache](PARTITION_STATS_CACHE_2026-09-09.md),
[RF candidates](../../ensemble/bench/RF_COLUMN_TILES.md),
[ET evidence](../../bench/results/et_shared_score_2026-09-09/README.md),
[CUDA stream audit](RF_CUDA_STREAM_AUDIT_2026-09-09.md),
[broader tree inventory](../TREE_ALPHA_FEATURE_STATUS.md).

## Performance work: preserve the learner

Latest steering: prioritize measured end-to-end speed over developer ergonomics.
The [dedicated GPU measurement procedure](TREE_GPU_MEASUREMENT_NEXT.md)
compares sampled allocation paths with identical model outputs.
The [reuse audit](TREE_CODE_REUSE_AUDIT.md) separates safe shared primitives
from algorithm differences. Optional feature sampling changes the learner; its
packing/cache costs need whole-fit measurements, not a histogram-count claim.


| ID / priority | Reference and implementation approach in Mojo | Acceptance gate |
| --- | --- | --- |
| P1 / now: RF histogram tiles | cuML batched shared histograms already underpin RF. Measure current two/four-column kernels that amortize row/label gathers and size shared storage to actual bins/classes; profile before adding other layouts. | Full model, prediction and OOB fingerprints; positive dispatch witnesses; stable NVIDIA IDENTICAL competitor timing (MacBook FAST separately); no default selection from noisy runs. |
| P2 / next device: RF tree overlap | Mirror cuML independent-tree stream scheduling. First prove selectable CUDA stream/context views; give each slot private Builder, split staging and scratch; retain original tree IDs, RNG and output order. | Serial/concurrent equality, uneven tree completion, bootstrap/weights/OOB, cleanup and copy dependencies; trace showing actual overlapping queues; whole-fit speed/memory measurements. |
| P3 / next device: ET shared histograms | Reuse the shared-count organization, preserving ET's random thresholds and candidate sampling. Tune sharding/occupancy by vendor and class count; do not replace random thresholds with RF quantile candidates. | Existing independent count/score oracles, full-forest fingerprints and repeated class-boundary timings; measure binary separately. |
| P4 / next device: symmetric partitioning | Validate existing single-pass stable scatter on NVIDIA at genuinely large leaves, then compare row-index movement with stationary statistics where supported. Preserve row order and all read-after-write dependencies. | Leaves above 500k rows, empty/all-left/all-right/ragged cases, complete trained-model identity and stage/full-fit timings; keep regressions opt-in. |
| P5 / after profile: retained non-symmetric work | Extend the existing CatBoost-derived histogram reuse/cache only where profiling still shows rescans. Retain unchanged leaf candidates; invalidate on changed statistics, sampling masks or eligibility. Evaluate a stable best-leaf queue if host selection is material. | Same winning leaf and tie order at every step; no arbitrary batching of Lossguide decisions; both split children invalidated; sabotage checks and full models. |
| P6 / next: prepared pools and workspaces | Follow LightGBM Dataset/XGBoost quantized-matrix lifetime ideas using owned Mojo buffers. Expose current numeric GBDT pool to Python, then safely reuse capacity for fit scratch. Assess RF quantiles and ET raw-input pools separately. | Explicit frozen data/border/seed contract, mutation/lifetime tests, repeated independent fits, memory plateau, public fit timing including one-time preparation and amortization. |
| P7 / after profile: host overhead | Profile Python packing, quantization, allocations, transfers and result materialization separately. Avoid duplicate copies and object conversion; use existing contiguous/device input paths where present. | End-to-end improvement with unchanged inputs/models, dtype/layout/NaN validation, ownership tests; report kernel-only time separately. |
| P8 / later: inference | RF/ET public prediction currently reconstructs trees and copies rows into native lists each call. First traverse borrowed flat buffers directly, then add owned resident GPU models inspired by cuML forest inference. Preserve tree accumulation order and postprocessing; audit GBDT separately. | Public prediction/probability/apply/save-load equivalence, cold/warm and small/large batches, measured memory/throughput; no reassociation in IDENTICAL. |
| P9 / later: heterogeneous histograms | Pack feature histogram offsets using actual bin counts instead of padding every feature to the maximum, where profiling shows wasted memory/work. Extend existing layouts rather than duplicating histogram subtraction. | Boundary/bin-offset oracle, constant/one-hot/unequal-bin fixtures, full models and memory/fit measurements in every enabled mode. |
| P10 / after profile: small frontiers | Single trees cannot benefit from cross-tree overlap; ET best-first expands at most one node per tree per cycle. Profile frontier downloads/launches, then consider fused small-node kernels and device frontier compaction without changing priority. | One-tree and uneven forest workloads, same frontier order/ties and exact leaf budgets, end-to-end latency and identity. |

Single-tree native engines share relevant ET builder paths, but standalone
public DecisionTree exports are still a separate API task. Test one-tree native
models explicitly rather than assuming forest tests cover their dispatch.
RF already builds quantiles/bins once per forest: prepared RF work targets
reuse across fits. ET pools retain raw inputs; node-local random thresholds are still drawn per fit.
The binding overhead audit points to `bindings/_mojolearn_rf.mojo`
(`_rebuild_trees`, prediction and `_forest_out`) and
`bindings/_mojolearn_trees.mojo`; typed bulk model export can also replace
per-node Python scalar construction during fit.

## Growth features: explicit changes to the learner

| ID / priority | Reference | Mojo implementation and dependency | Required evidence |
| --- | --- | --- | --- |
| F1 / implemented bounded slice: minimum child Hessian | LightGBM `min_sum_hessian_in_leaf`; XGBoost `min_child_weight` | Apply eligibility to each candidate before winner selection using actual objective curvature, not a score denominator mislabeled as Hessian. First slice supports NewtonL2/NewtonCosine with RMSE/Logloss/CrossEntropy and explicit weighting/bootstrap semantics. Other scores/losses need a separate curvature plane or audit. Default disabled; plumb through native, prepared, Python and binding APIs. | Analytic weighted regression/logistic cases, equality boundary, highest-score-ineligible but second-best-valid split, default fingerprints, mode readback and public fits. |
| F2 / bounded per-tree slice implemented; node/level later | LightGBM per-tree/per-node fractions; XGBoost `colsample_*` | Deterministic masks keyed by seed/tree/node and a specified stable feature order. Numeric per-tree sampling now reuses the existing searchers with selected-bin projection. Node/level sampling remains queued; measure packing/cache overhead before speed claims. Specify intersection with categorical sources, interaction constraints and shared symmetric-depth splits. | Fraction-one default equivalence, known RNG vectors and selected-feature witnesses, no empty masks, all policies/modes, cross-device masks, work counters and quality/time comparison. |
| F3 / next: interaction constraints | XGBoost/LightGBM permitted interaction groups | Carry allowed-feature state along each path. Respect overlapping groups; intersect with sampling masks. Define symmetric-tree shared-split eligibility explicitly. | Exhaustive tiny path oracle, overlapping/disjoint groups, invalid feature IDs, unseen features, save/load and constraint verification over every model path. |
| F4 / later: monotonic constraints | XGBoost constrained split evaluation; LightGBM monotone bounds | Start with numeric features and a bounded objective/leaf-estimation profile. Carry descendant lower/upper bounds, score feasible leaves and enforce bounds in final estimation. Split filtering alone is insufficient. | Independent ordered-pair predictions, descendant-bound oracle, weighted/NaN cases, repeated estimation steps, round trips and cross-device identity. |
| F5 / later: L1 and bounded updates | XGBoost `reg_alpha`/`max_delta_step`; LightGBM leaf regularization | Specify objective scaling and apply soft-thresholding/bounds coherently to gain and leaf estimation. Integrate with monotonic bounds and iterative leaf updates; do not only change the score kernel. | Closed-form one-leaf/two-leaf cases, zero/default equivalence, threshold edges, weighted losses and independent loss checks. |
| F6 / later: learned missing direction | XGBoost/LightGBM missing-value split routing | Preserve current CatBoost Min/Max/Forbidden defaults. Optional per-split route requires separate missing statistics, eligibility/scoring both directions, model serialization and inference routing. | All-missing/mixed/absent-at-train cases, tie rule, both directions chosen in witnesses, constraints interaction and round trips. |
| F7 / later: multiclass non-symmetric growth | LightGBM/XGBoost multiclass boosting designs | Choose/document classwise scalar trees versus vector leaves; neither follows automatically from current symmetric multiclass support. Implement dimensions, objective statistics, scoring and prediction together. | Independent multiclass loss/probabilities, class weights, per-policy trees, memory scaling and mode/cross-vendor checks. |
| F8 / assess: stricter leaf-size controls | LightGBM/XGBoost-style child eligibility, contrasted with current CatBoost parent terminal size | If requested, add a separate explicit both-child count/weight constraint rather than silently redefining `min_data_in_leaf`. Reuse F1 candidate-filter machinery where correct. | Tiny skewed partitions distinguishing parent/child semantics, counts versus Hessians, equality and weighted cases. |

Feature controls may reduce search work, but changing the fitted model is not
a performance-only win. Report heldout quality and tree size alongside timing.
LightGBM's node feature sampling documentation explicitly cautions that it
does not necessarily accelerate training; our implementation must measure
whether candidate pruning actually avoids histogram work.

## Broader parity backlog

These items are tracked, not a commitment to implement every competitor API
before shipping the core GPU growth controls.

| Area | Existing boundary and how to extend it |
| --- | --- |
| Categorical ingestion and combinations | Existing coded categories, one-hot/simple CTRs and experimental depth-two FeatureFreq are partial. Mirror CatBoost stable category mappings and permutation-aware CTR construction in explicit stages: raw-value encoding/unseen handling, arbitrary-depth combination state, training/prediction tables and leakage gates. Learned LightGBM/XGBoost categorical partitions would be a separate algorithm option. |
| General Ordered boosting | Existing numeric single-permutation OrderedRMSE is bounded. Extend objective derivatives and prefix-state estimation before categorical/multiple-permutation composition; retain independent leakage and zero-mass-prefix tests. |
| Ranking | Add group/query/pair contracts before objective names. Start with one chosen ranking loss and independent gradient/metric reference; implement GPU derivatives, sampling, estimation and group-aware validation. CatBoost pairwise and LightGBM/XGBoost LambdaRank variants are different algorithms, not aliases. |
| Multi-target / uncertainty losses | Specify approximation dimensions, independent target calculus and model representation; add corresponding prediction semantics. Do not reinterpret scalar losses or merely accept names. |
| Leaf weights and inspection | Audit available serialized metadata; add retained leaf mass/count semantics where absent, then stable inspection APIs. Separate counts, effective weights and Hessians. |
| Custom objectives / callbacks / continuation | Define owned gradient/Hessian buffers, validation and device transfer boundaries; version model/training-state metadata for continuation. Specify callback frequency and cancellation. Python callbacks are orchestration, not a CPU tree learner. |
| SHAP / attribution and interchange | Evaluate TreeSHAP GPU traversal after leaf/path metadata is reliable; check additivity with an independent reference. Export only representable categorical/missing/objective semantics and reject lossy conversions explicitly. |
| Sparse input | Add CSR/CSC/device sparse contracts, sparse-aware histogram construction and explicit absent-value semantics. Avoid hidden dense expansion; measure memory and fit quality. |
| Distributed / multi-GPU | Separate later project: partitioning, deterministic collective reduction schedule, synchronized winning splits and failure recovery. Single-GPU identity does not imply distributed identity. |
| Additional sampling/boosters | Assess GOSS/gradient-based sampling, DART, feature weights/fixed splits only for concrete workloads. Each changes learning behavior and needs documented RNG, estimator math and quality gates. |
| RF/ET public API completion | RF dict/balanced class weights now reuse the existing native engine; see [scope and qualification](RF_CLASS_WEIGHT.md). OOB remains refused pending GPU accumulation rather than exposing the current host traversal. Expose existing standalone tree engines with their actual semantics, not an implied sklearn CART equivalence. ET sample weights, missing routing and additional regression criteria require genuine engine work; weighted counts need a defined replacement for its current exact integer/rational score contract. |
| Packaging | Rebuild each affected native extension for its mode/vendor; exercise installed public fit/predict/save/load and record actual binary mode. Source checks do not certify wheels. |

## Measurement and promotion protocol

1. Record source commit plus any source diff, compiler/driver/device, build
   definitions, binary hash, dataset/split hash and exact parameters. Keep raw
   samples and full model fingerprints; preserve failed/noisy attempts too.
2. Start with bounded synthetic diagnostic workloads. For competitor claims,
   use real HIGGS and Year plus multiclass/regression workloads on the same
   device/session, matched data and explicit parameter differences. Measure
   heldout quality as well as time. cuML is the RF comparator; ET has no
   assumed equivalent cuML learner.
3. Warm up kernels and full fits, synchronize timers, alternate arm order and
   monitor reference/canary drift. Separate construction/upload, native fit,
   full public fit and inference. The current RF driver rejects canary spread
   above 1.1; a gain smaller than observed drift remains inconclusive even
   below that cutoff. Bound retries rather than collecting until a win appears.
4. A default change requires repeated gains beyond noise, representative
   sizes/classes/bins and regression checks on the targeted vendor. Follow
   the project campaign's five interleaved rounds/three sizes requirement.
   An opt-in experiment may finish with a useful negative result.
5. Correctness checks include independent histogram/gradient references,
   dispatch witnesses and negative controls, complete model/prediction/OOB
   checks as applicable, serialization, and actual public native exports.
   Exercise FAST, DETERMINISTIC and IDENTICAL, then NVIDIA/AMD/Apple for the
   scope claimed. Compare to the old path within mode before cross-vendor
   comparison; preserve prior arithmetic and tie schedules.
6. Historical HIGGS RF time ratio 5762/3314 ms = 1.74 is one workload, not a
   general gap or proof of its cause. No 1.0–1.2x cuML parity floor is promised.

## Reference map

- **cuML:** pinned `265b9da6a0e75dbef071a3168398b993a5ff6f0e`,
  [RF scheduler](https://github.com/rapidsai/cuml/blob/265b9da6a0e75dbef071a3168398b993a5ff6f0e/cpp/src/randomforest/randomforest.cuh),
  [batched histogram kernels](https://github.com/rapidsai/cuml/blob/265b9da6a0e75dbef071a3168398b993a5ff6f0e/cpp/src/decisiontree/batched-levelalgo/kernels/builder_kernels_impl.cuh).
  Mojo owners: `ensemble/randomforest.mojo`, `ensemble/decisiontree/`,
  `extratrees/impl/decisiontree/` for applicable ET ideas.
- **CatBoost:** audited local pin `54a8143a5904ea1cfe98fe0d84c31d48cf13369b`,
  [GPU non-symmetric registry](https://github.com/catboost/catboost/blob/54a8143a5904ea1cfe98fe0d84c31d48cf13369b/catboost/cuda/train_lib/pointwise_non_symmetric.cpp),
  [growth controls](https://catboost.ai/docs/en/concepts/parameter-tuning).
  Mojo owners: `gbdt/methods/greedy_subsets_searcher/`, `gbdt/targets/`,
  `gbdt/train.mojo` and `gbdt/prepared.mojo`.
- **LightGBM:** local source pin `3d1cf3011adfed7209ba54bfeb05e8b2309040e4`,
  [parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html),
  [column sampler](https://github.com/microsoft/LightGBM/blob/3d1cf3011adfed7209ba54bfeb05e8b2309040e4/src/treelearner/col_sampler.hpp),
  [CUDA split finder](https://github.com/microsoft/LightGBM/blob/3d1cf3011adfed7209ba54bfeb05e8b2309040e4/src/treelearner/cuda/cuda_best_split_finder.hpp).
  Use as eligibility/sampling reference, not as proof of identical scoring.
- **XGBoost:** [parameters](https://xgboost.readthedocs.io/en/stable/parameter.html)
  for child Hessian, feature sampling, interactions and bounded updates;
  pin the implementation revision before a source-derived port.
- **Existing project scope:** [GPU-only decisions](TREE_GROWTH_SCOPE.md),
  [historical tree handoff](HANDOFF_trees.md),
  [project roadmap](../../ROADMAP.md). Older historical artifact claims retain
  their original commit/device scope; this plan does not broaden them.

## Current execution ledger

| Work | State as of 2026-09-10 |
| --- | --- |
| Consolidated roadmap | Written in this file; keep status and evidence links current |
| RF timing refresh | Completed: all four timing windows failed stability; 120 model fingerprints match. [Sep10 evidence](../../bench/results/rf_column_tiles_2026-09-10/EVIDENCE.md). Defaults unchanged. |
| Minimum child Hessian | Implemented for the three audited scalar losses with Newton scoring; native/public checks pass in all modes on M4. [Feature and evidence](GBDT_MIN_CHILD_HESSIAN.md). Cross-vendor qualification remains. |
| Sub-byte layout gate | Retained for live histogram layouts; named pixi task and hardware-matrix entry both pass all three internal negative controls. [Audit and commands](SUB_BYTE_LAYOUT_GATE.md). |
| Pipeline expansion | A1 unweighted Float32 MSE/MAE/RMSE, A2 unweighted confusion/precision/recall/F1, and B1 RF/ET sklearn protocol implemented; see their contracts in the [detailed plan](GPU_PIPELINE_PLAN.md). [GPU log loss](GPU_LOG_LOSS.md) is implemented with build/smoke validation only; full numerical qualification is pending. [Binary ROC-AUC/PR curves](GPU_RANKING_METRICS.md) are implemented with local build/smoke checks only; they do not add learning-to-rank. [GPU MinMaxScaler](GPU_MINMAX_SCALER.md) adds bounded finite Float32 preprocessing with limited local checks; [GPU StandardScaler](GPU_STANDARD_SCALER.md) adds centered Float32 population variance under the same limited validation scope. [B2 GBDT adapters](GBDT_SKLEARN_ADAPTER_PLAN.md) now cover RMSE regression and binary Logloss classification with GPU Float32 probabilities, preserving the legacy learner API; local build/smoke evidence is recorded. Bounded [serial cross-validation](GPU_CROSS_VALIDATION.md) is implemented using optional sklearn splitting/cloning; native splitters and broader model selection remain planned. Keep DETERMINISTIC pending measurements. |
| Code reuse / GPU-only training | Shared RF/ET log helper landed with six mode/objective checks. ET CPU public dispatch retired; native public fits delegate GPU, explicit host references remain for checks. [Reuse audit](TREE_CODE_REUSE_AUDIT.md). Host RF/ET inference remains a separate GPU migration. |
| Numeric per-tree sampling | Implemented all three modes/policies with unchanged default fingerprints. [Contract](GBDT_FEATURE_FRACTION.md). Exploratory M4 timing establishes no speed gain; prioritize buffer reuse and dedicated GPU evidence. |
| Other features above | Planned; prioritize measured speed and bounded feature work as described above |

## September 10 NVIDIA baseline update

The [H100 IDENTICAL baseline](../../bench/results/nvidia_identical_trees_2026-09-10/README.md)
records RF/cuML, symmetric GBDT/CatBoost and standalone ET. Five measured fits
per MojoLearn learner retained model/prediction hashes. Competitor timing
spreads were too large for accepted parity claims; no performance defaults
changed. The CUDA stream capability probe passed. NVIDIA remains IDENTICAL
versus competitors for every learner; FAST is a MacBook decision-tree target.

The [symmetric CatBoost comparison audit](SYMMETRIC_CATBOOST_COMPARISON.md)
records the corrected no-noise profile, different structure-search dispatch,
and the evidence required before accepting a speed comparison. Historical H100
artifacts remain unchanged.
