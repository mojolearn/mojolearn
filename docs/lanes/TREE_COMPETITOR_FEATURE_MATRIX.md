# Decision-tree competitor feature matrix and delivery queue

Audited 2026-09-10 against MojoLearn lane base `f84f8e6b`, local source and
upstream documentation. This is part of the [execution roadmap](DECISION_TREE_ROADMAP.md).

## Where we stand

MojoLearn has substantial dense scalar GBDT coverage, all three growth policies,
RF/ET engines and bounded sklearn-compatible tree adapters. Broad CatBoost,
LightGBM and XGBoost product parity is incomplete. There is no meaningful
percentage without a weighted workload and backend definition: ranking or
categorical ingestion can decide usability even when many scalar options exist.

Compare three separate things: public feature availability, algorithm semantics,
and qualification/performance. A matching option name proves neither matching
models nor matching speed. IDENTICAL means qualified cross-vendor identity
within MojoLearn, not bitwise equality to a competitor. New controls should use
shared implementation in FAST, DETERMINISTIC and IDENTICAL; the latter pins
arithmetic, RNG and ties. Do not remove DETERMINISTIC without measurements.

Competitor cells below describe their documented library surface, **not blanket
GPU support**. CatBoost CPU/GPU and growth-policy restrictions, LightGBM OpenCL
`gpu` versus `cuda`, and XGBoost tree method/objective restrictions must be
checked at implementation time. No CPU learner is planned for MojoLearn.

## Growth and regularization

References for this table: [LightGBM parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html),
[XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html),
[CatBoost tuning](https://catboost.ai/docs/en/concepts/parameter-tuning) and
[CatBoost common parameters](https://catboost.ai/docs/en/references/training-parameters/common).

| Feature | LightGBM | XGBoost | CatBoost | MojoLearn and roadmap |
| --- | --- | --- | --- | --- |
| Growth policy | Leafwise, depth cap | Depthwise / lossguide | Symmetric / Depthwise / Lossguide | Implemented all three; CatBoost-derived semantics |
| Depth / leaf budgets | Both | Both, method-dependent | Policy-dependent | Implemented, policy-dependent |
| Child eligibility | Row/Hessian limits | Child Hessian | Terminal leaf row count | Parent terminal count plus bounded child Hessian; separate strict child counts F8 |
| Minimum split improvement | `min_gain_to_split` | `gamma` | Different score controls | Bounded non-symmetric `min_split_gain`; score units differ |
| Column sampling | Tree/node fractions | Tree/level/node fractions | `rsm`, GPU restricted | Numeric per-tree fraction implemented; node/level F2 remains |
| Row sampling | Bagging / GOSS | Uniform / gradient-based | Bootstrap families | Supported bootstrap subset; not GOSS or general gradient-based sampling |
| L2 leaf regularization | Yes | Yes | Yes | Implemented |
| L1 / maximum update | Both | Both | No directly equivalent general pair | Missing; F5 |
| Interaction constraints | Feature groups | Feature groups | No counterpart in audited common API | Missing; F3, choose exact overlap semantics |
| Monotonic constraints | Yes | Yes | CPU-only documented | Missing; F4, GPU extension beyond CatBoost baseline |
| Missing values | Native handling | Learned default route | Min / Max / Forbidden | Implemented CatBoost modes; learned per-split route F6 |
| Feature preference / forced splits | Contributions / forced splits | Feature sampling weights | Weights / fixed binary splits | General public controls missing; F9 below |
| Linear leaves | `linear_tree` | Not tree leaf equivalent of `gblinear` | No counterpart in audited API | Missing; deferred F10 |

Important boundaries:

- CatBoost documents `rsm` on GPU for pairwise ranking, not ordinary scalar
  regression/classification, and monotonic constraints on CPU. Neither is a
  claim of missing CatBoost GPU scalar parity. LightGBM documents linear trees
  for `cpu`/`gpu`, not `cuda`. Source links above govern these restrictions.
- Our child Hessian option currently supports RMSE/Logloss/CrossEntropy with
  Newton scores. Our minimum gain uses the selected CatBoost-derived score,
  not XGBoost gamma units. Neither replaces a strict both-child row limit.
- Lossguide already chooses the next best leaf; a depth cap does not make
  LightGBM a levelwise learner. We do not need another growth-policy name.

## Objectives, data and model lifecycle

References: [LightGBM features](https://lightgbm.readthedocs.io/en/latest/Features.html),
[LightGBM Python API](https://lightgbm.readthedocs.io/en/latest/Python-API.html),
[XGBoost GPU support](https://xgboost.readthedocs.io/en/stable/gpu/index.html),
[XGBoost categorical data](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html),
[XGBoost multiple outputs](https://xgboost.readthedocs.io/en/stable/tutorials/multioutput.html),
[CatBoost objectives](https://catboost.ai/docs/en/concepts/loss-functions) and
[CatBoost model API](https://catboost.ai/docs/en/concepts/python-reference_catboost).
General objective names also come from the parameter references above.

| Feature | LightGBM | XGBoost | CatBoost | MojoLearn and roadmap |
| --- | --- | --- | --- | --- |
| Dense regression / binary classification | Yes | Yes | Yes | Implemented; bounded sklearn adapters RMSE / binary Logloss |
| Multiclass | Yes | Yes | Yes, policy restrictions | Symmetric legacy API; non-symmetric extension F7 and broader adapters missing |
| Scalar objective breadth | Regression families | Regression families | Regression families | 14 total standard losses including multiclass; 11 GPU non-symmetric losses; not every competitor loss |
| Ranking | LambdaRank / XE-NDCG | Ranking objectives | Query / pairwise families | No group/pair trainer; R1 |
| Multi-target regression | No general native target-matrix objective in audited list | Experimental multiple outputs | MultiRMSE | Missing; R2 |
| Survival / predictive uncertainty | Not established by this audit | Survival objectives; objective-specific outputs | Separate objective families; backend audit pending | Missing; R2 requires objective-specific design |
| Native categories | Category partitions | One-hot / partitions | One-hot / CTR combinations | Coded one-hot/simple CTRs; raw schema and general combinations missing; C1/C2 |
| Ordered boosting | No CatBoost equivalent | No CatBoost equivalent | Yes | Bounded numeric OrderedRMSE only; C3 |
| Text / embedding feature processing | Not a comparable built-in pipeline | Not a comparable built-in pipeline | Feature-specific processing | Missing; C4 later |
| Sparse input | Yes | Yes | Input-format dependent | Main GBDT surface lacks sparse contract; S1 |
| Reusable training data | Dataset | (Quantile)DMatrix | Pool / quantized Pool | Native numeric prepared GBDT only; Python ownership P6 |
| Heldout stopping | Yes | Yes | Yes | Implemented bounded eval/stopping; general callbacks missing |
| Save / load / inspection | Yes | Yes | Yes | Legacy inference archives and some leaf metadata; adapter archive and richer inspection L1 |
| Continuation / refit | API-specific | API-specific | API-specific | General training-state continuation absent; L2 |
| Custom objectives / callbacks | APIs available | APIs available | APIs, backend restrictions | General public extension interfaces absent; L3 |
| SHAP / attribution | Contributions | GPU SHAP | SHAP APIs | General public SHAP missing; L4 |
| Distributed / multi-GPU | Distributed learners | Distributed GPU | GPU / distributed facilities | Missing production support; S2 separate project |
| Alternative boosters | RF / DART | Parallel trees / DART | Ordered / other controls | RF/ET separate engines; no DART; R3 |

Do not treat every row as an obligation to reproduce every API. In particular,
RF-like boosting modes are not proof of identical random-forest semantics;
Extra Trees random thresholds must remain distinct. cuML remains our RF
performance reference. Native RF OOB/class-weight capabilities need public
wiring; standalone DecisionTree exports and ET weighting/missing/criteria
support are tracked separately in the main roadmap.

MojoLearn evidence: [public tree inventory](../TREE_ALPHA_FEATURE_STATUS.md),
[non-symmetric scope](NON_SYMMETRIC_GPU_PARITY.md),
[GBDT adapters](GBDT_SKLEARN_ADAPTER_PLAN.md),
[prepared data](PREPARED_GBDT.md), and
[pipeline implementation ledger](GPU_PIPELINE_PLAN.md). GPU non-symmetric
objective coverage matches the audited CatBoost registry at
`54a8143a5904ea1cfe98fe0d84c31d48cf13369b`; it does not establish whole-library
parity. Existing M4 checks do not qualify new features on CUDA/HIP.

## Implementation order and concrete ownership

Yes: implement the useful gaps, in bounded slices. Retain current defaults and
models when a new option is disabled. Priorities below describe the next
feature sequence; they do not suspend already queued performance/pipeline work.

| Order / ID | Deliverable and how | Source / dependency / acceptance |
| --- | --- | --- |
| 1 / F2a | Numeric GBDT per-tree feature fraction, then F2b node/level sampling. Filter actual histogram/candidate work; keep original feature IDs in models. | Mirror LightGBM `ColSampler` and its CUDA consumer, audited below. Thread configuration through native train/prepared, binding ABI, legacy estimator and adapters. Freeze sample count, eligible-feature order and RNG; default-one byte equivalence plus selected-feature witnesses in all modes. |
| 2 / F3 | Path-aware interaction masks, initially numeric Depthwise/Lossguide. | Reuse F2 candidate-mask plumbing, but separate sampling from allowed features. Choose one upstream's overlap semantics. Exhaustive tiny-path oracle, invalid IDs, serialization; define symmetric shared-depth behavior separately. |
| 3 / F5 then F4 | Coherent L1/bounded leaf updates, followed by monotonic descendant bounds. | Pin XGBoost implementation first; modify both split scoring and final/iterative leaf estimation. Start scalar RMSE/Logloss. Closed-form optimum and monotonic prediction checks; split filtering alone cannot guarantee monotonicity. |
| 4 / F8 then F6 | Optional strict child counts, then learned missing direction. | Extend candidate eligibility; add missing-statistics alternatives, deterministic route ties, serialized route and inference support. Preserve CatBoost defaults. |
| Parallel foundation / P6 | Python prepared data and owned reusable workspace. | Extend existing Mojo prepared implementation, not a second trainer. Training-fold-only quantization, lifetime/refit identity and actual repeated-fit time. |
| Next / C1 then C2/C3 | Stable category schema/unseen handling; general combinations; broader Ordered boosting. | CatBoost mapping/table and permutation-state algorithms. Persist dictionaries/tables, prove train/eval separation; arbitrary-depth combinations are not a wrapper-only change. |
| Next / F7 | Multiclass Depthwise/Lossguide and public multiclass adapters. | Explicit classwise versus vector-leaf design; loss, dimensions, score and inference together. An extension beyond the audited CatBoost GPU non-symmetric registry. |
| Workload-selected / R1 | One ranking objective with groups/pairs and group-aware splitting. | Choose CatBoost or LambdaRank algorithm explicitly; GPU derivatives and metrics; independent gradients, ties and group integrity. AUC is a classification score, not this feature. |
| Later / L1–L4 | Portable adapter archives/inspection, continuation, extension hooks, SHAP. | Version training state and retained metadata before continuation/attribution. Preserve mode and category schema. |
| Later / F9 | Feature weights, fixed splits and feature-use penalties. | Distinguish score multipliers from sampling weights and penalties; pin the chosen upstream dispatch. |
| Deferred / F10, R2–R3, C4, S1–S2 | Linear leaves; objective-by-objective multi-target/survival/uncertainty; DART/GOSS; text/embeddings; sparse; distributed. | Separate designs and workload justification. Keep these visible without labeling general CPU support as a GPU port obligation. |

### First source-ready slice: column sampling

Local LightGBM pin `3d1cf3011adfed7209ba54bfeb05e8b2309040e4` was inspected:
[`col_sampler.hpp`](https://github.com/microsoft/LightGBM/blob/3d1cf3011adfed7209ba54bfeb05e8b2309040e4/src/treelearner/col_sampler.hpp)
lines 34–54 compute a rounded eligible-feature count with a nonempty minimum;
77–94 reset the tree mask; 96–186 combine node selection with constraints.
The actual
[CUDA tree learner](https://github.com/microsoft/LightGBM/blob/3d1cf3011adfed7209ba54bfeb05e8b2309040e4/src/treelearner/cuda/cuda_single_gpu_tree_learner.cpp)
lines 153–154 reset/pass the per-tree mask and 618–621 obtain node masks.
This establishes a real GPU-path reference, including permitted host metadata
work. The existing Mojo feature enumeration and histogram layout still need
an insertion-point audit before implementation. Do not merely zero the winning
score after all histograms were built and claim a training speedup.

RNG translation must be declared: reproduce the reference integer sampler and
its draw schedule, or explicitly document our different portable seeded mapping.
Neither choice promises same-seed models across libraries. Start with numeric
features; category expansions require an explicit original-feature selection
contract. Sampling is a learning change; measure quality as well as speed.

### Interaction constraints are not interchangeable

The inspected LightGBM `GetByNode` requires the accumulated branch features to
fit a permitted group when extending that group. XGBoost's
[documented overlapping-group examples](https://xgboost.readthedocs.io/en/stable/tutorials/feature_interaction_constraint.html)
include different allowed-feature expansion behavior. Pin and inspect its
implementation before choosing a contract; do not combine rules from the two.
Monotonic constraints likewise need inherited leaf bounds, as illustrated by
[XGBoost's constraint documentation](https://xgboost.readthedocs.io/en/stable/tutorials/monotonic.html).

## Qualification and scope

All three modes remain supported through shared source. New floating-point
work must use existing mode-aware primitives and preserve reduction order in
IDENTICAL. Integer masks still need stable feature order, RNG and tie rules.
Cross-device identity is a later evidence gate, not a synonym for compilation.

Update: numeric F2a is implemented; see the [feature contract](GBDT_FEATURE_FRACTION.md)
for its limited local checks and exploratory timing. Default-one behavior is
retained. Buffer reuse and dedicated AMD/NVIDIA evidence take priority over
additional feature knobs. Continue to commit/push completed slices to the lane.
Main integration remains a separate reviewed step.
