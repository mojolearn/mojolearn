# Apple FAST trees: source-only A/B programme, 2026-10-06

This programme concerns **TREES ONLY, Apple GPU, FAST numeric mode**. The
owner requested a new worktree, a thorough idea list before implementation,
parallel implementation, and commit/push. **Do not compile, verify, test,
benchmark, measure, launch queues, or promote defaults in this session.**
Every new candidate is opt-in. There are no performance or quality results.
Bits may change from the previous version; predictive quality may not degrade.

Base: `07f2b3c00`; branch: `ideas/apple-fast-trees-20261006`.
Source notes used: `docs/apple-fast/PLAN-trees.md`, `NEXT_PASS_TREES.md`,
`notes/sym-hist.md`, `notes/sym-est.md`, and existing switch comments. Historical
notes are leads, not fresh evidence. Already rejected experiments must not be
reintroduced unchanged under a new name. Existing selectable mechanisms can be
used as prerequisites, but their provenance and previous outcomes stay explicit.

## Common experimental contract

The A arm is the frozen base algorithm with **no new experiment defines**;
the B arm adds the card's define(s), with otherwise identical input, settings,
seed, stopping policy, tree count, leaf budget, depth, borders, loss, weights,
categorical schema, ranking queries, and output requests. Controls for an
interaction hold all prerequisites equal in both arms. Never reduce training
rows, features, classes, trees, iterations, candidate splits, query pairs,
background rows or SHAP features to make B faster. Scheduling, representation,
fusion and reduction order are fair experiments. No benchmark-dimension or
dataset-name routing. Block, chunk and memory limits need a hardware or cost
explanation applying to neighboring shapes. Do not add Python runtime work.

New switches use `MOJOLEARN_AFT_<ID>` and must require both
`GLOBAL_NUMERIC_MODE == NUMERIC_FAST` and `has_apple_gpu_accelerator()`.
Related alternative defines are mutually exclusive. A source implementation
does not establish that it compiles, preserves quality, or improves speed.
If Mojo lacks a required capability, record the Modular ask; do not patch the
compiler or create an unsupported workaround. Each card's final implementation
record must identify exact code, defines, prerequisites and limitations.

Future qualification, **not authorized to run now**: use the full datasets and
actual public estimators saved in `tools/bench_board.py`,
`bench/speed/forest_speed_arm.py`, `tools/speed_gbdt_arm.py`, and for expanded
trees `tools/bench_board_algos.py`. Before any later timing, retain dataset
version/hash, split and actual dimensions, estimator settings, source/binary/
compiler/hardware provenance, toggles and resource policy. Audit internal caps;
`--rows full` is insufficient. Missing full recipes remain pending. Include
preparation, fit, required synchronization and consumed outputs; distinguish
fit, first prediction, repeated prediction, explanation preparation and
explanation use. A/B must include all affected estimators, neighboring shapes
and a full non-board dataset. Record losers and failed or incomplete cells.
Evaluate combinations as well as isolated switches. Never infer whole-workload
wins from a component. Use board tools only for later admitted results.

Quality gate keys used below:

- **C**: held-out classification logloss, AUC/accuracy and calibration under the
  existing task rule, weighted and unweighted; multiclass probability sums and
  class ordering; no new nonfinite outputs or weakened input refusal.
- **R**: held-out regression RMSE/MAE and existing estimator/task acceptance;
  constant/duplicate features, target magnitude, weights and empty leaves.
- **Q**: full-query NDCG and pairwise objective, group boundaries and seeded
  randomization; never split or truncate a query to fit a scratch budget.
- **O**: ordered-prefix causality, no target leakage, fixed permutations and
  categorical unknown-value behavior, plus C/R as applicable.
- **I**: isolation anomaly ordering/ROC/PR where labels exist, path-length
  normalization, score/decision signs and contamination threshold semantics.
- **P**: public prediction shapes, probabilities, thresholds, missing direction,
  weights and model export/reload; C/R/I for the underlying model.
- **S**: TreeSHAP local additivity, expected value, per-feature attribution
  error under the existing contract, background semantics, repeated features,
  zero covers, tree scales and multiclass outputs.

Numerical changes require the same quality gates as exact-looking changes.
Keep baseline repeat spread and quality uncertainty visible; do not invent a
tolerance or treat a single rounded metric as proof of no degradation.

## F: forests, decision trees and isolation forest (implementation lane F)

Applies to RF/ET classifier and regressor, standalone decision trees and tree
embeddings when they reach the changed builder, plus isolation forest as named.
Sources: `ensemble/decisiontree/`, `ensemble/randomforest.mojo`,
`extratrees/impl/`, `isolation_forest/impl/`.

| ID | B mechanism versus A | Why it might save time; quality/coverage obligations |
| --- | --- | --- |
| F01 | RF histogram feature packing: an alternate number of columns per block, selected by compile define | Rebalance shared-memory occupancy and reuse without removing any columns; C/R, narrow and wide inputs. |
| F02 | RF histogram row work per block: alternate batch size from byte/work budget | Reduce scheduling overhead or improve occupancy at the same bin precision; C/R, shallow and deep levels. |
| F03 | RF small-node split scheduling: alternate cooperative block size or occupancy policy | Avoid idle lanes when a node has few samples; C/R, mixed node sizes, retain min-leaf and tie semantics. |
| F04 | RF device partition/scan scheduling with alternate work per group | Change traffic/launch balance without changing sampled row multiplicities; C/R, uneven and empty segments. |
| F05 | RF bootstrap/gather loop fusion or vectorized row copying | Reuse sampled index and feature loads; preserve RNG draws, OOB membership and weights; C/R. |
| F06 | ExtraTrees split-score block geometry and cooperative reduction | Reduce idle threads/shared pressure while scoring every original random threshold; C/R, all criteria. |
| F07 | ExtraTrees partition chunk geometry | Amortize scan/scatter launches using an alternate byte budget; C/R, one-row and skewed children. |
| F08 | ExtraTrees multi-node work batching | Increase concurrent nodes subject to scratch memory, retain original feature sampling; C/R. |
| F09 | ExtraTrees feature min/max scan work decomposition | Reuse row/feature addressing and expose more parallel reduction work; C/R, constant features and finite extremes. |
| F10 | Isolation build threadgroup geometry | Rebalance shared memory and lanes while retaining the same sample and branching logic; I, varied subsample sizes. |
| F11 | Isolation scoring: multiple rows per thread or alternate row/tree decomposition | Reuse node loads and reduce per-row scheduling cost; I/P, single and large prediction batches. |
| F12 | Isolation row-major gather vectorization or per-thread feature tiling | Coalesce the existing sampled-row gather without a host sample loop or changing sampled observations; I, odd-width tails. |

## G: symmetric GBDT, quantization, categorical and ordered (lane G)

Applies to symmetric and symmetric-1000, categorical, ordered and multiclass
callers wherever a shared path is reached. Sources: `gbdt/gpu_data/`,
`gbdt/ctrs/`, `gbdt/methods/kernel/`, `gbdt/methods/ordered*`,
`gbdt/methods/leaves_estimation/`. Retain existing accepted algorithms.

| ID | B mechanism versus A | Why it might save time; quality/coverage obligations |
| --- | --- | --- |
| G01 | Histogram fold scans: cooperative work across bin lanes | Replace serial bins per feature where independent feature scans admit it; C/R/O, mixed bin counts and tails. |
| G02 | Histogram work multiplier derived from alternative occupancy target | Redistribute identical row work; use a different policy from the previously rejected fixed-multiplier retry; C/R, shallow/deep trees. |
| G03 | Split winner reduction: share the winner across a block or split resolve/apply | Avoid recomputing global candidate folds for every row; C/R, tie order and categorical policies. |
| G04 | Partition-statistics alternate reduction geometry | Trade lanes/chunks and memory reuse without rereading more planes; C/R, imbalance, empty leaves. |
| G05 | Device float-border/quantization kernel row tiling | Reuse border data and packed-word state; same borders and comparisons, all features and odd tails; C/R/O. |
| G06 | Compressed-index packing alternate rows per thread or block width | Improve packing occupancy at unchanged bit layout; C/R/O, binary/half-byte/one-byte features. |
| G07 | Leaf estimation fused derivative/statistic work or an alternate reduction tile | Avoid writing/reloading intermediates or reduce shared pressure; C/R, Newton vs gradient, weighted and multi-iteration cases. |
| G08 | Leaf update apply tiling, reusing a leaf value across several rows | Amortize metadata fetch and expose contiguous stores; C/R, all leaves, learning rate unchanged. |
| G09 | CTR exact count/sum preparation work decomposition | Reuse segment/index loads without changing categories, priors, borders or prefix exclusion; O/C/R, unseen categories. |
| G10 | CTR row lookup/binarization tiling | Coalesce lookup and output or increase independent queries per lane; O/C/R, skewed cardinalities and long segments. |
| G11 | Ordered fold statistics batched reduction geometry | Improve parallelism across permutations/prefix tasks while preserving causality and seed mapping; O/C/R. |
| G12 | Ordered estimation/apply task tiling | Reuse task descriptors and avoid launch/metadata overhead without skipping any prefix model; O/C/R, task count and weights. |

## N: non-symmetric GBDT and ranking (lane N)

Applies to depthwise, lossguide, pairlogit, YetiRank, and shared non-symmetric
categorical/multiclass paths. Sources: `gbdt/methods/greedy_subsets_searcher/`,
`gbdt/targets/kernel/`, corresponding ranking leaf-estimation paths.

| ID | B mechanism versus A | Why it might save time; quality/coverage obligations |
| --- | --- | --- |
| N01 | Exact Lossguide candidate batch geometry | Increase/decrease concurrent scoring scratch from cost reasoning; preserve global best-first ordering and leaf budget; C/R/Q. |
| N02 | Non-symmetric histogram work multiplier/feature grouping | Rebalance occupancy under the existing exact candidate set; C/R, neighboring feature counts and depths. |
| N03 | Partition split flag/scan tile geometry | Reduce redundant memory traffic while preserving stable row coverage; C/R/Q, empty and skewed children. |
| N04 | Non-symmetric histogram/statistics reduction geometry | Increase independent work without dropping Hessian/weight information; C/R/Q, large and small active leaves. |
| N05 | Multi-leaf split application tiling/fusion | Reuse split descriptors and row indices, reduce per-row indexing work; C/R/Q, early stop and leaf constraints. |
| N06 | Leaf cursor/prediction update tiling | Reuse selected leaf values and improve writes in the training loop; C/R/Q, multiclass dimensions. |
| N07 | PairLogit gradient pair tile | Rebalance independent pair work while retaining every pair and its weight; Q, short/long queries and ties. |
| N08 | PairLogit gather/accumulation work decomposition | Reuse document/query indexing and intermediate loads; Q, repeated documents and extreme margins. |
| N09 | YetiRank task block/worker scheduling | Tune work sharing across complete tasks without changing permutations or draw count; Q, variable query lengths. |
| N10 | YetiRank sorting traversal or merge tiling | Improve scratch/register traffic in the existing exact composite-key sort; Q, ties and long-query fallback. |
| N11 | Ranking estimation buffer fill/update fusion or tiling | Remove redundant passes while preserving the objective and every estimation iteration; Q, backtracking where supported. |
| N12 | Querywise reduction geometry | Parallelize query statistics with unchanged group membership and weighting; Q/R, singleton and highly uneven groups. |

## P: resident inference, TreeSHAP and tree-ensemble orchestration (root)

Includes trees exposed through expanded algorithms: DART classifier/regressor,
bagging, AdaBoost, random-trees embedding, and exported forests when they reach
the changed kernels. Kernel-SHAP and permutation-SHAP over non-tree models are
out of scope. Sources: `core/forest_inference.mojo`, `gbdt/resident_model.mojo`,
`gbdt/models/kernel/add_bin_values.mojo`, `xtrees/shap_device.mojo`,
`xtrees/dart_device.mojo`.

| ID | B mechanism versus A | Why it might save time; quality/coverage obligations |
| --- | --- | --- |
| P01 | Forest grove launch block width | Match more/fewer 32-tree groves to a threadgroup without changing tree order; P/C/R. |
| P02 | Forest ordered inference multiple rows per thread | Reuse forest metadata with independent row accumulators; P, strict-order/export callers and odd tails. |
| P03 | Forest grove shared-row staging with an alternative capacity policy | Reduce repeated input loads; compare against original staging only with equal prerequisites, P/C/R. |
| P04 | Forest class argmax row tiling | Reuse class-loop control across rows; preserve first-index ties and label mapping; P/C. |
| P05 | GBDT packed prediction rows per worker | Traverse identical splits for multiple independent rows, retain tree order; P/C/R/Q. |
| P06 | GBDT prediction postprocessing fused row work or block geometry | Reduce overhead in sigmoid/softmax/class mapping while retaining public output semantics; P/C. |
| P07 | TreeSHAP row quartet | Extend the existing row-pair concept with a distinct four-row tile; reuse tree/table metadata, S, odd row counts. |
| P08 | TreeSHAP kernel block width | Trade active tree/row tasks against register pressure; S, deep and shallow trees, all output classes. |
| P09 | TreeSHAP contribution chunk byte budget | Change the memory/launch tradeoff using a fixed byte budget, no dataset routing; S, peak memory and full backgrounds. |
| P10 | TreeSHAP row-to-output fold tiling | Reuse slot/feature routing across several rows while preserving all tree contributions; S, repeated features and scales. |
| P11 | DART residual/ensemble row tiling | Reuse dropout/tree metadata while retaining every selected tree and exact dropout draws; C/R/P. |
| P12 | DART add-tree/normalization update fusion or tiling | Amortize elementwise control and memory traffic; same normalization and learning rate; C/R/P, empty/full dropout sets. |

## Interactions: explicit A/B experiments, no extra default umbrella

These are planned factorial comparisons, not permission to run measurements.
Each cell needs its own complete define set. Never assume independent wins add.

| ID | Compare | Interaction being tested |
| --- | --- | --- |
| X01 | F01, F02 separately and together against A | Histogram occupancy versus row-batch scratch pressure. |
| X02 | F06, F07, F08 singles, pairwise, combined | ExtraTrees scoring concurrency versus partition working set. |
| X03 | F10, F12, then F11 on the resulting model | Isolation build/gather resources and end-to-end fit+score. |
| X04 | G01, G02, G03 separately and together | Histogram throughput versus scan/resolve launch overhead. |
| X05 | G05, G06, G09, G10 preparation singles and combined | Input/CTR preparation versus training and cold prediction. |
| X06 | G07, G08, G11, G12 singles and compatible pairs | Leaf estimation and ordered task memory pressure. |
| X07 | N01..N06 by grow policy, then relevant ranking N07..N12 | Tree search changes can interact with loss-specific work. |
| X08 | P01..P06 compatible inference combinations | Throughput, first-use latency and model preparation. |
| X09 | P07, P08, P09, P10 singles, pairs, combined | SHAP row reuse versus register/memory footprint. |
| X10 | P11, P12 separately and together | DART update fusion versus dropout-stage cost. |
| X11 | Best quality-qualified fit switches plus inference switches | Whole public fit+consume result, model handoff and memory. |
| X12 | Proposed complete default configuration versus frozen A | Required final full-workload decision; no promotion in this work. |

## Coverage and exclusions

No reduction in precision, approximate exp/log, reduced bin count, approximate
split search, early tree stopping, weaker SHAP method or fewer random draws is
silently bundled with scheduling changes. Those are different algorithmic
experiments with broader quality risk. This programme prioritizes removing
redundancy and changing work decomposition at unchanged model settings.

Do not build GPU graph replay, custom Metal code, private unified-memory
pinning, undocumented subgroup operations or compiler-IR patches. If future
profiling calls for unsupported functionality, record a precise Modular ask.
No host/CPU race, AMD/NVIDIA FAST tuning or IDENTICAL tuning belongs here.

## Implementation handoff rules

Keep logs out of context: save complete output to files, use targeted rg/grep
with bounded surrounding lines and short tails, and summarize exit status,
coverage, failures, and evidence paths. Expand only relevant diagnostic blocks;
never hide failures or infer full success from filtered output. Propagate this
reminder to any further delegated agent.

All implementation records must say **uncompiled, unverified, unmeasured**.
Do not run even syntax checks, manifest checks, format checks, diff checks,
test discovery, smoke drivers, imports, hooks that verify code, or dry-run
build/benchmark commands. Reading source and Git bookkeeping are allowed.
Root commits/pushes the shared worktree; agents do not commit or push.
