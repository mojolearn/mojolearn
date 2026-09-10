# GPU tree growth scope and working memory

Pipeline and mode decision (2026-09-10): [GPU_PIPELINE_PLAN.md](GPU_PIPELINE_PLAN.md).
Keep DETERMINISTIC supported. Latest priority is competitor-relative tree
performance; metrics and interoperability remain implemented foundations.
A1 unweighted Float32 errors, A2 unweighted GPU confusion/precision/recall/F1,
and B1 RF/ET sklearn compatibility are implemented on the tree lane; contracts and remaining work are tracked in the pipeline plan.

Consolidated implementation sequence and backlog: [DECISION_TREE_ROADMAP.md](DECISION_TREE_ROADMAP.md).

Decision recorded 2026-09-09 at the user's request. Keep this scope when
continuing tree work in later sessions.

- Prioritize large-data GPU training. Base training-speed decisions and
  performance defaults on representative large real datasets (HIGGS 1M is one
  example), recording memory pressure and held-out quality. Choose scale using
  features, classes, bins, depth and device capacity as well as row count;
  there is no universal minimum row threshold. Small correctness/smoke and
  diagnostic fixtures are welcome but are not large-data speed evidence.
- Optimize and extend the GPU learner. Do not build a CPU tree-training
  backend or pursue CPU-only CatBoost parity for its own sake.
- CPU preparation and coordination are intentional parts of the GPU pipeline:
  Python input packing, host border selection and metadata/control logic
  coexist with GPU histograms, scoring, partitioning and leaf estimation.
  A separate CPU learner might benefit small datasets or machines without a
  GPU, but no measurement here establishes a benefit to our GPU workloads.
- Use CatBoost GPU source as the baseline for existing algorithm behavior.
  Its current non-symmetric GPU registration includes the same 11 objectives
  as MojoLearn. Multiclass and Lq absence from that registry is not a missing
  port of an existing CatBoost GPU registration. Broader CatBoost feature
  parity remains incomplete (categorical combinations, ranking, etc.).
- IDENTICAL means MojoLearn's cross-device numeric contract, not bitwise
  agreement with CatBoost, LightGBM or XGBoost. Preserve existing default
  model bits. FAST and DETERMINISTIC should support new GPU features too,
  subject to their own numerical guarantees and mode-specific checks.
- Borrow useful LightGBM/XGBoost capabilities as explicit, documented GPU
  options. A shared option name does not imply the same objective scaling,
  split scoring, default, tie handling or resulting model.

## Initial September 9 work

These slices have been implemented or measured; the consolidated roadmap
records their current qualification and remaining device work.

1. Retain unchanged IDENTICAL leaf statistics; recompute both split children
   with the original pinned reduction schedule. Validate against a full-sweep
   reference with complete model/prediction/loss fingerprints.
2. Benchmark symmetric FAST row-index splitting. Do not flip defaults from
   noisy results. Validate NVIDIA stable single-pass partition on leaves
   exceeding 500,000 rows when an available device can actually run it.
3. Reuse explicit quantized numeric datasets and device buffers across native
   fits. Freeze quantization independently of training hyperparameters; do
   not implicitly cache mutable caller pointers or target-dependent CTRs.
4. Add optional minimum split-gain control to Depthwise/Lossguide. Disabled
   must retain the existing CatBoost-derived behavior. The threshold uses
   our selected split-score improvement, not XGBoost gamma's units.

Update 2026-09-10: optional [minimum child Hessian](GBDT_MIN_CHILD_HESSIAN.md)
is implemented for Depthwise/Lossguide with Newton scoring and
RMSE/Logloss/CrossEntropy. Native and public checks pass on M4 in all three
modes; this does not certify other vendors or all objective/score pairs.

## Subsequent GPU capabilities to assess

- Broader child-Hessian/weight constraints and feature subsampling can reduce
  search work as well as regularize trees. They change the learner when
  enabled and require explicit deterministic sampling/reduction rules.
- Interaction constraints restrict eligible features along each path.
- Monotonic constraints require consistent split eligibility, descendant
  bounds and final leaf estimation. Filtering split candidates alone is not
  sufficient to claim monotonic predictions.
- L1 regularization and bounded leaf updates affect both split scores and
  estimation; they need a coherent objective-specific implementation.

These are priorities, not claims of implemented support. Finish and measure
current work before broadening the implementation.

References: [CatBoost growth policies](https://catboost.ai/docs/en/concepts/parameter-tuning),
[CatBoost GPU registry](https://github.com/catboost/catboost/blob/master/catboost/cuda/train_lib/pointwise_non_symmetric.cpp),
[LightGBM parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html),
[XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html).

## Shared remote training protection

The user has an ongoing/planned RunPod training run for tomorrow. Concurrent
SSH sessions can share one endpoint; a second shell is not a second training
job. Use read-only connection/process diagnostics on that pod. Do not restart
it, alter SSH/network configuration, replace its environment or run competing
GPU benchmarks/builds without establishing that the training work is protected.
An unavailable direct TCP endpoint can have a separate RunPod basic SSH proxy
or browser-terminal route; connection refusal does not prove another session
has occupied the endpoint. See [RunPod SSH](https://docs.runpod.io/pods/configuration/use-ssh).

## GPU-only and code reuse decision — reaffirmed 2026-09-10

The user explicitly reaffirmed GPU access as a requirement. Do not add a
separate CPU training arm, automatic CPU fallback, or duplicate GPU learners
for each numeric mode. Host validation, feature-mask metadata, packing and
scheduling remain intentional orchestration; host test oracles remain allowed.

Random Forest's native owner is `ensemble/randomforest.mojo`, with its tree
builder in `ensemble/decisiontree/`. Extra Trees lives in `extratrees/` and
GBDT in `gbdt/`. Python entry points are `randomforest.py`, `extratrees.py`
and `ensemble.py` respectively; Python `ensemble.py` does not own the native
random-forest engine.

Before adding tree functionality, audit existing shared primitives and both
callers. Share identical data structures, arithmetic helpers, RNG primitives,
validation/packing and orchestration when contracts match. Retain explicit
algorithm differences: RF binned threshold search, ET random thresholds and
GBDT gradient/Hessian scoring are not interchangeable. Keep borrowed upstream
provenance and numeric schedules when extracting helpers. Similar filenames
alone are not evidence that entire builders should be merged.

Track concrete findings and bounded extractions in
[TREE_CODE_REUSE_AUDIT.md](TREE_CODE_REUSE_AUDIT.md). New per-tree GBDT feature
sampling should have one implementation reused across growth policies and
native/prepared/public callers, with mode-specific arithmetic policy rather
than independent copies. Refuse unsupported configurations instead of silently
falling back to a CPU learner or a different algorithm.

## Speed priority — latest user steering, 2026-09-10

Prioritize end-to-end training/inference speed over developer ergonomics.
Code reuse is valuable when it prevents divergent algorithms or supports faster
execution; cosmetic architectural unification is not the primary work queue.
Feature sampling is an optional learning control: packing fewer features can
reduce histograms while adding projection/allocation/synchronization and cache
rebuild cost. Measure the whole fit and heldout quality before presenting it
as a performance improvement. Investigate original-layout masked histograms
or reusable projection plans where they can preserve the sampling contract.
Keep default-one behavior unchanged and make no speed claim from work counts.

Clarification in the same turn: shared code and cheap dispatch remain welcome;
engineering convenience is not inherently a runtime cost. Distinguish a
one-time branch from actual data movement. The current feature-sampling
projection runs once per sampled boosting tree, not at every depth/leaf split
and not merely once per fit or growth-policy selection. Its cost can amortize
over a deep tree. Decide from whole-fit measurements rather than rejecting
packing on its name or assuming it is negligible.

## Performance priority update — corrected 2026-09-10

IDENTICAL versus GPU competitors on NVIDIA is the user's performance target
EVERYWHERE, for all learners. FAST matters only for decision trees on the
MacBook. This supersedes the earlier NVIDIA FAST comparison priority. Do not
optimize an internal mode ratio or claim that identity is free or causes a
speedup. DETERMINISTIC is not integer-only: integer statistics coexist with
floating scoring and regression work. Similar timings cannot isolate the cost
of identity across differently optimized implementations.

Use cuML RF and CatBoost GPU symmetric GBDT as the appropriate NVIDIA
comparators. ET needs a real equivalent competitor before claiming ET parity.
HIP correctness/identity remains in scope; cuML is NVIDIA-only.

Tree overlap is an audited design, not a production feature or flag. Implement
an opt-in concurrency control first, with private scratch, fixed RNG/output
ordering and a serial fallback. Preserve IDENTICAL's floating reduction order.
Promote only when whole-fit timing and memory evidence support it.
