# GPU tree growth scope and working memory

Consolidated implementation sequence and backlog: [DECISION_TREE_ROADMAP.md](DECISION_TREE_ROADMAP.md).

Decision recorded 2026-09-09 at the user's request. Keep this scope when
continuing tree work in later sessions.

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

## Current work

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

## Subsequent GPU capabilities to assess

- Minimum child Hessian/weight constraints and feature subsampling can reduce
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
