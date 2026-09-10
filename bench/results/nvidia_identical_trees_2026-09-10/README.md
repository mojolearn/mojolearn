# NVIDIA IDENTICAL tree baseline — 2026-09-10

## Outcome

The current IDENTICAL RF, ET and symmetric GBDT bindings built and ran on a
separate RunPod H100. Each learner retained its model-array and prediction
hashes across five measured fits. This is same-device repeatability for these
fixtures, not cross-vendor qualification of every new tree feature.

RF/cuML and symmetric GBDT/CatBoost timing spreads exceeded the declared 1.10
max/min threshold. Retain all samples, including outliers; these are exploratory
baselines, not accepted parity results or evidence for a default change. ET's
standalone timing passed that threshold but has no equivalent GPU competitor
in this run. No performance kernel or scheduling default was changed.

| Workload | MojoLearn IDENTICAL median fit | GPU competitor median fit | Timing accepted? |
| --- | ---: | ---: | --- |
| RF, HIGGS 1M x 28, 100 trees, depth 16 | 4904.681 ms | cuML default streams: 3527.001 ms; one stream: 3859.076 ms | No; all three spreads > 1.10 |
| Symmetric GBDT, HIGGS 1M x 28, 100 trees, depth 6 | 471.659 ms | CatBoost GPU: 895.264 ms | No; both spreads > 1.10 |
| ET, Covertype 90K x 54, 100 trees, depth 16 | 1469.566 ms | None | Standalone timing only |

RF heldout log loss/AUC: MojoLearn 0.538850/0.809906; cuML both stream settings
0.538814/0.809834. Symmetric GBDT: MojoLearn 0.542067/0.800716; CatBoost
0.542525/0.800430. ET heldout accuracy: 0.870300. These quality comparisons do
not imply identical algorithms or models across libraries. A lower observed
MojoLearn median is not evidence that identity is free or causes acceleration.

## Scope and provenance

- H100 80GB HBM3; driver 580.126.09; Mojo 1.0.0 ed45d567. Full versions,
  native-mode/vendor witnesses, binary/source/data hashes and parameters are
  in the JSON/log files and manifests. CatBoost 1.2.10, cuML 26.8.0.
- Source archive started at `45d65d55`; the RF mode getter and focused benchmark
  files were overlaid to `e4cd8a6b` before the full measurements. RF was built
  after its getter overlay. The subsequent unrelated main merges were not
  part of this remote build. No claim of whole-main or wheel validation.
- Each full run used one warmup plus five timed rounds. RF alternated ours,
  cuML default, cuML one stream; GBDT alternated ours/CatBoost. ET was alone.
  The runner uses fixed arm order within each round; it does not rotate the
  starting arm. Future noise diagnosis should examine order, CPU/NUMA and
  GPU clock effects rather than repeatedly sampling until a win appears.
- The timer includes constructor, host input packing, fit and synchronization.
  Model hashing and prediction/metrics are outside it. Prediction performance
  was not isolated. RF/ET public prediction still uses host traversal.
- RF and GBDT use the same fixed 500K HIGGS heldout tail. The requested
  Covertype cap is 100K total; the existing loader splits it into 90K training
  and 10K heldout rows. Dataset fallback is disabled.
- JSON `mode=identical` and inherited FSPEED headers identify the MojoLearn
  experiment. They do not claim CatBoost/cuML run an IDENTICAL mode. Competitor
  arithmetic and scheduling remain native; one-stream cuML is named separately.
- The early one-round synthetic RF/GBDT smoke JSON came from the initial helper
  and labels repeatability true after rescoring the same fitted model. It is
  **not** an independent repeat-fit or performance result. The final helper
  emits null for that one-round field. Only the three five-round runs above
  support the repeated-fit statements.
- Tcmalloc reported NUMA `mbind` warnings. Their timing impact is not isolated.
  No other work ran on this rented GPU during the full timing loops. The
  optional LightGBM CUDA build was intentionally terminated first (exit 143),
  because it was outside this comparison. Its failed probe is setup history,
  not a learner failure in the measured arms.

## CUDA streams

`rf_stream_api.log` / `.exit` record a passing real H100 probe: one initial
stream, two after creation, and successful selection/synchronization of the
new context view. This resolves the prior device-capability uncertainty.
It does not prove kernel buffer ownership, race-free tree overlap or a speedup.
The next scheduler implementation remains opt-in, with private per-slot scratch,
fixed tree IDs/RNG/output slots, pinned floating aggregation and a serial fallback.

## Rental lifecycle

Dedicated pod `3r2toker2hjtep`, created 12:11:24 UTC, with a 60-minute on-pod
API termination watchdog armed before work. Advertised pod rate: $3.49/hour.
Evidence was copied locally before manual termination at 12:28:05 UTC. DELETE
returned HTTP 204 and subsequent GET returned HTTP 404: removal verified.
The protected Samba training pod was not modified or used. Approximate compute
cost for 17 minutes is $0.99, excluding storage; this is not a final bill.

## Next work

1. Diagnose timing variance with per-arm clock/CPU/memory telemetry and stage
   profiles before promoting a speed claim. Preserve these unfiltered samples.
2. Implement an opt-in RF tree-overlap scheduler using the now-proven CUDA
   context views; validate private buffer ownership and full models before timing.
3. Evaluate existing RF column tiles and ET shared-count candidates on NVIDIA
   IDENTICAL, keeping defaults until both correctness and stable timing pass.
4. Implement shared GPU RF/ET inference without changing forest aggregation
   order. FAST performance work remains restricted to MacBook decision trees.
