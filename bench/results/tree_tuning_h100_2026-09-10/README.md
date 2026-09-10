# H100 tree histogram tuning — 2026-09-10

Dedicated NVIDIA H100 80GB HBM3, driver 580.126.09. All learner artifacts
in this campaign use IDENTICAL. This compares implementation candidates
within that mode; it does not measure the cost of identity or attribute a
speedup to the identity contract.

## Completed ET experiment

Full Covertype: 522,911 training rows × 54 features, seven classes, 58,101
held-out rows; 100 trees, depth 16, Gini, sqrt feature sampling, no bootstrap,
seed 0xACC2021. Same-source private integer counts (mask 0) versus shared
integer counts (mask 15), actual CUDA/mode getter checks in every process.

ABBA passes, each one warmup plus three measured fits. All 16 exported models
and every training/test probability matched bit for bit. Test accuracy
0.6694204919020327 in both arms. Baseline median 6024.075 ms; candidate
5556.022 ms: **7.77% less fit time** in this measured configuration.
Max/min spreads 1.0577 and 1.0428; pass-median drifts 1.0499 and 1.0102.
See [raw samples](tuning/et-covtype/timing.json) and
[configuration/provenance](tuning/et-covtype/config.json).

The reduced real-data smoke also passed exact model/probability checks; its
two timed samples per arm are not performance evidence. The size reminder
labels full Covertype below its one-million-row/256-MiB heuristic. It is a
useful complete multiclass companion, not proof across million-row or
high-memory-pressure workloads. No production default changes follow from
this one configuration.

## Campaign provenance and remaining collection

The rental received source archive `78027916`, then the new ET benchmark from
`47ae46ef` and RF benchmark revisions through `ab5735a8`. These are harness
overlays; native variants all build from the same archived learner sources.
The archive marker alone does not describe those overlays. Configuration
source hashes and the eventual campaign manifest distinguish them.

The initial RF run was deliberately interrupted after discovering duplicate
host prediction in its verification path. Its driver is retained as
`interrupted-rf-driver.py`; the corrected driver uses one full prediction for
both quality and hashing. Training timers are unchanged. The queued retry
uses a fresh output directory. RF results and final rental teardown are still
being collected; this intermediate evidence commit makes no RF speed claim.

Prepared `.npy` inputs are excluded from Git; their shapes, hashes and the
shared real-data loader are retained. No synthetic fallback was used. Build
logs, binary witnesses, telemetry and final rental status will be retained
when the campaign completes. The exact bounded run scripts are alongside
this file. GPU timings are serialized; no protected training pod is used.
