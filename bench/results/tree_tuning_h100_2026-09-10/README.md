# H100 tree tuning and GPU inference — 2026-09-10

Dedicated NVIDIA H100 80GB HBM3, driver 580.126.09. All NVIDIA learner artifacts
measured here use IDENTICAL. These experiments compare implementations within
that mode and relevant GPU competitors; they do not measure the cost of
identity or attribute acceleration to the identity contract.

## Training: ET shared counts

Full Covertype: 522,911 training rows × 54 features, seven classes, 58,101
held-out rows; 100 trees, depth 16, Gini, sqrt features, no bootstrap,
seed 0xACC2021. Private integer counts (mask 0) versus shared integer counts
(mask 15), actual CUDA/mode getter checks in every process.

ABBA passes, each one warmup plus three measured fits. All 16 exported models
and every training/test probability matched bit for bit. Test accuracy
0.6694204919020327 in both arms. Baseline median **6024.075 ms**; candidate
**5556.022 ms**, or **7.77% less fit time** in this configuration. Max/min
spreads 1.0577/1.0428; pass-median drifts 1.0499/1.0102. See
[raw samples](tuning/et-covtype/timing.json) and
[configuration/provenance](tuning/et-covtype/config.json).

The reduced smoke also passed exact model/probability checks; its two timed
samples per arm are not performance evidence. Full Covertype falls below the
one-million-row/256-MiB reminder heuristic. It is a useful complete multiclass
companion, not evidence across million-row and high-memory-pressure workloads.
No production histogram default is changed by this one configuration.

## Training: RF histogram tiles and cuML

HIGGS: 1M training rows × 28 features, fixed 500k held-out rows, 100 trees,
depth 16, 128 bins, sqrt features, bootstrap and seed 7. The three same-source
bindings differ only in the two/four-column histogram definitions. Isolated
one-tree probes proved the actual selected launch paths. The corrected driver
used six balanced permutations and one warmup per arm.

| Histogram path | Median fit (ms) | Max/min spread |
| --- | ---: | ---: |
| Reference | 5892.856 | 1.0501 |
| Two columns | 5766.476 | 1.0678 |
| Four columns | 5744.565 | 1.0841 |

All 21 full models and full native Float32 held-out probability arrays matched
exactly. The measured median reductions are only 2.14% and 2.52%; no default
promotion from this single window. [Raw results](tuning/rf-higgs-1m-single-prediction/summary.json)
retain predictions, quality, source/binary hashes and per-fit verification time.

A separate rotating RF/cuML comparison on the same dataset retained default
cuML streams and an explicitly labeled one-stream arm:

| Trainer | Median fit (ms) | Max/min spread |
| --- | ---: | ---: |
| MojoLearn IDENTICAL, reference histogram | 5838.652 | 1.0480 |
| cuML default streams | 3519.783 | 1.0627 |
| cuML one stream | 4254.789 | 1.0913 |

All five measured samples per arm met the 10% spread threshold. MojoLearn took
**1.66× cuML default fit time** on this cell; RF training parity is not achieved.
MojoLearn logloss/AUC remained 0.538850/0.809906; the comparator quality and
resolved parameters are in [the full result](tuning/rf-cuml-higgs-1m.json).
No claim about a typical dataset or another GPU follows from this HIGGS cell.

## New inference algorithm: parallel groves

The user requested retaining the existing algorithm and adding one that uses
GPU parallelism. Public names are `sequential` (existing host prediction,
default) and `parallel_groves` (new shared RF/ET GPU engine, opt-in).
The [engine contract](../../../docs/FOREST_INFERENCE_ENGINES.md) describes the
fixed 32-group graph, numerical differences, versioned archives and remaining
work. Training algorithms and defaults are unchanged.

Kernel checks passed on H100 CUDA and Apple M4 Metal. All **144 recorded scalar
fixture outputs**, for both ordered and grove kernels, matched across devices
(288 scalar bit comparisons). The cases include tree counts 1/31/32/33,
scalar/vector leaves, equality, signed zeros and subnormals. A cancellation
case independently verifies that grove and sequential association can differ.
Graph/bounds/nonfinite refusal and empty-row checks also passed. The original
compile error (platform-sized Int kernel parameters) and corrected logs are
retained; the fix uses explicit Int32 kernel arguments.

Public CUDA checks cover RF/ET classifiers and regressors: 33-tree fits,
independent NumPy fixed-graph oracle, repeated calls, versioned save/load and
nearby sequential outputs. Differences from sequential in these four fixtures
were at most 9.54e-7. See [public checks](inference/public-check.log).
These bounded CUDA/Metal checks are not broad HIP or large-model cross-vendor
qualification; the option remains experimental.

Large RF inference comparison: same MojoLearn forest trained on HIGGS 1M,
500k prediction rows × 28 features, 100 trees depth 16. Five measured public
calls after warmup, rotating arm order; fit excluded, host input/output transfers
included. cuML predicts with its separately trained, cached GPU forest.

| Inference algorithm | Median call (ms) | Range (ms) | Max/min spread |
| --- | ---: | ---: | ---: |
| Sequential | 25830.090 | 25599–26087 | 1.0191 |
| Parallel groves | 127.132 | 112–140 | **1.2458** |
| cuML GPU | 12.741 | 12.64–13.36 | 1.0568 |

The GPU engine's times are **exploratory**, with no certified speed ratio due
to its spread. Full prediction hashes repeated within each MojoLearn engine;
new/old probability arrays agreed within the explicit 2e-6 tolerances, not
bit for bit. [All samples](inference/rf-higgs-inference.json) are retained.
The new path is still materially behind cuML GPU. Persistent device models,
borrowed/staged-input ownership, vector-leaf traversal reuse and profiling of
boundary work remain next steps. Do not attribute this gap to a measured
identity cost; none was isolated.

## Provenance, failures and rental closure

The rental received archive `78027916`, then benchmark overlays from `47ae46ef`
and RF revisions through `ab5735a8`. Native histogram variants use the same
archived learner sources. The original [source hash manifest](source-sha256.json)
and [packed binaries](binary-manifest.json) belong to the training campaign.
Later inference source/binary overlays are separately retained, including exact
executed source, in [the inference manifest](inference/artifacts/manifest.json).
The archive marker alone does not describe either overlay set. Other concurrent
main-branch work was integrated locally, not compiled wholesale on this pod.

The first RF tile run was deliberately interrupted because verification called
host prediction twice. Its partial logs, exit 254 and executed driver are
retained. The replacement uses one prediction array for both shared quality
scoring and full-bit hashing; training timers were unchanged and the replacement
used a fresh directory. No samples from the interrupted run enter the results.

Prepared public-data `.npy` inputs are excluded from Git; shapes, hashes and
shared loader code are retained. No synthetic fallback was used. tcmalloc NUMA
binding warnings remain in raw logs and were not causally isolated. No clocks
were altered. GPU timings were serialized, separate from builds/downloads.
Nsight Systems was unavailable; the optional CuPy NVTX marker probe passed,
but there is no collected kernel timeline. One-second telemetry is not a
substitute for causal profiling.

Pod `8w0hv6x2l2ps0e` was created at 13:13:59 UTC with a 60-minute API teardown
armed before work. Manual DELETE returned 204 at 14:13:43, and GET confirmed
404 at 14:13:44. Approximate compute **$3.48** at $3.49/hour, excluding storage
and not an invoice. [Rental record](rental.json). The protected training pod
was untouched. No rented resource from this campaign remains running.
