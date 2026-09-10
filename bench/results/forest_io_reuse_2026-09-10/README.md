# RF/ET device I/O workspace reuse — September 10, 2026

Public `parallel_groves` now retains one exact-size device input/output pair.
Stable large-data pairs show modest gains. This changes allocation lifetime,
not training, threshold routing, the fixed reduction graph, or archive format.
The estimator's `sequential` inference default remains unchanged.

The main remaining performance finding is RF/HIGGS: NVIDIA IDENTICAL takes
**21.094 ms/call versus cuML GPU's 10.698 ms/call**, or **1.972 times the time**,
in the qualified eight-call throughput comparison. This is one cell, with
independently trained forests; it is neither same-model equivalence nor evidence
that IDENTICAL arithmetic causes the gap. Single-call RF timing was unstable.
No cuML Extra Trees comparison is claimed.

## Qualified allocation/reuse pairs

Times are medians. Each arm has eight retained samples and passes the existing
max/min <= 1.10 gate. Full output hashes match across all MojoLearn grove arms.
Single-call and throughput measurements are different scopes.

| Device/mode | Forest/data | Scope | Allocate each call, ms | Reuse, ms | Time reduction |
| --- | --- | --- | ---: | ---: | ---: |
| H100 IDENTICAL | RF/HIGGS | Eight calls/block, normalized per call | 21.485 | 21.094 | 1.82% |
| H100 IDENTICAL | ET/HIGGS | Single call | 25.926 | 24.510 | 5.46% |
| H100 IDENTICAL | ET/Year | Single call | 5.882 | 5.776 | 1.80% |
| H100 IDENTICAL | ET/Covtype | Single call | 4.294 | 4.221 | 1.70% |
| Apple M4 FAST | RF/HIGGS | Single call | 522.380 | 514.894 | 1.43% |

Reuse was faster in 6/8 RF CUDA throughput rounds, 7/8 ET/HIGGS single-call
rounds, 6/8 ET/Year rounds and 8/8 ET/Covtype rounds. These counts are descriptive,
not independent statistical confidence intervals. The changes are small in most
cells; do not extrapolate a large speedup or universal win from this grid.

The NVIDIA RF single-call pair, all three ET throughput pairs, and Metal
ET/Year failed stability. Their raw results remain included. ET throughput
medians were nearly equal; **no ET throughput gain is certified**. An unstable
transient or staged reference also means some complete multi-arm grids fail even
when the allocation/reuse pair above passes. No timing retries were run.

## Workloads and method

All cells use 100 trees, maximum depth 16. These are public host-input,
host-output predictions; they include input checks, transfers and output
construction. Fit, data loading and output hashing are outside prediction timing.

| Data | Training rows | Prediction rows | Features | Outputs | Nodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| RF/HIGGS | 1,000,000 | 500,000 | 28 | 2 | 4,019,922 |
| ET/HIGGS | 1,000,000 | 500,000 | 28 | 2 | 1,884,036 |
| ET/Year | 463,811 | 51,534 | 90 | 1 | 3,814,254 |
| ET/Covtype | 522,911 | 58,101 | 54 | 7 | 927,052 |

HIGGS uses the first 1M original training rows and original final 500k held-out
rows, verified by array hashes before testing. No rows were repeated to create
scale. Year and Covtype use the existing loader splits. Raw JSON records model,
data, binding and source hashes. CUDA reads back IDENTICAL/cuda and actual vector
selection; Metal reads back FAST/metal. Scalar Year retains scalar traversal.

The same fitted MojoLearn forest runs transient model upload, staged resident,
borrowed allocation, and reuse arms. Arm order rotates within one process. The
RF cuML reference is independently trained with cuML 26.8.0 and uses cached
nvForest GPU inference. The pinned dispatch audit is in
[GPU_FOREST_INFERENCE_NEXT](../../../docs/lanes/GPU_FOREST_INFERENCE_NEXT.md).
The predeclared CUDA grid contains single-call samples followed by one
throughput companion (eight calls/block, eight blocks/arm). Every retained
prediction is checked after timing. Separate staged calls are instrumented and
must not be described as kernel-only timing. Metal runs a separate two-cell,
single-call companion under the build and benchmark locks.

The dedicated CUDA pod runs setup, compilation, correctness and timing
sequentially, with no competing workload launched on it. Its telemetry is
retained; no clock controls were applied. Do not infer a cause for jitter from
NUMA `mbind` warnings or host logs. The setup log retains unrelated preinstalled
Torch dependency conflicts; this campaign does not use Torch.

## Ownership, validation and default selection

The workspace retains `4 * rows * (features + outputs)` bytes: 60,000,000 bytes
on HIGGS, 18,758,376 on Year, and 14,176,644 on Covtype. This is calculated
residency, not measured peak memory. Resizing replaces the old pair; zero-row
calls retain it; releasing/refitting the resident model frees it. In this A/B,
reuse workspace stays resident during later baseline arms, so telemetry is not
an isolated per-arm memory comparison. Inputs still upload and outputs still
download each call; this is not GPU-array I/O.

The qualified pairs above justify selecting reuse through the shared Python
protocol. No new public engine or numeric mode was added. The explicit
`forest_predict_resident_into_gpu` allocation reference and `--reuse-io`
benchmark remain reachable. CPU learning was not added.

Native CUDA IDENTICAL checks passed both policies (RF/ET), explicit allocation
and reuse, changed input values, same-size reuse, resizing, empty batches,
release/stale handles, finite input/output refusal, malformed graphs and cleanup.
Pre-rental Metal FAST/IDENTICAL lifecycle checks passed the same source behavior.
Public CUDA IDENTICAL checks passed all four estimators with both paths and an
independent fixed-graph oracle, repeat calls, pickle and versioned archives.
After selecting the default, public CUDA IDENTICAL and Metal FAST checks passed
with `reuse_io_override=false` and `io_entrypoint=forest_predict_resident_reuse_gpu`.
The existing 67 forest host tests passed. HIP and broader large-model
cross-vendor qualification remain open.

## Provenance and rental

Measured source: commit `591c82a9b541121b4fed845dbf42877b47e1fae6`.
The final change selects the already measured reuse entrypoint in
`python/mojolearn/_forest_protocol.py`; native implementation changes afterward
are comments only. `cuda/final-wrapper-sha256.txt` pins the promoted wrapper
used by the final public check. Native binary hashes and archived binary contents
were verified. `summarize.py` regenerates `summary.json` from raw cells.

`campaign.json` records the separate H100 pod, $3.49/hour rate, 55-minute
on-pod API teardown lease, and verified termination (DELETE 204, subsequent
GET 404). Estimated compute was **$0.91**, excluding storage, not an invoice.
The unrelated training pod was untouched. Pricing was checked with RunPod's
[live GPU query](https://docs.runpod.io/sdks/graphql/manage-pods); the returned
offer is retained in `offer.json`.

`artifact-manifest.json` identifies the source, result and binary archives held
outside git under `mojolearn-evidence/forest-io-reuse-2026-09-10`. Readable results
are committed here; packaged binaries and tarballs are not. `SHA256SUMS` covers
all committed files in this report directory except itself.

## Next performance target

Profile and compare nvForest-style packed node storage and compact leaf-vector
storage against the existing separate arrays. Pack once per resident snapshot,
retain tree IDs, RF/ET equality/subnormal semantics and the fixed grove graph,
and measure preparation as well as large public predictions. The source audit
now gives a bounded implementation plan. Training histogram traffic and
independent-tree overlap remain separate targets; this campaign makes no new
training-speed or CatBoost claim.
