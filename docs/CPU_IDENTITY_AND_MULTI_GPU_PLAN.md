# CPU identity and useful multi-GPU completion plan

Created 2026-09-18 from `50faa970b`, in `lane/next-wheel-coverage`.
This is an implementation plan, not evidence that its gates have passed.

## Objective and scope

CPU execution supplies an independent arithmetic reference for supported GPU
algorithms. It may remain behind the private verifier training context; public
CPU training and distributed CPU drivers are not completion requirements.
Existing public CPU inference remains supported. A CPU implementation is
independent execution, not proof of mathematical correctness by itself: use
negative controls, semantic checks and independent GPU records as well.

Multi-GPU work targets useful compute distribution, capacity, or independent-job
throughput. Label those outcomes separately. A `devices=(0, 1)` argument, two
visible GPUs, or two processes is not proof that both GPUs performed work.

The frozen 0.8.7 artifact remains separate. Ship it only after its existing
release gates pass; these new implementations belong in a subsequently built
and qualified wheel. Do not merge current main into the frozen release branch.

## A. CPU identity work remaining

### A1. Finish the current release proof

- Resolve nine UMAP saved-model expectation failures using fresh GPU recordings
  after the intentional transform batch-invariance change. Compare host outputs
  with those records. Preserve historical records and explicitly supersede the
  old expectations; never manufacture a GPU reference from the CPU answer.
- Finish clean and sabotage certification on ARM64 Linux, x86-64 Linux and
  macOS. Inspect each failure independently; refusal is not numerical agreement.
- Replay the installed verifier and saved-model checks against the exact wheel
  bytes to publish; retain source, native-binary and wheel hashes.

Status when written: release run 35350125464's Mac build/verification job passed;
ARM64 certification is running its sabotage stage after the earlier UMAP
failure. The other CPU matrix jobs are queued. Fresh Apple reference capture
and the already-merged RBF CPU route tests are queued for the local slot.
No new publication claim is made here.

### A2. Qualify 17 implemented ordinary CPU routes

| Group | Routes | Missing work |
| --- | --- | --- |
| Neural (5) | mamba3, transformer, transformer-window, samba, samba-untied-dropout-accum | Current all-fixture independent witnesses for every retained property; scoped admission currently rejects missing batchgrad, batchscale, ragged and stepfull in the new GPU records |
| Classical (12) | gp-optimize, gp-optimize-restarts, svc-poly, gbdt-query-rmse, gmm-random-init-sample, gmm-sample, gp-normalize-y, gp-sample-y, gp-sample-y-normalize, gpc, gpc-multiclass, ivf-extend | Complete current NVIDIA/AMD witnesses, all applicable properties, negative controls and installed replay |

For each route, use all nine current fixtures and at least two repeats.
Capture train, inference, save/reload and batch checks plus applicable gradient,
batch-scale, ragged, full-sequence-versus-step, and sampler/trainer properties.
Compare CPU, Apple, NVIDIA and AMD under the applicable reference admission
policy. Preserve the current table's properties when merging references.
Remove a hold only after admission and an observed installed CPU replay pass.
Do not treat N/A, OWED, REFUSED, or stable-but-wrong repeated results as passes.

Deliverable: a per-lane/per-property completion ledger linked to retained
records, with remaining holds named explicitly. The 17 are routes, not 17
missing algorithm implementations and not an exhaustive parameter inventory.

### A3. Close real CPU oracle/inference gaps

1. **KernelRidge / Nystroem kernels.** Host saved-model inference currently
   accepts linear and RBF only; polynomial, sigmoid and Laplacian are refused
   by `_HostKernelMethod._host_refusals`. Inventory which of those kernels the
   GPU surface actually supports, implement matching host prediction/transform
   arithmetic for that scope, and gate fitted-model reloads and batch invariance.
   Add fit/oracle arithmetic only where needed to independently verify the GPU
   fit; public CPU `.fit()` is not required. Precomputed-kernel support is not
   implicitly added.
2. **Whole loaded CausalLM.** Add a verifier lane covering checkpoint parsing,
   tensor mapping, block composition, prefill, carried-state decode, cache reset
   and greedy output. Use tiny architecture-specific fixtures with nontrivial
   weights, tied/untied heads and supported low-bit formats; compare logits
   bitwise before comparing token IDs. Add a real supported-checkpoint smoke
   separately. Block evidence alone cannot certify loader/composition behavior.
3. **Property audit.** Reconcile every supported ordinary lane with the shipped
   table, including low-bit variants: current fixture revision, real numerical
   references, device witnesses, batch and applicable decode/gradient properties,
   and a negative control that was observed to trigger. Publish exact remaining
   cells rather than claiming completeness from the empty saved-model registry.
4. Finish the queued RBF route regression, retaining failures/skips honestly.
   Do not expand this into implementing the other parallel CPU drivers.

For GPU-distributed paths, compare against the canonical one-GPU result and its
CPU arithmetic oracle where available. Test GPU placement, transport, ordering
and ownership directly on GPUs. A distributed CPU clone is not required.

## B. Shared physical multi-GPU verification gate

Build this gate before admitting any new multi-GPU claim; use it to refresh
existing drivers too. Historical records cover substantial NVIDIA/AMD scope,
but are not a current-artifact certificate for all 50 parallel harness lanes.

- Pin one wheel/source and inputs for one-GPU and two-GPU runs on each vendor.
  Record physical UUIDs, architectures, visible-to-physical index mapping and
  the devices actually used by each worker/session.
- Record logical shard/layer/index ownership and evidence that each assigned
  device executes its assigned work. Reject duplicate physical devices and a
  silently unused second device when the fixture is meant to exercise both.
- Compare full outputs and fitted state, plus relevant gradients, statistics,
  optimizer state, checkpoint/reload, caches and intermediate traces. Preserve
  canonical global IDs, random draws, tie order and reduction order.
- Exercise uneven partitions, final partial tiles, empty work on a device where
  allowed, non-default device order, ties, supported weights/masks and repeated
  runs. Include AMD transfers below and above the historical 1 MiB boundary.
- Negative controls: wrong shard order/global offset, dropped work, altered
  transferred bytes, or wrong state ownership must be caught. A native refusal
  qualifies as a control only when that exact refusal is explicitly expected.
- Gate both NVIDIA and AMD. Apple/CPU provide reference answers where useful;
  neither substitutes for physical multi-GPU execution.
- Measure throughput/latency separately from bitwise correctness, including
  transfer/assembly overhead. Measure per-device peak memory and host memory.
  Claim increased capacity only after a controlled workload exceeds the
  admitted one-device memory budget and completes with the intended distributed
  residency. Host staging/offload must be identified explicitly.

Use bounded cloud jobs, retained receipts, watchdogs and verified deletion.
Respect the existing numerical-worker limits; do not overlap new rentals with
active CPU certification. Hardware absence leaves the gate pending, never passed.

## C. Six implementation work packages

### C1. Generic loaded-language-model inference across GPUs

Priority: high capacity value. Depends on A3's whole-model baseline and B.

Start with an explicit layer-to-device map for supported CausalLM architectures.
Keep each layer's weights and KV/recurrent state on its owner; transfer hidden
activations at boundaries. Preserve absolute positions, masks, sequence lengths,
state reset and tied embedding/head semantics. Use the canonical block kernels.
Define checkpoint materialization and host-memory limits explicitly.

First scope: prefill, step and greedy generation on two GPUs; FP32 first, then
supported BF16/int8. This is layer/model parallel inference, not a claim of
within-layer tensor parallelism. A layer, head or an individual layer's cache
that exceeds its owner remains unsupported in this first scope.

Acceptance: exact single-device logits and state across split points, multi-step
cache reuse, batch/ragged checks, reload and placement controls on both vendors;
a model or total cache that exceeds one-device capacity; end-to-end memory and
latency report. Tensor-parallel giant layers and cache sharding within a layer
require a subsequent design and separate proof.

Likely seams: `models/causal_lm.py`, block inference/state classes, device/session
ownership and transport code. Expose an explicit public entry only with its
identity lane and installed-wheel gate.

### C2. Distributed IVF index storage and search

Priority: high capacity value. Depends on B and the existing IVF oracle.

First implement query scheduling over replicated indexes as a small correctness
milestone, labelled throughput-only. The capacity milestone partitions index
lists/vectors with stable global IDs. Share the canonical coarse centroids and
query routing; search the same selected lists as single-device IVF, then merge
candidates using the existing distance/tie ordering. Do not compare an
approximate IVF contract with an unrelated exhaustive-search contract.

Preserve nprobe, metric semantics, empty lists, uneven list sizes, k exceeding
available candidates and extend/save/load behavior. Define extension ownership
and duplicate-ID rejection. Distributed build follows only after search/storage
works: reuse canonical quantizer training and preserve assignment/CSR ordering;
independent per-shard quantizers are not equivalent.

Acceptance: exact distances/IDs and index reload against single-device IVF,
both vendors, transfer/merge negative controls, and an index larger than one
GPU's capacity with measured residency. Replication alone does not close C2.

Likely seams: `_ivf_impl.py`, native IVF build/search/storage, worker transport.

### C3. GaussianProcessClassifier

Priority: useful compute distribution; two separately admitted milestones.

First distribute independent multiclass one-versus-rest fits/predictions and
assemble estimators in original class order. Preserve each class's initialization,
convergence and the final probability normalization. Binary GPC gets no benefit
from this class-level split and must be described that way.

Then assess covariance/factorization partitioning within an individual fit for
large binary problems. Reusing GP row work alone does not pool the root matrix
or factor; treat those as explicit capacity dependencies, not completed by the
class scheduler.

Acceptance: exact per-class state, iteration counts, probabilities and labels
for binary/multiclass controls, class-order/tie/imbalance cases, reload and
physical two-GPU records. Claim binary capacity only after the matrix-state
milestone passes a capacity gate.

Likely seams: `_gpc_impl.py`, `parallel_classical.py`, GP and Cholesky bindings.

### C4. Distributed ARIMA / Holt-Winters prediction

Priority: bounded throughput feature; a suitable first implementation after B.

Shard independent series from fitted models, carry the correct per-series state
and reassemble predictions in original series order. Cover in-sample prediction,
forecast horizons, seasonal variants and supported exogenous inputs. Preserve
existing startup NaN behavior. Do not imply that this repairs the separate
parallel ARIMA-fit exogenous-input refusal.

Acceptance: exact one-vs-two-GPU forecasts over uneven series partitions,
short/long horizons, restored models and a deliberate series-order fault on
both vendors. Benchmark enough independent series to assess transfer overhead.

Likely seams: `parallel_classical.py`, forecast model state and worker protocol.

Implementation audit (2026-09-18): Holt-Winters `predict` currently invokes
`bindings/holtwinters_host_predict.mojo` for in-sample arithmetic even in the
GPU binding; `forecast` and wholly out-of-sample `predict` use the device
forecast entry. Merely dispatching the existing in-sample method to GPU worker
processes would still execute host arithmetic. C4 must either implement and
verify that arithmetic on the assigned GPUs or explicitly leave in-sample
prediction outside its GPU claim. Physical worker placement alone is not
evidence of GPU computation. Preserve its startup NaNs in either case.

### C5. Multi-GPU cross-validation scheduling

Priority: bounded job-throughput feature; can follow C4 before the larger builds.

Add an explicit device scheduler for independent folds. Preserve fold definitions,
estimator cloning, seeds, scoring, weights and output order. Bind each worker's
GPU before runtime import. Define n_jobs/device oversubscription and failure
cleanup; preserve the existing single-job default and estimator parameters.
Do not share mutable estimator state between folds.

Acceptance: fold scores and relevant fitted-state witnesses match sequential
runs; no data leakage or seed drift; both GPUs execute folds; uneven folds and
one failing fold clean up workers correctly. Expose and test the supported
n_jobs/device contract. Label this parallel jobs, not distributed estimator fit
or beyond-one-GPU capacity.

Likely seams: `model_selection.py` and the persistent device worker pool.

### C6. General distributed matrix multiplication

Priority: workload-driven, after model/index baseline measurements.

Start with output-row tiles while preserving each output cell's complete K
reduction and numeric contract. Keep the right operand replicated initially;
state that capacity limit. Reuse existing identical GEMM kernels and proven
transport, without silently changing kernel/reduction behavior at tile shapes.
Do not partition K in the first implementation: that would introduce a new
cross-device sum contract.

Acceptance: exact results for transposes, odd dimensions, tails, subnormals and
supported dtypes/low-bit profiles; tile-order/offset negative controls and both
vendors. Compare against the existing one-GPU GEMM and CPU oracle. Enable only
an explicit API initially; avoid rerouting every estimator through it.

Proceed to two-dimensional operand/output distribution only for a demonstrated
capacity need. Report crossover sizes and transfer overhead; no speedup promise
before measurements.

## D. Execution order and release acceptance

1. **Evidence first:** A1/A2, A3 baseline audits, and B's shared physical gate.
   Keep 0.8.7 publication independent of the expansion roadmap.
2. **Bounded GPU features:** C4 forecasting and C5 fold scheduling establish the
   new gate on simpler partitions. No parallel CPU product implementation.
3. **Capacity priorities:** C1 loaded models and C2 IVF; C3 class-level GPC can
   be developed separately, with rentals still serialized under the budget.
4. **Deeper partitions:** C3 binary matrix state, C6 GEMM, then measured root
   memory bottlenecks in existing forest/clustering/kernel/mixture drivers.

Use separate worktrees and small independently reviewable commits. Integrate
completed changes with current main without overwriting concurrent work. A
planning commit or an implemented API does not close its qualification box.
For each package retain: supported configurations, out-of-scope cases, exact
wheel/source hashes, all gate receipts, negative controls, and any performance
or capacity measurements claimed. Include the public API and verifier lane in
the wheel, run the installed artifact, and only then mark it release-qualified.

Overall completion means the stated six scopes and CPU evidence backlog meet
these gates. It does not mean every possible parameter, topology, GPU model,
checkpoint architecture or input size has been verified.

## Execution update: 2026-09-18

- Kernel CPU implementation merged at bd0533295: six new polynomial/sigmoid/
  Laplacian KernelRidge/Nystroem routes, including public saved-model inference.
  31 runtime checks passed, 216 applicable CPU/Apple properties matched, and
  54 training sabotage cells moved. Independent NVIDIA/AMD and installed-wheel
  qualification remain owed; these six routes are still pending by design.
- Main now declares 234 harness routes: 161 default public CPU, 23 ordinary
  pending (17 prior + six new kernels), and 50 parallel routes. This is route
  inventory, not a count of distinct algorithms or complete proofs. The 18
  logical CPU parallel routes are not 18 physical GPU certifications.
- ARM64 clean release sweep passed. Classical UMAP expectations and sabotage
  certification failed. The byte-LM sabotage loader opt-in and sharded-neighbor
  numerical mismatch reporting are repaired and confirmed by targeted native
  controls on Apple M4. Full release certification must still be repeated with
  correct current GPU references and compatible qualification witnesses.
- Whole loaded-CausalLM proof, vendor admission for pending lanes, and the six
  GPU capabilities remain open. No new multi-GPU capability is claimed here.

Details and retained evidence: docs/lanes/LANE_STATUS_cpu-proof-followup.md.
