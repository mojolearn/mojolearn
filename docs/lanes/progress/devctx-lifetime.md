# devctx-lifetime: one process-lifetime DeviceContext for every binding

Lane `lane/devctx-lifetime`, cut from origin/main cf35966cd on 2026-09-28.
Background: docs/lanes/progress/merged.md, "M2 Pro Metal ROOT FIX". A binding
that makes a new `DeviceContext()` on every call exhausts Metal command queues
in one process on the M2 Pro; a later call then refuses ("Failed to create
Metal command queue") or returns output the GPU never wrote.

## Audit (at cf35966cd, every .mojo outside `*_main.mojo`)

398 files contain `DeviceContext()` (372 of them as `= DeviceContext()`).
Reach was computed from each `bindings/_mojolearn*.mojo` import closure.

| class | files | what |
|---|---|---|
| per-call binding entry (THE HAZARD), fixed here | 28 | 118 sites, listed below |
| already process-lifetime | 11 | core/neural_context.mojo (GP, SVM, KernelRidge, GMM, Cholesky, embedding, byte LM, mamba, training, transformer), the byte LM keeper, x_ann, x_cluster, x_cnn, x_decomp, x_linear, x_metrics, x_neighbors, x_prep, sequence/exec_device |
| one-shot main (a `def main`) | 335 | checks 114, training 27, extratrees 25, ensemble 24, transformer 19, gemm 19, neighbors 13, mamba 13, bench 13, umap 11, and 30 more families with 1 to 6 each |
| check or bench library no binding reaches | 19 | checks/{batch_invariance,boosting,oracle,vendor_correctness}_check, checks/{launch,mixed_hist}_probe, checks/level_bench, cluster/checks/{estimator,kmeans}_check, dbscan/checks/dbscan_check, decomposition/checks/svd_full_check, neighbors/checks/{ball_cover,ball_cover_knn,estimator,knn,radius,warpsort}_check, bench/knn_smallk_{dispatch,price}_fixture |
| other | 5 | core/forest_inference_model.mojo (no binding imports it; tools only), mamba/host/gen/*_prefill_backward.mojo x3 (a HOST shim struct named DeviceContext, no GPU), bindings/_mojolearn_training_host.mojo (the same shim) |

Fixed sites, by binding .so and slot (`Mojo<Slot>Context{Identical,Fast}`):

- Core: bindings/_mojolearn.mojo (11), neighbors/resident_index.mojo (1)
- Estimators: bindings/_mojolearn_estimators.mojo (16), kde/estimator.mojo (2), kde/resident_fit.mojo (1)
- Gbdt: bindings/_mojolearn_gbdt.mojo (5), gbdt/binary_prediction.mojo (1), gbdt/resident_model.mojo (1)
- Hdbscan: bindings/_mojolearn_hdbscan.mojo (4)
- Ivf: bindings/_mojolearn_ivf.mojo (4)
- Linalg: bindings/_mojolearn_linalg.mojo (7), decomposition/linalg_public_device.mojo (3; x_decomp calls only the ctx-taking forms)
- Metrics: bindings/_mojolearn_metrics.mojo (3), metrics/estimator.mojo (20), spectral/estimator.mojo (5)
- Rf: bindings/_mojolearn_rf.mojo (3)
- Solver: bindings/_mojolearn_solver.mojo (3)
- Trees: bindings/_mojolearn_trees.mojo (3), extratrees/estimator.mojo (2)
- Tsa: holtwinters/estimator.mojo (2), tsa/estimator.mojo (2)
- Arima: arima/estimator.mojo (2)
- Preprocessing: preprocessing/estimator.mojo (6)
- Resample: resample/estimator.mojo (6)
- Svm (the existing slot): isolation_forest/estimator.mojo (1)
- NeuralMamba (the existing slot): mamba/impl/modeling/modeling_mamba_prefill_backward.mojo, mamba/impl/modules/mamba{2,3}_prefill_backward.mojo (1 each)

NOT in this class and left as they are: 27 multi-GPU sites that make
`DeviceContext(device_id=rank)` per call (cluster, core gram/householder,
cholesky, solver, svm kernel matrices, dbscan, mixture, isolation forest, glm,
gbdt shards, resample owners, training pools). On a Mac they run one rank, so
each par-* call still makes one context. Owed: a per-rank slot in the same
accessor if a par-* lane ever shows the queue class.

## Fix

ONE accessor, `process_ctx[name]()` in core/neural_context.mojo (the slot the
neural and GP lanes already used; `neural_ctx` stays as the same function).
Each fixed file imports it and names its binding's slot with the numeric tier:
`comptime _DEVCTX_SLOT = "Mojo<Slot>ContextIdentical" if _DEVCTX_MODE ==
_DEVCTX_IDENTICAL else "Mojo<Slot>ContextFast"`. Library modules take the
name of the one GPU binding that reaches them, so each .so holds one context.

- No arithmetic changes: same kernels, same launches, same order on one
  stream.
- The context no longer dies at the end of a call, so nothing may lean on
  that. A binding entry whose block ended without a synchronize now calls
  `ctx.synchronize()` at the end of that block (37 sites); every other site
  already synchronizes or returns host values it downloaded.
- CPU-only: the accessor runs only when a device path calls it; host bindings
  never reach it and nothing runs at import, so no context is made there.

## Lane check: par-* CPU arm

tools/algos_lane_check.py now reads a `par-*` lane whose CPU arm refuses
EVERY fixture with the declared sentence "no CPU implementation of the
cooperative multi-GPU driver" (python/mojolearn/_parallel_pool.py
`_cpu_refusal`) as KNOWN REFUSAL, provided the GPU arm exited 0 with a real
train hash on every fixture. It compares nothing, says so in the RESULT line,
and is never a DISAGREE under `--sabotage`. Any other refusal, a mix of
refused and hashed CPU cells, or a GPU refusal still fails.

## Evidence

(filled in as the runs land)
