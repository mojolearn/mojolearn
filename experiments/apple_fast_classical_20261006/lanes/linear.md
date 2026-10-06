# Linear, decomposition and kernel candidates: AFCL-L01–L14

All fourteen controls have source implementations in production call paths. They
are **OFF by default**, guarded by Apple GPU plus `NUMERIC_FAST`, and carry
`NEVER RUN — PENDING MEASUREMENT` comments beside their switches. Status for
every entry is `source_written_uncompiled_unverified_unmeasured`. No compilation,
verification, test, measurement, hook or GPU job was run. There are no timing or
quality results, and no source is promoted by this delivery.

The machine-readable A/B configurations, callers, paths and restrictions are in
[linear.json](linear.json). Define arrays contain bare compiler define names.
Build both arms for the same Apple FAST target and retain any listed prerequisite
in both arms. Other experiment toggles should be absent during a single-card A/B.
The listed configuration is a future experiment specification, not an executed run.

| Card | A schedule | B schedule | Exact production scope |
| --- | --- | --- | --- |
| L01 | Existing 8192/2048-row Gram chunks | 4096/1024-row chunks | Shared linear/class-covariance grid Gram |
| L02 | One dependent accumulator per chunk fold | Four stripes and a pairwise finish | Means, Gram, X'Y and symmetric Gram final folds |
| L03 | One held-out row at a time per lane | Two independent predictions sharing path coefficients | LassoCV/ElasticNetCV held-out MSE |
| L04 | Default 512-row Huber partials | 256-row partials | Resident Huber loss/gradient |
| L05 | 2048-row nominal partials, limit 256 | 1024-row nominal partials, limit 512 | Existing QN `fast_xtdz` gradient product |
| L06 | 12 KiB moved-row feature page | 16 KiB page | Fused RBF/linear SMO gradient updates |
| L07 | One compensated chain per covariance cell | Bounded row partials and compensated finish | Opt-in compensated PCA covariance |
| L08 | 16-deep shared operand slab | 32-deep slab | Opt-in x_decomp tiled products |
| L09 | 256-row reflector-product partials | 128-row partials | Opt-in grid Householder QR |
| L10 | Generic one-cell OP_MUZ launch | Specialized four-stripe OP_MUZ launch | Frobenius multiplicative NMF updates |
| L11 | 256-query variance groups | 128-query groups | GPR/GPC predictive variance |
| L12 | One epilogue cell per thread | One component across four rows per thread | Separate RBFSampler cosine epilogue |
| L13 | 256-thread support selection per trial | 128-thread support selection per trial | MinCovDet/EllipticEnvelope C steps |
| L14 | 256-thread minibatch row blocks | 128-thread minibatch row blocks | Separate classical SGD row prediction/loss kernels |

## Reach and prerequisites

L01/L02 reach more callers than the initial linear idea names: unweighted
Ridge/RidgeClassifier/RidgeCV, Lars/LassoLars, BayesianRidge, ARDRegression,
LDA/QDA and other classical x_prep stages entering the shared grid Gram. Keep
the existing grid-Gram routes enabled in both arms. Weighted routes that bypass
that Gram remain outside the implementation. Workspace sizing follows the same
row partition used by the launches. The inherited tile-pair dispatch is unchanged;
there is no new dataset or exact-dimension rule.

L03 uses `MOJOLEARN_X_LINEAR_ENETCV_FAST=1` in both environments, preserving the
existing contiguous-fold eligibility and fallback. Alpha search, fold count,
coordinate-descent stopping and final refit remain intact. Only held-out prediction
work is grouped. L04 requires the existing Huber resident route and default
512-row baseline to stay enabled; it does not change line-search or iteration
budgets, tolerance, batching or witness replay policy.

L05 needs `MOJOLEARN_QN_FAST_COALESCED_OFF` in both arms: otherwise the default
coalesced product can bypass `fast_xtdz`. Keep `QN_FAST_XTDZ` enabled. Reach includes
eligible binary and multinomial logistic gradients, linear SVC/SVR and QN
squared/absolute regression gradients. Existing staging/register capacity gates
and distributed routing remain. L06 requires the existing fused SMO route; only
RBF/linear kernels within that route's feature-register capacity change. The new
shared page plus norms/deltas uses at most 24 KiB, calculated at the smallest
four-feature padded row, below Apple's 32 KiB shared-memory floor.

L07 requires `MOJOLEARN_PCA_FAST_COMPENSATED_COV` in both arms. The new schedule
retains centering before multiplication and compensation within every partial and
the final fold. Nominal chunks contain 2048 rows; the partial allocation is bounded
by 16 MiB or one covariance matrix, whichever is larger. This is an extension of
an existing unpromoted compensated candidate. It is not evidence against the
default matrix-unit covariance route, and other PCA solvers are outside its reach.

L08 requires `MOJOLEARN_DECOMP_FAST_GEMM_TILED` in both arms. The output tile,
transpose rules, precision and partial-K coverage stay fixed; operand shared memory
grows from 4 KiB to 8 KiB. It affects the classical x_decomp product entry, including
NMF, FactorAnalysis, PLS/CCA, ALS, randomized SVD and least-squares callers where
they select it. All further classical `DKit.mm`/`DevExec.gemm` consumers must be
mapped before admission. Ordered precision-sensitive products bypass it. No global
GEMM or neural product implementation was changed.

L09 requires `MOJOLEARN_QR_FAST_DEV` in both arms. Source reading narrowed the
initial TSQR idea to the supported grid Householder implementation in `fast_qr`.
It changes reflector-product row chunks, keeping scaled norms, reflector signs,
all columns and updates. No blocked-TSQR implementation was changed; a caller that
selects another QR route cannot establish coverage for this candidate.

L10 is intentionally scoped to `solver="mu", beta_loss="frobenius"`. OP_MUZ already
fuses multiply/divide and zero-denominator replacement, so the new mechanism is
four coalesced cell stripes per lane with an operation-specialized dispatch. It
calls the existing arithmetic cell, retaining broadcast indexing, regularization,
stopping checks and nonnegative update semantics. Coordinate-descent and KL/IS
update formulas bypass OP_MUZ. No newly eliminated intermediate is claimed.

L11 narrows whole-prediction batching to the existing variance query groups.
Training-axis sums, GPR clamp flags/std, GPC probability integration and query
coverage remain intact. Mean-only prediction, training, kernel formation and
triangular solves do not change. IDENTICAL segmented variance and sabotage routing
remain outside the new control.

L12 reuses one offset across four rows while retaining the projection, random
draws and exact epilogue arithmetic expressions. It reaches both the resident
RBFSampler pipeline and ordinary transforms that use a separate epilogue. Keep
`MOJOLEARN_RBF_FUSED` absent in both arms of the latter comparison; existing fused
projection routes bypass it. Nystroem and KernelRidge are outside this cosine
epilogue and are not claimed as implemented coverage. The production module's
`checks/` directory name does not imply that a check was executed.

L13 changes the integer support-selection schedule on the existing accepted
MCD batched route. Its eight radix passes and ascending-index tie selection remain;
the private histogram and scan page shrinks from 17.125 KiB to 8.625 KiB per trial.
This trades longer per-thread row walks for more concurrently resident trials.
Initial and iterative selection use the same new width. Keep accepted
`MCD_BATCH_MMA`/`MCD_BMMA` defaults enabled in both arms; no failed legacy covariance
arm is revived. Trial counts, random starts, C steps, covariance factors,
eigensolves and reweighting are unchanged. The older active-compaction and
covariance-reuse candidates remain independent and off for an isolated comparison.

L14's reachable classical SGD implementation is in `x_linear/device.mojo`, not
the initially suggested `glm` directory. It changes the separate minibatch
row prediction/loss launch to 128 threads; grid sizing and witness accounting
use one shared helper. Each row keeps the same predictor and the same gradient
and update sequence. Batch size, examples, seeded order, epochs, sample/class
weights, learning-rate progression and stopping observations are unchanged.
SGDClassifier/Regressor and any other classical SGD caller selecting this kernel
are affected. Per-sample SGD and the separate single-block chunked minibatch
route bypass it, as do sparse routes; those settings are not claimed as coverage.

## Future qualification, explicitly not performed

Each future arm needs the full affected estimator dataset and settings, dataset
version/hash, actual uncapped dimensions, source/binary/compiler/Apple hardware,
numeric mode and all toggles recorded before timing. Use the retained recipe
locations in the parent inventory and `experiments/performance_ideas/README.md`.
Unmapped transitive consumers and intrinsic lane caps are pending coverage.

The whole-operation boundary must include preparation, fit/training, required
synchronization and consumed outputs. Report fitting and inference, cold and
repeated use separately when applicable. Use one excluded warmup and one scored
sample per arm under repository policy, and preserve failures and pending cells.
No component timing admits a default. All sample counts here are zero.

Quality gates include the existing opponent/task floors, full-data objective or
held-out prediction metrics, convergence status and refusals. Focus additionally
on cancellation and large offsets in L01/L02/L07; selected regularization and
fold ties in L03; outliers and weights in L04; extreme logits and imbalance in
L05; margins/support vectors and SVR paired updates in L06; orthogonality, rank and
reconstruction in L07–L09; zero denominators and fixed-H transforms in L10;
calibration, clamp placement and query-batch invariance in L11; seeded feature-map
and downstream quality in L12. Neighboring shapes, a non-board workload and ragged
tails belong in future coverage. Bits may differ across versions; quality may not
be weakened.

L13 additionally needs robust fitted-state, exact tie-support and anomaly-ranking
coverage, including singular and inactive trials. L14 needs full-epoch task quality,
seeded shuffle/weight semantics and final partial-batch coverage. Neither source
schedule constitutes a quality result before the prohibited work is separately
authorized and performed.

After isolated qualification, assess L01+L02, L07 with relevant PCA products,
L08+L09, L08+L10 and the complete proposed configuration end to end. Changes to
association, occupancy and scratch can interact. Keep unsupported, failed, neutral
and rejected outcomes beside the switches; all candidates stay OFF until the
required evidence exists. No board was updated in this source-only delivery.

## Source handoff

Modified implementation files:

- `x_linear/fast_gram.mojo`
- `x_linear/enetcv_fast.mojo`
- `x_linear/huber_fast.mojo`
- `x_linear/device.mojo`
- `glm/impl/qn/fast_xtdz.mojo`
- `svm/impl/fast_update_f.mojo`
- `decomposition/impl/linalg/detail/pca.mojo`
- `x_decomp/fast_gemm.mojo`
- `x_decomp/fast_qr.mojo`
- `x_decomp/device.mojo`
- `x_decomp/mcd_fast.mojo`
- `gaussian_process/checks/kernels.mojo`
- `gaussian_process/gpc_device_var.mojo`
- `kernel_methods/checks/random_features.mojo`

New implementation file: `gaussian_process/afcl_prediction.mojo`.
New retained source records: this file and `linear.json`.

No known unsupported Mojo feature was introduced intentionally, and no toolchain
workaround was written. Source compatibility and behavior are expressly unproven
because the owner prohibited compiling and verifying. The parent owns the final
commit and push on `lane/apple-fast-classical-ideas-20261006`; this lane made no
commit. These source records are the retained evidence; there are no build/test/GPU
logs because those commands were not run.
