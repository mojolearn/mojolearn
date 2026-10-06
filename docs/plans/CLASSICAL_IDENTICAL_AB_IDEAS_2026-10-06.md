# Classical ML IDENTICAL A/B ideas — 2026-10-06

Planning baseline: `35b2d4976`. Branch: `ideas/classical-identical-20261006`.

Sixty candidate cards, including separately attributable sub-arms, were written before implementation fan-out. This is a design inventory, not a performance ranking or an identity claim. New source remains disabled and uncompiled/unverified by the owner's request.

## Contract and scope

- **Scope:** Classical ML only, including classical preprocessing, metrics, model selection, matrix factorizations, graph/manifold estimators and classical forecasting. Excludes neural models, neural optimizers, transformers, Mamba, CNNs, language models and Apple FAST tuning.
- **Identity:** Within EACH arm/version, NVIDIA, AMD, Apple and the host must produce bitwise-identical promised outputs/state. A and B may have different bits. Matching B's old bits is not a quality gate for a versioned profile. Each arm must independently satisfy task quality; compare every admitted model attribute, convergence/selection decision, error and seed contract.
- **Default:** All new candidates default OFF; no promotion, merge to main, build, test, static verification, benchmark, remote job or board-result mutation in this request.
- **Performance:** A future default decision requires full-dataset end-to-end A/B on both NVIDIA and AMD for every affected estimator and relevant interaction, combined faster with neither materially slower; Apple/host establish identity and Apple times do not vote. One excluded warmup and one scored sample, serial cells per GPU, separate vendor boxes. Reuse accepted matching build/identity evidence.
- **Workload:** Resolve dataset version/hash/split, actual dimensions, settings, numeric profile, flags and timed boundaries before any future timing. Audit intrinsic lane caps; --rows full is not proof. Missing mappings remain pending. Fit/preparation/synchronization/consumed outputs are included; cold, repeated use and inference are separate.
- **Routing:** No exact benchmark dimensions, names, seeds or adjacent-to-board thresholds. Explain every route using hardware/work/bytes and cover neighboring shapes and one non-board dataset. For removing a targeted incumbent rule, A is the general rule and B explicitly retains the old rule.
- **Implementation:** Mojo performs all runtime data processing. Python here is metadata/orchestration only. Use supported Mojo facilities; record upstream asks and park unsupported functionality. No compiler-output edits, unsupported build modes or speculative backend shims.
- **Resources:** Any future CPU race gets all allocated cores (actual Linux cgroup allocation; Apple unrestricted), arms serially, preserving algorithm semantics and documented nested pools. Record worker effective pools.
- **Evidence:** Retain winners, losers, neutral/inconclusive cells, refusals and pending cells with source/binary/harness/hardware provenance. Use board tools only when real measurements exist. No opponent ratio from own-only A/B. A source draft is not a compile pass, identity proof, quality pass, speed win or complete full-workload harness.

## Full-workload recipe map

Each card names its affected estimator family. Resolve each estimator to these saved recipes; exact dataset hashes/dimensions and cap audits remain pending because no campaign is authorized. Shared primitives require transitive caller coverage. A component driver never substitutes for this map.

- **classical:** `tools/classical_two_datasets.py: LANES, BLOCK_OF, preparation and lane row constants`
- **more:** `tools/bench_board_more.py: LANES, LANE_CONFIG, preparation and workload constants`
- **expanded:** `tools/bench_board_algos.py: LANES, _add calls, sub/block definitions, settings and timed spans (classical lanes only)`
- **trees:** `tools/bench_board.py: TREE_LANES, TREE_TASK_DATASETS, plan_races; bench/speed/forest_speed_arm.py`
- **neighbors:** `tools/knn_datasets.py: index/query blocks and preparation`

## Priority and arm convention

P0: prioritize traffic/launch reductions with an existing arithmetic contract. P1: larger caller integration or an explicit new profile. P2: expensive/irregular work requiring careful design. These are source-based hypotheses, not measured bottleneck claims.

A is the candidate and B the frozen incumbent. Hold other switches at the incumbent configuration, not an all-off configuration. Expose sub-arms individually. Compare A's NVIDIA/AMD/Apple/host words with A, and B's columns with B; never reject a valid version revision just because A differs from B. Quality still compares A with the incumbent and applicable accepted references.

## Inventory

| ID | Priority | Lane | Type | Experiment |
| --- | --- | --- | --- | --- |
| [C01](#c01) | P1 | shared | profile | Versioned fixed-leaf reduction profiles |
| [C02](#c02) | P0 | shared | schedule | Fuse independent column statistics with shared input loads |
| [C03](#c03) | P0 | shared | schedule | Fuse finite checks and extrema production |
| [C04](#c04) | P1 | shared | schedule | Apply centering and scaling at the consumer load |
| [C05](#c05) | P0 | shared | schedule | Retain bounded classical scratch across phases |
| [C06](#c06) | P0 | shared | schedule | Batch row norms without changing feature folds |
| [C07](#c07) | P1 | shared | schedule | Stable radix layout for classical preparation |
| [C08](#c08) | P0 | shared | schedule | Reuse category dictionaries and inverse maps |
| [C09](#c09) | P0 | shared | schedule | Fused regression and weighted metric reports |
| [C10](#c10) | P1 | shared | schedule | Reuse stable score sorting for ranking metrics |
| [C11](#c11) | P1 | shared | schedule | Generate resampling indices at their consumer |
| [C12](#c12) | P1 | shared | schedule | Sparse exact classification counts before dense output |
| [C13](#c13) | P1 | linear | profile | Shared sufficient statistics across CV folds |
| [C14](#c14) | P0 | linear | schedule | Reuse one factorization for multiple right-hand sides |
| [C15](#c15) | P1 | linear | profile | Solve against factors instead of forming inverses |
| [C16](#c16) | P0 | linear | schedule | Fuse GLM response, residual and objective production |
| [C17](#c17) | P0 | linear | schedule | Batch independent OVR and line-search tasks |
| [C18](#c18) | P1 | linear | schedule | Coordinate-descent residual tile reuse |
| [C19](#c19) | P0 | linear | schedule | Persistent ordered online linear updates |
| [C20](#c20) | P1 | linear | schedule | Bounded SVM kernel-row cache and shared pair loads |
| [C21](#c21) | P0 | linear | schedule | Fuse canonical SMO extrema selection |
| [C22](#c22) | P1 | linear | profile | Produce symmetric Gram/kernel triangles once |
| [C23](#c23) | P1 | linear | profile | Stream centered covariance for PCA and discriminants |
| [C24](#c24) | P2 | linear | profile | Versioned TSQR merge tree and panel sizes |
| [C25](#c25) | P1 | linear | schedule | Reuse randomized projection and decomposition panels |
| [C26](#c26) | P1 | linear | schedule | Reuse NMF sufficient products within each accepted step |
| [C27](#c27) | P2 | linear | schedule | Batch independent ICA and factor-analysis component work |
| [C28](#c28) | P1 | linear | schedule | Bucket independent sparse-coding and ALS solves |
| [C29](#c29) | P0 | graph | schedule | Streaming exact distance-to-top-k |
| [C30](#c30) | P1 | graph | profile | Versioned direct-difference distance arithmetic |
| [C31](#c31) | P0 | graph | schedule | Bucket ragged CSR canonicalization by work |
| [C32](#c32) | P0 | graph | schedule | Fuse exact radius threshold and compact emission |
| [C33](#c33) | P0 | graph | schedule | Freeze first graph convergence state on device |
| [C34](#c34) | P1 | graph | schedule | Canonical parallel MST edge selection |
| [C35](#c35) | P0 | graph | schedule | IVF long-list chunking and short-list packing |
| [C36](#c36) | P0 | graph | schedule | Reuse KMeans centroid tiles through assignment |
| [C37](#c37) | P1 | graph | profile | Versioned fixed-row centroid accumulation |
| [C38](#c38) | P0 | graph | schedule | KMeans++ nearest-distance reuse |
| [C39](#c39) | P1 | graph | schedule | Retain mini-batch and bisecting clustering state |
| [C40](#c40) | P1 | graph | schedule | Share MeanShift distance tiles across seed updates |
| [C41](#c41) | P2 | graph | schedule | OPTICS fused canonical reachability minima |
| [C42](#c42) | P1 | graph | schedule | Compact active agglomerative distance work |
| [C43](#c43) | P1 | graph | schedule | Resident normalized graph operators |
| [C44](#c44) | P2 | graph | schedule | Seeded UMAP graph and sampling-state reuse |
| [C45](#c45) | P0 | trees_stats | schedule | Fuse tree split statistics and candidate scoring |
| [C46](#c46) | P0 | trees_stats | schedule | Exact histogram sibling subtraction |
| [C47](#c47) | P0 | trees_stats | schedule | Tree frontier tasks by live work and histogram bytes |
| [C48](#c48) | P0 | trees_stats | schedule | Stable tree partition fused with child bookkeeping |
| [C49](#c49) | P1 | trees_stats | schedule | Batch independent forest construction with logical RNG IDs |
| [C50](#c50) | P0 | trees_stats | schedule | Packed forest inference and fixed tree-output reduction |
| [C51](#c51) | P1 | trees_stats | schedule | TreeSHAP shared path metadata and bounded batching |
| [C52](#c52) | P1 | trees_stats | profile | Versioned stable pair-combine KDE reduction |
| [C53](#c53) | P0 | trees_stats | schedule | GMM reusable centered tiles and fused sufficient statistics |
| [C54](#c54) | P1 | trees_stats | schedule | GP factor and batched prediction reuse |
| [C55](#c55) | P0 | trees_stats | schedule | Naive Bayes class-statistic pass fusion |
| [C56](#c56) | P1 | trees_stats | schedule | Discriminant class solves and prediction fusion |
| [C57](#c57) | P2 | trees_stats | schedule | Robust covariance candidate state reuse |
| [C58](#c58) | P1 | trees_stats | schedule | Batch independent classical forecast candidates |
| [C59](#c59) | P1 | trees_stats | schedule | ARIMA device-resident order and likelihood trials |
| [C60](#c60) | P1 | trees_stats | schedule | Reuse classical stationarity preparation and lag work |

<a id="c01"></a>
## C01 — Versioned fixed-leaf reduction profiles

**Affected estimators:** Classical column statistics; GLM objectives; covariance; regression metrics.

**Source anchors:** `core/pinned_reduce.mojo`, `core/column_stats.mojo`, `core/device_fold.mojo`.

**A:** A new classical-only logical leaf size and adjacent-pair merge, independent of physical block and warp size. Separate leaf sizes into separate arms.

**B:** The incumbent caller's specified serial/halving reduction.

**Identity/version contract:** New version: fix leaf membership, per-leaf addition/FMA order, FTZ, odd-tail carry and signed zero; use one scalar helper in host and device. GPU geometry may schedule but never redefine leaves.

**Quality gates:** Objective, residual and downstream task metrics must not degrade; inspect cancellation, overflow, subnormals, constant and ragged columns.

**Reject or defer when:** Reject register spills, larger scratch, poor error growth or any incomplete host/caller migration; no global neural-mode change.

**Prior work:** I04 proposes a GEMM fold revision; this is a classical scalar-statistic profile, separately gated.

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c02"></a>
## C02 — Fuse independent column statistics with shared input loads

**Affected estimators:** StandardScaler; GaussianNB; PCA preparation; GLM centering.

**Source anchors:** `core/column_stats.mojo`, `preprocessing/standard.mojo`, `glm/impl/center_device.mojo`.

**A:** Calculate independent sums/counts or centered second moments in a single tiled read, maintaining separate accumulators and the incumbent order for each output.

**B:** Separate passes over identical input for each statistic.

**Identity/version contract:** Keep two-pass centered variance where required; merging passes is not permission to replace it with E[x²]-E[x]².

**Quality gates:** Every fitted mean, variance, scale, count and downstream output; zero-variance, weighted and missing-data semantics.

**Reject or defer when:** Reject increased spills or occupancy loss, duplicated reads on tails, and any changed missing-value denominator.

**Prior work:** I24 leaves broader preprocessing pass fusion open.

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c03"></a>
## C03 — Fuse finite checks and extrema production

**Affected estimators:** MinMaxScaler; MaxAbsScaler; classical estimator input preparation.

**Source anchors:** `core/input_device.mojo`, `preprocessing/minmax.mojo`, `x_prep/prims.mojo`.

**A:** Produce validity status and required extrema from one input walk, with canonical first-error indexing.

**B:** Separate validity and extrema scans.

**Identity/version contract:** Exact status/min/max selections use an explicit total order and original public NaN/Inf refusal rules; publication still waits for validation.

**Quality gates:** Same refusal type/index, fitted extrema, signed zeros, empty-input and all-missing behavior.

**Reject or defer when:** Reject early mutation before a deferred error or any policy that silently drops nonfinite rows.

**Prior work:** Extension of I24.

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c04"></a>
## C04 — Apply centering and scaling at the consumer load

**Affected estimators:** PCA; OLS; Ridge; GaussianNB; normalized classical distances.

**Source anchors:** `glm/impl/center_device.mojo`, `core/row_norms.mojo`, `preprocessing/standard.mojo`.

**A:** Pass immutable mean/scale buffers to a Mojo consumer and reproduce subtract/divide operations at load, avoiding an intermediate matrix.

**B:** Materialize centered/scaled input and then read it in the consumer.

**Identity/version contract:** Keep the stored-intermediate rounding and FTZ seams explicit even when no matrix is stored; do not contract subtraction and multiplication into an FMA.

**Quality gates:** Fitted state, predictions, residuals and input ownership; readonly, strided and repeated callers.

**Reject or defer when:** Reject recomputation when multiple consumers make materialization cheaper; report cold and repeated use separately.

**Prior work:** New extension

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c05"></a>
## C05 — Retain bounded classical scratch across phases

**Affected estimators:** Iterative classical estimators; repeated predict/transform.

**Source anchors:** `core/scratch_pool.mojo`, `core/device_pool.mojo`, `core/device_liveness.mojo`.

**A:** Reuse explicitly owned scratch sized from live bytes for sequential stages; separate retained fit scratch from optional repeated-call caches.

**B:** Allocate and release each temporary at each phase/call.

**Identity/version contract:** Scheduling only: initialize every consumed cell and preserve device/context ownership, asynchronous lifetime, error release and concurrent estimator isolation.

**Quality gates:** Same outputs, failures and repeated-fit state; cover growing/shrinking shapes and context changes.

**Reject or defer when:** Reject unbounded retention, stale-data dependence or a cold-use regression hidden by hot timings.

**Prior work:** I02 covers GEMM scratch; extend only to classical caller-owned buffers.

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c06"></a>
## C06 — Batch row norms without changing feature folds

**Affected estimators:** KMeans; KNN; KDE; kernel matrices; Normalizer.

**Source anchors:** `core/row_norms.mojo`, `core/cosine_rows.mojo`, `core/expand_distances.mojo`.

**A:** Assign multiple independent rows to one block or tile, reusing dispatch and coalesced loads with each row's logical lanes unchanged.

**B:** One incumbent row task per block.

**Identity/version contract:** Feature order, squared terms, pinned fold and portable square root remain fixed; schedule choices follow bytes and available parallelism, not benchmark dimensions.

**Quality gates:** Norm and final distance words, underflow, overflow, signed zero, tails and downstream task quality.

**Reject or defer when:** Reject reduced parallelism, shared-memory expansion or accidental use of physical warp width.

**Prior work:** New extension

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c07"></a>
## C07 — Stable radix layout for classical preparation

**Affected estimators:** QuantileTransformer; KBinsDiscretizer; categorical encoders; isotonic regression.

**Source anchors:** `core/stable_radix_sort.mojo`, `core/stable_radix_digits.mojo`, `core/segmented_sort.mojo`, `x_prep/dsort.mojo`.

**A:** Independently vary digit width, keys per task, and ragged-segment bucket scheduling under a scratch/shared-memory cost model.

**B:** Incumbent stable radix schedule with identical key encoding.

**Identity/version contract:** Stable secondary row index, NaN policy, signed-zero ordering and per-segment boundaries are preserved; no approximate quantiles.

**Quality gates:** Exact permutations, category IDs, quantile cutpoints and public transforms; duplicates and highly skewed segments.

**Reject or defer when:** Reject descriptors or histogram clears costing more than passes saved; do not rerun a previously decided identical configuration.

**Prior work:** Reuse I19 infrastructure and distinguish new geometries from existing digit-width arms.

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c08"></a>
## C08 — Reuse category dictionaries and inverse maps

**Affected estimators:** OneHotEncoder; OrdinalEncoder; TargetEncoder; LabelEncoder; Naive Bayes preparation.

**Source anchors:** `core/label_encode_device.mojo`, `core/label_rows_device.mojo`, `x_prep/cat_cls2.mojo`.

**A:** Build a canonical dictionary and inverse map once per fit, then share them among counts and encodings; evaluate independent grouped output emission.

**B:** Repeated dictionary/sort/inverse work for each consumer.

**Identity/version contract:** No dictionary reordering; unknown/missing-category behavior and supervised fold isolation remain unchanged.

**Quality gates:** Vocabulary, inverse map, output shape/layout and target-encoding leakage guards; unseen labels and repeated fits.

**Reject or defer when:** Reject cache keys based only on pointer address or shape; no reuse across mutated data without an explicit immutable owner.

**Prior work:** I19 supplies dictionary primitives; this card concerns caller lifetime and reuse.

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c09"></a>
## C09 — Fused regression and weighted metric reports

**Affected estimators:** MSE; MAE; R²; regression scoring; sample-weighted metrics.

**Source anchors:** `x_metrics/reg_epi.mojo`, `x_metrics/group.mojo`, `metrics/estimator.mojo`.

**A:** Share prediction/target loads while maintaining one pinned accumulator stream per metric; finish a requested metric bundle with one status/readback boundary.

**B:** Separate public metric calls or separate native passes.

**Identity/version contract:** No reassociation between metrics; denominator, multioutput policy, force-finite and zero-weight rules are part of the contract.

**Quality gates:** All requested values and warning/error states; degenerate targets and large weight range.

**Reject or defer when:** Reject extra unrequested work or an API that times a smaller output set than the control.

**Prior work:** I24 combines classification reports; extend to regression.

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c10"></a>
## C10 — Reuse stable score sorting for ranking metrics

**Affected estimators:** ROC AUC; average precision; precision-recall and ROC curves.

**Source anchors:** `x_metrics/ranking.mojo`, `core/segmented_sort.mojo`.

**A:** Compute a single stable score order and tie boundaries, then emit all requested rank metrics from shared canonical prefix counts.

**B:** Each metric sorts and scans independently.

**Identity/version contract:** Tie blocks, positive-label encoding, weights and threshold endpoint conventions remain exact; weighted prefixes keep a fixed arithmetic tree.

**Quality gates:** Curves and scalar values for all ties, missing classes, weights and multiclass averaging.

**Reject or defer when:** Reject retained sorted state without input ownership or assumptions that tied scores can be permuted arbitrarily for weighted sums.

**Prior work:** New extension

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c11"></a>
## C11 — Generate resampling indices at their consumer

**Affected estimators:** Bootstrap; permutation tests; forest bootstrap; cross-validation row gathers.

**Source anchors:** `resample/estimator.mojo`, `resample/device_post.mojo`, `core/shuffle_iterator.mojo`, `core/philox.mojo`.

**A:** Regenerate deterministic counter-indexed draws directly in gather/statistic tasks or retain one compact permutation for multiple consumers.

**B:** Materialize full index arrays and reread them for each operation.

**Identity/version contract:** Seed, draw number, replicate, rejection sampling and ordering must produce the same sequence; stateful RNG cannot be replaced without a separately versioned contract.

**Quality gates:** Index sequence, multiplicities, confidence intervals and hypothesis-test statistic/p-value; no data-dependent RNG draw consumption.

**Reject or defer when:** Reject repeated RNG cost exceeding memory savings or altered dependence between replicates.

**Prior work:** New extension

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c12"></a>
## C12 — Sparse exact classification counts before dense output

**Affected estimators:** Confusion matrix; precision/recall/F-score; class-count preparation.

**Source anchors:** `x_metrics/cm_epi.mojo`, `core/label_encode_device.mojo`, `naive_bayes/nb.mojo`.

**A:** Accumulate exact integer occupied label-pair counts in canonical sorted segments, then materialize the requested dense or sparse report once.

**B:** Clear and update a full class-by-class scratch table for each report.

**Identity/version contract:** Unweighted integer counts only in the first arm; weighted FP counts need a separate fixed-fold profile. Public class ordering and empty cells are unchanged.

**Quality gates:** All confusion and averaging outputs, absent classes, high class counts, integer overflow/refusal and warning cells.

**Reject or defer when:** Reject sort overhead for dense occupancy; route only from documented storage/work estimates and preserve public output allocation cost.

**Prior work:** Build on I24 exact report APIs; this is occupied-key accumulation.

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c13"></a>
## C13 — Shared sufficient statistics across CV folds

**Affected estimators:** RidgeCV; LassoCV; ElasticNetCV; LogisticRegressionCV preparation.

**Source anchors:** `x_linear/cd.mojo`, `x_linear/cd_grid.mojo`, `x_linear/device_ops.mojo`.

**A:** Compute disjoint fold statistics once and combine the retained folds in a specified order for each training set; keep solver selection and fold IDs.

**B:** Rescan every training fold independently.

**Identity/version contract:** New version for all columns: per-fold means and centered Gram construction must have a complete host/device fold contract. Never assume total-minus-heldout has old bits.

**Quality gates:** Validation losses, selected alpha/C/l1 ratio, coefficients, convergence and ill-conditioned residuals; include leakage checks.

**Reject or defer when:** Reject cancellation, workspace inflation or fewer optimization iterations masquerading as a scheduling gain.

**Prior work:** Explicit remaining I12 Gram/shared-fold proposal.

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c14"></a>
## C14 — Reuse one factorization for multiple right-hand sides

**Affected estimators:** OLS; Ridge; multioutput regression; Gaussian processes; repeated linear solves.

**Source anchors:** `core/householder_qr.mojo`, `glm/impl/ridge_multi.mojo`, `cholesky`.

**A:** Factor once and apply the same canonical factor to grouped target/RHS columns with bounded workspace.

**B:** Refactor or repeatedly reload identical factors for each RHS.

**Identity/version contract:** Same factor, pivot/rank decisions and solve folds; caches belong to immutable fitted state and include regularization/preprocessing settings.

**Quality gates:** Residuals, coefficients, rank, predictions and stale-factor invalidation; singular and nearly singular cases.

**Reject or defer when:** Reject memory growth or latency losses for a single RHS.

**Prior work:** Reuse I22 factor-state work; extend actual multioutput callers without claiming new initial implementation.

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c15"></a>
## C15 — Solve against factors instead of forming inverses

**Affected estimators:** BayesianRidge; ARD; Ridge; covariance consumers with solve-only needs.

**Source anchors:** `x_linear`, `glm/impl/ridge.mojo`, `cholesky`, `x_decomp`.

**A:** Replace inverse-then-product with canonical triangular solves where the public operation does not need a dense inverse; retain separately requested uncertainty outputs.

**B:** Materialize and multiply by an explicit inverse.

**Identity/version contract:** New numerical profile across host and all devices, fixed solve ordering and pivot policy; do not replace required inverse-derived diagnostics with approximations.

**Quality gates:** Prediction, posterior variance/evidence, condition sensitivity and residuals at least as good as control.

**Reject or defer when:** Reject loss of required fitted attributes or worse uncertainty calibration.

**Prior work:** New extension

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c16"></a>
## C16 — Fuse GLM response, residual and objective production

**Affected estimators:** LogisticRegression; PoissonRegressor; GammaRegressor; TweedieRegressor; LinearSVC/SVR.

**Source anchors:** `glm/impl/qn/glm_base.mojo`, `glm/impl/qn/glm_logistic.mojo`, `glm/impl/qn/glm_softmax.mojo`, `x_linear`.

**A:** Reuse each linear response and portable nonlinear evaluation to produce residuals and loss leaves in one pass; preserve separate canonical objective fold.

**B:** Separate response, nonlinear link, residual and objective kernels.

**Identity/version contract:** Same stable link branches, FMA boundaries, regularization and trial-vector association; line search sees exactly the intended candidate state.

**Quality gates:** Objective/gradient, first accepted step, convergence iterations, coefficients, probabilities and deviance.

**Reject or defer when:** Reject changed branch behavior at extreme logits or stale objective/gradient reuse.

**Prior work:** Extends I12 beyond accepted-iterate scheduling.

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c17"></a>
## C17 — Batch independent OVR and line-search tasks

**Affected estimators:** Multiclass linear models; independent optimizer trials.

**Source anchors:** `glm/impl/qn/qn_linesearch.mojo`, `x_linear/sgd.mojo`, `x_linear/device_ops.mojo`.

**A:** Schedule independent classes or explicitly materialized trial vectors together; independently attribute OVR and bounded speculative line search.

**B:** Sequential dispatch of the same classes/trials.

**Identity/version contract:** Preserve first accepted trial, logical evaluation counts, class order and each model's iteration dependencies.

**Quality gates:** All fitted states, probabilities, accepted steps, refusals and convergence; count physical speculative work separately.

**Reject or defer when:** Reject extra evaluations or expanded live state erasing the launch savings.

**Prior work:** Reuse I12 existing OVR and bounded4trial code; new work is full-caller selection/combination mapping.

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c18"></a>
## C18 — Coordinate-descent residual tile reuse

**Affected estimators:** Lasso; ElasticNet; linear sparse coding.

**Source anchors:** `solver`, `x_linear/cd.mojo`, `x_linear/cd_grid.mojo`.

**A:** Keep one bounded feature/residual tile resident through adjacent ordered coordinate updates, or fuse residual write with its next ordered consumer.

**B:** Separate residual update and next-coordinate scans.

**Identity/version contract:** Coordinate updates remain sequential in the declared order; simultaneous Jacobi updates are a different algorithm and excluded.

**Quality gates:** Duality gaps, stopping iteration, sparsity pattern, coefficients and predictions; correlated columns.

**Reject or defer when:** Reject changed residual rounding or silently weakened convergence tolerance.

**Prior work:** I12 independent-coordinate infrastructure is reusable; this card targets memory passes.

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c19"></a>
## C19 — Persistent ordered online linear updates

**Affected estimators:** SGDClassifier/Regressor; Perceptron; PassiveAggressive; SGDOneClassSVM.

**Source anchors:** `x_linear/sgd.mojo`, `x_linear/device_ops.mojo`, `core/shuffle_iterator.mojo`.

**A:** Run bounded consecutive samples in one device task with resident coefficients; reuse existing step-fusion where available and extend equivalent online solvers.

**B:** One launch/dispatch boundary per logical update.

**Identity/version contract:** No mini-batch averaging: retain exact sample permutation, learning-rate time index, clipping, penalties and sequential updates.

**Quality gates:** Coefficients, intercept, epochs, loss, averaging state and seeded repeatability.

**Reject or defer when:** Reject device watchdog pressure, lost parallelism or changed sample semantics.

**Prior work:** I12 supplies SGD fusion; do not duplicate its decided controls.

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c20"></a>
## C20 — Bounded SVM kernel-row cache and shared pair loads

**Affected estimators:** SVC; SVR; OneClassSVM.

**Source anchors:** `svm/impl/kernelcache.mojo`, `svm/impl/distance/kernel_matrices.mojo`, `svm/impl/smosolver.mojo`.

**A:** Reuse canonical kernel rows across working-set updates with deterministic eviction, and fuse independent loads of selected row pairs.

**B:** Recompute/reload kernel rows under incumbent cache organization.

**Identity/version contract:** Cache only computed values; working-set selection, row precision, gamma, exp, error updates and stopping remain identical.

**Quality gates:** Support vectors, dual coefficients, decision values, probabilities and iteration count; repeated rows and near ties.

**Reject or defer when:** Reject cache memory dominating fit or a hardware-specific eviction policy affecting arithmetic rather than storage.

**Prior work:** New extension

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c21"></a>
## C21 — Fuse canonical SMO extrema selection

**Affected estimators:** SVC; SVR; OneClassSVM.

**Source anchors:** `svm/impl/workingset.mojo`, `svm/impl/ws_util.mojo`, `svm/impl/grid_fold.mojo`.

**A:** Compute multiple independent KKT extrema from one scan and reduce explicit value/index tuples together.

**B:** Separate scans/reductions for each working-set statistic.

**Identity/version contract:** Exact total order with stable index ties, NaN refusals and eligibility masks; no heuristic working-set change.

**Quality gates:** Selected indices, KKT violations, full SMO trajectory and final model.

**Reject or defer when:** Reject altered ties or register pressure exceeding memory savings.

**Prior work:** New extension

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c22"></a>
## C22 — Produce symmetric Gram/kernel triangles once

**Affected estimators:** KernelRidge; KernelPCA; SVC training; covariance and classical Gram users.

**Source anchors:** `kernel_methods/impl/distance/kernel_matrices.mojo`, `core/gram_splitk.mojo`, `svm/impl/distance/kernel_matrices.mojo`.

**A:** Evaluate one canonical orientation of each symmetric pair, then mirror once or let a triangular consumer read it.

**B:** Compute both matrix halves independently.

**Identity/version contract:** If current orientations differ in floating evaluation order, establish a versioned canonical orientation across all columns; keep diagonal and signed-zero semantics explicit.

**Quality gates:** Full matrix and downstream model state, symmetry, PSD/residual diagnostics and quality.

**Reject or defer when:** Reject triangular indexing overhead or assuming all configured kernels are symmetric.

**Prior work:** New extension

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c23"></a>
## C23 — Stream centered covariance for PCA and discriminants

**Affected estimators:** PCA; IncrementalPCA; covariance estimators; LDA/QDA.

**Source anchors:** `decomposition/impl`, `core/column_stats.mojo`, `naive_bayes/da.mojo`, `x_decomp`.

**A:** Accumulate explicitly centered fixed row panels and merge their covariance blocks in one versioned tree, avoiding a centered full matrix.

**B:** Materialize centered data then call the incumbent covariance contraction.

**Identity/version contract:** One panel partition and merge order on host/NVIDIA/AMD/Apple; preserve centering before products rather than unstable raw-moment subtraction.

**Quality gates:** Explained variance, orthogonality, reconstruction, rank, discriminant accuracy and conditioning.

**Reject or defer when:** Reject altered statistical normalization, inaccurate tail handling or quality loss.

**Prior work:** Related to I22 blocked linalg, but a separate covariance numerical profile.

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c24"></a>
## C24 — Versioned TSQR merge tree and panel sizes

**Affected estimators:** OLS; PCA/SVD; least squares; QR consumers.

**Source anchors:** `core/householder_qr.mojo`, `x_decomp/qr_sliced.mojo`, `glm/impl/linalg/detail/lstsq.mojo`.

**A:** Compare binary and fixed higher-arity logical QR merge trees and panel widths as independent arithmetic profiles, with immutable column sign conventions.

**B:** Current TSQR logical tree and panel profile.

**Identity/version contract:** All columns must implement the identical reflector formation/application, pivot/rank and sign rules; vendor scheduling may differ only below this tree.

**Quality gates:** R/Q words across vendors, reconstruction/backward error, orthogonality, rank and downstream prediction quality.

**Reject or defer when:** Reject higher workspace, increased error or reliance on unsupported compiler features; no revival of quarantined block-Jacobi code.

**Prior work:** Explicit unimplemented profile extension of I22.

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c25"></a>
## C25 — Reuse randomized projection and decomposition panels

**Affected estimators:** TruncatedSVD; randomized SVD; random projection; Nystroem/RBFSampler.

**Source anchors:** `x_decomp`, `decomposition/impl`, `kernel_methods/impl/random/rng_device.mojo`.

**A:** Reuse immutable projection/basis tiles across outputs and fuse adjacent pointwise transforms without changing iteration count or rank.

**B:** Reload or rematerialize the same projection/basis intermediates.

**Identity/version contract:** RNG coordinate mapping, oversampling, power iterations, sign canonicalization and reduction order remain fixed.

**Quality gates:** Transform words, reconstruction/subspace quality, downstream scores and seeded model state.

**Reject or defer when:** Reject dropping power iterations or reducing embedding dimension to obtain the speed gain.

**Prior work:** New extension

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c26"></a>
## C26 — Reuse NMF sufficient products within each accepted step

**Affected estimators:** NMF; nonnegative dictionary learning.

**Source anchors:** `x_decomp/device.mojo`, `x_decomp`.

**A:** Cache WᵀW/WᵀX or HHᵀ/XHᵀ only while the corresponding factor is unchanged, fuse elementwise update/clamp passes with explicit rounding.

**B:** Recompute products or materialize each update intermediate independently.

**Identity/version contract:** Same alternating update dependency, normalization, epsilon and zero-lock behavior; never reuse across a factor mutation.

**Quality gates:** Reconstruction error, objective trajectory, iterations and nonnegative fitted factors.

**Reject or defer when:** Reject stale products or a changed epsilon/tolerance/rank.

**Prior work:** New extension

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c27"></a>
## C27 — Batch independent ICA and factor-analysis component work

**Affected estimators:** FastICA; FactorAnalysis; CCA; PLS.

**Source anchors:** `x_decomp/device.mojo`, `x_decomp`.

**A:** Share centered input tiles across independent component score/nonlinearity calculations; batch independent solves and defer only redundant drains.

**B:** Separate scans and temporary matrices for each component.

**Identity/version contract:** Preserve deflation dependencies, orthogonalization order and sign/tie conventions; only truly independent work can execute concurrently.

**Quality gates:** Convergence, reconstruction, component subspace, likelihood/correlation and predictions.

**Reject or defer when:** Reject altered deflation or an aggregate quality metric hiding a failed component.

**Prior work:** New extension

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c28"></a>
## C28 — Bucket independent sparse-coding and ALS solves

**Affected estimators:** SparseCoder; DictionaryLearning; ImplicitALS.

**Source anchors:** `x_decomp/als_dev.mojo`, `x_decomp/device.mojo`, `x_decomp`.

**A:** Group independent row/item solves by nonzero count or active-set workspace and reuse a shared dictionary/Gram tile.

**B:** One uniform task shape and repeated dictionary loads for all rows.

**Identity/version contract:** Stable row/nonzero order, identical regularization and canonical solves; scheduling changes cannot alter active-set ties or the ALS half-step barrier.

**Quality gates:** Objective, support, recommendation ranking and full factor state; empty and extremely long rows.

**Reject or defer when:** Reject sorting/descriptor cost, peak scratch or silently approximated solves.

**Prior work:** New extension

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c29"></a>
## C29 — Streaming exact distance-to-top-k

**Affected estimators:** NearestNeighbors; KNNClassifier/Regressor; LOF; exact graph construction.

**Source anchors:** `neighbors/impl/detail/knn_brute_force.mojo`, `neighbors/impl/detail/fused_l2_knn.mojo`.

**A:** Fuse distance tiles with local exact top-k and a bounded canonical merge, avoiding a full query-by-reference distance matrix.

**B:** Materialize distances then select.

**Identity/version contract:** Exact (distance,index) total order, same metric arithmetic and self-neighbor policy. Approximate pruning is excluded unless independently certified with exact fallback.

**Quality gates:** Every neighbor/index/distance and downstream predictions or LOF score; large k, ties, duplicates and tails.

**Reject or defer when:** Reject merge overhead, oversized local lists or unproven exclusion bounds.

**Prior work:** Reuse I15/A06 selection machinery; split new full-caller tile/merge arms from prior experiments.

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c30"></a>
## C30 — Versioned direct-difference distance arithmetic

**Affected estimators:** KNN; KMeans; KDE; RBF kernels; radius graphs.

**Source anchors:** `core/expand_distances.mojo`, `core/row_norms.mojo`, `neighbors/impl/detail/fused_l2_knn.mojo`.

**A:** Use fixed-fold sum((x-y)²) in bounded tiles, independently compare register lane counts with one logical arithmetic profile.

**B:** Expanded ||x||²+||y||²-2x·y with the incumbent clamp.

**Identity/version contract:** This deliberately changes version bits: subtraction, square/FMA, FTZ, fold and distance clamp must be identical in every column; metric semantics remain exact Euclidean.

**Quality gates:** Neighbor membership, cancellation-sensitive distances, clustering quality, densities and model metrics; nearly equal large vectors.

**Reject or defer when:** Reject degraded task quality, overflow behavior or replacing a configured metric.

**Prior work:** New extension

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c31"></a>
## C31 — Bucket ragged CSR canonicalization by work

**Affected estimators:** RadiusNeighbors; DBSCAN; HDBSCAN graph preparation.

**Source anchors:** `neighbors`, `dbscan`, `core/segmented_sort.mojo`.

**A:** Compact row tasks by required merge levels/degree and schedule stable merges only for active runs.

**B:** Uniform merge levels driven by the longest row.

**Identity/version contract:** Sort only; preserve membership, multiplicity, CSR offsets and stable neighbor index order.

**Quality gates:** Complete canonical CSR words and final cluster states; singleton, empty, huge and skewed rows.

**Reject or defer when:** Reject max-degree readback or descriptors erasing the gain.

**Prior work:** Follow-up to I13 stable canonical merge, not its initial implementation.

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c32"></a>
## C32 — Fuse exact radius threshold and compact emission

**Affected estimators:** RadiusNeighbors; DBSCAN; graph building.

**Source anchors:** `neighbors`, `dbscan`, `experiments/performance_ideas/N08/fused_threshold.mojo`.

**A:** Distance evaluation writes threshold flags/counts directly, then emits canonical accepted edges using bounded scans; attribute count-only and emit fusion separately.

**B:** Write full distance tiles, then threshold and compact.

**Identity/version contract:** Same inclusive/exclusive threshold rule and portable distance arithmetic; emit sorted indices or retain canonicalization.

**Quality gates:** Exact edge sets, offsets, distances and boundary-equal cases plus DBSCAN output.

**Reject or defer when:** Reject duplicate distance evaluation costing more than traffic saved or a moved epsilon boundary.

**Prior work:** Reuse N08; extend vendor-neutral caller integration.

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c33"></a>
## C33 — Freeze first graph convergence state on device

**Affected estimators:** DBSCAN; connected components; label propagation/spreading.

**Source anchors:** `dbscan`, `core/device_fold.mojo`, `x_neighbors`.

**A:** Queue a bounded chunk of iterations under a device live mask, preserving the first converged state and performing one status readback.

**B:** Host polls after every iteration.

**Identity/version contract:** Frozen inactive state, first stopping iteration and canonical label assignment; later speculative rounds cannot modify converged buffers.

**Quality gates:** Labels, graph probabilities, stopping counts, long chains, disconnected and nearly converged graphs.

**Reject or defer when:** Reject excessive masked work or observing only the last round of a chunk.

**Prior work:** Reuse I14 convergence machinery; include full graph callers and interactions.

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c34"></a>
## C34 — Canonical parallel MST edge selection

**Affected estimators:** HDBSCAN; single-linkage clustering.

**Source anchors:** `hdbscan/impl/detail/reachability.mojo`, `hdbscan/impl/detail/tree_device.mojo`, `hierarchy`.

**A:** Reduce candidate edges in parallel with a total (weight,min_endpoint,max_endpoint) key and deterministically emit chosen edges/components.

**B:** Incumbent repeated edge scans or serial canonical selection.

**Identity/version contract:** Mutual-reachability arithmetic and tie policy fixed; if tie contract changes, all model/host columns must move in one new version.

**Quality gates:** MST, condensation, cluster labels/probabilities, stability, prediction and outlier outputs.

**Reject or defer when:** Reject floating atomic ordering, disconnected-graph mishandling or claiming identical labels alone proves equivalent hierarchy.

**Prior work:** New extension

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c35"></a>
## C35 — IVF long-list chunking and short-list packing

**Affected estimators:** IVFFlat search.

**Source anchors:** `ivf/resident.mojo`, `ivf`.

**A:** Split long selected lists into independent chunks and pack small ones, merging exact top-k results in canonical order.

**B:** One fixed task per query/list.

**Identity/version contract:** Centroids, nprobe, list membership, distances, ties and index training unchanged; no reduced recall allowance.

**Quality gates:** Full search IDs/distances and declared recall, giant/empty lists and batch tails.

**Reject or defer when:** Reject task setup or merge cost and do not change nprobe to obtain speed.

**Prior work:** Extend I16 scheduling with recorded scope; no duplication of already supplied staging arms.

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c36"></a>
## C36 — Reuse KMeans centroid tiles through assignment

**Affected estimators:** KMeans; MiniBatchKMeans; BisectingKMeans; IVF training.

**Source anchors:** `cluster/impl/detail/kmeans.mojo`, `cluster/impl/detail/min_cluster_distance_compute.mojo`.

**A:** Share centroid/norm tiles across multiple independent row assignments and fuse nearest-centroid selection with distances.

**B:** Separate distance production and assignment selection with repeated centroid reads.

**Identity/version contract:** Keep canonical centroid index ties, incumbent feature folds, inertia and empty-cluster handling.

**Quality gates:** Labels, centers, inertia, iterations and initialization trajectories across multiple seeds.

**Reject or defer when:** Reject register pressure or different center assignment at near ties.

**Prior work:** New extension

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c37"></a>
## C37 — Versioned fixed-row centroid accumulation

**Affected estimators:** KMeans; MiniBatchKMeans; BisectingKMeans; IVF training.

**Source anchors:** `cluster/impl/detail/kmeans.mojo`, `cluster/impl/detail/kmeans_common.mojo`.

**A:** Accumulate row panels with fixed logical membership and a canonical merge; separate cluster-feature task geometry from the arithmetic partition.

**B:** Current centroid sum accumulation profile.

**Identity/version contract:** New profile only if logical row partition changes; host and all GPUs share partition, sample-weight multiplication and normalization order.

**Quality gates:** Centers, inertia and task quality with imbalanced/empty clusters, large weight ranges and cancellation.

**Reject or defer when:** Reject atomic floating sums, hardware-dependent leaves or a single-vendor win with material loss on the other.

**Prior work:** A08 tested a geometry candidate; reuse evidence only when source/profile actually match.

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c38"></a>
## C38 — KMeans++ nearest-distance reuse

**Affected estimators:** KMeans initialization; IVF index training; BisectingKMeans.

**Source anchors:** `cluster/impl/detail/kmeans.mojo`.

**A:** Update the minimum-distance vector only against newly added centers and reuse candidate potential partials with bounded device status handling.

**B:** Recompute distances against all chosen centers or drain each independent trial separately.

**Identity/version contract:** Same random draws, candidate order, potential fold, first minimum tie and selected centers.

**Quality gates:** Exact chosen seed centers and full fit quality; duplicates and exhausted nonzero distance mass.

**Reject or defer when:** Reject stale minimum vectors, altered random draw counts or speculative trials changing selection.

**Prior work:** Existing incremental init/PSI switches need selective recipe coverage; do not reinvent them.

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c39"></a>
## C39 — Retain mini-batch and bisecting clustering state

**Affected estimators:** MiniBatchKMeans; BisectingKMeans.

**Source anchors:** `x_cluster/minibatch.mojo`, `x_cluster/minibatch_cells.mojo`, `x_cluster/bisect_fast.mojo`.

**A:** Retain counts, center state, subset row IDs and statistics across ordered batches/splits in IDENTICAL Mojo paths.

**B:** Repeated transfers, reconstruction or independent temporary allocation per batch/split.

**Identity/version contract:** Seed stream, batch membership/order, center reassignment, split priority and stopping remain fixed; FAST code is a reference for storage only.

**Quality gates:** Centers, labels, counts and batch/iteration histories; skewed and empty partitions.

**Reject or defer when:** Reject importing FAST arithmetic, state leakage or hidden batch-size changes.

**Prior work:** New extension

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c40"></a>
## C40 — Share MeanShift distance tiles across seed updates

**Affected estimators:** MeanShift.

**Source anchors:** `x_cluster/meanshift.mojo`, `x_cluster/meanshift_idn.mojo`.

**A:** Batch independent live seeds with bounded distance tiles and ordered in-band sums; compact inactive seeds without changing their logical IDs.

**B:** Per-seed scans with uniform work for completed seeds.

**Identity/version contract:** Bandwidth, inclusive boundary, update fold, stopping and final mode suppression/ties unchanged.

**Quality gates:** Centers, cluster assignment and convergence histories on uneven seed densities.

**Reject or defer when:** Reject changing bin seeding or bandwidth estimation sample counts.

**Prior work:** New extension

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c41"></a>
## C41 — OPTICS fused canonical reachability minima

**Affected estimators:** OPTICS.

**Source anchors:** `x_cluster/optics.mojo`, `x_cluster/optics_xi_device.mojo`, `x_cluster/device_ops.mojo`.

**A:** Fuse eligibility/reachability updates and compact minimum selection; reuse computed neighbor rows where immutable.

**B:** Separate full scans and redundant neighbor calculations per expansion.

**Identity/version contract:** Expansion order, predecessor ties, core distances and Xi extraction must remain canonical; no approximate queue ordering.

**Quality gates:** Ordering, reachability, predecessor and extracted hierarchy/labels.

**Reject or defer when:** Reject task overhead or a change to the first equal-reachability row.

**Prior work:** New extension

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c42"></a>
## C42 — Compact active agglomerative distance work

**Affected estimators:** AgglomerativeClustering; linkage methods.

**Source anchors:** `x_cluster/agglo.mojo`, `x_cluster/device_tree.mojo`, `hierarchy`.

**A:** Store one canonical triangle and maintain deterministic active-cluster task lists while updating only affected distance rows.

**B:** Scan/update full square workspaces including inactive entries.

**Identity/version contract:** Preserve linkage recurrence arithmetic, merge ties, cluster numbering and connectivity constraints.

**Quality gates:** Children, distances, cut labels and full dendrogram on equal-distance and disconnected cases.

**Reject or defer when:** Reject compaction costs or use of a nearest-neighbor-chain algorithm without proving the required tie contract.

**Prior work:** New extension

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c43"></a>
## C43 — Resident normalized graph operators

**Affected estimators:** SpectralEmbedding/Clustering; LabelPropagation; PageRank; Isomap/LLE graph preparation.

**Source anchors:** `spectral`, `x_cluster/spectral_assign.mojo`, `umap`, `x_neighbors`.

**A:** Reuse CSR degree/normalization buffers and fuse pointwise scaling into fixed-fold sparse matvec consumers.

**B:** Materialize normalized matrices and repeat host/device preparation or separate scaling passes.

**Identity/version contract:** Per-row neighbor order and reduction tree fixed; eigenvector sign/order, solver stopping and graph topology unchanged.

**Quality gates:** Residuals, embeddings/subspace quality, graph outputs and downstream labels; isolated and high-degree rows.

**Reject or defer when:** Reject changed eigen-solver tolerance or graph sparsification as a supposed scheduling optimization.

**Prior work:** New extension

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c44"></a>
## C44 — Seeded UMAP graph and sampling-state reuse

**Affected estimators:** UMAP; manifold graph preparation.

**Source anchors:** `umap`, `core/philox.mojo`, `core/segmented_sort.mojo`.

**A:** Cache immutable canonical fuzzy-graph metadata and generate independent edge-sampling descriptors on device, reusing them across ordered optimization steps.

**B:** Repeated metadata construction or transfer for the same graph/epoch.

**Identity/version contract:** Preserve RNG draw mapping, edge order, negative samples and dependent coordinate update order; parallel conflicting SGD updates are excluded.

**Quality gates:** Seeded embedding words, trustworthiness/neighbor preservation, graph and epoch state.

**Reject or defer when:** Reject dropping epochs, samples or edges, or using unseeded execution to get parallelism.

**Prior work:** New extension

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c45"></a>
## C45 — Fuse tree split statistics and candidate scoring

**Affected estimators:** GBDT symmetric/depthwise/lossguide; RandomForest; ExtraTrees.

**Source anchors:** `gbdt`, `ensemble/decisiontree/batched_levelalgo/split.mojo`, `extratrees`.

**A:** Consume ordered histogram prefixes directly in score/tie selection without writing all intermediate prefix/score arrays.

**B:** Separate prefix, gain and best-split kernels.

**Identity/version contract:** Same candidate set, prefix arithmetic, regularization, missing direction, score rounding and feature/threshold tie order.

**Quality gates:** Split sequence, leaf values, predictions and task quality, including categorical/missing inputs.

**Reject or defer when:** Reject register pressure, altered tie selection or fewer candidate thresholds.

**Prior work:** Builds on I17 frontier scheduling; different pass-fusion hypothesis.

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c46"></a>
## C46 — Exact histogram sibling subtraction

**Affected estimators:** RandomForest; ExtraTrees; bounded exact tree histograms.

**Source anchors:** `ensemble/decisiontree/batched_levelalgo/exact_histogram_subtract.mojo`, `ensemble/decisiontree/batched_levelalgo/builder.mojo`.

**A:** Compute the smaller child and recover the sibling from retained parent exact integer/limb counts under proven range bounds.

**B:** Build both child histograms independently.

**Identity/version contract:** Only exact bounded accumulators qualify; floating gradient sums require a distinct coordinated numerical profile and are not enabled by this card.

**Quality gates:** Histogram cells, split/leaf/model identity, overflow/refusal and sampled-feature semantics.

**Reject or defer when:** Reject unbounded parent retention or subtracting mismatched feature/row populations.

**Prior work:** Reuse I18 implementation; extend applicability only with explicit accumulator proofs.

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c47"></a>
## C47 — Tree frontier tasks by live work and histogram bytes

**Affected estimators:** GBDT; RandomForest; ExtraTrees; IsolationForest.

**Source anchors:** `gbdt`, `ensemble/decisiontree/batched_levelalgo/builder.mojo`, `extratrees`, `isolation_forest`.

**A:** Bucket independent node-feature tasks by rows and bounded histogram footprint, compacting inactive nodes and retaining stable logical node IDs.

**B:** Uniform task geometry across a skewed frontier.

**Identity/version contract:** Scheduling cannot alter row membership, feature RNG mapping, split order or arithmetic partitions.

**Quality gates:** Tree structures and predictions for deep/skewed and broad trees, peak memory and task counts.

**Reject or defer when:** Reject descriptor overhead, uncontrolled frontiers or thresholds fitted to board dimensions.

**Prior work:** Extends I17/A07/N07; attribute new geometry independently of existing candidates.

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c48"></a>
## C48 — Stable tree partition fused with child bookkeeping

**Affected estimators:** GBDT; RandomForest; ExtraTrees; IsolationForest.

**Source anchors:** `ensemble/decisiontree/batched_levelalgo/builder.mojo`, `extratrees`, `gbdt`.

**A:** Use one stable flag/scan/scatter to produce child row indices, counts and offsets; avoid separate metadata and redundant row scans.

**B:** Independent partition and child-statistic bookkeeping passes.

**Identity/version contract:** Stable original row order within each child, exact offsets, missing direction and overflow handling.

**Quality gates:** Child row lists, sampled weights, all descendant splits and public model words.

**Reject or defer when:** Reject unstable atomics or consuming partially published offsets.

**Prior work:** New extension

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c49"></a>
## C49 — Batch independent forest construction with logical RNG IDs

**Affected estimators:** RandomForest; ExtraTrees; IsolationForest.

**Source anchors:** `ensemble/randomforest.mojo`, `extratrees/estimator.mojo`, `isolation_forest`.

**A:** Share immutable training tiles among bounded independent tree tasks; key random draws by existing tree/node/draw identities.

**B:** Build one tree at a time with repeated input staging.

**Identity/version contract:** Keep bootstrap, feature sampling, random thresholds and forest output tree order fixed; no worker-completion-order RNG.

**Quality gates:** Per-tree structures, OOB state, predictions and anomaly scores across seeds.

**Reject or defer when:** Reject memory multiplied by forest width or an implicit reduction-order change in forest averaging.

**Prior work:** I18 multi-tree frontier was still open; distinguish construction from inference batching.

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c50"></a>
## C50 — Packed forest inference and fixed tree-output reduction

**Affected estimators:** GBDT; RandomForest; ExtraTrees; IsolationForest.

**Source anchors:** `core/forest_inference.mojo`, `core/forest_inference_model.mojo`, `core/gbdt_host_predict.mojo`.

**A:** Compare structure-of-arrays node storage and bounded row/tree tiles, retaining a fixed logical per-row tree accumulation order.

**B:** Incumbent model layout and per-tree traversal schedule.

**Identity/version contract:** No threshold quantization or leaf compression changing values; preserve missing/categorical branches, class order and averaging arithmetic.

**Quality gates:** Decision paths, leaf IDs, scores/probabilities and predict outputs; unbalanced depths and repeated/cold use.

**Reject or defer when:** Reject model repacking cost hiding in untimed setup or altered reductions across tree batches.

**Prior work:** New extension

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c51"></a>
## C51 — TreeSHAP shared path metadata and bounded batching

**Affected estimators:** TreeSHAP; forest feature contributions.

**Source anchors:** `xtrees/shap_device.mojo`, `xtrees/shap_tab.mojo`, `xtrees/shap_host.mojo`.

**A:** Reuse immutable path/cover metadata and batch independent row/tree explanations while reducing contributions in canonical tree order.

**B:** Reconstruct path metadata and issue per-row/tree work separately.

**Identity/version contract:** Exact SHAP semantics, repeated-feature path unwinding, zero covers and bias contribution retained; no approximate SHAP substitution.

**Quality gates:** All contribution words across vendors, additivity residual, interactions if supported and zero-cover behavior.

**Reject or defer when:** Reject exponential scratch growth or change to feature attribution semantics.

**Prior work:** New extension

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c52"></a>
## C52 — Versioned stable pair-combine KDE reduction

**Affected estimators:** KernelDensity score_samples and scoring.

**Source anchors:** `kde`, `experiments/performance_ideas/I20`.

**A:** Represent each fixed reference tile by (maximum,scaled_sum) and combine pairs with portable exp/log in a declared tree, sharing distance/kernel work.

**B:** Incumbent chunked log-sum-exp fold and temporary layout.

**Identity/version contract:** New host/device profile includes -Inf/empty tiles, zero contributions, rescaling order and tail carry; same metric, kernel and bandwidth.

**Quality gates:** Log-density accuracy, held-out likelihood, extreme bandwidths and all supported kernels/metrics; no quality regression.

**Reject or defer when:** Reject extra exponentials, overflow or silently dropping small nonzero contributions.

**Prior work:** I20 schedule work exists; this is its separate arithmetic-profile proposal.

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c53"></a>
## C53 — GMM reusable centered tiles and fused sufficient statistics

**Affected estimators:** GaussianMixture; BayesianGaussianMixture.

**Source anchors:** `mixture`, `x_cluster/bgmm_device.mojo`, `x_cluster/bgmm_kernels.mojo`.

**A:** Share centered tiles/responsibility reads across independent component statistics with bounded batching; separate each fusion from component stacking.

**B:** Reread/recenter inputs for each statistic/component.

**Identity/version contract:** Keep responsibility normalization, covariance fold, regularization, empty-component policy and first convergence state; changed statistic fold requires its own version profile.

**Quality gates:** Likelihood, iterations, weights/means/covariances/precisions and predictions; all covariance types and near singularities.

**Reject or defer when:** Reject recreating the documented AMD E-step stacking loss or cheaper iterations that increase full-fit time.

**Prior work:** Reuse I21 centered-load/fusion work and retained loser evidence.

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c54"></a>
## C54 — GP factor and batched prediction reuse

**Affected estimators:** GaussianProcessRegressor; GaussianProcessClassifier; sparse variational GP.

**Source anchors:** `gaussian_process/gpr_resident.mojo`, `gaussian_process/gpc_resident_k.mojo`, `gaussian_process/gp_optim.mojo`.

**A:** Retain fitted kernel/factor state for grouped mean/variance RHS tasks and reuse shared kernel tiles across independent class/hyperparameter evaluations.

**B:** Rebuild/reupload factors or compute mean and variance kernel tiles separately.

**Identity/version contract:** Hyperparameters, exact/variational model choice, jitter/pivot policy and optimizer trial selection unchanged; caches invalidate on theta or training-data changes.

**Quality gates:** Predictive means/variances, log marginal likelihood, gradients, probabilities and calibration.

**Reject or defer when:** Reject replacing exact GP with sparse approximation or timing only mean when variance is requested.

**Prior work:** New extension

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c55"></a>
## C55 — Naive Bayes class-statistic pass fusion

**Affected estimators:** GaussianNB; MultinomialNB; BernoulliNB; ComplementNB; CategoricalNB.

**Source anchors:** `naive_bayes/nb.mojo`, `core/label_rows_device.mojo`.

**A:** Reuse canonical label grouping and input loads across counts/means/centered moments, emitting only required fitted statistics.

**B:** Separate class and feature scans and intermediate arrays.

**Identity/version contract:** Integer counts exact; weighted floating moments use the same fixed partition or an explicitly versioned common profile. Smoothing and prior semantics unchanged.

**Quality gates:** Counts, means/variance, log probabilities and predictions; absent classes, zero counts and partial_fit where supported.

**Reject or defer when:** Reject raw-moment variance cancellation or double-applying smoothing during reuse.

**Prior work:** New extension

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c56"></a>
## C56 — Discriminant class solves and prediction fusion

**Affected estimators:** LinearDiscriminantAnalysis; QuadraticDiscriminantAnalysis.

**Source anchors:** `naive_bayes/da.mojo`, `cholesky`, `core/column_stats.mojo`.

**A:** Batch independent class factorizations/solves and share prediction input tiles with fixed per-class quadratic forms.

**B:** Per-class staging, launch and centered intermediate matrices.

**Identity/version contract:** Same covariance, shrinkage, priors, factor/rank thresholds and quadratic-form fold; class batching changes scheduling only.

**Quality gates:** Fitted factors, decision values, posterior probabilities, accuracy and singular-class refusals.

**Reject or defer when:** Reject changed rank truncation or posterior normalization.

**Prior work:** New extension

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c57"></a>
## C57 — Robust covariance candidate state reuse

**Affected estimators:** MinCovDet; EllipticEnvelope; covariance estimators.

**Source anchors:** `x_decomp`, `x_prep`.

**A:** Reuse immutable row tiles across independent robust-subset candidates and batch candidate covariance/distance work; retain canonical subset selection.

**B:** Repeated full staging/scans for each candidate.

**Identity/version contract:** Seeded subset membership, C-step update order, support-size settings, determinant ties and reweighting unchanged.

**Quality gates:** Location, covariance, support, distances, anomaly labels and robust objective.

**Reject or defer when:** Reject fewer trials, changed contamination/support fraction or a covariance shortcut that degrades robustness.

**Prior work:** New extension

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c58"></a>
## C58 — Batch independent classical forecast candidates

**Affected estimators:** ETS; Holt-Winters; Theta; GARCH; independent forecast series.

**Source anchors:** `holtwinters/impl/holtwinters.mojo`, `tsa`, `sequence`.

**A:** Schedule independent series and parameter trials in bounded tasks, sharing immutable observations and retaining scalar recurrence order within a series.

**B:** Per-series/trial dispatch and repeated observation transfers.

**Identity/version contract:** Only classical forecast callers; no neural sequence code changes. Preserve first selected trial, initialization, recurrence arithmetic and convergence.

**Quality gates:** Likelihood/SSE, selected parameters, fitted states, forecasts and forecast errors; short and ragged seasonal series.

**Reject or defer when:** Reject changed search grid or fewer optimizer trials.

**Prior work:** Use I23 independent-work infrastructure; do not duplicate existing forecast arms.

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c59"></a>
## C59 — ARIMA device-resident order and likelihood trials

**Affected estimators:** ARIMA; AutoARIMA; SARIMA.

**Source anchors:** `arima`, `tsa/impl/auto_arima.mojo`, `tsa/impl/timeSeries/arima_helpers.mojo`.

**A:** Retain differenced observations and independent likelihood trial states, batching supported order/trial evaluations with deterministic acceptance selection.

**B:** Recreate or transfer immutable preparation per trial and poll each independent evaluation.

**Identity/version contract:** Order search sequence, AIC/BIC ties, likelihood/Kalman arithmetic, stationarity constraints and optimizer stopping unchanged.

**Quality gates:** Chosen orders, coefficients, innovations, likelihood and forecast quality; singular/refused candidates.

**Reject or defer when:** Reject approximate Kalman covariance, skipped orders or unsupported device features.

**Prior work:** Follow-up to I23 batching; profile-changing affine scans remain separate future work.

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

<a id="c60"></a>
## C60 — Reuse classical stationarity preparation and lag work

**Affected estimators:** ADF; KPSS; AutoARIMA differencing selection; autocorrelation features.

**Source anchors:** `tsa/impl/stationarity.mojo`, `tsa/impl/timeSeries/kpss_fused.mojo`, `tsa/impl/select_d_fast.mojo`.

**A:** Share canonical detrending/differencing buffers and lagged product tiles across requested tests/lags, fusing independent scalar report work.

**B:** Recompute each preparation and rescan series per test/lag.

**Identity/version contract:** Same test definitions, lag selection, trend, null, critical-value interpolation and differencing stopping; FAST code only informs storage scheduling.

**Quality gates:** Statistics, p-values, selected lags/differencing order and downstream forecast quality.

**Reject or defer when:** Reject truncated series, changed lag caps or altered critical-value policy.

**Prior work:** New extension

**Future measurement:** full affected-estimator operation on saved full datasets, including preparation, synchronization and consumed outputs; separate fit/inference/cold/repeated use as applicable. Record launches, bytes, peak scratch and solver iterations as diagnostics. Cover neighboring shapes and a non-board dataset; missing/capped recipes remain pending.

**Current evidence:** design only at fan-out; see lane handoffs for subsequently drafted source. Compilation, identity, quality and timing intentionally not run.

## Interaction rounds

- C01 × C02/C09/C13/C23/C37/C52: arithmetic profile plus pass fusion; test each alone before combined.
- C03/C04/C05/C08 × all affected estimators: input preparation, scratch retention and data ownership; include cold/repeated fit.
- C06/C29/C30/C32/C36: norms, direct/expanded distances, selection and threshold fusion; ensure metric route reach.
- C07/C08/C10/C12/C31: shared sort/scan storage and canonical key contracts; avoid unbounded combined scratch.
- C13/C14/C15/C18/C23/C24: CV statistics, factorization and solve profile; selected model can change only with preserved quality.
- C16/C17/C19: objective fusion, independent trials and persistent updates; never speculate dependent SGD samples.
- C20/C21/C22: SVM cache, extrema and symmetric kernels; check complete trajectory and memory together.
- C29/C31/C32/C33/C34/C43/C44: graph membership/order, convergence, hierarchy and manifold quality.
- C36/C37/C38/C39: KMeans assignment, centroid fold, initialization and state retention; full-fit acceptance.
- C45/C46/C47/C48/C49/C50/C51: complete tree construction/prediction/explanation configurations.
- C02/C04/C14/C23/C53/C54/C55/C56/C57: covariance/factor consumers and their uncertainty/quality outputs.
- C58/C59/C60: shared forecasting preparation, candidate batching and order selection.

## Deliberately excluded shortcuts

No lower precision/TF32 substitution, approximate exact-neighbor search, fewer trees/iterations/restarts/epochs, lower rank, larger convergence tolerance, reduced datasets, changed seeds, unseeded UMAP, removed uncertainty outputs, CPU work hidden inside GPU timing, Python runtime computation, or board-specific routing. A separately supported algorithm option is not evidence for accelerating the same estimator contract.

## Implementation handoff

Four disjoint lanes: shared C01–C12; linear/decomposition C13–C28; graph/clustering C29–C44; trees/probabilistic/classical-time-series C45–C60. Implement actual opt-in Mojo candidates or wire existing candidates into attributable recipes. Do not substitute flags, empty wrappers or manifests for an implementation. Record precisely when only a component is drafted and a public caller remains unwired. Unsupported capabilities become explicit Modular asks. No compile, verification or timing is authorized in this worktree.

> Keep logs out of context: save complete output to files, use targeted rg/grep with bounded surrounding lines and short tails, and summarize exit status, coverage, failures, and evidence paths. Expand only relevant diagnostic blocks; never hide failures or infer full success from filtered output.
