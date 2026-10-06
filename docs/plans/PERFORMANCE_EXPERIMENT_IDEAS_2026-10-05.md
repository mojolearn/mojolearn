# Performance experiment ideas for AMD and NVIDIA IDENTICAL and Apple FAST

These 60 proposals target faster public operations, with priority on shared GEMM scheduling, device residency, deterministic irregular work, and Apple FAST kernels that already have usable experimental implementations. They are ideas for future experiments. This change implements no kernels, changes no defaults, runs no benchmarks, and makes no new speedup claims.

The source baseline is `integration/identical-all-20261004` at `f867b50e8f09c493ef8c2c939aded6b3a9f0910e`, October 5, 2026. The planning branch is `codex/performance-experiment-ideas-20261005`. Source links below are pinned to that baseline so this document also works on older branches. Later experimental branches are named explicitly; their code is not part of the baseline.

## What deserves attention first

Start by establishing which existing candidates help on both NVIDIA and AMD. Several earlier roadmap ideas are now implemented, and a global all-on versus all-off result cannot attribute a gain to each switch. For Apple FAST, prioritize real callers of the existing GEMM and neural candidates, with quality gates and timing through completion and the caller's first read.

Do not rank work using old opponent ratios as though they describe today's source. Historical profiles help choose what to inspect; the next frozen source needs its own stage attribution. For example, the [neural roadmap][neural-roadmap] cites an older AMD LM profile dominated by GEMM. That supports inspecting GEMM first, not claiming the current LM has the same breakdown.

| Order | Experiments | Why start here | First useful result |
| --- | --- | --- | --- |
| 1 | I01, A01, N01 | Existing schedules with broad neural reach | Actual route coverage and an isolated NVIDIA plus AMD comparison |
| 2 | I02, I03, A02, N02 | Scratch traffic, allocations, and resource pressure can erase arithmetic gains | Lower total operation time with bounded workspace |
| 3 | I08, I09, F07, F08 | Existing neural candidates address substantial repeated work | Forward and training quality plus whole-operation timing |
| 4 | I13, I14, I17 | Dense neighborhoods and tree frontiers can amplify otherwise small costs | Better scaling on dense or skewed inputs without changing the answer |
| 5 | F01, F02, F05 | Apple GEMM screens have leads but incomplete caller admission | A real estimator win with the intended kernel actually reached |
| 6 | I10, I12, I19, F03 | Launch and synchronization overhead may dominate smaller operations | Fewer waits or passes with identical error and lifetime behavior |
| 7 | I18, I20, I21, F11, F12, F17 | Broader classical and time-series opportunities | Family-specific quality and memory results |
| 8 | I04, I22, I23, F16 | Higher design cost or numerical risk | A valid numerical formulation before performance work |

Priority is conditional on the next profile. A reduction in a stage occupying fraction `f` of total time has a maximum whole-operation speedup of `1 / ((1 - f) + f / s)` when that stage becomes `s` times faster. Use that bound before investing in a complicated microkernel.

## Rules for interpreting and eventually testing the ideas

### IDENTICAL numerical scope

NVIDIA and AMD are the performance targets. Apple and the host are identity witnesses; Apple performance does not vote on IDENTICAL defaults. Same-version outputs must match across all four columns. Old-version bits are not sacred: an explicitly revised numerical profile may change the fold or algorithmic evaluation order if every column, host oracle, contract, and affected caller changes together.

The cards distinguish **schedule** changes, which aim to preserve the existing arithmetic graph, from **profile** changes, which need a coordinated numerical revision. These are design claims to verify, not evidence of identity. A hardware-specific schedule may differ; hardware-specific numerical answers may not. FP32 remains FP32; these proposals do not authorize replacing it with TF32, BF16, or FP16.

Use the actual [FP32 contract][fp32-contract]: the logical partition depends on the contraction dimension and profile, with specified FMA, flush, adjacent-pair fold, and odd-tail carry behavior. Preserving only the final reduction tree is insufficient if a leaf's arithmetic or a signed-zero seam changes. Canonical ties, RNG mapping, convergence iteration, refusals, model state, and backward/checkpoint behavior belong in the affected identity checks.

### Apple FAST numerical scope

Apple FAST is judged by speed and task quality. Different bits, a new fold, or run-to-run variation alone are not failures. Require no material quality loss against FAST main and quality at least as good as the admitted best opponent under the repository's metric policy. Preserve API semantics, finite/error handling, fitted attributes, and buffer ownership. Do not infer quality from output hashes or a kernel-only oracle.

Where prior notes demand zero numerical difference, use the later policy in the [experiment ledger][apple-ledger]. A real quality failure remains a failure. A neutral timing or a missing quality metric is not a promotion.

### General constraints

- Runtime data processing stays in Mojo, on the GPU for GPU routes. Python is API glue. No CPU or NumPy runtime branch is proposed to make an experiment look faster.
- A route rule must follow a documented hardware, workload-size, memory, or cost argument. No exact benchmark dimensions, dataset names, seed recognition, or handpicked boundaries near board rows. Test neighboring shapes and at least one non-board workload.
- Reuse supported Mojo and Modular capabilities. If an intrinsic, stream/event operation, graph facility, or required device property is unavailable, park that implementation and record the upstream request. Do not rewrite compiler output or invent an unsupported backend.
- One hypothesis per first comparison. Existing flag names are cited only where found. New cards have idea IDs, not claimed implementations or registered defines.
- Prefer a simple consistent small improvement over an elaborate fragile one. The [acceptance policy][acceptance] has no fixed minimum speedup percentage.

### A future measurement round

1. Freeze one source SHA. Record every define, binding, compiler, artifact hash, hardware identity, input/settings hash, and actual dispatched route. Compile on the established cheap build machines before using timing GPUs.
2. Run unscored correctness/quality and reach checks first. A compiled flag or import is not proof the candidate ran. Check the fallback and the disabled arm too.
3. Use current production as the control and change one mechanism. Most opt-in candidates require the normal on configuration in both arms; using `MOJOLEARN_IDN_ALL_OFF` as the ordinary control confounds all other optimizations. For removal of a benchmark-fitted rule, retain the old rule as the designated B arm, per [AGENTS.md][agents].
4. Respect the established one scored run per arm policy and matched warmup. Alternate arm order across separately frozen rounds where appropriate. Do not silently add repetitions. One sample cannot establish statistical confidence for a small difference; record an inconclusive result and a proposed confirmation round.
5. Measure the public call through completion and required output consumption. Add stage diagnostics, launch/wait counts, transfer bytes, peak scratch, and resource counters separately. Resident-call measurements are a different contract from cold public calls; report both explicitly when relevant.
6. IDENTICAL acceptance requires a combined NVIDIA/AMD benefit and neither materially slower, with the new version's four-column identity established. Report each vendor and workload individually as well as a predeclared aggregate; do not hide a regression in the aggregate. Apple IDENTICAL board measurements remain reporting work, not an optimization vote.
7. Reuse admitted opponent results only when the input, settings, hardware, and measurement contract match. A separate opponent measurement is needed for a genuinely new comparison. Do not rerun opponents as a side effect of candidate A/B work.
8. Only actual future results belong in the execution ledger and board tools. No board values, measured verdicts, or execution jobs are created by this proposal.

## Existing evidence that changes the queue

| Area | State at the reviewed baseline or named branch | Consequence |
| --- | --- | --- |
| AMD smaller MFMA body and NVIDIA stepped packed body | Dispatch already contains `IDN_GEMM_MFMA16`, `IDN_GEMM_AMD_BAND_MFMA`, and `IDN_GEMM_NV_STEP_KPACK` | Attribute and extend them; do not propose their initial implementation again |
| GEMM schedule campaign | `lane/identical-speed-ab-20261005@a006da73d` defines one-page, slack2, slack8, body-tiles, and packed64 profiles against `acbac5221` | Reuse/rebase the campaign design for the next freeze; this is not evidence that the profiles won |
| IDENTICAL ledger | Nine records include a measured global wave, KMeans accumulate-256 marked KEEP but not promoted, two DROP entries, and incomplete qualifications | Check current source and evidence before changing a default; ledger and recipe status can lag one another |
| GMM stack and OCSVM chunk256 | Ledger records GMM E-step stack slower on AMD; OCSVM chunk256 neutral there | A retry needs a changed mechanism or a clearly different support region |
| Candidate recipes | 75 entries; 71 marked unqualified, three block-eigh quarantined, one convergence-failure disabled | Recipes are coverage information, not a ready-to-run or successful queue |
| RBC canonical sorting | Baseline has quadratic rank counting; `codex/rbc-canonical-merge-20261005@d49bce462` contains a bounded merge candidate | I13 starts from that newer candidate, without claiming it is qualified or merged |
| Apple GEMM catalog | Resident screen has individual leads, but no universal geometry; square-NN candidates lost in that screen | F01 measures actual callers and adjacent shapes; no global GEMM replacement |
| Apple neural candidates | The ledger records FLASH, SSD MMA, and LM backward candidates as merged unmeasured opt-ins | F07–F10 are qualification and extension experiments |
| Apple scoped PCA and ARIMA | PCA fit and original Kalman variants have quality holds; compensated K3 has kernel-only evidence | Numerical/model repair and full caller quality precede timing or promotion |
| Apple output and gather experiments | Pinned output shifted cost into first read; GPU resample gather had mixed size-dependent results | Measure caller consumption; do not revive host runtime work or use dataset-specific switches |

Sources: [IDENTICAL ledger][idn-ledger], [candidate recipes][recipes], [Apple ledger][apple-ledger], [resident GEMM screen][resident-screen], and [Apple candidate queue][apple-queue]. Older roadmap statements about missing implementations or zero promotions are historical and are not carried forward as current facts.

## Experiments shared by AMD and NVIDIA IDENTICAL

### I01 Attribute the existing GEMM schedules

**Starting point:** Existing candidates in the [GEMM dispatcher][gemm]; schedule experiment, small implementation effort, high breadth.

**Experiment:** Compare one-page, group-slack2, group-slack8, actual-body tile counting, and packed64 individually with current dispatch. Use the existing campaign branch rather than recreating flags. Cover NN, NT, TN, ragged tails, narrow outputs, long contractions, and real forward/dWeight/dInput calls. After selecting useful singles, try only the interactions suggested by their resource measurements.

**Judge:** Whole LM/transformer steps and representative Gram callers, plus partial-plane bytes, fold time, spills, and route counts. Stop pursuing a profile if the real caller does not reach it or if saved fold traffic is offset by lost parallelism. Every schedule must retain the current contract.

### I02 Size and retain GEMM scratch by live use

**Starting point:** Group workspace reuse already exists in [GEMM][gemm]. This is a caller-coverage and memory-lifetime extension, not a new pooling proposal.

**Experiment:** Compare current high-water workspace provisioning with per-session live-range sizing and a bounded reusable scratch arena. Separate eliminating undersized-workspace fallbacks from reducing over-allocation. Exercise alternating large/small calls, concurrent independent sessions, exceptions, and model destruction; count allocations and hidden synchronization.

**Judge:** Public operation latency, peak memory, and allocation/wait counts. Schedule only. Reject reuse that aliases in-flight consumers, retains unbounded peak capacity, or merely moves allocation cost outside the scored interval without benefiting the stated public contract.

### I03 Batch independent GEMM jobs without changing contractions

**Starting point:** Follow-up to the multi-job idea in the [neural roadmap][neural-roadmap] and [training tensor operations][dev-tensors]. Medium effort, broad potential.

**Experiment:** Batch compatible Q/K/V, gate/up, or independent gradient products as separate grid jobs with shared staging where valid. Compare independent launches, job batching, and shared-operand staging separately. Keep each output's original contraction dimension, leaf boundaries, and fold; do not concatenate a backward contraction and assume it preserves bits.

**Judge:** End-to-end forward/backward timing, achieved device fill, staging bytes, and capacity. Reject if descriptor/staging overhead dominates small batches, scratch grows excessively, or synchronization serializes jobs that currently overlap.

### I04 Test a revised fold profile only after proving the current bottleneck

**Starting point:** A research hypothesis, not the default next GEMM task. The [candidate audit][candidate-audit] raised fold revisions, while the later [neural roadmap][neural-roadmap] found no established throughput mechanism for them.

**Experiment:** Only if I01–I03 show leaf dependency or fold traffic is limiting, compare a small set of analytically motivated leaf lengths and balanced-tree partitions. Define one profile shared by all four columns and preserve partition independence from output batch dimensions. Start with scalar/reference evaluation and adversarial conditioning before device code.

**Judge:** Public quality, same-version identity, numerical error, scratch, and whole-operation speed. Stop if the bottleneck is scheduling, if there is no implementable hardware mechanism, or if any caller cannot adopt the revised profile coherently. Native matrix instructions with undocumented internal arithmetic are not an identity proof.

### I05 Fuse only the arithmetic already adjacent to GEMM outputs

**Starting point:** [GEMM][gemm], [Llama forward][llama], and [transformer backward][transformer-backward]; mixed existing fusions and proposed extensions.

**Experiment:** Examine residual add, bias, output scaling, and activation consumers. Compare one selected epilogue with the separate pass while reproducing every intermediate rounding and flush seam. A register-resident value must be explicitly rounded where the materialized baseline rounded it. Evaluate several independent output tiles rather than one device-wide serial owner.

**Judge:** Full layer time and bytes written/read, with intermediate and final identity. Reject a fusion that increases register pressure enough to slow the dominant product. Any intended arithmetic reassociation becomes a separate profile experiment, not an accidental fusion side effect.

### I06 Reuse attention tiles while preserving canonical row arithmetic

**Starting point:** [Attention implementation][attention] and prior neural experiments; proposed extension, medium/high effort.

**Experiment:** Compare current query-tile staging against sharing K/V loads across compatible query heads or query tiles. Keep each query's score, max, denominator, output, and gradient order fixed. Include GQA with real KV sharing, full attention, causal tails, mask boundaries, and small batches. First inspect the existing packed/stash and cooperative dK/dV rejection evidence.

**Judge:** Attention plus layer time, scratch, load reuse, and register pressure. Stop if extra barriers or replicated accumulators erase the gain. Changing to an online softmax recurrence is a numerical-profile project with forward and backward witnesses, not automatically a schedule optimization.

### I07 Choose stored versus recomputed attention state by a memory model

**Starting point:** [Attention][attention], [transformer backward][transformer-backward], and existing attention evidence. Proposed scoped revisit.

**Experiment:** Compare retaining canonical probabilities or exponent state with recomputing it from retained inputs using exactly the original evaluation. Choose experiments from a bytes-versus-recomputation model, not a board sequence length. Keep dropout/RNG coordinates and checkpoint behavior unchanged; include memory pressure and training batch capacity.

**Judge:** Complete train step, including forward storage, backward recomputation, and allocation. Reject if a backward-only win makes the full step slower, changes refusal timing/state, or reduces usable batch capacity. Historical aliasing and cooperative-gradient losses are controls to understand, not candidates to blindly restore.

### I08 Attribute and extend Mamba tile reuse

**Starting point:** [SSD][ssd] contains `IDN_M2_SSD_TILES`; [Mamba SISO][siso] has related tiling. Existing implementation plus follow-up, schedule class.

**Experiment:** Isolate existing tiling from subsequent proposals to retain `G⊙L`, reused decay values, or shared state tiles. Measure separate variants for reused intermediates versus avoided recomputation. Include short and long sequences, multiple state widths, causal tails, and backward consumers before dropping unused triangular work.

**Judge:** Mamba/Samba operation time, memory bytes, redundant exponential calls, and occupancy. Stop if retained intermediates cost more bandwidth than recomputation or if a forward-only dead-value assumption breaks backward or trace consumers.

### I09 Parallelize recurrences with explicit versioned boundaries

**Starting point:** [Selective scan][selective-scan], [Mamba backward][mamba-backward], [sequence recurrent code][recurrent]; existing partial work and higher-risk extensions.

**Experiment:** First qualify already token-parallel convolution and scan routes. Separately propose fixed absolute-position chunking for remaining long recurrences and shared suffix reductions for repeated backward suffix work. The chunk/merge order is a numerical profile, shared with host replay, decode, checkpoint, and backward. Do not reuse a FAST chunk rule that depends on total sequence length without addressing prefix consistency.

**Judge:** Train and inference scaling, gradients, prefix/decode agreement, and same-version bits. Stop if chunk overhead dominates short sequences or if the revised recurrence changes model quality beyond the allowed contract.

### I10 Finish training steps with one well-defined status reduction

**Starting point:** [Byte LM][byte-lm], [optimizer checks][optimizer], and existing scan-fusion work. Schedule/lifetime extension, medium effort.

**Experiment:** Compare existing separate state scans with per-block integer status contributions emitted while updated values are already live. Read final loss/status together where dependencies allow. Preserve the first offending field/index and whether state is committed or rolled back; explicitly compare normal, nonfinite-input, nonfinite-gradient, and update-failure paths.

**Judge:** Full step latency and bytes scanned, not just scan-kernel time. Reject any variant that reports errors one step later, commits a failed update, or suppresses a required check. A status reduction cannot replace a validation that must happen before unsafe memory access.

### I11 Sort embedding updates by canonical token order

**Starting point:** [Embedding kernels][embedding] and [embedding sort][embedding-sort]; roadmap extension.

**Experiment:** Compare the current scan/bitonic paths with stable radix grouping on `(token_id, original_position)` and processing only touched rows. Preserve each embedding row's original accumulation order, zero initialization for untouched outputs, duplicate multiplicity, and padding semantics. Test skewed vocabularies, all-identical IDs, mostly unique IDs, and alternating vocab sizes.

**Judge:** Full backward time, sorting workspace, and dense-output initialization cost. Reject when sorting costs exceed the avoided scan or when a sparse intermediate simply postpones a required dense materialization. This is not permission to use unordered floating atomics.

### I12 Reduce linear solver passes without changing accepted iterates

**Starting point:** [SGD][sgd], [linear device operations][linear-device], and [coordinate descent][cd]; several related optimizations already exist.

**Experiment:** Isolate batch-step fusion, parallel independent one-vs-rest classes, and speculative line-search evaluations from each trial parameter vector. Preserve the first acceptable step and convergence decision. Treat Gram-vs-row-sweep and shared-fold CV statistics as separate numerical-profile variants; account for all preparation cost.

**Judge:** Whole fit, epochs/iterations, validation metric, coefficients/predictions, and bytes through X. Stop if a faster fit merely does less optimization, changes the accepted line-search candidate, or suffers ill-conditioned Gram cancellation. Do not include block-local SGD averaging without a separate algorithm decision.

### I13 Bound canonical neighborhood sorting on dense graphs

**Starting point:** [RBC canonicalization][rbc] is quadratic at the baseline. The newer `codex/rbc-canonical-merge-20261005@d49bce462` implements stable bottom-up merges. High-priority qualification plus follow-up.

**Experiment:** Compare the bounded candidate with rank counting on safe small controls; use operation counts and bounded inputs instead of forcing pathological quadratic runs. Then compare compact degree buckets with one global merge schedule, so a single long row does not make every short row execute all merge levels. A bucketed route is proposed, not present in that branch.

**Judge:** DBSCAN/radius-neighbor time, canonical CSR words, duplicates, empty rows, degree skew, peak scratch, and maximum-degree readback overhead. Reject if bucketing costs dominate or if ties, row boundaries, or multiplicity change. This changes ordering work, never graph membership.

### I14 Make graph convergence polling cheaper without skipping the stopping state

**Starting point:** [DBSCAN][dbscan], [HDBSCAN switches][hdb-switches], and existing chunk/split recipes. Existing candidates plus controlled extensions.

**Experiment:** Compare connected-component chunk sizes and edge partitions individually. Retain the first converged state using device flags/masks when multiple rounds are queued. Test long chains, disconnected components, dense graphs, and highly skewed degree. For HDBSCAN, separately examine MST tie handling, condensation, prediction, and soft-clustering paths rather than treating them as one optimization.

**Judge:** Full fit/predict time, rounds executed, edge visits, and canonical outputs. Stop if an optimization observes only a later state, changes equal-weight choices, or adds enough inactive work to negate fewer waits. Approximate graph construction is outside this schedule experiment.

### I15 Stream exact neighbor selection instead of materializing all distances

**Starting point:** [Brute-force neighbors][knn] and [certified candidate machinery][certified-knn]; existing pieces, integration/qualification work.

**Experiment:** Compare fused tile-local selection, hierarchical exact top-k merging, and certified coarse candidates followed by canonical exact rescoring. Keep the total `(distance, index)` order and distance arithmetic. Certification must prove excluded points cannot enter the answer, with full exact fallback whenever the bound is inconclusive. Include near ties, duplicates, high-dimensional cancellation, and large k.

**Judge:** Public query latency, distance bytes avoided, certification/fallback rate, and exact neighbors/distances. Reject if almost every query falls back, temporary candidate lists exceed the distance-memory savings, or a loose numerical bound silently becomes approximate search.

### I16 Balance IVF work by list length while retaining query semantics

**Starting point:** [IVF-flat search][ivf-search] has grouped/staged concepts and device-side support; scoped scheduling investigation.

**Experiment:** Compare one fixed query/list assignment with chunked long-list tasks and compact batches of short lists. Keep nprobe, centroids, selected lists, distance arithmetic, and final stable top-k identical. Separate staging from task balancing. Test empty lists, one giant list, uniform occupancy, tail dimensions, and batch-size changes.

**Judge:** Whole search time, task imbalance, scratch, and output identity. Stop if task descriptors dominate small queries or if result merging becomes the new bottleneck. Changing index training, nprobe, or recall is a different algorithm experiment, not a scheduling win.

### I17 Attribute batched tree frontiers and retain their device state

**Starting point:** [Depthwise/Lossguide][gbdt-depth] already contains default-enabled `LG_EXACT_ID` and partition inheritance. Existing work needs attribution; further residency is a hypothesis.

**Experiment:** Compare those mechanisms individually using their rollback controls, then examine whether frontier statistics, model fragments, and partition bookkeeping can stay resident through more of a tree. Bound capacity from live leaves, features, statistics, and bytes. Include symmetric/depthwise/lossguide cases, skewed trees, categorical inputs, and multiple fits.

**Judge:** Whole boosting fit, wait count, scratch high-water mark, split sequence, leaf values, and model predictions. Reject if larger frontiers inflate memory or speculative work enough to lose, or if categorical arena limits fail on a different tree shape.

### I18 Use histogram algebra only where exactness is established

**Starting point:** [Forest builder][forest-builder], [random forest][rf], and tree roadmap. Proposed extension, medium/high effort.

**Experiment:** Consider sibling histogram subtraction only for accumulators proven exact and bounded, with compatible feature sampling. Compare computing both children with computing one plus exact parent-minus-child. Separately compare a bounded multi-tree frontier. Weighted floating gradient/statistic histograms must not be assumed subtractable with the same bits; they need a distinct profile and quality analysis.

**Judge:** Histogram work, complete fit, memory, overflow witnesses, sampled-feature semantics, and canonical tie decisions. Stop if extra parent retention overwhelms bandwidth savings or exactness cannot be proved over the supported data range.

### I19 Reuse deterministic radix infrastructure across data preparation

**Starting point:** [Preparation radix][prep-radix], [segmented sort][segmented-sort], and existing quantile/select routes. Qualification plus generalization.

**Experiment:** Compare digit widths and segmented task layouts derived from shared-memory footprint and key width. Distinguish full sort from selection-only callers. Exercise quantiles, categorical encoders, bootstrap order statistics, and embedding grouping with stable secondary keys where required. Include signed zero, NaNs under the public policy, repeated keys, ragged segments, and tails.

**Judge:** End-to-end transform/fit, passes, scratch, and exact outputs/indices. Reject a shared implementation that forces every caller to materialize unnecessary full permutations or gives a broad name to a narrower supported key contract.

### I20 Bound KDE tiles and reuse stable log-sum-exp partials

**Starting point:** [KDE kernel density][kde] and [chunked validation][kde-check]; chunked fold behavior already exists. This is scheduling/working-set work first.

**Experiment:** Compare query-by-reference tile shapes and reusable partial buffers under the existing logical fold. Profile distance evaluation, kernel transform, log-sum-exp, and output separately. Test all supported kernels/metrics, extreme bandwidths, sparse effective contributions, long reference sets, and small query batches.

**Judge:** `score_samples` latency, peak distance/partial storage, and same-version log densities. Reject if staging saves memory but causes excessive launches or repeated exponentials. A different stable pair-combine reduction is a separate profile experiment with its own oracle, not evidence from a mathematically equivalent formula.

### I21 Fuse GMM work within components before stacking more components

**Starting point:** [GMM E-step][gmm-estep] and [M-step][gmm-mstep]; existing fused candidates and a documented AMD loss for E-step stacking.

**Experiment:** Test bounded component batches and reuse of centered tiles/precision data, separately from fused Cholesky, triangular solve, and log determinant. Keep the baseline's responsibility, covariance, and lower-bound arithmetic where claiming a schedule change. Cover component counts, covariance types, ill conditioning, empty/effectively empty components, and sample-weight extremes.

**Judge:** Whole EM fit, iteration count, likelihood, fitted covariance state, memory, and bits. Stop if register or scratch pressure recreates the stack regression. A larger component batch is not inherently better, and a cheaper iteration is not a win if it takes more iterations.

### I22 Replace repeated dense linear algebra work with bounded blocked work

**Starting point:** [Decomposition device operations][decomp], [TSQR][tsqr], [Cholesky][cholesky], and [triangular solves][trsm]. Mixed schedule and profile experiments; high effort.

**Experiment:** Investigate strip-based trailing updates, TSQR caller coverage, different TSQR combine arity, and reuse of identical factorizations. Keep them as separate arms. A skipped structural-zero operation still needs signed-zero/nonfinite analysis; a changed QR tree is a profile revision. Reuse the existing Householder and pinned arithmetic infrastructure rather than reviving quarantined block Jacobi.

**Judge:** Solve/reconstruction residuals, rank decisions, downstream estimator quality, identity, and total factorization/solve time. Stop if a kernel-only gain adds copies or changes pivot/rank behavior without a complete new contract.

### I23 Parallelize independent time-series work before changing filters

**Starting point:** [ARIMA batching][arima-batch], [sequence execution][sequence-exec], and existing Theta/ETS/GARCH work. Mixed existing candidates and extensions.

**Experiment:** First batch independent series, order candidates, or optimizer trial parameters while preserving selection order, tie-breaking, convergence, and failure handling. Then consider a separately versioned affine scan for recurrences where the model admits it. Do not assume a Kalman covariance update is safely replaceable by a rank-one approximation.

**Judge:** Complete fit/order selection/forecast time, selected models, likelihood/criteria, forecast quality, same-version bits, and scratch. Stop if batching wastes most work on inactive candidates, alters early-stop semantics, or changes initialization/differencing to improve a timing.

### I24 Move remaining metric and preprocessing work into bounded device passes

**Starting point:** [Runtime route debt][host-baseline], [metric implementation][metrics], and [preparation operations][prep-device]. Prioritize measured hot paths, not baseline-row count.

**Experiment:** Group related counts, finite checks, scaling statistics, and output transformations only when they share input passes. Compare resident Mojo device outputs with the current public route, preserving error order and observable fitted attributes. Measure upload, compute, download, and output consumption separately so a tiny kernel does not conceal a full-array round trip.

**Judge:** Public call latency, passes, bytes, quality/identity, and no Python data computation. Stop when fusion increases scratch or synchronization more than it saves. Removing runtime Python debt and demonstrating a performance improvement are separate conclusions.

## AMD specific IDENTICAL experiments

AMD cards refine the shared proposals. Their acceptance still requires the NVIDIA comparison and four-column identity. The official [HIP performance guide](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html) supports investigating memory access, register pressure, LDS use, and occupancy; it does not establish that any particular mojolearn kernel is limited by those resources.

### A01 Qualify smaller MFMA bodies across the stepped dispatch region

**Starting point:** `IDN_GEMM_MFMA16` and `IDN_GEMM_AMD_BAND_MFMA` already exist in [GEMM][gemm]. High-priority existing-candidate experiment.

**Experiment:** Isolate small-body routing from group count, then cover the transition between scalar, packed, small MFMA, and full MFMA plans. Sample neighboring output sizes and contraction leaves in all orientations. Retain the established arithmetic primitive; do not substitute a wider native contraction merely because it has higher peak throughput.

**Judge:** Real LM/Gram caller time, reached plans, active workgroups, VGPR/spill counts, and identity. Reject any region justified only by one historical shape. The result should explain a general fill/resource rule or leave the current route in place.

### A02 Compare one LDS page with deeper staging only where resources permit

**Starting point:** `MOJOLEARN_GEMM_ONE_PAGE` exists; [GEMM][gemm] supports staged bodies. Small effort, scheduling only.

**Experiment:** Compare one versus two pages and a bounded staging-depth variant as independent arms. Predict resource occupancy using actual compiled registers and LDS, then verify it on the device. Include long enough work to amortize staging and smaller calls where extra barriers matter.

**Judge:** GEMM and full caller time, occupancy, memory stalls, and spills. Stop if registers already limit residency so removing one page admits no extra workgroup, or if lost prefetch outweighs additional residency. Higher occupancy alone is not the acceptance metric.

### A03 Redesign operand staging around measured LDS access conflicts

**Starting point:** [GEMM loaders][gemm] and [SSD/SISO staging][ssd]; new scoped hypothesis, medium effort.

**Experiment:** Compare current staging with padding or an invertible swizzle derived from lane addresses and the target's documented memory organization. Separate global-load coalescing from LDS layout so their effects are attributable. Keep logical operands, accumulation order, and tail masks identical; include transposed and ragged products.

**Judge:** Relevant bank-conflict/stall counters, staging bytes, LDS footprint, and full operation time. Reject if reduced conflicts consume enough extra LDS or address arithmetic to lose. Use supported Mojo operations; backend assembly surgery is out of scope.

### A04 Map two logical groups onto a wave only with independent membership

**Starting point:** [RBC canonicalization][rbc] documents a failed narrow-ballot assumption; [sequence execution][sequence-exec] and selection kernels offer other candidates.

**Experiment:** Where a kernel uses logical 32-element groups, compare explicit subwave index/mask handling with a full-wave mapping. Start with supported shuffle/reduction primitives and local fixtures that isolate each group's membership. Do not reuse the unsupported RBC ballot form or assume NVIDIA warp semantics transfer to AMD.

**Judge:** Exact group outputs, empty/tail behavior, wave utilization, and end-to-end speed. Park any form the installed compiler cannot express. Reject if cross-group values leak or if extra masking erases utilization gains.

### A05 Reduce VGPR live ranges in reduction and attention bodies

**Starting point:** [GEMM][gemm], [attention][attention], and [GMM][gmm-estep]; proposed resource experiment.

**Experiment:** Split address calculations from accumulator lifetimes, shorten temporary vector live ranges, and compare compact versus wider per-thread output tiles. Change one body at a time. Retain the exact accumulator sequence; do not exchange a spill for a different reduction. Record actual compiled VGPR and local-memory usage rather than inferring pressure from source length.

**Judge:** Caller time and measured spills/occupancy. Stop if compiler allocation is unchanged or instruction overhead exceeds saved spill traffic. This is particularly important before retrying component stacking that already lost on AMD.

### A06 Test fused exact low-dimensional neighbor selection on AMD

**Starting point:** Existing fused-select machinery in [neighbors][knn] and [kernel matrix][kernel-matrix]; an AMD qualification of I15.

**Experiment:** Compare distance materialization with fused distance/top-k over a range of dimensions and k values. Derive tile capacity from register/LDS limits and candidate count. Include slightly larger dimensions than the intended support region, duplicate-heavy inputs, and long query batches.

**Judge:** Exact ordered neighbors and distances, public query time, materialized bytes, and spill counts. Reject a selector that only wins for a single dimension or becomes bandwidth-heavy when merging local candidates. Do not advertise the old board ratio as the expected gain.

### A07 Schedule independent graph and histogram tiles to reduce the long tail

**Starting point:** [RBC][rbc], [DBSCAN][dbscan], and [forest builder][forest-builder]; proposed extension after I13/I18.

**Experiment:** Compare static equal-row assignment with compact tasks bounded by edge count or histogram work. Use independently writable outputs and explicit canonical reduction where needed. Buckets must represent actual work, not dataset identity. Test one heavy segment among many light segments and uniformly dense inputs.

**Judge:** Tail workgroup duration, load balance, descriptor overhead, complete fit, and identity. Reject a global queue or persistent-loop design that depends on unsupported synchronization or sacrifices progress. A simple multi-launch task schedule is preferable if it captures the same benefit.

### A08 Revisit KMeans accumulator geometry around the retained 256-row candidate

**Starting point:** The [IDENTICAL ledger][idn-ledger] marks accumulate-256 KEEP but not promoted. Qualification and interaction work, not a new discovery.

**Experiment:** Compare current baseline, the retained candidate, and at most one resource-motivated alternative. Then isolate interaction with nearest-center assignment and convergence polling. Include few/many centers, imbalanced clusters, broad feature widths, and multiple initialization seeds; audit exact integer-sum bounds where used.

**Judge:** Whole fit, iterations, centers/labels, scratch, bandwidth, and both vendors. Stop if a kernel benefit disappears in fit or only appears under the all-off baseline. Check the current promotion state before scheduling duplicate work.

## NVIDIA specific IDENTICAL experiments

These are architecture-scoped schedules, not an alternative NVIDIA numerical contract. The official [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) motivates examining transfers, coalescing, register use, shared-memory staging, and occupancy. Availability in CUDA does not establish availability through this repository's Mojo toolchain.

### N01 Attribute packed64 and actual-body tile counting

**Starting point:** Existing packed64/body-tiles campaign profiles and `IDN_GEMM_NV_STEP_KPACK` in [GEMM][gemm]. High priority, small effort.

**Experiment:** Compare each mechanism independently across stepped-down calls, then their interaction. Count blocks using the body that actually executes and report allocated partial planes. Cover forward, dWeight, and dInput independently; a transpose label does not imply the same resource bottleneck.

**Judge:** Public layer/step time, workspace, registers, and actual plan selection. Reject if smaller tiles create excessive duplicate loads or if additional blocks merely increase partial reduction cost. Preserve leaf and fold semantics across plan transitions.

### N02 Specialize fold stack capacity from the actual logical tree

**Starting point:** The generalized FS4 route already exists in [GEMM][gemm]. Extend evidence before adding more stack variants.

**Experiment:** Compare the existing compact stack with the full stack over contraction/group combinations requiring different depths. Only consider another specialization when a statically proven tree-depth bound changes storage or spills. Include odd leaf counts, maximum supported depths, tails, and group boundaries.

**Judge:** Compiler local-memory usage, spills, kernel and whole-step time, and exact carry behavior. Reject excessive template/code growth for a negligible caller gain. Never restore an exact-shape dispatch rule to recover an old isolated win.

### N03 Overlap operand loading with computation using supported primitives

**Starting point:** [GEMM][gemm] staging and [attention][attention]; conditional research proposal.

**Experiment:** Compare supported asynchronous global-to-shared copy with existing loads only if Mojo exposes the required alignment, completion, and barrier semantics. Keep a plain supported-load control and test aligned plus ragged paths. Measure whether reducing register-mediated copies helps the actual limiting resource.

**Judge:** Correct completion, no stale reads, stage overlap, register demand, and public operation time. Park the proposal if the installed toolchain lacks support. No hand-written PTX patching, compiler-output edits, or unsupported build mode is part of this idea.

### N04 Separate transpose staging choices by access pattern

**Starting point:** [GEMM][gemm] already handles orientations through strides. New comparison, not a proposal to add unconditional transpose copies.

**Experiment:** Compare current outer-contiguous and gather staging with a layout designed for each access pattern. Hold arithmetic fixed and vary vector width only when alignment/tail rules justify it. Include small rectangles, tall matrices, and long-K products; time any explicit packing or retained layout construction.

**Judge:** Coalescing efficiency, redundant loads, staging instructions, and full caller cost. Reject if a faster inner loop depends on an uncounted transpose or packing pass. Persistent packing requires mutation/version invalidation and a separately reported reuse contract.

### N05 Batch compatible launches only where current traces show launch gaps

**Starting point:** [Training tensor operations][dev-tensors], [Llama][llama], and I03. Medium effort, workload-dependent.

**Experiment:** Compare a small grouped launch over independent operations with current launches. If supported graph/replay facilities exist, evaluate them as a separate mechanism with fixed addresses and correct changing-input semantics; otherwise leave that arm parked. Do not hide a large serial loop in a persistent kernel.

**Judge:** Whole-operation latency, device idle gaps, cold preparation, replay amortization, and state/lifetime correctness. Reject if GEMM already saturates the device or if launch savings require a new public usage contract without showing that contract separately.

### N06 Reevaluate attention gradient tile geometry with its old rejection as control

**Starting point:** [Attention backward][attention] and [cooperative dK/dV rejection][attention-rejection]. That rejection was measured on Apple IDENTICAL, not NVIDIA; it is a work-decomposition warning, not evidence of a NVIDIA loss. A revised experiment needs a specific resource hypothesis.

**Experiment:** Compare compact gradient tiles and recomputation/stash choices under the canonical gradient order. Before coding, identify why the earlier cooperative form lost: resource pressure, work duplication, or insufficient independent blocks. Include GQA, sequence tails, several head widths, and complete backward consumers.

**Judge:** Complete train step, scratch, spill/load counters, and all gradients. Stop if the proposed mechanism does not address the earlier loss. A better attention microbenchmark alone does not justify replacing a faster complete training route.

### N07 Stream histogram partitions with bounded shared-memory replication

**Starting point:** [Forest builder][forest-builder] and [GBDT depthwise][gbdt-depth]; proposed resource comparison related to I18.

**Experiment:** Compare a few per-warp/private histogram layouts and bounded feature chunks, retaining the existing exact accumulator representation and final ordering. Derive widths from bin count, statistics, shared memory, and concurrent blocks. Include high/low cardinality, weighted inputs, and deep sparse frontiers.

**Judge:** Full tree fit, histogram traffic, atomic contention where integer atomics are valid, memory, and identity. Reject if initialization/merge work or shared-memory pressure exceeds the reduced contention. Floating atomic accumulation is not introduced into IDENTICAL.

### N08 Fuse distance evaluation and threshold filtering for exact graph work

**Starting point:** [Neighbor distance paths][knn], [RBC][rbc], and [DBSCAN][dbscan]; proposed memory-traffic extension.

**Experiment:** Compare writing full distance tiles with immediately consuming canonical distances into exact threshold decisions or top-k candidates. Keep count/scan/fill consistency, public epsilon comparison, signed-zero treatment, and final canonical graph ordering. Test thresholds at representable neighbors of actual distances and highly dense graphs.

**Judge:** Full graph build/fit, bytes avoided, count/fill passes, peak memory, and exact CSR/labels. Reject if the fused path evaluates distances twice at greater total cost or if numerical shortcut bounds drop boundary edges.

## Apple FAST experiments

These 20 cards use FAST quality gates, not IDENTICAL bit gates. The [Apple Metal guidance](https://developer.apple.com/videos/play/wwdc2020/10631/) explains SIMD-group/threadgroup distinctions and the need for correct synchronization. It does not establish that a Metal capability is exposed by the installed Mojo compiler. Existing source, actual dispatch, and public caller quality remain the admission requirements.

### F01 Select GEMM geometry from actual caller behavior

**Starting point:** G1–G10 exist in the [Apple queue][apple-queue]; the [resident screen][resident-screen] has individual leads and square-NN losses. High-priority qualification.

**Experiment:** Start with the strongest relevant direct/staged candidates on tall/narrow projection, low-width NT, Gram, and odd dimensions. Compare against each real caller's current kernel, then cover neighboring shapes. Separate cold public calls from resident-input calls. Do not rerun the full ten-variant catalog on every estimator.

**Judge:** Estimator quality, actual route counters, completion plus first read, scratch, and total time. Reject a global selector from one matrix screen. A deterministic cost rule must explain where the candidate wins and where incumbent geometry remains preferable.

### F02 Share GEMM infrastructure without discarding specialized operations

**Starting point:** [GEMM reach audit][gemm-reach] and [scoped dispatch][scoped-dispatch]; existing adapters and further integration work.

**Experiment:** First compare the shared candidate at both unfused SDK entrances. Then independently adapt decomp/PCA, LU subtract, Cholesky triangular subtract, and active batched MCD products, retaining strides, epilogues, split policy, compensation, and batching. Count which products actually reach each adapter.

**Judge:** Caller quality and full time including any additional pass. Stop if a generic interface removes a fused epilogue or turns one batched operation into many launches. Similar tile names or imports are not evidence of equivalent arithmetic or runtime coverage.

### F03 Retain session buffers where the real caller still pays setup cost

**Starting point:** C1–C4 lifecycle probes in the [Apple queue][apple-queue]; some behavior already overlaps pooled VAR. Proposed real-caller extension.

**Experiment:** Choose an actual estimator with measured allocation/context overhead and compare bounded retained input, scratch, and output slots with its incumbent. Invalidate cached state on mutable model attributes or inputs; test reuse, alternating sizes, two independent models, exceptions, and teardown. Compare cold and repeated calls separately.

**Judge:** Public call through first read, peak retained memory, and semantic behavior. Reject no-op probe timing as the baseline and reject pinned output if its cost merely reappears when the caller consumes it.

### F04 Pair independent completions without delaying required decisions

**Starting point:** C2 `wait_pair` and grouped readback ideas in the [Apple queue][apple-queue]. Medium effort, lifecycle-focused.

**Experiment:** Find two real independent operations whose results are not needed between launches. Compare separate waits/readbacks with one completion and a packed result transfer. Keep both source buffers alive and preserve per-operation errors; include one failure, empty outputs, and asymmetric operation sizes.

**Judge:** Caller wall time and wait/transfer counts. Stop if dependency analysis shows the calls are not independent, packing adds a full extra copy, or the slower operation unnecessarily delays a latency-sensitive result. Unified memory does not remove synchronization or ownership obligations.

### F05 Qualify narrow softmax GEMM inside the full optimizer

**Starting point:** [Softmax narrow candidate][softmax-narrow] and its source-ready state in the queue. Existing candidate, high breadth if qualified.

**Experiment:** Compare the small direct matrix candidate with the current multiclass/softmax product at actual forward, gradient, and line-search callers. Include few and many classes, imbalanced targets, extreme logits, regularization, and ill-conditioned features. Keep optimizer settings and stopping policy fixed.

**Judge:** Whole fit time, log loss/accuracy or the applicable board metric, iteration count, margins, and fitted state. Stop if a matrix lead disappears under optimizer iterations or changes solver stability. Matrix error/hash checks alone cannot certify this candidate.

### F06 Revisit MCD batching after separating neutral geometry from quality failures

**Starting point:** [Batched MCD][mcd], the G1/ordered-covariance neutral records, and earlier quality holds in the [Apple ledger][apple-ledger].

**Experiment:** Compare active-candidate compaction, bounded candidate batches, and covariance reuse as separate mechanisms. Reuse a valid incumbent covariance/rank path. Test support selection, reweighting, contamination levels, singular and nearly singular data, and prediction/score consumers.

**Judge:** Complete robust-covariance/elliptic-envelope time, fitted location/covariance/support, downstream quality, and peak memory. Stop if compaction costs exceed inactive work or fitted-state quality worsens. Do not present another G1-only run as a new mechanism after the neutral result.

### F07 Qualify existing FLASH and GQA reuse on meaningful attention shapes

**Starting point:** [Apple attention][apple-attention] has opt-in `AFN_ATTN_FLASH` and `AFN_ATTN_GQA_TILE`; the ledger records merged unmeasured work.

**Experiment:** Compare FLASH alone first, then GQA tile sharing where multiple query heads actually share K/V. Include short/long contexts, causal tails, masks, real GQA ratios, and full model outputs. Forward candidates need their own inference quality; training requires separate backward qualification rather than assuming it exists.

**Judge:** Model quality, memory, and full forward time. Stop if attention is no longer the dominant cost, if the GQA fixture has no shared heads and therefore tests no reuse, or if softmax rescaling creates a real quality loss. Different FAST bits are acceptable.

### F08 Qualify LM backward fusion and parameter views independently

**Starting point:** [Byte LM FAST controls][byte-lm-fast] and [LM A/B brief][apple-lm-brief] contain existing opt-ins; verify the actual Apple FAST binding is available first.

**Experiment:** Compare backward wait removal, existing fused backward routes, and parameter/gradient views independently before a bundle. Exercise handle swaps after optimizer steps, repeated training, checkpoint/resume, and recovery from a failed update. Do not score a forward row for a flag that only changes backward.

**Judge:** Whole train-step time, loss/gradient behavior, short held-out learning curves, memory traffic, and rollback semantics. Stop on stale views, hidden IDENTICAL fallback, missing binding reach, or quality loss. Avoid a forward-only claim for a training improvement.

### F09 Compare fused cross entropy with a memory-bounded LM head

**Starting point:** [Byte LM FAST][byte-lm-fast] and [chunked head design][chunked-head]; existing mechanisms with distinct numerical contracts.

**Experiment:** Compare full logits plus fused CE with a bounded vocabulary-tile head that preserves stable normalization and supplies required gradients. Include large vocabularies, low-probability targets, extreme logits, tail tokens, and both loss-only and logits-returning APIs. Public logits APIs must still materialize requested outputs.

**Judge:** Full train step, peak memory, held-out loss/perplexity, gradients, and output semantics. Stop if repeated GEMMs outweigh memory savings or if an apparent win comes from omitting requested logits. Do not require equality with IDENTICAL as the FAST quality criterion.

### F10 Qualify SSD MMA and isolate Mamba fusion interactions

**Starting point:** [Apple SSD MMA][apple-ssd] and [Mamba controls][mamba-controls] already exist.

**Experiment:** Compare SSD MMA, token-parallel convolution/input fusion, and buffer arena changes separately. Add combinations only after their individual reach and quality are known. For Mamba-1 chunk scan, examine prefix/decode behavior and long-sequence drift; for Mamba-3, keep angle/state changes separate from elementwise fusion.

**Judge:** Full Mamba/Samba forward and supported training quality, memory, and latency over lengths/state widths. Stop if a local matrix win expands intermediates excessively, a recurrence loses quality, or a combination merely reruns the same effective route.

### F11 Use TSQR and compensated products where decomposition quality needs them

**Starting point:** [Decomposition][decomp], [TSQR][tsqr], and the scoped PCA HOLD in the [inline outcomes][gemm-outcomes]. Higher effort, quality first.

**Experiment:** Compare valid TSQR/range-finder paths and selectively compensated products against the current FAST caller. Separate projection/transform from covariance construction and fit; a transform gain does not admit a fit change. Include nearly collinear data, low rank, clustered singular values, and downstream LLE/PCA reconstruction.

**Judge:** Reconstruction, singular values/rank/noise estimates, embedding quality where applicable, and complete operation time. Stop if the selected path masks an existing failure, uses looser quality thresholds, or adds compensation everywhere when only one product needs it.

### F12 Keep more tree search and leaf estimation resident

**Starting point:** [Apple tree planning][apple-tree-plan], [GBDT depthwise][gbdt-depth], and [ordered tree search][ordered-tree]. Existing broad work; proposed remaining-gap experiments.

**Experiment:** Profile the current tree first, then compare device-resident candidate scores, winner selection, leaf-estimation inputs, or partition reuse individually. For ordered boosting, preserve document-ID-keyed randomness while experimenting with coalesced permutation-order storage. Cover categorical, ranking, depthwise, and lossguide workloads.

**Judge:** Whole fit, quality metric, model state, waits, and peak categorical scratch. Stop if more speculative candidates or wider state cause memory pressure, or if a speed gain depends on changing boosting iterations, sampling semantics, or quality.

### F13 Compare forest prediction layouts under real traversal divergence

**Starting point:** [Forest inference][forest-inference] and [TreeSHAP device code][shap]; proposed scheduling and storage experiments.

**Experiment:** Compare bounded groups of trees/rows, compact node layouts, and reused split metadata. Consider SHAP path work separately, with deep forests and repeated queries. Include shallow/deep, balanced/skewed, multiclass, and sparse-feature-access models; count staging and any model conversion cost.

**Judge:** Predict/SHAP latency, memory bandwidth, accuracy or attribution checks, and cold versus retained-model cost. Stop if larger groups cause divergent traversal or register spills. Historical SHAP pipeline losses require a new mechanism, not simply another pipeline flag.

### F14 Make candidate selection cheaper without changing the advertised neighbor task

**Starting point:** [Neighbors][knn], [IVF][ivf-search], and Apple ANN paths. Medium effort, split exact and approximate contracts.

**Experiment:** For exact APIs, compare streaming selection and certified rescoring. For an already approximate API, compare candidate buffering/compaction at the same requested search settings, reporting recall and downstream metric. Use distinct experiments and labels; an exact method cannot quietly become approximate because FAST permits different bits.

**Judge:** Query/index-build time as appropriate, memory, exactness or recall under the declared API, and public quality. Stop if omitted candidates lower quality, if candidate growth dominates, or if index construction cost is hidden in a search-only comparison.

### F15 Reduce clustering work while preserving stopping and quality

**Starting point:** [MiniBatchKMeans][minibatch] and [cluster device operations][cluster-device]; related existing candidates and defaults.

**Experiment:** Separate label assignment, center accumulation, stopping scans, and scratch reuse. Compare batched independent starts only with unchanged seeds and selection policy. For DBSCAN/HDBSCAN, profile graph build and later graph stages separately; an improved distance kernel may not address the dominant work.

**Judge:** Whole fit, iterations, inertia/silhouette or applicable clustering quality, memory, and repeat behavior. Stop if extra batched work overwhelms fewer launches, a nominally faster result stops early, or a tiny kernel improvement is noise in the public operation.

### F16 Repair and integrate compensated Kalman candidates before timing them

**Starting point:** [Scalar compensated ARIMA candidate][arima-scalar] has kernel-only evidence; original scalar/full-Gaussian forms have holds in the [queue][apple-queue]. High risk, full quality prerequisite.

**Experiment:** Integrate the compensated candidate into an actual supported AutoARIMA route while preserving initialization, differencing, covariance model, optimizer inputs, and failure handling. Compare likelihoods and gradients to independent references, then selected orders and forecasts. A blocked full-covariance scan is a separate mathematical experiment after its oracle passes.

**Judge:** Forecast quality, optimizer stability, model selection, full fit time, and memory. Stop on structural covariance/gradient errors; do not waive them because FAST permits different bits. Faster likelihood evaluation alone is not a complete estimator result.

### F17 Batch independent time-series candidates with bounded state

**Starting point:** [ARIMA batching][arima-batch] and [sequence execution][sequence-exec]; safer scheduling companion to F16.

**Experiment:** Compare batching series/order candidates or speculative optimizer points with current launches. Compact inactive candidates on device only when the saved work can amortize compaction. Keep the number of requested orders, search policy, initialization, and forecast horizon fixed. Include small panels and heterogeneous series lengths.

**Judge:** Complete fit/search/forecast latency, selected model quality, wasted inactive work, and state memory. Stop if padding or extra evaluations dominate or if a supposedly parallel VAR/ETS route is actually one serial block over runtime-sized work.

### F18 Reuse resident preparation and scoring without hiding data movement

**Starting point:** [KDE][kde], [preparation][prep-device], and the no-op/lifecycle cautions in the [Apple queue][apple-queue]. Proposed caller-specific work.

**Experiment:** Select a repeated transform or score path with measurable copy/setup cost. Compare retained immutable inputs and bounded scratch, then optionally fuse a same-input statistic/transform pass. Account for changing fitted attributes, strides, finite checks, and outputs requested by the API. Include a cold call and realistic repeated-call count.

**Judge:** Actual public quality and time through output consumption. Stop if caching relies on unsafe pointer identity, mutation invalidation dominates, or a device-side win is overwhelmed by mandatory output copies. Do not revive the previously slower KDE sample fusion without a changed explanation.

### F19 Make resampling a bandwidth experiment with an all-Mojo runtime

**Starting point:** [Resample estimator][resample] and the mixed GPU gather result in the [ledger][apple-ledger]. The older narrow hybrid proposal used host/NumPy work and is not the design proposed here.

**Experiment:** Compare direct device gather, coalesced tiled gather, and device-generated indices into caller-owned output. Keep all GPU-route draw/gather work in Mojo on the GPU. Derive variants from row bytes, reuse, number of arrays, and measured transfer costs; include narrow and wide rows, weighted/replacement modes, and output consumption.

**Judge:** Whole resample time, exact sampling semantics, memory, and transfer bytes. Stop if wide rows remain transfer-bound or any claimed gain excludes the output read. No dataset-specific width switch and no CPU runtime fallback to rescue a losing case.

### F20 Fuse optimizer and normalization work using task-level quality gates

**Starting point:** [Apple optimizer candidates][apple-optim], [sequence LayerNorm][layernorm], and existing attention normalization candidates. Qualification and scoped extension.

**Experiment:** Compare row-parallel normalization, batched parameter updates, and fused status checks independently. For optimizers, include a short fixed training task with the same hyperparameters and data order, plus state/update diagnostics and failure recovery. For normalization, include cancellation, large magnitude, tails, and downstream model outputs.

**Judge:** Training quality and full step/layer time, memory passes, and refusal semantics. Stop if a standalone optimizer lacks a meaningful quality judge, if fusion increases resource pressure enough to lose, or if real model quality drops. A moved digest alone is not a FAST failure.

## Experiment matrix and stopping discipline

The cards identify the mechanism; this matrix makes the intended support region reviewable before any run.

| Workload family | Required variation | Main diagnostic | Correctness or quality witness |
| --- | --- | --- | --- |
| GEMM and dense algebra | Orientations, adjacent/ragged sizes, short/long K, narrow/wide outputs, cold/resident calls | Compute versus memory stalls, registers, shared memory, partial planes | Contract seams and host oracle; solve/reconstruction/rank quality |
| Attention and sequence | Batch, length, heads/KV heads, width, causal tails, decode/prefill, backward | Live state, redundant loads, launches and waits | Full outputs/gradients/state plus downstream task quality |
| Trees | Depth, leaf count, bins, classes, categorical cardinality, weights, skew | Frontier/histogram work, scratch, host waits | Splits, leaf values, RNG/ties, fitted model and validation metric |
| Neighbors and graphs | Dimensions, k/radius/nprobe, duplicates, near ties, dense and skewed graphs | Materialized distances, task balance, canonicalization, fallback rate | Exact output order/membership, or declared approximate recall |
| Clustering and mixtures | Components/clusters, covariance type, skew, condition, seeds, convergence | Assignment/statistics stages, EM iterations, scratch | Centers/responsibilities/covariances and applicable quality |
| Preparation and resampling | Row bytes, number of arrays, repeated keys, weights, output size, reuse | Passes, transfers, first read, initialization | Sampling/index semantics, fitted attributes, exact mode contract |
| Time series | Series count/length, orders, missing/refused inputs, horizon, conditioning | Active candidate ratio, recurrence versus setup | Likelihood, gradients, order choice, forecast quality |

Use naturally adjacent shapes and at least one independent workload; do not choose validation inputs merely to preserve a desired dispatch boundary. For a profile revision, identity is candidate NVIDIA = candidate AMD = candidate Apple = candidate host. Candidate = old baseline is required only when claiming unchanged arithmetic.

For a schedule experiment, save the route counts and relevant intermediate witnesses alongside final outputs. A final digest can accidentally pass when the candidate is unreachable, when both arms load the same binary, or when a fixture's exact sums hide a changed order. Reuse existing sabotage/negative controls where applicable.

Stop a candidate early when it has no runtime reach, fails semantics/quality, cannot be expressed with supported tooling, or has a memory model that clearly loses at the public call. Mark noisy or incomplete evidence inconclusive. Keep a failed attempt's scope and explanation so it is not proposed again under a new name.

## Ideas deliberately deferred

- **Uncontrolled tensor-core substitution in IDENTICAL.** A native FP32 label does not establish the pinned FMA/fold/flush behavior. TF32/BF16/FP16 substitutions also change the requested precision contract. A wider matrix instruction requires a complete supported arithmetic proof or a separately agreed profile, not a throughput assumption.
- **Unordered floating atomics in IDENTICAL.** Use exact bounded integer accumulation where valid, or a canonical reduction. Atomics can be considered in FAST only with its actual quality and semantics gates.
- **Block Jacobi eigensolver revival.** Existing quarantine and convergence failures remain meaningful. A new convergence argument and independent quality evidence would be prerequisites, not another threshold sweep.
- **Algorithm changes awaiting decisions in earlier roadmaps.** Block-local SGD averaging, approximate replacement of exact CAGRA construction, tournament LU pivoting, and cancellation-sensitive Gram SSE are not quietly bundled into the schedule queue.
- **Compiler/backend workarounds.** AMD portable kernel output, unavailable generic targets, unsupported graph replay, or missing intrinsics wait for Modular support. Backend output patching is not an experiment in this plan.
- **Multi-GPU scaling and reduced-precision product modes.** Potentially valuable separate projects, but outside this AMD/NVIDIA IDENTICAL and Apple FP32 FAST experiment backlog.
- **Repeating a rejected arm unchanged.** Packed AMD bodies with an established loss, neutral MCD geometry, slower KDE sample fusion, pinned output, and host/NumPy gather proposals need a new causal mechanism before reconsideration.

## Suggested first frozen rounds

**Round one would resolve existing shared schedules:** I01 plus the platform-specific attribution in A01/N01, then I02 where allocation fallbacks are observed. Qualify the same frozen source on both timing vendors and the identity witnesses. Keep profile revisions out of this round so a failure has a narrow cause.

**Round two would target the largest measured non-GEMM cost:** choose I08, I10, I13, or I17 based on the refreshed stage profile. Reuse existing implementation branches for qualification, including the newer RBC candidate, rather than treating this document as a reason to write duplicate code.

**The separate Apple FAST round would begin with actual-caller GEMM and neural candidates:** F01/F05 and F07/F08/F10, selected by confirmed reach and available quality judges. F16 remains quality-first; its kernel evidence does not authorize estimator timing. No full FAST board rerun is implied.

After individual successes, run a small, explicitly chosen interaction round. Particularly useful pairs are group size × scratch reuse, tile size × staging pages, attention state retention × training memory, tree frontier width × histogram memory, and Apple GEMM geometry × fused epilogue. A bundle's win does not retroactively validate every component.

## Pinned source references

The repository links below refer to the reviewed baseline, not whichever branch happens to be checked out. The document is intended to remain usable after a documentation-only merge to older branches.

[agents]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/AGENTS.md
[acceptance]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/docs/PERFORMANCE_ACCEPTANCE.md
[fp32-contract]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/gemm/IDENTICAL_FP32_CONTRACT.md
[neural-roadmap]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/docs/plans/IDENTICAL_NEURAL_ROADMAP.md
[candidate-audit]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/docs/identical/candidate-audit/candidate-audit-2026-10-04.md
[idn-ledger]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/docs/identical/optimization-ledger.json
[recipes]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/tools/identical_candidate_recipes.json
[apple-ledger]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/docs/apple-fast/EXPERIMENTS.md
[apple-queue]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/docs/apple-fast/CONSOLIDATED_CANDIDATE_QUEUE_2026-10-04.md
[resident-screen]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/docs/apple-fast/RESIDENT_GEMM_RESULTS_2026-10-04.md
[gemm]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/gemm/checks/gemm_identical.mojo
[dev-tensors]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/training/dev_tensors.mojo
[llama]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/transformer/impl/llama/modeling_llama.mojo
[transformer-backward]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/transformer/checks/transformer_backward.mojo
[attention]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/transformer/impl/llama/fused_attention.mojo
[ssd]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/mamba/impl/modules/ssd_minimal.mojo
[siso]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/mamba/impl/ops/mamba3_siso.mojo
[selective-scan]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/mamba/impl/ops/selective_scan_interface.mojo
[mamba-backward]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/mamba/impl/modules/mamba3_backward.mojo
[recurrent]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/sequence/recurrent.mojo
[byte-lm]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/training/byte_lm.mojo
[optimizer]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/training/checks/optimizer.mojo
[embedding]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/embedding/checks/embedding_identical.mojo
[embedding-sort]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/embedding/checks/embedding_sort.mojo
[sgd]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/x_linear/sgd.mojo
[linear-device]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/x_linear/device.mojo
[cd]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/x_linear/cd.mojo
[rbc]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/neighbors/checks/ball_cover_canonical_order.mojo
[dbscan]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/dbscan/impl/dbscan.mojo
[hdb-switches]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/hdbscan/impl/detail/idn_switches.mojo
[knn]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/neighbors/impl/detail/knn_brute_force.mojo
[certified-knn]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/neighbors/impl/detail/certified_mma_knn.mojo
[ivf-search]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/ivf/impl/neighbors/ivf_flat/ivf_flat_search.mojo
[gbdt-depth]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo
[forest-builder]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/ensemble/decisiontree/batched_levelalgo/builder.mojo
[rf]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/ensemble/randomforest.mojo
[prep-radix]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/x_prep/dradix.mojo
[segmented-sort]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/core/segmented_sort.mojo
[kde]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/kde/impl/neighbors/kernel_density.mojo
[kde-check]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/kde/checks/kde_chunked_check.mojo
[gmm-estep]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/mixture/checks/estep.mojo
[gmm-mstep]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/mixture/checks/mstep.mojo
[decomp]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/x_decomp/device.mojo
[tsqr]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/x_decomp/tsqr_device.mojo
[cholesky]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/cholesky/checks/potrf.mojo
[trsm]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/cholesky/checks/trsm.mojo
[arima-batch]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/arima/impl/batched_arima.mojo
[sequence-exec]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/sequence/exec_device.mojo
[host-baseline]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/tools/hooks/host_routes_baseline.tsv
[metrics]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/x_metrics/cls_epi.mojo
[prep-device]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/x_prep/device.mojo
[kernel-matrix]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/checks/kernel_matrix.mojo
[attention-rejection]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/bench/evidence/2026-09-20_attention_v2_cooperative_dkdv_rejected.md
[gemm-reach]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/docs/apple-fast/GEMM_REACH_AUDIT_2026-10-04.md
[scoped-dispatch]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/experiments/apple_fast/gemm/scoped_dispatch.mojo
[softmax-narrow]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/experiments/apple_fast/gemm/softmax_narrow.mojo
[mcd]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/x_decomp/mcd_bmma.mojo
[apple-attention]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/transformer/impl/llama/afn_apple_fast.mojo
[byte-lm-fast]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/training/byte_lm_afn.mojo
[apple-lm-brief]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/docs/apple-fast/ab-neural/lm.md
[chunked-head]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/training/CHUNKED_LM_HEAD_V2.md
[apple-ssd]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/mamba/impl/modules/afn_ssd_mma.mojo
[mamba-controls]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/mamba/impl/modules/afn_defines.mojo
[gemm-outcomes]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/docs/apple-fast/GEMM_INLINE_OUTCOMES.md
[apple-tree-plan]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/docs/apple-fast/NEXT_PASS_TREES.md
[ordered-tree]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/gbdt/methods/oblivious_tree_doc_parallel_structure_searcher.mojo
[forest-inference]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/core/forest_inference.mojo
[shap]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/xtrees/shap_device.mojo
[minibatch]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/x_cluster/minibatch_fast.mojo
[cluster-device]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/x_cluster/device_ops.mojo
[arima-scalar]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/arima/impl/fast_scalar_df.mojo
[resample]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/resample/estimator.mojo
[apple-optim]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/training/afn_optim.mojo
[layernorm]: https://github.com/mojolearn/mojolearn/blob/f867b50e8f09c493ef8c2c939aded6b3a9f0910e/sequence/layernorm.mojo
