# Geometry and clustering source-only candidates

All fourteen AFCL-G cards have source changes in the existing production
call paths. **NEVER RUN — PENDING MEASUREMENT.** They are uncompiled,
unverified and unmeasured. No compiler, tests, checker, identity run,
benchmark, GPU job or performance/quality qualification was run here.
No result or speedup is claimed. Parent owns the eventual commit/push.

Every new control is a compile define with default OFF. Presence enables
the candidate only in Apple `NUMERIC_FAST`; omission is A. Setting a define
to `0` still enables an `is_defined` switch, so omit it to disable it.
All existing flags and runtime settings must be identical in both arms.
Bits may change; training budgets, tolerances, seeds, candidate sets and
task-quality requirements may not be weakened.

| Card | A → B source mechanism | Actual reach and limitations |
| --- | --- | --- |
| G01 | MMA kNN: 8 → 4 SIMD groups per block | Exact kNN MMA route, including classical consumers; narrow and chunked-feature kernels. Existing feature/k register limits unchanged. |
| G02 | Scalar streaming top-k: 64 → 128 staged index candidates | Exact scalar fused kNN route. Keep `MOJOLEARN_KNN_FAST_MMA_OFF` in both arms; otherwise the earlier MMA dispatch takes precedence. Final merge and number of slices are unchanged. |
| G03 | DBSCAN MMA epsilon search: 8 → 4 SIMD groups per block | Existing eligible MMA epsilon-neighborhood route, retaining the exact epsilon-boundary fallback. Wider-feature fallback unchanged. |
| G04 | HDBSCAN tiled core distances: 64 → 32 reference rows per tile | Keep `MOJOLEARN_HDB_CORE_TILE` in both arms. Existing register k/feature eligibility unchanged. This does not turn that unqualified older route on by itself. |
| G05 | Exhaustive centroid assignment: 16 → 8 normal row threads, 8 → 4 skinny row threads | Fused KMeans assignment, including callers that use it for initialization/quantization. Column threads, feature tile, alignment ladder and distance semantics retained. |
| G06 | Blocked centroid sums: current 256 → 128 input rows per partial | Keep `MOJOLEARN_EXPERIMENTAL_KMEANS_BLOCK_ACC` in both arms. Existing accumulator override flags must be absent for the stated 256-row A. Int32 accumulation and quantization policy retained; partial table grows. |
| G07 | Resident MiniBatchKMeans: 256 → 128 threads and chunk rows | Assignment, compacted center sum and resident finish/reassignment schedules share the width. Group length, batch draws and stopping observations unchanged. Different sums may change convergence trajectory. |
| G08 | GMM E-step: 256 → 128 fused row threads, 128 → 64 ordinary row threads | GaussianMixture fit and E-step inference/score callers. Component traversal, Mahalanobis sums, logsumexp and mean-likelihood policy unchanged. Explicit callers overriding `row_tpb` retain that override. |
| G09 | IVF scan: 4 → 2 query SIMD groups per block | Existing batched FAST IVF-Flat scan and its ANN consumers. Admission feature cap deliberately stays fixed at the baseline cap; no probe/list/candidate reduction or balanced-task replacement. |
| G10 | UMAP FAST epochs: 128 → 64 CSR head rows per block | Dense-graph and sparse graph FAST optimizers, including the existing fused 2D/3D epoch kernel. Each head retains edge order, snapshot and counter-derived negative samples. |
| G11 | Spectral Lanczos SpMV: 8 → 4 SIMD-owned CSR rows per block | SpectralEmbedding and SpectralClustering on the existing long-row SpMV route. Graph, solver tolerances and row sum partition retained. LLE's separate dense solve is outside this implemented scope. |
| G12 | MeanShift: 256 → 512 input rows per seed partial | Existing grid-based all-neighbor MeanShift. Two strided bandwidth-test passes amortize seed loading and halve partial-table/final-fold work. Seeds, bandwidth and iterations retained. |
| G13 | KDE tiled stable scoring: 4 → 2 queries per thread | Existing Apple FAST dimension-tiled scoring for KernelDensity. Query tile and distance accumulator bank halve; all train contributions and max-rescaled logsumexp remain. Query-derived chunk count can change reduction association. |
| G14 | MMA Boruvka MST: 128 → 64 staged reference rows | Agglomerative single-linkage Euclidean pairwise route and HDBSCAN consumers of the same search. Scalar boundary recomputation, every candidate and tie ordering remain; existing unsupported linkage/metric semantics remain refused. |

The narrower blocks trade per-block reuse against occupancy; the larger
streaming/MeanShift tiles trade synchronization and scratch traffic against
serial work. These are hypotheses, including plausible losses. All sizes
derive from full SIMD groups, shared-storage budgets or bounded row-chunk
work, and apply to neighboring input shapes. No board dimension or dataset
name selects an arm. No shared neural GEMM or Python numerical runtime was
changed. `geometry.json` records the exact controls, source/caller paths,
dependencies and future quality scope for every card.

Future qualification must use the full saved workloads for every affected
estimator, with an explicitly mapped dataset/version/hash, actual dimensions,
uncapped coverage, settings and whole-operation boundary. A component win
does not admit a switch. Include preparation, fit, synchronization and
consumed outputs, and report inference/cold/repeated usage separately where
applicable. Required quality is the existing task contract and accepted
quality floor, including neighboring shapes and a non-board workload.

Particularly relevant combinations are G01+G05 for neighborhood/quantizer
consumers, G05+G06 for KMeans, G05+G08 for GMM initialization plus E-step,
G05+G09 for IVF training plus queries, and G01+G11 for spectral graph plus
eigensolver. G01 and G02 are alternative distance routes; an interaction
does not mean enabling mutually preempted arms and claiming both executed.
G04 is distinct from HDBSCAN's normal kNN core route. G04+G14 also needs its
own HDBSCAN evaluation when both tiled core distance and MMA MST search
execute. G13 must cover every supported density kernel/metric, weights and
extreme bandwidths; G14 must cover both hierarchy and HDBSCAN consumers,
ties, degenerate points and eligible/fallback boundaries. Existing workload caps
and all performance/quality coverage remain pending; no board was changed.

Only source and planning artifacts exist for this lane. There are no build,
test, timing, identity or GPU logs to report. No validation claim should be
inferred from the presence of the manifest or implementation comments.
