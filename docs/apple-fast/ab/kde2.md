# lane/apple-fast-kde2: KernelDensity.score_samples on the Apple GPU (FAST)

Board lane `kde` (AFC_FAMILY=classical, binding `estimators`), datasets istella (d = 220, the
regression: FAST 1,539 ms against IDENTICAL 181 ms) first, then taxi (d = 11, FAST 60 against
IDENTICAL 112). Request lines in `kde2.txt`; the profile and the cause in
`docs/apple-fast/notes/kde2.md`. Every define is FAST + Apple only and default OFF; IDENTICAL compiles
main's code unchanged; FAST without a define runs main's fused pass unchanged. Bits: FAST only, the
log-sum-exp fold association changes (per-thread partials, a block merge, a chunk merge) as FAST may.
Quality (the board's mean log-likelihood of the queries) must stay within FAST run-to-run spread on
every arm; the gaussian x euclidean cell arithmetic is main's except under NORM_FUSED (below).

Files: `kde/impl/neighbors/kernel_density.mojo` (the `lane/apple-fast-kde2` section above
`kde_score_samples_device`, and the hook inside it), `kde/resident_fit.mojo` (SAMPLE_FUSED).

## -D MOJOLEARN_KDE_DIMTILE (the fix; the other defines turn it on too)

Mechanism. Main's FAST pass (`kde_fused_logsumexp_kernel` / `kde_fused_wide_kernel`) is one thread
per query over every train row: 16 threadgroups at 2,000 queries, each thread a serial chain of
`n_train x d` FMAs, and for d > 64 the query row is re-read from global memory per 13-row tile with
the accumulators in thread-private memory. `kde2_dimtile_kernel` is the GEMM shape instead: a block
of 128 threads owns 64 queries x 64 train rows per step, each thread 4 x 8 cells in registers
(stride 16 over queries, 8 over train rows, so a simdgroup's loads of one feature are consecutive
words); both tiles are staged 32 features at a time, feature-major with a one-word pad
(bank-conflict-free fill and inner loop), 21 KB of threadgroup memory under a compile-time fits
gate; the train rows are chunked over `grid.y` so about 2,048 blocks run (32 query blocks x 64
chunks on the board shape) instead of 16. Each thread folds its cells into an online log-sum-exp
`(m, s)` per query (the fused kernel's per-cell arithmetic: `v = acc * -1/(2h^2)` for gaussian x
euclidean, `compute_log_kernel` otherwise, `+ log w` when weighted), the 8 train groups of a block
merge through threadgroup memory, and `kde2_merge_kernel` (one thread per query) folds the chunk
pairs: `lse = M + log(sum_c s_c * exp(m_c - M))`, `-inf` when every chunk is `-inf` (DEVIATION 603
kept). Then main's `normalize_scores_kernel`. All six kernels and all six metrics run on this one
path (cosine takes its row norms from `kde2_row_sqnorm_kernel`); there is no d limit and no wide
arm. Launches per unweighted call: tile, merge, normalize (3; main: 2) plus the two norm launches for
cosine; scratch: `part_m`, `part_s` (n_chunks x n_query floats each, 512 KB each on the board), `lse`.

Expected. istella: the per-thread chain drops from 22 M FMAs to about 44 K (100,000 x 220 / 2,048
blocks / ... per cell set), the whole GPU is busy, and the inner loop is 2 threadgroup loads per
8 FMAs. The cells are 44 GFMA at d = 220 (about 3 ms at the M3 Ultra's rate) plus 200 M `exp`
calls in the epilog; the tile traffic is about 5.5 GB. Tens of ms, against 1,539 (and 181
IDENTICAL). taxi: the same parallelism gain over 16 blocks; the epilog (one `exp` per cell)
dominates at d = 11, so the gain is smaller but should still beat 60 ms.

Risk. A register-heavy kernel (32 accumulators + 12 staged values + the LSE state) can spill on
Metal; the compiler's choice is only visible on the box. The per-query fold order differs from
main's, so the digests differ (FAST may); the mean log-likelihood must not.

## -D MOJOLEARN_KDE_LSE_FUSED (on top of DIMTILE)

Mechanism. `kde2_merge_kernel[FINISH=True]`: the chunk merge applies `normalize_scores_kernel`'s two
subtractions (`lse - log(sum w)`, then `- norm`) and writes the scores directly. One launch instead
of two, no `lse` buffer (one fewer live Metal buffer per call).

Expected. About one launch + wait (~0.2 ms) and one buffer lifetime per call; visible only once
DIMTILE has brought the call to tens of ms. Risk: none to quality; the two roundings are the same.

## -D MOJOLEARN_KDE_NORM_FUSED (on top of DIMTILE)

Mechanism. For euclidean and sqeuclidean, `kde2_row_sqnorm_kernel` computes `|q|^2` per query and
`|t|^2` per train row once per call (one thread per row), the tile kernel's inner loop accumulates
the dot `q . t` only (one FMA per feature instead of a subtract and an FMA), and the epilog expands
`max(0, |q|^2 + |t|^2 - 2 q.t)`. This is the expanded L2 cuML's pairwise distance uses.

Expected. The inner loop halves its arithmetic; if the tile kernel is ALU-bound at d = 220 this is up
to the whole inner-loop cost, if it is load-bound it is nothing. Two extra launches and two buffers
(n_query + n_train floats).

Risk. Quality: the expanded form cancels when `|q - t|^2 << |q|^2`; on the board's standardized data
`|x|^2 ~ d` and the error on `dist^2` is ~1e-7 x 440, i.e. ~2e-5 on the log-kernel at h = 1,
far inside run-to-run spread. On un-standardized data with large offsets it would not be; this
define must be judged on the quality column as well as the time, and it is the one define here
that changes a cell's arithmetic.

## -D MOJOLEARN_KDE_KERNEL_VARIANTS (on top of DIMTILE)

Mechanism. The tile kernel is instantiated per metric at compile time (`METRIC_C`: the six
metrics, plus `GAUSS_C = 1` for gaussian x euclidean), seven instantiations, dispatched once on the
host; the default DIMTILE build has one instantiation that reads the metric at run time and
branches per feature chunk (a uniform branch outside the feature loop) and per cell in the epilog.
With the variant, no metric or gaussian branch exists in the kernel body, so the compiler sees one
inner loop and one epilog: fewer instructions, fewer live registers, no branch per cell.

Expected. A few percent on the tile kernel at most (the runtime branches are uniform and sit
outside the feature loop); its value is in whether the single-instantiation kernel spills where the
specialized one does not. Risk: compile time (seven kernels); none to quality.

## -D MOJOLEARN_KDE_SAMPLE_FUSED (independent of DIMTILE)

Mechanism. `kde_score_samples_resident` (the board's path: the fit set is device-resident, the
call uploads the queries and scores) drains twice per call on main: once inside the fused flow
(`ctx.synchronize()` at the end of `_kde_score_samples_fused`) and once after the pinned download.
Under this define it enqueues the score, enqueues the download straight into the caller's rows
(`enqueue_copy(dst_ptr=scores, src_buf=dout)`, no pinned `hout` buffer), and synchronizes ONCE; the
scratch buffers outlive that drain in a list and are freed after it. Unweighted calls also create
no dummy log-weight buffer (main allocates a 1-float `logw` per call). Without DIMTILE it runs
main's fused kernels through the same single-drain flow (`_kde2_launch_main_fused`), so it can be
A/B'd alone against main; with DIMTILE it runs the tile pass.

Expected. One drain (~0.2 ms on Apple) and two buffer lifetimes per call. Risk: the raw-pointer
download (8 KB) instead of the pinned one; measured ~21 ms per 64 MB on Apple for raw downloads,
so negligible at this size, but the A/B decides.

## -D MOJOLEARN_KDE2_ALL

All five. Request lines compare it against main ("" arm) and against DIMTILE alone, on istella.
