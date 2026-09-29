# x_cluster: the cluster expansion lane

MiniBatchKMeans, BisectingKMeans, MeanShift, OPTICS, AffinityPropagation and
BayesianGaussianMixture (full covariance), public through
`python/mojolearn/_expansion_cluster.py`.

## Shape

One generic driver per algorithm over the `ClusterOps` trait
(`ops.mojo`). `DeviceOps` (`device_ops.mojo`, the GPU binding
`_mojolearn_x_cluster`) runs each primitive as one kernel; `HostOps`
(`host/host_ops.mojo`, the CPU binding `_mojolearn_x_cluster_host`) runs the
same per-element body in a plain loop. Every body is in `bodies.mojo`, once;
host control code (sampling, sorting, sequential loops, float64 k-sized
updates) is one source compiled into both bindings. Entries are in
`entries.mojo`; the one Python call is `x_cluster_call(which, ...)`.

## Seams (IDENTITY_PATHS.md rows 110-119)

| DEVIATION | seam | body | check |
|---|---|---|---|
| 5100 / 5101 | squared distance: features ascending, the product pinned | `sq_dist_rows` | `checks/dist_check.mojo` |
| 5102 | nearest-row argmin: the lowest index on a tie | `nearest_row` | `checks/nearest_check.mojo` |
| 5103 | row order statistic by a bisection on the bits | `kth_smallest_row` | `checks/kth_check.mojo` |
| 5104 | mean-shift fold over the rows ascending | `meanshift_seed` | `checks/meanshift_check.mojo` |
| 5105 | affinity propagation damping: two pinned products, one add (`ap_r_update`, one spelling); on the device one block per row, the max and second max with their lowest indices by an integer max of `(float order, -index)` keys (lane/cluster-apple) | `ap_responsibility_row`, `device_ops._ap_r_kernel` | `checks/ap_check.mojo` (n = 48; n = 600 with planted ties; arms 5105_ap_damping, 5105_ap_rmax_tie) |
| 5106 | availability column fold ascending | `ap_availability_col` | `checks/ap_check.mojo` |
| 5107 | bisecting tree descent: the left child on a tie | `tree_descend` | `checks/descend_check.mojo` |
| 5108 | mixture Mahalanobis fold, the difference first | `gauss_q_cell` | `checks/gauss_check.mojo` |
| 5109 | E-step log-sum-exp: first max, ascending portable exp | `resp_row` | `checks/gauss_check.mojo` |
| 5110 | mixture M-step moments: nk rows ascending; means and covariances through the identical GEMM at OP_TN, as `mixture/` (DEVIATION 1729); revised 2026-09-29, the rows-ascending chain lost Istella-S's small covariance directions (`host/moments_gemm.mojo`) | `nk_cell`; `host/moments_gemm.mojo`; `device_ops._moments_gemm` | `checks/moments_check.mojo` |
| 5111 | OPTICS's other metrics: features ascending, portable pow/sqrt | `pdist_cell` | `checks/pdist_check.mojo` |
| 5117 | agglomerative Lance-Williams update: Float64 of the Float32 matrix, pinned products, one quotient | `lance_williams` | `checks/agglo_check.mojo` |
| 5118 | agglomerative merge order: the lowest live pair, the lowest i then j on a tie (host loop) | `agglo.mojo::agglo_tree` | `checks/agglo_check.mojo` |
| 5119 | SpectralClustering discretize / cluster_qr: a one-sided Jacobi SVD (column sums rows ascending, pinned products) for LAPACK's (host code) | `spectral_assign.mojo::jacobi_svd` | `checks/spectral_assign_check.mojo` |
| 5120 | the device row order statistic: a four-pass radix select on the bits, one block per row, integer histograms (the same value as 5103's bisection) | `device_ops.mojo::_kth_kernel` | `checks/kth_check.mojo` (long rows) |
| 5121 | FAST only since 2026-09-29 (IDENTICAL takes 5110's GEMM fold): the device M-step moments: a row tile's addends formed by every thread into shared memory, then each fold one thread's register chain over them rows ascending (the `*_term` / `chain_add` / final functions of 5110) | `device_ops.mojo::_moments_pass_kernel` | `checks/moments_check.mojo` (tiles, chains, fallback) |
| 5122 | AffinityPropagation's tie noise by the draw's counter (draw j = mix(seed + j * gamma)), one cell per thread, the float32 of the unit draw rounded to nearest-even from the 53-bit integer (no Float64: Apple GPUs have none) | `bodies.ap_noise_cell`, `splitmix_at`, `unit_f32` | `checks/ap_noise_check.mojo` |

Each check first shows its fixture SEPARATES the pinned spelling from the
alternative (VACUOUS otherwise), then holds the device and the CPU column to
the host oracle (`checks/oracles.mojo`) bit for bit. Each seam has a
source-patch sabotage arm that makes its check fail
(`checks/sabotage/`, paired with its driver in
`tools/identity_lanes/cluster.checks`; `tools/algos_lane_check.sh --pass 2`
applies, runs and reverses each). The host binding's negative control
is `-D MOJOLEARN_HOST_SABOTAGE=1` (the `REV` arm of the bodies).

What is not carried is in `NOT_IMPLEMENTED.tsv`.
