# x_neighbors: the neighbors + kernel expansion lane

LocalOutlierFactor, NearestCentroid, OneClassSVM, KernelPCA,
PolynomialCountSketch, AdditiveChi2Sampler, SkewedChi2Sampler,
LabelPropagation, LabelSpreading, KNNImputer, PageRank,
connected_components, Louvain and SVGP (`python/mojolearn/_expansion_neighbors.py`).

## How it is built

Every floating-point operation is an ITEM FUNCTION in `items.mojo`: the work
of one output cell, one row, or one whole sequential solve, written once.
`gen.py` generates, from its one op table, the GPU driver (`device_ops.mojo`:
upload, one thread per item, download), the host driver (`host_ops.mojo`: a
plain loop over the same items) and both bindings
(`bindings/_mojolearn_x_neighbors.mojo`, `bindings/_mojolearn_x_neighbors_host.mojo`),
so the GPU column and the CPU column run the same statements over the same
address contract. The dense symmetric eigenproblem (KernelPCA) is the host
Jacobi of `spectral/checks/symmetric_eig_host.mojo` in both bindings. Python
only moves buffers and keeps exact integer bookkeeping.

## Seams (IDENTITY_PATHS.md rows 120-129)

| DEVIATION | seam | move | check / sabotage arm |
|---|---|---|---|
| 5200 | one-class SMO (`ocsvm_smo_item`) | float32, one sequential item; libsvm's `>=` / `<=` scans, the LAST index on a tie | `checks/model_check.mojo`, `5200_smo_first_index.patch` |
| 5201 | NearestCentroid shrink scale m*s == 0 | deviation 0 instead of a computed NaN | `model_check.mojo`, `5201_shrink_centroid_fold_reversed.patch` |
| 5202 | KernelPCA centering order, svd_flip sign row; every eigen_solver served by the dense Jacobi | their subtract-subtract-add order; the FIRST row of largest magnitude | `model_check.mojo`, `5202_svd_flip_last_row.patch` |
| 5203 | PolynomialCountSketch convolution | summed directly, shift ascending (no FFT) | `checks/sketch_check.mojo`, `5203_pcs_shift_reversed.patch` |
| 5204 | Louvain | sequential, nodes ascending, ties to the lowest community (networkx shuffles by seed) | `checks/graph_check.mojo`, `5204_louvain_order_reversed.patch` |
| 5205 | SVGP | q(u) at its closed-form optimum for the caller's hyperparameters; Cholesky columns left to right (a launch per column, a thread per row), folds ascending; y^T y blocked | `graph_check.mojo`, `5205_cholesky_fold_reversed.patch` |
| 5206 | distances (sqdist, nan_euclidean, L1) | features ascending on the pinned fma; `/present` then `*d` | `checks/dist_check.mojo`, `5206_sqdist_fold_reversed.patch` |
| 5207 | k-NN selection | strict `<` insertion: equal values keep the lower column | `dist_check.mojo`, `5207_select_tie_high.patch` |
| 5208 | kernel epilogues | the pinned fold, portable exp / tanh / sqrt | `dist_check.mojo`, `5208_rbf_fold_reversed.patch` |
| 5209 | dense folds (matmul, sums, group means, variance, normalize, softmax) | inner index ascending, one thread per cell; whole-array folds (variance, sum abs difference) blocked by `XN_FOLD_BLOCK`, partials ascending | `checks/fold_check.mojo`, `5209_matmul_fold_reversed.patch` |
| 5210 | LOF densities | ranks ascending | `fold_check.mojo`, `5210_lrd_ranks_reversed.patch` |
| 5212 | NearestCentroid std / shrink / discriminant | rows ascending; the distance square-rooted then squared, as theirs | `model_check.mojo`, `5212_nc_std_rows_reversed.patch` |
| 5213 | chi-squared samplers | cosh as the mean of two exps, tan as sin / cos, sqrt(step / cosh) | `sketch_check.mojo`, `5213_sech_factor_split.patch` |
| 5214 | label propagation / spreading | soft clamp as product then add; Laplacian `/ w_j` then `/ w_i` | `checks/semi_check.mojo`, `5214_laplacian_divide_wi_first.patch` |
| 5215 | KNNImputer donors | nearest first, the lower row on a tie | `semi_check.mojo`, `5215_donor_tie_high.patch` |
| 5216 | PageRank step | x @ Q ascending, the dangling mass, the teleport, pinned fmas | `graph_check.mojo`, `5216_pagerank_fold_reversed.patch` |
| 5217 | connected components | the MIN label (integers) | `graph_check.mojo`, `5217_cc_max_label.patch` |

Run one check: `tools/with_identical_mode.sh pixi run mojo run -I . x_neighbors/checks/<name>_check.mojo`.
All of them with their arms: `sh tools/algos_lane_check.sh <lanes> --pass 2` (listing:
`tools/identity_lanes/neighbors.checks`). What is not carried: `NOT_IMPLEMENTED.tsv`.
