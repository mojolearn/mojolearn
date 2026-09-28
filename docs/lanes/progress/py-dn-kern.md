# lane/py-dn-kern (sub-lane of py-decomp-nbrs): kernel chains and spectral loops

Brief: ~/mojolearn-evidence/py_work_brief.md. Audit rows: neighbors row 8,
cluster rows 1, 2, 7, decomp rows 5, 6 (~/mojolearn-evidence/python_work_audit.md).
Base: lane/py-decomp-nbrs (on lane/apple2-merged 342469dae). Base columns:
/root/mojolearn-py-dn-base on the shared NVIDIA pod.

## What moved

| where | before | after |
|---|---|---|
| `KernelPCA.transform` | 5 `xn_*` calls; the nq x n_fit kernel matrix downloaded after `kernel`, re-uploaded for `rowsum` and `kpca_center`, the centered copy downloaded and re-uploaded for `matmul` | one `xn_kpca_transform`: the same items (kernel, rowsum, scale_div, kpca_center, matmul) per device-resident row tile (at most 2^24 cells), only (nq, c) downloaded |
| `OneClassSVM.score_samples` (not precomputed) | `kernel` then `matmul`, the matrix across the bus twice | one `xn_kernel_matmul` per row tile |
| `SVGP.fit` | `Kfu` and `Kuf` both computed and downloaded, two `matmul` calls re-uploading them | one `xn_svgp_stats`: Kfu per tile, B = Kuf Kfu and b = Kuf y folded across tiles by `matmul_tn_acc_item` (the float32 accumulator carried; same pinned fma sequence); Kuf is never formed |
| `SVGP.predict_f` | `kernel`, `unary`, `matmul`, `svgp_var`, each a round trip | one `xn_svgp_predict` per row tile |
| `_spectral_impl._DenseCOO` (rbf and precomputed kNN routes of SpectralEmbedding and SpectralClustering) | Python double loop over n^2 float32 cells | `nonzero_f32_count/fill` (core/dense_coo.mojo, base binding and core host binding), DEVIATION 2489's float64 scan over float32 |
| `SpectralEmbedding._precomputed_knn_affinity` | `tolist`, per-row Python tuples and `sorted`, n^2 symmetrize loop | `knn_affinity_f32` (core/dense_coo.mojo): per-row integer sort of (float32 bits, column) keys over host tasks, C, then 0.5 (C + C^T) |
| `_expansion_decomp.py` graph helpers | `_knn_order`, `_knn_connectivity`, `_symmetrize`, `_normed_laplacian`, `_sign_flip_rows`, never called | deleted |

Kuf == transpose(Kfu) was checked in the item before relying on it: the rbf
item forms `_sub(x_f, z_f) = ftz(ftz(a) - ftz(b))`, and IEEE subtraction is
antisymmetric (a - b == -(b - a) exactly), so the squares, the fold, the
scale and the exp are the same bits; the unary scale follows.

A/B arm: `MOJOLEARN_XN_OLD_ITEMS=1` keeps the unfused chains (the switch the
x_neighbors lane already uses for its replaced forms).

Negative control for the new device path:
`x_neighbors/checks/sabotage/fused_chain_device.patch` (the fused device
drivers add 1e-3 to their first output; the host column is untouched).

Tests: `python/mojolearn/tests/test_native_dense_coo.py` holds the new scan
and affinity to byte equality with the retired Python loops (kept there as
oracles), planted zeros, -0.0, NaN, ties, duplicate COO entries, refusals.

## Before / after (same box, one job)

PENDING. The first shared pod went down at about 19:20Z on 2026-09-28. On
the new pod (nvc1, 2x A40) the four touched bindings BUILD (`_mojolearn`,
`_mojolearn_core_host`, `_mojolearn_x_neighbors`, `_mojolearn_x_neighbors_host`,
built by `sh`, no GPU). The head columns job nvc1-0012 was cancelled at
19:52:54Z together with every other py-* job on the queue (not by this lane);
it produced no verdict. Job scripts are on the pod: /root/ev-py-dn-kern/head1.sh
(columns + ab_diff against /root/ev-py-decomp-nbrs/base), sab1.sh (the fused
device sabotage, once), kern_timing.sh (base tree vs this tree, GPU then CPU;
bench ~/mojolearn-evidence/py-dn-kern/kern_bench.py). Nothing below is measured yet.

| machine | column | case | base s | lane s | digest equal |
|---|---|---|---|---|---|

## Bits (base columns vs lane columns, and GPU == CPU)

PENDING (same reason). Lanes: x-neighbors-kpca, x-neighbors-ocsvm,
x-neighbors-svgp, spectral, spectral-precomputed, spectral-embedding,
par-graph-spectral, x-cluster-spectral-affinities, x-decomp-spectral-rbf.

## DEVIATION changes

None added or retired. DEVIATION 2489's intent (the dense scan in Mojo) is
restored for the float32 routes `_DenseCOO` had reintroduced in Python.
Row 139's "keyed sorts" permission loses the precomputed kNN per-row sort
(now in Mojo).

## Unproven

Everything in this file until the pod jobs above run: the build of
`_mojolearn`, `_mojolearn_core_host`, `_mojolearn_x_neighbors` and
`_mojolearn_x_neighbors_host`, the lane columns, the sabotage arm and the
timings.
