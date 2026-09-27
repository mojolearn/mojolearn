# decomp: progress

Design (pass 1): every float operation is a cell in `x_decomp/cells.mojo`, run
by `x_decomp/device.mojo` (GPU binding `_mojolearn_x_decomp`, one thread per
output) or `x_decomp/host.mojo` (host binding, same cell in a loop); the entry
points are written once in `x_decomp/api.mojo`. Python
(`python/mojolearn/_expansion_decomp.py`, `_Kit`) holds control flow and data
movement only. eigh reuses `decomposition/` (device_eigh / host_eigh).

| algorithm | lane | commit | pod check |
|---|---|---|---|
| IncrementalPCA | x-decomp-ipca | 9135f7adb | AGREE: compared batch 9, infer 9, train 9 (A40 cuda vs x86 CPU) |
| GaussianRandomProjection / SparseRandomProjection | x-decomp-grp, x-decomp-srp | (see git log: "decomp lane: GaussianRandomProjection / SparseRandomProjection") | AGREE batch 9, infer 9, train 9 each |
| SpectralEmbedding | -- | already public | the existing _spectral_impl.py class (cuML route); its missing affinity='rbf' is added in the next row |
| NMF | x-decomp-nmf | (see git log: "decomp lane: NMF") | AGREE infer 9, train 9 (NMF has no batch part: transform's stopping test is over the whole batch); ipca/grp/srp re-AGREE after the new cells |
| FastICA | x-decomp-fastica | (see git log: "decomp lane: FastICA") | AGREE batch 9, infer 9, train 9 |
| FactorAnalysis | x-decomp-factor-analysis | (see git log: "decomp lane: FactorAnalysis") | AGREE batch 9, infer 9, train 9. Sanity: fixed-iteration trajectories match sklearn (psi within 4e-5); at tol=1e-2 the stopping test sits under the float32 resolution of the log-likelihood (noise ~0.03 at n=1000) so the stop iteration can differ from sklearn's float64 run |
| SpectralEmbedding affinity='rbf' | x-decomp-spectral-rbf | (see git log: "decomp lane: SpectralEmbedding affinity='rbf'") | AGREE train 9; --sabotage (host sqdist features descending) PASS: AGREE, DISAGREE 9, AGREE; verify spectral-embedding/spectral/spectral-precomputed VERIFIED before and after (67 parts, 0 divergent) |
| LU solve (lu_factor, lu_solve, solve) | x-decomp-lu | (see git log: "decomp lane: LU solve (lu_factor, lu_solve, solve)") | AGREE train 9; pivots equal scipy's including exact ties |
| lstsq / randomized_svd | x-decomp-lstsq-rsvd | (see git log: "decomp lane: lstsq / randomized_svd") | AGREE train 9; all 9 decomp lanes re-AGREE after the new MGS2 orth cell |
| PLSRegression / PLSCanonical / CCA | x-decomp-pls | (see git log: "decomp lane: PLSRegression / PLSCanonical / CCA") | AGREE batch 9, infer 9, train 9; sklearn match 2e-6 (CCA mode B needs the new QR + one-sided Jacobi svd cell; _thin_svd and FactorAnalysis moved to it, FA now stops at 60 vs sklearn 62 iterations); fa/lstsq-rsvd/nmf/ipca/fastica re-AGREE |
| SparsePCA / MiniBatchSparsePCA / DictionaryLearning (+ MiniBatchDictionaryLearning, sparse_encode) | x-decomp-sparse-pca, x-decomp-dict-learning | (see git log: "decomp lane: LatentDirichletAllocation ...", same commit) | AGREE batch 9, infer 9, train 9 each; sklearn match 1e-6 (DictionaryLearning, SparsePCA, MiniBatch* with shuffle=False) |
| LatentDirichletAllocation | x-decomp-lda | (same commit) | AGREE batch 9, infer 9, train 9; given one model, transform and score equal sklearn's to 1e-7; fit reaches sklearn's optimum on 3 of 6 seeds in 20 iterations (the Philox Gamma draws start it elsewhere) |
| LatentDirichletAllocation, Isomap / MDS / ClassicalMDS / LocallyLinearEmbedding, SparsePCA / MiniBatchSparsePCA / DictionaryLearning | x-decomp-manifold (Isomap, MDS, ClassicalMDS, LLE) | (see git log: "decomp lane: LatentDirichletAllocation, Isomap / MDS / ClassicalMDS / LocallyLinearEmbedding, SparsePCA / MiniBatchSparsePCA / DictionaryLearning") | AGREE batch 9, infer 9, train 9; sklearn match 1e-6 (Isomap, ClassicalMDS, MDS), 1e-4 (LLE through the SVD of I - W); test_lane_select OK |

Next: EllipticEnvelope / MinCovDet, then ALS
