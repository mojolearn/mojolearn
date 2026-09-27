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
| SpectralEmbedding | -- | SKIPPED | already public (python/mojolearn/_spectral_impl.py, lane spectral-embedding); only affinity='rbf' is missing and that file is not this lane's |
| NMF | x-decomp-nmf | (see git log: "decomp lane: NMF") | AGREE infer 9, train 9 (NMF has no batch part: transform's stopping test is over the whole batch); ipca/grp/srp re-AGREE after the new cells |
| FastICA | x-decomp-fastica | (see git log: "decomp lane: FastICA") | AGREE batch 9, infer 9, train 9 |
| FactorAnalysis | x-decomp-factor-analysis | (see git log: "decomp lane: FactorAnalysis") | AGREE batch 9, infer 9, train 9. Sanity: fixed-iteration trajectories match sklearn (psi within 4e-5); at tol=1e-2 the stopping test sits under the float32 resolution of the log-likelihood (noise ~0.03 at n=1000) so the stop iteration can differ from sklearn's float64 run |

Next: SpectralEmbedding affinity='rbf' (policy: fix it in _spectral_impl.py), then LU solve
