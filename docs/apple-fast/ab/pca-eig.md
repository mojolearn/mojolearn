# lane/apple-fast-pca-eig: PCA / tSVD eigen step on the grid (PLAN.md item 7, PLAN-classical.md 3)

Written without a Mojo toolchain (cloud peer); the first M3 build (binding `estimators`,
bindings/build_estimators.sh) is the compile check. Every switch is compiled under FAST only
(`comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL`) and defaults OFF; IDENTICAL compiles
the old code unchanged. The switches are env vars read on the host at dispatch.

| switch | kind | site | what it changes under FAST |
|---|---|---|---|
| `MOJOLEARN_PCA_FAST_EIG=1` | env | `decomposition/impl/linalg/detail/pca.mojo` `eig_and_truncate` -> `_pca_fast_rr_eigh` | the round-robin Jacobi of `x_decomp/jacobi_par.mojo` (`eigh_par_cs_kernel`, `eigh_par_update_kernel`, `eigh_par_off_kernel`, the x_decomp kit's own default eigh order since lane/neural-pass104) on a device copy of `cov`: m - 1 rounds a sweep, each round's m / 2 disjoint rotations across the grid, the sweep test folded on the device (`pf_sweep_test_kernel`) and read back as 4 floats once a sweep. Not converged in 30 sweeps, or ||A||_F moved: `cov` untouched, the cyclic kernel runs as before and its refusal keeps its name |
| `MOJOLEARN_PCA_FAST_TOPK=1` | env | same file, `_pca_fast_topk_tail` | the descending order of the n eigenvalues (`pf_rank_kernel`, one thread per value, rank by count, ties to the lower index) and the gather of the n_components columns of V (`pf_gather_kernel`) on the device; n + n_components x n floats and n ints come back instead of two n x n matrices; the float64 ratios, square roots and noise mean are the host's, in `order_truncate_spectrum`'s order |
| `MOJOLEARN_PCA_FAST_NO_ALIAS=1` | env | `decomposition/estimator.mojo` `pca_fit_host` | when `compute_covariance` takes the fused split-K arm (`gram_splitk_applies`, the board shape), `xa2` is allocated at 1 float instead of n x d and `xa` at the split-K partials scratch size (`gram_splitk_chunk_count() * d * d`, `gram_splitk_scratch_covers`) instead of n x d (3.5 GB each at Istella's 4,000,000 x 220) |

Cause: `eig_and_truncate` launched `jacobi_eigh_kernel[JACOBI_ROT_TPB]` with grid_dim=(1, 1, 1): one block
of 256 threads ran every one of the n (n - 1) / 2 cyclic rotations of a sweep behind a barrier (24,090 a
sweep at 220 columns, up to 15 sweeps) while the GPU idled, then copied both n x n matrices to the host and
ordered, truncated and gathered there. `pca_fit_host` allocated two unused n x d buffers on the fused arm.
Board: pca Istella FAST 988 ms vs IDENTICAL 814 vs best opponent ~206; tsvd shares `eig_and_truncate`.

Notes
- The round-robin kernels are vendor-neutral (x_decomp runs them on every column), so the switch is gated on
  FAST only, not on Apple; NVIDIA / AMD FAST builds of `estimators` compile them too (they already do in the
  x_decomp binding).
- `incremental-pca` (bench_board_algos, datasets taxi, istella) is on the x_decomp lane (`_gram_svd` ->
  `k.eigh`, already the round-robin default there); its lines are a flat-check baseline, not an A/B of this change.
- The `pca-eig-all-istella` line passes three envs in one single-quoted envB; if the queue does not keep the
  quotes, run it by hand or drop it (the three single-switch lines are the measurement).
- Compile risks to watch: `enqueue_fill(ctx, order_buf, Int32(-1))` on a DType.int32 buffer;
  `ctx.enqueue_copy(dst_buf=info_buf, src_ptr=h_set.unsafe_ptr())` from a HostBuffer;
  the module-level `from x_decomp.jacobi_par import ...` in pca.mojo (x_decomp/device.mojo imports pca.mojo's
  `sign_flip_kernel`; jacobi_par itself imports only cells, rr and jacobi_eigh_device, so no cycle).

Keep rule: a switch becomes the FAST default when its arm is faster on the M3 and held-out quality
(explained-variance ratio of the components) stays within FAST's run-to-run spread; then the env read goes
and the arm is the code.
