# PCA / TruncatedSVD / randomized_svd: fit-path trace and ideas (read-only analyst, 2026-10-09, main 42d1e42c6)

## 0. Shapes and bytes (from bench/, not the brief)

The brief's "istella 3.3M x 220" is wrong for the board: `tools/classical_two_datasets.py:32-45` says the big block is the
cached Istella-S TRAIN SPLIT, 2,043,304 rows (the 3.4M count includes the validation file the cache never decodes);
taxi is 4,000,000 rows x 11 numeric columns. tsvd and randomized-svd run the algos "tsvd" block: 1,000,000 stride rows
(`tools/bench_board_algos.py:1596`). All fp32 C-order, `n_components=10` (pca/tsvd, `tools/bench_board_harness.py:92-94`),
randomized-svd `n_components=8, n_oversamples=10, n_iter=4` (`bench_board_algos.py:484-488`; torch = `svd_lowrank(q=18, niter=4)`).

| lane | dataset | X bytes | uploads inside our clock | torch/cuML clock |
|---|---|---|---|---|
| pca | istella | 2,043,304 x 220 x 4 = **1,798,107,520 B (1.80 GB)** | 1 | torch kernel-only (`TorchPCA`, classical_two_datasets.py:1408-1426: mean, xc = x - mean, xc.T @ xc, `torch.linalg.eigh` = cuSOLVER syevd) |
| pca | taxi | 4,000,000 x 11 x 4 = **176,000,000 B** | 1 | same |
| tsvd | istella | 1,000,000 x 220 x 4 = **880,000,000 B** | **2** (tsvd_fit + tsvd_explained each copy X) | cuML tsvdFit = Gram + eig, inputs on device |
| tsvd | taxi | 1,000,000 x 11 x 4 = **44,000,000 B** | **2** | same |
| randomized-svd | istella | 880 MB | 1 device upload + **2 host copies** (`_M.from_input`: `a.tobytes()` then `array.frombytes`) | torch svd_lowrank on a device tensor |
| randomized-svd | taxi | 44 MB | same | same |

Copy rate on the L40S box: the board recorded torch's pageable upload of the taxi 176 MB at 9.19 ms (`board.json
/races/classical/pca/taxi/.../torch-gpu/upload_ms_untimed`) = **19.2 GB/s**. Our `ctx.enqueue_copy(dst_buf=x, src_ptr=x_ptr)`
from the NumPy pointer is the same pageable path, so the copy inside our clock is about:
pca taxi **9.2 of 12.7 ms (72%)**; pca istella **94 of 231 ms (41%)**; tsvd istella **2 x 46 = 92 of 96 ms**; tsvd taxi
**4.6 of 6.7 ms**. The copy dominates taxi PCA and both tsvd cells outright; on istella PCA it is the largest single stage but
the kernel side (~137 ms vs torch 16.9) is still an 8x deficit by itself. No `upload_ms_separate` for our cells is in this
checkout (bench/results holds only the opponent cells), so these are byte/rate estimates; the probe settles them.

## 1. PCA fit trace (IDENTICAL, svd_solver='covariance_eigh', the board arm)

Python `PCA.fit` (`python/mojolearn/decomposition.py:313-416`): `as_f32_c(_dense(X))` (no copy for a C-order f32 array),
allocates the five output arrays, one binding call `pca_fit` (`bindings/_mojolearn_estimators.mojo:280-308`) ->
`pca_fit_host` (`decomposition/estimator.mojo:115-192`):

| # | stage | code | launches / grid | bytes (istella) | host sync |
|---|---|---|---|---|---|
| 1 | device buffers: `x` (n*d), `xa` (n*d), `xa2` (n*d), `mu`, `cov` | estimator.mojo:127-135 | 3 x 1.8 GB allocations per fit; `xa` is never read on the >128-column path (gemm_tn_identical_v1 uses only `xt2`, and `need <= k*m` so it reuses it) | 5.4 GB resident | - |
| 2 | H2D copy of X (pageable) | estimator.mojo:136-137 | 1 | 1.8 GB | sync |
| 3 | column mean | `column_mean_launch` -> IDN_XTY_TILED `xty_tiled` (core/xtdz_coalesced.mojo:228-256,364-385) | 2 launches: tile partials (tiles=n/256 x d cells, 256 TPB) + per-column fold | 1.8 GB read | sync |
| 4 | center X in place | `shift_columns_kernel` (pca.mojo:339-346), 1 elem/thread | 1 launch, 1.76M blocks | 1.8 GB r + 1.8 GB w | - |
| 5 | Gram | `gemm_tn` (core/gemm.mojo:529-562): `gram_splitk_applies` is False at 220 > GRAM_MAX_COLS=128 (core/gram_splitk.mojo:231,477) -> `gemm_tn_identical_v1` -> `identical_gemm_into(OP_TN)` -> `choose_gemm_plan_tiles` (gemm/checks/gemm_identical.mojo:4265-4318): `contract_partition(k)`: K_LEAF_MIN=128, MAX_LEAVES=1024 (gemm/contract.mojo:133,161) -> L=1996 rows, P=1024 leaves; 220x220=48,400 cells <= cap 128K (NV) / 1M (AMD) -> **PLAN_SPLIT_64_4X4**: 4x4=16 output tiles x 1024 leaves = 16,384 blocks, each loading two 1996x64 slabs of X | X slabs loaded ~9x (16 tile-blocks x 2 operands / (220/64 cols, padded to 256)) = **~17 GB of loads** (L2 may absorb part), partials 1024 x 48,400 x 4 = 198 MB write + fold read | - |
| 6 | scale cov by 1/(n-1) | `scale_in_place_kernel` | 1 | 194 KB | - |
| 7 | **restore X** (`restore_input = not PCA_FAST_GRAM_MMA` = True on NVIDIA/AMD, estimator.mojo:159) | `shift_columns_kernel` +mu | 1 | 1.8 GB r + 1.8 GB w, **nobody reads x afterwards** | sync |
| 8 | eigensolver | `eig_and_truncate` (pca.mojo:702-803): `enqueue_es_scale` (power-of-two range scale) + sync; `PCA_RR_EIGH` (decomposition/pca_rr_switch.mojo:20, IDENTICAL default) -> `_eig_rr_device` (pca.mojo:618-693): per sweep 3 test launches (`eigh_par_off_part`, `_fold`, `pca_rr_gate`) + **host read of the 6-word flag (sync, PCA_RR_FLAG_TEST)** + (m-1)=219 rounds x 2 launches (`pca_rr_cs_kernel`: 1 block; `pca_rr_update_kernel`: (h*h + n*h)/256 = 142 blocks) | **441 launches + 1 host sync per sweep**; sweeps needed at 220 are unrecorded (pca_rr_switch.mojo:25-35; the budget is 60); a two-sided Jacobi to 1e-7 typically takes 6-10 sweeps -> **2,600-4,400 launches, 6-10 syncs**; each launch moves ~200 KB | per sweep |
| 9 | unscale diag, `sign_flip_kernel` (220 blocks x 32) | pca.mojo:744-751 | 2 | - | sync |
| 10 | D2H cov (194 KB), vecs (194 KB), info; **host** `order_truncate_spectrum` (pca.mojo:469-518): O(d^2) selection sort of eigenvalues, components gathered on the host in Float64 lists, then stored to the Python arrays elementwise | 3 copies | 388 KB | sync |
| 11 | D2H mean | estimator.mojo:177-181 | 1 | 880 B | sync |

The cyclic alternative (`-D MOJOLEARN_PCA_RR_EIGH_OFF`, `jacobi_eigh_kernel`, grid 1 x 256 threads,
decomposition/checks/jacobi_eigh_device.mojo:90-100) is one launch but n(n-1)/2 = 24,090 serial rotations per sweep on
one SM: worse at 220. Host-column twin: `decomposition/host/pca_oracle.mojo` (`host_eig_and_truncate`, same switch).

Cost model, L40S (800 GB/s, ~5 us per small launch): copy ~94 ms; stages 3+4+7 = 9 GB of traffic ~11 ms; Gram 5-20 ms
(L2 hit rate on the slab reloads decides); RR Jacobi **25-45 ms** (launch-bound: 441 launches x 6-10 sweeps x ~8 us incl.
the tiny kernels' own time, plus the per-sweep readback); 3 x 1.8 GB allocations and 7 synchronizes: a few to tens of ms.
Sum 150-200 ms of the 231. torch: mean + (x - mean) materialized + cuBLAS TN GEMM = 7.2 GB ~9 ms, syevd(220) 2-3 ms, = 16.9.
On the MI325X (6 TB/s, launch ~10-15 us) the traffic stages shrink to ~2 ms and the **Jacobi launch chain (40-90 ms) and the
copy** are nearly all of the 288 ms; that is why the AMD ratio (31x) is worse than NVIDIA's (13.7x).

taxi (11 columns): `gram_splitk_applies` is True -> `gram_centered_splitk_into` (fused centering, X read-only, no stages
4/7): `_enqueue_partial_centered` grid = PINNED_GRAM_SPLITK_CHUNKS = **128 blocks x GRAM_TPB** (core/gram_splitk.mojo:287,
919-940) + reduce. 128 blocks on 142 SMs for 176 MB is under-occupied but ~1 ms. Jacobi at n=11: m=12 -> 25 launches +
1 sync per sweep x ~6 sweeps = ~150 launches, 1.5 ms. Kernel-only ~3.5 ms vs torch 1.43: allocations (3 x 176 MB), 7 syncs,
the Jacobi chain. AMD taxi (20.3 vs 1.15): launch latency again.

## 2. TruncatedSVD trace (IDENTICAL, algorithm='covariance_eigh')

`TruncatedSVD.fit` (decomposition.py:636-700): `_tsvd_tsqr_components` returns None under IDENTICAL (`TSVD_QFIX` is
FAST-only, x_decomp/qfix.mojo:67) -> binding `tsvd_fit` -> `tsvd_fit_host` (estimator.mojo:293-340): allocate x, gram, xa,
xa2 (3 x n*d), **upload X**, `gemm_tn` (same PLAN_SPLIT_64_4X4 at 220 / split-K Gram at 11; no centering), sync,
`eig_and_truncate` (the same RR Jacobi chain, singular_scale 1), host tail. Then a SECOND binding call `tsvd_explained` ->
`tsvd_explained_host` (estimator.mojo:437-521): allocate x again, **upload X again**, upload components, sync,
`gemm_nt` X.Vt (core/gemm.mojo `pinned_gemm_nt_kernel`, one thread per output cell, k=220 serial loop, 1M x 10 outputs),
`_column_variance(xt)` and `_column_variance(x)` (estimator.mojo:367-405): each = mean pass (xty_tiled) + `shift_columns`
in place (r+w) + `square_in_place` (r+w) + mean pass = for x: 880 MB x 5 = 4.4 GB; `tsvd_finish_kernel` (1 block), 2 D2H, sync.

istella: 2 uploads 92 ms (est.) + Gram ~5-10 ms + Jacobi 25-45 ms + explained passes ~6 ms = the 96.4 is copy-dominated and
the kernel side (~40-50 ms) is still above cuML's 35.3 (cuML's clock also has inputs on the device). taxi 6.7: 4.6 ms copies.

## 3. randomized_svd trace (IDENTICAL; `mojolearn.randomized_svd`, python/mojolearn/_expansion_decomp.py:2922-3025)

`_rsvd_direct_input` is skipped (RSVD_FAST_DIRECT_IN is a FAST+Apple w4 flag) -> `_M.from_input(M)` (:226-238):
**`a.tobytes()` then `array.array.frombytes`: two full host copies of X** (880 MB each; ~0.15-0.3 s of memcpy inside
fit). Then `_rsvd_core` (:2973-3025), every op a binding call on the resident kit (`Kit.mm/orth/svd/...`, `_use` -> device
path because the operand is >= _RES_MIN): first `k.mm(A, Q)` uploads A once (`_did`, :646-657, pageable 880 MB ~46 ms).
Per power iteration (4) + 1 range pass: `k.mm(A, Q)` (1M x 220 . 220 x 18), `orth`, `k.mm(A, Q, ta=True)` (220 x 1M . 1M x 18),
`orth` -> 9 GEMMs, **9 orthonormalizations**; then colsum/abs/count_gt (a D2H sync), `B = Q^T A` (18 x 220), `_thin_svd(B)`
(`k.svd` one-sided Jacobi of the 220 x 18 transpose + a 1M x 18 . 18 x 18 product), `_flip_u` (two absmax_signs reductions over
U), `.out()` downloads U (32 MB), S, Vt.

* GEMM route (x_decomp/device.mojo:1977-2009 `launch_gemm`; DECOMP_FAST_GEMM_TILED/MMA are FAST defines): `gemm_part_kernel`
  with FOLD_BLOCK=4096 (x_decomp/cells.mojo:365), one thread per (chunk, i, j) cell, scalar serial loop, TPB=128. A.Q: 18M
  threads x 220 loads (A rows broadcast across the 18 j-threads; fine, ~5-15 ms). **A^T.Q: 245 chunks x 3,960 cells = 970k
  threads each doing 4096 dependent strided loads of A (adjacent threads differ in j, so a warp touches ~2 columns of one
  row: 16 useful bytes per 128 B line)**: latency-bound, est. 20-60 ms each x 5.
* Orthonormalization (`orth_on_device` -> `orth_on_device_diag`, x_decomp/device.mojo:2450-2545): two passes of the SLICED
  Householder QR `qr_factor` (core/householder_qr.mojo:660-700 -> `qr_split_enqueue` :612-640): `qr_slice_count` = 64 slices
  (QR_MAX_SLICES=64, :183,196-200); per column j one `qr_reflector_kernel` (64 blocks x QR_TPB=32 threads) and one
  `qr_apply_kernel` ((17-j)/8 x 64 blocks); 18 columns -> ~37 launches per pass, **2,048 threads working a 1M x 18 matrix**,
  column reads at stride 18 floats (uncoalesced), the matrix read ~18 times per pass; then the stacked-R QR, a 1-thread guard,
  `trsm_kernel` (1M/128 blocks), sync. ~154 launches and 6 syncs per orth (the classical-structural note agrees). Est.
  **40-100 ms per orth x 9 = 0.4-0.9 s: the dominant randomized-svd cost, and independent of d**, which is why taxi (44 MB)
  still costs 215 ms. torch: `torch.linalg.qr` (cuSOLVER geqrf/orgqr) on 1M x 18 is ~1 ms per orth, cuBLAS for the products.

## 4. Existing switches and rows touching these lanes (do not repeat)

* `experiments/six_lane_integration/grid_controls/classical-decomp.json`: `C01_MEAN` (bm_column_mean; neutral),
  `PCA_COV` = c04 (`bm_centered_gram_panels`) / c23 (`bm_onepass_covariance`, Chan merge; slower istella, faster taxi, being
  deleted), `TSVD_FUSED_STATS` (`bm_tsvd_variances`; 1.44x slower; it STILL re-uploads X, estimator.mojo:453-457).
  All three use `bm_leaf_gram_kernel` (grid leaves x 105 tile pairs, each thread chaining `leaf` rows serially; at istella's
  48,400 cells `bm_onepass_leaf_rows` grows the leaf to 8192 rows so leaves x cells <= 2^24, core/blocked_moments_ops.mojo:36-67)
  -> long dependent chains per thread: that is the istella loss, not the one-pass idea itself.
* `classical-structural.json`: `rsvd_tsqr_ortho` (`MOJOLEARN_CLASSICAL_RSVD_TSQR_ORTHO`, x_decomp/tsqr_core.mojo:149; default
  off; RUN OWED for randomized-svd on taxi, istella, nv + amd): replaces the 9 sliced-QR orths with one blocked TSQR pass each
  (~25 launches / 1 sync). It targets exactly the dominant rsvd stage; `C24_TSQR` (panel8 / rows2048 / tree4) tunes that TSQR.
  `gap-tsqr.json`: `tsqr_wy_pair`, `tsqr_tree_par`, `xd_svd_group` (the small SVD) reach `linalg.svd`, not these board lanes.
* `docs/apple-fast/EXPERIMENTS.md`: PCA_FAST_EIG / NO_ALIAS / TOPK (DROPPED-noise, Apple FAST), PCA_FAST_GRAM_MMA (KEPT,
  FAST+Apple: it also skips the restore pass), DECOMP_FAST_GEMM_MMA (KEPT, FAST+Apple, rsvd -25%), DECOMP_FAST_GEMM_TILED
  (DROPPED), SVD_FAST_CHOLQR (+763% on `svd`, Apple FAST; the per-column `chol_step_kernel` chain, not CholQR as such).
  None of these change the NVIDIA/AMD IDENTICAL route.

## 5. Ideas

### PCA (both datasets; NVIDIA and AMD IDENTICAL)

| # | idea | define | change | effect (cost model) | bits | identity risk | effort | Opus-able |
|---|---|---|---|---|---|---|---|---|
| P1 | **One launch per Jacobi sweep.** For n^2 floats that fit a block's working set from global memory, run all m-1 rounds of a sweep in ONE block of 1024 threads with `barrier()` between rounds, applying `rr_cs`/`rr_block`/`rr_vrow` (x_decomp/rr.mojo) per 2x2 block exactly as `pca_rr_update_kernel` does, test fold inside the same kernel at the sweep end (the gate state stays). Rule from size: n*n*4 B <= a per-SM L1/L2-resident budget, not a board width. | `MOJOLEARN_IDN_PCA_RR_ONE_BLOCK` | pca.mojo:618-693 (`_eig_rr_device`), host column unchanged (same words) | 441 launches + 1 sync per sweep -> 1-2 launches; at 220: Jacobi 25-45 ms -> 5-10 ms NV, 40-90 -> ~10 ms AMD; at 11: 150 launches -> ~6 | none (same cell functions, same round order) | low; `check_jacobi_*` and the host `host_eigh_rr` already pin the words | M | yes |
| P1b | Cheaper half-step if P1 is too big: fold `pca_rr_cs_kernel` into `pca_rr_update_kernel` (each block recomputes the h rotations' (c, s) from `a` before updating; `rr_cs` is a few flops per pair) and fuse the three test launches into one. | `MOJOLEARN_IDN_PCA_RR_FUSED_ROUND` | pca.mojo:554-586 | launches per sweep 441 -> 220; Jacobi ~halved | none | low | S | yes |
| P2 | **Skip the restore pass and the dead alias buffer.** `x` is the fit's private device copy; nothing reads it after the Gram (the FAST arm already passes `restore_input=False` for this reason, estimator.mojo:157-159). Also allocate `xa`/`xa2` to what the chosen route needs: `identical_gemm_workspace_max_floats` (198 MB) on the v1 path, `n_chunks*m*m` on the split-K path, instead of two more n*d buffers. | `MOJOLEARN_IDN_PCA_LEAN_SCRATCH` | estimator.mojo:127-135,159; pca.mojo:339-366 | -3.6 GB traffic (~4.5 ms NV, ~1 ms AMD), -3.6 GB of allocations per fit (allocation time + memory pressure) | none (the restored words are never read; `pca.jacobi.a` traces unchanged) | none | S | yes |
| P3 | **Fused centering in the v1 TN Gram** for widths past the split-K kernel: subtract `mu[col]` at the tile load of `identical_gemm_into(OP_TN)` (a `center` pointer argument, as `_enqueue_partial_centered` does for <=128), removing stage 4 as well and keeping X read-only. Same fp32 subtraction the shift kernel stored, so the Gram words are unchanged (the <=128 twin is proven cell by cell by `check_gram_centered_fused`). | `MOJOLEARN_IDN_GEMM_TN_CENTERED` | gemm/checks/gemm_identical.mojo PLAN_SPLIT_* loaders (OP_TN only), core/gemm.mojo:529-562, pca.mojo:331-346 | another -3.6 GB (~4.5 ms NV); with P2, X is read once for the mean and once for the Gram | none | low (the tile loader changes, the fold does not); the host column (`centered_gram_v1_cell`) already computes centered words | M | yes |
| P4 | **Pinned, chunked upload overlapping the mean pass.** Stage X through a pinned host buffer in chunks of whole 256-row mean tiles (and, for the Gram, whole 1996-row leaves): the copy runs at pinned rate (~25 GB/s on PCIe4) and the `xty_tile_partial` of chunk i runs while chunk i+1 copies. The Gram still waits for `mu` (two passes), so only the mean hides under the copy. | `MOJOLEARN_IDN_UPLOAD_PINNED_CHUNKS` | estimator.mojo:136-137 (+ a shared helper for tsvd/ols) | istella copy 94 -> ~70 ms, taxi 9.2 -> ~7 ms; mean pass hidden | none (same leaf/tile partition: chunk boundaries are multiples of the tile and leaf sizes) | none | M | yes |
| P5 | **Device-side spectrum order and gather.** Replace `order_truncate_spectrum` (host O(d^2) sort over Float64 lists, components gathered on the host) with a device rank kernel (argsort of 220 values by value desc, index asc on ties: the ridge svdEig row in EXPERIMENTS already does this) + a gather of the top k columns + D2H of only k x d. Removes 2 of the 4 D2H round trips and the host loops; small on NV, visible on AMD/taxi where the fit is sub-15 ms. | `MOJOLEARN_IDN_PCA_DEVICE_TRUNCATE` | pca.mojo:469-518,756-803; estimator.mojo:161-176 | ~1-2 ms per fit | none if ties are ordered as today (first index wins) | low | S | yes |

Not worth it: a different Gram plan at 220 (the split plan is already parallel; its ~9x slab reload is L2-served when the
16 tile blocks of a leaf are co-scheduled; check block ordering before touching it). The copy itself cannot go below bytes/pinned
rate; the board's kernel/kernel column (upload_ms_separate) is what shows the rest.

### TruncatedSVD

| # | idea | define | change | effect | bits | identity risk | effort | Opus-able |
|---|---|---|---|---|---|---|---|---|
| T1 | **One upload, one binding call.** `tsvd_fit` + `tsvd_explained` as one `tsvd_fit_explained` host function that keeps the device X between the Gram and the variances (the Python class calls both back to back, decomposition.py:684-700). | `MOJOLEARN_IDN_TSVD_ONE_UPLOAD` | estimator.mojo:293-340 + 437-521 merged; decomposition.py:684-700; bindings/_mojolearn_estimators.mojo:1547-1549 (additive export) | -880 MB H2D istella (~46 ms of 96), -44 MB taxi (~2.3 of 6.7); -1 n*d allocation | none (same kernels, same order) | none | S | yes |
| T2 | **Read-only two-pass variance.** Replace mean + shift(w) + square(w) + mean with mean (xty_tiled) + one kernel folding `ftz((x - mu)^2)` in registers in the SAME tile order as `xty_tiled` (the current second mean pass folds exactly those stored words). | `MOJOLEARN_IDN_COLVAR_FUSED` | estimator.mojo:367-405 `_column_variance` (also used by `_column_variance(xt)`) | -2 r/w passes over X and over X.Vt (~3.5 GB istella, ~5 ms NV) | none (same words into the same fold) | low; `check_variance_identity` exists (decomposition/checks/variance_identity_check.mojo) | S | yes |
| T3 | **Variances from the Gram, no second pass over X.** var(X V^T)_j = (v_j^T G v_j - (v_j^T s)^2 / n) / n and var(X)_j = (G_jj - s_j^2 / n) / n with G = X^T X already on the device and s = column sums (one xty_tiled pass). Removes the X.Vt GEMM and both variance passes: the whole `tsvd_explained` becomes k x d work. | `MOJOLEARN_IDN_TSVD_GRAM_VARIANCE` | estimator.mojo:437-521 + host column `decomposition/host/pca_oracle.mojo` (tsvd variances) | tsvd istella kernel side 40-50 -> ~30 ms (Gram + Jacobi remain) | **changes** (a different formula; host column moves with it) | medium: cancellation when column means are large relative to the spread (uncentered data); needs the quality metric (explained_variance_ratio_sum, reconstruction error) gate; cuML uses the transformed form | M | yes, with the quality gate stated |
| T4 | P1/P1b/P2/P4 apply unchanged (`eig_and_truncate` and the Gram are shared). | as above | - | Jacobi 25-45 -> 5-10 ms at 220 | - | - | - | - |

### randomized_svd (and PCA/TruncatedSVD algorithm='randomized')

| # | idea | define | change | effect | bits | identity risk | effort | Opus-able |
|---|---|---|---|---|---|---|---|---|
| R1 | **Upload straight from the caller's buffer on every column.** `_rsvd_direct_input` (an existing route, _expansion_decomp.py:2939-2970) is gated on the FAST+Apple w4 flag; make the IDENTICAL binding export the resident entries' direct upload too (no `tobytes`/`frombytes`), keeping the host finiteness scan. | `MOJOLEARN_IDN_RSVD_DIRECT_IN` (flag bit in `x_decomp_w4_flags` or a sibling) | x_decomp w4 flags; _expansion_decomp.py:2931-2933 | **-2 host copies of X: ~150-300 ms istella, ~10-15 ms taxi**; the device upload stays | none (a copy is not arithmetic; the same words reach the device) | none | S | yes |
| R2 | **Run the existing `rsvd_tsqr_ortho` control first** (its RUN OWED ID + RACE on nv/amd for taxi, istella). It replaces the dominant stage (9 x ~154 launches on 2,048 threads) with 9 blocked TSQR passes. Nothing new to write; flag so no lane duplicates it. | `MOJOLEARN_CLASSICAL_RSVD_TSQR_ORTHO` (exists) | x_decomp/tsqr_core.mojo:149; device.mojo:2458-2461 | est. 0.4-0.9 s -> ~0.05-0.1 s on istella and taxi alike | changes (recorded in its control; host column moves) | covered by its ID check | 0 | - |
| R3 | **CholeskyQR2 orth for the sketch** as the alternative if R2 loses: R from the pinned split-K Gram of the m x 18 sketch (`gemm_tn_splitk`, 18 <= 128, 128 chunks), the IDENTICAL Cholesky cells (OLS's `chol_diag`/`chol_col_elem`, 18 columns), `trsm_kernel` row-parallel; twice. ~8 launches, 1 sync, 4 reads of the 72 MB sketch per orth. | `MOJOLEARN_IDN_RSVD_CHOLQR2` | x_decomp/device.mojo:2450-2545 (`orth_on_device`), host `HostExec.orth` | orth 40-100 ms -> ~1-2 ms each | changes (host column with it) | low-medium: a rank-deficient sketch needs the guard -> keep R2's zero-column contract; Apple's SVD_FAST_CHOLQR loss was the serial per-column chol on a 220-wide `svd`, not this 18-wide case | M | yes |
| R4 | **Staged tall GEMM in the kit, same fold.** Keep `gemm_part_kernel`'s FOLD_BLOCK=4096 chain per cell (bits) but stage A's 4096 x 18 (ta) / 220 x 18 (nn) chunk tiles through shared memory and give each thread a 2x2 or 4x1 register tile so a warp's loads are contiguous along the 220-wide rows; or route (ta) products to the pinned `identical_gemm_into` PLAN_SPLIT_* plans (bits change: contract leaves instead of 4096-row chunks; host column `HostExec` gemm must follow). | `MOJOLEARN_IDN_XD_GEMM_STAGED` (same bits) / `MOJOLEARN_IDN_XD_GEMM_PLANS` (bits change) | x_decomp/device.mojo:1977-2009, cells.mojo gemm_part_kernel | A^T.Q 20-60 ms -> 3-8 ms each x 5 | staged: none; plans: changes | staged: none; plans: the pinned plans are the IDENTICAL GEMM already | M / S | yes |
| R5 | Drop the host-visible `count_gt` sync and the `cs`/`dead` compaction when `nlive == full` is decided on the device (a flag word read once with the final download), and fuse `_flip_u`'s two `absmax_signs` passes over U with the `U = Q Uh` product's epilogue. Small (a few ms); do after R1-R4. | `MOJOLEARN_IDN_RSVD_TAIL_FUSED` | _expansion_decomp.py:2996-3024 + kit entries | ~2-5 ms | none | low | S | yes |

Order of expected board impact: randomized-svd R1 + R2 (0.94 s -> ~0.15-0.25 s istella, 215 -> ~40 ms taxi; then R4
towards torch's 66 / 32 ms); tsvd T1 (+T2) (96 -> ~45 ms, 6.7 -> ~4 ms); pca P1 + P2 (+P3) (231 -> ~170-180 ms NV with the
copy still inside, kernel side 137 -> ~80 ms; AMD 288 -> ~200). None of P1-P5/T1-T2/R1/R4-staged move bits, so they need no
host-column change and can be ID-checked as no-ops on the output hash.
