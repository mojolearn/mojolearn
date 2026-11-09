# Executor N7 ops: Cholesky, eigh, sort, compaction (API note for N8-N10)

Lane `n7-device-linalg-ops` (2026-11-08), plan row "Cholesky / eigh / sort / compaction as executor ops"
(docs/plans/EXECUTOR_CONSOLIDATION_AND_CLEANUP.md section 4, third wave). Ops only; the family lanes N8-N10 move
their loops onto them and delete the old paths.

## The contract every op keeps

- **Enqueue only.** No op reads device memory back or drains the stream. Scratch an op allocates for itself is
  released in stream order (the `DevExec._rr_batch_enq` pattern); scratch a family reuses per iteration is a
  struct the family allocates once (`SortScratch`, `CompactScratch`).
- **The gate.** Every op takes `gate: DeviceBuffer[DType.int32]` whose word 0 is the loop's done word. Inside a
  bounded device loop (`autograd/program_loop_device.mojo`, a `LoopStep`) pass `registers.words`; outside a loop
  pass `step_gate_open_word(ctx)` (core/step_gate.mojo). The caller's outputs are written only while the gate is
  open, so iterations enqueued past the stop leave them as the stopping iteration left them.
- **The failure word.** An op that can fail on the device (Cholesky pivot, unconverged eigh) writes a positive
  code in its status word. `step_fail_joins_loop(ctx, registers.words, status, it)` (or the per-op aliases
  `chol_op_info_joins_loop`, `eigh_op_status_joins_loop`) stops the loop at that iteration; enqueue it right after
  the op, before the body's kernels that consume the result. After the loop, `registers.failure()` is the code
  (register word `STEP_FAIL` = 6; 0 = none); a nonzero code means `LoopOutcome.converged` is the failure's stop,
  not convergence: check `failure()` first and raise by name. Outside a loop, read the status word at the
  family's own sync.
- **Host column.** Each op has a host twin with the same algorithm and words, parallel over its independent units
  through `host_cells`. Use the twin in the family's `HostExec` arm.

## Cholesky (`cholesky/chol_ops.mojo`)

| call | does | host twin |
|---|---|---|
| `chol_op_begin(ctx, gate, info)` | info[0] = 0 (gate open) or `CHOL_OP_GATED` = -1 (done) | n/a |
| `chol_op_factor(ctx, a, info, n)` | `A = L L^T` in place, lower, upper triangle +0.0; info[0] = k > 0 on a failed minor | `chol_host_factor_lower` (cholesky/host/chol_oracle.mojo) |
| `chol_op_solve_lower / _upper(ctx, l, b, info, n, nrhs)` | `L X = B` / `L^T X = B` in place, skipped when info != 0 | `chol_host_trsm_lower / _upper` |
| `chol_op_cho_solve(ctx, l, b, info, n, nrhs)` | potrs (forward then back) | `chol_host_solve` |
| `chol_op_logdet(ctx, l, info, scratch, out, n)` | out[0] = 2 sum log L_ii, stored only when info == 0; scratch `chol_op_logdet_scratch_floats(n)` | `CholHostFactor.logdet` |
| `chol_op_info_joins_loop(ctx, words, info, it)` | the pivot failure stops the loop | n/a |
| `chol_op_zero_on_fail(ctx, info, buf, count)` | `buf[0, count) = +0.0` when info != 0 (a failed fit's zero dual), read-free (lane N8) | n/a |

Per iteration: `chol_op_begin`, the family builds `a` (kernel + ridge, its own kernel), `chol_op_factor`,
`chol_op_info_joins_loop`, then solves / logdet / the family's gradient. The factor schedule per column: strips
(`chol_strips_enqueue`) on NVIDIA, AMD and Apple FAST; left-looking (`chol_left_enqueue_deferred`) on Apple
IDENTICAL; the one guarded panel below n = 32. These are `potrf_lower`'s own launches, so `CHOL_DEFER_INFO` now
holds on every vendor and the profile's words are unchanged. A failed factor's matrix is the factor frozen at the
failing panel (not LAPACK's completed partial factor): nothing may consume it, which the gate guarantees.

## eigh (`x_decomp/eigh_ops.mojo`, host `x_decomp/eigh_ops_host.mojo`)

| call | does | host twin |
|---|---|---|
| `eigh_op_dense(ctx, gate, a, w, v, status, batch, n)` | `batch` dense symmetric n x n problems (a CONSUMED) by the round-robin Jacobi, one block per problem (`rr_batch_gated_kernel`), range scale, sign flip, ascending; status[0] = problems not converged | `eigh_op_dense_host` (`host_eigh_rr_sorted` per problem, problems in parallel) |
| `eigh_op_sym(ctx, gate, a, w, v, status, n)` (lane N9) | ONE dense problem by `DevExec.eigh`'s route: from n = TD_MIN_N the Householder tridiagonal solve on a copy with the Jacobi of `a` enqueued behind it, chosen on the device on a refusal; `eigh_op_dense` below; status[0] = 1 when no route holds. `DevExec.eigh_enq` wraps it with an open gate | `eigh_op_sym_host` (`HostExec.eigh`) |
| `eigh_op_tridiag(ctx, gate, d, e, w, z, status, n)` | tridiagonal T (d n words, e n-1 words, read only): bisection + twisted vectors (`td_solve_enqueue`), sign flip, ascending; a refusal falls back on the device to the dense Jacobi of T; status[0] = 1 when neither holds | `eigh_op_tridiag_host` (`host_td_solve`, then the Jacobi of T on a refusal) |
| `eigh_op_status_joins_loop(ctx, words, status, it)` | an unconverged solve stops the loop | n/a |

Outputs: w ascending (batch x n), vectors in COLUMNS of a row-major n x n (batch x n x n), sign-flipped. A family
that wants descending order keeps w / v and reads them backwards in its own kernel, or reorders with the sort op
below (`sort_op_f32` with `descending=True`, then `sort_op_gather_rows_f32`); no host reverse.

Which op for which family (plan section 4): FastICA symmetric decorrelation, FA EM / rotation, IPCA, LLE local
spectra (batch = samples), BayesianRidge (p x p), MCD / PLS `pinvh`, Spectral discretize (k x k), dictionary
learning: `eigh_op_dense`. KernelPCA / Isomap / SpectralEmbedding Lanczos (the T of alphas / betas the Lanczos step
already keeps on the device, `x_decomp/lanczos_dev.mojo`): `eigh_op_tridiag` on (alphas, betas), then the Ritz
vectors as one GEMM of the basis with z.

## Sort (`core/sort_ops.mojo`, host `core/sort_ops_host.mojo`)

| call | does | host twin |
|---|---|---|
| `SortScratch(ctx, capacity)` | the radix pairs, ping-pong and counts, once per fit | n/a |
| `sort_op_f32(ctx, gate, keys, order, n, descending, scratch)` | order[r] = input index of the r-th key: the unique stable order of `sort_key_bits` (float order, -0.0 before +0.0, NaNs by sign at the ends; descending inverts the key, ties keep input order) by the 8-bit LSD radix (`fast_radix_sort_pairs_u32`) | `sort_op_f32_host` (same tiles and passes) |
| `sort_op_gather_rows_f32(ctx, gate, src, order, dst, rows, cols)` | dst[r] = src[order[r]] row by row | `sort_op_gather_rows_f32_host` |

SVM working set: its sort is already this radix on every GPU; NMF CD permutation (Philox draw + this sort),
MDS isotonic: use these.

## Compaction (`core/compact_ops.mojo`, host `core/compact_ops_host.mojo`)

| call | does | host twin |
|---|---|---|
| `CompactScratch(ctx, capacity)` | offsets and block sums, once per fit | n/a |
| `compact_op_indices(ctx, gate, flags, n, scratch, out, count)` | out[0, count) = flagged indices ascending, count[0] on the device | `compact_indices_host` |
| `compact_op_i32 / _f32(ctx, gate, src, flags, n, scratch, out, count)` | the flagged elements in index order | `compact_i32_host / _f32_host` |
| `compact_scan_enqueue(...)` | the scan and count alone (svm `SelectScratch.scan` = this + one read) | n/a |
| `compact_flag_nonzero_kernel`, `compact_count_bound(count, t)` | flags from a float; a consumer kernel launched over the capacity tests `t < count[0]` | n/a |

Flags are bytes 0 / 1. SMO (`svm/impl/smosolver.mojo`): replace the drained `select_i32` count with
`compact_op_*` and launch the later grids over `n_ws` with `compact_count_bound`; then the drain per outer
iteration goes, and so does `SelectScratch` once its last caller moves.

## What the family lanes delete when they move

- GP / GPC / KRR (N8): the per-panel `info` reads and the `info == 0` host branches around
  `copy_vector_from_matrix_diagonal_kernel` + `enqueue_logdet` + `cho_solve` (gaussian_process/gp_optim.mojo,
  gpr_resident.mojo, classifier.mojo, gpc_ovr.mojo, the KRR fit). `potrf_lower` stays for the trace / sabotage
  checks and the multi-GPU owner set only.
  DONE 2026-11-09 (lane n8-gp-krr-smo): the optimizer runs every start on the device (run sequencer
  `gp_opt_advance_kernel`, done word read per 8 evaluations); the GPC Newton loop is one `bounded_device_loop`
  (gaussian_process/gpc_newton.mojo); the fits, lml_grad, sample_y and KRR read `info` at their own sync. KRR's
  identity-trace route (km_check's `chol.panelNNN.*` records) keeps `potrf_lower`. `potrf_lower`'s other routes
  stay: the public Cholesky (FAST_NOSYNC / FAST_TALL), GMM's M-step, x_decomp/device.mojo and kit_device.mojo, the
  checks and the multi-GPU owner set still call it.
- Decomposition (N9): the host round trips of w, V (kit_device.mojo eigh, ipca_owner.mojo, lanczos_dev.mojo /
  mani_decomp_native.mojo projected solve, lle_iter_dev.mojo), `HostExec.eigh_batch`'s serial loop over problems
  (replace with `eigh_op_dense_host`), and `DevExec._eigh_par_on`'s per-sweep three-scalar read for the k x k
  callers.
- SMO (assigned to N8 on 2026-11-08; NOT moved by lane n8-gp-krr-smo, owed to its own lane): the drained count in
  `select_i32` is the only drain of an outer iteration, and the host reads `diff` / `inner` (`return_buff`) on that
  same drain for `check_stopping_condition` (diff_prev, n_small_diff, n_increased_diff and the max_iter budget that
  `launch_block_solve` takes as a launch argument). Removing the count drain alone saves nothing. The move is all of:
  the stop rule's state as device words with one criterion kernel; the block solve reading its iteration budget from
  a device word; `fold_order_rank_kernel`, the row gather and both fused update_f kernels (`fast_update_f`,
  `idn_update_f`) bounded by `compact_count_bound` over n_ws; and the non-fused tile path (poly / sigmoid /
  precomputed / multi-device), whose `kernel_matrix_dispatch` over nnz rows would have to run over n_ws rows: up to
  n_ws / nnz times the tile GEMM per outer iteration, which can cost more than the drain it removes. Measure that
  arm (svc / svr board rows, late iterations where nnz << n_ws) before choosing it.
- SMO, Spectral discretize, dictionary learning, ARIMA lock-step (N10): the drained count, the host n x k loops
  of `x_cluster/spectral_assign.mojo:232-275` (`jacobi_svd` on the host).

## Bits

The ops reuse the profile kernels, so a family that moves an existing device Cholesky, solve, compaction or sort
keeps its words. Bits change (every column together, host twin included) when a family moves:
a host LAPACK / NumPy / host-Jacobi eigh onto `eigh_op_*`; a dense eigh at n >= TD_MIN_N from `DevExec.eigh`'s
tridiagonal route onto `eigh_op_dense` (the Jacobi); a Lanczos T solve onto `eigh_op_tridiag`. Identity lanes
to re-record then: kernel-pca, isomap, spectral-embedding, spectral-clustering, fast-ica, factor-analysis,
incremental-pca, locally-linear-embedding, bayesian-ridge, ard-regression, min-cov-det / elliptic-envelope,
pls, dictionary-learning (each lane whose family moved). Apple FAST cholesky / gaussian-process words change
when they move (strip schedule and identical solves instead of the vendor-GEMM panel route and
`fast_cho_solve`); FAST needs only quality.

## RUN OWED

- Apple FAST (M3, the FAST peer): cholesky synthetic and gaussian-process rows, op route (strips + identical
  solves) against `potrf_lower`'s FAST route; keep the faster, delete the other.
- NVIDIA + AMD: none for this lane alone (no consumer yet; existing paths keep their launches). Each family lane
  measures its own rows when it moves.
- Compile: `checks/executor_n7_ops_compile.mojo` on NVIDIA, AMD and Apple targets, IDENTICAL and FAST.
