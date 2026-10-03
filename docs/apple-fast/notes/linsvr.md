# linsvr: LinearSVR on the Apple GPU (FAST), profile by reading (2026-10-03)

Lane af-linsvr, branch lane/apple-fast-linsvr off origin/main 8897404da. Board lane `linearsvr`
(tools/bench_board_more.py, AFC_FAMILY=classical2): taxi, 1,000,000 fit rows x 11 features,
penalty='l2', loss='epsilon_insensitive' (QN_LOSS_SVR_L1), epsilon=0, C=1, tol=1e-4, fit_intercept.
M3 Ultra board 0834: FAST 183 ms, IDENTICAL 110 ms, sklearn (liblinear, CPU) 83,935 ms.

The brief names binding `svm`. The code says otherwise: `python/mojolearn/svm.py:191` LinearSVR
inherits `_LinearSVMBase._BINDING = "_mojolearn_estimators"` and calls
`linear_model._qn_fit_one_target` -> `_mojolearn_estimators.qn_fit` -> `glm/estimator.mojo:471
qn_fit_host` -> `glm/impl/qn/qn.mojo qn_fit_x` -> L-BFGS (`qn_solvers.mojo min_lbfgs`, Armijo
backtracking `qn_linesearch.mojo ls_backtrack`) over the objective `glm/impl/qn/glm_base.mojo
GLMWithData.evaluate_pen`. `svm/impl/svr_impl.mojo` is the kernel SVR (SMO) and is not on this
lane's path. Every change here is in `glm/impl/qn/` and the A/B lines build binding `estimators`.
The same objective code serves logreg and linearsvc (C == 1 losses), so the defines move them too.

## 1. Why FAST is 1.7x slower than IDENTICAL on the same GPU

One objective evaluation (`evaluate_pen`, l2 != 0) on taxi, launches in queue order:

| step | IDENTICAL on Apple (main) | FAST on Apple (main) |
|---|---|---|
| g <- 0, reg grad + value | `enqueue_memset`, `tikhonov_reg_grad_kernel` (1 block) | same |
| z = X w | `pinned_gemv_n_kernel`, 1 thread per row (`core/gemm.mojo:733`) | the SAME kernel: FAST + Apple takes it when k <= 64 (`core/gemm.mojo:746-760`) |
| z += b | `add_bias_kernel` | same |
| loss map, dZ | `svr_l1_loss_dz_kernel`, no sum (`enqueue_loss_and_dz(..., False)`) | `svr_l1_loss_dz_kernel`, then `sum_terms_kernel`: ONE block of 256 threads striding all 1M rows (`glm_base.mojo:214`) |
| X^T dZ, sum dZ, sum loss | `qnt_partial_kernel`: 3907 tiles x 13 outputs = 50,791 threads in 199 blocks, X read once row-coalesced (`glm_base.mojo:485`) | `xtdz_partial_kernel` (`core/xtdz_coalesced.mojo:112`): s = 256 // 11 = 23 residues per block, 12 blocks of 253 threads, each thread walking 3,907 rows; then `xtdz_fold_kernel` (11 blocks) |
| epilogue, bias mean | `qnt_fold_kernel`: 13 blocks, each folds 3,907 partials; alpha/beta epilogue, bias mean, loss store inside | `gemm_epilogue_kernel` (1 block), then `mean_kernel`: ONE block of 256 threads striding all 1M rows again (`glm_base.mojo:240`) |
| gradient norm | `nrm1_kernel` (1 block over 12 words) | same |
| readback | `read_scalars`: one copy, one synchronize | same |
| launches / row sweeps | 9 launches; every sweep over the rows uses >= 199 blocks | 12 launches; THREE sweeps over 1M rows run on 1 block (sum_terms, mean) or 12 blocks (xtdz_partial) |

Cause: `QN_TILED` (`glm_base.mojo:464`, lane/gap-linear-nv) is compiled under
`GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL` only. FAST on Apple never got it and still runs the
pre-tiled objective: the loss sum and the bias mean are one threadgroup each (one GPU core of the
M3 Ultra's 80 reads 4 MB serially, 3,907 dependent loads per thread even with the 32-wide unroll
of `core/strided_walk.mojo`), and the gradient's `xtdz_coalesced` at d = 11 puts only 12 blocks
on the device. Per evaluation that is on the order of a millisecond of near-serial work that
IDENTICAL does not do; L-BFGS on the non-smooth epsilon-insensitive loss evaluates the objective
once per line-search candidate, tens to a hundred-plus times per fit, which is the 73 ms gap.
Nothing else differs: the gemv kernel, the loss kernel, the direction kernel, the number of
synchronizes and the uploads are the same in both tiers.

## 2. The rest of the fit (both tiers)

Per L-BFGS iteration with the first candidate accepted (Armijo, step 1): `lbfgs_dir_kernel`
(one block: S/Y update, two-loop recursion, next xp/gradp saves, dg_init; `qn_util.mojo:439`),
`axpy_kernel` (x = xp + step * drt), one `evaluate_pen` (above), and ONE synchronize carrying the
loss, the regularizer value, the gradient norm (speculative, `_gnorm_kind`), dg_init and the
direction's verdict (`read_scalars`, 4 words). Every rejected candidate adds `axpy` + `evaluate_pen`
+ one synchronize. The convergence test (`check_convergence`) runs on the host on numbers that
already came home with that one synchronize: there is no separate per-iteration readback to move
to the device. A device-side convergence or line search would have to run whole iterations without
the host, which the host-driven L-BFGS structure does not allow cheaply; not attempted (see
docs/apple-fast/ab/linsvr.md).

Per fit, fixed: `qn_fit_host` uploads X (44 MB) and y, memsets w, synchronizes (`glm/estimator.mojo
~511`); `GLMWithData.__init__` creates 8 buffers and synchronizes; `min_lbfgs` creates 10 buffers
plus 2m sub-buffer views and synchronizes; the initial `evaluate` + `nrm2` (two synchronizes); the
final copy of w home. About 35 live Metal buffers per fit: launch cost growth is negligible here.

## 3. Candidates (each its own define, FAST + Apple only; docs/apple-fast/ab/linsvr.md explains)

1. `MOJOLEARN_LSVR_FASTPATH_FIX`: FAST on Apple takes the tiled objective (`qn_tiled_applies` true
   under the define): the IDENTICAL launch structure above, FAST arithmetic. Expected: FAST ~=
   IDENTICAL (110 ms).
2. `MOJOLEARN_LSVR_FUSED_GRAD`: one pass over X per evaluation: each thread owns 16 rows, holds the
   11 features of a row in registers, computes z, the loss term and dZ, and accumulates the 11
   gradient cells, sum dZ and sum loss in registers; the tile partials are folded by the tiled
   fold kernel. Replaces gemv + bias + loss map + partial (two reads of X, three round trips of z
   and loss_terms through device memory) with one kernel. d <= 32 (taxi), C == 1.
3. `MOJOLEARN_LSVR_LINESEARCH_BATCH`: the fused pass also evaluates the loss at the next three
   backtracking steps (step/2, step/4, step/8: z_c = x . xp + step_c * x . drt from the same
   registers) and the device adds each candidate's l2 value; the host reads all four behind the
   one synchronize and, when step 1 fails Armijo, jumps straight to the first passing candidate
   with one evaluation instead of one evaluation + synchronize per rejected step.
4. `MOJOLEARN_LSVR_EVAL_SLIM`: the four one-block launches around the objective (memset g,
   Tikhonov gradient + value, gradient norm, OWL-QN l1 term) become one launch after the fold:
   g += l2 w, reg value, norm, l1 term, all from one block.
5. `MOJOLEARN_LSVR_ALL`: 1 + 2 + 3 + 4.

Not done: `MOJOLEARN_LSVR_DEVICE_CONVERGE` (no separate readback exists, section 2) and
`MOJOLEARN_LSVR_DUAL_CD` (liblinear's dual coordinate descent is sequential per coordinate; a
parallel block variant changes the solution path and would need its own convergence proof against
the primal tolerance; not simple, not attempted).
