# lane/apple-fast-linsvr: LinearSVR (the GLM quasi-Newton objective) on the Apple GPU, FAST

Off origin/main 8897404da (0.8.35). Binding `estimators` (LinearSVR answers from
`_mojolearn_estimators.qn_fit`, not from the svm binding; `python/mojolearn/svm.py:36`). Every
switch is a `-D` build define, compiled under FAST + Apple only, default OFF; IDENTICAL compiles
main's code unchanged. The same objective serves logreg and linearsvc (every C == 1 loss), so a
win here moves those lanes too. Profile and launch tables: docs/apple-fast/notes/linsvr.md.
Request lines: docs/apple-fast/ab/linsvr.txt (taxi first, the deciding dataset; istella rows for
the switches that reach d = 220).

Board 0834 (M3 Ultra): linearsvr taxi FAST 183 ms, IDENTICAL 110 ms, sklearn-cpu 83,935 ms.

## The finding: why FAST was slower than IDENTICAL

`QN_TILED` (`glm/impl/qn/glm_base.mojo`, lane/gap-linear-nv) rewrote the IDENTICAL C == 1 objective
as one row-coalesced pass (`qnt_partial_kernel`, 199 blocks on taxi) plus a 13-block fold, and it is
compiled under `GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL` only. FAST on Apple kept the older
sequence: `sum_terms_kernel` (the loss value) and `mean_kernel` (the bias gradient) are ONE
threadgroup of 256 threads each striding all 1,000,000 rows, and the gradient's `xtdz_coalesced` at
d = 11 runs 12 blocks of 253 threads, each thread walking 3,907 rows. Three near-serial sweeps of the
rows per objective evaluation, one evaluation per line-search candidate, tens to a hundred-plus
evaluations per fit. Everything else (the one-thread-per-row gemv, the loss kernel, the fused
L-BFGS direction kernel, one synchronize per evaluation, the uploads) is the same in both tiers.

## The switches

| switch | kind | site | what it changes under FAST on Apple |
|---|---|---|---|
| `-D MOJOLEARN_LSVR_FASTPATH_FIX` (`QN_FAST_TILED`) | define, binding `estimators` | `glm/impl/qn/glm_base.mojo` `qn_tiled_applies` | FAST takes the tiled objective: forward, loss map, `qnt_partial_kernel` (every 256-row tile's chain of every output, X read once) and `qnt_fold_kernel` (fold + epilogue + bias mean + loss store), FAST arithmetic. Removes the two one-block row sweeps and the 12-block gradient. Expected: FAST ~= IDENTICAL (about 110 ms). Risk: none beyond FAST's words changing (a different fold order). |
| `-D MOJOLEARN_LSVR_FUSED_GRAD` (`QN_FAST_FUSED`) | define | `glm_base.mojo` `qnf_partial_kernel`, `qnf_fold_kernel`, `GLMWithData.enqueue_fused` | one pass over X per evaluation: block b owns 4,096 rows, thread t takes rows t, t + 256, ... of them; the row's d features sit in registers, z = x . w + b, the loss term and dZ follow, and the d gradient cells, the dZ sum and the loss sum accumulate in registers over the thread's 16 rows; nothing per row is written back. Tile partials (62,720 tiles x 13 outputs on taxi) are folded by one 13-block kernel with the epilogue. Replaces gemv + bias + loss map + partial (two reads of X, z and loss_terms written and re-read) with one launch; C == 1 and d <= 32 (taxi d = 11 takes the 16-register instantiation; istella d = 220 falls back to the tiled path when FASTPATH_FIX is also on, else to main's). Risk: register pressure at the d <= 32 instantiation; the loss-id dispatch is a uniform branch per row. |
| `-D MOJOLEARN_LSVR_LINESEARCH_BATCH` (`QN_FAST_LS_BATCH`, implies FUSED_GRAD) | define | `glm_base.mojo` `qnf_partial_kernel[.., 4]`, `evaluate_batch`; `qn_linesearch.mojo` `ls_backtrack_batched` | the fused pass also forms, from the same registers, z at the next three backtracking steps (`x . xp + step/2^c (x . drt)`), sums their loss, and the fold adds each candidate's l2 value; all four objectives come home behind the one synchronize. The host walks them exactly as `ls_backtrack` walks candidates (Armijo, min/max step, `ls_iters`, `step *= 0.5`); when step 1 fails it jumps straight to the first passing candidate and materializes it with ONE evaluation, instead of one evaluation + synchronize per rejected step. When step 1 passes (the common L-BFGS case) the cost is the extra per-row work (two more 11-term dot products and three loss values), no extra launch or wait. If all four fail with budget left, the plain walk continues. Armijo only (the shipped line search); OWL-QN's projected search is untouched. Risk: the decision for candidates 2..4 uses the batch's objective, which can differ from a full evaluation at that point in the last bits (FAST promises no bits); the epsilon-insensitive loss backtracks often, which is where this pays. |
| `-D MOJOLEARN_LSVR_EVAL_SLIM` (`QN_FAST_SLIM`) | define | `glm_base.mojo` `qn_slim_epilogue_kernel`, `evaluate_pen` | the objective's fold writes the loss gradient (beta = 0) and ONE block then adds the Tikhonov gradient `l2 w`, folds its value, the gradient norm (`_gnorm_kind`: sum g^2, sum abs g, or max abs g) and, under OWL-QN, the l1 term into slots 1..3: one launch where there were four (memset g, `tikhonov_reg_grad_kernel`, the norm kernel, `nrm1_kernel`). Independent of the other three (it wraps any objective path). n_param <= 256. Expected: a few launches per evaluation, small. Risk: the g words are read by the thread that wrote them, so no device fence is needed; the folds are the block primitives. |
| `-D MOJOLEARN_LSVR_DEVICE_CONVERGE` (`QN_FAST_DCONV`, implies the LINESEARCH_BATCH and EVAL_SLIM kernels) | define | `glm/impl/qn/qn_dconv.mojo` (`dc_dir_kernel`, `dc_axpy_kernel`, `dc_ls_kernel`, `dc_check_kernel`, `dconv_run`); `glm_base.mojo` gated `qnf_*_gated_kernel` entries, `GLMWithData.enqueue_dconv_eval`; `qn_solvers.mojo min_lbfgs` | from iteration 2 the device runs whole L-BFGS iterations: the direction kernel with `end` / `n_vec` kept in a device state block, candidate 0's axpy, the fused batch pass (step 1 plus the next three backtracking steps), the Armijo walk on the device, a materialize pass only when a later candidate won, and `update_and_check` + `check_convergence` (gradient norm vs `epsilon * max(fx, epsilon)`, the `past`-deep objective-change test, max_iterations) on the device. The host reads the 32-word state block once per k = 8 iterations instead of synchronizing once per evaluation. Freeze: state word 0 is the stop flag and every kernel returns at its first line once it is set, so the up to 7 iterations enqueued after the stop are no-ops (no row work) and x, fx, k stay at the stopping iterate: the answer the host loop returns at that iteration. Anything the device does not decide (dg_init > 0, all four candidates rejected, a min/max-step stop, the line-search budget) hands the iteration back to the host before touching its start state; the host reruns it with the shipped line search and re-enters. Expected: one synchronize per 8 iterations where there was one per evaluation (~0.2 ms each plus the GPU idle bubble), at the price of up to 4 gated no-op launches per iteration when step 1 is accepted. Risk: the Armijo product is formed in device code (a boundary candidate can go the other way; FAST promises no bits); identity trace runs keep the host loop. |
| `-D MOJOLEARN_LSVR_DUAL_CD` (`QN_FAST_DUAL_CD`) | define | `glm/impl/qn/lsvr_dual.mojo` (`dcd_local_kernel`, `dcd_fold_kernel`, `dcd_pick_kernel`, `lsvr_dual_warm_start`); `qn_solvers.mojo qn_minimize` | a dual coordinate-descent warm start for the SVR losses (epsilon-insensitive and squared epsilon-insensitive), then the shipped L-BFGS from that point to its own tolerance. Each round: every thread runs liblinear's exact CD step (`solve_l2r_l1l2_svr`) over its 16 rows against a register copy of w (w lives on the device only), the tiles' delta-w fold, and a global step gamma picked from 2^0 .. 2^-18 and 1/K (K = 62,720 tiles on taxi) by the dual objective change; w += gamma dw, and the next round commits gamma to beta. 16 rounds, 3 launches each, no synchronize; one at the end. Applies to SVR L1 / L2, one target, l2 > 0, d <= 32 (taxi yes; istella d = 220 falls through to main's path, so its row is a control and should read equal). Expected: fewer L-BFGS iterations from a near-optimal start; the 16 dual rounds cost about 16 objective passes. Risk: on dense low-d data the parallel tiles conflict and gamma may sit near 1/K, making the warm start weak (then the arm is slower by the rounds' cost); the answer is still certified by L-BFGS's own test, so quality cannot degrade beyond the solver tolerance. |
| `-D MOJOLEARN_LSVR_ALL` | define | all of the above except DUAL_CD | FASTPATH_FIX, FUSED_GRAD, LINESEARCH_BATCH, EVAL_SLIM and DEVICE_CONVERGE: fused one-pass objective with batch candidates, slim epilogue, device-side line search and convergence; the tiled path for d > 32. DUAL_CD is not in ALL: it replaces the starting point of the primal solver rather than a kernel of it, so it is its own arm. |

Earlier draft of this file skipped DEVICE_CONVERGE and DUAL_CD; both are now built.

Why DUAL_CD is a damped parallel form and not liblinear's CD: liblinear updates
one coordinate at a time against the current w. Updating many rows at once
against the same w diverges on dense data (every taxi row shares all 11
features, so the parallel steps add up along the same directions; a safe
Jacobi damping is about the number of rows updated at once). The CoCoA-style
form here keeps each thread's 16 rows exactly sequential and damps only the
sum, by a step picked from a grid that contains 1/K: CoCoA's averaging step
(gamma = 1/K) decreases the dual every round at a linear rate, every smaller
positive step decreases it too by convexity, so the picked step is never worse
and the rounds are monotone. The unpenalized intercept would put sum beta = 0
on the dual, which coordinate steps cannot keep; the dual phase uses
liblinear's augmented bias feature (value 1) and the L-BFGS polish removes
that penalty, so the final objective and its tolerance are the shipped
solver's. Shrinking is not done: a row is 44 bytes inside a row-coalesced tile,
so skipping bound rows saves ALU, not bandwidth.

Compile owed: peer. DEVICE_CONVERGE, DUAL_CD and the changed ALL are
uncompiled (Andrew 2026-10-03: compile slots jammed, the M3 peer compiles).
Builds owed: FAST + DEVICE_CONVERGE, FAST + DUAL_CD, FAST + ALL, FAST off,
IDENTICAL; the four request lines below build them anyway.

New request lines (docs/apple-fast/ab/linsvr.txt): `linsvr-dconv-taxi`,
`linsvr-dconv-vs-batch-taxi` (DEVICE_CONVERGE against LINESEARCH_BATCH +
EVAL_SLIM, the same kernels with the host loop: isolates the device loop),
`linsvr-dualcd-taxi`, `linsvr-dualcd-istella` (a control: d = 220 does not
take the dual path). The existing `linsvr-all-*` lines now measure ALL with
DEVICE_CONVERGE in it.

## Keep rule

A switch becomes the FAST default when its arm is faster on the M3 and held-out R2 / RMSE stay
within FAST's run-to-run spread (the board's `linearsvr` quality columns); then the define goes and
the arm is the code. FASTPATH_FIX is the floor (IDENTICAL's structure); FUSED_GRAD supersedes it
where d <= 32; LINESEARCH_BATCH and EVAL_SLIM stack on top. `n_iter_` should match across arms up
to the batch's last-bit decisions.

Compile risks to watch: `qnf_partial_kernel`'s `comptime for` over `InlineArray` registers with a
runtime `if j < d` guard and the parametric `[DMAX, K]` launches; `comptime if not X: raise ... else:`
method bodies; the `or slim` / `not slim` runtime guards around main's launches (constant False
outside FAST + Apple); the new import of `pinned_block_max` and of `QN_FAST_LS_BATCH`, `QNF_LS_K`
into `qn_linesearch.mojo`.

Compile risks for the two new defines (uncompiled): the `qnf_*` kernels are now
`@always_inline` bodies with a plain and a gated kernel entry each; `qn_dconv.mojo`
imports `_block_dot_bcast`, `_two_loop`, `_dev_barrier` from `qn_util.mojo` and
`_qnb_barrier` from `glm_base.mojo`; `continue` inside `comptime if QN_FAST_DCONV` in
`min_lbfgs`; Float32 <-> Int state words via `cast[DType.int32]`; `lsvr_dual.mojo`'s
`dcd_local_kernel[DMAX]` register arrays (`InlineArray[Float32, DCD_G]` with
`comptime for`) and its shared `stack_allocation` in `dcd_pick_kernel`.
