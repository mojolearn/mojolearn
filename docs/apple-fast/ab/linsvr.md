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
| `-D MOJOLEARN_LSVR_ALL` | define | all of the above | the four together: fused one-pass objective with batch candidates, slim epilogue; the tiled path for d > 32. |

Not implemented, and why: `MOJOLEARN_LSVR_DEVICE_CONVERGE`: the convergence test already runs on
numbers that come home with the ONE synchronize per evaluation (loss, regularizer, gradient norm,
dg_init and the direction's verdict share `read_scalars`); there is no separate per-iteration
readback to remove, and running whole iterations device-side would need a device-driven L-BFGS,
not a switch. `MOJOLEARN_LSVR_DUAL_CD`: liblinear's dual coordinate descent is sequential per
coordinate; a parallel block variant changes the solution path and needs its own convergence
argument against the primal tolerance; not simple, not attempted.

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
