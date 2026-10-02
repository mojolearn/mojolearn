# lane/apple-fast-robust: huber, perceptron, sgd-ocsvm, ocsvm, elliptic-envelope, min-cov-det

Written without a Mojo toolchain (cloud peer); the first M3 build of `x_linear` (`bindings/build_x_linear.sh`, FAST) is
the compile check. Every change is compiled under FAST + Apple only and defaults OFF; IDENTICAL compiles main's code
unchanged. Branch cut from origin/main and merged with main at 2d7eade5b (the lane/cg-integrate merge).

| switch | kind | site | what it changes under FAST on Apple |
|---|---|---|---|
| `-D MOJOLEARN_HUBER_DEVICE_LBFGS=1` | define, `HUBER_DEVICE_LBFGS` comptime alias in `x_linear/huber_fast.mojo` (FAST and `has_apple_gpu_accelerator()` and the define) | `fit_device` in `x_linear/device.mojo`, inside the existing `comptime if GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()` block, before main's `huber_fit_grid` dispatch | main's grid fit (`x_linear/huber_grid.mojo`) runs `lbfgs` on the host and pays two host waits per objective evaluation (theta upload, three launches, gradient and sums home, the witness read, `synchronize`), up to 41 evaluations per iteration for 100 iterations. The board row (taxi 168 s, 8x IDENTICAL's CPU) predates the grid fit; the grid fit's taxi time is unknown (the `afc-dbg` line failed on a quoting error). Here the minimizer's state (theta, trial point, g, gn, dir, the 10-pair ring, rho, alpha, f, slope, tt) lives in one device buffer; an evaluation is four launches (`hf_map_kernel` a thread per row, `hf_part_kernel` a thread per (FOLD_BLOCK block, task), main's `hg_fold_kernel`, `hf_step_kernel` one block of 256: huber_finish, the Armijo / noise + curvature decision, the pair update, the two-loop direction on the lead with ascending `_dot`s, `tn = theta + tt dir` across the block). The host enqueues HF_BATCH = 16 evaluations and reads the 8-word state once per batch; evaluations after the stop are no-ops. The objective, folds and dots are the grid fit's statements in the same order, so the coefficients should be the grid fit's bits. A batch is one Metal witness unit with a device-side snapshot and restore (x_linear/device.mojo `_sgd_ps_grid`'s pattern). |
| `-D MOJOLEARN_HUBER_FAST_BLOCK512=1` (with the define above) | define, `HF_FOLD` comptime in `x_linear/huber_fast.mojo` | `hf_part_kernel` / `hf_fold_blocks` | the partials over 512-row blocks instead of FOLD_BLOCK = 4096: eight times the threads of the partials launch (taxi: (d + 6) tasks x n / 4096 blocks is a few thousand threads, each a 4096-row serial chain); a different fold order (FAST bits only, quality is the bar). |
| `-D MOJOLEARN_MCD_DEVICE_CSTEPS=1` | define, `MCD_DEVICE_CSTEPS` comptime alias in `x_decomp/mcd_fast.mojo` (FAST and `has_apple_gpu_accelerator()` and the define) | the top of `fast_mcd_dev` in `x_decomp/kit_device.mojo` (the GPU binding's `x_decomp_mcd`); `fast_mcd_fast` returns False for d > 64 and main's search runs | main's `fast_mcd_dev` runs mcd.mojo's search one candidate and one C-step at a time, each C-step 10-15 kit launches with host syncs for the distances (the h-smallest selection on the host), the LU log determinant and the Jacobi eigenpairs: on taxi (100,000 x 16, mid subsample) 333 subsets x 10 trials x 2 steps, then 3,330 candidates x up to 30 steps on 1,500 rows, then 10 x 30 on all rows. Here the search is three phases of candidates stepping together on the device: the two `_perm`s by a bitonic sort of the draws' (float image, index) keys (`mf_keys_kernel`, `mf_bitonic_kernel`); per step the masked moments per (candidate, 256-row tile, feature) and (candidate, tile, feature pair) with a fold per (candidate, cell) (`colmean`, `emp_cov`'s 1/h); one thread per candidate for `fast_logdet` (partial-pivot LU, the sign rule, logs floored at FLT_MIN) and the C-step control (`det < prev_det and iters > 0 and det != -inf`, sklearn's `use_prev`) recording the chosen parity; one thread per candidate for `pinvh` (cyclic Jacobi, the `|w| > max|w| d eps` cut, V diag(1/w) V^T); one thread per (candidate, row) for `mahal`; one block per candidate for the h smallest by (distance image, index) (8-pass radix select over the 32-bit image, then a block scan taking the first `need` ties in index order) as the next parity's mask; the stop word read once per step; `_order_by_det` as one thread per candidate counting the keys below its own; the kept candidates' (mean, covariance) handed to the next phase in rank order; the best candidate's support and distances scattered on the device. Same statements as mcd.mojo within float32 (blocked folds, a one-thread Jacobi instead of the kit's eigh, the cut in float32 since Metal has no float64): not bit for bit. X uploaded once; NaN distances and a no-pinvh stop raise as mcd.mojo does (error words read at the end). Not moved: the Python reweighting after `_fast_mcd` (`_consistency_factor`, `_chi2_quantile`, `_masked_cov`, `_pinvh`, `_mahal`: a dozen resident-kit calls) and EllipticEnvelope's host percentile of `dist_`. |

Not changed (reasons):

- perceptron (istella 1.1x): main's `_sgd_mb_grid` already runs a chunk of batches per launch (`sgd_mb_chunk_kernel`) and
  syncs once per epoch (20 epochs); what is left is the serial batch chain of minibatch SGD itself.
- sgd-ocsvm (istella 1.8x, taxi 1.05x): the board runs the per-sample fit (`batch_size=0`, `_sgd_ps_grid`: one block
  walks the samples in order, 2048 per launch, one sync per epoch). The per-sample order is the estimator's quality
  (the minibatch form flags 0.37 of istella at batch 256 vs sklearn 0.055, `_expansion_linear.py`); a faster form
  changes the algorithm, not a host route. Left alone.
- ocsvm (taxi 1.4x): main now runs the one-class SMO on the grid (`x_neighbors/ocsvm_dev.mojo`, c-svm, state read once
  per OCSVM_CHUNK iterations); the board row predates it. Needs a re-run before any further change.
- elliptic-envelope / min-cov-det: see the `MCD_DEVICE_CSTEPS` row (pass 2). EllipticEnvelope is MinCovDet's fit
  plus a host percentile of `dist_`; both lanes race the same device search.

## Risky compile sites

- `x_decomp/mcd_fast.mojo`: `struct MfPhase(Movable)` holds DeviceBuffer fields (the witness.mojo idiom) and is built from a
  borrowed `DeviceContext`; kernels take `F32Ptr` / `I32Ptr` / `U64Ptr` (`MutPointer[.., MutAnyOrigin]`) built from
  `buf.unsafe_ptr()` addresses (x_decomp/device.mojo `_p`'s idiom); `mf_select_kernel` uses
  `stack_allocation[MF_SH_INT, Scalar[DType.int32], address_space = AddressSpace.SHARED]()` (17.5 KB, the ALS kernel's
  idiom) and `barrier()`; `InlineArray[Int32, 16](fill=Int32(0))` and `InlineArray[Float32, 64]` per thread; the
  Jacobi/LU use plain Float32 ops and `sqrt0` (no Float64 in any kernel); `DeviceBuffer[DType.uint64]` for the sort keys;
  `_down` / `_down_i` imported from x_decomp/device.mojo for the outputs; `mf_select_kernel` returns early per block
  (uniform) before its barriers.

- `x_linear/huber_fast.mojo` `hf_step_kernel`: `var gamma: Float32` assigned in both branches (lbfgs's idiom); `var k`
  declared in several sibling scopes; `_dot` and `LBFGS_M` imported from `x_linear/lbfgs.mojo` (an underscore name);
  `team_barrier()` between the block's phases (device-memory barrier on Apple); the `if run:` body is uniform (the stop
  word is read into a register before the first barrier).
- `huber_fit_fast`: `var ctx = ctx0.copy()` so `Witness(ctx, ...)` and `wit.ok(ctx, ...)` get a mutable context
  (huber_grid's `wctx`); kernels take `FP` / `IP` and are launched with `buf.unsafe_ptr()` exactly as the sibling
  `hg_*` launches; no `MutAnyOrigin` parameters were added; device-to-device `enqueue_copy(dst_buf=, src_buf=)` for the
  snapshot and restore (the `_sgd_ps_grid` idiom).
- The dispatch is nested `comptime if HUBER_DEVICE_LBFGS and not X_LINEAR_SERIAL_FOLDS:` inside the FAST + Apple block of
  `fit_device`.

## Request lines (`robust.txt`)

- `robust-huber-dev-taxi`: huber taxi, arm A main's grid fit, arm B the device L-BFGS. Bar: faster, r2 within FAST
  run-to-run spread (the bits should match arm A).
- `robust-huber-blk-taxi`: huber taxi, both arms the device L-BFGS, arm B with 512-row partial blocks. Queue after the
  first line wins.
- RUN OWED after a taxi win: the same two lines on istella (`robust-huber-dev-istella`, `robust-huber-blk-istella`).
- `robust-mcd-taxi`, `robust-ee-taxi`: min-cov-det and elliptic-envelope taxi, arm A main's resident search, arm B the
  device C-steps (binding `x_decomp`). Bar: faster; `covariance_` / `support_` / `dist_` within float32 of arm A
  (the selection is a total order on (distance, index), so the supports should match exactly unless a distance tie
  moves under the blocked folds).

## ocsvm baseline re-run (no switch)
`robust-ocsvm-base-taxi` runs arm A only (afc_ab.sh with no envB): main's current ocsvm FAST route (x_neighbors/ocsvm_dev.mojo, SMO on the
grid, brought in by the cg-integrate merge) at this branch's head, which carries no ocsvm change. The board row (taxi 246 ms vs sklearn
181 ms) predates it; this line gives the manager the current FAST time to replace it.
