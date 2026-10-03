# gap-cls2: FAST Apple rows slower than the best opponent (lane/apple-fast-gap-cls2)

Board: docs/apple-fast/BOARD_M3_FAST.md (origin/main 18dc09a7a). The board's ms is the FIT only
(tools/bench_board_algos.py `round`: `runner.fit()` is timed, `runner.infer()` is not), so for the
transforms (gaussian-rp, minmax-scaler, onehot, ordinal) only fit counts.

Every switch below is FAST + Apple + an explicit `-D`, default OFF. IDENTICAL and the other vendors
compile main's paths. No CPU route, no host compute added; every candidate keeps the output words
(the A/B digests must match arm A's).

## gaussian-rp (istella 53.7 vs 24.3 ms; taxi 3.3 vs 1.6)

Hypothesis: the fit's time is the host finiteness walk over X, not the projection matrix.
- python/mojolearn/_expansion_decomp.py `_RandomProjection.fit` -> `_M.shape_of_input` (:207) ->
  `_host_all_finite` (:135) -> bindings/host_helpers.mojo:47 `all_finite_f32_binding`: one host thread,
  a branch per word (no vectorization) over 900k x 220 words on Istella (~790 MB), ~14 on taxi.
- The matrix itself: `x_decomp_dev_rand` + `ew scale` (two launches, 10 x d) and one 10 x d
  download + synchronize (`x_decomp_dev_download`, :1240 area). Small; matters on taxi.
- scikit-learn's fit is its own `check_array` finite pass (one numpy sum), hence 24 ms.

Candidates (x_decomp/resident.mojo GRP_CLS2_*, read back by `x_decomp_grp_cls2`):
- `-D MOJOLEARN_XD_FAST_CLS2_GRP_NOSCAN`: fit reads the shape only; transform's device projection
  (`dev_project_py`, already on main) flags a non-finite X. Precedent: SparseRandomProjection's
  `-D MOJOLEARN_SPARSE_RP_DEVICE` (x_neighbors/kapprox_dev.mojo:53).
- `-D MOJOLEARN_XD_FAST_CLS2_GRP_DEVSCAN`: the refusal stays in fit as a device scan
  (`dev_first_nonfinite_py`: upload from the caller's buffer into a pooled buffer,
  core/device_scan.mojo `device_first_nonfinite`, one 4 B result).
- `-D MOJOLEARN_XD_FAST_CLS2_GRP_LAZY` (with NOSCAN): components_ downloaded on first read
  (python property), so fit ends with no synchronize.

## minmax-scaler (istella 106 vs 63.0 ms)

Hypothesis: preprocessing/estimator.mojo `minmax_fit_direct` (:240 area) allocates a FRESH device
buffer of n*d (880 MB at the board's shape) per fit (`_upload_direct`), then runs TWO full reads of X
with a synchronize each: `device_first_nonfinite` (core/device_scan.mojo:180, its own partials buffer,
host buffer, sync) and the extrema pass (preprocessing/minmax.mojo `minmax_fit_fast`, sync on download).
Candidates (preprocessing/minmax.mojo PREP_CLS2_MINMAX_*):
- `-D MOJOLEARN_PREP_FAST_CLS2_MINMAX_FUSED`: the nonfinite flag computed inside the extrema pass
  (`extrema_rows_flag_kernel`, `extrema_finalize_flag_kernel`: a 6th row in the one download): one read
  of X, one synchronize.
- `-D MOJOLEARN_PREP_FAST_CLS2_MINMAX_POOL`: X's device buffer from core/device_pool.mojo
  `pool_take`/`pool_give` (kept between fits).
- both together.

## onehot (taxi 28.0 vs 19.0 ms), ordinal (taxi 24.8 vs 18.8 ms)

Hypothesis: python/mojolearn/_expansion_prep.py `_fit_categories` (:723 area):
- `uo = pr.alloc(n * d)`: the distinct values go to a HOST arena region of n*d words (5M words, 20 MB
  at taxi's 1M x 5) that `_Prog.run` zero-allocates on the host (`_zero_words`) and the ranges runner
  DOWNLOADS whole (everything but the inputs comes back), for a few hundred distinct values.
- `sort_cols`: a 4-pass radix sort of every column (x_prep/dradix.mojo) to find distinct small integers.
Candidates (x_prep/cat_cls2.mojo, ops 157-161, read back by `x_prep_cls2_cat`):
- `-D MOJOLEARN_X_PREP_FAST_CLS2_PACK`: the run scan writes device scratch; `cat_pack` copies each
  column's distinct words into a small host region (64K words); overflow reruns main's program.
- `-D MOJOLEARN_X_PREP_FAST_CLS2_PRESENT` (with PACK): no sort. Presence flags per column for words
  that are integers in [0, 4096) (by bits; -0.0 is 0), a chunked flag scan writes the present integers
  ascending (= the sorted distinct canonical words). Any other word in a column reruns main's program.

## minibatch-kmeans (istella 257 vs 124 ms; taxi 44.9 vs 43.9)

Hypothesis: x_cluster/minibatch_fast.mojo `minibatch_fast_steps`: up to max_iter*n/batch = 24,414
steps; each step is FOUR launches (`_mbf_assign_kernel`, `_mbf_sum_kernel`, `_mbf_finish_kernel`,
`_mbf_reassign_kernel`) and every MBF_GROUP = 32 steps a synchronize (the convergence read). Launch +
sync overhead per step dominates the kernels (4096 x 8 x 220 per step). Plus a fresh 880 MB X buffer
per fit (x_cluster/minibatch_ptr.mojo `xbuf`).
Candidates (x_cluster/minibatch_fast.mojo MBK_CLS2_*):
- `-D MOJOLEARN_X_CLUSTER_FAST_CLS2_MBK_G128`: 128 steps per convergence read.
- `-D MOJOLEARN_X_CLUSTER_FAST_CLS2_MBK_FIN`: finish + reassign as ONE launch
  (`_mbf_finish_reassign_kernel`: 256 threads stride the k x d cells, device-memory barrier, then the
  reassignment as before): three launches a step. Same arithmetic, same draws.
- `-D MOJOLEARN_X_CLUSTER_FAST_CLS2_MBK_POOL`: X's device buffer pooled between fits.

## ocsvm (taxi 372 vs 181 ms; regressed from 246 in the last refresh)

Hypothesis: python/mojolearn/_expansion_neighbors.py `OneClassSVM.fit` forms the 10,000 x 10,000 Gram
with `xn_kernel` (x_neighbors/device_ops.mojo:179 op_kernel), DOWNLOADS its 400 MB into a fresh host
array (~3 GB/s download + first-touch page faults), and x_neighbors/ocsvm_dev.mojo `op_ocsvm` UPLOADS
it back (`_buf(ctx, q, n * n, True)`). Then the SMO: three launches per iteration
(`ocsvm_sel_i_kernel`, `ocsvm_sel_j_kernel`, `ocsvm_step_kernel`), a synchronize every 64.
Candidates (x_neighbors/ocsvm_dev.mojo OCSVM_CLS2_*):
- `-D MOJOLEARN_XN_FAST_CLS2_OCSVM_RES`: `op_ocsvm_x` (`x_neighbors_ocsvm_resident`) forms the same Gram
  with the same kernel on the device and solves over it there: no 400 MB round trip.
- `-D MOJOLEARN_XN_FAST_CLS2_OCSVM_2L`: two launches per iteration: the step kernel also writes the
  next iteration's sel_i partials (ping-pong buffers), sel_j commits the pending pair. Same total-order
  maxima/minima: the same alpha bits.
- `-D MOJOLEARN_XN_FAST_CLS2_OCSVM_CHUNK256`: 256 iterations per stop-flag read instead of 64.

## Compiles (M2, m2compile.sh) and queued A/Bs

M2 compiles, all rc=0 (~/mojolearn-evidence/gap-cls2/m2c_*.log): fast + every define set (combined and
single), fast default, identical for x_decomp, x_cluster, x_prep, x_neighbors, preprocessing.

M3 queue lines 990-1013 (afc_ab_def.sh, 1 rep x 2 rounds, A = main FAST, B = the define[s]):
gapcls2-{noscan,noscanlazy,devscan}-grp-{istella,taxi}; gapcls2-{g128,fin,pool}-mbk-{istella,taxi},
gapcls2-all3-mbk-istella; gapcls2-{fused,pool,fusedpool}-minmax-istella;
gapcls2-{pack,present}-{onehot,ordinal}-taxi; gapcls2-{res,2l,chunk256,all3}-ocsvm-taxi.
