# lane/apple-fast-select: feature selection (NEXT_PASS.md unowned gaps)

Written without a Mojo toolchain (cloud peer); the first M3 build is the compile check.
Every switch is a build define, compiled under FAST + Apple only (`GLOBAL_NUMERIC_MODE == NUMERIC_FAST and
has_apple_gpu_accelerator()` and `is_defined[...]`), default OFF; IDENTICAL compiles main's code unchanged.
All four lanes are `tools/bench_board_algos.py` lanes (AFC_FAMILY=algos; select-d is one of its extra lanes).

| define | binding | lanes | site | what it changes under FAST on Apple |
|---|---|---|---|---|
| `-D MOJOLEARN_SELECT_FREG=1` | x_prep | select-f-regression, select-r-regression | `x_prep/device.mojo` run loop, op 64 -> `x_prep/select_fast.mojo` `select_freg_device` | the `f_regression` stage (one thread per column over every row) becomes row x feature tiles: 512-row x 32-column threadgroups (8 row lanes x 32 column lanes, 64 rows per thread, a tile row one coalesced read), a tree over the row lanes into per-tile partials in the sort scratch, then a threadgroup per column folds the partials. Pass 1 the column sums and y's (when centring), pass 2 the centred cross and square sums, then the unit's own tail (r, F, `f_sf` p-value, force_finite words). |
| `-D MOJOLEARN_SELECT_FCLS=1` | x_prep | select-f-classif | `x_prep/device.mojo` run loop, ops 63 and 16 -> `select_fcls_device`, `select_cstats_device` | the `f_classif` stage becomes the same tiles (within-class squares about the class means, then F and p per column); and in a program that contains op 63, the `class_stats` stage ahead of it (K*d threadgroups each over every row) becomes class-sum tiles: per-class column sums and counts in threadgroup memory (K <= 16 classes; the board's cls block is binary), then a threadgroup per (class, column) folds the tiles into the means and counts. chi2 / naive Bayes / DA programs keep their kernels. |
| `-D MOJOLEARN_SELECT_D=1` | tsa | select-d | `tsa/impl/auto_arima.mojo` `select_d` -> `tsa/impl/select_d_fast.mojo` | the host-controlled first-stationary loop (per round: a host scan of the whole input for non-finite values with a wait, eight launches, a wait, the flags downloaded and masked on the host) becomes one stream: a non-finite flag kernel, every round's differencing and the primitive's KPSS kernels back to back on one workspace, a kernel choosing each series' first stationary order, one download of the orders and the flag (the non-finite refusal is raised after it, by the same name). |

Bits: the FAST scores change only by fold order (pairwise instead of row order; the arithmetic is x_prep/prims.mojo's
and the edge rules the units'); select-d's statistics are the primitive's own kernels, so the orders are the same words.

Risky compile sites (no toolchain here): `x_prep/select_fast.mojo` kernels take the scratch as `MutPointer[UInt32,
MutAnyOrigin]` and `bitcast[Float32]()` it (x_prep/common.mojo's idiom), threadgroup arrays are `stack_allocation[..,
address_space = AddressSpace.SHARED]` as x_prep/fastred.mojo; `select_d_fast.mojo` passes `w.unsafe_ptr() + offset`
and `res.unsafe_ptr() + d * batch` as kernel arguments (x_prep/device.mojo's `dq.unsafe_ptr() + ..` idiom) and keeps
every buffer alive past the one `synchronize` (`_ = buf^` after it).
