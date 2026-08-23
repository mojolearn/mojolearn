# solver: coordinate descent (Lasso / ElasticNet), from cuML `cpp/src/solver/`

Seventh section. **COPY, DO NOT IMPROVE.** `cdFit` and `cdPredict`
(`upstream/cuml-v26.08.00/cpp/src/solver/cd.cuh`) and every RAFT primitive
their control plane calls, file for file, under `solver/ported/`; the one
thing cuML never needed -- a row-length reduction with ONE fold shape on
three vendors -- under `solver/mojo_only/`. DEVIATION 610 is the identity
construction; 611 is reserved and NOT spent (see `shuffle` below); 612-619
are unused.

## Status: CONSTRUCTION plus one Apple device's gates; no second vendor has run this.

Built and gated 2026-08-23 on one M4. Nothing here has been compiled for
or run on NVIDIA or AMD. No performance was measured and no timing is
printed by any file in this directory; the cross-vendor leg and the
benchmark arena are the identity lane's.

    == solver/mojo_only/cd_check.mojo [IDENTICAL] ALL PASSED ==
    == solver/mojo_only/cd_check.mojo [FAST] ALL PASSED ==

## Commands

    # the gates, both arms (every line carries the mode the binary COMPILED in)
    tools/with_build_lock.sh     pixi run mojo run -I . solver/mojo_only/cd_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . solver/mojo_only/cd_check.mojo

    # the driver: one Lasso fit on the planted fixture, the oracle's verdict, predict
    tools/with_build_lock.sh     pixi run mojo run -I . solver/cd_main.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . solver/cd_main.mojo

    # the identity card
    MOJOLEARN_IDENTITY_TRACE=/tmp/cd.apple.card \
        tools/with_identical_mode.sh pixi run mojo run -I . solver/cd_main.mojo
    python3 tools/identity_trace_diff.py /tmp/cd.apple.card /tmp/cd.other.card

`cd_main.mojo` reads `MOJOLEARN_CD_ROWS/COLS/ALPHA/L1_RATIO/FIT_INTERCEPT/
EPOCHS/TOL` (defaults 2048 / 16 / 0.01 / 1.0 / 1 / 1000 / 1e-3). No pixi
task is registered (pixi.toml is not this lane's); the lines above are the
whole interface.

## What is ported

| ours | theirs | what |
|---|---|---|
| `solver/ported/solver/cd.mojo` | `cuml cpp/src/solver/cd.cuh` | `cdFit` line for line (guards, preprocess, `l1_alpha`/`l2_alpha`, colNorm + addScalar, the five-operation coordinate step with DEVICE-pointer alphas, one host read per epoch, the stopping rule, postprocess), `cdUpdateCoefKernel` transcribed, `cdPredict` |
| `solver/ported/solver/shuffle.mojo` | `cd solver/shuffle.h` | `initShuffle` (the identity permutation); `shuffle` REFUSED, see below |
| `solver/ported/solvers/params.mojo` | `cuml include/cuml/solvers/params.hpp` | the three enums |
| `solver/ported/glm/preprocess.mojo` | `cuml cpp/src/glm/preprocess.cuh` | `preProcessData` / `postProcessData`, unweighted arms, IN PLACE as theirs (HAND-OFF: it belongs under `glm/ported/`) |
| `solver/ported/functions/linear_reg.mojo` | `cuml cpp/src_prims/functions/linearReg.cuh` | `linearRegH` only |
| `solver/ported/linalg/coalesced_reduction.mojo` | `raft linalg/detail/coalesced_reduction-inl.cuh` | `coalescedSumMediumKernel<256>` with its per-thread Kahan-Babushka-Neumaier sum and two `BlockReduce.Sum`s -- the FAST arm of colNorm and mean |
| `solver/ported/linalg/norm.mojo` | `raft linalg/detail/norm.cuh` | `colNorm<L2Norm, rowMajor=false>` (a SUM OF SQUARES; no sqrt) |
| `solver/ported/stats/mean.mojo` | `raft stats/detail/mean.cuh`, `mean_center.cuh` | `mean<false>` = sum times `1/N` (a multiply), `meanCenter`/`meanAdd` as `matrixVectorOp` along columns |
| `solver/ported/linalg/axpy.mojo` | `raft linalg/detail/axpy.cuh` | `axpy<T, DevicePointerMode=true>` -- cuBLAS axpy with a DEVICE alpha; MAX has no axpy, so the mirror is one thread per row reading alpha from device memory |
| `solver/mojo_only/profile_dot.mojo` | none | DEVIATION 610: every `n_rows` reduction under IDENTICAL is the `mojolearn.identical.gemm.fp32.v1` `OP_NT` cell at `1 x 1 x n_rows`, CALLED from the gemm lane (`identical_gemm_with_plan`) -- no second fold is spelled; the sum for the means is the dot against ones |
| `solver/mojo_only/cd_oracle.mojo` | none | the host oracle (device arithmetic, serial), the whole-row serial variant, the Float64 reference, the hashed fixtures |
| `solver/mojo_only/cd_check.mojo` | none | the gates, below |
| `solver/cd_main.mojo` | none | the driver and the card |

Upstream pins: cuML `v26.08.00` = `265b9da`, RAFT `v26.08.00` = `ebf9268`
(`solver/PORTED_MAP.tsv`).

## cuML versus scikit-learn: the objective is the SAME, the stopping rule is not

`cd.cuh:77-84` documents `f = 1/2||y - Xw||^2 + 1/2 alpha (1-l1_ratio)||w||^2
+ alpha l1_ratio ||w||_1`, but `cd.cuh:168-169` scales both penalties by
`n_rows`:

    l2_alpha = (1 - l1_ratio) * alpha * n_rows
    l1_alpha =      l1_ratio  * alpha * n_rows

so the minimized function is `n_rows` times scikit-learn's

    (1/(2n)) ||y - Xw||^2 + alpha * l1_ratio * ||w||_1 + (alpha/2)(1 - l1_ratio) ||w||^2

and `alpha`, `l1_ratio` mean the SAME THING in `cuml.ElasticNet`,
`sklearn.linear_model.ElasticNet` and here. Their docstring is off by the
factor `n`; the code is what is ported. scikit-learn's coordinate update
(`sklearn/linear_model/_cd_fast.pyx::enet_coordinate_descent`) uses the
same `l1_reg = alpha * l1_ratio * n`, `l2_reg = alpha * (1 - l1_ratio) * n`
and the same soft threshold over `norm_cols_X[j] + l2_reg`, so the per-
coordinate step is the same formula. Where the three differ:

| | cuML `cdFit` (ported here) | scikit-learn `ElasticNet`/`Lasso` |
|---|---|---|
| stopping rule | after each epoch, `coefMax < tol OR diffMax / coefMax < tol` (`cd.cuh:236`) -- a coefficient-CHANGE test only | the same change test with `<=` as a gate (`_cd_fast.pyx:461-462`), THEN the duality gap must be `<= tol * ||y||^2` (`:392, :471`); the gap is what is certified |
| `tol` default | `1e-3` (Python `ElasticNet(tol=1e-3)`); `_params_from_cpu` multiplies a scikit-learn `tol` by 10 | `1e-4` |
| soft-threshold guard | `squared > 1e-5 ? r / squared : 0` (`cd.cuh:62`), an ABSOLUTE threshold on a sum of squares that scales with `n` and the data (the `OLS_NONZERO_THRESH` class; carried as theirs, recorded) | `norm_cols_X[j] == 0.0` skips the coordinate (`_cd_fast.pyx:440`) |
| coordinate order | cyclic (`selection='cyclic'`); `'random'` is `std::shuffle` of `std::mt19937(0)` per epoch | cyclic, or `random` from its own RNG with `random_state` |
| warm start | `coef` is read as the starting point (cuML's Python passes zeros) | `warm_start` |
| `positive`, `precompute`, `warm_start=True` | not supported (`UnsupportedOnGPU`) | supported |
| the design matrix | F-order (column-major), MUTATED IN PLACE under `fit_intercept` (centered, then un-centered, which is not bitwise the original) | untouched |
| `fit_intercept` | `preProcessData`: mean-center X and y on the device; intercept = `mean(y) - mu_X . coef`; `meanAdd` to undo | the same centering (`_preprocess_data`) |
| `normalize` | removed in 26.08 (no arm to port; not exposed) | removed |

scikit-learn's `Lasso`/`ElasticNet` have no Array API path (`ROADMAP.md`),
so this section races scikit-learn-on-CPU, not scikit-learn-on-Metal.

## DEVIATION 610: cuML's iteration count is a function of the card it ran on

`cdFit` performs four row-length reductions and ships them on THREE fold
shapes:

- the column norms and, under `fit_intercept`, the column and label means,
  on RAFT's `coalescedReduction`, whose DISPATCH reads
  `getMultiProcessorCount()` (`coalesced_reduction-inl.cuh:497`): `D <=
  512 || (N >= 16 numSMs && D < 2048)` takes the Thin kernel (logical warps
  of the hardware width), `N < numSMs && D >= 2^17` the Thick one, the rest
  the Medium one -- so WHICH kernel folds, and in what shape, depends on the
  SM count; the Medium and Thick kernels compensate per thread (KBN) and
  then fold the compensations with CUB;
- the per-coordinate `dot(X[:, ci], residual)` on cuBLAS `gemv`, closed.

The coordinate update branches on those bits (`coef > l1_alpha`), the
stopping rule branches on them (`diffMax / coefMax < tol`), and `n_iter` is
what the Python estimator returns. cuML ships one backend and accepts that.
Here, under IDENTICAL:

- ALL FOUR reductions are the `mojolearn.identical.gemm.fp32.v1` dot
  (`solver/mojo_only/profile_dot.mojo`): `L = contract_leaf_size(n_rows)`,
  `P = ceil(n_rows / L)`, serial ascending `fma` inside a leaf with every
  seam flushed, a fixed balanced tree over the leaves -- a pure function of
  `n_rows`, and the gemm lane's own gates and launch-invariance stand behind
  it. The means are the dot against ones (`fma(x, 1, acc) == x + acc`
  exactly). No fold is spelled in this section;
- the two axpys are `ftz(identical_mul_add(ftz(alpha), ftz(x), ftz(r)))`,
  one thread per row, order-free; `cdUpdateCoefKernel` flushes its
  quotient, its `diff` and its `|r|` (the `ConvState` the host reads);
  `meanCenter`/`meanAdd`/`addScalar` flush their stores;
- so `coef`, `residual`, `ConvState`, the EPOCH COUNT and the intercept
  are a function of the inputs alone, and the card carries each per epoch.

Under FAST the arms are the vendor spellings: `linalg.gemv.gemv_gpu` for
the dot (`core/gemm.mojo::gemv_n`, the cuBLAS mirror), the ported Medium
kernel for the norms and means (Thin and Thick are NOT ported, and the
SM-count dispatch with them: `UNPORTED.tsv`), and the same elementwise
kernels with the naive multiply-add. On this M4 FAST and IDENTICAL differ
in 2 of 16 coefficients on the planted fixture (`cd_main.mojo`, both modes
printed), which is the measured size of the fold's reach on one device.

`predict` (`cdPredict` -> `linearRegH`) is the gemm profile's `OP_TN` at
`m = n_rows, n = 1, k = n_cols` under IDENTICAL (the column-major design is
already the `k x m` row-major operand; no transpose) and a device transpose
plus `gemv_gpu` under FAST (`linalg.matmul` does not write at `n = 1`).

## The card

    0  cd.input.x      f32  n_rows*n_cols     the inputs, so a cross-vendor
    1  cd.input.y      f32  n_rows            diff starts from the same bits
    2  cd.mu_input     f32  n_cols            [fit_intercept only]
    3  cd.mu_labels    f32  1                 [fit_intercept only]
    4  cd.l1_alpha     f32  1                 host Float32 products
    5  cd.l2_alpha     f32  1
    6  cd.colnorm      f32  n_cols            sum of squares per column
    7  cd.squared      f32  n_cols            + l2_alpha (== colnorm at l1_ratio 1)
    8  cd.sweep000.coef   f32  n_cols         per epoch, in order
    9  cd.sweep000.resid  f32  n_rows
    10 cd.sweep000.conv   f32  3              {-last r, coefMax, diffMax}
       ...
       cd.final.coef   f32  n_cols
       cd.intercept    f32  1
       cd.n_iter       i32  1                 THE INTEGER STAGE

A diff that first moves at `cd.n_iter` says the two machines disagreed
about how long to run; at a `.conv` it says the stopping scalars moved; at
a `.resid` it names the epoch and (with `MOJOLEARN_IDENTITY_TRACE_DUMP`)
the row. On the planted fixture (2048 x 16, Lasso alpha 0.01) the card has
20 records over 3 epochs; `check_cd_card_is_emitted` asserts the list and a
run-to-run control.

## The gates (`solver/mojo_only/cd_check.mojo`), and what each one saw on the M4

    check_cd_refuses_by_name              shuffle=true, sample_weight, loss=HINGE, n_cols=0,
                                          n_rows=1, alpha<0, l1_ratio=1.5, predict loss=HINGE:
                                          each raises naming the parameter
    check_cd_recovers_the_planted_support Lasso alpha 0.01 on hashed X in [-0.5,0.5), 4 of 16
                                          planted nonzeros: support recovered EXACTLY; worst
                                          |device - float64| 2.8e-07 (tol 2e-3); n_iter device 3
                                          = oracle 3 = float64 3; final diffMax 1.22e-4 /
                                          coefMax 2.6656 = 4.58e-5 < 0.001
    check_cd_serial_fold_is_a_different_answer
                                          n_rows 100 (one leaf): profile oracle == serial oracle
                                          bit for bit; n_rows 2048 (P = 16): they differ at 8/8
                                          colnorms, 2/8 coefficients, 2044/2048 residual cells
                                          (first: 0x400bd882 vs 0x400bd886) -- the tree is reached
    check_cd_device_equals_oracle         THE GATE, three fixtures, IDENTICAL asserts:
                                            planted  2048x16 intercept, alpha 0.01 l1 1.0: 0 cells,
                                                     20-record cards agree on every stage
                                            denormal 1024x4, y = X[:,0]*1e-37, alpha 0: 0 cells,
                                                     12 records agree
                                            large    20000x4 (P = 157), alpha 0.005 l1 0.5: 0 cells,
                                                     17 records agree
                                          FAST REPORTS: 2/16 coef, 2014/2048 resid (planted);
                                          2/4, 965/1024 (denormal; the FAST oracle keeps
                                          denormals the device flushes); 0/4, 1077/20000 (large)
    check_cd_is_launch_invariant          THE HEADLINE: coef + residual + intercept + n_iter bytes
                                          (2066 floats) identical across axpy block 256/64, 1-D/2-D
                                          axpy grid, gemm plans AUTO/FLAT/SPLITK_STAGED/SPLITK for
                                          the dot, 0/37/5 floats of buffer padding, poisons
                                          -987654 / +13.5 / NaN / -0.0; the padding is checked
                                          untouched after the fit; A vs A' run-to-run control 0.
                                          IDENTICAL asserts; FAST reports (also 0 on this M4)
    check_cd_elasticnet_arms_reach        ||w||_1: lasso 4.1413, enet(0.5) 3.9062, alpha 0 4.6500
                                          (float64 4.6500); fit_intercept=False returns 0.0 and
                                          moves 2 cells; l2 arm moves 2 cells, alpha=0 arm 8
    check_cd_predict_matches_host         IDENTICAL: 0 of 1000 rows differ from the gemm OP_TN cell
                                          + flushed add; FAST: 494 differ, worst 5.5e-6 relative
    check_cd_card_is_emitted              20 records, the tag list above, control identical
    check_cd_soft_threshold_operand_order a REPORT, meaningful in the SOFT_SWAP build (below)

On batch composition: a per-coordinate solver has no batch axis of its own
-- each dot is one `1 x 1` cell and each column norm one launch -- so the
composition arm is the gemm lane's (`check_device_is_batch_invariant` there)
and what this section varies is the plan, the grid and the padding.

## Sabotages (each a build define; each restored; each output verbatim)

| define | what it breaks | result |
|---|---|---|
| `MOJOLEARN_GEMM_SABOTAGE_FOLD_SERIAL=1` | the dot's fold ACROSS leaves goes serial (the gemm lane's switch; this section owns no fold) | `check_cd_device_equals_oracle FAILED under IDENTICAL: 24700 disagreements` -- planted: coef 2/16, resid 2009/2048, first card divergence `cd.mu_input`; denormal: first divergence `cd.colnorm`; large: coef 1/4, resid 19880/20000 |
| `MOJOLEARN_GEMM_SABOTAGE_LEAF_ROTATE=1` | the leaf at tree position `t` is ROTATED BY THE BLOCK INDEX (the brief's "rotate the reduction start by block index") | `FAILED ... 24981 disagreements` -- large (P = 157, the SPLITK leaf kernel spans two blocks): coef 1/4, resid 20000/20000, first divergence `cd.mu_input`; planted and denormal (P <= 16, one block, rotation by 0) UNCHANGED, as the mechanism predicts |
| `MOJOLEARN_CD_SABOTAGE_NO_FTZ_RESID=1` | the ORACLE's residual flush dropped (store and re-read). On Apple the device's flush is hardware, so the host side is where the seam's reach can be shown on this column | `FAILED ... 1025 disagreements` -- denormal fixture ONLY: resid 1024/1024, `first resid cell 0 device 0x00000000 oracle 0x80019511`, first card divergence `7 cd.sweep000.resid`; planted and large unchanged (no denormal is reachable there). Found while building it: dropping only the STORE-side flush is invisible at `n_cols > 1`, because the next coordinate's axpy flushes the denormal residual on load; the seam is the pair and the sabotage drops both |
| `MOJOLEARN_CD_SABOTAGE_SOFT_SWAP=1` | `-(l1_alpha - coef)` for `coef - l1_alpha`, `l1_alpha + coef` for `coef + l1_alpha` in `cdUpdateCoefKernel` | `REPORT: 0 coef cells and 0 residual cells moved` -- IEEE subtraction is exactly anticommutative and addition commutative in round-to-nearest; the operand order is not a bit |

A fused-versus-unfused sabotage of the axpy is NOT listed because it cannot
be seen on this column: Metal through MAX contracts `a*b+c` by default, so
`identical_mul_add` is bit-inert here (gemm contract 4.1). The pin's value
is the first non-contracting backend; the reach proof there is the device
oracle arm the gemm lane describes.

## UNPORTED (the short form; `solver/UNPORTED.tsv` is the parameter-by-parameter list)

- `shuffle = true` / `selection='random'`: REFUSED BY NAME in both modes.
  `cd.cuh:164-166` seeds `std::mt19937 g(rand())` and then `initShuffle`
  RESEEDS it with `random_state = 0`, so their permutation sequence is
  `std::shuffle` over `mt19937(0)` -- and the C++ standard does not specify
  `std::shuffle`'s algorithm (libstdc++ and libc++ draw differently), so
  the permutation is not a pure function of the seed across toolchains. An
  exact port (mt19937 is fully specified; libstdc++'s `std::shuffle` would
  have to be transcribed and NAMED as the algorithm) would be DEVIATION
  611 with its own gate; it is reserved, not spent.
- `sample_weight`: the weighted `preProcessData`, the `sqrt`-weight scaling
  of `input` and `labels` and its undo (`cd.cuh:136-163, 240-251`).
- `loss != SQRD_LOSS`: theirs asserts too.
- RAFT `coalescedReduction`'s Thin and Thick kernels and the SM-count
  dispatch (FAST uses Medium at every shape).
- `linearRegLossGrads`, `penalty.cuh`: the SGD solver's.

## HAND-OFF TO THE IDENTITY LANE

1. **Estimators.** `mojolearn.Lasso(alpha=1.0, fit_intercept=True,
   max_iter=1000, tol=1e-3, selection='cyclic')` and
   `mojolearn.ElasticNet(alpha=1.0, l1_ratio=0.5, ...)` over
   `solver/ported/solver/cd.mojo::cd_fit` / `cd_predict` (F-order input,
   `coef` zeroed by the caller, `selection='random'` and `sample_weight`
   refused by name, `solver='qn'` refused by name, `positive`/`precompute`/
   `warm_start` refused by name as cuML's `_params_from_cpu` does). The
   binding goes beside `LinearRegression`/`Ridge` in
   `bindings/_mojolearn_estimators.mojo` and `python/mojolearn/linear_model.py`;
   the card prefix is `cd`.
2. **`solver/ported/glm/preprocess.mojo` belongs under `glm/ported/glm/`**
   (it is cuML's `glm/preprocess.cuh`, which `glm/UNPORTED.tsv` lists as
   NOT PORTED). Move it and delete that UNPORTED entry in the same commit;
   `glm/estimator.mojo`'s host-side centering could then call it.
3. **pixi lines** (pixi.toml is not this lane's):

        check-cd = "tools/with_build_lock.sh pixi run mojo run -I . solver/mojo_only/cd_check.mojo"
        check-cd-identity = "tools/with_identical_mode.sh pixi run mojo run -I . solver/mojo_only/cd_check.mojo"
        cd-main = "tools/with_build_lock.sh pixi run mojo run -I . solver/cd_main.mojo"

4. **The cross-vendor leg**: `solver/cd_main.mojo` under
   `MOJOLEARN_IDENTITY_TRACE` on the H100 and the MI300X/MI325X at the
   default fixture, diffed against the Mac card with
   `tools/identity_trace_diff.py`; and `cd_check.mojo` under IDENTICAL on
   both. The expected first divergence, if any, is `cd.input.x` (a host
   fixture is assembled in Float64 and rounded once per cell, so it should
   not move) and then `cd.colnorm`.
5. **`SUPPORT_MATRIX.md` / `IDENTITY_PATHS.md`**: the row below.

## ROW TEXT FOR THE IDENTITY LANE

| 40 | **coordinate descent (Lasso / ElasticNet): the row-length reductions and the epoch count** -- `solver/ported/solver/cd.mojo` (cuML `solver/cd.cuh`): `colNorm` and the means on RAFT `coalescedReduction`, whose kernel is CHOSEN BY SM COUNT (`-inl.cuh:497`) and compensates per thread then folds with CUB; `dot(X[:,ci], residual)` on cuBLAS `gemv`; `coef > l1_alpha` and `diffMax / coefMax < tol` branch on those bits, so `n_iter` is per card | yes: three fold shapes, one of them closed, one of them a device property | DEVIATION 610: all four reductions are the `gemm.fp32.v1` `OP_NT` cell at `1 x 1 x n_rows` (`solver/mojo_only/profile_dot.mojo`, CALLED from the gemm lane; the means are the dot against ones); axpys `identical_mul_add` + `ftz` at the residual (store and load); `cdUpdateCoefKernel` flushes quotient, diff, |r|; card `cd.colnorm/squared/sweepNNN.coef/resid/conv/final.coef/intercept/n_iter` | **CONSTRUCTION plus one Apple device's gates, 2026-08-23**: device == host oracle bit for bit on every stage and cell of three fixtures incl. a denormal-residual one; launch-invariant across 4 plans / 2 block sizes / 2 grids / 3 paddings; FAST vs IDENTICAL differ in 2 of 16 coefficients on the planted fixture; three sabotages fail where predicted (FOLD_SERIAL everywhere, LEAF_ROTATE only at P = 157, NO_FTZ_RESID only on the denormal fixture); `shuffle=true` refused by name (611 reserved); no second vendor has run this |
