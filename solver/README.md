# solver: coordinate descent (Lasso / ElasticNet), from cuML `cpp/src/solver/`

Seventh section. **COPY, DO NOT IMPROVE.** `cdFit` and `cdPredict`
(`upstream/cuml-v26.08.00/cpp/src/solver/cd.cuh`) and every RAFT primitive
their control plane calls, file for file, under `solver/ported/`; the one
thing cuML never needed -- a row-length reduction with ONE fold shape on
three vendors -- under `solver/mojo_only/`. DEVIATION 610 is the identity
construction; 611 is reserved and NOT spent (see `shuffle` below); 612 is
the card's NaN canonicalization and 613 the NaN/inf parameter refusals
(the ROW 39 AUDIT section); 614-619 are unused.

## Status: CERTIFIED Apple M4 <-> NVIDIA H100 at leg 11 (commit 144aa5b, judged by tools/e3_round_judge.sh section 7 on 2026-08-23): the IDENTICAL card is bit-identical across the two vendors, 20 stages; the FAST cards differ, recorded, the shipped arm makes no cross-vendor claim; AMD MI325X is OWED (that leg was not run).

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
    # the same through the registered pixi tasks (no *-identity task exists by design)
    pixi run check-cd
    tools/with_identical_mode.sh pixi run check-cd

    # the driver: one Lasso fit on the planted fixture, the oracle's verdict, predict
    tools/with_build_lock.sh     pixi run mojo run -I . solver/cd_main.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . solver/cd_main.mojo
    pixi run cd-main

    # a sabotage build (the define goes right after `mojo run`)
    tools/with_identical_mode.sh pixi run mojo run -D MOJOLEARN_CD_SABOTAGE_ZERO_FOLD_MAX=1 -I . solver/mojo_only/cd_check.mojo

    # the identity card
    MOJOLEARN_IDENTITY_TRACE=/tmp/cd.apple.card \
        tools/with_identical_mode.sh pixi run mojo run -I . solver/cd_main.mojo
    python3 tools/identity_trace_diff.py /tmp/cd.apple.card /tmp/cd.other.card

`cd_main.mojo` reads `MOJOLEARN_CD_ROWS/COLS/ALPHA/L1_RATIO/FIT_INTERCEPT/
EPOCHS/TOL` (defaults 2048 / 16 / 0.01 / 1.0 / 1 / 1000 / 1e-3). The pixi
tasks `check-cd` and `cd-main` are registered by the identity lane; the
lines above are the whole interface.

## What is ported

| ours | theirs | what |
|---|---|---|
| `solver/ported/solver/cd.mojo` | `cuml cpp/src/solver/cd.cuh` | `cdFit` line for line (guards, preprocess, `l1_alpha`/`l2_alpha`, colNorm + addScalar, the five-operation coordinate step with DEVICE-pointer alphas, one host read per epoch, the stopping rule, postprocess), `cdUpdateCoefKernel` transcribed, `cdPredict` |
| `solver/ported/solver/shuffle.mojo` | `cd solver/shuffle.h` | `initShuffle` (the identity permutation); `shuffle` REFUSED, see below |
| `solver/ported/solvers/params.mojo` | `cuml include/cuml/solvers/params.hpp` | the three enums |
| `glm/ported/glm/preprocess.mojo (moved there 2026-08-23 by the identity lane, per the hand-off)` | `cuml cpp/src/glm/preprocess.cuh` | `preProcessData` / `postProcessData`, unweighted arms, IN PLACE as theirs (HAND-OFF: it belongs under `glm/ported/`) |
| `solver/ported/functions/linear_reg.mojo` | `cuml cpp/src_prims/functions/linearReg.cuh` | `linearRegH` only |
| `solver/ported/linalg/coalesced_reduction.mojo` | `raft linalg/detail/coalesced_reduction-inl.cuh` | `coalescedSumMediumKernel<256>` with its per-thread Kahan-Babushka-Neumaier sum and two `BlockReduce.Sum`s -- the FAST arm of colNorm and mean |
| `solver/ported/linalg/norm.mojo` | `raft linalg/detail/norm.cuh` | `colNorm<L2Norm, rowMajor=false>` (a SUM OF SQUARES; no sqrt) |
| `solver/ported/stats/mean.mojo` | `raft stats/detail/mean.cuh`, `mean_center.cuh` | `mean<false>` = sum times `1/N` (a multiply), `meanCenter`/`meanAdd` as `matrixVectorOp` along columns |
| `solver/ported/linalg/axpy.mojo` | `raft linalg/detail/axpy.cuh` | `axpy<T, DevicePointerMode=true>` -- cuBLAS axpy with a DEVICE alpha; MAX has no axpy, so the mirror is one thread per row reading alpha from device memory |
| `solver/mojo_only/profile_dot.mojo` | none | DEVIATION 610: every `n_rows` reduction under IDENTICAL is the `mojolearn.identical.gemm.fp32.v1` `OP_NT` cell at `1 x 1 x n_rows`, CALLED from the gemm lane (`identical_gemm_with_plan`) -- no second fold is spelled; the sum for the means is the dot against ones |
| `solver/mojo_only/cd_oracle.mojo` | none | the host oracle (device arithmetic, serial), the whole-row serial variant, the Float64 reference, the hashed fixtures |
| `solver/mojo_only/record_canon.mojo` | none | DEVIATION 612: every float stage of the card is hashed through a copy whose NaNs are rewritten to the one payload `0x7FC00000` (IDENTITY_PATHS row 39 FACT 2) |
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
                                          n_rows=1, alpha<0, l1_ratio=1.5, predict loss=HINGE,
                                          alpha=NaN, alpha=inf, l1_ratio=NaN, tol=NaN (DEVIATION
                                          613): each raises naming the parameter
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
    check_cd_signed_zero_coefficients     ROW 39: alpha 1e36 / l1_ratio 0 / tol 0 on 256x8 flushes
                                          every quotient to a zero SIGNED by its dot: 4 -0.0 and
                                          4 +0.0 coefficients, coef[0] -0.0 in the `pos` fixture
                                          and +0.0 in the negated one (both orders); device card ==
                                          oracle card on every stage (incl. `.conv`), cells bit for
                                          bit, coefMax/diffMax +0.0 (0x00000000). IDENTICAL asserts;
                                          FAST RECORDED (the FAST oracle keeps the subnormal
                                          quotient 0x801d0718 the Apple device flushes)
    check_cd_nan_payload_is_canonical     DEVIATION 612: a +inf label puts a COMPUTED NaN in 256/256
                                          residual cells (device 0x7fc00000, arm64 host 0x7fc00000),
                                          a 0x7FC0BEEF label a PROPAGATED one in 1/256 (device
                                          0x7fc00000, host 0x7fc0beef); canonicalized cards agree on
                                          every stage, cells agree after canonicalization. IDENTICAL
                                          asserts; FAST RECORDED

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
| `MOJOLEARN_CD_SABOTAGE_ZERO_FOLD_MAX=1` | the `coefMax` fold as the hardware `max(conv, cand)` with `cand` = the SIGNED `r` when `r` is a zero (row 39) | `check_cd_signed_zero_coefficients FAILED under IDENTICAL (zero-fold sabotage ZERO_FOLD_MAX): 1 of 2 fixtures` -- `signed_zero_neg` (whose LAST coordinate is -0.0): `first card divergence: 8 cd.sweep000.conv`; `signed_zero_pos` (last coordinate +0.0) inert. Apple returns the SECOND operand of `max(+0,-0)`; NVIDIA/AMD's IEEE-2019 maximum returns +0.0, so this sabotage is PREDICTED INERT there -- one source, two cards |
| `MOJOLEARN_CD_SABOTAGE_ZERO_FOLD_MAX_SWAPPED=1` | the same with the operands swapped, `max(cand, conv)` | `check_cd_signed_zero_coefficients OK [IDENTICAL] ... (zero-fold sabotage ZERO_FOLD_MAX_SWAPPED)` -- APPLE-INERT (the +0.0 seed is the second operand) and predicted inert on NVIDIA/AMD too (IEEE maximum): a spelling that is right everywhere by accident of operand order, which is exactly why a hardware max is not the spelling -- the two sabotages differ on Apple alone |
| `MOJOLEARN_CD_SABOTAGE_NO_NAN_CANON=1` | both cards hash raw NaN bits (DEVIATION 612 off) | `check_cd_nan_payload_is_canonical FAILED under IDENTICAL (canon sabotage NO_NAN_CANON): 1 of 2 fixtures` -- `nan_propagated`: `first cell 9 device 0x7fc00000 oracle 0x7fc0beef; first card divergence: 7 cd.sweep000.resid`; `nan_computed` INERT on this machine (the M4's GPU and its arm64 host both default to 0x7fc00000) and predicted to FAIL on the NVIDIA box (device 0x7fffffff against an x86 host's 0xffc00000) |

A fused-versus-unfused sabotage of the axpy is NOT listed because it cannot
be seen on this column: Metal through MAX contracts `a*b+c` by default, so
`identical_mul_add` is bit-inert here (gemm contract 4.1). The pin's value
is the first non-contracting backend; the reach proof there is the device
oracle arm the gemm lane describes.

## ROW 39 AUDIT (2026-08-23): signed zero, NaN payloads, FAST-mode gates

IDENTITY_PATHS row 39, measured the same day on Apple M4 / NVIDIA H100 /
AMD MI325X: `max(+0.0, -0.0)` is -0.0 on Apple (the second operand) and
+0.0 on NVIDIA/AMD (IEEE-2019 maximum); a computed NaN's payload is the
vendor's (0x7fc00000 / 0x7fffffff / 0xffc00000); the phase-6 gate scripts
run the FAST pass too. Every float `max`/`min`/`abs`/clamp/compare-fold
site in `solver/` was read against those three facts.

### Sites reviewed

| site | what it is | can +-0.0 / NaN reach it | verdict |
|---|---|---|---|
| `solver/ported/solver/cd.mojo` `cd_update_coef_kernel`, the soft threshold `c > l1_alpha ? c - l1_alpha : (c < -l1_alpha ? c + l1_alpha : 0)` | a branch, no max/min | `c` can be -0.0 (a flushed negative dot): both compares are false, `r = +0.0` literal; a nonzero arm's result is strictly signed (`c > a` implies `c - a > 0`); a NaN `c` takes the `0` arm (both compares false) | no hardware max/min; UNCHANGED. The SOFT_SWAP sabotage (below) already shows the operand order is not a bit |
| the same kernel, `r = r / sq` then `ftz(r)` | a quotient that CAN flush to a SIGNED zero | yes: `dot / squared` below the normal floor is -0.0 for a negative dot (Apple's hardware flush and `ftz` agree, row 10); stored to `coef` and negated into `conv[0]` | IEEE division, negation and the signed flush are vendor-invariant; the -0.0 is a RECORDED bit (`cd.sweepNNN.coef`, `.conv`) and `check_cd_signed_zero_coefficients` plants it and compares the card bit for bit. PROVEN in a comment at the site |
| the same kernel, `diffMax`/`coefMax` folds: `if conv[2] < diff: conv[2] = diff`, `if conv[1] < absv: conv[1] = absv` | two max folds spelled as strict compares | candidates are `abs()` values (`abs(-0.0)` is +0.0, the sign bit is cleared), seed is the +0.0 `enqueue_memset` wrote, strict `<` keeps the EARLIER value on a tie by position, `x < NaN` is false so a NaN never enters | PROVEN in a comment at the site; no hardware max; UNCHANGED. The -0.0 fixture reaches both folds with both zeros in both orders; the ZERO_FOLD_MAX sabotage shows what a hardware-max spelling would do |
| `solver/mojo_only/cd_oracle.mojo` `cd_oracle_fit`, the same folds on the host | the oracle's spelling | identical reasoning | comment added; asserted in both modes that the sign bit of `coefMax`/`diffMax` is never set |
| `solver/ported/linalg/coalesced_reduction.mojo:84` `if abs(sum) >= abs(cur)` | RAFT's KBN branch selecting the compensation FORMULA | both operands `abs()`; on a magnitude tie the two formulas compute the same `c`; NaN takes the second arm as RAFT's does | PROVEN in the docstring; FAST-only (comptime assert); UNCHANGED |
| `solver/ported/stats/mean.mojo`, `glm/preprocess.mojo`, `linalg/norm.mojo`, `linalg/axpy.mojo`, `functions/linear_reg.mojo` | sums, a multiply by `1/N`, `x - mu`, `fma`, `x + s`; `norm` is a SUM OF SQUARES with NO sqrt | no clamp, no max/min, no sqrt, no division by a data value anywhere (`ratio = 1/N` with `n_rows >= 2`); a zero column gives `colnorm = 0`, `squared = l2_alpha`, and `cd.cuh:62`'s `1e-5` guard zeroes the coefficient; a zero-variance column under `fit_intercept` centers to the zero column | nothing to change |
| `solver/mojo_only/profile_dot.mojo:64,88` `if p > w`, `if p < 0` | Int | not floats | nothing |
| `solver/mojo_only/cd_check.mojo` `if e > worst` (tolerance scans) | host Float64 `abs()` maxima | report values only | nothing |

No `max(`/`min(` on a float, no `.clamp()`, no `reduce_max/min`, no
`pinned_block_max/min`, no `Atomic.max/min`, no `copysign` exists in
`solver/`. No clamp was rewritten because there is none.

### FACT 2, the recorded stages (DEVIATION 612, `solver/mojo_only/record_canon.mojo`)

cuML's `cdFit` has no finiteness guard. With FINITE inputs of moderate
scale no stage can hold a NaN (a zero column, a zero-variance column,
`alpha = 0`, `l1_ratio` at either end, `tol = 0`, `max_iter = 0` were each
traced: zeros and infs at worst, never `0/0`). Two things CAN put one on
the card: a non-finite label or design cell, and an overflow of
`x * residual` past FLT_MAX (`inf - inf` in a dot; `(-inf) * x + inf` in
the second axpy). The soft threshold maps a NaN dot to a `0` coefficient
(both compares false) but the residual keeps the NaN, and the payload is
the vendor's. Decision per stage:

| stage | NaN reachable | handled |
|---|---|---|
| `cd.input.x`, `cd.input.y` | a user NaN (its own payload, the same bits on every vendor) | canonicalized at the record (so a user payload and a propagated one hash alike) |
| `cd.l1_alpha`, `cd.l2_alpha` | only from a NaN/inf parameter | **DEVIATION 613**: `alpha` non-finite, `l1_ratio` NaN, `tol` NaN are REFUSED BY NAME (`cd.mojo`, the block banner there); after that they are finite-times-finite, finite or +inf, never NaN; recorded raw |
| `cd.mu_input`, `cd.mu_labels`, `cd.colnorm`, `cd.squared` | from a non-finite input or an overflowing sum of squares (`inf`, not NaN, unless the input holds a NaN) | canonicalized |
| `cd.sweepNNN.coef/resid/conv`, `cd.final.coef` | yes (the two mechanisms above) | canonicalized |
| `cd.intercept` | `mu_labels - mu_input . coef` with an inf coefficient | canonicalized (`canon_nan_f32` on the host scalar) |
| `cd.n_iter` | integer | no |

`record_device_canon` copies the buffer, rewrites every NaN to
`0x7FC00000` (a bit WRITE, the payload numerics.mojo's `identical_log`/
`identical_sqrt` already use) and hashes the copy; the live buffers are
untouched (a NaN payload steers no downstream bit: every consumer is a
payload-blind comparison or an op that yields a NaN again). The oracle
card does the same through `canon_nan_list`. Gate:
`check_cd_nan_payload_is_canonical` (two fixtures, outputs in the gates
list above); sabotage `NO_NAN_CANON` (table above): FAILS on this machine
on the PROPAGATED fixture (the M4's GPU canonicalizes a propagated payload
to 0x7fc00000, the arm64 host keeps 0x7fc0beef), INERT here on the
COMPUTED fixture (GPU and arm64 host both default to 0x7fc00000), and
predicted to fail on the NVIDIA box on both (device 0x7fffffff, x86 host
0xffc00000).

### The -0.0 fixture and its sabotages

`fixture_signed_zero` (256 x 8, alpha 1e36, l1_ratio 0, tol 0, 2 epochs):
`squared` is ~2.56e38 and every `dot / squared` flushes to a zero signed
by its dot -- 4 coefficients -0.0 and 4 +0.0, `coef[0]` -0.0 in the `pos`
fixture and +0.0 in the negated one, so the folds see both zeros in both
orders. Under IDENTICAL: device card == oracle card on every stage, every
cell bit for bit, `coefMax`/`diffMax` 0x00000000 in every sweep. Under
FAST: RECORDED (the host oracle keeps the subnormal quotient 0x801d0718
that the Apple device flushes; a denormal-keeping column would keep it on
the device too).

- `ZERO_FOLD_MAX` (hardware `max(conv, cand)`, the signed zero as a
  candidate): FAILS on Apple on the fixture whose LAST coordinate is -0.0
  (`first card divergence: 8 cd.sweep000.conv`), inert on the other; Apple
  returns the second operand. PREDICTED INERT on NVIDIA and AMD (IEEE
  maximum returns +0.0 whatever the order) -- the same source would give
  two different cards, which is the defect the `abs()`-and-strict-`<`
  spelling does not have.
- `ZERO_FOLD_MAX_SWAPPED` (`max(cand, conv)`): APPLE-INERT (the +0.0 seed
  is the second operand) and predicted inert on NVIDIA/AMD. Right
  everywhere by accident of operand order; the two sabotages differ on
  Apple alone, which is the whole point.

For a MAX fold of `abs()` values the correct answer is +0.0 and IEEE
maximum agrees with it, so no sabotage of THIS site can fail on NVIDIA/
AMD; a `min` fold or a `max` over signed values would have the mirror
image. Nothing in `solver/` spells either.

### FAST demotions (FACT 3)

`_record_or_raise` in `cd_check.mojo`: asserts under IDENTICAL, prints
`RECORDED [FAST] ...` and continues under FAST. Demoted: the 2e-3
tolerance against the Float64 reference (coef and intercept,
`check_cd_recovers_the_planted_support`; alpha=0 coef in
`check_cd_elasticnet_arms_reach`), the 1e-4 relative predict tolerance
(`check_cd_predict_matches_host`), the run-to-run control at one launch
(`check_cd_is_launch_invariant`) and the run-to-run card equality and
`n_iter` (`check_cd_card_is_emitted`) -- all of them claims about a
vendor gemv's rounding or repeatability. Kept as assertions in both modes:
every refusal by name, the record count and tag list, the planted-support
decision (a ~50x margin: `l1_alpha` 20.48 against a null column's dot of
~0.4), `n_iter` being a converged count, the norm ordering and bit-movement
reach checks, `fit_intercept=False` returning the 0.0 literal, and every
host-only computation.

### Outputs

    tools/with_build_lock.sh     pixi run mojo run -I . solver/mojo_only/cd_check.mojo
    == solver/mojo_only/cd_check.mojo [FAST] ALL PASSED ==
    tools/with_identical_mode.sh pixi run mojo run -I . solver/mojo_only/cd_check.mojo
    == solver/mojo_only/cd_check.mojo [IDENTICAL] ALL PASSED ==
    tools/with_identical_mode.sh pixi run mojo run -I . solver/cd_main.mojo
    [IDENTICAL] oracle: n_iter=3 intercept 0x3ad72d10; coefficients differing from the device: 0 of 16
    (card: 20 records)

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
2. **`glm/ported/glm/preprocess.mojo (moved there 2026-08-23 by the identity lane, per the hand-off)` belongs under `glm/ported/glm/`**
   (it is cuML's `glm/preprocess.cuh`, which `glm/UNPORTED.tsv` lists as
   NOT PORTED). Move it and delete that UNPORTED entry in the same commit;
   `glm/estimator.mojo`'s host-side centering could then call it.
3. (done by the identity lane: `check-cd` and `cd-main` are registered; FAST
   via `pixi run check-cd`, IDENTICAL via `tools/with_identical_mode.sh pixi
   run check-cd`.)

4. **The cross-vendor leg**: `solver/cd_main.mojo` under
   `MOJOLEARN_IDENTITY_TRACE` on the H100 and the MI300X/MI325X at the
   default fixture, diffed against the Mac card with
   `tools/identity_trace_diff.py`; and `cd_check.mojo` under IDENTICAL on
   both. The expected first divergence, if any, is `cd.input.x` (a host
   fixture is assembled in Float64 and rounded once per cell, so it should
   not move) and then `cd.colnorm`.
5. **`SUPPORT_MATRIX.md` / `IDENTITY_PATHS.md`**: the row below (row 41 in
   the ledger).

## ROW TEXT FOR THE IDENTITY LANE (row 41)

| 41 | **coordinate descent (Lasso / ElasticNet): the row-length reductions and the epoch count** -- `solver/ported/solver/cd.mojo` (cuML `solver/cd.cuh`): `colNorm` and the means on RAFT `coalescedReduction`, whose kernel is CHOSEN BY SM COUNT (`-inl.cuh:497`) and compensates per thread then folds with CUB; `dot(X[:,ci], residual)` on cuBLAS `gemv`; `coef > l1_alpha` and `diffMax / coefMax < tol` branch on those bits, so `n_iter` is per card | yes: three fold shapes, one of them closed, one of them a device property | DEVIATION 610: all four reductions are the `gemm.fp32.v1` `OP_NT` cell at `1 x 1 x n_rows` (`solver/mojo_only/profile_dot.mojo`, CALLED from the gemm lane; the means are the dot against ones); axpys `identical_mul_add` + `ftz` at the residual (store and load); `cdUpdateCoefKernel` flushes quotient, diff, |r|; card `cd.colnorm/squared/sweepNNN.coef/resid/conv/final.coef/intercept/n_iter` | **CONSTRUCTION plus one Apple device's gates, 2026-08-23**: device == host oracle bit for bit on every stage and cell of three fixtures incl. a denormal-residual one; launch-invariant across 4 plans / 2 block sizes / 2 grids / 3 paddings; FAST vs IDENTICAL differ in 2 of 16 coefficients on the planted fixture; three sabotages fail where predicted (FOLD_SERIAL everywhere, LEAF_ROTATE only at P = 157, NO_FTZ_RESID only on the denormal fixture); `shuffle=true` refused by name (611 reserved); ROW 39 AUDIT 2026-08-23: no float max/min/clamp in the section, the coefMax/diffMax folds are abs()-and-strict-`<` (proven at the site, -0.0 planted in both orders and card-compared), the card's NaN stages are canonicalized to 0x7FC00000 (DEVIATION 612, planted computed and propagated NaN), non-finite alpha / NaN l1_ratio / NaN tol refused by name (DEVIATION 613), vendor-shaped FAST claims RECORDED; CERTIFIED Apple M4 <-> NVIDIA H100 at leg 11 (commit 144aa5b, judged by tools/e3_round_judge.sh section 7 on 2026-08-23): the IDENTICAL card is bit-identical across the two vendors, 20 stages; the FAST cards differ, recorded, the shipped arm makes no cross-vendor claim; AMD MI325X is OWED (that leg was not run) |
