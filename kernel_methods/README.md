# kernel_methods: kernel ridge, Nystroem and random Fourier features, FP32

Opened 2026-08-25. Three estimators that are almost entirely made of
primitives this repository already has: a pinned GEMM, a pinned Cholesky, a
pinned Jacobi eigensolver, a ported RBF expansion, a ported L1 distance and a
ported Philox generator. **DEVIATIONS 1660-1689 are this lane's and all
thirty are spent**; the next lane in this range needs its own block.

## Status

CORRECTED 2026-09-01, by diffing the cards rather than reading a status block: RAN ON AMD under IDENTICAL on the 2026-08-28 leg, 18 checks OK. Two cards exist on Apple at md5 18f2892e926f0d1b, 29 stages; no card was emitted on AMD, so there is no byte comparison yet.  The sentence that stood here said no second vendor had run this lane, and commit 89c1920c fixed the adjacent false sentence in the same paragraph while leaving this one standing.
HAS RUN THIS UNDER IDENTICAL, so there is no identity card outside the M4.**
An NVIDIA H100 compiled and ran all three estimators under FAST on
2026-08-26
(`bench/results/fast_speed/2026-08-26_040100-nvidia-classical.md`), which is
a speed leg and not an identity one.
`pixi run check-kernel-methods` is green in FAST and green
under `tools/with_identical_mode.sh`, **18 checks each**, 13 sabotage arms
driven at run time with no source edited and no rebuild.

    check_nystroem_full_equals_exact_kernel OK: at q = n the embedding's
      Gram equals the exact kernel matrix BIT FOR BIT on FIX_KM_ORTHO,
      64 cells, in both modes
    check_kernel_ridge_planted_linear OK: at alpha = 0 the 8 dual
      coefficients, the 8 training predictions and all 32 PRIMAL weights
      equal their hand-derived values BIT FOR BIT
    check_launch_invariance OK [IDENTICAL]: dual coefficients, Nystroem
      embedding, random weights and feature map byte-identical across two
      threads-per-block choices, and one query row transformed ALONE
      equals the same row inside a batch, for all three estimators

**The one measured win this lane has, and it is a FAST one.** On an NVIDIA
H100 on 2026-08-26, kernel ridge at fixture `RBF.16x5` ran in 0.336 ms
against cuML's GPU `KernelRidge` at 2.256 ms, which is **6.71x FASTER**;
fit plus predict are inside the clock on both sides, with `kernel`, `gamma`
and `alpha` passed explicitly
(`bench/results/fast_speed/2026-08-26_040100-nvidia-classical.md`). Nystroem
ran at 0.424 ms (`RBF.16x5q8`) and RBFSampler at 0.295 ms (`3x5q8`), both
with NO OPPONENT ON THIS BOX, because RAPIDS ships neither and the
scikit-learn CPU arm is refused under GPU-PATH-ONLY. That artifact's header
records that at these fixture sizes both arms are dominated by launch and
dispatch cost, so the 6.71x is a fixed-cost figure and not a kernel verdict.
See WHAT THIS WILL COST.

### FIVE ARMS THAT COULD NOT FIRE, and what each one settled

**`KMSAB_RIDGE_RELATIVE` was inert at the RBF kernel**, which is the
`cholesky/` lane's finding arriving here exactly as that lane predicted.
A relative ridge is `A_ii + alpha * A_ii`; on any correlation kernel
`A_ii = k(x, x) = 1` exactly, so it IS the absolute ridge. Zero cells moved
on all five fixtures. The arm is now driven at the LINEAR kernel, whose
diagonal is `x . x`, and it moves.

**`KMSAB_STD_TRANSCENDENTAL` is inert under FAST and asserted under
IDENTICAL.** It swaps `identical_exp` for `std.math.exp`, and under FAST
`identical_exp` IS the vendor exponential, so the arm substitutes a call for
the same call. Same classification `gaussian_process/` gives its
`GP_SAB_STD_EXP` and the same shape as `hierarchy/`'s `LINK_SAB_STD_SQRT`.
The arm-count floor is mode dependent for this reason.

**The alone-versus-batch arm is IDENTICAL-only.** Under FAST the cross-kernel
product rides MAX's `linalg.matmul`, which may pick a different tile shape
and k-split at `nq = 1` than at `nq = 9`, and a k-split is a summation order.
Measured: 0xbea6b00f alone against 0xbea6b001 in a batch of 9. Under
IDENTICAL the product goes through `identical_gemm_into`, whose partition is
a function of the shape alone, and the compare passes.

**The random-feature weights are not bit-identical to the host map in both
modes, and the claim that they were is deleted.** The index arithmetic is
integer, but the Box-Muller transform on top is `log`, `sqrt` and `cos`.
Measured under FAST: device 0x3f93ac2a against host 0x3f93ac29, one ulp, at
flat index 0. Asserted under IDENTICAL, reported under FAST.

**A planted `-0.0` cannot reach the dual coefficients and no fixture fixes
it.** Each dual cell is a sum over every row, and `-0.0 + a == a` annihilates
the sign at the first nonzero addend. The reachable signed-zero gates are the
ones upstream of the solve, and those are asserted.

### The sigmoid kernel is not Mercer, and the refusal is correct

`tanh(gamma x.y + c0)` is positive semi-definite only for particular
parameter ranges, so `K + alpha I` stays indefinite until `alpha` exceeds the
most negative eigenvalue. Measured: refused at `info = 6` on a 12-row
fixture. cuML instead falls back to a least-squares solve behind a
`warnings.warn`, returning a different estimator than the caller asked for.
This lane refuses by name, and the check now asserts that refusal as the
expected outcome rather than tuning the ridge until it goes away.


**BUILT AND GATED ON ONE APPLE M4 IN BOTH MODES, 2026-08-25, AND RUN ON AN
NVIDIA H100 UNDER FAST ON 2026-08-26.**

Eighteen checks and thirteen run-time sabotage arms are written and green on
the M4 in both modes. What does NOT exist is a cross-vendor identity claim:
the NVIDIA leg was a FAST speed leg
(`bench/results/fast_speed/2026-08-26_040100-nvidia-classical.md`), so no
card outside the M4 has ever been compared against another. Where a table
below is an expectation rather than a transcript it is marked as one.

    pixi run check-kernel-methods                                              # FAST
    tools/with_build_lock.sh     pixi run mojo run -I . kernel_methods/checks/km_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . kernel_methods/checks/km_check.mojo

    MOJOLEARN_IDENTITY_TRACE=/tmp/mac.km.identical.card \
        tools/with_identical_mode.sh pixi run mojo run -I . kernel_methods/kernel_methods_main.mojo

## The upstream findings, per estimator

The brief that opened this lane asked for a check of the pinned cuML
checkout. Here is what is actually there at `265b9da`
(`upstream/cuml-v26.08.00`).

| estimator | is it in cuML? | what we do |
|---|---|---|
| **KernelRidge** | **YES, and it is PURE PYTHON.** `python/cuml/cuml/kernel_ridge/kernel_ridge.py`, 349 lines over cupy. There is NO `cpp/src/kernel_ridge/` at this pin. `_solve_cholesky_kernel` and `_safe_solve` are the whole algorithm and both are readable | **MIRRORED FILE FOR FILE** at `impl/kernel_ridge/kernel_ridge.mojo`, branch for branch, with the two closed calls underneath it (`cupyx.lapack.posv` = `cusolverDnpotrf` + `potrs`) replaced by `cholesky/`'s pinned pair. COPY DO NOT IMPROVE applies and is followed |
| **Nystroem** | **NO.** `grep -rn 'Nystroem' upstream/cuml-v26.08.00` returns NOTHING. cuVS has none either | **NO UPSTREAM EXISTS IN THE PINNED CHECKOUTS.** scikit-learn's `sklearn/kernel_approximation.py::Nystroem` is treated as ORACLE-ONLY and is not transliterated; `estimator.mojo` follows its `fit` step for step and cites it, and `DERIVATION_MAP.tsv` records the file as `mojo-only` rather than as a port |
| **RBFSampler** | **NO.** No `RBFSampler`, no `kernel_approximation`. `python/cuml/cuml/random_projection/` exists and is a JOHNSON-LINDENSTRAUSS random projection, which is a different estimator with a different purpose (dimensionality reduction preserving distances, not a kernel approximation), and its `johnson_lindenstrauss_min_dim` literally calls scikit-learn's | **NO UPSTREAM EXISTS IN THE PINNED CHECKOUTS**, same treatment |

**What DOES have an upstream, and is ported:**

- **The polynomial and sigmoid kernel epilogues.** cuVS `6ba2ce2` ships
  `cpp/src/distance/detail/kernels/kernel_matrices.cu` with four kernel
  types. `svm/impl/distance/kernel_matrices.mojo` already mirrors its
  LINEAR and RBF halves and refuses the other two by name. This lane ports
  those two, beside its own caller, and `DERIVATION_MAP.tsv` records the result
  as **a second partial mirror of one upstream file, neither half complete on
  its own.** DEVIATION 1664. `svm/` is not edited.
- **`next_float` and `box_muller_transform`.** RAFT `ebf9268`,
  `raft/random/detail/rng_device.cuh:481-487` and `:133-146`. `core/philox.mojo`
  ports RAFT's Philox generator, its `next_u64`, its `next_double` and its
  Lemire `uniformInt`, but not those two, because nothing here needed a
  float32 uniform or a Gaussian before. Ported at `impl/random/rng_device.mojo`.

**One correction to the brief, made against the checkout.** The brief said to
"check cuML too" for Nystroem and RBFSampler and to report. The report is
above and the answer is that neither exists; the file that a search for
"random features in cuML" lands on, `random_projection/random_projection.py`,
is `GaussianRandomProjection` / `SparseRandomProjection`, which approximate
DISTANCES rather than a kernel and share no code with an RFF map. It is named
here so the next person does not re-derive the search.

## WHAT THIS LANE REUSES RATHER THAN REWRITES

Nothing in the list below was re-implemented here, and each was checked for
before a line was written. **This is the section to read before adding
anything to this directory.**

| what | the file it lives in, and the exact entry point | why not a second copy |
|---|---|---|
| **Every matrix product**: the linear kernel, the polynomial and sigmoid dot, the Nystroem normalization, the Nystroem embedding, kernel ridge's prediction, and the random-feature projection | `gemm/checks/gemm_identical.mojo::identical_gemm_into` at `OP_NN` and `OP_NT`, sized by `identical_gemm_workspace_max_floats`, ops from `gemm/checks/gemm_oracle.mojo` | profile `mojolearn.identical.gemm.fp32.v1` is gated at 62 shapes across eight execution plans with six sabotages. **There is no hand-written contraction anywhere in `kernel_methods/`**, and `linalg.matmul` is refused on every path |
| **The normative answer for those products**, in the oracle | `gemm/checks/gemm_oracle.mojo::gemm_oracle` | `gemm_identical.mojo::contract_partition` records that this repository already shipped one re-spelling of the leaf rule and got it wrong |
| **The Cholesky factorization and both solves** | `cholesky/checks/potrf.mojo::potrf_lower`, `chol_workspace_floats`, `add_jitter`, `chol_jitter_pinned`; `cholesky/checks/trsm.mojo::cho_solve` | profile `mojolearn.identical.cholesky.fp32.v1`, gated green on an Apple M4 in both modes at `1339da7`. Kernel ridge IS a Cholesky solve and writing a second one would be writing a second `potrf` |
| **The float32 replay and the float64 reference of that solve** | `cholesky/checks/cholesky_oracle.mojo::oracle_potrf_lower`, `oracle_cho_solve`, `reference_potrf_lower_f64`, `reference_solve_f64` | `check_kernel_ridge_vs_oracle` and `check_signed_zero_reach` compare against that lane's own second spelling. No factorization is written in `kernel_methods/checks/km_oracle.mojo` |
| **The LINEAR and RBF kernel matrices**, including the expansion epilogue and the squared row norms | `svm/impl/distance/kernel_matrices.mojo::kernel_op`, `row_norms_l2sq`, `kernel_workspace_floats` -- a port of cuVS `kernel_matrices.cu` under that lane's DEVIATION 630 | already ported, already gated. **And the polynomial and sigmoid kernels reuse it too**: they need `X Y^T` and nothing else before their epilogue, which is what `kernel_op` computes at a `KERNEL_LINEAR` parameter block, so this lane obtains the dot by CALLING IT rather than by issuing its own GEMM |
| **`KernelParams` itself** (`{kernel, degree, gamma, coef0}` and the kernel enumeration) | `svm/impl/svm/svm_parameter.mojo` | one struct, one set of enum values, so a kernel id cannot mean two things in one repository. This lane adds exactly ONE value, `KM_KERNEL_LAPLACIAN = 5`, and adds it in its own file because `svm/` is not editable here |
| **The Manhattan distance the laplacian kernel needs** | `kde/impl/distance/distance.mojo::pairwise_distance` at `DIST_L1` (RAFT's `l1.cuh`, one thread per cell, ascending, every seam already flushed) | a second L1 distance would be a second thing to pin. The laplacian kernel here is that call plus a four-token `exp` epilogue |
| **The Jacobi eigensolver** | `decomposition/checks/jacobi_eigh_device.mojo::jacobi_eigh_kernel`, `JACOBI_TPB`, `JACOBI_TOL`, `JACOBI_SWEEPS`; and `decomposition/checks/jacobi_eigh.mojo::jacobi_eigh` for the float64 oracle | Nystroem's eigenstep IS a symmetric eigendecomposition. That file's block dim is a CONTRACT (its `pinned_block_sum` slab is `JACOBI_TPB` wide) and this lane passes the constant, as its three other call sites do |
| **THE EIGENVECTOR SIGN CONVENTION** | `decomposition/impl/linalg/detail/pca.mojo::sign_flip_kernel`, `SIGNFLIP_TPB` -- RAFT's `signFlipKernel`, `raft/matrix/detail/math.cuh:367` | **THE BRIEF ASKED THIS LANE TO PIN THE SIGN BY "A STATED RULE (for example, force the largest-magnitude component of each vector positive, ties broken on index)". THAT EXACT RULE ALREADY EXISTS, PORTED, WITH ITS OWN GATE AND ITS OWN THREE-WAY-TIE FIXTURE.** Inventing a second one would have been the single largest duplication available in this lane. DEVIATION 1668 |
| **The Philox generator, the derived key and the position map** | `core/philox.mojo::PhiloxState`, `custom_next_uniform_int_u32`, `philox_next_u64`; `resample/checks/index_map.mojo::resample_key`, `position_subsequence`, `key_lo`, `key_hi`, `key_join`, `draw_permutation_key`, `draw_uniform_in`, `permutation_key_lt` | `index_map.mojo`'s DEVIATION 1690 is exactly the property this lane's brief demands and is already argued and gated. Re-spelling five of its functions here would be five copies of a construction whose whole value is that there is one. **The cost is a shared kind-byte namespace and it is named** (DEVIATION 1679); `check_km_kind_bytes_are_disjoint` gates it |
| **Transcendentals and the arithmetic pins**: `identical_exp`, `identical_log`, `identical_tanh`, `identical_cos`, `identical_sin`, `identical_sqrt`, `identical_div`, `identical_mul`, `identical_mul_add`, `ftz` | `checks/numerics.mojo` | IDENTITY_PATHS rows 9, 10, 12 and 49. The only `std.math` in this directory is inside the float64 oracle and inside the `KMSAB_STD_TRANSCENDENTAL` sabotage arm, both by design |
| **Stage hashing and the differ** | `core/identity_trace.mojo::IdentityTrace`, `record_device`, `record_list_f32`, `record_list_i32`, `record_scalar_f32`, `first_divergence`, `read_trace_lines` | one hash function per repository |
| **The fixture hash and the RBF Gram helper** | `cholesky/checks/cholesky_fixture.mojo::chol_mix64`, `bits_value`, `exact_offdiag` | **THIS PAYS DOWN A DEBT RATHER THAN ADDING TO ONE.** That file's header names four existing copies of splitmix64 in this tree and calls it a debt. `kernel_methods/checks/km_fixture.mojo` makes no fifth copy; it imports. The brief blessed the cross-lane import explicitly for `rbf_gram` and the same argument covers the three functions its point sets are built from |
| **The pinned block fold**, considered and NOT used | `core/pinned_reduce.mojo` | named here because a reviewer will look for it. **No kernel this lane owns folds across threads at all** -- every one is one thread per output cell -- so there is no fold shape to pin and importing one would suggest there is. The three folds below these entry points (`identical_gemm`'s, the Jacobi's `pinned_block_sum`, `potrf_lower`'s trailing update) are each their own lane's |
| **Fixed-point accumulation**, considered and NOT used | `checks/fixed_point.mojo` | same. It exists to REPLACE float atomics, and this lane has none. There is no float atomic, no `Atomic.fetch_add`, no warp shuffle, ballot or vote, and no `block.sum` anywhere in `kernel_methods/` |
| **Top-k, k-means, tree builders** | `neighbors/`, `cluster/`, `gbdt/` | a kernel method needs none of them |

**One duplication is taken and it is named rather than hidden.**
`km_sabotage.mojo::sabotage_rbf_epilogue_kernel` is a copy of
`svm/impl/distance/kernel_matrices.mojo::rbf_kernel_expanded_kernel`, a
kernel in another lane. It has to be: `KMSAB_STD_TRANSCENDENTAL` must reach
the RBF exponential, `svm/` may not be edited here, and `svm/`'s own
`SAB_STD_EXP` is a compile-time define that cannot be selected at run time.
**If svm's epilogue changes, this copy is stale and the gate silently becomes
a comparison of two of our own old ideas.** `KMSAB_COPY_ONLY` and
`check_km_sabotage_copies_agree` exist for exactly that: the copies are
driven with no arm engaged and must reproduce the production kernels bit for
bit before any arm is believed.

**One name collides and it is not a mistake.**
`kernel_methods/checks/kernel_matrix.mojo` and the repository root's
`checks/kernel_matrix.mojo` are unrelated: the root one is the per-vendor
TUNABLES matrix (`lib_block_size_for`, `TARGET_COLUMN`), this one is about
kernel matrices in the machine-learning sense. The brief named the path, so
it is kept, and both file headers carry the disambiguation.

## The identity table (row text for `IDENTITY_PATHS.md`)

| n | path | what is vendor-dependent in the ordinary spelling | what we did | status |
|---|---|---|---|---|
| 67 | **`kernel_methods/` -- THE RIDGE AND THE JITTER ARE ONE KNOB, NOT TWO** (`impl/kernel_ridge/kernel_ridge.mojo::kernel_ridge_solve`) | a kernel-ridge solver has TWO plausible places to perturb the diagonal: the user's `alpha` and the linear-algebra layer's numerical jitter. If both fire, the factored matrix is `K + (alpha + jitter) I` while every docstring, every oracle and every user says `K + alpha I`, and the discrepancy is invisible because both answers look like ridge regression | **PIN.** `alpha` IS the ridge; `potrf_lower` is handed `+0.0`. Cited rather than re-derived: `cholesky/README.md` measured that the absolute and relative jitter policies COINCIDE on a unit diagonal, which is every correlation-shaped kernel. DEVIATION 1660 | **CONSTRUCTION 2026-08-25, NOTHING RUN.** `KMSAB_RIDGE_PLUS_JITTER` and `KMSAB_RIDGE_RELATIVE`, both driven over a fixture SWEEP because the second is nearly inert on a unit diagonal |
| 68 | **`kernel_methods/` -- THE POLYNOMIAL POWER** (`impl/distance/kernel_matrices.mojo::polynomial_epilogue_kernel`) | `pow(base, degree)` on a device is a vendor's `powf`, and the base of a polynomial kernel is NEGATIVE on roughly half of a centered Gram matrix. `checks/numerics.mojo::identical_pow` is `exp(p log x)` and returns NaN there by its own contract | **REPLACE.** An ASCENDING SERIAL repeated product at an INTEGER degree, exact for a dyadic base and one pinned rounding sequence otherwise. A non-integer or out-of-range degree RAISES BY NAME. A repeated-squaring ladder is a SECOND order and is refused. DEVIATION 1663 | **CONSTRUCTION, NOTHING RUN.** `KMSAB_POLY_VIA_POW` on `FIX_KM_MIXED`, where it produces NaN across roughly half the matrix rather than a slightly different number |
| 69 | **`kernel_methods/` -- WHICH RBF** (`checks/kernel_matrix.mojo`) | there are TWO upstream RBF spellings and they disagree: cuML's is `exp(-gamma * sqeuclidean)`, unexpanded; cuVS's is `exp(-gamma * (\|x\|^2 + \|y\|^2 - 2 x.y))`, a GEMM plus an epilogue. The expansion cancels catastrophically for nearby rows, so the diagonal of a correlation kernel is NOT exactly 1 | **FOLLOW THE ONE ALREADY PORTED.** cuVS's expanded form through `svm/`'s `kernel_op`, with no clamp at zero, exactly as theirs has none. The consequence for row 67's argument is stated rather than discovered: the unit-diagonal claim is about the MATHEMATICAL diagonal. DEVIATION 1666 | **CONSTRUCTION, NOTHING RUN.** `check_kernel_matrix_vs_oracle` prints the worst \|K_ii - 1\| and the worst gap against a float64 UNEXPANDED reference |
| 70 | **`kernel_methods/` -- THE EIGENVECTOR SIGN, AND WHERE IT IS INERT** (`estimator.mojo::nystroem_fit_host`) | an eigenvector is determined only up to sign, so two implementations of one eigendecomposition legitimately return different bases and every downstream comparison is void | **REUSE THE PINNED ONE.** `decomposition/impl/linalg/detail/pca.mojo::sign_flip_kernel`, RAFT's `signFlipKernel`, called on the device before the ordering. **AND THE FINDING**: the sign CANCELS BITWISE in the symmetric product `Q diag(w) Q^T`, because flipping column `k` negates both factors of term `k` and `(-a)(-b) == ab` exactly. So the pin is load bearing for the RECORDED eigenvector stage and for any caller who builds the ASYMMETRIC embedding, and INERT in the shipped normalization. DEVIATION 1668 | **CONSTRUCTION, NOTHING RUN.** `check_eigen_sign_is_pinned` asserts the convention, asserts that negating the data moves no bit, asserts that `KMSAB_NO_SIGN_FLIP` flips at least one column, AND asserts that it moves NO bit of the normalization or the embedding, which is the finding checked rather than argued |
| 71 | **`kernel_methods/` -- THE EIGENVALUE ORDER IS A SUMMATION ORDER** (`estimator.mojo::_eigen_order_f32`) | a Jacobi does not sort, so the caller does, and "descending by eigenvalue" is not a total order on a spectrum with a repeated eigenvalue. The order is the `k` axis of the normalization product, so two orders are two float32 matrices | **PIN.** Descending by value, ASCENDING INDEX on a tie, by selection sort so the result is a function of the values and the indices and nothing else. It matches `raft::matrix::colReverse`'s intent, `cub::ArgMax`'s tie rule, `np.argmax`'s and cuML's thrust loop's. DEVIATION 1669 | **CONSTRUCTION, NOTHING RUN.** `KMSAB_EIGEN_ORDER_ASCENDING` and `KMSAB_EIGEN_TIE_UNSTABLE`; the second is inert without a repeated eigenvalue, which is why `FIX_KM_ORTHO` plants FOUR-WAY ties |
| 72 | **`kernel_methods/` -- THE POSITION-MAPPED DRAWS** (`checks/random_features.mojo`) | scikit-learn's basis sample is `rnd.permutation(n)[:q]`, a Fisher-Yates over a sequential stream, and its random weights are `normal(size=(d, D))` filled row-major from that stream. Both make a value a function of HOW MANY values were asked for, which is `resample/`'s DEVIATION 1690 in a new estimator | **REPLACE, through the map that already exists.** Row `j`'s basis key and weight `(f, j)` are pure functions of `(seed, ...)` and of nothing else. Prefix stability in `n_components` and in `n_features` follows, and both are gated bit for bit. One Box-Muller pair serves two components so both draws are spent. DEVIATIONS 1671, 1675, 1677, 1679 | **CONSTRUCTION, NOTHING RUN.** `check_random_features_prefix_stability` at `D` = 1, 64, **65** and 256 (the ODD width, because one pair serves two components), `check_nystroem_basis_prefix_stability` at `q` = 2, 4, 12 and `n`, and `KMSAB_RF_STREAM_DRAW`, which is INERT at one width by construction |
| 73 | **`kernel_methods/` -- `log(0)` IN A GAUSSIAN DRAW** (`impl/random/rng_device.mojo::km_guard_unit`) | RAFT's `box_muller_transform` opens with `sqrt(-2 log(val1))` and `next_float`'s range is `[0, 1)`, CLOSED AT ZERO. At `2^-24` probability per pair, a `128 x 1024` weight matrix produces a non-finite weight one fit in 256, and the feature map is then NaN for every row -- a wrong answer with no error | **FIX, NOT PORT** (`PORTING_RULES 0c`, `assume-our-code-is-broken`). `+0.0` becomes `2^-24`, EXACTLY the smallest value the generator itself can return, so the substituted draw is one the generator produces and one point of a `2^24`-point grid moves onto its neighbour. Every other input untouched on every vendor. DEVIATION 1676 | **CONSTRUCTION, NOTHING RUN.** `check_boxmuller_guard` drives it DIRECTLY at a planted `u1 = +0.0` (a sweep cannot reach a `2^-24` event) and requires the unguarded pair non-finite, the guarded pair finite, and the guard inert on 4096 other draws |
| 74 | **`kernel_methods/` -- row 39 in a kernel matrix**: where signed zeros are ERASED and where they SURVIVE | a `-0.0` is invisible to every tolerance comparison, and the two places it can appear here behave OPPOSITELY | signed zeros in the DATA are erased by the kernel matrix, because `identical_gemm`'s fold is seeded `+0.0` and row 39 says a sum is `-0.0` only when every term is; a signed-zero TARGET survives into the dual coefficients, where the forward substitution's `fma(-l, b, t)` carries the sign. `mu = +0.0` in the Box-Muller transform turns a `-0.0` weight into `+0.0`, deterministically, and that is THEIR line | **CONSTRUCTION, NOTHING RUN.** `check_signed_zero_reach` ASSERTS the erasure (a version that preserved them would be the surprising one), RAISES if no negative zero reaches the dual coefficients, and compares device against replay BY SIGN BIT |

Row numbers 67-74 are proposed, not claimed: `IDENTITY_PATHS.md` is not this
lane's file and the orchestrator assigns them. `cholesky/README.md` already
proposes 60-66.

## The deviations

| # | what |
|---|---|
| **1660** | **`alpha` IS THE RIDGE and the Cholesky profile's jitter is `+0.0`. The two are never both applied.** The headline; `impl/kernel_ridge/kernel_ridge.mojo`'s first banner carries the argument in full |
| 1661 | cuML widens the kernel matrix to float64 before `posv`; there is no device float64 on this column, so the only conditioning available is `alpha`. Measured against a float64 host solve of the SAME float32 matrix, so the number isolates the solver |
| **1662** | cuML's least-squares fallback on a singular kernel matrix is NOT ported; a failed factorization RAISES BY NAME with `alpha` as the closure |
| **1663** | **the polynomial power is an ASCENDING REPEATED PRODUCT at an integer degree**, because `identical_pow` is `exp(p log x)` and a polynomial kernel's base is routinely negative. A non-integer or out-of-range degree refuses |
| 1664 | the polynomial and tanh epilogues complete a mirror `svm/impl/distance/kernel_matrices.mojo` leaves partial. Two partial mirrors of one upstream file, in two lanes, and neither is complete alone |
| 1665 | the laplacian kernel has no cuVS epilogue; it is `identical_exp(-gamma * L1)` over `kde/`'s ported `pairwise_distance` at `DIST_L1` |
| **1666** | **which RBF**: cuVS's EXPANDED form, not cuML's unexpanded one, with no clamp at zero. The consequence -- the correlation diagonal is not exactly 1 -- is stated rather than discovered later |
| 1667 | Nystroem's eigenstep substitutes `decomposition/`'s Jacobi for sklearn's `xp.linalg.svd`; for a symmetric PSD basis kernel they coincide, and cuSOLVER's `gesvd` is closed |
| **1668** | **the eigenvector sign convention is `sign_flip_kernel`'s, REUSED, plus the proof that the sign CANCELS BITWISE in the symmetric normalization** -- so the pin is load bearing for the recorded stage and inert in the shipped product, and that asymmetry is gated |
| **1669** | the eigenvalue order is DESCENDING with an ascending-index tie break, and it is a SUMMATION ORDER rather than a presentation choice |
| 1670 | sklearn's `clip(S, 1e-12, None)` is copied BY VALUE; a negative eigenvalue is clipped rather than refused, exactly as theirs |
| **1671** | **the basis row sample is POSITION-MAPPED**: a rank prefix over per-row Philox keys, not a Fisher-Yates over a stream. Prefix stability in `n_components` follows |
| 1672 | the O(n^2) rank pass is bounded at `KM_MAX_BASIS_POOL` and refuses above it with the closure |
| 1673 | `n_components > n_samples` RAISES where scikit-learn warns and clamps (their own comment on that branch is "XXX should we just bail?") |
| **1674** | the embedding is `embedded @ normalization.T`, and the transpose is NOT free: `normalization` is mathematically symmetric and NOT bitwise symmetric |
| 1675 | the normals are RAFT's Box-Muller over a counter-based generator, not numpy's ziggurat, so no bit comparison with scikit-learn is possible and none is attempted |
| **1676** | RAFT's Box-Muller does not guard `log(0)`; we do, by substituting `2^-24`, the smallest value `next_float` itself returns |
| 1677 | one Box-Muller PAIR serves components `2p` and `2p+1`, so both draws are spent and prefix stability survives an odd `n_components` |
| 1678 | `sqrt(2 gamma)`, `2 pi` and `sqrt(2 / D)` are host constants computed once and passed as kernel arguments |
| **1679** | this lane IMPORTS `resample/checks/index_map.mojo` and shares its Philox KIND-BYTE NAMESPACE. A collision would be a correlation, not an error, so it is gated |
| 1680 | kernel ridge's prediction is `identical_gemm_into` at `OP_NN`; `linalg.matmul` is refused, and the `k` axis there is `n_samples`, the longest reduction in the estimator |
| 1681 | MULTI-TARGET kernel ridge falls out FREE through `cho_solve`'s `nrhs` and is SHIPPED rather than deferred; cuML's per-target `alpha` arm is not ported |
| 1682 | `sample_weight` is not ported |
| 1683 | `kernel='precomputed'` is refused by name: it is a shape contract, not a kernel, and there is no oracle for it |
| 1684 | a training Gram uploads `X` TWICE, because Mojo refuses one buffer as two mutable kernel arguments. A special diagonal path would be a second code path reached only in the square case, which is rule 8's failure |
| 1685 | the ridge is added by this lane's own kernel, not `cholesky/`'s `add_jitter`, because `add_jitter` refuses any value outside its profile's two and `alpha` is a caller's free parameter. Same three tokens, deliberately |
| 1686 | non-finite input, a negative `alpha`, a non-positive `gamma`, an out-of-range `degree`, `n_components` out of range and an unsupported kernel each RAISE BY NAME on the host before any launch |
| 1687 | the sabotage arms are runtime-selectable through a `sabotage` argument in a separate file never reached at `KMSAB_NONE`, plus a `KMSAB_COPY_ONLY` control that proves the copies faithful |
| 1688 | the eigenvalue ordering and the basis rank pass run on the HOST, O(q^2) and O(n^2) in the SAMPLE count, never O(n_samples * n_features), exactly as `eig_and_truncate`'s ordering does |
| 1689 | the eigenvector column scaling is a DIVIDE by `sqrt(s)`, never a multiply by a precomputed reciprocal. sklearn writes `U / sqrt(S)`, and `cholesky/`'s DEVIATION 1643 refuses the same shape for the same reason: a reciprocal-then-multiply is two roundings where a divide is one |

**All thirty numbers 1660-1689 are spent.** None is reserved.

## WHAT THE ORCHESTRATOR MUST WIRE

Nothing outside `kernel_methods/` was edited by this lane. These lines are
wanted in `pixi.toml`, in the file's existing format, beside the other
classical lanes' tasks (they sit naturally after `check-cholesky` /
`cholesky-main`):

    check-kernel-methods = "mojo run -I . kernel_methods/checks/km_check.mojo"
    kernel-methods-main = "mojo run -I . kernel_methods/kernel_methods_main.mojo"

The IDENTICAL pass is the injector, exactly as for every other gate:
`tools/with_identical_mode.sh pixi run check-kernel-methods`. **No
`*-identity` task is wanted.**

Also owed by the orchestrator, and none of it is this lane's to do:

1. **`IDENTITY_PATHS.md` rows 67-74**, from the identity table above.
   `cholesky/README.md` proposes 60-66, so the orchestrator has two lanes'
   rows to place and should renumber both together.
2. **`PORTING.md` / `ROADMAP.md` / `SUPPORT_MATRIX.md`**: `kernel_methods/`
   is a new section and appears in none of them.
3. **`UNWIRED.md`**: `kernel_methods/estimator.mojo` has no caller in
   `bindings/_mojolearn_estimators.mojo` or `python/mojolearn/`.
4. **The Python surface**, when a binding is wanted:
   `kernel_ridge_fit_host`, `kernel_ridge_predict_host`,
   `nystroem_fit_host`, `nystroem_transform_host`, `rbf_sampler_fit_host`,
   `rbf_sampler_transform_host`. Refuse `float64` inputs, `kernel='precomputed'`,
   `sample_weight` and `gamma='scale'` in Python, by name, so a user meets
   the refusal at the surface they typed at.
5. **`CARD_GAPS.md`**: this lane emits a card and it has never been emitted.
6. **A CHANGE THIS LANE BELIEVES `core/` NEEDS AND DID NOT MAKE.**
   `impl/random/rng_device.mojo` ports RAFT's `next_float` and
   `box_muller_transform`, which belong in `core/philox.mojo` beside the rest
   of RAFT's generator. Moving them there would also delete a duplication:
   `resample/checks/index_map.mojo::draw_unit_float` inlines `next_float`'s
   two lines for its own use. **Three copies of two lines is the current
   state and one file is the fix**; `core/` is not this lane's directory.

## SABOTAGES TO PERFORM

All thirteen are selected at RUN TIME through the `sabotage` argument
(`kernel_methods/checks/km_sabotage.mojo`), copying
`cholesky/checks/chol_sabotage.mojo`'s construction. **No source edit and
no rebuild is required for any of them**; `check_km_sabotages` drives them in
one run and prints a line per arm. These arms were driven on the Apple M4 on
2026-08-25 in both modes (Status). The classifications below are what each
arm is expected to do rather than a transcript of what it did.

| check it targets | sabotage | exactly what it corrupts | what must move |
|---|---|---|---|
| `check_kernel_matrix_vs_oracle`, `check_rbf_sampler_vs_oracle` | `KMSAB_STD_TRANSCENDENTAL` | the RBF and laplacian `exp`, the sigmoid `tanh` and the feature map's `cos` go through `std.math`. The largest transcendental surface in the lane: one call per cell of an `n x n` matrix | the kernel matrix's bits under IDENTICAL. MUST FAIL |
| `check_kernel_matrix_vs_oracle` | `KMSAB_POLY_VIA_POW` | the polynomial power becomes `identical_pow` = `exp(p log x)` | on `FIX_KM_MIXED` it returns NaN across roughly half the matrix rather than a different number, which is DEVIATION 1663 made visible. MUST FAIL |
| `check_kernel_ridge_vs_oracle` | `KMSAB_RIDGE_RELATIVE` | `K_ii += alpha * K_ii` instead of `K_ii += alpha` | the dual coefficients. **MUST FAIL, SWEPT** -- nearly inert on RBF and laplacian, which have a near-unit diagonal, and this is exactly the trap `cholesky/README.md` recorded |
| `check_kernel_ridge_vs_oracle` | `KMSAB_RIDGE_PLUS_JITTER` | the Cholesky profile's `2^-20` ridge applied IN ADDITION to `alpha`. DEVIATION 1660's forbidden state | the dual coefficients. MUST FAIL |
| `check_eigen_sign_is_pinned` | `KMSAB_NO_SIGN_FLIP` | `sign_flip_kernel` is not launched | **TWO-SIDED.** At least one eigenvector column's sign MUST move, AND the normalization and the embedding MUST NOT. Both asserted; the second is DEVIATION 1668's proof |
| `check_nystroem_full_equals_exact_kernel` | `KMSAB_EIGEN_ORDER_ASCENDING` | the eigenvalue order is ascending: same multiset, different `k` axis | the normalization's bits. MUST FAIL |
| `check_nystroem_full_equals_exact_kernel` | `KMSAB_EIGEN_TIE_UNSTABLE` | the tie break keeps the HIGHER index, so the order stops being total | **MUST FAIL, SWEPT** -- inert on any fixture without a repeated eigenvalue, which is why `FIX_KM_ORTHO` plants four-way ties |
| `check_nystroem_full_equals_exact_kernel` | `KMSAB_NO_EIGEN_CLIP` | sklearn's `clip(S, 1e-12)` dropped | **REPORT, and if it is inert on every fixture the check SAYS SO and calls it a coverage gap owed a genuinely singular basis.** An arm that cannot fire is not silently counted as passing |
| `check_launch_invariance`, `check_nystroem_basis_prefix_stability` | `KMSAB_BASIS_FROM_LAUNCH` | the basis rows become a LAUNCH-STRIDED slice instead of the rank prefix | the basis indices and everything below them. MUST FAIL |
| `check_random_features_prefix_stability` | `KMSAB_RF_STREAM_DRAW` | `W[f][j]` is drawn at the sequential position `f * n_components + j`, which is what numpy does | **MUST FAIL, AT TWO WIDTHS.** Inert at one `n_components` by construction, so it is driven at 48 and 96 and the first 48 components must DIFFER |
| `check_boxmuller_guard` | `KMSAB_NO_BOXMULLER_GUARD` | DEVIATION 1676's `log(0)` guard dropped | **REPORT** in the fixture sweep -- a `2^-24` event will not occur -- and driven DIRECTLY at a planted `u1 = +0.0`, where the unguarded pair must be non-finite |
| `check_nystroem_full_equals_exact_kernel` | `KMSAB_EMBED_OP_NN` | the embedding uses `@ normalization` instead of `@ normalization.T` | the embedding's bits. MUST FAIL -- and it would be INERT if `normalization` were bitwise symmetric, which is precisely why DEVIATION 1674 exists |
| `check_rbf_sampler_vs_oracle` | `KMSAB_RF_SCALE_IN_KERNEL` | `sqrt(2/D)` recomputed per thread instead of once on the host | **REPORT, EXPECTED INERT.** Driven anyway: an inert result proves the constant is not a fold and not launch shaped, and an arm that ever MOVES is a finding about that column's division |
| everything | `KMSAB_COPY_ONLY` | **NOT AN ARM.** Routes the drivers through the sabotage file's COPIES with no arm engaged | nothing. `check_km_sabotage_copies_agree` requires bit equality, which is what makes a failing arm attributable to the arm and not to a stale copy of another lane's kernel |

**Two arms the brief asked for have no object here and it is worth saying
why.** "Swap the two fold levels" and "change the block reduction's width":
**there is no fold across threads anywhere in `kernel_methods/`.** Every
kernel this lane owns is one thread per output cell; the only cross-thread
combinations below these entry points live inside `identical_gemm_into`,
inside the Jacobi's `pinned_block_sum` and inside `potrf_lower`'s trailing
update, and each carries its own lane's sabotages. That absence is the reason
launch invariance is a property of SHAPE here rather than a property a check
happens to observe, and `check_launch_invariance`'s docstring says so.

## What the checks are expected to establish

| check | what it would establish |
|---|---|
| `check_km_kind_bytes_are_disjoint` | this lane's three Philox kind bytes are pairwise distinct and disjoint from `resample/`'s three, so the shared key namespace carries no correlated streams |
| `check_km_refusals` | thirteen refusals fire BY NAME **and** every paired legal value is ACCEPTED, so the validator is not simply always raising |
| `check_ortho_fixture_is_exact` | `FIX_KM_ORTHO`'s three claims -- disjoint supports, a diagonal Gram with power-of-four entries, `y = X w` exactly -- re-derived on the host BEFORE anything is asserted from them. A fixture whose properties are assumed is a gate with no floor |
| `check_nystroem_full_equals_exact_kernel` | **THE HEADLINE, three arms.** (a) At `q == n` on the exact fixture, `Phi Phi^T` equals the exact kernel matrix BIT FOR BIT in BOTH modes. (b) A tolerance REPORT on the inexact fixtures, plus per-stage gaps against the float64 Nystroem -- **not asserted, because the eigendecomposition is an iterative float32 algorithm and a bitwise demand there would be a gate that cannot pass.** (c) DETERMINISM across two launch geometries on every fixture, asserted in both modes |
| `check_kernel_ridge_planted_linear` | the dual coefficients, the training predictions and the PRIMAL weights all equal their hand-derived values bit for bit at `alpha = 0`, and target 0 of a 3-target fit equals the 1-target fit |
| `check_kernel_matrix_vs_oracle` | five kernels x five fixtures, square AND rectangular (`n_query != n_samples` on purpose), equal the float32 second spelling at every cell under IDENTICAL; plus the float64 gap and the worst `\|K_ii - 1\|` as REPORTS |
| `check_kernel_ridge_vs_oracle` | the dual coefficients equal `cholesky/`'s own float32 replay at every cell, the two `info` values AGREE about solvability, and the float64 gap and the prediction gap are REPORTED |
| `check_eigen_sign_is_pinned` | the convention holds on every column of every fixture; negating `X` moves no bit; the sabotage flips at least one sign; and it moves NO bit of the normalization or the embedding. **Its docstring also states what it cannot do**: a deterministic solver cannot exhibit a spontaneous flip, so a cross-implementation demonstration is owed |
| `check_nystroem_basis_prefix_stability` | the basis at `q` = 2, 4, 12 is the prefix of the basis at `q = n`, and the `q == n` sample is a bijection |
| `check_random_features_prefix_stability` | component `j` is identical at `D` = 1, 64, **65** and 256 and row `f` at `d` = 8 and 16; and the DEVICE weights equal the host map bit for bit |
| `check_boxmuller_guard` | the unguarded RAFT transform at `u1 = +0.0` is non-finite, the guard substitutes `2^-24` for BOTH zeros, the guarded transform is finite, and the guard is inert on 4096 other draws |
| `check_rbf_sampler_vs_oracle` | the feature map equals the float32 replay of OUR draws at every cell. **Never compared with scikit-learn's numbers**, and DEVIATION 1675 says why |
| `check_rbf_sampler_approximates_kernel` | **REPORT.** The Monte Carlo error at `D` = 64, 256, 1024, printed and not asserted, because it is a random variable with an `O(1/sqrt(D))` scale. The only assertion is that every estimate is finite |
| `check_launch_invariance` | nothing moves across two threads-per-block choices at every kernel of all three estimators, and one query row transformed ALONE equals the same row inside a batch -- the arm that would catch a driver sizing a grid or a workspace from the wrong dimension of a RECTANGULAR cross-kernel |
| `check_signed_zero_reach` | planted negative zeros are ERASED by the kernel matrix (asserted, because the `+0.0`-seeded fold requires it), negative zeros DO reach the dual coefficients (the check RAISES if none do), and device and replay agree BY SIGN BIT |
| `check_km_sabotage_copies_agree` | the sabotage file's copies reproduce the production kernels bit for bit with no arm engaged, INCLUDING the copy of `svm/`'s RBF epilogue that this lane cannot edit and cannot otherwise notice drifting |
| `check_card_is_emitted` | the stage list, in order, with a sane record count, and two runs of one fixture producing an IDENTICAL card |
| `check_km_sabotages` | all thirteen arms driven at run time, each classified MUST FAIL or REPORT in advance, each MUST FAIL arm SWEEPING the fixtures with the moving fixture and the inert count printed |

## WHAT THIS WILL COST

**READ THIS BEFORE PUTTING A NUMBER NEXT TO THIS LANE.**

`bench/results/SPEED_LANE_2026-08-25.md` section 3.4 measured this
repository's identical FP32 GEMM at **2% of FP32 peak on an Apple M4 --
0.078 TF/s against MPS's 2.33 TF/s, a factor of about 30** -- and at 3 to 5%
of peak on an H100 and an MI325X. That report's own conclusion is that "the
gap is structural to the kernel rather than an accident of one backend".

**Kernel ridge and Nystroem are GEMM bound and Cholesky bound.** Kernel ridge
is one `n x n x d` product, one `O(n^3/3)` factorization, one `O(n^2 t)`
solve and one `q x t x n` product. Nystroem is a `q x q x d` product, an
`O(q^3)` Jacobi, a `q x q x q` product and an `n x q x q` product. Every one
of those rides the same 2%-of-peak contraction, and the Cholesky rides it
too (its trailing update IS `identical_gemm_into`) on top of a `trsm` that
`cholesky/README.md` records as `O(n^2 nrhs)` global loads with no reuse.

**So the plain statement, and it is a statement rather than a hedge: under
IDENTICAL, kernel ridge and Nystroem are EXPECTED TO BE SLOWER THAN
scikit-learn ON A CPU until the GEMM gap closes.** scikit-learn's kernel
ridge calls LAPACK `posv` through a tuned BLAS at a large fraction of a
modern CPU's peak, and a GPU running its contractions at 2% of peak does not
have thirty times the headroom to spend. The only number taken so far is
the FAST NVIDIA row above, which compares kernel ridge against cuML's GPU
arm and not against scikit-learn on a CPU under IDENTICAL, so it does not
test this prediction; what is predicted is the SIGN.

Three things follow and all three are policy rather than opinion:

1. **Any speed claim about this lane lives in FAST, not IDENTICAL**, and
   `mojotrees-switches-must-flip` applies to it the same way it applies
   everywhere: a measured bit-identical win flips the default in the same
   session, and a decline carries a priced cost.
2. **IDENTICAL is a REPRODUCIBILITY MODE WITH A PRICED COST**, and the
   standing rule `IDENTITY IS NOT FREE` is the frame: conforming costs on
   every vendor, and the cost here is a contraction running at 2 to 5% of
   peak on all three columns. Saying "identity is free on Apple" is banned
   and so is implying it by silence.
3. **The fix is not in this directory.** Closing the gap means the gemm
   lane's kernel, a blocked `trsm`, a staged Cholesky panel and a `syrk`
   output shape -- all four are in `cholesky/README.md`'s and the gemm lane's
   WHAT IS OWED. Nothing this lane can do to its own source changes the
   number materially.

**`RBFSampler` IS THE ONE ARM WHOSE ARITHMETIC INTENSITY MIGHT SURVIVE, AND
THAT IS A HYPOTHESIS RATHER THAN A MEASUREMENT.** Its `fit` touches no data
at all: it is `n_features x n_components` independent Box-Muller draws, each
one twenty Philox rounds of register-resident integer arithmetic plus four
transcendentals, writing one float. Nothing is read, nothing is folded,
nothing is shared. Its `transform` is a single `n x D x d` GEMM plus a
one-thread-per-cell epilogue, so it inherits the 2% number for the projection
and nothing else. The reason the draws MIGHT be competitive is that
`resample/checks/index_map.mojo`'s DEVIATION 1690 already names the price
that makes them compute bound: a fresh per-position generator costs TWO
Philox block evaluations for one pair where a sequential stream amortises one
over four draws, which is roughly 8x the RNG arithmetic and ZERO extra memory
traffic. **That is exactly the trade that turns a bandwidth-bound kernel into
a compute-bound one, and it is why this arm is worth measuring first.**

**THE HYPOTHESIS HAS NOT BEEN TESTED.** The one number RBFSampler has is
0.295 ms at `3x5q8` on an NVIDIA H100 under FAST, 2026-08-26, with no
opponent on that box
(`bench/results/fast_speed/2026-08-26_040100-nvidia-classical.md`), so
nothing has compared the pinned per-position generator against a sequential
stream. The hypothesis is written down so that it can be falsified rather
than assumed, and the honest prior is that a feature map which spends 8x the
RNG arithmetic to buy reproducibility may well be slower than numpy's
ziggurat too.

## WHAT IS OWED

1. **THE IDENTICAL-MODE LEG ON A SECOND VENDOR.** This lane has been built
   and executed. It is built and gated on one Apple M4 in both modes
   (Status, 2026-08-25), and an NVIDIA H100 compiled and ran all three
   estimators under FAST on 2026-08-26, kernel ridge 6.71x faster than
   cuML's GPU arm at `RBF.16x5`
   (`bench/results/fast_speed/2026-08-26_040100-nvidia-classical.md`). A
   FAST leg is a speed leg and buys no identity; there is still no
   IDENTICAL-mode card from any vendor except the M4, which is item 2.
2. **The second and third vendor legs.** An NVIDIA H100 and an AMD MI325X run
   of `kernel_methods_main.mojo` under `tools/with_identical_mode.sh`, and
   `tools/identity_trace_diff.py` over the three cards. Until then there is
   no cross-vendor claim of any kind. Two arms are expected to behave
   differently there and are recorded that way rather than as passes:
   `KMSAB_STD_TRANSCENDENTAL` (NVIDIA's and AMD's `exp`, `tanh` and `cos` are
   not Cephes) and `KMSAB_RF_SCALE_IN_KERNEL` (a column whose device division
   differs from its host one).
3. **A FIXTURE WHOSE NYSTROEM BASIS KERNEL IS ACTUALLY SINGULAR AT THE
   SAMPLED ROWS.** `FIX_KM_DUP` has duplicate ROWS, but the basis sample may
   or may not draw both members of a duplicate pair at a given `q` and seed,
   so `KMSAB_NO_EIGEN_CLIP` may be inert on every fixture. The check SAYS SO
   when that happens and calls it a coverage gap rather than a pass. Closing
   it needs a fixture that plants a duplicate INSIDE the first `q` ranks,
   which means planting the basis keys rather than the data.
4. **A CROSS-IMPLEMENTATION DEMONSTRATION OF THE SIGN CONVENTION.** A
   deterministic solver run twice on identical bits cannot exhibit a
   spontaneous sign flip, so `check_eigen_sign_is_pinned` can only show that
   the convention is reached, that removing it moves a recorded stage, and
   that it moves nothing else. Showing it PREVENTING a divergence needs a
   second eigensolver -- the float64 host Jacobi at a different sweep budget
   is the cheapest candidate.
5. **A NUMBER UNDER IDENTICAL, AND A THROUGHPUT NUMBER OF ANY KIND.** The
   FAST NVIDIA row in Status is the whole of this lane's timing evidence and
   it is a fixed-cost row at 16 rows. Nothing measures what IDENTICAL costs
   here, and nothing measures a shape large enough to be a throughput
   statement. See WHAT THIS WILL COST for what is expected and why.
6. **`sample_weight`, the per-target `alpha` arm, `gamma='scale'`, and the
   three remaining pairwise kernels** (`cosine`, `chi2`, `additive_chi2`).
   `NOT_IMPLEMENTED.tsv` carries a row for each with what it would take.
7. **A device-resident hyperparameter sweep.** `kernel_ridge_fit_host` forms
   the kernel matrix once per `alpha`; a cross-validation over twenty alphas
   forms it twenty times and need form it once. The device-level entry points
   exist and are documented for exactly this, and no driver here uses them
   that way.
8. **Larger `n`.** The largest fixture is 16 rows and the largest kernel
   matrix any check forms is 16 x 16. Nothing here says what happens at
   `n = 4096`, where the kernel matrix is 64 MB, the Cholesky walks 128
   panels with a host round trip each, and `KM_MAX_BASIS_POOL` refuses the
   Nystroem basis pass outright.
