# cholesky: cross-vendor bit-identical dense FP32 Cholesky, and the solves on it

Opened 2026-08-25. The missing primitive that gates Gaussian processes,
kernel ridge regression and Gaussian mixture models, all of which are
separate lanes and all of which will call this one. **DEVIATIONS 1630-1646
are this lane's**; 1647-1659 are reserved and unspent.

**The profile is `mojolearn.identical.cholesky.fp32.v1`.** Changing
`CHOL_NB_PINNED`, the jitter, the panel column order, the inner summation
order or the trailing update's fold creates a v2; it does not amend v1. Same
discipline as `mojolearn.identical.gemm.fp32.v1`.

## Status

**BUILT AND GATED ON ONE APPLE M4, BOTH MODES, 2026-08-25. NO SECOND
VENDOR HAS RUN THIS UNDER IDENTICAL, so there is no identity card outside
the M4.** An NVIDIA H100 compiled and ran this lane under FAST on
2026-08-26; the row is in
`bench/results/fast_speed/2026-08-26_040100-nvidia-classical.md` and it
settles speed, not identity.

`pixi run check-cholesky` is green in FAST and green under
`tools/with_identical_mode.sh`, twelve checks in each mode, and all ten
sabotage arms were driven at run time through the `sabotage` argument with
no source edited. Under IDENTICAL:

    check_potrf_vs_oracle OK: 6 of 6 fixtures bit-equal to the host oracle
      at every cell; the 48x48 planted factor equals its PLANT bit for bit;
      worst |device - float64 reference| 2.68e-07
    check_launch_invariance OK: 6 fixtures byte-identical across panel_tpb
      128/32, elem_tpb 256/64, pad 0/37, two poisons, run twice, and alone
      versus the leading block of a block-diagonal matrix of twice the size
    check_pivot_failure_is_identical OK: the singular fixture stops at
      info=38 on device and oracle, partial factor equal at all 2304 cells
    check_block_size_is_pinned OK: nb_hint=16 refused by name; the run
      reports nb=32 and the card carries it

Under FAST two of those become REPORTS and both move, which is the evidence
the pins are load bearing. `check_potrf_vs_oracle` reports 2 of 6 fixtures
diverging, and the mechanism is named rather than waved at: `ftz` compiles
away under FAST, so the host oracle keeps a subnormal at cell [3, 1]
(0x00000200) that Metal flushes to zero. That is the documented denormal
policy and it is exactly the divergence the IDENTICAL pin removes.

**One timing exists and it is a FAST one.** On an NVIDIA H100 on
2026-08-26, fixture `RBF.64x64r4`, our median was 0.316 ms against
torch-gpu's 0.215 ms, so we are 1.47x SLOWER
(`bench/results/fast_speed/2026-08-26_040100-nvidia-classical.md`; the
vendor arm is `torch.linalg.cholesky` plus `torch.cholesky_solve`, which is
cuSOLVER `potrf` and `potrs`, with the ridge, the logdet and the solve
inside the clock on both sides). That artifact's own header records that
this fixture is small enough that both arms are dominated by launch and
dispatch cost, so the ratio is a fixed-cost figure and not a kernel
verdict. No IDENTICAL-mode timing exists on any vendor.

### Two corrections the gate forced, both worth keeping

**The RBF fixture cannot test the jitter policy.** `CHOL_SAB_JITTER_RELATIVE`
replaces the absolute ridge `A_ii + jitter` with `A_ii + jitter * A_ii`, and
on an RBF Gram matrix those are THE SAME NUMBER because `A_ii = exp(-0) =
1.0` exactly. `rbf_gram`'s own docstring says so. The arm was reached and
provably inert, and the gate refused to pass rather than let it look
verified. This generalizes and matters downstream: **every correlation-shaped
kernel matrix has a unit diagonal, so on the shapes the Gaussian-process and
kernel-ridge lanes will actually feed this factorization, the two jitter
policies COINCIDE.** DEVIATION 1637's choice is load bearing only off that
diagonal.

**`CHOL_SAB_LOGDET_PAIRWISE` was inert on RBF too**, for an unrelated reason,
and the two together are why `check_cholesky_sabotages` now SWEEPS the
fixtures instead of naming one. An arm inert on the fixture its author
happened to pick is indistinguishable from an arm that is unreached, and this
lane produced one of each. The requirement is now that each arm move on at
least one fixture that factors, and the print names which fixture and how
many earlier ones were inert to it.

Two things follow and both are deliberate. First, every sentence in this
directory that could be mistaken for a measurement is marked as expected,
predicted or owed. Second, the two places where a prediction could be wrong
in an interesting way are named rather than smoothed over: whether
`CHOL_SAB_NO_FTZ_PIVOT` is inert on Apple (predicted yes, because Metal
flushes in hardware) and whether `FIX_ILL` fails without a ridge (not
predicted at all -- the check asserts device-equals-oracle either way and
prints the `info` it got).

    pixi run check-cholesky                                                    # FAST
    tools/with_build_lock.sh     pixi run mojo run -I . cholesky/checks/cholesky_check.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . cholesky/checks/cholesky_check.mojo

    MOJOLEARN_IDENTITY_TRACE=/tmp/mac.chol.identical.card \
        tools/with_identical_mode.sh pixi run mojo run -I . cholesky/cholesky_main.mojo

## The upstream pin

| upstream | checkout | commit | what it contributed |
|---|---|---|---|
| RAFT | `/Users/andrewhendel/CascadeProjects/upstream/raft-v26.08.00` | `ebf9268` | `linalg/cholesky_r1_update.cuh` and its `detail/`; `matrix/detail/matrix.cuh`'s four triangular and diagonal helpers. **The only portable Cholesky source in the three checkouts** |
| cuML | `/Users/andrewhendel/CascadeProjects/upstream/cuml-v26.08.00` | `265b9da` | READ; contributes NO code. cuML implements no Cholesky. Its one consumer is `cpp/src/solver/lars_impl.cuh:315-320` |
| cuVS | `/Users/andrewhendel/CascadeProjects/upstream/cuvs-v26.08.00` | `6ba2ce2` | READ; contributes NO code. Its only from-scratch factorization is `cpp/src/neighbors/scann/detail/scann_avq.cuh:179-200`, which is `cusolverDnpotrf` followed by `cusolverDnpotrs` |

**The upstream situation, recorded honestly.** cuML and cuVS do not implement
Cholesky at all. Every factorization in either tree is a cuSOLVER call, and
cuSOLVER is a closed vendor library with no source. There is nothing to
transliterate, so `cholesky/checks/potrf.mojo` and `trsm.mojo` are NOT
ports and their headers say so instead of citing line numbers they cannot
have. `VENDOR_LIBS.md`'s surviving exception -- call the platform equivalent
where the dispatch reaches a CLOSED library, because there is nothing to port
-- does not resolve it either, because MAX ships no `potrf`. So the move is
the third one `IDENTITY_PATHS.md`'s opening rule allows applied to a gap
rather than to a pathway: write the factorization with every numeric decision
named. `DERIVATION_MAP.tsv`'s header names the cuSOLVER or cuBLAS call each entry
point replaces, which is what `VENDOR_LIBS.md` requires of a substitution.

**One correction to the brief that opened this lane, made against the
checkout.** The rank-one update was described as "the rank-one Cholesky
update cuML's SVM uses". It is not. `grep -rn cholesky
upstream/cuml-v26.08.00/cpp/src/svm/` returns nothing at this pin; cuML's SVM
solver is SMO and touches no factorization. The consumer is
`ML::Solver::Lars::updateCholesky`, growing the Gram matrix of an active set
one column per LARS step. PORTING_RULES rule 1: the file wins, and the false
sentence is deleted rather than annotated.

## WHAT THIS LANE REUSES RATHER THAN REWRITES

Nothing in the list below was re-implemented here, and each was checked for
before a line was written.

| what | the file it lives in | why not a second copy |
|---|---|---|
| **The matrix product** in the trailing update `A22 -= L21 L21^T` | `gemm/checks/gemm_identical.mojo::identical_gemm_into`, sized by `identical_gemm_workspace_max_floats`, at `OP_NT` from `gemm/checks/gemm_oracle.mojo` | profile `mojolearn.identical.gemm.fp32.v1` is gated at 62 shapes across eight execution plans with six sabotages. A hand-written contraction here would be a second contraction in one repository and would have to earn that certificate again |
| **The normative answer for that product**, in the oracle | `gemm/checks/gemm_oracle.mojo::gemm_oracle` | `gemm_identical.mojo::contract_partition` is explicit that a second spelling of the leaf rule is a second thing that can be wrong, and records that the shape table already shipped one such re-spelling and got it wrong |
| **Transcendentals and the arithmetic pins**: `identical_sqrt`, `identical_log`, `identical_div`, `identical_mul`, `identical_mul_add`, `ftz` | `checks/numerics.mojo` | rows 9, 10, 12 and 49 of `IDENTITY_PATHS.md`. Nothing in this directory calls `std.math` on a numeric path; the only `std.math` here is `sqrt` and `log` inside the FLOAT64 host reference and inside the `CHOL_SAB_STD_SQRT` sabotage arm, both by design |
| **Stage hashing and the differ** | `core/identity_trace.mojo`: `IdentityTrace`, `record_device`, `record_host`, `record_list_i32`, `fnv1a64_bytes`, `first_divergence`, `read_trace_lines` | one hash function per repository. The tag-uniqueness invariant is what forced the three-digit panel numbering |
| **The pinned block fold**, considered and NOT used | `core/pinned_reduce.mojo` | named here because a reviewer will look for it. No kernel in this lane folds across threads at all -- the panel, both solves and the log-determinant each keep every sum inside one thread -- so there is no fold shape to pin and importing one would suggest there is |
| **Fixed-point accumulation**, considered and NOT used | `checks/fixed_point.mojo` | same. It exists to REPLACE float atomics, and this lane has none. There is no float atomic, no `Atomic.fetch_add`, no warp shuffle, ballot or vote, and no `block.sum` anywhere in `cholesky/` |
| **Top-k, distance kernels, RNG** | `neighbors/`, `cluster/`, `gbdt/gpu_util/kernel/random_gen.mojo` | a Cholesky needs none of them. The only randomness in the lane is the fixture hash |
| **The float32 GEMM used to CHECK the reconstruction** | `cholesky/checks/cholesky_fixture.mojo::gram_from_lower`, which is host code the fixtures already needed | rather than a second device product for the checking path |

**One duplication is taken, and it is named rather than hidden.**
`chol_mix64` in `cholesky_fixture.mojo` is the same three lines as
`kde/checks/kde_fixture.mojo:18`, `holtwinters/checks/hw_fixture.mojo:30`
and `isolation_forest/checks/if_fixture.mojo:32`, with the same splitmix64
constants. That is the established per-lane convention in this tree; the
alternative is a cross-lane import of another lane's fixture file, which
`core/pinned_reduce.mojo` argues against at length for hot files. Four copies
of one hash is a debt and this sentence is the record of it.

## How this relates to `gbdt/lapack/linear_system.mojo`

That file is this repository's existing Cholesky: a HOST, FLOAT64, UNBLOCKED
`dposv` -- factor plus two substitutions -- transliterated from CatBoost's
`SolveLinearSystemCholesky`, solving a `(numClasses - 1) x (numClasses - 1)`
system once per LEAF for the MultiClass Newton step. Six by six for a
seven-class problem. It is not edited by this lane and nothing here imports
it.

**Should it eventually call this lane? No, and the reason is a measurement
its own DEVIATION 74 already records.** It runs on the host, once per leaf,
on a matrix small enough to sit in registers; routing it through
`potrf_lower` would be a device round trip per leaf to solve a 6x6, which is
the same shape of mistake `PORTING_RULES` rule 2's corollary describes ("nine
drains per level became two by DELETING our inventions"). It is also FLOAT64,
where this lane is FLOAT32 by DEVIATION 1635, so the substitution would
change its answers.

**What they should share is the SPELLING of the pivot decision, and today
they do not.** `linear_system.mojo` writes `if s <= 0.0`, this lane writes
`if not (s > 0.0)`. The two differ on exactly one input, a NaN `s`, which
their float64 host path can reach through a saturated multinomial
probability at `l2_leaf_reg = 0` -- their spelling PASSES a NaN pivot and
proceeds to `sqrt(NaN)`. That is an observation about another lane's file
made while reading it and it is recorded here rather than acted on, because
that file belongs to the GBDT lane and its DEVIATION 74 is explicit that
CatBoost's failure behavior is copied on purpose. **It is the GBDT lane's
call, and it is named so that the call can be made.**

## The identity table (row text for `IDENTITY_PATHS.md`)

| n | path | what is vendor-dependent in the ordinary spelling | what we did | status |
|---|---|---|---|---|
| 60 | **`cholesky/` -- the panel BLOCK SIZE `NB`** (`cholesky/checks/potrf.mojo::chol_nb_for`) | a blocked factorization's block size looks like a tuning knob and IS a summation order: it partitions the `k` axis of the trailing update, so two values of it bracket the same sum differently and return two different float32 factors. Every LAPACK and cuSOLVER implementation picks it from the shape and the device, per vendor | **PIN.** `CHOL_NB_PINNED = 32` for every shape under IDENTICAL, and a caller hint asking for anything else RAISES BY NAME rather than being ignored. NOT derived from `n`, so the factor of a leading block equals the corresponding part of the factor of the whole. Under FAST it is free. DEVIATION 1630 | **CONSTRUCTION 2026-08-25, NOTHING RUN.** `check_block_size_is_pinned` reads the block size back from the run and from the card's own `chol.nb` stage under IDENTICAL, and under FAST requires two `NB` values to produce DIFFERENT bits |
| 61 | **`cholesky/` -- the PIVOT DECISION** (`panel_factor_kernel`) | Cholesky failure is DATA-DEPENDENT. The value compared is a float sum and the comparison is spelled `s <= 0` in LAPACK, `isnan(sqrt(...))` in RAFT. A subnormal pivot is positive on a column that keeps subnormals and zero on one that flushes, so two vendors can disagree about whether the input is positive definite at all -- one returns a factor, the other returns an error, and no downstream bitwise gate ever runs | **PIN, both halves.** The value is `ftz`-flushed and folded ascending through `identical_mul_add`; the comparison is `not (s > 0.0)`, so NaN fails, both zeros fail, and a flushed subnormal fails on every column. `info` follows LAPACK and is read back per panel. DEVIATION 1634 | **CONSTRUCTION, NOTHING RUN.** `check_pivot_failure_is_identical` (an exactly singular fixture stopping at `info = 38` by arithmetic, with the PARTIAL factor compared cell by cell) and `check_signed_zero_and_denormal` (a subnormal pivot refused at `info = 36`) |
| 62 | **`cholesky/` -- the TRAILING UPDATE** (`potrf_lower`) | `A22 -= L21 L21^T` is a matrix product, and a vendor GEMM's k-split is a per-vendor summation order that nothing in this repository can pin, read or check | **REUSE THE PINNED ONE.** `identical_gemm_into` at `OP_NT`, profile `mojolearn.identical.gemm.fp32.v1`, plus a pinned elementwise subtract over the lower triangle. `linalg.matmul` is refused. At `NB = 32 <= CONTRACT_K_LEAF_MIN` the profile's partition is one leaf, so the fold is an ascending chain. DEVIATION 1636 | **CONSTRUCTION, NOTHING RUN.** `CHOL_SAB_VENDOR_MATMUL` swaps `linalg.matmul` in so the gate can be shown to see it |
| 63 | **`cholesky/` -- the TRIANGULAR SOLVES** (`trsm.mojo`) | `cublastrsm` and `cusolverDnpotrs` are closed; their blocking is a summation order | **ONE THREAD PER RIGHT-HAND-SIDE COLUMN**, `k` ascending in both substitutions, every divide an `identical_div` and never a reciprocal-times. No float crosses a thread boundary, so launch and batch invariance are properties of the kernel's shape. DEVIATIONS 1631, 1643 | **CONSTRUCTION, NOTHING RUN.** `check_cho_solve_residual` (a planted solution recovered BIT FOR BIT on an exactly-representable fixture) and `check_launch_invariance` |
| 64 | **`cholesky/` -- the LOG-DETERMINANT** (`chol_logdet`) | `2 sum log(L_jj)` is a fold (row 21) over a device `log` (row 12), and three downstream lanes will otherwise each compute it their own way | **PIN AND CENTRALIZE.** One thread, ascending, `identical_log`, and it is an ENTRY POINT so a GP, a KRR and a GMM cannot each invent one. `CholeskyFactor` carries it. DEVIATION 1639 | **CONSTRUCTION, NOTHING RUN.** `check_logdet` against a hand-derived closed form (`2 * (count of planted 2.0 diagonals) * log 2`) and against the float64 reference |
| 65 | **`cholesky/` -- the JITTER** (`add_jitter`) | every FP32 kernel matrix needs a diagonal ridge, and if each caller picks its own then reproducibility depends on a number nobody writes down. A RELATIVE ridge additionally makes the perturbation a function of the data | **PIN.** Exactly two accepted values under IDENTICAL, `+0.0` and `2^-20` (bits `0x35800000`, written as bits because `String(Float32)` does not round trip); anything else RAISES BY NAME; a relative policy is refused outright. DEVIATION 1637 | **CONSTRUCTION, NOTHING RUN.** `check_cholesky_refusals` and the `CHOL_SAB_JITTER_RELATIVE` arm |
| 66 | **`cholesky/` -- row 39 in a factorization**: signed zeros in the factor, subnormal pivots, and the strict upper triangle | a `-0.0` in the factor is invisible to every tolerance comparison; an uninitialized upper triangle hashes differently run to run on ONE machine; a subnormal diagonal is the divergent-outcome case of row 61 | signed zeros are ARITHMETIC (IEEE fma and add sign rules on operands the oracle also has) and are compared BY SIGN BIT device against oracle; the strict upper triangle is written `+0.0` once at the end (DEVIATION 1640); subnormal pivots are flushed before the compare and refused everywhere. Non-finite and non-symmetric input refused by name on the host first (DEVIATION 1638) | **CONSTRUCTION, NOTHING RUN.** `check_signed_zero_and_denormal`, with a fixture that PLANTS both zeros because no ordinary input mixes them |

Row numbers 60-66 are proposed, not claimed: `IDENTITY_PATHS.md` is not this
lane's file and the orchestrator assigns them.

## The deviations

| # | what |
|---|---|
| **1630** | **`NB` is a NUMERIC parameter, pinned under IDENTICAL, and a caller hint is refused.** The headline. `potrf.mojo`'s first banner carries the argument in full |
| 1631 | cuSOLVER `potrf` / `potrs` and `cublastrsm` are CLOSED, so `potrf.mojo` and `trsm.mojo` are NOT ports and do not pretend to be. The only portable Cholesky source in the checkouts is the RAFT rank-one update, and its consumer is LARS rather than the SVM |
| 1632 | the rank-one update's three cuBLAS calls become this lane's pinned kernels; the HOST round trip is COPIED rather than optimized away |
| 1633 | RAFT's `eps` clamp is a numerical policy, so an unpinned `eps` is refused under IDENTICAL; and the pivot is tested BEFORE the square root rather than after, which changes the answer on a subnormal and on nothing else |
| **1634** | LAPACK's `info` contract, and the pivot decision itself is pinned in both halves -- the value compared and the comparison |
| 1635 | float32 on the device, float64 only in the host oracle. No `math_t = double` instantiation |
| **1636** | the trailing update is `identical_gemm_into`; `linalg.matmul` is refused; the symmetric half is computed and thrown away rather than introducing a triangular GEMM shape |
| **1637** | the jitter policy is part of the profile, not a caller's free choice; an unpinned value refuses |
| 1638 | non-finite and non-symmetric input refused by name, on the host, before any launch, naming the cell and both values by bits |
| 1639 | `chol_logdet` is a device single-thread ascending fold through `identical_log`, and it is an entry point so three lanes cannot each invent one |
| 1640 | the strict upper triangle of the factor is zeroed with `+0.0`, where LAPACK leaves it untouched, so `chol.factor` hashes the factor and nothing else |
| 1641 | the panel is ONE block with a pinned serial column order and no float crossing a thread boundary; no threadgroup staging |
| 1642 | the sabotage arms are runtime-selectable through a `sabotage` argument, in a separate file, never reached by any driver |
| 1643 | every diagonal division is a divide and never a reciprocal-times, so RAFT's `getDiagonalInverseMatrix` shape is not used on any identity path |
| 1644 | the ported RAFT matrix helpers are ROW-major where theirs are column-major: a relabelling, not an algorithm change |
| 1645 | `matrix_diagonal_inverse` is ported and deliberately uncalled here |
| 1646 | the rank-one update's strided copy is contiguous in row-major storage and is KEPT anyway, so a failed update leaves the caller's matrix untouched; their 256-byte workspace align is not ported |
| 1647-1659 | RESERVED, unspent |

## WHAT THE ORCHESTRATOR MUST WIRE

Nothing outside `cholesky/` was edited by this lane. These lines are wanted
in `pixi.toml`, in the file's existing format, beside the other classical
lanes' tasks:

    check-cholesky = "mojo run -I . cholesky/checks/cholesky_check.mojo"
    cholesky-main = "mojo run -I . cholesky/cholesky_main.mojo"

The IDENTICAL pass is the injector, exactly as for every other gate:
`tools/with_identical_mode.sh pixi run check-cholesky`. No `*-identity` task
is wanted.

Also owed by the orchestrator, and none of it is this lane's to do:

1. **`IDENTITY_PATHS.md` rows 60-66**, from the identity table above.
2. **`PORTING.md` / `ROADMAP.md`**: `cholesky/` is a new section and appears
   in neither.
3. **`UNWIRED.md`**: `cholesky/estimator.mojo` has no caller in
   `bindings/_mojolearn_estimators.mojo` or `python/mojolearn/`, and
   `cholesky/impl/matrix/detail/matrix.mojo`'s
   `get_lower_triangular_kernel`, `copy_vector_to_matrix_diagonal_kernel` and
   `matrix_diagonal_inverse_kernel` have no caller in this lane (the last one
   deliberately, DEVIATION 1645; the first two are ported-and-unwired and
   belong on the list).
4. **The Python surface**, when a binding is wanted:
   `cholesky/estimator.mojo::cholesky_factor_host`, `cholesky_solve_host`,
   `cholesky_logdet_host`, `cholesky_rank1_update_host`,
   `cholesky_profile_jitter`. Refuse `float64` inputs and any `nb` keyword in
   Python, by name.

## SABOTAGES TO PERFORM

All ten are selected at RUN TIME through the `sabotage` argument
(`cholesky/checks/chol_sabotage.mojo`), copying
`hierarchy/checks/sabotage_tile.mojo`'s construction and how
`linkage_check.mojo` drives it. **No source edit and no rebuild is required
for any of them**; `check_cholesky_sabotages` drives all ten in one run and
prints a line per arm. All ten were driven on the Apple M4 on 2026-08-25 in
both modes. The column below is what each arm MUST move; it is not a
transcript of what it did, and the two corrections that run forced are
recorded above.

| check it targets | sabotage | exactly what it corrupts | what must move |
|---|---|---|---|
| `check_block_size_is_pinned`, `check_launch_invariance` | `CHOL_SAB_NB_FROM_LAUNCH` | the driver sets `NB = panel_tpb` instead of the pin, so the block size reaches the numerical parameter | `CholRun.nb` must report a value that is not `CHOL_NB_PINNED`, proving the read-back sees the run rather than the driver's intention. MUST FAIL |
| `check_potrf_vs_oracle` | `CHOL_SAB_PANEL_DESCENDING` | the within-column and panel-solve sums walk `k` DESCENDING: same multiset, different order | the factor's bits, on any inexact fixture. MUST FAIL |
| `check_launch_invariance` | `CHOL_SAB_PANEL_ROTATE` | the panel solve's inner sum starts at `k0 = block_idx.x % span` and wraps, so the summation order is a function of launch geometry | the factor's bits, but ONLY when the panel solve runs at more than one block. Driven at `solve_tpb = 8`. MUST FAIL there; INERT at one block, which is the point |
| `check_potrf_vs_oracle` | `CHOL_SAB_STD_SQRT` | `std.math.sqrt` on the pivot instead of `identical_sqrt` | REPORT. Apple's and AMD's `sqrt` are correctly rounded so this is expected INERT there; NVIDIA's PTX `sqrt` is off by one ulp on 180,714 of 2^20 patterns (DEVIATION 258) and every diagonal entry goes through it |
| `check_signed_zero_and_denormal` | `CHOL_SAB_NO_FTZ_PIVOT` | `ftz` dropped on the pivot value before the comparison | APPLE-INERT, and this is the arm that matters most. Metal flushes in hardware so no Apple bit is expected to move; on NVIDIA and AMD the unflushed `2^-140` pivot is positive, `FIX_DENORMAL_PIVOT` SUCCEEDS instead of returning `info = 36`, and the two vendors disagree about whether the input is positive definite. RECORDED, never claimed |
| `check_pivot_failure_is_identical` | `CHOL_SAB_PIVOT_GE` | the pivot test becomes `s < 0`, so an exactly-zero pivot passes | `info` on the exactly-singular fixture, from 38 to something else. MUST FAIL |
| `check_potrf_vs_oracle` | `CHOL_SAB_VENDOR_MATMUL` | the trailing update calls `core/gemm.mojo::gemm_nt`, i.e. MAX `linalg.matmul` | the factor's bits under IDENTICAL. If it moves NO bit, that is RECORDED and is not evidence the swap is safe: it means the vendor happened to pick this profile's order at this shape on this device, which is the thing no vendor guarantees |
| `check_cho_solve_residual`, `check_potrf_vs_oracle` | `CHOL_SAB_TRSM_RECIPROCAL` | the solves multiply by `1 / L_ii` instead of dividing, which is RAFT's own `matrixDiagonalInverse` shape | the factor's bits (the panel solve carries the arm). MUST FAIL |
| `check_logdet` | `CHOL_SAB_LOGDET_PAIRWISE` | the log-determinant folds PAIRWISE instead of ascending serial | `chol.logdet`'s bits. MUST FAIL |
| `check_potrf_vs_oracle` on `FIX_ILL` | `CHOL_SAB_JITTER_RELATIVE` | the ridge becomes `jitter * A_ii`, the policy a GP library would plausibly choose for itself | the factor's bits. MUST FAIL |

**One arm the brief asked for has no object here and it is worth saying why:
"swap the two fold levels" and "change the block reduction's width". There
is no block reduction anywhere in this lane.** The panel keeps every sum
inside one thread, both solves keep every sum inside one thread, the
log-determinant runs in one thread, and the only cross-thread combination in
the whole factorization lives inside `identical_gemm_into`, where the gemm
lane's own six sabotages already cover it. That absence is the reason
launch invariance is a property of shape here rather than a property a check
happens to observe.

## What the checks are expected to establish

| check | what it would establish |
|---|---|
| `check_cholesky_refusals` | NaN, infinity, asymmetry, a bad dimension, a negative or NaN jitter, an unpinned jitter under IDENTICAL, an `NB` hint under IDENTICAL, and a singular matrix each refuse BY NAME; a symmetric matrix, both pinned jitters and the pinned `NB` are accepted, so the refusals are not simply always firing |
| `check_potrf_vs_oracle` | the device factor equals the float32 host replay bit for bit at every cell of all six fixtures under IDENTICAL (a REPORT under FAST), the `info` values agree, and the 48x48 PLANTED factor equals its own plant bit for bit in BOTH modes -- a hand-derivable answer, possible only because that fixture's arithmetic is exact end to end |
| `check_potrf_reconstructs` | `L L^T` equals the jittered input: on the planted fixture bit for bit, elsewhere to a relative cell error the check prints. Independent of the oracle comparison -- a transposed index or a dropped panel passes that one and fails this |
| `check_cho_solve_residual` | a planted `X` comes back BIT FOR BIT from `B = A X` on the exact fixture, device and oracle agreeing; a float64 residual on the RBF fixture |
| `check_logdet` | the device scalar equals the closed form `2 * (count of planted 2.0 diagonals) * log 2` and the float64 reference |
| `check_pivot_failure_is_identical` | the singular fixture stops at `info = 38` on device and oracle, with every cell of the PARTIAL factor equal and every card stage equal |
| `check_launch_invariance` | **the headline.** The factor does not move across `panel_tpb` 128/32, `elem_tpb` 256/64, `solve_tpb` 256/8 (one block against many), 0/37 floats of padding, two poisons, the same run twice, or between the matrix factored alone and the same matrix as the leading block of a block-diagonal matrix of twice the size. That last arm is DEVIATION 1630's gate as much as a scheduling one: if `NB` were derived from `n`, the 2n run would partition differently and it would fail while everything else passed |
| `check_block_size_is_pinned` | under IDENTICAL, that the pinned `NB` is what RAN -- read back from `CholRun` and from the card's own `chol.nb` stage, and cross-checked against the panel count -- and that a hint refuses. Under FAST, that `NB = 32` and `NB = 16` give DIFFERENT bits on at least one swept fixture, which is what makes the pin load-bearing rather than decorative. **If neither fixture separates them the check RAISES**, and that is the correct outcome |
| `check_card_is_emitted` | the stage list, in order, with the right record count, and two runs of one fixture producing an identical card |
| `check_signed_zero_and_denormal` | negative zeros actually REACH the factor (the check raises if none do, because then the agreement proves nothing), every zero matches the oracle by sign bit, and a subnormal pivot is refused at `info = 36` where the float64 reference succeeds |
| `check_r1_update_equals_potrf` | the ported RAFT rank-one update and the blocked factorization agree bit for bit up to the panel width -- two DEVICE spellings of one arithmetic, which is a stronger claim than a device-versus-host comparison -- and REPORT beyond it, where they are entitled to differ |
| `check_cholesky_sabotages` | all ten arms driven at run time, each classified MUST FAIL, APPLE-INERT or REPORT in advance |

## The card

`cholesky/cholesky_main.mojo` emits, for `n` and `NB` giving `P` panels,
`2 + 3(P-1) + 1 + 4 + 2` stages:

    chol.input, chol.jittered,
    chol.panel000.{factored,solved,trailing} ... chol.panel<P-1>.factored,
    chol.factor, chol.nb, chol.diag, chol.logdet,
    chol.solve.forward, chol.solve.back

The ORDER is the product, not the length. A card that diverges has an address
and the address is the diagnosis: `chol.jittered` moving means the fixture or
the ridge moved; a `panelNNN.factored` moving with its predecessor identical
means the panel's own arithmetic moved (`identical_sqrt`, the `fma` pin, the
flush); a `panelNNN.trailing` moving with `solved` identical means the GEMM
moved, which is the gemm lane's certificate and not this one's; `chol.nb`
differing means two runs used two different NUMERIC parameters and nothing
below it is comparable at all; `chol.logdet` moving with `chol.diag`
identical is `identical_log` and nothing else.

**No card has been emitted.** The stage list above is what the source
records, not a transcript.

## WHAT IS OWED

1. **THE IDENTICAL-MODE LEG ON A SECOND VENDOR.** This lane has been
   compiled and executed. It is built and gated on one Apple M4 in both
   modes (Status, 2026-08-25), and an NVIDIA H100 compiled and ran it under
   FAST on 2026-08-26 at fixture `RBF.64x64r4`, median 0.316 ms against
   torch-gpu's 0.215 ms
   (`bench/results/fast_speed/2026-08-26_040100-nvidia-classical.md`). A
   FAST leg is a speed leg and buys no identity; there is still no
   IDENTICAL-mode card from any vendor except the M4, which is item 2.
2. **The second and third vendor legs.** An NVIDIA H100 and an AMD MI325X
   run of `cholesky_main.mojo` under `tools/with_identical_mode.sh`, and
   `tools/identity_trace_diff.py` over the three cards. Until then there is
   no cross-vendor claim of any kind. Two arms are expected to behave
   differently there and are recorded that way rather than as passes:
   `CHOL_SAB_STD_SQRT` (NVIDIA's approximate PTX `sqrt`, DEVIATION 258) and
   `CHOL_SAB_NO_FTZ_PIVOT` (NVIDIA and AMD keep subnormals).
3. **Whether `FIX_ILL` fails without a ridge.** Deliberately not predicted.
   A rank-deficient Gram matrix has a zero pivot in exact arithmetic and
   whatever float32 rounding makes of zero in practice; the check asserts
   device-equals-oracle in both the jittered and un-jittered runs and prints
   the two `info` values. The orchestrator records what it got.
4. **A blocked `trsm`.** Today each right-hand side reads `i` floats of `L`
   per row with no reuse: `O(n^2 nrhs)` global loads. A blocked solve turns
   it into GEMM work. It is a speed idea, it would introduce a second fold
   to pin, and the blocked solve has never been measured against the
   present one, so it is named rather than attempted.
5. **A staged panel.** Same shape of debt (DEVIATION 1641).
6. **A `syrk` shape in the gemm profile.** DEVIATION 1636 computes the whole
   symmetric product and throws half away. Closing it means a triangular
   output arm of `mojolearn.identical.gemm.fp32.v1`, which is the gemm
   lane's call and not this one's.
7. **Larger `n`.** The largest fixture is 64 x 64 and the largest matrix any
   check factors is the 128 x 128 block-diagonal embedding. Nothing here
   says what happens at 4096, where the workspace is `n^2 + n NB` floats
   beside the matrix and the per-panel host round trip is 128 drains.
8. **The `float64` device path**, which does not exist and cannot on Metal.
   A caller needing FP64 conditioning has DEVIATION 1637's ridge and nothing
   else.
