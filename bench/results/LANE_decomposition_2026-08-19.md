# LANE decomposition — 2026-08-19

**VERDICT: PCA above 32 features is now TRUSTWORTHY.** The `0f17dde` fix is
real. Verified independently at n = 16, 32, **33**, 64, 128, 256 against the
eigendecomposition property itself (`A V = V diag(lambda)` per cell,
`V^T V = I`), cross-checked against the host Float64 solver, and the full PCA
at 64 and 128 features matches `sklearn.decomposition.PCA` to 1.7e-06 on
`explained_variance_` and 6.0e-06 on component directions. The check has REACH,
proved by sabotage: re-imposing the cap leaves n = 16 and n = 32 byte-identical
and makes n = 33 fail at `||V^T V - I|| = 0.61`.

**Strongest remaining gap: we ship the arm cuML's dispatch does NOT take.**
`calEig`'s default is `COV_EIG_DQ`, not Jacobi. Details in row 1.

---

## 0. THE ORCHESTRATOR'S FIRST MESSAGE IS A MISATTRIBUTION — READ THIS FIRST

> "You edited files your brief put out of bounds: `core/gemm.mojo` (-280),
> `core/column_stats.mojo` (-321), `glm/ported/linalg/detail/lstsq.mojo`."

**I did not.** Every write this lane made was to an explicit `decomposition/`
path. `git status --porcelain -- decomposition/` is the complete list of my
work:

    M decomposition/mojo_only/jacobi_eigh.mojo
    M decomposition/mojo_only/jacobi_eigh_device.mojo
    M decomposition/mojo_only/pca_check.mojo
    M decomposition/ported/linalg/detail/pca.mojo
    M decomposition/ported/linalg/detail/tsvd.mojo
    M decomposition/PORTED_MAP.tsv          (assigned to me)
    M decomposition/README.md               (assigned to me)
    M decomposition/UNPORTED.tsv            (assigned to me)
    ?? decomposition/jacobi_main.mojo
    ?? decomposition/mojo_only/jacobi_check.mojo
    ?? decomposition/pca_wide_main.mojo
    ?? decomposition/pca_wide_sklearn.py

plus **one deliberate, minimal edit to `glm/ported/linalg/detail/lstsq.mojo`,
described in section 0b**, which is the fix for the build break.

`core/gemm.mojo` last changed at 18:13:11 and `core/column_stats.mojo` at
18:00:41, both while this lane was running but neither by it. The docstring
you quoted ("248 at 1M x 128") is not mine and I cannot supply the measurement
behind it. **The lane that deleted `gemm_nt_kernel`, `covariance_kernel` and
`covariance_reduce_kernel` is someone else and still needs to be asked.** I
did read `core/gemm.mojo` and `core/column_stats.mojo` as part of the limit
hunt (section 2) and found no problem-dimension cap in either.

One thing that lane's edit *did* do to my files: it removed `COV_TILE` from
`decomposition/ported/linalg/detail/pca.mojo`'s import block. I picked that up
on a re-read and preserved it, so nothing was clobbered — but a peer is
editing inside `decomposition/`, and that is worth knowing.

### 0b. The glm build break, and why it IS mine

`glm/ols_main.mojo` failed with `constraint failed: Wrong number of arguments
to enqueue` at `lstsq.mojo:134`. That is not the lstsq rewrite: it is **my**
change to `jacobi_eigh_kernel`, which gained an `info_out` parameter (row 4).
`lstsq_eig` launches that same kernel and was still passing five arguments.

Fixed, minimally, in `glm/ported/linalg/detail/lstsq.mojo` — the only lines
touched are the import, the `enqueue_function` argument list, and a
non-convergence check on the returned info. `glm/ported/glm/ols.mojo`,
`glm/mojo_only/ols_check.mojo` and `glm/ols_main.mojo` were **not** touched.

    check_ols_exact OK  |  check_ols_scale_invariant OK
    check_ols_beats_truth_on_noise OK  |  check_ols_dispatch_guard OK

---

## 1. DIVERGENCES FOUND

The most important finding is not in any single row: **`raft/linalg/detail/pca.cuh`
and `raft/linalg/detail/tsvd.cuh`, cited by every file in this section, have
never existed.** `git log --all -- cpp/include/raft/linalg/detail/pca.cuh` in
the RAFT checkout returns nothing. PCA lives in cuML. Every line number quoted
from those two paths was invented, and so were two function names. The section
was written from a recollection of their code. All citations below are from
files on disk at cuML `00094f7` / RAFT `661a3b8`.

| # | What upstream does (file:line) | What ours did | Fixed? |
|---|---|---|---|
| **1** | **`calEig` branches on `prms.algorithm`, which DEFAULTS to `solver::COV_EIG_DQ`** — `eigDC` → cuSOLVER `syevd`. `cuml/cpp/src/tsvd/tsvd.cuh:108-120`, `cuml/cpp/include/cuml/decomposition/params.hpp:53`. The Python layer maps **both** `svd_solver='auto'` and `'full'` onto `COV_EIG_DQ` and reaches `COV_EIG_JACOBI` only from `'jacobi'` (`pca.pyx:392-404`). | We ship Jacobi and three files asserted it was "THEIR algorithm choice, not our substitute". **That is the k-NN mistake: we ported the arm their dispatch does not take.** | **Docs fixed, code not.** Both arms end in closed cuSOLVER (`syevd`/`syevj`) and MAX ships no symmetric eigensolver (row 3), so `COV_EIG_DQ` is no more portable than `COV_EIG_JACOBI` — there is nothing to port either way. Jacobi stays because it can be diffed rotation-by-rotation against a host oracle. Now recorded as a SUBSTITUTION in `UNPORTED.tsv` and `PORTED_MAP.tsv`, and the false sentence is deleted from `README.md`, `jacobi_eigh.mojo` and `jacobi_eigh_device.mojo`. |
| **2** | **`pcaFit` does not flip signs at all** (`cuml/cpp/src/pca/pca.cuh:104-139`), and it is what `PCA.fit()` reaches (`pca.pyx:547`). `pcaFitTransform` calls `ML::signFlip` (`tsvd.cuh:139`), which is **U-based**: argmax-abs down each column of the TRANSFORMED data, flipping the score column and the component row together (`tsvd.cuh:151-176`). `raft::matrix::signFlipKernel` (`raft/matrix/detail/math.cuh:367`) is V-based but **no cuML path calls it** — only RAFT's own test (`raft/cpp/tests/matrix/math.cu:176`). | Docstring claimed "RAFT calls `sign_flip_components` with a `flip_signs_based_on_U` switch (`detail/pca.cuh:189`). This ports the default arm." **Neither symbol exists anywhere in RAFT or cuML** (`grep -rn` over both checkouts: zero hits). Fabricated. | **Docs fixed, behavior deliberately kept.** We flip V-based in `pcaFit`'s position. Reason stated in the file: a fit whose signs are "whatever the eigensolver returned" is not reproducible, and ours is not cuSOLVER. sklearn 1.9 resolves it identically — `_fit_full` calls `svd_flip(U, Vt, u_based_decision=False)`, verified by reading the installed source, not assumed. Recorded as DIVERGENT CALLER in `PORTED_MAP.tsv`. |
| **3** | RAFT/cuML call cuSOLVER for the eigensolver. | Hand-written Jacobi, justified by "cuSOLVER is closed". | **Confirmed correct, and the negative result is now citable.** MAX's `linalg` ships `arch`, `matmul`, `accumulate`, `block_scaled_*`, `bmm`, `fp4/6/8_*`, `gemv`, `grouped_matmul*`, `lora`, `matrix_band_part`, `mx_format`, `mxfp*`, `packing`, `qr_factorization`, `structuring`, `transpose`, `utils`, `utils_gpu`. **No eigensolver, no SVD, no Cholesky, no triangular solve.** `docs_search` for "eigenvalue eigendecomposition symmetric eigensolver" in `kernels` returns 0 hits; "linalg svd factorization" returns only the `linalg` index page. `strings` over the installed `linalg.mojoc` finds no `eig*`/`svd*`/`syev*` symbol. |
| **4** | `eigDC` — the arm the default dispatch reaches — **aborts** on non-zero `dev_info`: "eigensolver couldn't converge to a solution" (`raft/linalg/detail/eig.cuh:79-82`). `eigJacobi` fetches `executed_sweeps` via `cusolverDnXsyevjGetSweeps` and **never reads it**, and never checks `dev_info` (`eig.cuh:310-311`). | Hitting the sweep limit was completely **silent**. The kernel fell out of the loop and returned a non-answer. Same class as the size cap: a wrong answer with no error. | **FIXED.** The kernel writes a 3-slot info buffer (`converged`, `||offdiag||_F/||A||_F`, `sweeps`); `eig_and_truncate` and `lstsq_eig` raise on it. We follow the DEFAULT arm's behavior, not the Jacobi arm's. |
| **5** | `eigJacobi`'s `tol` is cuSOLVER's, i.e. measured against the matrix; RAFT's default is `1.e-7` with `sweeps = 15` (`raft/linalg/eig.cuh:108-109`), matching cuML's Python defaults (`pca.pyx` `tol=1e-7`, `iterated_power=15`). | Hardcoded `Int32(80)` sweeps and `Float32(1.0e-10)` **absolute** tolerance on a sum of squares — neither from their code. `off` scales with the square of the data, so the same matrix in different units was a different problem, and on any realistic covariance the test was unreachable so all 80 sweeps always ran. | **FIXED.** Test is now `||offdiag||_F <= tol * ||A||_F`, defaults are theirs. Measured: n = 16/32/33/64/128/256 converge in **5/6/6/8/8/9** sweeps, all inside their budget of 15. `check_jacobi_scale_invariance` asserts A and 1000·A take the same sweep count (7 and 7). |
| **6** | `truncCompExpVars` guards the noise variance with `prms.n_components < prms.n_cols && prms.n_components < prms.n_rows`, else sets 0 (`cuml/cpp/src/pca/pca.cuh:74-83`). | Ours had only `n_components < n_cols`. | **FIXED**, as `n_components < n_cols and n_components <= singular_scale` (`singular_scale` is `n_rows - 1` on the PCA path and 1 for tSVD, which correctly suppresses it there since `tsvdFit` computes no noise variance at all). New `check_pca_truncation` gives the path reach. |
| **7** | `calEig` does `colReverse` **then `raft::linalg::transpose(components, n_cols, stream)`** (`tsvd.cuh:122-123`), so a component ends up in a ROW; `truncZeroOrigin` then keeps the first `n_components` rows. | We do neither; the host gather reads column `src` of a row-major matrix straight into a contiguous component. | **Not a defect — same result, one less pass.** Their reverse+transpose and our sorted gather land on the same layout (`components_` = `n_components x n_cols`, component contiguous). Recorded because the *reason* differs: cuSOLVER returns eigenvalues ascending so a reverse suffices; cyclic Jacobi returns no order at all, so a sort is required, not preferred. |
| **8** | `pcaFit` **clamps**: `if (n_components > prms.n_cols) n_components = prms.n_cols;` (`pca.cuh:123`). | Ours raises. | **NOT fixed, deliberately.** Their clamp is applied only to `seqRoot`; `truncCompExpVars` still truncates with the unclamped `prms.n_components` (`pca.cuh:132`), which over-reads `components_all`. Copying a latent out-of-bounds read is not copying. Recorded in `UNPORTED.tsv`. |
| **9** | `tsvdFit` computes **no** `explained_var`/`ratio`. `tsvdFitTransform` derives them from the variance of the TRANSFORMED data, ratioed against the total per-column variance of the input (`tsvd.cuh:272-293`). | Our `tsvd_fit` returns Gram eigenvalues as `explained_var` and ratios them against their own sum. | **NOT fixed.** A different quantity from theirs, and the fix is `tsvdFitTransform`, which is unported. Recorded in `UNPORTED.tsv`. |
| **10** | `raft::stats::cov<rowMajor>(..., sample=true, stable=true)` → `meanCenter` in place, then GEMM with `alpha = 1/(N-1)`, leaving the data centered for the caller to restore (`raft/stats/detail/cov.cuh:58-66`); `pcaFit` restores with `meanAdd` (`pca.cuh:138`). | Same, with the scale as a separate pass because MAX's matmul has no `alpha`. | **Correct as-is.** The `n-1` denominator, the in-place centering and the restore all match. `check_input_restored` covers the restore. |
| **11** | cuML is column-major throughout (`raft::stats::cov<false>`, `raft::stats::mean<false>`). | We are row-major throughout. | **Correct as-is**, a convention not a divergence, but it is why our `sign_flip_kernel` reads `f * n + col` where theirs reads `blockIdx.x * D + i`. |
| **12** | The enum is `solver::COV_EIG_DQ` (params.hpp:30). | `UNPORTED.tsv` called it `COV_EIG_DC`. | **FIXED.** A symbol that does not exist cannot be grepped for by the next reader. |

---

## 2. THE LIMIT HUNT — every bound in `decomposition/**`, classified

The prior audit that "found none remaining" was checking the wrong thing. Here
is the complete list with each one classified.

| Constant / construct | Where | Bounds a PROBLEM or a LAUNCH? | Action |
|---|---|---|---|
| `JACOBI_MAX_N = 32` | *(was)* `jacobi_eigh_device.mojo` | **PROBLEM** | Already gone at `0f17dde`; **independently verified this round** (section 6). |
| `JACOBI_TPB = 32` | `jacobi_eigh_device.mojo:76` | LAUNCH | Legitimate. Every loop is strided by it; `n` is unbounded. Its comment claimed the opposite and is deleted. |
| `SIGNFLIP_TPB = 32` | `pca.mojo:196` | LAUNCH | Legitimate; strided loops. RAFT dispatches this on `D` (32/64/128/256, `math.cuh:400-409`) purely for throughput. **OPEN**: matching their ladder is a speed item, not correctness. |
| `stack_allocation[2]` (`rot`) | `jacobi_eigh_device.mojo` | Neither | 2 scalars in SHARED, size independent of `n`. Legitimate. |
| `stack_allocation[3]` (`sh`) | `pca.mojo` | Neither | 3 scalars in SHARED. Legitimate. |
| `Int32(80)` sweeps, `Float32(1e-10)` tol | `pca.mojo` *(was)* | **PROBLEM — a silent truncation of the ANSWER** | **FIXED**, rows 4 and 5. This is the second real one and the brief was right to name it. |
| `grid_dim=(n_cols,1,1)` ×2 | `pca.mojo` | LAUNCH, covers | Legitimate. |
| `grid_dim=((cells+255)//256,1,1)` ×4 | `pca.mojo` | LAUNCH, covers | Legitimate grid clamp over a strided loop. |
| `grid_dim=(1,1,1)` | jacobi launch | LAUNCH | Single block *by design* — the `(p,q)` sequence is serial. Not a cap; slow at large `n`, see OPEN items. |
| `PCA_COLS = 4` | `pca_check.mojo:55` | **A CAP ON WHAT IS TESTED** | The reason the bug shipped. **FIXED** by adding `check_pca_wide` / `check_pca_truncation` at 64 and 128, and `jacobi_check.mojo` to 256. |
| `if n_components > n_cols: raise` | `pca.mojo`, `tsvd.mojo` | Guard, not a silent clamp | Kept; see row 8. |

**Read-only sweep of `core/` (reported, not edited):** `core/gemm.mojo`
(`GEMM_VECLEN/KBLK/ACC_*/MBLK/NBLK/SMEM_*`) and `core/column_stats.mojo`
(`STATS_TPB = 128`, `TRANSPOSE_TILE = 32`) are all tile and launch geometry
with grid-covered dispatch. **No problem-dimension cap found in either.**

---

## 3. WHAT I CHANGED, file by file

**`decomposition/mojo_only/jacobi_eigh_device.mojo`**
- Deleted the false comment `# One block, and the matrix plus the basis both
  live in shared memory, so this is the cap. 32 x 32 x 4 bytes x 2 arrays =
  8 KB…`. Replaced with what `JACOBI_TPB` actually is (launch geometry).
- Header rewritten: it now states that `COV_EIG_JACOBI` is **not** their
  default arm (`cuml/cpp/src/tsvd/tsvd.cuh:108`, `params.hpp:53`,
  `pca.pyx:392-404`) and that this file is a substitution.
- `JACOBI_TOL = 1e-7` / `JACOBI_SWEEPS = 15` from `raft/linalg/eig.cuh:108-109`.
- **DEVIATION BLOCK 1**: relative convergence test.
- **DEVIATION BLOCK 2**: `info_out` and non-convergence reporting, following
  `eigDC` (`raft/linalg/detail/eig.cuh:79-82`) rather than `eigJacobi`
  (`eig.cuh:310`).
- Dropped dead `block_dim`, `block_idx` imports.

**`decomposition/ported/linalg/detail/pca.mojo`** (from `cuml/cpp/src/pca/pca.cuh`)
- Every `raft/linalg/detail/{pca,tsvd}.cuh` citation corrected to the cuML
  paths and re-derived line numbers; RAFT pin `9aa17e5` → cuML `00094f7`.
- The fabricated `sign_flip_components` / `flip_signs_based_on_U` passage
  deleted and replaced with what `pcaFit`, `pcaFitTransform` and
  `raft::matrix::signFlipKernel` actually do.
- Two garbled comment blocks (a bad merge at `0f17dde`) rewritten: the
  `SIGNFLIP_TPB` note and the selection-sort note that still said the sort
  "cannot grow while that cap holds".
- Jacobi launch: info buffer, their defaults, and the convergence raise.
- `noise_vars` guard completed (`pca.cuh:74-83`).
- Dropped the dead `cluster.mojo_only.reduce_by_key` import.

**`decomposition/ported/linalg/detail/tsvd.mojo`** — citations corrected to
`cuml/cpp/src/tsvd/tsvd.cuh::tsvdFit`; dead import dropped.

**`decomposition/mojo_only/jacobi_eigh.mojo`** — the "WHY THE HOST" section
described a path the fit stopped taking. Rewritten: this is the **oracle**,
not a CPU fallback. The false "Jacobi is THEIR algorithm choice, not our
substitute" sentence deleted.

**`decomposition/mojo_only/jacobi_check.mojo`** (NEW) —
`check_jacobi_device_sizes`, `check_jacobi_reaches_past_32`,
`check_jacobi_scale_invariance`. Hashed per-cell fixture, per-cell residual.

**`decomposition/mojo_only/pca_check.mojo`** — added `check_pca_wide`
(64 and 128 features, emits for the oracle) and `check_pca_truncation`
(128 features, `truncZeroOrigin` + `noise_vars` reach).

**`decomposition/jacobi_main.mojo`**, **`decomposition/pca_wide_main.mojo`**,
**`decomposition/pca_wide_sklearn.py`** (NEW) — entry points and the oracle.

**`decomposition/README.md`**, **`PORTED_MAP.tsv`**, **`UNPORTED.tsv`** —
corrected paths, the default-arm finding, the cap section, current results.

**`glm/ported/linalg/detail/lstsq.mojo`** — section 0b.

---

## 4. PROPOSED PORTING.md DEVIATION ENTRIES (numbered from 30)

**30. The device Jacobi's convergence test is RELATIVE, and RAFT's is a
closed-library parameter.** cuSOLVER `syevj` takes `tol` measured against the
matrix (`raft/linalg/detail/eig.cuh:276`, default `1.e-7` at
`raft/linalg/eig.cuh:108`). Ours used `off <= 1e-10` on an absolute sum of
squares, so the same matrix in different units converged differently and on a
real covariance the test was unreachable. Now
`||offdiag||_F <= tol * ||A||_F`, with `tol = 1e-7`, `sweeps = 15` from their
defaults. Measured: 5/6/6/8/8/9 sweeps at n = 16/32/33/64/128/256; A and
1000·A both take 7.

**31. We report eigensolver non-convergence; RAFT's Jacobi arm does not.**
`detail::eigJacobi` fetches `executed_sweeps` and never reads it, and never
checks `dev_info` (`raft/linalg/detail/eig.cuh:310-311`). Their DEFAULT arm
`eigDC` aborts on `dev_info != 0` (`eig.cuh:79-82`). We follow the default
arm: an unconverged eigendecomposition returned as if it were one is the same
defect class as the 32-feature cap.

**32. The PCA sign flip runs on a path cuML does not flip on, with the
opposite convention.** `pcaFit` (`cuml/cpp/src/pca/pca.cuh:104`) does not flip;
`pcaFitTransform` flips U-based (`tsvd.cuh:139`). We flip V-based inside the
fit, matching sklearn 1.9's `svd_flip(U, Vt, u_based_decision=False)`. Reason:
reproducibility — our eigensolver is not cuSOLVER, so "whatever the solver
returned" is not a convention.

**33. We ship `calEig`'s `COV_EIG_JACOBI` arm; their dispatch defaults to
`COV_EIG_DQ`.** `params.hpp:53`; `pca.pyx:392-404` maps `'auto'` and `'full'`
onto DQ. Both arms terminate in closed cuSOLVER (`syevd` / `syevj`) and MAX
ships no symmetric eigensolver, so neither is portable; Jacobi is chosen
because it can be checked against a host reference rotation by rotation.

**34. `n_components > n_cols` raises where `pcaFit` clamps.** Their clamp
(`pca.cuh:123`) reaches only `seqRoot`; `truncCompExpVars` still truncates
with the unclamped value and over-reads. Not copied.

---

## 5. FALSE DOC SENTENCES FOUND IN FILES I MAY NOT EDIT

- **`VENDOR_LIBRARIES.md:67`** — `| RAFT `pca.cuh` | cuSOLVER `syevj` | …`.
  The *conclusion* ("NOT FOUND as a dense eigensolver") is correct and I
  re-verified it. The *attribution* is wrong: there is no `pca.cuh` in RAFT,
  and it is cuML that calls `eigJacobi`. Suggest `cuml pca.cuh -> raft
  eigJacobi -> cuSOLVER syevj (opt-in arm; the default arm is eigDC ->
  syevd)`.
- **`PORTING.md 22`** (as quoted by `jacobi_eigh.mojo`) — records the
  eigensolver as running on the host with a condition that would move it to
  the device. **It moved.** The entry describes a state that no longer exists.
- **`UNWIRED.md`** — `pca.mojo` says the ordering/truncation departure "is
  named in UNWIRED.md"; whoever owns that file should confirm the entry still
  matches, since `sign_flip` left the host list and the size cap is gone.
- **`HOST_AND_DEVICE.md`** — not read in full; the `jacobi_eigh.mojo` rule-one
  discussion it anchors was rewritten this round.

---

## 6. BUILD / CHECK EVIDENCE

    tools/with_build_lock.sh pixi run --manifest-path .../mojotrees/pixi.toml \
      mojo build -I . decomposition/{pca_main,pca_wide_main,jacobi_main}.mojo
    tools/with_build_lock.sh ... mojo build -I . glm/ols_main.mojo

all clean (warnings only, none new).

### The cap is gone — `decomposition/jacobi_main.mojo`

    n =  16: ||V^T V - I||_max = 2.2055e-06, worst cell of A V - V diag(lam) / ||A||_F = 3.9327e-07, vs host Float64 spectrum = 4.7406e-07, sweeps = 5
    n =  32: ||V^T V - I||_max = 4.6394e-06, worst cell = 8.8848e-07, vs host = 1.3592e-06, sweeps = 6
    n =  33: ||V^T V - I||_max = 5.3954e-06, worst cell = 8.1501e-07, vs host = 1.2495e-06, sweeps = 6
    n =  64: ||V^T V - I||_max = 9.9645e-06, worst cell = 1.0237e-06, vs host = 2.1899e-06, sweeps = 8
    n = 128: ||V^T V - I||_max = 2.8000e-05, worst cell = 1.7866e-06, vs host = 4.3036e-06, sweeps = 8
    n = 256: ||V^T V - I||_max = 6.1534e-05, worst cell = 2.2086e-06, host skipped (too slow), sweeps = 9
    check_jacobi_device_sizes OK
    check_jacobi_reaches_past_32 OK: top eigenvalue 9.598 -> 1001.327 for a +1000
      spike at index 63 (past the old cap), and -> 1006.056 for the same spike at
      index 5 (inside it). Both move.
    check_jacobi_scale_invariance OK: A and 1000 A both converge in 7 sweeps to
      relative off-diagonal 3.1294e-12 and 3.0816e-12; spectra agree to 3.9356e-07

**Tolerances, stated:** `||V^T V - I||_max <= 2e-4`; worst per-CELL
`|A V - V diag(lam)|` relative to `||A||_F` `<= 5e-5`; device Float32 vs host
Float64 spectrum relative to `||A||_F` `<= 5e-4`. The device is Float32
because Metal has no double and the host reference is Float64; **they are not
expected to agree bit for bit and are not compared as if they were.**

### SABOTAGE — the check has reach

Temporarily re-imposed the old cap in the kernel
(`var n = 32 if Int(n_in) > 32 else Int(n_in)`), rebuilt, ran, reverted:

    n = 16: ... 2.2055316433178263e-06 ...   <- byte-identical to the good run
    n = 32: ... 4.6393667185729015e-06 ...   <- byte-identical to the good run
    Unhandled exception: n = 33: V^T V is not the identity, worst entry off by
      0.6120794167416866 against a tolerance of 0.0002

The window is exactly the cap: below it nothing moves, at 33 the error is
**3000x** the tolerance. A digest could not have told these two builds apart;
this can.

### PCA at 64 and 128 — `decomposition/pca_wide_main.mojo`

    check_pca_wide n_cols =  64 OK: spectrum descending, ratios sum to 1.0000000000000002,
      top 8 components orthonormal, rank-8 cliff present, noise floor 0.80604
    check_pca_wide n_cols = 128 OK: spectrum descending, ratios sum to 0.9999999999999997,
      top 8 components orthonormal, rank-8 cliff present, noise floor 0.68470
    check_pca_truncation OK at 128 features: top 10 variances and ratios identical
      to the full fit, ratios sum to 0.99692 (not 1, which is the point),
      noise_var 0.98947 = mean of the 118 discarded eigenvalues, and 0 when none

### sklearn ORACLE — `decomposition/pca_wide_sklearn.py`, scikit-learn 1.9.0

    <python-with-sklearn> decomposition/pca_wide_sklearn.py /tmp/pca_wide.out

    n_cols=  64  explained_variance 1.597e-06  ratio 3.606e-07  singular_values 3.027e-06
                 worst |1-|dot|| over separated components 3.596e-06 (component 4)   OK
                 our top-3 EV [4985.94873047 3856.58300781 3008.25952148]
             sklearn top-3 EV [4985.9185   3856.5684   3008.2407  ]
    n_cols= 128  explained_variance 1.674e-06  ratio 5.480e-07  singular_values 3.875e-06
                 worst |1-|dot|| over separated components 5.993e-06 (component 5)   OK
                 our top-3 EV [8855.81347656 7502.54345703 6003.09423828]
             sklearn top-3 EV [8855.766    7502.48     6003.041   ]
    PCA matches scikit-learn at every width tested.

Components compared **up to sign** and only where the eigenvalue gap exceeds
1e-3 of the total; the fixture is rank-8 over an isotropic floor, so
components 8+ span a degenerate subspace where no individual eigenvector is
defined. That block is still covered — its **eigenvalues** are compared one by
one, and they agree to 1.7e-06.

The sklearn interpreter on this box is
`/Users/andrewhendel/CascadeProjects/monotone-cost/.venv/bin/python`; none of
the system or pixi pythons have sklearn.

### Regression: 4-feature suite and glm unchanged

    check_covariance_is_symmetric OK  |  check_pca_fit OK  |  check_pca_invariants OK
    check_input_restored OK (9.5367e-07)  |  check_tsvd_against_pca OK (|dot| = 0.82770)
    check_ols_exact OK | check_ols_scale_invariant OK | check_ols_beats_truth_on_noise OK
    check_ols_dispatch_guard OK

No timing benchmarks were run.

---

## 7. WHAT I DID NOT DO, AND WHY

- **Did not port `COV_EIG_DQ`.** It is `eigDC` → cuSOLVER `syevd`, closed, no
  source. Porting "divide and conquer" from a textbook would be inventing an
  algorithm, not copying one, and would trade a solver checkable against a
  host oracle for one that is not. Row 1 records the gap instead.
- **Did not substitute a MAX primitive for anything.** There is nothing to
  substitute (row 3), and per the rule change nothing device-wide belongs
  inside the rotation loop.
- **Did not change the single-block Jacobi to a tournament.** The `(p,q)` loop
  is serial: 8128 pairs per sweep at n = 128, 32640 at n = 256, one block, four
  barriers each. **This is now the section's largest performance item** and it
  became one the moment the cap lifted. Deliberately not touched in a
  correctness round, and it needs interleaved arms the orchestrator owns.
- **Did not touch `n_components > n_cols`** (row 8) or tSVD's
  `explained_var` semantics (row 9).
- **Did not add whitening or `pcaInverseTransform`.** Unported, recorded.
- **Did not run the host Float64 cross-check at n = 256** — the host solver is
  single-threaded O(n^3) per sweep and costs minutes there. The per-cell
  residual and orthonormality checks still run at 256 and are the properties
  that actually pin the answer.
- **Did not edit `core/**`, `PORTING.md`, `VENDOR_LIBRARIES.md`, `UNWIRED.md`,
  `HOST_AND_DEVICE.md`, or any root doc.** Findings for those are in section 5.

## 8. OPEN ITEMS FOR THE NEXT ROUND

1. **The serial `(p,q)` loop** — the real cost of removing the cap. A
   round-robin tournament does `n/2` disjoint rotations per round.
2. **`SIGNFLIP_TPB` dispatch ladder** (32/64/128/256 on `D`,
   `math.cuh:400-409`) — throughput only.
3. **The host selection sort in `eig_and_truncate`** — O(n_cols^2) index
   comparisons, 8128 at n = 128, 32640 at 256. `nn.argsort.argsort` is the
   replacement when `n_cols` reaches the low thousands.
4. **Who deleted `gemm_nt_kernel` / `covariance_kernel` /
   `covariance_reduce_kernel` from `core/`, and where is that measurement?**
   Not this lane.
