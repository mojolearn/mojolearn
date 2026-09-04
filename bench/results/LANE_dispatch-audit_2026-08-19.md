# LANE dispatch-audit, 2026-08-19

**One question, asked of every entry point in `cluster/ dbscan/ decomposition/
glm/`: for the parameters this repo actually runs, which function does the
incumbent's dispatch call, and is that the function we ported?**

Read-mostly lane. No kernel body, launch geometry, or algorithm was changed.
Everything below marked NOT FIXED is written up with the exact upstream
`file:line` and the exact change required.

Upstream read at the pinned checkouts: cuML `00094f7`, cuVS `94c2819`,
RAFT `661a3b8`.

---

## 0. THE TABLE

Parameters held fixed throughout: **float32 data, int32 labels/indices, single
GPU, row-major, L2, and the defaults in our own params structs.**

| # | entry point (their public door) | their dispatch target at OUR parameters | ours | same? | file:line |
|---|---|---|---|---|---|
| 1 | `cuvs::cluster::kmeans::fit` (C++) | `detail::kmeans_fit` -> validate, weight fill, **`checkWeight`**, n_init loop, init dispatch, `kmeans_fit_main` | `cluster/gbdt/cluster/kmeans.mojo::fit` -> `kmeans_fit_main`, which has the wrapper folded in | **YES except `checkWeight`** | `cuvs .../detail/kmeans.cuh:812-951`; call at `:872` |
| 2 | k-means init, `init=KMeansPlusPlus` + `oversampling_factor=2.0` (both defaults) | **`initScalableKMeansPlusPlus` (k-means\|\|)**, NOT `kmeansPlusPlus` | RAISES by name | **NO — and we say so** | `detail/kmeans.cuh:910-915`; defaults `kmeans.hpp:75,105` |
| 3 | k-means assignment, `metric=L2Expanded` (default) | `is_fused == true` -> **`fusedDistanceNNMinReduce`** (the fused SIMT arm) | `simt_kernel.mojo`, launched by `min_cluster_and_distance_compute` | **YES** | `kmeans_common.cuh:378-379`, fused arm `:430-449` |
| 4 | k-means Lloyd loop, `inertia_check=false` (default) | cost reduction + `delta > 1-tol` test **skipped**; only `sqrdNormError < tol` stops it | same, gated on the same flag | **YES** | `detail/kmeans.cuh:468-492`; default `kmeans.hpp:120` |
| 5 | `ML::Dbscan::fit`, `eps_nn_method=BRUTE_FORCE` (default, C++ and Python) | `epsUnexpL2SqNeighborhood` (fused, unexpanded, `eps2`) | `dbscan/gbdt/neighbors/epsilon_neighborhood.mojo`, reachable | **YES** | `dbscan.hpp:74`; `vertexdeg/algo.cuh:225,229-230` |
| 6 | `ML::Dbscan::fit`, `eps_nn_method=RBC` **at int32 labels** | **STILL `epsUnexpL2SqNeighborhood`.** RBC is disabled `constexpr` for `Index_ == int32_t` | **we default to RBC and run it** | **NO — deliberate, measured, documented** | `runner.cuh:143-150`, index build guard `:235` |
| 7 | DBSCAN final labelling, `algo_ccl` | `algo_ccl` is hardcoded 2 -> `final_relabel` (make_monotonic) **always**, then `relabelForSkl` | `make_monotonic` then `relabel_for_skl_kernel`, same order | **YES** | `dbscan.cuh:122`; `runner.cuh:410-416` |
| 8 | DBSCAN batch policy | `compute_batch_size` takes `eps_nn_method`; the `MAX_LABEL/n_rows` clamp is **skipped in RBC mode** | same: `eps_nn_method` threaded from `dbscan_fit_impl`, clamp gated (fixed by LANE_dbscan-batching_2026-08-19; was unconditional) | **YES** | `dbscan.cuh:71` |
| 9 | `PCA.fit`, `svd_solver='auto'` (Python default) | `COV_EIG_DQ` -> `raft::linalg::eigDC` -> cuSOLVER `syevd` | device cyclic Jacobi | **NO — closed cuSOLVER both arms, nothing to port** | `params.hpp:53`; `pca.pyx:394-395`; `tsvd.cuh:119` |
| 10 | `PCA.fit` sign of components | **no flip at all.** `pcaFit` never calls `signFlip` | V-based `signFlipKernel` | **NO — deliberate, sklearn-aligned** | `pca.cuh:104-139` (no flip); `math.cuh:367` |
| 11 | `PCA.fit_transform` (Python) | `self.fit(X).transform(X)`. **`pcaFitTransform` and its U-based flip are UNREACHABLE from Python** | n/a | n/a — the U-based arm is not a missing default | `pca.pyx:582-588`; only caller `cpp/tests/sg/pca_test.cu:165` |
| 12 | `PCA.fit` singular values | `seqRoot(..., set_neg_zero=true)` — negative eigenvalue -> 0 | `sqrt(lam*scale)`, no clamp -> NaN | **NO** | `pca.cuh:136`; `math.cuh:86-95` |
| 13 | `TruncatedSVD.fit` (`tsvdFit`) | outputs **only** `components` + `singular_vals`; no `explained_var`/`ratio`/`noise` | returns all five via the PCA-shaped `eig_and_truncate` | **NO** | `tsvd.cuh:190-238` |
| 14 | `TruncatedSVD.fit_transform` | `tsvdFitTransform`: flips signs, and computes `explained_var` from **`vars(trans_input)`**, not the eigenvalues | n/a (not ported); our `tsvd_fit` returns eigenvalues | **NO** | `tsvd.pyx:414`; `tsvd.cuh:270-276` |
| 15 | `LinearRegression` (Python), `algorithm='eig'` (default) | `algo=1` -> `lstsqEig` | `lstsq_eig` | **YES (solver)** | `linear_regression.pyx:309,336-343`; `ols.cuh:120` |
| 16 | `LinearRegression` (Python), `fit_intercept=True` (**default**) | `preProcessData` -> guard -> `lstsqEig` -> `postProcessData` | RAISES on `fit_intercept`; defaults False | **NO — we mirror the non-default branch** | `linear_regression.pyx:309`; `preprocess.cuh:98,100,115,117,148-176` |
| 17 | `olsFit` (C++), `algo` default | `algo = 0` -> `lstsqSvdJacobi` | `algo` defaults to 1 | **NO — recorded in glm/UNPORTED.tsv** | `ols.cuh:67` |
| 18 | `olsFit` shape guard | `n_cols > n_rows \|\| n_cols == 1` -> force algo 0 | copied exactly; raises because algo 0 is unported | **YES** | `ols.cuh:112-113` |
| 19 | `lstsqEig` eigensolver | `raft::linalg::eigDC` (cuSOLVER `syevd`), aborts on `dev_info != 0` | device Jacobi, raises on non-convergence | **NO — closed cuSOLVER; abort behaviour copied** | `lstsq.cuh:315`; `detail/eig.cuh:149-151` |
| 20 | DBSCAN `calc_core_sample_indices` (Python default **True**) | `thrust::copy_if` compaction of the core mask | not ported | **NO — extra output, labels unaffected** | `dbscan.pyx:304,382`; `runner.cuh:419-442` |

Every row above is **CHECKED**: each cited line was opened in the checkout.

Rows I did **NOT CHECK**: `cluster_cost` / `predict` / `fit_predict` beyond the
fact that they reach the same `minClusterAndDistanceCompute` as `fit`; the
`neighbors/` section (another lane owns it); anything under `core/`.

---

## 1. DIVERGENCES FOUND

| # | what upstream does (file:line) | what ours did | fixed? |
|---|---|---|---|
| D1 | `runner.cuh:143-150` disables RBC **`constexpr`** for `Index_ == int32_t`; `:235` builds the index only for `float && int64_t`. We are int32-label, so **cuML's dispatch can never send our parameters to RBC.** | `dbscan/gbdt/dbscan/runner.mojo` defaults to RBC and its DEVIATION 35 note claimed "We are float32, int32-label and L2 only, so those restrictions **cost us nothing**", and the inline fallback list claimed "the first two cannot fire here". Both are the exact opposite of `:143`. | **DOC FIXED.** The default is left at RBC (its 27x measurement and its point-for-point equality check stand), but the file now says plainly that their dispatch never reaches this arm at our parameters and that "we follow their dispatch" is NOT among the reasons. |
| D2 | `dbscan.cuh:71`: the `batch_size <= MAX_LABEL/n_rows` clamp is inside `if (eps_nn_method != RBC)`. RBC emits CSR directly and never materializes the `N*batch_size` dense adjacency the clamp protects. | `compute_batch_size` had no `eps_nn_method` parameter and clamped unconditionally. | **FIXED** by LANE_dbscan-batching_2026-08-19: parameter threaded in their order, clamp gated, reach checked on both sides (`check_dbscan_max_mbytes_moves_the_batch`). On this device the clamp never bound a default-budget run (the 80% budget cuts first at every n), so the fix changes dispatch fidelity and user-raised budgets, not the recorded timings. |
| D3 | `detail/kmeans.cuh:872`: `kmeans_fit` calls `checkWeight` **unconditionally** on the default path. It is not an assertion — it cub-reduces the weights and rescales all of them by `n_samples/sum` when they disagree (`kmeans_common.cuh:160-172`). The rescale cancels in the centroids (`div_checkzero_op` at `detail/kmeans.cuh:326-331`) but **not** in the inertia, which is weighted at `:521-526` and is what `n_init` selects on and what `fit` reports. | `kmeans_fit_main` takes the caller's weights as given. | **NOT FIXED** (needs code; see §4 F2). `cluster/UNPORTED.tsv` row rewritten — it previously said "RAFT_EXPECTS", which is not what the function does. |
| D4 | `pca.cuh:136`: `seqRoot(..., set_neg_zero=true)` turns a negative eigenvalue into a **zero** singular value (`math.cuh:86-95`). Reachable on any rank-deficient or badly scaled design. `tsvdFit` passes no such flag (`tsvd.cuh:237`). | `eig_and_truncate` does `sqrt(lam * scale)` with no clamp -> **NaN** where cuML returns 0, on both paths. | **NOT FIXED** (arithmetic; see §4 F3). Documented in `pca.mojo`. |
| D5 | `tsvdFit` (`tsvd.cuh:190-238`) outputs **only** components and singular values. `explained_var` for tSVD exists only in `tsvdFitTransform` and is `raft::stats::vars` of the **transformed** data (`tsvd.cuh:272-276`) — a different quantity from the Gram eigenvalue on uncentered data. | `tsvd_fit` returns a full `PCAResult` with eigenvalue-derived `explained_var`/`ratio`/`noise_var`. | **NOT FIXED** (arithmetic; see §4 F4). Documented in `tsvd.mojo`. Already an `UNPORTED.tsv` row; the row is accurate. |
| D6 | `linear_regression.pyx:309`: `fit_intercept=True` is the **Python default**, so cuML's default fit is `preProcessData -> lstsqEig -> postProcessData`, not `lstsqEig` alone. | `ols.mojo` said only "A user calling cuML from Python gets the solver we have" and listed `fit_intercept` as unported without saying it is the default. | **DOC FIXED** in `ols.mojo` and `glm/UNPORTED.tsv`. Porting it is §4 F5. |
| D7 | `dbscan.pyx:304` `calc_core_sample_indices=True` is the **Python default** and `:382` allocates the buffer, so `core_sample_indices` is non-null on every default fit. | `dbscan/UNPORTED.tsv` claimed it is "nullptr in every cuML default path". | **DOC FIXED** (the row now states the correction). Still deferred: it is an extra output and does not touch labels. |
| D8 | `raft::argmin_op` (`raft/core/operators.hpp:198-205`) is the reducer with the key tie-break, reached from the unfused arm at `kmeans_common.cuh:474-489`. `MinAndDistanceReduceOpImpl` (`helper_structs.cuh:47-97`) is **value-only**, `:52`/`:59`. | `simt_kernel.mojo` attributed our tie-break to `MinAndDistanceReduceOpImpl` at `helper_structs.cuh:39-62` — wrong struct, wrong lines, and that struct has no tie-break. | **DOC FIXED.** The comparator itself is unchanged and is still right; only the attribution was wrong. |

**Confirmations (checked and matching), because an unchecked "matches" is worth
less than an admitted gap:**

- **`calEig`'s two arms differ in the eigensolver call and in NOTHING ELSE.**
  The branch is `tsvd.cuh:108-120`; `colReverse` (`:122`), the in-place
  `transpose` (`:123`) and `rowReverse` (`:125`) run on **both** arms.
  Scaling, `trunc_comp_exp_vars`, component order and sign handling are all
  downstream of the join. The only other difference is that `prms.tol` and
  `prms.n_iterations` are dead on the DQ arm and live on the Jacobi arm — and
  through the **C++** door `paramsSolver::tol` is `0.0` (`params.hpp:44`)
  while through the **Python** door it is `1e-7` (`pca.pyx:358,413`). Our
  `JACOBI_TOL = 1e-7` / `JACOBI_SWEEPS = 15` match the Python door exactly.
- **`inertia_check` is the only default-false gate in the k-means fit loop.**
  Nothing else in `kmeans_fit_main` (`detail/kmeans.cuh:352-542`) is
  conditioned on a flag we run unconditionally or vice versa.
- **Every RBC downgrade in `runner.cuh:139-201` is accounted for**: `:143-150`
  (double or int32 — fires for us, deliberately not copied, D1), `:152-156`
  (non-L2Sqrt metric — cannot fire, L2 only), `:194-200` (`D > MAX_LABEL/N` —
  can fire, **is** copied at `runner.mojo`).
- **`algo_vd`, `algo_adj`, `algo_ccl` are fixed constants, not user options**
  (`dbscan.cuh:118-122`), and `algo_ccl = 2` makes `final_relabel`
  unconditional. Ours runs it unconditionally. Correct.
- **DBSCAN's default metric `'euclidean'` maps to `L2SqrtExpanded`**
  (`dbscan.pyx:112`), but `vertexdeg/algo.cuh:224-231`'s non-Cosine branch
  calls the **unexpanded** kernel with `eps2 = eps*eps` for it regardless.
  Our unexpanded port is their kernel, not an approximation of it.
- **k-means's fused arm is chosen by metric alone** (`kmeans_common.cuh:378-379`),
  with no compute-capability term. Nothing about our hardware changes which arm
  their dispatch takes.

---

## 2. THE FABRICATION SWEEP (line ranges, not just paths)

Method: every `` `file.ext:NNN` `` and bare `` `:NNN` `` in `cluster/`,
`dbscan/`, `decomposition/`, `glm/` was extracted by script, resolved against
the three checkouts, and the cited lines printed and read. **189 distinct
citations resolved and opened.**

**Wrong, and corrected in place — 12:**

| citation as written | points at | should be | file |
|---|---|---|---|
| `helper_structs.cuh:28-33` = `KVPMinReduceImpl` (x2) | `#include <raft/util/device_atomics.cuh>` | `:39-44` | `cluster/.../simt_kernel.mojo` |
| `helper_structs.cuh:39-62` = `MinAndDistanceReduceOpImpl` (x2) | `KVPMinReduceImpl` | `:47-97`, and that struct is value-only — the tie-break is `raft/core/operators.hpp:202` | `cluster/.../simt_kernel.mojo` |
| `reduce_rows_by_key.cuh:300` = "lands with a global atomic add" | `IdxT ncols,` (a parameter) | `:287` (`raft::myAtomicAdd`) | `cluster/mojo_only/reduce_by_key.mojo` |
| `reduce_rows_by_key.cuh:292` = "`gid = threadIdx.x + ...`" | `typename WeightT,` (a template param) | `:280` | `cluster/mojo_only/reduce_by_key.mojo` |
| `reduce_rows_by_key.cuh:270-288` = the kernel | starts on template params | `:272-288` | `cluster/.../detail/kmeans.mojo` |
| `runner.cuh:365` = the core-point `filter_op` | `n_points,` (an argument) | `:384-386` | `dbscan/gbdt/sparse/detail/csr.mojo` |
| `detail/pca.cuh:189` = `sign_flip_components` | **file has never existed** — a live comment still carried the fabrication the same file's docstring denounces | deleted; replaced with `math.cuh:367` | `decomposition/.../pca.mojo` |
| `detail/eig.cuh:79-82` = "`eigDC` aborts on `dev_info`" (x3) | `eigDC_legacy`'s ASSERT — nothing calls that function | `:149-151` | `lstsq.mojo`, `pca.mojo`, `decomposition/README.md` |
| `ols.cuh:105` = "cuML's OLS solver algo = 1" | `n_rows,` (an argument) | `:120` | `glm/.../lstsq.mojo` |
| `ols.cuh:75-76` = the two ASSERTs | second ASSERT + a blank line | `:74-75` | `glm/.../ols.mojo` |
| `ols.cuh:122-124` = "their default arm" | last case + start of default | `:123-125` | `glm/.../ols.mojo` |
| `dbscan.cuh:147-151` = the quoted memory estimate | the `if` and the `cudaMemGetInfo` decl | `:157-158` | `dbscan/.../dbscan.mojo` |

Plus 5 tightened but not wrong: `ols.cuh:98-110` -> `:99-110`;
`ols.cuh:120-125` -> `:116-126` (the switch); `algo.cuh:131`/`:146` ->
`:132`/`:143`/`:161`; `algo.cuh:137-153` -> `:137-144`;
`runner.cuh:195-200` -> `:194-200`; `tsvd.cuh:139` -> `:140`.

**A RAFT commit hash that is not a valid object:** `lstsq.mojo` said
`PORT OF ... at RAFT '9aa17e5'`. `git cat-file -t 9aa17e5` in the RAFT
checkout returns `fatal: Not a valid object name`. Corrected to `661a3b8`,
the pinned commit. (This is the same class of defect as round 1's cuVS
`2140532c`.)

**Checked and CORRECT — the rest.** Spot-worth naming because they were the
load-bearing ones: `params.hpp:53`, `kmeans.hpp:28-121`/`:44-60`/`:120`,
`detail/kmeans.cuh:825-835`/`:837-846`/`:877-883`/`:910-915`/`:395-399`/`:468-492`,
`kmeans_common.cuh:176-181`/`:183-188`/`:378-380`/`:430-449`/`:450-491`,
`dbscan.hpp:32`/`:74`/`:88`/`:103`/`:117`, `dbscan.pyx:300`/`:371-372`,
`dbscan.cuh:34`/`:101`/`:122`/`:197`, `runner.cuh:59`/`:169-177`/`:245-246`/
`:281`/`:384`/`:410-416`/`:419-442`, `adj_to_csr.cuh:35`/`:67`/`:75`/`:94-101`/`:157-166`,
`csr.cuh:61-63`/`:72`/`:133`/`:149`, `epsilon_neighborhood.cuh:93-116`/`:218`,
`pca.cuh:74-83`/`:104-139`/`:136`/`:138`/`:160`, `tsvd.cuh:122`/`:125`/`:151-176`,
`pca.pyx:392-404`/`:547`, `math.cuh:367`/`:378`/`:400-409`,
`raft/cpp/tests/matrix/math.cu:176`, `linear_regression.pyx:309`/`:336-343`/`:390-394`,
`ols.cuh:67`/`:112-113`, `kmeans.pyx:98-129`/`:218-219`/`:229-231`,
`eig.cuh:108-109`/`:276`/`:310-311`, `l2_exp.cuh:124-135`/`:132-134`.

**Ambiguous but not wrong (NOT FIXED, low value):** a run of bare `` `:NNN` ``
citations in `cluster/gbdt/cluster/detail/kmeans.mojo` and
`kmeans_common.mojo` follow a `kmeans.hpp` citation but mean
`detail/kmeans.cuh`. Every value is correct against `detail/kmeans.cuh`; only
the implied file is ambiguous to a reader.

**One more false doc sentence corrected:** `pca.mojo` recommended
`nn.argsort.argsort` as the device replacement for its host selection sort,
citing `archive/reference/VENDOR_LIBRARIES.md`'s AVAILABLE row. Per this round's addenda
`nn.argsort[target="gpu"]` is non-monotone above 256 elements — i.e. wrong at
exactly the `n_cols` this note is about, silently. The recommendation is
deleted and the reason recorded.

---

## 3. WHAT I CHANGED, file by file

Docstrings, comments and TSV rows only. No kernel body, no launch geometry, no
algorithm, nothing under `neighbors/`, `core/`, `bench/`, `gbdt/`,
`mojo_only/`, or any root doc.

1. **`dbscan/gbdt/dbscan/runner.mojo`** — DEVIATION 35 rewritten: the claim
   that cuML's RBC restrictions "cost us nothing" is deleted and replaced with
   the `constexpr` downgrade at `runner.cuh:143-150` and the index-build guard
   at `:235`, stating that their dispatch never reaches RBC at int32 labels.
   The inline fallback list's "the first two cannot fire here" is deleted for
   the same reason. Query-view citations `algo.cuh:131`/`:146` -> `:132`/`:143`/`:161`;
   `:137-153` -> `:137-144`; `:195-200` -> `:194-200`.
2. **`dbscan/gbdt/dbscan/dbscan.mojo`** — "brute force" removed from the
   PORT OF scope line (the default is RBC now); their fixed `algo_vd`/
   `algo_adj`/`algo_ccl` recorded from `dbscan.cuh:118-122` with the note that
   `algo_ccl = 2` makes `final_relabel` non-optional; D2 written up on
   `compute_batch_size` from `dbscan.cuh:71`; memory-estimate citation
   `:147-151` -> `:157-158`.
3. **`dbscan/gbdt/sparse/detail/csr.mojo`** — `runner.cuh:365` -> `:384-386`.
4. **`decomposition/gbdt/linalg/detail/pca.mojo`** — new section answering
   the DQ-vs-Jacobi question from `tsvd.cuh:98-126` (nothing downstream of the
   branch differs); new section on `set_neg_zero` (D4) from `pca.cuh:136` +
   `math.cuh:86-95`; sign-flip section extended with the fact that
   `pcaFitTransform` is unreachable from Python (`pca.pyx:582-588`) while
   tSVD's is not (`tsvd.pyx:414`); the live `sign_flip_components` /
   `detail/pca.cuh:189` fabrication deleted; `eig.cuh:79-82` -> `:149-151`;
   the `nn.argsort` recommendation deleted; tol/sweeps provenance pinned to
   `pca.pyx:356-358,412-413` with the C++ `tol = 0.0` contrast from
   `params.hpp:44-45`.
5. **`decomposition/gbdt/linalg/detail/tsvd.mojo`** — D5 written up from
   `tsvd.cuh:190-238` and `:272-276`, plus the `set_neg_zero` asymmetry.
6. **`decomposition/README.md`** — `detail/eig.cuh:79-82` -> `:149-151`.
7. **`glm/gbdt/glm/ols.mojo`** — D6 written up from
   `linear_regression.pyx:309` and `preprocess.cuh:98,100,115,117,148-176`,
   naming `ols.cuh:156` as the branch we do mirror; `:75-76` -> `:74-75`,
   `:120-125` -> `:116-126`, `:122-124` -> `:123-125`, `:98-110` -> `:99-110`.
8. **`glm/gbdt/linalg/detail/lstsq.mojo`** — RAFT commit `9aa17e5`
   (not a valid object) -> `661a3b8`; `ols.cuh:105` -> `:120`;
   `eig.cuh:79-82` -> `:149-151` with `lstsq.cuh:315` naming the call site.
9. **`cluster/gbdt/distance/fused_distance_nn/simt_kernel.mojo`** — D8:
   `helper_structs.cuh:28-33` -> `:39-44`, and the tie-break re-attributed to
   `raft::argmin_op` (`raft/core/operators.hpp:198-205`, reached at
   `kmeans_common.cuh:474-489`) with a note that `MinAndDistanceReduceOpImpl`
   at `:47-97` is value-only.
10. **`cluster/mojo_only/reduce_by_key.mojo`** — `:300` -> `:287`, `:292` -> `:280`.
11. **`cluster/gbdt/cluster/detail/kmeans.mojo`** — `:270-288` -> `:272-288`.
12. **`cluster/UNPORTED.tsv`** — `checkWeight` row rewritten (D3).
13. **`dbscan/UNPORTED.tsv`** — `core_indices` row rewritten (D7).
14. **`glm/UNPORTED.tsv`** — `preprocess.cuh` row rewritten (D6).

---

## 4. FINDINGS THAT NEED CODE (written up, not made)

**F1 — DBSCAN batch policy ignores `eps_nn_method`. DONE**
(LANE_dbscan-batching_2026-08-19): `eps_nn_method` added in their parameter
order, clamp wrapped in `if eps_nn_method != EPS_NN_RBC:` mirroring
`dbscan.cuh:71`, threaded from `dbscan_fit_impl`. The hold-back concern was
empty: the clamp never bound a default-budget fit on this device (at n =
50,000 the budget gives 40,669 rows against a clamp of 42,949, and a clamped
batch costs ~5*MAX_LABEL ~ 10.7 GB at every n, above the ~10.2 GB default
budget), so no recorded timing's batch count moves.

**F2 — `checkWeight` is not ported and is on cuVS's default path.**
`cluster/gbdt/cluster/detail/kmeans.mojo::kmeans_fit_main`, immediately
before the `n_init` loop. Sum `weights` on the device; if the sum differs from
`n_samples`, scale every weight by `n_samples / sum`. Mirror of
`detail/kmeans.cuh:872` -> `kmeans_common.cuh:136-173`. Effect: our reported
inertia stops being off by `sum/n_samples` for any weight vector that does not
already sum to `n_samples`. No effect on centroids.

**F3 — `set_neg_zero` on the PCA singular values.**
`decomposition/gbdt/linalg/detail/pca.mojo::eig_and_truncate`, the
`singular_vals.append(sqrt(lam * Float64(singular_scale)))` line. Mirror
`math.cuh:86-95`: `0.0 if lam < 0.0 else sqrt(lam * scale)`. It must be
conditional on the caller — `pcaFit` passes `true` (`pca.cuh:136`), `tsvdFit`
passes nothing (`tsvd.cuh:237`) — so the flag has to arrive alongside
`singular_scale`. Effect: a NaN becomes a 0 on rank-deficient input.

**F4 — tSVD's `explained_var` is a different quantity from ours.**
Their value requires the transform, so it cannot be produced from
`eig_and_truncate` at all: `tsvdFitTransform` computes
`vars(trans_input)` after the sign flip (`tsvd.cuh:272-276`). Two honest
options: (a) port `tsvdTransform` + `signFlip` and compute it their way, or
(b) have `tsvd_fit` return only `components` and `singular_vals`, which is
exactly what `tsvdFit` returns (`tsvd.cuh:190-238`), and stop populating three
fields their function does not have. (b) is the smaller change and the more
faithful one.

**F5 — `fit_intercept` is cuML's Python default and we refuse it.**
`glm/gbdt/glm/ols.mojo`. Needs `preProcessData`/`postProcessData` from
`preprocess.cuh`; the non-`normalize`, no-`sample_weight` arm is four steps:
column means of X (`:98`), mean-center X (`:100`), mean of y (`:115`),
mean-center y (`:117`); then after the solve, `intercept = mean(y) - mu_X . coef`
(`:148-161`) and un-center both in place (`:163-176`). `core/column_stats.mojo`
already has the mean and shift kernels. This is the single largest remaining
gap between what a cuML Python user gets and what this repo gives them.

---

## 5. PROPOSED ROWS

`glm/UNPORTED.tsv` (already applied to the section file; repeated for the
root map if it duplicates them):

```
cuml glm/preprocess.cuh	NOT PORTED, AND fit_intercept IS ON THEIR DEFAULT PATH	LinearRegression(fit_intercept=True) is cuML's PYTHON DEFAULT (linear_regression.pyx:309), so their default fit is preProcessData -> lstsqEig -> postProcessData, not lstsqEig alone
```

New rows proposed for `decomposition/UNPORTED.tsv`:

```
cuml cpp/src/pca/pca.cuh::seqRoot set_neg_zero	NOT PORTED, AND IT IS ON THEIR DEFAULT PATH	pcaFit passes set_neg_zero=true (pca.cuh:136) so a negative eigenvalue becomes a ZERO singular value (math.cuh:86-95). Ours takes sqrt of it and returns NaN. Reachable on any rank-deficient or badly scaled design. tsvdFit passes no flag (tsvd.cuh:237), so the clamp is PCA-only
```

New row proposed for `dbscan/UNPORTED.tsv`:

```
cuml dbscan/dbscan.cuh::compute_batch_size eps_nn_method	DIVERGENT	theirs gates the batch_size <= MAX_LABEL/n_rows clamp on eps_nn_method != RBC (dbscan.cuh:71) because the RBC arm never materializes the dense N x batch_size adjacency. Ours has no eps_nn_method parameter and clamps unconditionally, so RBC fits batch more finely than cuML's would. Slowdown, not a wrong answer
```

No `PORTED_MAP.tsv` rows are proposed: this lane ported nothing.

---

## 6. PROPOSED archive/reference/PORTING.md DEVIATION ENTRIES (numbered from 30; renumber)

**30. DBSCAN defaults to RBC where cuML's dispatch cannot reach RBC at all.**
Theirs: `eps_nn_method` defaults to `BRUTE_FORCE` (`dbscan.hpp:74`), and even
when a caller asks for RBC it is disabled `constexpr` for `Index_ == int32_t`
(`runner.cuh:143-150`) with the index built only for `float && int64_t`
(`:235`). Ours: `EPS_NN_RBC` is the default and runs, at int32 labels.
Reason: measured 27.5x at n = 200,000 on an M4 with the arms interleaved, and
`check_dbscan_rbc_matches_brute` shows the two labellings identical point for
point. This is a deviation with a measured reason and **not** a case of
following their dispatch; the file says so.

**31. PCA/tSVD return NaN where cuML returns 0 for a negative eigenvalue.**
Theirs: `seqRoot(..., set_neg_zero=true)` on the PCA path (`pca.cuh:136`,
`math.cuh:86-95`). Ours: unconditional `sqrt`. Reason: none — this is an
unfixed gap, recorded so it is counted. See lane finding F3.

**32. k-means does not normalize sample weights.** Theirs: `checkWeight`
rescales the weight vector to sum to `n_samples` on every fit
(`detail/kmeans.cuh:872`, `kmeans_common.cuh:160-172`). Ours: takes them as
given. Consequence: reported inertia differs by `sum/n_samples`; centroids do
not. Reason: none — unfixed gap. See lane finding F2.

---

## 7. FALSE DOC SENTENCES IN FILES I MAY NOT EDIT

- **`archive/reference/VENDOR_LIBRARIES.md`** records `nn.argsort.argsort` as AVAILABLE and
  device-capable. Per this round's addenda it is non-monotone above 256
  elements and returns a well-formed but wrong permutation. `pca.mojo` had
  taken that row at face value and recommended it as a replacement for its
  host sort; I deleted the recommendation there, but the source row is not
  mine to edit.
- **`bench/results/VENDOR_PATH_2026-08-19.md`** carries the stale
  `linalg.gemv.gemv` sentence that `glm/.../lstsq.mojo` already flags as
  wrong (the GPU symbol is `gemv_gpu`). Not mine.
- **`UNWIRED.md` / root `README.md`**: not audited by this lane. If either
  describes DBSCAN as brute-force-by-default it is now false — the default is
  RBC as of this round.

---

## 8. BUILD AND CHECK EVIDENCE

All six section entry points built through `tools/with_build_lock.sh`; only
pre-existing warnings, no errors.

```
mojo build -I . decomposition/pca_main.mojo      -> warnings only
mojo build -I . decomposition/pca_wide_main.mojo -> warnings only
mojo build -I . decomposition/jacobi_main.mojo   -> warnings only
mojo build -I . glm/ols_main.mojo                -> warnings only
mojo build -I . dbscan/dbscan_main.mojo          -> warnings only
mojo build -I . cluster/kmeans_main.mojo         -> warnings only
```

Checks, all passing:

```
pca      check_covariance_is_symmetric / check_pca_fit / check_pca_invariants /
         check_input_restored / check_tsvd_against_pca            5/5 OK
ols      check_ols_exact / check_ols_scale_invariant /
         check_ols_beats_truth_on_noise / check_ols_dispatch_guard 4/4 OK
dbscan   check_fused_eps_agrees_with_materialized / check_dbscan /
         check_dbscan_eps_sensitivity / check_exclusive_scan_beyond_the_old_cap /
         check_dbscan_batching_agrees / check_dbscan_rbc_matches_brute 6/6 OK
kmeans   check_reach_by_sabotage / check_kmeans_fit /
         check_device_inclusive_scan / check_kmeans_plus_plus_init /
         check_fused_reduction_across_lanes                        5/5 OK
```

**SABOTAGE — reach evidence for the D1 finding.** D1 is only worth reporting
if the RBC arm is the code that actually runs at the shipped default. Scaled
the RBC query radius by 0.6 at `runner.mojo:267` (`eps_radius = Float32(eps) *
0.6`), rebuilt, ran:

```
check_dbscan                  OK   (eps=2; 0.6*2 = 1.2 still separates the blobs)
check_dbscan_eps_sensitivity  FAIL "at eps = 12 the three blobs did NOT merge;
                                    point 200 has a different label"
```

That is the predicted shape: a smaller radius cannot merge the blobs at
eps = 12 but is still large enough to keep them separated at eps = 2 — a
window big enough to move the answer, small enough not to destroy the
property. Sabotage reverted from a scratchpad copy, rebuilt, all 6 dbscan
checks OK again (output above).

---

## 9. WHAT I DID NOT DO, AND WHY

- **No code change for F1-F5.** All five are arithmetic or structural and this
  lane is read-mostly by brief; F1 in particular would move every DBSCAN
  timing taken this round, and the RBC default it interacts with was set by a
  concurrent lane.
- **Did not audit `neighbors/`.** Another lane owns it, and the k-NN dispatch
  defect it was named for was already found and fixed in round 1.
- **Did not audit `core/`, `gbdt/`, `mojo_only/`, `bench/` or any root doc.**
  Out of scope; peer session holds several of them.
- **Did not run timing benchmarks.** The 27.5x RBC table quoted in the DBSCAN
  deviation is a prior lane's measurement, restated, not re-measured.
- **Did not re-derive the `nn.argsort` 256-element bug.** Taken from this
  round's addenda as given, per "do not rediscover these".
- **Did not fix the ambiguous bare `` `:NNN` `` runs** in
  `cluster/gbdt/cluster/detail/kmeans.mojo` and `kmeans_common.mojo`. Every
  value is correct against `detail/kmeans.cuh`; only the implied filename is
  ambiguous, and rewriting ~15 comments in a file a peer lane is editing was
  the wrong trade.
- **`cluster/predict`, `fit_predict` and `cluster_cost` dispatch: NOT CHECKED**
  past confirming they reach the same `minClusterAndDistanceCompute` as `fit`.
