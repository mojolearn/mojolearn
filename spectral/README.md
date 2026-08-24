# spectral: a port of cuVS spectral embedding and cuML spectral clustering

**Same strategy as `cluster/` and `gemm/`, three upstreams.** This section
mirrors the RAFT thick-restart Lanczos, the RAFT graph Laplacian, the cuVS
spectral embedding driver and the cuML surface above it, file for file, so
that "did we port this?" is answered by `ls` rather than by reading.

Read `IDENTICAL_SPECTRAL_CONTRACT.md` first. It is not a design note: it
is the part of the numerical plan without which nothing else here means
anything, and section 0 below says why.

## 0. Why this lane needed a contract before it needed a gate

A GEMM has one right answer up to rounding order. So does a Mamba block.
**An eigenproblem does not.** If `v` is a unit eigenvector of a symmetric
matrix then so is `-v`, and the two are equally correct and utterly
different downstream: the sign propagates into the Ritz vectors, into the
NEXT Lanczos pass (the Ritz vectors literally become the new basis at the
restart), and into the embedding a user reads. If two eigenvalues are
equal then every orthonormal basis of their shared subspace is equally
correct and there are infinitely many. cuSOLVER picks one, LAPACK picks
another, a Jacobi sweep picks a third, and none of them is wrong.

So **a bit-identity claim over an unpinned convention is not a claim**,
and RAFT pins nothing, because RAFT ships one backend and has never had
to. `IDENTICAL_SPECTRAL_CONTRACT.md` pins the sign rule (DEVIATION 770),
the ordering rule and the tie rule (DEVIATION 778), the restart and
reorthogonalization decisions, and every seam's fused-or-unfused choice.
It also states the limit honestly: the profile claims the same BITS on
every vendor inside a degenerate subspace, and does NOT claim that basis
is STABLE under perturbation, because it is not.

## THE STATE OF THIS LANE, ABOVE ANYTHING ELSE

**Read this before you read a single claim below.**

This section arrived as 4,536 lines of UNTRACKED work from a fan-out agent
that died before committing anything. Two earlier rounds died the same
way. It landed at `de4d388`, audited and renumbered, and the audit's
findings are in `PORTED_MAP.tsv`, `UNPORTED.tsv` and section 3 here.

**No gate in this lane is GREEN, and none may be cited as one.** What is
true, and the only thing that is:

  * Both binaries compiled clean once, on 2026-08-23, before this lane
    lost its compile slot to a box whose load average had hit 20.
  * On that one run, under IDENTICAL on one M4, ELEVEN of the fifteen
    checks passed, including the headline one:

        check_spectral_device_equals_oracle OK [IDENTICAL]: path, ring,
          hashed, hashed_unnorm(n=300), blobs(dataset path): every stage
          hash (L, diag, v0, each step's alpha/beta, each restart's Ritz
          values and residual, the Ritz vectors, the embedding) and every
          embedding cell bit for bit

  * **NO SABOTAGE HAS EVER BEEN RUN.** Not one. A green check whose path
    has not been sabotaged proves nothing, which is this repository's
    oldest standing rule, so the sentence above is EVIDENCE and not a
    gate. All eight arms exist in source; all eight are unrun.
  * The run stopped at `check_spectral_blobs_separate` (13 of 144 points
    nearer another blob's centroid in the 2-column embedding) and the four
    checks after it, including the rung-2 clustering one, never executed.
    That failure is UNDIAGNOSED. See "What is owed".

## 1. Path mirroring

Three roots, each with its constant prefix dropped:

    raft   cpp/include/raft/   ->  spectral/ported/
    cuml   cpp/src/            ->  spectral/ported/
    cuvs   cpp/src/            ->  spectral/ported/cuvs/

so `raft/sparse/solver/detail/lanczos.cuh` is
`spectral/ported/sparse/solver/detail/lanczos.mojo` and can be diffed
against it side by side. The cuVS root keeps its `cuvs/` component
because two of the three upstreams have a file called `spectral.cuh` and
they are different algorithms (`UNPORTED.tsv` says which).

## 2. THE PINS, and the one that is missing

| upstream | pin | checkout |
|---|---|---|
| RAFT | **v26.08.00** | `~/CascadeProjects/upstream/raft-v26.08.00` |
| cuML | **v26.08.00** | `~/CascadeProjects/upstream/cuml-v26.08.00` |
| cuVS | **v26.08.00**, `6ba2ce2` | `~/CascadeProjects/upstream/cuvs-v26.08.00` |

**`upstream/raft` and `upstream/cuml` are 25.08 checkouts and are NOT what
this lane was read against.** That is not a pedantic distinction. The two
raft trees are different files, not near-copies:

    sparse/solver/detail/lanczos.cuh    2176 lines at 25.08, 799 at 26.08
    sparse/linalg/detail/laplacian.cuh   160 lines at 25.08, 285 at 26.08
    sparse/matrix/detail/diagonal.cuh    165 lines at 25.08, 255 at 26.08

and 25.08's `laplacian.cuh` has **no COO overload of
`compute_graph_laplacian` and no `zero_to_one_functor` at all**, both of
which this lane ports. An audit against the wrong tree produces confident
nonsense.

`upstream/cuvs` is a 25.08 checkout and is likewise NOT what this lane
reads. **The cuVS 26.08 checkout landed on 2026-08-23** and settled five
open questions at once; what it settled, including one finding that went
against this lane, is section 3.0.

## 3.0 WHAT THE cuVS 26.08 CHECKOUT SETTLED, INCLUDING AGAINST US

For one round this README said the two files this lane's cuVS layer cites
did not exist and that every cuVS citation was unverified. That was an
artifact of reading 25.08. With `cuvs-v26.08.00` (`6ba2ce2`) in hand:

  * **Both files exist exactly where the headers said**, at
    `cpp/src/preprocessing/spectral/detail/spectral_embedding.cuh` (225
    lines) and `cpp/src/cluster/detail/spectral.cuh` (82 lines), and **18
    of this lane's 20 cuVS line citations are EXACT**. The two that were
    wrong were both `params` struct ranges and are corrected. The old
    finding 1 is WITHDRAWN in full.
  * **The Laplacian overload is COO, which is what this lane ported.**
    `spectral_embedding.cuh:127` and `:219` instantiate
    `create_laplacian<..., raft::device_coo_matrix<...>>` explicitly, and
    26.08 has no `coo_to_csr_matrix` at all. The old finding 8's self-loop
    rounding hazard DOES NOT ARISE and nothing is invalidated.
  * **AND THE ONE AGAINST US: DEVIATION 780 CLAIMED THREE BOUNDS THAT ARE
    VERBATIM UPSTREAM.** `max_iterations = 10 * n_samples` (`:64`), the
    `RAFT_EXPECTS(n_samples - n_components > 0)` (`:65-66`),
    `ncv = min(n_samples - n_components, max(2 * n_components + 1, 20))`
    (`:67`) and `config.tolerance = spectral_embedding_config.tolerance`
    (`:68`, and the field is real at
    `preprocessing/spectral_embedding.hpp:59`). Every one of those was
    recorded as OURS. The `n - k` clamp this README described as REPAIRING
    a hole in theirs IS THEIRS, message string included. **C1, C2 and C3
    are STRUCK. They were never deviations.** Reading the wrong tree does
    not only invent defects in our code, it invents ORIGINALITY we do not
    have, and claiming a deviation we did not make is exactly as bad as
    missing one.
  * What survives of DEVIATION 780 is two clauses, and both live in code
    that stands where a CLOSED vendor library does rather than in the
    mirrored driver: the host Jacobi's sweep cap of 60 that returns an
    unconverged basis instead of raising (C4), and this lane's own
    admissibility guard admitting `ncv == n` (C6), which the ported driver
    cannot even reach because their `ncv` is always below `n`.
  * The `NCV` and `MAXITER` sabotage arms keep their value and change
    their meaning. They no longer test a choice of ours; they inject cuVS
    **25.08**'s older spelling and so test that this lane mirrors the
    26.08 one.

## 3. THE AUDIT, symbol by symbol

### 3.1 What checks out

**`ported/sparse/solver/detail/lanczos.mojo` is the strong file.** Every
one of its roughly forty RAFT line citations was opened and confirmed:
`lanczos_aux` (:247-399), `lanczos_solve_ritz` (:128-245),
`lanczos_smallest` (:401-754), `lanczos_compute_eigenpairs` (:756-796),
the three kernels at :100-126, the two triangular kernels at :72-98, the
clamp constants at :374/:385-386/:388-389, the restart's every step from
:538 to :746. It is a MIRROR, not a reinvention: the loop structure, the
`(i - 1 + ncv) % ncv` wraparound, the order of norm-then-clamp-vector-
then-clamp-scalar, the `if (i >= end_idx - 1) break` placement, the
thick-restart's five copies, and the `iter += ncv - nEigVecs` accounting
are all theirs, line for line.

The RAFT Laplacian, diagonal, symmetrize, sort/filter/csr and COO files
check out too, and so do both cuML files. The one arithmetic detail worth
naming because it is easy to get wrong and was got RIGHT: theirs is
`row_scale * values[idx] * col_scale`, which in C++ is
`(row_scale * values) * col_scale`, two roundings in that order, and the
port keeps two roundings in that order (`diagonal.cuh:216`).

### 3.2 What was invented, missing or silently divergent

  1. ~~**The cuVS layer's citations point at files that do not exist.**~~
     **WITHDRAWN**, see section 3.0. The files exist and 18 of 20
     citations were exact. Struck rather than deleted, because a finding
     this lane got wrong is worth as much to the next reader as one it got
     right.
  2. ~~**`ncv`, `max_iterations` and a plumbed `tolerance` are OURS.**~~
     **WITHDRAWN, AND THIS IS THE ONE THAT WENT AGAINST US.** All three
     are verbatim `spectral_embedding.cuh:64-68`. C1, C2 and C3 of
     DEVIATION 780 are STRUCK; C4 and C6 survive. Section 3.0.
  3. **DEVIATION 777, a defect that stopped the kNN path dead.** The
     repeated-key refusal was spelled inside `coo_sort`, and the dataset
     path raised `repeated (row, col) pair (0, 0)` before a single bit was
     ever compared. `coo_symmetrize` writes into `2 * nnz` slots and
     **RAFT ITSELF zero-fills that output** (`symmetrize.cuh:205-209`), so
     the result legitimately carries a run of `(0, 0, 0.0)` padding, and
     cuVS SORTS THAT and only then compacts it with
     `coo_remove_scalar(0)`. A sort that refuses repeats cannot stand
     where theirs sorts. `coo_sort` is now their sort and nothing else;
     `refuse_repeated_keys` carries the refusal at the Laplacian's input,
     which is the one place a repeat is a tie-break theirs never defined.
  4. **Eight wrong upstream citations, all corrected in place** rather
     than left to mislead: the symmetrize kernel's extent (`:44-107`,
     truly `:44-115`) and the wrong `coo_symmetrize` OVERLOAD (`:119-`,
     truly the `raft::resources` one at `:168-234`); `sort.h:35-49` (that
     is `TupleComp`, `coo_sort` is `:62-70`); `filter.cuh:39-120` (the
     kernel is `:39-85` and cuVS calls the handle overload at `:197-250`);
     `lanczos_types.hpp:40-70` (truly `:39-68`); and four off-by-one to
     off-by-two ranges in the two cuML files.
  5. **`u -= V^T uu` is two roundings here and one call in theirs**
     (DEVIATION 779). Theirs is a single `gemv` with `alpha=-1, beta=1`
     free to fuse its epilogue; ours is a gemm-v1 contraction plus a
     separate subtraction, one extra rounding per coordinate. Deliberate,
     because the contraction belongs to `gemm.fp32.v1` which owns its own
     accumulator, but it had never been written down.
  6. **The Jacobi `ROTATE` is two fmas where NR's C is four roundings**
     (DEVIATION 781), and **the sweep cap of 60 returns silently** where
     NR uses 50 and raises. Both CHOSEN, neither previously recorded.
  7. **Two upstream oddities NOT ported, recorded in `UNPORTED.tsv`**: a
     dead `V_k_T` transpose theirs computes and never reads (`:644-647`),
     and transposed launch bounds at `:161-162` that happen to cover every
     row and move no bit.
  8. ~~**Which Laplacian overload is the path is UNSETTLED.**~~
     **WITHDRAWN: SETTLED, AND THIS LANE PORTED THE RIGHT ONE.**
     `spectral_embedding.cuh:127` and `:219` instantiate
     `create_laplacian<..., raft::device_coo_matrix<...>>` explicitly and
     26.08 has no `coo_to_csr_matrix`. The self-loop rounding hazard
     between the two overloads does not arise on this path. It is still
     worth knowing that the two overloads round differently, so
     `UNPORTED.tsv` keeps the mechanism and drops the alarm.
  9. **NEW, from the 26.08 tree: their dataset `fit_predict` hands
     `create_connectivity_graph` a PARTLY UNINITIALIZED params struct.**
     `cluster/detail/spectral.cuh:73-74` default-constructs
     `spectral_embedding::params` and sets only `n_neighbors`;
     `n_components`, `norm_laplacian` and `drop_first` are POD and are
     read by nobody on that path, so it is harmless in theirs. Ours passes
     a fully-initialized struct. Recorded because a reader diffing the two
     will see a difference that is not one, and because an upstream edit
     that made that function read one of those fields would be a live bug
     there and not here.

The classic failure the audit was told to hunt, one variable of theirs
split into two of ours, **was not found**. The one place it could have
hidden, `alpha_i` living as both a device scalar and a host copy across
`:315-330`, is a copy in theirs too and is spelled as one value here.

## 4. The k-means at the end is `cluster/`'s, not a second copy

`ported/cuvs/cluster/detail/spectral.mojo` calls
`cluster/ported/cluster/kmeans.mojo::fit_predict` with
`oversampling_factor = 0.0` (classic k-means++, their dispatch's arm) and
copies the device setup `cluster/estimator.mojo::kmeans_fit` performs.
`cluster/` is READ and IMPORTED and is never edited from this lane.

## 5. HAND-OFF: what belongs to a caller and is not performed here

  * `n_components + 1` when `drop_first` is set. cuML's Python does that
    arithmetic (`spectral_embedding.pyx:294`) and this lane's `transform`
    receives the already-incremented number, exactly as cuVS's does.
  * `n_components` defaulting to `n_clusters` when `None`, `n_neighbors=10`,
    `n_init=10`, `eigen_tol='auto'` meaning `1e-5`, and the refusals of
    `affinity` outside `{nearest_neighbors, precomputed}` and
    `assign_labels != 'kmeans'`.

## 6. The sabotages, and what each is honestly expected to do

All eight are build defines, none on by default, **none has been run**.
Three of them state in advance that they are expected NOT to fail a gate,
so that a pass cannot later be misread as a success.

| define | expected |
|---|---|
| `..._SIGN_FLIP` | FAIL device == oracle at the first Ritz-vector stage |
| `..._LAPLACIAN_SEAM` | FAIL at `spectral.L.vals` (reassociates seam L6, no fusion change) |
| `..._SPMV_ROTATE` | FAIL device == oracle (makes the matvec order a function of `blockIdx`) |
| `..._NCV` | FAIL, and STRUCTURALLY: a different stage count. **Mirror fidelity, not a choice of ours**: it reverts `ncv` to cuVS 25.08's spelling |
| `..._SWEEP_CAP` | FAIL `check_spectral_path_exact`. NOT device == oracle: the solver is SHARED by both arms |
| `..._MAXITER` | EXPECTED INERT. Every fixture converges long before either bound. **Mirror fidelity**: it reverts `max_iterations` to cuVS 25.08's literal `1000` |
| `..._ROTATE_UNFUSED` | EXPECTED REACHED BUT INERT, **and that is a hole**: seam J4 has no gate with teeth until a certificate lands |
| `..._STD_SQRT` | REPORT. Inert on a host with a correctly rounded `sqrt`; it exists to be measured per host |

## 7. What is owed, and every item needs a compile slot

  1. **Run the eight sabotage arms.** Nothing in this lane is gated until
     they run and each behaves as the table above predicts. An arm that
     surprises the table is a finding either way.
  2. **Diagnose `check_spectral_blobs_separate`.** 13 of 144 points land
     nearer another blob's centroid. Two hypotheses, untested: the kNN
     graph of three well-separated blobs has three connected components,
     so `L`'s three smallest eigenvalues are all zero and the whole
     2-column embedding sits inside a triple-degenerate subspace, in which
     case the check is asserting a convention that does not exist and the
     CHECK is wrong; or the embedding really is not separating and the
     PORT is wrong. Contract section 3's degeneracy clause predicts the
     first. Nothing may be concluded until it is measured.
  3. **Run the four checks that never executed**, including the rung-2
     clustering one.
  4. **A CERTIFICATE gate for `symmetric_eig_host`** so seam J4 and the
     sweep cap have teeth (`UNPORTED.tsv`).
  5. **A fixture that makes `max_iterations` bite**, so C2 is testable.
  6. **Re-run everything under FAST** and record, not assert.
  7. **The cross-vendor legs.** Nothing here has run anywhere but one M4,
     so no cross-vendor claim exists.
  8. ~~A cuVS 26.08 checkout.~~ **DONE**, 2026-08-23, `6ba2ce2`. It
     settled five questions at once and cost this lane three deviation
     clauses it should never have claimed. Section 3.0.

The pixi task line and the IDENTITY_PATHS row this lane needs are in the
lane report rather than applied here, because `pixi.toml` and
`IDENTITY_PATHS.md` belong to the orchestrator.

## 8. No performance number appears in this section, ever

By Andrew's order for this lane. The fixtures are deliberately small.
