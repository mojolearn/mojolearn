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
findings are in `DERIVATION_MAP.tsv`, `NOT_IMPLEMENTED.tsv` and section 3 here.

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

  * **ALL NINE SABOTAGE ARMS HAVE NOW RUN**, one build each, 2026-08-23,
    IDENTICAL, one M4. **Seven bite.** Section 6 carries the measured
    table: which check each arm failed, which stage moved first, how many
    stages and cells moved, and WHICH FIXTURE reached it. Three of the
    seven live arms are carried by ONE FIXTURE EACH, which is the finding
    to read before any of the green ones.
  * `check_spectral_blobs_separate` is DIAGNOSED and fixed, and the four
    checks it used to block now run. **ALL 18 CHECKS PASS**, under
    IDENTICAL **and under FAST**, on one M4. The verdict is section 3.05
    and it went against the CHECK, not the port.
  * **FAST is RECORDED, never asserted.** Under FAST the identity clauses
    print `RECORDED [FAST] agreement on this column (no identity claim
    under FAST)` instead of raising, per the metrics lane's leg-11 lesson.
    The device and oracle arms did agree bitwise under FAST on this Apple
    column, which is an OBSERVATION about this column and not a property
    of the profile: FAST is unversioned and makes no identity claim.

## 1. Path mirroring

Three roots, each with its constant prefix dropped:

    raft   cpp/include/raft/   ->  spectral/derived/
    cuml   cpp/src/            ->  spectral/derived/
    cuvs   cpp/src/            ->  spectral/derived/cuvs/

so `raft/sparse/solver/detail/lanczos.cuh` is
`spectral/derived/sparse/solver/detail/lanczos.mojo` and can be diffed
against it side by side. The cuVS root keeps its `cuvs/` component
because two of the three upstreams have a file called `spectral.cuh` and
they are different algorithms (`NOT_IMPLEMENTED.tsv` says which).

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

### 3.05 THE BLOBS VERDICT: the check was wrong, and the reason is a real
### property of the algorithm being mirrored

`check_spectral_blobs_separate` failed 13 of 144 for four rounds. Two
hypotheses were on the table with opposite consequences. **The check was
wrong**, and the evidence is structural rather than a judgement call:

  1. **The kNN graph of three well-separated blobs at `n_neighbors = 10`
     has EXACTLY THREE CONNECTED COMPONENTS and zero cross-blob edges.**
     Union-find over the graph's own index arrays, asserted in
     `check_spectral_disconnected_graph_records_the_limit`. The blob
     centers are 34.7 to 43.9 apart with clouds of radius under 2, and
     each blob holds 48 points, so a 10-neighbor search never leaves its
     own blob. The graph does not become connected until `k = 49`.
  2. **On RAFT's actual normalized Laplacian that gives eigenvalue ZERO
     WITH MULTIPLICITY THREE.** Note RAFT normalizes by `sqrt(diag(L))`,
     NOT by `sqrt(degree)` (`laplacian.cuh:267-270`), then forces the
     diagonal to `1.0` (`:276`). An exact float64 eigendecomposition of
     that matrix gives `0, 0, 0, 0.16515296, 0.18516985, ...`.
  3. **A single-vector Lanczos cannot return three copies of a triple
     eigenvalue.** `K(A, v0)` contains only the projection of `v0` onto
     each eigenspace, so it holds ONE direction in a three-dimensional
     null space, and RAFT reorthogonalizes fully at every step
     (`lanczos.cuh:343-369`), removing the rounding noise that would let
     the other copies re-emerge. The solver returns the degenerate value
     once and fills the remaining slots with HIGHER eigenvectors, which
     are not constant on a component and do not separate the blobs.
  4. **Measured, and the two arms disagree in exactly the way the theory
     predicts.** The Float32 arm returns two near-zeros plus the spurious
     `0.16515295`; the Float64 arm returns ONE near-zero plus `0.16515295`
     and `0.18516985`. Different counts, because which copies re-emerge is
     precisely what rounding decides. The device and oracle Float32 arms
     agree with each other BIT FOR BIT throughout, so the identity claim
     was never in question.

So the port was faithful and the check was asserting a property of the
EXACT eigendecomposition rather than of the algorithm. **This lane's own
contract already said so** (section 3: a `k`-column embedding of a
`c`-component graph with `k <= c` sits entirely inside a degenerate
subspace) and the check had been written without honoring the clause.

THE FIX, in two parts:
  * `check_spectral_blobs_separate` now runs at `n_neighbors = 52`. Not
    arbitrary: the graph connects at 49, and 52 is the first value that
    also leaves a comfortable gap, with L's three smallest eigenvalues
    `0, 0.0463, 0.1208`, all SIMPLE. It passes 144 of 144.
    `check_spectral_clustering_labels` moved for the same reason and
    recovers the planted partition exactly.
  * `check_spectral_disconnected_graph_records_the_limit` keeps the
    `n_neighbors = 10` case and turns it into a regression test on the
    ALGORITHM: it ASSERTS the component count, which is structural, exact
    and vendor-independent, and only RECORDS how many near-zeros came
    back, because asserting that number would be asserting our own tally
    of a rounding accident. It also stands as a guard against a future
    author "fixing" this with a block Lanczos, which would be a
    reinvention rather than a port.

### 3.1 What checks out

**`derived/sparse/solver/detail/lanczos.mojo` is the strong file.** Every
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
  7. **Two upstream oddities NOT ported, recorded in `NOT_IMPLEMENTED.tsv`**: a
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
     `NOT_IMPLEMENTED.tsv` keeps the mechanism and drops the alarm.
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

`derived/cuvs/cluster/detail/spectral.mojo` calls
`cluster/derived/cluster/kmeans.mojo::fit_predict` with
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

All nine are build defines, none on by default. **All nine RAN**, one
build each, 2026-08-23, IDENTICAL, one M4. Seven bite.

| arm | verdict | which check, and which FIXTURE reached it | stages moved | cells moved |
|---|---|---|---|---|
| `SIGN_FLIP` | **BITES** | `device == oracle`, **`hashed` ONLY** | 2/119, first `spectral.ritz.vectors` | 144/144 |
| `LAPLACIAN_SEAM` | **BITES** | `device == oracle`, **`hashed` ONLY** | 101/119, first `spectral.L.vals` | 143/144 |
| `SPMV_ROTATE` | **BITES** | `device == oracle`, **`hashed_unnorm` (n=300) ONLY** | 205/230, first `spectral.lanczos.step0003.beta` | 735/900 |
| `NCV` | **BITES** | `device == oracle`, **`tiny_ncv_clamp` (n=22) ONLY** | **STRUCTURAL, 85 vs 77** | 64/66 |
| `SWEEP_CAP` | **BITES** | `check_tsolve_against_float64_jacobi` | n/a, host solver | worst eigenvalue error 6.9e-4 against a 2e-6 budget |
| `ROTATE_UNFUSED` | **BITES** | `check_spectral_ring_exact` | n/a, closed form | breaks the ring's DEGENERATE PAIR: 5 distinct Ritz values where `{0, l1, l1, l2, l2}` is required |
| `TIE_REVERSE` | **BITES** | `check_tsolve_tie_order_is_stable` | n/a, host | eigenvector column 0 is no longer `e_1` |
| `MAXITER` | **BITES** | `device == oracle`, **ALL SIX fixtures** | **1 stage, `spectral.lanczos.config`** | **0 cells** |
| `STD_SQRT` | **INERT** | nothing | 0 | 0. **A MEASUREMENT**: this host's `std.math.sqrt` is correctly rounded on every value this lane feeds it |


**REACH IS PER FIXTURE, and three of the seven live arms are carried by ONE
FIXTURE EACH.** That is the most important thing this run produced, because
it means deleting one fixture would silently disarm a clause:

  * `SIGN_FLIP` is inert on `path`, `ring`, `hashed_unnorm` and `blobs`.
    MECHANISM: after a restart the projected matrix built from `-beta_k` is
    exactly `D T D` with `D = diag(-1 for i < k, +1 otherwise)`, so its
    eigenvectors differ from the unsabotaged ones by a sign pattern that
    `pin_column_signs` then re-canonicalizes. Whether the final flip
    survives to the embedding depends on the fixture, and only `hashed`
    keeps it. **The device-side instrument for DEVIATION 770 rests on one
    fixture.** (The sign rule also has two host instruments that this arm
    does not target, so the clause is not undefended, but the DEVICE path
    is.)
  * `LAPLACIAN_SEAM` is unreached on four. `ring` and `hashed_unnorm` run
    `norm_laplacian = false`, so seam L6 never executes at all; `path` and
    `blobs` carry weights that are exactly `1.0` or `0.5`, and multiplying
    by a power of two is exact, so the reassociation moves no bit. Only a
    fixture with NON-POWER-OF-TWO weights AND a normalized Laplacian
    reaches it, which is exactly what `hashed_graph_fixture`'s docstring
    claims it exists for.
  * `SPMV_ROTATE` is unreached on four FOR A BORING REASON: at
    `LANCZOS_TPB = 256` the graphs with n = 22, 48, 64 and 144 all fit in
    ONE BLOCK, where `blockIdx.x` is 0 and the rotation is the identity.
    Only n = 300 spans two blocks. A launch-invariance arm that only ever
    runs one block tests nothing.
  * `NCV` was inert on all five original fixtures and is now carried by a
    sixth added for it. `ncv = min(n - k, max(2k + 1, 20))` takes the
    `n - k` branch only when `n < k + 20`, and every fixture here was
    larger, so the clamp never bound. `tiny_ncv_clamp` is n=22, k=4, where
    it is `min(18, 20) = 18`; the arm then takes `min(20, 22) = 20` and the
    two runs take a DIFFERENT NUMBER OF LANCZOS STEPS, which is the
    structural shape a changed bound has.
  * `MAXITER` WAS a reach failure too, and was fixed WITHOUT the heavy
    fixture it was thought to need. Recording `spectral.lanczos.config`
    means a changed bound is caught AT THE BOUND: the arm now moves ONE
    stage and ZERO CELLS on all six fixtures. An arm that moves no number
    at all is still a real catch when the DECISION it changes is itself a
    recorded stage.
  * The one remaining inert arm, `STD_SQRT`, is a MEASUREMENT and not a
    failure: this host's `std.math.sqrt` is correctly rounded on every
    value this lane feeds it, which is exactly what that arm exists to
    find out per host.

**TWO PREDICTIONS IN THIS DOCUMENT WERE WRONG, BOTH IN THE LANE'S FAVOR,
and they are corrected rather than quietly updated.**

  1. `SWEEP_CAP` was predicted to fail `check_spectral_path_exact`. It
     fails EARLIER, at `check_tsolve_against_float64_jacobi`, which is a
     tighter and more direct independent reference.
  2. `ROTATE_UNFUSED` was declared **EXPECTED REACHED BUT INERT, AND THAT
     IS A HOLE**. IT BITES. It fails `check_spectral_ring_exact` by
     BREAKING THE RING'S DEGENERATE PAIR: with NR's four roundings the
     solver returns five distinct Ritz values where the doubled spectrum
     requires `{0, l1, l1, l2, l2}`. So seam J4 does have an instrument,
     and it is the degeneracy this lane had been treating only as a hazard
     to be RECORDED. A degenerate spectrum turns out to be the most
     sensitive test surface in the lane, not the least trustworthy one.


**The VACUOUS negative controls did not fire in any of the ten runs**,
which is the correct outcome and also means the controls are themselves
unexercised. A self-test that forces one to fire is OWED.

## 6.5 MORE HASHES: the decisions this pipeline makes, and which are recorded

Andrew asked whether there are more hashes to take. There were. DEVIATION
778's tie rule was a DECISION with no instrument until one was built for
it, so the whole pipeline was re-read for others.

**ADDED THIS ROUND**, both host-side integers, no new kernel and no new
cross-thread fold:

  * `spectral.lanczos.config` -- `n`, `k`, `ncv`, `max_iterations`,
    `which`, as one integer stage. The solver's SHAPE is chosen before any
    float moves and was invisible in every other stage: two runs could
    agree on every alpha and beta and still have been asked different
    questions. This is what turned `MAXITER` from inert into a live arm.
  * `spectral.lanczos.restartNNNN.sweeps` -- how many Jacobi sweeps the
    projected solve took. The sweep cap is DEVIATION 780's other surviving
    clause and this is the only stage that can see it directly.

**ALREADY RECORDED, checked rather than assumed**: how many Lanczos steps
ran and which criterion stopped the loop (the `converged_restarts_iter`
triple makes `res <= tol` against `iter >= maxIter` derivable); whether a
restart broke down (it raises by name); whether `kernel_normalize`'s
`beta == 0` branch fired (`beta` is a recorded per-step scalar, so the
branch is derivable from it); whether `coo_symmetrize` found a transpose
(0.5 against 1.0 in the recorded `spectral.W.vals`); how many entries the
zero-compaction dropped (implied by the recorded `W.rows` length).

**HANDED OVER, NOT ADDED, and the reason is the same for both.** Each
needs a COUNT over a device buffer, which is a cross-thread reduction and
therefore a NEW fold whose own order would have to be pinned before it
could be trusted. **An instrument that is itself an identity hazard is not
an improvement**, so these are named for the orchestrator rather than
taken unilaterally:

  * **whether the `u` clamp at `1e-7` fired, and on how many entries.**
    `u` is never a card stage, so this clamp is the one seam in the
    Lanczos with no instrument at all.
  * **whether `zero_to_one_functor` substituted a `1.0` for a zero
    degree.** `spectral.diag` records the post-substitution value, so a
    `1.0` there is currently ambiguous between a genuine unit degree and a
    substituted zero.

## 7. What is owed, and every item needs a compile slot

  1. ~~Run the eight sabotage arms.~~ **DONE**, and a ninth added for the
     tie rule. **EIGHT of nine bite**; the one that does not is a
     measurement. Both follow-ups from that run are closed too: the
     VACUOUS self-test landed, and `MAXITER`'s reach failure was fixed by
     recording the decision rather than by building a heavy fixture.
  2. ~~Diagnose `check_spectral_blobs_separate`.~~ **DONE, section 3.05.
     The CHECK was wrong, the port was faithful.**
  3. ~~Run the four checks that never executed.~~ **DONE. ALL 18 CHECKS
     PASS**, including rung-2 clustering, launch invariance, the card
     emission and the five device refusals.
  4. ~~A CERTIFICATE gate for `symmetric_eig_host` so seam J4 and the
     sweep cap have teeth.~~ **NO LONGER BLOCKING**: the sweep cap is
     caught by `check_tsolve_against_float64_jacobi`, seam J4 by
     `check_spectral_ring_exact`, and the tie rule by the new
     `check_tsolve_tie_order_is_stable`. A certificate would still be
     cheaper and sharper than three indirect references, so it stays a
     nice-to-have rather than a hole.
  5. ~~A fixture that makes `max_iterations` bite.~~ **NO LONGER NEEDED,
     and it was the heavy item.** Recording `spectral.lanczos.config`
     catches a changed bound directly, so the arm bites on all six
     existing fixtures with zero cells moved.
  6. ~~Re-run everything under FAST and record, not assert.~~ **DONE.**
     All 18 pass under FAST; the identity clauses RECORD rather than
     assert, and the recorded outcome on this Apple column is agreement.
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
