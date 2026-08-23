# The unsupervised identity lane: k-means, k-NN, DBSCAN

Opened 2026-08-23, when Andrew asked whether bitwise identity was worth it
for the algorithms that are not GBDT, and answered yes for k-means and k-NN
("the cheapest closures you have left") and yes with more work for DBSCAN.

`IDENTITY_PATHS.md` is the ledger and stays the ledger: rows 19-26 are this
lane's and every claim below is a row there. This file is the LANE file --
what was found, what it cost, what is owed, and how to run it.

## The state before this round, in one number

`cluster/`, `neighbors/` and `dbscan/` contained **zero** identity
constructions. Not one `ftz`, not one `identical_mul_add`, not one pinned
fold, not one mode gate. `NUMERIC_IDENTICAL` compiled and changed nothing
there, so "the identical column" was a statement about `gbdt/` wearing a
repository-wide name.

## What the round found before it fixed anything

Three findings that were not the work, and matter more than most of it.

### 1. `check-ieee-arith`'s contraction verdict was an artifact (row 9)

The ledger said, and every downstream argument assumed, that **Apple's FAST
codegen does not contract**: *"fused 0 / unfused 1,046,394 of 2^20"*. Both
halves of that are true and the conclusion does not follow. Of those 2^20
hashed patterns, **zero** separate a fused `a*b+c` from an unfused one --
random exponents put the product and the addend far enough apart that both
spellings round identically -- and the counting arm tested `got == unfused`
FIRST, so all 1,046,394 ties were scored as evidence for UNFUSED. A backend
that contracts every multiply-add scores exactly the same.

The check now carries a BUILT-TO-SEPARATE arm: `a` and `b` with half-width
mantissas, `c = -fl(a*b)`, so an unfused evaluation is exactly `+0.0` and a
fused one is the product's rounding error. **Metal through MAX: FUSED on
1,629 of 1,629.** An isolated two-kernel probe agrees (a kernel containing
only `a*a + c` returns the fma bits).

Consequences, in order of importance:

- `identical_mul_add` is **bit-inert on Apple**, like `ftz` on an FTZ
  backend. This lane's sweep therefore moved no shipped bits, which was
  verified rather than assumed: the three suites print byte-identical
  output at `HEAD` and after (a detached worktree, same environment).
- The pin's value is the backend that does NOT contract by default. Row 9's
  REMEDY is unchanged; only its Apple sentence was wrong.
- Any argument in this repository resting on "Apple is unfused" has to be
  re-derived. That includes the sentence about IDENTICAL differing from
  FAST on Apple by design.

### 2. The library block-size row was feeding three float folds (row 21)

`lib_block_size_for` is documented SCHEDULING. For `K_LIB_ROW_NORM`,
`K_LIB_REDUCE_BY_KEY` and `K_LIB_PLUS_PLUS` the block size is the WIDTH OF A
FLOAT FOLD -- and for the third it also sets the chunk count of the float
scan that decides WHICH SAMPLE k-means++ draws. This is item 2 of the
ledger's own founding findings ("a scheduling row was feeding a float sum"),
one table over, and it is bit-inert TODAY only because every column happens
to carry 128. The table's docstring actively invites a vendor number to land
there "WITHOUT touching a kernel".

### 3. The fused k-NN arm moves a tie ON ONE DEVICE

`check_knn_fused_tie_set_is_geometry_invariant`, first run, FAST mode: with
40 identical queries, row 16 came back with a different neighbour than row
0. Same point, same index, same process. That is `updateSortedWarpQ`'s mutex
merge resolving an equidistant tie by arrival order, and it is intermittent
-- a later run showed the movement between query counts instead. Row 11 was
a documented refusal about cross-vendor reproducibility; it was also a
within-run one.

## What landed

| # | move | where |
|---|---|---|
| 500 | composite `(distance, index)` radix key: the tie class stops existing | `neighbors/mojo_only/select_radix_identical.mojo` |
| 501 | ranked output placement: slots come from the key, not from `atomicAdd` | same file |
| 502 | `grid_x = 1` and the ARM pinned to cuVS's own dispatch; 64-lane columns REFUSED | `fused_l2_knn.mojo`, `knn_brute_force.mojo` |
| 503 | k-means assignment: contraction + flush pins | `cluster/.../simt_kernel.mojo`, `unfused_distance_nn.mojo` |
| 504 | `pinned_block_sum`: a halving fold with no lane primitive in it | `core/pinned_reduce.mojo` |
| 505 | the tiled k-NN arm's distances off the vendor matmul | `neighbors/mojo_only/pinned_distance_tile.mojo` |
| 506 | DBSCAN's eps accumulators, brute-force AND ball-cover | `epsilon_neighborhood.mojo`, `ball_cover/common.mojo` |
| 507 | a truncated label propagation raises instead of returning | `csr.mojo`, `merge_labels.mojo` |
| 508 | the float-fold library rows pinned at the comptime accessor | `mojo_only/kernel_matrix.mojo` |
| 509 | under IDENTICAL, AUTO pins to the **TILED** arm on every column, superseding 502's pin to fused | `knn_brute_force.mojo` |
| 510 | `K_LIB_COLUMN_STATS` added to the float-fold classifier (the fourth row 508 missed) | `mojo_only/kernel_matrix.mojo` |
| 512 | AUTO may not choose an arm this column does not have, in EITHER mode | `knn_brute_force.mojo` |
| 513 | DBSCAN's batch-count check reads the real batch count and gates that its budgets moved it | `dbscan_identity_check.mojo`, `dbscan/estimator.mojo` |
| 514 | the mode flip holds the shared build lock for its whole window | `tools/with_identical_mode.sh` |

(511 is reserved for `jacobi_eigh_device.mojo`'s two Float32 `block.sum`
folds, named below and not taken; 510's kernel half was landed by the
decomposition lane the same day.)

## The second round, 2026-08-23: what the AMD column found before AMD did

The lane's first owed item was "a second vendor". Half of it turns out not
to need one, and that half found three defects.

`TARGET_COLUMN` is a COMPTIME choice (`mojo_only/kernel_matrix.mojo`), so
`-D MOJOLEARN_COLUMN_AMD=1` compiles this source against AMD's block sizes,
64-wide lane width, LDS budget and 110-core occupancy **on the attached
device**. The arithmetic stays Metal's -- contraction, denormals and device
transcendentals (rows 9, 10, 12) cannot fail this way -- but everything
that is a source-level response to a vendor CONSTANT is exercised, and that
is most of rows 8, 21, 22 and 23.

### 1. The identical column pinned k-NN to an arm that does not exist on AMD

First AMD-column build of `knn_identity_check`:

    Unhandled exception: fusedL2kNN: the FAISS warp queue is a 32-lane
    bitonic network and this target column's lane width is 64.

That refusal is row 23's and it is correct. What was wrong is what reached
it. DEVIATION 502 had pinned AUTO to the FUSED arm under IDENTICAL, so on a
64-wide wavefront **the identical build of k-NN did not lose its tie
guarantee, it raised** -- the mode whose entire purpose is one answer on
three vendors was pinned to a kernel that exists on two. DEVIATION 509
pins AUTO to the TILED arm on every column instead: the arm whose tie set
is NAMED (lowest index, by the composite key), whose distances come from a
per-cell contraction rather than a vendor matmul, and which contains no
warp primitive at all.

DEVIATION 512 is the same hole in the SHIPPED mode and is a plain
correctness bug rather than an identity one: under FAST, AUTO chose the
fused arm by geometry, so on AMD every `k <= 64` row-major query at a
`grid_x == 1` shape -- the common case -- raised for a caller who never
named an arm. AUTO now takes tiled wherever the fused arm cannot exist.
An EXPLICIT `KNN_METHOD_FUSED` still refuses, which is the honest answer to
someone who asked for that arm by name.

### 2. The DBSCAN batch-count check had never checked a batch count

`dbscan_fit`'s docstring said "Returns the batch count". It returns
`dbscan_fit_impl`'s value, which that function's own docstring names
correctly: the total propagation passes. `check_dbscan_batch_count_
invariance` believed the wrong sentence, called the value `batches` and
printed it as one -- so its headline number was a different quantity, and,
the part that matters, **nothing verified that its four budgets produced
more than one distinct batch count.** A run where every budget gave one
batch would have printed the same reassuring line.

DEVIATION 513: the check reads the batch count from the runner's own trace
header (not a second copy of the sizing arithmetic) and REFUSES unless the
budgets moved it. The real numbers on this box are 1/6/3/1 batches; the
2/12/6/2 the check used to print were pass counts. This matters for the AMD
leg specifically: an MI300X has 192 GB, so the default "ask the device"
budget is one batch there, and a check that could not tell would have
called it green.

### 3. The mode flip is a shared-file mutation with no lock

`tools/with_identical_mode.sh` rewrites `mojo_only/numerics.mojo` for the
length of a command that runs for minutes, and this checkout is worked by
parallel sessions. The failure is not a merge conflict, it is a **silently
mislabelled measurement**: a build that lands inside another session's flip
window compiles the other arm and reports the arm it was compiled against,
because the mode is a comptime constant and every label in the binary
agrees with the binary.

Both halves were observed in one afternoon. The refusal fired on a tree
`grep` showed as FAST a second later, and a verification pass meant to be
the FAST arm came back reporting `mode IDENTICAL` on both columns.
DEVIATION 514 holds `/tmp/cbsym-build.lock` across the whole flip-run-revert
window, `check_unsupervised_identity.sh`'s FAST arm takes the same lock, and
`check_column_invariance.sh` READS THE MODE BACK from each run rather than
assuming it from the flip.

## The column-invariance gate, and the number that gives it teeth

    pixi run check-column-invariance

k-means, k-NN and DBSCAN, under IDENTICAL, at the APPLE, NVIDIA and AMD
columns, twice each: eighteen runs, and every card and every output hash
must be the same one.

| | runs | columns | distinct answers |
|---|---|---|---|
| k-NN under **FAST** | 3 | APPLE only | **3** |
| k-NN under **IDENTICAL** | 3 | APPLE + AMD | **1** |

The FAST row is the measurement that makes the gate worth running. Three
consecutive runs of one binary, one fixture, one device gave three
different sorted index sets -- `updateSortedWarpQ`'s mutex merge resolving
an equidistant tie by arrival order. Row 11 was written as a refusal about
cross-VENDOR reproducibility; on this fixture the shipped default is not
reproducible across RUNS.

Under FAST the two columns also diverge, and the shape of the divergence is
worth recording because it is row 11's signature exactly: first divergence
at `knn.out_dist`, **every sorted distance identical**, sorted indices
different. Same distances, different neighbours. `knn_search` now records
`knn.sorted_dist` / `knn.sorted_idx` as separate stages so that diagnosis
is one line of a card diff instead of an afternoon.

WHAT THIS GATE IS NOT: every run is on one backend, so contraction (row 9),
the denormal policy (row 10) and device transcendentals (row 12) cannot
fail it. E1 is not shortened by it. It is the part of the debt that can be
paid without renting anything, and it caught two defects and one tooling
hazard on the day it was written.

Plus the instrument: `IdentityTrace` checkpoints in the k-means fit
(`restartNN.iterMM.` stages), `knn_search` (norms, distances, INDICES --
separated because two runs whose distances agree and whose indices do not
have diverged in the selector and nowhere else) and the DBSCAN runner.

**The DBSCAN trace records three stages and no per-batch ones, deliberately.**
With `max_mbytes_per_batch = 0` -- the default, and cuML's -- the batch count
comes from the device's FREE MEMORY, so `batch03.core` exists on one machine
and not on another and the differ would align two disjoint tag sets. What the
per-batch records would have tested is gated directly instead
(`check_dbscan_batch_count_invariance`).

## How to run it

    pixi run check-unsupervised-identity     # both modes, all six files
    pixi run price-unsupervised-identity     # the cost, alternated

Both flip `mojo_only/numerics.mojo` session-locally through
`tools/with_identical_mode.sh`, which reverts on EXIT/INT/TERM. The flip must
never be committed (E1_RUNBOOK's preconditions).

A traced run is ONE FIT: the tags carry algorithm positions, two fits in one
process would repeat them, and `tools/identity_trace_diff.py` refuses a file
whose sequence numbers restart. That is the instrument stating its contract.

## The price, measured (STALE for `knn.auto` -- see owed item 7)

`tools/price_unsupervised_identity.sh`, three alternated rounds, M4, one
device, medians of nine samples per arm:

| arm | FAST | IDENTICAL | ratio |
|---|---|---|---|
| `kmeans.fit` (100k x 32, k=16, 10 iters) | 15.60 ms | 11.86 ms | 0.76x |
| `knn.auto` (20k index, 1k queries, k=10) | 10.88 ms | 13.61 ms | 1.25x |
| `knn.tiled` (same shape, forced) | 8.04 ms | 22.92 ms | **2.85x** |
| `dbscan.fit` (20k x 8) | 144.05 ms | 203.74 ms | 1.41x |

Read as a band, not to four figures: the modes are two BINARIES, so they
cannot be interleaved inside one process the way `bench/` interleaves two
arms, and this box drifts up to 1.7x across twenty minutes. The k-means row
coming out BELOW 1.0 is inside that band and should not be reported as
identity being free -- what it honestly says is that the k-means pins cost
nothing measurable at this shape.

`knn.tiled`'s 2.85x is the only real price and it is the only place where
identity buys a different KERNEL rather than a different rounding: the
vendor matmul is replaced by a per-cell contraction. Under IDENTICAL the
default arm does not go there (DEVIATION 502 pins it to fused), so 2.85x is
what `k > 64` costs.

## What is owed

1. ~~**A second vendor.**~~ **DONE 2026-08-23 for the Apple<->AMD pair.**
   An MI300X (ROCm 6.4.1) and the M4 produce bit-identical outputs AND
   bit-identical cards for all three algorithms -- 86 matched stage pairs,
   zero divergences, 0 findings on either leg, commit parity proven by a
   source-tree hash rather than by a `commit.txt` the rented box could not
   write. **NVIDIA is still owed**, and `E1_RESULTS.md`'s own honesty note
   is why that matters: for the tree ensembles Apple<->AMD agreed through
   every stage while an H100 diverged. See `E1U_RESULTS.md`. What remains
   of the original item: `check-column-invariance` pays the
   source-level half; the arithmetic half is E1's and is untouched by it.
   Rows 19-26 have not been to a real MI300X. `bench/unsupervised_trace_
   main.mojo` and E1_RUNBOOK's Phase 3u are the payload, ready to run; what
   is missing is the box. The GBDT rows HAVE been -- Apple<->AMD agree and
   an H100 diverges at `tree001.winners.scores` -- which is the measured
   proof that two backends agreeing closes nothing, and by the same argument
   three columns on one backend agreeing closes less.
   **Run `check-ieee-arith` FIRST on the real AMD box.** The AMD leg of
   2026-08-23 recorded "a*b+c is UNFUSED on this backend" from the counting
   arm that was later shown to be an artifact (row 9's correction); AMD's
   actual contraction behaviour is therefore UNMEASURED, and it is the
   behaviour `identical_mul_add` exists for.
2. **`core/column_stats.mojo`'s float `block.sum` pair** is row 20's defect
   in the same directory, reached by PCA and OLS. One import from the fix,
   left named rather than taken, because `decomposition/` is another lane.
3. **The `pinned_block_sum` merge.** `gbdt/targets/kernel/pointwise_targets`
   has a same-shape twin. Two implementations of one identity primitive is a
   thing to get wrong; the merge is one import when that lane next opens the
   file.
4. **The fused k-NN queue's tie set is UNNAMED, and is no longer the
   default's problem.** DEVIATION 509 moved AUTO off that arm under
   IDENTICAL on every column, so the shipped identical answer no longer
   depends on it; naming it is now only owed for an explicit
   `KNN_METHOD_FUSED`. The original text stands: The
   tiled arm returns the lowest indices (a total order); the fused arm
   returns whatever its distance-only comparator's network settles on --
   reproducibly, but it is `{5, 900}` where tiled gives `{5, 100}` on this
   lane's fixture. Naming it needs a `(distance, index)` comparator threaded
   through `faiss_select`'s merge network and a `warp_v_top` beside
   `warp_k_top` in `add_thread_q`. Not done; the arm pin (502) is what makes
   the shipped answer well-defined without it.
5. **`k > SELECT_BLOCK` in the identical selector.** Refused at the
   launcher, not silent. The rank pass needs a loop.
6. **`decomposition/mojo_only/jacobi_eigh_device.mojo`** folds two Float32
   sums (`fro2`, `off`) through the library `block.sum` at `JACOBI_TPB`,
   and those folds decide the SWEEP COUNT, so a last-bit difference changes
   the eigenvectors rather than perturbing them. `K_LIB_JACOBI_EIGH`'s 32 is
   the lane width, not a scheduling choice, so the fix is the fold and not
   the matrix row. Reserved as DEVIATION 511; `decomposition/` is the other
   lane's and rows 27-31 are open there.
7. **`price-unsupervised-identity` is stale.** The table below was measured
   with AUTO pinned to the fused arm. Under DEVIATION 509 the IDENTICAL
   default IS the tiled arm, so `knn.auto` now costs what `knn.tiled` costs
   at `k <= 64` and the 1.25x row no longer describes the shipped default.
   Re-measure before quoting it.
