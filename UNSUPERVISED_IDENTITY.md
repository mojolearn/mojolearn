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

## The price, measured

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

1. **A second vendor.** Everything above is construction plus transcription
   certified on ONE device. Rows 19-26 have not been to E1. The GBDT rows
   have -- Apple<->AMD agree and an H100 diverges at
   `tree001.winners.scores` -- which is the measured proof that two backends
   agreeing closes nothing.
2. **`core/column_stats.mojo`'s float `block.sum` pair** is row 20's defect
   in the same directory, reached by PCA and OLS. One import from the fix,
   left named rather than taken, because `decomposition/` is another lane.
3. **The `pinned_block_sum` merge.** `gbdt/targets/kernel/pointwise_targets`
   has a same-shape twin. Two implementations of one identity primitive is a
   thing to get wrong; the merge is one import when that lane next opens the
   file.
4. **The fused k-NN queue's tie set is deterministic but UNNAMED.** The
   tiled arm returns the lowest indices (a total order); the fused arm
   returns whatever its distance-only comparator's network settles on --
   reproducibly, but it is `{5, 900}` where tiled gives `{5, 100}` on this
   lane's fixture. Naming it needs a `(distance, index)` comparator threaded
   through `faiss_select`'s merge network and a `warp_v_top` beside
   `warp_k_top` in `add_thread_q`. Not done; the arm pin (502) is what makes
   the shipped answer well-defined without it.
5. **`k > SELECT_BLOCK` in the identical selector.** Refused at the
   launcher, not silent. The rank pass needs a loop.
