# neighbors: exact brute-force k-NN from cuVS and RAFT, and the k-NN classifier and regressor over it from cuML

Third section, second and third upstream (and, since 2026-08-23, cuML as a
fourth, for the classifier and regressor -- see the last section). Same
rule as `gbdt/` and `cluster/`: **COPY, DO NOT IMPROVE.**

## The claim being tested

Brute force is a matrix multiply plus a top-k. Both are arithmetic-dense and
embarrassingly parallel, which is the shape a GPU wants. On a machine where
the competition's GPU arms do not run at all, exact brute force on the GPU can
plausibly beat APPROXIMATE k-NN on the CPU, and that is strictly stronger than
being faster, because it returns the RIGHT neighbors.

## Two upstreams, and a refinement to the layering rule

`cluster/README.md` said a RAFT call is not a `gbdt/` file, because RAFT is
a general library this tree does not mirror. That is still right for a call we
merely stand in for. It is wrong for a file we read and transliterate, which
is what `select_radix.mojo` is, and which makes it a derivative work of RAFT.

    a RAFT call we stand in for   ->  checks/, naming the call
    a RAFT file we transliterate  ->  gbdt/,  with raft as its upstream

`DERIVATION_MAP.tsv` names the upstream per row for that reason.

## Both top-k families are ported now

RAFT ships two. `select_radix.cuh` has **zero** warp intrinsics:
`__syncthreads()` plus CUB block collectives. `select_warpsort.cuh` is the
FAISS WarpSelect design and has 14, which is why radix went first.

This section used to say warpsort was **not expressible**, because Mojo 1.0
was believed to have no warp primitives. **That was false** (`PORTING.md 2`,
`VENDOR_LIBRARIES.md`): they are under `std.gpu.primitives.warp`, one
namespace level below where four searches looked. `select_warpsort.mojo`
ports `warp_sort_immediate`, its base queue, the block tree-merge and the
dense `block_kernel`; the three `warp_sort_filtered` / `_distributed` /
`_distributed_ext` queues are still out, for one specific reason recorded in
that file's docstring and in `NOT_IMPLEMENTED.tsv`.

**Their own dispatch prefers warpsort.** `select_k-inl.cuh:38` sends
`2 < k <= 256` to it and only `k > 256` to radix, which is every k a user
actually asks for. Radix was RAFT's second choice across the whole practical
range and this tree ran it alone until DEV 1922 (2026-08-28,
`bench/results/LANE_knn-speed-campaign_2026-08-28.md`): the kernel-matrix
row `knn_warpsort_select_for` now hands the tiled path's `2 < k <= 256`
band to warpsort on the NVIDIA column under FAST, pending the
orchestrator's H100 A/B. On Apple nothing has yet MEASURED the two against
each other, so radix stays that column's default and warpsort sits beside
it so the two can be diffed; the row is where Apple flips if a Mac window
says otherwise. AMD (64-lane) is excluded by the same row — `WARP_LANES`
is a pinned 32 — and IDENTICAL keeps the composite-key radix everywhere.

## `core/` now exists, and it was earned rather than planned

`core/gemm.mojo` and `core/row_norms.mojo` were written for the k-means port
and moved up unchanged the moment this section needed them. That is the
evidence `PLAN.md` asked for about whether the substrate was real or was
quietly shaped by the first algorithm through it.

`core/expand_distances.mojo` is new here for a real reason: k-means fuses the
distance epilogue into its reduction because the reduction consumes each
element as it is formed, and a top-k cannot, because every distance has to
survive.

## Status

**Launched and passing.**

    check_knn OK: 64 queries x k=8 over 4096 index points,
      every returned neighbor is in the exact true set
    check_knn_reach_by_sabotage OK: index_norm moved 512/512 neighbors;
      query_norm offset moved 0 sets, which is the predicted shape

The truth is computed on the host in **Float64 with the DIRECT formula**, so
the GPU's expanded-identity answer is checked against an independent
computation and not against a rearrangement of itself.

Never benchmarked. No timing of any kind exists.

## Three things this section paid for in failed runs

**1. A pointer conditional picked the wrong branch.** `PORTING.md 19`.

**2. The expanded identity cannot rank collinear points in float32.** The
first fixture put 4096 points on a line; norms were about 1e10, distances
about 1e3, and float32's ulp at 1e10 is roughly 1024, so every distance
collapsed onto a coarse grid. **Rescaling does not help**: for N collinear
points the ratio of closest-pair squared distance to norm is about `1/N^2`,
which at N=4096 is below float32 precision at any scale. cuVS defaults to
`L2Expanded` and this is what the GEMM formulation costs.

**3. A reach sabotage has a WINDOW.** Large enough that the result must
visibly move, small enough that it does not destroy the property being
asserted. Both k-NN sabotages failed once for being outside it, and the
kernel was right both times.

## The k-NN classifier and regressor, 2026-08-23

`mojolearn.KNeighborsClassifier` / `KNeighborsRegressor` are cuML's
`ML::knn_classify` / `knn_class_proba` / `knn_regress` over the search
above, ported file for file where cuML keeps them:

    cuml cpp/src/knn/knn.cu:328-389          ->  impl/knn/knn.mojo
    cuml cpp/src_prims/selection/knn.cuh     ->  impl/selection/knn.mojo
    raft label/detail/classlabels.cuh        ->  impl/label/classlabels.mojo

The vote is `class_probs_kernel` (each of the `k` neighbour slots adds `1/k`
to its class, serially, per query) then `class_vote_kernel` (first maximum,
so a tie goes to the LOWEST class in the sorted unique-label set, which is
what scikit-learn's `mode` does); the mean is `regress_avg_kernel`. Both
folds are serial per row in cuML already, so FAST and IDENTICAL run ONE
arithmetic and the mode changes only the `ftz` seams (DEVIATION 542;
IDENTITY_PATHS row 32). The host policy that composes search and vote
under one identity trace is `neighbors/estimator.mojo` policies 5-7;
DEVIATIONS 541-544 are in the DEVIATION BLOCKs of the two ported files and
the estimator (540 was taken by the Gram/TF32 lane the same day and is
not used here).

Gates: `checks/knn_classify_check.mojo`, `checks/knn_regress_check.mojo`
(host transcription bit for bit, planted vote ties, reach by one flipped
label / one moved target, `k = 0` refused, multi-output layout, run twice),
run by `pixi run check-knn`; `tools/knn_sklearn_oracle.py` against
scikit-learn in both modes; E2U cells `knn_clf_k5`, `knn_clf_k15_3class`,
`knn_clf_ties`, `knn_reg_k5`, `knn_reg_k50`, `knn_clf_weights_distance_refused`
(`tools/e2u_matrix_fit.py`, first passes under
`bench/results/e2u/2026-08-23_knn_clf_reg/`).

What is not ported is in `NOT_IMPLEMENTED.tsv` (the MNMG `precomp_lbls` arm;
`weights='distance'` WAS in that list and is not any more, see below).

## Six metrics, general-p Minkowski, and distance weights, 2026-09-01

Four refusals closed and one kept for a stated reason.

**The distance abstraction.** `impl/distance/detail/distance_ops.mojo` is
cuVS's `distance_ops/` directory: one file holding the `core()` and
`epilog()` of every op this tree computes, plus their `DistanceType`
enumerators with THEIR values (`cuvs/distance/distance.h:22-69`). It was
built rather than adding a branch per metric per lane because there were
already two copies of the same three cores -- `kde/impl/distance/
distance_ops.mojo` had `l2_unexp` / `l1` / `l_inf` and the k-NN tile had the
expanded L2 identity inline -- and cosine plus Minkowski would have made
four. The three cores MOVED there, body for body; no KDE bit moves.

**Which metrics can use the expanded trick and which cannot**, because it
decides the kernel shape:

    L2Expanded / L2SqrtExpanded  EXPANDED.  ||x||^2 + ||y||^2 - 2 x.y
    CosineExpanded               EXPANDED, DIFFERENTLY. Same inner product,
                                 SQRT'd norms instead of squared, and a
                                 `1 - dot/(nx*ny)` epilogue. cuVS's own
                                 brute force says so out loud: it REWRITES
                                 the metric to `InnerProduct`
                                 (`knn_brute_force.cuh:141`) and fixes it
                                 up by hand at `:221`.
    L1 / Linf / L2SqrtUnexpanded UNEXPANDED. Nothing factors.
    LpUnexpanded                 UNEXPANDED AT EVERY p, and that is not a
                                 limitation of the port: `sum |x-y|^p` does
                                 not factor into per-row terms for any p
                                 but 2, which is why cuVS's own op is
                                 called `lp_UNEXP`.

So cosine rides the two arms the expanded L2 path already had (vendor
matmul plus epilogue under FAST, the pinned per-cell fold under IDENTICAL)
and Minkowski rides the per-cell fold in both.

**The numeric tier: none of the three refuses anywhere.** Every primitive
the new ops need was already pinned and gated before this lane started --
`identical_mul_add` (row 9), `identical_div` and `identical_mul` (rows 49,
9), `identical_pow` (row 12, which is `portable_expf(p * portable_logf(z))`
built from operations that are correctly rounded on all three columns),
`ftz` (row 10). `pow` being a transcendental is NOT an obstacle and bought
no deviation of its own. What general p COSTS is ACCURACY, not sameness,
and the cost is measured rather than asserted:
`check_metric_matches_float64_reference` prints the worst relative error
per metric against a float64 reference computed a THIRD way.

**Three numbered deviations, each for a stated reason.**

- **552** `metric_arg` (p) is refused non-finite, `<= 0`, and SUBNORMAL.
  cuVS never checks, and their own default-constructed index has
  `metric_arg_ = 0` (`brute_force.cu:35`), so their default Lp computes
  `pow(diff, 0) == 1` per feature and returns garbage. The subnormal clause
  is the identity one: a subnormal p flushes to zero on Metal and not on
  CUDA, so the same call would return `z^0 = 1` on one column and `z^tiny`
  on another.
- **553** a ZERO-NORM ROW under cosine is refused by name. Their epilogue
  divides by `||x|| ||y||` with no guard, so a zero row is `0/0 = NaN`, and
  a NaN then enters a top-k where the two selectors this lane ships sort it
  to OPPOSITE ends -- radix's `twiddle_in` key puts it above every finite
  distance, the FAISS queue's `<` never admits it. Two arms, two different
  wrong answers, no error either way.
- **554/555/556** the distance weights: computed on the host (554, the zero
  test is a per-row any-reduction), a row whose weights all underflow is
  refused (555, the normalizer would be `+0` on an FTZ column and normal
  elsewhere), and the optional parameter on the three ported cuML
  functions (556).

**`weights='distance'` is ORIGINAL WORK and nothing here credits an
upstream for it.** cuML refuses it in the Python layer above the kernels,
so there is no C++ entry and no kernel to transliterate.
`NOT_IMPLEMENTED.tsv` used to give "there is no upstream GPU kernel to
port" as the reason; that reason was withdrawn on 2026-09-01, because a
refusal is legitimate only when the thing is genuinely impossible here or
when refusing IS the right behaviour for the input. scikit-learn's
`_base.py:74-114` is the SEMANTICS source and the definition is four lines.
The one clause everybody gets wrong is the third: a row containing an exact
zero distance is REPLACED WHOLESALE by its zero mask, so exact matches get
1.0 and every other neighbour in that row gets 0.0. A per-element port
agrees with sklearn on every fixture that has no duplicate point, which is
why `check_knn_distance_weights` plants one.

**`algorithm='kd_tree'` STAYS REFUSED, and the reason is engineering. What
the caller actually wanted now EXISTS: `algorithm='rbc'`.** cuVS shipping
no GPU kd-tree is NOT the reason and must not be given as one. A kd-tree
query is a per-query stack walk with data-dependent branching and divergent
memory access -- the shape a GPU is worst at -- and above roughly 15
dimensions the pruning bound stops firing and it degenerates to a full
scan, so it would lose to the brute force in this directory on exactly the
workloads it claims to help. The structure that DOES help in low dimensions
was already here and is exact: cuVS's RANDOM BALL COVER, one level over
sqrt(n) landmarks, no stack, no divergence, already serving
`RadiusNeighbors`. The sentence that used to close this paragraph -- "what
is actually owed is `rbc_knn_query`, not a tree" -- was true until
2026-09-01 and is not any more. `impl/neighbors/ball_cover/knn.mojo` is
that query, `NearestNeighbors(algorithm='rbc')` is its surface, and the
answer is EXACT: gated per query row against a host brute force with no
tolerance under IDENTICAL. DEVIATIONS 558 to 567 are in that file.

**The ball cover is no longer Euclidean-only, and the refusal is now
exactly as wide as its own argument.** Its pruning IS the triangle
inequality on the landmark radii, so the honest scope is THE METRICS:
Euclidean, L1/Manhattan, Linf/Chebyshev and Minkowski at p >= 1 (that is
Minkowski's inequality, which is what the name refers to). What stays
refused is what genuinely fails the inequality -- cosine, Lp at p < 1, and
SQUARED Euclidean, which is not a metric even though Euclidean distance is
-- and every refusal message now says the inequality is what the pruning
rests on. The bounds could not simply be re-pointed at a new distance
function: they were written in SQUARED Euclidean space and an Lp radius has
no squared form, so they were restructured into a per-metric COMPARISON
space (`impl/neighbors/ball_cover/common.mojo`, DEVIATIONS 564 and 565).
On the Euclidean arm that is the same arithmetic in the same order, so no
Euclidean bit moves and `check-ball-cover` / `check-radius` /
`check-dbscan` are the proof.

**The ball-cover gates**: `pixi run check-ball-cover-knn` and
`check-ball-cover-knn-identical` (`checks/ball_cover_knn_check.mojo`).
Eight gates. The exhaustive one is 6 metrics x 7 values of k x 2 fixtures,
per query row, against a host brute force spelled a second time; the
zero-tolerance claim is the IDENTICAL run, because under FAST
`identical_mul_add` is a bare `a * b + c` and `identical_pow` is the
stdlib `**` on both sides, so the FAST arm carries a stated 64-ulp
tolerance and prints its tie flips. Four sabotage arms and two non-vacuity
counts -- one its own gate, one the precondition inside the sweep:

    seam                        arm                          must move
    --------------------------  ---------------------------  ---------
    the pruning branch RAN      `dist_count` at the SHIPPED  PRUNED count
    (non-vacuity, per fixture)  scale, on the sweep's own    STRICTLY > 0
                                fixture at its own k
    the pruning threshold       `prune_scale` swept down     the answer
                                                             BREAKS, and
                                                             the largest
                                                             breaking
                                                             scale is
                                                             PRINTED
    the four-ulp slack          `prune_scale` above 1        NOTHING
                                (DEVIATION 567)
    `R_radius`                  scaled to 0.3x               LOSES
                                                             neighbours,
                                                             never gains
    `metric` reaching the       judge an L1 run by the       they DIFFER
    kernel                      EUCLIDEAN oracle
    (non-vacuity) `dist_count`  candidate distances vs       < 90% of
                                brute force's                brute force

A ONE-ULP TIGHTENING CANNOT MOVE THIS KERNEL AND THE GATE SAYS SO. The
shipped threshold carries a deliberate four-ulp slack precisely so that
float rounding can only ADMIT a candidate and never drop one, so a one-ulp
tightening lands inside it. The downward sweep replaces that arm, and the
printed breaking scale is the honest measure of how much tightening the
fixture can detect.

**WAS RED, CLOSED 2026-09-01, and the cause was the FIXTURE.**
`check_rbc_knn_prune_is_load_bearing` REFUSED: no `prune_scale` down to
0.400 moved a single row. The two arms above it were the evidence that the
kernel was not at fault -- `check_rbc_knn_prunes_work` passes, so
`dist_count` is below 90% of brute force and the query is not a full scan,
and `check_rbc_knn_radius_is_read` passes, so a perturbed landmark radius
does lose neighbours on the SCATTERED fixture. The sweep was the ONLY arm
running the LATTICE fixture, where a query at `(a + 0.5, b + 0.5)` has
SIXTEEN index rows at exactly `sqrt(0.5)` competing for k = 8 slots, so a
pruned neighbour is replaced by a row at a ZERO-ulp-identical distance and
the FAST 64-ulp tie tolerance forgives it as a tie flip. **That fixture
could not witness this sabotage at any scale.** The sweep now runs the
tie-free SCATTERED fixture and asserts a strictly positive pruned count on
that fixture first.

MEASURED, Apple M4, both `pixi run check-ball-cover-knn` and
`check-ball-cover-knn-identical`:

    PRUNED 25141 of 36864 candidate distances (computed 11723)
    survives a one-ulp tightening (the DEVIATION 567 slack)
    BREAKS at prune_scale 0.95, on 1 of 96 query rows

So the pruning branch runs, it is load bearing, and the arm is now shown
capable of going red. The printed breaking scale is the honest measure of
how much tightening this fixture can detect; a smaller number would mean a
weaker gate. The exactness arms never rested on this one and are unchanged.

**The gates**: `pixi run check-metric` and `check-metric-identical`
(`checks/metric_check.mojo`, with `checks/metric_oracle.mojo`'s three
levels of truth). Eight gates and THREE sabotage arms, one per seam whose
value looks right when it is wrong:

    seam                        arm                          must move
    --------------------------  ---------------------------  ---------
    cosine's SQRT'd norm        swap the sqrt flag on the     >= 1 cell
                                norm kernel
    Minkowski's `metric_arg`    four p values, pairwise       every cell
                                compared
    the weight rule's           a per-ELEMENT zero rule       >= 1 entry
    ROW-LEVEL replacement       beside the row-level one

Each arm PREDICTS what happens when its seam is broken (recorded in the
gate's own docstring); OBSERVED is filled in from the run.

**Not measured.** No timing of any kind exists for the new metrics, as for
the rest of this directory.
