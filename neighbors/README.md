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

What is not ported is in `NOT_IMPLEMENTED.tsv` (the MNMG `precomp_lbls` arm,
`weights='distance'`, which cuML refuses too).
