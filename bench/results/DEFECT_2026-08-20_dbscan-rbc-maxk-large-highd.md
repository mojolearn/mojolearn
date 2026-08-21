# DBSCAN RBC crashes at 400,000 x d=32. New cell, never run before.

    pixi run -e bench python bench/run_bench.py --mojo-bin build/dsweep_main \
        --sklearn-script dsweep_sklearn.py --rounds 1

    Unhandled exception caught during execution: dbscan rbc: batch 79 was
    bounded at 1 columns by loop 1 and came back with 0; given maximum
    rowsize was not sufficient

`dbscan/ported/dbscan/runner.mojo:577`. Deterministic; the run dies there
every time and takes the whole sweep with it, so **no scikit-learn arm ran
and this sweep produced no ratios at all.**

## Why this cell existed for the first time today

Every `d >= 32` cell on the board ran at n = 200,000. Every large-n cell ran
at d = 8. "We win at width" and "we lose at size in their best corner" were
measured on disjoint slices and the intersection was assumed, never run.
The first time anything ran at both, it crashed.

## What the assertion is

Mirrors cuVS `ASSERT(max_k == data.max_k, "given maximum rowsize was not
sufficient")`, `algo.cuh:135`. It is an EQUALITY on purpose: loop 1 measures
the longest neighbor row per batch on those exact rows, so loop 2's one-pass
query cannot legitimately disagree. A mismatch means the CSR in `col_ind` is
truncated garbage, so raising is correct and the assertion is not the defect.

## The leading hypothesis, NOT yet measured

`maxklen[79] == 1` and the loop-2 query returned `0`. A max degree of 1 over
a whole batch means exactly one point in 5,000-odd had exactly one neighbor
within eps. At d=32 the fixture uses eps = 0.30*sqrt(32/8) = 0.6 in a
[0,4]^32 cube, where essentially every point is isolated -- so a batch's
maximum is decided by a SINGLE point, and one point moving across the eps
boundary flips the batch max between 1 and 0.

That points at the two paths computing the boundary differently:

- loop 1 takes `maxklen[i]` from a device max-reduce over `vd`, which the
  vertexdeg kernel wrote (`runner.mojo:449-462`)
- loop 2 takes `actual` from `rbc_eps_nn_query_max_k`

If those two disagree on a point at distance exactly eps, or on the
self-neighbor, dense data hides it (the batch max is set by a well-interior
point) and sparse data cannot (the batch max IS the boundary point). Same
shape as [[uniform-test-data-hides-permutation]]: the regime where every
cell holds the same value is the regime that cannot see the disagreement.

**This is a hypothesis from two integers. It has not been probed.**

## The cheap next step

Print `vd`'s max and `rbc_eps_nn_query_max_k`'s answer for the same batch,
side by side, at the failing cell, plus the count of points at distance
within one ULP of eps. That distinguishes "boundary round-off" from "the two
kernels disagree structurally" in one run. Do NOT fix by relaxing the
equality to `<=`; that hides a truncated-CSR bug behind a silent pass.

## What did complete, and why it is not a result

Ours only, no ratios, one round, medians of three:

    knn_d32@200,000        205.7 ms
    knn_d64@200,000        511.2
    knn_d128@200,000     1,016.0
    knn_d32@1,000,000    1,356.1     <- new cell, ran clean
    dbscan_d8@200,000      669.2
    dbscan_d32@200,000   5,179.0
    dbscan_d64@200,000  20,717.0

These run ~1.4x slower than the same cells in the recorded board
(dbscan_d32 3,430.6, dbscan_d64 15,290.3). Peer sessions were active on this
box. **Absolute ms across windows compare to nothing** -- that is the
standing rule here and it applies to this table. The numbers are recorded
only to show which cells executed.

`knn_d32@1,000,000` is the one genuinely new cell that ran clean. It has no
scikit-learn counterpart yet because the sweep died before their arm.
