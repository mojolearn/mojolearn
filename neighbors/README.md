# neighbors: exact brute-force k-NN, from cuVS and RAFT

Third section, second and third upstream. Same rule as `ported/` and
`cluster/`: **COPY, DO NOT IMPROVE.**

## The claim being tested

Brute force is a matrix multiply plus a top-k. Both are arithmetic-dense and
embarrassingly parallel, which is the shape a GPU wants. On a machine where
the competition's GPU arms do not run at all, exact brute force on the GPU can
plausibly beat APPROXIMATE k-NN on the CPU, and that is strictly stronger than
being faster, because it returns the RIGHT neighbors.

## Two upstreams, and a refinement to the layering rule

`cluster/README.md` said a RAFT call is not a `ported/` file, because RAFT is
a general library this tree does not mirror. That is still right for a call we
merely stand in for. It is wrong for a file we read and transliterate, which
is what `select_radix.mojo` is, and which makes it a derivative work of RAFT.

    a RAFT call we stand in for   ->  mojo_only/, naming the call
    a RAFT file we transliterate  ->  ported/,  with raft as its upstream

`PORTED_MAP.tsv` names the upstream per row for that reason.

## Why radix and not warpsort

RAFT ships two top-k families. `select_warpsort.cuh` is the FAISS WarpSelect
design with 14 warp intrinsics, and Mojo 1.0 has none, so it is not
expressible. `select_radix.cuh` has **zero**: `__syncthreads()` plus CUB block
collectives, which is exactly the pair Mojo provides.

**Their own dispatch prefers the one we cannot have.** `select_k-inl.cuh:38`
sends `2 < k <= 256` to warpsort, which is every k a user actually asks for.
We are on RAFT's second choice by necessity. The bar is not WarpSelect on an
NVIDIA card, because that card cannot run here; the bar is `argpartition` on a
CPU.

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
