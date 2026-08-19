# Porting the FUSED kernel. 2.8x on k-means, and the lesson that got it.

## The instruction that produced this

> always do what the other library does regardless of what your test code
> does. what is happening is they are doing deeper customizations and you are
> looking surface level and reverting.

That is exactly what had happened, twice, and it is worth writing down
because the pattern is subtle.

## The pattern

cuVS has two distance paths. The FUSED one keeps a running `(value, key)`
minimum per row **in registers** and never writes the distance matrix. The
UNFUSED one writes the whole `m x n` tile to memory and reduces it after.

I ported the unfused one. My stated reason was true: their fused arm is
CUTLASS, CUTLASS is NVIDIA tensor cores, and it does not port. I also noted
their own selector prefers unfused on Blackwell.

**Both facts are true and neither was the point.** There are two fused
implementations, not one. `fused_distance_nn/simt_kernel.cuh` is the SIMT
path, plain CUDA cores, no tensor-core instruction anywhere, and it is the
one that runs on most hardware. It ports. I skipped it because the unfused
path was simpler, and then justified the choice with a fact about a different
file.

Then, when the first benchmark went badly, I blamed the GEMM and tuned it.
That is the surface-level move: the GEMM was not the design difference. The
FUSION was.

## The numbers

    arm       naive   tiled   tiled+SIMD   FUSED    sklearn
    kmeans   450.83  542.92       491.05  175.30      58.64
    knn       21.67   30.88        22.26   22.28      33.43
    pca       16.94   16.92        17.00   16.87       6.23
    dbscan    15.52   20.38        17.08   16.85       8.88
    ols       28.18   27.67        28.14   27.81     105.74

**k-means 491 to 175 ms.** Every kernel-tuning round before this moved it by
less than 15 percent. Porting what they actually do moved it by 2.8x.

The arithmetic is identical in both. What changed is that at 200,000 rows and
16 clusters the unfused path wrote 3.2 million floats to global memory and
read them straight back, every Lloyd iteration, to extract one minimum per
row. The fused kernel does the same multiplies and moves none of it.

## Where it still stands

We win k-NN (1.50x) and OLS (3.80x). We lose k-means (0.33x), PCA (0.37x) and
DBSCAN (0.53x).

And the next two are the same lesson again, already visible:

- **PCA has not moved by one millisecond across four rounds.** Its cost is
  `core/column_stats.mojo::covariance_kernel`, which is still the naive 16x16
  tile, and RAFT computes that product through the same contraction policy
  everything else uses. There is a second contraction-shaped kernel in
  `core/` and it has never been ported.
- **DBSCAN materializes its full `n x n` adjacency matrix.** cuML does not:
  it batches rows so the adjacency stays bounded, and `dbscan/UNPORTED.tsv`
  has said so since the section was written.

Both are cases of theirs doing something deeper that we have written down and
not yet done.

## Standing rule, added

**Port the path they actually run, not the path that is easiest to port. If
a path is skipped, the reason has to be a property of THAT path, not of a
different one.** "CUTLASS does not port" was not a reason to skip
`simt_kernel.cuh`.
