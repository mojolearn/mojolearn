# Putting MAX's tuned matmul behind core/gemm.mojo

## What was established

`linalg.matmul` from MAX runs on the GPU, takes a `TileTensor` over a
`DeviceBuffer`, supports `transpose_b`, and has an `elementwise_lambda_fn`
epilogue hook. Verified against a host reference: **0 mismatches of 3072**.

Wired behind `core/gemm.mojo::gemm_nt`, which every section now calls.

## What it bought, from a CLEAN run

    arm      ported contraction   MAX matmul
    knn                  22.28        16.11     -> 1.93x vs sklearn
    dbscan               16.85        14.53
    kmeans              175.30       175.83     (unchanged, see below)
    pca                  16.87        16.99     (unchanged, see below)
    ols                  27.81        27.87     (unchanged, see below)

## THE HARD LIMIT, which explains all three unchanged rows

    max/kernels/src/linalg/matmul/__init__.mojo:110:9:
    note: constraint failed: transpose_a not yet supported

MAX's matmul does the **N-T** shape and not the **T-N** shape.

Every distance computation is N-T: rows against centroids or index points.
Those moved.

Every covariance is T-N: `raft::stats::cov`, `lstsqEig`'s first step and
`tsvd_fit` all ask cuBLAS for `CUBLAS_OP_T, CUBLAS_OP_N`, contracting down
the ROW axis. Those cannot use it, and `core/column_stats.mojo::
covariance_kernel` stays the only implementation of that shape.

**That is the whole answer to why PCA and OLS never moved across six rounds.**
OLS in particular sat at 28 ms through naive, tiled, tiled+SIMD, fused and
vendor rounds, because nothing any of them changed was on its path. k-means
is unchanged for a different reason: it now takes the FUSED kernel, which
does not call `gemm_nt` at all.

Two ways forward, both measurable, neither guessed: an explicit transpose
(MAX ships `linalg.transpose`) followed by an N-T matmul, or applying the
register-tiling port to `covariance_kernel` directly.

## One more limit found

`matmul` with `n = 1` produced zeros for some coefficients. RAFT does not
call gemm there either: `lstsqEig` uses `raft::linalg::gemv` for
`w <- covA Ab`, because a matrix against ONE vector is a different BLAS
routine. MAX ships `linalg.gemv`; until it is wired, that step stays on the
ported kernel. Reading their line again is what found it.

## The standing rule this round establishes

**Where the incumbent calls a vendor primitive, call ours. Do not hand-write
one.** cuBLAS, cuSOLVER, CUB and Thrust sit under every one of these
libraries, and MAX ships `linalg`, `algorithm`, `nn` and `layout` as the
equivalents. Hand-writing a replacement is only correct where no equivalent
exists, and that has to be checked rather than assumed. I assumed twice.

The same table applies to the boosting port, which has not been touched:

    their file            vendor call                       ours today
    split_points.cu       cub::DeviceRadixSort::SortPairs   hand-written partition
    split_points.cu       cub::DeviceScan::ExclusiveSum     hand-written offsets
    compute_scores.cu     cub::BlockReduce                  hand-written tree reduce
    histogram_utils.cu    cub::WarpScan                     hand-written serial scan
    cuda_util/reduce.cu   cub::DeviceSegmentedReduce        hand-written reduce

Five more substitutions, same principle, none attempted yet.

## A caveat on the last benchmark run

The final run was thermally noisy: scikit-learn's own ranges tripled within
it (k-means [78, 287] ms, k-NN [36, 88] ms) and three arms came back
INDISTINGUISHABLE. The clean numbers above are from the earlier run in the
same session. **Do not quote the noisy run.** The harness reported the
overlap correctly, which is what it is for.
