# After the substitution passes. Every arm moved.

> **THE 8.33x IS AGAINST THE WRONG ARM. Corrected 2026-08-19, late.**
>
> `LinearRegression` is LAPACK `gelsd` -- an SVD of the full design matrix.
> Ours is `lstsq_eig`: form `X^T X` (32 x 32 here) and eigendecompose it. Those
> are different algorithms doing different amounts of work, so 8.33x is a
> comparison of ALGORITHMS wearing the costume of a comparison of hardware.
>
> **The honest number is the `ols_normal_eq` arm** --
> `Ridge(alpha=0, solver="cholesky")`, which forms `X^T X` and solves it, the
> same route as ours. It is now emitted by `bench_sklearn.py` beside the
> default arm. Until that has been run, THIS FILE HAS NO DEFENSIBLE OLS RATIO.
>
> The accuracy side of that trade is priced in `bench/ols_conditioning.py`: in
> float32 -- which is all Metal has -- the normal-equations route is 32x less
> accurate than `gelsd` at condition number 10 and 834x at 100, and past 1e3
> float32 cannot hold the problem at all. Speed bought with precision, and both
> halves belong next to the number.

Apple M4, 3 rounds alternating our process with scikit-learn's, 15 samples
per arm per side, pooled from one invocation of `bench/run_bench.py`.

    arm       first run   fused round   AFTER SUBSTITUTIONS   sklearn   verdict
    kmeans       450.83        175.30                120.85     54.57   sklearn faster
    knn           21.67         22.28                 15.85     31.61   OURS, 1.99x
    pca           16.94         16.87                 10.48      6.26   sklearn faster
    dbscan        15.52         16.85                 14.52      9.28   sklearn faster
    ols           28.18         27.81                 11.63     96.87   OURS, 8.33x

## The two that matter most

**OLS 27.81 -> 11.63 ms, and 8.33x over scikit-learn.** It had sat at 28 ms
across SIX rounds while every other kernel was tuned, because nothing any of
those rounds touched was on its path. The fix was `ColKernelPolicy`.

**PCA moved for the first time, 16.87 -> 10.48 ms.** Same cause. Both were
blocked on the same false belief: that the T-N shape had no route because
MAX's matmul refuses `transpose_a`. RAFT ships a second policy and a
row-major matrix viewed column-major turns `X^T X` into the N-T shape.

k-NN is now a clean 1.99x, from the fused kernel plus MAX's matmul plus the
block scan replacing 16 barriers per radix pass per row.

k-means is 3.7x faster than the first run and still loses. DBSCAN barely
moved and still loses.

## What we still lose, and it is honest

Three of five. k-means at 0.45x, PCA at 0.60x, DBSCAN at 0.64x.

The comparison remains our untuned GPU kernels against **Apple Accelerate
dispatching to the AMX coprocessor**, which is dedicated matrix hardware on
the CPU die. We inherit the incumbents' ARCHITECTURE and write our own inner
loops, because cuBLAS, cuSOLVER, CUB and CUTLASS are closed or CUDA-only.

And the OLS win is still partly an ALGORITHM difference. scikit-learn's
`LinearRegression` uses LAPACK `gelsd`, an SVD of the full 500000 x 32
matrix; ours forms a 32 x 32 `A^T A` and eigendecomposes that. Normal
equations are less work than an SVD. Real for a user, not evidence that our
kernels are faster than theirs.

## What produced the movement, in order of size

1. **`ColKernelPolicy`** for the Gram shape. OLS and PCA both.
2. **The fused distance kernel** (`simt_kernel.cuh`), which never writes the
   `n x k` distance matrix. k-means 450 -> 175 on its own.
3. **MAX's `linalg.matmul`** behind `core/gemm.mojo::gemm_nt`, N-T shape.
4. **`block.prefix_sum`** replacing a Hillis-Steele scan in `select_radix`:
   16 barriers per radix pass per row down to one call.
5. **Warp shuffles** for the key-carrying reductions, after the "Mojo has no
   warp primitives" claim was retracted.
6. `block.sum` for seven hand-written tree reductions.
7. `gemv_gpu`, `signFlipKernel`, the k-means++ scan-and-binary-search.

Every one of those except (6) and (7) came from reading THEIR source and
finding we had solved the same problem a different way, not from finding a
missing library call.

## Not in these numbers

`select_warpsort` is ported and cannot be wired: instantiating it at a launch
site crashes `mojo build`. RAFT's dispatch prefers that family for every k a
k-NN user asks for, so the 1.99x is on their SECOND choice.

DBSCAN's batching is in and costs a second distance pass, which is why it
moved least. Its `MergeLabels` route would remove that pass.
