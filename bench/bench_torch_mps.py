#!/usr/bin/env python3
"""THE GPU BASELINE THAT ACTUALLY EXISTS ON THIS MACHINE.

Andrew's rule, 2026-08-20: if a GPU library for the algorithm runs on the
MacBook, benchmark against THAT. A CPU-only baseline is permitted only when
no GPU option exists.

scikit-learn is not that library -- measured in
`SKLEARN_GPU_BASELINE_2026-08-20.md`, and `KMeans` in particular declares
`array_api_support: False` and refuses a torch tensor on any device. But
PyTorch runs on MPS, and a competent Mac user reaching for GPU k-means or
GPU brute-force k-NN writes exactly what is below: `cdist` + `argmin` +
`index_add_` for Lloyd, `cdist` + `topk` for k-NN, `X^T X` + `eigh` for PCA.
Thirty lines each, no exotic knowledge. That is the honest opponent and it
is a much harder one than scikit-learn's CPU.

SCALE. Andrew, same day: never test without scale, and row count alone is
not scale. The standing board ran 4,000,000 x 32 with k=64, which is large
in rows and small in every dimension that decides the cost. Real k-means
builds an IVF coarse quantizer at k = 1k-64k over embeddings of width
384/768/1536; real k-NN searches a 1M-1B index at those widths. The shapes
here are chosen from that, not from what fits comfortably.

TILING IS NOT A HANDICAP, IT IS THE IMPLEMENTATION. A 1M x 1024 distance
matrix is 4 GB and a 10k x 1M one is 40 GB, so both sides tile. Doing it in
chunks is what every real implementation does and it is not a thumb on the
scale; refusing to tile and calling their OOM a win would be.

    pixi run -e skgpu python bench/bench_torch_mps.py [--quick]
"""
import argparse
import time

import torch

DEVICE = "mps"
REPEATS = 3

# Tile widths. A `cdist` result is materialized, so these cap the largest
# intermediate: KM_CHUNK x k floats for k-means, NN_CHUNK x n_index for k-NN.
# At the full shapes that is 16384 x 1024 x 4 = 67 MB and 128 x 1,000,000 x 4
# = 2 GB. Both sides of this comparison tile; a 40 GB distance matrix is
# not an implementation.
KM_CHUNK = 16_384
NN_CHUNK = 1024

M1, M2, M3 = 0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB


def u01(rows, cols, salt, gen):
    """Deterministic fixture, generated on the device to keep host RAM free."""
    return torch.rand((rows, cols), generator=gen, device=DEVICE,
                      dtype=torch.float32)


def sync():
    torch.mps.synchronize()


def emit(name, seconds):
    print(f"ARM {name} {seconds * 1000:.4f}", flush=True)


# --------------------------------------------------------------------------
# k-means, Lloyd. Same shape of computation as ours: assign every row to its
# nearest centroid, then recompute centroids as the mean of their members.
# `index_add_` is the scatter-add; `cdist` is the assignment distance.
# --------------------------------------------------------------------------
def kmeans(x, init, n_iter, chunk):
    """EXPANDED L2, NOT `cdist`. Corrected 2026-08-20 before any ratio was
    quoted from this file.

    The first version called `torch.cdist`, which materializes pairwise
    distances directly. That read 13.9 s at 1M x 384, k=1024 -- and it is
    NOT what a competent implementation does, nor what we do. Ours runs
    `METRIC_L2_EXPANDED`: ||x-c||^2 = ||x||^2 - 2 x.c + ||c||^2, so the
    whole assignment becomes ONE GEMM, which is the operation a GPU is
    actually built for.

    Benchmarking our GEMM against their `cdist` would have been an
    ALGORITHM difference reported as a hardware one -- the identical trap
    we deleted the `LinearRegression` and `dbscan_brute` arms for earlier
    the same day. `||x||^2` is dropped because it is constant across
    centroids and cannot move an argmin; that is standard and both sides
    do it."""
    n, d = x.shape
    k = init.shape[0]
    c = init.clone()
    for _ in range(n_iter):
        labels = torch.empty(n, dtype=torch.long, device=DEVICE)
        c2 = (c * c).sum(dim=1)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            labels[s:e] = torch.addmm(
                c2.unsqueeze(0), x[s:e], c.T, beta=1.0, alpha=-2.0
            ).argmin(dim=1)
        sums = torch.zeros((k, d), device=DEVICE, dtype=torch.float32)
        counts = torch.zeros(k, device=DEVICE, dtype=torch.float32)
        sums.index_add_(0, labels, x)
        counts.index_add_(0, labels, torch.ones(n, device=DEVICE))
        nonempty = counts > 0
        c = torch.where(nonempty[:, None], sums / counts.clamp(min=1)[:, None], c)
    return c


# --------------------------------------------------------------------------
# brute-force k-NN. `cdist` + `topk`, queries tiled.
# --------------------------------------------------------------------------
def knn(index, queries, k, chunk, index_sq=None, index_tile=65_536):
    """One GEMM per tile, tiled over BOTH queries and index, top-k merged.

    THREE VERSIONS OF THIS WERE MEASURED before any ratio was quoted, and
    the fastest is kept, because handing the opponent a bad implementation
    and then beating it is the whole failure mode this file exists to avoid:

        cdist, query tile 128                    35-50 s
        one GEMM over the FULL index, tile 512   475 s   <- 10x WORSE
        one GEMM, index tiled, top-k merged      this

    The middle one is the instructive failure. Switching `cdist` for a GEMM
    is right, but doing it against the whole 1,000,000-row index makes each
    `topk` run over a 512 x 1,000,000 (2 GB) matrix, and `topk` on MPS
    degrades catastrophically at that width -- it cost more than the GEMM
    saved. Tiling the index keeps every intermediate at
    `chunk x index_tile` and merges partial top-k results, which is what any
    real brute-force implementation does and what ours does.

    `||q||^2` is dropped (constant within a query row, cannot move the
    top-k). `||index||^2` is hoisted out of the timed loop the way a real
    implementation would."""
    if index_sq is None:
        index_sq = (index * index).sum(dim=1)
    n_q = queries.shape[0]
    out_d = torch.empty((n_q, k), device=DEVICE)
    out_i = torch.empty((n_q, k), dtype=torch.long, device=DEVICE)
    for s in range(0, n_q, chunk):
        e = min(s + chunk, n_q)
        q = queries[s:e]
        best_d = None
        best_i = None
        for a in range(0, index.shape[0], index_tile):
            b = min(a + index_tile, index.shape[0])
            d = torch.addmm(index_sq[a:b].unsqueeze(0), q, index[a:b].T,
                            beta=1.0, alpha=-2.0)
            td, ti = torch.topk(d, min(k, b - a), dim=1, largest=False)
            ti = ti + a
            if best_d is None:
                best_d, best_i = td, ti
            else:
                cd = torch.cat((best_d, td), dim=1)
                ci = torch.cat((best_i, ti), dim=1)
                best_d, sel = torch.topk(cd, k, dim=1, largest=False)
                best_i = torch.gather(ci, 1, sel)
        out_d[s:e], out_i[s:e] = best_d, best_i
    return out_d, out_i


# --------------------------------------------------------------------------
# PCA by covariance eigendecomposition -- OUR route, and sklearn's `auto`.
# The GEMM is the cost and runs on MPS. `eigh` of the d x d matrix is tiny
# and `aten::_linalg_eigh` is unimplemented on MPS, so it goes to the CPU --
# which is what any sane implementation would do regardless of support.
# --------------------------------------------------------------------------
def pca(x, n_comp):
    mean = x.mean(dim=0, keepdim=True)
    xc = x - mean
    cov = (xc.T @ xc) / (x.shape[0] - 1)
    sync()
    evals, evecs = torch.linalg.eigh(cov.cpu())
    return evecs[:, -n_comp:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="small shapes, to prove the harness runs")
    args = ap.parse_args()

    if args.quick:
        km_n, km_d, km_k, km_iter = 100_000, 128, 256, 5
        nn_n, nn_q, nn_d, nn_k = 100_000, 2_000, 128, 10
    else:
        # k-means: an IVF coarse quantizer over 1M sentence-transformer-width
        # embeddings. k=1024 is the low end of what FAISS would pick for this
        # corpus size (~sqrt(N) to 16*sqrt(N)).
        km_n, km_d, km_k, km_iter = 1_000_000, 384, 1024, 20
        # k-NN: a 1M-vector index at the same width, 10k queries, k=10.
        nn_n, nn_q, nn_d, nn_k = 1_000_000, 10_000, 384, 10

    print(f"# torch {torch.__version__} on {DEVICE}", flush=True)
    print(f"# kmeans {km_n:,} x {km_d}, k={km_k}, {km_iter} iters", flush=True)
    print(f"# knn    {nn_n:,} x {nn_d} index, {nn_q:,} queries, k={nn_k}",
          flush=True)

    gen = torch.Generator(device=DEVICE).manual_seed(7)
    km_x = u01(km_n, km_d, 0, gen) * 10.0
    km_init = u01(km_k, km_d, 5, gen) * 10.0
    sync()

    # UNTIMED WARM-UP, both arms. The smoke run read 559 ms on rep 0 against
    # 70 ms on reps 1-2 for the identical call: MPS compiles its kernels on
    # first use. Ours warms up too (`bench_main.mojo` runs an untimed pass),
    # so timing their cold start would be a thumb on the scale.
    kmeans(km_x, km_init, 2, chunk=KM_CHUNK)
    pca(km_x, 8)
    sync()

    for _ in range(REPEATS):
        t = time.perf_counter()
        kmeans(km_x, km_init, km_iter, chunk=KM_CHUNK)
        sync()
        emit("kmeans", time.perf_counter() - t)

        t = time.perf_counter()
        pca(km_x, 8)
        sync()
        emit("pca", time.perf_counter() - t)

    del km_x, km_init
    torch.mps.empty_cache()

    nn_idx = u01(nn_n, nn_d, 1, gen)
    nn_q_t = u01(nn_q, nn_d, 2, gen)
    sync()

    knn(nn_idx, nn_q_t[:NN_CHUNK], nn_k, chunk=NN_CHUNK)   # warm-up, untimed
    sync()

    for _ in range(REPEATS):
        t = time.perf_counter()
        knn(nn_idx, nn_q_t, nn_k, chunk=NN_CHUNK)
        sync()
        emit("knn", time.perf_counter() - t)


if __name__ == "__main__":
    main()
