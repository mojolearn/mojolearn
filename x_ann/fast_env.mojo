# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The ann family's FAST-on-Apple trial switches, read from the environment
on the HOST at dispatch time (lane/apple-fast-ann, 2026-10-02).

Every switch defaults OFF and is read only in a FAST build on the Apple
column (`ANN_FAST_APPLE`); an IDENTICAL build compiles each reader to
`False`, so the old path is the only path there and its bits never move.
`MOJOLEARN_<LANE>_FAST_<NAME>=1` turns a switch on for one process. The
A/B rows that measure them are docs/apple-fast/ab/ann.txt; a switch that
wins (faster on the M3, held-out quality within FAST's run-to-run spread)
becomes the FAST default and its reader goes."""
from std.os import getenv
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

#: the compile gate of every switch here: FAST (not IDENTICAL) on Apple
comptime ANN_FAST_APPLE = GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL and has_apple_gpu_accelerator()


def tsne_fast_zsum() -> Bool:
    """t-SNE: Z from one threadgroup of 128 threads (`sum_team_kernel`)
    instead of one thread adding the n row sums (`sum_kernel`)."""
    comptime if ANN_FAST_APPLE:
        return String(getenv("MOJOLEARN_TSNE_FAST_ZSUM")) == "1"
    return False


def tsne_fast_split() -> Bool:
    """t-SNE: the repulsion's candidate rows split into TS_SPLIT stripes per
    row, one threadgroup per (row block, stripe), joined in stripe order."""
    comptime if ANN_FAST_APPLE:
        return String(getenv("MOJOLEARN_TSNE_FAST_SPLIT")) == "1"
    return False


def cagra_fast_team() -> Bool:
    """CAGRA search: one threadgroup of 32 threads per query
    (`cg_search_team_kernel`) instead of one thread per query."""
    comptime if ANN_FAST_APPLE:
        return String(getenv("MOJOLEARN_CAGRA_FAST_TEAM")) == "1"
    return False


def ann_fast_knn_bigd() -> Bool:
    """The exact k-NN graph (CAGRA build, t-SNE affinities) for rows wider
    than 64 features: candidate rows staged in threadgroup memory a feature
    chunk at a time (`knn_tiled_bigd_kernel`) instead of the untiled cell."""
    comptime if ANN_FAST_APPLE:
        return String(getenv("MOJOLEARN_ANN_FAST_KNN_BIGD")) == "1"
    return False


def ivf_fast_scan_select() -> Bool:
    """The IVF-PQ / SQ / RaBitQ scan's top-k in one launch per chunk
    (`select_group_kernel`) instead of the partial lists, seven pair joins
    and the merge."""
    comptime if ANN_FAST_APPLE:
        return String(getenv("MOJOLEARN_IVF_FAST_SCAN_SELECT")) == "1"
    return False


def ivfpq_fast_device_codebooks() -> Bool:
    """IVF-PQ build: every subspace codebook trained by one batched Lloyd
    loop on the device (`x_ann/pq_kmeans_device.mojo`) over the residuals
    already there, instead of pq_dim serial host-driven k-means fits over a
    downloaded residual matrix."""
    comptime if ANN_FAST_APPLE:
        return String(getenv("MOJOLEARN_IVFPQ_FAST_DEVICE_CODEBOOKS")) == "1"
    return False


def ivf_fast_device_trainset() -> Bool:
    """IVF coarse quantizer: the FAST training sample gathered on the device
    from the uploaded rows and its fixed-point scale from device column
    sums, instead of a host gather, a host column pass and a second upload."""
    comptime if ANN_FAST_APPLE:
        return String(getenv("MOJOLEARN_IVF_FAST_DEVICE_TRAINSET")) == "1"
    return False


def ivf_fast_device_csr() -> Bool:
    """IVF build: the CSR lists (offsets, carried ids, permuted vectors) by
    device histogram, scan and ranked scatter instead of the host passes of
    `build_list_layout`."""
    comptime if ANN_FAST_APPLE:
        return String(getenv("MOJOLEARN_IVF_FAST_DEVICE_CSR")) == "1"
    return False
