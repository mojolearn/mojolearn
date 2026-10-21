# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The GPU binding's resident half of randomized PCA (x_decomp/pca_randomized.mojo):
each method is the typed body of the resident entry the Python kit called
(`x_decomp_dev_*`), with that entry's build flags: the pooled orth when
DECOMP_FAST_ORTH_WS is compiled in (`s_flags` bit 2), the resident small SVD
when IDN_SVD_RESIDENT is (`x_decomp_dev_svd` registered)."""
from x_decomp.cells import F32Ptr
from x_decomp.pca_randomized import RDev
from x_decomp.resident import (
    IDN_SVD_RESIDENT,
    dev_absmax_run,
    dev_colsum_run,
    dev_download_run,
    dev_ew_run,
    dev_gemm_run,
    dev_move_run,
    dev_orth_run,
    dev_rand_run,
    dev_rowsum_run,
    dev_svd_run,
    dev_upload_run,
    pool_alloc,
    pool_free,
)
from x_decomp.s_linalg_fast import DECOMP_FAST_ORTH_WS, dev_orth_ws_run
from x_decomp.select_dev import dev_order_f_run, dev_order_small_run, dev_reduce_run


struct DevRDev(RDev):
    @staticmethod
    def alloc(n: Int) raises -> Int:
        return pool_alloc(n)

    @staticmethod
    def free(id: Int) raises:
        pool_free(id)

    @staticmethod
    def upload(id: Int, src: F32Ptr, n: Int) raises:
        _ = dev_upload_run(id, src, n)

    @staticmethod
    def download(id: Int, dst: F32Ptr, n: Int) raises:
        _ = dev_download_run(id, dst, n)

    @staticmethod
    def ew(
        a: Int, b: Int, c: Int, dst: Int, op: Int, count: Int, d: Int, lb: Int, bm: Int, lc: Int, cm: Int,
        s: Float64,
    ) raises:
        _ = dev_ew_run(a, b, c, dst, op, count, d, lb, bm, lc, cm, s)

    @staticmethod
    def gemm(a: Int, b: Int, c: Int, m: Int, k: Int, n: Int, ta: Bool, tb: Bool) raises:
        _ = dev_gemm_run(a, b, c, m, k, n, ta, tb)

    @staticmethod
    def colsum(a: Int, dst: Int, n: Int, d: Int) raises:
        _ = dev_colsum_run(a, dst, n, d)

    @staticmethod
    def rowsum(a: Int, dst: Int, n: Int, d: Int) raises:
        _ = dev_rowsum_run(a, dst, n, d)

    @staticmethod
    def move(
        src: Int, idx: Int, dst: Int, op: Int, count: Int, a1: Int, a2: Int, a3: Int, ist: Int, ioff: Int,
        nsrc: Int, ndst: Int, nidx: Int,
    ) raises:
        _ = dev_move_run(src, idx, dst, op, count, a1, a2, a3, ist, ioff, nsrc, ndst, nidx)

    @staticmethod
    def rand(dst: Int, count: Int, seed: Int, stream: Int, kind: Int) raises:
        _ = dev_rand_run(dst, count, seed, stream, kind)

    @staticmethod
    def orth(a: Int, dst: Int, m: Int, l: Int) raises:
        comptime if DECOMP_FAST_ORTH_WS:
            _ = dev_orth_ws_run(a, dst, m, l)
        else:
            _ = dev_orth_run(a, dst, m, l)

    @staticmethod
    def absmax(a: Int, dst: Int, n: Int, d: Int, by_col: Bool) raises:
        _ = dev_absmax_run(a, dst, n, d, by_col)

    @staticmethod
    def has_svd() -> Bool:
        return IDN_SVD_RESIDENT

    @staticmethod
    def svd(a: Int, s: F32Ptr, v: F32Ptr, m: Int, n: Int) raises:
        _ = dev_svd_run(a, s, v, m, n)

    @staticmethod
    def reduce(a: Int, dst: Int, count: Int, op: Int) raises:
        _ = dev_reduce_run(a, dst, count, op)

    @staticmethod
    def order_f(a: Int, skip: Int, dst: Int, cnt: Int, n: Int, neg: Int, has: Int) raises:
        _ = dev_order_f_run(a, skip, dst, cnt, n, neg, has)

    @staticmethod
    def order_small(a: Int, dst: Int, n: Int) raises:
        _ = dev_order_small_run(a, dst, n)
