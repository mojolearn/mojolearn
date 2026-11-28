# SPDX-License-Identifier: Apache-2.0
"""Harmony's device kernels (lane sc-stats-1): one thread per cell of each body in x_cluster/harmony_bodies.mojo.
Only the GPU binding imports this file (x_cluster/device_ops.mojo `DeviceOps.hm_*` launch them)."""
from max.gpu import block_dim, block_idx, thread_idx
from x_cluster.bodies import FPtr, IPtr
from x_cluster.harmony_bodies import (
    hm_codes_cell, hm_rownorm_row, hm_count_cell, hm_scale_row, hm_eo_cell, hm_block_eo_cell, hm_block_r_cell,
    hm_objective_row, hm_centroid_cell, hm_ridge_a_cell, hm_ridge_rhs_cell, hm_chol_block, hm_solve_cell,
    hm_correct_cell,
)


@always_inline
def _hm_tid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


def hm_codes_kernel(raw: IPtr, off: IPtr, n: Int32, c: Int32, dst: IPtr):
    var t = _hm_tid()
    if t < Int(n):
        hm_codes_cell(raw, off, Int(n), Int(c), dst, t)


def hm_rownorm_kernel(src: FPtr, dst: FPtr, n: Int32, d: Int32):
    var t = _hm_tid()
    if t < Int(n):
        hm_rownorm_row(src, dst, Int(d), t)


def hm_count_kernel(codes: IPtr, n: Int32, c: Int32, nb: Int32, dst: FPtr):
    var t = _hm_tid()
    if t < Int(nb):
        hm_count_cell(codes, Int(n), Int(c), dst, t)


def hm_scale_kernel(dist: FPtr, n: Int32, k: Int32, sigma: Float32, dst: FPtr, normalize: Int32):
    var t = _hm_tid()
    if t < Int(n):
        hm_scale_row(dist, Int(k), sigma, dst, normalize != 0, t)


def hm_eo_kernel(r: FPtr, codes: IPtr, pr: FPtr, n: Int32, k: Int32, c: Int32, nb: Int32, e: FPtr, o: FPtr):
    var t = _hm_tid()
    if t < Int(k) * Int(nb):
        hm_eo_cell(r, codes, pr, Int(n), Int(k), Int(c), Int(nb), e, o, t)


def hm_block_eo_kernel(
    r: FPtr, codes: IPtr, pr: FPtr, n: Int32, k: Int32, c: Int32, nb: Int32, a: Int64, c0: Int64, lo: Int32,
    hi: Int32, sign: Float32, e: FPtr, o: FPtr,
):
    var t = _hm_tid()
    if t < Int(k) * Int(nb):
        hm_block_eo_cell(r, codes, pr, Int(n), Int(k), Int(c), Int(nb), Int(a), Int(c0), Int(lo), Int(hi), sign,
                         e, o, t)


def hm_block_r_kernel(
    sd: FPtr, e: FPtr, o: FPtr, codes: IPtr, theta: FPtr, n: Int32, k: Int32, c: Int32, nb: Int32, a: Int64,
    c0: Int64, lo: Int32, hi: Int32, r: FPtr,
):
    var t = _hm_tid()
    if t < Int(hi) - Int(lo):
        hm_block_r_cell(sd, e, o, codes, theta, Int(n), Int(k), Int(c), Int(nb), Int(a), Int(c0), Int(lo), r, t)


def hm_objective_kernel(
    r: FPtr, dist: FPtr, e: FPtr, o: FPtr, codes: IPtr, theta: FPtr, n: Int32, k: Int32, c: Int32, nb: Int32,
    sigma: Float32, dst: FPtr,
):
    var t = _hm_tid()
    if t < Int(n):
        hm_objective_row(r, dist, e, o, codes, theta, Int(k), Int(c), Int(nb), sigma, dst, t)


def hm_centroid_kernel(r: FPtr, z: FPtr, n: Int32, k: Int32, d: Int32, y: FPtr):
    var t = _hm_tid()
    if t < Int(k) * Int(d):
        hm_centroid_cell(r, z, Int(n), Int(k), Int(d), y, t)


def hm_ridge_a_kernel(r: FPtr, codes: IPtr, lamb: FPtr, n: Int32, k: Int32, c: Int32, b1: Int32, am: FPtr):
    var t = _hm_tid()
    if t < Int(k) * Int(b1) * Int(b1):
        hm_ridge_a_cell(r, codes, lamb, Int(n), Int(k), Int(c), Int(b1), am, t)


def hm_ridge_rhs_kernel(r: FPtr, z: FPtr, codes: IPtr, n: Int32, k: Int32, d: Int32, c: Int32, b1: Int32,
                        rhs: FPtr):
    var t = _hm_tid()
    if t < Int(k) * Int(b1) * Int(d):
        hm_ridge_rhs_cell(r, z, codes, Int(n), Int(k), Int(d), Int(c), Int(b1), rhs, t)


def hm_chol_kernel(am: FPtr, k: Int32, b1: Int32):
    var t = _hm_tid()
    if t < Int(k):
        hm_chol_block(am, Int(b1), t)


def hm_solve_kernel(am: FPtr, rhs: FPtr, k: Int32, b1: Int32, d: Int32, w: FPtr):
    var t = _hm_tid()
    if t < Int(k) * Int(d):
        hm_solve_cell(am, rhs, Int(b1), Int(d), w, t)


def hm_correct_kernel(z: FPtr, r: FPtr, w: FPtr, codes: IPtr, n: Int32, k: Int32, d: Int32, c: Int32, b1: Int32,
                      dst: FPtr):
    var t = _hm_tid()
    if t < Int(n) * Int(d):
        hm_correct_cell(z, r, w, codes, Int(k), Int(d), Int(c), Int(b1), dst, t)
