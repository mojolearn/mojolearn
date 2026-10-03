# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/apple-fast-gap-cls1 (2026-10-03): NearestCentroid without the
Python passes over every row.

  * NC_CLS1_LABELS (FAST + Apple default, off with
    -D MOJOLEARN_NC_FAST_CLS1_LABELS_OFF; default since the M3 A/B, n=1,
    nearest-centroid taxi 105 -> 20.9 ms, quality identical): `fit` takes the
    native label encoder's int32 codes as they are and the class counts from
    the device (`op_nc_counts`: a block per NCC1_ROWS rows, a thread per class,
    then a thread per class over the blocks) instead of `_labels_of` (tolist,
    a dict lookup per row) and a Python counting loop.
The Python side reads the switch from `x_neighbors_cls1_flags` (bit 1)."""
from std.gpu import block_idx, thread_idx
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from std.python import PythonObject
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from x_neighbors.items import FP, IP
from x_neighbors.device_ops import xn_ctx, _buf, _buf_i, _down

comptime NC_CLS1_FAST_APPLE = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
comptime NC_CLS1_LABELS = NC_CLS1_FAST_APPLE and not is_defined["MOJOLEARN_NC_FAST_CLS1_LABELS_OFF"]()
comptime NCC1_ROWS = 4096
comptime NCC1_MAX_C = 256


def nc_cls1_flags_binding() raises -> PythonObject:
    """Bit 1: NC_CLS1_LABELS."""
    var f = 0
    comptime if NC_CLS1_LABELS:
        f |= 1
    return PythonObject(f)


def nc_counts_part_kernel(lab: IP, n: Int64, c_n: Int64, parts: IP):
    """Block b, thread c: rows of class c in row block b."""
    var b = Int(block_idx.x)
    var c = Int(thread_idx.x)
    var nn = Int(n)
    if c < Int(c_n):
        var lo = b * NCC1_ROWS
        var hi = min(nn, lo + NCC1_ROWS)
        var cnt = 0
        for i in range(lo, hi):
            if Int(lab.unsafe_load(i)) == c:
                cnt += 1
        parts.unsafe_store(b * Int(c_n) + c, Int32(cnt))


def nc_counts_red_kernel(parts: IP, nb: Int64, c_n: Int64, nk: FP):
    """Thread c: the count of class c over the row blocks, as float32."""
    var c = Int(block_idx.x) * NCC1_MAX_C + Int(thread_idx.x)
    if c < Int(c_n):
        var t = 0
        for b in range(Int(nb)):
            t += Int(parts.unsafe_load(b * Int(c_n) + c))
        nk.unsafe_store(c, Float32(t))


def op_nc_counts(lab: Int, nk: Int, n: Int, n_classes: Int) raises:
    """nk[c] = rows whose int32 label is c (exact below 2**24 rows a class)."""
    comptime if not NC_CLS1_LABELS:
        raise Error("nc_counts needs NC_CLS1_LABELS (FAST, Apple, no -D MOJOLEARN_NC_FAST_CLS1_LABELS_OFF)")
    else:
        if n_classes < 1 or n_classes > NCC1_MAX_C:
            raise Error("nc_counts: 1 to 256 classes")
        var ctx = xn_ctx()
        var d_lab = _buf_i(ctx, lab, n, True)
        var nb = max((n + NCC1_ROWS - 1) // NCC1_ROWS, 1)
        var d_parts = ctx.enqueue_create_buffer[DType.int32](nb * n_classes)
        var d_nk = _buf(ctx, nk, n_classes, False)
        ctx.enqueue_function[nc_counts_part_kernel](
            d_lab.unsafe_ptr(), Int64(n), Int64(n_classes), d_parts.unsafe_ptr(),
            grid_dim=nb, block_dim=NCC1_MAX_C,
        )
        ctx.enqueue_function[nc_counts_red_kernel](
            d_parts.unsafe_ptr(), Int64(nb), Int64(n_classes), d_nk.unsafe_ptr(),
            grid_dim=1, block_dim=NCC1_MAX_C,
        )
        _down(ctx, d_nk, nk, n_classes)
        ctx.synchronize()
        _ = d_lab^
        _ = d_parts^
        _ = d_nk^
