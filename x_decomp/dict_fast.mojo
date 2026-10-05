# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The dictionary update of (MiniBatch)DictionaryLearning on resident
matrices, FAST on Apple (lane/apple-fast-gap-clus3, 2026-10-03). Switch:
`DECOMP_FAST_DICT_DEV`, default on in FAST (Apple since apple-fast-gap-clus3, every vendor since cpu2-l8-decomp)
(`-D MOJOLEARN_DECOMP_FAST_DICT_DEV_OFF` turns it off): only that build exports `x_decomp_dev_dict_update`, and
python/mojolearn/_expansion_decomp.py `_update_dict` takes it when the
binding has it. IDENTICAL compiles none of this.

Cause: `_update_dict`'s atom loop (sklearn `_update_dict`, block coordinate
descent over the atoms in order) builds the dictionary on the HOST each
atom: `_vstack(*rows)` downloads the row the previous atom made, and
`A.rows(j)`, `B.cols(j)` and the stacked D go up again for a 1 x nc by
nc x m product, then about ten elementwise launches; ~16 round trips a step
at n_components = 16, and MiniBatchDictionaryLearning runs up to
max_iter * n / batch_size steps (3,910 on Istella).

Here D (nc x m), A (nc x nc) and B (m x nc) stay on the device; atom j is
two launches with no sync:
  * `_dd_update_kernel`: a thread per feature f,
    v = D[j, f] + (B[f, j] - sum_i A[j, i] D[i, f]) / A[j, j] (i ascending,
    the rows before j already updated, as the loop does), written in place
    (thread f alone reads and writes column f), and its block's sum of v^2;
  * `_dd_norm_kernel`: a thread per feature folds the block sums in block
    order, D[j, f] /= max(sqrt(sum), 1).
The caller checks the diagonal of A first (one read of A) and keeps the
loop for an unused atom (A[j, j] <= 1e-6, the Philox resample) and for
positive_dict. FAST: the product's fold order and the norm's differ from
the elementwise chain's (quality, not bits).
"""
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from std.python import PythonObject
from std.sys.compile import is_defined
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, ftz, identical_div, identical_mul_add, identical_sqrt
from x_decomp.cells import F32Ptr
from x_decomp.device import xd_ctx
from x_decomp.resident import _id, _n, _ptr, pool_alloc, pool_free, X_DECOMP_POOL

comptime DECOMP_FAST_DICT_DEV = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and not is_defined["MOJOLEARN_DECOMP_FAST_DICT_DEV_OFF"]()
)  # FAST + Apple default since the M3 A/B clus3-mbdl-dictdev-istella (n=1):
# mb-dict-learning istella 6832 -> 5193 ms (-24%), sparsity .08636 and
# reconstruction error .6483 identical; -D MOJOLEARN_DECOMP_FAST_DICT_DEV_OFF turns it off.
# Lane cpu2-l8-decomp (2026-10-04, re-audit L8): FAST on every vendor (NVIDIA
# and AMD FAST ran the per-atom host loop); IDENTICAL keeps the kit loop of
# `_update_dict`, now resident (same words as the host column).
comptime DD_TPB = 64


def _dd_update_kernel(dm: F32Ptr, a: F32Ptr, b: F32Ptr, nc: Int32, m: Int32, j: Int32, part: F32Ptr):
    var tid = Int(thread_idx.x)
    var f = Int(block_idx.x) * DD_TPB + tid
    var NC = Int(nc)
    var M = Int(m)
    var J = Int(j)
    var red = stack_allocation[DD_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var v = Float32(0)
    if f < M:
        var acc = Float32(0)
        for i in range(NC):
            acc = ftz(identical_mul_add(ftz(a[J * NC + i]), ftz(dm[i * M + f]), acc))
        var upd = ftz(ftz(b[f * NC + J]) - acc)
        v = ftz(ftz(dm[J * M + f]) + ftz(identical_div(upd, ftz(a[J * NC + J]))))
        dm[J * M + f] = v
    red[tid] = ftz(v * v)
    barrier()
    var off = DD_TPB // 2
    while off > 0:
        if tid < off:
            red[tid] = ftz(red[tid] + red[tid + off])
        barrier()
        off //= 2
    if tid == 0:
        part[Int(block_idx.x)] = red[0]


def _dd_norm_kernel(dm: F32Ptr, m: Int32, j: Int32, part: F32Ptr, nblk: Int32):
    var f = Int(block_idx.x) * DD_TPB + Int(thread_idx.x)
    var M = Int(m)
    if f < M:
        var s = Float32(0)
        for q in range(Int(nblk)):
            s = ftz(s + part[q])
        var nrm = identical_sqrt(s)
        if nrm < Float32(1):
            nrm = Float32(1)
        var J = Int(j)
        dm[J * M + f] = ftz(identical_div(dm[J * M + f], nrm))


def dict_update_dev(isrc: Int, ia: Int, ib: Int, idst: Int, nc: Int, m: Int) raises -> Int:
    """dst (nc x m) = src updated atom by atom (see the module docstring);
    p = [nc, m]. Every A[j, j] must be > 1e-6 (the caller checks)."""
    comptime if DECOMP_FAST_DICT_DEV:
        var cells = nc * m
        if cells <= 0:
            return 0
        _ = _ptr(isrc, cells)
        var pd = _ptr(idst, cells)
        var pa = _ptr(ia, nc * nc)
        var pb = _ptr(ib, m * nc)
        var ctx = xd_ctx()
        var pool = X_DECOMP_POOL.get_or_create_ptr()
        ctx.enqueue_copy(
            dst_buf=pool[].bufs[idst].create_sub_buffer[DType.float32](0, cells),
            src_buf=pool[].bufs[isrc].create_sub_buffer[DType.float32](0, cells),
        )
        var nblk = (m + DD_TPB - 1) // DD_TPB
        var sid = pool_alloc(nblk)
        var pp = _ptr(sid, nblk)
        for j in range(nc):
            ctx.enqueue_function[_dd_update_kernel](
                pd, pa, pb, Int32(nc), Int32(m), Int32(j), pp, grid_dim=nblk, block_dim=DD_TPB,
            )
            ctx.enqueue_function[_dd_norm_kernel](
                pd, Int32(m), Int32(j), pp, Int32(nblk), grid_dim=nblk, block_dim=DD_TPB,
            )
        # the stream orders a later reuse of the scratch behind these launches
        pool_free(sid)
        return cells
    raise Error("x_decomp: dev_dict_update needs a FAST build without -D MOJOLEARN_DECOMP_FAST_DICT_DEV_OFF")

def dev_dict_update_py(
    src: PythonObject, a: PythonObject, b: PythonObject, dst: PythonObject, p: PythonObject
) raises -> PythonObject:
    """dst (nc x m) = src updated atom by atom (see the module docstring);
    p = [nc, m]. Every A[j, j] must be > 1e-6 (the caller checks)."""
    return PythonObject(dict_update_dev(_id(src), _id(a), _id(b), _id(dst), _n(p, 0), _n(p, 1)))
