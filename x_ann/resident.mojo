# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The device-resident ann indexes (lane/py-dn-ann, 2026-09-28): IVF-PQ,
IVF-SQ, IVF-RaBitQ and CAGRA, the x_ann half of DEVIATION 1804's closure.

Each `search` used to copy the whole fitted index into host lists and upload
it (centroids, offsets, ids, codes, codebooks or the CAGRA dataset and
graph), plus an all-ones filter of one int32 per row when there was no
filter, and CAGRA checked every graph entry again. A handle does that once:
`x_ann_index_prepare` uploads the arrays (and, for CAGRA, admits the graph
with the same refusal), keeps a device all-ones filter, and every later
`x_ann_index_search` uploads only its queries (and a caller's filter, which
is per-call data). The search bodies are `ivf_pq_search_on`,
`ivf_sq_search_on`, `ivf_rabitq_search_on` and `cagra_search_on`, the same
functions the one-shot entries now call after their own upload, so the
bits are the same by construction.

The registry is `neighbors/resident_index.mojo`'s pattern: a process-wide
`_Global` keyed by an increasing integer handle, the extension holding the
GIL across prepare, search and release. The buffers live on the x_ann
process-lifetime context (`x_ann/device_ctx.mojo`), which outlives them.

THE ADDRESSES (mirrored in python/mojolearn/_expansion_ann.py):

  prepare(kind, addrs, params) -> handle
    kind 0 IVF-PQ   addrs centers, offsets, list_indices, codebooks, codes
                    params n, dim, n_lists, pq_dim, pq_bits
    kind 1 IVF-SQ   addrs centers, offsets, list_indices, vmin, delta, codes
                    params n, dim, n_lists
    kind 2 RaBitQ   addrs centers, offsets, list_indices, codes, norms, ips
                    params n, dim, n_lists, seed
    kind 3 CAGRA    addrs dataset, graph
                    params n, d, graph_degree
  search(handle, addrs, params)
    IVF kinds       addrs queries, out_d, out_i, out_n[, filter (n int32)]
                    params m, k, n_probes
    CAGRA           addrs queries, out_d, out_i
                    params m, k, itopk_size, search_width, max_iterations, n_seeds[, rs]
  release(handle)
"""
from std.ffi import _Global
from std.python import Python, PythonObject
from x_ann.switches import ANN3_PREPARE

from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE
from x_ann.abi import a_int, check_search, in_f32, in_i32, out_f32, out_i32, p_int
from x_ann.device_ctx import x_ann_ctx
from x_ann.io import upload_f32, upload_i32
from core.abs_sum_blocked import device_any_index_out_of_range
from x_ann.ivf_pq_core import F32P, I32P, IvfPqIndex, pq_len_of
from x_ann.ivf_rabitq_core import rq_pow2
from x_ann.ivf_pq_device import ivf_pq_search_on, ivf_sq_search_on, ivf_rabitq_search_on
from x_ann.ivf_pq_device import IvfPqDevice
from x_ann.ivf_scan_device import scan_gather_f32, scan_gather_i32
from x_ann.cagra_device import cagra_search_on

comptime KIND_PQ = 0
comptime KIND_SQ = 1
comptime KIND_RABITQ = 2
comptime KIND_CAGRA = 3


def _p[dt: DType](mut b: DeviceBuffer[dt]) -> MutPointer[Scalar[dt], MutAnyOrigin]:
    return b.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


struct AnnResident(Movable):
    """One index on the device. Unused slots hold a one-word buffer."""

    var kind: Int
    var n: Int
    var dim: Int
    var n_lists: Int
    var a: Int
    var b: Int
    var f0: DeviceBuffer[DType.float32]
    var f1: DeviceBuffer[DType.float32]
    var f2: DeviceBuffer[DType.float32]
    var i0: DeviceBuffer[DType.int32]
    var i1: DeviceBuffer[DType.int32]
    var i2: DeviceBuffer[DType.int32]
    var ones: DeviceBuffer[DType.int32]
    var offsets: List[Int32]
    # lane ann-apple3, behind `ANN3_PREPARE` (x_ann/switches.mojo): the codes
    # (and RaBitQ's norms and factors) in list order, gathered ONCE here by
    # the launches every search makes otherwise (`ivf_scan_search`); `pre`
    # says they are there (an index whose lists hold every row once: the
    # all-ones filter is then its own list-order copy).
    var pre: Bool
    var g_codes: DeviceBuffer[DType.int32]
    var g_a: DeviceBuffer[DType.float32]
    var g_b: DeviceBuffer[DType.float32]

    def __init__(out self, kind: Int, addrs: PythonObject, params: PythonObject) raises:
        var ctx = x_ann_ctx()
        self.kind = kind
        self.n = p_int(params, 0)
        self.dim = p_int(params, 1)
        self.a = 0
        self.b = 0
        var empty_f = List[Float32]()
        var empty_i = List[Int32]()
        if kind == KIND_CAGRA:
            self.n_lists = 0
            var deg = p_int(params, 2)
            if self.n < 2 or self.dim <= 0 or deg < 1:
                raise Error("CAGRA: need at least two rows, d >= 1 and graph_degree >= 1")
            self.a = deg
            var g = in_i32(addrs, 1, self.n * deg)
            # cpu3-neighbors: the graph's row-id refusal is one device pass
            # over the uploaded graph (one word back), not a host walk
            self.i0 = upload_i32(ctx, g)
            if device_any_index_out_of_range(ctx, self.i0, self.n * deg, self.n):
                raise Error("CAGRA search: the graph names a row outside the dataset")
            self.f0 = upload_f32(ctx, in_f32(addrs, 0, self.n * self.dim))
            self.f1 = upload_f32(ctx, empty_f)
            self.f2 = upload_f32(ctx, empty_f)
            self.i1 = upload_i32(ctx, empty_i)
            self.i2 = upload_i32(ctx, empty_i)
            self.ones = upload_i32(ctx, empty_i)
            self.offsets = List[Int32]()
            self.pre = False
            self.g_codes = upload_i32(ctx, empty_i)
            self.g_a = upload_f32(ctx, empty_f)
            self.g_b = upload_f32(ctx, empty_f)
            return
        self.n_lists = p_int(params, 2)
        var n = self.n
        var dim = self.dim
        var n_lists = self.n_lists
        if n < 1 or dim < 1 or n_lists < 1:
            raise Error("x_ann: an index needs n, dim and n_lists of at least 1")
        self.offsets = in_i32(addrs, 1, n_lists + 1)
        self.f0 = upload_f32(ctx, in_f32(addrs, 0, n_lists * dim))
        self.i0 = upload_i32(ctx, self.offsets)
        # lane ann-apple3: the ids and the codes are held in locals until the
        # list-order gathers are enqueued, then moved into the entry
        var ids = upload_i32(ctx, in_i32(addrs, 2, n))
        var n_slots = Int(self.offsets[n_lists])
        var pre = n_slots == n
        comptime if not ANN3_PREPARE:
            pre = False
        var gs = n_slots if pre else 0
        self.pre = pre
        if kind == KIND_PQ:
            var pq_dim = p_int(params, 3)
            var pq_bits = p_int(params, 4)
            self.a = pq_dim
            self.b = pq_bits
            var pq_len = pq_len_of(dim, pq_dim)
            self.f1 = upload_f32(ctx, in_f32(addrs, 3, pq_dim * (1 << pq_bits) * pq_len))
            self.f2 = upload_f32(ctx, empty_f)
            var codes = upload_i32(ctx, in_i32(addrs, 4, n * pq_dim))
            self.g_codes = scan_gather_i32(ctx, _p(codes), _p(ids), gs, pq_dim)
            self.g_a = upload_f32(ctx, empty_f)
            self.g_b = upload_f32(ctx, empty_f)
            self.i2 = codes^
            self.i1 = ids^
        elif kind == KIND_SQ:
            self.f1 = upload_f32(ctx, in_f32(addrs, 3, dim))
            self.f2 = upload_f32(ctx, in_f32(addrs, 4, dim))
            var codes = upload_i32(ctx, in_i32(addrs, 5, n * dim))
            self.g_codes = scan_gather_i32(ctx, _p(codes), _p(ids), gs, dim)
            self.g_a = upload_f32(ctx, empty_f)
            self.g_b = upload_f32(ctx, empty_f)
            self.i2 = codes^
            self.i1 = ids^
        elif kind == KIND_RABITQ:
            self.a = p_int(params, 3)
            var words = (rq_pow2(dim) + 31) // 32
            var codes = upload_i32(ctx, in_i32(addrs, 3, n * words))
            var norms = upload_f32(ctx, in_f32(addrs, 4, n))
            var ips = upload_f32(ctx, in_f32(addrs, 5, n))
            self.g_codes = scan_gather_i32(ctx, _p(codes), _p(ids), gs, words)
            self.g_a = scan_gather_f32(ctx, _p(norms), _p(ids), gs)
            self.g_b = scan_gather_f32(ctx, _p(ips), _p(ids), gs)
            self.i2 = codes^
            self.i1 = ids^
            self.f1 = norms^
            self.f2 = ips^
        else:
            raise Error("x_ann_index_prepare: unknown index kind " + String(kind))
        # the no-filter filter, once: one int32 1 per row, what
        # `_ann_mask(None)` sent on every search
        self.ones = upload_i32(ctx, List[Int32](length=n, fill=Int32(1)))

    def __init__(out self, *, var built: IvfPqIndex, var dev: IvfPqDevice) raises:
        """lane fg-ivf A7 (`IVF_PQ_RESIDENT_FIT`): a KIND_PQ entry from a
        resident build (`ivf_pq_build_device_resident`): the centres, the
        codebooks and the row-order codes are the build's own device buffers
        (moved in, the words prepare would upload from the downloaded
        arrays); the offsets and the ids go up from the build's host lists
        (n_lists + 1 and n words); then the same list-order gather and
        all-ones filter as prepare's KIND_PQ branch, so a search reads the
        same words either way."""
        var ctx = x_ann_ctx()
        var empty_f = List[Float32]()
        self.kind = KIND_PQ
        self.n = built.n_rows
        self.dim = built.dim
        self.n_lists = built.n_lists
        self.a = built.pq_dim
        var pq_bits = 0
        while (1 << pq_bits) < built.n_codes:
            pq_bits += 1
        self.b = pq_bits
        var n = self.n
        if n < 1 or self.dim < 1 or self.n_lists < 1:
            raise Error("x_ann: an index needs n, dim and n_lists of at least 1")
        if len(built.offsets) != self.n_lists + 1 or len(built.list_indices) != n:
            raise Error("x_ann_ivf_pq_build_resident: the built lists disagree with n and n_lists")
        self.offsets = built.offsets.copy()
        self.i0 = upload_i32(ctx, self.offsets)
        var ids = upload_i32(ctx, built.list_indices)
        var n_slots = Int(self.offsets[self.n_lists])
        var pre = n_slots == n
        comptime if not ANN3_PREPARE:
            pre = False
        var gs = n_slots if pre else 0
        self.pre = pre
        self.f0 = ctx.enqueue_create_buffer[DType.float32](1)
        self.f1 = ctx.enqueue_create_buffer[DType.float32](1)
        self.i2 = ctx.enqueue_create_buffer[DType.int32](1)
        swap(self.f0, dev.dcenters)
        swap(self.f1, dev.dcb)
        swap(self.i2, dev.dcodes)
        self.f2 = upload_f32(ctx, empty_f)
        self.g_codes = scan_gather_i32(ctx, _p(self.i2), _p(ids), gs, built.pq_dim)
        self.g_a = upload_f32(ctx, empty_f)
        self.g_b = upload_f32(ctx, empty_f)
        self.i1 = ids^
        self.ones = upload_i32(ctx, List[Int32](length=n, fill=Int32(1)))
        _ = dev^
        _ = built^


struct AnnRegistry(Movable):
    var entries: Dict[Int, AnnResident]
    var next_id: Int

    def __init__(out self):
        self.entries = Dict[Int, AnnResident]()
        self.next_id = 1


comptime ANN_REGISTRY = _Global[
    StorageType=AnnRegistry,
    name=(
        "MojoXAnnResidentIdentical" if GLOBAL_NUMERIC_MODE == 1 else
        "MojoXAnnResidentDeterministic" if GLOBAL_NUMERIC_MODE == 2 else
        "MojoXAnnResidentFast"
    ),
    init_fn=AnnRegistry.__init__,
]


def x_ann_index_prepare_binding(kind: PythonObject, addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    var state = ANN_REGISTRY.get_or_create_ptr()
    if state[].next_id == 9223372036854775807:
        raise Error("resident ann index handle space exhausted")
    var entry = AnnResident(Int(py=kind), addrs, params)
    var handle = state[].next_id
    state[].next_id += 1
    state[].entries[handle] = entry^
    return PythonObject(handle)


def x_ann_register_pq_built(var built: IvfPqIndex, var dev: IvfPqDevice) raises -> Int:
    """lane fg-ivf A7: register a resident IVF-PQ build; returns its handle
    (the same handle space and release as `x_ann_index_prepare`)."""
    var state = ANN_REGISTRY.get_or_create_ptr()
    if state[].next_id == 9223372036854775807:
        raise Error("resident ann index handle space exhausted")
    var entry = AnnResident(built=built^, dev=dev^)
    var handle = state[].next_id
    state[].next_id += 1
    state[].entries[handle] = entry^
    return handle


def x_ann_index_export_codes_binding(handle: PythonObject, address: PythonObject, count: PythonObject) raises -> PythonObject:
    """lane fg-ivf A7: write a resident IVF-PQ index's row-order codes
    (n x pq_dim int32) into the caller's array: one device copy, outside fit
    and search (`IVFPQIndex.codes_` on first read)."""
    var state = ANN_REGISTRY.get_or_create_ptr()
    var h = Int(py=handle)
    if h not in state[].entries:
        raise Error("unknown or released resident ann index handle")
    ref e = state[].entries[h]
    if e.kind != KIND_PQ:
        raise Error("x_ann_index_export_codes: the handle is not an IVF-PQ index")
    var c = Int(py=count)
    if c != e.n * e.a:
        raise Error("x_ann_index_export_codes: count must be n * pq_dim")
    var ctx = x_ann_ctx()
    ctx.enqueue_copy(dst_ptr=I32P(unsafe_from_address=Int(py=address)), src_buf=e.i2)
    ctx.synchronize()
    _ = ctx^
    return PythonObject(0)


def x_ann_index_release_binding(handle: PythonObject) raises -> PythonObject:
    var state = ANN_REGISTRY.get_or_create_ptr()
    var h = Int(py=handle)
    if h not in state[].entries:
        raise Error("unknown or released resident ann index handle")
    var released = state[].entries.pop(h)
    _ = released^
    return PythonObject(0)


def x_ann_index_search_binding(handle: PythonObject, addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    var state = ANN_REGISTRY.get_or_create_ptr()
    var h = Int(py=handle)
    if h not in state[].entries:
        raise Error("unknown or released resident ann index handle")
    ref e = state[].entries[h]
    var ctx = x_ann_ctx()
    var m = p_int(params, 0)
    var k = p_int(params, 1)
    var od = List[Float32]()
    var oi = List[Int32]()
    if e.kind == KIND_CAGRA:
        var L = p_int(params, 2)
        var width = p_int(params, 3)
        var max_iter = p_int(params, 4)
        var n_seeds = p_int(params, 5)
        var rs = p_int(params, 6) if len(params) > 6 else 0
        if rs < 0:
            raise Error("CAGRA search: rs must be >= 0")
        if m <= 0 or k <= 0 or L < k or width < 1 or max_iter < 1 or n_seeds < 1 or n_seeds > e.n:
            raise Error("CAGRA search: need k >= 1, itopk_size >= k, search_width >= 1, max_iterations >= 1, 1 <= n_seeds <= n")
        var q = in_f32(addrs, 0, m * e.dim)
        cagra_search_on(ctx, _p(e.f0), e.n, e.dim, _p(e.i0), e.a, q, m, k, L, width, max_iter, n_seeds, od, oi, rs)
        out_f32(od, addrs, 1)
        out_i32(oi, addrs, 2)
        _ = ctx^
        return PythonObject(m)
    var n_probes = p_int(params, 2)
    check_search(e.n_lists, m, k, n_probes)
    var q = in_f32(addrs, 0, m * e.dim)
    var on = List[Int32]()
    var filtered = len(addrs) > 4
    var dmask = upload_i32(ctx, in_i32(addrs, 4, e.n) if filtered else List[Int32]())
    var mask = _p(dmask) if filtered else _p(e.ones)
    # the prepared list-order arrays; an unfiltered search's mask is the
    # all-ones filter, which is its own list-order copy
    var mask_pre = e.pre and not filtered
    if e.kind == KIND_PQ:
        ivf_pq_search_on(ctx, _p(e.f0), _p(e.i0), _p(e.i1), _p(e.f1), _p(e.i2), mask, e.offsets, e.n_lists,
                         e.dim, e.a, e.b, q, m, k, n_probes, od, oi, on,
                         e.pre, _p(e.g_codes), _p(e.g_a), _p(e.g_b), mask_pre)
    elif e.kind == KIND_SQ:
        ivf_sq_search_on(ctx, _p(e.f0), _p(e.i0), _p(e.i1), _p(e.f1), _p(e.f2), _p(e.i2), mask, e.offsets,
                         e.n_lists, e.dim, q, m, k, n_probes, od, oi, on,
                         e.pre, _p(e.g_codes), _p(e.g_a), _p(e.g_b), mask_pre)
    else:
        ivf_rabitq_search_on(ctx, _p(e.f0), _p(e.i0), _p(e.i1), _p(e.i2), _p(e.f1), _p(e.f2), mask, e.offsets,
                             e.n_lists, e.dim, e.a, q, m, k, n_probes, od, oi, on,
                             e.pre, _p(e.g_codes), _p(e.g_a), _p(e.g_b), mask_pre)
    out_f32(od, addrs, 1)
    out_i32(oi, addrs, 2)
    out_i32(on, addrs, 3)
    _ = dmask^
    _ = ctx^
    return PythonObject(m)
