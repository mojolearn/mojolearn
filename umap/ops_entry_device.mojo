# SPDX-License-Identifier: Apache-2.0
"""The device column of the UMAP operations table
(core/neighbors_entry_table_umap.mojo): the UMAP bodies of the metrics
binding (bindings/_mojolearn_metrics.mojo), with the same refusals, the
same kernels and the same words. Both the table entry and the binding's
Python exports (`umap_fit_transform`, `umap_fit_transform_ex`,
`umap_transform`, kept for the unconverted callers named there) run
`umap_ops_device_words`. Called with the GIL already released; no Python.
"""
from std.memory import unsafe_memcpy
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.hostptr import f32_ptr, read_f32
from core.neural_context import process_ctx
from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from std.math import ceildiv
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot
from core.neighbors_entry_table_umap import (
    UMAP_OP_FIT, UMAP_OP_FIT_EX, UMAP_OP_TRANSFORM, UMAP_OP_ROWS_READ, umap_ops_check_layout, umap_ops_check_spans,
)
from metrics.metrics_entries_impl import device_finite_f32_list
from umap.estimator import fit_transform as umap_fit_transform
from umap.sparse_estimator import sparse_fit_transform, sparse_fit_transform_device
from umap.transform import transform as umap_transform, transform_device as umap_transform_device
from max.gpu.host import DeviceBuffer
from core.fold_x_word import fold_x_slot
from core.metrics_store import metrics_store_buffer
from umap.ops_entry_words import (
    UmapWordPtr, UmapFloatPtr, umap_fit_config, umap_fit_ex_config, umap_transform_config,
)

#: The metrics image's ONE process-lifetime DeviceContext: the same slot name
#: metrics/estimator.mojo and the spectral modules use, so UMAP and the other
#: metrics entries share the one context.
comptime UMAP_DEVCTX_SLOT = "MojolearnMetricsContextIdenticalV3" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnMetricsContextFastV3"


def _load(address: Int, n: Int) raises -> List[Float32]:
    # The binding's `_load_f32`.
    return read_f32(address, max(0, n))


def _device_finite(values: List[Float32]) raises -> Bool:
    # The binding's `_device_finite_f32`: one device scan, one status word.
    if len(values) == 0:
        return True
    return device_finite_f32_list(process_ctx[UMAP_DEVCTX_SLOT](), values)


def _store(output: MutPointer[Float32, MutUntrackedOrigin], values: List[Float32]):
    # The binding's `_store_f32_list`: one memcpy, no per-value loop.
    if len(values) > 0:
        unsafe_memcpy(dest=output, src=values.unsafe_ptr(), count=len(values))
    _ = len(values)


comptime _SW_TPB = 256
comptime _SW_U32P = MutPointer[UInt32, MutAnyOrigin]
comptime _SW_I32P = MutPointer[Int32, MutAnyOrigin]


def umap_words_differ_kernel(x: _SW_U32P, y: _SW_U32P, flag: _SW_I32P, count: Int32):
    """flag[0] = 1 when any of the first `count` 32-bit words differ (the bytes compare of `mani_same_words`)."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(count):
        if x.unsafe_load(t) != y.unsafe_load(t):
            flag.unsafe_store(0, Int32(1))


def _same_words_device(ctx: DeviceContext, x: DeviceBuffer[DType.float32], y: DeviceBuffer[DType.float32],
                       count: Int) raises -> Bool:
    """`x.tobytes() == y.tobytes()` over the first `count` words, on the device: one launch, one word home."""
    if count <= 0:
        return True
    var dflag = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_memset(dflag, Int32(0))
    ctx.enqueue_function[umap_words_differ_kernel](
        _SW_U32P(unsafe_from_address=Int(x.unsafe_ptr())), _SW_U32P(unsafe_from_address=Int(y.unsafe_ptr())),
        _SW_I32P(unsafe_from_address=Int(dflag.unsafe_ptr())), Int32(count),
        grid_dim=ceildiv(count, _SW_TPB), block_dim=_SW_TPB,
    )
    var hflag = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_buf=hflag, src_buf=dflag)
    ctx.synchronize()
    var differ = hflag.unsafe_ptr().unsafe_load(0) != Int32(0)
    _ = hflag^
    _ = dflag^
    _ = x.unsafe_ptr()
    _ = y.unsafe_ptr()
    return not differ


def _rows_buffer(ctx: DeviceContext, word: Int, count: Int) raises -> DeviceBuffer[DType.float32]:
    """A row block's device buffer: a metrics-store slot (negative word), or the host words uploaded once."""
    if word < 0:
        return metrics_store_buffer(fold_x_slot(word), count)
    var buf = ctx.enqueue_create_buffer[DType.float32](max(count, 1))
    if count > 0:
        ctx.enqueue_copy(dst_buf=buf, src_ptr=f32_ptr(word))
    return buf^


def umap_ops_device_run(operation: Int, a: UmapWordPtr, p: UmapWordPtr, np: Int, f: UmapFloatPtr) raises -> Int:
    if operation == UMAP_OP_FIT:
        var config = umap_fit_config(p, np, f)
        var n = p[0]
        var d = p[1]
        var output = f32_ptr(a[1])
        var ctx = process_ctx[UMAP_DEVCTX_SLOT]()
        var embedding: List[Float32]
        if a[0] < 0:
            # THE SLOT FORM (lane vdf-scope-cluster7): X is a metrics-store slot (a fold taken on the device);
            # the same body as the host form after its upload, so the same bits.
            var d_x = metrics_store_buffer(fold_x_slot(a[0]), n * d)
            embedding = sparse_fit_transform_device(ctx, d_x, n, d, config)
            _ = d_x^
        else:
            var x = _load(a[0], n * d)
            embedding = umap_fit_transform(ctx, x, n, d, config)
        ctx.synchronize()
        if len(embedding) != n * config.n_components:
            raise Error("UMAP returned an unexpected embedding shape")
        if not _device_finite(embedding):
            raise Error("UMAP returned a non-finite embedding")
        _store(output, embedding)
        return config.n_components
    if operation == UMAP_OP_FIT_EX:
        var config = umap_fit_ex_config(a, p, f)
        var n = p[0]
        var d = p[1]
        var tkind = p[8]
        var tdims = p[9]
        var tk = p[10]
        var tw = Float32(f[9])
        var init_addr = a[2]
        var target_addr = a[3]
        var init = List[Float32]()
        if init_addr != 0:
            init = _load(init_addr, n * config.n_components)
            if not _device_finite(init):
                raise Error("UMAP initial embedding must be finite")
        var target = List[Float32]()
        if tkind != 0:
            target = _load(target_addr, n * tdims)
        var output = f32_ptr(a[1])
        var embedding = List[Float32]()
        if a[0] < 0:
            # THE SLOT FORM (lane vdf-scope-cluster7): X is a metrics-store slot.
            var d_x = metrics_store_buffer(fold_x_slot(a[0]), n * d)
            with process_ctx[UMAP_DEVCTX_SLOT]() as ctx:
                embedding = sparse_fit_transform_device(ctx, d_x, n, d, config, init, target, tkind, tdims, tk, tw)
            _ = d_x^
        else:
            var x = _load(a[0], n * d)
            with process_ctx[UMAP_DEVCTX_SLOT]() as ctx:
                embedding = sparse_fit_transform(ctx, x, n, d, config, init, target, tkind, tdims, tk, tw)
        if len(embedding) != n * config.n_components:
            raise Error("UMAP returned an unexpected embedding shape")
        if not _device_finite(embedding):
            raise Error("UMAP returned a non-finite embedding")
        _store(output, embedding)
        return config.n_components
    if operation == UMAP_OP_ROWS_READ:
        # a metrics-store row slot into host words (a fold-fitted state's first export of its rows)
        if a[0] >= 0:
            raise Error("umap ops: a rows read takes a store slot word")
        var count = p[0]
        if count > 0:
            var ctx = process_ctx[UMAP_DEVCTX_SLOT]()
            var held = metrics_store_buffer(fold_x_slot(a[0]), count)
            var head = held.create_sub_buffer[DType.float32](0, count)
            ctx.enqueue_copy(dst_ptr=f32_ptr(a[1]), src_buf=head)
            ctx.synchronize()
            _ = head^
            _ = held^
        return count
    # UMAP_OP_TRANSFORM (the layout check refused any other id)
    var config = umap_transform_config(p, np, f)
    var n = p[0]
    var rows = p[1]
    var d = p[2]
    var fitted = _load(a[1], n * config.n_components)
    var output = f32_ptr(a[3])
    var embedding = List[Float32]()
    if a[0] < 0 or a[2] < 0:
        # THE SLOT FORM (lane vdf-scope-cluster7): the training rows (a fitted-rows slot) and / or the queries (a
        # fold slot) are metrics-store slots; a host word of the other goes up once. `transform_device` is the
        # host form's own body after its uploads.
        var ctx = process_ctx[UMAP_DEVCTX_SLOT]()
        var dx = _rows_buffer(ctx, a[0], n * d)
        var dq = _rows_buffer(ctx, a[2], rows * d)
        if rows == n and _same_words_device(ctx, dx, dq, n * d):
            # the member's same-input reuse (`umap_transform_reuses_fit`), made here on the device words
            embedding = fitted.copy()
        else:
            embedding = umap_transform_device(ctx, dx, dq, fitted, n, rows, d, config)
        _ = dx^
        _ = dq^
    else:
        var training = _load(a[0], n * d)
        var queries = _load(a[2], rows * d)
        with process_ctx[UMAP_DEVCTX_SLOT]() as ctx:
            embedding = umap_transform(ctx, training, fitted, queries, n, rows, d, config)
    if len(embedding) != rows * config.n_components:
        raise Error("UMAP transform returned an unexpected shape")
    if not _device_finite(embedding):
        raise Error("UMAP transform returned a non-finite embedding")
    _store(output, embedding)
    return config.n_components


def umap_ops_device_words(operation: Int, addresses: List[Int], ints: List[Int], floats: List[Float64]) raises -> Int:
    """The binding exports' path: the same layout check and body as the
    table entry, over the words they unpacked."""
    umap_ops_check_layout(operation, len(addresses), len(ints), len(floats))
    var a = UmapWordPtr(unsafe_from_address=Int(addresses.unsafe_ptr()))
    var p = UmapWordPtr(unsafe_from_address=Int(ints.unsafe_ptr()))
    var f = UmapFloatPtr(unsafe_from_address=Int(floats.unsafe_ptr()))
    var columns = umap_ops_device_run(operation, a, p, len(ints), f)
    _ = addresses
    _ = ints
    _ = floats
    return columns


def umap_ops_device_v1(operation: Int, aa: Int, na: Int, pp: Int, np: Int,
    ff: Int, nf: Int, result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        umap_ops_check_spans(operation, aa, na, pp, np, ff, nf, result)
        var a = UmapWordPtr(unsafe_from_address=aa)
        var p = UmapWordPtr(unsafe_from_address=pp)
        var f = UmapFloatPtr(unsafe_from_address=ff)
        var r = UmapWordPtr(unsafe_from_address=result)
        r[] = umap_ops_device_run(operation, a, p, np, f)
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct UmapOpsDeviceAPI(Defaultable, Movable):
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(umap_ops_device_v1)


comptime _OPS_SLOT = "MojolearnUmapOpsDeviceIdenticalV1" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnUmapOpsDeviceFastV1"
# The publisher identity of this image's table (its entry list's address).
comptime UMAP_OPS_DEVICE_API = SharedSlot[UmapOpsDeviceAPI, _OPS_SLOT]
