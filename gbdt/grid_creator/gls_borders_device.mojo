"""The device driver of `gls_borders.mojo` (lane hr2-gbdt-host): the float
columns' grids, every step on the device, the columns in chunks so the
staging stays bounded. Returns, per float column, its borders WITH the NaN
sentinel (`calc_quantization`'s output) and the NaN mode it resolved to.

cpu-gpu-cleanup w2-trees: every `feature_border_type` runs here. The
subsample, keys, sort and NaN modes are GreedyLogSum's; the per-column
search is `gls_column` for GreedyLogSum and `border_types.mojo`'s
`border_type_column` for the six others (its table built by
`border_table_kernel`, its scratch sized per type)."""
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from core.device_zero import enqueue_fill
from gbdt.gpu_data.sym_feat_switches import GBDT_QUANT_DEVICE
from gbdt.options.data_processing_options import (
    NAN_MODE_FORBIDDEN,
    NAN_MODE_MAX,
    NAN_MODE_MIN,
)
from gbdt.gpu_util.kernel.reorder_one_bit import REORDER_BLOCK
from gbdt.gpu_util.kernel.segmented_sort import launch_segmented_radix_sort
from gbdt.grid_creator.binarization import (
    BORDER_TYPE_GREEDY_LOG_SUM,
    BORDER_TYPE_MAX_LOG_SUM,
    BORDER_TYPE_MIN_ENTROPY,
)
from gbdt.grid_creator.border_types import (
    border_table_kernel,
    border_types_columns_kernel,
    exact_i32_words,
)
from gbdt.grid_creator.gls_borders import (
    GLS_BLOCK,
    GLS_MODE_REFUSED,
    border_keys_kernel,
    border_sample_kernel,
    gls_budget_kernel,
    gls_columns_kernel,
    gls_log_table_kernel,
)


def _blocks(n: Int) -> Int:
    return max(1, min((n + GLS_BLOCK - 1) // GLS_BLOCK, 65535))


def device_float_borders(
    ctx: DeviceContext,
    cols: List[MutPointer[Float32, MutUntrackedOrigin]],
    n_rows: Int,
    sample_n: Int,
    border_count: Int,
    nan_mode_option: Int,
    sample_key: UInt64,
    border_type: Int = BORDER_TYPE_GREEDY_LOG_SUM,
    # lane/apple-fast-sym-feat (`GBDT_QUANT_DEVICE`): the float columns
    # already resident on the device, column-major in `cols` order; the
    # chunk loop then reads them in place instead of uploading `cols`.
    dev_cols: Optional[MutPointer[Float32, MutAnyOrigin]] = None,
    # lane cpu4-gbdt: separately held resident columns (one buffer each,
    # `n_rows` long; `train`'s CTR columns), copied device to device
    dev_col_bufs: List[DeviceBuffer[DType.float32]] = List[
        DeviceBuffer[DType.float32]
    ](),
) raises -> Tuple[List[List[Float32]], List[Int]]:
    """`dev_col_bufs` (lane cpu4-gbdt): when non-empty, the columns are
    already on the device (each `n_rows` long) and `cols` is ignored; each chunk
    copies them device to device instead of uploading from host memory.
    Same kernels after the copy, so the same borders."""
    var resident = len(dev_col_bufs) > 0
    var n_float = len(dev_col_bufs) if resident else len(cols)
    var borders = List[List[Float32]]()
    var modes = List[Int]()
    if n_float == 0 or n_rows <= 0:
        return (borders^, modes^)
    var use_dev = False
    comptime if GBDT_QUANT_DEVICE:
        use_dev = dev_cols.__bool__()
    var sn = min(sample_n, n_rows)
    var sampled = sn < n_rows
    var span = max(n_rows, sn)
    var chunk = max(1, min(n_float, (1 << 25) // span))
    var heap_cap = border_count + 2
    var out_cap = border_count + 1
    var gls = border_type == BORDER_TYPE_GREEDY_LOG_SUM
    var exact = (
        border_type == BORDER_TYPE_MAX_LOG_SUM
        or border_type == BORDER_TYPE_MIN_ENTROPY
    )
    # the exact dynamic program's scratch per column (`exact_column`):
    # 4 * sn soft doubles, the Int32 planes, sn floats; the chunk shrinks
    # so the chunk's scratch stays under 1 GiB
    var f64_stride = 4 * sn if exact else 0
    var i32_stride = exact_i32_words(sn, border_count) if exact else 0
    var uniq_stride = sn if exact else 0
    if exact:
        var per_col = 8 * f64_stride + 4 * i32_stride + 4 * uniq_stride
        chunk = max(1, min(chunk, (1 << 30) // max(1, per_col)))

    # the type's table: `log(w + 1e-8)` for GreedyLogSum, the type's score
    # or penalty for the others (`border_table_entry`)
    var d_logtab = ctx.enqueue_create_buffer[DType.uint64](sn + 1)
    if gls:
        ctx.enqueue_function[gls_log_table_kernel](
            d_logtab.unsafe_ptr(), Int32(sn + 1),
            grid_dim=_blocks(sn + 1), block_dim=GLS_BLOCK,
        )
    else:
        ctx.enqueue_function[border_table_kernel](
            d_logtab.unsafe_ptr(), Int32(sn + 1), Int32(border_type),
            grid_dim=_blocks(sn + 1), block_dim=GLS_BLOCK,
        )
    var d_f64s = ctx.enqueue_create_buffer[DType.uint64](max(1, chunk * f64_stride))
    var d_i32s = ctx.enqueue_create_buffer[DType.int32](max(1, chunk * i32_stride))
    var d_uniq = ctx.enqueue_create_buffer[DType.float32](max(1, chunk * uniq_stride))
    var d_idx = ctx.enqueue_create_buffer[DType.uint32](max(sn, 1))
    if sampled:
        ctx.enqueue_function[border_sample_kernel](
            d_idx.unsafe_ptr(), Int32(sn), Int32(n_rows), sample_key,
            grid_dim=_blocks(sn), block_dim=GLS_BLOCK,
        )
    var d_cols = ctx.enqueue_create_buffer[DType.float32](
        1 if use_dev else chunk * n_rows
    )
    var d_keys = ctx.enqueue_create_buffer[DType.uint32](chunk * sn)
    var d_vals = ctx.enqueue_create_buffer[DType.uint32](chunk * sn)
    var d_tkeys = ctx.enqueue_create_buffer[DType.uint32](chunk * sn)
    var d_tvals = ctx.enqueue_create_buffer[DType.uint32](chunk * sn)
    var d_offs = ctx.enqueue_create_buffer[DType.int32](chunk * sn)
    var blocks_wide = max(1, (sn + REORDER_BLOCK - 1) // REORDER_BLOCK)
    var d_bsums = ctx.enqueue_create_buffer[DType.int32](chunk * blocks_wide)
    var d_seg_off = ctx.enqueue_create_buffer[DType.uint32](chunk)
    var d_seg_size = ctx.enqueue_create_buffer[DType.uint32](chunk)
    var d_nan_s = ctx.enqueue_create_buffer[DType.int32](chunk)
    var d_nan_c = ctx.enqueue_create_buffer[DType.int32](chunk)
    var d_valid = ctx.enqueue_create_buffer[DType.int32](chunk)
    var d_budget = ctx.enqueue_create_buffer[DType.int32](chunk)
    var d_mode = ctx.enqueue_create_buffer[DType.int32](chunk)
    var d_hs = ctx.enqueue_create_buffer[DType.int32](chunk * heap_cap)
    var d_he = ctx.enqueue_create_buffer[DType.int32](chunk * heap_cap)
    var d_hp = ctx.enqueue_create_buffer[DType.int32](chunk * heap_cap)
    var d_hsc = ctx.enqueue_create_buffer[DType.uint64](chunk * heap_cap)
    var d_out = ctx.enqueue_create_buffer[DType.float32](chunk * out_cap)
    var d_counts = ctx.enqueue_create_buffer[DType.int32](chunk)
    var h_seg = ctx.enqueue_create_host_buffer[DType.uint32](2 * chunk)
    var h_out = ctx.enqueue_create_host_buffer[DType.float32](chunk * out_cap)
    var h_counts = ctx.enqueue_create_host_buffer[DType.int32](chunk)
    var h_mode = ctx.enqueue_create_host_buffer[DType.int32](chunk)
    ctx.synchronize()
    var hs = h_seg.unsafe_ptr()
    for c in range(chunk):  # small-loop(chunk: columns of the chunk): two segment words per column, launch arguments
        hs[c] = UInt32(c * sn)
        hs[chunk + c] = UInt32(sn)
    ctx.enqueue_copy(
        dst_buf=d_seg_off, src_ptr=hs
    )
    ctx.enqueue_copy(
        dst_buf=d_seg_size, src_ptr=hs + chunk
    )
    var base = 0
    while base < n_float:
        var width = min(chunk, n_float - base)
        var cols_ptr = d_cols.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        if use_dev:
            cols_ptr = dev_cols.value() + base * n_rows
        else:
            for c in range(width):
                var view = d_cols.create_sub_buffer[DType.float32](
                    c * n_rows, n_rows
                )
                if resident:
                    ctx.enqueue_copy(
                        dst_buf=view, src_buf=dev_col_bufs[base + c]
                    )
                    continue
                ctx.enqueue_copy(
                    dst_buf=view,
                    src_ptr=rebind[UnsafePointer[Float32, MutAnyOrigin]](
                        cols[base + c]
                    ),
                )
        enqueue_fill(ctx, d_nan_s, Int32(0))
        enqueue_fill(ctx, d_nan_c, Int32(0))
        var key_span = n_rows if sampled else sn
        ctx.enqueue_function[border_keys_kernel](
            cols_ptr, d_idx.unsafe_ptr(), d_keys.unsafe_ptr(),
            d_vals.unsafe_ptr(), d_nan_s.unsafe_ptr(), d_nan_c.unsafe_ptr(),
            Int32(width), Int32(n_rows), Int32(sn), Int32(1 if sampled else 0),
            grid_dim=_blocks(width * key_span), block_dim=GLS_BLOCK,
        )
        launch_segmented_radix_sort(
            ctx, width * sn, width, sn, 0, 32,
            d_keys, d_vals, d_tkeys, d_tvals, d_seg_off, d_seg_size,
            d_offs, d_bsums,
        )
        ctx.enqueue_function[gls_budget_kernel](
            d_nan_s.unsafe_ptr(), d_nan_c.unsafe_ptr(), Int32(sn),
            Int32(1 if sampled else 0), Int32(width), Int32(border_count),
            Int32(nan_mode_option), Int32(NAN_MODE_FORBIDDEN),
            d_valid.unsafe_ptr(), d_budget.unsafe_ptr(), d_mode.unsafe_ptr(),
            grid_dim=_blocks(width), block_dim=GLS_BLOCK,
        )
        if gls:
            ctx.enqueue_function[gls_columns_kernel](
                d_keys.unsafe_ptr(), Int32(sn), Int32(width),
                d_valid.unsafe_ptr(), d_budget.unsafe_ptr(), d_logtab.unsafe_ptr(),
                d_hs.unsafe_ptr(), d_he.unsafe_ptr(), d_hp.unsafe_ptr(),
                d_hsc.unsafe_ptr(), Int32(heap_cap), d_out.unsafe_ptr(),
                Int32(out_cap), d_counts.unsafe_ptr(),
                grid_dim=max(1, (width + 63) // 64), block_dim=64,
            )
        else:
            ctx.enqueue_function[border_types_columns_kernel](
                Int32(border_type), d_keys.unsafe_ptr(), Int32(sn), Int32(width),
                d_valid.unsafe_ptr(), d_budget.unsafe_ptr(), d_logtab.unsafe_ptr(),
                d_hs.unsafe_ptr(), d_he.unsafe_ptr(), d_hp.unsafe_ptr(),
                d_hsc.unsafe_ptr(), Int32(heap_cap),
                d_f64s.unsafe_ptr(), Int64(f64_stride),
                d_i32s.unsafe_ptr(), Int64(i32_stride),
                d_uniq.unsafe_ptr(), Int64(uniq_stride),
                d_out.unsafe_ptr(), Int32(out_cap), d_counts.unsafe_ptr(),
                grid_dim=max(1, (width + 63) // 64), block_dim=64,
            )
        ctx.enqueue_copy(dst_ptr=h_out.unsafe_ptr(), src_buf=d_out)
        ctx.enqueue_copy(dst_ptr=h_counts.unsafe_ptr(), src_buf=d_counts)
        ctx.enqueue_copy(dst_ptr=h_mode.unsafe_ptr(), src_buf=d_mode)
        ctx.synchronize()
        for c in range(width):  # small-loop(width: columns of the chunk): collects each column's fitted borders (at most out_cap) and mode, model parameters, no row data
            var m = Int(h_mode.unsafe_ptr()[c])
            if m == Int(GLS_MODE_REFUSED):
                raise Error(
                    "There are nan factors and nan values for float features are"
                    " not allowed. Set nan_mode != Forbidden."
                )
            var nb = Int(h_counts.unsafe_ptr()[c])
            var bs = List[Float32](capacity=nb + 1)
            if m == NAN_MODE_MIN:
                bs.append(Float32(-3.4028234663852886e38))
            for j in range(nb):  # small-loop(nb: borders of one column): copies one column's fitted borders (at most out_cap), model parameters
                bs.append(h_out.unsafe_ptr()[c * out_cap + j])
            if m == NAN_MODE_MAX:
                bs.append(Float32(3.4028234663852886e38))
            borders.append(bs^)
            modes.append(m)
        base += width
    _ = d_f64s^
    _ = d_i32s^
    _ = d_uniq^
    return (borders^, modes^)
