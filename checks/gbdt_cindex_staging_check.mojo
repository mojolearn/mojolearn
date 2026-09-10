# SPDX-License-Identifier: Apache-2.0
"""WP5: ring wrap/tail, mixed layouts, NaN modes, and source reach.
Build each numeric mode under tools/with_build_lock.sh; no speed claim.
"""
from max.gpu.host import DeviceContext
from std.memory import bitcast
from gbdt.train import _build_cindex_from_floats, _build_cindex_from_columns
from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.data.quantization import nan_substitution


def check(ctx: DeviceContext, n_rows: Int, n_features: Int) raises:
    var flat = List[Float32]()
    var columns = List[List[Float32]]()
    var borders = List[List[Float32]]()
    var folds = List[Int]()
    var treatment = List[Int]()
    for f in range(n_features):
        var count = 1
        if f % 3 == 1:
            count = 15
        elif f % 3 == 2:
            count = 31
        if f % 11 == 3:
            count = 0
        var bs = List[Float32]()
        for b in range(count):
            bs.append(Float32(b) - 8.0)
        borders.append(bs^)
        folds.append(count)
        treatment.append(f % 3)
        var col = List[Float32]()
        for r in range(n_rows):
            var v = Float32((r * 17 + f * 13) % 45) - 12.0
            if r % 19 == 7 and f % 3 != 0:
                v = bitcast[DType.float32](UInt32(0x7fc01234))
            elif r % 23 == 2:
                v = bitcast[DType.float32](UInt32(0x80000000))
            col.append(v)
            flat.append(v)
        columns.append(col^)
    var lay = build_layout(folds)
    var expected = List[UInt32]()
    expected.resize(n_rows * lay.columns, UInt32(0))
    for f in range(n_features):
        if len(borders[f]) == 0:
            continue
        ref cf = lay.features[f]
        for r in range(n_rows):
            var v = flat[f*n_rows+r]
            if v != v:
                v = nan_substitution(treatment[f])
            var bin = UInt32(0)
            for b in range(len(borders[f])):
                if v > borders[f][b]:
                    bin += 1
            expected[Int(cf.offset)*n_rows+r] |= (bin & cf.mask) << cf.shift
    var actual = _build_cindex_from_floats(ctx, flat, n_rows, borders, folds, treatment)
    var twin = _build_cindex_from_columns(ctx, columns, n_rows, borders, folds, treatment)
    var ha = ctx.enqueue_create_host_buffer[DType.uint32](len(expected))
    var hb = ctx.enqueue_create_host_buffer[DType.uint32](len(expected))
    ctx.enqueue_copy(dst_buf=ha, src_buf=actual)
    ctx.enqueue_copy(dst_buf=hb, src_buf=twin)
    ctx.synchronize()
    for i in range(len(expected)):
        if ha.unsafe_ptr().unsafe_load(i) != expected[i] or hb.unsafe_ptr().unsafe_load(i) != expected[i]:
            raise Error("WP5 compressed index differs from per-cell oracle")
    # Change an active source cell across its sole border: proves source reach.
    flat[0] = 100.0
    var changed = _build_cindex_from_floats(ctx, flat, n_rows, borders, folds, treatment)
    ctx.enqueue_copy(dst_buf=ha, src_buf=changed)
    ctx.synchronize()
    if ha.unsafe_ptr().unsafe_load(Int(lay.features[0].offset)*n_rows) == expected[Int(lay.features[0].offset)*n_rows]:
        raise Error("WP5 negative control did not move")
    # Refuse an unseen NaN after a full ring has already been queued.
    var bad_feature = ((n_features - 1) // 3) * 3
    if len(borders[bad_feature]) == 0:
        bad_feature -= 3
    flat[bad_feature*n_rows] = bitcast[DType.float32](UInt32(0x7fc00001))
    var refused = False
    try:
        var bad = _build_cindex_from_floats(ctx, flat, n_rows, borders, folds, treatment)
    except:
        refused = True
    if not refused:
        raise Error("WP5 lost unseen-NaN refusal")
    print("PASS WP5 rows", n_rows, "features", n_features)


def main() raises:
    var ctx = DeviceContext()
    check(ctx, 1, 19)
    check(ctx, 257, 35)
    check(ctx, 8193, 67)
