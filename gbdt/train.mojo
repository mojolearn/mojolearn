"""The user-facing surface: train from raw floats, predict raw floats.

NOT A PORT -- this is the convenience layer over ported machinery, the
`fit(X, y)` shape callers actually hold. Everything under it is the
transliterated pipeline: borders from `grid_creator.binarization`
(their GreedyLogSum, heap semantics included), device quantization
through `binarize_float_feature_kernel` (their BinarizeFloatFeatureImpl,
the same kernel their own predict quantizes with), the compressed index
through `write` layout rules, and `doc_parallel_boosting.fit`.

One stated difference from CatBoost's default pipeline: their quantizer
subsamples large datasets before computing borders; this computes them
from ALL rows. Same rule, more data -- the grids agree wherever theirs
did not subsample, and `binarization_check` holds the border parity on
the oracle fixture.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from gbdt.gpu_data.compressed_index_builder import build_layout
from gbdt.gpu_data.kernel.binarize import (
    BINARIZE_BLOCK_SIZE,
    BINARIZE_DOCS_PER_THREAD,
    binarize_float_feature_kernel,
)
from gbdt.grid_creator.binarization import best_split
from gbdt.methods.doc_parallel_boosting import TAdditiveModel, fit, predict
from gbdt.options.catboost_options import SCORE_FUNCTION_COSINE


@fieldwise_init
struct TrainedModel(Movable):
    """The ensemble plus the quantization grid it was trained on, which
    is what applying it to NEW raw floats requires -- their model carries
    its borders for the same reason."""

    var model: TAdditiveModel
    var fold_counts: List[Int]
    var one_hot: List[Bool]
    var borders: List[List[Float32]]
    var losses: List[Float64]


def _build_cindex_from_floats(
    ctx: DeviceContext,
    x_colmajor: List[Float32],
    n_rows: Int,
    borders: List[List[Float32]],
) raises -> DeviceBuffer[DType.uint32]:
    var n_features = len(borders)
    var fold_counts = List[Int]()
    for f in range(n_features):
        fold_counts.append(len(borders[f]))
    var lay = build_layout(fold_counts)
    var cindex = ctx.enqueue_create_buffer[DType.uint32](
        n_rows * lay.columns
    )
    ctx.enqueue_memset(cindex, UInt32(0))

    var xdev = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var hx = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    # the count-first borders layout `binarize_float_feature_kernel`
    # reads (their `sharedBorders[0]` broadcast)
    var hbo = ctx.enqueue_create_host_buffer[DType.float32](256)
    var bdev = ctx.enqueue_create_buffer[DType.float32](256)
    comptime BIN_GRID = BINARIZE_BLOCK_SIZE * BINARIZE_DOCS_PER_THREAD
    for f in range(n_features):
        if len(borders[f]) == 0:
            continue
        ref cf = lay.features[f]
        for r in range(n_rows):
            hx.unsafe_ptr().unsafe_store(r, x_colmajor[f * n_rows + r])
        ctx.enqueue_copy(dst_buf=xdev, src_ptr=hx.unsafe_ptr())
        hbo.unsafe_ptr().unsafe_store(0, Float32(len(borders[f])))
        for b in range(len(borders[f])):
            hbo.unsafe_ptr().unsafe_store(1 + b, borders[f][b])
        ctx.enqueue_copy(dst_buf=bdev, src_ptr=hbo.unsafe_ptr())
        ctx.enqueue_function[binarize_float_feature_kernel](
            Int32(Int(cf.offset) * n_rows), cf.mask, cf.shift,
            xdev.unsafe_ptr(), Int32(n_rows),
            bdev.unsafe_ptr(), cindex.unsafe_ptr(),
            grid_dim=(n_rows + BIN_GRID - 1) // BIN_GRID,
            block_dim=(BINARIZE_BLOCK_SIZE, 1, 1),
        )
        ctx.synchronize()
    return cindex^


def train(
    ctx: DeviceContext,
    x_colmajor: List[Float32],
    y: List[Float32],
    n_rows: Int,
    n_features: Int,
    border_count: Int = 128,
    n_estimators: Int = 100,
    max_depth: Int = 6,
    learning_rate: Float32 = Float32(0.03),
    l2_leaf_reg: Float32 = Float32(3.0),
    one_hot: List[Bool] = List[Bool](),
    bootstrap_bayesian: Bool = False,
    bagging_temperature: Float32 = Float32(1.0),
    random_seed: UInt64 = UInt64(0),
    score_function: Int = SCORE_FUNCTION_COSINE,
) raises -> TrainedModel:
    """Borders -> device quantization -> fit, one call.

    `x_colmajor` is `[feature * n_rows + row]`. A feature marked in
    `one_hot` skips border search: its values ARE dense category codes
    `0..k-1` and its fold count is `max + 1`, split by equality.
    """
    if len(x_colmajor) != n_rows * n_features:
        raise Error("x_colmajor size mismatch")
    if len(y) != n_rows:
        raise Error("y size mismatch")

    var borders = List[List[Float32]]()
    var fold_counts = List[Int]()
    for f in range(n_features):
        var flagged = len(one_hot) == n_features and one_hot[f]
        if flagged:
            var maxc = 0
            for r in range(n_rows):
                var c = Int(x_colmajor[f * n_rows + r])
                if c > maxc:
                    maxc = c
            if maxc > 254:
                raise Error("one-hot feature " + String(f)
                            + " has more than 255 categories")
            # synthetic integer 'borders' 0.5, 1.5, ... so the SAME
            # quantize kernel maps code k to bin k
            var bs = List[Float32]()
            for c in range(maxc):
                bs.append(Float32(c) + Float32(0.5))
            fold_counts.append(len(bs) + 1 if len(bs) > 0 else 0)
            borders.append(bs^)
        else:
            var col = List[Float32]()
            for r in range(n_rows):
                col.append(x_colmajor[f * n_rows + r])
            var bs = best_split(col^, border_count)
            fold_counts.append(len(bs))
            borders.append(bs^)

    # one-hot features occupy folds+? -- for ordered features CatBoost's
    # fold count IS the border count; a one-hot feature has k categories
    # = k bins reached by k-1 synthetic borders, and its fold count must
    # cover bin k-1 for the equality candidates, hence len+1 above.

    var cindex = _build_cindex_from_floats(ctx, x_colmajor, n_rows, borders)

    var targets = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var weights = ctx.enqueue_create_buffer[DType.float32](n_rows)
    var ht = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    var hw = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    for r in range(n_rows):
        ht.unsafe_ptr().unsafe_store(r, y[r])
        hw.unsafe_ptr().unsafe_store(r, Float32(1.0))
    ctx.enqueue_copy(dst_buf=targets, src_ptr=ht.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=weights, src_ptr=hw.unsafe_ptr())
    ctx.synchronize()

    var model = TAdditiveModel()
    var losses = fit(
        model, ctx, n_rows, fold_counts, max_depth, cindex, targets,
        weights, False, n_estimators, learning_rate, l2_leaf_reg, True,
        bootstrap_bayesian=bootstrap_bayesian,
        bagging_temperature=bagging_temperature,
        random_seed=random_seed,
        one_hot=one_hot,
        score_function=score_function,
    )
    return TrainedModel(
        model^, fold_counts^, one_hot.copy(), borders^, losses^
    )


def predict_floats(
    ctx: DeviceContext,
    tm: TrainedModel,
    x_colmajor: List[Float32],
    n_rows: Int,
) raises -> List[Float32]:
    """Apply a trained model to NEW raw floats: quantize against the
    model's own grid (as their predict does internally), then the
    tree-wise apply the probe suite pins to the learn cursor."""
    var n_features = len(tm.fold_counts)
    if len(x_colmajor) != n_rows * n_features:
        raise Error("x_colmajor size mismatch")
    var cindex = _build_cindex_from_floats(
        ctx, x_colmajor, n_rows, tm.borders
    )
    var cursor = ctx.enqueue_create_buffer[DType.float32](n_rows)
    predict(
        tm.model, ctx, n_rows, tm.fold_counts, cindex, cursor,
        one_hot=tm.one_hot,
    )
    var hc = ctx.enqueue_create_host_buffer[DType.float32](n_rows)
    ctx.enqueue_copy(dst_ptr=hc.unsafe_ptr(), src_buf=cursor)
    ctx.synchronize()
    var out = List[Float32]()
    for r in range(n_rows):
        out.append(hc.unsafe_ptr().unsafe_load(r))
    return out^
