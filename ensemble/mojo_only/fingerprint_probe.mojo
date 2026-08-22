"""Print a bit-exact fingerprint of five fixed forests, one line per config.

    pixi run mojo run -I . ensemble/mojo_only/fingerprint_probe.mojo

THE GATE FOR CONTROL-PLANE EDITS. A change that only reorders host work --
pooling a workspace, deleting a synchronize whose ordering the queue already
guarantees -- must not move any output bit. This probe folds every node
field, every leaf value and the OOB vector of five differently-shaped fits
into one FNV-1a64 per config, over BIT PATTERNS (never decimal text -- Mojo's
`String(Float32)` is one ULP wrong 0.46% of the time). Record the lines
before the edit, diff after. Any moved hash is a real output change.

The five configs are chosen for REACH, one per branch this lane's
control-plane work touches (reach is per-branch; an unexercised arm is
unchecked):

  CLF-BOOT    the default Philox bootstrap arm of `RowSampler._sample_rows`
  CLF-OOB     `store_bootstrap_mask` + `compute_oob_score`
  CLF-NOBOOT  the no-bootstrap identity arm (host staging)
  REG-BOOT    `Builder` generic over the REGRESSION objective
  CLF-DEEP    max_depth 14, across their depth >= 13 allocation branch
              (`builder.cuh:221-224`), multi-batch trees

This is a PROBE with recorded expected output, not a self-contained check:
it cannot fail on its own, it fails when its output moves between two runs
that should agree. Sabotage discipline for the gate lives with the edits
that use it -- the first recorded use deliberately flipped `max_features`
in one config and watched exactly that line move.
"""

from max.gpu.host import DeviceContext

from ensemble.decisiontree.batched_levelalgo.bins import (
    ClassificationBin,
    RegressionBin,
)
from ensemble.decisiontree.batched_levelalgo.objectives import (
    ClassificationObjectiveFunction,
    RegressionObjectiveFunction,
)
from ensemble.decisiontree.decisiontree import DecisionTreeParams, GINI, MSE
from ensemble.randomforest import (
    RF_params,
    RandomForestMetaData,
    fit_forest,
)

comptime DT = DType.float32
comptime CLT = DType.int32
comptime RLT = DType.float32
comptime ClsObj = ClassificationObjectiveFunction[DT, CLT, ClassificationBin]
comptime RegObj = RegressionObjectiveFunction[DT, RLT, RegressionBin]

comptime FNV_OFFSET: UInt64 = 0xCBF29CE484222325
comptime FNV_PRIME: UInt64 = 0x100000001B3


@always_inline
def _mix(x: UInt64) -> UInt64:
    """splitmix64 finalizer; the fixture generator."""
    var h = x
    h ^= h >> 33
    h *= 0xFF51AFD7ED558CCD
    h ^= h >> 33
    h *= 0xC4CEB9FE1A85EC53
    h ^= h >> 33
    return h


@always_inline
def _fold(mut h: UInt64, v: UInt64):
    h ^= v
    h *= FNV_PRIME


def _rf_params(
    n_trees: Int,
    bootstrap: Bool,
    max_features: Float32,
    max_depth: Int,
    max_n_bins: Int,
    criterion: Int,
) -> RF_params:
    return RF_params(
        n_trees=Int32(n_trees),
        bootstrap=bootstrap,
        max_samples=Float32(1.0),
        seed=UInt64(1234),
        n_streams=Int32(1),
        tree_params=DecisionTreeParams(
            max_depth=Int32(max_depth),
            max_leaves=Int32(-1),
            max_features=max_features,
            max_n_bins=Int32(max_n_bins),
            min_samples_leaf=Int32(1),
            min_samples_split=Int32(2),
            split_criterion=criterion,
            min_impurity_decrease=Float32(0.0),
            max_batch_size=Int32(128),
        ),
    )


def _fingerprint[
    dt: DType, lt: DType
](forest: RandomForestMetaData[dt, lt]) -> UInt64:
    var h = FNV_OFFSET
    _fold(h, UInt64(len(forest.trees)))
    for t in range(len(forest.trees)):
        ref tree = forest.trees[t]
        _fold(h, UInt64(len(tree.sparsetree)))
        for i in range(len(tree.sparsetree)):
            ref node = tree.sparsetree[i]
            _fold(h, UInt64(UInt32(Int(node.ColumnId()))))
            _fold(h, UInt64(node.QueryValue().to_bits()))
            _fold(h, UInt64(node.BestMetric().to_bits()))
            _fold(h, UInt64(Int(node.LeftChildId()) & 0xFFFFFFFF))
            _fold(h, UInt64(UInt32(Int(node.InstanceCount()))))
        for i in range(len(tree.vector_leaf)):
            _fold(h, UInt64(tree.vector_leaf[i].to_bits()))
    if forest.has_oob:
        _fold(h, UInt64(forest.oob_score_.to_bits()))
        for i in range(len(forest.oob_decision_function_)):
            _fold(h, UInt64(forest.oob_decision_function_[i].to_bits()))
        for i in range(len(forest.oob_prediction_)):
            _fold(h, UInt64(forest.oob_prediction_[i].to_bits()))
    return h


def _fit_clf(
    ctx: DeviceContext,
    n_rows: Int,
    n_cols: Int,
    n_classes: Int,
    mut p: RF_params,
    oob: Bool,
    salt: UInt64,
) raises -> UInt64:
    var hx = ctx.enqueue_create_host_buffer[DT](n_rows * n_cols)
    var hy = ctx.enqueue_create_host_buffer[CLT](n_rows)
    for r in range(n_rows):
        var cls = Int(_mix(salt + UInt64(r) * 7919) % UInt64(n_classes))
        hy.unsafe_ptr().unsafe_store(r, Int32(cls))
        for c in range(n_cols):
            var u = _mix(salt ^ (UInt64(r) * 1315423911 + UInt64(c)))
            var noise = Float32(Int(u % 10000)) / Float32(10000.0)
            # Columns 0..2 carry the class signal; the rest are noise.
            var v = noise
            if c < 3:
                v = Float32(cls) + noise * Float32(0.8)
            hx.unsafe_ptr().unsafe_store(r * n_cols + c, v)
    var dx = ctx.enqueue_create_buffer[DT](n_rows * n_cols)
    ctx.enqueue_copy(dst_buf=dx, src_ptr=hx.unsafe_ptr())
    var dy = ctx.enqueue_create_buffer[CLT](n_rows)
    ctx.enqueue_copy(dst_buf=dy, src_ptr=hy.unsafe_ptr())
    var dsw = ctx.enqueue_create_buffer[DT](1)
    ctx.synchronize()
    var forest = fit_forest[ClsObj](
        ctx, dx, dy, dsw, n_rows, n_cols, n_classes, p, oob_score=oob
    )
    _ = dx^
    _ = dy^
    _ = dsw^
    _ = hx^
    _ = hy^
    return _fingerprint(forest)


def _fit_reg(
    ctx: DeviceContext,
    n_rows: Int,
    n_cols: Int,
    mut p: RF_params,
    salt: UInt64,
) raises -> UInt64:
    var hx = ctx.enqueue_create_host_buffer[DT](n_rows * n_cols)
    var hy = ctx.enqueue_create_host_buffer[RLT](n_rows)
    for r in range(n_rows):
        var acc = Float32(0.0)
        for c in range(n_cols):
            var u = _mix(salt ^ (UInt64(r) * 2654435761 + UInt64(c)))
            var v = Float32(Int(u % 10000)) / Float32(10000.0)
            hx.unsafe_ptr().unsafe_store(r * n_cols + c, v)
            if c < 3:
                acc += v
        hy.unsafe_ptr().unsafe_store(r, acc)
    var dx = ctx.enqueue_create_buffer[DT](n_rows * n_cols)
    ctx.enqueue_copy(dst_buf=dx, src_ptr=hx.unsafe_ptr())
    var dy = ctx.enqueue_create_buffer[RLT](n_rows)
    ctx.enqueue_copy(dst_buf=dy, src_ptr=hy.unsafe_ptr())
    var dsw = ctx.enqueue_create_buffer[DT](1)
    ctx.synchronize()
    var forest = fit_forest[RegObj](ctx, dx, dy, dsw, n_rows, n_cols, 1, p)
    _ = dx^
    _ = dy^
    _ = dsw^
    _ = hx^
    _ = hy^
    return _fingerprint(forest)


def main() raises:
    var ctx = DeviceContext()

    var p1 = _rf_params(8, True, Float32(0.5), 8, 32, GINI)
    print("CLF-BOOT   ", hex(_fit_clf(ctx, 4000, 20, 4, p1, False, 11)))

    var p2 = _rf_params(8, True, Float32(0.5), 8, 32, GINI)
    print("CLF-OOB    ", hex(_fit_clf(ctx, 4000, 20, 4, p2, True, 11)))

    var p3 = _rf_params(4, False, Float32(0.5), 8, 32, GINI)
    print("CLF-NOBOOT ", hex(_fit_clf(ctx, 4000, 20, 4, p3, False, 22)))

    var p4 = _rf_params(6, True, Float32(0.5), 8, 32, MSE)
    print("REG-BOOT   ", hex(_fit_reg(ctx, 4000, 20, p4, 33)))

    var p5 = _rf_params(6, True, Float32(0.3), 14, 32, GINI)
    print("CLF-DEEP   ", hex(_fit_clf(ctx, 8000, 20, 4, p5, False, 44)))
