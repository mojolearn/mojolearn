"""Does every node split on a column the SAMPLER actually drew for it?

    pixi run mojo run -I . ensemble/mojo_only/sampled_cols_check.mojo

THE INVARIANT THIS EXISTS FOR. `sample_features` writes node `nid`'s drawn
columns at `column_samples[nid * k .. nid * k + k)` where `k` is the round's
sampled-column count (`builder.cuh:509` passes it as the argument `k`), and
both split kernels read `column_samples[nid * dataset.n_sampled_cols +
colIndex]` (`kernels/builder_kernels_impl.cuh:310`, `:373`). In THEIR source
those are the same variable -- `builder.cuh:240` sets the field and `:439-440`
narrows it per round -- so writer and reader cannot disagree.

Ours held the sampled count on the Builder and left the DatasetView field at
whatever the caller passed, which was `n_cols`. Node 0 coincided; every later
node in a batch read another node's columns. WHY NOTHING CAUGHT IT: every
check in this directory ran at `max_features = 1.0` or set the field to
`n_cols`, and under either the two strides are equal. `forest_check` arm B
does run `max_features = 0.25`, but it only asserts that trees DIFFER, which
was true under the bug too. That is the uniform-fixture failure: an expected
value that is the same in every cell verifies nothing about placement.

So this check compares PER CELL against an independent tally -- the sampler's
own output, read back off the device -- rather than against a total.

  A. THE INVARIANT. Fit with `max_features` well below 1.0 and a fixture in
     which EVERY column separates the classes, so both children of the root
     split whichever columns they happen to draw. Then read `column_samples`
     back and require each node's chosen `colid` to be a member of that
     node's own slot. A wrong stride puts it in a neighbour's slot.

  B. SABOTAGE. Restore the pre-fix value through `Builder.train`'s hook 8
     and require the run to MOVE -- either the invariant breaks or the tree
     changes. A check whose sabotage changes nothing is not testing the path.
"""

from max.gpu.host import DeviceContext

from ensemble.decisiontree.batched_levelalgo.bins import ClassificationBin
from ensemble.decisiontree.batched_levelalgo.objectives import (
    ClassificationObjectiveFunction,
)
from ensemble.decisiontree.batched_levelalgo.builder import Builder
from ensemble.decisiontree.batched_levelalgo.dataset import DatasetView
from ensemble.decisiontree.batched_levelalgo.quantiles import (
    compute_quantiles,
    Quantiles,
)
from ensemble.decisiontree.decisiontree import DecisionTreeParams, GINI

comptime DT = DType.float32
comptime LT = DType.int32

comptime N_ROWS = 400
comptime N_COLS = 16
comptime N_CLASSES = 4
comptime MAX_FEATURES = Float32(0.25)   # -> k = 4 of 16


def _jitter(i: Int, j: Int) -> Float32:
    """A hashed value in [0, 1). Distinct per (row, column), so each column
    is a DIFFERENT separating column and which one a node drew is
    observable."""
    var h = UInt32(i * 2654435761 + j * 40503)
    h ^= h >> 15
    h = h * UInt32(2246822519)
    h ^= h >> 13
    return Float32(Int(h % UInt32(1000))) / Float32(1000.0)


struct FitOut(Copyable, Movable):
    var colid: List[Int32]
    var is_leaf: List[Bool]
    var samples: List[Int32]
    var k: Int

    def __init__(out self):
        self.colid = List[Int32]()
        self.is_leaf = List[Bool]()
        self.samples = List[Int32]()
        self.k = 0


def _fit[sabotage: Int](ctx: DeviceContext) raises -> FitOut:
    var params = DecisionTreeParams(
        max_depth=Int32(2),
        max_leaves=Int32(-1),
        max_features=MAX_FEATURES,
        max_n_bins=Int32(16),
        min_samples_leaf=Int32(1),
        min_samples_split=Int32(2),
        split_criterion=GINI,
        min_impurity_decrease=Float32(0.0),
        max_batch_size=Int32(128),
    )

    # EVERY column separates the classes, each by a different permutation:
    # class index dominates, the hashed jitter never crosses a class band.
    var hx = ctx.enqueue_create_host_buffer[DT](N_ROWS * N_COLS)
    for j in range(N_COLS):
        for i in range(N_ROWS):
            var cls = i % N_CLASSES
            hx.unsafe_ptr().unsafe_store(
                j * N_ROWS + i, Float32(cls) + Float32(0.5) * _jitter(i, j)
            )
    var dx = ctx.enqueue_create_buffer[DT](N_ROWS * N_COLS)
    ctx.enqueue_copy(dst_buf=dx, src_ptr=hx.unsafe_ptr())

    var hy = ctx.enqueue_create_host_buffer[LT](N_ROWS)
    for i in range(N_ROWS):
        hy.unsafe_ptr().unsafe_store(i, Int32(i % N_CLASSES))
    var dy = ctx.enqueue_create_buffer[LT](N_ROWS)
    ctx.enqueue_copy(dst_buf=dy, src_ptr=hy.unsafe_ptr())

    var hr = ctx.enqueue_create_host_buffer[DType.int32](N_ROWS)
    for i in range(N_ROWS):
        hr.unsafe_ptr().unsafe_store(i, Int32(i))
    var dr = ctx.enqueue_create_buffer[DType.int32](N_ROWS)
    ctx.enqueue_copy(dst_buf=dr, src_ptr=hr.unsafe_ptr())

    var dsw = ctx.enqueue_create_buffer[DT](1)
    ctx.synchronize()

    var qr = compute_quantiles(
        ctx, dx, Int(params.max_n_bins), N_ROWS, N_COLS, seed=UInt64(7)
    )
    var quantiles = Quantiles[DT](
        qr.quantiles_array.unsafe_ptr()
        .unsafe_origin_cast[MutUntrackedOrigin](),
        qr.n_bins_array.unsafe_ptr()
        .unsafe_origin_cast[MutUntrackedOrigin](),
    )

    var dataset = DatasetView[DT, LT](
        dx.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        dy.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        dsw.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        Int64(N_ROWS),
        Int64(N_COLS),
        Int64(1),
        Int64(N_ROWS),
        Int32(N_ROWS),
        Int32(N_COLS),        # what randomforest.mojo passes: n_cols
        dr.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        Int32(N_CLASSES),
        False,
        # DEVIATION 314: raw path.
        dx.unsafe_ptr()
        .unsafe_origin_cast[MutUntrackedOrigin]()
        .unsafe_bitcast[UInt8](),
        False,
    )

    var builder = Builder[
        ClassificationObjectiveFunction[DT, LT, ClassificationBin]
    ](
        ctx, params, Int32(0), UInt64(42), N_ROWS, N_COLS,
        Int32(N_CLASSES),
    )
    var tree = builder.train[sabotage](ctx, dataset, quantiles)

    var out = FitOut()
    out.k = builder.original_n_sampled_cols
    for i in range(len(tree.sparsetree)):
        ref n = tree.sparsetree[i]
        out.colid.append(n.ColumnId())
        out.is_leaf.append(n.IsLeaf())

    # The sampler's own output for the LAST batch -- nodes 1 and 2.
    var n_read = 2 * out.k
    var hs = ctx.enqueue_create_host_buffer[DType.int32](n_read)
    ctx.enqueue_copy(dst_buf=hs, src_buf=builder.column_samples)
    ctx.synchronize()
    for i in range(n_read):
        out.samples.append(hs.unsafe_ptr().unsafe_load(i))

    _ = qr^
    _ = hs^
    _ = dx^
    _ = dy^
    _ = dr^
    _ = dsw^
    _ = hx^
    _ = hy^
    _ = hr^
    _ = builder^
    return out^


def _violations(f: FitOut) raises -> Int:
    """Per cell: node 1 must have split on a column in slot 0, node 2 on a
    column in slot 1. Nodes 1 and 2 are the root's children and are exactly
    the work items of the last batch, in push order (left, then right)."""
    var wrong = 0
    for node in range(1, 3):
        if node >= len(f.colid) or f.is_leaf[node]:
            raise Error(
                "fixture is wrong: node "
                + String(node)
                + " did not split, so the invariant is untested"
            )
        var want_slot = node - 1
        var found = False
        for c in range(f.k):
            if f.samples[want_slot * f.k + c] == f.colid[node]:
                found = True
        print(
            "    node",
            node,
            ": split on col",
            f.colid[node],
            "| its sampled columns",
            end=" ",
        )
        for c in range(f.k):
            print(f.samples[want_slot * f.k + c], end=" ")
        print("->", "IN SET" if found else "NOT IN ITS OWN SLOT")
        if not found:
            wrong += 1
    return wrong


def main() raises:
    var ctx = DeviceContext()
    print("sampled_cols_check --", N_COLS, "columns, max_features 0.25")
    print()

    print("ARM A -- the invariant, shipped path")
    var good = _fit[0](ctx)
    print("    k =", good.k, "of", N_COLS, "columns per node")
    var bad_cells = _violations(good)
    if bad_cells != 0:
        raise Error(
            "arm A FAILED: "
            + String(bad_cells)
            + " node(s) split on a column outside their own sampled slot"
        )
    print("  arm A OK: every node split on a column its own draw contained")
    print()

    print("ARM B -- sabotage 8, the pre-fix stride")
    var sab = _fit[8](ctx)
    var sab_wrong = _violations(sab)
    var tree_moved = False
    if len(sab.colid) != len(good.colid):
        tree_moved = True
    else:
        for i in range(len(good.colid)):
            if sab.colid[i] != good.colid[i]:
                tree_moved = True
    if sab_wrong == 0 and not tree_moved:
        raise Error(
            "arm B FAILED: restoring the wrong stride changed NOTHING, so"
            " this check does not reach the field it claims to test"
        )
    print(
        "  arm B OK: the wrong stride moved the run --",
        sab_wrong,
        "node(s) off their slot, tree changed:",
        tree_moved,
    )
    print()
    print("sampled_cols_check: ALL OK")
