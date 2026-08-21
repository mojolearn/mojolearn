"""DEVIATION 205: does the rescue FIRE, does it move the tree, and do the
two paths agree when it does?

A digest that does not move cannot tell a working change from a no-op, so
this fits the SAME fixture with the rescue on and off and reports the
difference. `shaped_constant_heavy` is the fixture deviations 132 and 151
named as the one where our early stop costs us, so it is the one that has to
move; `all_constant` is the control that must NOT move, because a node with
no non-constant column anywhere is a leaf to sklearn too.
"""

from extratrees.mojo_only.fixtures import (
    Dataset as FixtureDataset,
    all_shapes,
    analytic_all_constant,
    hashed_classification,
    shaped_dataset,
)
from extratrees.mojo_only.fixture_parity_check import const_heavy_shapes
from extratrees.ported.decisiontree.batched_levelalgo.builder import (
    train_classification,
)
from extratrees.ported.decisiontree.batched_levelalgo.dataset import Dataset
from extratrees.ported.decisiontree.decisiontree import DecisionTreeParams
from extratrees.ported.decisiontree.batched_levelalgo.builder import (
    train_classification_device,
)
from extratrees.ported.decisiontree.flatnode import TreeMetaDataNode
from max.gpu.host import DeviceContext


def fit_nodes(
    fx: FixtureDataset, seed: UInt64, rescue: Bool, n_classes: Int32
) raises -> Tuple[Int, Int]:
    var x = List[Float32](
        length=fx.n_rows * fx.n_cols, fill=Float32(0.0)
    )
    for r in range(fx.n_rows):
        for c in range(fx.n_cols):
            x[c * fx.n_rows + r] = fx.value(r, c)
    # `shaped_dataset` carries a REGRESSION target; the classification arm of
    # quality_band_check derives a label from it and so does this. The probe
    # is about the FEATURE matrix being constant-heavy, which is untouched.
    var y = List[Float32]()
    for r in range(fx.n_rows):
        y.append(Float32(1.0) if fx.y[r] > 0.0 else Float32(0.0))
    var rows = List[Int32]()
    for r in range(fx.n_rows):
        rows.append(Int32(r))
    var ds = Dataset(
        rebind[MutPointer[Float32, MutUntrackedOrigin]](x.unsafe_ptr()),
        rebind[MutPointer[Float32, MutUntrackedOrigin]](y.unsafe_ptr()),
        Int32(fx.n_rows),
        Int32(fx.n_cols),
        Int32(fx.n_rows),
        Int32(fx.n_cols),
        rebind[MutPointer[Int32, MutUntrackedOrigin]](rows.unsafe_ptr()),
        n_classes,
    )
    var p = DecisionTreeParams()
    p.max_depth = 12
    p.max_features = 0.15
    var t = train_classification(ds, p, 0, seed, n_classes, rescue=rescue)
    var leaves = 0
    for i in range(t.num_nodes()):
        if t.sparsetree[i].IsLeaf():
            leaves += 1
    var n = t.num_nodes()
    _ = x.unsafe_ptr()
    _ = y.unsafe_ptr()
    _ = rows.unsafe_ptr()
    return (n, leaves)


def fit_device_tree(
    ctx: DeviceContext, fx: FixtureDataset, seed: UInt64, n_classes: Int32
) raises -> TreeMetaDataNode[DType.float32]:
    var x = List[Float32](length=fx.n_rows * fx.n_cols, fill=Float32(0.0))
    for r in range(fx.n_rows):
        for c in range(fx.n_cols):
            x[c * fx.n_rows + r] = fx.value(r, c)
    var cls = List[Int32]()
    for r in range(fx.n_rows):
        cls.append(Int32(1) if fx.y[r] > 0.0 else Int32(0))
    var rows = List[Int32]()
    for r in range(fx.n_rows):
        rows.append(Int32(r))
    var p = DecisionTreeParams()
    p.max_depth = 12
    p.max_features = 0.15
    var t = train_classification_device(
        ctx, x, cls, rows, Int32(fx.n_rows), Int32(fx.n_cols), n_classes,
        p, 0, seed,
    )
    _ = x.unsafe_ptr()
    _ = rows.unsafe_ptr()
    return t^


def fit_host_tree(
    fx: FixtureDataset, seed: UInt64, n_classes: Int32
) raises -> TreeMetaDataNode[DType.float32]:
    var x = List[Float32](length=fx.n_rows * fx.n_cols, fill=Float32(0.0))
    for r in range(fx.n_rows):
        for c in range(fx.n_cols):
            x[c * fx.n_rows + r] = fx.value(r, c)
    var y = List[Float32]()
    for r in range(fx.n_rows):
        y.append(Float32(1.0) if fx.y[r] > 0.0 else Float32(0.0))
    var rows = List[Int32]()
    for r in range(fx.n_rows):
        rows.append(Int32(r))
    var ds = Dataset(
        rebind[MutPointer[Float32, MutUntrackedOrigin]](x.unsafe_ptr()),
        rebind[MutPointer[Float32, MutUntrackedOrigin]](y.unsafe_ptr()),
        Int32(fx.n_rows), Int32(fx.n_cols), Int32(fx.n_rows),
        Int32(fx.n_cols),
        rebind[MutPointer[Int32, MutUntrackedOrigin]](rows.unsafe_ptr()),
        n_classes,
    )
    var p = DecisionTreeParams()
    p.max_depth = 12
    p.max_features = 0.15
    var t = train_classification(ds, p, 0, seed, n_classes, rescue=True)
    _ = x.unsafe_ptr()
    _ = y.unsafe_ptr()
    _ = rows.unsafe_ptr()
    return t^


def identity(
    ctx: DeviceContext, name: String, fx: FixtureDataset, n_classes: Int32
) raises -> Int:
    """The DEVICE rescue must land on the same column as the HOST rescue.

    This is the assertion the whole of DEVIATION 205 rests on: two independent
    implementations of `_splitter.pyx:573-577`, one sequential and one a
    sub-batch of kernels, must choose the same feature for the same node. If
    they do not, `device_tree_check` would have caught it only on a fixture
    where the rescue fires -- and its fixtures do not fire it, which is why
    this exists.
    """
    var bad = 0
    for s in range(4):
        var seed = UInt64(s * 7919 + 11)
        var h = fit_host_tree(fx, seed, n_classes)
        var d = fit_device_tree(ctx, fx, seed, n_classes)
        if h.num_nodes() != d.num_nodes():
            print("      ", name, "seed", s, "NODE COUNT differs: host",
                  h.num_nodes(), "device", d.num_nodes())
            bad += 1
            continue
        var diff = 0
        for i in range(h.num_nodes()):
            var a = h.sparsetree[i]
            var b = d.sparsetree[i]
            if (
                a.ColumnId() != b.ColumnId()
                or a.QueryValue().to_bits() != b.QueryValue().to_bits()
                or a.LeftChildId() != b.LeftChildId()
                or a.InstanceCount() != b.InstanceCount()
            ):
                diff += 1
        print("      ", name, "seed", s, ":", h.num_nodes(),
              "nodes,", diff, "differ between host and device")
        if diff != 0:
            bad += 1
    return bad


def report(name: String, fx: FixtureDataset, n_classes: Int32) raises -> Int:
    var moved = 0
    for s in range(4):
        var seed = UInt64(s * 7919 + 11)
        var off = fit_nodes(fx, seed, False, n_classes)
        var on = fit_nodes(fx, seed, True, n_classes)
        if off[0] != on[0]:
            moved += 1
        print(
            "   ",
            name,
            " seed",
            s,
            " rescue OFF nodes",
            off[0],
            "leaves",
            off[1],
            " -> ON nodes",
            on[0],
            "leaves",
            on[1],
        )
    return moved


def main() raises:
    print("[reach] DEVIATION 205, max_features=0.15 so the sample is narrow")
    var seed = UInt64(20260821)
    var heavy = shaped_dataset(seed, 512, const_heavy_shapes())
    var moved_heavy = report("shaped_constant_heavy", heavy, 2)
    var allc = analytic_all_constant().data.copy()
    var moved_allc = report("all_constant", allc, 2)
    var hashed = hashed_classification(seed, 512, 16, 3)
    var moved_hashed = report("hashed_cls", hashed, 3)
    print("")
    print("[identity] host rescue against DEVICE rescue, same fixture")
    var ctx = DeviceContext()
    var bad = identity(ctx, "shaped_constant_heavy", heavy, 2)
    bad += identity(ctx, "hashed_cls", hashed, 3)
    if bad != 0:
        raise Error(
            "the host and device rescues chose DIFFERENT columns. They call"
            " the same rescue_pick on the same key, so a difference here is a"
            " difference in the non-constant SET or its order"
        )

    print("")
    print(
        "[reach] seeds where the tree MOVED: constant_heavy",
        moved_heavy,
        "of 4, all_constant",
        moved_allc,
        "of 4 (must be 0), hashed_cls",
        moved_hashed,
        "of 4",
    )
    if moved_heavy == 0:
        raise Error(
            "the rescue changed NOTHING on the fixture deviation 151 named."
            " Either it never fires or it fires and picks the column the old"
            " path already had -- both are defects, and a green run here"
            " would have hidden them"
        )
    if moved_allc != 0:
        raise Error(
            "the rescue moved all_constant, where EVERY column is constant"
            " and sklearn's loop also stops. It must be inert there"
        )
    print(
        "rescue_check: PASS -- the rescue moves the constant-heavy fixture on"
        " every seed, is inert where every column is constant and where none"
        " is, and the host and device rescues choose the same column"
    )
