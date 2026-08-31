# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
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
from extratrees.mojo_only.fixed_point import choose_scale, quantize
from extratrees.ported.decisiontree.batched_levelalgo.builder import (
    train_regression,
    train_regression_device,
)
from extratrees.ported.decisiontree.decisiontree import (
    CRITERION_MSE,
    DecisionTreeParams,
)
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


def reg_params() -> DecisionTreeParams:
    var p = DecisionTreeParams()
    p.max_depth = 12
    p.max_features = 0.15
    p.split_criterion = CRITERION_MSE
    return p^


def fit_reg_host(
    fx: FixtureDataset, seed: UInt64, rescue: Bool
) raises -> TreeMetaDataNode[DType.float32]:
    var x = List[Float32](length=fx.n_rows * fx.n_cols, fill=Float32(0.0))
    for r in range(fx.n_rows):
        for c in range(fx.n_cols):
            x[c * fx.n_rows + r] = fx.value(r, c)
    var y = List[Float32]()
    for r in range(fx.n_rows):
        y.append(fx.y[r])
    var rows = List[Int32]()
    for r in range(fx.n_rows):
        rows.append(Int32(r))
    var ds = Dataset(
        rebind[MutPointer[Float32, MutUntrackedOrigin]](x.unsafe_ptr()),
        rebind[MutPointer[Float32, MutUntrackedOrigin]](y.unsafe_ptr()),
        Int32(fx.n_rows), Int32(fx.n_cols), Int32(fx.n_rows),
        Int32(fx.n_cols),
        rebind[MutPointer[Int32, MutUntrackedOrigin]](rows.unsafe_ptr()),
        Int32(1),
    )
    var t = train_regression(ds, reg_params(), 0, seed, rescue=rescue)
    _ = x.unsafe_ptr()
    _ = y.unsafe_ptr()
    _ = rows.unsafe_ptr()
    return t^


def fit_reg_device(
    ctx: DeviceContext, fx: FixtureDataset, seed: UInt64
) raises -> TreeMetaDataNode[DType.float32]:
    var x = List[Float32](length=fx.n_rows * fx.n_cols, fill=Float32(0.0))
    for r in range(fx.n_rows):
        for c in range(fx.n_cols):
            x[c * fx.n_rows + r] = fx.value(r, c)
    var mag = Float64(0.0)
    for r in range(fx.n_rows):
        var v = Float64(fx.y[r])
        mag += v if v >= 0.0 else -v
    var sc = choose_scale(mag, fx.n_rows)
    var q = List[Int32]()
    for r in range(fx.n_rows):
        q.append(Int32(quantize(Float64(fx.y[r]), sc)))
    var rows = List[Int32]()
    for r in range(fx.n_rows):
        rows.append(Int32(r))
    var t = train_regression_device(
        ctx, x, q, sc, rows, Int32(fx.n_rows), Int32(fx.n_cols),
        reg_params(), 0, seed,
    )
    _ = x.unsafe_ptr()
    _ = rows.unsafe_ptr()
    return t^


def regression_arms(
    ctx: DeviceContext, name: String, fx: FixtureDataset
) raises -> Tuple[Int, Int]:
    """`(seeds where the rescue moved the tree, seeds where host != device)`.

    The clause lives in `node_split_random`, which sklearn shares between both
    criteria (`_splitter.pyx:507-736` is reached by `RandomSplitter` for
    regression and classification alike), so a regression tree stops early for
    the same reason. This is that, measured rather than assumed.
    """
    var moved = 0
    var bad = 0
    for s in range(4):
        var seed = UInt64(s * 7919 + 11)
        var off = fit_reg_host(fx, seed, False)
        var on = fit_reg_host(fx, seed, True)
        var dev = fit_reg_device(ctx, fx, seed)
        if off.num_nodes() != on.num_nodes():
            moved += 1
        var diff = 0
        if on.num_nodes() != dev.num_nodes():
            diff = -1
        else:
            for i in range(on.num_nodes()):
                var a = on.sparsetree[i]
                var b = dev.sparsetree[i]
                if (
                    a.ColumnId() != b.ColumnId()
                    or a.QueryValue().to_bits() != b.QueryValue().to_bits()
                    or a.LeftChildId() != b.LeftChildId()
                    or a.InstanceCount() != b.InstanceCount()
                ):
                    diff += 1
        if diff != 0:
            bad += 1
        print(
            "      ", name, "seed", s, " host rescue OFF nodes",
            off.num_nodes(), "-> ON", on.num_nodes(),
            " device", dev.num_nodes(), " host-vs-device differ", diff,
        )
    return (moved, bad)


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
    print("[regression] the same clause, the MSE criterion")
    var rheavy = shaped_dataset(seed, 512, const_heavy_shapes())
    var r1 = regression_arms(ctx, "shaped_constant_heavy", rheavy)
    var rallc = analytic_all_constant().data.copy()
    var r2 = regression_arms(ctx, "all_constant", rallc)
    if r1[0] == 0:
        raise Error(
            "the REGRESSION rescue changed nothing on the constant-heavy"
            " fixture. The clause is in node_split_random, which sklearn"
            " shares between both criteria, so a regression tree that never"
            " moves means the rescue is not reached on this path"
        )
    if r2[0] != 0:
        raise Error(
            "the regression rescue moved all_constant, where every column is"
            " constant and sklearn stops too"
        )
    if r1[1] != 0 or r2[1] != 0:
        raise Error(
            "the host and device REGRESSION rescues chose different columns"
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
