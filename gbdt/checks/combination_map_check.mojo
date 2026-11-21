# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Native check of the combination map (`gbdt/models/combination_map.mojo`),
host only; `gbdt/checks/combination_map_check_main.mojo` runs it.

One fixture: four raw input features (a float, a categorical of 5 codes, a
one-hot categorical of 3 codes, a categorical of 4 codes) and a combination
of four members in fit order: `float > 0.5`, `one-hot == 1`, the 5-code
feature, the 4-code feature. Checked:

  * the map built from the fit's groups (`tree_ctr_groups_host`, the
    numbering the device builder's segments share) gives every learn row its
    own group id back as its derived code, on a column-major and on a
    row-major input: the table a fit tallied by group id is read at the same
    entry at predict;
  * a tuple the learn pool never held, a category the fit never saw and a
    NaN on the float member (it takes `nan_side`) land where the statement
    says; a categorical value that is not a whole number is refused;
  * the tree-CTR model columns of `expand_combination_columns` equal the
    simple table's own `value_for` at the row's group, bit for bit, and an
    unseen row takes `empty_value` (the prior);
  * the text records read back to the same map, and a text whose tuples do
    not ascend is refused.
"""

from std.memory import bitcast
from gbdt.checks.tree_ctr_check import fixture_codes
from gbdt.ctrs.ctr import CTR_BORDERS, CTR_FEATURE_FREQ, TCtrConfig, TPrior
from gbdt.ctrs.tree_ctr import TreeU32Ptr, tree_ctr_groups_host
from gbdt.models.combination_map import (
    CombF32Ptr,
    CombU32Ptr,
    TCombinationCtrs,
    TCombinationMap,
    TCombinationMember,
    combination_cat_member,
    combination_codes_into,
    combination_ctr_tables,
    combination_float_member,
    combination_map_from_groups,
    combination_maps_from_text,
    combination_maps_text,
    combination_one_hot_member,
    expand_combination_columns,
)

#: raw input features of the fixture
comptime COMB_FIXTURE_RAW = 4
#: model columns of the fixture: the four raw ones, then two tree-CTR columns
comptime COMB_FIXTURE_COLUMNS = 6


def _need(ok: Bool, what: String) raises:
    if not ok:
        raise Error("combination map check: " + what)


def _u32t(values: List[UInt32]) -> TreeU32Ptr:
    return TreeU32Ptr(unsafe_from_address=Int(values.unsafe_ptr()))


def _u32c(values: List[UInt32]) -> CombU32Ptr:
    return CombU32Ptr(unsafe_from_address=Int(values.unsafe_ptr()))


def _f32c(values: List[Float32]) -> CombF32Ptr:
    return CombF32Ptr(unsafe_from_address=Int(values.unsafe_ptr()))


struct CombinationFixture(Movable):
    """The learn pool of the check (column-major raw input), the fit's
    groups, the map, the model section and the binarized target."""

    var n: Int
    var x: List[Float32]
    var group_ids: List[UInt32]
    var n_groups: Int
    var cmap: TCombinationMap
    var ctrs: TCombinationCtrs
    var target: List[UInt8]

    def __init__(
        out self,
        n: Int,
        var x: List[Float32],
        var group_ids: List[UInt32],
        n_groups: Int,
        var cmap: TCombinationMap,
        var ctrs: TCombinationCtrs,
        var target: List[UInt8],
    ):
        self.n = n
        self.x = x^
        self.group_ids = group_ids^
        self.n_groups = n_groups
        self.cmap = cmap^
        self.ctrs = ctrs^
        self.target = target^


def combination_fixture_members() -> List[TCombinationMember]:
    var members = List[TCombinationMember]()
    members.append(combination_float_member(0, Float32(0.5), 1))
    members.append(combination_one_hot_member(2, 1))
    members.append(combination_cat_member(1))
    members.append(combination_cat_member(3))
    return members^


def combination_fixture(n: Int) raises -> CombinationFixture:
    var f0 = fixture_codes(n, 7, 31)
    var f1 = fixture_codes(n, 5, 32)
    var f2 = fixture_codes(n, 3, 33)
    var f3 = fixture_codes(n, 4, 34)
    var tcodes = fixture_codes(n, 2, 35)
    var x = List[Float32](length=COMB_FIXTURE_RAW * n, fill=Float32(0.0))
    var side0 = List[UInt32](length=n, fill=UInt32(0))
    var side2 = List[UInt32](length=n, fill=UInt32(0))
    var target = List[UInt8](capacity=n)
    for r in range(n):
        var value = Float32(Int(f0[r])) / Float32(7.0)
        x[r] = value
        x[n + r] = Float32(Int(f1[r]))
        x[2 * n + r] = Float32(Int(f2[r]))
        x[3 * n + r] = Float32(Int(f3[r]))
        if value > Float32(0.5):
            side0[r] = UInt32(1)
        if f2[r] == UInt32(1):
            side2[r] = UInt32(1)
        target.append(UInt8(Int(tcodes[r])))

    # the fit's grouping, members in fit order
    var tree_members = List[TreeU32Ptr]()
    tree_members.append(_u32t(side0))
    tree_members.append(_u32t(side2))
    tree_members.append(_u32t(f1))
    tree_members.append(_u32t(f3))
    var cards: List[Int] = [2, 2, 5, 4]
    var groups = tree_ctr_groups_host(tree_members, cards, n)

    var member_codes = List[CombU32Ptr]()
    member_codes.append(_u32c(side0))
    member_codes.append(_u32c(side2))
    member_codes.append(_u32c(f1))
    member_codes.append(_u32c(f3))
    var cmap = combination_map_from_groups(
        combination_fixture_members(), member_codes, _u32c(groups.ids),
        groups.count, n,
    )

    var configs = List[TCtrConfig]()
    configs.append(TCtrConfig(CTR_BORDERS, TPrior(Float32(0.5), Float32(1.0)), 0, 0))
    configs.append(TCtrConfig(CTR_FEATURE_FREQ, TPrior(Float32(0.0), Float32(1.0)), 0, 1))
    var tables = combination_ctr_tables(
        groups.ids, groups.count, configs, target, 2, 0, COMB_FIXTURE_RAW
    )
    var ctrs = TCombinationCtrs()
    ctrs.add(cmap.copy(), tables^)
    ctrs.check(COMB_FIXTURE_RAW, COMB_FIXTURE_COLUMNS)
    # read through raw pointers above
    _ = len(side0)
    _ = len(side2)
    _ = len(f1)
    _ = len(f3)
    var n_groups = groups.count
    var ids = groups.ids.copy()
    return CombinationFixture(n, x^, ids^, n_groups, cmap^, ctrs^, target^)


def combination_predict_rows() -> List[Float32]:
    """Five predict rows, ROW-major, four raw features each:
    0  a tuple of fitted values (learn row 0's, by construction below)
    1  the 5-code feature at 5: a category the fit never saw
    2  a NaN on the float member (side 1) with fitted categories
    3  the one-hot feature at 7: unseen code, side 0
    4  the 4-code feature at 9: unseen."""
    var rows: List[Float32] = [
        0.0, 0.0, 0.0, 0.0,
        0.9, 5.0, 1.0, 2.0,
        0.0, 1.0, 1.0, 1.0,
        0.9, 2.0, 7.0, 3.0,
        0.1, 3.0, 0.0, 9.0,
    ]
    rows[8] = bitcast[DType.float32](UInt32(0x7FC00000))
    return rows^


def codes_reproduce_groups() raises:
    var n = 997
    var fx = combination_fixture(n)
    var codes = List[UInt32](length=n, fill=UInt32(0))
    combination_codes_into(
        fx.cmap, _f32c(fx.x), n, COMB_FIXTURE_RAW, False, _u32c(codes)
    )
    for r in range(n):
        _need(codes[r] == fx.group_ids[r], "a learn row's derived code is its group id (column-major)")
    # the same pool row-major
    var xr = List[Float32](length=COMB_FIXTURE_RAW * n, fill=Float32(0.0))
    for r in range(n):
        for f in range(COMB_FIXTURE_RAW):
            xr[r * COMB_FIXTURE_RAW + f] = fx.x[f * n + r]
    var codes_r = List[UInt32](length=n, fill=UInt32(7))
    combination_codes_into(
        fx.cmap, _f32c(xr), n, COMB_FIXTURE_RAW, True, _u32c(codes_r)
    )
    for r in range(n):
        _need(codes_r[r] == fx.group_ids[r], "a learn row's derived code is its group id (row-major)")
    _need(fx.cmap.groups() == fx.n_groups, "one tuple per group")
    _ = len(xr)
    print("  combination codes reproduce the fit's group ids")


def unseen_and_refusals() raises:
    var n = 997
    var fx = combination_fixture(n)
    var g = fx.cmap.groups()
    var rows = combination_predict_rows()
    # row 0 takes learn row 0's raw values
    for f in range(COMB_FIXTURE_RAW):
        rows[f] = fx.x[f * n]
    var codes = List[UInt32](length=5, fill=UInt32(0))
    combination_codes_into(
        fx.cmap, _f32c(rows), 5, COMB_FIXTURE_RAW, True, _u32c(codes)
    )
    _need(codes[0] == fx.group_ids[0], "a fitted tuple finds its group")
    _need(Int(codes[1]) == g, "a category the fit never saw is the unseen code")
    _need(Int(codes[4]) == g, "an unseen category on the last member is the unseen code")
    # rows 2 and 3 are whatever the pool held for (1, 1, 1, 1) and
    # (1, 0, 2, 3): compare against a scan of the map's tuples
    var want2 = g
    var want3 = g
    for i in range(g):
        ref t = fx.cmap.tuples
        if t[4 * i] == UInt32(1) and t[4 * i + 1] == UInt32(1) and (
            t[4 * i + 2] == UInt32(1) and t[4 * i + 3] == UInt32(1)
        ):
            want2 = i
        if t[4 * i] == UInt32(1) and t[4 * i + 1] == UInt32(0) and (
            t[4 * i + 2] == UInt32(2) and t[4 * i + 3] == UInt32(3)
        ):
            want3 = i
    _need(Int(codes[2]) == want2, "a NaN on a float member takes nan_side")
    _need(Int(codes[3]) == want3, "an unseen one-hot code is side 0")

    rows[5] = Float32(1.5)
    var refused = False
    try:
        combination_codes_into(
            fx.cmap, _f32c(rows), 5, COMB_FIXTURE_RAW, True, _u32c(codes)
        )
    except error:
        refused = True
    _need(refused, "a categorical value that is not a whole number is refused")
    _ = len(rows)
    _ = len(codes)
    print("  unseen tuples take the unseen code; a non-code is refused")


def columns_equal_table_values() raises:
    var n = 997
    var fx = combination_fixture(n)
    var dst = List[Float32](length=COMB_FIXTURE_COLUMNS * n, fill=Float32(-1.0))
    expand_combination_columns(
        fx.ctrs, _f32c(fx.x), n, COMB_FIXTURE_RAW, False,
        COMB_FIXTURE_COLUMNS, _f32c(dst),
    )
    for t in range(len(fx.ctrs.tables)):
        ref tab = fx.ctrs.tables[t]
        _need(tab.column == COMB_FIXTURE_RAW + t, "the tables fill the last model columns")
        for r in range(n):
            var want = tab.value_for(Int(fx.group_ids[r]))
            var got = dst[tab.column * n + r]
            _need(
                bitcast[DType.uint32](want) == bitcast[DType.uint32](got),
                "a tree-CTR column is its table's value at the row's group",
            )
    for i in range(COMB_FIXTURE_RAW * n):
        _need(dst[i] == Float32(-1.0), "the other model columns are not written")

    # an unseen row takes the prior
    var rows = combination_predict_rows()
    var out = List[Float32](length=COMB_FIXTURE_COLUMNS * 5, fill=Float32(-1.0))
    expand_combination_columns(
        fx.ctrs, _f32c(rows), 5, COMB_FIXTURE_RAW, True,
        COMB_FIXTURE_COLUMNS, _f32c(out),
    )
    for t in range(len(fx.ctrs.tables)):
        ref tab = fx.ctrs.tables[t]
        var empty = tab.empty_value()
        _need(
            bitcast[DType.uint32](out[tab.column * 5 + 1])
            == bitcast[DType.uint32](empty),
            "an unseen combination takes the table's empty value",
        )
    _ = len(dst)
    _ = len(rows)
    _ = len(out)
    print("  tree-CTR model columns equal the simple table values")


def text_round_trip() raises:
    var fx = combination_fixture(211)
    var maps = List[TCombinationMap]()
    maps.append(fx.cmap.copy())
    var text = combination_maps_text(maps)
    var back = combination_maps_from_text(text)
    _need(len(back) == 1, "one map read back")
    _need(back[0].same(fx.cmap), "the text reads back to the same map")

    var descending = String(
        "combination 0 members 1 groups 2\n"
        "combination_member 0 0 cat 1\n"
        "combination_group 0 0 3\n"
        "combination_group 0 1 2\n"
    )
    var refused = False
    try:
        var bad = combination_maps_from_text(descending)
        _ = len(bad)
    except error:
        refused = True
    _need(refused, "tuples that do not ascend are refused")

    var short = String(
        "combination 0 members 2 groups 1\n"
        "combination_member 0 0 cat 1\n"
    )
    refused = False
    try:
        var bad = combination_maps_from_text(short)
        _ = len(bad)
    except error:
        refused = True
    _need(refused, "a map missing a member is refused")
    print("  combination map text round trip")


def run_checks() raises:
    codes_reproduce_groups()
    unseen_and_refusals()
    columns_equal_table_values()
    text_round_trip()
