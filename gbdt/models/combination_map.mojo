# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The combination map: what a model keeps of a tree CTR (a feature
combination) so that a row it has never seen can be scored.

A tree CTR is the target statistic of a COMBINATION of members: categorical
features and the 0/1 sides of binary splits (`gbdt/ctrs/tree_ctr.mojo`). At
predict it is a SIMPLE CTR over a derived categorical column:

    derived code of a row = the rank of the row's member tuple among the
                            tuples the learn pool held (its group id)
    value                 = the ordinary `TCtrValueTable` lookup at that code

so the table, its arithmetic, its unseen-category value and its device
statement are the simple CTR's, unchanged. What is new is this map: the
members and the ascending list of member tuples present, one per group.

MEMBERS, in the order the fit added them to the grouping (the order is part
of the model: the member added LAST is the most significant, which is the
order of the device builder's segments and of `tree_ctr_refine_host`'s
ranks, so the group ids a fit tallied its table under are the positions of
the tuples here):

    cat      the dense category code of raw input feature f
    float    1 when raw input feature f's value is above the border, else 0
             (a NaN takes `nan_side`: the side the model's own NaN
             substitution puts it on)
    one_hot  1 when raw input feature f's dense code equals `value`, else 0

Members read the RAW input, never a model column, so the map needs neither
the model's borders nor an earlier CTR column.

UNSEEN. A tuple the learn pool never held (which includes any category the
fit never saw: step 1 gives it the code one past the fitted ones) takes the
code `groups`, one past the table, and the table answers with its
`empty_value`: the prior, as the reference does for an unseen combination.
There is no hash, so no two combinations share an entry.

ONE STATEMENT, BOTH COLUMNS. `combination_row_code` is the whole per-row
computation (member values, the lexicographic binary search) on plain
pointers. The host column calls it across the workers
(`combination_codes_into`); the device kernel
(`gbdt/models/kernel/combination_code.mojo`) calls the same function, one
thread per row. It is comparisons and integer arithmetic only, so the two
give the same codes on every vendor.

PARALLEL PLAN. Rows are the independent units: every row pass here
(`combination_codes_into`, `expand_combination_columns`,
`combination_map_from_groups`, the ascending check) is split across the
host's workers by `host_parallelize`, the worker count from
`host_predict_task_count_work`. The loops over members, maps and tables are
loops over model structure.

GPU-free: no device import.
"""

from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.host_predict_threads import (
    host_predict_chunk,
    host_predict_task_count_work,
)
from gbdt.ctrs.ctr import CTR_BORDERS, CTR_COUNTER, CTR_FEATURE_FREQ, TCtrConfig
from gbdt.models.ctr_value_table import (
    TCtrValueTable,
    build_ctr_tables,
    build_ctr_tables_from_counts,
    dense_category_code,
)

comptime COMB_MEMBER_CAT = 0
comptime COMB_MEMBER_FLOAT = 1
comptime COMB_MEMBER_ONE_HOT = 2

#: words per member in the flat form the row statement reads:
#: `[kind, raw feature, one-hot value, nan side]`
comptime COMB_MEMBER_WORDS = 4

#: the pointer types of the row statement. `MutAnyOrigin` because a kernel's
#: arguments are these; the host hands it list storage under the same type.
comptime CombF32Ptr = MutPointer[Float32, MutAnyOrigin]
comptime CombU32Ptr = MutPointer[UInt32, MutAnyOrigin]
comptime CombI64Ptr = MutPointer[Int64, MutAnyOrigin]


# ===========================================================================
# THE ROW STATEMENT (host workers and the device kernel)
# ===========================================================================


@always_inline
def combination_member_value(
    x: CombF32Ptr,
    n: Int,
    n_raw: Int,
    row_major: Int32,
    r: Int,
    mi: CombI64Ptr,
    mf: CombF32Ptr,
    j: Int,
) -> Int:
    """Member `j`'s value for row `r`: a dense code, or a side 0/1; -1 when
    a categorical or one-hot member's input is not a dense category code
    (`dense_category_code`'s three tests: finite, in `[0, 2^32)`, a whole
    number). Every Float32 at or above 2^24 is a whole number, and below it
    the Int-to-Float32 conversion is exact, so the test is the same word on
    every device."""
    var kind = Int(mi[COMB_MEMBER_WORDS * j])
    var f = Int(mi[COMB_MEMBER_WORDS * j + 1])
    var v: Float32
    if row_major != Int32(0):
        v = x[r * n_raw + f]
    else:
        v = x[f * n + r]
    if kind == COMB_MEMBER_FLOAT:
        if v != v:
            return Int(mi[COMB_MEMBER_WORDS * j + 3])
        if v > mf[j]:
            return 1
        return 0
    # a NaN fails both comparisons
    if not (v >= Float32(0.0) and v < Float32(4294967296.0)):
        return -1
    var c = Int(v)
    if v < Float32(16777216.0) and Float32(c) != v:
        return -1
    if kind == COMB_MEMBER_ONE_HOT:
        if c == Int(mi[COMB_MEMBER_WORDS * j + 2]):
            return 1
        return 0
    return c


@always_inline
def combination_row_code(
    x: CombF32Ptr,
    n: Int,
    n_raw: Int,
    row_major: Int32,
    r: Int,
    mi: CombI64Ptr,
    mf: CombF32Ptr,
    m: Int,
    tuples: CombU32Ptr,
    g: Int,
) -> Int:
    """Row `r`'s derived code: the position of its member tuple in `tuples`
    (`g` tuples of `m` words, ascending with the LAST member most
    significant), or `g` when the tuple is not there. `-(1 + f)` when raw
    feature `f` is the lowest feature of the row whose value is not a dense
    category code."""
    var bad = -1
    for j in range(m):  # small-loop(m: members of one combination, bounded by max_ctr_complexity plus the tree depth): one row's refusal test
        if combination_member_value(x, n, n_raw, row_major, r, mi, mf, j) < 0:
            var f = Int(mi[COMB_MEMBER_WORDS * j + 1])
            if bad < 0 or f < bad:
                bad = f
    if bad >= 0:
        return -(1 + bad)
    var lo = 0
    var hi = g
    while lo < hi:
        var mid = (lo + hi) // 2
        # is tuples[mid] below the row's tuple?
        var below = False
        var j = m - 1
        while j >= 0:
            var a = combination_member_value(
                x, n, n_raw, row_major, r, mi, mf, j
            )
            var b = Int(tuples[mid * m + j])
            if b < a:
                below = True
                break
            if b > a:
                break
            j -= 1
        if below:
            lo = mid + 1
        else:
            hi = mid
    if lo >= g:
        return g
    for j in range(m):  # small-loop(m: members of one combination): one row's equality test
        if Int(tuples[lo * m + j]) != combination_member_value(
            x, n, n_raw, row_major, r, mi, mf, j
        ):
            return g
    return lo


@no_inline
def _calc_host(
    count_in_class: Float32,
    total_count: Float32,
    prior_num: Float32,
    prior_denom: Float32,
    shift: Float32,
    scale: Float32,
) -> Float32:
    """`TCtrValueTable.calc`, on plain values (a worker task cannot raise,
    and `value_for` can). A function boundary, as that one is."""
    var ctr = (count_in_class + prior_num) / (total_count + prior_denom)
    return (ctr + shift) * scale


# ===========================================================================
# THE MAP
# ===========================================================================


@fieldwise_init
struct TCombinationMember(Copyable, ImplicitlyCopyable, Movable):
    """One member of a combination; see the module note."""

    var kind: Int
    #: the raw input feature the member reads
    var feature: Int
    #: float: the border the value is compared above
    var border: Float32
    #: one_hot: the dense code whose rows are on side 1
    var value: Int
    #: float: the side a NaN takes (0 or 1)
    var nan_side: Int


def combination_cat_member(feature: Int) -> TCombinationMember:
    return TCombinationMember(COMB_MEMBER_CAT, feature, Float32(0.0), 0, 0)


def combination_float_member(
    feature: Int, border: Float32, nan_side: Int
) -> TCombinationMember:
    return TCombinationMember(COMB_MEMBER_FLOAT, feature, border, 0, nan_side)


def combination_one_hot_member(feature: Int, value: Int) -> TCombinationMember:
    return TCombinationMember(
        COMB_MEMBER_ONE_HOT, feature, Float32(0.0), value, 0
    )


def _tuples_ascend(tuples: CombU32Ptr, m: Int, g: Int) -> Bool:
    """Every tuple strictly above the one before it (last member most
    significant), the pairs split across the workers."""
    if g <= 1:
        return True
    var pairs = g - 1
    var units = max(1, host_predict_task_count_work(pairs, pairs * m))
    var chunk = host_predict_chunk(pairs, units)
    var flags = List[UInt32](length=units, fill=UInt32(0))
    var fp = CombU32Ptr(unsafe_from_address=Int(flags.unsafe_ptr()))

    def scan(unit: Int) {imm chunk, imm pairs, imm tuples, imm m, imm fp}:
        var wrong = UInt32(0)
        for i in range(unit * chunk, min((unit + 1) * chunk, pairs)):
            var above = False
            var j = m - 1
            while j >= 0:
                var a = tuples[i * m + j]
                var b = tuples[(i + 1) * m + j]
                if b > a:
                    above = True
                    break
                if b < a:
                    break
                j -= 1
            if not above:
                wrong = UInt32(1)
        fp[unit] = wrong

    host_parallelize(scan, units)
    var ok = True
    for u in range(units):
        if flags[u] != UInt32(0):
            ok = False
    return ok


struct TCombinationMap(Copyable, Movable):
    """The members of one combination and the member tuples the learn pool
    held, `groups()` tuples of `len(members)` words, ascending with the last
    member most significant. A row's derived code is its tuple's position."""

    var members: List[TCombinationMember]
    var tuples: List[UInt32]

    def __init__(
        out self, var members: List[TCombinationMember], var tuples: List[UInt32]
    ) raises:
        var m = len(members)
        if m < 1:
            raise Error("a combination map needs at least one member")
        if len(tuples) == 0 or len(tuples) % m != 0:
            raise Error(
                "a combination map of " + String(m) + " members holds "
                + String(len(tuples)) + " tuple words"
            )
        var g = len(tuples) // m
        for j in range(m):
            var kind = members[j].kind
            if (
                kind != COMB_MEMBER_CAT
                and kind != COMB_MEMBER_FLOAT
                and kind != COMB_MEMBER_ONE_HOT
            ):
                raise Error("combination member " + String(j) + " has an unknown kind")
            if members[j].feature < 0:
                raise Error("combination member " + String(j) + " names no input feature")
            if kind == COMB_MEMBER_FLOAT:
                if members[j].border != members[j].border:
                    raise Error("combination member " + String(j) + " has a NaN border")
                if members[j].nan_side != 0 and members[j].nan_side != 1:
                    raise Error("combination member " + String(j) + " has a NaN side that is not 0 or 1")
            if kind == COMB_MEMBER_ONE_HOT and members[j].value < 0:
                raise Error("combination member " + String(j) + " has a negative one-hot code")
            if kind != COMB_MEMBER_CAT:
                # a split's side is 0 or 1 in every tuple
                for i in range(g):  # small-loop(g: tuples of one map, once per model load): load-time check
                    if tuples[i * m + j] > UInt32(1):
                        raise Error(
                            "combination tuple " + String(i)
                            + " holds a split side that is not 0 or 1"
                        )
        var tp = CombU32Ptr(unsafe_from_address=Int(tuples.unsafe_ptr()))
        if not _tuples_ascend(tp, m, g):
            raise Error(
                "combination tuples must ascend strictly (last member most"
                " significant): the position of a tuple is its code"
            )
        self.members = members^
        self.tuples = tuples^

    def groups(self) -> Int:
        return len(self.tuples) // len(self.members)

    def max_feature(self) -> Int:
        var top = 0
        for j in range(len(self.members)):
            if self.members[j].feature > top:
                top = self.members[j].feature
        return top

    def flat_words(self) -> List[Int64]:
        """The members as the row statement reads them."""
        var out = List[Int64](capacity=COMB_MEMBER_WORDS * len(self.members))
        for j in range(len(self.members)):
            out.append(Int64(self.members[j].kind))
            out.append(Int64(self.members[j].feature))
            out.append(Int64(self.members[j].value))
            out.append(Int64(self.members[j].nan_side))
        return out^

    def flat_borders(self) -> List[Float32]:
        var out = List[Float32](capacity=len(self.members))
        for j in range(len(self.members)):
            out.append(self.members[j].border)
        return out^

    def same(self, other: Self) -> Bool:
        if len(self.members) != len(other.members):
            return False
        if len(self.tuples) != len(other.tuples):
            return False
        for j in range(len(self.members)):
            ref a = self.members[j]
            ref b = other.members[j]
            if a.kind != b.kind or a.feature != b.feature or a.value != b.value:
                return False
            if a.nan_side != b.nan_side:
                return False
            if bitcast[DType.uint32](a.border) != bitcast[DType.uint32](b.border):
                return False
        for i in range(len(self.tuples)):
            if self.tuples[i] != other.tuples[i]:
                return False
        return True

    def to_text(self, ordinal: Int) -> String:
        """The map's records (see `TCombinationMapReader` for the grammar).
        A border is written as its 32 bits in decimal: exact, and read back
        without a float parser."""
        var m = len(self.members)
        var g = self.groups()
        var out = (
            String("combination ") + String(ordinal) + " members "
            + String(m) + " groups " + String(g) + "\n"
        )
        for j in range(m):
            ref member = self.members[j]
            var head = (
                String("combination_member ") + String(ordinal) + " "
                + String(j) + " "
            )
            if member.kind == COMB_MEMBER_CAT:
                out += head + "cat " + String(member.feature) + "\n"
            elif member.kind == COMB_MEMBER_FLOAT:
                out += (
                    head + "float " + String(member.feature) + " border_bits "
                    + String(Int(bitcast[DType.uint32](member.border)))
                    + " nan_side " + String(member.nan_side) + "\n"
                )
            else:
                out += (
                    head + "one_hot " + String(member.feature) + " value "
                    + String(member.value) + "\n"
                )
        for i in range(g):
            var line = (
                String("combination_group ") + String(ordinal) + " " + String(i)
            )
            for j in range(m):
                line += " " + String(Int(self.tuples[i * m + j]))
            out += line + "\n"
        return out^


def combination_map_from_groups(
    var members: List[TCombinationMember],
    member_codes: List[CombU32Ptr],
    group_ids: CombU32Ptr,
    n_groups: Int,
    n_rows: Int,
) raises -> TCombinationMap:
    """The map of a fitted combination from the fit's own per-row values:
    `member_codes[j][r]` is member `j`'s value for learn row `r` (a dense
    code or a side) and `group_ids[r]` the row's group under the members
    added in that order (`tree_ctr_groups_host`, or the device builder's
    segment number). Every row writes its tuple into its group's slot; the
    rows of a group hold the same tuple, so the tasks never disagree on a
    word. The ascending order is then checked, not assumed."""
    var m = len(members)
    if m < 1 or len(member_codes) != m:
        raise Error("combination map: one code column per member")
    if n_groups < 1 or n_rows < 1:
        raise Error("combination map: no groups")
    var tuples = List[UInt32](length=n_groups * m, fill=UInt32(0))
    var tp = CombU32Ptr(unsafe_from_address=Int(tuples.unsafe_ptr()))
    var units = max(1, host_predict_task_count_work(n_rows, n_rows))
    var chunk = host_predict_chunk(n_rows, units)
    # driver-loop: over the members of the combination
    for j in range(m):
        var codes = member_codes[j]

        def fill(unit: Int) {imm chunk, imm n_rows, imm codes, imm group_ids, imm tp, imm m, imm j, imm n_groups}:
            for r in range(unit * chunk, min((unit + 1) * chunk, n_rows)):
                var gid = Int(group_ids[r])
                if gid < n_groups:
                    tp[gid * m + j] = codes[r]

        host_parallelize(fill, units)
    return TCombinationMap(members^, tuples^)


def combination_codes_into(
    cmap: TCombinationMap,
    x: CombF32Ptr,
    n_rows: Int,
    n_raw: Int,
    row_major: Bool,
    dst: CombU32Ptr,
) raises:
    """`dst[r]` = row `r`'s derived code under `cmap` (`cmap.groups()` for an
    unseen tuple), rows across the workers. `x` is the raw input, `n_rows *
    n_raw` values, row- or column-major. A categorical or one-hot member
    whose input is not a dense category code is refused with
    `dense_category_code`'s message for the lowest such feature and its
    lowest row, the cell the device statement reports."""
    if cmap.max_feature() >= n_raw:
        raise Error(
            "a combination member reads input feature "
            + String(cmap.max_feature()) + " of " + String(n_raw)
        )
    if n_rows <= 0:
        return
    var words = cmap.flat_words()
    var borders = cmap.flat_borders()
    var mi = CombI64Ptr(unsafe_from_address=Int(words.unsafe_ptr()))
    var mf = CombF32Ptr(unsafe_from_address=Int(borders.unsafe_ptr()))
    var tp = CombU32Ptr(unsafe_from_address=Int(cmap.tuples.unsafe_ptr()))
    var m = len(cmap.members)
    var g = cmap.groups()
    var rm = Int32(1) if row_major else Int32(0)
    var units = max(
        1, host_predict_task_count_work(n_rows, n_rows * m * 8)
    )
    var chunk = host_predict_chunk(n_rows, units)
    # per task: the lowest refused feature and its lowest row, or -1
    var bad = List[Int64](length=2 * units, fill=Int64(-1))
    var bp = CombI64Ptr(unsafe_from_address=Int(bad.unsafe_ptr()))

    def rows(unit: Int) {imm chunk, imm n_rows, imm x, imm n_raw, imm rm, imm mi, imm mf, imm m, imm tp, imm g, imm dst, imm bp}:
        var bad_f = -1
        var bad_r = -1
        for r in range(unit * chunk, min((unit + 1) * chunk, n_rows)):
            var code = combination_row_code(
                x, n_rows, n_raw, rm, r, mi, mf, m, tp, g
            )
            if code < 0:
                var f = -(code + 1)
                if bad_f < 0 or f < bad_f:
                    bad_f = f
                    bad_r = r
                code = g
            dst[r] = UInt32(code)
        bp[2 * unit] = Int64(bad_f)
        bp[2 * unit + 1] = Int64(bad_r)

    host_parallelize(rows, units)
    # the scratch lists are read through raw pointers above
    _ = len(words)
    _ = len(borders)
    var first_f = -1
    var first_r = -1
    for u in range(units):
        var f = Int(bad[2 * u])
        if f >= 0 and (first_f < 0 or f < first_f):
            # tasks own ascending row ranges: the first task holding the
            # lowest feature holds its lowest row
            first_f = f
            first_r = Int(bad[2 * u + 1])
    if first_f >= 0:
        var v: Float32
        if row_major:
            v = x[first_r * n_raw + first_f]
        else:
            v = x[first_f * n_rows + first_r]
        _ = dense_category_code(v, first_f, first_r)
        raise Error(
            "categorical feature " + String(first_f) + " row "
            + String(first_r) + " is not a dense category code"
        )


# ===========================================================================
# THE MODEL SECTION: maps and the tables over them
# ===========================================================================


struct TCombinationCtrs(Copyable, Movable):
    """A model's tree CTRs: the maps, and one `TCtrValueTable` per tree-CTR
    model column. A table's `source_feature` is the ORDINAL OF ITS MAP here
    (not a raw input feature: its input is the derived column) and its
    `column` the model column it feeds. The tables ascend by column and
    occupy the model's last `len(tables)` columns, after every raw and
    simple-CTR column."""

    var maps: List[TCombinationMap]
    var tables: List[TCtrValueTable]

    def __init__(out self):
        self.maps = List[TCombinationMap]()
        self.tables = List[TCtrValueTable]()

    def is_empty(self) -> Bool:
        return len(self.tables) == 0

    def add(
        mut self, var cmap: TCombinationMap, var tables: List[TCtrValueTable]
    ) raises:
        """One chosen combination: its map and its tables (one per config
        kept by the trees), `source_feature` already the next map ordinal
        (`combination_ctr_tables`)."""
        var ordinal = len(self.maps)
        for i in range(len(tables)):
            if tables[i].source_feature != ordinal:
                raise Error("a combination table must name its own map")
            if tables[i].categories() != cmap.groups():
                raise Error(
                    "a combination table holds " + String(tables[i].categories())
                    + " entries for " + String(cmap.groups()) + " groups"
                )
            self.tables.append(tables[i].copy())
        self.maps.append(cmap^)

    def first_column(self, n_columns: Int) -> Int:
        """The first tree-CTR model column of a model of `n_columns`."""
        return n_columns - len(self.tables)

    def check(self, n_raw: Int, n_columns: Int) raises:
        """What a loaded section must satisfy before any row is scored, so
        that no worker task and no kernel meets a table it cannot read."""
        var first = n_columns - len(self.tables)
        if first < 0:
            raise Error("more combination tables than model columns")
        for i in range(len(self.maps)):
            if self.maps[i].max_feature() >= n_raw:
                raise Error(
                    "combination " + String(i) + " reads input feature "
                    + String(self.maps[i].max_feature()) + " of " + String(n_raw)
                )
        for i in range(len(self.tables)):
            ref tab = self.tables[i]
            if tab.column != first + i:
                raise Error(
                    "combination tables must fill the model's last columns in"
                    " order; table " + String(i) + " names column "
                    + String(tab.column)
                )
            if tab.source_feature < 0 or tab.source_feature >= len(self.maps):
                raise Error("a combination table names no map")
            if tab.ctr_type == CTR_BORDERS:
                if tab.target_classes_count < 2 or tab.target_border_idx < 0 or (
                    tab.target_border_idx >= tab.target_classes_count
                ):
                    raise Error("a Borders combination table has invalid target classes")
                if len(tab.counts) % tab.target_classes_count != 0:
                    raise Error("a Borders combination table is not whole histograms")
            elif tab.ctr_type != CTR_FEATURE_FREQ and tab.ctr_type != CTR_COUNTER:
                raise Error("a combination table of an unsupported CTR type")
            if tab.categories() != self.maps[tab.source_feature].groups():
                raise Error("a combination table and its map disagree on the group count")


def combination_ctr_tables(
    group_ids: List[UInt32],
    n_groups: Int,
    ctr_configs: List[TCtrConfig],
    binarized_target: List[UInt8],
    target_classes_count: Int,
    map_ordinal: Int,
    first_column: Int,
) raises -> List[TCtrValueTable]:
    """The tables of one combination from the learn rows' group ids: the
    simple CTR's builder with the group id as the category code (host
    column)."""
    return build_ctr_tables(
        group_ids, n_groups, ctr_configs, binarized_target,
        target_classes_count, map_ordinal, first_column,
    )


def combination_ctr_tables_from_counts(
    counts: List[Int],
    histogram: List[Int],
    n_groups: Int,
    ctr_configs: List[TCtrConfig],
    target_classes_count: Int,
    map_ordinal: Int,
    first_column: Int,
    n_rows: Int,
) raises -> List[TCtrValueTable]:
    """The same tables from counts the device tallied over the builder's
    per-row segment ids (`gbdt/ctrs/fast_prep.mojo`'s histogram launches)."""
    return build_ctr_tables_from_counts(
        counts, histogram, n_groups, ctr_configs, target_classes_count,
        map_ordinal, first_column, n_rows,
    )


def expand_combination_columns(
    ctrs: TCombinationCtrs,
    x: CombF32Ptr,
    n_rows: Int,
    n_raw: Int,
    row_major: Bool,
    n_columns: Int,
    dst: CombF32Ptr,
) raises:
    """The host column's tree-CTR model columns: `dst` is the model's
    column-major matrix (`n_columns * n_rows`), and every tree-CTR column
    `c` gets `dst[c * n_rows + r]` = its table's value at row `r`'s derived
    code. One code pass per map, one value pass per table, rows across the
    workers; the other columns of `dst` are not touched."""
    ctrs.check(n_raw, n_columns)
    if n_rows <= 0 or ctrs.is_empty():
        return
    var codes = List[UInt32](length=n_rows, fill=UInt32(0))
    var cp = CombU32Ptr(unsafe_from_address=Int(codes.unsafe_ptr()))
    var units = max(1, host_predict_task_count_work(n_rows, n_rows * 4))
    var chunk = host_predict_chunk(n_rows, units)
    # driver-loop: over the model's combination maps
    for mo in range(len(ctrs.maps)):
        combination_codes_into(ctrs.maps[mo], x, n_rows, n_raw, row_major, cp)
        # driver-loop: over the model's tree-CTR tables (those of this map)
        for ti in range(len(ctrs.tables)):
            ref tab = ctrs.tables[ti]
            if tab.source_feature != mo:
                continue
            var counts = CombI64Ptr(unsafe_from_address=Int(tab.counts.unsafe_ptr()))
            var borders_type = tab.ctr_type == CTR_BORDERS
            var classes = tab.target_classes_count
            var border_idx = tab.target_border_idx
            var cats = tab.categories()
            var denom = Float32(tab.counter_denominator)
            var pn = tab.prior_num
            var pd = tab.prior_denom
            var sh = tab.shift
            var sc = tab.scale
            var column = dst + tab.column * n_rows

            def values(unit: Int) {imm chunk, imm n_rows, imm cp, imm counts, imm borders_type, imm classes, imm border_idx, imm cats, imm denom, imm pn, imm pd, imm sh, imm sc, imm column}:
                for r in range(unit * chunk, min((unit + 1) * chunk, n_rows)):
                    var code = Int(cp[r])
                    var value: Float32
                    if not borders_type:
                        var count = Float32(0.0)
                        if code < cats:
                            count = Float32(Int(counts[code]))
                        value = _calc_host(count, denom, pn, pd, sh, sc)
                    elif code >= cats:
                        value = _calc_host(
                            Float32(0.0), Float32(0.0), pn, pd, sh, sc
                        )
                    else:
                        var base = code * classes
                        var total = 0
                        var good = 0
                        for k in range(border_idx + 1):  # small-loop(border_idx: target classes of one Borders table): one row's histogram
                            total += Int(counts[base + k])
                        for k in range(border_idx + 1, classes):  # small-loop(classes: target classes of one Borders table): one row's histogram
                            good += Int(counts[base + k])
                        total += good
                        value = _calc_host(
                            Float32(good), Float32(total), pn, pd, sh, sc
                        )
                    column[r] = value

            host_parallelize(values, units)
    _ = len(codes)


# ===========================================================================
# TEXT
# ===========================================================================


struct TCombinationMapReader(Movable):
    """Reads the map records of a model text, one split line at a time:

        combination <ordinal> members <m> groups <g>
        combination_member <ordinal> <j> cat <feature>
        combination_member <ordinal> <j> float <feature> border_bits <u32> nan_side <0|1>
        combination_member <ordinal> <j> one_hot <feature> value <code>
        combination_group <ordinal> <i> <word> x m

    Strict: ordinals, member and group indices are the positions they land
    in, and a map is complete (every member and every group present) before
    the next begins and before `finish`."""

    var maps: List[TCombinationMap]
    var members: List[TCombinationMember]
    var tuples: List[UInt32]
    var want_members: Int
    var want_groups: Int
    var open: Bool

    def __init__(out self):
        self.maps = List[TCombinationMap]()
        self.members = List[TCombinationMember]()
        self.tuples = List[UInt32]()
        self.want_members = 0
        self.want_groups = 0
        self.open = False

    def _close(mut self) raises:
        if not self.open:
            return
        if len(self.members) != self.want_members:
            raise Error(
                "combination " + String(len(self.maps)) + " declares "
                + String(self.want_members) + " members and holds "
                + String(len(self.members))
            )
        if len(self.tuples) != self.want_groups * self.want_members:
            raise Error(
                "combination " + String(len(self.maps)) + " declares "
                + String(self.want_groups) + " groups and holds "
                + String(len(self.tuples) // max(self.want_members, 1))
            )
        var members = List[TCombinationMember]()
        var tuples = List[UInt32]()
        swap(members, self.members)
        swap(tuples, self.tuples)
        self.maps.append(TCombinationMap(members^, tuples^))
        self.open = False

    def feed(mut self, t: List[String]) raises -> Bool:
        """Take one record. False when the record is not a map record (the
        caller's own keywords)."""
        if len(t) == 0:
            return False
        if t[0] == String("combination"):
            self._close()
            if len(t) != 6 or t[2] != String("members") or t[4] != String("groups"):
                raise Error("malformed `combination` record")
            if Int(t[1]) != len(self.maps):
                raise Error(
                    "`combination` records must arrive in order; expected "
                    + String(len(self.maps)) + ", got " + t[1]
                )
            self.want_members = Int(t[3])
            self.want_groups = Int(t[5])
            if self.want_members < 1 or self.want_groups < 1:
                raise Error("a `combination` needs members and groups")
            self.open = True
            return True
        if t[0] == String("combination_member"):
            if not self.open or len(t) < 5:
                raise Error("a `combination_member` outside its `combination`")
            if Int(t[1]) != len(self.maps) or Int(t[2]) != len(self.members):
                raise Error("`combination_member` records must arrive in order")
            if len(self.tuples) != 0:
                raise Error("a `combination_member` after the first `combination_group`")
            var feature = Int(t[4])
            if t[3] == String("cat") and len(t) == 5:
                self.members.append(combination_cat_member(feature))
            elif t[3] == String("float") and len(t) == 9:
                if t[5] != String("border_bits") or t[7] != String("nan_side"):
                    raise Error("malformed float `combination_member`")
                var bits = Int(t[6])
                if bits < 0 or bits > 4294967295:
                    raise Error("`border_bits` is not 32 bits")
                self.members.append(
                    combination_float_member(
                        feature,
                        bitcast[DType.float32](UInt32(bits)),
                        Int(t[8]),
                    )
                )
            elif t[3] == String("one_hot") and len(t) == 7:
                if t[5] != String("value"):
                    raise Error("malformed one_hot `combination_member`")
                self.members.append(
                    combination_one_hot_member(feature, Int(t[6]))
                )
            else:
                raise Error("unknown `combination_member` kind '" + t[3] + "'")
            return True
        if t[0] == String("combination_group"):
            if not self.open or len(self.members) != self.want_members:
                raise Error("a `combination_group` before its members")
            if len(t) != 3 + self.want_members:
                raise Error("a `combination_group` holds one word per member")
            if Int(t[1]) != len(self.maps) or (
                Int(t[2]) * self.want_members != len(self.tuples)
            ):
                raise Error("`combination_group` records must arrive in order")
            if Int(t[2]) >= self.want_groups:
                raise Error("more `combination_group` records than declared")
            for j in range(self.want_members):
                var word = Int(t[3 + j])
                if word < 0 or word > 4294967295:
                    raise Error("a `combination_group` word is not 32 bits")
                self.tuples.append(UInt32(word))
            return True
        return False

    def finish(mut self) raises -> List[TCombinationMap]:
        """Close the last map and hand the maps over."""
        self._close()
        var out = List[TCombinationMap]()
        swap(out, self.maps)
        return out^


def combination_maps_text(maps: List[TCombinationMap]) -> String:
    var out = String("")
    for i in range(len(maps)):
        out += maps[i].to_text(i)
    return out^


def combination_maps_from_text(text: String) raises -> List[TCombinationMap]:
    """The inverse of `combination_maps_text` on a text holding only map
    records (the model reader feeds its own lines to the reader instead)."""
    var reader = TCombinationMapReader()
    for line in text.split("\n"):
        var t = List[String]()
        for piece in line.split(" "):
            var s = String(piece)
            if s.byte_length() > 0:
                t.append(s)
        if len(t) == 0:
            continue
        if not reader.feed(t):
            raise Error("not a combination record: '" + String(line) + "'")
    return reader.finish()
