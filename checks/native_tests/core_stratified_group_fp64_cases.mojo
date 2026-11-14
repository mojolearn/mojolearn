# SPDX-License-Identifier: Apache-2.0
"""StratifiedGroupKFold's group-fold plan under its explicit fused FP64
policy, moved from test_stratified_group_fp64_identity.py (2026-11-14; the
whole file).

The split door (bindings/msel_split_door.mojo msel_strat_group_split, no
shuffle) encodes the classes by first appearance, counts the group x class
table (bincount2_i32: core/hotpath_impl_rows.mojo on the host column,
core/hotpath_device.mojo device_bincount2_i32 on a GPU) and plans the folds
on the host (strat_group_plan_i32). The independent oracle is the former
test's `_reference_group_plan`, written here on its own: groups ordered by
descending class-count deviation (stable), each placed in the fold whose
mean per-class deviation of the normalized fold distribution is smallest,
ties to the smaller fold size and then the lower fold; every deviation sums
squares by fma(delta, delta, total) in binary64.

Cases (fixture, rows, groups, folds): (base, 600, 23, 3), (wide, 600, 23,
3), (ties, 600, 23, 3), (hashed, 600, 23, 3), (base, 577, 17, 4), (base,
631, 29, 3); groups = (row // 7) % groups. The labels are the identity
harness's metrics labels of those fixtures, stored by
checks/native_tests/fixtures/stratified_group/generate.py. Each group's fold
must equal the oracle's, so every row is tested exactly once and train and
test never share a row. The per-fold row lists themselves stay the door's.
"""
from std.math import fma, sqrt
from std.memory import bitcast

comptime STRATIFIED_GROUP_FIXTURES = 6
comptime _DIR = "checks/native_tests/fixtures/stratified_group/"


def require(condition: Bool, message: String) raises:
    if not condition:
        raise Error(message)


@fieldwise_init
struct StratCase(Copyable, Movable):
    var name: String
    var rows: Int
    var groups: Int
    var folds: Int


def strat_case(unit: Int) -> StratCase:
    if unit == 0:
        return StratCase("base", 600, 23, 3)
    if unit == 1:
        return StratCase("wide", 600, 23, 3)
    if unit == 2:
        return StratCase("ties", 600, 23, 3)
    if unit == 3:
        return StratCase("hashed", 600, 23, 3)
    if unit == 4:
        return StratCase("base", 577, 17, 4)
    return StratCase("base", 631, 29, 3)


def load_labels(c: StratCase) raises -> List[Int32]:
    var name = c.name + "-" + String(c.rows) + ".i32"
    var out = List[Int32](length=c.rows, fill=0)
    with open(_DIR + name, "r") as file:
        var bytes = file.read_bytes()
        require(len(bytes) == 4 * c.rows, "stratified label fixture extent changed: " + name)
        for i in range(c.rows):
            var w = UInt32(bytes[4 * i]) | (UInt32(bytes[4 * i + 1]) << 8) | (UInt32(bytes[4 * i + 2]) << 16) | (UInt32(bytes[4 * i + 3]) << 24)
            out[i] = bitcast[DType.int32](w)
    return out^


def group_codes(c: StratCase) -> List[Int32]:
    """(row // 7) % groups; the first appearances ascend, so these are the
    door's first-seen group codes too."""
    var out = List[Int32](length=c.rows, fill=0)
    for i in range(c.rows):
        out[i] = Int32((i // 7) % c.groups)
    return out^


def first_seen(labels: List[Int32]) -> Tuple[List[Int32], Int]:
    """Class codes renumbered by first appearance (the oracle's own
    `dict.fromkeys` order)."""
    var rank = Dict[Int, Int]()
    var out = List[Int32](length=len(labels), fill=0)
    for i in range(len(labels)):
        var v = Int(labels[i])
        if v not in rank:
            rank[v] = len(rank)
        out[i] = Int32(rank.get(v, 0))
    return (out^, len(rank))


def _deviation(values: List[Float64]) -> Float64:
    var mean = Float64(0)
    for v in values:
        mean += v
    mean = mean / Float64(len(values))
    var total = Float64(0)
    for v in values:
        var delta = v - mean
        total = fma(delta, delta, total)
    return sqrt(total / Float64(len(values)))


def oracle_plan(enc: List[Int32], k: Int, groups: List[Int32], m: Int, folds: Int) -> List[Int]:
    """The former `_reference_group_plan`: the fold of each group."""
    var table = List[Int](length=m * k, fill=0)
    for i in range(len(enc)):
        table[Int(groups[i]) * k + Int(enc[i])] += 1
    var counts = List[Int](length=k, fill=0)
    for g in range(m):
        for c in range(k):
            counts[c] += table[g * k + c]
    var dev = List[Float64](length=m, fill=0)
    for g in range(m):
        var row = List[Float64](length=k, fill=0)
        for c in range(k):
            row[c] = Float64(table[g * k + c])
        dev[g] = _deviation(row)
    # stable sort by -deviation (insertion sort keeps equal keys in order)
    var order = List[Int](length=m, fill=0)
    for g in range(m):
        order[g] = g
    for i in range(1, m):
        var g = order[i]
        var j = i - 1
        while j >= 0 and -dev[order[j]] > -dev[g]:
            order[j + 1] = order[j]
            j -= 1
        order[j + 1] = g
    var distribution = List[Int](length=folds * k, fill=0)
    var sizes = List[Int](length=folds, fill=0)
    var assignment = List[Int](length=m, fill=-1)
    for t in range(m):
        var g = order[t]
        var best = -1
        var best_score = Float64(0)
        for f in range(folds):
            var score = Float64(0)
            for c in range(k):
                var col = List[Float64](length=folds, fill=0)
                for j in range(folds):
                    var v = distribution[j * k + c] + (table[g * k + c] if j == f else 0)
                    col[j] = Float64(v) / Float64(counts[c])
                score += _deviation(col)
            score = score / Float64(k)
            if best < 0 or score < best_score or (score == best_score and sizes[f] < sizes[best]):
                best = f
                best_score = score
        assignment[g] = best
        for c in range(k):
            distribution[best * k + c] += table[g * k + c]
            sizes[best] += table[g * k + c]
    return assignment^


def check_plan(c: StratCase, enc: List[Int32], k: Int, groups: List[Int32], native: List[Int32]) raises:
    var tag = c.name + " rows " + String(c.rows) + " groups " + String(c.groups) + " folds " + String(c.folds)
    var want = oracle_plan(enc, k, groups, c.groups, c.folds)
    require(len(native) == c.groups, tag + ": plan extent")
    for g in range(c.groups):
        require(Int(native[g]) == want[g], tag + ": group " + String(g) + " fold " + String(native[g])
                + ", the fused FP64 oracle says " + String(want[g]))
    var seen = List[Int](length=c.rows, fill=0)
    for f in range(c.folds):
        for i in range(c.rows):
            if Int(native[Int(groups[i])]) == f:
                seen[i] += 1
    for i in range(c.rows):
        require(seen[i] == 1, tag + ": row " + String(i) + " is tested " + String(seen[i]) + " times")
