# SPDX-License-Identifier: Apache-2.0
"""Shuffle-split row order against an independent counter-key oracle, moved
from test_native_cv_row_order.py (lane tmig-msel-metrics-tsa-1, 2026-11-14).

The splitters' draws are x_metrics programs (x_metrics/split.mojo:
`permute`, `rows64`, `group_sort`, `approx_mode`, `permute_drawn`,
`strat_take`, `take_rows64`) run through the metrics program entry. Each
fixture builds the program the model-selection door builds for the public
splitter (bindings/msel_split_door.mojo `msel_shuffle_split`,
`msel_strat_shuffle_split`, `MselRng`) and compares its rows with an
oracle written here: draw k of seed s sorts the rows by
mix((row + 1) * GOLDEN + (salt + 1) * MULT), ties by row, salt =
mix(s * GOLDEN + k + 1), all in wrapping unsigned 64-bit words.

Fixtures (CV_ROW_ORDER_FIXTURES):
  0-8   ShuffleSplit(4, test_size=1, train_size=n - 1, random_state=seed),
        n in {60, 64, 72} x seed in {0, 7, 29}: test then train is draw k's
        order and a permutation of 0..n-1; some draw's salt has bit 31 set
        (the seeds exercise the signed low and high parameter words)
  9-14  StratifiedShuffleSplit(3, train_size=22, test_size=11, random_state=
        seed), five classes of nine rows, seed in {0, 7, 2^31, 2^32 + 2^31,
        2^63 + 2^31, 2^64 - 2^31}: the per-class approximate-mode counts (a
        tie-cut group consumes one draw), each class's rows permuted by its
        own draw, the final train and test orders, every split continuing the
        draw count; train and test are disjoint
The group permutation of permutation_test_score (public GaussianNB fits and
the public metric) stays in Python.
"""
from bindings.arena_prog_run import MetricArenaOwner, MetricProgramInput, metrics_plan_execute
from core.metrics_program_plan import MetricProgramPlan
from core.native_descriptor_abi import NativeDescriptorCall

comptime CV_ROW_ORDER_FIXTURES = 15
comptime SHUFFLE_FIXTURES = 9
comptime GOLDEN: UInt64 = 0x9E3779B97F4A7C15
comptime MULTIPLIER: UInt64 = 0xBF58476D1CE4E5B9
comptime NONE = -1


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


# ---------------------------------------------------------------- the oracle


def mix(v: UInt64) -> UInt64:
    var x = v
    x = (x ^ (x >> 30)) * MULTIPLIER
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB
    return x ^ (x >> 31)


def draw_salt(seed: UInt64, draw: UInt64) -> UInt64:
    return mix(seed * GOLDEN + draw + 1)


def oracle_order(size: Int, seed: UInt64, draw: UInt64) -> List[Int]:
    """Rows 0..size-1 sorted by (key, row): an insertion sort of the oracle,
    independent of the product's heapsort."""
    var salt = draw_salt(seed, draw)
    var keys = List[UInt64](length=size, fill=0)
    for row in range(size):
        keys[row] = mix(UInt64(row + 1) * GOLDEN + (salt + 1) * MULTIPLIER)
    var rows = List[Int](length=size, fill=0)
    for i in range(size):
        var row = i
        var j = i
        while j > 0 and (keys[rows[j - 1]] > keys[row] or (keys[rows[j - 1]] == keys[row] and rows[j - 1] > row)):
            rows[j] = rows[j - 1]
            j -= 1
        rows[j] = row
    return rows^


struct OracleDraws(Movable):
    var seed: UInt64
    var draw: UInt64

    def __init__(out self, seed: UInt64):
        self.seed = seed
        self.draw = 0

    def permutation(mut self, size: Int) -> List[Int]:
        var result = oracle_order(size, self.seed, self.draw)
        self.draw += 1
        return result^

    def allocation(mut self, available: List[Int], wanted: Int) raises -> List[Int]:
        """scikit-learn's approximate mode in exact integers; a cut group of
        equal remainders takes the classes first in one draw's order."""
        var total = 0
        for value in available:
            total += value
        var assigned = List[Int]()
        var remainders = List[Int]()
        var given = 0
        for value in available:
            assigned.append(value * wanted // total)
            remainders.append(value * wanted % total)
            given += value * wanted // total
        var left = wanted - given
        var last = total  # remainders are below total: walk them descending
        while left > 0:
            var best = -1
            for r in remainders:
                if r < last and r > best:
                    best = r
            require(best >= 0, "allocation: the remainders ran out")
            last = best
            var tied = List[Int]()
            for index in range(len(remainders)):
                if remainders[index] == best:
                    tied.append(index)
            if left >= len(tied):
                for index in tied:
                    assigned[index] += 1
                left -= len(tied)
            else:
                var order = self.permutation(len(tied))
                for i in range(left):
                    assigned[tied[order[i]]] += 1
                left = 0
        var check = 0
        for value in assigned:
            check += value
        require(check == wanted, "allocation: the counts do not sum to the request")
        return assigned^


# ---------------------------------------------------------------- the program


def signed32(v: UInt64) -> Int:
    """A 32-bit word as the signed stage parameter the door records."""
    var w = Int(v & 0xFFFFFFFF)
    return w - (1 << 32) if w >= (1 << 31) else w


def run_plan(entry: NativeDescriptorCall, plan: MetricProgramPlan) raises -> MetricArenaOwner:
    """The program's constant inputs as records, the whole arena run."""
    var records = List[MetricProgramInput]()
    for k in range(len(plan.input_offsets)):  # driver-loop: over the program's input records
        require(plan.input_consts[k] >= 0, "run_plan: a check program holds only constant inputs")
        records.append(MetricProgramInput(plan.input_offsets[k], plan.const_address(k),
                                          len(plan.consts[plan.input_consts[k]])))
    var arena = metrics_plan_execute(entry, plan, records, False, False)
    _ = records^
    return arena^


def read_rows(arena: MetricArenaOwner, at: Int, count: Int) -> List[Int]:
    var rows = List[Int](length=count, fill=0)
    for i in range(count):
        rows[i] = arena.i64(at + 2 * i)
    return rows^


def same_rows(actual: List[Int], expected: List[Int], what: String) raises:
    require(len(actual) == len(expected), what + ": extent " + String(len(actual)) + " vs " + String(len(expected)))
    for i in range(len(actual)):
        require(actual[i] == expected[i], what + ": row " + String(i) + " is " + String(actual[i]) + ", the oracle "
                + String(expected[i]))


def shuffle_fixture(entry: NativeDescriptorCall, unit: Int) raises:
    var sizes: List[Int] = [60, 64, 72]
    var seeds: List[UInt64] = [0, 7, 29]
    var n = sizes[unit // 3]
    var seed = seeds[unit % 3]
    var n_test = 1
    var n_train = n - 1
    var plan = MetricProgramPlan()
    var words = List[Int]()
    for draw in range(4):  # the door's permutation_rows: one permute + rows64 per split
        var salt = draw_salt(seed, UInt64(draw))
        var o = plan.alloc(n)
        plan.stage_op("permute", 1, [n, o, signed32(salt), signed32(salt >> 32)])
        var w = plan.want(plan.alloc(2 * n), 2 * n, -1, 1)
        plan.stage_op("rows64", n, [o, w])
        words.append(w)
    var arena = run_plan(entry, plan)
    var what = "ShuffleSplit n=" + String(n) + " seed=" + String(seed)
    var high = False
    for draw in range(4):
        var perm = read_rows(arena, words[draw], n)
        # the door's split: test = perm[0:n_test], train = perm[n_test:n_test + n_train]
        var actual = List[Int]()
        for i in range(n_test):
            actual.append(perm[i])
        for i in range(n_train):
            actual.append(perm[n_test + i])
        same_rows(actual, oracle_order(n, seed, UInt64(draw)), what + " draw " + String(draw))
        var seen = List[Int](length=n, fill=0)
        for row in actual:
            require(row >= 0 and row < n, what + ": row out of range")
            seen[row] += 1
        for row in range(n):
            require(seen[row] == 1, what + ": not a permutation of the rows")
        if (draw_salt(seed, UInt64(draw)) & (1 << 31)) != 0:
            high = True
    _ = arena^
    require(high, what + ": no draw sets bit 31 of its salt")


def stratified_fixture(entry: NativeDescriptorCall, unit: Int) raises:
    var seeds: List[UInt64] = [0, 7, 0x80000000, 0x180000000, 0x8000000080000000, 0xFFFFFFFF80000000]
    var seed = seeds[unit - SHUFFLE_FIXTURES]
    var counts: List[Int] = [9, 9, 9, 9, 9]
    var k = len(counts)
    var n = 45
    var n_train = 22
    var n_test = 11
    var codes = List[Int32](length=n, fill=0)
    var offsets = List[Int]()
    var at = 0
    for c in range(k):
        offsets.append(at)
        for _ in range(counts[c]):
            codes[at] = Int32(c)
            at += 1
    var counts32 = List[Int32]()
    for c in range(k):
        counts32.append(Int32(counts[c]))
    var oracle = OracleDraws(seed)
    var draws: UInt64 = 0  # the door's MselRng.draws
    var what = "StratifiedShuffleSplit seed=" + String(seed)
    for split in range(3):  # the door's driver loop: one program per split
        var plan = MetricProgramPlan()
        var key = plan.input_words(codes.copy())
        var cnt = plan.input_words(counts32.copy())
        var off = plan.alloc(k + 1)
        var ordr = plan.alloc(n)
        plan.stage_op("group_sort", 1, [key, n, k, off, ordr])
        var modes = plan.alloc(2 * k)
        var used = plan.want(plan.alloc(2), 2, -1, 1)
        var w0 = signed32(seed)
        var w1 = signed32(seed >> 32)
        var w2 = signed32(draws)
        var w3 = signed32(draws >> 32)
        plan.stage_op("approx_mode", k, [k, cnt, NONE, n_train, modes, NONE, used, w0, w1, w2, w3])
        plan.stage_op("approx_mode", k, [k, cnt, modes, n_test, modes + k, used, used + 1, w0, w1, w2, w3])
        var drawn = plan.alloc(n)
        for c in range(k):
            var d = draws + UInt64(c)
            plan.stage_op("permute_drawn", 1, [counts[c], drawn + offsets[c], w0, w1, signed32(d), signed32(d >> 32),
                                               used, 2])
        var train = plan.alloc(n_train)
        var test = plan.alloc(n_test)
        plan.stage_op("strat_take", n, [k, ordr, off, drawn, modes, train, test])
        var last_tr = plan.alloc(n_train)
        var last_te = plan.alloc(n_test)
        var d_tr = draws + UInt64(k)
        var d_te = draws + UInt64(k + 1)
        plan.stage_op("permute_drawn", 1, [n_train, last_tr, w0, w1, signed32(d_tr), signed32(d_tr >> 32), used, 2])
        plan.stage_op("permute_drawn", 1, [n_test, last_te, w0, w1, signed32(d_te), signed32(d_te >> 32), used, 2])
        var tr_words = plan.want(plan.alloc(2 * n_train), 2 * n_train, -1, 1)
        var te_words = plan.want(plan.alloc(2 * n_test), 2 * n_test, -1, 1)
        plan.stage_op("take_rows64", n_train, [train, last_tr, tr_words])
        plan.stage_op("take_rows64", n_test, [test, last_te, te_words])
        var arena = run_plan(entry, plan)
        var got_train = read_rows(arena, tr_words, n_train)
        var got_test = read_rows(arena, te_words, n_test)
        draws += UInt64(arena.i32(used) + arena.i32(used + 1) + k + 2)
        _ = arena^

        # the oracle, in the generator's draw order
        var train_counts = oracle.allocation(counts, n_train)
        var rest = List[Int]()
        for c in range(k):
            rest.append(counts[c] - train_counts[c])
        var test_counts = oracle.allocation(rest, n_test)
        var wanted_train = List[Int]()
        var wanted_test = List[Int]()
        for c in range(k):
            var perm = oracle.permutation(counts[c])
            var stop = train_counts[c]
            for i in range(stop):
                wanted_train.append(offsets[c] + perm[i])
            for i in range(stop, stop + test_counts[c]):
                wanted_test.append(offsets[c] + perm[i])
        var final_train = oracle.permutation(len(wanted_train))
        var final_test = oracle.permutation(len(wanted_test))
        var expected_train = List[Int]()
        for i in range(len(final_train)):
            expected_train.append(wanted_train[final_train[i]])
        var expected_test = List[Int]()
        for i in range(len(final_test)):
            expected_test.append(wanted_test[final_test[i]])
        same_rows(got_train, expected_train, what + " split " + String(split) + " train")
        same_rows(got_test, expected_test, what + " split " + String(split) + " test")
        require(UInt64(oracle.draw) == draws, what + " split " + String(split) + ": the program consumed "
                + String(draws) + " draws, the oracle " + String(oracle.draw))
        for a in got_train:
            for b in got_test:
                require(a != b, what + " split " + String(split) + ": train and test share row " + String(a))


def run_row_order_fixture(entry: NativeDescriptorCall, unit: Int) raises:
    if unit < SHUFFLE_FIXTURES:
        shuffle_fixture(entry, unit)
    else:
        stratified_fixture(entry, unit)
