# SPDX-License-Identifier: Apache-2.0
"""RUN half of the splitters' x_metrics programs (lane py-msel-prep-cnn,
2026-11-14): the K-fold row table (`fold_rows`), its shuffled form and
StratifiedKFold's one-program folds (`group_sort`, the per-class `permute`
draws, `strat_codes`, `fold_rows`), and the counter RNG's draw stage, built
on a typed `MetricProgramPlan` (core/metrics_program_plan.mojo) from typed
values only.

bindings/msel_split_door.mojo builds these programs through here (its
`MselRng` draws through `MselCounterRng`; the door keeps the pins of the
caller buffers and the Python result lists), and a native check builds the
same plan and runs it with `metrics_plan_execute` (bindings/arena_prog_run.mojo)
on its own buffers. Same arena layout, same stage records, same draws in the
same order as the door built before: the rows do not move. The one change:
the stratified fold-by-class offsets are program-owned constant words
(`input_words`) instead of a pinned int32 Array, the same words at the same
offset.

No Python object is read, built or called here.
"""
from core.metrics_program_plan import MetricProgramPlan

#: `_expansion_metrics._FOLD_ROWS_BOUND`: the largest K-fold row table.
comptime MSEL_SPLIT_FOLD_ROWS_BOUND = 1 << 27
comptime MSEL_SPLIT_GOLDEN: UInt64 = 0x9E3779B97F4A7C15
comptime MSEL_SPLIT_NONE = -1

comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]


def msel_split_mix64(var x: UInt64) -> UInt64:
    """splitmix64's finalizer (`_expansion_metrics._mix64`), in wrapping
    64-bit arithmetic."""
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB
    return x ^ (x >> 31)


def msel_split_signed32(v: UInt64) -> Int:
    """A 32-bit word as the signed stage parameter."""
    var w = Int(v & 0xFFFFFFFF)
    return w - (1 << 32) if w >= (1 << 31) else w


@fieldwise_init
struct MselCounterRng(ImplicitlyCopyable, Movable):
    """`CounterRng`: draw k of seed s is a device permutation keyed by
    `_mix64(s * GOLDEN + k + 1)`."""

    var seed: UInt64
    var draws: UInt64

    def salt(mut self) -> UInt64:
        var s = msel_split_mix64(self.seed * MSEL_SPLIT_GOLDEN + self.draws + 1)
        self.draws += 1
        return s

    def permute_stage(mut self, mut plan: MetricProgramPlan, n: Int) raises -> Int:
        """The next draw as a `permute` stage of `plan`; returns its slot."""
        var s = self.salt()
        var out = plan.alloc(max(n, 1))
        plan.stage_op("permute", 1, [n, out, msel_split_signed32(s), msel_split_signed32(s >> 32)])
        return out


@fieldwise_init
struct MselFoldRows(ImplicitlyCopyable, Movable):
    """Where a `fold_rows` stage leaves its table: fold f's test rows (int64)
    at word `out + 2*n*f`, then its train rows; fold sizes (int32) at
    `sizes`."""

    var n: Int
    var k: Int
    var out: Int
    var sizes: Int

    def size(self, base: Int, f: Int) -> Int:
        """Fold f's test row count, read from the run arena at `base`."""
        return Int(_I32P(unsafe_from_address=base + 4 * self.sizes)[f])

    def test_rows(self, base: Int, f: Int) -> Int:
        """The address of fold f's ascending test rows (int64)."""
        return base + 4 * (self.out + 2 * self.n * f)

    def train_rows(self, base: Int, f: Int) -> Int:
        """The address of fold f's ascending train rows (int64)."""
        return base + 4 * (self.out + 2 * self.n * f + 2 * self.size(base, f))


def msel_fold_rows_fits(n: Int, k: Int) -> Bool:
    """Whether the K-fold row table program applies (else the caller takes
    its other route, no draw made)."""
    return n >= 2 and k >= 1 and 2 * n * k <= MSEL_SPLIT_FOLD_ROWS_BOUND


def msel_fold_rows_stage(mut plan: MetricProgramPlan, n: Int, k: Int, code: Int, order: Int) raises -> MselFoldRows:
    """The `fold_rows` stage and its two declared outputs."""
    var out_slot = plan.alloc(2 * n * k)
    var out = plan.want(out_slot, 2 * n * k, -1, 1)
    var sz_slot = plan.alloc(k)
    var sz = plan.want(sz_slot, k, -1, 1)
    plan.stage_op("fold_rows", 1, [n, k, code, order, out, sz])
    return MselFoldRows(n, k, out, sz)


def msel_shuffled_fold_rows_plan(mut plan: MetricProgramPlan, n: Int, k: Int,
                                 mut rng: MselCounterRng) raises -> MselFoldRows:
    """`fold_rows(n, k, rng=rng)`: one permutation draw, then the table.
    The caller checks `msel_fold_rows_fits` first."""
    var order = rng.permute_stage(plan, n)
    var code = plan.scratch(n)
    return msel_fold_rows_stage(plan, n, k, code, order)


def msel_stratified_fold_rows_plan(mut plan: MetricProgramPlan, enc_slot: Int, n: Int, counts: List[Int],
                                   table: Int, k: Int, mut rng: MselCounterRng,
                                   shuffle: Bool) raises -> MselFoldRows:
    """`stratified_fold_rows`: StratifiedKFold's folds in one program. The
    caller has made the n int32 class codes (classes in first-seen order)
    the plan input at `enc_slot`; `counts` are the m class sizes and `table`
    the address of the (k, m) int64 per-fold class allocation. The caller
    checks `msel_fold_rows_fits` first."""
    var m = len(counts)
    var off = plan.alloc(m + 1)
    var ordr = plan.scratch(n)
    plan.stage_op("group_sort", 1, [enc_slot, n, m, off, ordr])
    var pb = MSEL_SPLIT_NONE
    if shuffle:
        var at = 0
        for c in range(m):  # driver-loop: over classes (one permutation stage each)
            var o = rng.permute_stage(plan, counts[c])
            if c == 0:
                pb = o
            if o != pb + at:
                raise Error("RuntimeError: stratified_fold_rows: the class permutations are not contiguous")
            at += counts[c]
    var alloc = _I64P(unsafe_from_address=table)
    var cum = List[Int32](capacity=m * (k + 1))
    for c in range(m):  # driver-loop: over classes (fold-by-class offsets)
        var acc = 0
        cum.append(0)
        for f in range(k):  # driver-loop: over folds
            acc += Int(alloc[f * m + c])
            cum.append(Int32(acc))
    var cum_slot = plan.input_words(cum^)
    var code = plan.scratch(n)
    plan.stage_op("strat_codes", n, [enc_slot, ordr, off, pb, cum_slot, k, code])
    return msel_fold_rows_stage(plan, n, k, code, MSEL_SPLIT_NONE)
