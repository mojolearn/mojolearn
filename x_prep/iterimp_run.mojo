# SPDX-License-Identifier: Apache-2.0
"""RUN of IterativeImputer with the built-in Bayesian ridge (lane
bm-rest-iterimp, 2026-11-23; docs/plans/PYTHON_AT_THE_BOUNDARY.md).

The door (bindings/prep_iterimp_native_door.mojo) is UNPACK (the parameters,
the initial SimpleImputer's words and, for transform, the fitted fields and
the addresses of the fitted vectors, read once into `IterImpFit` /
`IterImpApply`), then `iterimp_fit` / `iterimp_transform` here, then PACK
(the fitted attributes and the result, read from the arenas the runner
holds). These functions build the same x_prep programs, with the same
allocations, stages and order, as the door's former builders over its
Python-holding `EncProg`, so no bit moves:

  fit        1 the initial SimpleImputer fit (bindings/prep_encode_imputer_run.mojo
               `imputer_fit_stage`), 2 its transform (`imputer_transform_stage`),
             3 the statistics, the scale and the imputation orders,
             4 the n_nearest_features draws (when on), 5 every round of the
               ridge in one program, 6 the missing indicator (add_indicator)
  transform  the initial transform, the fitted steps in one program, the indicator

Programs run through an `IterImpRunner`: the door's runner over the x_prep
transport of the selected image (X as its Python owner, the GIL released
around the run), or a Mojo check's runner over the host transport. Owner 0
of a program is X; owner k >= 1 is the native buffer at `seconds[k - 1]`
(a vector this RUN holds, or a fitted vector the door keeps alive); -1 the
program's own parameter words. A runner keeps each run's host arena until
`release`; PACK reads the arenas the result names.

A SUPPLIED ESTIMATOR (`estimator=`, lane mf-iterimp, 2026-11-23) whose
class is a native member (core/member_abi.mojo) runs `iterimp_member_rounds`
after `iterimp_prepare`: the rounds of the door's former user-estimator loop
in the same program order (per (round, feature) step one gather program of
the step's training and predict blocks, the member's FIT on the gathered
training block and its PREDICT (PREDICT_STD with the same query's mean, for
sample_posterior) on the predict block, one scatter program into Xt; per
round the stop program), every member state kept per step for the door's
PACK (`imputation_sequence_`: a clone of the estimator per step written by
`member_pack`). The member operations go through the runner
(`IterImpRunner.member`), which releases the GIL around them when the
member's kind allows it.

A CLASSIFIER member (lane mf-iterimp3, 2026-11-23) is fitted as
scikit-learn's IterativeImputer fits one: `estimator.fit(X_train, y_train)`
on the float target column, so the classifier's own target check reads it
(`check_classification_targets`: a non-integral value is a continuous
target and refused in its words) and its classes are `np.unique(y_train)`.
Here the step's classes and int32 codes are made natively
(`iterimp_class_values`, `iterimp_class_codes`, across the host cores), the
member fits on the codes with the step's class count, and its predicted
codes are mapped back to the class values before the scatter (clipped, as
sklearn clips every prediction).

DEGENERATE SHAPES, as scikit-learn's `fit_transform` and
`_initial_imputation` do (sklearn/impute/_iterative.py, 1.9):
  * fewer than two kept features: `n_iter_ = 0`, no step, the initial
    imputation (`iterimp_initial_only`), for every estimator (none is
    fitted or called);
  * a kept column with no observed value (keep_empty_features): marked NOT
    missing (`mask_missing_values[:, empty] = False`), its values the initial
    fill. Its missing count is 0 (the orders and skip_complete read that),
    the mask programs mark it observed (`iterimp_unmark_empty`), and the
    tolerance scale reads its fill as an observed value. A supplied
    estimator's step for it fits on every row (y = the fill) and predicts
    none.

Serial by nature: the programs run one after the other, each reading what
the one before it wrote (the orders, the draws, the filled block); the
rounds are Gauss-Seidel sweeps over the features, so the features of one
round are NOT independent (each step's fit reads the imputation the step
before it wrote), and the sample_posterior draws advance one stream. The
stages themselves run across the image's cores (or the device), and so does
each member's own fit and predict and a classifier step's label work. No
Python import (tools/python_free_dirs.sh)."""
from std.math import isfinite, floor
from std.memory import unsafe_memcpy
from std.builtin.sort import sort
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from bindings.prep_encode_prog_run import (
    EncProgram, enc_input, enc_col_stats, enc_mark_missing, enc_floats_of, enc_ii_rows)
from bindings.prep_encode_kbins_run import enc_seed_words
from bindings.prep_encode_imputer_run import (
    ImputerFit, ImputerApply, IMP_CONSTANT, imputer_fit_stage, imputer_transform_stage)
from bindings.normal_dist_helpers_run import truncnorm_draws
from core.member_class_weight import member_cw_from_values, member_cw_values_refusal
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, MEMBER_OP_FIT, MEMBER_OP_PREDICT, MEMBER_OP_PREDICT_STD,
    MEMBER_F32, MEMBER_F64, MEMBER_I32, MEMBER_I64,
    member_new, member_call_of, member_own, member_free,
)
from core.member_notices import member_notices_take
from core.preprocessing_program import prep_program_stage as _stage

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]
comptime _IntP = MutPointer[Int, MutUntrackedOrigin]

#: scikit-learn's classifier target refusal (sklearn/utils/multiclass.py
#: `check_classification_targets`), the words a classifier's own fit raises
#: for a float target with a non-integral value.
comptime II_CONTINUOUS_WORDS = "Unknown label type: continuous. Maybe you are trying to fit a classifier, which expects discrete classes on a regression target with continuous values."

# x_prep stage codes (_expansion_prep._OPS).
comptime OP_EIGH = 18
comptime OP_II_MEAN = 53
comptime OP_II_GRAM = 54
comptime OP_II_SUB = 55
comptime OP_II_BR = 56
comptime OP_II_PREDICT = 57
comptime OP_II_SNAPSHOT = 58
comptime OP_II_CONV = 59
comptime OP_NAN_MASK = 60
comptime OP_II_SIGMA = 99
comptime OP_II_POST = 100
comptime OP_II_ROWABS = 107
comptime OP_II_GATHER = 138
comptime OP_II_SCATTER = 139
comptime OP_HCAT = 140
comptime OP_ABSCORR_CELL = 211
comptime OP_ABSCORR_NORM = 212
comptime OP_F2_WPICK = 234
comptime OP_C2_II_MISS = 258
comptime OP_C2_II_POS = 259
comptime OP_C2_II_ORD = 260
comptime OP_C2_II_RAND = 261
comptime OP_C2_COLMAX = 262

comptime II_NONE = -1
comptime II_GOLDEN: UInt64 = 0x9E3779B97F4A7C15
# `IterImpFit.order`: the C2_II_POS codes, then the random order.
comptime II_ASCENDING = 0
comptime II_DESCENDING = 1
comptime II_ROMAN = 2
comptime II_ARABIC = 3
comptime II_RANDOM = 4


trait IterImpRunner:
    """Runs one x_prep program (owner 0 X, owner k >= 1 the address
    `seconds[k - 1]`, -1 the program's own words) and keeps its host arena
    until `release`. Returns the arena's handle."""

    def run(mut self, mut prog: EncProgram, seconds: List[Int]) raises -> Int:
        ...

    def words(self, handle: Int) -> Int:
        """The address of the handle's host arena (float32 / int32 words)."""
        ...

    def release(mut self, handle: Int):
        ...

    def member(mut self, plan: MemberPlan, operation: Int, mut call: MemberCall) raises -> Int:
        """One member operation of a supplied estimator's rounds
        (`member_call`): the door's runner releases the GIL around it for a
        kind with MEMBER_GIL_FREE or MEMBER_GIL_DOOR (THE GIL RULE), any
        other kind is called with the GIL held; a check's runner calls it."""
        ...


@fieldwise_init
struct IterImpFit(Movable):
    """IterativeImputer.fit of the (n, d) block of owner 0. `init`: the
    initial SimpleImputer's request (missing_values, initial_strategy,
    fill_value, keep_empty_features). `mark`, `missing`: this estimator's
    numeric missing_values (marked NaN first). `stats_blocked`: the image's
    blocked column statistics. `work_on`: the R3 `work` region. `region`:
    the image has an output region. `order`: II_ASCENDING .. II_RANDOM.
    `max_iter`, `nnf` (n_nearest_features, II_NONE: None), `tol` (before the
    scale), `rng` (random_state as its 64-bit word), `lo` / `hi`
    (min_value / max_value per column, as given)."""
    var init: ImputerFit
    var n: Int
    var d: Int
    var mark: Bool
    var missing: Float64
    var stats_blocked: Bool
    var work_on: Bool
    var region: Bool
    var order: Int
    var skip: Bool
    var posterior: Bool
    var add_indicator: Bool
    var max_iter: Int
    var nnf: Int
    var tol: Float64
    var rng: UInt64
    var lo: List[Float64]
    var hi: List[Float64]


struct IterImpFitted(Movable):
    """What the fit leaves for PACK. The initial imputer: `statistics`,
    `fill` (the same values when `shared`: one fitted object), `keep`.
    This estimator: `bounds` (the kept columns' min / max pairs),
    `indicator`, `n_missing`, `n_iter`, `rng`, `post_step`; the finished
    steps' feature `seq_j` and coef / intercept offsets in arena `rounds`;
    their posterior (`post_has_nbl`, `post_nbl`, sigma / alpha / means
    offsets in `rounds`, sigma side `post_side`). The result: (n, `cols`)
    float32 words at `out_offset` of arena `out_handle` (an output-region
    offset is the region's, read by PACK). `empty`: the kept POSITIONS
    (0 .. dk-1) of the kept columns with no observed value
    (keep_empty_features), which scikit-learn marks not missing (PACK keeps
    them as `_empty_kept` for transform)."""
    var statistics: List[Float32]
    var fill: List[Float32]
    var shared: Bool
    var keep: List[Int]
    var empty: List[Int]
    var bounds: List[Float64]
    var indicator: List[Int]
    var n_missing: Int
    var n_iter: Int
    var rng: UInt64
    var post_step: Int
    var rounds: Int
    var seq_j: List[Int]
    var seq_coef: List[Int]
    var seq_inter: List[Int]
    var post_has_nbl: List[Bool]
    var post_nbl: List[List[Int]]
    var post_sig: List[Int]
    var post_al: List[Int]
    var post_means: List[Int]
    var post_side: List[Int]
    var out_handle: Int
    var out_offset: Int
    var cols: Int

    def __init__(out self):
        self.statistics = List[Float32]()
        self.fill = List[Float32]()
        self.shared = False
        self.keep = List[Int]()
        self.empty = List[Int]()
        self.bounds = List[Float64]()
        self.indicator = List[Int]()
        self.n_missing = 0
        self.n_iter = 0
        self.rng = 0
        self.post_step = 0
        self.rounds = II_NONE
        self.seq_j = List[Int]()
        self.seq_coef = List[Int]()
        self.seq_inter = List[Int]()
        self.post_has_nbl = List[Bool]()
        self.post_nbl = List[List[Int]]()
        self.post_sig = List[Int]()
        self.post_al = List[Int]()
        self.post_means = List[Int]()
        self.post_side = List[Int]()
        self.out_handle = II_NONE
        self.out_offset = 0
        self.cols = 0


@fieldwise_init
struct IterImpStep(Copyable, Movable):
    """One fitted imputation step, as the door read it: feature `j`, the
    addresses of its coef (dk words) and intercept (1 word); with
    sample_posterior its neighbour list (`has_nbl`), and the addresses and
    word counts of its sigma, alpha and means vectors."""
    var j: Int
    var coef: Int
    var inter: Int
    var has_nbl: Bool
    var nbl: List[Int]
    var sig: Int
    var sig_words: Int
    var al: Int
    var al_words: Int
    var means: Int
    var means_words: Int


struct IterImpApply(Movable):
    """IterativeImputer.transform of the (n, d) block of owner 0. `init`: the
    fitted initial imputer's transform (its fill vector is owner 1, at
    `fill_address`). `mark`, `missing`: this estimator's missing_values.
    `keep`, `bounds`, `indicator`: the fitted fields. `seed`: the fitted
    rng's low 31 bits; `post_step`: the fitted posterior step count.
    `empty`: the fitted `_empty_kept` positions (marked not missing; set by
    the caller after construction, none by default)."""
    var init: ImputerApply
    var fill_address: Int
    var n: Int
    var d: Int
    var mark: Bool
    var missing: Float64
    var work_on: Bool
    var region: Bool
    var keep: List[Int]
    var bounds: List[Float64]
    var indicator: List[Int]
    var posterior: Bool
    var seed: Int
    var post_step: Int
    var steps: List[IterImpStep]
    var empty: List[Int]

    def __init__(out self, var init: ImputerApply, fill_address: Int, n: Int, d: Int, mark: Bool, missing: Float64,
                 work_on: Bool, region: Bool, var keep: List[Int], var bounds: List[Float64],
                 var indicator: List[Int], posterior: Bool, seed: Int, post_step: Int,
                 var steps: List[IterImpStep]):
        self.init = init^
        self.fill_address = fill_address
        self.n = n
        self.d = d
        self.mark = mark
        self.missing = missing
        self.work_on = work_on
        self.region = region
        self.keep = keep^
        self.bounds = bounds^
        self.indicator = indicator^
        self.posterior = posterior
        self.seed = seed
        self.post_step = post_step
        self.steps = steps^
        self.empty = List[Int]()


@fieldwise_init
struct IterImpApplied(Movable):
    """The transform's result: (n, `cols`) float32 words at `out_offset` of
    arena `out_handle`, and the posterior step count to store."""
    var out_handle: Int
    var out_offset: Int
    var cols: Int
    var post_step: Int


# ---------------------------------------------------------------- small helpers
def _one(value: Float64) -> List[Float64]:
    var out = List[Float64]()
    out.append(value)
    return out^


def _ones(nbl: List[Int], dk: Int) -> List[Float64]:
    # `[1 if a in nbl else 0 for a in range(dk)]`
    var out = List[Float64]()
    for a in range(dk):  # driver-loop: over the kept features (one step's neighbour mask)
        var hit = 0.0
        for i in range(len(nbl)):  # driver-loop: over one step's drawn features
            if nbl[i] == a:
                hit = 1.0
        out.append(hit)
    return out^


def _copy(address: Int, offset: Int, count: Int) -> List[Float32]:
    # A copy of `count` arena words at `offset` (the door's `_take1`).
    var out = List[Float32](length=max(count, 0), fill=0.0)
    if count > 0:
        unsafe_memcpy(dest=_F32P(unsafe_from_address=Int(out.unsafe_ptr())),
                      src=_F32P(unsafe_from_address=address) + offset, count=count)
    return out^


def _word(address: Int, offset: Int) -> Float32:
    return _F32P(unsafe_from_address=address)[offset]


def _iword(address: Int, offset: Int) -> Int:
    return Int(_I32P(unsafe_from_address=address)[offset])


def _seed_put(mut prog: EncProgram, value: UInt64) -> Int:
    # `_seed_words(pr, seed)`: the low and high words as signed int32.
    return prog.put_ints(enc_seed_words(Int(value & 0xFFFFFFFF), Int(value >> 32)))


def _x(mut prog: EncProgram, n: Int, d: Int, mark: Bool, missing: Float64) -> Int:
    # `_mark_missing(p, p.put_x(values), n * d, missing)`: X is owner 0.
    return enc_mark_missing(prog, prog.input(n * d, 0), n * d, mark, missing)


def iterimp_unmark_empty(mut prog: EncProgram, mo: Int, n: Int, dk: Int, empty: List[Int]):
    """scikit-learn's `_initial_imputation` with keep_empty_features:
    `mask_missing_values[:, empty] = False`. Stages appended after the
    program's NAN_MASK of the (n, dk) mask at `mo`: per kept position c in
    `empty`, its missing rows (`enc_ii_rows`) and one scatter of zero words
    into them (ii_scatter, no clip). Every row of such a column is missing at
    fit; at transform the rows past the count are the zero words of the row
    list, row 0, whose mask word is written 0 again. No host loop over rows."""
    if len(empty) == 0 or n <= 0 or dk <= 0:
        return
    var zeros = prog.alloc(n)  # arena words start zero (the n_nearest_features draws' mask reads them so)
    for e in range(len(empty)):  # driver-loop: over the kept columns with no observed value at fit
        var c = empty[e]
        var rows = enc_ii_rows(prog, mo, n, dk, c, True)
        _stage(prog.program, OP_II_SCATTER, n, mo, dk, rows[0], c, zeros, zeros, 0)


def _run_split(rows: Int, work: Int) -> Tuple[Int, Int]:
    # (tasks, rows per task) of a host split over `rows` with `work` scalar operations in all.
    var units = max(1, host_predict_task_count_work(rows, work))
    return (units, host_predict_chunk(rows, units))


def _unique_run(p: _F32P, n: Int) -> Int:
    """Deduplicate an ascending run in place; the number kept."""
    if n <= 0:
        return 0
    var kept = 1
    for i in range(1, n):  # small-loop(one task's sorted rows, deduplicated in place)
        if p[i] != p[kept - 1]:
            p[kept] = p[i]
            kept += 1
    return kept


def iterimp_class_values(y: Int, m: Int) raises -> List[Float32]:
    """A classifier step's classes: scikit-learn's classifier fit on the
    float32 target `y` (m words): a NaN or infinite value is refused
    (`_assert_all_finite`), a non-integral one is a continuous target
    (`check_classification_targets`, II_CONTINUOUS_WORDS), else the classes
    are `np.unique(y)` ascending. Parallel: each task checks, sorts and
    deduplicates its rows; the runs (one word per distinct value and task)
    are merged once."""
    if m <= 0:
        return List[Float32]()
    var split = _run_split(m, m * 16)
    var units = split[0]
    var chunk = split[1]
    var work = List[Float32](unsafe_uninit_length=m)
    var kept = List[Int](length=units, fill=0)
    var flags = List[Int](length=units, fill=0)
    var src = _F32P(unsafe_from_address=y)
    var wp = _F32P(unsafe_from_address=Int(work.unsafe_ptr()))
    var kp = _IntP(unsafe_from_address=Int(kept.unsafe_ptr()))
    var fp = _IntP(unsafe_from_address=Int(flags.unsafe_ptr()))

    def _part(unit: Int) {imm m, imm chunk, imm src, imm wp, imm kp, imm fp}:
        var lo = min(unit * chunk, m)
        var hi = min(lo + chunk, m)
        var flag = 0
        for i in range(lo, hi):  # cpu-route: one task's rows of a classifier step's target (the label check)
            var v = src[i]
            if v != v:
                flag = flag | 1
            elif not isfinite(v):
                flag = flag | 2
            elif floor(v) != v:
                flag = flag | 4
            wp[i] = v
        fp[unit] = flag
        if hi > lo and (flag & 1) == 0:
            var part = Span(unsafe_ptr=wp + lo, length=hi - lo)
            sort(part)
            kp[unit] = _unique_run(wp + lo, hi - lo)

    host_parallelize(_part, units)
    var seen = 0
    for u in range(units):  # driver-loop: over the label tasks (one flag word each)
        seen = seen | flags[u]
    if (seen & 1) != 0:
        raise Error("Input y contains NaN.")
    if (seen & 2) != 0:
        raise Error("Input y contains infinity or a value too large for dtype('float32').")
    if (seen & 4) != 0:
        raise Error(II_CONTINUOUS_WORDS)
    var merged = List[Float32]()
    for u in range(units):  # driver-loop: over the label tasks (their distinct values)
        var lo = min(u * chunk, m)
        for i in range(kept[u]):  # driver-loop: over one task's distinct values (classes, not rows)
            merged.append(work[lo + i])
    _ = len(work)  # the tasks' runs stay alive past the merge
    sort(merged)
    var count = _unique_run(_F32P(unsafe_from_address=Int(merged.unsafe_ptr())), len(merged))
    var classes = List[Float32](length=count, fill=0.0)
    for c in range(count):  # driver-loop: over the step's classes
        classes[c] = merged[c]
    return classes^


def iterimp_class_codes(y: Int, m: Int, classes: List[Float32]) -> List[Int32]:
    """The int32 codes of the float32 target `y` (m words) over the
    ascending `classes` it was read for (`np.unique`'s inverse): a binary
    search per row, rows split across the host cores."""
    var codes = List[Int32](length=max(m, 1), fill=0)
    var k = len(classes)
    if m <= 0 or k == 0:
        return codes^
    var split = _run_split(m, m * 8)
    var units = split[0]
    var chunk = split[1]
    var src = _F32P(unsafe_from_address=y)
    var cp = _F32P(unsafe_from_address=Int(classes.unsafe_ptr()))
    var dp = _I32P(unsafe_from_address=Int(codes.unsafe_ptr()))

    def _part(unit: Int) {imm m, imm chunk, imm k, imm src, imm cp, imm dp}:
        var lo = min(unit * chunk, m)
        var hi = min(lo + chunk, m)
        for i in range(lo, hi):  # cpu-route: one task's rows of a classifier step's target (its codes)
            var v = src[i]
            var a = 0
            var b = k - 1
            while a < b:
                var mid = (a + b) // 2
                if cp[mid] < v:
                    a = mid + 1
                else:
                    b = mid
            dp[i] = Int32(a)

    host_parallelize(_part, units)
    _ = len(classes)  # the classes the tasks read stay alive past them
    return codes^


def _class_of_codes(raw: Int, dtype: Int, m: Int, classes: List[Float32], dst: Int) raises:
    """A classifier member's predicted codes (m words of `dtype` at `raw`)
    written at `dst` as their class values (float32), rows split across the
    host cores; a code outside 0 .. classes-1 is refused."""
    var k = len(classes)
    if m <= 0:
        return
    var split = _run_split(m, m)
    var units = split[0]
    var chunk = split[1]
    var bad = List[Int](length=units, fill=0)
    var cp = _F32P(unsafe_from_address=Int(classes.unsafe_ptr()))
    var dp = _F32P(unsafe_from_address=dst)
    var bp = _IntP(unsafe_from_address=Int(bad.unsafe_ptr()))

    def _part(unit: Int) {imm m, imm chunk, imm k, imm raw, imm dtype, imm cp, imm dp, imm bp}:
        var lo = min(unit * chunk, m)
        var hi = min(lo + chunk, m)
        for i in range(lo, hi):  # cpu-route: one task's predicted codes mapped to the step's class values
            var code = 0
            if dtype == MEMBER_I64:
                code = Int(_I64P(unsafe_from_address=raw)[i])
            elif dtype == MEMBER_F32:
                code = Int(_F32P(unsafe_from_address=raw)[i])
            elif dtype == MEMBER_F64:
                code = Int(_F64P(unsafe_from_address=raw)[i])
            else:
                code = Int(_I32P(unsafe_from_address=raw)[i])
            if code < 0 or code >= k:
                bp[unit] = 1
                dp[i] = Float32(0.0)
            else:
                dp[i] = cp[code]

    host_parallelize(_part, units)
    _ = len(classes)  # the classes the tasks read stay alive past them
    for u in range(units):  # driver-loop: over the code tasks (one flag word each)
        if bad[u] != 0:
            raise Error("RuntimeError: mojolearn: IterativeImputer: the classifier member predicted a class code "
                        + "outside its classes")


def iterimp_bounds(lo: List[Float64], hi: List[Float64], keep: List[Int]) raises -> List[Float64]:
    """`_bounds(d)` interleaved (zip stops at the shorter), then the kept
    columns' (lower, upper) pairs."""
    var bounds = List[Float64]()
    for i in range(min(len(lo), len(hi))):  # driver-loop: over the per-column bound pairs
        bounds.append(lo[i])
        bounds.append(hi[i])
    var out = List[Float64]()
    for c in range(len(keep)):  # driver-loop: over the kept columns
        for h in range(2):  # small-loop(h: the lower and upper bound)
            var at = 2 * keep[c] + h
            if at >= len(bounds):
                raise Error("IndexError: list index out of range")
            out.append(bounds[at])
    return out^


struct IterImpDraws(Movable):
    """`_NeighbourDraws`: every (round, feature) step's n_nearest_features
    draw, handed out in fit order; each hand-out moves the stream past the
    words the steps taken so far own (k per step)."""
    var lists: List[List[Int]]
    var at: Int
    var k: Int
    var start: UInt64
    var on: Bool

    def __init__(out self):
        self.lists = List[List[Int]]()
        self.at = 0
        self.k = 0
        self.start = 0
        self.on = False

    def next(mut self, mut rng: UInt64) -> List[Int]:
        var picks = self.lists[self.at].copy()
        self.at += 1
        rng = self.start + UInt64(self.at) * UInt64(self.k) * II_GOLDEN
        return picks^


def _neighbours[R: IterImpRunner](mut runner: R, xf: Int, n: Int, dk: Int, js: List[Int], k: Int,
                                  rng: UInt64, work_on: Bool) raises -> List[List[Int]]:
    """`_neighbours_device`: the |corr| matrix on the device and one thread per
    step drawing its k features (f2_wpick, step c takes splitmix64 words
    c*k .. c*k + k - 1 of the stream). Returns each step's ascending picks."""
    var p = EncProgram(work_on)
    var fo = enc_input(p, n * dk, 1)
    var mz = p.alloc(n * dk)
    var means = p.alloc(dk)
    var cnt = p.alloc(1)
    var g = p.alloc(dk * dk)
    var flag = p.alloc(1)
    _stage(p.program, OP_II_MEAN, dk, fo, n, dk, mz, 0, means, cnt, flag)
    _stage(p.program, OP_II_GRAM, dk * dk, fo, n, dk, mz, 0, means, g, flag)
    var m = p.alloc(dk * dk)
    _stage(p.program, OP_ABSCORR_CELL, dk * dk, g, dk, m)
    _stage(p.program, OP_ABSCORR_NORM, dk, m, dk)
    var calls = len(js)
    var fl = p.work(calls * dk)
    var picked = p.alloc(calls * (k + 1))
    var jo = p.put_ints(js)
    var so = _seed_put(p, rng)
    _stage(p.program, OP_F2_WPICK, calls, m, dk, jo, k, so, fl, picked)
    var seconds: List[Int] = [xf]  # the initial fill, in its held arena
    var h = runner.run(p, seconds)
    _ = len(p.consts_i)  # the program's own words too
    var a = runner.words(h)
    var lists = List[List[Int]]()
    for c in range(calls):  # driver-loop: one feature list per step
        var got = _iword(a, picked + c * (k + 1) + k)
        if got < k:
            runner.release(h)
            raise Error("mojolearn: IterativeImputer: no feature left to draw")
        var picks = List[Int]()
        for i in range(got):  # driver-loop: over one step's drawn features
            picks.append(_iword(a, picked + c * (k + 1) + i))
        lists.append(picks^)
    runner.release(h)
    return lists^


def iterimp_indicator[R: IterImpRunner](mut runner: R, n: Int, d: Int, dk: Int, mark: Bool, missing: Float64,
                                     indicator: List[Int], xt_address: Int, work_on: Bool,
                                     region: Bool) raises -> Tuple[Int, Int]:
    """`_with_indicator`: Xt (the (n, dk) words at `xt_address`), then
    MissingIndicator's columns. Returns (handle, offset) of the (n, dk + m)
    result."""
    var m = len(indicator)
    var p = EncProgram(work_on)
    var xo = _x(p, n, d, mark, missing)
    var mo = p.alloc(n * m)
    var io = p.put_floats(enc_floats_of(indicator))
    _stage(p.program, OP_NAN_MASK, n * m, xo, n, d, io, m, mo)
    var to = enc_input(p, n * dk, 1)
    var hc = p.output(n * (dk + m), region)
    _stage(p.program, OP_HCAT, n * (dk + m), to, dk, mo, m, hc)
    var seconds: List[Int] = [xt_address]
    var h = runner.run(p, seconds)
    _ = len(p.consts_f)  # the program's own words stay alive past the run
    return (h, hc)


# ---------------------------------------------------------------- fit
struct IterImpPlan(Movable):
    """What programs 1 to 4 leave for the rounds (the built-in ridge's here,
    a supplied estimator's in the door): the rounds' feature orders, the
    n_nearest_features draws, the rng word, the scaled tol, and the initial
    fill's (n, dk) block at `xf_offset` of the held arena `xf_handle`
    (`xf_address` its first word)."""
    var orders: List[List[Int]]
    var any_order: Bool
    var draws: IterImpDraws
    var rng: UInt64
    var tol: Float64
    var xf_handle: Int
    var xf_offset: Int
    var xf_address: Int

    def __init__(out self):
        self.orders = List[List[Int]]()
        self.any_order = False
        self.draws = IterImpDraws()
        self.rng = 0
        self.tol = 0.0
        self.xf_handle = II_NONE
        self.xf_offset = 0
        self.xf_address = 0


def iterimp_fit[R: IterImpRunner](mut runner: R, r: IterImpFit) raises -> IterImpFitted:
    """IterativeImputer.fit_transform with the built-in Bayesian ridge. The
    result keeps two arenas held: `rounds` (the fitted steps) and
    `out_handle` (the result; the same arena without an indicator)."""
    var fitted = IterImpFitted()
    var plan = iterimp_prepare(runner, r, fitted)
    if len(fitted.keep) < 2:
        iterimp_initial_only(runner, r, plan, fitted)
        return fitted^
    iterimp_rounds(runner, r, plan, fitted)
    return fitted^


def iterimp_initial_only[R: IterImpRunner](mut runner: R, r: IterImpFit, mut plan: IterImpPlan,
                                           mut fitted: IterImpFitted) raises:
    """scikit-learn's fit with fewer than two kept features ("Edge case: a
    single feature, we return the initial imputation"; no kept feature is
    the same return through `np.all` of an empty mask): `n_iter_ = 0`, no
    imputation step, Xt the initial fill (then the indicator). No estimator
    is fitted or called, the built-in one or a supplied one. After
    `iterimp_prepare`; `rounds` names the result's arena (PACK reads no step
    from it)."""
    var dk = len(fitted.keep)
    fitted.n_iter = 0
    fitted.rng = plan.rng
    fitted.post_step = 0
    if len(fitted.indicator) == 0:
        fitted.out_handle = plan.xf_handle
        fitted.out_offset = plan.xf_offset
        fitted.cols = dk
    else:
        var made = iterimp_indicator(runner, r.n, r.d, dk, r.mark, r.missing, fitted.indicator, plan.xf_address,
                                     r.work_on, r.region)
        runner.release(plan.xf_handle)  # the initial fill, copied into the indicator program's arena
        fitted.out_handle = made[0]
        fitted.out_offset = made[1]
        fitted.cols = dk + len(fitted.indicator)
    plan.xf_handle = II_NONE
    fitted.rounds = fitted.out_handle


def iterimp_prepare[R: IterImpRunner](mut runner: R, r: IterImpFit, mut fitted: IterImpFitted) raises -> IterImpPlan:
    """Programs 1 to 4: the initial imputer (its fitted values, the kept
    columns), the bounds, the indicator columns, the orders and the draws
    (into `fitted` and the plan). The initial fill's arena stays held."""
    var n = r.n
    var d = r.d
    var made = IterImpPlan()
    # 1: the initial SimpleImputer's fit
    var p1 = EncProgram(r.work_on)
    var plan = imputer_fit_stage(p1, r.init)
    var none = List[Int]()
    var h1 = runner.run(p1, none)
    _ = len(p1.consts_f)  # the program's own words stay alive past the run
    var a1 = runner.words(h1)
    var any_empty = False
    for j in range(d):  # driver-loop: over the per-column non-missing counts (feature metadata)
        any_empty = any_empty or Int(_word(a1, plan.stats + j)) == 0
    fitted.shared = not (r.init.strategy == IMP_CONSTANT or any_empty)
    if fitted.shared:
        fitted.statistics = _copy(a1, plan.source, d)
        fitted.fill = fitted.statistics.copy()
    else:
        fitted.statistics = _copy(a1, plan.statistics, d)
        fitted.fill = _copy(a1, plan.fill, d)
    for j in range(d):  # driver-loop: over the kept columns (feature metadata)
        var observed = Int(_word(a1, plan.stats + j))
        if r.init.keep_empty or observed != 0:
            if observed == 0:
                fitted.empty.append(len(fitted.keep))  # its kept position: marked not missing (sklearn)
            fitted.keep.append(j)
    runner.release(h1)
    var keep = fitted.keep.copy()
    var dk = len(keep)
    # 2: its transform (no indicator: the initial imputer has none)
    var p2 = EncProgram(r.work_on)
    var apply = ImputerApply(n, d, 0, r.init.mark, r.init.missing, 1, d, keep.copy(), List[Int]())
    var o2 = imputer_transform_stage(p2, apply, r.region)
    var fill_seconds: List[Int] = [Int(fitted.fill.unsafe_ptr())]
    var h2 = runner.run(p2, fill_seconds)
    _ = len(fitted.fill)  # the fill vector the run read stays alive past it
    _ = len(p2.consts_f)
    # the filled block stays in its arena (read by programs 4 and 5 there)
    made.xf_handle = h2
    made.xf_offset = o2
    made.xf_address = runner.words(h2) + 4 * o2
    fitted.bounds = iterimp_bounds(r.lo, r.hi, keep)
    var rng = r.rng
    # 3: the statistics, the scale and the imputation orders: one program
    var p = EncProgram(r.work_on)
    var xo = _x(p, n, d, r.mark, r.missing)
    var st = p.alloc(6 * d)
    enc_col_stats(p, r.stats_blocked, xo, n, d, st, True)
    var rounds = r.max_iter
    var R_ = max(rounds, 1)
    var ordw = p.alloc(max(R_ * dk, 1))
    var lens = p.alloc(R_)
    var sc = p.alloc(1)
    _stage(p.program, OP_C2_COLMAX, 1, st + 5 * d, d, sc)
    if dk > 0:
        var missd = p.work(dk)
        var ko = p.put_ints(keep)
        _stage(p.program, OP_C2_II_MISS, dk, st, ko, n, missd)
        if len(fitted.empty) > 0:
            # sklearn marks an empty kept column not missing: its missing count
            # is 0 (int32 0 is the float32 word 0.0 the scatter stores)
            var eo = p.put_ints(fitted.empty)
            var zo = p.alloc(len(fitted.empty))
            _stage(p.program, OP_II_SCATTER, len(fitted.empty), missd, 1, eo, 0, zo, zo, 0)
        if r.order == II_RANDOM:
            var so = _seed_put(p, rng)
            _stage(p.program, OP_C2_II_RAND, R_, missd, dk, 1 if r.skip else 0, so, ordw, lens)
        else:
            var pos = p.work(dk)
            _stage(p.program, OP_C2_II_POS, dk, missd, dk, r.order, pos)
            _stage(p.program, OP_C2_II_ORD, dk, missd, dk, pos, R_, ordw, lens, 1 if r.skip else 0)
    var h3 = runner.run(p, none)
    _ = len(p.consts_i)  # the program's own words stay alive past the run
    var a3 = runner.words(h3)
    var scale = Float64(_word(a3, sc))
    if not isfinite(scale) and scale == scale and scale > 0.0:
        runner.release(h3)
        raise Error("Input X contains infinity or a value too large for dtype('float64').")
    for e in range(len(fitted.empty)):  # driver-loop: over the kept columns with no observed value at fit
        # sklearn's `np.max(np.abs(X[~mask_missing_values]))` reads such a
        # column's fill as observed values (`Xt[:, empty] = X_filled[:, empty]`)
        scale = max(scale, abs(Float64(fitted.fill[keep[fitted.empty[e]]])))
    if r.add_indicator:
        for j in range(d):  # driver-loop: over the per-column non-missing counts (feature metadata)
            if Int(_word(a3, st + j)) < n:
                fitted.indicator.append(j)
    var ln = List[Int]()
    for i in range(R_):  # driver-loop: over the imputation rounds
        ln.append(_iword(a3, lens + i) if dk > 0 else 0)
    for i in range(rounds):  # driver-loop: the device's orders as control lists (imputation rounds)
        var order = List[Int]()
        for c in range(ln[i]):  # driver-loop: over one round's feature order
            order.append(_iword(a3, ordw + i * dk + c))
        made.any_order = made.any_order or len(order) > 0
        made.orders.append(order^)
    runner.release(h3)
    fitted.n_missing = ln[0]
    if r.order == II_RANDOM:
        var m = ln[0] if r.skip else dk
        rng = rng + UInt64(rounds) * UInt64(max(m - 1, 0)) * II_GOLDEN
    # 4: the n_nearest_features draws, every step up front in one program
    if r.nnf != II_NONE and r.nnf < dk:
        made.draws.on = True
        made.draws.k = r.nnf
        made.draws.start = rng
        var js = List[Int]()
        for i in range(len(made.orders)):  # driver-loop: the (round, feature) step list
            for c in range(len(made.orders[i])):  # driver-loop: over one round's feature order
                js.append(made.orders[i][c])
        if len(js) > 0:
            made.draws.lists = _neighbours(runner, made.xf_address, n, dk, js, r.nnf, rng, r.work_on)
    made.rng = rng
    made.tol = r.tol * scale
    return made^


def iterimp_rounds[R: IterImpRunner](mut runner: R, r: IterImpFit, mut plan: IterImpPlan,
                                     mut fitted: IterImpFitted) raises:
    """Programs 5 and 6 of the built-in Bayesian ridge: every round in one
    program over the initial fill (read in its held arena, released after),
    then the indicator."""
    var n = r.n
    var d = r.d
    var keep = fitted.keep.copy()
    var dk = len(keep)
    var rounds = r.max_iter
    var rng = plan.rng
    var tol = plan.tol
    var any_order = plan.any_order
    # 5: the built-in Bayesian ridge: every round in one program
    var q = EncProgram(r.work_on)
    var qx = _x(q, n, d, r.mark, r.missing)
    var fo = enc_input(q, n * dk, 1)
    q.inout(fo, n * dk)
    var ko = q.put_floats(enc_floats_of(keep))
    var mo = q.alloc(n * dk)
    _stage(q.program, OP_NAN_MASK, n * dk, qx, n, d, ko, dk, mo)
    iterimp_unmark_empty(q, mo, n, dk, fitted.empty)
    var bo = q.put_floats(fitted.bounds)
    var to = q.put_floats(_one(tol))
    var flag = q.alloc(1)
    var niter = q.alloc(1)
    var prev = q.alloc(n * dk)
    var cnt = q.alloc(1)
    var g = q.alloc(dk * dk)
    var p1w = max(dk - 1, 1)
    var gs = q.alloc(p1w * p1w)
    var eig = q.alloc(p1w)
    var vec = q.alloc(p1w * p1w)
    var w = q.alloc(p1w)
    var seq_j = List[Int]()
    var seq_coef = List[Int]()
    var seq_inter = List[Int]()
    var post_has = List[Bool]()
    var post_nbl = List[List[Int]]()
    var post_sig = List[Int]()
    var post_al = List[Int]()
    var post_means = List[Int]()
    var post_pp = List[Int]()
    var post_step = 0
    var seed = Int(rng & 0x7FFFFFFF)
    var conv = not r.posterior
    var rowabs = -1
    for i in range(rounds):  # driver-loop: drives the device imputation rounds (imputation rounds)
        if len(plan.orders[i]) > 0 and conv:
            _stage(q.program, OP_II_SNAPSHOT, n * dk, fo, prev, flag)
        for c in range(len(plan.orders[i])):  # driver-loop: drives one device regression per feature (per-round feature order)
            var j = plan.orders[i][c]
            var coef = q.alloc(dk)
            var inter = q.alloc(1)
            var means = q.alloc(dk)
            var nb1 = 0
            var pp = dk - 1
            var nbl = List[Int]()
            if plan.draws.on:
                nbl = plan.draws.next(rng)
                nb1 = q.put_floats(_ones(nbl, dk)) + 1
                pp = len(nbl)
            var al = q.alloc(2) if r.posterior else -1
            seq_j.append(j)
            seq_coef.append(coef)
            seq_inter.append(inter)
            _stage(q.program, OP_II_MEAN, dk, fo, n, dk, mo, j, means, cnt, flag)
            _stage(q.program, OP_II_GRAM, dk * dk, fo, n, dk, mo, j, means, g, flag)
            if pp > 0:
                _stage(q.program, OP_II_SUB, 1, g, dk, j, gs, flag, nb1)
                _stage(q.program, OP_EIGH, 1, gs, pp, 0, eig, vec, 1)  # q[5]=1: cyclic eigh_unit, not round-robin
            _stage(q.program, OP_II_BR, 1, g, dk, j, eig, vec, means, cnt, coef, inter, flag, w, nb1, al + 1)
            if r.posterior:
                var sig = q.alloc(max(pp, 1) * max(pp, 1))
                _stage(q.program, OP_II_SIGMA, pp * pp, eig, vec, pp, al, sig, flag)
                var key_words: List[Int] = [seed, post_step]
                var key = q.put_ints(key_words)
                post_step += 1
                _stage(q.program, OP_II_POST, n, fo, n, dk, mo, j, coef, inter, bo, flag, means, sig, al, nb1, key)
                post_has.append(plan.draws.on)
                post_nbl.append(nbl^)
                post_sig.append(sig)
                post_al.append(al)
                post_means.append(means)
                post_pp.append(pp)
            else:
                _stage(q.program, OP_II_PREDICT, n, fo, n, dk, mo, j, coef, inter, bo, flag)
        if len(plan.orders[i]) > 0 and conv:
            if rowabs < 0:
                rowabs = q.alloc(n)
            _stage(q.program, OP_II_ROWABS, n, fo, prev, dk, rowabs, flag)
            _stage(q.program, OP_II_CONV, 1, fo, prev, n * dk, to, flag, niter, dk, rowabs + 1)
    var xf_seconds: List[Int] = [plan.xf_address]
    var h5 = runner.run(q, xf_seconds)
    runner.release(plan.xf_handle)  # the initial fill, copied up by the run
    plan.xf_handle = II_NONE
    _ = len(q.consts_f)
    _ = len(q.consts_i)
    var a5 = runner.words(h5)
    var done = 0
    if any_order:
        done = Int(_word(a5, niter)) if conv else rounds
    fitted.n_iter = done if any_order else min(1, rounds)
    var steps = 0
    for i in range(min(done, len(plan.orders))):  # driver-loop: count of finished imputation steps (imputation rounds)
        steps += len(plan.orders[i])
    for s in range(min(steps, len(seq_j))):  # driver-loop: fitted imputation sequence offsets (finished imputation steps)
        fitted.seq_j.append(seq_j[s])
        fitted.seq_coef.append(seq_coef[s])
        fitted.seq_inter.append(seq_inter[s])
    for s in range(min(steps, len(post_sig))):  # driver-loop: fitted posterior offsets (finished imputation steps)
        fitted.post_has_nbl.append(post_has[s])
        fitted.post_nbl.append(post_nbl[s].copy())
        fitted.post_sig.append(post_sig[s])
        fitted.post_al.append(post_al[s])
        fitted.post_means.append(post_means[s])
        fitted.post_side.append(max(post_pp[s], 1))
    fitted.rng = rng
    fitted.post_step = post_step
    fitted.rounds = h5
    # 6: the missing indicator, reading Xt where the rounds left it
    if len(fitted.indicator) == 0:
        fitted.out_handle = h5
        fitted.out_offset = fo
        fitted.cols = dk
        return
    var made = iterimp_indicator(runner, n, d, dk, r.mark, r.missing, fitted.indicator, a5 + 4 * fo, r.work_on, r.region)
    fitted.out_handle = made[0]
    fitted.out_offset = made[1]
    fitted.cols = dk + len(fitted.indicator)


# ---------------------------------------------------------------- a supplied estimator's member
struct IterImpMember(Copyable, Movable):
    """A supplied estimator as a native member, admitted once per fit by the
    door's UNPACK: its `plan`; `borrows` its kind keeps pointers into its
    fit buffers (MEMBER_BORROWS_FIT: each state is told to own them right
    after its fit, since the step's gather arena is released at the step's
    end); `posterior` sample_posterior: the predict block is answered by
    MEMBER_OP_PREDICT_STD with the same query's mean written to `call.y`
    (the GaussianProcessRegressor and BayesianRidge / ARDRegression members;
    the door admits no other family for it). `classifier` (set by the door
    after construction): a classifier kind, fitted on each step's int32
    codes with the step's class count; its PREDICT writes codes of
    `code_dtype` (the member's MEMBER_OP_OUTPUTS answer, asked once).
    `cw_refusal` (set by the door, lane mf-left4): the estimator's own words
    for a class_weight it refuses, read at UNPACK; a classifier step raises
    them before its fit, where the estimator's own fit raised them ("": none).
    `marker_word` (set by the door, lane mf-iterimp7): for a kind that reads
    -1 in y as unlabeled (LabelPropagation / LabelSpreading), the plan word
    each classifier step writes its -1 class's code into
    (`iterimp_marker_code`), as model selection's units write it after their
    encode; -1: a kind without one."""
    var plan: MemberPlan
    var borrows: Bool
    var posterior: Bool
    var classifier: Bool
    var code_dtype: Int
    var cw_refusal: String
    var marker_word: Int

    def __init__(out self, var plan: MemberPlan, borrows: Bool, posterior: Bool):
        self.plan = plan^
        self.borrows = borrows
        self.posterior = posterior
        self.classifier = False
        self.code_dtype = MEMBER_I32
        self.cw_refusal = String("")
        self.marker_word = -1


def iterimp_marker_code(classes: List[Float32]) -> Int:
    """The code of the unlabeled marker among a classifier step's ascending
    class values: the class equal to -1 (the estimator's own test, `c == -1`
    on a number; lane mf-iterimp7), or -1 when the step has none. The step's
    fit and the door's PACK read it from the same classes."""
    for c in range(len(classes)):  # driver-loop: over one step's classes (not rows)
        if classes[c] == Float32(-1.0):
            return c
    return -1


struct IterImpMemberFit(Movable):
    """What `iterimp_member_rounds` leaves for PACK: per finished step its
    member state (`states`, freed by `iterimp_member_free`), feature `seq_j`
    and neighbour list `seq_nbl`, and for a classifier its ascending class
    values (`classes`, empty for a regressor: PACK publishes them as the
    estimator's labels and sets the plan's class count); every step's fit
    notices in step order; `n_iter`, `rng`; the result's (n, `cols`) words
    at `out_offset` of arena `out_handle`."""
    var states: List[Int]
    var seq_j: List[Int]
    var seq_nbl: List[List[Int]]
    var classes: List[List[Float32]]
    var notices: List[Float64]
    var n_iter: Int
    var rng: UInt64
    var out_handle: Int
    var out_offset: Int
    var cols: Int

    def __init__(out self):
        self.states = List[Int]()
        self.seq_j = List[Int]()
        self.seq_nbl = List[List[Int]]()
        self.classes = List[List[Float32]]()
        self.notices = List[Float64]()
        self.n_iter = 0
        self.rng = 0
        self.out_handle = II_NONE
        self.out_offset = 0
        self.cols = 0


def iterimp_member_free(member: IterImpMember, mut fit: IterImpMemberFit):
    """Release every member state of the fit the door's PACK did not hand to
    its estimator (a 0 word: `member_held_adopt` took it); never raises
    (after PACK, or on an error path)."""
    for s in range(len(fit.states)):  # driver-loop: over the fitted imputation steps (one state each)
        if fit.states[s] == 0:
            continue
        try:
            var call = member_call_of(member.plan, fit.states[s])
            member_free(member.plan, call)
        except:
            pass  # member_call_of builds words only; a state that cannot be freed is the store's
    fit.states = List[Int]()


def _others(dk: Int, j: Int) -> List[Int]:
    # `[a for a in range(dk) if a != j]`
    var out = List[Int]()
    for a in range(dk):  # driver-loop: over the kept features
        if a != j:
            out.append(a)
    return out^


def _member_stop[R: IterImpRunner](mut runner: R, work_on: Bool, xt: Int, prev: Int, n: Int, dk: Int,
                                   tol: Float64) raises -> Bool:
    """`_ii_stop`: the round's stop, the inf-norm of Xt - prev below tol
    (Xt and prev are the (n, dk) words at their addresses)."""
    var p = EncProgram(work_on)
    var xo = enc_input(p, n * dk, 1)
    var po = enc_input(p, n * dk, 2)
    var to = p.put_floats(_one(tol))
    var flag = p.alloc(1)
    var niter = p.alloc(1)
    var rowabs = p.alloc(n)
    _stage(p.program, OP_II_ROWABS, n, xo, po, dk, rowabs, flag)
    _stage(p.program, OP_II_CONV, 1, xo, po, n * dk, to, flag, niter, dk, rowabs + 1)
    var seconds: List[Int] = [xt, prev]
    var h = runner.run(p, seconds)
    _ = len(p.consts_f)  # the program's own words stay alive past the run
    var stop = _word(runner.words(h), flag) != 0
    runner.release(h)
    return stop


struct _Gathered(Movable):
    """What `_member_gather` leaves: the arena `handle` (its address `a`),
    the offsets of the missing rows' int32 words (`mis_rows`) and of the
    predict block (`gm`), the missing count `m`; with the fit blocks the
    training block (`go`), its targets (`gy`) and the observed count `mob`."""
    var handle: Int
    var a: Int
    var mis_rows: Int
    var gm: Int
    var m: Int
    var go: Int
    var gy: Int
    var mob: Int

    def __init__(out self):
        self.handle = II_NONE
        self.a = 0
        self.mis_rows = 0
        self.gm = 0
        self.m = 0
        self.go = 0
        self.gy = 0
        self.mob = 0


def _member_gather[R: IterImpRunner](mut runner: R, work_on: Bool, xt: Int, mask: Int, n: Int, dk: Int, j: Int,
                                     nbl: List[Int], with_fit: Bool) raises -> _Gathered:
    """`_ii_take`'s program: feature j's missing rows and the predict block
    Xt[mis][:, nbl] and, `with_fit`, the training blocks Xt[obs][:, nbl] and
    Xt[obs, j] (the same stages in the same order the fit's step and the
    door's transform rounds ran). The arena stays held: the caller releases
    `handle` after the scatter read the rows."""
    var nc = len(nbl)
    var p = EncProgram(work_on)
    var xo = enc_input(p, n * dk, 1)
    var mo = enc_input(p, n * dk, 2)
    var cols = enc_floats_of(nbl) if nc > 0 else _one(0.0)
    var co = p.put_floats(cols)
    var jo = p.put_floats(_one(Float64(j)))
    var mis = enc_ii_rows(p, mo, n, dk, j, True)
    var gm = p.alloc(n * nc)
    if nc > 0:
        _stage(p.program, OP_II_GATHER, n * nc, xo, dk, mis[0], mis[1], co, nc, gm)
    var out = _Gathered()
    if with_fit:
        var obs = enc_ii_rows(p, mo, n, dk, j, False)
        var go = p.alloc(n * nc)
        var gy = p.alloc(n)
        if nc > 0:
            _stage(p.program, OP_II_GATHER, n * nc, xo, dk, obs[0], obs[1], co, nc, go)
        _stage(p.program, OP_II_GATHER, n, xo, dk, obs[0], obs[1], jo, 1, gy)
        out.go = go
        out.gy = gy
        out.mob = obs[1]  # an offset until the run below
    var take_seconds: List[Int] = [xt, mask]
    var h = runner.run(p, take_seconds)
    _ = len(p.consts_f)  # the program's own words stay alive past the run
    var a = runner.words(h)
    out.handle = h
    out.a = a
    out.mis_rows = mis[0]
    out.gm = gm
    out.m = Int(_word(a, mis[1]))
    if with_fit:
        out.mob = Int(_word(a, out.mob))
    return out^


def _member_put[R: IterImpRunner](mut runner: R, work_on: Bool, bounds: List[Float64], plan: MemberPlan,
                                  state: Int, posterior: Bool, classes: List[Float32], code_dtype: Int,
                                  gathered: _Gathered, xt: Int, n: Int, dk: Int,
                                  j: Int, nc: Int, mut rng: UInt64) raises -> Tuple[Int, Int]:
    """`_ii_put`: the member's PREDICT (or PREDICT_STD and the mean, then
    the truncated normal draws) of state `state` on the gathered predict
    block, then the scatter program (the clipped float32 store into
    Xt[rows, j]; a draw is stored unclipped). A classifier (`classes` not
    empty) predicts codes of `code_dtype`, written as their class values
    (`estimator.predict` returns the labels). Returns the new Xt's (handle,
    offset). The gathered arena is released after the scatter read it."""
    var m = gathered.m
    var a = gathered.a
    var lo = bounds[2 * j]
    var hi = bounds[2 * j + 1]
    var values = List[Float32](length=m, fill=0.0)
    var clip = 1
    var ask = member_call_of(plan, state)
    ask.x = MemberMatrix(a + 4 * gathered.gm, m, nc)
    if posterior:
        var std = List[Float32](length=m, fill=0.0)
        ask.dst = MemberMatrix(Int(std.unsafe_ptr()), m, 1)
        ask.y = MemberMatrix(Int(values.unsafe_ptr()), m, 1)  # the same query's mean
        _ = runner.member(plan, MEMBER_OP_PREDICT_STD, ask)
        var mu = List[Float64](length=m, fill=0.0)
        var sd = List[Float64](length=m, fill=0.0)
        for i in range(m):  # cpu-route: one feature's posterior mean and std widened (exact) for the serial draw stream below
            mu[i] = Float64(values[i])
            sd[i] = Float64(std[i])
        _ = len(std)  # the std buffer the member wrote stays alive past its reads
        var drawn = List[Float64](length=m, fill=0.0)
        # Serial by nature: only drawing entries advance the one stream.
        rng = truncnorm_draws(Int(mu.unsafe_ptr()), Int(sd.unsafe_ptr()), m, lo, hi, rng, Int(drawn.unsafe_ptr()))
        _ = len(mu)
        _ = len(sd)
        for i in range(m):  # cpu-route: one feature's posterior draws rounded once to float32 (the door's `_f32_vector`)
            values[i] = Float32(drawn[i])
        clip = 0
    elif len(classes) > 0:
        # the codes in the member's own element type (8 bytes a row holds any of them)
        var raw = List[Int64](length=m, fill=0)
        ask.dst = MemberMatrix(Int(raw.unsafe_ptr()), m, 1, code_dtype)
        _ = runner.member(plan, MEMBER_OP_PREDICT, ask)
        _class_of_codes(Int(raw.unsafe_ptr()), code_dtype, m, classes, Int(values.unsafe_ptr()))
        _ = len(raw)  # the codes the member wrote stay alive past their reads
    else:
        ask.dst = MemberMatrix(Int(values.unsafe_ptr()), m, 1)
        _ = runner.member(plan, MEMBER_OP_PREDICT, ask)
    # `_ii_put`'s program: Xt in-out, the rows' int32 words where the gather
    # program left them, the values, the bounds
    var q = EncProgram(work_on)
    var qx = enc_input(q, n * dk, 1)
    q.inout(qx, n * dk)
    var ro = enc_input(q, m, 2)
    var vo = enc_input(q, m, 3)
    var lohi: List[Float64] = [lo, hi]
    var bo = q.put_floats(lohi)
    _stage(q.program, OP_II_SCATTER, m, qx, dk, ro, j, vo, bo, clip)
    var put_seconds: List[Int] = [xt, a + 4 * gathered.mis_rows, Int(values.unsafe_ptr())]
    var h2 = runner.run(q, put_seconds)
    _ = len(values)  # the values the program read stay alive past it
    _ = len(q.consts_f)
    runner.release(gathered.handle)  # the gather arena (rows, blocks), read by the scatter above
    return (h2, qx)


def _member_step[R: IterImpRunner](mut runner: R, work_on: Bool, bounds: List[Float64], member: IterImpMember,
                                   mut fit: IterImpMemberFit, xt: Int, mask: Int, n: Int, dk: Int, j: Int,
                                   nbl: List[Int], mut rng: UInt64) raises -> Tuple[Int, Int]:
    """One (round, feature) step: the gather program (`_member_gather` with
    the training blocks), a new member state FIT on the training blocks
    where the arena holds them (kept in `fit.states`), then `_member_put`
    (its PREDICT or PREDICT_STD on the predict block and the scatter).
    Returns the new Xt's (handle, offset), or (II_NONE, 0) when feature j
    has no missing row (Xt unchanged, as the door's `_ii_put` returned it).
    A classifier step first reads the target's classes and codes
    (`iterimp_class_values` refuses as the classifier's own fit would) and
    fits on the codes with the step's class count."""
    var nc = len(nbl)
    var gathered = _member_gather(runner, work_on, xt, mask, n, dk, j, nbl, True)
    var a = gathered.a
    var classes = List[Float32]()
    if not member.classifier:
        # FIT: a new state on the gathered training rows (host float32 views)
        var call = member_new(member.plan)
        fit.states.append(call.state)  # freed by iterimp_member_free on every path
        fit.classes.append(List[Float32]())
        call.x = MemberMatrix(a + 4 * gathered.go, gathered.mob, nc)
        call.y = MemberMatrix(a + 4 * gathered.gy, gathered.mob, 1)
        _ = runner.member(member.plan, MEMBER_OP_FIT, call)
        if member.borrows:
            member_own(member.plan, call.state)  # THE LIFETIME RULE: the arena goes at the step's end
        member_notices_take(member.plan, call.state, fit.notices)
        if gathered.m == 0:
            runner.release(gathered.handle)
            return (II_NONE, 0)
        return _member_put(runner, work_on, bounds, member.plan, call.state, member.posterior, classes,
                           member.code_dtype, gathered, xt, n, dk, j, nc, rng)
    # A classifier: the step's classes and codes, then FIT on the codes
    var y = a + 4 * gathered.gy
    classes = iterimp_class_values(y, gathered.mob)
    var codes = iterimp_class_codes(y, gathered.mob, classes)
    if member.cw_refusal.byte_length() > 0:
        # A class_weight the estimator refuses (lane mf-left4): its own words,
        # at this first classifier fit, after the target's class read (the
        # arena is left as a refusing fit leaves it).
        raise Error(member.cw_refusal)
    var step_plan = member.plan.copy()
    step_plan.classes = len(classes)
    if member.marker_word >= 0:
        # LabelPropagation / LabelSpreading: the -1 class is the unlabeled
        # marker (its rows join the graph, not classes_), as the estimator's
        # own fit reads the step's target (lane mf-iterimp7)
        var marker = iterimp_marker_code(classes)
        if marker >= 0:
            step_plan.params[member.marker_word] = Float64(marker)
    # A class_weight dict keyed by the target's values (lane mf-adm3): the
    # table over this step's classes, as the step's own fit reads the dict
    # against its labels ('balanced' reads the codes in the fit itself).
    member_cw_from_values(step_plan, classes)
    # A key no class of this step holds, with a class left unweighted:
    # scikit-learn's words naming the unweighted classes, raised where the
    # step's own fit raises them (before it; lane mf-gate6).
    var cw_absent = member_cw_values_refusal(step_plan, classes)
    if cw_absent.byte_length() > 0:
        raise Error(cw_absent)
    var call = member_new(step_plan)
    fit.states.append(call.state)  # freed by iterimp_member_free on every path
    fit.classes.append(classes.copy())
    call.x = MemberMatrix(a + 4 * gathered.go, gathered.mob, nc)
    call.y = MemberMatrix(Int(codes.unsafe_ptr()), gathered.mob, 1, MEMBER_I32)
    _ = runner.member(step_plan, MEMBER_OP_FIT, call)
    if member.borrows:
        member_own(step_plan, call.state)  # THE LIFETIME RULE: the codes and the arena go at the step's end
    _ = len(codes)  # the codes the fit read stay alive past it (and past the own copy above)
    member_notices_take(step_plan, call.state, fit.notices)
    if gathered.m == 0:
        runner.release(gathered.handle)
        return (II_NONE, 0)
    return _member_put(runner, work_on, bounds, step_plan, call.state, False, classes, member.code_dtype, gathered,
                       xt, n, dk, j, nc, rng)


def iterimp_member_rounds[R: IterImpRunner](mut runner: R, r: IterImpFit, mut plan: IterImpPlan,
                                            fitted: IterImpFitted, member: IterImpMember,
                                            mut fit: IterImpMemberFit) raises:
    """The rounds of a supplied estimator that is a native member, after
    `iterimp_prepare` (programs 1 to 4): the missing mask program, then per
    round each feature's step (`_member_step`: gather, member FIT and
    PREDICT, scatter) and, without sample_posterior, the stop program, then
    the indicator. Xt starts as the initial fill in its held arena; each
    step's scatter leaves the next Xt in a new arena (the one before it is
    released unless it is the round's start, which the stop reads). The
    features of a round are not independent (each step reads the Xt the
    step before it wrote): they run in order. On an error every state made
    so far is released before the error is raised."""
    var n = r.n
    var d = r.d
    var dk = len(fitted.keep)
    var rng = plan.rng
    # `_mask_arr`: the (n, dk) float32 missing mask (1.0 missing)
    var mp = EncProgram(r.work_on)
    var mx = _x(mp, n, d, r.mark, r.missing)
    var mo = mp.alloc(n * dk)
    var ko = mp.put_floats(enc_floats_of(fitted.keep))
    _stage(mp.program, OP_NAN_MASK, n * dk, mx, n, d, ko, dk, mo)
    iterimp_unmark_empty(mp, mo, n, dk, fitted.empty)
    var none = List[Int]()
    var mh = runner.run(mp, none)
    _ = len(mp.consts_f)  # the program's own words stay alive past the run
    var mask = runner.words(mh) + 4 * mo
    var xt_h = plan.xf_handle
    var xt_off = plan.xf_offset
    var xt = plan.xf_address
    plan.xf_handle = II_NONE  # the initial fill's arena is this run's from here
    var done = 0
    try:
        for i in range(len(plan.orders)):  # driver-loop: over the imputation rounds (Gauss-Seidel sweeps; serial by nature)
            var check = not member.posterior and len(plan.orders[i]) > 0
            var prev_h = xt_h
            var prev = xt
            for c in range(len(plan.orders[i])):  # driver-loop: over the round's features in order (each reads the last imputation)
                var j = plan.orders[i][c]
                var nbl = plan.draws.next(rng) if plan.draws.on else _others(dk, j)
                var made = _member_step(runner, r.work_on, fitted.bounds, member, fit, xt, mask, n, dk, j, nbl, rng)
                fit.seq_j.append(j)
                fit.seq_nbl.append(nbl^)
                if made[0] != II_NONE:
                    if xt_h != prev_h:
                        runner.release(xt_h)
                    xt_h = made[0]
                    xt_off = made[1]
                    xt = runner.words(xt_h) + 4 * xt_off
            done = i + 1
            var stop = False
            if check:
                stop = _member_stop(runner, r.work_on, xt, prev, n, dk, plan.tol)
            if prev_h != xt_h:
                runner.release(prev_h)
            if stop:
                break
        runner.release(mh)
        fit.n_iter = done if plan.any_order else min(1, len(plan.orders))
        fit.rng = rng
        if len(fitted.indicator) == 0:
            fit.out_handle = xt_h
            fit.out_offset = xt_off
            fit.cols = dk
            return
        var placed = iterimp_indicator(runner, n, d, dk, r.mark, r.missing, fitted.indicator, xt, r.work_on,
                                       r.region)
        runner.release(xt_h)  # Xt, copied into the indicator program's arena
        fit.out_handle = placed[0]
        fit.out_offset = placed[1]
        fit.cols = dk + len(fitted.indicator)
    except error:
        iterimp_member_free(member, fit)
        raise error^


# ---------------------------------------------------------------- transform with a supplied estimator
struct IterImpHeldStep(Copyable, Movable):
    """One fitted step of a supplied estimator for transform: feature `j`,
    its neighbour list `nbl`, and the HELD member state of the step's fitted
    estimator (`plan`, `state`: bindings/member_held_door.mojo, read once
    per transform by the door's UNPACK; the estimator owns it). A
    classifier step (set by the door after construction): its ascending
    class values `classes` (empty for a regressor) and the element type of
    its predicted codes `code_dtype`."""
    var j: Int
    var nbl: List[Int]
    var plan: MemberPlan
    var state: Int
    var classes: List[Float32]
    var code_dtype: Int

    def __init__(out self, j: Int, var nbl: List[Int], var plan: MemberPlan, state: Int):
        self.j = j
        self.nbl = nbl^
        self.plan = plan^
        self.state = state
        self.classes = List[Float32]()
        self.code_dtype = MEMBER_I32


def iterimp_member_transform[R: IterImpRunner](mut runner: R, r: IterImpApply, steps: List[IterImpHeldStep],
                                               mut rng: UInt64) raises -> IterImpApplied:
    """IterativeImputer.transform with a supplied estimator whose fitted
    steps are held native member states (lane mf-iterimp2): the initial
    transform, the missing mask, then per fitted step in order the gather
    program (the missing rows and the predict block), the member's PREDICT
    (PREDICT_STD and the mean, then the draws, with sample_posterior) and
    the scatter program, the same programs the door's transform rounds ran
    over the public estimators; then the indicator. Serial by nature: each
    step reads the Xt the step before it wrote; each member call runs
    across the cores itself. `rng` advances as the draws use it. The
    result's arena stays held."""
    var n = r.n
    var d = r.d
    var dk = len(r.keep)
    var p0 = EncProgram(r.work_on)
    var o0 = imputer_transform_stage(p0, r.init, r.region)
    var fill_seconds: List[Int] = [r.fill_address]
    var h0 = runner.run(p0, fill_seconds)
    _ = len(p0.consts_f)
    # `_mask_arr`: the (n, dk) float32 missing mask (1.0 missing)
    var mp = EncProgram(r.work_on)
    var mx = _x(mp, n, d, r.mark, r.missing)
    var mo = mp.alloc(n * dk)
    var ko = mp.put_floats(enc_floats_of(r.keep))
    _stage(mp.program, OP_NAN_MASK, n * dk, mx, n, d, ko, dk, mo)
    iterimp_unmark_empty(mp, mo, n, dk, r.empty)
    var none = List[Int]()
    var mh = runner.run(mp, none)
    _ = len(mp.consts_f)  # the program's own words stay alive past the run
    var mask = runner.words(mh) + 4 * mo
    var xt_h = h0
    var xt_off = o0
    var xt = runner.words(h0) + 4 * o0
    for s in range(len(steps)):  # driver-loop: over the fitted imputation steps in order (each reads the last imputation)
        ref step = steps[s]
        var nc = len(step.nbl)
        var gathered = _member_gather(runner, r.work_on, xt, mask, n, dk, step.j, step.nbl, False)
        if gathered.m == 0:
            runner.release(gathered.handle)
            continue
        var made = _member_put(runner, r.work_on, r.bounds, step.plan, step.state, r.posterior, step.classes,
                               step.code_dtype, gathered, xt, n, dk, step.j, nc, rng)
        runner.release(xt_h)
        xt_h = made[0]
        xt_off = made[1]
        xt = runner.words(xt_h) + 4 * xt_off
    runner.release(mh)
    if len(r.indicator) == 0:
        return IterImpApplied(xt_h, xt_off, dk, r.post_step)
    var placed = iterimp_indicator(runner, n, d, dk, r.mark, r.missing, r.indicator, xt, r.work_on, r.region)
    runner.release(xt_h)  # Xt, copied into the indicator program's arena
    return IterImpApplied(placed[0], placed[1], dk + len(r.indicator), r.post_step)


# ---------------------------------------------------------------- transform
def iterimp_transform[R: IterImpRunner](mut runner: R, r: IterImpApply) raises -> IterImpApplied:
    """IterativeImputer.transform with the built-in Bayesian ridge: the
    initial transform, the fitted steps in one program, the indicator. The
    result's arena stays held."""
    var n = r.n
    var d = r.d
    var dk = len(r.keep)
    var p0 = EncProgram(r.work_on)
    var o0 = imputer_transform_stage(p0, r.init, r.region)
    var fill_seconds: List[Int] = [r.fill_address]
    var h0 = runner.run(p0, fill_seconds)
    _ = len(p0.consts_f)
    var p = EncProgram(r.work_on)
    var xo = _x(p, n, d, r.mark, r.missing)
    var fo = enc_input(p, n * dk, 1)
    p.inout(fo, n * dk)
    var ko = p.put_floats(enc_floats_of(r.keep))
    var mo = p.alloc(n * dk)
    _stage(p.program, OP_NAN_MASK, n * dk, xo, n, d, ko, dk, mo)
    iterimp_unmark_empty(p, mo, n, dk, r.empty)
    var bo = p.put_floats(r.bounds)
    # owner 1 the filled block; then each step's fitted vectors in order
    var seconds: List[Int] = [runner.words(h0) + 4 * o0]  # the initial fill, in its held arena
    var post_step = r.post_step
    for s in range(len(r.steps)):  # driver-loop: drives one device regression per step (fitted imputation steps)
        ref step = r.steps[s]
        var j = step.j
        seconds.append(step.coef)
        var co = enc_input(p, dk, len(seconds))
        seconds.append(step.inter)
        var io = enc_input(p, 1, len(seconds))
        if r.posterior:
            var nb1 = 0
            if step.has_nbl:
                nb1 = p.put_floats(_ones(step.nbl, dk)) + 1
            var key_words: List[Int] = [r.seed, post_step]
            var key = p.put_ints(key_words)
            post_step += 1
            seconds.append(step.means)
            var mo2 = enc_input(p, step.means_words, len(seconds))
            seconds.append(step.sig)
            var so2 = enc_input(p, step.sig_words, len(seconds))
            seconds.append(step.al)
            var ao2 = enc_input(p, step.al_words, len(seconds))
            _stage(p.program, OP_II_POST, n, fo, n, dk, mo, j, co, io, bo, II_NONE, mo2, so2, ao2, nb1, key)
        else:
            _stage(p.program, OP_II_PREDICT, n, fo, n, dk, mo, j, co, io, bo, II_NONE)
    var h = runner.run(p, seconds)
    runner.release(h0)  # the initial fill, copied up by the run
    _ = len(p.consts_f)
    _ = len(p.consts_i)
    if len(r.indicator) == 0:
        return IterImpApplied(h, fo, dk, post_step)
    var made = iterimp_indicator(runner, n, d, dk, r.mark, r.missing, r.indicator, runner.words(h) + 4 * fo,
                              r.work_on, r.region)
    runner.release(h)
    return IterImpApplied(made[0], made[1], dk + len(r.indicator), post_step)
