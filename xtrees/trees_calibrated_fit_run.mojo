# SPDX-License-Identifier: Apache-2.0
"""RUN of CalibratedClassifierCV.fit over NATIVE members (lane mf-cal,
2026-11-23; docs/plans/PYTHON_AT_THE_BOUNDARY.md). No Python: no Python
object parameter, field, local or import.

The UNPACK (bindings/trees_public_meta_door.mojo `_cal_native`) admits one
clone of the base estimator per cv fold (and, for ensemble=False, one more
over every row) into ONE member set (core/member_set_run.mojo): fold f is
unit f, reading the set's X through its train rows as a row draw (an index
view for a family that reads one, else the worker's own gathered block) and
the ensemble's int32 codes with the ensemble's class count k; the full-X
unit (ensemble=False) is unit `folds`, with no draw.

RUN (`calibrated_fit_run`, the whole of it: the door takes the pins, then
releases the GIL ONCE around this call when no member's kind needs it held,
and returns the pins after it):

  * THE FOLD CLASSES (lane mf-cal2, 2026-11-23; `cal_fold_classes_run`): one
    pass per fold over the codes of its train rows (the folds across the
    host cores). A fold that holds every class is fitted as admitted. A fold
    that lacks some (two or more present) is fitted, as scikit-learn fits
    it, on the classes it holds: its member is re-coded onto them (fold
    code c is the ensemble code `fold_classes[f][c]`, ascending: the
    classes_ its own fit door encoded) with that class count, and its
    scores are scattered into the ensemble's columns at those codes (the
    door's `_sub_cols`). A fold holding one class is left to its door unit
    (`uncovered[f]`: a native member fits two or more; its fit door raises
    or fits in the estimator's own words). The set's own class-cover pass
    is not made (`skip_lacking`).
  * the set's fits (core/member_set_run.mojo: the worker members in
    parallel on the host cores, one at a time when a member's route is a
    GPU, then the other members' turns on this thread);

  * per native fold, in fold order: its test rows of X (the `_gather` glue
    op over every column, or the host row copy under the door's
    MOJOLEARN_CALIB_FAST_HOST_FOLDS B arm: the same words), the fold
    member's scores through ONE member call on its fitted state
    (decision_function when the estimator has one, else predict_proba; a
    float64 answer rounded to float32 word by word, as the door's
    admission of the member's output rounded it: `_member_f32`), then the
    column picks of the door's `_cal_scores`: binary (k = 2) one column
    (one-column output, else the positive column: code 1, the member's
    classes being the codes 0 .. k-1 the set packed), else every column
    scattered at the member's class codes (0 .. k-1, or the fold's classes)
    with `x_trees_accumulate_cols`; a two-class member's one-column
    decision (a fold of a k > 2 ensemble holding two classes) is its
    positive class's score and goes to that class's column alone (the door
    read two columns of the one-column block before mf-cal2: past its end);
  * ensemble=True: the fold's calibrators over its test codes
    (`cal_calibrators_run`: the door's `_fit_calibrators`, the same glue ops
    xt_transpose_f64 / xt_indicator_codes / xt_platt_fit / xt_isotonic_fit
    in the same order);
  * ensemble=False: the fold's scores placed at its test rows of the
    out-of-fold block (float32 words, as the door admitted them, then
    `x_trees_scatter`), and once every fold is placed the calibrators over
    the whole block and every code.

SERIAL BY RULE over the folds in the scoring step: each member call splits
its rows over the host cores itself, and the glue ops run on the x_trees
image's device on a GPU install (the door gated them one fold at a time
before this split for the same reason). The fits are the parallel units.

A fold member whose call refuses (its kind refuses a non-finite row, or does
not say its element type) is returned by index, and a failing fit returns
`CAL_FIT_FAILED` with every state released; nothing is published: the
door releases the set and takes its door path for every fold (the fallback
rule), which raises the estimator's own words.

Same programs over the same words in the same order as the door before this
split: no bits change (tools/mf_cal_digests.py proves it on the CPU column),
except the one-column two-class decision above, which the door read past
the block's end (bits change there: the old words were not defined).
"""
from std.memory import unsafe_memcpy
from core.entry_table import EntryRef
from core.host_parallel import host_parallelize
from core.member_abi import (
    member_call_of, member_outputs_of, MEMBER_OP_PREDICT_PROBA, MEMBER_OP_DECISION, MEMBER_F32, MEMBER_F64,
    MEMBER_I64,
)
from core.member_set_run import (
    MemberSetRun, member_set_prepare, member_set_fit_free, member_set_turn, member_set_first_failure,
    member_set_run_release, MEMBER_SET_WORKER,
)
from core.admitted_rows_gather import admitted_gather_rows_f32, ARG_OUT_OF_RANGE
from xtrees.trees_meta_vote_run import VoteMember, VoteRows, _member_f32
from xtrees.glue_calls import (
    xt_accumulate, xt_accumulate_cols, xt_gather_f32, xt_gather_i32, xt_scatter, xt_transpose_f64,
    xt_indicator_codes, xt_platt_fit, xt_isotonic_fit, xt_iota_i32,
)

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]

#: `calibrated_fit_scores_run` returns this when every native fold answered.
comptime CAL_ALL_ANSWERED = -1
#: `calibrated_fit_run` returns this when a member's fit failed (every state
#: released; the door's fallback rule).
comptime CAL_FIT_FAILED = -2


struct CalFitPlan(Movable):
    """What the RUN reads, built by UNPACK. `x`: X, float32 C-order `n` x `d`
    host words; `codes`: the ensemble's int32 class codes (n), `k` classes;
    `operation`: MEMBER_OP_DECISION or MEMBER_OP_PREDICT_PROBA; `sigmoid`:
    Platt calibrators (else isotonic); `ensemble`: one (member, calibrators)
    pair per fold (else the out-of-fold block and one pair over every row);
    `host_rows`: the fold's test rows are copied on the host (the door's B
    arm) instead of the `_gather` glue op. Per fold: the address and count of
    its int32 test rows. The caller keeps every buffer alive through RUN.

    Written by the RUN (`cal_fold_classes_run`), per fold: `fold_classes[f]`
    the ensemble codes its train rows hold, ascending, when it lacks some
    (empty: every class), and `fold_codes[f]` its re-coded targets (n
    words, the fold code at each of its train rows; empty: the ensemble's
    codes). THE LIFETIME RULE: a member that borrows its fit buffers reads
    `fold_codes[f]` again at PACK, so the caller keeps the plan alive until
    the set's last pack or release."""
    var x: Int
    var n: Int
    var d: Int
    var codes: Int
    var k: Int
    var operation: Int
    var sigmoid: Bool
    var ensemble: Bool
    var host_rows: Bool
    var test_rows: List[Int]
    var test_counts: List[Int]
    var fold_classes: List[List[Int32]]
    var fold_codes: List[List[Int32]]

    def __init__(out self, x: Int, n: Int, d: Int, codes: Int, k: Int, operation: Int, sigmoid: Bool,
                 ensemble: Bool, host_rows: Bool):
        self.x = x
        self.n = n
        self.d = d
        self.codes = codes
        self.k = k
        self.operation = operation
        self.sigmoid = sigmoid
        self.ensemble = ensemble
        self.host_rows = host_rows
        self.test_rows = List[Int]()
        self.test_counts = List[Int]()
        self.fold_classes = List[List[Int32]]()
        self.fold_codes = List[List[Int32]]()

    def fold(mut self, rows: Int, count: Int):
        """The next fold's test rows (int32 numbers at `rows`, `count` of them)."""
        self.test_rows.append(rows)
        self.test_counts.append(count)
        self.fold_classes.append(List[Int32]())
        self.fold_codes.append(List[Int32]())

    def folds(self) -> Int:
        return len(self.test_rows)


struct CalFitOut(Movable):
    """What the RUN wrote. Calibrator j: Platt (`isotonic[j]` False: `a[j]`,
    `b[j]`) or isotonic (`m[j]` knots, `knots_x[j]` / `knots_y[j]`, float64).
    Per fold: `native[f]` (its pair was made here; False: its door unit makes
    it) and its calibrators `first[f] .. first[f] + count[f]` (ensemble).
    ensemble=False: `scores`, the out-of-fold block (n x c float64, c = 1 for
    two classes, else k), and, when `final_ready`, the final calibrators
    `final_first .. final_first + final_count`; not ready when a fold was
    left to its door: the door places that fold's scores into `scores` and
    calls `cal_calibrators_run` itself."""
    var isotonic: List[Bool]
    var a: List[Float64]
    var b: List[Float64]
    var m: List[Int]
    var knots_x: List[List[Float64]]
    var knots_y: List[List[Float64]]
    var native: List[Bool]
    var first: List[Int]
    var count: List[Int]
    var scores: List[Float64]
    var final_ready: Bool
    var final_first: Int
    var final_count: Int

    def __init__(out self):
        self.isotonic = List[Bool]()
        self.a = List[Float64]()
        self.b = List[Float64]()
        self.m = List[Int]()
        self.knots_x = List[List[Float64]]()
        self.knots_y = List[List[Float64]]()
        self.native = List[Bool]()
        self.first = List[Int]()
        self.count = List[Int]()
        self.scores = List[Float64]()
        self.final_ready = False
        self.final_first = 0
        self.final_count = 0

    def calibrators(self) -> Int:
        return len(self.isotonic)

    def add_sigmoid(mut self, a: Float64, b: Float64):
        self.isotonic.append(False)
        self.a.append(a)
        self.b.append(b)
        self.m.append(0)
        self.knots_x.append(List[Float64]())
        self.knots_y.append(List[Float64]())

    def add_isotonic(mut self, var knots_x: List[Float64], var knots_y: List[Float64], m: Int):
        self.isotonic.append(True)
        self.a.append(0.0)
        self.b.append(0.0)
        self.m.append(m)
        self.knots_x.append(knots_x^)
        self.knots_y.append(knots_y^)


def cal_columns(k: Int) -> Int:
    """The score columns of a k-class calibration: 1 for two classes, else k."""
    return 1 if k == 2 else k


def cal_calibrators_run(glue: EntryRef, scores: Int, n: Int, c: Int, codes: Int, sigmoid: Bool,
                        mut res: CalFitOut) raises:
    """`CalibratedClassifierCV._fit_calibrators` over the float64 `n` x `c`
    scores at `scores` and the int32 codes at `codes` (n): one calibrator per
    class column appended to `res` (column j's 0/1 target is code 1 for
    c = 1, else code j). The door's glue ops in the door's order."""
    var st = scores
    var transposed = List[Float64]()
    if c > 1:
        transposed = List[Float64](unsafe_uninit_length=c * n)
        st = Int(transposed.unsafe_ptr())
        _ = xt_transpose_f64(glue, scores, st, n, c)
    # Zero-filled as the door's admitted vectors were (indicator writes all n).
    var yj = List[Int32](length=n, fill=Int32(0))
    var y64 = List[Float64]()
    var y64_at = 0
    if not sigmoid:
        y64 = List[Float64](length=n, fill=0.0)
        y64_at = Int(y64.unsafe_ptr())
    var yj_at = Int(yj.unsafe_ptr())
    for j in range(c):  # driver-loop: over the class columns (one calibrator each)
        var cls = 1 if c == 1 else j
        var f = st + 8 * j * n
        _ = xt_indicator_codes(glue, codes, yj_at, y64_at, n, cls)
        if sigmoid:
            var ab = List[Float64](length=2, fill=0.0)
            _ = xt_platt_fit(glue, f, yj_at, Int(ab.unsafe_ptr()), n)
            res.add_sigmoid(ab[0], ab[1])
        else:
            var kx = List[Float64](length=n, fill=0.0)
            var ky = List[Float64](length=n, fill=0.0)
            var m = xt_isotonic_fit(glue, f, y64_at, Int(kx.unsafe_ptr()), Int(ky.unsafe_ptr()), n)
            # The door kept `kx[0:m]`, `ky[0:m]` (`_copy_block`).
            var keep_x = List[Float64](unsafe_uninit_length=m)
            var keep_y = List[Float64](unsafe_uninit_length=m)
            if m > 0:
                unsafe_memcpy(dest=keep_x.unsafe_ptr(), src=kx.unsafe_ptr(), count=m)
                unsafe_memcpy(dest=keep_y.unsafe_ptr(), src=ky.unsafe_ptr(), count=m)
            _ = len(kx)  # read through the fit and the copy above
            _ = len(ky)
            res.add_isotonic(keep_x^, keep_y^, m)
    # Integer addresses carry no lifetime: the work buffers live to here.
    _ = len(transposed)
    _ = len(yj)
    _ = len(y64)


@always_inline
def _cal_row(draw: Int, i: Int, wide: Bool) -> Int:
    """Row `i` of a fold's draw (int32 or int64 numbers at `draw`)."""
    if wide:
        return Int(MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=draw)[i])
    return Int(MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=draw)[i])


def cal_fold_classes_run(mut run: MemberSetRun, mut plan: CalFitPlan):
    """THE FOLD CLASSES (module note), before the set's fits: fold f is unit
    f of `run`, its train rows the unit's row draw over `plan.codes`. Marks
    the classes each fold's train rows hold (the folds across the host
    cores, one pass over its codes each: the pass the set's class-cover rule
    made, which `skip_lacking` now leaves out), then per fold: every class,
    nothing; two or more but not all, `fold_classes[f]` and the re-coded
    `fold_codes[f]` (written across the host cores, one fold per task), the
    unit's target and class count pointed at them; one class,
    `uncovered[f]` (its door unit). Never raises."""
    var folds = plan.folds()
    var k = plan.k
    var n = plan.n
    run.skip_lacking = True
    if folds == 0 or k <= 0:
        return
    var wide = run.index_dtype == MEMBER_I64
    var codes = plan.codes
    var seen = List[UInt8](length=folds * k, fill=UInt8(0))
    var seen_at = Int(seen.unsafe_ptr())
    var run_at = Int(UnsafePointer(to=run))

    def _mark(f: Int) {imm seen_at, imm run_at, imm codes, imm k, imm n, imm wide}:
        var r = MutPointer[MemberSetRun, MutUntrackedOrigin](unsafe_from_address=run_at)
        var draw = r[].picks[4 * f]
        var count = r[].picks[4 * f + 1]
        var rows = count if count > 0 else n  # count 0: the unit reads every row
        var y = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=codes)
        var marks = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=seen_at + f * k)
        for i in range(rows):
            var c = Int(y[_cal_row(draw, i, wide) if count > 0 else i])
            if c >= 0 and c < k:
                marks[c] = UInt8(1)

    host_parallelize(_mark, folds)
    # ensemble code -> fold code, per re-coded fold (k words each)
    var maps = List[Int32](length=folds * k, fill=Int32(0))
    var recoded = List[Int]()
    var targets = List[Int]()
    for f in range(folds):  # driver-loop: over the cv folds (each fold's k class marks)
        var present = List[Int32]()
        for c in range(k):  # driver-loop: over the classes (one mark each)
            if seen[f * k + c] != UInt8(0):
                maps[f * k + c] = Int32(len(present))
                present.append(Int32(c))
        var held = len(present)
        if held == k:
            continue
        if held < 2:
            run.uncovered[f] = UInt8(1)  # one class: its door unit (a native member fits two or more)
            continue
        plan.fold_classes[f] = present^
        plan.fold_codes[f] = List[Int32](length=n, fill=Int32(0))
        var at = Int(plan.fold_codes[f].unsafe_ptr())
        run.targets[f] = at
        run.plans[f].classes = held
        recoded.append(f)
        targets.append(at)
    if len(recoded) > 0:
        var recoded_at = Int(recoded.unsafe_ptr())
        var targets_at = Int(targets.unsafe_ptr())
        var maps_at = Int(maps.unsafe_ptr())

        def _recode(i: Int) {imm recoded_at, imm targets_at, imm maps_at, imm run_at, imm codes, imm k, imm n,
                             imm wide}:
            var f = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=recoded_at)[i]
            var dst = MutPointer[Int32, MutUntrackedOrigin](
                unsafe_from_address=MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=targets_at)[i])
            var r = MutPointer[MemberSetRun, MutUntrackedOrigin](unsafe_from_address=run_at)
            var draw = r[].picks[4 * f]
            var count = r[].picks[4 * f + 1]
            var rows = count if count > 0 else n
            var y = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=codes)
            var to = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=maps_at + 4 * f * k)
            for j in range(rows):
                var row = _cal_row(draw, j, wide) if count > 0 else j
                dst[row] = to[Int(y[row])]

        host_parallelize(_recode, len(recoded))
    # Integer addresses carry no lifetime: the work lists live to here.
    _ = len(seen)
    _ = len(maps)
    _ = len(recoded)
    _ = len(targets)


def _cal_fold_scores(glue: EntryRef, mut run: MemberSetRun, plan: CalFitPlan, f: Int, iota: Int, xte: Int,
                     nte: Int, class_cols: Int, mut s: List[Float64]) raises -> Bool:
    """Fold f's float64 nte x c scores (the door's `_cal_scores`) from ONE
    call of its fitted member over its test rows at `xte`, into `s` (zero,
    nte x c). False when the member does not serve the call (no float
    answer, a width other than the door's, or a refusal)."""
    var k = plan.k
    var call = member_call_of(run.plans[f], run.states[f])
    var cols = member_outputs_of(run.plans[f], call, plan.operation)
    var dtype = call.out_dtype
    # The member's classes in the ensemble's columns: 0 .. k-1, or the
    # classes a fold that lacks some holds (`_sub_cols`).
    var ks = k
    var class_at = class_cols
    if len(plan.fold_classes[f]) > 0:
        ks = len(plan.fold_classes[f])
        class_at = Int(plan.fold_classes[f].unsafe_ptr())
    # A classifier that does not say its element type is not served (the
    # door's `member_held_open` rule); the door's widths: predict_proba one
    # column per class, k > 2 one column per class, but for a two-class
    # member's one-column decision: its positive class's (its second code's)
    # column alone.
    if dtype != MEMBER_F32 and dtype != MEMBER_F64:
        return False
    if cols < 1 or (plan.operation == MEMBER_OP_PREDICT_PROBA and cols != ks):
        return False
    if k > 2:
        if cols == 1 and ks == 2 and plan.operation == MEMBER_OP_DECISION:
            class_at += 4
            ks = 1
        elif cols != ks:
            return False
    var keep = List[List[Float32]]()
    var member = VoteMember(Int(run.plans.unsafe_ptr() + f), run.states[f], plan.operation, cols, dtype, 0, 1.0)
    var blk = _member_f32(member, VoteRows(xte, nte, plan.d), keep)
    if blk == 0:
        return False
    var acc = Int(s.unsafe_ptr())
    if k == 2:
        if cols == 1:
            _ = xt_accumulate(glue, acc, blk, nte, 1.0)
        else:
            # `_one_col`: the positive column (code 1) as an nte x 1 block, the
            # door's `_gather` over the rows arange(nte) and the column [1].
            var one: List[Int32] = [1]
            var col = List[Float32](length=nte, fill=0.0)
            _ = xt_gather_f32(glue, blk, iota, Int(one.unsafe_ptr()), Int(col.unsafe_ptr()), nte, cols, nte, 1)
            _ = xt_accumulate(glue, acc, Int(col.unsafe_ptr()), nte, 1.0)
            _ = len(one)
            _ = len(col)
    else:
        _ = xt_accumulate_cols(glue, acc, blk, class_at, nte, k, ks, 1.0)
    _ = len(keep)  # the native block is read through the ops above
    _ = len(s)
    _ = len(plan.fold_classes[f])  # read through `class_at` above
    return True


def calibrated_fit_scores_run(glue: EntryRef, mut run: MemberSetRun, plan: CalFitPlan,
                              mut res: CalFitOut) raises -> Int:
    """RUN after the set's fits (module note): every native fold's scores
    and calibrators (ensemble) or placed scores and the final calibrators
    (ensemble=False). Returns CAL_ALL_ANSWERED, or the first fold whose
    member did not serve its call (nothing then is to be published)."""
    var n = plan.n
    var d = plan.d
    var k = plan.k
    var c = cal_columns(k)
    var folds = plan.folds()
    # The door's `_arange`s, made once: every column of X, the rows of the
    # largest fold, the class codes 0 .. k-1 (`_sub_cols` of a member whose
    # classes are the ensemble's codes).
    var widest = 1
    for f in range(folds):  # driver-loop: over the cv folds (their test counts)
        widest = max(widest, plan.test_counts[f])
    var cols = List[Int32](length=max(d, 1), fill=Int32(0))
    var iota = List[Int32](length=widest, fill=Int32(0))
    var class_cols = List[Int32](length=k, fill=Int32(0))
    _ = xt_iota_i32(glue, Int(cols.unsafe_ptr()), d)
    _ = xt_iota_i32(glue, Int(iota.unsafe_ptr()), widest)
    _ = xt_iota_i32(glue, Int(class_cols.unsafe_ptr()), k)
    if not plan.ensemble:
        res.scores = List[Float64](length=n * c, fill=0.0)
    var all_native = True
    for f in range(folds):  # driver-loop: over the cv folds, in fold order (serial by rule: module note)
        res.native.append(False)
        res.first.append(res.calibrators())
        res.count.append(0)
        if run.uncovered[f] != UInt8(0) or run.states[f] == 0:
            all_native = False  # this fold's train rows lack a class: its door unit
            continue
        var te = plan.test_rows[f]
        var nte = plan.test_counts[f]
        var xte = List[Float32](length=nte * d, fill=0.0)
        if plan.host_rows:
            if admitted_gather_rows_f32[DType.int32](plan.x, n, d, te, nte, Int(xte.unsafe_ptr())) == ARG_OUT_OF_RANGE:
                raise Error("x_trees gather: row index out of range")
        else:
            _ = xt_gather_f32(glue, plan.x, te, Int(cols.unsafe_ptr()), Int(xte.unsafe_ptr()), n, d, nte, d)
        var s = List[Float64](length=nte * c, fill=0.0)
        if not _cal_fold_scores(glue, run, plan, f, Int(iota.unsafe_ptr()), Int(xte.unsafe_ptr()), nte,
                                Int(class_cols.unsafe_ptr()), s):
            return f
        _ = len(xte)
        if plan.ensemble:
            var codes_te = List[Int32](length=nte, fill=Int32(0))
            _ = xt_gather_i32(glue, plan.codes, te, Int(codes_te.unsafe_ptr()), n, nte)
            cal_calibrators_run(glue, Int(s.unsafe_ptr()), nte, c, Int(codes_te.unsafe_ptr()), plan.sigmoid, res)
            res.count[f] = res.calibrators() - res.first[f]
            _ = len(codes_te)
        else:
            # The door admitted the fold's float64 scores as float32 before
            # the scatter (exact: each is a float32 word, or zero, widened).
            var s32 = List[Float32](unsafe_uninit_length=nte * c)
            for i in range(nte * c):  # native input marshaling: the door's float32 admission of one fold's scores
                s32[i] = s[i].cast[DType.float32]()
            _ = xt_scatter(glue, Int(res.scores.unsafe_ptr()), Int(s32.unsafe_ptr()), te, n, c, nte, c, 0)
            _ = len(s32)
        _ = len(s)
        res.native[f] = True
    if not plan.ensemble and all_native:
        res.final_first = res.calibrators()
        cal_calibrators_run(glue, Int(res.scores.unsafe_ptr()), n, c, plan.codes, plan.sigmoid, res)
        res.final_count = res.calibrators() - res.final_first
        res.final_ready = True
    _ = len(cols)
    _ = len(iota)
    _ = len(class_cols)
    return CAL_ALL_ANSWERED


def calibrated_fit_run(glue: EntryRef, mut run: MemberSetRun, mut plan: CalFitPlan, mut res: CalFitOut) raises -> Int:
    """The whole RUN (module note): the fold classes, the set's fits (the
    worker members on the host cores, then the other members' turns on this
    thread), then `calibrated_fit_scores_run`. The door calls it with the
    GIL released when no member's kind needs it held (THE GIL RULE word of
    every member is MEMBER_SET_WORKER or MEMBER_SET_TURN_RELEASED), else
    held; a Mojo test or another RUN calls it with no GIL. Returns
    CAL_ALL_ANSWERED with the fitted states left in `run` for the caller's
    PACK (or `member_set_run_release`); CAL_FIT_FAILED (states released) or
    the fold whose member did not serve its call: nothing to publish."""
    cal_fold_classes_run(run, plan)
    var turns = member_set_prepare(run)
    _ = member_set_fit_free(run)  # skip_lacking: no cover pass, so it cannot answer False
    if turns:
        for u in range(len(run.plans)):  # driver-loop: over the members that are not worker members
            if run.gil_free[u] != MEMBER_SET_WORKER:
                member_set_turn(run, u)
    if member_set_first_failure(run) >= 0:
        member_set_run_release(run)
        return CAL_FIT_FAILED
    return calibrated_fit_scores_run(glue, run, plan, res)
