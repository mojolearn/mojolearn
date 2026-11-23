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
    that lacks some is fitted, as scikit-learn fits it
    (calibration.py `_fit_classifier_calibrator_pair`: `estimator.fit` on
    the fold's train rows, whatever classes they hold), on the classes it
    holds: its member is re-coded onto them (fold code c is the ensemble
    code `fold_classes[f][c]`, ascending: the classes_ its own fit door
    encoded) with that class count, and its scores are scattered into the
    ensemble's columns at those codes (the door's `_sub_cols`). Since lane
    mf-cal3 that includes a fold holding ONE class: its member's fit is
    given one class, as the estimator's own fit is, and decides as it
    does: the families that refuse one class raise their fit door's words
    (the set records them; the door raises the first in fold order, as
    scikit-learn's fold loop raises the first), the families that fit one
    class (naive Bayes, k-NN, ...) fit it and are scored like any re-coded
    fold. The set's own class-cover pass is not made (`skip_lacking`).
  * the set's fits (core/member_set_run.mojo: the worker members in
    parallel on the host cores, one at a time when a member's route is a
    GPU, then the other members' turns on this thread);

  * per native fold, in fold order: its test rows of X (the `_gather` glue
    op over every column, or the host row copy under the door's
    MOJOLEARN_CALIB_FAST_HOST_FOLDS B arm: the same words), the fold
    member's scores through ONE member call on its fitted state
    (decision_function when the estimator has one, else predict_proba; a
    float64 answer rounded to float32 word by word, as the door's
    admission of the member's output rounded it: `member_f32_block`), then
    placed by THE PLACEMENT RULE (`cal_fold_place`, lane mf-cal3b: scikit-learn
    1.9's calibration.py and `_enforce_prediction_order`, read, not guessed):
    a fold that holds every class as before (binary: one column, the
    one-column output or the positive column, code 1; else every column at
    its code); a fold that lacks a class as scikit-learn places it (ensemble:
    one calibrator per score column against the held class zip pairs it
    with, a two-class fold's one score against its FIRST class, a one-class
    predict_proba refused in scikit-learn's words; ensemble=False: zeros, or
    float32's lowest word for a decision_function, at the classes it lacks,
    and scikit-learn's refusals of an imbalanced decision fold). Before,
    mf-cal2 put a two-class fold's score at its second class and every
    lacking fold had k calibrators (bits change on those folds only);
  * ensemble=True: the fold's calibrators (the placement's `calibrate`
    columns) over its test codes
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

A fold member whose scoring call refuses raises its own words out of the
RUN (lane mf-cal3: no door path is left to raise them), and a failing fit
returns `CAL_FIT_FAILED` with every state released and the failing members'
words in the set's `errors`; nothing is published and the door raises the
first failing fold's words (bindings/trees_public_meta_door.mojo
`_cal_native`). An answer of an element type or width the door's
`_cal_scores` does not read is a member ABI defect and raises a
RuntimeError (no family answers one: proba and decision are float words,
one column per class, or one for a two-class decision).

Same programs over the same words in the same order as the door before this
split: no bits change (tools/mf_cal_digests.py proves it on the CPU column)
for a fold that holds every class. A fold that lacks one follows THE
PLACEMENT RULE since lane mf-cal3b (bits change there, and scikit-learn's
refusals replace answers it never gave; the door's `_cal_scores` and the
predict program follow the same rule).
"""
from std.memory import unsafe_memcpy
from core.entry_table import EntryRef
from core.host_parallel import host_parallelize
from core.member_abi import (
    member_call_of, member_outputs_of, MEMBER_OP_PREDICT_PROBA, MEMBER_OP_DECISION, MEMBER_F32, MEMBER_F64,
    MEMBER_I64,
)
from core.member_class_weight import member_cw_subset
from core.member_set_run import (
    MemberSetRun, member_set_prepare, member_set_fit_free, member_set_turn, member_set_first_failure,
    member_set_run_release, MEMBER_SET_WORKER,
)
from core.admitted_rows_gather import admitted_gather_rows_f32, ARG_OUT_OF_RANGE
from xtrees.trees_meta_vote_run import VoteMember, VoteRows, member_f32_block
from xtrees.glue_calls import (
    xt_accumulate, xt_accumulate_cols, xt_gather_f32, xt_gather_i32, xt_scatter, xt_transpose_f64,
    xt_indicator_codes, xt_platt_fit, xt_isotonic_fit, xt_iota_i32,
)

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]

#: `calibrated_fit_run` returns this when every fold was fitted and scored.
comptime CAL_ALL_ANSWERED = -1
#: `calibrated_fit_run` returns this when a member's fit failed (every state
#: released; the failing members' words in the set's `errors`).
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
    # Written by the RUN on a failure (lane mf-cal3b): the failing fold (or
    # unit) in fold order, -1 none; True when its scores were refused
    # (`cal_fold_place`), False for a failed fit.
    var failed_fold: Int
    var failed_scoring: Bool

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
        self.failed_fold = -1
        self.failed_scoring = False

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
    Per fold: its calibrators `first[f] .. first[f] + count[f]` (ensemble).
    ensemble=False: `scores`, the out-of-fold block (n x c float64, c = 1 for
    two classes, else k), and the final calibrators `final_first ..
    final_first + final_count`. (Lane mf-cal3 removed `native[f]` and
    `final_ready`: every fold is the RUN's now, none is left to a door.)"""
    var isotonic: List[Bool]
    var a: List[Float64]
    var b: List[Float64]
    var m: List[Int]
    var knots_x: List[List[Float64]]
    var knots_y: List[List[Float64]]
    var first: List[Int]
    var count: List[Int]
    var scores: List[Float64]
    var final_first: Int
    var final_count: Int

    def __init__(out self):
        self.isotonic = List[Bool]()
        self.a = List[Float64]()
        self.b = List[Float64]()
        self.m = List[Int]()
        self.knots_x = List[List[Float64]]()
        self.knots_y = List[List[Float64]]()
        self.first = List[Int]()
        self.count = List[Int]()
        self.scores = List[Float64]()
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


#: The words scikit-learn's cross_val_predict adds to its imbalanced-fold
#: refusals (model_selection/_validation.py `_enforce_prediction_order`).
comptime CAL_STRATIFY = "To fix this, use a cross-validation technique resulting in properly stratified folds"


def cal_lacking_warning(m: Int, k: Int) -> String:
    """The RuntimeWarning scikit-learn's `_enforce_prediction_order` warns for
    a fold of `cross_val_predict` (ensemble=False) whose train rows hold `m`
    of the `k` classes."""
    return ("Number of classes in training fold (" + String(m) + ") does not match total number of classes ("
            + String(k) + "). Results may not be appropriate for your use case. " + CAL_STRATIFY)


struct CalPlace(Movable):
    """WHERE A MEMBER'S SCORES GO, as scikit-learn's calibration.py places them
    (lane mf-cal3b, 2026-11-23; `cal_fold_place`). The member's score block
    has `cols` columns; the calibration's scores S have `cal_columns(k)`.

      * k == 2: S's one column is block column `pick` (-1: S stays zero).
      * k > 2: block column i is added at S column `place[i]`
        (`x_trees_accumulate_cols` with ks = len(place)); every other S column
        holds `fill` (0, or float32's lowest word for a decision_function
        out-of-fold block, ensemble=False).
      * `calibrate`: the S columns that get a calibrator, in order (k == 2:
        [0], whose 0/1 target is code 1). For k > 2 it is the LAST
        len(calibrate) entries of `place` (THE PREDICT RULE: calibrator j of a
        pair reads and writes column `cols[ks - count + j]`,
        xtrees/calibrated_apply.mojo)."""
    var pick: Int
    var place: List[Int32]
    var calibrate: List[Int]
    var fill: Float64
    # The member answer's width and element type (`_cal_member_place` writes them).
    var cols: Int
    var dtype: Int

    def __init__(out self):
        self.pick = 0
        self.place = List[Int32]()
        self.calibrate = List[Int]()
        self.fill = 0.0
        self.cols = 0
        self.dtype = MEMBER_F32


def _cal_shape(rows: Int, cols: Int) -> String:
    """A NumPy shape tuple as scikit-learn's messages print it."""
    return "(" + String(rows) + ", " + String(cols) + ")"


def cal_fold_place(k: Int, held: List[Int32], operation: Int, cols: Int, ensemble: Bool, rows: Int) raises -> CalPlace:
    """THE PLACEMENT RULE of one calibration member (`CalPlace`), read from
    scikit-learn 1.9's calibration.py and model_selection/_validation.py.
    `held`: the member's classes as ensemble codes, ascending (its classes_:
    0 .. k-1 for a fold that holds every class); `operation`: decision_function
    (MEMBER_OP_DECISION) or predict_proba; `cols`: its answer's width (a
    two-class decision is one column, scikit-learn's 1-D answer); `rows`: the
    test rows (for the words). Raises scikit-learn's refusals in its words,
    and a RuntimeError for a width no family answers (a member ABI defect).

    ensemble=True (`_fit_classifier_calibrator_pair`, `_fit_calibrator`,
    `_CalibratedClassifier.predict_proba`): the scores are
    `_get_response_values` of the member: two held classes give ONE column
    (the decision, or the probability of its second class); one held class:
    a predict_proba of one column is refused ("Got predict_proba of shape
    ..., but need classifier with two classes."). Calibrator i is fitted on
    score column i against the class `pos_class_indices[i]` (zip: as many as
    score columns): every class of a full fold; each held class of a fold
    holding more than two; the FIRST held class of a fold holding two (fed
    the score of the second: scikit-learn's zip). k == 2 one calibrator.

    ensemble=False (`cross_val_predict` then `_enforce_prediction_order`): a
    fold that lacks a class has its predict_proba columns placed at the held
    classes, zeros elsewhere (k == 2: S is the column of code 1, zero when
    the fold holds only code 0); its decision_function is refused when it
    holds two classes or fewer, or when its width is not its class count
    ("Irregular ..."), else placed with float32's lowest word elsewhere (the
    scores are float32 words, `member_f32_block`). The caller warns
    `cal_lacking_warning` for such a fold. No per-fold calibrators."""
    var got = CalPlace()
    var m = len(held)
    var decision = operation == MEMBER_OP_DECISION
    var bad = cols < 1 or (not decision and cols != m) or (decision and cols != m and not (cols == 1 and m <= 2))
    # an irregular decision of a fold that lacks a class is scikit-learn's own refusal below (ensemble=False)
    if bad and not (decision and not ensemble and m != k and cols >= 1):
        raise Error("RuntimeError: mojolearn: CalibratedClassifierCV: a member holding " + String(m)
                    + " classes answered " + String(cols) + " columns (member ABI)")
    if m == k:
        # Every class: the placement every fold had before this rule.
        if k == 2:
            got.pick = 0 if cols == 1 else 1
            got.calibrate.append(0)
            return got^
        for c in range(k):  # driver-loop: over the classes (one column each)
            got.place.append(held[c])
            got.calibrate.append(Int(held[c]))
        return got^
    if ensemble:
        if m == 1:
            if not decision:
                raise Error("ValueError: Got predict_proba of shape " + _cal_shape(rows, cols)
                            + ", but need classifier with two classes.")
            # scikit-learn negates a one-class decision (`_process_decision_function`); no family with a
            # decision_function fits one class, so no pair can hold such a member.
            raise Error("RuntimeError: mojolearn: CalibratedClassifierCV: a one-class member answered "
                        + "decision_function (no family fits one class with one)")
        if k == 2:
            raise Error("RuntimeError: mojolearn: CalibratedClassifierCV: a two-class fold lacks no class")
        if m == 2:
            # One score column (the second held class's), calibrated against the FIRST held class.
            if cols == 1:
                got.place.append(held[0])
            else:
                got.place.append(held[1])  # block column 0: placed, never calibrated nor read
                got.place.append(held[0])  # block column 1, the second class's probability
            got.calibrate.append(Int(held[0]))
            return got^
        for c in range(m):  # driver-loop: over the held classes (one column each)
            got.place.append(held[c])
            got.calibrate.append(Int(held[c]))
        return got^
    # ensemble=False: `_enforce_prediction_order` over the fold's answer.
    if decision:
        if cols != 1 and cols != m:
            raise Error("ValueError: Output shape " + _cal_shape(rows, cols) + " of decision_function does not "
                        + "match number of classes (" + String(m) + ") in fold. Irregular decision_function "
                        + "outputs are not currently supported by cross_val_predict")
        if m <= 2:
            raise Error("ValueError: Only " + String(m) + " class/es in training fold, but " + String(k)
                        + " in overall dataset. This is not supported for decision_function with imbalanced "
                        + "folds. " + CAL_STRATIFY)
        if cols != m:
            raise Error("RuntimeError: mojolearn: CalibratedClassifierCV: a member holding " + String(m)
                        + " classes answered " + String(cols) + " columns (member ABI)")
        got.fill = Float64(Float32.MIN_FINITE)
    if k == 2:
        # one held class: its probability column at its code, then the column of code 1
        got.pick = 0 if held[0] == 1 else -1
        return got^
    for c in range(m):  # driver-loop: over the held classes (one column each)
        got.place.append(held[c])
    return got^


def cal_calibrators_run(glue: EntryRef, scores: Int, n: Int, c: Int, codes: Int, sigmoid: Bool,
                        which: List[Int], mut res: CalFitOut) raises:
    """`CalibratedClassifierCV._fit_calibrators` over the float64 `n` x `c`
    scores at `scores` and the int32 codes at `codes` (n): one calibrator per
    class column of `which` (empty: every column), in its order, appended to
    `res` (column j's 0/1 target is code 1 for c = 1, else code j: scikit-learn's
    `label_binarize` column). The door's glue ops in the door's order."""
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
    var count = len(which) if len(which) > 0 else c
    for q in range(count):  # driver-loop: over the calibrated class columns (one calibrator each)
        var j = which[q] if len(which) > 0 else q
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
    nothing; one or more but not all, `fold_classes[f]` and the re-coded
    `fold_codes[f]` (written across the host cores, one fold per task), the
    unit's target and class count pointed at them (one class: its fit
    decides, as the estimator's own fit does; module note). Every fold's
    train rows are non-empty (the door's UNPACK refuses an empty fold
    before the RUN). Never raises."""
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
        if held == k or held == 0:
            continue  # every class (held 0: no train rows, which UNPACK refuses before the RUN)
        # A class_weight dict's table narrowed to the classes the fold holds
        # (lane mf-adm3; core/member_class_weight.mojo): the fold's own fit
        # reads the dict against the fold's labels.
        member_cw_subset(run.plans[f], present)
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


def cal_held_classes(plan: CalFitPlan, f: Int) -> List[Int32]:
    """Fold f's member classes as ensemble codes, ascending: the classes its
    train rows hold when it lacks some (`fold_classes[f]`), else 0 .. k-1."""
    if len(plan.fold_classes[f]) > 0:
        return plan.fold_classes[f].copy()
    var every = List[Int32](capacity=plan.k)
    for c in range(plan.k):  # driver-loop: over the classes
        every.append(Int32(c))
    return every^


def _cal_member_place(mut run: MemberSetRun, plan: CalFitPlan, f: Int) raises -> CalPlace:
    """Fold f's fitted member: its answer's width and element type (one
    OUTPUTS question of its state) and `cal_fold_place` over its classes. A
    non-float answer is a member ABI defect (RuntimeError); scikit-learn's
    refusals raise in its words."""
    var call = member_call_of(run.plans[f], run.states[f])
    var cols = member_outputs_of(run.plans[f], call, plan.operation)
    var dtype = call.out_dtype
    if dtype != MEMBER_F32 and dtype != MEMBER_F64:
        raise Error("RuntimeError: mojolearn: CalibratedClassifierCV: fold " + String(f) + "'s member answered "
                    + String(cols) + " columns of element type " + String(dtype) + " (member ABI)")
    var place = cal_fold_place(plan.k, cal_held_classes(plan, f), plan.operation, cols, plan.ensemble,
                               plan.test_counts[f])
    place.cols = cols
    place.dtype = dtype
    return place^


def _cal_fold_scores(glue: EntryRef, mut run: MemberSetRun, plan: CalFitPlan, f: Int, iota: Int, xte: Int,
                     nte: Int, mut s: List[Float64]) raises -> List[Int]:
    """Fold f's float64 nte x c scores (the door's `_cal_scores`) from ONE
    call of its fitted member over its test rows at `xte`, into `s` (zero,
    nte x c), placed by scikit-learn's rule (`cal_fold_place`, lane mf-cal3b);
    returns the S columns that get a calibrator. A refusal of the call raises
    the member's own words."""
    var k = plan.k
    var place = _cal_member_place(run, plan, f)
    var cols = place.cols
    var dtype = place.dtype
    var keep = List[List[Float32]]()
    var member = VoteMember(Int(run.plans.unsafe_ptr() + f), run.states[f], plan.operation, cols, dtype, 0, 1.0)
    var blk = member_f32_block(member, VoteRows(xte, nte, plan.d), keep)
    var acc = Int(s.unsafe_ptr())
    if k == 2:
        if place.pick == 0 and cols == 1:
            _ = xt_accumulate(glue, acc, blk, nte, 1.0)
        elif place.pick >= 0:
            # `_one_col`: column `pick` (the positive column, code 1) as an nte
            # x 1 block, the door's `_gather` over the rows arange(nte).
            var one: List[Int32] = [Int32(place.pick)]
            var col = List[Float32](length=nte, fill=0.0)
            _ = xt_gather_f32(glue, blk, iota, Int(one.unsafe_ptr()), Int(col.unsafe_ptr()), nte, cols, nte, 1)
            _ = xt_accumulate(glue, acc, Int(col.unsafe_ptr()), nte, 1.0)
            _ = len(one)
            _ = len(col)
        # pick -1: S stays zero (a fold holding only code 0, ensemble=False)
    else:
        _ = xt_accumulate_cols(glue, acc, blk, Int(place.place.unsafe_ptr()), nte, k, len(place.place), 1.0)
        if place.fill != 0.0:
            # `_enforce_prediction_order`'s default of the classes the fold
            # lacks (ensemble=False decision_function: float32's lowest word)
            var mark = List[Bool](length=k, fill=True)
            for i in range(len(place.place)):  # driver-loop: over the placed columns
                mark[Int(place.place[i])] = False
            for r in range(nte):  # native input marshaling: one fill pass over one fold's unplaced columns
                for c in range(k):
                    if mark[c]:
                        s[r * k + c] = place.fill
    _ = len(keep)  # the native block is read through the ops above
    _ = len(s)
    return place.calibrate.copy()


def calibrated_fit_scores_run(glue: EntryRef, mut run: MemberSetRun, plan: CalFitPlan,
                              mut res: CalFitOut) raises:
    """RUN after the set's fits (module note): every fold's scores and
    calibrators (ensemble) or placed scores and the final calibrators
    (ensemble=False). A fold member's refused call raises its words."""
    var n = plan.n
    var d = plan.d
    var k = plan.k
    var c = cal_columns(k)
    var folds = plan.folds()
    # The door's `_arange`s, made once: every column of X, the rows of the
    # largest fold.
    var widest = 1
    for f in range(folds):  # driver-loop: over the cv folds (their test counts)
        widest = max(widest, plan.test_counts[f])
    var cols = List[Int32](length=max(d, 1), fill=Int32(0))
    var iota = List[Int32](length=widest, fill=Int32(0))
    _ = xt_iota_i32(glue, Int(cols.unsafe_ptr()), d)
    _ = xt_iota_i32(glue, Int(iota.unsafe_ptr()), widest)
    if not plan.ensemble:
        res.scores = List[Float64](length=n * c, fill=0.0)
    for f in range(folds):  # driver-loop: over the cv folds, in fold order (serial by rule: module note)
        res.first.append(res.calibrators())
        res.count.append(0)
        var te = plan.test_rows[f]
        var nte = plan.test_counts[f]
        var xte = List[Float32](length=nte * d, fill=0.0)
        if plan.host_rows:
            if admitted_gather_rows_f32[DType.int32](plan.x, n, d, te, nte, Int(xte.unsafe_ptr())) == ARG_OUT_OF_RANGE:
                raise Error("x_trees gather: row index out of range")
        else:
            _ = xt_gather_f32(glue, plan.x, te, Int(cols.unsafe_ptr()), Int(xte.unsafe_ptr()), n, d, nte, d)
        var s = List[Float64](length=nte * c, fill=0.0)
        var which = _cal_fold_scores(glue, run, plan, f, Int(iota.unsafe_ptr()), Int(xte.unsafe_ptr()), nte, s)
        _ = len(xte)
        if plan.ensemble:
            var codes_te = List[Int32](length=nte, fill=Int32(0))
            _ = xt_gather_i32(glue, plan.codes, te, Int(codes_te.unsafe_ptr()), n, nte)
            cal_calibrators_run(glue, Int(s.unsafe_ptr()), nte, c, Int(codes_te.unsafe_ptr()), plan.sigmoid, which,
                                res)
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
    if not plan.ensemble:
        res.final_first = res.calibrators()
        cal_calibrators_run(glue, Int(res.scores.unsafe_ptr()), n, c, plan.codes, plan.sigmoid, List[Int](), res)
        res.final_count = res.calibrators() - res.final_first
    _ = len(cols)
    _ = len(iota)


def calibrated_fit_run(glue: EntryRef, mut run: MemberSetRun, mut plan: CalFitPlan, mut res: CalFitOut) raises -> Int:
    """The whole RUN (module note): the fold classes, the set's fits (the
    worker members on the host cores, then the other members' turns on this
    thread), then `calibrated_fit_scores_run`. The door calls it with the
    GIL released when no member's kind needs it held (THE GIL RULE word of
    every member is MEMBER_SET_WORKER or MEMBER_SET_TURN_RELEASED), else
    held; a Mojo test or another RUN calls it with no GIL. Returns
    CAL_ALL_ANSWERED with the fitted states left in `run` for the caller's
    PACK (or `member_set_run_release`); CAL_FIT_FAILED with the states
    released and the failing members' words in `run.errors` (the first in
    fold order, `member_set_first_failure`, is the one to raise). A refused
    scoring call raises its member's words (the caller releases the set).

    FOLD ORDER OF THE REFUSALS (lane mf-cal3b): scikit-learn fits and scores
    fold after fold and raises the first failure, of a fit or of the fold's
    scores (`cal_fold_place`'s refusals). So before any score, every fold
    before the first failed fit is placed in fold order: the first refused
    placement raises (its fold in `plan.failed_fold`, `plan.failed_scoring`
    True, every state released), else a failed fit returns CAL_FIT_FAILED
    (`plan.failed_fold` its unit)."""
    plan.failed_fold = -1
    plan.failed_scoring = False
    cal_fold_classes_run(run, plan)
    var turns = member_set_prepare(run)
    _ = member_set_fit_free(run)  # skip_lacking: no cover pass, so it cannot answer False
    if turns:
        for u in range(len(run.plans)):  # driver-loop: over the members that are not worker members
            if run.gil_free[u] != MEMBER_SET_WORKER:
                member_set_turn(run, u)
    var first = member_set_first_failure(run)
    var placed = plan.folds() if first < 0 else min(first, plan.folds())
    for f in range(placed):  # driver-loop: over the cv folds before the first failed fit (one OUTPUTS question each)
        try:
            _ = _cal_member_place(run, plan, f)
        except error:
            plan.failed_fold = f
            plan.failed_scoring = True
            member_set_run_release(run)
            raise error^
    if first >= 0:
        plan.failed_fold = first
        member_set_run_release(run)
        return CAL_FIT_FAILED
    calibrated_fit_scores_run(glue, run, plan, res)
    return CAL_ALL_ANSWERED
