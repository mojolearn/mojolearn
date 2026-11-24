# SPDX-License-Identifier: Apache-2.0
"""CLASS WEIGHTS OF A CLASSIFIER MEMBER (lane mf-adm3, 2026-11-23): the
typed RUN of `class_weight` inside a member's FIT (core/member_abi.mojo), so
that a classifier with `class_weight='balanced'` or a dict is a native member
and answers as its own fit door answers. No Python: no Python object
parameter, field, local or import.

Before this lane every classifier admit refused a class_weight (the labels a
dict is keyed by never cross the ABI) and the callers kept their door paths.
Now:

  * 'balanced' needs no label: the member FIT computes it from the class
    codes it is handed (`call.y`, codes 0 .. classes-1 of the fit's own
    classes), with the bodies the doors run: the class counts of
    xtrees/glue_device.mojo `code_counts_host` (integers, split over the
    host cores here; the sum of integer counts is exact in any order), the
    ratio of core/array_helpers_impl.mojo `class_ratio_f64` (scikit-learn's
    n / (k * count_c), one binary64 division per class; the same function),
    then each family's own step (below).
  * a dict is read at UNPACK, where the caller's labels are
    (bindings/member_class_weight_unpack.mojo `member_cw_fill`): one weight
    and one "named" word per caller code, in the parameter words. A fit
    whose rows hold only some classes (a calibration fold re-coded onto the
    classes it holds, xtrees/trees_calibrated_fit_run.mojo
    `cal_fold_classes_run`) narrows the table to those classes
    (`member_cw_subset`); a named class the fit does not hold is the
    refusal the estimator's own fit raises for a key absent from its y,
    when the narrowed table also leaves a class unweighted
    (core/class_weight_plan.mojo `class_weight_dict_refused`).
  * a dict whose labels are the target's own numeric VALUES, known only in
    RUN (an IterativeImputer classifier step: its classes are the step's
    distinct values, x_prep/iterimp_run.mojo `iterimp_class_values`) is
    read at UNPACK into a KEYED tail, one (key, weight) pair per dict entry
    in the estimator's own conversion (`member_cw_fill_keyed`); the step
    turns it into the table over its classes (`member_cw_from_values`): a
    class matches the key equal to its value (Python's numeric `==` of the
    key and the float32 label: one float64 compare), and a key no class
    matches is the absent refusal. A key that is not a number has no entry
    and marks every step's table absent (no numeric class equals it: the
    step's fit refuses it, as the estimator's own fit does; lane mf-left4).
  * a value the estimator refuses (MEMBER_CW_INVALID, lane mf-left4) is a
    pending tail no fit reads: the FIT caller's UNPACK raises the
    estimator's own words (`member_cw_fill_plan`).

THE TAIL, appended to the family's NEW words at `member_cw_start(...)`:
  [0] mode (MEMBER_CW_BALANCED, MEMBER_CW_DICT); [1] K, the table's class
  count (0 for 'balanced'; -1 for a dict whose UNPACK has not read the
  caller's labels yet: such a member is never fitted); [2] absent (1: the
  table names a class the fit's rows do not hold); then per class c of the
  table, two words: [3 + 2c] its weight (the family's own conversion of the
  dict value; 1.0 for a class the dict does not name), [4 + 2c] named (1: a
  key of the dict).
SGD (word 18) and LogisticRegressionCV (word 5) keep their own 'balanced'
word (1); a dict there is word 2 and this tail. Every other kind's
'balanced' is the tail alone.

THE FAMILY STEPS (each the door's, read from it):
  * rows (LogisticRegression, LinearSVC: bindings/logistic_native_door.mojo
    `_row_weights`, bindings/svc_linear_boundary_unpack.mojo
    `_qn_row_weights`): the per-class plan of core/class_weight_plan.mojo
    `class_weight_rows_plan` (its Float32 narrowing and its refusals), then
    per row (sample weight, or 1) x the class word, ONE float32 product, as
    xtrees/ops.mojo `mul_f32` multiplies the gathered rows. Every class 1:
    the sample weights themselves, or ones (a fit with a class_weight is
    always a weighted fit there).
  * float64 per class (SVC: bindings/svc_kernel_params_unpack.mojo
    `svm_weights_unpack`): the ratios or the dict's floats, for the solver's
    per-row bounds.
  * float32 per class (SGD*, RidgeClassifier, LogisticRegressionCV:
    bindings/x_linear_common_unpack.mojo `xl_class_weights_unpack`): the
    dict's floats narrowed to float32; 'balanced' is the x_linear class
    step's own (XL_OP_CLASS_PREP), as before.
Bits: the same words in the same order as each door; nothing kept serial.
"""
from core.array_helpers_impl import class_ratio_f64
from core.class_weight_plan import (
    ClassWeightInputs, class_weight_rows_plan, class_weight_dict_refused, class_weight_absent_values,
)
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from core.member_abi import MemberPlan, MEMBER_I32, MEMBER_I64, MEMBER_F32
from core.member_families import (
    MEMBER_FAMILY_LOGISTIC, MEMBER_FAMILY_XLINEAR, MEMBER_FAMILY_SGD, MEMBER_FAMILY_LINEAR_SVM,
    MEMBER_FAMILY_KERNEL_SVM,
)

comptime MEMBER_CW_NONE = 0
comptime MEMBER_CW_BALANCED = 1
comptime MEMBER_CW_DICT = 2
#: A dict keyed by the target's values: [3, P, absent, (key, weight) x P]
#: (absent 1: the dict holds a key that is not a number, which no step's
#: numeric class equals: lane mf-left4).
comptime MEMBER_CW_KEYED = 3
#: A class_weight value its estimator refuses (neither None, 'balanced' nor
#: an exact dict): [4, -1, 0]. Never fitted: the FIT caller's UNPACK raises
#: the estimator's own words (bindings/member_class_weight_unpack.mojo
#: `member_cw_fill_plan`), as the estimator's fit door raises them (lane
#: mf-left4). A caller that only predicts reads no tail.
comptime MEMBER_CW_INVALID = 4
#: Words before the per-class table: mode, K, absent.
comptime MEMBER_CW_HEAD = 3
#: Words per class of the table: weight, named.
comptime MEMBER_CW_PER_CLASS = 2

comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]


def member_cw_start(family: Int, kind: Int, params: List[Float64]) -> Int:
    """The first word of the class-weight tail in the NEW words of a
    classifier kind that takes one, else -1. The family layouts (their
    module notes): LogisticRegression 10 words, LinearSVC 11, SVC 17,
    SGDClassifier / Perceptron / PassiveAggressiveClassifier (SGD kinds
    2..4) 20, RidgeClassifier (x_linear kind 9) 3, LogisticRegressionCV
    (x_linear kind 13) 7 + its C count (word 6)."""
    if family == MEMBER_FAMILY_LOGISTIC and kind == 0:
        return 10
    if family == MEMBER_FAMILY_LINEAR_SVM and kind == 0:
        return 11
    if family == MEMBER_FAMILY_KERNEL_SVM and kind == 0:
        return 17
    if family == MEMBER_FAMILY_SGD and kind >= 2 and kind <= 4:
        return 20
    if family == MEMBER_FAMILY_XLINEAR and kind == 9:
        return 3
    if family == MEMBER_FAMILY_XLINEAR and kind == 13 and len(params) > 6:
        return 7 + Int(params[6])
    return -1


def member_cw_mode(params: List[Float64], start: Int) -> Int:
    """MEMBER_CW_NONE when the words have no tail at `start`."""
    if start < 0 or len(params) < start + MEMBER_CW_HEAD:
        return MEMBER_CW_NONE
    return Int(params[start])


def member_cw_balanced_words(mut words: List[Float64]):
    """UNPACK: append the 'balanced' tail."""
    words.append(Float64(MEMBER_CW_BALANCED))
    words.append(0.0)
    words.append(0.0)


def member_cw_pending_words(mut words: List[Float64]):
    """UNPACK: append a dict's tail before the caller's labels are read
    (K = -1: `member_cw_fill` replaces it, or the member is not fitted)."""
    words.append(Float64(MEMBER_CW_DICT))
    words.append(-1.0)
    words.append(0.0)


def member_cw_invalid_words(mut words: List[Float64]):
    """UNPACK: append the tail of a class_weight value the estimator refuses
    (MEMBER_CW_INVALID: the FIT caller's UNPACK raises its words)."""
    words.append(Float64(MEMBER_CW_INVALID))
    words.append(-1.0)
    words.append(0.0)


def member_cw_pending(params: List[Float64], start: Int) -> Bool:
    """The tail still needs the FIT caller's UNPACK: a dict before its
    labels were read, or an invalid value whose refusal that UNPACK raises."""
    var mode = member_cw_mode(params, start)
    return (mode == MEMBER_CW_DICT and params[start + 1] < 0.0) or mode == MEMBER_CW_INVALID


def member_cw_table_words(mut params: List[Float64], start: Int, values: List[Float64], named: List[Bool]):
    """UNPACK: replace the tail at `start` by a dict table over the caller's
    codes (one weight and one named word per class)."""
    var words = List[Float64](capacity=start + MEMBER_CW_HEAD + MEMBER_CW_PER_CLASS * len(values))
    for i in range(start):  # driver-loop: over the family's own NEW words (kept)
        words.append(params[i])
    words.append(Float64(MEMBER_CW_DICT))
    words.append(Float64(len(values)))
    words.append(0.0)
    for c in range(len(values)):  # driver-loop: over the caller's classes (one table entry each)
        words.append(values[c])
        words.append(1.0 if named[c] else 0.0)
    params = words^


def member_cw_subset(mut plan: MemberPlan, present: List[Int32]):
    """RUN, before the fit: the table of a member whose fit is re-coded onto
    `present` (its classes in code order, codes of the table) narrowed to
    those classes; a named class not among them marks the tail absent (the
    fit then refuses, as the estimator's own fit refuses a class_weight key
    absent from its y). 'balanced' and no tail: nothing (they read the fit's
    own codes). Never raises."""
    var start = member_cw_start(plan.family, plan.kind, plan.params)
    if member_cw_mode(plan.params, start) != MEMBER_CW_DICT:
        return
    var k = Int(plan.params[start + 1])
    if k < 0:
        return
    var absent = plan.params[start + 2]
    var inside = List[Bool](length=k, fill=False)
    for i in range(len(present)):  # driver-loop: over the classes the fit holds
        var c = Int(present[i])
        if c >= 0 and c < k:
            inside[c] = True
    for c in range(k):  # driver-loop: over the table's classes (named and not held)
        if plan.params[start + MEMBER_CW_HEAD + MEMBER_CW_PER_CLASS * c + 1] != 0.0 and not inside[c]:
            absent = 1.0
    var words = List[Float64](capacity=start + MEMBER_CW_HEAD + MEMBER_CW_PER_CLASS * len(present))
    for i in range(start):  # driver-loop: over the family's own NEW words (kept)
        words.append(plan.params[i])
    words.append(Float64(MEMBER_CW_DICT))
    words.append(Float64(len(present)))
    words.append(absent)
    for i in range(len(present)):  # driver-loop: over the classes the fit holds (their table entries)
        var c = Int(present[i])
        var at = start + MEMBER_CW_HEAD + MEMBER_CW_PER_CLASS * c
        words.append(plan.params[at] if c >= 0 and c < k else 1.0)
        words.append(plan.params[at + 1] if c >= 0 and c < k else 0.0)
    plan.params = words^


def member_cw_keyed_words(mut params: List[Float64], start: Int, keys: List[Float64], values: List[Float64],
    foreign: Bool):
    """UNPACK: replace the tail at `start` by a KEYED tail (module note).
    `foreign`: the dict also holds a key that is not a number (it has no
    entry here; every step's table is then marked absent)."""
    var words = List[Float64](capacity=start + MEMBER_CW_HEAD + MEMBER_CW_PER_CLASS * len(keys))
    for i in range(start):  # driver-loop: over the family's own NEW words (kept)
        words.append(params[i])
    words.append(Float64(MEMBER_CW_KEYED))
    words.append(Float64(len(keys)))
    words.append(1.0 if foreign else 0.0)
    for e in range(len(keys)):  # driver-loop: over the dict's entries (one pair each)
        words.append(keys[e])
        words.append(values[e])
    params = words^


def member_cw_from_values(mut plan: MemberPlan, classes: List[Float32]):
    """RUN, before a fit whose classes are the target's values `classes`
    (ascending, code order): a KEYED tail becomes the dict table over them
    (module note). Anything else: nothing. Never raises."""
    var start = member_cw_start(plan.family, plan.kind, plan.params)
    if member_cw_mode(plan.params, start) != MEMBER_CW_KEYED:
        return
    var entries = Int(plan.params[start + 1])
    var k = len(classes)
    var matched = List[Bool](length=max(entries, 0), fill=False)
    var words = List[Float64](capacity=start + MEMBER_CW_HEAD + MEMBER_CW_PER_CLASS * k)
    for i in range(start):  # driver-loop: over the family's own NEW words (kept)
        words.append(plan.params[i])
    words.append(Float64(MEMBER_CW_DICT))
    words.append(Float64(k))
    # A key that is not a number (KEYED word 2) equals no numeric class: the
    # step's fit refuses it as absent, as the estimator's own fit does.
    words.append(1.0 if plan.params[start + 2] != 0.0 else 0.0)
    for c in range(k):  # driver-loop: over the step's classes (one key search each; a dict is short)
        var value = 1.0
        var named = 0.0
        var label = Float64(classes[c])
        for e in range(entries):  # driver-loop: over the dict's entries
            if plan.params[start + MEMBER_CW_HEAD + MEMBER_CW_PER_CLASS * e] == label:
                value = plan.params[start + MEMBER_CW_HEAD + MEMBER_CW_PER_CLASS * e + 1]
                named = 1.0
                matched[e] = True
                break
        words.append(value)
        words.append(named)
    for e in range(entries):  # driver-loop: over the dict's entries (a key no class holds)
        if not matched[e]:
            words[start + 2] = 1.0
    plan.params = words^


def member_cw_values_refusal(plan: MemberPlan, classes: List[Float32]) -> String:
    """RUN, after `member_cw_from_values`: scikit-learn's refusal words for a
    table that names a key no class of the step holds while leaving a class
    unweighted (core/class_weight_plan.mojo `class_weight_dict_refused` and
    `class_weight_absent_values`: "The classes, [0, 2], are not in
    class_weight", sklearn/utils/class_weight.py:114-120), listing the
    step's unweighted classes in class order; "" when the fit may proceed.
    Lane mf-gate6: before it the member fit raised its family's fixed absent
    words, which name no class (gate mf_left4 imp_foreign rows)."""
    var start = member_cw_start(plan.family, plan.kind, plan.params)
    if member_cw_mode(plan.params, start) != MEMBER_CW_DICT:
        return String("")
    var k = Int(plan.params[start + 1])
    if k != len(classes) or plan.params[start + 2] == 0.0:
        return String("")
    var unweighted = List[Float64]()
    for c in range(k):  # driver-loop: over the step's classes (the named words)
        if plan.params[start + MEMBER_CW_HEAD + MEMBER_CW_PER_CLASS * c + 1] == 0.0:
            unweighted.append(Float64(classes[c]))
    if not class_weight_dict_refused(1, len(unweighted)):
        return String("")
    return class_weight_absent_values(unweighted)


def member_cw_dict_ready(params: List[Float64], start: Int, classes: Int, absent: String) raises:
    """FIT: a dict table that this fit can read: filled at UNPACK, one entry
    per class of the fit, and naming no class the fit's rows lack (else
    `absent`, the estimator's own words for a key absent from its y)."""
    if Int(params[start]) == MEMBER_CW_KEYED:
        raise Error("TypeError: mojolearn: a member with a class_weight dict keyed by its target's values is "
            + "fitted only by a caller that turns it into its step's table")
    if Int(params[start]) == MEMBER_CW_INVALID:
        raise Error("TypeError: mojolearn: a member with a class_weight its estimator refuses is refused at "
            + "its caller's UNPACK, never fitted")
    var k = Int(params[start + 1])
    if k < 0:
        raise Error("TypeError: mojolearn: a member with a class_weight dict is fitted only by a caller that "
            + "hands over its labels")
    if params[start + 2] != 0.0:
        # A key the fit's rows lack is refused only when the table also
        # leaves a class unweighted (`class_weight_dict_refused`, the door's
        # rule; scikit-learn 1.9.1 compute_class_weight).
        var unweighted = 0
        for c in range(k):  # driver-loop: over the table's classes (the named words)
            if params[start + MEMBER_CW_HEAD + MEMBER_CW_PER_CLASS * c + 1] == 0.0:
                unweighted += 1
        if class_weight_dict_refused(1, unweighted):
            raise Error(absent)
    if k != classes:
        raise Error("member ABI: the class_weight table has " + String(k) + " entries for " + String(classes)
            + " classes")


def member_cw_value(params: List[Float64], start: Int, c: Int) -> Float64:
    return params[start + MEMBER_CW_HEAD + MEMBER_CW_PER_CLASS * c]


@always_inline
def _code(codes: Int, dtype: Int, r: Int) -> Int:
    if dtype == MEMBER_I64:
        return Int(_I64P(unsafe_from_address=codes)[r])
    if dtype == MEMBER_F32:
        return Int(_F32P(unsafe_from_address=codes)[r])
    return Int(_I32P(unsafe_from_address=codes)[r])


def member_cw_counts(codes: Int, dtype: Int, n: Int, k: Int) raises -> List[Int64]:
    """The k class counts of `n` codes (int32, int64 or float32), the
    integers of xtrees/glue_device.mojo `code_counts_host` and its refusal,
    the rows split over the host cores (per-task counts summed in task order:
    integers, exact in any order)."""
    var counts = List[Int64](length=k, fill=Int64(0))
    if n <= 0 or k <= 0:
        return counts^
    var tasks = host_predict_task_count_work(n, n)
    var chunk = host_predict_chunk(n, tasks)
    var partial = List[Int64](length=tasks * k, fill=Int64(0))
    var bad = List[Int](length=tasks, fill=0)
    var pp = partial.unsafe_ptr()
    var bp = bad.unsafe_ptr()

    def task(worker: Int) {imm codes, imm dtype, imm n, imm k, imm chunk, imm pp, imm bp}:
        var failed = False
        for r in range(worker * chunk, min((worker + 1) * chunk, n)):
            var c = _code(codes, dtype, r)
            if c < 0 or c >= k:
                failed = True
            else:
                pp[worker * k + c] = pp[worker * k + c] + Int64(1)
        bp[worker] = Int(failed)

    host_parallelize(task, tasks)
    for worker in range(tasks):  # fixed worker summaries after the join
        if bad[worker] != 0:
            raise Error("x_trees code_counts: a class code outside [0, n_classes)")
        for c in range(k):  # driver-loop: over the classes (one integer sum each)
            counts[c] = counts[c] + partial[worker * k + c]
    _ = partial^
    _ = bad^
    return counts^


def _balanced_ratios(codes: Int, dtype: Int, n: Int, k: Int) raises -> List[Float64]:
    """scikit-learn's 'balanced' n / (k * count_c) per class: the counts,
    then core/array_helpers_impl.mojo `class_ratio_f64` (mode 0), the
    doors' own ratio function. A class with no row is the doors' division
    refusal (the fit's own classes all hold rows, so it does not occur)."""
    var counts = member_cw_counts(codes, dtype, n, k)
    var ratios = List[Float64](length=k, fill=0.0)
    var zeros = class_ratio_f64(_I64P(unsafe_from_address=Int(counts.unsafe_ptr())),
        _F64P(unsafe_from_address=Int(ratios.unsafe_ptr())), k, n, 0)
    _ = counts^
    if zeros != 0:
        raise Error("ZeroDivisionError: division by zero")
    return ratios^


def member_cw_rows(params: List[Float64], start: Int, codes: Int, dtype: Int, n: Int, k: Int, sample: Int,
    absent: String, mut rows: List[Float32]) raises -> Int:
    """FIT, the rows step (LogisticRegression, LinearSVC; module note): the
    address of the fit's float32 row weights. `sample` the caller's float32
    sample weights (0: none); `rows` holds the words when they are made here.
    Every class weight 1: `sample` itself, or `n` ones."""
    var balanced = member_cw_mode(params, start) == MEMBER_CW_BALANCED
    var inputs = ClassWeightInputs()
    if balanced:
        inputs.values = _balanced_ratios(codes, dtype, n, k)
    else:
        member_cw_dict_ready(params, start, k, absent)
        for c in range(k):  # driver-loop: over the classes (the table read at UNPACK, already checked)
            var value = member_cw_value(params, start, c)
            inputs.append_validated(value, value == 1.0)
    var plan = class_weight_rows_plan(inputs, balanced)
    _ = inputs^
    if plan.all_one:
        if sample != 0:
            return sample
        rows = List[Float32](length=n, fill=Float32(1.0))
        return Int(rows.unsafe_ptr())
    rows = List[Float32](length=n, fill=Float32(0.0))
    if n == 0:
        return Int(rows.unsafe_ptr())
    var words = Int(plan.words.unsafe_ptr())
    var dst = Int(rows.unsafe_ptr())
    var tasks = host_predict_task_count_work(n, n)
    var chunk = host_predict_chunk(n, tasks)

    def task(worker: Int) {imm codes, imm dtype, imm n, imm chunk, imm words, imm dst, imm sample}:
        var table = _F32P(unsafe_from_address=words)
        var out = _F32P(unsafe_from_address=dst)
        var sw = _F32P(unsafe_from_address=sample)
        for r in range(worker * chunk, min((worker + 1) * chunk, n)):
            # the class word gathered, then ONE float32 product with the
            # row's weight (ones without sample weights), as the doors'
            # gather and `mul_f32`
            var s = sw[r] if sample != 0 else Float32(1.0)
            out[r] = s * table[_code(codes, dtype, r)]

    host_parallelize(task, tasks)
    _ = plan^
    return Int(rows.unsafe_ptr())


def member_cw_table_f64(params: List[Float64], start: Int, codes: Int, dtype: Int, n: Int, k: Int,
    absent: String) raises -> List[Float64]:
    """FIT, the float64 step (SVC; module note): the k class weights."""
    if member_cw_mode(params, start) == MEMBER_CW_BALANCED:
        return _balanced_ratios(codes, dtype, n, k)
    member_cw_dict_ready(params, start, k, absent)
    var values = List[Float64](length=k, fill=1.0)
    for c in range(k):  # driver-loop: over the classes (the table read at UNPACK)
        values[c] = member_cw_value(params, start, c)
    return values^


def member_cw_table_f32(params: List[Float64], start: Int, k: Int, absent: String) raises -> List[Float32]:
    """FIT, the float32 step of a dict (x_linear kinds; module note): the
    k class weights the x_linear class step reads."""
    member_cw_dict_ready(params, start, k, absent)
    var values = List[Float32](length=k, fill=Float32(1.0))
    for c in range(k):  # driver-loop: over the classes (the table read at UNPACK)
        values[c] = Float32(member_cw_value(params, start, c))
    return values^
