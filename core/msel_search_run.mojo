# SPDX-License-Identifier: Apache-2.0
"""THE TYPED SEARCH RUN (lane mf-search4, 2026-11-23): GridSearchCV and
RandomizedSearchCV fitted over NATIVE MEMBERS with Mojo values only, the
body of the search META kind (core/ensemble_member.mojo,
`META_SEARCH_CLASSIFIER` / `META_SEARCH_REGRESSOR`) and callable from a Mojo
test. No Python: no Python object parameter, field, local or import.

What the UNPACK hands over (`MselSearchPlan`): one `MemberPlan` per
CANDIDATE (the base estimator's clone with the candidate's parameters set,
admitted as a member by the boundary: its parameter words ARE the
candidate's typed row), each candidate kind's capability mask, the fit's X
(host float32, rows x cols), y (int32 class codes 0 .. classes-1 for a
classifier, float32 targets for a regressor; one column), the int cv's fold
count, the scoring word and the metric program route.

The RUN, in the door's order (bindings/msel_search_door.mojo
`msel_search_fit` over bindings/msel_cv_door.mojo `msel_cv_units`):
  1. THE FOLDS of the int cv, `check_cv(cv, y, classifier)`: StratifiedKFold
     over the codes for a classifier, KFold for a regressor, unshuffled, by
     xtrees/glue_calls.mojo `xt_device_folds` (the folds CalibratedClassifierCV
     and Stacking already take). Each fold's test rows (and train rows when
     train scores are asked) are gathered ONCE and shared by every candidate.
  2. THE UNITS: every (candidate, fold), candidate-major (unit u = c * folds +
     f, the door's order), fitted on the fold's train rows as ONE native
     member set per WAVE (core/member_set_run.mojo: the GIL-free kinds on host
     workers; a wave holds as many units as there are host cores, or the
     search's n_jobs, so at most one gathered train block per worker lives).
  3. THE SCORES of each unit on its test rows (and train rows): scoring None,
     the member's own score (`MEMBER_OP_SCORE`: the estimator's `score`, what
     the door's member route reads); scoring 'accuracy', the member's predicted
     codes and THE NATIVE METRIC RUN (bindings/arena_prog_run.mojo
     `metric_inputs_run` over core/labels_accuracy_plan.mojo
     `label_accuracy_plan`, the program `accuracy_score` runs). The member
     calls run on host workers when every kind of the wave is GIL-free, else
     one after the other on this thread; the metric programs run one after
     the other (each is one program of the x_metrics runner).
  4. A FAILING UNIT: with error_score='raise' (`tolerate` False) the first
     failing unit's words, in unit order, after the wave's states are
     released; with a number, the failing fit's (or side's) scores are that
     value, as the door's `_fit_and_score` writes them.
  5. THE TABLES (bindings/msel_search_stats_run.mojo, the door's own passes):
     per candidate the mean and standard deviation over the folds, the
     'min' ranks, the best candidate (the first of rank 1) and its mean.
  6. THE REFIT (`refit`): the best candidate's plan fitted on every row (a
     set of one); its state is the result's `best_state`, owned by the
     caller. Every fold state was released (scikit-learn keeps none).
The fit notices of every unit (in unit order) and of the refit are handed
over for the caller's PACK.

SERIAL BY NATURE: the waves follow one another (each bounds the gathered
blocks); inside a wave every unit is independent and runs on the workers.
"""
from bindings.native_unit_os_threads_run import native_unit_cores
from bindings.arena_prog_run import MetricRoute, metric_inputs_run
from bindings.msel_search_stats_run import msel_row_means, msel_row_stds, msel_rank_scores, msel_rank_best
from core.entry_table import EntryRef
from core.host_parallel import host_parallelize
from core.labels_accuracy_plan import label_accuracy_plan
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_call_of,
    MEMBER_F32, MEMBER_I32, MEMBER_I64, MEMBER_OP_SCORE, MEMBER_OP_PREDICT, member_output_dtype,
    MEMBER_CAN_TEST_ROWS, MEMBER_BORROWS_FIT, MEMBER_GIL_FREE, MEMBER_GIL_DOOR,
)
from core.member_notices import member_notices_take
from core.member_set_run import (
    MemberSetRun, member_set_prepare, member_set_fit_free, member_set_turn, member_set_run_release,
    member_set_first_failure, MEMBER_SET_WORKER, MEMBER_SET_TURN_RELEASED, MEMBER_SET_TURN_HELD, MEMBER_SET_LACKING,
)
from xtrees.glue_calls import xt_device_folds, xt_gather_f32

#: `MselSearchPlan.scoring`: the member's own score (scoring=None), or
#: 'accuracy' through the native metric program.
comptime SEARCH_SCORE_MEMBER = 0
comptime SEARCH_SCORE_ACCURACY = 1

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _RunPtr = MutPointer[MemberSetRun, MutUntrackedOrigin]
comptime _StrPtr = MutPointer[String, MutUntrackedOrigin]
comptime _U8P = MutPointer[UInt8, MutUntrackedOrigin]


struct MselSearchPlan(Movable):
    """Everything typed the search RUN reads (module note). Built by the
    caller's UNPACK; `plans[c]` is candidate c's member plan."""
    var plans: List[MemberPlan]
    var masks: List[Int]
    var x: Int
    var rows: Int
    var cols: Int
    var y: Int
    var classes: Int        # 0: a regressor
    var n_splits: Int
    var scoring: Int
    var metric: MetricRoute
    var tolerate: Bool      # a numeric error_score
    var error_value: Float64
    var return_train: Bool
    var refit: Bool
    var cap: Int            # the search's n_jobs bound, 0: none
    var glue: EntryRef      # the x_trees glue entry of the fit's image (the folds, the gathers)
    # THE 2-D y (core/member_abi.mojo layout 4): a multi-output base's targets
    # are rows x `y_cols` words of `y_dtype`, with each output's class count
    # (a classifier); the folds are then KFold's (`check_cv` of a
    # multi-output target) and the scoring is the member's own.
    var y_dtype: Int
    var y_cols: Int
    var class_counts: List[Int]

    def __init__(out self, var plans: List[MemberPlan], var masks: List[Int], x: Int, rows: Int, cols: Int, y: Int,
                 classes: Int, n_splits: Int, glue: EntryRef, metric: MetricRoute):
        self.plans = plans^
        self.masks = masks^
        self.x = x
        self.rows = rows
        self.cols = cols
        self.y = y
        self.classes = classes
        self.n_splits = n_splits
        self.scoring = SEARCH_SCORE_MEMBER
        self.metric = metric
        self.tolerate = False
        self.error_value = 0.0
        self.return_train = False
        self.refit = True
        self.cap = 0
        self.glue = glue
        self.y_dtype = MEMBER_I32 if classes > 0 else MEMBER_F32
        self.y_cols = 1
        self.class_counts = List[Int]()


struct MselSearchResult(Movable):
    """What the search RUN answers: per candidate its fold scores (test,
    and train when asked), the door's tables, the best candidate and the
    refit state (0: no refit), and the fit notices for the PACK."""
    var n_splits: Int
    var test: List[List[Float64]]
    var train: List[List[Float64]]
    var means: List[Float64]
    var stds: List[Float64]
    var ranks: List[Int]
    var train_means: List[Float64]
    var best: Int
    var best_score: Float64
    var best_state: Int
    var notices: List[List[Float64]]

    def __init__(out self):
        self.n_splits = 0
        self.test = List[List[Float64]]()
        self.train = List[List[Float64]]()
        self.means = List[Float64]()
        self.stds = List[Float64]()
        self.ranks = List[Int]()
        self.train_means = List[Float64]()
        self.best = -1
        self.best_score = 0.0
        self.best_state = 0
        self.notices = List[List[Float64]]()


def _set_word(mask: Int) -> UInt8:
    """THE GIL RULE word of a unit in its set (core/ensemble_member.mojo's rule)."""
    if (mask & MEMBER_GIL_FREE) != 0:
        return MEMBER_SET_WORKER
    if (mask & MEMBER_GIL_DOOR) != 0:
        return MEMBER_SET_TURN_RELEASED
    return MEMBER_SET_TURN_HELD


def _fit_wave(mut run: MemberSetRun) raises:
    """core/member_set_run.mojo `member_set_fit` without its raise on a
    failing member: every failure stays in `run.failed` / `run.errors` for
    the caller's error rule. A member whose rows lack a class raises
    `MEMBER_SET_LACKING` (stratified folds never lack one)."""
    var turns = member_set_prepare(run)
    if not member_set_fit_free(run):
        member_set_run_release(run)
        raise Error(MEMBER_SET_LACKING)
    if turns:
        for u in range(len(run.plans)):  # driver-loop: over the units that are not worker members
            if run.gil_free[u] != MEMBER_SET_WORKER:
                member_set_turn(run, u)


def _gather_words(src: Int, rows: Int, count: Int, width: Int, dst: Int):
    """`count` rows of `width` 4-byte words of `src` at the int32 row numbers
    `rows`, copied as bits (codes or float32 targets, one word per output)."""
    var s = _I32P(unsafe_from_address=src)
    var r = _I32P(unsafe_from_address=rows)
    var d = _I32P(unsafe_from_address=dst)
    for i in range(count):
        var at = Int(r[i]) * width
        for c in range(width):
            d[i * width + c] = s[at + c]


def _score_side(run_at: Int, v: Int, x: Int, y: Int, count: Int, cols: Int, y_dtype: Int, y_cols: Int,
                scoring: Int, pred: Int, pred_dtype: Int) raises -> Float64:
    """Unit `v` of the wave on one side's gathered block: the member's own
    score, or its predicted codes (scoring 'accuracy'; the metric runs after
    the wave): written in the member's own `pred_dtype` into the 8-byte words
    at `pred`, then narrowed in place to int32 codes at the same address."""
    var run = _RunPtr(unsafe_from_address=run_at)
    var call = member_call_of(run[].plans[v], run[].states[v])
    call.x = MemberMatrix(x, count, cols, MEMBER_F32)
    if scoring == SEARCH_SCORE_MEMBER:
        call.y = MemberMatrix(y, count, y_cols, y_dtype)
        _ = member_call(run[].plans[v].member, MEMBER_OP_SCORE, call)
        return call.value
    call.dst = MemberMatrix(pred, count, 1, pred_dtype)
    _ = member_call(run[].plans[v].member, MEMBER_OP_PREDICT, call)
    if pred_dtype == MEMBER_I64:
        var wide = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=pred)
        var narrow = _I32P(unsafe_from_address=pred)
        for i in range(count):  # ascending: word i is read (8 bytes at 8i) before any later write reaches it
            narrow[i] = Int32(wide[i])
    return 0.0


def _accuracy(route: MetricRoute, truth: Int, pred: Int, count: Int, classes: Int) raises -> Float64:
    """`accuracy_score(y_true, y_pred)` of one side through the native metric
    program (the plan `labels_accuracy` runs; codes in 0 .. classes-1)."""
    var acc = label_accuracy_plan(count, classes, False)
    var addresses: List[Int] = [truth, pred]
    var sizes: List[Int] = [count, count]
    var arena = metric_inputs_run(route, acc.program, addresses, sizes)
    var at = arena.base() + 4 * (acc.result_offset + 8)
    var value = _F64P(unsafe_from_address=at)[0]
    _ = arena^
    _ = acc^
    return value


def msel_search_run(plan: MselSearchPlan) raises -> MselSearchResult:
    """The whole search fit over native members (module note). Raises the
    door's words for empty candidates, the folds' refusals and (error_score
    'raise') the first failing unit's words."""
    var n = plan.rows
    var d = plan.cols
    var m = len(plan.plans)
    var k = plan.classes
    var classifier = k > 0
    var y_dtype = plan.y_dtype
    var y_cols = max(plan.y_cols, 1)
    if y_cols > 1 and (classifier or plan.scoring != SEARCH_SCORE_MEMBER):
        raise Error("member ABI: a search over a multi-output member folds by KFold and scores by the member")
    if m == 0:
        raise Error("No fits were performed. Was the CV iterator empty? Were there no candidates?")
    if len(plan.masks) != m:
        raise Error("member ABI: a search needs one capability mask per candidate")
    var n_splits = plan.n_splits
    if n_splits < 2:
        raise Error("ValueError: k-fold cross-validation requires at least one train/test split by setting "
                    + "n_splits=2 or more, got n_splits=" + String(n_splits) + ".")
    if n_splits > n:
        raise Error("ValueError: Cannot have number of splits n_splits=" + String(n_splits)
                    + " greater than the number of samples: n_samples=" + String(n) + ".")
    # 1. The folds (fold f: its n - c train rows, then its c test rows).
    var fold_counts = List[Int32](length=n_splits + 1, fill=Int32(0))
    var fold_rows = List[Int32](length=max(n_splits * n, 1), fill=Int32(0))
    var rows_at = Int(fold_rows.unsafe_ptr())
    var codes_at = plan.y if classifier else Int(fold_counts.unsafe_ptr())
    var status = xt_device_folds(plan.glue, codes_at, rows_at, Int(fold_counts.unsafe_ptr()), n, n_splits, k)
    if status == 1:
        raise Error("ValueError: n_splits=" + String(n_splits)
                    + " cannot be greater than the number of members in each class.")
    if status != 0:
        raise Error("ValueError: y codes outside [0, n_classes)")
    var used = 0
    for f in range(n_splits):  # driver-loop: over the cv folds
        if Int(fold_counts[f]) > 0:
            used = f + 1
    var sides = 2 if plan.return_train else 1
    # Each candidate's predicted code type (scoring 'accuracy'), asked once.
    var pred_dtypes = List[Int](length=m, fill=MEMBER_I32)
    if plan.scoring == SEARCH_SCORE_ACCURACY:
        for c in range(m):  # driver-loop: over the candidates (one OUTPUTS question each)
            var probe = plan.plans[c].copy()
            probe.classes = k
            var dtype = member_output_dtype(probe, MEMBER_OP_PREDICT, d, 1, MEMBER_I32)
            if dtype != MEMBER_I32 and dtype != MEMBER_I64:
                raise Error("member ABI: a search scored by accuracy needs a member that predicts class codes")
            pred_dtypes[c] = dtype
    var pred_dtypes_at = Int(pred_dtypes.unsafe_ptr())
    # Per (fold, side): its row numbers, its count and its gathered X and y.
    var side_rows = List[Int](length=used * sides, fill=0)
    var side_count = List[Int](length=used * sides, fill=0)
    var side_x = List[List[Float32]](capacity=used * sides)
    var side_y = List[List[Int32]](capacity=used * sides)
    for f in range(used):  # driver-loop: over the cv folds (their sides' rows)
        var c = Int(fold_counts[f])
        side_rows[f * sides] = rows_at + 4 * (f * n + n - c)
        side_count[f * sides] = c
        if sides == 2:
            side_rows[f * sides + 1] = rows_at + 4 * (f * n)
            side_count[f * sides + 1] = n - c
    for s in range(used * sides):  # driver-loop: over the fold sides (their blocks, allocated once)
        side_x.append(List[Float32](length=max(side_count[s] * d, 1), fill=0.0))
        side_y.append(List[Int32](length=max(side_count[s] * y_cols, 1), fill=Int32(0)))
    var columns = List[Int32](length=max(d, 1), fill=Int32(0))
    for j in range(d):  # small-loop(the column numbers 0 .. d-1 of a side's gather, once per fit)
        columns[j] = Int32(j)
    for s in range(used * sides):  # driver-loop: over the fold sides (one glue gather of X each, parallel inside)
        _ = xt_gather_f32(plan.glue, plan.x, side_rows[s], Int(columns.unsafe_ptr()), Int(side_x[s].unsafe_ptr()),
                          n, d, side_count[s], d)
    var side_rows_at = Int(side_rows.unsafe_ptr())
    var side_count_at = Int(side_count.unsafe_ptr())
    var y_src = plan.y
    var y_dst = List[Int](length=used * sides, fill=0)
    for s in range(used * sides):  # driver-loop: over the fold sides (the target block addresses)
        y_dst[s] = Int(side_y[s].unsafe_ptr())
    var y_dst_at = Int(y_dst.unsafe_ptr())

    def _gather_y(s: Int) {imm side_rows_at, imm side_count_at, imm y_src, imm y_dst_at, imm y_cols}:
        var rows = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=side_rows_at)[s]
        var count = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=side_count_at)[s]
        var dst = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=y_dst_at)[s]
        _gather_words(y_src, rows, count, y_cols, dst)

    host_parallelize(_gather_y, used * sides)
    # 2-4. The units, wave by wave.
    var units = m * used
    var test = List[Float64](length=max(units, 1), fill=0.0)
    var train = List[Float64](length=max(units, 1), fill=0.0)
    var res = MselSearchResult()
    res.n_splits = used
    var wave = max(native_unit_cores(), 1)
    if plan.cap > 0:
        wave = min(wave, plan.cap)
    var u0 = 0
    while u0 < units:  # driver-loop: over the waves of units (each bounds the gathered train blocks)
        var u1 = min(units, u0 + wave)
        var width = u1 - u0
        var run = MemberSetRun(plan.x, n, d, y_dtype, MEMBER_I32)
        run.cap = plan.cap
        run.y_cols = y_cols
        var all_free = True
        for u in range(u0, u1):  # driver-loop: over the wave's units (admission into the set, once each)
            var c = u // used
            var f = u - c * used
            var p = plan.plans[c].copy()
            p.classes = k
            p.class_counts = plan.class_counts.copy()
            var mask = plan.masks[c]
            if (mask & MEMBER_GIL_FREE) == 0:
                all_free = False
            var count = Int(fold_counts[f])
            run.add(p^, plan.y, 0, rows_at + 4 * (f * n), n - count, 0, 0,
                    UInt8(1) if (mask & MEMBER_CAN_TEST_ROWS) != 0 else UInt8(0),
                    UInt8(1) if (mask & MEMBER_BORROWS_FIT) != 0 else UInt8(0), _set_word(mask))
        _fit_wave(run)
        if not plan.tolerate:
            var first = member_set_first_failure(run)
            if first >= 0:
                var words = run.errors[first].copy()
                member_set_run_release(run)
                raise Error(words)
        # The scores: one task per (unit, side), every member call of the wave.
        var tasks = width * sides
        var values = List[Float64](length=max(tasks, 1), fill=0.0)
        var failed = List[UInt8](length=max(tasks, 1), fill=UInt8(0))
        var errors = List[String](length=max(tasks, 1), fill=String(""))
        var preds = List[List[Int64]](capacity=tasks)
        var pred_at = List[Int](length=max(tasks, 1), fill=0)
        for t in range(tasks):  # driver-loop: over the wave's (unit, side) tasks (their code buffers)
            var f = (u0 + t // sides) % used
            var count = side_count[f * sides + t % sides]
            preds.append(List[Int64](length=max(count, 1) if plan.scoring == SEARCH_SCORE_ACCURACY else 1,
                                     fill=Int64(0)))
            pred_at[t] = Int(preds[t].unsafe_ptr())
        var x_at = List[Int](length=used * sides, fill=0)
        for s in range(used * sides):  # driver-loop: over the fold sides (their X block addresses)
            x_at[s] = Int(side_x[s].unsafe_ptr())
        var run_at = Int(UnsafePointer(to=run))
        var values_at = Int(values.unsafe_ptr())
        var failed_at = Int(failed.unsafe_ptr())
        var errors_at = Int(errors.unsafe_ptr())
        var pred_list = Int(pred_at.unsafe_ptr())
        var x_list = Int(x_at.unsafe_ptr())
        var scoring = plan.scoring

        def _score(t: Int) {imm run_at, imm values_at, imm failed_at, imm errors_at, imm pred_list, imm x_list,
                            imm y_dst_at, imm side_count_at, imm sides, imm used, imm u0, imm d, imm y_dtype,
                            imm y_cols, imm scoring, imm pred_dtypes_at}:
            var v = t // sides
            var side = t % sides
            var s = ((u0 + v) % used) * sides + side
            var rp = _RunPtr(unsafe_from_address=run_at)
            if rp[].failed[v] != UInt8(0) or rp[].states[v] == 0:
                _U8P(unsafe_from_address=failed_at)[t] = UInt8(1)
                return
            try:
                var value = _score_side(run_at, v, MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=x_list)[s],
                                        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=y_dst_at)[s],
                                        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=side_count_at)[s],
                                        d, y_dtype, y_cols, scoring,
                                        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pred_list)[t],
                                        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=pred_dtypes_at)[
                                            (u0 + v) // used])
                _F64P(unsafe_from_address=values_at)[t] = value
            except error:
                _U8P(unsafe_from_address=failed_at)[t] = UInt8(2)
                _StrPtr(unsafe_from_address=errors_at)[t] = String(error)

        if all_free:
            host_parallelize(_score, tasks)
        else:
            for t in range(tasks):  # driver-loop: over the wave's tasks (THE GIL RULE: kinds that are not GIL-free)
                _score(t)
        # The metric programs (scoring 'accuracy'), one after the other.
        if scoring == SEARCH_SCORE_ACCURACY:
            for t in range(tasks):  # driver-loop: over the wave's tasks (one native metric program each)
                if failed[t] != UInt8(0):
                    continue
                var s = ((u0 + t // sides) % used) * sides + t % sides
                try:
                    values[t] = _accuracy(plan.metric, y_dst[s], pred_at[t], side_count[s], k)
                except error:
                    failed[t] = UInt8(2)
                    errors[t] = String(error)
        # The error rule, in unit order.
        for t in range(tasks):  # driver-loop: over the wave's tasks (their scores)
            var u = u0 + t // sides
            var side = t % sides
            var value = values[t]
            if failed[t] != UInt8(0):
                if not plan.tolerate:
                    var words = errors[t].copy()
                    member_set_run_release(run)
                    raise Error(words)
                value = plan.error_value
            if side == 0:
                test[u] = value
            else:
                train[u] = value
        # The units' fit notices, in unit order; the states go.
        for v in range(width):  # driver-loop: over the wave's units (their fit notices)
            if run.states[v] != 0:
                var words = List[Float64]()
                try:
                    member_notices_take(run.plans[v], run.states[v], words)
                except:
                    pass
                if len(words) > 0:
                    res.notices.append(words^)
        member_set_run_release(run)
        _ = len(preds)  # the code buffers were read through the tasks above
        _ = len(values)
        _ = len(failed)
        _ = len(errors)
        _ = len(x_at)
        _ = len(pred_at)
        u0 = u1
    _ = len(pred_dtypes)
    _ = len(side_x)  # the blocks were read through every wave above
    _ = len(side_y)
    _ = len(y_dst)
    _ = len(side_rows)
    _ = len(side_count)
    _ = len(fold_rows)
    _ = len(fold_counts)
    _ = len(columns)
    # 5. The tables.
    for c in range(m):  # driver-loop: over the candidates (their fold scores)
        var row = List[Float64](capacity=used)
        var row_train = List[Float64](capacity=used)
        for f in range(used):  # small-loop(one candidate's folds)
            row.append(test[c * used + f])
            row_train.append(train[c * used + f])
        res.test.append(row^)
        res.train.append(row_train^)
    res.means = msel_row_means(res.test)
    res.stds = msel_row_stds(res.test, res.means)
    res.ranks = msel_rank_scores(res.means)
    if plan.return_train:
        res.train_means = msel_row_means(res.train)
    res.best = msel_rank_best(res.ranks)
    res.best_score = res.means[res.best]
    # 6. The refit of the best on every row (a set of one).
    if plan.refit:
        var best = plan.plans[res.best].copy()
        best.classes = k
        best.class_counts = plan.class_counts.copy()
        var mask = plan.masks[res.best]
        var run = MemberSetRun(plan.x, n, d, y_dtype, MEMBER_I32)
        run.cap = plan.cap
        run.y_cols = y_cols
        run.add(best^, plan.y, 0, 0, 0, 0, 0,
                UInt8(1) if (mask & MEMBER_CAN_TEST_ROWS) != 0 else UInt8(0),
                UInt8(1) if (mask & MEMBER_BORROWS_FIT) != 0 else UInt8(0), _set_word(mask))
        _fit_wave(run)
        var first = member_set_first_failure(run)
        if first >= 0:
            var words = run.errors[first].copy()
            member_set_run_release(run)
            raise Error(words)
        res.best_state = run.states[0]
        var words = List[Float64]()
        try:
            member_notices_take(run.plans[0], run.states[0], words)
        except:
            pass
        if len(words) > 0:
            res.notices.append(words^)
        run.states[0] = 0  # the caller owns the refit state
        member_set_run_release(run)
    return res^
