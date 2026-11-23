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
     the door's member route reads); a named scorer or SEVERAL (multi-metric,
     `scorers`; lane mf-metric4), each scorer's member response (predict,
     predict_proba or decision_function: asked ONCE per unit and side for
     every scorer that reads it, as scikit-learn's multimetric scorer caches
     it) and THE SCORER'S METRIC PROGRAM (core/msel_scorer_run.mojo
     `msel_scorer_value`: the builder the metric's own door calls, through
     bindings/arena_prog_run.mojo `metric_inputs_run`). The member calls run
     on host workers when every kind of the wave is GIL-free, else one after
     the other on this thread; the metric programs run one after the other
     (each is one program of the x_metrics runner, on its one context).
  4. A FAILING UNIT: with error_score='raise' (`tolerate` False) the first
     failing unit's words, in unit order, after the wave's states are
     released; with a number, the failing fit's (or side's) scores are that
     value, as the door's `_fit_and_score` writes them.
  5. THE TABLES (bindings/msel_search_stats_run.mojo, the door's own passes):
     per scorer and candidate the mean and standard deviation over the folds,
     the 'min' ranks; the best candidate (the first of rank 1) and its mean
     by the refit scorer (`refit_scorer`, refit=<name>; the one scorer
     otherwise).
  6. THE REFIT (`refit`; refit=False fits nothing): the best candidate's
     plan fitted on every row (a set of one); its state is the result's
     `best_state`, owned by the caller. Every fold state was released
     (scikit-learn keeps none).
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
from core.msel_scorer_run import (
    MselScorer, msel_scorer_row, msel_scorer_value, msel_scorer_regression, msel_scorer_binary_column,
    SCORER_MEMBER, SCORER_ACCURACY,
)
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, member_call, member_call_of, member_outputs_of,
    MEMBER_F32, MEMBER_F64, MEMBER_I32, MEMBER_I64, MEMBER_OP_SCORE, MEMBER_OP_PREDICT, member_output_dtype,
    MEMBER_CAN_TEST_ROWS, MEMBER_BORROWS_FIT, MEMBER_GIL_FREE, MEMBER_GIL_DOOR,
)
from core.member_notices import member_notices_take
from core.member_set_run import (
    MemberSetRun, member_set_prepare, member_set_fit_free, member_set_turn, member_set_run_release,
    member_set_first_failure, MEMBER_SET_WORKER, MEMBER_SET_TURN_RELEASED, MEMBER_SET_TURN_HELD, MEMBER_SET_LACKING,
)
from xtrees.glue_calls import xt_device_folds, xt_gather_f32

#: `MselSearchPlan.scoring`: the member's own score (scoring=None), or a
#: scorer code of core/msel_scorer_run.mojo ('accuracy' is 1).
comptime SEARCH_SCORE_MEMBER = SCORER_MEMBER
comptime SEARCH_SCORE_ACCURACY = SCORER_ACCURACY

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _RunPtr = MutPointer[MemberSetRun, MutUntrackedOrigin]
comptime _StrPtr = MutPointer[String, MutUntrackedOrigin]
comptime _IntP = MutPointer[Int, MutUntrackedOrigin]
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
    # MULTI-METRIC (lane mf-metric4): the scorer codes of a dict or list
    # `scoring` in the caller's name order (empty: the one `scoring`), and
    # the index among them whose table chooses the best (refit=<name>).
    var scorers: List[Int]
    var refit_scorer: Int

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
        self.scorers = List[Int]()
        self.refit_scorer = 0


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
    # Per scorer (in `scorers` order; one entry for one scorer): its fold
    # scores, test and train, and its tables. `test` .. `train_means` above
    # are the refit scorer's.
    var scorer_test: List[List[List[Float64]]]
    var scorer_train: List[List[List[Float64]]]
    var scorer_means: List[List[Float64]]
    var scorer_stds: List[List[Float64]]
    var scorer_ranks: List[List[Int]]
    var scorer_train_means: List[List[Float64]]

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
        self.scorer_test = List[List[List[Float64]]]()
        self.scorer_train = List[List[List[Float64]]]()
        self.scorer_means = List[List[Float64]]()
        self.scorer_stds = List[List[Float64]]()
        self.scorer_ranks = List[List[Int]]()
        self.scorer_train_means = List[List[Float64]]()


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


def _member_score(run_at: Int, v: Int, x: Int, y: Int, count: Int, cols: Int, y_dtype: Int,
                  y_cols: Int) raises -> Float64:
    """Unit `v` of the wave on one side's gathered block: the member's own
    score (scoring None)."""
    var run = _RunPtr(unsafe_from_address=run_at)
    var call = member_call_of(run[].plans[v], run[].states[v])
    call.x = MemberMatrix(x, count, cols, MEMBER_F32)
    call.y = MemberMatrix(y, count, y_cols, y_dtype)
    _ = member_call(run[].plans[v].member, MEMBER_OP_SCORE, call)
    return call.value


def _respond(run_at: Int, v: Int, op: Int, x: Int, count: Int, cols: Int, dst: Int, capacity: Int,
             dtype: Int) raises -> Int:
    """Unit `v`'s response `op` (predict, predict_proba or decision_function)
    on one side's gathered block, written in the member's own `dtype` into
    the 8-byte words at `dst` (`capacity` words), then narrowed in place to
    the scorer's words: int32 codes (an int64 predict) or float32 (a float64
    output; the rounding the metric doors' admission applies). Answers the
    output's columns per row."""
    var run = _RunPtr(unsafe_from_address=run_at)
    var call = member_call_of(run[].plans[v], run[].states[v])
    call.x = MemberMatrix(x, count, cols, MEMBER_F32)
    var width = 1
    if op != MEMBER_OP_PREDICT:
        width = max(member_outputs_of(run[].plans[v], call, op), 1)
    if count * width > capacity:
        raise Error("member ABI: a member response wider than its classes")
    call.dst = MemberMatrix(dst, count, width, dtype)
    _ = member_call(run[].plans[v].member, op, call)
    var words = count * width
    if dtype == MEMBER_I64:
        var wide = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=dst)
        var narrow = _I32P(unsafe_from_address=dst)
        for i in range(words):  # ascending: word i is read (8 bytes at 8i) before any later write reaches it
            narrow[i] = Int32(wide[i])
    elif dtype == MEMBER_F64:
        var wide = _F64P(unsafe_from_address=dst)
        var narrow = _F32P(unsafe_from_address=dst)
        for i in range(words):  # ascending: word i is read (8 bytes at 8i) before any later write reaches it
            narrow[i] = Float32(wide[i])
    return width


def _scorer_side(route: MetricRoute, row: MselScorer, truth: Int, response: Int, count: Int, classes: Int,
                 cols: Int) raises -> Float64:
    """One scorer on one side: the `_BINARY_PROBA` column pick of a
    two-class member's probabilities (`predict_proba[:, 1]`, as the scorer
    hands it), then core/msel_scorer_run.mojo `msel_scorer_value`."""
    if msel_scorer_binary_column(row) and classes == 2 and cols == 2:
        var column = List[Float32](length=max(count, 1), fill=0.0)
        var src = _F32P(unsafe_from_address=response)
        for i in range(count):  # driver-loop: over the side's rows (column 1 of the probabilities)
            column[i] = src[2 * i + 1]
        var value = msel_scorer_value(route, row, truth, Int(column.unsafe_ptr()), count, classes, 1)
        _ = len(column)  # read by the program above
        return value
    return msel_scorer_value(route, row, truth, response, count, classes, cols)


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
    var codes = plan.scorers.copy()
    if len(codes) == 0:
        codes.append(plan.scoring)
    if y_cols > 1 and (classifier or len(codes) != 1 or codes[0] != SEARCH_SCORE_MEMBER):
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
    # The scorers (resolved once) and the member responses they read: each
    # distinct response is asked once per unit and side.
    var member_score = len(codes) == 1 and codes[0] == SCORER_MEMBER
    var rows_of = List[MselScorer]()
    var ops = List[Int]()
    var op_of = List[Int]()
    if not member_score:
        for j in range(len(codes)):  # driver-loop: over the scorers (once per fit)
            if codes[j] == SCORER_MEMBER:
                raise Error("member ABI: the member's own score cannot be one of several scorers")
            var row = msel_scorer_row(codes[j])
            if msel_scorer_regression(row) == classifier:
                raise Error("member ABI: scorer '" + String(row.name) + "' does not score this kind of estimator")
            var at = -1
            for o in range(len(ops)):  # small-loop(the distinct responses, at most three)
                if ops[o] == row.op:
                    at = o
            if at < 0:
                at = len(ops)
                ops.append(row.op)
            op_of.append(at)
            rows_of.append(row)
    var nsc = len(codes)
    var nops = len(ops)
    # Each candidate's response element types, asked once.
    var out_dtypes = List[Int](length=max(m * nops, 1), fill=MEMBER_I32)
    for c in range(m):  # driver-loop: over the candidates (one OUTPUTS question per response)
        for o in range(nops):  # small-loop(the distinct responses)
            var probe = plan.plans[c].copy()
            probe.classes = k
            var unsaid = MEMBER_I32 if ops[o] == MEMBER_OP_PREDICT and classifier else MEMBER_F32
            var dtype = member_output_dtype(probe, ops[o], d, 1, unsaid)
            if ops[o] == MEMBER_OP_PREDICT and classifier:
                if dtype != MEMBER_I32 and dtype != MEMBER_I64:
                    raise Error("member ABI: a search scored by labels needs a member that predicts class codes")
            elif dtype != MEMBER_F32 and dtype != MEMBER_F64:
                raise Error("member ABI: a search scorer needs float32 or float64 member outputs")
            out_dtypes[c * nops + o] = dtype
    var out_dtypes_at = Int(out_dtypes.unsafe_ptr())
    var ops_at = Int(ops.unsafe_ptr())
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
    var test = List[Float64](length=max(units * nsc, 1), fill=0.0)
    var train = List[Float64](length=max(units * nsc, 1), fill=0.0)
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
        var values = List[Float64](length=max(tasks * nsc, 1), fill=0.0)
        var failed = List[UInt8](length=max(tasks * nsc, 1), fill=UInt8(0))
        var errors = List[String](length=max(tasks * nsc, 1), fill=String(""))
        var preds = List[List[Int64]](capacity=max(tasks * nops, 1))
        var pred_at = List[Int](length=max(tasks * nops, 1), fill=0)
        var pred_cols = List[Int](length=max(tasks * nops, 1), fill=1)
        var pred_cap = List[Int](length=max(tasks * nops, 1), fill=0)
        for t in range(tasks):  # driver-loop: over the wave's (unit, side) tasks (their response buffers)
            var f = (u0 + t // sides) % used
            var count = side_count[f * sides + t % sides]
            for o in range(nops):  # small-loop(the distinct responses)
                var width_o = 1 if ops[o] == MEMBER_OP_PREDICT else max(k, 1)
                preds.append(List[Int64](length=max(count * width_o, 1), fill=Int64(0)))
                pred_at[t * nops + o] = Int(preds[t * nops + o].unsafe_ptr())
                pred_cap[t * nops + o] = count * width_o
        var x_at = List[Int](length=used * sides, fill=0)
        for s in range(used * sides):  # driver-loop: over the fold sides (their X block addresses)
            x_at[s] = Int(side_x[s].unsafe_ptr())
        var run_at = Int(UnsafePointer(to=run))
        var values_at = Int(values.unsafe_ptr())
        var failed_at = Int(failed.unsafe_ptr())
        var errors_at = Int(errors.unsafe_ptr())
        var pred_list = Int(pred_at.unsafe_ptr())
        var cols_list = Int(pred_cols.unsafe_ptr())
        var cap_list = Int(pred_cap.unsafe_ptr())
        var x_list = Int(x_at.unsafe_ptr())

        def _score(t: Int) {imm run_at, imm values_at, imm failed_at, imm errors_at, imm pred_list, imm cols_list,
                            imm cap_list, imm x_list, imm y_dst_at, imm side_count_at, imm sides, imm used, imm u0,
                            imm d, imm y_dtype, imm y_cols, imm member_score, imm nsc, imm nops, imm ops_at,
                            imm out_dtypes_at}:
            var v = t // sides
            var side = t % sides
            var s = ((u0 + v) % used) * sides + side
            var rp = _RunPtr(unsafe_from_address=run_at)
            if rp[].failed[v] != UInt8(0) or rp[].states[v] == 0:
                for j in range(nsc):  # small-loop(the scorers)
                    _U8P(unsafe_from_address=failed_at)[t * nsc + j] = UInt8(1)
                return
            var x = _IntP(unsafe_from_address=x_list)[s]
            var count = _IntP(unsafe_from_address=side_count_at)[s]
            try:
                if member_score:
                    _F64P(unsafe_from_address=values_at)[t] = _member_score(
                        run_at, v, x, _IntP(unsafe_from_address=y_dst_at)[s], count, d, y_dtype, y_cols)
                    return
                var c = (u0 + v) // used
                for o in range(nops):  # small-loop(the distinct responses)
                    var at = t * nops + o
                    _IntP(unsafe_from_address=cols_list)[at] = _respond(
                        run_at, v, _IntP(unsafe_from_address=ops_at)[o], x, count, d,
                        _IntP(unsafe_from_address=pred_list)[at], _IntP(unsafe_from_address=cap_list)[at],
                        _IntP(unsafe_from_address=out_dtypes_at)[c * nops + o])
            except error:
                for j in range(nsc):  # small-loop(the scorers)
                    _U8P(unsafe_from_address=failed_at)[t * nsc + j] = UInt8(2)
                    _StrPtr(unsafe_from_address=errors_at)[t * nsc + j] = String(error)

        if all_free:
            host_parallelize(_score, tasks)
        else:
            for t in range(tasks):  # driver-loop: over the wave's tasks (THE GIL RULE: kinds that are not GIL-free)
                _score(t)
        # The metric programs (named scorers), one after the other.
        if not member_score:
            for t in range(tasks):  # driver-loop: over the wave's tasks (one native metric program per scorer)
                var s = ((u0 + t // sides) % used) * sides + t % sides
                for j in range(nsc):  # small-loop(the scorers)
                    var at = t * nsc + j
                    if failed[at] != UInt8(0):
                        continue
                    var o = op_of[j]
                    try:
                        values[at] = _scorer_side(plan.metric, rows_of[j], y_dst[s], pred_at[t * nops + o],
                                                  side_count[s], k, pred_cols[t * nops + o])
                    except error:
                        failed[at] = UInt8(2)
                        errors[at] = String(error)
        # The error rule, in unit order (then scorer order).
        for t in range(tasks):  # driver-loop: over the wave's tasks (their scores)
            var u = u0 + t // sides
            var side = t % sides
            for j in range(nsc):  # small-loop(the scorers)
                var value = values[t * nsc + j]
                if failed[t * nsc + j] != UInt8(0):
                    if not plan.tolerate:
                        var words = errors[t * nsc + j].copy()
                        member_set_run_release(run)
                        raise Error(words)
                    value = plan.error_value
                if side == 0:
                    test[j * units + u] = value
                else:
                    train[j * units + u] = value
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
        _ = len(preds)  # the response buffers were read through the tasks above
        _ = len(pred_cols)
        _ = len(pred_cap)
        _ = len(values)
        _ = len(failed)
        _ = len(errors)
        _ = len(x_at)
        _ = len(pred_at)
        u0 = u1
    _ = len(out_dtypes)
    _ = len(ops)
    _ = len(side_x)  # the blocks were read through every wave above
    _ = len(side_y)
    _ = len(y_dst)
    _ = len(side_rows)
    _ = len(side_count)
    _ = len(fold_rows)
    _ = len(fold_counts)
    _ = len(columns)
    # 5. The tables, per scorer; the best by the refit scorer.
    if plan.refit_scorer < 0 or plan.refit_scorer >= nsc:
        raise Error("member ABI: the refit scorer is not one of the search's scorers")
    for j in range(nsc):  # driver-loop: over the scorers (their tables)
        var rows_test = List[List[Float64]](capacity=m)
        var rows_train = List[List[Float64]](capacity=m)
        for c in range(m):  # driver-loop: over the candidates (their fold scores)
            var row = List[Float64](capacity=used)
            var row_train = List[Float64](capacity=used)
            for f in range(used):  # small-loop(one candidate's folds)
                row.append(test[j * units + c * used + f])
                row_train.append(train[j * units + c * used + f])
            rows_test.append(row^)
            rows_train.append(row_train^)
        var means = msel_row_means(rows_test)
        res.scorer_stds.append(msel_row_stds(rows_test, means))
        res.scorer_ranks.append(msel_rank_scores(means))
        if plan.return_train:
            res.scorer_train_means.append(msel_row_means(rows_train))
        else:
            res.scorer_train_means.append(List[Float64]())
        res.scorer_means.append(means^)
        res.scorer_test.append(rows_test^)
        res.scorer_train.append(rows_train^)
    var r = plan.refit_scorer
    res.test = res.scorer_test[r].copy()
    res.train = res.scorer_train[r].copy()
    res.means = res.scorer_means[r].copy()
    res.stds = res.scorer_stds[r].copy()
    res.ranks = res.scorer_ranks[r].copy()
    res.train_means = res.scorer_train_means[r].copy()
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
