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
THE RESIDENT ROUTE (lane mf-search5; `msel_search_rows`): when the caller
hands a resident scope and X's pin (`scope`, `pin`: the search's own fit door
opens the scope the old DeviceRows units opened, and the boundary reads it
with the search's own `device`), a candidate whose member reads row numbers
(`MEMBER_CAN_ROW_INDEX`) takes EVERY side, the fit's and each score's, as the
whole X plus the side's int32 rows: no host gather, and its device gathers
the rows out of one resident base (the old route's device take). The pin is
offered on those calls only for the candidates of ONE image (the first
candidate that keeps X resident, `MEMBER_CAN_SCOPE`, and every candidate of
the same family, kind and route): their calls share one scope cache, which
publishes X at most once, so the caller's ONE reference covers the whole
fit. The moment a cache takes it, the RUN writes the token at `kept_at`
(THE PIN RULE, core/member_abi.mojo). Without a scope every side is the
gathered block, as before.

A CLUSTERER BASE (lane mf-cluster5): a non-classifier search whose scorers
are all clustering scorers (core/msel_scorer_run.mojo `msel_scorer_cluster`)
folds by KFold as any non-classifier, asks its members' predict as cluster
ids (int32), and scores them against the targets' float32 words of integer
labels (the label-pair scorers through the metrics image's cluster entry,
`cluster`, resolved by the UNPACK).

The fit notices of every unit (in unit order) and of the refit are handed
over for the caller's PACK; each unit's words hold, after its fit notices,
the warnings its scores' public metrics raise (lane mf-score5: test side,
then train side, in the scorers' order; core/member_notices.mojo
MEMBER_NOTICE_METRIC_*).

THE CV TOOLS (lane mf-msel5): cross_validate, cross_val_score,
learning_curve and cross_val_predict are this RUN over ONE candidate (the
door's carrier search, cv = the door's own folds as given folds;
learning_curve's are its size-major (train[:size], test) units). The RUN
then also answers, per unit, its fit's and its test scores' wall-clock
seconds (`fit_seconds`, `score_seconds`: scikit-learn's `_fit_and_score`
clocks: the fit, then the test side's scoring only); with `keep` every
unit's fitted state, OWNED by the caller (`unit_states`; return_estimator;
a kind that borrows its fit buffers owns its rows first, MEMBER_EXPORT_OWN,
since its gathered train block goes with the wave); with `indices` each fold's train and test rows
(`fold_train`, `fold_test`); with `respond` (a response op, -1: none) no
scorer at all: each unit's test-side response (cross_val_predict's method)
written straight into the caller's result buffer at the unit's test rows
(`respond_dst`, lane mf-msel5b: `_place_response`, scikit-learn's placement
and its `_enforce_prediction_order` for a fold that lacks a class; no copy
between the member's answer and the result). Given folds whose test rows are
the same rows (a learning curve's sizes of one fold) share ONE gathered test
block (`_same_rows`), as the door's units reused it; a side that is one run
of consecutive rows is a view of X (no gather; the old MSEL_CV_SLICE_BIT
route's views).

FOLDS THAT LACK A CLASS and FITS WITHOUT A TARGET (lane mf-msel5b): the
section notes at their code in `msel_search_run`.

SERIAL BY NATURE: the waves follow one another (each bounds the gathered
blocks); inside a wave every unit is independent and runs on the workers.
"""
from bindings.native_unit_os_threads_run import native_unit_cores
from bindings.arena_prog_run import MetricRoute, metric_inputs_run
from bindings.msel_search_stats_run import msel_row_means, msel_row_stds, msel_rank_scores, msel_rank_best
from core.entry_table import EntryRef, entry_ref_unresolved
from core.host_parallel import host_parallelize
from core.msel_scorer_run import (
    MselScorer, MselScorerOptions, msel_scorer_row, msel_scorer_value, msel_scorer_pos_words, msel_scorer_regression, msel_scorer_binary_column,
    msel_scorer_float_labels, msel_scorer_float_proba,
    msel_scorer_positive, msel_scorer_ap, msel_scorer_cluster, msel_scorer_cluster_entry,
    msel_scorer_labels_from_truth, msel_scorer_log_loss, msel_scorer_brier,
    SCORER_MEMBER, SCORER_ACCURACY, SCORER_POSITIVE_ABSENT, SCORER_POSITIVE_KIND, SCORER_POSITIVE_MIXED,
    SCORER_MIXED_WORDS, SCORER_NUMBER_WORDS,
)
from core.member_abi import (
    MemberPlan, MemberCall, MemberMatrix, MemberRef, member_call, member_call_of, member_outputs_of, member_free, member_own,
    MEMBER_F32, MEMBER_F64, MEMBER_I32, MEMBER_I64, MEMBER_OP_SCORE, MEMBER_OP_PREDICT, member_output_dtype,
    MEMBER_CAN_TEST_ROWS, MEMBER_BORROWS_FIT, MEMBER_GIL_FREE, MEMBER_GIL_DOOR, MEMBER_CAN_ROW_INDEX,
    MEMBER_CAN_SCOPE, MEMBER_SCOPE_NONE, member_pin_settle, MEMBER_OP_DECISION, MEMBER_READS_Y_GIVEN,
    MEMBER_OP_PREDICT_PROBA, member_outputs,
)
from core.member_class_weight import member_cw_subset
from core.member_notices import (
    member_notices_take, member_notice_add, MEMBER_NOTICE_FOLD_LACKING, MEMBER_LACKING_STRATIFY,
)
from checks.numerics import identical_log, identical_log64
from std.time import perf_counter_ns
from core.member_set_run import (
    MemberSetRun, member_set_prepare, member_set_fit_free, member_set_turn, member_set_run_release,
    member_set_first_failure, MEMBER_SET_WORKER, MEMBER_SET_TURN_RELEASED, MEMBER_SET_TURN_HELD, MEMBER_SET_LACKING,
)
from xtrees.glue_calls import xt_device_folds, xt_gather_f32

#: `MselSearchPlan.scoring`: the member's own score (scoring=None), or a
#: scorer code of core/msel_scorer_run.mojo ('accuracy' is 1).
comptime SEARCH_SCORE_MEMBER = SCORER_MEMBER
comptime SEARCH_SCORE_ACCURACY = SCORER_ACCURACY

#: `msel_search_rows` per candidate (THE RESIDENT ROUTE, module note): every
#: side a gathered host block; every side the whole X plus its rows (no pin);
#: the same with X's pin offered (the one image that keeps X resident).
comptime SEARCH_ROWS_GATHERED = 0
comptime SEARCH_ROWS_INDEXED = 1
comptime SEARCH_ROWS_RESIDENT = 2

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _RunPtr = MutPointer[MemberSetRun, MutUntrackedOrigin]
comptime _StrPtr = MutPointer[String, MutUntrackedOrigin]
comptime _IntP = MutPointer[Int, MutUntrackedOrigin]
comptime _U8P = MutPointer[UInt8, MutUntrackedOrigin]


#: Lane mf-rows8b (row :3484): a candidate its UNPACK refused (a nested
#: meta-estimator whose admission raised) with a numeric error_score: its
#: mask carries (label word index + 1) << SEARCH_REFUSED_SHIFT, the word its
#: refusal (`label_words`); its plan is another candidate's, never fitted.
#: Each of its units fails with those words before any fit (scored
#: error_score, as scikit-learn's failing fits are); a refit of it raises them.
comptime SEARCH_REFUSED_SHIFT = 48


struct MselOtherTier(Copyable, Movable, Defaultable):
    """LANE mf-r9d (door :3789): THE OTHER numeric_mode TIER of a search's
    scorers. A make_scorer whose `numeric_mode` names the tier that is not
    the process default runs its metric on that tier's x_metrics image, as
    the native scorer door does (bindings/msel_scorer_native_door.mojo
    `MselMetricPrepared.prepare`: `family_msel_metric_bindings[mode]`). The
    UNPACK resolves that image's program runner ONCE (`route`, `resolved`;
    its connection pinned with the plan entry), its cluster label-pair entry
    when a label-pair scorer of that tier is asked (`cluster`), and, per
    candidate, whether its estimator takes its fold rows on the host with the
    fast tier asked (`cpu_fast`: 1 = each of that candidate's other-tier
    sides raises the shared device='cpu' refusal, as the door's prepare does
    per score). A scorer of this tier says so in its options word group
    (core/msel_scorer_run.mojo `MselScorerOptions.tier`)."""
    var route: MetricRoute
    var resolved: Bool
    var cluster: EntryRef
    var cpu_fast: List[UInt8]

    def __init__(out self):
        self.route = MetricRoute(entry_ref_unresolved().call, False, False)
        self.resolved = False
        self.cluster = entry_ref_unresolved()
        self.cpu_fast = List[UInt8]()


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
    # THE POSITIVE CLASS (lane mf-search4b item 3): per scorer (the one
    # `scoring`, or each of `scorers`), the class code of its pos_label
    # among the fit's labels, mapped at UNPACK (-1: the scorer reads none).
    var positives: List[Int]
    # THE SCORERS' OPTIONS (lane mf-tail5b; core/msel_scorer_run.mojo module
    # note): per scorer a make_scorer's kwargs (empty: every row's own), and
    # the `labels` codes they index (int32, kept by the plan through the RUN).
    var options: List[MselScorerOptions]
    var selections: List[Int32]
    # THE GIVEN FOLDS (lane mf-search4b item 6): one of our splitters' folds
    # or a user's index list, as int32 row views the caller keeps alive
    # (empty: the int cv's KFold / StratifiedKFold below).
    var fold_train_at: List[Int]
    var fold_train_n: List[Int]
    var fold_test_at: List[Int]
    var fold_test_n: List[Int]
    # THE RESIDENT ROUTE (lane mf-search5, module note): the caller's scope
    # handle word (MEMBER_SCOPE_NONE: none), X's pin token (0: none; ONE
    # reference the caller took) and the address of the Int the RUN writes
    # the token into when a cache takes that reference (0: nowhere).
    var scope: Int
    var pin: Int
    var kept_at: Int
    # Lane mf-cluster5: the metrics image's cluster label-pair entry
    # (core/metrics_cluster_table.mojo; resolved by the UNPACK when a
    # label-pair scorer is asked, else unresolved) and the fit's labels'
    # reprs by class code (the binary scorers' refusal words).
    var cluster: EntryRef
    var label_words: List[String]
    # THE CV TOOLS (lane mf-msel5, module note): keep every unit's fitted
    # state for the caller; answer each fold's rows; answer each unit's
    # test-side response `respond` (a MEMBER_OP_*; -1: score as above).
    var keep: Bool
    var indices: Bool
    var respond: Int
    # THE CV TOOLS' FINAL BUFFER (lane mf-msel5b): with `respond`, each
    # unit's test-side response is written straight into the caller's
    # result, `respond_dst` (rows x `respond_width` 8-byte words of
    # `respond_dtype`, MEMBER_I64 or MEMBER_F64), at the unit's test rows
    # (cross_val_predict's placement); `respond_log`: predict_log_proba (the
    # log of each probability, in the member's own precision).
    var respond_dst: Int
    var respond_width: Int
    var respond_dtype: Int
    var respond_log: Bool
    # THE GIVEN y (lane mf-tail5; core/member_abi.mojo `MemberCall.y_given`):
    # a regressor's targets as the caller was given them (`given_dtype`:
    # MEMBER_F64 / _I64 / _I32 words, rows x `y_cols`), 0: none. Handed to
    # each unit's and the refit's fit whose candidate kind reads it
    # (MEMBER_READS_Y_GIVEN), with the fold's rows, as `y` is.
    var given: Int
    var given_dtype: Int
    # LANE mf-r9d: the other numeric_mode tier's metric route (`MselOtherTier`;
    # unresolved: no scorer of that tier)
    var other: MselOtherTier

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
        self.positives = List[Int]()
        self.options = List[MselScorerOptions]()
        self.selections = List[Int32]()
        self.fold_train_at = List[Int]()
        self.fold_train_n = List[Int]()
        self.fold_test_at = List[Int]()
        self.fold_test_n = List[Int]()
        self.scope = MEMBER_SCOPE_NONE
        self.pin = 0
        self.kept_at = 0
        self.cluster = entry_ref_unresolved()
        self.label_words = List[String]()
        self.keep = False
        self.indices = False
        self.respond = -1
        self.respond_dst = 0
        self.respond_width = 1
        self.respond_dtype = MEMBER_F64
        self.respond_log = False
        self.given = 0
        self.given_dtype = MEMBER_F64
        self.other = MselOtherTier()


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
    # THE CV TOOLS (lane mf-msel5, module note), per unit (candidate-major,
    # u = c * folds + f): its fit's and its test scores' seconds; with
    # `keep` its fitted state (0: its fit failed), owned by the caller (with
    # `respond` the responses are in the caller's own buffer, lane
    # mf-msel5b). Per fold (with `indices`): its train and test rows.
    var fit_seconds: List[Float64]
    var score_seconds: List[Float64]
    var unit_states: List[Int]
    var fold_train: List[List[Int32]]
    var fold_test: List[List[Int32]]

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
        self.fit_seconds = List[Float64]()
        self.score_seconds = List[Float64]()
        self.unit_states = List[Int]()
        self.fold_train = List[List[Int32]]()
        self.fold_test = List[List[Int32]]()


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


@fieldwise_init
struct _Side(ImplicitlyCopyable, Movable):
    """Where one side's rows are (THE RESIDENT ROUTE, module note): `index`
    0, the gathered block `x` / `y` of `count` rows; else the whole X
    (`base_x`, `base_rows` rows) and target (`base_y`) with the side's
    `count` int32 row numbers at `index`, X's pin `pin` offered (0: none) in
    the caller's `scope`. `took`: the address of the byte written 1 when a
    cache took the pin's reference."""
    var x: Int
    var y: Int
    var count: Int
    var index: Int
    var base_x: Int
    var base_y: Int
    var base_rows: Int
    var scope: Int
    var pin: Int
    var took: Int


def _side_set(mut call: MemberCall, side: _Side, cols: Int, y_dtype: Int, y_cols: Int, with_y: Bool):
    """Point `call` at one side (`_Side`); the pin is offered only on a
    resident side."""
    call.scope = side.scope
    call.pin = 0
    # A fit without a target (lane mf-msel5b: an unsupervised base, y=None):
    # the call carries no y.
    if side.index == 0:
        call.x = MemberMatrix(side.x, side.count, cols, MEMBER_F32)
        if with_y and side.y != 0:
            call.y = MemberMatrix(side.y, side.count, y_cols, y_dtype)
        return
    call.x = MemberMatrix(side.base_x, side.base_rows, cols, MEMBER_F32)
    call.row_index = MemberMatrix(side.index, side.count, 1, MEMBER_I32)
    if with_y and side.base_y != 0:
        call.y = MemberMatrix(side.base_y, side.base_rows, y_cols, y_dtype)
    call.pin = side.pin


def _offered(member: MemberRef, op: Int, mut call: MemberCall, side: _Side) raises:
    """One member call over the side `call` points at; the member's answer
    about an offered pin is read whether the call returns or raises (THE PIN
    RULE)."""
    if call.pin == 0:
        _ = member_call(member, op, call)
        return
    try:
        _ = member_call(member, op, call)
    finally:
        if member_pin_settle(call) != 0:
            _U8P(unsafe_from_address=side.took)[0] = UInt8(1)


def _member_score(run_at: Int, v: Int, side: _Side, cols: Int, y_dtype: Int, y_cols: Int) raises -> Float64:
    """Unit `v` of the wave on one side (`_Side`): the member's own score
    (scoring None)."""
    var run = _RunPtr(unsafe_from_address=run_at)
    var call = member_call_of(run[].plans[v], run[].states[v])
    _side_set(call, side, cols, y_dtype, y_cols, True)
    _offered(run[].plans[v].member, MEMBER_OP_SCORE, call, side)
    return call.value


def _respond(run_at: Int, v: Int, op: Int, side: _Side, cols: Int, dst: Int, capacity: Int,
             dtype: Int, narrow: Bool, predict_cols: Int) raises -> Int:
    """`_respond` of a response that is not predict_log_proba."""
    return _respond(run_at, v, op, side, cols, dst, capacity, dtype, narrow, predict_cols, False)


def _respond(run_at: Int, v: Int, op: Int, side: _Side, cols: Int, dst: Int, capacity: Int,
             dtype: Int, narrow: Bool, predict_cols: Int, log: Bool) raises -> Int:
    """Unit `v`'s response `op` (predict, predict_proba or decision_function)
    on one side (`_Side`), written in the member's own `dtype` into
    the 8-byte words at `dst` (`capacity` words), then narrowed in place to
    the scorer's words: int32 codes (an int64 predict) or float32 (a float64
    output; the rounding the metric doors' admission applies) unless
    `narrow` is False (THE CV TOOLS' `respond`: the member's own words).
    Answers the output's columns per row (a predict: `predict_cols`, one
    per output of a multi-output member)."""
    var run = _RunPtr(unsafe_from_address=run_at)
    var call = member_call_of(run[].plans[v], run[].states[v])
    var count = side.count
    var width = max(predict_cols, 1)
    if op != MEMBER_OP_PREDICT:
        width = max(member_outputs_of(run[].plans[v], call, op), 1)
    if count * width > capacity:
        raise Error("member ABI: a member response wider than its classes")
    _side_set(call, side, cols, MEMBER_F32, 1, False)
    call.dst = MemberMatrix(dst, count, width, dtype)
    _offered(run[].plans[v].member, op, call, side)
    var words = count * width
    if log:
        # LANE mf-r9d (door :3789): a scorer whose response is
        # predict_log_proba. scikit-learn 1.9 hands `np.log(predict_proba(X))`
        # to the metric as it hands predict_proba (utils/_response.py lines
        # 217-223: the same `_process_predict_proba`); the log is taken in the
        # member's own precision through the checks/numerics seam (`_log_word`,
        # as the cv tools' `respond_log`), BEFORE the narrowing below, which
        # then rounds it as the metric doors' admission rounds a float64.
        if op != MEMBER_OP_PREDICT_PROBA or (dtype != MEMBER_F32 and dtype != MEMBER_F64):
            raise Error("member ABI: predict_log_proba is the log of a float predict_proba")
        if dtype == MEMBER_F64:
            var p64 = _F64P(unsafe_from_address=dst)
            for i in range(words):  # driver-loop: over the side's probabilities (their logs, float64)
                p64[i] = _log_word(p64[i], MEMBER_F64)
        else:
            var p32 = _F32P(unsafe_from_address=dst)
            for i in range(words):  # driver-loop: over the side's probabilities (their logs, float32)
                p32[i] = Float32(_log_word(Float64(p32[i]), MEMBER_F32))
    if not narrow:
        return width
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


def _held_words(label_words: List[String], held: Int, held_n: Int) -> String:
    """A fold estimator's `classes_` as scikit-learn's messages print a NumPy
    array: the labels' reprs, space-separated, in brackets."""
    var h = _I32P(unsafe_from_address=held)
    var shown = String("[")
    for j in range(held_n):  # small-loop(the fold's held classes)
        if j > 0:
            shown += " "
        var c = Int(h[j])
        if c < len(label_words):
            shown += label_words[c]
        else:
            shown += String(c)
    return shown + "]"


def _array_words(label_words: List[String], seen: List[Bool]) -> String:
    """The labels `seen` marks as a NumPy array prints them in scikit-learn's
    messages (`np.unique(y_true)`): reprs, space-separated, in brackets."""
    var out = String("[")
    var first = True
    for c in range(len(seen)):  # small-loop(the class codes)
        if not seen[c]:
            continue
        if not first:
            out += " "
        first = False
        out += label_words[c] if c < len(label_words) else String(c)
    return out + "]"


def _int_word(word: String) -> Bool:
    """Lane mf-rows8a: a label word that is an integer's repr (digits, an
    optional leading '-')."""
    var bytes = word.as_bytes()
    if len(bytes) == 0:
        return False
    for i in range(len(bytes)):  # small-loop(the word's bytes)
        var c = Int(bytes[i])
        if not ((c >= 48 and c <= 57) or (c == 45 and i == 0 and len(bytes) > 1)):
            return False
    return True


def _classes_words(label_words: List[String], classes: Int) -> String:
    """Lane mf-rows8a: the fit's `classes` labels (codes 0 .. classes-1) as
    `f"{classes}"` prints a classifier's `classes_` array: space-separated
    in brackets; an integer array's entries right-aligned to the widest, as
    NumPy pads them (a string or object array is not padded)."""
    var n = min(classes, len(label_words))
    var ints = n > 0
    var width = 0
    for c in range(n):  # small-loop(the fit's labels)
        ints = ints and _int_word(label_words[c])
        width = max(width, label_words[c].byte_length())
    var out = String("[")
    for c in range(n):  # small-loop(the fit's labels)
        if c > 0:
            out += " "
        if ints:
            for _ in range(width - label_words[c].byte_length()):  # small-loop(the padding)
                out += " "
        out += label_words[c]
    return out + "]"


def _plain_word(word: String) -> String:
    """A label's repr as `str` of a NumPy scalar prints it (`'{0}'.format`):
    a string label without its quotes."""
    var n = word.byte_length()
    if n >= 2 and word.startswith("'") and word.endswith("'"):
        return String(word[byte=1 : n - 1])
    return word


def _fold_truth_checks(label_words: List[String], row: MselScorer, truth: Int, count: Int, classes: Int,
                       view_cols: Int, positive: Int) raises:
    """A FOLD THAT LACKS A CLASS scored by 'average_precision' or
    'neg_brier_score' (lane mf-msel5c): scikit-learn's checks of the side's
    `y_true` (its `classes` ensemble codes; `present` of them on this side)
    against the `view_cols` columns the fold estimator handed, in their
    words (scikit-learn 1.9).
    average_precision_score (metrics/_ranking.py): more than two present
    labels need y_score of shape (n_samples, present) (`label_binarize` over
    the present labels); at most two take the binary branch, which refuses a
    pos_label=1 not among two present labels, then `column_or_1d(y_score)`.
    brier_score_loss (metrics/_classification.py): one column is the binary
    branch (its run raises those words); several are
    `_validate_multiclass_probabilistic_prediction` with labels=None, which
    refuses one present label and a present count other than the columns."""
    var seen = List[Bool](length=max(classes, 1), fill=False)
    var t = _I32P(unsafe_from_address=truth)
    var present = 0
    for i in range(count):  # driver-loop: over the side's true codes (the labels it holds)
        var c = Int(t[i])
        if c >= 0 and c < classes and not seen[c]:
            seen[c] = True
            present += 1
    if msel_scorer_ap(row):
        if present > 2:
            if view_cols != present:
                var shape = (String("(") + String(count) + ",)") if view_cols == 1 else _shape_words(count, view_cols)
                raise Error("ValueError: `y_score` needs to be of shape `(n_samples, n_classes)`, since `y_true` "
                            + "contains multiple classes. Got `y_score.shape=" + shape + "`.")
            return
        if present == 2 and (positive < 0 or positive >= classes or not seen[positive]):
            raise Error("ValueError: pos_label=" + msel_scorer_pos_words(row, label_words, positive)
                        + " is not a valid label. It should be one of " + _array_words(label_words, seen))
        if view_cols != 1:
            raise Error("ValueError: y should be a 1d array, got an array of shape " + _shape_words(count, view_cols)
                        + " instead.")
        return
    if view_cols == 1:
        return
    if present == 1:
        var only = String("")
        for c in range(classes):  # small-loop(the one class present)
            if seen[c]:
                only = _plain_word(label_words[c] if c < len(label_words) else String(c))
        raise Error("ValueError: y_true contains only one label (" + only + "). Please provide the list of all "
                    + "expected class labels explicitly through the labels argument.")
    if present != view_cols:
        raise Error("ValueError: y_true and y_prob contain different number of classes: " + String(present) + " vs "
                    + String(view_cols) + ". Please provide the true labels explicitly through the labels argument. "
                    + "Classes found in y_true: " + _array_words(label_words, seen))


def _fold_view_side(route: MetricRoute, cluster: EntryRef, label_words: List[String], row: MselScorer, truth: Int,
                    response: Int, count: Int, classes: Int, cols: Int, positive: Int, held: Int, held_n: Int,
                    mut notices: List[Float64]) raises -> Float64:
    """THE FOLD THAT LACKS A CLASS, scored by probabilities or decisions
    (lane mf-kmeans5): what scikit-learn's scorer hands the metric, from the
    FOLD ESTIMATOR's own `classes_` (the `held_n` held classes, ascending
    ensemble codes at `held`) and its own `cols` columns, never scattered into
    the ensemble's `classes`. `_get_response_values` (sklearn/utils/
    _response.py): a pos_label outside the fold's classes is refused in its
    words; a fold of at most two classes is a BINARY estimator: its
    predict_proba of one column is refused ("need classifier with two
    classes"), of two columns gives the column of the positive (classes[-1]
    without one), and its 1-D decision is negated when the positive is
    classes[0] (`_process_decision_function`). Then the metric: one whose
    labels come from the side's `y_true` (`msel_scorer_labels_from_truth`:
    roc_auc, log_loss, top_k) reads the side's codes ranked among the labels
    the side holds, with that label count; log_loss with a different count
    raises its words. 'average_precision' and 'neg_brier_score' (lane
    mf-msel5c, `_fold_truth_checks`) first make scikit-learn's checks of
    the side's `y_true` against the columns handed, in its words; then their
    metric reads the ensemble's codes (its runs rank the classes the side
    holds, as `label_binarize(y_true, classes=present)` and
    `LabelBinarizer().fit(y_true)` do)."""
    var h = _I32P(unsafe_from_address=held)
    var pos_col = held_n - 1  # pos_label None: classes[-1]
    if positive >= 0 or positive == SCORER_POSITIVE_ABSENT or positive == SCORER_POSITIVE_KIND:
        var at = -1
        for j in range(held_n):  # small-loop(the fold's held classes)
            if Int(h[j]) == positive:
                at = j
        # `_get_response_values` checks pos_label for a BINARY estimator only
        # (`type_of_target(classes) == "binary"`: at most two held classes).
        # lane mf-rows8a: a brier / D^2 brier scorer's given pos_label too
        if (msel_scorer_ap(row) or msel_scorer_positive(row) or (msel_scorer_brier(row) and row.pos_given)) and (
                held_n <= 2):
            if at < 0:
                raise Error("ValueError: pos_label=" + msel_scorer_pos_words(row, label_words, positive)
                            + " is not a valid label: It should be one of " + _held_words(label_words, held, held_n))
            pos_col = at
    var src = _F32P(unsafe_from_address=response)
    var column = List[Float32]()
    var view = response
    var view_cols = cols
    if held_n <= 2:
        if row.op == MEMBER_OP_PREDICT_PROBA:
            if cols < 2:
                raise Error("ValueError: Got predict_proba of shape (" + String(count) + ", " + String(cols)
                            + "), but need classifier with two classes.")
            column = List[Float32](length=max(count, 1), fill=0.0)
            for i in range(count):  # driver-loop: over the side's rows (the positive's probability column)
                column[i] = src[cols * i + pos_col]
            view = Int(column.unsafe_ptr())
            view_cols = 1
        elif pos_col == 0 and cols == 1:
            column = List[Float32](length=max(count, 1), fill=0.0)
            for i in range(count):  # driver-loop: over the side's rows (the decision of the first class)
                column[i] = -src[i]
            view = Int(column.unsafe_ptr())
    if msel_scorer_ap(row) or msel_scorer_brier(row):
        _fold_truth_checks(label_words, row, truth, count, classes, view_cols, positive)
    var labels_at = truth
    var label_count = classes
    var ranked = List[Int32]()
    if msel_scorer_labels_from_truth(row):
        var seen = List[Int32](length=max(classes, 1), fill=Int32(-1))
        var t = _I32P(unsafe_from_address=truth)
        for i in range(count):  # driver-loop: over the side's true codes (the labels it holds)
            var c = Int(t[i])
            if c >= 0 and c < classes:
                seen[c] = Int32(0)
        label_count = 0
        for c in range(classes):  # small-loop(the classes: each held one's rank)
            if seen[c] == Int32(0):
                seen[c] = Int32(label_count)
                label_count += 1
        ranked = List[Int32](length=max(count, 1), fill=Int32(0))
        for i in range(count):  # driver-loop: over the side's true codes (their ranks)
            var c = Int(t[i])
            ranked[i] = seen[c] if (c >= 0 and c < classes) else Int32(c)
        labels_at = Int(ranked.unsafe_ptr())
        var predicted = 2 if view_cols == 1 else view_cols
        if msel_scorer_log_loss(row) and label_count != predicted:
            var present = String("[")
            var first = True
            for c in range(classes):  # small-loop(the classes the side holds, in order)
                if seen[c] >= Int32(0):
                    if not first:
                        present += " "
                    if c < len(label_words):
                        present += label_words[c]
                    else:
                        present += String(c)
                    first = False
            raise Error("ValueError: y_true and y_pred contain different number of classes " + String(label_count)
                        + ", " + String(predicted) + ". Please provide the true labels explicitly through the "
                        + "labels argument. Classes found in y_true: " + present + "]")
    var value = msel_scorer_value(route, cluster, label_words, row, labels_at, view, count, label_count, view_cols,
                                  -1 if msel_scorer_labels_from_truth(row) else positive, notices)
    _ = len(column)  # read by the program above
    _ = len(ranked)
    return value


def _scorer_side(route: MetricRoute, cluster: EntryRef, label_words: List[String], row: MselScorer, truth: Int,
                 response: Int, count: Int, classes: Int, cols: Int, positive: Int, held: Int, held_n: Int,
                 mut notices: List[Float64]) raises -> Float64:
    """One scorer on one side: the `_BINARY_PROBA` column pick of a
    two-class member's probabilities as the scorer hands it
    (`_get_response_values`: `predict_proba[:, 1]`, or the positive's column
    for 'average_precision', whose decision values are negated when its
    positive is the first class; lane mf-score5), then
    core/msel_scorer_run.mojo `msel_scorer_value` (its metric warnings to
    `notices`). The member's response buffer is shared by the scorers that
    read it: a pick is a copy."""
    var pick = -1
    var negate = False
    if positive == SCORER_POSITIVE_MIXED:
        # MIXED-KIND LABELS (lane mf-msel5b): the metric doors' refusal of a
        # target holding strings and integers, before any pos_label check.
        raise Error(SCORER_MIXED_WORDS)
    if held_n > 0 and row.op != MEMBER_OP_PREDICT:
        # a fold estimator holding `held_n` of the classes (lane mf-kmeans5)
        return _fold_view_side(route, cluster, label_words, row, truth, response, count, classes, cols, positive,
                               held, held_n, notices)
    if msel_scorer_ap(row) and classes == 2 and (positive == SCORER_POSITIVE_ABSENT
                                                 or positive == SCORER_POSITIVE_KIND):
        # AP-ABSENT (lane mf-msel5): two labels and no label 1. scikit-learn's
        # scorer asks `_get_response_values(..., pos_label=1)`, which refuses a
        # pos_label not among the classifier's classes, printing them as a
        # NumPy array: the labels' reprs, space-separated, in brackets (lane
        # mf-rows8a: the fit's labels only, `_classes_words`).
        raise Error("ValueError: pos_label=" + msel_scorer_pos_words(row, label_words, positive)
                    + " is not a valid label: It should be one of " + _classes_words(label_words, classes))
    if msel_scorer_binary_column(row) and classes == 2 and cols == 2:
        pick = 1
        if msel_scorer_ap(row) or (msel_scorer_brier(row) and row.pos_given):
            # the positive's column (lane mf-tail5b: a given pos_label on brier too)
            if positive < 0 or positive > 1:
                # lane mf-rows8a: a given pos_label that is not one of the
                # binary member's classes (SCORER_POSITIVE_ABSENT): scikit-learn's
                # `_get_response_values` words (sklearn 1.9 utils/_response.py
                # lines 206-210), raised before any metric
                raise Error("ValueError: pos_label=" + msel_scorer_pos_words(row, label_words, positive)
                            + " is not a valid label: It should be one of " + _classes_words(label_words, classes))
            pick = positive
    elif msel_scorer_ap(row) and classes == 2 and cols == 1 and positive == 0:
        negate = True
    if pick >= 0 or negate:
        var column = List[Float32](length=max(count, 1), fill=0.0)
        var src = _F32P(unsafe_from_address=response)
        if negate:
            for i in range(count):  # driver-loop: over the side's rows (the decision of the first class)
                column[i] = -src[i]
        else:
            for i in range(count):  # driver-loop: over the side's rows (one column of the probabilities)
                column[i] = src[2 * i + pick]
        var value = msel_scorer_value(route, cluster, label_words, row, truth, Int(column.unsafe_ptr()), count,
                                      classes, 1, positive, notices)
        _ = len(column)  # read by the program above
        return value
    return msel_scorer_value(route, cluster, label_words, row, truth, response, count, classes, cols, positive,
                             notices)


def _same_image(a: MemberPlan, b: MemberPlan) -> Bool:
    """Two candidates run on one image of one family's kind (the same
    member entry and route words): their calls share one scope cache."""
    return (a.family == b.family and a.kind == b.kind and a.route.device == b.route.device
            and a.route.mode == b.route.mode and a.route.vendor == b.route.vendor)


def msel_search_rows(plan: MselSearchPlan) -> List[Int]:
    """THE RESIDENT ROUTE per candidate (module note), one
    `SEARCH_ROWS_*` word each, decided once per fit: GATHERED without a
    scope and pin or for a member that reads no row numbers; INDEXED for one
    that does (its kind's mask and its entry's answer both); RESIDENT for
    those of them on the image of the first candidate that keeps X resident
    (`MEMBER_CAN_SCOPE`): only they are offered X's pin."""
    var m = len(plan.plans)
    var out = List[Int](length=max(m, 1), fill=SEARCH_ROWS_GATHERED)
    if plan.scope == MEMBER_SCOPE_NONE or plan.pin == 0 or len(plan.masks) != m:
        return out^
    var first = -1
    for c in range(m):  # driver-loop: over the candidates (one route word each, once per fit)
        var mask = plan.masks[c]
        if (mask & MEMBER_CAN_ROW_INDEX) == 0 or not plan.plans[c].member.can(MEMBER_CAN_ROW_INDEX):
            continue
        out[c] = SEARCH_ROWS_INDEXED
        if (mask & MEMBER_CAN_SCOPE) == 0:
            continue
        if first < 0:
            first = c
        if _same_image(plan.plans[c], plan.plans[first]):
            out[c] = SEARCH_ROWS_RESIDENT
    return out^


def _note_kept(plan: MselSearchPlan):
    """A candidate's cache took X's reference: the caller's `pin_kept`
    (THE PIN RULE)."""
    if plan.kept_at != 0:
        _IntP(unsafe_from_address=plan.kept_at)[0] = plan.pin


def _same_rows(a: Int, b: Int, count: Int) -> Bool:
    """Two int32 row views of `count` rows hold the same rows in the same
    order (THE CV TOOLS: one gathered test block per distinct test side)."""
    if a == b:
        return True
    var p = _I32P(unsafe_from_address=a)
    var q = _I32P(unsafe_from_address=b)
    for i in range(count):  # driver-loop: over one test side's rows (stops at the first difference)
        if p[i] != q[i]:
            return False
    return True


def _drop_kept(plan: MselSearchPlan, mut res: MselSearchResult):
    """THE CV TOOLS: release every kept unit state before a raise (the
    caller owns them only when the RUN returns). Never raises."""
    var used = max(res.n_splits, 1)
    for u in range(len(res.unit_states)):  # driver-loop: over the kept units (one release each)
        if res.unit_states[u] != 0:
            var c = u // used
            var call = MemberCall(plan.plans[c].route, res.unit_states[u])
            member_free(plan.plans[c], call)
            res.unit_states[u] = 0


@always_inline
def _word_at(src: Int, i: Int, dtype: Int) -> Float64:
    """Word `i` of a member response written in its own element type (int
    codes and float32 values exact in float64)."""
    if dtype == MEMBER_I64:
        return Float64(MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=src)[i])
    if dtype == MEMBER_I32:
        return Float64(_I32P(unsafe_from_address=src)[i])
    if dtype == MEMBER_F64:
        return _F64P(unsafe_from_address=src)[i]
    return Float64(_F32P(unsafe_from_address=src)[i])


@always_inline
def _log_word(value: Float64, dtype: Int) -> Float64:
    """predict_log_proba's word (`np.log(predict_proba(X))`): the log in the
    member's own precision (a float32 probability's log is a float32),
    through the checks/numerics.mojo seam (cross-vendor bits; was raw libm
    `log`, MATHLINT w3 gate row 8)."""
    if dtype == MEMBER_F32:
        return Float64(identical_log(Float32(value)))
    return identical_log64(value)


@always_inline
def _lowest(dtype: Int) -> Float64:
    """`np.finfo(predictions.dtype).min` of the member's response type."""
    if dtype == MEMBER_F32:
        return Float64(Float32.MIN_FINITE)
    return Float64.MIN_FINITE


def _shape_words(rows: Int, cols: Int) -> String:
    """A NumPy shape as scikit-learn's messages print it."""
    return "(" + String(rows) + ", " + String(cols) + ")"


def _place_response(src: Int, src_dtype: Int, cols: Int, count: Int, rows: Int, dst: Int, width: Int,
                    dst_dtype: Int, op: Int, log_words: Bool, held: Int, m: Int, k: Int) raises:
    """THE CV TOOLS' FINAL BUFFER (lane mf-msel5b): one unit's test-side
    response (`count` rows x `cols` words of `src_dtype` at `src`) written
    straight into the caller's result `dst` (8-byte words of `dst_dtype`,
    `width` per row) at the side's int32 rows `rows`: scikit-learn's
    `cross_val_predict` placement `predictions[inv_test_indices]`, with no
    copy between. A fold whose train rows hold only `m` of the `k` classes
    (`held`: their ensemble codes, ascending; 0: every class) is placed as
    `_enforce_prediction_order` (model_selection/_validation.py, scikit-learn
    1.9): a predict's fold codes mapped to the classes; predict_proba (and
    its log) columns at the held classes, 0 (log: the dtype's lowest) for
    the others; decision_function refused for an irregular width or two
    held classes or fewer, else placed with the dtype's lowest elsewhere."""
    var r = _I32P(unsafe_from_address=rows)
    var whole = held == 0
    var place = cols
    var fill = 0.0
    if not whole and op != MEMBER_OP_PREDICT:
        if op == MEMBER_OP_DECISION:
            if cols != 1 and cols != m:
                raise Error("ValueError: Output shape " + _shape_words(count, cols) + " of decision_function does "
                            + "not match number of classes (" + String(m) + ") in fold. Irregular "
                            + "decision_function outputs are not currently supported by cross_val_predict")
            if m <= 2:
                raise Error("ValueError: Only " + String(m) + " class/es in training fold, but " + String(k)
                            + " in overall dataset. This is not supported for decision_function with imbalanced "
                            + "folds. " + MEMBER_LACKING_STRATIFY)
            fill = _lowest(src_dtype)
        elif log_words:
            fill = _lowest(src_dtype)
        if cols != m or width != k:
            raise Error("member ABI: a fold member holding " + String(m) + " classes answered " + String(cols)
                        + " columns for " + String(k) + " placed")
    elif cols != width:
        raise Error("member ABI: a cv tool response of " + String(cols) + " columns for " + String(width) + " placed")
    var h = _I32P(unsafe_from_address=held)
    var wide = dst_dtype == MEMBER_I64
    var di = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=dst)
    var df = _F64P(unsafe_from_address=dst)
    for i in range(count):  # driver-loop: over the side's rows (each written once, at its own row)
        var at = Int(r[i]) * width
        if not whole and op != MEMBER_OP_PREDICT:
            for c in range(width):
                df[at + c] = fill
        for c in range(place):
            var value = _word_at(src, i * cols + c, src_dtype)
            var to = c
            if not whole:
                if op == MEMBER_OP_PREDICT:
                    value = Float64(h[Int(value)])
                else:
                    to = Int(h[c])
            if log_words:
                value = _log_word(value, src_dtype)
            if wide:
                di[at + to] = Int64(value)
            else:
                df[at + to] = value


def _held_codes(codes: Int, rows: Int, count: Int, k: Int) -> List[Int32]:
    """The classes the `count` int32 rows `rows` of the codes at `codes`
    hold, ascending (empty when they hold all `k`)."""
    return _held_codes_col(codes, rows, count, k, 1, 0)


def _held_codes_col(codes: Int, rows: Int, count: Int, k: Int, width: Int, column: Int) -> List[Int32]:
    """`_held_codes` of output `column` of rows x `width` codes (lane
    mf-tail5: a multi-output classifier's folds, each output on its own; one
    body for both)."""
    var seen = List[UInt8](length=max(k, 1), fill=UInt8(0))
    var y = _I32P(unsafe_from_address=codes)
    var r = _I32P(unsafe_from_address=rows)
    var found = 0
    for i in range(count):  # driver-loop: over a fold's train rows (stops when every class is seen)
        var c = Int(y[Int(r[i]) * width + column])
        if c >= 0 and c < k and seen[c] == UInt8(0):
            seen[c] = UInt8(1)
            found += 1
            if found == k:
                return List[Int32]()
    var held = List[Int32]()
    for c in range(k):  # small-loop(the classes)
        if seen[c] != UInt8(0):
            held.append(Int32(c))
    return held^


def _map_multi_codes(buffer: Int, dtype: Int, count: Int, outputs: Int, held_at: Int, f: Int):
    """A MULTI-OUTPUT CLASSIFIER'S LACKING FOLD (lane mf-tail5): its unit's
    predicted fold codes (`count` rows x `outputs`, int32 or int64 at
    `buffer`) mapped in place, per output, to the classes' codes (output j
    of fold f's held classes, `held_at`'s list f * outputs + j; empty: that
    output holds every class, its codes unchanged), as each output's fold
    estimator predicts its own labels."""
    var lists = MutPointer[List[Int32], MutUntrackedOrigin](unsafe_from_address=held_at)
    for j in range(outputs):  # small-loop(the outputs)
        var h = Int(lists[f * outputs + j].unsafe_ptr())
        var m = len(lists[f * outputs + j])
        if m == 0:
            continue
        var held = _I32P(unsafe_from_address=h)
        for i in range(count):  # driver-loop: over the side's rows (output j's code)
            if dtype == MEMBER_I64:
                var w = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=buffer)
                w[i * outputs + j] = Int64(held[Int(w[i * outputs + j])])
            else:
                var w = _I32P(unsafe_from_address=buffer)
                w[i * outputs + j] = held[Int(w[i * outputs + j])]


def _place_multi_proba(src: Int, src_dtype: Int, cols: Int, count: Int, rows: Int, dsts: Int, outputs: Int,
                       counts_at: Int, held_at: Int, f: Int, log_words: Bool) raises:
    """THE CV TOOLS' PER-OUTPUT BUFFERS (lane mf-tail5): a multi-output
    classifier unit's test-side predict_proba (`count` rows x `cols` words
    of `src_dtype` at `src`: each output's columns in turn, the fold
    estimator's own classes) written into output j's result (the float64
    rows x `counts[j]` block whose address is word j at `dsts`; `counts`
    the outputs' class counts, Int words at `counts_at`) at the
    side's int32 rows: scikit-learn's cross_val_predict of a list response,
    each output placed by `_enforce_prediction_order` over that output's
    classes (a lacking output's columns at its held classes, 0 elsewhere;
    `held_at`: fold f's per-output held lists, 0: none lack). `log_words`
    (lane mf-rows8b): predict_log_proba, each output's `np.log` of its
    block in the member's precision, the dtype's lowest at a lacking
    output's other classes (`_enforce_prediction_order`'s float_min)."""
    var r = _I32P(unsafe_from_address=rows)
    var counts = _IntP(unsafe_from_address=counts_at)
    var addresses = _IntP(unsafe_from_address=dsts)
    var lists = MutPointer[List[Int32], MutUntrackedOrigin](unsafe_from_address=held_at)
    var total = 0
    for j in range(outputs):  # small-loop(the outputs: the member's columns of each)
        var m = len(lists[f * outputs + j]) if held_at != 0 else 0
        total += m if m > 0 else counts[j]
    if total != cols:
        raise Error("member ABI: a multi-output fold member answered " + String(cols) + " probability columns for "
                    + String(total) + " held")
    var offset = 0
    for j in range(outputs):  # small-loop(the outputs: one result block each)
        var k = counts[j]
        var m = len(lists[f * outputs + j]) if held_at != 0 else 0
        var width = m if m > 0 else k
        var held = _I32P(unsafe_from_address=Int(lists[f * outputs + j].unsafe_ptr()) if m > 0 else 0)
        var df = _F64P(unsafe_from_address=addresses[j])
        for i in range(count):  # driver-loop: over the side's rows (each written once, at its own row)
            var at = Int(r[i]) * k
            if m > 0:
                for c in range(k):
                    df[at + c] = _lowest(src_dtype) if log_words else 0.0
            for c in range(width):
                var to = Int(held[c]) if m > 0 else c
                var value = _word_at(src, i * cols + offset + c, src_dtype)
                df[at + to] = _log_word(value, src_dtype) if log_words else value
        offset += width


def _multi_lacking_score(run_at: Int, v: Int, side: _Side, cols: Int, outputs: Int, held_at: Int, f: Int) raises -> Float64:
    """A MULTI-OUTPUT CLASSIFIER'S OWN SCORE on a fold that lacks a class in
    some output (lane mf-tail5): `MultiOutputClassifier.score`,
    `np.mean(np.all(y == y_pred, axis=1))`, of the unit's predict mapped
    back to the classes' codes per output (`_map_multi_codes`), against the
    side's codes (rows x `outputs`)."""
    var count = side.count
    var buffer = List[Int32](length=max(count * outputs, 1), fill=Int32(0))
    _ = _respond(run_at, v, MEMBER_OP_PREDICT, side, cols, Int(buffer.unsafe_ptr()), count * outputs, MEMBER_I32,
                 False, outputs)
    _map_multi_codes(Int(buffer.unsafe_ptr()), MEMBER_I32, count, outputs, held_at, f)
    var ix = _I32P(unsafe_from_address=side.index)
    var hits = 0
    for i in range(count):  # driver-loop: over the side's rows (all outputs equal)
        var row = Int(ix[i]) if side.index != 0 else i
        var src = _I32P(unsafe_from_address=side.y if side.index == 0 else side.base_y)
        var equal = True
        for j in range(outputs):
            if buffer[i * outputs + j] != src[row * outputs + j]:
                equal = False
        if equal:
            hits += 1
    return Float64(hits) / Float64(count) if count > 0 else 0.0


@always_inline
def _run_start(rows: Int, count: Int) -> Int:
    """The first row when the int32 rows at `rows` are one ascending run of
    consecutive rows (the side is then a view of X), else -1 (lane
    mf-msel5b: the slice route's views, for every fold source)."""
    if count <= 0:
        return -1
    var r = _I32P(unsafe_from_address=rows)
    var first = Int(r[0])
    if Int(r[count - 1]) - first != count - 1:
        return -1
    for i in range(1, count):  # driver-loop: over one side's rows (stops at the first gap)
        if Int(r[i]) != first + i:
            return -1
    return first


def _multi_output_scorers(codes: List[Int], plan: MselSearchPlan, multi_cls: Bool) raises -> Bool:
    """Lane mf-rows8b: the member's own score alone, or every scorer one
    that reads the member's predict (a regression, label or clustering row:
    a multi-output target's search). Replaces `_regression_scorers` (lane
    mf-score8 round b: regression rows only), removed. Lane mf-r9c: a
    MULTI-OUTPUT CLASSIFIER (`multi_cls`) also by its probability rows (each
    side scikit-learn's reading of the per-output list,
    core/msel_scorer_run.mojo `_multi_proba_value`)."""
    if len(codes) == 1 and codes[0] == SEARCH_SCORE_MEMBER:
        return True
    for j in range(len(codes)):  # small-loop(the scorers)
        if codes[j] == SEARCH_SCORE_MEMBER:
            return False
        var row = msel_scorer_row(codes[j], plan.options[j] if j < len(plan.options) else MselScorerOptions(),
                                  Int(plan.selections.unsafe_ptr()))
        if row.op != MEMBER_OP_PREDICT and not (multi_cls and row.op == MEMBER_OP_PREDICT_PROBA):
            return False
    return True


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
    # A MULTI-OUTPUT CLASSIFIER (lane mf-tail5): its codes per output, each
    # output's class count in `class_counts` (`classes` is 0).
    var multi_cls = y_cols > 1 and y_dtype == MEMBER_I32 and len(plan.class_counts) == y_cols
    var responding = plan.respond >= 0
    var codes = plan.scorers.copy()
    if responding:
        # THE CV TOOLS' `respond` (module note): no scorer; one placeholder
        # table of zeros keeps the tables' shape.
        # A multi-output target (lane mf-msel5b): a regressor's predict, one
        # column per output.
        # A multi-output classifier's predict_proba (lane mf-tail5): one
        # result block per output (`respond_width` = minus the outputs,
        # `respond_dst` the address of their addresses; module note).
        # lane mf-rows8b: predict_log_proba too (its log per output block)
        var listed = multi_cls and plan.respond == MEMBER_OP_PREDICT_PROBA
        if len(codes) != 0 or (y_cols > 1 and (classifier or (plan.respond != MEMBER_OP_PREDICT and not listed))):
            raise Error("member ABI: a search that answers its units' responses takes no scorer and one output")
        if listed and plan.respond_width != -y_cols:
            raise Error("member ABI: a multi-output predict_proba needs one result block per output")
        if plan.respond_dst == 0 or (plan.respond_width < 1 and not listed):
            raise Error("member ABI: a search that answers its units' responses needs the caller's result buffer")
        codes.append(SEARCH_SCORE_MEMBER)
    elif len(codes) == 0:
        codes.append(plan.scoring)
    if y_cols > 1 and (classifier or not _multi_output_scorers(codes, plan, multi_cls)):
        # lane mf-score8 round b: a multi-output REGRESSOR may also be scored
        # by regression rows (their programs take the outputs). Lane
        # mf-rows8b: and by the label and clustering rows (each side raises
        # scikit-learn's multi-output words, `MselScorer.outs`); a
        # multi-output classifier by the same rows (its codes read as their
        # labels' values, `MselScorer.counts_at`). Not a probability or
        # decision row (the admission keeps those).
        raise Error("member ABI: a search over a multi-output member folds by KFold and scores by the member")
    if m == 0:
        raise Error("No fits were performed. Was the CV iterator empty? Were there no candidates?")
    if len(plan.masks) != m:
        raise Error("member ABI: a search needs one capability mask per candidate")
    var given = len(plan.fold_test_at)
    if (len(plan.fold_train_at) != given or len(plan.fold_train_n) != given
            or len(plan.fold_test_n) != given):
        raise Error("member ABI: a search's given folds need a train and a test side each")
    var n_splits = plan.n_splits if given == 0 else given
    if given == 0 and n_splits < 2:
        raise Error("ValueError: k-fold cross-validation requires at least one train/test split by setting "
                    + "n_splits=2 or more, got n_splits=" + String(n_splits) + ".")
    if given == 0 and n_splits > n:
        raise Error("ValueError: Cannot have number of splits n_splits=" + String(n_splits)
                    + " greater than the number of samples: n_samples=" + String(n) + ".")
    # 1. The folds: per fold its train and test rows (int32 row views). The
    # int cv's (fold f: its n - c train rows, then its c test rows), or the
    # given ones (the caller's views, checked by its UNPACK / fit).
    var fold_counts = List[Int32](length=n_splits + 1, fill=Int32(0))
    var fold_rows = List[Int32](length=max(n_splits * n if given == 0 else 1, 1), fill=Int32(0))
    var rows_at = Int(fold_rows.unsafe_ptr())
    var tr_at = List[Int](capacity=n_splits)
    var tr_n = List[Int](capacity=n_splits)
    var te_at = List[Int](capacity=n_splits)
    var te_n = List[Int](capacity=n_splits)
    var used = 0
    if given == 0:
        var codes_at = plan.y if classifier else Int(fold_counts.unsafe_ptr())
        var status = xt_device_folds(plan.glue, codes_at, rows_at, Int(fold_counts.unsafe_ptr()), n, n_splits, k)
        if status == 1:
            raise Error("ValueError: n_splits=" + String(n_splits)
                        + " cannot be greater than the number of members in each class.")
        if status != 0:
            raise Error("ValueError: y codes outside [0, n_classes)")
        for f in range(n_splits):  # driver-loop: over the cv folds
            if Int(fold_counts[f]) > 0:
                used = f + 1
        for f in range(used):  # driver-loop: over the cv folds (their sides' views)
            var c = Int(fold_counts[f])
            tr_at.append(rows_at + 4 * (f * n))
            tr_n.append(n - c)
            te_at.append(rows_at + 4 * (f * n + n - c))
            te_n.append(c)
    else:
        used = given
        for f in range(given):  # driver-loop: over the given folds (their sides' views)
            tr_at.append(plan.fold_train_at[f])
            tr_n.append(plan.fold_train_n[f])
            te_at.append(plan.fold_test_at[f])
            te_n.append(plan.fold_test_n[f])
    var res = MselSearchResult()
    res.n_splits = used
    var sides = 2 if plan.return_train and not responding else 1
    if plan.indices:
        for f in range(used):  # driver-loop: over the folds (their rows, copied once for the caller)
            var tr = List[Int32](length=tr_n[f], fill=Int32(0))
            var te = List[Int32](length=te_n[f], fill=Int32(0))
            var tp = _I32P(unsafe_from_address=tr_at[f])
            var ep = _I32P(unsafe_from_address=te_at[f])
            for i in range(tr_n[f]):  # driver-loop: over the fold's train rows
                tr[i] = tp[i]
            for i in range(te_n[f]):  # driver-loop: over the fold's test rows
                te[i] = ep[i]
            res.fold_train.append(tr^)
            res.fold_test.append(te^)
    # The scorers (resolved once) and the member responses they read: each
    # distinct response is asked once per unit and side.
    var member_score = len(codes) == 1 and codes[0] == SCORER_MEMBER and not responding
    var rows_of = List[MselScorer]()
    var ops = List[Int]()
    # LANE mf-r9d: per distinct response, whether it is predict_log_proba (a
    # scorer's own response, `MselScorer.log`); False for every other
    var logs = List[Bool]()
    var op_of = List[Int]()
    var clusterer = False
    if responding:
        ops.append(plan.respond)
        logs.append(False)  # the cv tools' log is placed by `_place_response` (`respond_log`)
        op_of.append(0)
    elif not member_score:
        for j in range(len(codes)):  # driver-loop: over the scorers (once per fit)
            if codes[j] == SCORER_MEMBER:
                raise Error("member ABI: the member's own score cannot be one of several scorers")
            var row = msel_scorer_row(codes[j], plan.options[j] if j < len(plan.options) else MselScorerOptions(),
                                      Int(plan.selections.unsafe_ptr()))
            # A clusterer (lane mf-cluster5: no class codes) is scored by the
            # clustering scorers only, on its predicted cluster ids.
            if not classifier and msel_scorer_cluster(row):
                clusterer = True
            elif (msel_scorer_regression(row) == classifier and not (classifier and row.lv_n == k)
                  and not (not classifier and msel_scorer_float_labels(row))
                  and not (not classifier and y_cols == 1 and msel_scorer_float_proba(row))
                  and not (multi_cls and row.op == MEMBER_OP_PREDICT_PROBA)):
                # (lane mf-r9d: a probability or decision row on a non-classifier's
                # own columns, `msel_scorer_float_proba`)
                # (lane mf-r9c: a multi-output classifier's probability row,
                # `classes` 0, reads its outputs' label value words)
                # lane mf-score8: a regression scorer on a classifier reads its
                # labels' value words (core/msel_scorer_run.mojo module note)
                raise Error("member ABI: scorer '" + String(row.name) + "' does not score this kind of estimator")
            if row.tier == 1 and row.tier_word < 0:
                # LANE mf-r9d: a scorer of the other numeric_mode tier runs on
                # that tier's route (`MselOtherTier`), resolved by the UNPACK
                if not plan.other.resolved:
                    raise Error("member ABI: a scorer of the other numeric_mode tier needs that tier's metric route")
                if msel_scorer_cluster_entry(row) and plan.other.cluster.word == 0:
                    raise Error("member ABI: a label-pair cluster scorer of the other numeric_mode tier needs that "
                                + "tier's cluster entry")
            elif msel_scorer_cluster_entry(row) and row.tier == 0 and plan.cluster.word == 0:
                raise Error("member ABI: a label-pair cluster scorer needs the metrics image's cluster entry")
            var at = -1
            for o in range(len(ops)):  # small-loop(the distinct responses, at most four)
                # LANE mf-r9d: predict_log_proba is its own response (the log of
                # predict_proba in the member's precision, taken before the
                # scorer's float32 narrowing: `_respond`)
                if ops[o] == row.op and logs[o] == row.log:
                    at = o
            if at < 0:
                at = len(ops)
                ops.append(row.op)
                logs.append(row.log)
            op_of.append(at)
            rows_of.append(row)
    # lane mf-score8 round b: a clusterer scored by clustering scorers and
    # others together (its ids read as numbers or labels, `MselScorer.ids`).
    var nsc = len(codes)
    var nops = len(ops)
    var positives = List[Int](length=max(nsc, 1), fill=-1)
    for j in range(min(nsc, len(plan.positives))):  # small-loop(the scorers' positive class codes)
        positives[j] = plan.positives[j]
    for j in range(len(rows_of)):  # small-loop(the scorers: a binary one needs its positive code)
        # SCORER_POSITIVE_ABSENT / _KIND (lane mf-cluster5): pos_label=1 is not
        # one of the labels; each side raises the door's words (msel_scorer_run).
        var marked = (positives[j] == SCORER_POSITIVE_ABSENT or positives[j] == SCORER_POSITIVE_KIND
                      or positives[j] == SCORER_POSITIVE_MIXED)
        if y_cols > 1:
            continue  # lane mf-rows8b: a multi-output side refuses before any positive label is read
        if msel_scorer_positive(rows_of[j]) and not marked and (positives[j] < 0 or positives[j] >= k):
            raise Error("member ABI: a binary scorer needs its positive class code")
        if msel_scorer_ap(rows_of[j]) and k == 2 and not marked and (positives[j] < 0 or positives[j] >= k):
            raise Error("member ABI: 'average_precision' of a binary member needs its positive class code")
    # A FIT WITHOUT A TARGET (lane mf-msel5b): an unsupervised base fitted
    # with y=None, scored by its own score (scikit-learn's scorer=None:
    # `estimator.score(X_test)`) or answering its responses (cross_val_predict).
    var y_free = plan.y == 0
    if y_free and (classifier or not (member_score or responding)):
        raise Error("member ABI: a search without a target is scored by the member's own score")
    # THE FOLDS THAT LACK A CLASS (lane mf-msel5b): per fold, the classes its
    # train rows hold when they are not all k (one pass over its train codes
    # each, across the host cores: the class-cover pass the member set made,
    # which `skip_lacking` now leaves out), and its train codes re-coded onto
    # them (the fold's fit sees only its own classes, as the estimator's own
    # fit on those rows does). A lacking fold's responses are mapped back to
    # the classes (`_place_response`, or the codes of a predict a label
    # scorer reads); a scorer reading probabilities or decisions, or the
    # member's own score, of such a fold is scored by the fold estimator's own
    # columns (lane mf-kmeans5: `_fold_view_side`; a classifier's own score
    # as `ClassifierMixin.score`, the accuracy of its predict).
    var fold_held = List[List[Int32]]()
    var fold_codes = List[List[Int32]]()
    var any_lacking = False
    if classifier and y_cols == 1 and not y_free:
        for _ in range(used):  # small-loop(one held list per fold)
            fold_held.append(List[Int32]())
            fold_codes.append(List[Int32]())
        var held_at = Int(fold_held.unsafe_ptr())
        var fold_codes_at = Int(fold_codes.unsafe_ptr())
        var tr_at_at = Int(tr_at.unsafe_ptr())
        var tr_n_at = Int(tr_n.unsafe_ptr())
        var codes_y = plan.y

        def _mark(f: Int) {imm held_at, imm fold_codes_at, imm tr_at_at, imm tr_n_at, imm codes_y, imm k, imm n}:
            # One fold per task: its own held list and re-coded codes only.
            var rows = _IntP(unsafe_from_address=tr_at_at)[f]
            var count = _IntP(unsafe_from_address=tr_n_at)[f]
            var held = _held_codes(codes_y, rows, count, k)
            if len(held) == 0:
                return
            var to = List[Int32](length=max(k, 1), fill=Int32(0))
            for j in range(len(held)):  # small-loop(the fold's held classes)
                to[Int(held[j])] = Int32(j)
            var recoded = List[Int32](length=max(n, 1), fill=Int32(0))
            var src = _I32P(unsafe_from_address=codes_y)
            var r = _I32P(unsafe_from_address=rows)
            for i in range(count):  # driver-loop: over the fold's train rows (their fold codes)
                var row = Int(r[i])
                recoded[row] = to[Int(src[row])]
            MutPointer[List[Int32], MutUntrackedOrigin](unsafe_from_address=fold_codes_at)[f] = recoded^
            MutPointer[List[Int32], MutUntrackedOrigin](unsafe_from_address=held_at)[f] = held^

        host_parallelize(_mark, used)
        for f in range(used):  # small-loop(the folds' lacking marks)
            if len(fold_held[f]) > 0:
                any_lacking = True
    # A MULTI-OUTPUT CLASSIFIER'S FOLDS THAT LACK A CLASS (lane mf-tail5): per
    # fold and output the classes its train rows hold when they are not all
    # of that output's (`mheld[f * y_cols + j]`; empty: every class), and per
    # lacking fold its train codes re-coded output by output (`mcodes[f]`,
    # rows x outputs) with each output's fold class count (`mcounts[f]`): the
    # fold's MultiOutputClassifier fits each output's estimator on that
    # output's own classes, as scikit-learn's does. One task per fold.
    var mheld = List[List[Int32]]()
    var mcodes = List[List[Int32]]()
    var mcounts = List[List[Int]]()
    var multi_lacking = False
    if multi_cls and not y_free:
        for _ in range(used * y_cols):  # small-loop(one held list per fold and output)
            mheld.append(List[Int32]())
        for _ in range(used):  # small-loop(one block and one count list per fold)
            mcodes.append(List[Int32]())
            mcounts.append(List[Int]())
        var mheld_at = Int(mheld.unsafe_ptr())
        var mcodes_at = Int(mcodes.unsafe_ptr())
        var mcounts_at = Int(mcounts.unsafe_ptr())
        var counts_at = Int(plan.class_counts.unsafe_ptr())
        var mtr_at = Int(tr_at.unsafe_ptr())
        var mtr_n = Int(tr_n.unsafe_ptr())
        var mcodes_y = plan.y

        def _mark_multi(f: Int) {imm mheld_at, imm mcodes_at, imm mcounts_at, imm counts_at, imm mtr_at, imm mtr_n,
                                 imm mcodes_y, imm y_cols, imm n}:
            var rows = _IntP(unsafe_from_address=mtr_at)[f]
            var count = _IntP(unsafe_from_address=mtr_n)[f]
            var ks = _IntP(unsafe_from_address=counts_at)
            var lists = MutPointer[List[Int32], MutUntrackedOrigin](unsafe_from_address=mheld_at)
            var lacking = False
            for j in range(y_cols):  # small-loop(the outputs)
                var held = _held_codes_col(mcodes_y, rows, count, ks[j], y_cols, j)
                if len(held) > 0:
                    lacking = True
                lists[f * y_cols + j] = held^
            if not lacking:
                return
            var block = List[Int32](length=max(n * y_cols, 1), fill=Int32(0))
            var fold_counts = List[Int](length=y_cols, fill=0)
            var src = _I32P(unsafe_from_address=mcodes_y)
            var r = _I32P(unsafe_from_address=rows)
            for j in range(y_cols):  # small-loop(the outputs: each re-coded onto its own held classes)
                var m = len(lists[f * y_cols + j])
                fold_counts[j] = m if m > 0 else ks[j]
                var to = List[Int32](length=max(ks[j], 1), fill=Int32(0))
                for c in range(ks[j]):  # small-loop(output j's classes)
                    to[c] = Int32(c)
                for c in range(m):  # small-loop(output j's held classes)
                    to[Int(lists[f * y_cols + j][c])] = Int32(c)
                for i in range(count):  # driver-loop: over the fold's train rows (output j's fold codes)
                    var row = Int(r[i])
                    block[row * y_cols + j] = to[Int(src[row * y_cols + j])]
            MutPointer[List[Int32], MutUntrackedOrigin](unsafe_from_address=mcodes_at)[f] = block^
            MutPointer[List[Int], MutUntrackedOrigin](unsafe_from_address=mcounts_at)[f] = fold_counts^

        host_parallelize(_mark_multi, used)
        for f in range(used):  # small-loop(the folds' lacking marks)
            if len(mcodes[f]) > 0:
                multi_lacking = True
    var mheld_list = Int(mheld.unsafe_ptr()) if multi_lacking else 0
    var mlack = List[UInt8](length=max(used, 1), fill=UInt8(0))
    for f in range(len(mcodes)):  # small-loop(the folds' lacking marks, one byte each)
        if len(mcodes[f]) > 0:
            mlack[f] = UInt8(1)
    var mlack_at = Int(mlack.unsafe_ptr())
    var dst_counts = plan.class_counts.copy()
    if any_lacking and member_score:
        # A CLASSIFIER'S OWN SCORE on folds that lack a class (lane
        # mf-kmeans5): scikit-learn's `ClassifierMixin.score`,
        # `accuracy_score(y, fold_estimator.predict(X))`. A fold member knows
        # only its own classes, so every unit is scored as that: its predict
        # (fold codes mapped back to the classes, the label scorers' path
        # below) through the accuracy program, which the UNPACK resolves for
        # a classifier search scored by its members.
        member_score = False
        codes[0] = SCORER_ACCURACY
        var accuracy = msel_scorer_row(SCORER_ACCURACY)
        rows_of.append(accuracy)
        ops.append(accuracy.op)
        logs.append(False)
        op_of.append(0)
        nops = len(ops)
    var held_addr = List[Int](length=max(used, 1), fill=0)
    var held_count = List[Int](length=max(used, 1), fill=0)
    for f in range(len(fold_held)):  # small-loop(the folds' held class views)
        if len(fold_held[f]) > 0:
            held_addr[f] = Int(fold_held[f].unsafe_ptr())
            held_count[f] = len(fold_held[f])
    var held_addr_at = Int(held_addr.unsafe_ptr())
    var held_count_at = Int(held_count.unsafe_ptr())
    # THE RESIDENT ROUTE (module note): per candidate where its sides' rows
    # are; the X blocks are gathered only for a candidate that reads none.
    var rows_route = msel_search_rows(plan)
    var any_gathered = False
    for c in range(m):  # driver-loop: over the candidates (their route words)
        if rows_route[c] == SEARCH_ROWS_GATHERED:
            any_gathered = True
    var rows_route_at = Int(rows_route.unsafe_ptr())
    # Each candidate's response element types, asked once.
    var out_dtypes = List[Int](length=max(m * nops, 1), fill=MEMBER_I32)
    # LANE mf-r9d (door :3960): a NON-classifier's probability or decision
    # response is its own columns (a mixture model's components; scikit-learn
    # hands the raw answer to the metric), asked once per candidate before
    # any fit (core/member_abi.mojo `member_outputs`); 0: the classes' width
    var out_widths = List[Int](length=max(m * nops, 1), fill=0)
    for c in range(m):  # driver-loop: over the candidates (one OUTPUTS question per response)
        for o in range(nops):  # small-loop(the distinct responses)
            if not classifier and not responding and ops[o] != MEMBER_OP_PREDICT:
                out_widths[c * nops + o] = max(member_outputs(plan.plans[c], ops[o], d, 1), 1)
            var probe = plan.plans[c].copy()
            probe.classes = k
            var labels_out = ops[o] == MEMBER_OP_PREDICT and (classifier or clusterer)
            var unsaid = MEMBER_I32 if labels_out else MEMBER_F32
            var dtype = member_output_dtype(probe, ops[o], d, 1, unsaid)
            if responding and ops[o] == MEMBER_OP_PREDICT and not classifier:
                # cross_val_predict of a non-classifier (lane mf-msel5b): float
                # values, or a clusterer's integer ids, as the member writes them
                if dtype != MEMBER_I32 and dtype != MEMBER_I64 and dtype != MEMBER_F32 and dtype != MEMBER_F64:
                    raise Error("member ABI: a cv tool response of an element type the result cannot hold")
            elif labels_out:
                # lane mf-rows8b (row :3899): a non-classifier's float predict
                # under a clustering scorer is kept as floats; each side raises
                # the public cluster doors' refusal (`MselScorer.fpred`)
                var float_ids = not classifier and (dtype == MEMBER_F32 or dtype == MEMBER_F64)
                if dtype != MEMBER_I32 and dtype != MEMBER_I64 and not float_ids:
                    raise Error("member ABI: a search scored by labels needs a member that predicts class codes")
            elif ops[o] == MEMBER_OP_PREDICT and not classifier and (dtype == MEMBER_I32 or dtype == MEMBER_I64):
                pass  # lane mf-score8 round b: a non-classifier's integer ids (narrowed to int32)
            elif dtype != MEMBER_F32 and dtype != MEMBER_F64:
                raise Error("member ABI: a search scorer needs float32 or float64 member outputs")
            out_dtypes[c * nops + o] = dtype
    var out_dtypes_at = Int(out_dtypes.unsafe_ptr())
    var ops_at = Int(ops.unsafe_ptr())
    # LANE mf-r9d: one byte per distinct response, 1 = predict_log_proba
    var log_bytes = List[UInt8](length=max(nops, 1), fill=UInt8(0))
    for o in range(min(nops, len(logs))):  # small-loop(the distinct responses)
        if logs[o]:
            log_bytes[o] = UInt8(1)
    var logs_at = Int(log_bytes.unsafe_ptr())
    # Per (fold, side): its row numbers, its count and its gathered X and y.
    var side_rows = List[Int](length=used * sides, fill=0)
    var side_count = List[Int](length=used * sides, fill=0)
    var side_x = List[List[Float32]](capacity=used * sides)
    var side_y = List[List[Int32]](capacity=used * sides)
    for f in range(used):  # driver-loop: over the cv folds (their sides' rows)
        side_rows[f * sides] = te_at[f]
        side_count[f * sides] = te_n[f]
        if sides == 2:
            side_rows[f * sides + 1] = tr_at[f]
            side_count[f * sides + 1] = tr_n[f]
    # THE CV TOOLS (module note): a given fold's test side holding the same
    # rows as an earlier one's reads that side's blocks (alias_of: -1, its own).
    var alias_of = List[Int](length=max(used * sides, 1), fill=-1)
    if given > 0:
        for f in range(1, used):  # driver-loop: over the given folds (their test sides, once per fit)
            for g in range(f):  # driver-loop: over the earlier folds (the first with the same test rows)
                if te_n[g] == te_n[f] and alias_of[g * sides] < 0 and _same_rows(te_at[g], te_at[f], te_n[f]):
                    alias_of[f * sides] = g * sides
                    break
    # A SIDE THAT IS ONE RUN OF ROWS (lane mf-msel5b; the MSEL_CV_SLICE_BIT
    # route's row-range views, moved into this RUN for every fold source): a
    # view of X and of the target at its first row, gathered nowhere.
    var side_start = List[Int](length=max(used * sides, 1), fill=-1)
    var side_start_at = Int(side_start.unsafe_ptr())
    var side_rows_at0 = Int(side_rows.unsafe_ptr())
    var side_count_at0 = Int(side_count.unsafe_ptr())

    def _starts(s: Int) {imm side_rows_at0, imm side_count_at0, imm side_start_at}:
        _IntP(unsafe_from_address=side_start_at)[s] = _run_start(_IntP(unsafe_from_address=side_rows_at0)[s],
                                                                 _IntP(unsafe_from_address=side_count_at0)[s])

    host_parallelize(_starts, used * sides)
    for s in range(used * sides):  # driver-loop: over the fold sides (their blocks, allocated once)
        var own = alias_of[s] < 0 and side_start[s] < 0
        side_x.append(List[Float32](length=max(side_count[s] * d if any_gathered and own else 1, 1), fill=0.0))
        side_y.append(List[Int32](length=max(side_count[s] * y_cols if own and not y_free else 1, 1),
                                  fill=Int32(0)))
    var columns = List[Int32](length=max(d, 1), fill=Int32(0))
    for j in range(d):  # small-loop(the column numbers 0 .. d-1 of a side's gather, once per fit)
        columns[j] = Int32(j)
    if any_gathered:
        for s in range(used * sides):  # driver-loop: over the fold sides (one glue gather of X each, parallel inside)
            if alias_of[s] >= 0 or side_start[s] >= 0:
                continue
            _ = xt_gather_f32(plan.glue, plan.x, side_rows[s], Int(columns.unsafe_ptr()), Int(side_x[s].unsafe_ptr()),
                              n, d, side_count[s], d)
    var side_rows_at = Int(side_rows.unsafe_ptr())
    var side_count_at = Int(side_count.unsafe_ptr())
    var y_src = plan.y
    # Per side: where its X block and its target block are (its own, an
    # earlier side's with the same rows, or a view at its first row).
    var x_at = List[Int](length=max(used * sides, 1), fill=0)
    var y_dst = List[Int](length=max(used * sides, 1), fill=0)
    for s in range(used * sides):  # driver-loop: over the fold sides (their block addresses)
        var a = alias_of[s] if alias_of[s] >= 0 else s
        if side_start[a] >= 0:
            x_at[s] = plan.x + 4 * side_start[a] * d
            y_dst[s] = 0 if y_free else plan.y + 4 * side_start[a] * y_cols
        else:
            x_at[s] = Int(side_x[a].unsafe_ptr())
            y_dst[s] = 0 if y_free else Int(side_y[a].unsafe_ptr())
    var y_own = List[Int](length=max(used * sides, 1), fill=0)
    for s in range(used * sides):  # driver-loop: over the fold sides (1: the side gathers its own target block)
        y_own[s] = 1 if alias_of[s] < 0 and side_start[s] < 0 and not y_free else 0
    var y_own_at = Int(y_own.unsafe_ptr())
    var y_dst_at = Int(y_dst.unsafe_ptr())

    def _gather_y(s: Int) {imm side_rows_at, imm side_count_at, imm y_src, imm y_dst_at, imm y_cols, imm y_own_at}:
        if _IntP(unsafe_from_address=y_own_at)[s] == 0:
            return
        var rows = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=side_rows_at)[s]
        var count = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=side_count_at)[s]
        var dst = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=y_dst_at)[s]
        _gather_words(y_src, rows, count, y_cols, dst)

    host_parallelize(_gather_y, used * sides)
    # 2-4. The units, wave by wave.
    var units = m * used
    var test = List[Float64](length=max(units * nsc, 1), fill=0.0)
    var train = List[Float64](length=max(units * nsc, 1), fill=0.0)
    res.fit_seconds = List[Float64](length=units, fill=0.0)
    res.score_seconds = List[Float64](length=units, fill=0.0)
    if plan.keep:
        res.unit_states = List[Int](length=units, fill=0)
    var dst = plan.respond_dst
    var dst_width = plan.respond_width
    var dst_dtype = plan.respond_dtype
    var log_words = plan.respond_log
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
        run.scope = plan.scope
        # THE FOLDS THAT LACK A CLASS (above): the set's cover pass is ours.
        run.skip_lacking = len(fold_held) > 0
        var all_free = True
        for u in range(u0, u1):  # driver-loop: over the wave's units (admission into the set, once each)
            var c = u // used
            var f = u - c * used
            var p = plan.plans[c].copy()
            p.classes = k
            p.class_counts = plan.class_counts.copy()
            var target = plan.y
            if f < len(fold_held) and len(fold_held[f]) > 0:
                # the fold's own classes: its fit sees its re-coded codes, and a
                # class_weight dict narrowed to them (core/member_class_weight.mojo)
                p.classes = len(fold_held[f])
                member_cw_subset(p, fold_held[f])
                target = Int(fold_codes[f].unsafe_ptr())
            if f < len(mcodes) and len(mcodes[f]) > 0:
                # a multi-output fold lacking a class in some output (lane
                # mf-tail5): each output's own classes and re-coded codes
                p.class_counts = mcounts[f].copy()
                target = Int(mcodes[f].unsafe_ptr())
            var mask = plan.masks[c]
            if (mask & MEMBER_GIL_FREE) == 0:
                all_free = False
            run.add(p^, target, 0, tr_at[f], tr_n[f], 0, 0,
                    UInt8(1) if (mask & MEMBER_CAN_TEST_ROWS) != 0 else UInt8(0),
                    UInt8(1) if (mask & MEMBER_BORROWS_FIT) != 0 else UInt8(0), _set_word(mask))
            if plan.given != 0 and not classifier and (mask & MEMBER_READS_Y_GIVEN) != 0:
                run.given_target(plan.given, plan.given_dtype)  # THE GIVEN y, drawn with the fold's train rows
            if rows_route[c] == SEARCH_ROWS_RESIDENT:
                # the fit reads the fold's train rows by index (the set's
                # `by_index`) out of the whole X, X's pin offered
                run.pin_tokens[u - u0] = plan.pin
        for v in range(width):  # driver-loop: over the wave's units (a refused candidate's fail unfitted)
            var refused = (plan.masks[(u0 + v) // used] >> SEARCH_REFUSED_SHIFT) - 1
            if refused >= 0:
                run.failed[v] = UInt8(1)
                run.errors[v] = (plan.label_words[refused].copy() if refused < len(plan.label_words)
                                 else String("member ABI: a refused candidate's words are missing"))
        _fit_wave(run)
        for v in range(width):  # driver-loop: over the wave's units (their fits' seconds)
            res.fit_seconds[u0 + v] = run.seconds[v]
        for v in range(width):  # driver-loop: over the wave's units (did a cache take X's reference)
            if rows_route[(u0 + v) // used] == SEARCH_ROWS_RESIDENT and run.pin_tokens[v] == 0:
                _note_kept(plan)
        if not plan.tolerate:
            var first = member_set_first_failure(run)
            if first >= 0:
                var words = run.errors[first].copy()
                member_set_run_release(run)
                _drop_kept(plan, res)
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
                # a predict: one column per output (a multi-output regressor's
                # responses, lane mf-msel5b); else one per class
                var width_o = y_cols if ops[o] == MEMBER_OP_PREDICT else max(k, 1)
                var cand_o = (u0 + t // sides) // used
                if out_widths[cand_o * nops + o] > 0:
                    width_o = out_widths[cand_o * nops + o]  # lane mf-r9d: a non-classifier's own columns
                if multi_cls and ops[o] != MEMBER_OP_PREDICT:
                    # a multi-output classifier's proba: every output's columns (lane mf-tail5)
                    width_o = 0
                    for j in range(y_cols):  # small-loop(the outputs' class counts)
                        width_o += plan.class_counts[j]
                preds.append(List[Int64](length=max(count * width_o, 1), fill=Int64(0)))
                pred_at[t * nops + o] = Int(preds[t * nops + o].unsafe_ptr())
                pred_cap[t * nops + o] = count * width_o
        var run_at = Int(UnsafePointer(to=run))
        var values_at = Int(values.unsafe_ptr())
        var failed_at = Int(failed.unsafe_ptr())
        var errors_at = Int(errors.unsafe_ptr())
        var pred_list = Int(pred_at.unsafe_ptr())
        var cols_list = Int(pred_cols.unsafe_ptr())
        var cap_list = Int(pred_cap.unsafe_ptr())
        var x_list = Int(x_at.unsafe_ptr())
        # THE RESIDENT ROUTE: one byte per task, 1 when a cache took X's pin.
        var took = List[UInt8](length=max(tasks, 1), fill=UInt8(0))
        var took_at = Int(took.unsafe_ptr())
        # THE CV TOOLS: each test task's seconds (its member calls).
        var task_secs = List[Float64](length=max(tasks, 1), fill=0.0)
        var secs_at = Int(task_secs.unsafe_ptr())
        var narrow = not responding
        var base_x = plan.x
        var base_y = plan.y
        var scope = plan.scope
        var pin = plan.pin
        var dst_counts_at = Int(dst_counts.unsafe_ptr())

        def _score(t: Int) {imm run_at, imm values_at, imm failed_at, imm errors_at, imm pred_list, imm cols_list,
                            imm cap_list, imm x_list, imm y_dst_at, imm side_count_at, imm sides, imm used, imm u0,
                            imm d, imm y_dtype, imm y_cols, imm member_score, imm nsc, imm nops, imm ops_at,
                            imm out_dtypes_at, imm rows_route_at, imm side_rows_at, imm took_at, imm base_x,
                            imm base_y, imm n, imm scope, imm pin, imm secs_at, imm narrow, imm dst, imm dst_width,
                            imm dst_dtype, imm log_words, imm held_addr_at, imm held_count_at, imm k,
                            imm multi_cls, imm mheld_list, imm mlack_at, imm dst_counts_at, imm logs_at}:
            var v = t // sides
            var side = t % sides
            var started = perf_counter_ns()
            var s = ((u0 + v) % used) * sides + side
            var rp = _RunPtr(unsafe_from_address=run_at)
            if rp[].failed[v] != UInt8(0) or rp[].states[v] == 0:
                for j in range(nsc):  # small-loop(the scorers)
                    _U8P(unsafe_from_address=failed_at)[t * nsc + j] = UInt8(1)
                return
            var c = (u0 + v) // used
            var route = _IntP(unsafe_from_address=rows_route_at)[c]
            var index = 0 if route == SEARCH_ROWS_GATHERED else _IntP(unsafe_from_address=side_rows_at)[s]
            var place = _Side(_IntP(unsafe_from_address=x_list)[s], _IntP(unsafe_from_address=y_dst_at)[s],
                              _IntP(unsafe_from_address=side_count_at)[s], index, base_x, base_y, n, scope,
                              pin if route == SEARCH_ROWS_RESIDENT else 0, took_at + t)
            try:
                var mfold = (u0 + v) % used
                var mlacks = _U8P(unsafe_from_address=mlack_at)[mfold] != UInt8(0)
                if member_score and mlacks:
                    # a multi-output fold lacking a class (lane mf-tail5): the
                    # member's predict mapped back, scored as its own score
                    _F64P(unsafe_from_address=values_at)[t] = _multi_lacking_score(run_at, v, place, d, y_cols,
                                                                                   mheld_list, mfold)
                    if side == 0:
                        _F64P(unsafe_from_address=secs_at)[t] = Float64(perf_counter_ns() - started) / 1e9
                    return
                if member_score:
                    _F64P(unsafe_from_address=values_at)[t] = _member_score(run_at, v, place, d, y_dtype, y_cols)
                    if side == 0:
                        _F64P(unsafe_from_address=secs_at)[t] = Float64(perf_counter_ns() - started) / 1e9
                    return
                var fold = (u0 + v) % used
                var held = _IntP(unsafe_from_address=held_addr_at)[fold]
                for o in range(nops):  # small-loop(the distinct responses)
                    var at = t * nops + o
                    var op = _IntP(unsafe_from_address=ops_at)[o]
                    var buffer = _IntP(unsafe_from_address=pred_list)[at]
                    var dtype = _IntP(unsafe_from_address=out_dtypes_at)[c * nops + o]
                    var got = _respond(run_at, v, op, place, d, buffer, _IntP(unsafe_from_address=cap_list)[at],
                                       dtype, narrow, y_cols, _U8P(unsafe_from_address=logs_at)[o] != UInt8(0))
                    _IntP(unsafe_from_address=cols_list)[at] = got
                    if multi_cls and mlacks and op == MEMBER_OP_PREDICT:
                        # each output's fold codes as its classes' codes (lane mf-tail5)
                        _map_multi_codes(buffer, dtype, place.count, y_cols, mheld_list, fold)
                    if dst != 0 and multi_cls and op == MEMBER_OP_PREDICT_PROBA:
                        # THE PER-OUTPUT BUFFERS (lane mf-tail5): one block per output
                        _place_multi_proba(buffer, dtype, got, place.count, _IntP(unsafe_from_address=side_rows_at)[s],
                                           dst, y_cols, dst_counts_at, mheld_list if mlacks else 0, fold,
                                           log_words)
                    elif dst != 0:
                        # THE CV TOOLS' FINAL BUFFER: straight into the caller's
                        # result at the side's rows (no copy between).
                        _place_response(buffer, dtype, got, place.count, _IntP(unsafe_from_address=side_rows_at)[s],
                                        dst, dst_width, dst_dtype, op, log_words, held,
                                        _IntP(unsafe_from_address=held_count_at)[fold], k)
                    elif held != 0 and op == MEMBER_OP_PREDICT:
                        # a lacking fold's predicted fold codes as the classes'
                        # codes, in place, for the label scorers
                        var codes = _I32P(unsafe_from_address=buffer)
                        var h = _I32P(unsafe_from_address=held)
                        for i in range(place.count):  # driver-loop: over the side's predicted codes
                            codes[i] = h[Int(codes[i])]
                if side == 0:
                    _F64P(unsafe_from_address=secs_at)[t] = Float64(perf_counter_ns() - started) / 1e9
            except error:
                for j in range(nsc):  # small-loop(the scorers)
                    _U8P(unsafe_from_address=failed_at)[t * nsc + j] = UInt8(2)
                    _StrPtr(unsafe_from_address=errors_at)[t * nsc + j] = String(error)

        if all_free:
            host_parallelize(_score, tasks)
        else:
            for t in range(tasks):  # driver-loop: over the wave's tasks (THE GIL RULE: kinds that are not GIL-free)
                _score(t)
        for t in range(tasks):  # driver-loop: over the wave's tasks (did a cache take X's reference)
            if took[t] != UInt8(0):
                _note_kept(plan)
        # The metric programs (named scorers), one after the other; each
        # unit's metric warnings in its sides' order (test, then train) and
        # its scorers' order (lane mf-score5: raised at PACK after the unit's
        # fit notices, as the scorer raises them during the search).
        var metric_notes = List[List[Float64]](capacity=width)
        for _ in range(width):  # small-loop(the wave's units: one warning list each)
            metric_notes.append(List[Float64]())
        if responding:
            # The responses are in the caller's buffer already (the tasks
            # above). A unit of a fold that lacks a class warns as
            # `_enforce_prediction_order` warns, in unit order.
            # (lane mf-tail5) Only a response scikit-learn encodes warns:
            # predict_proba, its log and decision_function (`encode`), never
            # predict; a multi-output fold once per lacking output, in output
            # order (its list comprehension over `estimator.classes_`).
            for t in range(tasks):  # driver-loop: over the wave's tasks (one test side each)
                var fold = (u0 + t) % used
                if failed[t] != UInt8(0) or plan.respond == MEMBER_OP_PREDICT:
                    continue
                if held_count[fold] > 0:
                    member_notice_add(metric_notes[t], MEMBER_NOTICE_FOLD_LACKING, Float64(held_count[fold]),
                                      Float64(k))
                if multi_cls and mlack[fold] != UInt8(0):
                    for j in range(y_cols):  # small-loop(the outputs)
                        var m = len(mheld[fold * y_cols + j])
                        if m > 0:
                            member_notice_add(metric_notes[t], MEMBER_NOTICE_FOLD_LACKING, Float64(m),
                                              Float64(plan.class_counts[j]))
        elif not member_score:
            for t in range(tasks):  # driver-loop: over the wave's tasks (one native metric program per scorer)
                var s = ((u0 + t // sides) % used) * sides + t % sides
                var fold_of = (u0 + t // sides) % used
                var started = perf_counter_ns()
                for j in range(nsc):  # small-loop(the scorers)
                    var at = t * nsc + j
                    if failed[at] != UInt8(0):
                        continue
                    var o = op_of[j]
                    var notes = List[Float64]()
                    # lane mf-score8 round b: a non-classifier's predict of
                    # integer ids is read as ids (`MselScorer.ids`); one
                    # flag word set on a copy of the row per score
                    var scored = rows_of[j]
                    var dt = out_dtypes[((u0 + t // sides) // used) * nops + o]
                    scored.ids = (not classifier and scored.op == MEMBER_OP_PREDICT
                                  and (dt == MEMBER_I32 or dt == MEMBER_I64))
                    if y_cols > 1:
                        # lane mf-rows8b (rows :3777 / :3779): the side's outputs and,
                        # for a multi-output classifier, its outputs' class counts
                        scored.outs = y_cols
                        scored.counts_at = Int(plan.class_counts.unsafe_ptr()) if multi_cls else 0
                        # lane mf-r9c: the fold estimator's held classes per
                        # output (a probability row reads its `classes_`)
                        scored.mheld_at = mheld_list
                        scored.fold = fold_of
                    if not classifier and scored.op == MEMBER_OP_PREDICT and msel_scorer_cluster(scored):
                        # lane mf-rows8b (row :3899): float predictions (1 float32, 2 float64)
                        scored.fpred = 1 if dt == MEMBER_F32 else (2 if dt == MEMBER_F64 else 0)
                    if scored.tier == 1:
                        # LANE mf-r9d: the other numeric_mode tier; a candidate
                        # whose fold rows are on the host with the fast tier
                        # raises the shared device='cpu' refusal per side
                        var cand = (u0 + t // sides) // used
                        scored.cpu_fast = cand < len(plan.other.cpu_fast) and plan.other.cpu_fast[cand] != UInt8(0)
                    try:
                        values[at] = _scorer_side(plan.other.route if scored.tier == 1 else plan.metric,
                                                  plan.other.cluster if scored.tier == 1 else plan.cluster,
                                                  plan.label_words, scored, y_dst[s],
                                                  pred_at[t * nops + o], side_count[s], k, pred_cols[t * nops + o],
                                                  positives[j], held_addr[fold_of], held_count[fold_of], notes)
                    except error:
                        errors[at] = String(error)
                        # lane mf-score6: a scorer that returns no number
                        # raises whatever the error_score (SCORER_NUMBER_WORDS)
                        failed[at] = UInt8(3) if errors[at].startswith(SCORER_NUMBER_WORDS) else UInt8(2)
                    for i in range(len(notes)):  # small-loop(one score's warnings: a triple or two)
                        metric_notes[t // sides].append(notes[i])
                if t % sides == 0:
                    task_secs[t] += Float64(perf_counter_ns() - started) / 1e9
        # The error rule, in unit order (then scorer order).
        for t in range(tasks):  # driver-loop: over the wave's tasks (their scores)
            var u = u0 + t // sides
            var side = t % sides
            for j in range(nsc):  # small-loop(the scorers)
                var value = values[t * nsc + j]
                if failed[t * nsc + j] != UInt8(0):
                    if not plan.tolerate or failed[t * nsc + j] == UInt8(3):
                        var words = errors[t * nsc + j].copy()
                        member_set_run_release(run)
                        _drop_kept(plan, res)
                        raise Error(words)
                    value = plan.error_value
                if side == 0:
                    test[j * units + u] = value
                else:
                    train[j * units + u] = value
            if side == 0:
                res.score_seconds[u] = task_secs[t]
        # The units' fit notices, then their scores' metric warnings, in
        # unit order; the states go.
        for v in range(width):  # driver-loop: over the wave's units (their fit notices)
            var words = List[Float64]()
            if run.states[v] != 0:
                try:
                    member_notices_take(run.plans[v], run.states[v], words)
                except:
                    pass
            for i in range(len(metric_notes[v])):  # small-loop(the unit's metric warnings: a few triples)
                words.append(metric_notes[v][i])
            if len(words) > 0:
                res.notices.append(words^)
            if plan.keep and run.failed[v] == UInt8(0) and run.states[v] != 0:
                # THE CV TOOLS: the unit's state is the caller's now. A kind
                # that borrows its fit buffers first owns its rows (THE
                # LIFETIME RULE: the wave's gathered train block goes below).
                if (plan.masks[(u0 + v) // used] & MEMBER_BORROWS_FIT) != 0:
                    try:
                        member_own(run.plans[v], run.states[v])
                    except error:
                        member_set_run_release(run)
                        _drop_kept(plan, res)
                        raise error^
                res.unit_states[u0 + v] = run.states[v]
                run.states[v] = 0
        member_set_run_release(run)
        _ = len(preds)  # the response buffers were read through the tasks above
        _ = len(pred_cols)
        _ = len(pred_cap)
        _ = len(values)
        _ = len(failed)
        _ = len(errors)
        _ = len(pred_at)
        _ = len(took)
        _ = len(task_secs)
        u0 = u1
    _ = len(out_dtypes)
    _ = len(out_widths)
    _ = len(ops)
    _ = len(log_bytes)  # read through `logs_at` by every wave above
    _ = len(rows_route)
    _ = len(side_x)  # the blocks were read through every wave above
    _ = len(side_y)
    _ = len(y_dst)
    _ = len(x_at)
    _ = len(side_start)
    _ = len(fold_held)  # the folds' held classes and re-coded codes, read by every wave above
    _ = len(fold_codes)
    _ = len(mheld)  # (lane mf-tail5) the multi-output folds' held classes, codes and counts, read by every wave
    _ = len(mcodes)
    _ = len(mcounts)
    _ = len(mlack)
    _ = len(dst_counts)
    _ = len(held_addr)
    _ = len(held_count)
    _ = len(side_rows)
    _ = len(side_count)
    _ = len(alias_of)
    _ = len(y_own)
    _ = len(fold_rows)
    _ = len(fold_counts)
    _ = len(columns)
    # 5. The tables, per scorer; the best by the refit scorer.
    if plan.refit_scorer < 0 or plan.refit_scorer >= nsc:
        _drop_kept(plan, res)
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
    _ = len(tr_at)  # keep-alive: the fold-mark closures read it through tr_at_at / mtr_at
    res.best = msel_rank_best(res.ranks)
    res.best_score = res.means[res.best]
    # 6. The refit of the best on every row (a set of one).
    if plan.refit:
        var refused_best = (plan.masks[res.best] >> SEARCH_REFUSED_SHIFT) - 1
        if refused_best >= 0:
            # lane mf-rows8b: the best is a refused candidate (an error_score
            # above every other mean): scikit-learn's refit fit raises its words
            _drop_kept(plan, res)
            raise Error(plan.label_words[refused_best].copy() if refused_best < len(plan.label_words)
                        else String("member ABI: a refused candidate's words are missing"))
        var best = plan.plans[res.best].copy()
        best.classes = k
        best.class_counts = plan.class_counts.copy()
        var mask = plan.masks[res.best]
        var run = MemberSetRun(plan.x, n, d, y_dtype, MEMBER_I32)
        run.cap = plan.cap
        run.y_cols = y_cols
        run.scope = plan.scope
        run.add(best^, plan.y, 0, 0, 0, 0, 0,
                UInt8(1) if (mask & MEMBER_CAN_TEST_ROWS) != 0 else UInt8(0),
                UInt8(1) if (mask & MEMBER_BORROWS_FIT) != 0 else UInt8(0), _set_word(mask))
        if plan.given != 0 and not classifier and (mask & MEMBER_READS_Y_GIVEN) != 0:
            run.given_target(plan.given, plan.given_dtype)
        var resident_refit = rows_route[res.best] == SEARCH_ROWS_RESIDENT
        if resident_refit:
            run.pin_tokens[0] = plan.pin  # every row: the whole X, the resident base
        _fit_wave(run)
        if resident_refit and run.pin_tokens[0] == 0:
            _note_kept(plan)
        var first = member_set_first_failure(run)
        if first >= 0:
            var words = run.errors[first].copy()
            member_set_run_release(run)
            _drop_kept(plan, res)
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
