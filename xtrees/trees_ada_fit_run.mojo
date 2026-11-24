# SPDX-License-Identifier: Apache-2.0
"""RUN of AdaBoostClassifier (SAMME) and AdaBoostRegressor (AdaBoost.R2) `fit`
over a tree member (lane mf-ada, 2026-11-23; every tree base lane mf-ada2,
2026-11-23). No Python import: `tools/python_free_dirs.sh` covers this
directory.

The door's UNPACK (bindings/trees_public_boost_door.mojo `ada_fit`) reads the
estimator, the labels or targets, the starting weights, the base tree's
configuration (`ForestConfig`, ONCE per fit; the rounds differ only in the
member seed, slot `ADA_SEED_SLOT`), the image's forest entry and capability
bits, the data session it opened, the class_weight rows and the predict route
words into `AdaFitPlan`. `ada_fit_run` runs every round on typed values and
returns the kept rounds' export handles with their weights and errors
(`AdaFitResult`); the door's PACK builds one member per kept round, in round
order, from those handles (`forest_export_handle`) and the three lists.

Three routes, the ones the door ran before:
  device rounds (XF_CAP_ADA): XF_ADA_OPEN once (the weights, labels or targets
      and class_weight rows stay on the session's device), then per round
      XF_FIT_ADA_ROUND, which fits the member, predicts, steps and rescales
      there; only the step's four words come back.
  host rounds: per round the weights times n as float32 (`xt_scale_to_f32`),
      times the class_weight rows (the label weights multiply) for a
      classifier; the member fit on the session (XF_FIT_SESSION) or, without
      one, on the column-major X (XF_FIT); its predict of the training X on the
      member's own route (`ForestState`: sequential walk, strict or resident
      snapshot); the SAMME or R2 step (`xt_samme_step` / `xt_r2_step`), the
      early stops and the rescale (`xt_scale`). The regressor draws its rows
      from the weights (`xt_weighted_sample`) and fits them on the session
      (or, without one, on their column-major gather: the same fit body).
  member rounds (`member_route`, lane mf-ada2): every other tree base
      (RandomForest*, ExtraTrees*, a DecisionTree* with splitter='random'
      (its inner one-tree ExtraTrees forest) or a regression criterion the
      session does not fit; a RandomForest* with oob_score=True too, lane
      mf-oob3: its member computes the out-of-bag outputs in every fit,
      parameter word 11, and the door's PACK writes them). Each round's member is a NEW state of the
      base's native member (core/forest_member_run.mojo through
      core/member_abi.mojo; the door admits the base ONCE,
      `forest_member_admit_common` with the caller's class rows), differing
      from the first only in its seed words: the classifier fits the
      row-major X with the round's float32 weights times the class_weight
      rows (one correctly rounded product per row, the multiply the forest
      door's `trees_weighted_rows` ran), the regressor its drawn rows'
      row-major gather and targets, after the deviance criteria's target
      check of those targets (`forest_target_check`, the forest door's
      words). The member's own predict of the training X (its route: the
      estimator's predict rule) feeds the step. The kept states go to the
      door's PACK (`forest_member_pack_common`), the fit notices of every
      fitted round with them (`member_notices_take`).

THE GIL (lane mf-ada2): the door RELEASES it around the WHOLE RUN, every
round. Nothing here touches Python: the forest entry is marked GIL-free
(`entry_ref_gil_free`, the door), so the data session operations and the
fits and walks run bare (the session registries have their own lock words,
core/forest_entry.mojo), the glue and the label weights driver never touch
the GIL, and the forest member is `MEMBER_GIL_FREE`.

Serial by nature: each round's weights are the previous round's rescaled
weights, so round j + 1 cannot start before round j's step. The parallelism
is inside each launch (the image's fit and predict, the glue kernels).
"""
from std.memory import bitcast
from core.entry_table import EntryRef
from core.host_finite_scan import host_all_finite_f32
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from core.member_abi import (
    MemberPlan, MemberMatrix, member_new, member_call, member_call_of, member_free,
    MEMBER_OP_FIT, MEMBER_OP_PREDICT, MEMBER_F32, MEMBER_I32,
)
from core.member_notices import member_notices_take
from core.forest_calls import (
    xf_ada_open, xf_fit_ada_round, xf_fit_session, xf_export_sizes, xf_export_into, xf_export_release, xf_predict,
    xf_resident_prepare, xf_resident_predict, xf_resident_labels, xf_strict_prepare, xf_strict_predict,
    xf_snapshot_release,
)
from core.forest_fit_request import ForestFitArgs, XF_RESIDENT_RELEASE, XF_STRICT_RELEASE
from core.forest_member_run import _argmax_rows, FM_HEAD
from core.label_weights_native_api import LabelWeightsDriver, LabelWeightsRequest, LABEL_WEIGHTS_MULTIPLY
from bindings.forest_config_run import (
    ForestConfig, forest_request, forest_target_need, forest_target_check, FOREST_TARGET_ANY, FOREST_TARGET_POISSON,
)
from bindings.forest_predict_run import (
    ForestState, forest_labels_route, FOREST_ROUTE_SEQUENTIAL, FOREST_ROUTE_STRICT,
)
from xtrees.glue_calls import (
    xt_gather_f32, xt_r2_step, xt_samme_step, xt_scale, xt_scale_to_f32, xt_transpose_f32, xt_weighted_sample,
)
from xtrees.trees_forest_run import ForestFit, forest_fit_core, forest_export_counts
from xtrees.trees_boost_run import _sub_seed

#: The member seed's parameter slot (`random_state`) in the RF layout
#: (`_rf_config`: rows, cols, classes, n_estimators, max_depth, max_leaves,
#: fraction, n_bins, min_samples_leaf, min_samples_split,
#: min_impurity_decrease, bootstrap, max_samples, random_state, ...). The
#: members of one fit differ only there (`_member_like`).
comptime ADA_SEED_SLOT = 13
#: The seed slot's three words in a forest member's MEMBER_OP_NEW parameters
#: (core/forest_member_run.mojo: `FM_HEAD` head words, then three per slot:
#: the int reading's Int64 bits, the float reading, the no-reading flag).
#: Slot 13 is `random_state` in the ET layout too (`_et_config`), so one word
#: serves RandomForest, DecisionTree, ExtraTrees and a random-splitter
#: tree's inner forest.
comptime ADA_MEMBER_SEED_WORD = FM_HEAD + 3 * ADA_SEED_SLOT

#: The step's status words (`xt_samme_step` / `xt_r2_step` / XF_FIT_ADA_ROUND)
#: (0.0: go on; 1.0: a perfect member, stop after keeping it; 2.0: worse than
#: random, stop without it)
comptime ADA_STEP_PERFECT = 1.0
comptime ADA_STEP_WORSE = 2.0


struct AdaFitPlan(Movable):
    """One AdaBoost fit as typed values (see the module docstring).

    Addresses: `x` the row-major float32 X (n x d), `xcm` its column-major
    copy (classifier only; the fit without a session reads it), `codes` the
    int32 class codes 0..k-1 (classifier), `y` the float32 targets
    (regressor), `class_rows` the float32 class_weight row weights (0: none),
    `weights` the float64 boosting weights (n words, rescaled in place). The
    door keeps every buffer alive until `ada_fit_run` returns."""
    var classifier: Bool
    var n: Int
    var d: Int
    var k: Int
    var rounds: Int
    var rate: Float64
    var seed: Int
    var loss: Int
    #: the `tree_base` word of the steps: the base is a DecisionTree*
    var tree_base: Int
    var config: ForestConfig
    var glue: EntryRef
    var forest: EntryRef
    var caps: Int
    var session: Int
    var device_rounds: Bool
    #: the member's predict engine words (`ForestState`)
    var fast: Bool
    var groves: Bool
    #: the squared_error criterion code (the session fits only that criterion)
    var squared: Int
    var x: Int
    var xcm: Int
    var codes: Int
    var y: Int
    var class_rows: Int
    var weights: Int
    var multiply: Optional[LabelWeightsDriver]
    #: member rounds (module docstring): the admitted base's member plan
    #: (its seed words rewritten per round) and the regressor's target need
    #: (`forest_target_need` of `config`, which then holds the base's (or
    #: its inner forest's) configuration for the check's words)
    var member_route: Bool
    var member_plan: MemberPlan
    var target_need: Int

    def __init__(out self, classifier: Bool, n: Int, d: Int, k: Int, rounds: Int, rate: Float64, seed: Int,
                 var config: ForestConfig, glue: EntryRef, forest: EntryRef, caps: Int, session: Int):
        self.classifier = classifier
        self.n = n
        self.d = d
        self.k = k
        self.rounds = rounds
        self.rate = rate
        self.seed = seed
        self.loss = 0
        self.tree_base = 1
        self.config = config^
        self.glue = glue.copy()
        self.forest = forest.copy()
        self.caps = caps
        self.session = session
        self.device_rounds = False
        self.fast = False
        self.groves = False
        self.squared = -1
        self.x = 0
        self.xcm = 0
        self.codes = 0
        self.y = 0
        self.class_rows = 0
        self.weights = 0
        self.multiply = None
        self.member_route = False
        self.member_plan = MemberPlan()
        self.target_need = FOREST_TARGET_ANY


struct AdaFitResult(Movable):
    """The kept rounds, in round order (member j holds `_sub_seed(seed, j)`):
    each round's owned export handle, its estimator weight and error.
    `launched` marks a fit without a session (its export keeps the forest
    door's export selection); the others export "into". On the member
    route `handles` are the kept member STATES (`members` True) and
    `notices` the fit notice words of every fitted member round, in round
    order (the DecisionTree rounds have none)."""
    var handles: List[Int]
    var alphas: List[Float64]
    var errors: List[Float64]
    var launched: Bool
    var members: Bool
    var notices: List[Float64]

    def __init__(out self):
        self.handles = List[Int]()
        self.alphas = List[Float64]()
        self.errors = List[Float64]()
        self.launched = False
        self.members = False
        self.notices = List[Float64]()


def ada_release_handles(forest: EntryRef, mut handles: List[Int]):
    """Release every owned export handle (non-raising: error paths)."""
    for i in range(len(handles)):  # small-loop(kept rounds: at most n_estimators handles, error path only)
        try:
            xf_export_release(forest, handles[i])
        except:
            pass
    handles = List[Int]()


def ada_release_members(plan: MemberPlan, mut states: List[Int]):
    """Free every kept member state (non-raising: error paths and the PACK)."""
    for i in range(len(states)):  # small-loop(kept rounds: at most n_estimators states, error path only)
        if states[i] != 0:
            try:
                var call = member_call_of(plan, states[i])
                member_free(plan, call)
            except:
                pass  # non-raising release: a state whose call cannot be rebuilt is left to the process
    states = List[Int]()


def _release_one(plan: AdaFitPlan, handle: Int):
    """One round's export handle or member state back (non-raising)."""
    if handle == 0:
        return
    if plan.member_route:
        try:
            var call = member_call_of(plan.member_plan, handle)
            member_free(plan.member_plan, call)
        except:
            pass
        return
    try:
        xf_export_release(plan.forest, handle)
    except:
        pass


def _member_seed(mut plan: AdaFitPlan, index: Int):
    """The round's member seed in the member plan's seed slot words (an
    exact nonnegative int: both readings, no flag; `_member_like`)."""
    var s = _sub_seed(plan.seed, index)
    plan.member_plan.params[ADA_MEMBER_SEED_WORD] = bitcast[DType.float64](Int64(s))
    plan.member_plan.params[ADA_MEMBER_SEED_WORD + 1] = Float64(s)
    plan.member_plan.params[ADA_MEMBER_SEED_WORD + 2] = 0.0


def _member_fit(mut plan: AdaFitPlan, index: Int, x: Int, y: Int, weights: Int) raises -> Int:
    """One round's member: a new state of the admitted base with the
    round's seed, fitted on the row-major float32 X at `x` (n x d), the
    class codes or targets at `y` and the float32 row weights at `weights`
    (0: none); its state handle (the caller frees it)."""
    _member_seed(plan, index)
    var call = member_new(plan.member_plan)
    try:
        call.x = MemberMatrix(x, plan.n, plan.d, MEMBER_F32)
        if plan.classifier:
            call.y = MemberMatrix(y, plan.n, 1, MEMBER_I32)
            call.classes = plan.k
            if weights != 0:
                call.weights = MemberMatrix(weights, plan.n, 1, MEMBER_F32)
        else:
            call.y = MemberMatrix(y, plan.n, 1, MEMBER_F32)
        _ = member_call(plan.member_plan.member, MEMBER_OP_FIT, call)
    except error:
        member_free(plan.member_plan, call)
        raise error^
    return call.state


def _member_predict(plan: AdaFitPlan, state: Int, dst: Int) raises:
    """The round's member predicts the training X on its own route: int32
    codes (classifier) or float32 values at `dst`."""
    var call = member_call_of(plan.member_plan, state)
    call.x = MemberMatrix(plan.x, plan.n, plan.d, MEMBER_F32)
    call.dst = MemberMatrix(dst, plan.n, 1, MEMBER_I32 if plan.classifier else MEMBER_F32)
    _ = member_call(plan.member_plan.member, MEMBER_OP_PREDICT, call)


def _extreme(src: Int, rows: Int, largest: Bool) -> Float64:
    """The smallest (or largest) of `rows` >= 1 finite float32 words at
    `src` (the forest door's `reduce_stat`): per-task extremes, then theirs
    (any order gives the same value), split over the host workers."""
    var tasks = host_predict_task_count_work(rows, rows)
    var chunk = host_predict_chunk(rows, tasks)
    var source = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=src)
    var tops = List[Float32](length=max(tasks, 1), fill=source[0])
    var top_words = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(tops.unsafe_ptr()))

    def task(unit: Int) {imm source, imm top_words, imm chunk, imm rows, imm largest}:
        var first = unit * chunk
        var last = min(first + chunk, rows)
        if first >= last:
            return
        var top = source[first]
        for i in range(first + 1, last):  # native data loop (one host task's rows)
            if (source[i] > top) if largest else (source[i] < top):
                top = source[i]
        top_words[unit] = top

    host_parallelize(task, tasks)
    var best = tops[0]
    for t in range(1, len(tops)):  # small-loop(tasks): one extreme per host task
        if (tops[t] > best) if largest else (tops[t] < best):
            best = tops[t]
    return Float64(best)


def _target_check(plan: AdaFitPlan, y: Int, rows: Int) raises:
    """The deviance criteria's check of the round's drawn targets, as the
    forest door's `_target_check` ran it on each member's y: finite, the
    smallest and, for Poisson when that passed, the largest value."""
    if plan.target_need == FOREST_TARGET_ANY:
        return
    var finite = host_all_finite_f32(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=y), rows)
    var lowest = 0.0
    var highest = 1.0
    if finite and rows > 0:
        lowest = _extreme(y, rows, False)
        if plan.target_need == FOREST_TARGET_POISSON and not lowest < 0.0:
            highest = _extreme(y, rows, True)
    forest_target_check(plan.config, finite, lowest, highest)


def _seed_slot(mut plan: AdaFitPlan, index: Int):
    """The round's member seed in the configuration's seed slot (an exact
    nonnegative int: both readings, no flag), as `_member_like` gives the
    round's member its own `random_state`."""
    var s = _sub_seed(plan.seed, index)
    plan.config.ints[ADA_SEED_SLOT] = s
    plan.config.floats[ADA_SEED_SLOT] = Float64(s)
    plan.config.bad[ADA_SEED_SLOT] = UInt8(0)


def _round_request(mut plan: AdaFitPlan, index: Int, rows: Int, x: Int, y: Int, weights: Int) raises -> ForestFitArgs:
    """The round's typed request: the configuration read once, the round's
    seed, the data slots and addresses."""
    _seed_slot(plan, index)
    var classes = plan.k if plan.classifier else 0
    return forest_request(plan.config, plan.forest, rows, plan.d, classes, x, y,
                          ForestFit(plan.classifier, weights=weights), False)


def _predict_train(plan: AdaFitPlan, handle: Int, dst: Int) raises:
    """The round's member predicts the training X (row-major) on its own
    route (the estimator's `ForestState` rule): int32 class codes at `dst`
    for a classifier (the resident fused labels where served, else the
    first-max argmax of the votes), float32 values for a regressor. The model
    is the round's export, copied out of the image for the walk and dropped
    after it; the handle stays owned for the PACK."""
    var sizes = xf_export_sizes(plan.forest, handle)
    var counts = forest_export_counts(sizes)
    var offsets = List[Int32](length=counts[0], fill=0)
    var colid = List[Int32](length=counts[1], fill=0)
    var quesval = List[Float32](length=counts[2], fill=0.0)
    var left = List[Int32](length=counts[3], fill=0)
    var leaves = List[Float32](length=counts[4], fill=0.0)
    var model: List[Int] = [Int(offsets.unsafe_ptr()), Int(colid.unsafe_ptr()), Int(quesval.unsafe_ptr()),
                            Int(left.unsafe_ptr()), Int(leaves.unsafe_ptr())]
    xf_export_into(plan.forest, sizes, model)
    var trees = sizes.trees
    var outputs = sizes.outputs
    var state = ForestState(plan.forest, 0, plan.caps, plan.fast, plan.groves, plan.d, trees, outputs)
    var n = plan.n
    var votes = List[Float32]()
    var target = dst
    var labels = plan.classifier and forest_labels_route(state)
    if plan.classifier and not labels:
        votes = List[Float32](length=n * outputs, fill=0.0)
        target = Int(votes.unsafe_ptr())
    var wrote = 0
    if state.route == FOREST_ROUTE_SEQUENTIAL:
        wrote = xf_predict(plan.forest, model, plan.x, target, n, plan.d, trees, outputs, not plan.classifier)
    else:
        var release_op = XF_STRICT_RELEASE if state.route == FOREST_ROUTE_STRICT else XF_RESIDENT_RELEASE
        var snapshot = 0
        if state.route == FOREST_ROUTE_STRICT:
            snapshot = xf_strict_prepare(plan.forest, model, plan.d, trees, outputs)
        else:
            snapshot = xf_resident_prepare(plan.forest, model, plan.d, trees, outputs, state.ordered)
        try:
            if labels:
                wrote = xf_resident_labels(plan.forest, snapshot, plan.x, target, n, plan.d, outputs)
            elif state.route == FOREST_ROUTE_STRICT:
                wrote = xf_strict_predict(plan.forest, snapshot, plan.x, target, n, plan.d, outputs)
            else:
                wrote = xf_resident_predict(plan.forest, snapshot, plan.x, target, n, plan.d, outputs)
        finally:
            xf_snapshot_release(plan.forest, release_op, snapshot)
    _ = len(model)
    _ = len(offsets)  # the walk and the snapshot above read the five arrays through `model`
    _ = len(colid)
    _ = len(quesval)
    _ = len(left)
    _ = len(leaves)
    if wrote != n:
        raise Error("RuntimeError: forest prediction wrote " + String(wrote) + " of " + String(n) + " rows")
    if plan.classifier and not labels:
        _argmax_rows(target, dst, n, outputs)
    _ = len(votes)  # the argmax above read the votes through `target`


# Removed 2026-11-24 (lane w3-fix-h): `_export_notice`, the per-round export-size
# read of a DecisionTree round's notice word. A DecisionTree* (splitter='best')
# fit has no bootstrap and no max_samples, so its only notice was the n_bins
# clamp, which is no longer raised (ensemble/rf_notices.mojo); these rounds
# have no notices.


def _keep(mut result: AdaFitResult, handle: Int, alpha: Float64, error: Float64):
    result.handles.append(handle)
    result.alphas.append(alpha)
    result.errors.append(error)


def _round_fit(mut plan: AdaFitPlan, index: Int, rows: List[Int32], yr: List[Float32], w32: List[Float32],
               wt: List[Float32], xg: List[Float32], xc: List[Float32], all_cols: List[Int32], stats: Int,
               mut launched: Bool) raises -> Int:
    """One round's member fit (device round, session fit or the plain fit);
    returns its export handle. A device round also wrote the step at
    `stats`."""
    var n = plan.n
    var last = 1 if index == plan.rounds - 1 else 0
    if plan.device_rounds:
        var request = _round_request(plan, index, n, 0, 0, 0)
        if not plan.classifier:
            if request.criterion != plan.squared:
                raise Error("a forest data session fits squared_error members")
            request.round_seed = plan.seed
            request.round = index
            request.loss = plan.loss
        request.session = plan.session
        request.stats = stats
        request.rate = plan.rate
        request.last = last
        return xf_fit_ada_round(plan.forest, plan.classifier, request)
    if plan.classifier:
        var w_addr = Int(w32.unsafe_ptr())
        if plan.class_rows != 0:
            w_addr = Int(wt.unsafe_ptr())
        var xcm = plan.xcm  # locals: `plan` is passed mutably beside them
        var codes = plan.codes
        var request = _round_request(plan, index, n, xcm, codes, w_addr)
        if plan.session > 0:
            request.session = plan.session
            var handle = xf_fit_session(plan.forest, True, request)
            _ = len(w32)  # the fit above read the weights through their address
            _ = len(wt)
            return handle
        launched = True
        var handle = forest_fit_core(plan.forest, True, False, request, False)
        _ = len(w32)
        _ = len(wt)
        return handle
    if plan.session > 0:
        # The regressor's session fit on the drawn rows of the session's X.
        var request = _round_request(plan, index, n, Int(rows.unsafe_ptr()), Int(yr.unsafe_ptr()), 0)
        if request.criterion != plan.squared:
            raise Error("a forest data session fits squared_error members")
        request.session = plan.session
        var handle = xf_fit_session(plan.forest, False, request)
        _ = len(rows)  # the fit above read the row ids and targets through their addresses
        _ = len(yr)
        return handle
    # Without a session (an image without XF_CAP_SESSION, a Mojo check): the
    # drawn rows gathered (`xt_gather_f32`) and laid out column-major
    # (`xt_transpose_f32`), the bytes a host session's row gather writes
    # (core/mojolearn_rf_host_impl.mojo `rf_host_session_rows_x`), then the
    # plain unweighted column-major fit, the body the session fit runs.
    _ = xt_gather_f32(plan.glue, plan.x, Int(rows.unsafe_ptr()), Int(all_cols.unsafe_ptr()), Int(xg.unsafe_ptr()),
                      n, plan.d, n, plan.d)
    _ = xt_transpose_f32(plan.glue, Int(xg.unsafe_ptr()), Int(xc.unsafe_ptr()), n, plan.d)
    var request = _round_request(plan, index, n, Int(xc.unsafe_ptr()), Int(yr.unsafe_ptr()), 0)
    if request.criterion != plan.squared:
        raise Error("a forest data session fits squared_error members")
    launched = True
    var handle = forest_fit_core(plan.forest, False, False, request, False)
    _ = len(rows)  # the gather and the fit above read these words through their addresses
    _ = len(all_cols)
    _ = len(xg)
    _ = len(xc)
    _ = len(yr)
    return handle


def ada_fit_run(mut plan: AdaFitPlan) raises -> AdaFitResult:
    """The boosting rounds of one AdaBoost fit (see the module docstring).
    Refuses a classifier whose first member is worse than random in the
    estimator's words. On a refusal every owned handle (or member state) is
    released."""
    var result = AdaFitResult()
    result.members = plan.member_route
    var n = plan.n
    var stats = List[Float64](length=4, fill=0.0)
    var stats_addr = Int(stats.unsafe_ptr())
    var sp = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=stats_addr)
    var w32 = List[Float32]()
    var wt = List[Float32]()
    var pred_codes = List[Int32]()
    var pred_values = List[Float32]()
    var rows = List[Int32]()
    var yr = List[Float32]()
    var col0 = List[Int32](length=1, fill=0)
    var xg = List[Float32]()
    var xc = List[Float32]()
    var all_cols = List[Int32]()
    if not plan.device_rounds:
        w32 = List[Float32](length=max(n, 1), fill=0.0)
        if plan.classifier:
            pred_codes = List[Int32](length=max(n, 1), fill=0)
            if plan.class_rows != 0:
                wt = List[Float32](length=max(n, 1), fill=0.0)
        else:
            pred_values = List[Float32](length=max(n, 1), fill=0.0)
            rows = List[Int32](length=max(n, 1), fill=0)
            yr = List[Float32](length=max(n, 1), fill=0.0)
            if plan.session == 0 or plan.member_route:
                xg = List[Float32](length=max(n * plan.d, 1), fill=0.0)
                if not plan.member_route:
                    xc = List[Float32](length=max(n * plan.d, 1), fill=0.0)
                all_cols = List[Int32](length=max(plan.d, 1), fill=0)
                for c in range(plan.d):  # small-loop(the d column ids of the gather, once per fit)
                    all_cols[c] = Int32(c)
    var handle = 0
    try:
        if plan.device_rounds:
            xf_ada_open(plan.forest, plan.session, plan.weights, plan.codes if plan.classifier else plan.y,
                        plan.class_rows if plan.classifier else 0, n, plan.k if plan.classifier else 0)
        for it in range(plan.rounds):  # driver-loop: over the boosting rounds (serial: each round's weights are the last round's)
            var last = 1 if it == plan.rounds - 1 else 0
            if not plan.device_rounds:
                if plan.classifier:
                    _ = xt_scale_to_f32(plan.glue, plan.weights, Int(w32.unsafe_ptr()), n, Float64(n))
                    if plan.class_rows != 0:
                        _ = plan.multiply.value().run(LABEL_WEIGHTS_MULTIPLY, LabelWeightsRequest(
                            Int(w32.unsafe_ptr()), plan.class_rows, Int(wt.unsafe_ptr()), n, 0))
                else:
                    _ = xt_weighted_sample(plan.glue, plan.weights, Int(rows.unsafe_ptr()), n, n, plan.seed, it)
                    _ = xt_gather_f32(plan.glue, plan.y, Int(rows.unsafe_ptr()), Int(col0.unsafe_ptr()),
                                      Int(yr.unsafe_ptr()), n, 1, n, 1)
            if plan.member_route:
                if plan.classifier:
                    var w_addr = Int(wt.unsafe_ptr()) if plan.class_rows != 0 else Int(w32.unsafe_ptr())
                    var x_addr = plan.x  # locals: `plan` is passed mutably beside them
                    var codes_addr = plan.codes
                    handle = _member_fit(plan, it, x_addr, codes_addr, w_addr)
                else:
                    # the drawn rows' row-major gather (the forest door's
                    # `_gather`), checked and fitted unweighted
                    _ = xt_gather_f32(plan.glue, plan.x, Int(rows.unsafe_ptr()), Int(all_cols.unsafe_ptr()),
                                      Int(xg.unsafe_ptr()), n, plan.d, n, plan.d)
                    _target_check(plan, Int(yr.unsafe_ptr()), n)
                    handle = _member_fit(plan, it, Int(xg.unsafe_ptr()), Int(yr.unsafe_ptr()), 0)
                member_notices_take(plan.member_plan, handle, result.notices)
                if plan.classifier:
                    _member_predict(plan, handle, Int(pred_codes.unsafe_ptr()))
                    _ = xt_samme_step(plan.glue, plan.weights, Int(pred_codes.unsafe_ptr()), plan.codes, stats_addr,
                                      n, plan.k, plan.rate, last, plan.tree_base)
                else:
                    _member_predict(plan, handle, Int(pred_values.unsafe_ptr()))
                    _ = xt_r2_step(plan.glue, plan.weights, Int(pred_values.unsafe_ptr()), plan.y, stats_addr,
                                   n, plan.loss, plan.rate, last, plan.tree_base)
            else:
                var launched = False
                handle = _round_fit(plan, it, rows, yr, w32, wt, xg, xc, all_cols, stats_addr, launched)
                if launched:
                    result.launched = True
            if not plan.device_rounds and not plan.member_route:
                if plan.classifier:
                    _predict_train(plan, handle, Int(pred_codes.unsafe_ptr()))
                    _ = xt_samme_step(plan.glue, plan.weights, Int(pred_codes.unsafe_ptr()), plan.codes, stats_addr,
                                      n, plan.k, plan.rate, last, plan.tree_base)
                else:
                    _predict_train(plan, handle, Int(pred_values.unsafe_ptr()))
                    _ = xt_r2_step(plan.glue, plan.weights, Int(pred_values.unsafe_ptr()), plan.y, stats_addr,
                                   n, plan.loss, plan.rate, last, plan.tree_base)
            var status = sp[0]
            var total = sp[3]
            if status == ADA_STEP_WORSE:
                if len(result.handles) == 0:
                    if plan.classifier:
                        raise Error("BaseClassifier in AdaBoostClassifier ensemble is worse than random,"
                                    + " ensemble can not be fit.")
                    # AdaBoost.R2 keeps its first member with weight 0.
                    _keep(result, handle, 0.0, sp[2])
                    handle = 0
                    break
                _release_one(plan, handle)
                handle = 0
                break
            _keep(result, handle, sp[1], sp[2])
            handle = 0
            if status == ADA_STEP_PERFECT or not total > 0:
                break
            # A device round rescaled its resident weights by the total.
            if not plan.device_rounds and it < plan.rounds - 1:
                _ = xt_scale(plan.glue, plan.weights, n, total)
    except error:
        _release_one(plan, handle)
        if plan.member_route:
            ada_release_members(plan.member_plan, result.handles)
        else:
            ada_release_handles(plan.forest, result.handles)
        raise error^
    _ = len(stats)  # every step above wrote its four words through `stats_addr`
    _ = len(w32)
    _ = len(wt)
    _ = len(pred_codes)
    _ = len(pred_values)
    _ = len(rows)
    _ = len(yr)
    _ = len(col0)
    _ = len(xg)
    _ = len(xc)
    _ = len(all_cols)
    return result^
