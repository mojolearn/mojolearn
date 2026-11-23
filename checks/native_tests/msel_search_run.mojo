# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""THE TYPED SEARCH RUN (core/msel_search_run.mojo `msel_search_run`, lane mf-search4), in a process with no
Python, against the door's `cv_results_` words computed here independently: GridSearchCV over three
LogisticRegression candidates (l2 1.0, 0.1, 10.0; core/logistic_member.mojo on the CPU QN entry), cv = 3
(StratifiedKFold over the codes: the folds of xtrees/glue_calls.mojo `xt_device_folds`, which the door's
splitter writes for an int cv of a classifier).

THE ORACLE (the door's `_fit_and_score` per unit, `msel_search_fit`'s tables): for every candidate and fold the
member fitted directly on the fold's train rows (gathered here) and its own score (scoring None) or
`accuracy_score` of its predicted codes counted here (scoring 'accuracy') on the fold's test rows; the means and
deviations (`fsum / len`, the population deviation about the mean), the 'min' ranks, the best (the first of
rank 1), and the refit: the best candidate fitted on every row.

1. scoring None: every split score, mean, deviation, rank, the best index and score equal; the refit state's
   predict_proba equals the oracle's refit word for word.
2. scoring 'accuracy' (the native metric program over `metrics_program_host_v1`), return_train_score: every test
   and train split score equals the counted fraction; the best is the oracle's.
3. error_score: a candidate whose fit refuses (an l1 word with K = 3 classes: the member's NotImplementedError) raises its words with
   'raise', and with a number takes that value in every split while the others still score.

4. (lane mf-metric4) one named scorer, 'f1_macro' (predicted codes), then 'neg_log_loss' (predict_proba, float64
   narrowed to float32): every test split score equals the scorer's metric (core/msel_scorer_run.mojo
   `msel_scorer_value`, the builder the metric's door calls) over the member's own response on the fold's test rows.
5. (lane mf-metric4) multi-metric ['accuracy', 'balanced_accuracy', 'neg_log_loss'] with refit='balanced_accuracy'
   (refit_scorer 1) and refit=False: each scorer's table equals its oracle, the best follows scorer 1's ranks, and
   no refit state is made.
6. (lane mf-search4b) GIVEN FOLDS: the int cv's folds handed back as given int32 row views in REVERSED fold
   order (`fold_train_at` / `fold_test_at`, the path a splitter or an index list takes): every 'accuracy' split
   score is the oracle's of the reversed fold. THE BINARY SCORERS (`label_binary_run`, average='binary'): truth
   [1, 0, 1, 1] and predictions [1, 1, 0, 1] give precision, recall and f1 2/3 for positive code 1 and f1 0.0
   for positive code 0 (counted by hand), and three observed classes refuse in the door's words.

7. (lane mf-score5) multi-metric ['average_precision', 'neg_brier_score', 'top_k_accuracy',
   'normalized_mutual_info_score', 'adjusted_mutual_info_score'] (predict_proba / predict responses, K = 3): every
   scorer's split scores equal `msel_scorer_value` over the member's own response.

8. (lane mf-search5) THE RESIDENT ROUTE (`msel_search_rows`): with no scope every candidate's sides are gathered
   blocks; in a scope (word 0) with a pin, a candidate whose entry and kind read row numbers is INDEXED, and
   RESIDENT (offered the pin) when it keeps X resident and runs on the image of the first such candidate; one on
   another route stays INDEXED. Then the RUN in that scope over WRAPPED candidates (`_indexed_member_v2`: the
   logistic member behind an entry that answers MEMBER_CAN_ROW_INDEX | MEMBER_CAN_SCOPE, gathers the indexed rows
   itself and plays a scope cache that takes the first offered pin): every 'accuracy' test and train split score,
   and every member score (scoring None), equals the gathered run's over the bare members; every fit and score
   reads the whole X plus its rows; the pin is offered on every unit's fit and side and the refit (28 calls: 9
   fits, 18 sides, the refit) and the token is written at `kept_at` once; a mixed set (resident, indexed without
   the scope bit, gathered) gives the same scores and offers the pin to the resident candidate only.

9. (lane mf-cluster5) A CLUSTERER BASE: GridSearchCV over two KMeans candidates (n_clusters 2 and 3,
   core/kmeans_member.mojo on the CPU KMeans entry), classes 0 (KFold over the rows), y the class codes as float32
   words, multi-metric ['adjusted_rand_score', 'v_measure_score', 'normalized_mutual_info_score', 'rand_score']
   (the label-pair scorers over the CPU column's cluster entry, metrics/host/cluster_entry_host.mojo): every split
   score equals `msel_scorer_value` over KMeans fitted directly on the fold's train rows and its predicted cluster
   ids on the test rows; a clusterer scored by 'r2' beside them refuses.

10. (lane mf-msel5) THE CV TOOLS: one LogisticRegression candidate over a learning curve's given folds (each
   fold's train rows at two sizes, its first N/3 rows and all of them; size-major, so the two units of a fold
   share one test side), 'accuracy', return_train_score, `keep` and `indices`: every test and train split score
   equals the member fitted directly on that prefix and counted here; each unit has its fit and score seconds
   (finite, not negative); each kept state's predict_proba on X equals the direct fit's word for word; the fold
   rows come back as given. Then `respond` = predict_proba over the int cv (cross_val_predict): each unit's
   response words equal the direct fit's float64 probabilities on its test rows (K columns, MEMBER_F64).
11. (lane mf-msel5) AP-ABSENT: two classes whose labels are 0 and 2 (label words "0", "2"), scoring
   'average_precision' with the positive word SCORER_POSITIVE_ABSENT: with error_score='raise' the RUN raises
   scikit-learn's words "ValueError: pos_label=1 is not a valid label: It should be one of [0 2]"; with a number
   every split score is that number.

FAIL-PROOF EDITS (each must make a case fail): `side_rows[f * sides] = rows_at + 4 * (f * n)` (case 1: the train
rows are scored); `p.classes = k - 1` in the wave (cases 1, 2: the units' fits learn the wrong count);
`narrow[i] = Int32(wide[i]) + 1` in `_respond` (cases 2, 4); `value = 0.0` for a failed task (case 3);
`narrow[i] = Float32(wide[i] * 0.5)` in `_respond` (cases 4, 5: the probabilities); `var r = 0` in place of
`plan.refit_scorer` in the tables (case 5); `op_of.append(0)` for every scorer (case 5: log loss reads codes);
`used = 0` in the given-folds branch, or `te_at.append(plan.fold_train_at[f])` (case 6); `seen[positive] = 0`
removed, or `McEpi(2, 2, LBL_MACRO, ...)` in `label_binary_run` (case 6: the positive's value); in
core/msel_scorer_run.mojo `cluster_mi_run(..., row.metric != _CL_AMI, ...)` (case 7: NMI and AMI swapped); in
core/msel_search_run.mojo `out[c] = SEARCH_ROWS_RESIDENT` without the `_same_image` test, or
`call.row_index = MemberMatrix(side.index, side.count - 1, 1, MEMBER_I32)` in `_side_set`, or
`index = _IntP(unsafe_from_address=side_rows_at)[s]` taken from the train side for a test task, or
`_note_kept` removed after the fit wave (case 8); `var labels_out = ops[o] == MEMBER_OP_PREDICT and classifier`
in `msel_search_run` (case 9: KMeans' int32 ids refused as a float response), or `entry_ref_unresolved()` passed
for `plan.cluster` in `_scorer_side`'s call (case 9: the label-pair scorers refuse); in core/msel_search_run.mojo
`alias_of[f * sides] = g * sides` written for a test side of DIFFERENT rows (drop the `_same_rows` test), or
`res.unit_states[u0 + v] = run.states[v]` without `run.states[v] = 0` (the release frees it: case 10's kept
predict_proba reads a freed state), or `tr.append(Int32(0))` in place of the row copy (case 10: indices), or
`_widen(pred_at[at], side_count[s], dtype)` (case 10: one column only); `shown += " "` removed (case 11: the
words).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require
from checks.numerics import GLOBAL_NUMERIC_MODE
from bindings.arena_prog_run import MetricRoute
from bindings.msel_search_stats_run import msel_row_means, msel_row_stds, msel_rank_scores, msel_rank_best
from core.device_route import DEVICE_CPU
from core.entry_table import EntryRef
from core.logistic_member import logistic_member_v2
from core.logistic_native_api import QnNativeDriver
from core.logistic_route import LogisticRoute, logistic_route_pack
from core.member_abi import (
    MemberCall, MemberMatrix, MemberPlan, MemberRef, member_call, member_call_of, member_new, member_free,
    member_kind_capabilities, MEMBER_OP_CAPABILITIES, MEMBER_OP_FIT, MEMBER_OP_SCORE, MEMBER_OP_PREDICT,
    MEMBER_OP_PREDICT_PROBA, MEMBER_F64, MEMBER_I32, MEMBER_I64, MEMBER_CAN_ROW_INDEX, MEMBER_CAN_SCOPE,
    member_call_from_spans,
)
from core.msel_search_run import (
    MselSearchPlan, msel_search_run, msel_search_rows, SEARCH_SCORE_MEMBER, SEARCH_SCORE_ACCURACY,
    SEARCH_ROWS_GATHERED, SEARCH_ROWS_INDEXED, SEARCH_ROWS_RESIDENT,
)
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.msel_scorer_run import msel_scorer_code, msel_scorer_row, msel_scorer_value, SCORER_POSITIVE_ABSENT
from bindings.mcls_classify_run import label_binary_run, LBL_PRECISION, LBL_RECALL, LBL_F1
from core.trees_entry_table import TreesEntryKey
from glm.host.logistic_native_host import qn_host_native_v1
from x_metrics.descriptor_host import metrics_program_host_v1
from xtrees.glue_runs import publish_x_trees_glue
from xtrees.glue_calls import xt_ref, xt_device_folds
from std.memory import bitcast
from core.device_route import DeviceRoute
from core.kmeans_member import kmeans_member_v2
from core.kmeans_native_host import kmeans_host_native_v1
from core.kmeans_native_api import KMEANS_CAP_HOST
from core.metrics_cluster_table import cluster_ops_publish, cluster_ops_entry
from metrics.host.cluster_entry_host import METRICS_CLUSTER_HOST_API, metrics_cluster_host_v1

comptime N = 60
comptime D = 3
comptime K = 3
comptime F = 3
comptime CASES = 11


def _glue() raises -> EntryRef:
    return xt_ref(TreesEntryKey(Int(GLOBAL_NUMERIC_MODE), 0))


def _x() -> List[Float32]:
    var x = List[Float32](length=N * D, fill=0.0)
    for i in range(N):  # small-loop(fixture)
        for j in range(D):  # small-loop(fixture)
            x[i * D + j] = Float32((i * 7 + j * 5) % 11) * 0.25 - 1.0 + Float32(i % 3) * 0.5 * Float32(j + 1)
    return x^


def _codes() -> List[Int32]:
    var y = List[Int32](length=N, fill=Int32(0))
    for i in range(N):  # small-loop(fixture: every class has N / 3 rows)
        y[i] = Int32((i + i // 5) % K)
    return y^


def _logistic(l2: Float64, l1: Float64) raises -> MemberPlan:
    var plan = MemberPlan()
    plan.route = logistic_route_pack(LogisticRoute(DEVICE_CPU, Int(GLOBAL_NUMERIC_MODE), 0, False, False,
                                                   QnNativeDriver(qn_host_native_v1)))
    plan.member = MemberRef(EntryRef(logistic_member_v2, 0), 0)
    var probe = MemberCall(plan.route)
    plan.member.capabilities = member_call(plan.member, MEMBER_OP_CAPABILITIES, probe)
    plan.params = [0.0, 1.0, l1, l2, 100.0, 20.0, 10.0, 0.0, 1e-4, 1e-6]
    plan.classes = K
    return plan^


def _candidates(bad: Bool) -> List[MemberPlan]:
    var out = List[MemberPlan]()
    try:
        out.append(_logistic(1.0, 0.0))
        out.append(_logistic(0.1, 0.5 if bad else 0.0))
        out.append(_logistic(10.0, 0.0))
    except:
        pass
    return out^


def _masks(plans: List[MemberPlan]) raises -> List[Int]:
    var out = List[Int]()
    for c in range(len(plans)):  # small-loop(fixture: each candidate kind's mask, as the admission asks it)
        out.append(member_kind_capabilities(plans[c]))
    return out^


struct _Oracle(Movable):
    var test: List[List[Float64]]
    var train: List[List[Float64]]

    def __init__(out self):
        self.test = List[List[Float64]]()
        self.train = List[List[Float64]]()


def _rows_of(x: List[Float32], y: List[Int32], rows: Int, count: Int, mut xs: List[Float32], mut ys: List[Int32]):
    var r = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=rows)
    xs = List[Float32](length=count * D, fill=0.0)
    ys = List[Int32](length=count, fill=Int32(0))
    for i in range(count):  # small-loop(oracle: one fold side's rows)
        var at = Int(r[i])
        for j in range(D):  # small-loop(oracle)
            xs[i * D + j] = x[at * D + j]
        ys[i] = y[at]


def _side_score(plan: MemberPlan, state: Int, xs: List[Float32], ys: List[Int32], count: Int,
                scoring: Int) raises -> Float64:
    var call = member_call_of(plan, state)
    call.x = MemberMatrix(Int(xs.unsafe_ptr()), count, D)
    if scoring == SEARCH_SCORE_MEMBER:
        call.y = MemberMatrix(Int(ys.unsafe_ptr()), count, 1, MEMBER_I32)
        _ = member_call(plan.member, MEMBER_OP_SCORE, call)
        return call.value
    var wide = List[Int64](length=count, fill=Int64(0))
    call.dst = MemberMatrix(Int(wide.unsafe_ptr()), count, 1, MEMBER_I64)
    _ = member_call(plan.member, MEMBER_OP_PREDICT, call)
    var hits = 0
    for i in range(count):  # small-loop(oracle: accuracy_score, counted)
        if Int(wide[i]) == Int(ys[i]):
            hits += 1
    _ = wide^
    return Float64(hits) / Float64(count)


def _oracle(plans: List[MemberPlan], x: List[Float32], y: List[Int32], scoring: Int, train: Bool,
            tolerate: Bool, value: Float64) raises -> _Oracle:
    var counts = List[Int32](length=F + 1, fill=Int32(0))
    var rows = List[Int32](length=F * N, fill=Int32(0))
    var yy = y.copy()
    _ = xt_device_folds(_glue(), Int(yy.unsafe_ptr()), Int(rows.unsafe_ptr()), Int(counts.unsafe_ptr()), N, F, K)
    _ = len(yy)  # the fold program read the codes through their address
    var out = _Oracle()
    for c in range(len(plans)):  # small-loop(oracle: candidates)
        var test = List[Float64]()
        var tr = List[Float64]()
        for f in range(F):  # small-loop(oracle: folds)
            var t = Int(counts[f])
            var base = Int(rows.unsafe_ptr()) + 4 * (f * N)
            var xs = List[Float32]()
            var ys = List[Int32]()
            _rows_of(x, y, base, N - t, xs, ys)
            var call = member_new(plans[c])
            call.x = MemberMatrix(Int(xs.unsafe_ptr()), N - t, D)
            call.y = MemberMatrix(Int(ys.unsafe_ptr()), N - t, 1, MEMBER_I32)
            call.classes = K
            var fitted = True
            try:
                _ = member_call(plans[c].member, MEMBER_OP_FIT, call)
            except:
                fitted = False
            if not fitted:
                require(tolerate, "a failing candidate with error_score='raise' raises")
                test.append(value)
                tr.append(value)
                member_free(plans[c], call)
                continue
            var xt = List[Float32]()
            var yt = List[Int32]()
            _rows_of(x, y, base + 4 * (N - t), t, xt, yt)
            test.append(_side_score(plans[c], call.state, xt, yt, t, scoring))
            if train:
                tr.append(_side_score(plans[c], call.state, xs, ys, N - t, scoring))
            member_free(plans[c], call)
            _ = len(xs)  # the member's fit and calls read the fold rows through their addresses
            _ = len(ys)
            _ = len(xt)
            _ = len(yt)
        out.test.append(test^)
        out.train.append(tr^)
    _ = rows^
    _ = counts^
    return out^


def _plan(var plans: List[MemberPlan], x: List[Float32], y: List[Int32], scoring: Int) raises -> MselSearchPlan:
    var masks = _masks(plans)
    var plan = MselSearchPlan(plans^, masks^, Int(x.unsafe_ptr()), N, D, Int(y.unsafe_ptr()), K, F, _glue(),
                              MetricRoute(metrics_program_host_v1, False, False))
    plan.scoring = scoring
    return plan^


def check_member_score() raises:
    var x = _x()
    var y = _codes()
    var plans = _candidates(False)
    var want = _oracle(plans, x, y, SEARCH_SCORE_MEMBER, False, False, 0.0)
    var plan = _plan(plans.copy(), x, y, SEARCH_SCORE_MEMBER)
    var res = msel_search_run(plan)
    require(res.n_splits == F, "the folds")
    for c in range(len(plans)):  # small-loop(check: candidates)
        for f in range(F):  # small-loop(check: folds)
            require(res.test[c][f] == want.test[c][f], "split" + String(f) + "_test_score of candidate " + String(c))
    var means = msel_row_means(want.test)
    var stds = msel_row_stds(want.test, means)
    var ranks = msel_rank_scores(means)
    for c in range(len(plans)):  # small-loop(check: candidates)
        require(res.means[c] == means[c] and res.stds[c] == stds[c] and res.ranks[c] == ranks[c], "the tables")
    var best = msel_rank_best(ranks)
    require(res.best == best and res.best_score == means[best], "best_index_ / best_score_")
    # The refit: the best candidate on every row.
    var call = member_new(plans[best])
    call.x = MemberMatrix(Int(x.unsafe_ptr()), N, D)
    call.y = MemberMatrix(Int(y.unsafe_ptr()), N, 1, MEMBER_I32)
    call.classes = K
    _ = member_call(plans[best].member, MEMBER_OP_FIT, call)
    var a = List[Float64](length=N * K, fill=0.0)
    var b = List[Float64](length=N * K, fill=1.0)
    var p = member_call_of(plans[best], call.state)
    p.x = MemberMatrix(Int(x.unsafe_ptr()), N, D)
    p.dst = MemberMatrix(Int(a.unsafe_ptr()), N, K, MEMBER_F64)
    _ = member_call(plans[best].member, MEMBER_OP_PREDICT_PROBA, p)
    var q = member_call_of(plans[best], res.best_state)
    q.x = MemberMatrix(Int(x.unsafe_ptr()), N, D)
    q.dst = MemberMatrix(Int(b.unsafe_ptr()), N, K, MEMBER_F64)
    _ = member_call(plans[best].member, MEMBER_OP_PREDICT_PROBA, q)
    for i in range(N * K):  # small-loop(check)
        require(a[i] == b[i], "the refit's predict_proba")
    member_free(plans[best], call)
    var r = MemberCall(plans[best].route, res.best_state)
    member_free(plans[best], r)
    _ = x^
    _ = y^


def check_accuracy() raises:
    var x = _x()
    var y = _codes()
    var plans = _candidates(False)
    var want = _oracle(plans, x, y, SEARCH_SCORE_ACCURACY, True, False, 0.0)
    var plan = _plan(plans.copy(), x, y, SEARCH_SCORE_ACCURACY)
    plan.return_train = True
    var res = msel_search_run(plan)
    for c in range(len(plans)):  # small-loop(check: candidates)
        for f in range(F):  # small-loop(check: folds)
            require(res.test[c][f] == want.test[c][f], "accuracy split" + String(f) + " of candidate " + String(c))
            require(res.train[c][f] == want.train[c][f], "train accuracy split" + String(f))
    require(res.best == msel_rank_best(msel_rank_scores(msel_row_means(want.test))), "best_index_")
    var r = MemberCall(plans[res.best].route, res.best_state)
    member_free(plans[res.best], r)
    _ = x^
    _ = y^


def check_error_score() raises:
    var x = _x()
    var y = _codes()
    var plans = _candidates(True)
    var raised = False
    try:
        var plan = _plan(plans.copy(), x, y, SEARCH_SCORE_MEMBER)
        _ = msel_search_run(plan)
    except:
        raised = True
    require(raised, "error_score='raise': the failing candidate's words")
    var want = _oracle(plans, x, y, SEARCH_SCORE_MEMBER, False, True, -3.5)
    var plan = _plan(plans.copy(), x, y, SEARCH_SCORE_MEMBER)
    plan.tolerate = True
    plan.error_value = -3.5
    var res = msel_search_run(plan)
    for c in range(len(plans)):  # small-loop(check: candidates)
        for f in range(F):  # small-loop(check: folds)
            require(res.test[c][f] == want.test[c][f], "error_score split" + String(f) + " of candidate " + String(c))
    var r = MemberCall(plans[res.best].route, res.best_state)
    member_free(plans[res.best], r)
    _ = x^
    _ = y^


def _scorer_oracle(plans: List[MemberPlan], x: List[Float32], y: List[Int32],
                   codes: List[Int]) raises -> List[List[List[Float64]]]:
    """Per scorer, candidate and fold: the scorer's metric over the member's own response on the fold's test
    rows (the member fitted directly on the fold's train rows, as `_oracle`)."""
    var route = MetricRoute(metrics_program_host_v1, False, False)
    var counts = List[Int32](length=F + 1, fill=Int32(0))
    var rows = List[Int32](length=F * N, fill=Int32(0))
    var yy = y.copy()
    _ = xt_device_folds(_glue(), Int(yy.unsafe_ptr()), Int(rows.unsafe_ptr()), Int(counts.unsafe_ptr()), N, F, K)
    _ = len(yy)  # the fold program read the codes through their address
    var out = List[List[List[Float64]]]()
    for j in range(len(codes)):  # small-loop(oracle: scorers)
        out.append(List[List[Float64]]())
        for _ in range(len(plans)):  # small-loop(oracle: candidates)
            out[j].append(List[Float64]())
    for c in range(len(plans)):  # small-loop(oracle: candidates)
        for f in range(F):  # small-loop(oracle: folds)
            var t = Int(counts[f])
            var base = Int(rows.unsafe_ptr()) + 4 * (f * N)
            var xs = List[Float32]()
            var ys = List[Int32]()
            _rows_of(x, y, base, N - t, xs, ys)
            var call = member_new(plans[c])
            call.x = MemberMatrix(Int(xs.unsafe_ptr()), N - t, D)
            call.y = MemberMatrix(Int(ys.unsafe_ptr()), N - t, 1, MEMBER_I32)
            call.classes = K
            _ = member_call(plans[c].member, MEMBER_OP_FIT, call)
            var xt = List[Float32]()
            var yt = List[Int32]()
            _rows_of(x, y, base + 4 * (N - t), t, xt, yt)
            # the two responses, in the scorers' element types
            var wide = List[Int64](length=t, fill=Int64(0))
            var pc = member_call_of(plans[c], call.state)
            pc.x = MemberMatrix(Int(xt.unsafe_ptr()), t, D)
            pc.dst = MemberMatrix(Int(wide.unsafe_ptr()), t, 1, MEMBER_I64)
            _ = member_call(plans[c].member, MEMBER_OP_PREDICT, pc)
            var labels = List[Int32](length=t, fill=Int32(0))
            for i in range(t):  # small-loop(oracle)
                labels[i] = Int32(wide[i])
            var proba64 = List[Float64](length=t * K, fill=0.0)
            var qc = member_call_of(plans[c], call.state)
            qc.x = MemberMatrix(Int(xt.unsafe_ptr()), t, D)
            qc.dst = MemberMatrix(Int(proba64.unsafe_ptr()), t, K, MEMBER_F64)
            _ = member_call(plans[c].member, MEMBER_OP_PREDICT_PROBA, qc)
            var proba = List[Float32](length=t * K, fill=0.0)
            for i in range(t * K):  # small-loop(oracle: the float32 words the metric doors admit)
                proba[i] = Float32(proba64[i])
            for j in range(len(codes)):  # small-loop(oracle: scorers)
                var row = msel_scorer_row(codes[j])
                var resp = Int(proba.unsafe_ptr()) if row.op == MEMBER_OP_PREDICT_PROBA else Int(labels.unsafe_ptr())
                var cols = K if row.op == MEMBER_OP_PREDICT_PROBA else 1
                var notes = List[Float64]()
                out[j][c].append(msel_scorer_value(route, row, Int(yt.unsafe_ptr()), resp, t, K, cols, -1, notes))
            _ = len(labels)
            _ = len(proba)
            member_free(plans[c], call)
            _ = len(xs)  # the member's fit and calls read the fold rows through their addresses
            _ = len(ys)
            _ = len(xt)
            _ = len(yt)
    _ = rows^
    _ = counts^
    return out^


def check_named_scorer() raises:
    var x = _x()
    var y = _codes()
    var plans = _candidates(False)
    var names: List[String] = ["f1_macro", "neg_log_loss"]
    for i in range(len(names)):  # small-loop(check: the two scorers, one search each)
        var code = msel_scorer_code(names[i], False)
        require(code >= 2, names[i] + " is a served scorer")
        var want = _scorer_oracle(plans, x, y, [code])
        var plan = _plan(plans.copy(), x, y, code)
        var res = msel_search_run(plan)
        for c in range(len(plans)):  # small-loop(check: candidates)
            for f in range(F):  # small-loop(check: folds)
                require(res.test[c][f] == want[0][c][f], names[i] + " split" + String(f) + " of candidate " + String(c))
        require(res.best == msel_rank_best(msel_rank_scores(msel_row_means(want[0]))), names[i] + " best_index_")
        var r = MemberCall(plans[res.best].route, res.best_state)
        member_free(plans[res.best], r)
    _ = x^
    _ = y^


def check_multi_metric() raises:
    var x = _x()
    var y = _codes()
    var plans = _candidates(False)
    var codes: List[Int] = [msel_scorer_code("accuracy", False), msel_scorer_code("balanced_accuracy", False),
                            msel_scorer_code("neg_log_loss", False)]
    var want = _scorer_oracle(plans, x, y, codes)
    var plan = _plan(plans.copy(), x, y, SEARCH_SCORE_MEMBER)
    plan.scorers = codes.copy()
    plan.refit_scorer = 1
    plan.refit = False
    var res = msel_search_run(plan)
    require(len(res.scorer_test) == 3, "one table per scorer")
    for j in range(3):  # small-loop(check: scorers)
        for c in range(len(plans)):  # small-loop(check: candidates)
            for f in range(F):  # small-loop(check: folds)
                require(res.scorer_test[j][c][f] == want[j][c][f],
                        "scorer " + String(j) + " split" + String(f) + " of candidate " + String(c))
        var means = msel_row_means(want[j])
        for c in range(len(plans)):  # small-loop(check: candidates)
            require(res.scorer_means[j][c] == means[c], "scorer " + String(j) + " mean")
    var ranks = msel_rank_scores(msel_row_means(want[1]))
    require(res.best == msel_rank_best(ranks), "the best by the refit scorer (balanced_accuracy)")
    require(res.best_state == 0, "refit=False makes no refit state")
    _ = x^
    _ = y^


def check_given_folds() raises:
    var x = _x()
    var y = _codes()
    var plans = _candidates(False)
    var code = msel_scorer_code("accuracy", False)
    var want = _scorer_oracle(plans, x, y, [code])
    var counts = List[Int32](length=F + 1, fill=Int32(0))
    var rows = List[Int32](length=F * N, fill=Int32(0))
    var yy = y.copy()
    _ = xt_device_folds(_glue(), Int(yy.unsafe_ptr()), Int(rows.unsafe_ptr()), Int(counts.unsafe_ptr()), N, F, K)
    _ = len(yy)  # the fold program read the codes through their address
    var plan = _plan(plans.copy(), x, y, code)
    for g in range(F):  # small-loop(check: the folds, reversed)
        var f = F - 1 - g
        var t = Int(counts[f])
        plan.fold_train_at.append(Int(rows.unsafe_ptr()) + 4 * (f * N))
        plan.fold_train_n.append(N - t)
        plan.fold_test_at.append(Int(rows.unsafe_ptr()) + 4 * (f * N + N - t))
        plan.fold_test_n.append(t)
    var res = msel_search_run(plan)
    require(res.n_splits == F, "the given folds' count")
    for c in range(len(plans)):  # small-loop(check: candidates)
        for g in range(F):  # small-loop(check: folds)
            require(res.test[c][g] == want[0][c][F - 1 - g], "given split" + String(g) + " of candidate " + String(c))
    var r = MemberCall(plans[res.best].route, res.best_state)
    member_free(plans[res.best], r)
    _ = rows^
    _ = counts^
    _ = x^
    _ = y^
    # THE BINARY SCORERS, counted by hand.
    var route = MetricRoute(metrics_program_host_v1, False, False)
    var truth: List[Int32] = [1, 0, 1, 1]
    var pred: List[Int32] = [1, 1, 0, 1]
    var ta = Int(truth.unsafe_ptr())
    var pa = Int(pred.unsafe_ptr())
    var third = 2.0 / 3.0
    var warn = 0
    require(abs(label_binary_run(route, LBL_PRECISION, ta, pa, 4, 2, 1, warn) - third) < 1e-12, "binary precision")
    require(abs(label_binary_run(route, LBL_RECALL, ta, pa, 4, 2, 1, warn) - third) < 1e-12, "binary recall")
    require(abs(label_binary_run(route, LBL_F1, ta, pa, 4, 2, 1, warn) - third) < 1e-12, "binary f1")
    require(label_binary_run(route, LBL_F1, ta, pa, 4, 2, 0, warn) == 0.0, "binary f1 of positive code 0")
    var three: List[Int32] = [0, 1, 2, 1]
    var refused = False
    try:
        _ = label_binary_run(route, LBL_F1, Int(three.unsafe_ptr()), pa, 4, 3, 1, warn)
    except error:
        refused = String(error) == "average='binary' requires at most two observed classes"
    require(refused, "three observed classes refuse in the door's words")
    _ = len(truth)
    _ = len(pred)
    _ = len(three)


def check_score5_scorers() raises:
    """Case 7 (lane mf-score5): the curve, probability, top-k and clustering scorers in one multi-metric search."""
    var x = _x()
    var y = _codes()
    var plans = _candidates(False)
    var names: List[String] = ["average_precision", "neg_brier_score", "top_k_accuracy",
                               "normalized_mutual_info_score", "adjusted_mutual_info_score"]
    var codes = List[Int]()
    for i in range(len(names)):  # small-loop(check: the scorers, predict_proba responses)
        var code = msel_scorer_code(names[i], False)
        require(code >= 2, names[i] + " is a served scorer")
        codes.append(code)
    var want = _scorer_oracle(plans, x, y, codes)
    var plan = _plan(plans.copy(), x, y, SEARCH_SCORE_MEMBER)
    plan.scorers = codes.copy()
    plan.refit_scorer = 0
    var res = msel_search_run(plan)
    require(len(res.scorer_test) == len(codes), "one table per scorer")
    for j in range(len(codes)):  # small-loop(check: scorers)
        for c in range(len(plans)):  # small-loop(check: candidates)
            for f in range(F):  # small-loop(check: folds)
                require(res.scorer_test[j][c][f] == want[j][c][f],
                        names[j] + " split" + String(f) + " of candidate " + String(c))
    require(res.best == msel_rank_best(msel_rank_scores(msel_row_means(want[0]))), "the best by average_precision")
    var r = MemberCall(plans[res.best].route, res.best_state)
    member_free(plans[res.best], r)
    _ = x^
    _ = y^


# ---------------------------------------------------------------- 8. the resident route (lane mf-search5)


def _wrapped_call(operation: Int, address: Int, scoped: Bool) raises -> Int:
    """The fixture entry's body: the logistic member behind a row-index
    reader. The last parameter word is the address of four Int counters
    (pin offers, cache takes, indexed calls; the fourth is the RUN's
    `kept_at`), dropped before the
    logistic member reads its ten words; an indexed call's rows of X (and y)
    are gathered here, as a device take would, and handed on as a block; the
    first offered pin is taken (`pin_kept`), as a scope cache publishes X
    once."""
    var call = MutPointer[MemberCall, MutUntrackedOrigin](unsafe_from_address=address)
    var inner = MemberRef(EntryRef(logistic_member_v2, 0), 0)
    var counters = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=0)
    var stripped = False
    if call[].param_count > 10:
        var words = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call[].params)
        counters = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(words[call[].param_count - 1]))
        call[].param_count -= 1
        stripped = True
    if stripped and call[].pin != 0:
        counters[0] += 1
        if counters[1] == 0:
            counters[1] = 1
            call[].pin_kept = call[].pin
    var x_was = call[].x
    var y_was = call[].y
    var index_was = call[].row_index
    var xs = List[Float32]()
    var ys = List[Int32]()
    var answer = 0
    try:
        if call[].row_index.rows > 0:
            require(call[].row_index.dtype == MEMBER_I32, "the search hands int32 row numbers")
            if stripped:
                counters[2] += 1
            var count = call[].row_index.rows
            var cols = call[].x.cols
            var rows = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=call[].row_index.data)
            var src = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=call[].x.data)
            xs = List[Float32](length=max(count * cols, 1), fill=0.0)
            for i in range(count):  # small-loop(fixture: the indexed rows of X)
                for j in range(cols):  # small-loop(fixture)
                    xs[i * cols + j] = src[Int(rows[i]) * cols + j]
            call[].x = MemberMatrix(Int(xs.unsafe_ptr()), count, cols)
            if call[].y.rows > 0:
                var codes = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=call[].y.data)
                ys = List[Int32](length=max(count, 1), fill=Int32(0))
                for i in range(count):  # small-loop(fixture: the indexed rows of y)
                    ys[i] = codes[Int(rows[i])]
                call[].y = MemberMatrix(Int(ys.unsafe_ptr()), count, 1, y_was.dtype)
            call[].row_index = MemberMatrix()
        answer = member_call(inner, operation, call[])
        if operation == MEMBER_OP_CAPABILITIES:
            # the family's answer and a kind's (a live state) both
            answer |= MEMBER_CAN_ROW_INDEX
            if scoped:
                answer |= MEMBER_CAN_SCOPE
    finally:
        call[].x = x_was
        call[].y = y_was
        call[].row_index = index_was
        if stripped:
            call[].param_count += 1
        _ = len(xs)  # read by the logistic member above
        _ = len(ys)
    return answer


def _indexed_scoped_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        var word = _wrapped_call(operation, member_call_from_spans(operation, aa, na, np, nf), True)
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def _indexed_plain_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        var word = _wrapped_call(operation, member_call_from_spans(operation, aa, na, np, nf), False)
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def _wrapped(l2: Float64, scoped: Bool, counters: Int) raises -> MemberPlan:
    var plan = _logistic(l2, 0.0)
    if scoped:
        plan.member = MemberRef(EntryRef(_indexed_scoped_v2, 0), 0)
    else:
        plan.member = MemberRef(EntryRef(_indexed_plain_v2, 0), 0)
    var probe = MemberCall(plan.route)
    plan.member.capabilities = member_call(plan.member, MEMBER_OP_CAPABILITIES, probe)
    plan.params.append(Float64(counters))
    return plan^


def _same_scores(a: List[List[Float64]], b: List[List[Float64]], what: String) raises:
    require(len(a) == len(b), what + ": the candidates")
    for c in range(len(a)):  # small-loop(check: candidates)
        for f in range(len(a[c])):  # small-loop(check: folds)
            require(a[c][f] == b[c][f], what + " split" + String(f) + " of candidate " + String(c))


def _free_best(plan: MemberPlan, state: Int) raises:
    var r = MemberCall(plan.route, state)
    member_free(plan, r)


def check_resident_route() raises:
    var x = _x()
    var y = _codes()
    var counters = List[Int](length=4, fill=0)
    var at = Int(counters.unsafe_ptr())
    # The route words.
    var mixed: List[MemberPlan] = [_wrapped(1.0, True, at), _wrapped(0.1, False, at), _logistic(10.0, 0.0)]
    var words_plan = _plan(mixed.copy(), x, y, SEARCH_SCORE_ACCURACY)
    var none = msel_search_rows(words_plan)
    for c in range(3):  # small-loop(check: candidates)
        require(none[c] == SEARCH_ROWS_GATHERED, "no scope: every side gathered")
    words_plan.scope = 0
    words_plan.pin = 12345
    var got = msel_search_rows(words_plan)
    require(got[0] == SEARCH_ROWS_RESIDENT and got[1] == SEARCH_ROWS_INDEXED and got[2] == SEARCH_ROWS_GATHERED,
            "resident / indexed / gathered")
    var moved: List[MemberPlan] = [_wrapped(1.0, True, at), _wrapped(0.1, True, at)]
    moved[1].route.mode = moved[1].route.mode + 1
    var moved_plan = _plan(moved^, x, y, SEARCH_SCORE_ACCURACY)
    moved_plan.scope = 0
    moved_plan.pin = 12345
    var other = msel_search_rows(moved_plan)
    require(other[0] == SEARCH_ROWS_RESIDENT and other[1] == SEARCH_ROWS_INDEXED,
            "a candidate on another image is not offered the pin")
    # The gathered run over the bare members, then the resident run.
    var bare = _plan(_candidates(False), x, y, SEARCH_SCORE_ACCURACY)
    bare.return_train = True
    var want = msel_search_run(bare)
    var wrapped: List[MemberPlan] = [_wrapped(1.0, True, at), _wrapped(0.1, True, at), _wrapped(10.0, True, at)]
    var plan = _plan(wrapped.copy(), x, y, SEARCH_SCORE_ACCURACY)
    plan.return_train = True
    plan.scope = 0
    plan.pin = 12345
    plan.kept_at = at + 3 * 8  # the fourth counter
    for i in range(4):  # small-loop(fixture: the counters)
        counters[i] = 0
    var res = msel_search_run(plan)
    _same_scores(want.test, res.test, "resident accuracy")
    _same_scores(want.train, res.train, "resident train accuracy")
    require(res.best == want.best and res.best_score == want.best_score, "the best")
    require(counters[0] == 3 * F + 2 * 3 * F + 1, "the pin on every fit, every side and the refit")
    require(counters[1] == 1 and counters[3] == 12345, "one take, written at kept_at")
    require(counters[2] == 3 * F + 2 * 3 * F, "every fit and side reads the whole X plus its rows")
    _free_best(bare.plans[want.best], want.best_state)
    _free_best(plan.plans[res.best], res.best_state)
    # scoring None: the member's own score through the rows (y read by index).
    var bare_own = _plan(_candidates(False), x, y, SEARCH_SCORE_MEMBER)
    var want_own = msel_search_run(bare_own)
    var own = _plan(wrapped.copy(), x, y, SEARCH_SCORE_MEMBER)
    own.scope = 0
    own.pin = 777
    own.kept_at = at + 3 * 8
    counters[1] = 0
    var res_own = msel_search_run(own)
    _same_scores(want_own.test, res_own.test, "resident member score")
    require(counters[3] == 777, "the member-score run's take")
    _free_best(bare_own.plans[want_own.best], want_own.best_state)
    _free_best(own.plans[res_own.best], res_own.best_state)
    # The mixed set: the same scores; the pin offered to candidate 0 only.
    var mixed_plan = _plan(mixed.copy(), x, y, SEARCH_SCORE_ACCURACY)
    mixed_plan.return_train = True
    mixed_plan.scope = 0
    mixed_plan.pin = 99
    mixed_plan.kept_at = at + 3 * 8
    for i in range(4):  # small-loop(fixture: the counters)
        counters[i] = 0
    var res_mixed = msel_search_run(mixed_plan)
    _same_scores(want.test, res_mixed.test, "mixed accuracy")
    _same_scores(want.train, res_mixed.train, "mixed train accuracy")
    require(counters[0] == 3 * F + (1 if res_mixed.best == 0 else 0), "the pin on candidate 0's calls only")
    require(counters[2] == 2 * 3 * F, "candidates 0 and 1 read every side by index")
    require(counters[3] == 99, "the mixed run's take")
    _free_best(mixed_plan.plans[res_mixed.best], res_mixed.best_state)
    _ = len(counters)  # read and written by the fixture entry and the RUN through its address
    _ = x^
    _ = y^


# ---------------------------------------------------------------- 9. a clusterer base (lane mf-cluster5)


def _kmeans(clusters: Int) raises -> MemberPlan:
    var route = DeviceRoute(DEVICE_CPU)
    route.mode = Int(GLOBAL_NUMERIC_MODE)
    route.vendor = 0
    route.entry = EntryRef(kmeans_host_native_v1, 0)
    route.flags = KMEANS_CAP_HOST
    var plan = MemberPlan()
    plan.route = route
    plan.member = MemberRef(EntryRef(kmeans_member_v2, 0), 0)
    var probe = MemberCall(plan.route)
    plan.member.capabilities = member_call(plan.member, MEMBER_OP_CAPABILITIES, probe)
    # n_clusters, 'k-means++', n_init 1, max_iter 100, tol 1e-4, random_state 0 (its bits), euclidean, 2.0
    plan.params = [Float64(clusters), 0.0, 1.0, 100.0, 1e-4, bitcast[DType.float64](Int64(0)), 0.0, 2.0]
    plan.classes = 0
    return plan^


def check_clusterer_base() raises:
    var x = _x()
    var y = _codes()
    var yf = List[Float32](length=N, fill=0.0)
    for i in range(N):  # small-loop(fixture: the labels as the regressor kind's float32 target words)
        yf[i] = Float32(y[i])
    var mode = Int(GLOBAL_NUMERIC_MODE)
    var api = METRICS_CLUSTER_HOST_API.get_or_create_ptr()
    cluster_ops_publish(mode, 0, Int(api[].entries.unsafe_ptr()), metrics_cluster_host_v1)
    var cluster = cluster_ops_entry(mode, 0)
    var route = MetricRoute(metrics_program_host_v1, False, False)
    var names: List[String] = ["adjusted_rand_score", "v_measure_score", "normalized_mutual_info_score",
                               "rand_score"]
    var codes = List[Int]()
    for i in range(len(names)):  # small-loop(check: the scorers)
        codes.append(msel_scorer_code(names[i], False))
    var plans: List[MemberPlan] = [_kmeans(2), _kmeans(3)]
    var masks = _masks(plans)
    var plan = MselSearchPlan(plans.copy(), masks^, Int(x.unsafe_ptr()), N, D, Int(yf.unsafe_ptr()), 0, F, _glue(),
                              route)
    plan.scorers = codes.copy()
    plan.refit_scorer = 0
    plan.refit = False
    plan.cluster = cluster
    var res = msel_search_run(plan)
    # THE ORACLE: KFold's folds (k 0), KMeans fitted on each fold's train rows, its ids on the test rows.
    var counts = List[Int32](length=F + 1, fill=Int32(0))
    var rows = List[Int32](length=F * N, fill=Int32(0))
    _ = xt_device_folds(_glue(), Int(counts.unsafe_ptr()), Int(rows.unsafe_ptr()), Int(counts.unsafe_ptr()), N, F, 0)
    for c in range(len(plans)):  # small-loop(oracle: candidates)
        for f in range(F):  # small-loop(oracle: folds)
            var t = Int(counts[f])
            var base = Int(rows.unsafe_ptr()) + 4 * (f * N)
            var xs = List[Float32]()
            var ys = List[Int32]()
            _rows_of(x, y, base, N - t, xs, ys)
            var call = member_new(plans[c])
            call.x = MemberMatrix(Int(xs.unsafe_ptr()), N - t, D)
            _ = member_call(plans[c].member, MEMBER_OP_FIT, call)
            var xt = List[Float32]()
            var yt = List[Int32]()
            _rows_of(x, y, base + 4 * (N - t), t, xt, yt)
            var truth = List[Float32](length=max(t, 1), fill=0.0)
            for i in range(t):  # small-loop(oracle: the float32 target words)
                truth[i] = Float32(yt[i])
            var ids = List[Int32](length=max(t, 1), fill=Int32(0))
            var pc = member_call_of(plans[c], call.state)
            pc.x = MemberMatrix(Int(xt.unsafe_ptr()), t, D)
            pc.dst = MemberMatrix(Int(ids.unsafe_ptr()), t, 1, MEMBER_I32)
            _ = member_call(plans[c].member, MEMBER_OP_PREDICT, pc)
            for j in range(len(codes)):  # small-loop(check: scorers)
                var notes = List[Float64]()
                var want = msel_scorer_value(route, cluster, List[String](), msel_scorer_row(codes[j]),
                                             Int(truth.unsafe_ptr()), Int(ids.unsafe_ptr()), t, 0, 1, -1, notes)
                require(res.scorer_test[j][c][f] == want, names[j] + " split" + String(f) + " of candidate "
                        + String(c))
            member_free(plans[c], call)
            _ = len(xs)  # the member's fit and calls read the fold rows through their addresses
            _ = len(ys)
            _ = len(xt)
            _ = len(yt)
            _ = len(truth)
            _ = len(ids)
    require(res.best_state == 0, "refit=False keeps no state")
    var mixed = MselSearchPlan(plans.copy(), _masks(plans), Int(x.unsafe_ptr()), N, D, Int(yf.unsafe_ptr()), 0, F,
                               _glue(), route)
    var both: List[Int] = [codes[0], msel_scorer_code("r2", False)]
    mixed.scorers = both^
    mixed.refit = False
    mixed.cluster = cluster
    var refused = False
    try:
        _ = msel_search_run(mixed)
    except e:
        refused = String(e).find("clustering scorers only") >= 0
    require(refused, "a clusterer beside a regression scorer refuses")
    _ = rows^
    _ = counts^
    _ = x^
    _ = y^
    _ = yf^


def _direct_fit(plan: MemberPlan, x: List[Float32], y: List[Int32], rows: Int, count: Int) raises -> Int:
    """Oracle: the member fitted directly on `count` rows at the int32 row numbers `rows`."""
    var xs = List[Float32]()
    var ys = List[Int32]()
    _rows_of(x, y, rows, count, xs, ys)
    var call = member_new(plan)
    call.x = MemberMatrix(Int(xs.unsafe_ptr()), count, D)
    call.y = MemberMatrix(Int(ys.unsafe_ptr()), count, 1, MEMBER_I32)
    call.classes = plan.classes
    _ = member_call(plan.member, MEMBER_OP_FIT, call)
    _ = len(xs)  # the fit read the rows through their addresses
    _ = len(ys)
    return call.state


def _proba(plan: MemberPlan, state: Int, x: Int, rows: Int, classes: Int) raises -> List[Float64]:
    var out = List[Float64](length=max(rows * classes, 1), fill=0.0)
    var call = member_call_of(plan, state)
    call.x = MemberMatrix(x, rows, D)
    call.dst = MemberMatrix(Int(out.unsafe_ptr()), rows, classes, MEMBER_F64)
    _ = member_call(plan.member, MEMBER_OP_PREDICT_PROBA, call)
    return out^


def check_cv_tools() raises:
    var x = _x()
    var y = _codes()
    var one = _logistic(1.0, 0.0)
    var code = msel_scorer_code("accuracy", False)
    var counts = List[Int32](length=F + 1, fill=Int32(0))
    var rows = List[Int32](length=F * N, fill=Int32(0))
    var yy = y.copy()
    _ = xt_device_folds(_glue(), Int(yy.unsafe_ptr()), Int(rows.unsafe_ptr()), Int(counts.unsafe_ptr()), N, F, K)
    _ = len(yy)  # the fold program read the codes through their address
    # The learning curve's units, size-major: sizes N / 3 and every train row.
    var plan = _plan([one.copy()], x, y, code)
    plan.return_train = True
    plan.keep = True
    plan.indices = True
    plan.refit = False
    var sizes = List[Int]()
    for si in range(2):  # small-loop(fixture: the two sizes)
        for f in range(F):  # small-loop(fixture: folds)
            var t = Int(counts[f])
            var size = N // 3 if si == 0 else N - t
            plan.fold_train_at.append(Int(rows.unsafe_ptr()) + 4 * (f * N))
            plan.fold_train_n.append(size)
            plan.fold_test_at.append(Int(rows.unsafe_ptr()) + 4 * (f * N + N - t))
            plan.fold_test_n.append(t)
            sizes.append(size)
    var units = 2 * F
    var res = msel_search_run(plan)
    require(res.n_splits == units, "the curve's units")
    require(len(res.fit_seconds) == units and len(res.score_seconds) == units, "one fit and score time per unit")
    require(len(res.unit_states) == units and len(res.fold_train) == units, "kept states and rows per unit")
    for u in range(units):  # small-loop(check: units)
        var f = u % F
        var t = Int(counts[f])
        var train_at = Int(rows.unsafe_ptr()) + 4 * (f * N)
        var test_at = train_at + 4 * (N - t)
        require(res.fit_seconds[u] >= 0.0 and res.score_seconds[u] >= 0.0, "seconds of unit " + String(u))
        var state = _direct_fit(one, x, y, train_at, sizes[u])
        var xt = List[Float32]()
        var yt = List[Int32]()
        _rows_of(x, y, test_at, t, xt, yt)
        var xs = List[Float32]()
        var ys = List[Int32]()
        _rows_of(x, y, train_at, sizes[u], xs, ys)
        require(res.test[0][u] == _side_score(one, state, xt, yt, t, SEARCH_SCORE_ACCURACY), "curve test " + String(u))
        require(res.train[0][u] == _side_score(one, state, xs, ys, sizes[u], SEARCH_SCORE_ACCURACY),
                "curve train " + String(u))
        require(res.unit_states[u] != 0, "the kept state of unit " + String(u))
        var a = _proba(one, state, Int(x.unsafe_ptr()), N, K)
        var b = _proba(one, res.unit_states[u], Int(x.unsafe_ptr()), N, K)
        for i in range(N * K):  # small-loop(check)
            require(a[i] == b[i], "the kept estimator's predict_proba, unit " + String(u))
        var r = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=train_at)
        require(len(res.fold_train[u]) == sizes[u] and len(res.fold_test[u]) == t, "the rows' counts")
        for i in range(sizes[u]):  # small-loop(check: the train rows)
            require(res.fold_train[u][i] == r[i], "indices['train'] of unit " + String(u))
        var call = MemberCall(one.route, state)
        member_free(one, call)
        var kept = MemberCall(one.route, res.unit_states[u])
        member_free(one, kept)
        _ = len(xt)
        _ = len(yt)
        _ = len(xs)
        _ = len(ys)
    # cross_val_predict's method over the int cv.
    var plan2 = _plan([one.copy()], x, y, SEARCH_SCORE_MEMBER)
    plan2.respond = MEMBER_OP_PREDICT_PROBA
    plan2.refit = False
    var got = msel_search_run(plan2)
    require(len(got.outputs) == F, "one response per fold")
    for f in range(F):  # small-loop(check: folds)
        var t = Int(counts[f])
        var train_at = Int(rows.unsafe_ptr()) + 4 * (f * N)
        var state = _direct_fit(one, x, y, train_at, N - t)
        var xt = List[Float32]()
        var yt = List[Int32]()
        _rows_of(x, y, train_at + 4 * (N - t), t, xt, yt)
        var want = _proba(one, state, Int(xt.unsafe_ptr()), t, K)
        require(got.output_cols[f] == K and got.output_dtype[f] == MEMBER_F64, "the response's shape and type")
        require(len(got.outputs[f]) == t * K, "the response's words")
        for i in range(t * K):  # small-loop(check)
            require(got.outputs[f][i] == want[i], "predict_proba of fold " + String(f))
        var call = MemberCall(one.route, state)
        member_free(one, call)
        _ = len(xt)
        _ = len(yt)
    _ = rows^
    _ = counts^
    _ = x^
    _ = y^


def check_ap_absent() raises:
    var x = _x()
    var y = List[Int32](length=N, fill=Int32(0))
    for i in range(N):  # small-loop(fixture: two classes, N / 2 rows each)
        y[i] = Int32(i % 2)
    var one = _logistic(1.0, 0.0)
    one.classes = 2
    var code = msel_scorer_code("average_precision", False)
    require(code >= 0, "the average_precision row")
    var masks = _masks([one.copy()])
    var plan = MselSearchPlan([one.copy()], masks^, Int(x.unsafe_ptr()), N, D, Int(y.unsafe_ptr()), 2, F, _glue(),
                              MetricRoute(metrics_program_host_v1, False, False))
    plan.scoring = code
    plan.positives = [SCORER_POSITIVE_ABSENT]
    plan.label_words = ["0", "2"]
    plan.refit = False
    var words = String("")
    try:
        _ = msel_search_run(plan)
    except error:
        words = String(error)
    require(words == "ValueError: pos_label=1 is not a valid label: It should be one of [0 2]",
            "AP-absent refuses in scikit-learn's words, got: " + words)
    var masks2 = _masks([one.copy()])
    var plan2 = MselSearchPlan([one.copy()], masks2^, Int(x.unsafe_ptr()), N, D, Int(y.unsafe_ptr()), 2, F, _glue(),
                               MetricRoute(metrics_program_host_v1, False, False))
    plan2.scoring = code
    plan2.positives = [SCORER_POSITIVE_ABSENT]
    plan2.label_words = ["0", "2"]
    plan2.refit = False
    plan2.tolerate = True
    plan2.error_value = -1.0
    var res = msel_search_run(plan2)
    for f in range(F):  # small-loop(check: folds)
        require(res.test[0][f] == -1.0, "AP-absent takes the error score in split " + String(f))
    _ = x^
    _ = y^


def test_msel_search_run() raises:
    publish_x_trees_glue("cpu")
    def run_case(which: Int) raises:
        if which == 0:
            check_member_score()
        elif which == 1:
            check_accuracy()
        elif which == 2:
            check_error_score()
        elif which == 3:
            check_named_scorer()
        elif which == 4:
            check_multi_metric()
        elif which == 5:
            check_given_folds()
        elif which == 6:
            check_score5_scorers()
        elif which == 7:
            check_resident_route()
        elif which == 8:
            check_clusterer_base()
        elif which == 9:
            check_cv_tools()
        else:
            check_ap_absent()
    check_cases("msel_search_run", CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
