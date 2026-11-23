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
12. (lane mf-msel5b) A FOLD THAT LACKS A CLASS: two given folds, fold 0 training on rows without class 0. With
   `respond` = predict_proba into the caller's N x K result: fold 0's two columns land at classes 1 and 2 (the
   member fitted on the re-coded codes, two classes), class 0's column is 0.0, fold 1 is placed as it is, and one
   `_enforce_prediction_order` warning notice [12, 2, 3]; decision_function refuses in scikit-learn's words
   ("Only 2 class/es in training fold, but 3 in overall dataset. ..."); 'accuracy' scores fold 0 over its
   predicted fold codes mapped back to the classes; the member's own score of such a fold (lane mf-kmeans5) is
   `ClassifierMixin.score`: every unit scored as the accuracy of its predict, fold 0 over its mapped codes.
13. (lane mf-msel5b) A FIT WITHOUT A TARGET: cross_val_predict(KMeans(3), X) with y=None (plan.y 0, KFold):
   each fold's predicted cluster ids land at its rows of the caller's int64 result, equal to KMeans fitted
   directly on the fold's train rows and predicting its test rows (a view of X: one run of rows).
14. (lane mf-kmeans5) KMEANS SCORE (core/kmeans_member.mojo MEMBER_OP_SCORE over the host KMEANS_SCORE op): on
   the training X the member's score is minus the fit's inertia_ (its scalars block, word 0) bit for bit, for a
   unit-weight fit and for a weighted fit scored with the same weights; the same score through a host row index
   naming every row (the gathered route) is the same bits; a fit's labels are untouched by a score.
15. (lane mf-kmeans5) A KMEANS SEARCH SCORED BY ITS MEMBERS (scoring None, classes 0, KFold, y float32 words):
   every split score equals KMeans fitted directly on the fold's train rows and its own score on the test rows,
   i.e. minus the test rows' inertia; the best is the candidate of the higher mean.
16. (lane mf-kmeans5) A LACKING FOLD SCORED BY PROBABILITIES: 'neg_log_loss' over the two folds of case 12 (fold 0
   a binary fold estimator: its positive column, against a test side holding three labels) raises scikit-learn's
   words "ValueError: y_true and y_pred contain different number of classes 3, 2. ..."; with a number fold 0
   takes it and fold 1 equals the metric over the member's own probabilities.
17. (lane mf-msel5c) THE LACKING FOLD BY 'average_precision' AND 'neg_brier_score': case 12's fold 0 (a binary
   fold estimator) against a test side of three labels: AP refuses its (30,) scores in scikit-learn's
   `label_binarize` shape words; brier refuses a multiclass y_true for one column.
18. (lane mf-msel5c) FOUR classes, fold 0 a three-class fold estimator against four test labels: AP's shape words
   with (30, 3); brier's "different number of classes: 4 vs 3 ... Classes found in y_true: [0 1 2 3]".
20. (lane mf-tail5) THE GIVEN y (core/member_abi.mojo `MemberCall.y_given`): bindings/msel_split_spec_run.mojo
   `msel_target_codes_run` over float64 targets [1.0, 1.0 + 2^-40, 2.0, 1.0] given as MSEL_GIVEN_F64 codes them
   [0, 1, 2, 0] with m = 3 (np.unique's inverse over the given y), while their float32 words alone give m = 2;
   int64 [16777217, 16777216, 5] given as MSEL_GIVEN_I64 codes [2, 1, 0], m = 3 (equal in float32); int32
   [3, -1, 3] given as MSEL_GIVEN_I32 codes [1, 0, 1]; the MSEL_GIVEN_* words equal MEMBER_F64 / _I32 / _I64.
21. (lane mf-tail5) A MULTI-OUTPUT CLASSIFIER'S LACKING FOLD, the RUN's helpers: `_held_codes_col` over a rows x 2
   code block (output 0 holds classes {0, 2} of 3 on the train rows, output 1 all of 2) answers [0, 2] and [];
   `_map_multi_codes` maps output 0's fold codes [0, 1, 1] to [0, 2, 2] in place and leaves output 1 unchanged
   (int32 and int64 words); `_place_multi_proba` writes a unit's 2 + 2 probability columns into output 0's
   N x 3 block at classes 0 and 2 (class 1's column 0.0) and output 1's N x 2 block as they are, at the side's
   rows, and refuses a member answer of the wrong width.
19. (lane mf-msel5c) MAKE_SCORER KWARGS: top_k's k in the code (metric word, kept by the XOR flip), a binary row's
   given pos_label (its refusal names the label's str), and both refused on other rows.

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
`df[at + to] = value` written at `at + c` (case 10: rows in fold order, not the test rows' order); `shown += " "`
removed (case 11: the words); `to = c` (no held map) in `_place_response`, or `fill = 0.0` for a held fold's
proba replaced by the member's first column, or the `member_notice_add` of the lacking fold removed (case 12);
`codes[i] = h[Int(codes[i])]` removed (case 12: fold 0's accuracy); the `side.y != 0` test in `_side_set`
removed, or `y_dst[s] = plan.y + ...` for a target-free side (case 13: a read of address 0); `member_score =
False` removed from the lacking block (case 12: fold 0 scored by the member against fold codes); in
core/kmeans_member.mojo `call[].value = result.inertia` (cases 14, 15: the sign), or in
cluster/host/kmeans_oracle.mojo `host_kmeans_score` passing `SUM_MODE_PLAIN` (case 14's weighted fit), or the
`MEMBER_OP_SCORE` arm removed from `_host_indexed` (case 14: the indexed score reads the base X as the rows);
`label_count = classes` kept in `_fold_view_side` (case 16: no refusal, a log loss over three labels); in
`_fold_truth_checks` `if present > 2:` written `if present > 3:` (cases 17, 18: the AP shape words), or
`present != view_cols` written `present < view_cols` (case 18: brier's count words); in core/msel_scorer_run.mojo
`row.sign = -row.sign` removed, or `row.metric = opts.k` removed, or the
`msel_scorer_pos_words` call in `_binary_value` replaced by "1", or `opts.pos_word = Int(words[at + 4])` read
from `at + 5` (case 19); in bindings/mcls_classify_run.mojo `McEpi(2, chosen, avg, zd, fmode, b2)` written with
`2, 1.0` (case 22: fbeta 1/2), `chosen = sel_n` written `chosen = labels` (case 22: 2/3), or `zd` written 0
(case 22: zero_division 1/2); in bindings/msel_split_spec_run.mojo
`var kind = 0` in place of the given dtype in `msel_target_codes_run` (case 20: m = 2), or `Float64(srci64[i])`
read as `Float64(Float32(srci64[i]))` (case 20: the int64 pair merges); in core/msel_search_run.mojo
`_held_codes_col` reading `y[Int(r[i])]` without `* width + column` (case 21: output 0's held classes), `continue`
removed for an output with no held list in `_map_multi_codes` (case 21: output 1 read through an empty list),
`var to = c` in `_place_multi_proba` (case 21: output 0's columns not at their classes), or `offset += width`
removed (case 21: output 1 reads output 0's columns).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require
from checks.numerics import GLOBAL_NUMERIC_MODE
from bindings.arena_prog_run import MetricRoute
from bindings.msel_search_stats_run import msel_row_means, msel_row_stds, msel_rank_scores, msel_rank_best
from core.device_route import DEVICE_CPU
from core.entry_table import EntryRef, entry_ref_unresolved
from core.logistic_member import logistic_member_v2
from core.logistic_native_api import QnNativeDriver
from core.logistic_route import LogisticRoute, logistic_route_pack
from core.member_abi import (
    MemberCall, MemberMatrix, MemberPlan, MemberRef, member_call, member_call_of, member_new, member_free,
    member_kind_capabilities, MEMBER_OP_CAPABILITIES, MEMBER_OP_FIT, MEMBER_OP_SCORE, MEMBER_OP_PREDICT,
    MEMBER_OP_PREDICT_PROBA, MEMBER_F64, MEMBER_I32, MEMBER_I64, MEMBER_CAN_ROW_INDEX, MEMBER_CAN_SCOPE,
    member_call_from_spans, MEMBER_OP_DECISION, MEMBER_OP_EXPORT, MEMBER_BLOCK_BY_INDEX,
)
from core.msel_search_run import (
    MselSearchPlan, msel_search_run, msel_search_rows, SEARCH_SCORE_MEMBER, SEARCH_SCORE_ACCURACY,
    SEARCH_ROWS_GATHERED, SEARCH_ROWS_INDEXED, SEARCH_ROWS_RESIDENT,
    _held_codes_col, _map_multi_codes, _place_multi_proba,
)
from bindings.msel_split_spec_run import msel_target_codes_run, MSEL_GIVEN_F64, MSEL_GIVEN_I32, MSEL_GIVEN_I64
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.msel_scorer_run import (
    msel_scorer_code, msel_scorer_row, msel_scorer_value, SCORER_POSITIVE_ABSENT,
    msel_scorer_pos_words, MselScorer, MselScorerOptions, msel_scorer_group_words, msel_scorer_group_read,
    SCORER_GROUP_WORDS,
)
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
comptime CASES = 22


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
    # cross_val_predict's method over the int cv (lane mf-msel5b): written
    # straight into the caller's N x K float64 result at each fold's test rows.
    var plan2 = _plan([one.copy()], x, y, SEARCH_SCORE_MEMBER)
    plan2.respond = MEMBER_OP_PREDICT_PROBA
    plan2.refit = False
    var result = List[Float64](length=N * K, fill=-7.0)
    plan2.respond_dst = Int(result.unsafe_ptr())
    plan2.respond_width = K
    plan2.respond_dtype = MEMBER_F64
    _ = msel_search_run(plan2)
    for f in range(F):  # small-loop(check: folds)
        var t = Int(counts[f])
        var train_at = Int(rows.unsafe_ptr()) + 4 * (f * N)
        var state = _direct_fit(one, x, y, train_at, N - t)
        var xt = List[Float32]()
        var yt = List[Int32]()
        _rows_of(x, y, train_at + 4 * (N - t), t, xt, yt)
        var want = _proba(one, state, Int(xt.unsafe_ptr()), t, K)
        var r = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=train_at + 4 * (N - t))
        for i in range(t):  # small-loop(check: the fold's test rows)
            for c in range(K):  # small-loop(check: classes)
                require(result[Int(r[i]) * K + c] == want[i * K + c], "predict_proba of fold " + String(f))
        var call = MemberCall(one.route, state)
        member_free(one, call)
        _ = len(xt)
        _ = len(yt)
    for i in range(N * K):  # small-loop(check: every row written)
        require(result[i] != -7.0, "the result's word " + String(i) + " written")
    _ = len(result)
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


def _lacking_folds(y: List[Int32], mut train0: List[Int32], mut test0: List[Int32], mut train1: List[Int32],
                   mut test1: List[Int32]):
    """Two given folds: fold 0 tests rows [0, N/2) and trains on the rows of [N/2, N) whose code is not 0 (it
    lacks class 0); fold 1 tests [N/2, N) and trains on [0, N/2) (every class)."""
    for i in range(N):  # small-loop(fixture: the two folds)
        if i < N // 2:
            test0.append(Int32(i))
            train1.append(Int32(i))
        else:
            test1.append(Int32(i))
            if y[i] != 0:
                train0.append(Int32(i))


def _given(mut plan: MselSearchPlan, given: List[Int]):
    """The two given folds of `_lacking_folds` (their int32 views: train at, n, test at, n per fold)."""
    plan.fold_train_at = [given[0], given[4]]
    plan.fold_train_n = [given[1], given[5]]
    plan.fold_test_at = [given[2], given[6]]
    plan.fold_test_n = [given[3], given[7]]


def check_lacking_fold() raises:
    var x = _x()
    var y = _codes()
    var one = _logistic(1.0, 0.0)
    var train0 = List[Int32]()
    var test0 = List[Int32]()
    var train1 = List[Int32]()
    var test1 = List[Int32]()
    _lacking_folds(y, train0, test0, train1, test1)
    var given: List[Int] = [Int(train0.unsafe_ptr()), len(train0), Int(test0.unsafe_ptr()), len(test0),
                            Int(train1.unsafe_ptr()), len(train1), Int(test1.unsafe_ptr()), len(test1)]

    # The oracle of fold 0: the member fitted on its rows' codes re-coded onto the classes they hold (1 -> 0,
    # 2 -> 1), two classes; fold 1: every class.
    var y0 = y.copy()
    for i in range(N):  # small-loop(oracle: fold 0's codes)
        y0[i] = y[i] - 1 if y[i] > 0 else Int32(0)
    var two = one.copy()
    two.classes = 2
    var s0 = _direct_fit(two, x, y0, Int(train0.unsafe_ptr()), len(train0))
    var s1 = _direct_fit(one, x, y, Int(train1.unsafe_ptr()), len(train1))
    # cross_val_predict predict_proba: fold 0's two columns at classes 1 and 2, class 0's column 0.0.
    var plan = _plan([one.copy()], x, y, SEARCH_SCORE_MEMBER)
    _given(plan, given)
    plan.respond = MEMBER_OP_PREDICT_PROBA
    plan.refit = False
    var result = List[Float64](length=N * K, fill=-7.0)
    plan.respond_dst = Int(result.unsafe_ptr())
    plan.respond_width = K
    plan.respond_dtype = MEMBER_F64
    var res = msel_search_run(plan)
    var xt0 = List[Float32]()
    var yt0 = List[Int32]()
    _rows_of(x, y, Int(test0.unsafe_ptr()), len(test0), xt0, yt0)
    var want0 = _proba(two, s0, Int(xt0.unsafe_ptr()), len(test0), 2)
    for i in range(len(test0)):  # small-loop(check: fold 0's rows)
        var row = Int(test0[i])
        require(result[row * K] == 0.0, "the lacking class's column is 0.0")
        require(result[row * K + 1] == want0[i * 2] and result[row * K + 2] == want0[i * 2 + 1],
                "fold 0's columns placed at their classes")
    var xt1 = List[Float32]()
    var yt1 = List[Int32]()
    _rows_of(x, y, Int(test1.unsafe_ptr()), len(test1), xt1, yt1)
    var want1 = _proba(one, s1, Int(xt1.unsafe_ptr()), len(test1), K)
    for i in range(len(test1)):  # small-loop(check: fold 1's rows)
        for c in range(K):  # small-loop(check)
            require(result[Int(test1[i]) * K + c] == want1[i * K + c], "fold 1 placed as it is")
    # `_enforce_prediction_order`'s warning for fold 0 only: [code 12, held 2, classes 3].
    var warned = 0
    for i in range(len(res.notices)):  # small-loop(check: the units' notices)
        for j in range(len(res.notices[i]) // 3):  # small-loop(check: triples)
            if res.notices[i][3 * j] == 12.0:
                require(res.notices[i][3 * j + 1] == 2.0 and res.notices[i][3 * j + 2] == 3.0, "the warning's words")
                warned += 1
    require(warned == 1, "one lacking-fold warning")
    # decision_function of a fold holding two of three classes: scikit-learn's refusal.
    var plan2 = _plan([one.copy()], x, y, SEARCH_SCORE_MEMBER)
    _given(plan2, given)
    plan2.respond = MEMBER_OP_DECISION
    plan2.refit = False
    plan2.respond_dst = Int(result.unsafe_ptr())
    plan2.respond_width = K
    var words = String("")
    try:
        _ = msel_search_run(plan2)
    except error:
        words = String(error)
    require(words.startswith("ValueError: Only 2 class/es in training fold, but 3 in overall dataset."),
            "the decision refusal, got: " + words)
    # 'accuracy' over the same folds: fold 0's predicted fold codes mapped back to the classes.
    var plan3 = _plan([one.copy()], x, y, msel_scorer_code("accuracy", False))
    _given(plan3, given)
    plan3.refit = False
    var scored = msel_search_run(plan3)
    var call = member_call_of(two, s0)
    var wide = List[Int64](length=len(test0), fill=Int64(0))
    call.x = MemberMatrix(Int(xt0.unsafe_ptr()), len(test0), D)
    call.dst = MemberMatrix(Int(wide.unsafe_ptr()), len(test0), 1, MEMBER_I64)
    _ = member_call(two.member, MEMBER_OP_PREDICT, call)
    var hits = 0
    for i in range(len(test0)):  # small-loop(oracle: accuracy of the mapped codes)
        if Int(wide[i]) + 1 == Int(yt0[i]):
            hits += 1
    require(scored.test[0][0] == Float64(hits) / Float64(len(test0)), "fold 0's accuracy over mapped codes")
    require(scored.test[0][1] == _side_score(one, s1, xt1, yt1, len(test1), SEARCH_SCORE_ACCURACY), "fold 1")
    # the member's own score of a lacking fold (lane mf-kmeans5): `ClassifierMixin.score`, the accuracy of
    # each unit's predict (fold 0's mapped codes), the same values as 'accuracy' above
    var plan4 = _plan([one.copy()], x, y, SEARCH_SCORE_MEMBER)
    _given(plan4, given)
    plan4.refit = False
    var own = msel_search_run(plan4)
    require(own.test[0][0] == Float64(hits) / Float64(len(test0)), "fold 0's own score is its mapped accuracy")
    require(own.test[0][1] == _side_score(one, s1, xt1, yt1, len(test1), SEARCH_SCORE_ACCURACY),
            "fold 1's own score is its accuracy")
    var c0 = MemberCall(two.route, s0)
    member_free(two, c0)
    var c1 = MemberCall(one.route, s1)
    member_free(one, c1)
    _ = len(wide)
    _ = len(xt0)
    _ = len(yt0)
    _ = len(xt1)
    _ = len(yt1)
    _ = len(result)
    _ = len(train0)
    _ = len(test0)
    _ = len(train1)
    _ = len(test1)
    _ = x^
    _ = y^
    _ = y0^


def check_target_free() raises:
    """cross_val_predict(KMeans(3), X) with y=None: no target anywhere; each fold's predicted ids at its rows."""
    var x = _x()
    var km = _kmeans(3)
    var plans: List[MemberPlan] = [km.copy()]
    var masks = _masks(plans)
    var plan = MselSearchPlan(plans^, masks^, Int(x.unsafe_ptr()), N, D, 0, 0, F, _glue(),
                              MetricRoute(metrics_program_host_v1, False, False))
    plan.respond = MEMBER_OP_PREDICT
    plan.refit = False
    var result = List[Int64](length=N, fill=Int64(-7))
    plan.respond_dst = Int(result.unsafe_ptr())
    plan.respond_width = 1
    plan.respond_dtype = MEMBER_I64
    _ = msel_search_run(plan)
    # KFold over N rows, unshuffled: fold f tests rows [f * N / F, (f + 1) * N / F) (N divisible by F).
    var dummy = List[Int32](length=N, fill=Int32(0))
    for f in range(F):  # small-loop(check: folds)
        var lo = f * (N // F)
        var hi = lo + N // F
        var train = List[Int32]()
        for i in range(N):  # small-loop(oracle: the fold's train rows)
            if i < lo or i >= hi:
                train.append(Int32(i))
        var xs = List[Float32]()
        var ys = List[Int32]()
        _rows_of(x, dummy, Int(train.unsafe_ptr()), len(train), xs, ys)
        var call = member_new(km)
        call.x = MemberMatrix(Int(xs.unsafe_ptr()), len(train), D)
        _ = member_call(km.member, MEMBER_OP_FIT, call)
        var ids = List[Int32](length=N // F, fill=Int32(0))
        var q = member_call_of(km, call.state)
        q.x = MemberMatrix(Int(x.unsafe_ptr()) + 4 * lo * D, N // F, D)
        q.dst = MemberMatrix(Int(ids.unsafe_ptr()), N // F, 1, MEMBER_I32)
        _ = member_call(km.member, MEMBER_OP_PREDICT, q)
        for i in range(N // F):  # small-loop(check: the fold's rows, a view of X in the RUN)
            require(result[lo + i] == Int64(ids[i]), "the predicted id of row " + String(lo + i))
        member_free(km, call)
        _ = len(xs)
        _ = len(ys)
        _ = len(train)
        _ = len(ids)
    _ = len(result)
    _ = len(dummy)
    _ = x^


def _kmeans_inertia(plan: MemberPlan, state: Int) raises -> Float64:
    """Word 0 of the fitted state's scalars block (inertia_)."""
    var call = member_call_of(plan, state)
    call.block = 1
    call.flags = MEMBER_BLOCK_BY_INDEX
    _ = member_call(plan.member, MEMBER_OP_EXPORT, call)
    return MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)[0]


def _kmeans_score(plan: MemberPlan, state: Int, x: Int, rows: Int, weights: Int, index: Int) raises -> Float64:
    var call = member_call_of(plan, state)
    call.x = MemberMatrix(x, N if index != 0 else rows, D)
    if weights != 0:
        call.weights = MemberMatrix(weights, N if index != 0 else rows, 1)
    if index != 0:
        call.row_index = MemberMatrix(index, rows, 1, MEMBER_I32)
    _ = member_call(plan.member, MEMBER_OP_SCORE, call)
    return call.value


def check_kmeans_score() raises:
    """KMeans' member score on the training X is minus inertia_, unweighted and weighted, and through a row index."""
    var x = _x()
    var km = _kmeans(3)
    var call = member_new(km)
    call.x = MemberMatrix(Int(x.unsafe_ptr()), N, D)
    _ = member_call(km.member, MEMBER_OP_FIT, call)
    var inertia = _kmeans_inertia(km, call.state)
    require(inertia > 0.0, "a fit of three clusters on spread rows has a positive inertia")
    var score = _kmeans_score(km, call.state, Int(x.unsafe_ptr()), N, 0, 0)
    require(bitcast[DType.uint64](score) == bitcast[DType.uint64](-inertia), "score(X_train) == -inertia_")
    var every = List[Int32](length=N, fill=Int32(0))
    for i in range(N):  # small-loop(fixture: a row index naming every row in order)
        every[i] = Int32(i)
    var indexed = _kmeans_score(km, call.state, Int(x.unsafe_ptr()), N, 0, Int(every.unsafe_ptr()))
    require(bitcast[DType.uint64](indexed) == bitcast[DType.uint64](score), "the indexed score is the same bits")
    member_free(km, call)
    var w = List[Float32](length=N, fill=0.0)
    for i in range(N):  # small-loop(fixture: weights 1, 2, 3 in turn)
        w[i] = Float32(1 + i % 3)
    var wc = member_new(km)
    wc.x = MemberMatrix(Int(x.unsafe_ptr()), N, D)
    wc.weights = MemberMatrix(Int(w.unsafe_ptr()), N, 1)
    _ = member_call(km.member, MEMBER_OP_FIT, wc)
    var weighted = _kmeans_inertia(km, wc.state)
    var wscore = _kmeans_score(km, wc.state, Int(x.unsafe_ptr()), N, Int(w.unsafe_ptr()), 0)
    require(bitcast[DType.uint64](wscore) == bitcast[DType.uint64](-weighted),
            "a weighted score with the fit's weights == -inertia_")
    var unweighted = _kmeans_score(km, wc.state, Int(x.unsafe_ptr()), N, 0, 0)
    require(unweighted != wscore, "the weights change the score")
    member_free(km, wc)
    _ = len(every)
    _ = len(w)
    _ = x^


def check_kmeans_search() raises:
    """GridSearchCV(KMeans) with scoring None through the RUN: each split is the fold member's own score."""
    var x = _x()
    var y = _codes()
    var yf = List[Float32](length=N, fill=0.0)
    for i in range(N):  # small-loop(fixture: the target as the regressor kind's float32 words)
        yf[i] = Float32(y[i])
    var plans: List[MemberPlan] = [_kmeans(2), _kmeans(3)]
    var masks = _masks(plans)
    var plan = MselSearchPlan(plans.copy(), masks^, Int(x.unsafe_ptr()), N, D, Int(yf.unsafe_ptr()), 0, F, _glue(),
                              MetricRoute(metrics_program_host_v1, False, False))
    plan.scoring = SEARCH_SCORE_MEMBER
    plan.refit = False
    var res = msel_search_run(plan)
    var counts = List[Int32](length=F + 1, fill=Int32(0))
    var rows = List[Int32](length=F * N, fill=Int32(0))
    _ = xt_device_folds(_glue(), Int(counts.unsafe_ptr()), Int(rows.unsafe_ptr()), Int(counts.unsafe_ptr()), N, F, 0)
    var means = List[Float64]()
    for c in range(len(plans)):  # small-loop(oracle: candidates)
        var total = 0.0
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
            var want = _kmeans_score(plans[c], call.state, Int(xt.unsafe_ptr()), t, 0, 0)
            require(want < 0.0, "a KMeans score is minus a positive inertia")
            require(res.test[c][f] == want, "split" + String(f) + " of candidate " + String(c))
            total += want
            member_free(plans[c], call)
            _ = len(xs)
            _ = len(ys)
            _ = len(xt)
            _ = len(yt)
        means.append(total)
    require(res.best == (0 if means[0] >= means[1] else 1), "the best is the candidate of the higher mean")
    _ = rows^
    _ = counts^
    _ = x^
    _ = y^
    _ = yf^


def check_lacking_proba() raises:
    """'neg_log_loss' over case 12's folds: fold 0 is a binary fold estimator against three test labels."""
    var x = _x()
    var y = _codes()
    var one = _logistic(1.0, 0.0)
    var train0 = List[Int32]()
    var test0 = List[Int32]()
    var train1 = List[Int32]()
    var test1 = List[Int32]()
    _lacking_folds(y, train0, test0, train1, test1)
    var given: List[Int] = [Int(train0.unsafe_ptr()), len(train0), Int(test0.unsafe_ptr()), len(test0),
                            Int(train1.unsafe_ptr()), len(train1), Int(test1.unsafe_ptr()), len(test1)]
    var code = msel_scorer_code("neg_log_loss", False)
    var plan = _plan([one.copy()], x, y, code)
    _given(plan, given)
    plan.refit = False
    var words = String("")
    try:
        _ = msel_search_run(plan)
    except error:
        words = String(error)
    require(words.startswith("ValueError: y_true and y_pred contain different number of classes 3, 2."),
            "the lacking fold's log loss refusal, got: " + words)
    var tolerant = _plan([one.copy()], x, y, code)
    _given(tolerant, given)
    tolerant.refit = False
    tolerant.tolerate = True
    tolerant.error_value = -99.0
    var res = msel_search_run(tolerant)
    require(res.test[0][0] == -99.0, "fold 0 takes the error value")
    var s1 = _direct_fit(one, x, y, Int(train1.unsafe_ptr()), len(train1))
    var xt1 = List[Float32]()
    var yt1 = List[Int32]()
    _rows_of(x, y, Int(test1.unsafe_ptr()), len(test1), xt1, yt1)
    var proba = _proba(one, s1, Int(xt1.unsafe_ptr()), len(test1), K)
    var words32 = List[Float32](length=len(proba), fill=0.0)
    for i in range(len(proba)):  # small-loop(oracle: the RUN's float32 narrowing of the probabilities)
        words32[i] = Float32(proba[i])
    var notes = List[Float64]()
    var want = msel_scorer_value(MetricRoute(metrics_program_host_v1, False, False), msel_scorer_row(code),
                                 Int(yt1.unsafe_ptr()), Int(words32.unsafe_ptr()), len(test1), K, K, -1, notes)
    require(res.test[0][1] == want, "fold 1 is the metric over its member's probabilities")
    var c1 = MemberCall(one.route, s1)
    member_free(one, c1)
    _ = len(words32)
    _ = len(xt1)
    _ = len(yt1)
    _ = len(train0)
    _ = len(test0)
    _ = len(train1)
    _ = len(test1)
    _ = x^
    _ = y^


def _refusal(var plan: MselSearchPlan) -> String:
    """The words the RUN raises for `plan` ("" when it returns)."""
    try:
        _ = msel_search_run(plan)
    except error:
        return String(error)
    return String("")


def check_lacking_ap_brier() raises:
    """(lane mf-msel5c) case 12's folds (fold 0 a binary fold estimator, classes 1 and 2, its test side holding
    three labels) scored by 'average_precision' (predict_proba row, pos_label=1 = code 1) and 'neg_brier_score':
    scikit-learn's `label_binarize` shape refusal and brier's binary-branch refusal."""
    var x = _x()
    var y = _codes()
    var one = _logistic(1.0, 0.0)
    var train0 = List[Int32]()
    var test0 = List[Int32]()
    var train1 = List[Int32]()
    var test1 = List[Int32]()
    _lacking_folds(y, train0, test0, train1, test1)
    var given: List[Int] = [Int(train0.unsafe_ptr()), len(train0), Int(test0.unsafe_ptr()), len(test0),
                            Int(train1.unsafe_ptr()), len(train1), Int(test1.unsafe_ptr()), len(test1)]
    var ap = msel_scorer_code("average_precision", False)
    var plan = _plan([one.copy()], x, y, ap)
    _given(plan, given)
    plan.refit = False
    plan.positives = [1]
    plan.label_words = ["0", "1", "2"]
    var words = _refusal(plan^)
    require(words == "ValueError: `y_score` needs to be of shape `(n_samples, n_classes)`, since `y_true` contains "
            + "multiple classes. Got `y_score.shape=(" + String(N // 2) + ",)`.",
            "AP over a binary fold estimator against three labels, got: " + words)
    var brier = msel_scorer_code("neg_brier_score", False)
    var plan2 = _plan([one.copy()], x, y, brier)
    _given(plan2, given)
    plan2.refit = False
    plan2.positives = [0]
    plan2.label_words = ["0", "1", "2"]
    var words2 = _refusal(plan2^)
    require(words2.startswith("The type of the target inferred from y_true is multiclass but should be binary"),
            "brier over a binary fold estimator against three labels, got: " + words2)
    _ = len(train0)
    _ = len(test0)
    _ = len(train1)
    _ = len(test1)
    _ = x^
    _ = y^


def check_lacking_four() raises:
    """(lane mf-msel5c) FOUR classes, fold 0 trained without class 0 (a MULTICLASS fold estimator holding three
    classes) against a test side holding all four: 'average_precision' refuses the (30, 3) scores in scikit-learn's
    words, and 'neg_brier_score' refuses the class count with the labels found in y_true."""
    var x = _x()
    var y = List[Int32](length=N, fill=Int32(0))
    for i in range(N):  # small-loop(fixture: four classes)
        y[i] = Int32((i + i // 5) % 4)
    var one = _logistic(1.0, 0.0)
    one.classes = 4
    var train0 = List[Int32]()
    var test0 = List[Int32]()
    var train1 = List[Int32]()
    var test1 = List[Int32]()
    _lacking_folds(y, train0, test0, train1, test1)
    var given: List[Int] = [Int(train0.unsafe_ptr()), len(train0), Int(test0.unsafe_ptr()), len(test0),
                            Int(train1.unsafe_ptr()), len(train1), Int(test1.unsafe_ptr()), len(test1)]
    var words4: List[String] = ["0", "1", "2", "3"]
    var ap = msel_scorer_code("average_precision", False)
    var masks = _masks([one.copy()])
    var plan = MselSearchPlan([one.copy()], masks^, Int(x.unsafe_ptr()), N, D, Int(y.unsafe_ptr()), 4, F, _glue(),
                              MetricRoute(metrics_program_host_v1, False, False))
    plan.scoring = ap
    _given(plan, given)
    plan.refit = False
    plan.positives = [1]
    plan.label_words = words4.copy()
    var words = _refusal(plan^)
    require(words == "ValueError: `y_score` needs to be of shape `(n_samples, n_classes)`, since `y_true` contains "
            + "multiple classes. Got `y_score.shape=(" + String(N // 2) + ", 3)`.",
            "AP over a three-class fold estimator against four labels, got: " + words)
    var brier = msel_scorer_code("neg_brier_score", False)
    var masks2 = _masks([one.copy()])
    var plan2 = MselSearchPlan([one.copy()], masks2^, Int(x.unsafe_ptr()), N, D, Int(y.unsafe_ptr()), 4, F, _glue(),
                               MetricRoute(metrics_program_host_v1, False, False))
    plan2.scoring = brier
    _given(plan2, given)
    plan2.refit = False
    plan2.positives = [0]
    plan2.label_words = words4.copy()
    var words2 = _refusal(plan2^)
    require(words2 == "ValueError: y_true and y_prob contain different number of classes: 4 vs 3. Please provide "
            + "the true labels explicitly through the labels argument. Classes found in y_true: [0 1 2 3]",
            "brier over a three-class fold estimator against four labels, got: " + words2)
    _ = len(train0)
    _ = len(test0)
    _ = len(train1)
    _ = len(test1)
    _ = x^
    _ = y^


def _with(code: Int, opts: MselScorerOptions) raises -> MselScorer:
    return msel_scorer_row(code, opts, 0)


def check_scorer_words() raises:
    """(lanes mf-msel5c, mf-tail5b) MAKE_SCORER KWARGS AS WORDS: top_k_accuracy's k is the row's metric word
    (and survives a flip), a binary row's given pos_label marks the row and names the label in the refusal, a k
    or a given pos_label on another row refuses, and one scorer word group reads back what was written."""
    var top = msel_scorer_code("top_k_accuracy", False)
    var o = MselScorerOptions()
    o.k = 3
    var three = _with(top, o)
    require(three.metric == 3 and three.sign == 1.0, "top_k k=3 is the metric word")
    o.flip = True
    var flipped = _with(top, o)
    require(flipped.metric == 3 and flipped.sign == -1.0, "a flipped top_k keeps its k")
    var f1 = msel_scorer_code("f1", False)
    var g = MselScorerOptions()
    g.pos_given = True
    var given = _with(f1, g)
    var plain = msel_scorer_row(f1)
    var label_words: List[String] = ["'a'", "'b'", "'c'"]
    require(msel_scorer_pos_words(given, label_words, 2) == "c", "a given string pos_label prints its str")
    require(msel_scorer_pos_words(plain, label_words, 2) == "1", "the default pos_label prints 1")
    var truth: List[Int32] = [0, 0, 1, 1]
    var pred: List[Int32] = [0, 1, 1, 0]
    var notes = List[Float64]()
    var words = String("")
    try:
        _ = msel_scorer_value(MetricRoute(metrics_program_host_v1, False, False), entry_ref_unresolved(), label_words,
                              given, Int(truth.unsafe_ptr()), Int(pred.unsafe_ptr()), 4, 3, 1, 2, notes)
    except error:
        words = String(error)
    require(words == "pos_label=c is not a valid label. It should be one of ['a', 'b']",
            "a given pos_label absent from two present labels, got: " + words)
    # a given pos_label that is not one of the fit's labels: its own word (`pos_word`) after the fit's
    g.pos_word = 3
    var outside = _with(f1, g)
    var with_extra: List[String] = ["'a'", "'b'", "'c'", "'z'"]
    words = String("")
    try:
        _ = msel_scorer_value(MetricRoute(metrics_program_host_v1, False, False), entry_ref_unresolved(), with_extra,
                              outside, Int(truth.unsafe_ptr()), Int(pred.unsafe_ptr()), 4, 3, 1,
                              SCORER_POSITIVE_ABSENT, notes)
    except error:
        words = String(error)
    require(words == "pos_label=z is not a valid label. It should be one of ['a', 'b']",
            "a given pos_label outside the labels names its own word, got: " + words)
    var refused = False
    try:
        var k3 = MselScorerOptions()
        k3.k = 3
        _ = _with(f1, k3)
    except:
        refused = True
    require(refused, "a k on a binary row refuses")
    refused = False
    try:
        _ = _with(top, g)
    except:
        refused = True
    require(refused, "a given pos_label on top_k refuses")
    refused = False
    try:
        var b = MselScorerOptions()
        b.beta = 2.0
        _ = _with(msel_scorer_code("precision_macro", False), b)
    except:
        refused = True
    require(refused, "a beta on a row other than an F row refuses")
    # ONE WORD GROUP reads back what was written (the kind's layout, lane mf-tail5b)
    var w = MselScorerOptions()
    w.flip = True
    w.pos_given = True
    w.pos_word = 4
    w.k = 0
    w.beta = 2.0
    w.zd = 1
    var sels: List[Int32] = [7, 1, 0]
    w.sel_from = 1
    w.sel_n = 2
    var group = List[Float64]()
    group.append(99.0)  # a word before the group
    msel_scorer_group_words(group, f1, 5, w, sels)
    require(len(group) == 1 + SCORER_GROUP_WORDS + 2, "the group is its fixed words and its two labels codes")
    var code = 0
    var positive = 0
    var back = MselScorerOptions()
    var read = List[Int32]()
    var end = msel_scorer_group_read(group, 1, code, positive, back, read)
    require(end == len(group) and code == f1 and positive == 5, "the group's code and positive word")
    require(back.flip and back.pos_given and back.pos_word == 4 and back.beta == 2.0 and back.zd == 1
            and back.sel_n == 2 and back.sel_from == 0 and len(read) == 2 and read[0] == 1 and read[1] == 0,
            "the group's options and labels codes")
    _ = len(truth)
    _ = len(pred)


def check_scorer_options() raises:
    """(lane mf-tail5b) THE OPTIONS' VALUES: fbeta (beta=2) on the F row, an averaged row's `labels`, and
    zero_division, each against the hand value of scikit-learn's formula.
    truth [0, 0, 0, 1], pred [0, 1, 1, 1]: class 1 has tp 1, fp 2, fn 0 (precision 1/3, recall 1); class 0 tp 1,
    fp 0, fn 2 (precision 1, recall 1/3)."""
    var route = MetricRoute(metrics_program_host_v1, False, False)
    var truth: List[Int32] = [0, 0, 0, 1]
    var pred: List[Int32] = [0, 1, 1, 1]
    var notes = List[Float64]()
    # fbeta_score(beta=2) binary, pos_label 1: (1 + 4) p r / (4 p + r) = (5/3) / (7/3) = 5/7; f1 is 1/2
    var b2 = MselScorerOptions()
    b2.beta = 2.0
    var f1 = msel_scorer_code("f1", False)
    var fb = msel_scorer_value(route, _with(f1, b2), Int(truth.unsafe_ptr()), Int(pred.unsafe_ptr()), 4, 2, 1, 1,
                               notes)
    require(abs(fb - 5.0 / 7.0) < 1e-12, "fbeta(beta=2) = 5/7, got " + String(fb))
    var f = msel_scorer_value(route, msel_scorer_row(f1), Int(truth.unsafe_ptr()), Int(pred.unsafe_ptr()), 4, 2, 1,
                              1, notes)
    require(abs(f - 0.5) < 1e-12, "f1 = 1/2, got " + String(f))
    # precision_macro over both labels: (1 + 1/3) / 2 = 2/3; with labels=[1]: 1/3
    var pm = msel_scorer_code("precision_macro", False)
    var whole = msel_scorer_value(route, msel_scorer_row(pm), Int(truth.unsafe_ptr()), Int(pred.unsafe_ptr()), 4, 2,
                                  1, -1, notes)
    require(abs(whole - 2.0 / 3.0) < 1e-12, "precision_macro = 2/3, got " + String(whole))
    var one: List[Int32] = [1]
    var sel = MselScorerOptions()
    sel.sel_n = 1
    var picked = msel_scorer_value(route, msel_scorer_row(pm, sel, Int(one.unsafe_ptr())), Int(truth.unsafe_ptr()),
                                   Int(pred.unsafe_ptr()), 4, 2, 1, -1, notes)
    require(abs(picked - 1.0 / 3.0) < 1e-12, "precision_macro labels=[1] = 1/3, got " + String(picked))
    # zero_division=1: truth [0, 0], pred [0, 0] with labels=[0, 1]: label 1 has no prediction, its precision 1
    var t2: List[Int32] = [0, 0]
    var p2: List[Int32] = [0, 0]
    var both: List[Int32] = [0, 1]
    var zd = MselScorerOptions()
    zd.sel_n = 2
    var warned = List[Float64]()
    var loud = msel_scorer_value(route, msel_scorer_row(pm, zd, Int(both.unsafe_ptr())), Int(t2.unsafe_ptr()),
                                 Int(p2.unsafe_ptr()), 2, 2, 1, -1, warned)
    require(abs(loud - 0.5) < 1e-12 and len(warned) > 0, "zero_division 'warn': 1/2 and the warning")
    zd.zd = 1
    var quiet_notes = List[Float64]()
    var quiet = msel_scorer_value(route, msel_scorer_row(pm, zd, Int(both.unsafe_ptr())), Int(t2.unsafe_ptr()),
                                  Int(p2.unsafe_ptr()), 2, 2, 1, -1, quiet_notes)
    require(abs(quiet - 1.0) < 1e-12 and len(quiet_notes) == 0, "zero_division=1: 1.0 and no warning")
    _ = len(truth)
    _ = len(pred)
    _ = len(one)
    _ = len(t2)
    _ = len(p2)
    _ = len(both)



def check_given_targets() raises:
    """Case 20 (module note)."""
    require(MSEL_GIVEN_F64 == MEMBER_F64 and MSEL_GIVEN_I32 == MEMBER_I32 and MSEL_GIVEN_I64 == MEMBER_I64,
            "the given dtype words are the member ABI's")
    var close = 1.0 + 1.0 / 1099511627776.0  # 1 + 2^-40: one float32 word with 1.0
    var y64: List[Float64] = [1.0, close, 2.0, 1.0]
    var y32: List[Float32] = [Float32(1.0), Float32(close), Float32(2.0), Float32(1.0)]
    var given = msel_target_codes_run(Int(y32.unsafe_ptr()), 4, Int(y64.unsafe_ptr()), MSEL_GIVEN_F64)
    require(given.m == 3, "three given float64 values, three classes")
    require(given.codes[0] == 0 and given.codes[1] == 1 and given.codes[2] == 2 and given.codes[3] == 0,
            "np.unique's inverse over the given y")
    var alone = msel_target_codes_run(Int(y32.unsafe_ptr()), 4, 0, 0)
    require(alone.m == 2, "the float32 words alone merge the close pair")
    var i64: List[Int64] = [16777217, 16777216, 5]
    var f32: List[Float32] = [Float32(16777217.0), Float32(16777216.0), Float32(5.0)]
    var ints = msel_target_codes_run(Int(f32.unsafe_ptr()), 3, Int(i64.unsafe_ptr()), MSEL_GIVEN_I64)
    require(ints.m == 3 and ints.codes[0] == 2 and ints.codes[1] == 1 and ints.codes[2] == 0, "int64 given targets")
    var i32: List[Int32] = [3, -1, 3]
    var g32: List[Float32] = [Float32(3.0), Float32(-1.0), Float32(3.0)]
    var small = msel_target_codes_run(Int(g32.unsafe_ptr()), 3, Int(i32.unsafe_ptr()), MSEL_GIVEN_I32)
    require(small.m == 2 and small.codes[0] == 1 and small.codes[1] == 0 and small.codes[2] == 1, "int32 given targets")
    _ = len(y64)
    _ = len(y32)
    _ = len(i64)
    _ = len(f32)
    _ = len(i32)
    _ = len(g32)


def check_multi_lacking_helpers() raises:
    """Case 21 (module note)."""
    # rows x 2 codes: output 0 over 3 classes, output 1 over 2; train rows 0, 1, 3
    var codes: List[Int32] = [0, 0, 2, 1, 1, 0, 2, 1]
    var train: List[Int32] = [0, 1, 3]
    var h0 = _held_codes_col(Int(codes.unsafe_ptr()), Int(train.unsafe_ptr()), 3, 3, 2, 0)
    var h1 = _held_codes_col(Int(codes.unsafe_ptr()), Int(train.unsafe_ptr()), 3, 2, 2, 1)
    require(len(h0) == 2 and h0[0] == 0 and h0[1] == 2, "output 0 holds classes 0 and 2")
    require(len(h1) == 0, "output 1 holds every class")
    var held = List[List[Int32]]()
    held.append(h0.copy())
    held.append(List[Int32]())
    var pred32: List[Int32] = [0, 1, 1, 0, 1, 1]
    _map_multi_codes(Int(pred32.unsafe_ptr()), MEMBER_I32, 3, 2, Int(held.unsafe_ptr()), 0)
    require(pred32[0] == 0 and pred32[2] == 2 and pred32[4] == 2, "output 0's fold codes mapped to its classes")
    require(pred32[1] == 1 and pred32[3] == 0 and pred32[5] == 1, "output 1 unchanged")
    var pred64: List[Int64] = [1, 1, 0, 0]
    _map_multi_codes(Int(pred64.unsafe_ptr()), MEMBER_I64, 2, 2, Int(held.unsafe_ptr()), 0)
    require(pred64[0] == 2 and pred64[2] == 0 and pred64[1] == 1, "int64 codes mapped the same way")
    # a unit's answer for 2 test rows (rows 2 and 0 of N = 3): output 0's 2 held columns, then output 1's 2
    var src: List[Float64] = [0.25, 0.75, 0.5, 0.5, 0.875, 0.125, 0.0625, 0.9375]
    var rows: List[Int32] = [2, 0]
    var b0 = List[Float64](length=9, fill=-1.0)
    var b1 = List[Float64](length=6, fill=-1.0)
    var dsts: List[Int] = [Int(b0.unsafe_ptr()), Int(b1.unsafe_ptr())]
    var counts: List[Int] = [3, 2]
    _place_multi_proba(Int(src.unsafe_ptr()), MEMBER_F64, 4, 2, Int(rows.unsafe_ptr()), Int(dsts.unsafe_ptr()), 2,
                       Int(counts.unsafe_ptr()), Int(held.unsafe_ptr()), 0)
    require(b0[6] == 0.25 and b0[7] == 0.0 and b0[8] == 0.75, "row 2 of output 0 at classes 0 and 2")
    require(b0[0] == 0.875 and b0[1] == 0.0 and b0[2] == 0.125, "row 0 of output 0 at classes 0 and 2")
    require(b0[3] == -1.0, "row 1 is not this unit's")
    require(b1[4] == 0.5 and b1[5] == 0.5 and b1[0] == 0.0625 and b1[1] == 0.9375, "output 1 placed as it is")
    var refused = False
    try:
        _place_multi_proba(Int(src.unsafe_ptr()), MEMBER_F64, 5, 1, Int(rows.unsafe_ptr()), Int(dsts.unsafe_ptr()),
                           2, Int(counts.unsafe_ptr()), Int(held.unsafe_ptr()), 0)
    except:
        refused = True
    require(refused, "a member answer of the wrong width refuses")
    _ = len(codes)
    _ = len(train)
    _ = len(held)
    _ = len(src)
    _ = len(rows)
    _ = len(dsts)
    _ = len(b0)
    _ = len(b1)
    _ = len(counts)

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
        elif which == 10:
            check_ap_absent()
        elif which == 11:
            check_lacking_fold()
        elif which == 12:
            check_target_free()
        elif which == 13:
            check_kmeans_score()
        elif which == 14:
            check_kmeans_search()
        elif which == 15:
            check_lacking_proba()
        elif which == 16:
            check_lacking_ap_brier()
        elif which == 17:
            check_lacking_four()
        elif which == 18:
            check_scorer_words()
        elif which == 19:
            check_given_targets()
        elif which == 20:
            check_multi_lacking_helpers()
        else:
            check_scorer_options()
    check_cases("msel_search_run", CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
