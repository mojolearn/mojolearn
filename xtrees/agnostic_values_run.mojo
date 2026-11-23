# SPDX-License-Identifier: Apache-2.0
"""KernelExplainer and PermutationExplainer as typed RUNs (lane bm-rest-shap, 2026-11-23).

The chunk loops that bindings/trees_agnostic_native_door.mojo drove over Python objects: the coalition schedule,
the synthetic rows, the explained model on them, the background means, the weighted solve (Kernel SHAP) or the
permutation fold (Permutation SHAP) and the logit link. Every stage is the x_trees glue operation the door called
as a Python entry before (xtrees/glue_runs.mojo, `XT_KSHAP_*`, `XT_PSHAP_*`, `XT_AGN_*`, `XT_BLOCK_MEAN`,
`XT_LOGIT`), reached here through the resolved glue entry: no Python object, import or callback in this file.

THE MODEL is an `AgnModel`: a typed evaluator of the explained estimator on a float32 row buffer.
`AgnMember` is the NATIVE MODEL HANDLE: the estimator's held member state (core/member_abi.mojo) and the member
operation of the explained method (predict of a regressor kind, predict_proba of a classifier). The families
with a published member entry serve it: the forests (RandomForest, ExtraTrees, the decision trees), DART and
the GBDT host and device images, and every other family that publishes one. `AgnMeta` (lane mf-shap3,
2026-11-23) is a META-ESTIMATOR over such members (Voting soft and hard, Bagging, Stacking, OneVsRest,
CalibratedClassifierCV, AdaBoost, MultiOutputRegressor), evaluated by the RUN its public operation runs. The
door keeps one more evaluator for an estimator (or a member of one) whose class has no member entry (its compiled
door per chunk, with the GIL held).

PARALLEL. Chunks are independent (each row's coalitions and permutations are drawn from its own stream
(seed, row): chunk boundaries move no bit), and each chunk writes its own rows of phi. On the CPU column with a
member whose kind is `MEMBER_GIL_FREE`, the chunks run across the allocation's cores (`host_predict_workers`),
one evaluator, one synthetic buffer and one member call per worker; the in-flight synthetic words stay under the
one-chunk budget (`AGN_BUDGET`). A GPU glue image (one device context, the pooled device buffer of
MOJOLEARN_AGN_IDN_SYN_POOL) and every other evaluator run the chunks in order on the calling thread.
"""
from std.math import min, max
from core.entry_table import EntryRef
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_workers
from core.member_abi import (
    MemberRef, MemberPlan, MemberCall, MemberMatrix, member_call, member_call_of, MEMBER_F64, MEMBER_I32,
    MEMBER_I64,
)
from core.hotpath_impl import cast_elements, HP_F32, HP_F64, HP_I32, HP_I64
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from core.admitted_cast_elements import admitted_cast_elements
from xtrees.trees_agnostic_native_run import _chunk
from xtrees.trees_meta_vote_run import (
    VoteMember, VoteRows, VOTE_ALL_ANSWERED, vote_soft_run, vote_regress_run, vote_codes_run, vote_transform_run,
    bag_predict_run, stack_place_run, ovr_assemble_run, multioutput_proba_run, calibrated_proba_run, CalPair,
    CalFit, ada_classifier_run, ada_regressor_run, ADA_PROBA, member_f32_block, _member_codes_into,
)
from std.memory import unsafe_memcpy
from core.host_finite_scan import host_all_finite_f32
from core.linear_multi_calls import rm_predict
from core.neighbors_native_driver import XnDriver
from core.admitted_refusals import admitted_finite_refuse
from neighbors.label_propagation_run import LpQueryRequest, lp_query_run
from xtrees.glue_calls import (
    xt_kshap_schedule, xt_kshap_synth, xt_kshap_solve, xt_kshap_means, xt_kshap_solve_ey, xt_kshap_solve_model,
    xt_pshap_synth, xt_pshap_values, xt_pshap_dsynth, xt_pshap_dvalues, xt_pshap_values_linear,
    xt_pshap_values_model, xt_agn_model_load, xt_agn_model_release, xt_agn_pool_release, xt_block_mean, xt_logit,
    xt_gather_f32, xt_binary_proba, xt_scatter,
)

comptime _I64 = MutPointer[Int64, MutUntrackedOrigin]
comptime _F32 = MutPointer[Float32, MutUntrackedOrigin]

#: xtrees/agnostic.mojo AGN_BUDGET: synthetic words per explainer chunk (trees_agnostic_native_run `_chunk`).
comptime AGN_BUDGET = 1 << 25
#: MOJOLEARN_IDN_PSHAP_RESIDENT: 2^28 float32 words (1 GiB of synthetic rows on the device), a memory bound that
#: fits every supported GPU.
comptime AGN_BUDGET_RESIDENT = 1 << 28
#: x_trees_fast_switches bits (providers/python/xtrees_api_pyapi_door.mojo XTREES_FAST_SWITCHES).
comptime AGN_SW_KSHAP_FAST_BATCH = 8
comptime AGN_SW_IDN_SYN_POOL = 16
comptime AGN_SW_PSHAP_DELTA = 32


trait AgnModel(Movable & Deinitable):
    """The explained estimator on float32 C-order rows of `d` features, answering float32 (rows, k)."""

    def explained(mut self) raises -> Int:
        """The model on the explained rows (the request's `x`, `n` rows): the address of float32 (n, k) words,
        valid until the evaluator is destroyed."""
        ...

    def synthetic(mut self, count: Int, filled: Bool) raises -> Int:
        """A float32 input buffer of `count` words (zeroed when `filled`) for the next `rows` call; valid until
        the next `synthetic` call. A buffer of the same size is reused."""
        ...

    def rows(mut self, n: Int) raises -> Int:
        """The model on the first `n` rows of the synthetic buffer: the address of float32 (n, k) words, valid
        until the next `rows` or `synthetic` call."""
        ...

    def workers(self) -> Int:
        """How many evaluators may run at once (1: the calling thread only)."""
        ...

    def worker(self) raises -> Self:
        """A parallel worker's own evaluator over the same fitted state (only asked when `workers() > 1`)."""
        ...


def agn_labels_run(glue: EntryRef, labels: Int, n_labels: Int, codes: Int, n: Int, dst: Int, split: Bool) raises:
    """THE LABEL MAP of a classifier explained through predict (lane mf-shap2, 2026-11-23): `dst` (float32 n) =
    `labels[codes[i]]`, the float32 label of each int32 class code, through the x_trees glue gather (`labels` is a
    float32 `n_labels` x 1 table: the estimator's `classes_` narrowed to float32 once at UNPACK, the conversion the
    door's `_evaluate` made of the public predict's labels, element by element, so no bit differs). `split` (the
    CPU column's serial chunk): the rows in tasks across the host's cores, one gather per task; else one gather (a
    parallel worker's chunk: the workers are the split; a GPU image: the device kernel over every row). A code
    outside `0 .. n_labels-1` raises the gather's words before any cell is written (the caller's refusal rule)."""
    if n <= 0:
        return
    var col0 = List[Int32](length=1, fill=Int32(0))
    var c0 = Int(col0.unsafe_ptr())
    var tasks = host_predict_task_count_work(n, n) if split else 1
    if tasks <= 1:
        _ = xt_gather_f32(glue, labels, codes, c0, dst, n_labels, 1, n, 1)
        _ = col0^
        return
    var chunk = host_predict_chunk(n, tasks)
    var failed = List[Int](length=tasks, fill=0)
    var fp = failed.unsafe_ptr()

    def task(unit: Int) {imm glue, imm labels, imm n_labels, imm codes, imm n, imm dst, imm chunk, imm c0, imm fp}:
        var r0 = unit * chunk
        var rows = min(chunk, n - r0)
        if rows <= 0:
            return
        try:
            _ = xt_gather_f32(glue, labels, codes + 4 * r0, c0, dst + 4 * r0, n_labels, 1, rows, 1)
        except:
            fp.unsafe_store(unit, 1)

    host_parallelize(task, tasks)
    _ = col0^  # the column index is read through every task above
    for t in range(tasks):  # driver-loop: over the tasks, after the join
        if failed[t] != 0:
            raise Error("x_trees gather: row index out of range")
    _ = failed^


def agn_narrow(src: Int, dst: Int, count: Int, split: Bool) raises:
    """`dst` (float32 `count`) = the float64 words at `src` rounded to float32 element by element, the conversion
    the door's `_evaluate` made of a public float64 output (core/hotpath_impl.mojo `cast_elements`). `split`: the
    CPU column's serial chunk, across the host's cores; else on the calling thread (a parallel worker's chunk, a
    GPU image's). A word float32 cannot hold raises."""
    var bad = (admitted_cast_elements(src, HP_F64, dst, HP_F32, count) if split
               else cast_elements(src, HP_F64, dst, HP_F32, count))
    if bad != 0:
        raise Error("ValueError: the explained model output is not representable as float32")


def agn_ints_run(src: Int, src_type: Int, dst: Int, count: Int, split: Bool) raises:
    """`dst` (float32 `count`) = the int32 (`HP_I32`) or int64 (`HP_I64`) values at `src`, each converted to
    float32 (core/hotpath_impl.mojo `cast_elements`: the conversion the door's `_evaluate` made of a public integer
    output, `labels_cast` to "<f4", element by element, so no bit differs). `split`: the CPU column's serial chunk,
    across the host's cores; else on the calling thread (lane mf-shap3b, 2026-11-23)."""
    var bad = (admitted_cast_elements(src, src_type, dst, HP_F32, count) if split
               else cast_elements(src, src_type, dst, HP_F32, count))
    if bad != 0:
        raise Error("ValueError: the explained model output is not representable as float32")


def agn_select_run(glue: EntryRef, x: Int, n: Int, d: Int, cols: Int, width: Int, order: Int, dst: Int,
                   split: Bool) raises:
    """THE COLUMN SELECTION of a model that reads some of the explained columns (lane mf-shap3b, 2026-11-23; an
    RFE's support): `dst` (float32 n x width) = the columns at `cols` (int32, `width` of them) of the float32 n x d
    rows at `x`, through the x_trees glue gather (a copy of the words). `order`: int32 row numbers 0 .. n-1.
    `split` (the CPU column's serial chunk): row ranges in tasks across the host's cores, one gather per task (each
    range is gathered by the first row numbers of `order`); else one gather (a parallel worker's chunk, a GPU
    image's)."""
    if n <= 0:
        return
    var tasks = host_predict_task_count_work(n, n * max(1, width)) if split else 1
    if tasks <= 1:
        _ = xt_gather_f32(glue, x, order, cols, dst, n, d, n, width)
        return
    var chunk = host_predict_chunk(n, tasks)
    var failed = List[Int](length=tasks, fill=0)
    var fp = failed.unsafe_ptr()

    def task(unit: Int) {imm glue, imm x, imm n, imm d, imm cols, imm width, imm order, imm dst, imm chunk, imm fp}:
        var r0 = unit * chunk
        var rows = min(chunk, n - r0)
        if rows <= 0:
            return
        try:
            _ = xt_gather_f32(glue, x + 4 * r0 * d, order, cols, dst + 4 * r0 * width, rows, d, rows, width)
        except:
            fp.unsafe_store(unit, 1)

    host_parallelize(task, tasks)
    for t in range(tasks):  # driver-loop: over the tasks, after the join
        if failed[t] != 0:
            raise Error("x_trees gather: column index out of range")
    _ = failed^


struct AgnMember(AgnModel):
    """THE NATIVE MODEL HANDLE: one held member state of the explained estimator and the member operation of the
    explained method. `plan` is the address of the caller's `MemberPlan`; it outlives the RUN (the caller's
    UNPACK holds it). A member refusal (a kind that tests its rows refusing a non-finite chunk) raises the
    member's own words out of the RUN (lane mf-shap3, 2026-11-23: before, the caller then answered the whole
    call again through the model's door, `_DoorModel`, only to raise that door's words).
    `wide` (lane mf-shaprfe, 2026-11-23): the member writes the operation's output as float64 (or, on the label
    route, its codes as int64: lane mf-shap3) (the predict_proba
    of LogisticRegression, SVC, DART and GP classifiers, the naive Bayes and discriminant members on a
    float64-proba route): it writes into a float64 buffer of the chunk and the RUN reads that narrowed to
    float32, the conversion the door's `_evaluate` made of the public float64 output (core/hotpath_impl.mojo
    `cast_elements`, element by element, so no bit differs from the door's). `host`: the glue image is the CPU
    column (the request's `host`). Before, such a class was
    evaluated through the door, serially with the GIL held.
    `labels` (lane mf-shap2, 2026-11-23; 0: not this route): a classifier explained through predict (LinearSVC,
    RidgeClassifier, PassiveAggressiveClassifier, Perceptron, SGDClassifier with a hinge loss,
    SVC(probability=False), NearestCentroid). The member writes int32 class CODES into a reused buffer of the
    chunk and `agn_labels_run` maps them to the float32 label table at `labels` (`n_labels` words, the caller's
    UNPACK holds it) through the glue gather `glue`: one column (k is 1). Before, such a class was evaluated
    through the door (its public predict per chunk, serially with the GIL held).
    `ints` (lane mf-shap3b, 2026-11-23): a model that is not a classifier whose member writes integer values
    (IsolationForest's predict, int32 1 / -1; the other outlier detectors' predict, int64): the member writes them
    into a reused integer buffer of the chunk and the RUN widens them to float32 (`agn_ints_run`, the cast the
    door's `_evaluate` made of the public int output, element by element). Before, such a class had no native
    route. `cols` / `width` (lane mf-shap3b; 0: every column): the member reads `width` of the `d` explained
    columns, the int32 column numbers at `cols` (the caller's, alive through the RUN), gathered from each chunk's
    rows into a reused block (`agn_select_run`): an RFE explained through its fitted estimator over its support
    columns, exactly `RFE.predict` = `estimator_.predict(X[:, support_])` (the gather copies the float32 words)."""
    var member: MemberRef
    var plan: Int
    var state: Int
    var operation: Int
    var x: Int
    var n: Int
    var d: Int
    var k: Int
    var parallel: Bool
    var wide: Bool
    var host: Bool
    var glue: EntryRef
    var labels: Int
    var n_labels: Int
    var call: MemberCall
    var fx: List[Float32]
    var syn: List[Float32]
    var out: List[Float32]
    var wide_out: List[Float64]
    var codes: List[Int32]
    var codes64: List[Int64]
    var ints: Bool
    var cols: Int
    var width: Int
    var sub: List[Float32]
    var order: List[Int32]

    def __init__(out self, member: MemberRef, plan: Int, state: Int, operation: Int, x: Int, n: Int, d: Int, k: Int,
                 parallel: Bool, wide: Bool, host: Bool, glue: EntryRef, labels: Int, n_labels: Int,
                 ints: Bool = False, cols: Int = 0, width: Int = 0) raises:
        self.member = member
        self.plan = plan
        self.state = state
        self.operation = operation
        self.x = x
        self.n = n
        self.d = d
        self.k = k
        self.parallel = parallel
        self.wide = wide
        self.host = host
        self.glue = glue
        self.labels = labels
        self.n_labels = n_labels
        self.call = member_call_of(MutPointer[MemberPlan, MutUntrackedOrigin](unsafe_from_address=plan)[], state)
        self.fx = List[Float32]()
        self.syn = List[Float32]()
        self.out = List[Float32]()
        self.wide_out = List[Float64]()
        self.codes = List[Int32]()
        self.codes64 = List[Int64]()
        self.ints = ints
        self.cols = cols
        self.width = width
        self.sub = List[Float32]()
        self.order = List[Int32]()

    def _select(mut self, x: Int, n: Int) raises -> Int:
        """The member's columns of the chunk's rows (`cols`, `width` of them) gathered into the reused block."""
        var w = self.width
        if len(self.sub) < n * w:
            self.sub = List[Float32]()
            # Unfilled: the gather writes every cell before the member reads it.
            self.sub = List[Float32](unsafe_uninit_length=max(1, n * w))
        if len(self.order) < n:
            self.order = List[Int32]()
            self.order = List[Int32](unsafe_uninit_length=max(1, n))
            for i in range(n):  # small-loop(the row numbers the glue gather addresses, once per larger chunk)
                self.order[i] = Int32(i)
        var at = Int(self.sub.unsafe_ptr())
        agn_select_run(self.glue, x, n, self.d, self.cols, w, Int(self.order.unsafe_ptr()), at,
                       self.host and not self.parallel)
        return at

    def _run(mut self, x: Int, n: Int, dst: Int) raises:
        var d = self.d
        var k = self.k
        var member = self.member
        var operation = self.operation
        if self.cols != 0:
            self.call.x = MemberMatrix(self._select(x, n), n, self.width)
        else:
            self.call.x = MemberMatrix(x, n, d)
        if self.labels != 0 or self.ints:
            var count = n * k if self.ints else n
            if len(self.codes) < count:
                self.codes = List[Int32]()
                # Unfilled: the member writes every value before the label map or the widening reads it.
                self.codes = List[Int32](unsafe_uninit_length=count)
            if self.wide:
                # int64 codes (lane mf-shap3) or values: written here, narrowed to int32 before the label map, or
                # widened to float32 straight from int64.
                if len(self.codes64) < count:
                    self.codes64 = List[Int64]()
                    self.codes64 = List[Int64](unsafe_uninit_length=count)
                self.call.dst = MemberMatrix(Int(self.codes64.unsafe_ptr()), n, k if self.ints else 1, MEMBER_I64)
            else:
                self.call.dst = MemberMatrix(Int(self.codes.unsafe_ptr()), n, k if self.ints else 1, MEMBER_I32)
        elif not self.wide:
            self.call.dst = MemberMatrix(dst, n, k)
        else:
            if len(self.wide_out) < n * k:
                self.wide_out = List[Float64]()
                # Unfilled: the member writes every cell of dst before the narrowing reads it.
                self.wide_out = List[Float64](unsafe_uninit_length=n * k)
            self.call.dst = MemberMatrix(Int(self.wide_out.unsafe_ptr()), n, k, MEMBER_F64)
        # A refusal raises the member's own words out of the RUN.
        _ = member_call(member, operation, self.call)
        if self.ints:
            if self.wide:
                agn_ints_run(Int(self.codes64.unsafe_ptr()), HP_I64, dst, n * k, self.host and not self.parallel)
            else:
                agn_ints_run(Int(self.codes.unsafe_ptr()), HP_I32, dst, n * k, self.host and not self.parallel)
        elif self.labels != 0:
            if self.wide:
                # The int64 codes narrowed once, as the door's int32 cast of them narrowed (every code is a
                # class index; the gather below refuses one outside the table).
                _ = cast_elements(Int(self.codes64.unsafe_ptr()), HP_I64, Int(self.codes.unsafe_ptr()), HP_I32, n)
            agn_labels_run(self.glue, self.labels, self.n_labels, Int(self.codes.unsafe_ptr()), n, dst,
                           self.host and not self.parallel)
        elif self.wide:
            var src = Int(self.wide_out.unsafe_ptr())
            # The CPU column's serial chunk across the host's cores, as the door's cast did; a parallel
            # worker's chunk (the workers are the split) and a GPU image's (no host threads in a GPU
            # binding) on the calling thread.
            agn_narrow(src, dst, n * k, self.host and not self.parallel)

    def explained(mut self) raises -> Int:
        if len(self.fx) != self.n * self.k:
            # Unfilled: the member writes every cell of dst before anything reads it.
            self.fx = List[Float32](unsafe_uninit_length=self.n * self.k)
        var dst = Int(self.fx.unsafe_ptr())
        var x = self.x
        var n = self.n
        self._run(x, n, dst)
        return dst

    def synthetic(mut self, count: Int, filled: Bool) raises -> Int:
        if len(self.syn) != count:
            self.syn = List[Float32]()
            if filled:
                self.syn = List[Float32](length=count, fill=Float32(0))
            else:
                self.syn = List[Float32](unsafe_uninit_length=count)
        # A reused buffer is not zeroed again: every synthetic writer writes each cell the model then reads.
        return Int(self.syn.unsafe_ptr())

    def rows(mut self, n: Int) raises -> Int:
        if n * self.d > len(self.syn):
            raise Error("RuntimeError: agnostic explainer: synthetic rows exceed their buffer")
        if len(self.out) < n * self.k:
            self.out = List[Float32]()
            self.out = List[Float32](unsafe_uninit_length=n * self.k)
        var dst = Int(self.out.unsafe_ptr())
        var src = Int(self.syn.unsafe_ptr())
        self._run(src, n, dst)
        return dst

    def workers(self) -> Int:
        return host_predict_workers() if self.parallel else 1

    def worker(self) raises -> Self:
        return Self(self.member, self.plan, self.state, self.operation, self.x, self.n, self.d, self.k,
                    self.parallel, self.wide, self.host, self.glue, self.labels, self.n_labels, self.ints, self.cols,
                    self.width)


struct AgnLabelProp(AgnModel):
    """LabelPropagation / LabelSpreading AS THE MODEL (lane mf-shap3b, 2026-11-23): its `predict_proba`, the
    typed query body its public method runs (neighbors/label_propagation_run.mojo `lp_query_run`, over `driver`,
    the x_neighbors native operations entry of the image the estimator's own query calls), with `request` the
    fitted words read once at UNPACK (bindings/neighbors_lp_query_unpack.mojo `lp_query_unpack`, the one body
    the public query also reads them through; its `fit_x` / `distributions` addresses are the caller's, alive
    through the RUN). Each chunk writes float32 (rows, classes) in place, the public method's words. A non-finite
    chunk is refused by the query's deferred finite word in the public method's own words
    (`admitted_finite_refuse`, `name` its class). Before, the class had no native route.
    SERIAL CHUNKS: one query at a time on the calling thread (each query splits its own rows on the image; the
    driver entry has no parallel-worker trace), with the GIL released by the caller around the whole RUN, as the
    public method releases it around its query."""
    var driver: XnDriver
    var request: LpQueryRequest
    var name: String
    var x: Int
    var n: Int
    var d: Int
    var k: Int
    var fx: List[Float32]
    var syn: List[Float32]
    var out: List[Float32]

    def __init__(out self, driver: XnDriver, request: LpQueryRequest, name: String, x: Int, n: Int, d: Int):
        self.driver = driver.copy()
        self.request = request.copy()
        self.name = name.copy()
        self.x = x
        self.n = n
        self.d = d
        self.k = request.classes
        self.fx = List[Float32]()
        self.syn = List[Float32]()
        self.out = List[Float32]()

    def _run(mut self, x: Int, n: Int, dst: Int) raises:
        self.request.q = x
        self.request.rows = n
        self.request.features = self.d
        self.request.dst = dst
        self.request.labels = 0
        self.request.want_labels = False
        var status = lp_query_run(self.driver, self.request)
        if status != 0:
            admitted_finite_refuse(status, self.name)

    def explained(mut self) raises -> Int:
        if len(self.fx) != self.n * self.k:
            # Unfilled: the query writes every cell before anything reads it.
            self.fx = List[Float32](unsafe_uninit_length=max(1, self.n * self.k))
        var dst = Int(self.fx.unsafe_ptr())
        var x = self.x
        var n = self.n
        self._run(x, n, dst)
        return dst

    def synthetic(mut self, count: Int, filled: Bool) raises -> Int:
        if len(self.syn) != count:
            self.syn = List[Float32]()
            if filled:
                self.syn = List[Float32](length=count, fill=Float32(0))
            else:
                self.syn = List[Float32](unsafe_uninit_length=count)
        return Int(self.syn.unsafe_ptr())

    def rows(mut self, n: Int) raises -> Int:
        if n * self.d > len(self.syn):
            raise Error("RuntimeError: agnostic explainer: synthetic rows exceed their buffer")
        if len(self.out) < n * self.k:
            self.out = List[Float32]()
            self.out = List[Float32](unsafe_uninit_length=max(1, n * self.k))
        var dst = Int(self.out.unsafe_ptr())
        var src = Int(self.syn.unsafe_ptr())
        self._run(src, n, dst)
        return dst

    def workers(self) -> Int:
        return 1

    def worker(self) raises -> Self:
        raise Error("RuntimeError: the label propagation query runs its chunks on the calling thread")


#: `AgnMetaPlan.kind` (lane mf-shap3, 2026-11-23): the meta-estimator explained, by the RUN of its public
#: operation (xtrees/trees_meta_vote_run.mojo), one body per operation.
comptime AGN_META_SOFT = 0          # VotingClassifier(voting='soft').predict_proba: vote_soft_run
comptime AGN_META_REGRESS = 1       # VotingRegressor.predict: vote_regress_run
comptime AGN_META_HARD = 2          # VotingClassifier(voting='hard').predict: vote_codes_run, labels as float32
comptime AGN_META_BAG = 3           # BaggingClassifier.predict_proba, BaggingRegressor.predict: bag_predict_run
comptime AGN_META_STACK = 4         # StackingClassifier.predict_proba, StackingRegressor.predict: stack_place_run
comptime AGN_META_OVR = 5           # OneVsRestClassifier.predict_proba: ovr_assemble_run / the binary column
comptime AGN_META_CALIBRATED = 6    # CalibratedClassifierCV.predict_proba: calibrated_proba_run
comptime AGN_META_ADA = 7           # AdaBoostClassifier.predict_proba, AdaBoostRegressor.predict: ada_*_run
comptime AGN_META_COLUMNS = 8       # MultiOutputRegressor.predict: vote_transform_run, member j at column j
comptime AGN_META_RIDGE = 9         # MultiOutputRegressor over Ridge members: its one multi-output ridge predict


struct AgnMetaPlan(Copyable, Movable):
    """The typed UNPACK of a meta-estimator as the explained model (lane mf-shap3, 2026-11-23). Every member is a
    native `VoteMember` (`plan` the address of the caller's MemberPlan, held unmoved through the RUN); every table
    address (class columns, feature columns, labels, knots, coefficients) is the caller's, alive through the RUN.

    `k`: the explained output's columns. `classes`: the ensemble's class count (HARD, BAG, STACK, OVR,
    CALIBRATED, ADA). `classifier`: BAG, STACK and ADA explain predict_proba (else predict). `labels` /
    `n_labels`: HARD, the vote's `classes_` as float32. `features` / `widths`: BAG, per member its int32 feature
    columns (0: every column, `_BaggingBase._sub_X`'s own rule) and their count. `at`: OVR, per member its
    positive column. `pairs` / `cals`: CALIBRATED. `stack_kinds` / `binary` / `passthrough` / `meta_width` /
    `final_est` (one member, read over the meta features) / `final_tests` (its kind refuses a non-finite row
    itself): STACK. `stack_labels` / `stack_n_labels` (STACK, per member; lane mf-shap3b, 2026-11-23): a CLASSIFIER
    member the Stacking asks for predict (`stack_method_` 'predict'): the address of its `classes_` as a float32
    table and its length (0: any other member); its member writes int32 or int64 codes and its meta-feature
    column is their labels, as `_StackingBase._out` places the member's predict. `alphas`: ADA, the fsum'd member weights (`estimator_weights_`, every word). `ridge` (one
    entry) / `ridge_coef` / `ridge_icpt`: RIDGE (`k` outputs)."""
    var kind: Int
    var members: List[VoteMember]
    var k: Int
    var classes: Int
    var classifier: Bool
    var labels: Int
    var n_labels: Int
    var features: List[Int]
    var widths: List[Int]
    var at: List[Int]
    var pairs: List[CalPair]
    var cals: List[CalFit]
    var stack_kinds: List[Int]
    var binary: Bool
    var passthrough: Bool
    var meta_width: Int
    var final_est: List[VoteMember]
    var final_tests: Bool
    var stack_labels: List[Int]
    var stack_n_labels: List[Int]
    var alphas: List[Float64]
    var ridge: List[EntryRef]
    var ridge_coef: Int
    var ridge_icpt: Int

    def __init__(out self, kind: Int):
        self.kind = kind
        self.members = List[VoteMember]()
        self.k = 0
        self.classes = 0
        self.classifier = False
        self.labels = 0
        self.n_labels = 0
        self.features = List[Int]()
        self.widths = List[Int]()
        self.at = List[Int]()
        self.pairs = List[CalPair]()
        self.cals = List[CalFit]()
        self.stack_kinds = List[Int]()
        self.binary = False
        self.passthrough = False
        self.meta_width = 0
        self.final_est = List[VoteMember]()
        self.final_tests = False
        self.stack_labels = List[Int]()
        self.stack_n_labels = List[Int]()
        self.alphas = List[Float64]()
        self.ridge = List[EntryRef]()
        self.ridge_coef = 0
        self.ridge_icpt = 0


def _agn_refusal(m: VoteMember, x: Int, n: Int, d: Int) raises:
    """A member refused inside a meta RUN (the RUNs answer its index and write nothing more): the same member
    call again on the same rows, so that the member's own words are raised (the refusal path only)."""
    var plan = MutPointer[MemberPlan, MutUntrackedOrigin](unsafe_from_address=m.plan)
    var call = member_call_of(plan[], m.state)
    call.x = MemberMatrix(x, n, d)
    # Eight bytes per cell hold every element type a member writes.
    var scratch = List[Float64](length=max(1, n * max(1, m.cols)), fill=0.0)
    call.dst = MemberMatrix(Int(scratch.unsafe_ptr()), n, m.cols, m.dtype)
    _ = member_call(plan[].member, m.operation, call)
    _ = scratch^
    raise Error("RuntimeError: a member of the explained model refused the explained rows")


struct AgnMeta(AgnModel):
    """A META-ESTIMATOR AS THE MODEL (lane mf-shap3, 2026-11-23; the soft and regressor votes since lane mf-shap2):
    the explained output of `plan` (`AgnMetaPlan`) on float32 rows, by the RUN its public operation runs, so the
    fold, the order and the words are that operation's (no bit differs from its public predict). A float64
    public output (every vote, Bagging, Stacking classifier, OneVsRest, Calibrated, AdaBoost classifier,
    MultiOutputRegressor) is written into a float64 zero block of the chunk and narrowed to float32
    (`agn_narrow`, the conversion the door's `_evaluate` made of it); a float32 one (AdaBoostRegressor, the
    multi-output ridge, a StackingRegressor's final predict) is written in place; a hard vote's codes are mapped
    to its float32 labels (`agn_labels_run`). `glue` is the meta's own x_trees glue (the image its public
    operation folds with). A member that refuses raises its own words. `parallel`: every member's kind is
    `MEMBER_GIL_FREE` and the glue is the CPU column (the chunks may run on parallel workers). Before, every
    meta-estimator but the soft and regressor votes was evaluated through the door per chunk, serially with the
    GIL held."""
    var plan: AgnMetaPlan
    var x: Int
    var n: Int
    var d: Int
    var k: Int
    var parallel: Bool
    var host: Bool
    var glue: EntryRef
    var fx: List[Float32]
    var syn: List[Float32]
    var out: List[Float32]

    def __init__(out self, plan: AgnMetaPlan, x: Int, n: Int, d: Int, parallel: Bool, host: Bool, glue: EntryRef):
        self.plan = plan.copy()
        self.x = x
        self.n = n
        self.d = d
        self.k = plan.k
        self.parallel = parallel
        self.host = host
        self.glue = glue
        self.fx = List[Float32]()
        self.syn = List[Float32]()
        self.out = List[Float32]()

    def _split(self) -> Bool:
        return self.host and not self.parallel

    def _refused(self, j: Int, x: Int, n: Int, d: Int) raises:
        """Member `j` refused (`j` is not VOTE_ALL_ANSWERED): its own words."""
        if j != VOTE_ALL_ANSWERED:
            _agn_refusal(self.plan.members[j], x, n, d)

    def _order(self, n: Int) -> List[Int32]:
        """The row numbers 0 .. n-1 (int32) the gathers and scatters place by."""
        var order = List[Int32](unsafe_uninit_length=max(1, n))
        for i in range(n):  # small-loop(the row numbers the glue gather and scatter address, once per chunk)
            order[i] = Int32(i)
        return order^

    def _run(mut self, x: Int, n: Int, dst: Int) raises:
        var glue = self.glue
        var d = self.d
        var k = self.k
        var kind = self.plan.kind
        var rows = VoteRows(x, n, d)
        if kind == AGN_META_SOFT or kind == AGN_META_REGRESS:
            var width = k if kind == AGN_META_SOFT else 1
            # Zero, as the public operation's accumulator was (one block per chunk, as the door's per call).
            var acc = List[Float64](length=max(1, n * width), fill=0.0)
            var at = Int(acc.unsafe_ptr())
            var refused = (vote_soft_run(glue, rows, self.plan.members, k, at) if kind == AGN_META_SOFT
                           else vote_regress_run(glue, rows, self.plan.members, at))
            self._refused(refused, x, n, d)
            agn_narrow(at, dst, n * width, self._split())
            _ = acc^  # the fold is read through the narrowing above
            return
        if kind == AGN_META_HARD:
            var vk = self.plan.classes
            var acc = List[Float64](length=max(1, n * vk), fill=0.0)
            var codes = List[Int32](unsafe_uninit_length=max(1, n))
            var refused = vote_codes_run(glue, rows, self.plan.members, vk, False, Int(acc.unsafe_ptr()),
                                         Int(codes.unsafe_ptr()))
            self._refused(refused, x, n, d)
            # `VotingClassifier.predict` decodes the codes into `classes_`: here their float32 labels.
            agn_labels_run(glue, self.plan.labels, self.plan.n_labels, Int(codes.unsafe_ptr()), n, dst,
                           self._split())
            _ = acc^
            _ = codes^  # read through the label map above
            return
        if kind == AGN_META_BAG:
            self._bag(x, n, dst)
            return
        if kind == AGN_META_STACK:
            self._stack(x, n, dst)
            return
        if kind == AGN_META_OVR:
            self._ovr(x, n, dst)
            return
        if kind == AGN_META_CALIBRATED:
            var classes = self.plan.classes
            var acc = List[Float64](length=max(1, n * classes), fill=0.0)
            var refused = calibrated_proba_run(glue, rows, self.plan.members, self.plan.pairs, self.plan.cals,
                                               classes, Int(acc.unsafe_ptr()))
            self._refused(refused, x, n, d)
            agn_narrow(Int(acc.unsafe_ptr()), dst, n * classes, self._split())
            _ = acc^
            return
        if kind == AGN_META_ADA:
            self._ada(x, n, dst)
            return
        if kind == AGN_META_COLUMNS:
            var acc = List[Float64](length=max(1, n * k), fill=0.0)
            var refused = vote_transform_run(glue, rows, self.plan.members, 1, False, k, Int(acc.unsafe_ptr()))
            self._refused(refused, x, n, d)
            agn_narrow(Int(acc.unsafe_ptr()), dst, n * k, self._split())
            _ = acc^
            return
        if kind == AGN_META_RIDGE:
            # `MultiOutputRegressor.predict` over its fitted multi-output ridge: one call writes the float32
            # (n, outputs) block the public predict returns.
            rm_predict(self.plan.ridge[0], x, self.plan.ridge_coef, self.plan.ridge_icpt, dst, n, d, k)
            return
        raise Error("RuntimeError: agnostic explainer: unknown meta-estimator kind " + String(kind))

    def _bag(mut self, x: Int, n: Int, dst: Int) raises:
        """`_BaggingBase` predict: each member's feature columns of the rows (`_sub_X`: the rows themselves when
        it takes every column, else one glue gather), then `bag_predict_run` (the sums in member order, then
        / m), narrowed."""
        var m = len(self.plan.members)
        var d = self.d
        var classes = self.plan.classes if self.plan.classifier else 1
        var order = self._order(n)
        var keep = List[List[Float32]]()
        var xs = List[Int](capacity=m)
        var ds = List[Int](capacity=m)
        for j in range(m):  # driver-loop: over the member estimators (their feature subsets)
            if self.plan.features[j] == 0:
                xs.append(x)
                ds.append(d)
                continue
            var w = self.plan.widths[j]
            var block = List[Float32](unsafe_uninit_length=max(1, n * w))
            _ = xt_gather_f32(self.glue, x, Int(order.unsafe_ptr()), self.plan.features[j], Int(block.unsafe_ptr()),
                              n, d, n, w)
            xs.append(Int(block.unsafe_ptr()))
            ds.append(w)
            keep.append(block^)
        var acc = List[Float64](length=max(1, n * classes), fill=0.0)
        var refused = bag_predict_run(self.glue, self.plan.members, xs, ds, n, classes, self.plan.classifier,
                                      Int(acc.unsafe_ptr()), 0)
        if refused != VOTE_ALL_ANSWERED:
            _agn_refusal(self.plan.members[refused], xs[refused], n, ds[refused])
        agn_narrow(Int(acc.unsafe_ptr()), dst, n * classes, self._split())
        _ = acc^
        _ = keep^  # the gathered columns are read through the RUN above
        _ = order^

    def _stack(mut self, x: Int, n: Int, dst: Int) raises:
        """`_StackingBase.transform` (the members' outputs placed side by side into the float64 meta features,
        then X itself after them under passthrough), the meta features narrowed to float32
        (`_stacking_meta32`), then the final estimator on them: a classifier's probabilities placed at its class
        columns of a float64 zero block (`_probas_of`, `multioutput_proba_run`) and narrowed; a regressor's
        float32 predict written in place."""
        var d = self.d
        var width = self.plan.meta_width
        var order = self._order(n)
        var meta = List[Float64](length=max(1, n * width), fill=0.0)
        var counts = List[Int](length=len(self.plan.members), fill=0)
        var widths = List[Int]()
        # A classifier member asked for predict: its codes mapped to its float32 labels first (`agn_labels_run`),
        # then placed as an answered block (the public transform places that member's labels).
        var members = self.plan.members.copy()
        var labelled = List[List[Float32]]()
        var rows = VoteRows(x, n, d)
        for j in range(len(self.plan.stack_labels)):  # driver-loop: over members (the label members only)
            if self.plan.stack_labels[j] == 0:
                continue
            var codes = List[Int32](unsafe_uninit_length=max(1, n))
            if not _member_codes_into(members[j], rows, Int(codes.unsafe_ptr())):
                _agn_refusal(members[j], x, n, d)
            var block = List[Float32](unsafe_uninit_length=max(1, n))
            agn_labels_run(self.glue, self.plan.stack_labels[j], self.plan.stack_n_labels[j], Int(codes.unsafe_ptr()),
                           n, Int(block.unsafe_ptr()), self._split())
            _ = codes^  # read through the label map above
            members[j] = VoteMember(Int(block.unsafe_ptr()), 1, 0, 1.0)
            labelled.append(block^)
        var refused = stack_place_run(self.glue, rows, members, self.plan.stack_kinds,
                                      counts, self.plan.classes, self.plan.binary, Int(meta.unsafe_ptr()), n, width,
                                      Int(order.unsafe_ptr()), 0, widths, 0)
        _ = labelled^  # the label blocks are read through the placement above
        self._refused(refused, x, n, d)
        var col0 = 0
        for j in range(len(widths)):  # driver-loop: over members (the columns each took)
            col0 += widths[j]
        if self.plan.passthrough:
            _ = xt_scatter(self.glue, Int(meta.unsafe_ptr()), x, Int(order.unsafe_ptr()), n, width, n, d, col0)
        var xm = List[Float32](unsafe_uninit_length=max(1, n * width))
        agn_narrow(Int(meta.unsafe_ptr()), Int(xm.unsafe_ptr()), n * width, self._split())
        _ = meta^
        var final_est = self.plan.final_est[0]
        if not self.plan.final_tests and not host_all_finite_f32(_F32(unsafe_from_address=Int(xm.unsafe_ptr())),
                                                                 n * width):
            raise Error("ValueError: Input X contains NaN or infinity (the stacked meta features of the explained "
                        + "rows, which the final estimator reads)")
        var meta_rows = VoteRows(Int(xm.unsafe_ptr()), n, width)
        if self.plan.classifier:
            var classes = self.plan.classes
            var acc = List[Float64](length=max(1, n * classes), fill=0.0)
            var finals = List[VoteMember]()
            finals.append(final_est)
            var ks: List[Int] = [classes]
            var accs: List[Int] = [Int(acc.unsafe_ptr())]
            var got = multioutput_proba_run(self.glue, meta_rows, finals, ks, accs, 0)
            if got != VOTE_ALL_ANSWERED:
                _agn_refusal(final_est, meta_rows.x, n, width)
            agn_narrow(Int(acc.unsafe_ptr()), dst, n * classes, self._split())
            _ = acc^
        else:
            var keep = List[List[Float32]]()
            # A refusal raises the final estimator's own words (`member_f32_block`).
            var block = member_f32_block(final_est, meta_rows, keep)
            unsafe_memcpy(dest=_F32(unsafe_from_address=dst), src=_F32(unsafe_from_address=block), count=n)
            _ = keep^  # the final's predictions are read through the copy above
        _ = xm^  # the meta features are read through the final estimator above
        _ = order^

    def _ovr(mut self, x: Int, n: Int, dst: Int) raises:
        """`OneVsRestClassifier.predict_proba`: two classes, the first member's probability column of code 1
        (`_proba_positive`; zeros when it never saw code 1) as the rows (1 - p, p) (`x_trees_binary_proba`); more,
        `ovr_assemble_run` (finish 1: the normalized positive columns). Narrowed."""
        var d = self.d
        var classes = self.plan.classes
        var rows = VoteRows(x, n, d)
        var acc = List[Float64](length=max(1, n * classes), fill=0.0)
        if classes == 2:
            var keep = List[List[Float32]]()
            var first = self.plan.members[0]
            var block = member_f32_block(first, rows, keep)
            var p = List[Float32](length=max(1, n), fill=Float32(0))
            var at = self.plan.at[0]
            if at >= 0:
                var order = self._order(n)
                var col: List[Int32] = [Int32(at)]
                _ = xt_gather_f32(self.glue, block, Int(order.unsafe_ptr()), Int(col.unsafe_ptr()),
                                  Int(p.unsafe_ptr()), n, first.cols, n, 1)
                _ = order^
                _ = col^
            _ = xt_binary_proba(self.glue, Int(p.unsafe_ptr()), Int(acc.unsafe_ptr()), n)
            _ = keep^  # the member's block is read through the gather above
            _ = p^
        else:
            var refused = ovr_assemble_run(self.glue, rows, self.plan.members, self.plan.at, classes, 1,
                                           Int(acc.unsafe_ptr()))
            self._refused(refused, x, n, d)
        agn_narrow(Int(acc.unsafe_ptr()), dst, n * classes, self._split())
        _ = acc^

    def _ada(mut self, x: Int, n: Int, dst: Int) raises:
        """`AdaBoostClassifier.predict_proba` (each member's int32 codes, `ada_classifier_run` with ADA_PROBA:
        two classes into the margin pairs, more in place; narrowed) and `AdaBoostRegressor.predict` (each member's
        float32 predictions, the weighted median written in place)."""
        var d = self.d
        var rows = VoteRows(x, n, d)
        var m = len(self.plan.members)
        if not self.plan.classifier:
            var keep = List[List[Float32]]()
            var blocks = List[Int](capacity=m)
            for j in range(m):  # driver-loop: over the member estimators (every member answers first)
                blocks.append(member_f32_block(self.plan.members[j], rows, keep))
            ada_regressor_run(self.glue, blocks, self.plan.alphas, n, dst)
            _ = keep^  # the members' predictions are read through RUN
            return
        var classes = self.plan.classes
        var codes = List[List[Int32]]()
        var addrs = List[Int](capacity=m)
        for j in range(m):  # driver-loop: over the member estimators (their codes)
            var c = List[Int32](unsafe_uninit_length=max(1, n))
            if not _member_codes_into(self.plan.members[j], rows, Int(c.unsafe_ptr())):
                _agn_refusal(self.plan.members[j], x, n, d)
            addrs.append(Int(c.unsafe_ptr()))
            codes.append(c^)
        var acc = List[Float64](length=max(1, n * classes), fill=0.0)
        if classes == 2:
            var pairs = List[Float64](length=max(1, n * 2), fill=0.0)
            ada_classifier_run(self.glue, addrs, self.plan.alphas, n, classes, Int(acc.unsafe_ptr()), ADA_PROBA,
                               Int(pairs.unsafe_ptr()))
            agn_narrow(Int(pairs.unsafe_ptr()), dst, n * 2, self._split())
            _ = pairs^
        else:
            ada_classifier_run(self.glue, addrs, self.plan.alphas, n, classes, Int(acc.unsafe_ptr()), ADA_PROBA, 0)
            agn_narrow(Int(acc.unsafe_ptr()), dst, n * classes, self._split())
        _ = acc^
        _ = codes^  # the members' codes are read through RUN

    def explained(mut self) raises -> Int:
        if len(self.fx) != self.n * self.k:
            self.fx = List[Float32](unsafe_uninit_length=self.n * self.k)
        var dst = Int(self.fx.unsafe_ptr())
        var x = self.x
        var n = self.n
        self._run(x, n, dst)
        return dst

    def synthetic(mut self, count: Int, filled: Bool) raises -> Int:
        if len(self.syn) != count:
            self.syn = List[Float32]()
            if filled:
                self.syn = List[Float32](length=count, fill=Float32(0))
            else:
                self.syn = List[Float32](unsafe_uninit_length=count)
        return Int(self.syn.unsafe_ptr())

    def rows(mut self, n: Int) raises -> Int:
        if n * self.d > len(self.syn):
            raise Error("RuntimeError: agnostic explainer: synthetic rows exceed their buffer")
        if len(self.out) < n * self.k:
            self.out = List[Float32]()
            self.out = List[Float32](unsafe_uninit_length=n * self.k)
        var dst = Int(self.out.unsafe_ptr())
        var src = Int(self.syn.unsafe_ptr())
        self._run(src, n, dst)
        return dst

    def workers(self) -> Int:
        return host_predict_workers() if self.parallel else 1

    def worker(self) raises -> Self:
        return Self(self.plan, self.x, self.n, self.d, self.parallel, self.host, self.glue)


struct AgnRequest(Copyable, Movable):
    """One shap_values call, typed. `x` float32 n x d (the explained rows), `bg` float32 nb x d (the background),
    `phi` float64 n x d x k (the caller's output, every cell written). `switches`: the glue image's
    x_trees_fast_switches; `fast`: the FAST tier; `host`: the glue image is the CPU column. `forest`: the five
    flat-forest array addresses of MOJOLEARN_IDN_SHAP_DEVICE_MODEL (empty: not that route) with
    `forest_words` = [trees, nodes, k, d, nb, rf_input]."""
    var x: Int
    var bg: Int
    var phi: Int
    var n: Int
    var nb: Int
    var d: Int
    var k: Int
    var seed: Int
    var switches: Int
    var fast: Bool
    var host: Bool
    var forest: List[Int]
    var forest_words: List[Int]

    def __init__(out self, x: Int, bg: Int, phi: Int, n: Int, nb: Int, d: Int, k: Int, seed: Int, switches: Int,
                 fast: Bool, host: Bool):
        self.x = x
        self.bg = bg
        self.phi = phi
        self.n = n
        self.nb = nb
        self.d = d
        self.k = k
        self.seed = seed
        self.switches = switches
        self.fast = fast
        self.host = host
        self.forest = List[Int]()
        self.forest_words = List[Int]()

    def device_forest(self) -> Bool:
        return len(self.forest) == 5


def agn_expected_run(glue: EntryRef, y: Int, nb: Int, k: Int, logit: Bool, ev: Int) raises:
    """The explainer's expected value: ev (float64 k) = the background mean of the model outputs y (float32
    nb x k), then (KernelExplainer, link 'logit') the link: shap's `expected_value = link(fnull)`."""
    _ = xt_block_mean(glue, y, ev, 1, nb, k)
    if logit:
        _ = xt_logit(glue, ev, k)


struct _KshapSchedule(Movable):
    """`x_trees_kshap_schedule` for M features and the resolved sample count, and its words."""
    var size_off: List[Int64]
    var size_w: List[Float64]
    var cdf: List[Float64]
    var m: Int
    var nfixed: Int
    var nfull: Int
    var npaired: Int
    var L: Int
    var wbits: Int

    def __init__(out self, glue: EntryRef, M: Int, ns: Int) raises:
        var h = max(1, M // 2)
        self.size_off = List[Int64](length=h + 1, fill=Int64(0))
        self.size_w = List[Float64](length=h, fill=0.0)
        self.cdf = List[Float64](length=h, fill=0.0)
        var sched = List[Int64](length=6, fill=Int64(0))
        if M > 1:
            _ = xt_kshap_schedule(glue, Int(self.size_off.unsafe_ptr()), Int(self.size_w.unsafe_ptr()),
                                  Int(self.cdf.unsafe_ptr()), Int(sched.unsafe_ptr()), M, max(ns, 0))
        self.m = Int(sched[0])
        self.nfixed = Int(sched[1])
        self.nfull = Int(sched[2])
        self.npaired = Int(sched[3])
        self.L = Int(sched[4])
        self.wbits = Int(sched[5])

    def so(self) -> Int:
        return Int(self.size_off.unsafe_ptr())

    def sw(self) -> Int:
        return Int(self.size_w.unsafe_ptr())

    def cd(self) -> Int:
        return Int(self.cdf.unsafe_ptr())

    def synth_words(self, rows: Int, nb: Int, d: Int, r0: Int, seed: Int) -> List[Int]:
        var p: List[Int] = [rows, nb, d, self.nfixed, self.m, self.nfull, self.L, self.npaired, r0, seed, self.wbits]
        return p^

    def solve_words(self, rows: Int, nb: Int, d: Int, r0: Int, seed: Int, k: Int, link: Int) -> List[Int]:
        var p: List[Int] = [rows, nb, d, self.nfixed, self.m, self.nfull, self.L, self.npaired, r0, seed, self.wbits,
                            k, link]
        return p^


def _join(failed: List[Int], messages: List[String]) raises:
    for w in range(len(failed)):  # driver-loop: over the workers, after the join
        if failed[w] != 0:
            raise Error(messages[w])


def _plan_workers(n: Int, per_row: Int, available: Int) -> Tuple[Int, Int]:
    """(workers, rows per task) for a parallel chunk loop: at most `available` workers, never more than one chunk
    budget of synthetic words in flight across them (each task holds rows * per_row words), at least one row per
    task. A worker count moves no bit."""
    var w = min(available, max(1, AGN_BUDGET // max(1, per_row)))
    w = max(1, min(w, n))
    var rows = max(1, min((n + w - 1) // w, AGN_BUDGET // max(1, w * per_row)))
    var tasks = (n + rows - 1) // rows
    return (max(1, min(w, tasks)), rows)


def agn_kernel_values_run[M: AgnModel](glue: EntryRef, req: AgnRequest, ns: Int, fnull: Int, logit: Bool,
                                       mut model: M) raises:
    """KernelExplainer values of `req.n > 0` rows into req.phi. `ns`: the requested sample count ('auto' already
    resolved to 2 M + 2048), capped at 2^M - 2 for M <= 30 as shap does; `fnull` float64 k (linked); `logit`: the
    link meets every background mean and the model on the rows before the solve."""
    var n = req.n
    var d = req.d
    var k = req.k
    var nb = req.nb
    var Mf = d
    var nsc = ns
    if Mf <= 30:
        nsc = min(nsc, (1 << Mf) - 2)
    var sc = _KshapSchedule(glue, Mf, nsc)
    var m = sc.m
    var link = 1 if logit else 0
    var fx = model.explained()   # the explained rows, one model call (the device forest route needs only this)
    var R = _chunk(m * nb * d, n)
    var bits = req.switches
    if m > 0 and req.fast and (bits & AGN_SW_KSHAP_FAST_BATCH) != 0 and not req.device_forest():
        # MOJOLEARN_KSHAP_FAST_BATCH: one host buffer for every chunk's synthetic rows, each chunk's linked
        # background means into one float64 (n, m, k) block, then the solve over as many rows per call as the
        # budget holds (the same words as the per-chunk solve).
        var ey = List[Float64](length=n * m * k, fill=0.0)
        var e0 = Int(ey.unsafe_ptr())
        var r0 = 0
        while r0 < n:  # driver-loop: one glue call per row chunk
            var rows = min(R, n - r0)
            var syn = model.synthetic(rows * m * nb * d, True)
            _ = xt_kshap_synth(glue, req.x + 4 * r0 * d, req.bg, sc.so(), sc.sw(), sc.cd(), syn,
                               sc.synth_words(rows, nb, d, r0, req.seed))
            var out = model.rows(rows * m * nb)
            var means: List[Int] = [rows, nb, m, k, link]
            _ = xt_kshap_means(glue, out, e0 + 8 * r0 * m * k, means)
            r0 += rows
        var S = _chunk(m * d, n)
        var s0 = 0
        while s0 < n:  # driver-loop: solve-batch loop (one glue call per batch)
            var rows = min(S, n - s0)
            _ = xt_kshap_solve_ey(glue, e0 + 8 * s0 * m * k, fx + 4 * s0 * k, fnull, sc.so(), sc.sw(), sc.cd(),
                                  req.phi + 8 * s0 * d * k, sc.solve_words(rows, nb, d, s0, req.seed, k, link))
            s0 += rows
        _ = ey^
    elif req.device_forest():
        # The forest on the device: masks -> model -> means -> solve per chunk, nothing crossing but the
        # chunk's rows and phi.
        _ = xt_agn_model_load(glue, req.forest, req.bg, req.forest_words)
        try:
            var r0 = 0
            while r0 < n:  # driver-loop: one glue call per row chunk
                var rows = min(R, n - r0)
                _ = xt_kshap_solve_model(glue, req.x + 4 * r0 * d, fx + 4 * r0 * k, fnull, sc.so(), sc.sw(),
                                         sc.cd(), req.phi + 8 * r0 * d * k,
                                         sc.solve_words(rows, nb, d, r0, req.seed, k, link))
                r0 += rows
        finally:
            _ = xt_agn_model_release(glue)
    elif m > 0 and req.host and model.workers() > 1 and n > 1:
        # The CPU column, a member safe on parallel workers: the chunks across the allocation's cores.
        var per_row = m * nb * d
        var plan = _plan_workers(n, per_row, model.workers())
        var W = plan[0]
        var Rp = plan[1]
        var tasks = (n + Rp - 1) // Rp
        var failed = List[Int](length=W, fill=0)
        var messages = List[String](length=W, fill=String(""))
        var fp = failed.unsafe_ptr()
        var mp = messages.unsafe_ptr()
        var x0 = req.x
        var bg0 = req.bg
        var p0 = req.phi
        var seed = req.seed
        var so = sc.so()
        var sw = sc.sw()
        var cd = sc.cd()
        var nfixed = sc.nfixed
        var nfull = sc.nfull
        var L = sc.L
        var npaired = sc.npaired
        var wbits = sc.wbits

        def task(w: Int) {imm fp, imm mp, imm model, imm glue, imm x0, imm bg0, imm p0, imm seed, imm so, imm sw,
                          imm cd, imm nfixed, imm nfull, imm L, imm npaired, imm wbits, imm fx, imm fnull, imm n,
                          imm d, imm k, imm nb, imm m, imm link, imm Rp, imm tasks, imm W}:
            try:
                var mine = model.worker()
                var t = w
                while t < tasks:  # driver-loop: this worker's chunks (round robin)
                    var r0 = t * Rp
                    var rows = min(Rp, n - r0)
                    var syn = mine.synthetic(rows * m * nb * d, False)
                    var sp: List[Int] = [rows, nb, d, nfixed, m, nfull, L, npaired, r0, seed, wbits]
                    _ = xt_kshap_synth(glue, x0 + 4 * r0 * d, bg0, so, sw, cd, syn, sp)
                    var out = mine.rows(rows * m * nb)
                    var vp: List[Int] = [rows, nb, d, nfixed, m, nfull, L, npaired, r0, seed, wbits, k, link]
                    _ = xt_kshap_solve(glue, out, fx + 4 * r0 * k, fnull, so, sw, cd, p0 + 8 * r0 * d * k, vp)
                    t += W
                _ = mine^
            except problem:
                mp[unsafe_offset=w] = String(problem)
                fp.unsafe_store(w, 1)

        host_parallelize(task, W)
        _join(failed, messages)
        _ = failed^
        _ = messages^
    else:
        var reuse = (bits & AGN_SW_IDN_SYN_POOL) != 0   # the pooled device synthetic buffer (GPU glue)
        try:
            var r0 = 0
            while r0 < n:  # driver-loop: one glue call per row chunk
                var rows = min(R, n - r0)
                var y_addr = 0
                if m > 0:
                    # Unfilled (fb-infer/C, 2026-11-17): x_trees_kshap_synth writes every cell.
                    var syn = model.synthetic(rows * m * nb * d, False)
                    _ = xt_kshap_synth(glue, req.x + 4 * r0 * d, req.bg, sc.so(), sc.sw(), sc.cd(), syn,
                                       sc.synth_words(rows, nb, d, r0, req.seed))
                    y_addr = model.rows(rows * m * nb)
                _ = xt_kshap_solve(glue, y_addr, fx + 4 * r0 * k, fnull, sc.so(), sc.sw(), sc.cd(),
                                   req.phi + 8 * r0 * d * k, sc.solve_words(rows, nb, d, r0, req.seed, k, link))
                r0 += rows
        finally:
            if reuse:
                _ = xt_agn_pool_release(glue)   # the pooled device buffer does not outlive the explanation
    _ = sc^


def agn_permutation_values_run[M: AgnModel](glue: EntryRef, req: AgnRequest, npm: Int, linear_wb: Int,
                                            linear_link: Int, mut model: M) raises:
    """PermutationExplainer values of `req.n > 0` rows into req.phi with `npm >= 1` permutations per row.
    `linear_wb` (nonzero): MOJOLEARN_IDN_PSHAP_RESIDENT, the [w | b] float32 block of this library's linear
    predict `link(X W^T + b)` with link word `linear_link`: synthetic rows, the linear model, means and marginals
    on the device; the evaluator is not called."""
    var n = req.n
    var d = req.d
    var k = req.k
    var nb = req.nb
    var mm = npm * (2 * d + 1)
    var R = _chunk(mm * nb * d, n)
    if req.device_forest():
        # The forest on the device: permutations -> model -> means -> marginals per chunk.
        _ = xt_agn_model_load(glue, req.forest, req.bg, req.forest_words)
        try:
            var r0 = 0
            while r0 < n:  # driver-loop: one glue call per row chunk
                var rows = min(R, n - r0)
                var p: List[Int] = [rows, nb, d, npm, r0, req.seed, k]
                _ = xt_pshap_values_model(glue, req.x + 4 * r0 * d, req.phi + 8 * r0 * d * k, p)
                r0 += rows
        finally:
            _ = xt_agn_model_release(glue)
        return
    if linear_wb != 0:
        var Rl = max(1, min(n, AGN_BUDGET_RESIDENT // max(1, mm * nb * d)))
        var r0 = 0
        while r0 < n:  # driver-loop: one glue call per row chunk
            var rows = min(Rl, n - r0)
            var p: List[Int] = [rows, nb, d, npm, r0, req.seed, k, linear_link]
            _ = xt_pshap_values_linear(glue, req.x + 4 * r0 * d, req.bg, linear_wb, req.phi + 8 * r0 * d * k, p)
            r0 += rows
        return
    var bits = req.switches
    var idn_pool = (bits & AGN_SW_IDN_SYN_POOL) != 0
    if req.fast and (bits & AGN_SW_PSHAP_DELTA) != 0:
        # MOJOLEARN_PSHAP_DELTA: per chunk, the device builds only the synthetic rows that differ from their
        # previous coalition's into the front of one reused buffer (their count into `tot`), the model runs on
        # that prefix, and the device expands the outputs back to every (coalition, background row). Three Int64
        # words per (coalition, background row): bound them as if d were at least 16.
        var Rd = min(R, _chunk(mm * nb * max(d, 16), n))
        var tot = List[Int64](length=1, fill=Int64(0))
        var t0 = Int(tot.unsafe_ptr())
        var syn = model.synthetic(Rd * mm * nb * d, True)
        var r0 = 0
        while r0 < n:  # driver-loop: one model call per row chunk
            var rows = min(Rd, n - r0)
            var sp: List[Int] = [rows, nb, d, npm, r0, req.seed]
            _ = xt_pshap_dsynth(glue, req.x + 4 * r0 * d, req.bg, syn, t0, sp)
            var nv = Int(_I64(unsafe_from_address=t0)[0])
            if nv < 0 or nv > rows * mm * nb:
                raise Error("RuntimeError: x_trees_pshap_dsynth returned an invalid row count")
            # The model on the first nv rows of the buffer (read in place).
            var out = model.rows(nv)
            var vp: List[Int] = [rows, nb, d, npm, r0, req.seed, k]
            _ = xt_pshap_dvalues(glue, req.x + 4 * r0 * d, req.bg, out, req.phi + 8 * r0 * d * k, t0, vp)
            r0 += rows
        _ = tot^
        return
    if req.host and model.workers() > 1 and n > 1:
        # The CPU column, a member safe on parallel workers: the chunks across the allocation's cores.
        var per_row = mm * nb * d
        var plan = _plan_workers(n, per_row, model.workers())
        var W = plan[0]
        var Rp = plan[1]
        var tasks = (n + Rp - 1) // Rp
        var failed = List[Int](length=W, fill=0)
        var messages = List[String](length=W, fill=String(""))
        var fp = failed.unsafe_ptr()
        var mp = messages.unsafe_ptr()
        var x0 = req.x
        var bg0 = req.bg
        var p0 = req.phi
        var seed = req.seed

        def task(w: Int) {imm fp, imm mp, imm model, imm glue, imm x0, imm bg0, imm p0, imm seed, imm n, imm d,
                          imm k, imm nb, imm npm, imm mm, imm Rp, imm tasks, imm W}:
            try:
                var mine = model.worker()
                var t = w
                while t < tasks:  # driver-loop: this worker's chunks (round robin)
                    var r0 = t * Rp
                    var rows = min(Rp, n - r0)
                    var syn = mine.synthetic(rows * mm * nb * d, True)
                    var sp: List[Int] = [rows, nb, d, npm, r0, seed]
                    _ = xt_pshap_synth(glue, x0 + 4 * r0 * d, bg0, syn, sp)
                    var out = mine.rows(rows * mm * nb)
                    var vp: List[Int] = [rows, nb, d, npm, r0, seed, k]
                    _ = xt_pshap_values(glue, out, p0 + 8 * r0 * d * k, vp)
                    t += W
                _ = mine^
            except problem:
                mp[unsafe_offset=w] = String(problem)
                fp.unsafe_store(w, 1)

        host_parallelize(task, W)
        _join(failed, messages)
        _ = failed^
        _ = messages^
        return
    try:
        var r0 = 0
        while r0 < n:  # driver-loop: one model call per row chunk
            var rows = min(R, n - r0)
            var syn = model.synthetic(rows * mm * nb * d, True)
            var sp: List[Int] = [rows, nb, d, npm, r0, req.seed]
            _ = xt_pshap_synth(glue, req.x + 4 * r0 * d, req.bg, syn, sp)
            var out = model.rows(rows * mm * nb)
            var vp: List[Int] = [rows, nb, d, npm, r0, req.seed, k]
            _ = xt_pshap_values(glue, out, req.phi + 8 * r0 * d * k, vp)
            r0 += rows
    finally:
        if idn_pool:
            _ = xt_agn_pool_release(glue)   # the pooled device buffer does not outlive the explanation
