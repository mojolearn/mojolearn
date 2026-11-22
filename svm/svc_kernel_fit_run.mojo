# SPDX-License-Identifier: Apache-2.0
"""RUN of the kernel SVC's fit: the gamma, the per-row bounds, the binary
machine or the one-vs-one machines with scikit-learn's multiclass layout,
and libsvm's Platt cross-validation, as typed drivers over the SVM binding's
entry tables (core/svc_native_api.mojo SvcDrivers). No Python: the caller
(bindings/svc_kernel_native_door.mojo) releases the GIL around each run and
holds the owners of every address in the plan. Every launch is the one the
door always made, with the same operands in the same order.

The pair loop and the fold loop are serial by nature: each step launches on
the one device and the next reads what it wrote (a pair's rows, a fold's
held rows, the fold decisions). The independent units inside a launch (rows,
kernel cells) are parallel in the binding's owners.

Scratch is Mojo-owned and reused across the solves of one pair (the entries
read and write only the prefixes they report), so the worst-case buffers are
allocated once per pair, not once per solve and fold."""
from std.memory import unsafe_memcpy
from core.svc_native_api import SvcDrivers, SvcFitRequest, SvcQueryRequest
from core.scale_gamma_exact import scale_gamma_from_limbs
from svm.operation_metadata import SvcOperationSettings
from svm.svc_kernel_native_run import SVC_KERNEL_PRECOMPUTED, SVC_GLUE_GATHER

# `svc_pair_epilogue`'s row-glue modes (svm/impl/svc_rows.mojo GLUE_*).
comptime SVC_GLUE_SELECT = 7
comptime SVC_GLUE_C_ROWS = 8
comptime SVC_GLUE_FOLD = 9
comptime SVC_GLUE_FOLD_FINISH = 10
# `_scale_gamma`: the limb words.
comptime SVC_SCALE_GAMMA_SLOTS = 28
comptime SVC_GAMMA_VALUE = 0
comptime SVC_GAMMA_AUTO = 1
comptime SVC_GAMMA_SCALE = 2
comptime SVC_PLATT_FOLDS = 5


def _f32(values: List[Float32]) -> Int:
    return Int(values.unsafe_ptr()) if len(values) else 0


def _i32(values: List[Int32]) -> Int:
    return Int(values.unsafe_ptr()) if len(values) else 0


def _f64(values: List[Float64]) -> Int:
    return Int(values.unsafe_ptr()) if len(values) else 0


def _round_f32(value: Float64) -> Float64:
    return Float64(Float32(value))


def svc_gamma_run(drivers: SvcDrivers, kind: Int, value: Float64, x: Int, cells: Int, cols: Int) raises -> Float64:
    """`SVC._resolve_gamma` on host words: 'auto' 1 / n_cols; 'scale' the
    exact variance's once-rounded reciprocal (the binding's limbs, then
    core/scale_gamma_exact.mojo); else the float."""
    if kind == SVC_GAMMA_AUTO:
        return 1.0 / Float64(cols)
    if kind == SVC_GAMMA_SCALE:
        var limbs = List[Int64](length=SVC_SCALE_GAMMA_SLOTS, fill=Int64(0))
        _ = drivers.scale_gamma_limbs(x, cells, Int(limbs.unsafe_ptr()))
        return scale_gamma_from_limbs(limbs, cells, cols)
    return value


def svc_bounds_run(drivers: SvcDrivers, rows: Int, sample_weights: Int, codes: Int, class_weights: Int,
                   classes: Int, c: Float64, dst: Int) raises:
    """`_svm_impl._c_rows`: the per-row bounds C * class_weight[y_i] *
    sample_weight_i as float32 at `dst` (an absent factor, address 0, is 1)."""
    if rows > 0:
        _ = drivers.epilogue(SVC_GLUE_C_ROWS, sample_weights, codes if class_weights != 0 else 0, class_weights, dst,
                             rows, classes if class_weights != 0 else 0, 0, 0, c)


def _gather(drivers: SvcDrivers, x: Int, rows: Int, cols: Int, row_index: Int, out_rows: Int, col_index: Int,
            out_cols: Int, dst: Int) raises:
    """`x[rows][:, cols]` (an absent index list, address 0, keeps that axis
    whole), an exact copy, gathered in the binding."""
    if out_rows > 0 and out_cols > 0:
        _ = drivers.epilogue(SVC_GLUE_GATHER, x, row_index, col_index, dst, rows, cols, out_rows, out_cols)


struct SvcSolveScratch(Movable):
    """The worst-case buffers of one binary solve (`rows` x `cols`): duals,
    support indices, support rows and the info words."""
    var dual: List[Float32]
    var support: List[Int32]
    var vectors: List[Float32]
    var info: List[Float64]

    def __init__(out self, rows: Int, cols: Int):
        self.dual = List[Float32](length=max(rows, 1), fill=Float32(0))
        self.support = List[Int32](length=max(rows, 1), fill=Int32(0))
        self.vectors = List[Float32](length=max(rows * cols, 1), fill=Float32(0))
        self.info = List[Float64](length=8, fill=0.0)


def _solve(drivers: SvcDrivers, settings: SvcOperationSettings, x: Int, rows: Int, cols: Int, labels: Int,
           bounds: Int, scratch: SvcSolveScratch) raises -> Int:
    """ONE binary C-SVC solve in the short form (host words, a flagged X
    refused by the entry). Returns n_support; the model is in `scratch`."""
    return drivers.fit(SvcFitRequest(x, labels, _f32(scratch.dual), _i32(scratch.support), _f32(scratch.vectors),
        _f64(scratch.info), bounds, rows, cols, settings.kernel, settings.max_iter, settings.nochange_steps,
        settings.degree, -1, 0, -1, False, False, settings.gamma, settings.c, settings.tol, settings.coef0))


struct SvcPairFit(Copyable, Movable):
    """One fitted pair machine (class i against class j, j the solver's +1):
    its duals, its support rows (of the pair's rows; of the WHOLE kernel
    matrix for kernel='precomputed') and its support in X's row numbers."""
    var i: Int
    var j: Int
    var supports: Int
    var vector_cols: Int
    var dual: List[Float32]
    var vectors: List[Float32]
    var support: List[Int32]
    var bias: Float64
    var n_iter: Int

    def __init__(out self, i: Int, j: Int, supports: Int, vector_cols: Int, var dual: List[Float32],
                 var vectors: List[Float32], var support: List[Int32], bias: Float64, n_iter: Int):
        self.i = i
        self.j = j
        self.supports = supports
        self.vector_cols = vector_cols
        self.dual = dual^
        self.vectors = vectors^
        self.support = support^
        self.bias = bias
        self.n_iter = n_iter


struct SvcFitPlan(Movable):
    """One `SVC.fit`. X is the float32 host words at `x` (`rows` x `cols`),
    or, in the device form (`device_form`: one binary machine on a computed
    kernel), the host words or the fold (`base_id`, `base_rows`,
    `index_id`) the entry stages on the device, with `scale_gamma` resolved
    there. `labels`: the binary machine's float32 labels; `codes`: the int32
    class code of every row (the pairs and Platt read them). The bounds'
    operands are 0 when that factor is absent (`weighted`: any present).
    `seed`: the key of probability's fold shuffle."""
    var settings: SvcOperationSettings
    var x: Int
    var rows: Int
    var cols: Int
    var cells: Int
    var base_id: Int
    var base_rows: Int
    var index_id: Int
    var device_form: Bool
    var scale_gamma: Bool
    var gamma_kind: Int
    var gamma: Float64
    var classes: Int
    var labels: Int
    var codes: Int
    var weighted: Bool
    var sample_weights: Int
    var class_weights: Int
    var probability: Bool
    var seed: UInt64

    def __init__(out self, settings: SvcOperationSettings):
        self.settings = settings.copy()
        self.x = 0
        self.rows = 0
        self.cols = 0
        self.cells = 0
        self.base_id = -1
        self.base_rows = 0
        self.index_id = -1
        self.device_form = False
        self.scale_gamma = False
        self.gamma_kind = SVC_GAMMA_VALUE
        self.gamma = 1.0
        self.classes = 0
        self.labels = 0
        self.codes = 0
        self.weighted = False
        self.sample_weights = 0
        self.class_weights = 0
        self.probability = False
        self.seed = UInt64(0)


struct SvcFitResult(Movable):
    """What the fit produced, Mojo-owned until PACK. `status`: SVC_FIT_OK,
    or SVC_FIT_PAIR_LABELS (the solver's label pair of classes
    (`bad_i`, `bad_j`) was not (0.0, 1.0)). Binary: `solved` and its count.
    One-vs-one: `pairs`, then the layout (`support`, `n_support`,
    `dual_all`, `total` support vectors, `intercept` = -b per pair,
    `n_iter` per pair). Probability: `prob_a`, `prob_b` per pair."""
    var status: Int
    var bad_i: Int
    var bad_j: Int
    var gamma: Float64
    var bounds: List[Float32]
    var solved: SvcSolveScratch
    var supports: Int
    var pairs: List[SvcPairFit]
    var support: List[Int32]
    var n_support: List[Int32]
    var dual_all: List[Float32]
    var total: Int
    var intercept: List[Float32]
    var n_iter: List[Int32]
    var prob_a: List[Float64]
    var prob_b: List[Float64]

    def __init__(out self):
        self.status = SVC_FIT_OK
        self.bad_i = 0
        self.bad_j = 0
        self.gamma = 1.0
        self.bounds = List[Float32]()
        self.solved = SvcSolveScratch(0, 0)
        self.supports = 0
        self.pairs = List[SvcPairFit]()
        self.support = List[Int32]()
        self.n_support = List[Int32]()
        self.dual_all = List[Float32]()
        self.total = 0
        self.intercept = List[Float32]()
        self.n_iter = List[Int32]()
        self.prob_a = List[Float64]()
        self.prob_b = List[Float64]()


comptime SVC_FIT_OK = 0
comptime SVC_FIT_PAIR_LABELS = 1


def _fit_ovo(drivers: SvcDrivers, plan: SvcFitPlan, settings: SvcOperationSettings, bounds: Int,
             mut result: SvcFitResult) raises:
    """`SVC._fit_ovo` and `_set_ovo_native`: one binary machine per class
    pair (i < j, class j the solver's +1), then scikit-learn's multiclass
    layout from `svc_ovo_layout`."""
    var n = plan.rows
    var k = plan.classes
    var precomputed = settings.kernel == SVC_KERNEL_PRECOMPUTED
    var n_pairs = k * (k - 1) // 2
    result.intercept = List[Float32](length=max(n_pairs, 1), fill=Float32(0))
    result.n_iter = List[Int32](length=max(n_pairs, 1), fill=Int32(0))
    # The pair's row indices, its 0/1 labels and its bounds: their first m entries are the pair's.
    var index = List[Int32](length=max(n, 1), fill=Int32(0))
    var pair_labels = List[Float32](length=max(n, 1), fill=Float32(0))
    var pair_bounds = List[Float32](length=max(n, 1) if bounds != 0 else 0, fill=Float32(0))
    var total = 0
    var q_no = 0
    for i in range(k):  # driver-loop: over the class pairs (one binary machine each)
        for j in range(i + 1, k):  # driver-loop: over the class pairs (one binary machine each)
            var m = 0
            if n > 0:
                m = drivers.epilogue(SVC_GLUE_SELECT, plan.codes, bounds, _f32(pair_labels), _i32(index), n, i, j,
                                     _f32(pair_bounds))
            # the pair's kernel matrix (its rows AND its columns), or its rows of X
            var sub_cols = m if precomputed else plan.cols
            var sub = List[Float32](length=max(m * sub_cols, 1), fill=Float32(0))
            _gather(drivers, plan.x, n, plan.cols, _i32(index), m, _i32(index) if precomputed else 0, sub_cols, _f32(sub))
            var scratch = SvcSolveScratch(m, sub_cols)
            var n_sv = _solve(drivers, settings, _f32(sub), m, sub_cols, _f32(pair_labels), _f32(pair_bounds), scratch)
            _ = sub^  # keep alive: the call above reads it through an untracked pointer
            if _round_f32(scratch.info[3]) != 0.0 or _round_f32(scratch.info[4]) != 1.0:
                result.status = SVC_FIT_PAIR_LABELS
                result.bad_i = i
                result.bad_j = j
                return
            # the pair's support in X's row numbers, gathered in the binding
            var pair_support = List[Int32](length=n_sv, fill=Int32(0))
            _gather(drivers, _i32(index), m, 1, _i32(scratch.support), n_sv, 0, 1, _i32(pair_support))
            var vector_cols = plan.cols if precomputed else sub_cols
            var vectors = List[Float32](length=n_sv * vector_cols, fill=Float32(0))
            if precomputed:
                # the support rows of the WHOLE kernel matrix
                _gather(drivers, plan.x, n, plan.cols, _i32(pair_support), n_sv, 0, plan.cols, _f32(vectors))
            elif n_sv * vector_cols > 0:
                unsafe_memcpy(dest=vectors.unsafe_ptr(), src=scratch.vectors.unsafe_ptr(), count=n_sv * vector_cols)
            var dual = List[Float32](length=n_sv, fill=Float32(0))
            if n_sv > 0:
                unsafe_memcpy(dest=dual.unsafe_ptr(), src=scratch.dual.unsafe_ptr(), count=n_sv)
            var bias = _round_f32(scratch.info[0])
            var iters = Int(scratch.info[2])
            result.intercept[q_no] = Float32(-bias)
            result.n_iter[q_no] = Int32(iters)
            result.pairs.append(SvcPairFit(i, j, n_sv, vector_cols, dual^, vectors^, pair_support^, bias, iters))
            total += n_sv
            q_no += 1
    # `_set_ovo_native`: the support_ order, dual_coef_ cells and n_support_
    var cap = min(n, total)
    result.support = List[Int32](length=max(1, cap), fill=Int32(0))
    result.n_support = List[Int32](length=k, fill=Int32(0))
    result.dual_all = List[Float32](length=max(1, (k - 1) * cap), fill=Float32(0))
    var addresses: List[Int] = [_i32(result.support), _i32(result.n_support), _f32(result.dual_all)]
    var words: List[Int] = [k, n, n_pairs, cap]
    for q in range(len(result.pairs)):  # driver-loop: over the class pairs (their layout descriptors)
        addresses.append(_i32(result.pairs[q].support) if result.pairs[q].supports > 0 else 0)
        addresses.append(_f32(result.pairs[q].dual) if result.pairs[q].supports > 0 else 0)
        words.append(result.pairs[q].i)
        words.append(result.pairs[q].j)
        words.append(result.pairs[q].supports)
    result.total = drivers.ovo_layout(addresses, words)


def _platt(drivers: SvcDrivers, plan: SvcFitPlan, settings: SvcOperationSettings, bounds: Int,
           mut result: SvcFitResult) raises:
    """`SVC._fit_probability`: libsvm's `svm_binary_svc_probability` per
    class pair, five shuffled folds, then `sigmoid_train`: (A, B) per pair."""
    var k = plan.classes
    var n_all = plan.rows
    var precomputed = settings.kernel == SVC_KERNEL_PRECOMPUTED
    var n_pairs = k * (k - 1) // 2
    result.prob_a = List[Float64](length=n_pairs, fill=0.0)
    result.prob_b = List[Float64](length=n_pairs, fill=0.0)
    var index = List[Int32](length=max(n_all, 1), fill=Int32(0))
    var pair_labels = List[Float32](length=max(n_all, 1), fill=Float32(0))
    # SplitMix64's golden-ratio increment: the key of pair q's fold shuffle
    # is seed XOR (q * golden mod 2^64).
    var golden: UInt64 = 0x9E3779B97F4A7C15
    var pair_no = 0
    for ci in range(k):  # driver-loop: over the class pairs (one probability fit each)
        for cj in range(ci + 1, k):  # driver-loop: over the class pairs (one probability fit each)
            # the pair's rows in row order; libsvm's +1 is class ci
            var n = 0
            if n_all > 0:
                n = drivers.epilogue(SVC_GLUE_SELECT, plan.codes, 0, _f32(pair_labels), _i32(index), n_all, ci, cj, 0)
            var key = plan.seed ^ (UInt64(pair_no) * golden)
            var perm = List[Int32](length=max(n, 1), fill=Int32(0))
            if n > 0:
                _ = drivers.splitmix(_i32(perm), n, Int(key & UInt64(0xFFFFFFFF)), Int(key >> UInt64(32)))
            # Zeroed per pair: a fold that is skipped leaves its decisions 0 and its constants 0.
            var dec_perm = List[Float32](length=max(n, 1), fill=Float32(0))
            var consts = List[Float32](length=2 * SVC_PLATT_FOLDS, fill=Float32(0))
            # The pair's scratch, reused by its folds (each fold writes what it reads).
            var width = n if precomputed else plan.cols
            var rows_held = List[Int32](length=max(n, 1), fill=Int32(0))
            var fold_labels = List[Float32](length=max(n, 1), fill=Float32(0))
            var fold_bounds = List[Float32](length=max(n, 1) if bounds != 0 else 0, fill=Float32(0))
            var sub = List[Float32](length=max(n * width, 1), fill=Float32(0))
            var held = List[Float32](length=max(n * width, 1), fill=Float32(0))
            var cross = List[Float32](length=max(n * n, 1) if precomputed else 0, fill=Float32(0))
            var scratch = SvcSolveScratch(n, width)
            for f in range(SVC_PLATT_FOLDS):  # small-loop(Platt folds: five per pair)
                var begin = f * n // SVC_PLATT_FOLDS
                var end = (f + 1) * n // SVC_PLATT_FOLDS
                var h = end - begin
                var nt = n - h
                var n_pos = 0
                if n > 0:
                    n_pos = drivers.epilogue(SVC_GLUE_FOLD, _i32(index), _i32(perm), _f32(pair_labels), _i32(rows_held),
                                             n, begin, end, _f32(fold_labels))
                var n_neg = nt - n_pos
                if n_pos == 0 or n_neg == 0:
                    consts[2 * f] = Float32(1.0)
                    consts[2 * f + 1] = Float32(0.0) if nt == 0 else (Float32(1.0) if n_pos > 0 else Float32(-1.0))
                    continue
                if h == 0:
                    continue
                var train_rows = _i32(rows_held)
                var held_rows = train_rows + 4 * nt
                var cols = nt if precomputed else plan.cols
                _gather(drivers, plan.x, n_all, plan.cols, train_rows, nt, train_rows if precomputed else 0, cols, _f32(sub))
                if bounds != 0:
                    _gather(drivers, bounds, n_all, 1, train_rows, nt, 0, 1, _f32(fold_bounds))
                var n_sv = _solve(drivers, settings, _f32(sub), nt, cols, _f32(fold_labels), _f32(fold_bounds), scratch)
                _gather(drivers, plan.x, n_all, plan.cols, held_rows, h, train_rows if precomputed else 0, cols, _f32(held))
                var query = _f32(held)
                var features = cols
                var vectors = _f32(scratch.vectors)
                if precomputed and n_sv > 0:
                    # the held rows' cross-kernel columns at the fold's support vectors
                    _gather(drivers, _f32(held), h, nt, 0, h, _i32(scratch.support), n_sv, _f32(cross))
                    query = _f32(cross)
                    features = n_sv
                    vectors = _f32(scratch.dual)
                _ = drivers.query(SvcQueryRequest(query, _f32(scratch.dual) if n_sv > 0 else 0,
                    vectors if n_sv > 0 else 0, _f32(dec_perm) + 4 * begin, 0, h, features, n_sv, settings.kernel,
                    settings.degree, False, 0, 0, -1, 0, -1, False, _round_f32(scratch.info[0]), 0.0, 1.0,
                    settings.gamma, settings.cache_mib, settings.coef0))
            # libsvm's orientation (positive toward class i) and +1/-1
            # labels, back in pair row order, float64
            var decisions = List[Float64](length=max(n, 1), fill=0.0)
            var signs = List[Float64](length=max(n, 1), fill=0.0)
            if n > 0:
                _ = drivers.epilogue(SVC_GLUE_FOLD_FINISH, _f32(dec_perm), _i32(perm), _f32(pair_labels), _f64(decisions),
                                     n, _f32(consts), _f64(signs), 0)
            var ab = List[Float64](length=2, fill=0.0)
            _ = drivers.platt(_f64(decisions) if n > 0 else 0, _f64(signs) if n > 0 else 0, _f64(ab), n)
            _ = decisions^  # keep alive: the call above reads it through an untracked pointer
            _ = signs^  # keep alive: the epilogue / platt calls above read it through an untracked pointer
            _ = consts^  # keep alive: the epilogue / platt calls above read it through an untracked pointer
            _ = dec_perm^  # keep alive: the epilogue / platt calls above read it through an untracked pointer
            _ = perm^  # keep alive: the epilogue / platt calls above read it through an untracked pointer
            result.prob_a[pair_no] = ab[0]
            result.prob_b[pair_no] = ab[1]
            pair_no += 1


def svc_fit_run(drivers: SvcDrivers, plan: SvcFitPlan, mut result: SvcFitResult) raises:
    """Gamma, the bounds, then the binary machine (its model stays in
    `result.solved`) or the one-vs-one machines and their layout. A flagged
    device X and the binary label pair are the caller's to check before
    `svc_fit_finish_run`."""
    var settings = plan.settings.copy()
    result.gamma = plan.gamma
    if not plan.device_form:
        result.gamma = svc_gamma_run(drivers, plan.gamma_kind, plan.gamma, plan.x, plan.cells, plan.cols)
    settings.gamma = result.gamma
    if plan.weighted:
        result.bounds = List[Float32](length=max(plan.rows, 1), fill=Float32(0))
        svc_bounds_run(drivers, plan.rows, plan.sample_weights, plan.codes, plan.class_weights, plan.classes,
                       settings.c, _f32(result.bounds))
    var bounds = _f32(result.bounds) if plan.weighted else 0
    if plan.classes > 2:
        _fit_ovo(drivers, plan, settings, bounds, result)
        return
    result.solved = SvcSolveScratch(plan.rows, plan.cols)
    result.supports = drivers.fit(SvcFitRequest(plan.x, plan.labels, _f32(result.solved.dual),
        _i32(result.solved.support), _f32(result.solved.vectors), _f64(result.solved.info), bounds, plan.rows,
        plan.cols, settings.kernel, settings.max_iter, settings.nochange_steps, settings.degree, plan.base_id,
        plan.base_rows, plan.index_id, plan.scale_gamma, plan.device_form, settings.gamma, settings.c, settings.tol,
        settings.coef0))
    if plan.device_form:
        # the gamma the entry solved with (gamma='scale' from its device copy)
        result.gamma = result.solved.info[7]


def svc_fit_finish_run(drivers: SvcDrivers, plan: SvcFitPlan, support_vectors: Int, mut result: SvcFitResult) raises:
    """After the caller's checks: the one-vs-one `support_vectors_` (the
    rows of X at `result.support`, gathered into the public array at
    `support_vectors`), then the probability fit."""
    if plan.classes > 2 and result.total > 0:
        _gather(drivers, plan.x, plan.rows, plan.cols, _i32(result.support), result.total, 0, plan.cols, support_vectors)
    if plan.probability:
        var settings = plan.settings.copy()
        settings.gamma = result.gamma
        _platt(drivers, plan, settings, _f32(result.bounds) if plan.weighted else 0, result)
