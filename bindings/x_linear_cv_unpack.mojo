# SPDX-License-Identifier: Apache-2.0
# UNPACK of the cross-validated fits (RidgeCV, LassoCV, ElasticNetCV,
# LogisticRegressionCV; python/mojolearn/_expansion_linear.py): the refusals
# in the order and words the Python made them, X and y through the one
# admission, the folds, the checked weights and the typed requests, one
# function per public operation. RUN (x_linear/door_run.mojo `xl_fit_run`,
# `xl_logcv_fit_run`, `xl_ridge_cv_*_run`) holds no Python object; PACK is
# bindings/x_linear_cv_pack.mojo.
#
# The folds. An int (or None) cv takes the same integer fold ids the Python
# assigned (the base binding's `fold_ids` / `select_fold_i64`, or the
# binding's own StratifiedKFold ids); a user splitter object is the
# user-API boundary: its `split(X, y)` is called once, here, as the Python
# did, and its (train, test) index arrays are read natively.
#
# RidgeCV's split path gathers each split's rows here, split by split
# (`xl_ridge_cv_fold_unpack`): the row gather is the base binding's
# Python-callable `gather_rows_bytes` (the device gather on a GPU
# installation), which has no entry in the typed base helper table
# (core/base_entry_table.mojo), so the door drives the splits and RUN runs
# each split's alpha grid (fit, prediction, R^2) in one GIL release.
from std.math import isinf
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.native_descriptor_abi import NativeDescriptorCall
from core.entry_table import entry_ref_unresolved
from core.device_route import device_mode_code, device_vendor_code
from core.metrics_entry_table import MetricsEntryKey, metrics_r2_entry
from x_linear.door_run import (
    ALGO_ENETCV, ALGO_LOGCV, xl_run_parts, xl_parts_total, xl_lbfgs_work, XlSplitWords, xl_cv_fold_ids_run,
    xl_cv_roles_limit, xl_cv_roles_run, XlClassPrepRequest, XlRidgeCvRequest, XlRidgeFoldRequest, xl_ridge_out,
)
from bindings.x_linear_route_unpack import XLinearOps, xl_ops_require
from bindings.x_linear_common_unpack import (
    _b, _truth, _float, _int, _lt, _eq, _isinstance, _is_int, _name, _attr, _xl_scope, _xl_matrix, _xl_vector,
    _xl_with_weights, _xl_owned, XlFit, xl_fit_build, xl_class_weights_unpack,
)
from bindings.x_linear_sgd_unpack import _xl_class_codes
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_item
from bindings.base_helpers_key_unpack import base_helpers_key
from bindings.py_attr import py_attr
from bindings.py_typed_call import py_call_typed
from bindings.native_loaded_module_door import native_loaded_module
from bindings.classical_arrays_door import classical_wrap_array
from bindings.classical_warnings_unpack import classical_warning
from bindings.prep_native_estimator_door import prep_native_context, prep_native_split
from bindings.msel_inputs_unpack import msel_admit
from bindings.admitted_input_unpack import admit_weight_words, admitted_vector, admitted_storage_native, admit_ids
from bindings.py_optional import py_or_none, py_present
from bindings.x_linear_keys_unpack import x_linear_keys
from bindings.classical_labels_door import (
    labels_full_f32, labels_address, labels_cast, labels_real_type, labels_integral_type, labels_table,
)

# The three paths of RidgeCV.fit (`XlRidgeCv.path`).
comptime XL_RIDGE_CV_PLAIN = 0
comptime XL_RIDGE_CV_KFOLD = 1
comptime XL_RIDGE_CV_SPLITS = 2


# ------------------------------------------------------------ cv helpers


def _xl_helper(context: PythonObject, name: String) raises -> PythonObject:
    """A base binding helper (`_buffer._native(name)`) connected at import,
    or the refusal recorded then."""
    var nk = x_linear_keys()
    var entry = py_get(py_item(context, nk[].k_xl_helpers), PythonObject(name))
    if entry is None:
        raise Error("ImportError: mojolearn: the base binding has no `" + name + "`")
    if py_type(entry) is _b().str:
        raise Error(String(py=entry))
    return entry


def _xl_is_int_cv(cv: PythonObject, context: PythonObject) raises -> Bool:
    """`_is_int_cv`: an exact native integer that is not a bool (identity,
    never the numbers.Integral ABC hook)."""
    return labels_integral_type(py_type(cv), context)


def _xl_index_i64(values: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """`as_index_i64(values, name=name)`: row indices as a 1-D int64 Array
    (a list, tuple or range through `array('q', ...)` in C; a buffer through
    the native cast)."""
    var nk = x_linear_keys()
    var b = _b()
    var cls = py_type(values)
    if cls is b.list or cls is b.tuple or cls is b.range:
        var store = py_call(py_item(context, nk[].k_array_type), PythonObject("q"), values)
        return classical_wrap_array(store, Python.tuple(len(store)), "<i8", "C", context)
    # The one integer admission (bindings/admitted_input_unpack.mojo admit_ids):
    # one parallel read of the exporter's strides into fresh int64 words.
    return admit_ids(values, name, context)


def _xl_cv_splits(cv: PythonObject, x_source: PythonObject, y_source: PythonObject,
                  context: PythonObject) raises -> PythonObject:
    """`_cv_splits(cv, X, y)`: a list of (train, test) int64 Arrays from a
    splitter's `split(X, y)` (called once) or an iterable of pairs."""
    var nk = x_linear_keys()
    var b = _b()
    var pairs = PythonObject(None)
    if py_type(py_type(cv)) is b.type and py_contains(py_item(prep_native_context(context), nk[].k_family_msel_splitter_layouts), py_type(cv)):
        pairs = prep_native_split(cv, x_source, y_source, context)
    elif py_type(cv) is b.list or py_type(cv) is b.tuple:
        msel_admit(cv, prep_native_context(context))
        pairs = cv
    else:
        raise Error("TypeError: native linear CV requires a native splitter or list of index pairs; Python callbacks cannot execute")
    var listed = py_call(b.list, pairs)
    var splits = Python.list()
    for k in range(len(listed)):  # driver-loop: over the cv splits
        var pair = py_call(b.tuple, listed[k])
        if len(pair) != 2:
            if len(pair) > 2:
                raise Error("too many values to unpack (expected 2, got " + String(len(pair)) + ")")
            raise Error("not enough values to unpack (expected 2, got " + String(len(pair)) + ")")
        _ = py_call(splits.append, Python.tuple(_xl_index_i64(pair[0], "train", context),
                                                _xl_index_i64(pair[1], "test", context)))
    if len(splits) == 0:
        raise Error("mojolearn: cv produced no (train, test) split")
    return splits


def _xl_default_folds(n: Int, k: Int, context: PythonObject) raises -> PythonObject:
    """`KFold(n_splits=k).split(...)` unshuffled: contiguous folds by the
    core helpers (`fold_ids` with no codes, `select_fold_i64`)."""
    var fold_store = admitted_vector(n, "<i4", context)
    var fold_counts = admitted_vector(k, "<i8", context)
    var class_counts = admitted_vector(1, "<i8", context)
    _ = py_call_typed(_xl_helper(context, "fold_ids"), PythonObject(0), PythonObject(n), PythonObject(0), PythonObject(k),
                      PythonObject(labels_address(class_counts)), PythonObject(labels_address(fold_store)),
                      PythonObject(labels_address(fold_counts)))
    _ = class_counts  # keep the Python object alive through the native call above
    _ = fold_store  # keep the Python object alive through the native call above
    _ = fold_counts  # keep the Python object alive through the native call above
    var select = _xl_helper(context, "select_fold_i64")
    var counts = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=labels_address(fold_counts))
    _ = fold_counts  # keep the Python object alive through the native call above
    var out = Python.list()
    for fold in range(k):  # driver-loop: over the cv folds
        var size = Int(counts[fold])
        _ = fold_counts  # keep the Python object alive through the native call above
        var test = admitted_vector(size, "<i8", context)
        var train = admitted_vector(n - size, "<i8", context)
        var scratch = admitted_vector(n, "<i8", context)
        var got = _int(py_call_typed(select, PythonObject(labels_address(fold_store)), PythonObject(n), PythonObject(fold),
                                     PythonObject(labels_address(test) if size > 0 else labels_address(scratch)),
                                     PythonObject(labels_address(train) if n - size > 0 else labels_address(scratch))))
        _ = fold_store  # keep the Python object alive through the native call above
        _ = test  # keep the Python object alive through the native call above
        _ = scratch  # keep the Python object alive through the native call above
        _ = train  # keep the Python object alive through the native call above
        if got != size:
            raise Error("RuntimeError: mojolearn: select_fold_i64 disagrees with fold_ids")
        _ = scratch  # keep the Python object alive through the native calls above
        _ = train  # keep the Python object alive through the native calls above
        _ = test  # keep the Python object alive through the native calls above
        _ = fold_store  # keep the Python object alive through the native calls above
        _ = py_call(out.append, Python.tuple(train, test))
    return out


def _xl_kfold_ids_f32(n: Int, k: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_kfold_ids_f32(n, k)`: KFold's contiguous fold ids as float32."""
    if not _is_int(k) or _lt(k, PythonObject(2)):
        raise Error("mojolearn: cv must be None or an int in [2, n_samples]")
    var folds = _int(k)
    if folds > n:
        raise Error("Cannot have number of splits n_splits=" + String(folds) + " greater than the "
                    + "number of samples: n_samples=" + String(n) + ".")
    var ids = admitted_vector(n, "<i4", context)
    var counts = admitted_vector(folds, "<i8", context)
    _ = py_call_typed(_xl_helper(context, "fold_ids"), PythonObject(0), PythonObject(n), PythonObject(0),
                      PythonObject(folds), PythonObject(0), PythonObject(labels_address(ids)),
                      PythonObject(labels_address(counts)))
    _ = ids  # keep the Python object alive through the native call above
    _ = counts  # keep the Python object alive through the native call above
    var out = admitted_vector(n, "<f4", context)
    labels_cast(labels_address(ids), "<i4", labels_address(out), "<f4", n, context)
    _ = out  # keep the Python object alive through the native call above
    _ = ids  # keep the Python object alive through the native call above
    return out


def _xl_split_words(splits: PythonObject) raises -> XlSplitWords:
    """UNPACK of the cv splits: each side's int64 index address (0 when
    empty) and count. The caller keeps `splits` alive through RUN."""
    var words = XlSplitWords(List[Int](), List[Int](), List[Int](), List[Int]())
    for k in range(len(splits)):  # driver-loop: over the cv splits
        var tr = splits[k][0]
        var te = splits[k][1]
        var tr_size = Int(py=tr.size)
        var te_size = Int(py=te.size)
        words.train.append(labels_address(tr) if tr_size > 0 else 0)
        words.train_sizes.append(tr_size)
        words.test.append(labels_address(te) if te_size > 0 else 0)
        words.test_sizes.append(te_size)
    return words^


def _xl_full_i32(n: Int, value: Int, context: PythonObject) raises -> PythonObject:
    """`full((n,), value, '<i4')`: the store repeated in C."""
    var nk = x_linear_keys()
    var one = py_call(py_item(context, nk[].k_array_type), PythonObject("i"), Python.list(PythonObject(value)))
    var store = py_call(py_attr(one, "__mul__"), PythonObject(n))
    return classical_wrap_array(store, Python.tuple(n), "<i4", "C", context)


def _xl_cv_fold_ids_f32(splits: PythonObject, n: Int, name: String,
                        context: PythonObject) raises -> Tuple[PythonObject, PythonObject]:
    """`_cv_fold_ids_f32(splits, n, name)`: (fold ids, None) for a
    partition, else (the -(n + 1) words, the per-split roles).

    `_cv_fold_ids(splits, n)`: the int32 fold of every row when the test
    sides partition the rows and every train side is the complement of its
    test side (RUN: x_linear/door_run.mojo, on the base helper table).
    The split words and the helper key are read once for both RUNs, and
    both RUNs run with the GIL released: the former `assign_fold_i64`,
    `check_indices_i64`, `count_fold_hits_i64` and `mark_split_roles_f32`
    exports released it around their work (lane perf-audit-2-a)."""
    var nk = x_linear_keys()
    var key = base_helpers_key(py_item(context, nk[].k_xl_helpers))
    var words = _xl_split_words(splits)
    var ids = _xl_full_i32(n, -1, context)
    var ids_addr = labels_address(ids)
    var partition = False
    var py = Python()
    with GILReleased(py):
        partition = xl_cv_fold_ids_run(key, words, n, ids_addr)
    _ = ids  # keep the Python object alive through the native calls above
    _ = splits  # keep the index Arrays alive through the native calls above
    if partition:
        var out = admitted_vector(n, "<f4", context)
        labels_cast(labels_address(ids), "<i4", labels_address(out), "<f4", n, context)
        _ = out  # keep the Python object alive through the native call above
        _ = ids  # keep the Python object alive through the native call above
        return (out, PythonObject(None))
    # `_cv_roles_f32`: per-split roles (x_linear/cv_roles.mojo)
    xl_cv_roles_limit(n, name)
    var roles = admitted_vector(len(splits) * n, "<f4", context)
    var roles_addr = labels_address(roles)
    with GILReleased(py):
        xl_cv_roles_run(key, words, n, roles_addr, name)
    _ = words^
    _ = splits  # keep the index Arrays alive through the native calls above
    return (labels_full_f32(n, -Float64(n + 1), context), roles)


def _xl_parts(addrs: List[Int], counts: List[Int], context: PythonObject) raises -> PythonObject:
    """`_concat_f32_parts(parts)`: the parts' 32-bit words in order, one
    float32 vector (one byte copy per part)."""
    var out = admitted_vector(xl_parts_total(counts), "<f4", context)
    xl_run_parts(labels_address(out), addrs, counts)
    return out


def _xl_undefined_warning(message: String, context: PythonObject) raises:
    """`_expansion_metrics._undefined_warning(message)`: scikit-learn's
    UndefinedMetricWarning when sklearn.exceptions is loaded, else
    mojolearn's, at the public method's frame."""
    var nk = x_linear_keys()
    var category = py_item(context, nk[].k_xl_undefined_warning)
    var sk = py_get(py_item(context, nk[].k_modules), "sklearn.exceptions")
    if sk is not None:
        category = py_call(_b().getattr, sk, PythonObject("UndefinedMetricWarning"), category)
    var frame = py_call(native_loaded_module("sys")._getframe, PythonObject(0))
    var module = py_get(frame.f_globals, nk[].k___name__, "")
    classical_warning(message, category, context, Python.tuple(frame.f_code.co_filename, frame.f_lineno, module),
                      "UndefinedMetricWarning")


# ------------------------------------------------------------------ RidgeCV


struct XlRidgeCv(Movable):
    """One admitted RidgeCV.fit: its path, the typed request RUN reads and
    the objects whose words it addresses, kept alive until PACK returns: X,
    the targets (then the checked weights), the final fit's result words;
    the k-fold scores or the split path's (alpha, split) score table and
    its per-alpha means; and, for the split path, the (train, test) index
    Arrays, the base binding's row gather, and the scratch words every
    split's fits share (`fit_words`: one fit's result; `status_word`: the
    prediction's finite status). `store`: `store_cv_results` (cv=None)."""
    var path: Int
    var request: XlRidgeCvRequest
    var x: PythonObject
    var y: PythonObject
    var result: PythonObject
    var scores: PythonObject
    var means: PythonObject
    var splits: PythonObject
    var gather: PythonObject
    var fit_words: PythonObject
    var status_word: PythonObject
    var store: Bool

    def __init__(out self, path: Int, var request: XlRidgeCvRequest, var x: PythonObject, var y: PythonObject,
                 var result: PythonObject, var scores: PythonObject, var means: PythonObject,
                 var splits: PythonObject, var gather: PythonObject, var fit_words: PythonObject,
                 var status_word: PythonObject, store: Bool):
        self.path = path
        self.request = request^
        self.x = x^
        self.y = y^
        self.result = result^
        self.scores = scores^
        self.means = means^
        self.splits = splits^
        self.gather = gather^
        self.fit_words = fit_words^
        self.status_word = status_word^
        self.store = store


struct XlRidgeFold(Movable):
    """One cv split of RidgeCV's split path: the typed request of its alpha
    grid and the gathered rows it addresses (`pins`). `scored` False: the
    split has fewer than two test rows, its R^2 is undefined and it is not
    run."""
    var request: XlRidgeFoldRequest
    var scored: Bool
    var pins: PythonObject

    def __init__(out self, var request: XlRidgeFoldRequest, scored: Bool, var pins: PythonObject):
        self.request = request^
        self.scored = scored
        self.pins = pins^


def _xl_r2_entry(entries: XLinearOps, context: PythonObject) raises -> NativeDescriptorCall:
    """The typed R^2 entry of the metrics image the fold scores run on
    (core/metrics_entry_table.mojo): the `_mojolearn_metrics` image of the
    process default mode in the scope of the estimator's device, which is
    the image `metrics.r2_score(..., numeric_mode=None)` selected per fold
    and alpha through its Python entries. Read once per fit; an image that
    is not connected refuses in the metrics door's words, and one whose
    compiled mode or vendor is not the scope's has no table under this key
    and refuses in the table's."""
    var nk = x_linear_keys()
    var scope = _xl_scope(entries, context)
    var default_mode = py_item(py_item(scope, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)
    var selected = String(py=py_call(py_call(default_mode.strip).lower))
    _ = labels_table(scope, selected, "_mojolearn_metrics")
    var vendor = device_vendor_code(String(py=py_item(scope, PythonObject("vendor"))))
    return metrics_r2_entry(MetricsEntryKey(device_mode_code(selected), vendor))


def xl_fit_ridge_cv_unpack(model: PythonObject, fields: PythonObject, args: PythonObject, entries: XLinearOps,
                           context: PythonObject) raises -> XlRidgeCv:
    """RidgeCV.fit(X, y, sample_weight): cv=None (one fit over the alpha
    grid), `_fit_kfold` (an int cv without weights and with at least two
    rows a fold) or `_fit_splits` (any other cv: a fit, a prediction and an
    R^2 per split and alpha)."""
    var nk = x_linear_keys()
    var b = _b()
    if py_item(fields, nk[].k_scoring) is not None or _truth(py_item(fields, nk[].k_alpha_per_target)):
        raise Error("mojolearn RidgeCV: only scoring=None, alpha_per_target=False are implemented")
    var given = py_item(fields, nk[].k_alphas)
    var listed = given if Bool(py=py_call(b.hasattr, given, PythonObject("__len__"))) else Python.list(given)
    var floats = py_call(b.list, py_call(b.map, b.float, listed))
    var alphas = List[Float64]()
    for i in range(len(floats)):  # driver-loop: over the alpha grid
        alphas.append(Float64(py=floats[i]))
    var refuse = len(alphas) == 0
    for i in range(len(alphas)):  # driver-loop: over the alpha grid
        refuse = refuse or not (alphas[i] > 0)
    if refuse:
        raise Error("mojolearn RidgeCV: alphas must be positive")
    xl_ops_require(entries)
    var x = _xl_matrix(args[0], _name(model), context)
    var n = x.rows
    var d = x.cols
    var y = _xl_vector(args[1], n, context)
    var sample_weight = args[2]
    var cv = py_item(fields, nk[].k_cv)
    var A = len(alphas)
    var none = PythonObject(None)
    if cv is None:
        var has_sw = 0
        var yy = _xl_with_weights(y, sample_weight, n, has_sw, context)
        var fit_intercept = _truth(py_item(fields, nk[].k_fit_intercept))
        var result = admitted_vector(xl_ridge_out(d, A), "<f4", context)
        var store = _truth(py_item(fields, nk[].k_store_cv_results))
        var plain = XlRidgeCvRequest(entries.entry, x.address, n, d, yy.address, has_sw == 1, fit_intercept,
                                     alphas^, labels_address(result), 0, 0, 0, 0, entry_ref_unresolved().call,
                                     entry_ref_unresolved())
        return XlRidgeCv(XL_RIDGE_CV_PLAIN, plain^, x.array, yy.array, result, none, none, none, none, none, none,
                         store)
    var int_cv = _xl_is_int_cv(cv, context)
    if int_cv and sample_weight is None and _int(cv) >= 2 and n // _int(cv) >= 2:
        # `_fit_kfold`: the k-fold scores on the binding, the first best
        # finite score chosen in Mojo
        var folds = _int(cv)
        if not (2 <= folds and folds <= n):
            raise Error("mojolearn RidgeCV: cv must be in [2, n_samples]")
        if _truth(py_item(fields, nk[].k_store_cv_results)):
            raise Error("cv!=None and store_cv_results=True are incompatible")
        var fit_intercept = _truth(py_item(fields, nk[].k_fit_intercept))
        var scores = admitted_vector(A, "<f4", context)
        var result = admitted_vector(xl_ridge_out(d, 1), "<f4", context)
        var kfold = XlRidgeCvRequest(entries.entry, x.address, n, d, y.address, False, fit_intercept, alphas^,
                                     labels_address(result), folds, labels_address(scores), 0, 0,
                                     entry_ref_unresolved().call, entry_ref_unresolved())
        return XlRidgeCv(XL_RIDGE_CV_KFOLD, kfold^, x.array, y.array, result, scores, none, none, none, none, none,
                         False)
    # `_fit_splits`: a native fit, predict and r2_score per split and alpha
    if _truth(py_item(fields, nk[].k_store_cv_results)):
        raise Error("cv!=None and store_cv_results=True are incompatible")
    var splits = PythonObject(None)
    if int_cv:
        var folds = _int(cv)
        if not (2 <= folds and folds <= n):
            raise Error("mojolearn RidgeCV: cv must be in [2, n_samples]")
        splits = _xl_default_folds(n, folds, context)
    else:
        splits = _xl_cv_splits(cv, args[0], args[1], context)
    # The targets, then the checked weights, in one vector: the final fit
    # reads it as it is, and each split gathers its target rows from the
    # first n words and its weight rows from the second n.
    var has_sw = 0
    var yy = _xl_with_weights(y, sample_weight, n, has_sw, context)
    var fit_intercept = _truth(py_item(fields, nk[].k_fit_intercept))
    var r2 = _xl_r2_entry(entries, context)
    var helpers = base_helpers_key(py_item(context, nk[].k_xl_helpers))
    var gather = _xl_helper(context, "gather_rows_bytes")
    var k = len(splits)
    var table = admitted_vector(max(A * k, 1), "<f8", context)
    var means = admitted_vector(max(A, 1), "<f8", context)
    var result = admitted_vector(xl_ridge_out(d, 1), "<f4", context)
    var fit_words = admitted_vector(xl_ridge_out(d, 1), "<f4", context)
    var status_word = admitted_vector(1, "<i4", context)
    var request = XlRidgeCvRequest(entries.entry, x.address, n, d, yy.address, has_sw == 1, fit_intercept, alphas^,
                                   labels_address(result), 0, labels_address(table), labels_address(means), k, r2,
                                   helpers)
    return XlRidgeCv(XL_RIDGE_CV_SPLITS, request^, x.array, yy.array, result, table, means, splits, gather,
                     fit_words, status_word, False)


def _xl_gather_into(gather: PythonObject, src: Int, dst: PythonObject, offset: Int, rows: PythonObject, n: Int,
                    count: Int, width: Int) raises:
    """`take(src, idx, width)`: rows `rows` (`count` int64 indices) of the n
    rows of `width` float32 words at `src`, written from word `offset` of
    `dst` (the base binding's `gather_rows_bytes`)."""
    _ = py_call_typed(gather, PythonObject(src), PythonObject(labels_address(dst) + 4 * offset),
                      PythonObject(labels_address(rows)), PythonObject(n), PythonObject(count),
                      PythonObject(4 * width))
    _ = dst  # keep the Python object alive through the native call above
    _ = rows  # keep the Python object alive through the native call above


def xl_ridge_cv_fold_unpack(call: XlRidgeCv, s: Int, context: PythonObject) raises -> XlRidgeFold:
    """UNPACK of split `s` of RidgeCV's split path: its refusal, the
    undefined-R^2 warning of a split with fewer than two test rows (not
    run), and its gathered rows: X and the targets of the training rows
    (the training weights behind the targets when the fit is weighted), X,
    the targets and the weights of the test rows. Every gather writes all
    of its words, so the vectors are allocated unfilled."""
    var tr = call.splits[s][0]
    var te = call.splits[s][1]
    var n_tr = Int(py=tr.size)
    var n_te = Int(py=te.size)
    if n_tr == 0 or n_te == 0:
        raise Error("mojolearn RidgeCV: every cv split needs training and test rows")
    if n_te < 2:
        # scikit-learn's r2_score: NaN, with its warning, for fewer than
        # two test rows; every alpha's mean is then NaN (the first wins)
        _xl_undefined_warning("R^2 score is not well-defined with less than two samples.", context)
        return XlRidgeFold(XlRidgeFoldRequest(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, s), False, PythonObject(None))
    var n = call.request.n
    var d = call.request.d
    var weighted = call.request.weighted
    var x_train = admitted_storage_native([n_tr * d], "<f4", context, as_array=True)
    _xl_gather_into(call.gather, call.request.x, x_train, 0, tr, n, n_tr, d)
    var y_train = admitted_storage_native([2 * n_tr if weighted else n_tr], "<f4", context, as_array=True)
    _xl_gather_into(call.gather, call.request.y, y_train, 0, tr, n, n_tr, 1)
    var x_test = admitted_storage_native([n_te * d], "<f4", context, as_array=True)
    _xl_gather_into(call.gather, call.request.x, x_test, 0, te, n, n_te, d)
    var y_test = admitted_storage_native([n_te], "<f4", context, as_array=True)
    _xl_gather_into(call.gather, call.request.y, y_test, 0, te, n, n_te, 1)
    var w_test = PythonObject(None)
    var w_test_addr = 0
    if weighted:
        var weights = call.request.y + 4 * n
        _xl_gather_into(call.gather, weights, y_train, n_tr, tr, n, n_tr, 1)
        w_test = admitted_storage_native([n_te], "<f4", context, as_array=True)
        _xl_gather_into(call.gather, weights, w_test, 0, te, n, n_te, 1)
        w_test_addr = labels_address(w_test)
    var pred = admitted_storage_native([n_te], "<f4", context, as_array=True)
    var request = XlRidgeFoldRequest(labels_address(x_train), n_tr, labels_address(y_train), labels_address(x_test),
                                     n_te, labels_address(y_test), w_test_addr, labels_address(call.fit_words),
                                     labels_address(pred), labels_address(call.status_word), s)
    return XlRidgeFold(request^, True, Python.tuple(x_train, y_train, x_test, y_test, w_test, pred))


# ------------------------------------------------------ LassoCV / ElasticNetCV


struct XlEnetCvFit(Movable):
    """One admitted LassoCV / ElasticNetCV fit: the fit call, the l1_ratio
    grid as the user's own floats (`ratios`, `count` of them), the alpha
    grid's length and the folds."""
    var fit: XlFit
    var ratios: PythonObject
    var count: Int
    var grid: Int
    var folds: Int

    def __init__(out self, var fit: XlFit, var ratios: PythonObject, count: Int, grid: Int, folds: Int):
        self.fit = fit^
        self.ratios = ratios^
        self.count = count
        self.grid = grid
        self.folds = folds


def _xl_enetcv_unpack(model: PythonObject, fields: PythonObject, args: PythonObject, l1_ratios: PythonObject,
                      entries: XLinearOps, context: PythonObject) raises -> XlEnetCvFit:
    """`_enetcv_fit(est, X, y, l1_ratios, sample_weight)`, UNPACK: the
    alpha grid, the folds (ids, or per-split roles for splits that are not
    a partition) and the weights as one target vector, and the typed fit
    request of the path on the binding."""
    var nk = x_linear_keys()
    var name = _name(model)
    var b = _b()
    if not _eq(py_item(fields, nk[].k_selection), PythonObject("cyclic")):
        raise Error("mojolearn " + name + ": selection='random' is not implemented")
    var L = len(l1_ratios)
    for i in range(L):  # driver-loop: over the l1_ratio grid
        var r = Float64(py=l1_ratios[i])
        if not (r > 0 and r <= 1):
            raise Error("mojolearn " + name + ": l1_ratio must be in (0, 1]")
    xl_ops_require(entries)
    var x = _xl_matrix(args[0], _name(model), context)
    var n = x.rows
    var d = x.cols
    var y = _xl_vector(args[1], n, context)
    var cv = py_item(fields, nk[].k_cv)
    var ids = PythonObject(None)
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var roles = Optional[PythonObject]()
    var folds = 0
    if cv is None or _xl_is_int_cv(cv, context):
        var k = PythonObject(5) if cv is None else py_call(b.int, cv)
        ids = _xl_kfold_ids_f32(n, k, context)
        folds = _int(k)
    else:
        var splits = _xl_cv_splits(cv, args[0], args[1], context)
        folds = len(splits)
        var got = _xl_cv_fold_ids_f32(splits, n, name, context)
        ids = got[0]
        roles = py_present(got[1])
    var alphas = py_item(fields, nk[].k_alphas)
    var n_alphas = _attr(model, "n_alphas", PythonObject(None))
    var explicit = False
    var grid = 0
    var values = Python.list()
    if _is_int(alphas):
        grid = _int(alphas)
    elif alphas is None or py_contains(Python.tuple("warn", "deprecated"), alphas):
        grid = _int(n_alphas) if _is_int_plain(n_alphas) else 100
    else:
        explicit = True
        values = py_call(b.sorted, py_call(b.map, b.float, alphas), reverse=PythonObject(True))
        grid = len(values)
    if grid < 1:
        raise Error("mojolearn " + name + ": at least one alpha is required")
    var sample_weight = args[2]
    # A plain number is no weight vector (the former numbers.Number test,
    # by exact native type identity).
    if labels_real_type(py_type(sample_weight), context) or py_type(sample_weight) is _b().complex:
        sample_weight = PythonObject(None)
    var addrs = List[Int]()
    var counts = List[Int]()
    addrs.append(y.address)
    _ = y  # keep the Python object alive through the native call above
    counts.append(n)
    addrs.append(labels_address(ids))
    _ = ids  # keep the Python object alive through the native call above
    counts.append(n)
    var weighted = 0
    var w_keep = PythonObject(None)
    if Bool(roles):
        # per-split roles ride the weighted arm, after the weights
        if sample_weight is None:
            w_keep = labels_full_f32(n, 1.0, context)
        else:
            w_keep = _xl_owned(admit_weight_words(sample_weight, n, context), context)
        addrs.append(labels_address(w_keep))
        _ = w_keep  # keep the Python object alive through the native call above
        counts.append(n)
        addrs.append(labels_address(roles.value()))
        _ = roles  # keep the Python object alive through the native call above
        counts.append(Int(py=roles.value().size))
        weighted = 1
    elif sample_weight is not None:
        w_keep = _xl_owned(admit_weight_words(sample_weight, n, context), context)
        addrs.append(labels_address(w_keep))
        _ = w_keep  # keep the Python object alive through the native call above
        counts.append(n)
        weighted = 1
    var yy = _xl_parts(addrs, counts, context)
    _ = y  # keep the parts' owners alive through the copies above
    _ = ids  # keep the parts' owners alive through the copies above
    _ = w_keep  # keep the parts' owners alive through the copies above
    _ = roles  # keep the parts' owners alive through the copies above
    var fp = List[Float64]()
    fp.append(_float(py_item(fields, nk[].k_eps)))
    fp.append(_float(py_item(fields, nk[].k_tol)))
    for i in range(L):  # driver-loop: over the l1_ratio grid
        fp.append(Float64(py=l1_ratios[i]))
    if explicit:
        for i in range(grid):  # driver-loop: over the user alpha grid
            fp.append(Float64(py=values[i]))
    var ip: List[Int] = [_int(py_item(fields, nk[].k_max_iter)), 1 if _truth(py_item(fields, nk[].k_fit_intercept)) else 0, grid, folds, L,
                         1 if explicit else 0, 1 if _truth(py_item(fields, nk[].k_positive)) else 0, weighted]
    # the parts were copied into `yy`: X and `yy` are what the fit reads
    var fit = xl_fit_build(entries, ALGO_ENETCV, x.address, n * d, n, d, x.array, labels_address(yy),
                           Int(py=yy.size), yy, ip^, fp^, d + 4 + L * grid + L * grid * folds,
                           d * d + 4 * d + 3 + grid * (d + 2), 1, context)
    return XlEnetCvFit(fit^, l1_ratios, L, grid, folds)


def _is_int_plain(value: PythonObject) raises -> Bool:
    """`isinstance(value, int)` (bools included, as the Python test was)."""
    return _isinstance(value, _b().int)


def xl_fit_lasso_cv_unpack(model: PythonObject, fields: PythonObject, args: PythonObject, entries: XLinearOps,
                           context: PythonObject) raises -> XlEnetCvFit:
    """LassoCV.fit(X, y, sample_weight): the l1_ratio grid [1.0]."""
    return _xl_enetcv_unpack(model, fields, args, Python.list(PythonObject(1.0)), entries, context)


def xl_fit_elastic_net_cv_unpack(model: PythonObject, fields: PythonObject, args: PythonObject,
                                 entries: XLinearOps, context: PythonObject) raises -> XlEnetCvFit:
    """ElasticNetCV.fit(X, y, sample_weight): the l1_ratio grid of the
    estimator, as floats."""
    var nk = x_linear_keys()
    var b = _b()
    var given = py_item(fields, nk[].k_l1_ratio)
    var ratios = py_call(b.list, given) if Bool(py=py_call(b.hasattr, given, PythonObject("__len__"))) else Python.list(given)
    var floats = py_call(b.list, py_call(b.map, b.float, ratios))
    return _xl_enetcv_unpack(model, fields, args, floats, entries, context)


# ---------------------------------------------------- LogisticRegressionCV


struct XlLogCvFit(Movable):
    """One admitted LogisticRegressionCV.fit: the fit call, the class-weight
    step RUN makes before it (`prep`), the sorted class labels, the objects
    the step addresses (`pins`: the int32 codes, the class weights, the
    sample weights), the Cs grid, the fitted problems (`kp`), the folds and
    whether they are the binding's stratified ones (checked in RUN against
    the largest class)."""
    var fit: XlFit
    var prep: XlClassPrepRequest
    var classes: PythonObject
    var pins: PythonObject
    var cs: List[Float64]
    var kp: Int
    var folds: Int
    var stratified: Bool

    def __init__(out self, var fit: XlFit, var prep: XlClassPrepRequest, var classes: PythonObject,
                 var pins: PythonObject, var cs: List[Float64], kp: Int, folds: Int, stratified: Bool):
        self.fit = fit^
        self.prep = prep^
        self.classes = classes^
        self.pins = pins^
        self.cs = cs^
        self.kp = kp
        self.folds = folds
        self.stratified = stratified


def _xl_cs_grid(cs: PythonObject, context: PythonObject) raises -> List[Float64]:
    """The Cs grid: `np.logspace(-4, 4, m)` through `_pm.powr` (10 ** y by
    the decimal module at 50 digits, one correctly rounded conversion), or
    the user's values."""
    var nk = x_linear_keys()
    var grid = List[Float64]()
    if _is_int(cs):
        var m = _int(cs)
        if m <= 1:
            grid.append(1e-4)
            return grid^
        var ten = py_call(py_item(context, nk[].k_xl_decimal), PythonObject(10.0))
        var ctx = py_item(context, nk[].k_xl_pow_context)
        for i in range(m):  # driver-loop: over the Cs grid
            var exponent = -4.0 + Float64(8 * i) / Float64(m - 1)
            var value = Float64(py=py_call(_b().float, py_call(ctx.power, ten, py_call(py_item(context, nk[].k_xl_decimal), PythonObject(exponent)))))
            if isinf(value):
                raise Error("OverflowError: math range error")
            grid.append(value)
        return grid^
    var floats = py_call(_b().list, py_call(_b().map, _b().float, cs))
    for i in range(len(floats)):  # driver-loop: over the user Cs grid
        grid.append(Float64(py=floats[i]))
    return grid^


def xl_fit_logistic_cv_unpack(model: PythonObject, fields: PythonObject, args: PythonObject, entries: XLinearOps,
                              context: PythonObject) raises -> XlLogCvFit:
    """LogisticRegressionCV.fit(X, y, sample_weight): the class codes, the
    Cs grid, the folds, the weights and the typed requests of the
    class-weight step and of the fit. The class-weight step (its row
    weights and the largest class count the stratified folds are checked
    against) is RUN's (`xl_logcv_fit_run`)."""
    var nk = x_linear_keys()
    if not _eq(py_item(fields, nk[].k_penalty), PythonObject("l2")) or _truth(py_item(fields, nk[].k_dual)) or py_item(fields, nk[].k_l1_ratios) is not None:
        raise Error("mojolearn LogisticRegressionCV: only penalty='l2' (primal) is implemented")
    if py_item(fields, nk[].k_scoring) is not None or not _truth(py_item(fields, nk[].k_refit)):
        raise Error("mojolearn LogisticRegressionCV: scoring and refit=False are not implemented")
    if not py_contains(Python.tuple("lbfgs", "newton-cg", "newton-cholesky"), py_item(fields, nk[].k_solver)):
        raise Error("mojolearn LogisticRegressionCV: solver must be lbfgs (newton-* run L-BFGS too)")
    xl_ops_require(entries)
    var x = _xl_matrix(args[0], _name(model), context)
    var n = x.rows
    var d = x.cols
    var labels = _xl_class_codes(model, args[1], n, _xl_scope(entries, context))
    var K = len(labels.classes)
    var kp = 1 if K == 2 else K
    var Cs = _xl_cs_grid(py_item(fields, nk[].k_Cs), context)
    var cv = py_item(fields, nk[].k_cv)
    var explicit = not (cv is None or _xl_is_int_cv(cv, context))
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var roles = Optional[PythonObject]()
    var fold_ids = PythonObject(None)
    var folds = 0
    if explicit:
        var splits = _xl_cv_splits(cv, args[0], args[1], context)
        folds = len(splits)
        var got = _xl_cv_fold_ids_f32(splits, n, "LogisticRegressionCV", context)
        fold_ids = got[0]
        roles = py_present(got[1])
    else:
        folds = 5 if cv is None else _int(cv)
    var sample_weight = args[2]
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var raw = Optional[PythonObject]()
    if sample_weight is not None:
        raw = _xl_owned(admit_weight_words(sample_weight, n, context), context)
    var class_weight = py_item(fields, nk[].k_class_weight)
    var weigh = sample_weight is not None or class_weight is not None or Bool(roles)
    # `_class_prep(est, class_weight, classes, codes, sample_weight, True,
    # class_weight is not None)`: its class-weight words here, its call in RUN
    var weights = xl_class_weights_unpack(class_weight, labels.classes, context)
    var sw_addr = 0 if not raw else labels_address(raw.value())
    var codes = admitted_vector(n, "<f4", context)
    labels_cast(labels_address(labels.codes), "<i4", labels_address(codes), "<f4", n, context)
    _ = codes  # keep the Python object alive through the native call above
    var zeros = admitted_vector(n, "<f4", context)
    var addrs = List[Int]()
    var counts = List[Int]()
    addrs.append(labels_address(codes))
    _ = codes  # keep the Python object alive through the native call above
    counts.append(n)
    addrs.append(labels_address(fold_ids) if explicit else labels_address(zeros))
    _ = zeros  # keep the Python object alive through the native call above
    _ = fold_ids  # keep the Python object alive through the native call above
    counts.append(n)
    var has_sw = 0
    if weigh:
        if not raw:
            raw = labels_full_f32(n, 1.0, context)
        # The fit weights, then the raw weights. With class_weight the fit
        # weights are the class-weight step's row weights, which RUN writes
        # over this part (the third n words); without, they are the raw ones.
        addrs.append(labels_address(raw.value()))
        counts.append(n)
        addrs.append(labels_address(raw.value()))
        _ = raw  # keep the Python object alive through the native call above
        counts.append(n)
        has_sw = 1
    if Bool(roles):
        addrs.append(labels_address(roles.value()))
        _ = roles  # keep the Python object alive through the native call above
        counts.append(Int(py=roles.value().size))
    var yy = _xl_parts(addrs, counts, context)
    _ = codes  # keep the parts' owners alive through the copies above
    _ = zeros  # keep the parts' owners alive through the copies above
    _ = fold_ids  # keep the parts' owners alive through the copies above
    _ = roles  # keep the parts' owners alive through the copies above
    var rows_addr = labels_address(yy) + 4 * 2 * n if class_weight is not None else 0
    var prep = XlClassPrepRequest(True, labels_address(labels.codes), sw_addr, labels_address(weights[0]), rows_addr,
                                  n, K, weights[1], True)
    var p = kp * (d + 1)
    var nc = len(Cs)
    var ip: List[Int] = [_int(py_item(fields, nk[].k_max_iter)), 1 if _truth(py_item(fields, nk[].k_fit_intercept)) else 0, kp, nc, folds, has_sw]
    if explicit:
        ip.append(1)
    var fp = List[Float64]()
    fp.append(_float(py_item(fields, nk[].k_tol)))
    for i in range(nc):  # driver-loop: over the Cs grid
        fp.append(Cs[i])
    var fit = xl_fit_build(entries, ALGO_LOGCV, x.address, n * d, n, d, x.array, labels_address(yy), Int(py=yy.size),
                           yy, ip^, fp^, kp * d + kp + 2 + folds * nc, p + 1 + n * (kp + 1) + xl_lbfgs_work(p), 5,
                           context)
    return XlLogCvFit(fit^, prep^, labels.classes, Python.tuple(labels.codes, weights[0], py_or_none(raw)), Cs^, kp,
                      folds, not explicit)
