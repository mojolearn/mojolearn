# SPDX-License-Identifier: Apache-2.0
# Shared steps of the native classifier owner (bindings/prep_classify_native.mojo):
# the model's kind and numeric mode, the connected optional x_prep exports, the
# X admission of `_expansion_prep._x2d`, the CSR admission, the user parameter
# lists, the image's mode words and the partial_fit class codes (UNPACK; the
# statistics stage groups and the label staging are the typed builders of
# bindings/prep_classify_prog_run.mojo).
from std.os import getenv
from bindings.native_array_storage_door import native_array_empty, native_array_unfilled
from bindings.native_loaded_module_door import native_loaded_module
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.mode_answer_door import answer_mode_object
from core.mode_answer import MODE_FAMILY_PREP
from bindings.py_typed_call import py_call_typed
from bindings.classical_arrays_door import classical_address, classical_dtype_str, classical_c_order, classical_c_bytes, classical_wrap_array, classical_is_array
from bindings.admitted_input_unpack import admit_x
from bindings.classical_validate_unpack import classical_view
from bindings.x_prep_native_inputs_door import _rows_fields, _values
from bindings.py_admitted_registry import admitted_held_index
from bindings.admitted_meta_scope_unpack import admitted_held_verify
from core.host_finite_scan import host_all_finite_f32
from bindings.classical_labels_door import ClassicalLabels, classical_encode_labels
from bindings.prep_classify_prog_door import ClsProg, NONE, INT32_MAX, off_add, f32_array, i32_array
from bindings.prep_classify_prog_run import ClsDaMode, ClsLabels
from bindings.sequential_admission_door import _one_of
from bindings.prep_entry_cache_unpack import prep_entry_word, prep_entry_text
from bindings.prep_keys_unpack import prep_keys
from bindings.py_balanced import py_item

comptime I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime I64P = MutPointer[Int64, MutUntrackedOrigin]
comptime F32P = MutPointer[Float32, MutUntrackedOrigin]

# x_prep stage codes (_expansion_prep._OPS).
comptime OP_CSR_DENSE = 164
# _IDN_FAM bits.
comptime IDN_STATS_BLOCKED = 1
comptime IDN_CLASS_ONEPASS = 2
# _IDN_INT bits.
comptime IDN_NB_ONEPASS = 2
comptime IDN_NB_CSR = 4
comptime IDN_NB_CSR_DENSE = 8


def cls_builtins() raises -> PythonObject:
    return native_loaded_module("builtins")


def cls_name(model: PythonObject) raises -> String:
    return String(py_type(model).__name__)


def cls_repr(value: PythonObject) raises -> String:
    return String(py_call(cls_builtins().repr, value))


def cls_str(value: PythonObject) raises -> String:
    return String(py_call(cls_builtins().str, value))


def cls_kind(model: PythonObject, context: PythonObject) raises -> String:
    """The admitted classifier this model is (or subclasses), by its MRO."""
    var nk = prep_keys()
    var b = cls_builtins()
    var mro = py_call(py_item(b.type.__dict__, nk[].k___mro__).__get__, py_type(model))
    var kinds = py_item(context, nk[].k_cls_kinds)
    for k in range(len(mro)):  # driver-loop: over the model class's MRO
        var kind = py_get(kinds, mro[k])
        if kind is not None:
            return String(kind)
    raise Error("TypeError: the native classifier owner does not serve this class")


def cls_mode(context: PythonObject) raises -> PythonObject:
    """`_expansion_prep._mode()`: the process numeric mode."""
    var nk = prep_keys()
    return py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)


def cls_requested_mode(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    """The mode a classifier with an optional `numeric_mode` field fits in
    (GaussianNB): the process default when the field is None or absent (a
    model pickled before the parameter existed), else the validated word,
    answered (core/mode_answer.mojo, family PREP). Once per fit, in no loop;
    the None path makes the one default read `cls_mode` makes."""
    var nk = prep_keys()
    var mode = py_get(fields, nk[].k_numeric_mode)
    if mode is None:
        return py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE)
    if py_type(mode) is not cls_builtins().str:
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    mode = py_call(py_call(mode.strip).lower)
    var mode_word = String(mode)
    if mode_word != "fast" and mode_word != "identical":
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    answer_mode_object(mode, mode_word, MODE_FAMILY_PREP)
    return mode


def cls_vendor_used(model: PythonObject, context: PythonObject) raises -> PythonObject:
    """`vendor_used()` of the naive Bayes and discriminant-analysis
    classifiers: the vendor word the x_prep image of `context` (the context
    of the estimator's device route, `scaler_route_context`) reports, for
    the mode the estimator was fitted in, else its `numeric_mode`, else the
    process mode; None for an image built before the read-back. Metadata,
    in no loop."""
    var nk = prep_keys()
    var b = cls_builtins()
    var fields = py_fields(model)
    var mode = py_get(fields, "numeric_mode_")
    if mode is None:
        mode = py_get(fields, nk[].k_numeric_mode)
    if mode is None:
        mode = cls_mode(context)
    if py_type(mode) is not b.str:
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    mode = py_call(py_call(mode.strip).lower)
    var entries = py_get(py_item(context, nk[].k_cls_bindings), mode)
    if entries is None:
        raise Error("ValueError: numeric_mode must be fast, identical or None")
    if py_type(entries) is b.str:
        raise Error("ImportError: " + String(entries))
    var text = prep_entry_text(entries, "x_prep_vendor")
    if text.byte_length() == 0:
        return PythonObject(None)
    return PythonObject(text)


def cls_entry(mode: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """`_optional_prep_entry(_prep_binding(mode), name)`: the connected export,
    or None (a binding that did not connect fails in the run itself)."""
    var nk = prep_keys()
    var entries = py_get(py_item(context, nk[].k_cls_bindings), mode)
    if entries is None or py_type(entries) is cls_builtins().str:
        return PythonObject(None)
    return py_get(entries, PythonObject(name))


def _bits(mode: PythonObject, name: String, context: PythonObject) raises -> Int:
    var nk = prep_keys()
    if String(mode) != "identical":
        return 0
    var entries = py_get(py_item(context, nk[].k_cls_bindings), mode)
    if entries is None or py_type(entries) is cls_builtins().str:
        return 0
    return prep_entry_word(entries, name)  # read once per binding


def idn_int(mode: PythonObject, context: PythonObject) raises -> Int:
    return _bits(mode, "x_prep_idn_int", context)


def idn_fam(mode: PythonObject, context: PythonObject) raises -> Int:
    return _bits(mode, "x_prep_idn_fam", context)


def idn_fam2(mode: PythonObject, context: PythonObject) raises -> Int:
    return _bits(mode, "x_prep_idn_fam2", context)


def classical_shared(mode: PythonObject, context: PythonObject) raises -> Int:
    return _bits(mode, "x_prep_classical_shared", context)


def blocked() -> Bool:
    """`_expansion_prep._blocked()`: MOJOLEARN_XPREP_BLOCKED unset or not "0"."""
    return String(getenv("MOJOLEARN_XPREP_BLOCKED", "1").strip()) != "0"


def check_fitted(model: PythonObject, fields: PythonObject) raises:
    var nk = prep_keys()
    if not py_contains(fields, nk[].k_n_features_in_):
        raise Error("RuntimeError: mojolearn: this " + cls_name(model) + " instance is not fitted yet")


def check_width(model: PythonObject, fields: PythonObject, d: Int) raises:
    var nk = prep_keys()
    var fitted = py_item(fields, nk[].k_n_features_in_)
    if py_type(fitted) is not cls_builtins().int or Int(py=fitted) != d:
        raise Error("mojolearn: X has " + String(d) + " features, " + cls_name(model) + " was fitted with " + cls_str(fitted))


struct XIn(Movable):
    """An admitted X: a float32 C-ordered array, or a fold's float32 2-D
    DeviceRows kept as a handle (the runner gathers it on the device)."""
    var value: PythonObject
    var n: Int
    var d: Int

    def __init__(out self, value: PythonObject, n: Int, d: Int):
        self.value = value
        self.n = n
        self.d = d


def cls_x2d(source: PythonObject, context: PythonObject) raises -> XIn:
    """`_expansion_prep._x2d(X)`."""
    var nk = prep_keys()
    var b = cls_builtins()
    var value = source
    var n = 0
    var d = 0
    if py_type(source) is py_item(context, nk[].k_maxabs_rows_class):
        var fold = _rows_fields(source, context)
        # WHO REFUSES A NON-FINITE FOLD: model selection holds X with its
        # finite test deferred to the member doors (bindings/msel_cv_door.mojo),
        # and these programs hold no finite test (core/naive_bayes_member.mojo
        # `_capabilities`), so this door reads the held base's recorded verdict
        # (one scan per held X at most, then recorded: nothing per fold), and
        # on a thread that does not hold the base one parallel host scan of
        # its words (`host_all_finite_f32`, across the cores). A base that is
        # not proved finite takes the fold's host rows and the usual
        # admission, which refuses a non-finite row of THIS fold in the
        # estimator's words (a fold without one still fits). Proving suite
        # 2026-11-22: cross_validate over a GaussianNB fitted folds holding
        # NaN rows and scored them.
        var dense = fold.dtype == "<f4" and len(fold.trailing) == 1
        var proved = False
        if dense:
            var held = admitted_held_index(fold.base)
            if held >= 0:
                proved = admitted_held_verify(held)
            else:
                var words = F32P(unsafe_from_address=Int(py=classical_address(fold.base)))
                proved = host_all_finite_f32(words, fold.base_rows * fold.trailing[0])
            _ = fold.base  # the base is read through its address above: keep it alive past the scan
        if proved:
            n = fold.rows
            d = fold.trailing[0]
        else:
            # the host rows (DeviceRows.materialize) and the usual admission
            value = admit_x(_values(source, context, True), 2, "X", context)
            n = Int(py=value.shape[0])
            d = Int(py=value.shape[1])
    else:
        value = admit_x(source, 2, "X", context)
        n = Int(py=value.shape[0])
        d = Int(py=value.shape[1])
    if n == 0 or d == 0:
        raise Error("mojolearn: X must be a nonempty two-dimensional array")
    if n > INT32_MAX // d:
        raise Error("mojolearn: X exceeds the native Int32 indexing bound")
    return XIn(value, n, d)


def is_sparse(source: PythonObject) raises -> Bool:
    """A scipy.sparse container, recognized by its defining module."""
    var module = py_type(source).__module__
    return py_type(module) is cls_builtins().str and Bool(py=py_call(module.startswith, "scipy.sparse"))


def _i32_vector(source: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """`_as_typed(source, "<i4", "C", 1, name)` for a CSR index array."""
    var nk = prep_keys()
    var view = classical_view(source, context, name)
    if Int(py=view.ndim) != 1:
        raise Error(name + " must be 1-D")
    var n = Int(py=view.shape[0])
    var typestr = classical_dtype_str(view, context)
    if typestr != "<i4" and typestr != "<i8":
        raise Error("TypeError: " + name + " must hold int32 or int64 indices")
    # fb-infer/B (2026-11-17): a C-contiguous int32 index buffer (scipy's own
    # CSR form) is BORROWED, as 0.8.55's `_as_typed` returned it: the callers
    # only read it (the CSR kernels' and the program's uploads) while CsrIn
    # holds the view. d95f805cce built a zero-filled int32 array and memcpy'd
    # the indices into it on every call: two serial passes over 4 * nnz fresh
    # bytes per fit and per predict/predict_proba (MultinomialNB/ComplementNB
    # istella fit +290 ms, infer +63 ms; text, taxi alike). Removed: that
    # allocation and copy. A strided or int64 view still narrows into an
    # unfilled array (every word written before any read).
    if n == 0:
        return classical_wrap_array(native_array_empty("i"), Python.tuple(0), "<i4", "C", py_item(context, nk[].k_output_context))
    if typestr == "<i4" and classical_c_order(view, context):
        return view
    var out = classical_wrap_array(native_array_unfilled("i", n), Python.tuple(n), "<i4", "C",
                                   py_item(context, nk[].k_output_context))
    var dest = I32P(unsafe_from_address=Int(py=classical_address(out)))
    _ = out  # keep the Python object alive through the native call above
    # int64 words are one range-checked narrowing pass; only a strided view
    # takes the bytes copy first.
    var raw = PythonObject(None)
    var src = 0
    if classical_c_order(view, context):
        src = Int(py=classical_address(view))
    else:
        raw = native_array_empty(String("q") if typestr == "<i8" else String("i"))
        _ = py_call(raw.frombytes, classical_c_bytes(view, name, context))
        src = Int(py=py_call(raw.buffer_info)[0])
    for i in range(n):  # cpu-route: the CSR index words narrowed to int32 once at input, as _as_typed did
        var v = Int(I64P(unsafe_from_address=src)[i]) if typestr == "<i8" else Int(I32P(unsafe_from_address=src)[i])
        if v > INT32_MAX or v < -INT32_MAX - 1:
            raise Error(name + " holds an index beyond the int32 range")
        dest[i] = Int32(v)
        _ = out  # keep the Python object alive through the native call above
    _ = raw^
    _ = view  # keep the Python object alive through the reads above
    return out


struct CsrIn(Movable):
    """`_csr_input(X)`: (indptr, indices, data, n, d) of X.tocsr()."""
    var indptr: PythonObject
    var indices: PythonObject
    var data: PythonObject
    var n: Int
    var d: Int
    var nnz: Int

    def __init__(out self, indptr: PythonObject, indices: PythonObject, data: PythonObject, n: Int, d: Int, nnz: Int):
        self.indptr = indptr
        self.indices = indices
        self.data = data
        self.n = n
        self.d = d
        self.nnz = nnz


def csr_input(source: PythonObject, context: PythonObject) raises -> CsrIn:
    var csr = py_call(source.tocsr)
    var shape = csr.shape
    var n = Int(py=py_call(cls_builtins().int, shape[0]))
    var d = Int(py=py_call(cls_builtins().int, shape[1]))
    var nnz = Int(py=py_call(cls_builtins().int, csr.nnz))
    var data = admit_x(csr.data, 1, "data", context) if nnz > 0 else f32_array(0, context)
    return CsrIn(_i32_vector(csr.indptr, "indptr", context), _i32_vector(csr.indices, "indices", context), data, n, d, nnz)


def csr_dense_x(mut pr: ClsProg, csr: CsrIn, mode: PythonObject, context: PythonObject) raises -> Int:
    """`_csr_dense_x`: the offset of the X block the device builds from the
    CSR arrays, or -1 when the binding has no csr_dense (the caller densifies;
    -1 is never an arena or scratch offset)."""
    if (idn_int(mode, context) & IDN_NB_CSR_DENSE) == 0:
        return -1
    if csr.n * csr.d > INT32_MAX:
        raise Error("mojolearn: X exceeds the native Int32 indexing bound")
    var xo = pr.work(csr.n * csr.d)
    var ip = pr.put_words(csr.indptr)
    var ix = pr.put_words(csr.indices)
    var dv = pr.put(csr.data)
    pr.stage(OP_CSR_DENSE, csr.n, ip, ix, dv, csr.d, xo)
    return xo


def float_list(values: PythonObject, context: PythonObject) raises -> List[Float64]:
    """`[float(v) for v in (values.tolist() if hasattr(values, "tolist") else values)]`."""
    var b = cls_builtins()
    var items = values
    if py_type(values) is not b.list and py_type(values) is not b.tuple:
        var view = classical_view(values, context, "priors")
        var typestr = classical_dtype_str(view, context)
        var code = String("d")
        if typestr == "<f4":
            code = "f"
        elif typestr == "<i8":
            code = "q"
        elif typestr == "<i4":
            code = "i"
        elif typestr == "<u1" or typestr == "|u1" or typestr == "|b1":
            code = "B"
        elif typestr != "<f8":
            raise Error("TypeError: the values must be a sequence of real numbers")
        var store = native_array_empty(code)
        _ = py_call(store.frombytes, classical_c_bytes(view, "priors", context))
        items = py_call(store.tolist)
    var out = List[Float64]()
    for i in range(len(items)):  # driver-loop: over a user parameter list (class- or feature-count sized)
        out.append(Float64(py=py_call(b.float, items[i])))
    return out^


def given_priors(values: PythonObject, K: Int, who: String, check_sum: Bool, context: PythonObject) raises -> List[Float64]:
    """`_given_priors`: the user priors as floats, checked as the reference checks them."""
    var vals = float_list(values, context)
    if len(vals) != K:
        raise Error("mojolearn: " + who + ": number of priors must match number of classes")
    check_priors(vals, who, check_sum)
    return vals^


def check_priors(vals: List[Float64], who: String, check_sum: Bool) raises:
    """The value checks of `_given_priors` (its count check needs the class
    count, which a member's admission does not have yet)."""
    for i in range(len(vals)):  # driver-loop: over the user prior list (class-count sized)
        if vals[i] < 0:
            raise Error("mojolearn: " + who + ": priors must be non-negative")
    if check_sum:
        var total = prior_sum(vals)
        if abs(total - 1.0) > 1e-8 * max(1.0, abs(total)) and abs(total - 1.0) > 1e-5:
            raise Error("mojolearn: " + who + ": the sum of the priors should be 1")


def gnb_refuse(fields: PythonObject, context: PythonObject) raises:
    """GaussianNB's `var_smoothing` check, in the reference's words."""
    var nk = prep_keys()
    var vs = py_item(fields, nk[].k_var_smoothing)
    var b = cls_builtins()
    if py_type(vs) is b.bool or not is_real(vs, context) or not (cls_float(vs) >= 0):
        raise Error("The 'var_smoothing' parameter of GaussianNB must be a float in the range [0.0, inf). Got " + cls_repr(vs) + " instead.")


def prior_sum(vals: List[Float64]) -> Float64:
    # Python's sum(): left to right, binary64.
    var total = 0.0
    for i in range(len(vals)):  # driver-loop: over the user prior list (class-count sized)
        total += vals[i]
    return total


def is_real(value: PythonObject, context: PythonObject) raises -> Bool:
    var nk = prep_keys()
    var cls = py_type(value)
    if cls is py_type(PythonObject(0)) or cls is py_type(PythonObject(0.0)) or cls is py_type(PythonObject(True)):
        return True
    return (_one_of(cls, py_item(context, nk[].k_scalar_types))
            and not _one_of(cls, py_get(context, nk[].k_bool_types, Python.tuple(py_type(PythonObject(True))))))


def cls_float(value: PythonObject) raises -> Float64:
    return Float64(py=py_call(cls_builtins().float, value))


# ---------------------------------------------------------------- discriminant parameter checks
# (the door's fit and the member admission read them: one copy)


def da_shrinkage(shrinkage: PythonObject, context: PythonObject) raises -> Optional[Float64]:
    """`_shrinkage_value`: None, -1.0 for 'auto', else the float in [0, 1]."""
    var b = cls_builtins()
    if shrinkage is None:
        return None
    if Bool(py=py_call(b.isinstance, shrinkage, b.str)):
        if String(shrinkage) != "auto":
            raise Error("mojolearn: invalid shrinkage " + cls_repr(shrinkage))
        return Optional[Float64](-1.0)
    if not is_real(shrinkage, context) or not (cls_float(shrinkage) >= 0 and cls_float(shrinkage) <= 1):
        raise Error("mojolearn: shrinkage must be 'auto' or a float in [0, 1], got " + cls_repr(shrinkage))
    return Optional[Float64](cls_float(shrinkage))


def da_check_estimator_shrinkage(est: PythonObject, shrinkage: PythonObject) raises:
    if est is not None and shrinkage is not None and Bool(py=shrinkage != PythonObject(0)):
        raise Error("mojolearn: covariance_estimator and shrinkage parameters are not None. Only one of the two can be set.")


def da_solver_is(fields: PythonObject, name: String) raises -> Bool:
    var nk = prep_keys()
    var solver = py_item(fields, nk[].k_solver)
    return py_type(solver) is cls_builtins().str and String(solver) == name


# ---------------------------------------------------------------- the image's mode words


def cls_da_mode(mode: PythonObject, context: PythonObject) raises -> ClsDaMode:
    """UNPACK of the image's mode words a discriminant program is laid out
    from (bindings/prep_classify_prog_run.mojo `ClsDaMode`), read once per
    operation: the typed groups `cls_da_stats`, `cls_gram`, `cls_qda_cov`
    replaced the door groups that read them at every group (`_cs`, `_cls`,
    `_gram`, `_qda_cov`)."""
    if not blocked():
        return ClsDaMode(False, False, 0)
    var fam = idn_fam(mode, context)
    return ClsDaMode((fam & IDN_STATS_BLOCKED) != 0, (fam & IDN_CLASS_ONEPASS) != 0, idn_fam2(mode, context))


def nb_onepass(mode: PythonObject, context: PythonObject) raises -> Bool:
    return blocked() and (idn_int(mode, context) & IDN_NB_ONEPASS) != 0


# ---------------------------------------------------------------- labels


def encode_y(model: PythonObject, fields: PythonObject, y: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    """`_Classifier._encode_y`: classes_ set, the int32 codes returned."""
    var nk = prep_keys()
    var labels = classical_encode_labels(y, context)
    if labels.n != n:
        raise Error("mojolearn: X and y have different numbers of rows")
    py_setitem(fields, nk[].k_classes_, labels.classes)
    return labels.codes


struct PartialCodes(Movable):
    """`_partial_codes`' answer: the batch's class codes as int32 words
    (`codes`), or, for a numeric label buffer whose classes are exact
    ascending float32 words (`_DevCodes`), the raw label words the program
    looks up among the classes on the device."""
    var first: Bool
    var dev: Bool
    var codes: PythonObject
    var words: PythonObject
    var kind: Int
    var n: Int
    var cats: PythonObject
    var y: PythonObject

    def __init__(out self, first: Bool, dev: Bool, codes: PythonObject, words: PythonObject, kind: Int, n: Int,
                 cats: PythonObject, y: PythonObject):
        self.first = first
        self.dev = dev
        self.codes = codes
        self.words = words
        self.kind = kind
        self.n = n
        self.cats = cats
        self.y = y


def partial_walk(fields: PythonObject, y: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    """`_partial_walk`: the batch's codes among classes_, refusing (and naming)
    labels outside them. The batch is encoded once; its classes are looked up
    in classes_ (Python equality, as the dict did) and the row codes remapped."""
    var nk = prep_keys()
    var b = cls_builtins()
    var labels = classical_encode_labels(y, context)
    if labels.n != n:
        raise Error("mojolearn: X and y have different numbers of rows")
    var classes = py_item(fields, nk[].k_classes_)
    var index = Python.dict()
    for i in range(len(classes)):  # driver-loop: over the class index table (classes-sized)
        py_setitem(index, classes[i], PythonObject(i))
    var found = labels.classes
    var remap = List[Int]()
    var bad = Python.list()
    for k in range(len(found)):  # driver-loop: over the batch's distinct labels (classes-sized)
        var at = py_get(index, found[k])
        if at is None:
            _ = py_call(bad.append, py_call(b.repr, found[k]))
            remap.append(0)
        else:
            remap.append(Int(py=at))
    if len(bad) > 0:
        var named = py_call(b.sorted, py_call(b.set, bad))
        raise Error("mojolearn: The target label(s) " + cls_str(named) + " in y do not exist in the initial classes " + cls_str(classes))
    # The batch named every class in order (the common case): its codes are
    # already the codes among classes_, no remap pass (p3-prep).
    var identity = len(remap) == len(classes)
    for k in range(len(remap)):  # driver-loop: over the batch's distinct labels (classes-sized)
        if remap[k] != k:
            identity = False
    if identity and classical_is_array(labels.codes, context) and String(labels.codes.dtype) == "<i4" and n > 0:
        return labels.codes
    var out = i32_array(n, context)
    var dest = I32P(unsafe_from_address=Int(py=classical_address(out)))
    _ = out  # keep the Python object alive through the native call above
    var src = I32P(unsafe_from_address=Int(py=classical_address(labels.codes)))
    for i in range(n):  # cpu-route: Python object labels of any kind, the explicit label input step
        dest[i] = Int32(remap[Int(src[i])])
        _ = out  # keep the Python object alive through the native call above
    # the codes array src reads belongs to `labels`: named again here so it outlives the loop (it was freed at
    # its last named use above, and src read freed words: "index 910002915 is out of bounds"; nn-integrate)
    _ = labels.codes
    return out


def _label_kind(typestr: String) -> Tuple[Int, Int]:
    """`_LABEL_KIND`: (lab_load kind, words per label), or (-1, 0)."""
    if typestr == "<f4":
        return (0, 1)
    if typestr == "<i4":
        return (1, 1)
    if typestr == "<u4":
        return (2, 1)
    if typestr == "<i8":
        return (3, 2)
    if typestr == "<f8":
        return (4, 2)
    return (-1, 0)


def partial_codes(model: PythonObject, fields: PythonObject, y: PythonObject, classes: PythonObject, n: Int,
                  context: PythonObject) raises -> PartialCodes:
    """`_partial_codes(est, y, classes, n, defer=True)`."""
    var nk = prep_keys()
    var b = cls_builtins()
    var first = py_get(fields, nk[].k_classes_) is None
    if first and classes is None:
        raise Error("mojolearn: classes must be passed on the first call to partial_fit.")
    if classes is not None:
        var given = classical_encode_labels(classes, context).classes
        if not first and Bool(py=py_call(b.list, given) != py_call(b.list, py_item(fields, nk[].k_classes_))):
            raise Error("mojolearn: `classes=" + cls_str(given) + "` is not the same as on last call to partial_fit, was: " + cls_str(py_item(fields, nk[].k_classes_)))
        if first:
            py_setitem(fields, nk[].k_classes_, given)
    # `_partial_dev`: every class an exact int or float with an exact,
    # strictly ascending float32 word, and y a numeric label vector of n labels
    var cl = py_call(b.list, py_item(fields, nk[].k_classes_))
    var K = len(cl)
    var dev = K > 0
    var cats = f32_array(K, context)
    var cp = F32P(unsafe_from_address=Int(py=classical_address(cats)))
    _ = cats  # keep the Python object alive through the native call above
    for k in range(K):  # driver-loop: over the classes (classes-sized)
        var t = py_type(cl[k])
        if t is not b.int and t is not b.float:
            dev = False
            break
        var v = Float64(py=py_call(b.float, cl[k]))
        cp[k] = Float32(v)
        _ = cats  # keep the Python object alive through the native call above
        if Float64(cp[k]) != v or (k > 0 and not (Float64(cp[k - 1]) < Float64(cp[k]))):
            dev = False
            break
        _ = cats  # keep the Python object alive through the native calls above
    if dev:
        var words = label_words(y, n, context)
        if words[0] >= 0:
            return PartialCodes(first, True, PythonObject(None), words[2], words[0], n, cats, y)
    return PartialCodes(first, False, partial_walk(fields, y, n, context), PythonObject(None), 0, n, cats, y)


def label_words(y: PythonObject, n: Int, context: PythonObject) raises -> Tuple[Int, Int, PythonObject]:
    """`_label_buffer(y)` when it holds exactly n labels: (kind, n, the raw
    int32 words), or (-1, 0, None) for the walk."""
    var nk = prep_keys()
    var b = cls_builtins()
    var cls = py_type(y)
    if cls is b.list or cls is b.tuple or cls is b.str or cls is b.bytes:
        return (-1, 0, PythonObject(None))
    var view = PythonObject(None)
    try:
        view = classical_view(y, context, "y")
    except:
        return (-1, 0, PythonObject(None))
    var spec = _label_kind(classical_dtype_str(view, context))
    var size = Int(py=view.size)
    if spec[0] < 0 or size == 0 or Int(py=view.ndim) < 1 or size != n:
        return (-1, 0, PythonObject(None))
    var largest = 0
    for k in range(Int(py=view.ndim)):  # driver-loop: over array rank metadata
        largest = max(largest, Int(py=view.shape[k]))
    if size != largest or not classical_c_order(view, context):
        return (-1, 0, PythonObject(None))
    var nw = size * spec[1]
    if nw > 1073741824:
        return (-1, 0, PythonObject(None))
    var store = native_array_empty("i")
    _ = py_call(store.frombytes, classical_c_bytes(view, "y", context))
    return (spec[0], size, classical_wrap_array(store, Python.tuple(nw), "<i4", "C", py_item(context, nk[].k_output_context)))


def partial_labels(pc: PartialCodes) raises -> ClsLabels:
    """UNPACK of `_partial_codes`' answer for the typed builders
    (bindings/prep_classify_prog_run.mojo `cls_partial_codes`): the word
    counts of its buffers, which the caller owns in that order
    (`own_partial_labels`)."""
    if not pc.dev:
        return ClsLabels(False, pc.n, Int(py=pc.codes.size), 0, 0, 0)
    return ClsLabels(True, pc.n, 0, Int(py=pc.words.size), pc.kind, Int(py=pc.cats.size))


def own_partial_labels(mut pr: ClsProg, pc: PartialCodes):
    """The owners of `cls_partial_codes`' buffer records, in record order:
    the walk's codes, or the raw label words and the class words."""
    if not pc.dev:
        pr.own(pc.codes)
        return
    pr.own(pc.words)
    pr.own(pc.cats)


def partial_refuse(pr: ClsProg, neg: Int, fields: PythonObject, pc: PartialCodes, context: PythonObject) raises:
    """The program's post-run refusal of a device-coded batch: the walk names
    the unknown labels."""
    if neg == NONE:
        return
    if pr.value(neg) > 0:
        _ = partial_walk(fields, pc.y, pc.n, context)
        raise Error("mojolearn: a target label in y does not exist in the initial classes")
