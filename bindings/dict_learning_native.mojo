# SPDX-License-Identifier: Apache-2.0
# The native owner of the dictionary-learning family of
# python/mojolearn/_expansion_decomp.py: DictionaryLearning,
# MiniBatchDictionaryLearning, SparseCoder, SparsePCA, MiniBatchSparsePCA
# (fit, fit_transform, transform, inverse_transform) and the public
# `sparse_encode`. Each public operation is one call into dict_learning_c,
# which checks the parameters, converts the inputs once and runs the steps
# the Python shells ran (`_dict_learning`, `_sparse_encode`, the mini-batch
# loop, the SparsePCA ridge projection) through bindings/decomp_kit_native.mojo:
# the same x_decomp entries with the same words in the same order, so the
# bits are unchanged. fit_transform keeps X (host and device copies) and the
# fitted matrices between its fit and transform steps: X is converted and
# uploaded once, and the dictionary, mean and components are not downloaded
# and re-uploaded between the two.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from std.memory import memcpy
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING, optimizer_python_error
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_arrays import classical_typed, classical_address, classical_host_f32
from bindings.decomp_kit_native import (
    DecompKit, DecompF32, decomp_int, decomp_float, decomp_seed, decomp_seed_word, decomp_mode,
    decomp_entries, decomp_input, decomp_shape, XD_SPARSE_ALGOS,
    XD_SUB, XD_ADD, XD_MUL, XD_DIV, XD_MAXS, XD_MINS, XD_SCALE, XD_SOFT, XD_SQ, XD_SQRT, XD_SELECT,
)


def _name(model: PythonObject) raises -> String:
    return String(py=py_type(model).__name__)


def _is_str(value: PythonObject, text: String) raises -> Bool:
    var b = Python.import_module("builtins")
    return Bool(py=py_call(b.isinstance, value, b.str)) and String(py=value) == text


def _algo_index(value: PythonObject) raises -> Int:
    # The position in `_SPARSE_ALGOS`, -1 when not one of them.
    if _is_str(value, "lasso_lars"):
        return 0
    if _is_str(value, "lasso_cd"):
        return 1
    if _is_str(value, "lars"):
        return 2
    if _is_str(value, "omp"):
        return 3
    if _is_str(value, "threshold"):
        return 4
    return -1


def _check_algos(fit_algorithm: PythonObject, transform_algorithm: PythonObject) raises:
    # `_check_sparse_algos`.
    if not _is_str(fit_algorithm, "lars") and not _is_str(fit_algorithm, "cd"):
        raise Error("fit_algorithm must be 'lars' or 'cd'")
    if _algo_index(transform_algorithm) < 0:
        raise Error("transform_algorithm must be one of " + String(XD_SPARSE_ALGOS))


def _fit_index(fit_algorithm: PythonObject) raises -> Int:
    # `_SPARSE_ALGOS.index("lasso_" + fit_algorithm)` of a checked name.
    return 0 if _is_str(fit_algorithm, "lars") else 1


def _is_a(model: PythonObject, cls: PythonObject) raises -> Bool:
    # The model's class MRO, read through type's own descriptor, holds cls.
    var b = Python.import_module("builtins")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, py_type(model))
    for k in range(len(mro)):  # driver-loop: over the model class's MRO
        if mro[k] is cls:
            return True
    return False


def _check_fitted(model: PythonObject, fields: PythonObject) raises:
    # `_Base._check`.
    if not py_contains(fields, "components_"):
        raise Error("RuntimeError: " + _name(model) + " is not fitted; call fit first")


def _matrix(mut kit: DecompKit, source: PythonObject, name: String, context: PythonObject) raises -> Int:
    var array = decomp_input(source, name, context)
    return kit.input(array, Int(py=array.shape[0]), Int(py=array.shape[1]))


def _row(mut kit: DecompKit, source: PythonObject, name: String, context: PythonObject) raises -> Int:
    # A fitted 1-D float32 vector (mean_) as a 1 x n matrix.
    var array = classical_host_f32(source, 1, name, context)
    return kit.input(array, 1, Int(py=array.shape[0]))


def _copy_out(array: PythonObject, rows: Int, cols: Int, context: PythonObject) raises -> PythonObject:
    # `_M.out()` of a converted input: a fresh (rows, cols) float32 Array.
    var result = classical_typed(Python.tuple(rows, cols), "<f4", context["output_context"])
    memcpy(dest=DecompF32(unsafe_from_address=Int(py=classical_address(result))),
           src=DecompF32(unsafe_from_address=Int(py=classical_address(array))), count=rows * cols)
    _ = result  # keep the Python object alive through the native call above
    return result


def _split_sign(mut kit: DecompKit, code: Int) raises -> Int:
    # [max(code, 0), -min(code, 0)] side by side.
    var positive = kit.ew(XD_MAXS, code, scalar=0.0)
    var low = kit.ew(XD_MINS, code, scalar=0.0)
    var negative = kit.ew(XD_SCALE, low, scalar=-1.0)
    return kit.hstack([positive, negative])


def _sparse_encode(mut kit: DecompKit, x: Int, d: Int, algorithm: PythonObject, alpha: PythonObject,
                   nnz: PythonObject, init: Int, max_iter: PythonObject, positive: PythonObject) raises -> Int:
    # `_expansion_decomp._sparse_encode`.
    var index = _algo_index(algorithm)
    if index < 0:
        raise Error("algorithm=" + String(py=py_call(Python.import_module("builtins").repr, algorithm))
                    + " is not carried; one of " + String(XD_SPARSE_ALGOS))
    var n = kit.nrows(x)
    var m = kit.ncols(x)
    var kc = kit.nrows(d)
    var count = 0
    var reg = Float64(1.0)
    if index == 2 or index == 3:
        if nnz is not None:
            count = decomp_int(nnz)
        else:
            # int(min(max(m / 10, 1), kc))
            count = Int(min(max(Float64(m) / 10.0, Float64(1.0)), Float64(kc)))
    elif alpha is not None:
        reg = decomp_float(alpha)
    if index == 2:
        var g = kit.mm(d, d, tb=True)
        var q = kit.mm(x, d, tb=True)
        return kit.lars_rows(g, q, m, count)
    var q = kit.mm(x, d, tb=True)
    if index == 4:
        var code = kit.ew(XD_SOFT, q, scalar=reg)
        if Bool(py=positive):
            return kit.ew(XD_MAXS, code, scalar=0.0)
        return code
    var g = kit.mm(d, d, tb=True)
    if index == 3:
        return kit.omp_rows(g, q, count)
    var w = kit.copy_host(init) if (index == 1 and init >= 0) else kit.zeros(n, kc)
    return kit.lasso_rows(g, q, w, reg, decomp_int(max_iter), 1e-8, decomp_int(positive))


def _dict_learning(mut kit: DecompKit, x: Int, nc: Int, alpha: Float64, max_iter: PythonObject, tol: PythonObject,
                   method: Int, seed: PythonObject, ci: Int, di: Int, positive_dict: Bool, positive_code: Bool,
                   method_max_iter: PythonObject) raises -> Tuple[Int, Int, PythonObject, Int]:
    # `_expansion_decomp._dict_learning`: (code, D, errors, n_iter).
    var code: Int
    var d: Int
    if ci >= 0 and di >= 0:
        code = ci
        d = di
    else:
        var usv = kit.thin_svd(x, min(kit.nrows(x), kit.ncols(x)), True)
        code = usv[0]
        var st = kit.transpose(usv[1])
        d = kit.ew(XD_MUL, usv[2], st)
    var r = kit.nrows(d)
    if nc <= r:
        code = kit.cols_of(code, 0, nc)
        d = kit.rows_of(d, 0, nc)
    else:
        var code_pad = kit.zeros(kit.nrows(code), nc - r)
        code = kit.hstack([code, code_pad])
        var d_pad = kit.zeros(nc - r, kit.ncols(d))
        d = kit.vstack([d, d_pad])
    # The loop (encode, atom update, cost, stopping test) is ONE binding call
    # on host copies of the code and the dictionary, written in place.
    code = kit.copy_host(code)
    d = kit.copy_host(d)
    var iterations = decomp_int(max_iter)
    var slots = max(iterations, 1)
    var errors_store = classical_typed(Python.tuple(slots), "<f8", kit.context)
    var params = Python.list(kit.nrows(x), kit.ncols(x), nc, method, iterations, decomp_int(method_max_iter),
                             Int(positive_dict), Int(positive_code), decomp_seed_word(seed), -1, -1)
    var floats = Python.list(alpha, decomp_float(tol))
    var ax = kit.address(x)
    var acode = kit.address(code)
    var ad = kit.address(d)
    var result = py_call(kit.entries["x_decomp_dict_learning"], ax, acode, ad, classical_address(errors_store),
                         params, floats)
    _ = errors_store  # keep the Python object alive through the native call above
    kit.wrote_host(code)
    kit.wrote_host(d)
    var done = Int(py=result[0])
    var recorded = min(Int(py=result[1]), slots)
    var errors = Python.list()
    var words = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(errors_store)))
    _ = errors_store  # keep the Python object alive through the native call above
    for i in range(recorded):  # driver-loop: over the fit's recorded iterations
        _ = py_call(errors.append, PythonObject(words[i]))
        _ = errors_store  # keep the Python object alive through the native call above
    _ = errors_store^
    return (code, d, errors, done)


def _minibatch_fit(mut kit: DecompKit, x: Int, nc: Int, dict_init: Int, p: PythonObject) raises -> Tuple[Int, Int, Int]:
    # MiniBatchDictionaryLearning.fit's steps on X (n x m): (D, n_steps_, steps per pass).
    var n = kit.nrows(x)
    var m = kit.ncols(x)
    var batch = min(decomp_int(p["batch_size"]), n)
    var seed = decomp_seed(p["random_state"])
    var d: Int
    if dict_init >= 0:
        d = dict_init
    else:
        var usv = kit.thin_svd(x, min(n, m), True)
        var st = kit.transpose(usv[1])
        d = kit.ew(XD_MUL, usv[2], st)
    if nc <= kit.nrows(d):
        d = kit.rows_of(d, 0, nc)
    else:
        var pad = kit.zeros(nc - kit.nrows(d), m)
        d = kit.vstack([d, pad])
    var xt = x
    if Bool(py=p["shuffle"]):
        var draws = kit.rand(1, n, decomp_seed_word(seed), 50, 0)
        var order = kit.order(draws)
        xt = kit.take_rows(x, order)
    if batch <= 0:
        raise Error("integer division or modulo by zero")
    var per_pass = -((-n) // batch)
    var n_steps = p["max_iter"] * PythonObject(per_pass)
    # The step loop is ONE binding call on a host copy of the dictionary.
    d = kit.copy_host(d)
    var patience = p["max_no_improvement"]
    var params = Python.list(n, m, nc, _fit_index(p["fit_algorithm"]), 0, decomp_int(p["transform_max_iter"]),
                             Int(Bool(py=p["positive_dict"])), Int(Bool(py=p["positive_code"])),
                             decomp_seed_word(seed), -1, -1 if patience is None else decomp_int(patience), batch)
    _ = py_call(params.append, n_steps)
    var floats = Python.list(decomp_float(p["alpha"]), decomp_float(p["tol"]))
    var ax = kit.address(xt)
    _ = xt  # keep the Python object alive through the native call above
    var ad = kit.address(d)
    var step = decomp_int(py_call(kit.entries["x_decomp_dict_minibatch"], ax, ad, params, floats)) - 1
    kit.wrote_host(d)
    return (d, step + 1, per_pass)


def _normalize(mut kit: DecompKit, c: Int) raises -> Int:
    # `_BaseSparsePCA._normalize`: each row over its norm (a zero row kept).
    var squares = kit.ew(XD_SQ, c)
    var sums = kit.rowsum(squares)
    var norms = kit.ew(XD_SQRT, sums)
    var one = kit.const(1.0)
    var safe = kit.ew(XD_SELECT, norms, norms, one, 0.0)
    return kit.ew(XD_DIV, c, safe)


def _spca_project(mut kit: DecompKit, x: Int, mean: Int, comps: Int, ridge_alpha: PythonObject) raises -> Int:
    # `_BaseSparsePCA.transform`: ridge regression of the centered rows.
    var xc = kit.ew(XD_SUB, x, mean)
    var gram = kit.mm(comps, comps, tb=True)
    var alpha = kit.const(Float32(decomp_float(ridge_alpha)))
    var g = kit.diag_add(gram, alpha)
    var factor = kit.lu(g)
    var rhs = kit.mm(comps, xc, tb=True)
    var solved = kit.lu_solve(factor[0], factor[1], rhs)
    return kit.transpose(solved)


def _coding_encode(mut kit: DecompKit, x: Int, d: Int, fields: PythonObject) raises -> Int:
    # `_SparseCoding._encode`.
    var alpha = py_get(fields, "transform_alpha")
    if alpha is None:
        alpha = py_get(fields, "alpha", PythonObject(1.0))
    var code = _sparse_encode(kit, x, d, fields["transform_algorithm"], alpha, fields["transform_n_nonzero_coefs"],
                              -1, fields["transform_max_iter"], fields["positive_code"])
    if Bool(py=fields["split_sign"]):
        return _split_sign(kit, code)
    return code


def _coding_inverse(mut kit: DecompKit, var c: Int, d: Int, fields: PythonObject) raises -> PythonObject:
    # `_SparseCoding.inverse_transform`: code @ components_.
    if Bool(py=fields["split_sign"]):
        var h = kit.nrows(d)
        var plus = kit.cols_of(c, 0, h)
        var minus = kit.cols_of(c, h, 2 * h)
        c = kit.ew(XD_SUB, plus, minus)
    return kit.publish(kit.mm(c, d))


# ------------------------------------------------------------ the estimators
def dictionary_learning_fit(model: PythonObject, args: PythonObject, project: Bool,
                            context: PythonObject) raises -> PythonObject:
    # DictionaryLearning.fit / fit_transform.
    var fields = py_fields(model)
    var mode = decomp_mode(fields["numeric_mode"], context)
    py_setitem(fields, "numeric_mode_", mode)
    _check_algos(fields["fit_algorithm"], fields["transform_algorithm"])
    if py_get(fields, "callback") is not None:
        raise Error("NotImplementedError: callback is not carried")
    var kit = DecompKit(decomp_entries(mode, context), context["output_context"])
    try:
        var x = _matrix(kit, args[0], "X", context)
        var requested = fields["n_components"]
        var nc = kit.ncols(x) if requested is None else decomp_int(requested)
        var ci = -1 if fields["code_init"] is None else _matrix(kit, fields["code_init"], "code_init", context)
        var di = -1 if fields["dict_init"] is None else _matrix(kit, fields["dict_init"], "dict_init", context)
        var fit = _dict_learning(kit, x, nc, decomp_float(fields["alpha"]), fields["max_iter"], fields["tol"],
                                 _fit_index(fields["fit_algorithm"]), decomp_seed(fields["random_state"]), ci, di,
                                 Bool(py=fields["positive_dict"]), Bool(py=fields["positive_code"]),
                                 fields["transform_max_iter"])
        var d = fit[1]
        py_setitem(fields, "components_", kit.publish(d))
        py_setitem(fields, "error_", fit[2])
        py_setitem(fields, "n_iter_", PythonObject(fit[3]))
        py_setitem(fields, "n_features_in_", PythonObject(kit.ncols(x)))
        if not project:
            return model
        var code = fit[0]
        if Bool(py=fields["split_sign"]):
            code = _split_sign(kit, code)
        return kit.publish(code)
    finally:
        kit.close()


def minibatch_dictionary_fit(model: PythonObject, args: PythonObject, project: Bool,
                             context: PythonObject) raises -> PythonObject:
    # MiniBatchDictionaryLearning.fit / fit_transform (fit, then the transform
    # of the X and the dictionary the fit already holds).
    var fields = py_fields(model)
    var mode = decomp_mode(fields["numeric_mode"], context)
    py_setitem(fields, "numeric_mode_", mode)
    _check_algos(fields["fit_algorithm"], fields["transform_algorithm"])
    if py_get(fields, "callback") is not None:
        raise Error("NotImplementedError: callback is not carried")
    var kit = DecompKit(decomp_entries(mode, context), context["output_context"])
    try:
        var x = _matrix(kit, args[0], "X", context)
        var requested = fields["n_components"]
        var nc = kit.ncols(x) if requested is None else decomp_int(requested)
        var init = -1
        if fields["dict_init"] is not None:
            # (after the batch size and the seed are read, as the Python fit did)
            _ = min(decomp_int(fields["batch_size"]), kit.nrows(x))
            _ = decomp_seed(fields["random_state"])
            init = _matrix(kit, fields["dict_init"], "dict_init", context)
        var fit = _minibatch_fit(kit, x, nc, init, fields)
        var n_steps = fit[1]
        py_setitem(fields, "n_steps_", PythonObject(n_steps))
        py_setitem(fields, "n_iter_", PythonObject(-((-n_steps) // fit[2])))
        py_setitem(fields, "components_", kit.publish(fit[0]))
        py_setitem(fields, "n_features_in_", PythonObject(kit.ncols(x)))
        if not project:
            return model
        return kit.publish(_coding_encode(kit, x, fit[0], fields))
    finally:
        kit.close()


def coding_transform(model: PythonObject, args: PythonObject, inverse: Bool, context: PythonObject) raises -> PythonObject:
    # DictionaryLearning / MiniBatchDictionaryLearning transform and inverse_transform.
    var fields = py_fields(model)
    _check_fitted(model, fields)
    var kit = DecompKit(decomp_entries(py_get(fields, "numeric_mode_"), context), context["output_context"])
    try:
        if inverse:
            var c = _matrix(kit, args[0], "code", context)
            var d = _matrix(kit, fields["components_"], "components_", context)
            return _coding_inverse(kit, c, d, fields)
        var x = _matrix(kit, args[0], "X", context)
        var d = _matrix(kit, fields["components_"], "components_", context)
        return kit.publish(_coding_encode(kit, x, d, fields))
    finally:
        kit.close()


def _coder_dictionary(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    # SparseCoder's parameter check, then its dictionary as float32.
    var transform = fields["transform_algorithm"]
    if _algo_index(transform) < 0:
        raise Error("transform_algorithm must be one of " + String(XD_SPARSE_ALGOS))
    return decomp_input(fields["dictionary"], "dictionary", context)


def sparse_coder(model: PythonObject, operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    # SparseCoder fit / transform (fit_transform) / inverse_transform.
    var fields = py_fields(model)
    if operation == "coder_fit":
        var shape = decomp_shape(args[0], "X", context)
        var dictionary = _coder_dictionary(fields, context)
        var rows = Int(py=dictionary.shape[0])
        var cols = Int(py=dictionary.shape[1])
        if shape[1] != cols:
            raise Error("Dictionary and X have different numbers of features:dictionary.shape: ("
                        + String(rows) + ", " + String(cols) + ") X.shape: (" + String(shape[0]) + ", "
                        + String(shape[1]) + ")")
        py_setitem(fields, "numeric_mode_", decomp_mode(fields["numeric_mode"], context))
        py_setitem(fields, "components_", _copy_out(dictionary, rows, cols, context))
        py_setitem(fields, "n_components_", PythonObject(rows))
        py_setitem(fields, "n_features_in_", PythonObject(cols))
        return model
    var dictionary = _coder_dictionary(fields, context)
    var kit = DecompKit(decomp_entries(fields["numeric_mode"], context), context["output_context"])
    try:
        var d = kit.input(dictionary, Int(py=dictionary.shape[0]), Int(py=dictionary.shape[1]))
        if operation == "coder_inverse":
            var c = _matrix(kit, args[0], "code", context)
            return _coding_inverse(kit, c, d, fields)
        var x = _matrix(kit, args[0], "X", context)
        return kit.publish(_coding_encode(kit, x, d, fields))
    finally:
        kit.close()


def sparse_pca_fit(model: PythonObject, args: PythonObject, project: Bool, context: PythonObject) raises -> PythonObject:
    # SparsePCA / MiniBatchSparsePCA fit and fit_transform (the ridge
    # projection of the X, mean and components the fit holds).
    var fields = py_fields(model)
    var mode = decomp_mode(fields["numeric_mode"], context)
    py_setitem(fields, "numeric_mode_", mode)
    var minibatch = _is_a(model, context["minibatch_sparse_pca_class"])
    if not minibatch and not _is_str(fields["method"], "lars") and not _is_str(fields["method"], "cd"):
        raise Error("method must be 'lars' or 'cd'")
    var kit = DecompKit(decomp_entries(mode, context), context["output_context"])
    try:
        var x = _matrix(kit, args[0], "X", context)
        var cols = kit.ncols(x)
        var requested = fields["n_components"]
        var nc = cols if requested is None else decomp_int(requested)
        var mean = kit.colmean(x)
        var xc = kit.ew(XD_SUB, x, mean)
        var comps: Int
        if not minibatch:
            var ci = -1
            if fields["V_init"] is not None:
                ci = kit.transpose(_matrix(kit, fields["V_init"], "V_init", context))
            var di = -1
            if fields["U_init"] is not None:
                di = kit.transpose(_matrix(kit, fields["U_init"], "U_init", context))
            var xct = kit.transpose(xc)
            var fit = _dict_learning(kit, xct, nc, decomp_float(fields["alpha"]), fields["max_iter"], fields["tol"],
                                     _fit_index(fields["method"]), decomp_seed(fields["random_state"]), ci, di,
                                     False, False, PythonObject(1000))
            var code_t = kit.transpose(fit[0])
            comps = _normalize(kit, code_t)
            py_setitem(fields, "components_", kit.publish(comps))
            py_setitem(fields, "n_components_", PythonObject(kit.nrows(comps)))
            py_setitem(fields, "error_", fit[2])
            py_setitem(fields, "n_iter_", PythonObject(fit[3]))
        else:
            # MiniBatchDictionaryLearning(n_components=nc, fit_algorithm=method,
            # transform_algorithm="lasso_lars", transform_alpha=alpha, ...) on X^T.
            _check_algos(fields["method"], PythonObject("lasso_lars"))
            var xt = kit.transpose(xc)
            var p = Python.dict()
            py_setitem(p, "alpha", fields["alpha"])
            py_setitem(p, "max_iter", fields["max_iter"])
            py_setitem(p, "batch_size", fields["batch_size"])
            py_setitem(p, "shuffle", fields["shuffle"])
            py_setitem(p, "fit_algorithm", fields["method"])
            py_setitem(p, "random_state", fields["random_state"])
            py_setitem(p, "positive_dict", PythonObject(False))
            py_setitem(p, "positive_code", PythonObject(False))
            py_setitem(p, "transform_max_iter", PythonObject(1000))
            py_setitem(p, "tol", fields["tol"])
            py_setitem(p, "max_no_improvement", fields["max_no_improvement"])
            var fit = _minibatch_fit(kit, xt, nc, -1, p)
            var code = _sparse_encode(kit, xt, fit[0], PythonObject("lasso_lars"), fields["alpha"], PythonObject(None),
                                      -1, PythonObject(1000), PythonObject(False))
            var code_t = kit.transpose(code)
            comps = _normalize(kit, code_t)
            py_setitem(fields, "components_", kit.publish(comps))
            py_setitem(fields, "n_components_", PythonObject(kit.nrows(comps)))
            py_setitem(fields, "n_iter_", PythonObject(-((-fit[1]) // fit[2])))
        py_setitem(fields, "mean_", kit.out_shape(mean, Python.tuple(cols)))
        py_setitem(fields, "n_features_in_", PythonObject(cols))
        if not project:
            return model
        return kit.publish(_spca_project(kit, x, mean, comps, fields["ridge_alpha"]))
    finally:
        kit.close()


def sparse_pca_transform(model: PythonObject, args: PythonObject, inverse: Bool, context: PythonObject) raises -> PythonObject:
    # `_BaseSparsePCA.transform` / `inverse_transform`.
    var fields = py_fields(model)
    _check_fitted(model, fields)
    var kit = DecompKit(decomp_entries(py_get(fields, "numeric_mode_"), context), context["output_context"])
    try:
        var x = _matrix(kit, args[0], "X", context)
        var mean = _row(kit, fields["mean_"], "mean_", context)
        var comps = _matrix(kit, fields["components_"], "components_", context)
        if inverse:
            var product = kit.mm(x, comps)
            return kit.publish(kit.ew(XD_ADD, product, mean))
        return kit.publish(_spca_project(kit, x, mean, comps, fields["ridge_alpha"]))
    finally:
        kit.close()


def sparse_encode_call(args: PythonObject, context: PythonObject) raises -> PythonObject:
    # `sparse_encode(X, dictionary, algorithm, n_nonzero_coefs, alpha, init,
    # max_iter, positive, numeric_mode)`.
    var kit = DecompKit(decomp_entries(args[8], context), context["output_context"])
    try:
        var init = -1 if args[5] is None else _matrix(kit, args[5], "init", context)
        var x = _matrix(kit, args[0], "X", context)
        var d = _matrix(kit, args[1], "dictionary", context)
        return kit.publish(_sparse_encode(kit, x, d, args[2], args[4], args[3], init, args[6], args[7]))
    finally:
        kit.close()


def dict_learning_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        if len(call) != 4:
            raise Error("TypeError: dict_learning_native_call requires model, operation, arguments and connected context")
        var b = Python.import_module("builtins")
        if py_type(call[1]) is not b.str or py_type(call[2]) is not b.tuple or py_type(call[3]) is not b.dict:
            raise Error("TypeError: native dictionary learning requires an operation, an argument tuple and its context")
        var model = call[0]
        var operation = String(py=call[1])
        var args = call[2]
        var context = call[3]
        classical_connect_numpy(context)
        var result = PythonObject(None)
        if operation == "dl_fit" or operation == "dl_fit_transform":
            result = dictionary_learning_fit(model, args, operation == "dl_fit_transform", context)
        elif operation == "mbdl_fit" or operation == "mbdl_fit_transform":
            result = minibatch_dictionary_fit(model, args, operation == "mbdl_fit_transform", context)
        elif operation == "coding_transform" or operation == "coding_inverse":
            result = coding_transform(model, args, operation == "coding_inverse", context)
        elif operation == "coder_fit" or operation == "coder_transform" or operation == "coder_inverse":
            result = sparse_coder(model, operation, args, context)
        elif operation == "spca_fit" or operation == "spca_fit_transform":
            result = sparse_pca_fit(model, args, operation == "spca_fit_transform", context)
        elif operation == "spca_transform" or operation == "spca_inverse":
            result = sparse_pca_transform(model, args, operation == "spca_inverse", context)
        elif operation == "sparse_encode":
            result = sparse_encode_call(args, context)
        else:
            raise Error("unknown native dictionary learning operation")
        return result.steal_data()
    except error:
        if String(error) == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error^)
        if String(error).startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if String(error).startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        if String(error).startswith("AttributeError:"):
            return raise_python_exception(error_without_class(error, "AttributeError"), ExceptionType("PyExc_AttributeError"))
        if String(error).startswith("ImportError:"):
            return raise_python_exception(error_without_class(error, "ImportError"), ExceptionType("PyExc_ImportError"))
        if String(error).startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
