# SPDX-License-Identifier: Apache-2.0
# The native owner of the dictionary-learning family of
# python/mojolearn/_expansion_decomp.py: DictionaryLearning,
# MiniBatchDictionaryLearning, SparseCoder, SparsePCA, MiniBatchSparsePCA
# (fit, fit_transform, transform, inverse_transform) and the public
# `sparse_encode`. Each public operation is one call into dict_learning_c,
# which checks the parameters, converts the inputs once and runs the steps
# the Python shells ran (`_dict_learning`, `_sparse_encode`, the mini-batch
# loop, the SparsePCA ridge projection) over the typed kit (bindings/dict_learning_run.mojo):
# the same x_decomp entries with the same words in the same order, so the
# bits are unchanged. fit_transform keeps X (host and device copies) and the
# fitted matrices between its fit and transform steps: X is converted and
# uploaded once, and the dictionary, mean and components are not downloaded
# and re-uploaded between the two.
from std.python import Python, PythonObject
from bindings.cpython_table import cpython_table
from bindings.native_loaded_module import native_loaded_module
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from std.memory import unsafe_memcpy
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING, optimizer_python_error
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_arrays import classical_typed, classical_address
from bindings.admitted_input import admit_x
from bindings.decomp_public_lifecycle import decomp_public_begin, decomp_public_finish, decomp_coder_width
from x_decomp.decomp_public_lifecycle_run import decomp_public_method
from bindings.public_lifecycle_owner import lifecycle_end
from bindings.decomp_kit_native import (
    DecompDoor, DecompF32, decomp_int, decomp_float, decomp_seed, decomp_seed_word, decomp_mode,
    decomp_entries, decomp_input, decomp_shape, XD_SPARSE_ALGOS, decomp_kit_unpack, decomp_kit_publish,
    decomp_kit_out_shape, decomp_door_borrow, decomp_door_input, decomp_door_resident_hit,
    decomp_door_resident_input, decomp_door_resident_derived, decomp_door_resident_keep,
)
from bindings.decomp_kit_run import DKit
from bindings.dict_learning_run import (
    EncodeWords, Gram, DictWords, DictFit, MiniBatchWords, SpcaFit, dl_split_sign, dl_sparse_encode, dl_encode_run,
    dl_inverse_run, dl_dict_learning, dl_minibatch_fit, dl_spca_project, dl_spca_inverse, dl_spca_fit_run,
)
from bindings.py_lists import py_list_f64
from std.python._cpython import GILReleased


def _name(model: PythonObject) raises -> String:
    return String(py=py_type(model).__name__)


def _str_instance(value: PythonObject) raises -> Bool:
    # `isinstance(value, str)` through CPython's own native entry (the same
    # protocol, without a Python-level builtins.isinstance call).
    # An exact str (every setting a caller writes) answers by type identity,
    # with no symbol lookup or call (p3 rank 412).
    var text_type = native_loaded_module("builtins").str
    if py_type(value) is text_type:
        return True
    ref cpy = Python().cpython()
    var found = cpy.lib.get_function[Int32]("PyObject_IsInstance")(value._obj_ptr, text_type._obj_ptr)
    if found < 0:
        raise cpy.unsafe_get_error()
    return found == 1


def _is_str(value: PythonObject, text: String) raises -> Bool:
    return _str_instance(value) and String(py=value) == text


def _algo_index(value: PythonObject) raises -> Int:
    # The position in `_SPARSE_ALGOS`, -1 when not one of them. One instance
    # check and one string read for the five names.
    if not _str_instance(value):
        return -1
    var text = String(py=value)
    if text == "lasso_lars":
        return 0
    if text == "lasso_cd":
        return 1
    if text == "lars":
        return 2
    if text == "omp":
        return 3
    if text == "threshold":
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
    var b = native_loaded_module("builtins")
    if py_type(cls) is b.type:
        # CPython's own subtype test walks the same tp_mro by identity in one
        # native call (no MRO tuple fetch or per-row Python indexing).
        # Resolved once per process in the shared table (pass 2), not by name per call.
        var t = cpython_table()
        return t[].PyType_IsSubtype(py_type(model)._obj_ptr, cls._obj_ptr) != 0
    var mro = py_call(b.type.__dict__["__mro__"].__get__, py_type(model))
    for k in range(len(mro)):  # driver-loop: over the model class's MRO
        if mro[k] is cls:
            return True
    return False


def _check_fitted(model: PythonObject, fields: PythonObject) raises:
    # `_Base._check`.
    if not py_contains(fields, "components_"):
        raise Error("RuntimeError: " + _name(model) + " is not fitted; call fit first")


def _matrix(mut door: DecompDoor, mut kit: DKit, source: PythonObject, name: String,
            context: PythonObject) raises -> Int:
    return decomp_door_input(door, kit, source, name, context)


def _fitted_matrix(mut door: DecompDoor, mut kit: DKit, source: PythonObject, name: String,
                   context: PythonObject) raises -> Int:
    # UNPACK a fitted operand: on a query its device copy is kept across
    # calls (the model's record), else borrowed as _matrix.
    var hit = decomp_door_resident_hit(door, kit, source, name)  # frozen and resident: no re-admission
    if hit >= 0:
        return hit
    var array = decomp_input(source, name, context)
    var rows = Int(py=array.shape[0])
    var cols = Int(py=array.shape[1])
    var held = decomp_door_resident_input(door, kit, source, name, array, Int(py=classical_address(array)), array,
                                          rows, cols)
    if held >= 0:
        return held
    return decomp_door_borrow(door, kit, array, rows, cols)


def _fitted_row(mut door: DecompDoor, mut kit: DKit, source: PythonObject, name: String,
                context: PythonObject) raises -> Int:
    # A fitted 1-D float32 vector as a 1 x n matrix, resident as _fitted_matrix.
    var hit = decomp_door_resident_hit(door, kit, source, name)  # frozen and resident: no re-admission
    if hit >= 0:
        return hit
    var array = admit_x(source, 1, name, context)
    var cols = Int(py=array.shape[0])
    var held = decomp_door_resident_input(door, kit, source, name, array, Int(py=classical_address(array)), array,
                                          1, cols)
    if held >= 0:
        return held
    return decomp_door_borrow(door, kit, array, 1, cols)


def _copy_out(array: PythonObject, rows: Int, cols: Int, context: PythonObject) raises -> PythonObject:
    # `_M.out()` of a converted input: a fresh (rows, cols) float32 Array.
    var result = classical_typed(Python.tuple(rows, cols), "<f4", context["output_context"])
    unsafe_memcpy(dest=DecompF32(unsafe_from_address=Int(py=classical_address(result))),
           src=DecompF32(unsafe_from_address=Int(py=classical_address(array))), count=rows * cols)
    _ = result  # keep the Python object alive through the native call above
    return result


comptime _GRAM_NAME = "components_#gram"


def _gram(mut door: DecompDoor, mut kit: DKit, gram_source: PythonObject, words: EncodeWords) raises -> Gram:
    # UNPACK the Gram G = D D^T of a query's fitted components (N339, N333):
    # served from the model's resident record while `gram_source` (the
    # fitted components_ Array D was read from; None on every other call)
    # is the same frozen object at the same generation. -1 on a miss, a
    # host binding, or an algorithm that needs no Gram (threshold).
    if gram_source is None or words.algorithm == 4:
        return Gram(-1)
    return Gram(decomp_door_resident_derived(door, kit, gram_source, _GRAM_NAME))


def _keep_gram(mut door: DecompDoor, mut kit: DKit, gram_source: PythonObject, gram: Gram) raises:
    # PACK: a Gram RUN computed for a query goes to the model's record.
    if gram_source is not None and gram.computed >= 0:
        _ = decomp_door_resident_keep(door, kit, gram_source, _GRAM_NAME, gram.computed)


def _encode_words(m: Int, kc: Int, algorithm: PythonObject, alpha: PythonObject, nnz: PythonObject,
                  max_iter: PythonObject, positive: PythonObject, split_sign: Bool) raises -> EncodeWords:
    # UNPACK `_sparse_encode`'s options, each read only where its route reads
    # it (`m` the code's input width, `kc` the dictionary's atom count).
    var index = _algo_index(algorithm)
    if index < 0:
        raise Error("algorithm=" + String(py=py_call(native_loaded_module("builtins").repr, algorithm))
                    + " is not carried; one of " + String(XD_SPARSE_ALGOS))
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
    var iterations = 0
    var positive_word = 0
    var positive_flag = False
    if index == 4:
        positive_flag = Bool(py=positive)
    elif index == 0 or index == 1:
        iterations = decomp_int(max_iter)
        positive_word = decomp_int(positive)
    return EncodeWords(index, count, reg, iterations, positive_word, positive_flag, split_sign)


def _coding_words(m: Int, kc: Int, fields: PythonObject) raises -> EncodeWords:
    # UNPACK `_SparseCoding._encode`'s options.
    var alpha = py_get(fields, "transform_alpha")
    if alpha is None:
        alpha = py_get(fields, "alpha", PythonObject(1.0))
    var words = _encode_words(m, kc, fields["transform_algorithm"], alpha, fields["transform_n_nonzero_coefs"],
                              fields["transform_max_iter"], fields["positive_code"], False)
    words.split_sign = Bool(py=fields["split_sign"])
    return words^


def _dict_words(nc: Int, alpha: PythonObject, max_iter: PythonObject, tol: PythonObject, method: Int,
                random_state: PythonObject, positive_dict: Bool, positive_code: Bool,
                method_max_iter: PythonObject) raises -> DictWords:
    # UNPACK `_dict_learning`'s options (alpha and the seed's refusal first).
    var a = decomp_float(alpha)
    var seed = decomp_seed(random_state)
    return DictWords(nc, a, decomp_int(max_iter), decomp_float(tol), method, decomp_seed_word(seed), positive_dict,
                     positive_code, decomp_int(method_max_iter))


def _minibatch_words(n: Int, batch_size: PythonObject, random_state: PythonObject, shuffle: PythonObject,
                     max_iter: PythonObject, patience: PythonObject, fit_algorithm: PythonObject,
                     transform_max_iter: PythonObject, positive_dict: PythonObject, positive_code: PythonObject,
                     alpha: PythonObject, tol: PythonObject, mut per_pass: Int) raises -> MiniBatchWords:
    # UNPACK the mini-batch loop's options, in the order the Python fit read them.
    var batch = min(decomp_int(batch_size), n)
    var seed = decomp_seed(random_state)
    var mix = Bool(py=shuffle)
    if batch <= 0:
        raise Error("integer division or modulo by zero")
    per_pass = -((-n) // batch)
    var steps = Int(py=max_iter * PythonObject(per_pass))
    var limit = -1 if patience is None else decomp_int(patience)
    return MiniBatchWords(batch, decomp_seed_word(seed), mix, steps, limit, _fit_index(fit_algorithm),
                          decomp_int(transform_max_iter), Bool(py=positive_dict), Bool(py=positive_code),
                          decomp_float(alpha), decomp_float(tol))


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
    var entries = decomp_entries(mode, context)
    var door = DecompDoor(entries, context["output_context"])
    var kit = decomp_kit_unpack(entries)
    try:
        var x = _matrix(door, kit, args[0], "X", context)
        var requested = fields["n_components"]
        var nc = kit.ncols(x) if requested is None else decomp_int(requested)
        var ci = -1 if fields["code_init"] is None else _matrix(door, kit, fields["code_init"], "code_init", context)
        var di = -1 if fields["dict_init"] is None else _matrix(door, kit, fields["dict_init"], "dict_init", context)
        var method = _fit_index(fields["fit_algorithm"])
        var words = _dict_words(nc, fields["alpha"], fields["max_iter"], fields["tol"], method, fields["random_state"],
                                Bool(py=fields["positive_dict"]), Bool(py=fields["positive_code"]),
                                fields["transform_max_iter"])
        var split = project and Bool(py=fields["split_sign"])
        var fit = DictFit()
        var code = -1
        var py = Python()
        with GILReleased(py):
            fit = dl_dict_learning(kit, x, words, ci, di)
            if project:
                code = dl_split_sign(kit, fit.code) if split else fit.code
        py_setitem(fields, "components_", decomp_kit_publish(door, kit, fit.d))
        py_setitem(fields, "error_", py_list_f64(fit.errors))
        py_setitem(fields, "n_iter_", PythonObject(fit.done))
        py_setitem(fields, "n_features_in_", PythonObject(kit.ncols(x)))
        if not project:
            return model
        return decomp_kit_publish(door, kit, code)
    finally:
        kit.close()
        door.keep_alive()  # the borrowed owners outlive every RUN read


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
    var entries = decomp_entries(mode, context)
    var door = DecompDoor(entries, context["output_context"])
    var kit = decomp_kit_unpack(entries)
    try:
        var x = _matrix(door, kit, args[0], "X", context)
        var requested = fields["n_components"]
        var nc = kit.ncols(x) if requested is None else decomp_int(requested)
        var init = -1
        if fields["dict_init"] is not None:
            # (after the batch size and the seed are read, as the Python fit did)
            _ = min(decomp_int(fields["batch_size"]), kit.nrows(x))
            _ = decomp_seed(fields["random_state"])
            init = _matrix(door, kit, fields["dict_init"], "dict_init", context)
        var per_pass = 1
        var words = _minibatch_words(kit.nrows(x), fields["batch_size"], fields["random_state"], fields["shuffle"],
                                     fields["max_iter"], fields["max_no_improvement"], fields["fit_algorithm"],
                                     fields["transform_max_iter"], fields["positive_dict"], fields["positive_code"],
                                     fields["alpha"], fields["tol"], per_pass)
        var encode = EncodeWords(0, 0, 1.0, 0, 0, False, False)
        if project:
            # X against the nc atoms the fit keeps
            encode = _coding_words(kit.ncols(x), nc, fields)
        var d = -1
        var n_steps = 0
        var code = -1
        var py = Python()
        with GILReleased(py):
            var fit = dl_minibatch_fit(kit, x, nc, init, words)
            d = fit[0]
            n_steps = fit[1]
            if project:
                var gram = Gram(-1)
                code = dl_encode_run(kit, x, d, encode, gram)
        py_setitem(fields, "n_steps_", PythonObject(n_steps))
        py_setitem(fields, "n_iter_", PythonObject(-((-n_steps) // per_pass)))
        py_setitem(fields, "components_", decomp_kit_publish(door, kit, d))
        py_setitem(fields, "n_features_in_", PythonObject(kit.ncols(x)))
        if not project:
            return model
        return decomp_kit_publish(door, kit, code)
    finally:
        kit.close()
        door.keep_alive()  # the borrowed owners outlive every RUN read


def coding_transform(model: PythonObject, args: PythonObject, inverse: Bool, context: PythonObject) raises -> PythonObject:
    # DictionaryLearning / MiniBatchDictionaryLearning transform and inverse_transform.
    var fields = py_fields(model)
    _check_fitted(model, fields)
    var entries = decomp_entries(py_get(fields, "numeric_mode_"), context)
    var door = DecompDoor(entries, context["output_context"])
    # A query: the fitted dictionary stays on the device across calls.
    door.owner = model
    var kit = decomp_kit_unpack(entries)
    try:
        var py = Python()
        if inverse:
            var c = _matrix(door, kit, args[0], "code", context)
            var d = _fitted_matrix(door, kit, fields["components_"], "components_", context)
            var split = Bool(py=fields["split_sign"])
            var out = -1
            with GILReleased(py):
                out = dl_inverse_run(kit, c, d, split)
            return decomp_kit_publish(door, kit, out)
        var x = _matrix(door, kit, args[0], "X", context)
        var source = fields["components_"]
        var d = _fitted_matrix(door, kit, source, "components_", context)
        var words = _coding_words(kit.ncols(x), kit.nrows(d), fields)
        var gram = _gram(door, kit, source, words)
        var code = -1
        with GILReleased(py):
            code = dl_encode_run(kit, x, d, words, gram)
        _keep_gram(door, kit, source, gram)
        return decomp_kit_publish(door, kit, code)
    finally:
        kit.close()
        door.keep_alive()  # the borrowed owners outlive every RUN read


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
    var entries = decomp_entries(fields["numeric_mode"], context)
    var door = DecompDoor(entries, context["output_context"])
    var kit = decomp_kit_unpack(entries)
    try:
        var d = decomp_door_borrow(door, kit, dictionary, Int(py=dictionary.shape[0]), Int(py=dictionary.shape[1]))
        var py = Python()
        var out = -1
        if operation == "coder_inverse":
            var c = _matrix(door, kit, args[0], "code", context)
            var split = Bool(py=fields["split_sign"])
            with GILReleased(py):
                out = dl_inverse_run(kit, c, d, split)
            return decomp_kit_publish(door, kit, out)
        var x = _matrix(door, kit, args[0], "X", context)
        var words = _coding_words(kit.ncols(x), kit.nrows(d), fields)
        with GILReleased(py):
            var gram = Gram(-1)
            out = dl_encode_run(kit, x, d, words, gram)
        return decomp_kit_publish(door, kit, out)
    finally:
        kit.close()
        door.keep_alive()  # the borrowed owners outlive every RUN read


def sparse_pca_fit(model: PythonObject, args: PythonObject, project: Bool, context: PythonObject) raises -> PythonObject:
    # SparsePCA / MiniBatchSparsePCA fit and fit_transform (the ridge
    # projection of the X, mean and components the fit holds).
    var fields = py_fields(model)
    var mode = decomp_mode(fields["numeric_mode"], context)
    py_setitem(fields, "numeric_mode_", mode)
    var minibatch = _is_a(model, context["minibatch_sparse_pca_class"])
    if not minibatch and not _is_str(fields["method"], "lars") and not _is_str(fields["method"], "cd"):
        raise Error("method must be 'lars' or 'cd'")
    var entries = decomp_entries(mode, context)
    var door = DecompDoor(entries, context["output_context"])
    var kit = decomp_kit_unpack(entries)
    try:
        # UNPACK: X, the inits and every option, once.
        var x = _matrix(door, kit, args[0], "X", context)
        var cols = kit.ncols(x)
        var requested = fields["n_components"]
        var nc = cols if requested is None else decomp_int(requested)
        var vi = -1
        var ui = -1
        var dict_words = DictWords(nc, 0.0, 0, 0.0, 0, 0, False, False, 0)
        var mb_words = MiniBatchWords(1, 0, False, 0, -1, 0, 0, False, False, 0.0, 0.0)
        var encode = EncodeWords(0, 0, 1.0, 0, 0, False, False)
        var per_pass = 1
        if not minibatch:
            if fields["V_init"] is not None:
                vi = _matrix(door, kit, fields["V_init"], "V_init", context)
            if fields["U_init"] is not None:
                ui = _matrix(door, kit, fields["U_init"], "U_init", context)
            dict_words = _dict_words(nc, fields["alpha"], fields["max_iter"], fields["tol"], _fit_index(fields["method"]),
                                     fields["random_state"], False, False, PythonObject(1000))
        else:
            # MiniBatchDictionaryLearning(n_components=nc, fit_algorithm=method,
            # transform_algorithm="lasso_lars", transform_alpha=alpha, ...) on X^T.
            _check_algos(fields["method"], PythonObject("lasso_lars"))
            mb_words = _minibatch_words(cols, fields["batch_size"], fields["random_state"], fields["shuffle"],
                                        fields["max_iter"], fields["max_no_improvement"], fields["method"],
                                        PythonObject(1000), PythonObject(False), PythonObject(False), fields["alpha"],
                                        fields["tol"], per_pass)
            # lasso_lars on X^T (rows: the features) against the nc atoms
            encode = _encode_words(kit.nrows(x), nc, PythonObject("lasso_lars"), fields["alpha"],
                                   PythonObject(None), PythonObject(1000), PythonObject(False), False)
        var ridge = Float32(decomp_float(fields["ridge_alpha"])) if project else Float32(0.0)
        # RUN: the centering, the fit and the normalization (and the projection).
        var fit = SpcaFit()
        var py = Python()
        with GILReleased(py):
            fit = dl_spca_fit_run(kit, x, vi, ui, minibatch, dict_words, mb_words, encode, project, ridge)
        # PACK: the fitted fields, once.
        py_setitem(fields, "components_", decomp_kit_publish(door, kit, fit.comps))
        py_setitem(fields, "n_components_", PythonObject(kit.nrows(fit.comps)))
        if not minibatch:
            py_setitem(fields, "error_", py_list_f64(fit.errors))
            py_setitem(fields, "n_iter_", PythonObject(fit.done))
        else:
            py_setitem(fields, "n_iter_", PythonObject(-((-fit.steps) // per_pass)))
        py_setitem(fields, "mean_", decomp_kit_out_shape(door, kit, fit.mean, Python.tuple(cols)))
        py_setitem(fields, "n_features_in_", PythonObject(cols))
        if not project:
            return model
        return decomp_kit_publish(door, kit, fit.projected)
    finally:
        kit.close()
        door.keep_alive()  # the borrowed owners outlive every RUN read


def sparse_pca_transform(model: PythonObject, args: PythonObject, inverse: Bool, context: PythonObject) raises -> PythonObject:
    # `_BaseSparsePCA.transform` / `inverse_transform`.
    var fields = py_fields(model)
    _check_fitted(model, fields)
    var entries = decomp_entries(py_get(fields, "numeric_mode_"), context)
    var door = DecompDoor(entries, context["output_context"])
    # A query: the fitted mean and components stay on the device across calls.
    door.owner = model
    var kit = decomp_kit_unpack(entries)
    try:
        var x = _matrix(door, kit, args[0], "X", context)
        var mean = _fitted_row(door, kit, fields["mean_"], "mean_", context)
        var source = fields["components_"]
        var comps = _fitted_matrix(door, kit, source, "components_", context)
        var py = Python()
        var out = -1
        if inverse:
            with GILReleased(py):
                out = dl_spca_inverse(kit, x, mean, comps)
            return decomp_kit_publish(door, kit, out)
        var ridge = Float32(decomp_float(fields["ridge_alpha"]))
        var gram = Gram(decomp_door_resident_derived(door, kit, source, _GRAM_NAME))
        with GILReleased(py):
            out = dl_spca_project(kit, x, mean, comps, ridge, gram)
        _keep_gram(door, kit, source, gram)
        return decomp_kit_publish(door, kit, out)
    finally:
        kit.close()
        door.keep_alive()  # the borrowed owners outlive every RUN read


def sparse_encode_call(args: PythonObject, context: PythonObject) raises -> PythonObject:
    # `sparse_encode(X, dictionary, algorithm, n_nonzero_coefs, alpha, init,
    # max_iter, positive, numeric_mode)`.
    var entries = decomp_entries(args[8], context)
    var door = DecompDoor(entries, context["output_context"])
    var kit = decomp_kit_unpack(entries)
    try:
        var init = -1 if args[5] is None else _matrix(door, kit, args[5], "init", context)
        var x = _matrix(door, kit, args[0], "X", context)
        var d = _matrix(door, kit, args[1], "dictionary", context)
        var words = _encode_words(kit.ncols(x), kit.nrows(d), args[2], args[4], args[3], args[6], args[7], False)
        var out = -1
        var py = Python()
        with GILReleased(py):
            var gram = Gram(-1)
            out = dl_sparse_encode(kit, x, d, words, init, gram)
        return decomp_kit_publish(door, kit, out)
    finally:
        kit.close()
        door.keep_alive()  # the borrowed owners outlive every RUN read


def _dict_learning_dispatch(model: PythonObject, operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var result = PythonObject(None)
    if operation == "dl_fit" or operation == "dl_fit_transform":
        result = dictionary_learning_fit(model, args, operation == "dl_fit_transform", context)
    elif operation == "mbdl_fit" or operation == "mbdl_fit_transform":
        result = minibatch_dictionary_fit(model, args, operation == "mbdl_fit_transform", context)
    elif operation == "coding_transform" or operation == "coding_inverse":
        result = coding_transform(model, args, operation == "coding_inverse", context)
    elif operation == "coder_fit" or operation == "coder_transform" or operation == "coder_fit_transform" or operation == "coder_inverse":
        result = sparse_coder(model, operation, args, context)
    elif operation == "spca_fit" or operation == "spca_fit_transform":
        result = sparse_pca_fit(model, args, operation == "spca_fit_transform", context)
    elif operation == "spca_transform" or operation == "spca_inverse":
        result = sparse_pca_transform(model, args, operation == "spca_inverse", context)
    elif operation == "sparse_encode":
        result = sparse_encode_call(args, context)
    else:
        raise Error("unknown native dictionary learning operation")
    return result


def dict_learning_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        if len(call) != 4:
            raise Error("TypeError: dict_learning_native_call requires model, operation, arguments and connected context")
        var b = native_loaded_module("builtins")
        if py_type(call[1]) is not b.str or py_type(call[2]) is not b.tuple or py_type(call[3]) is not b.dict:
            raise Error("TypeError: native dictionary learning requires an operation, an argument tuple and its context")
        var model = call[0]
        var operation = String(py=call[1])
        var args = call[2]
        var context = call[3]
        classical_connect_numpy(context)
        if operation == "coder_transform" or operation == "coder_fit_transform":
            decomp_coder_width(model, args, context)
        var method = decomp_public_method(operation)
        var depth = decomp_public_begin(model, method, args, True, not operation.startswith("coder_"), context)
        var result = PythonObject(None)
        try:
            result = _dict_learning_dispatch(model, operation, args, context)
        finally:
            lifecycle_end(model, method, depth)
        result = decomp_public_finish(model, method, args, result, "", context)
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
