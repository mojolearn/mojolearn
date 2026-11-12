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
# The PACK half of the former `bindings/dict_learning_native.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/dict_learning_native_unpack.mojo`.
from std.python import Python, PythonObject
from bindings.py_cpython_table import cpython_table
from bindings.native_loaded_module_door import native_loaded_module
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from std.memory import unsafe_memcpy
from bindings.optimizer_public_words_unpack import PYTHON_ERROR_PENDING, optimizer_python_error
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_type, py_get, py_fields, py_setitem, py_contains
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.classical_arrays_door import classical_typed, classical_address
from bindings.admitted_input_unpack import admit_x
from bindings.decomp_public_lifecycle_unpack import decomp_public_begin, decomp_coder_width
from bindings.decomp_public_lifecycle_pack import decomp_public_finish
from x_decomp.decomp_public_lifecycle_run import decomp_public_method
from bindings.public_lifecycle_owner_door import lifecycle_end
from bindings.decomp_kit_native_door import (
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


from bindings.dict_learning_native_unpack import (
    _check_algos,
    _coder_dictionary,
    _coding_words,
    _dict_words,
    _encode_words,
    _fit_index,
    _is_a,
    _is_str,
    _matrix,
    _minibatch_words,
    coding_transform,
    sparse_encode_call,
    sparse_pca_transform,
)


def _copy_out(array: PythonObject, rows: Int, cols: Int, context: PythonObject) raises -> PythonObject:
    # `_M.out()` of a converted input: a fresh (rows, cols) float32 Array.
    var result = classical_typed(Python.tuple(rows, cols), "<f4", context["output_context"])
    unsafe_memcpy(dest=DecompF32(unsafe_from_address=Int(py=classical_address(result))),
           src=DecompF32(unsafe_from_address=Int(py=classical_address(array))), count=rows * cols)
    _ = result  # keep the Python object alive through the native call above
    return result


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
