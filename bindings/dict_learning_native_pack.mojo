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
    decomp_entries, decomp_entries_routed, decomp_kit_device_operation, decomp_input, decomp_shape, XD_SPARSE_ALGOS, decomp_kit_unpack, decomp_kit_publish,
    decomp_kit_out_shape, decomp_door_borrow, decomp_door_input, decomp_door_resident_hit,
    decomp_door_resident_input, decomp_door_resident_derived, decomp_door_resident_keep,
)
from bindings.decomp_kit_run import DKit
from bindings.dict_learning_run import (
    EncodeWords, Gram, DictWords, DictFit, MiniBatchWords, SpcaFit, dl_split_sign, dl_sparse_encode, dl_encode_run,
    dl_inverse_run, dl_dict_learning, dl_minibatch_fit, dl_spca_project, dl_spca_inverse, dl_spca_fit_run,
)
from bindings.py_lists import py_list_f64
from bindings.p3nc_fitted_freeze_unpack import fitted_owns_store_any
from std.python._cpython import GILReleased
from bindings.dict_keys_unpack import (
    dict_keys,
    DICT_OPN_DL_FIT,
    DICT_OPN_DL_FIT_TRANSFORM,
    DICT_OPN_MBDL_FIT,
    DICT_OPN_MBDL_FIT_TRANSFORM,
    DICT_OPN_CODING_TRANSFORM,
    DICT_OPN_CODING_INVERSE,
    DICT_OPN_CODER_FIT,
    DICT_OPN_CODER_TRANSFORM,
    DICT_OPN_CODER_FIT_TRANSFORM,
    DICT_OPN_CODER_INVERSE,
    DICT_OPN_SPCA_FIT,
    DICT_OPN_SPCA_FIT_TRANSFORM,
    DICT_OPN_SPCA_TRANSFORM,
    DICT_OPN_SPCA_INVERSE,
    DICT_OPN_SPARSE_ENCODE,
    DICT_OPN_DEVICE_SET,
    DICT_OPN_VENDOR_USED,
)
from bindings.py_keys import py_op_code
from bindings.py_balanced import py_item


from bindings.dict_learning_native_unpack import (
    DL_CALLBACK_WORDS,
    _check_algos,
    _coder_check,
    _coder_dictionary,
    _coding_words,
    _dict_words,
    _encode_words,
    _fit_index,
    _fitted_matrix,
    _gram,
    _keep_gram,
    _is_a,
    _is_str,
    _matrix,
    _minibatch_words,
    coding_transform,
    sparse_encode_call,
    sparse_pca_transform,
)


def _copy_out(array: PythonObject, rows: Int, cols: Int, context: PythonObject) raises -> PythonObject:
    var nk = dict_keys()
    # `_M.out()` of a converted input: a fresh (rows, cols) float32 Array.
    var result = classical_typed(Python.tuple(rows, cols), "<f4", py_item(context, nk[].k_output_context))
    unsafe_memcpy(dest=DecompF32(unsafe_from_address=Int(py=classical_address(result))),
           src=DecompF32(unsafe_from_address=Int(py=classical_address(array))), count=rows * cols)
    _ = result  # keep the Python object alive through the native call above
    return result


# ------------------------------------------------------------ the estimators
def dictionary_learning_fit(model: PythonObject, args: PythonObject, project: Bool,
                            context: PythonObject) raises -> PythonObject:
    var nk = dict_keys()
    # DictionaryLearning.fit / fit_transform.
    var fields = py_fields(model)
    var mode = decomp_mode(py_item(fields, nk[].k_numeric_mode), context)
    py_setitem(fields, nk[].k_numeric_mode_, mode)
    _check_algos(py_item(fields, nk[].k_fit_algorithm), py_item(fields, nk[].k_transform_algorithm))
    if py_get(fields, nk[].k_callback) is not None:
        raise Error(String(DL_CALLBACK_WORDS))
    var entries = decomp_entries_routed(model, fields, mode, context)
    var door = DecompDoor(entries, py_item(context, nk[].k_output_context))
    var kit = decomp_kit_unpack(entries)
    try:
        var x = _matrix(door, kit, args[0], "X", context)
        var requested = py_item(fields, nk[].k_n_components)
        var nc = kit.ncols(x) if requested is None else decomp_int(requested)
        var ci = -1 if py_item(fields, nk[].k_code_init) is None else _matrix(door, kit, py_item(fields, nk[].k_code_init), "code_init", context)
        var di = -1 if py_item(fields, nk[].k_dict_init) is None else _matrix(door, kit, py_item(fields, nk[].k_dict_init), "dict_init", context)
        var method = _fit_index(py_item(fields, nk[].k_fit_algorithm))
        var words = _dict_words(nc, py_item(fields, nk[].k_alpha), py_item(fields, nk[].k_max_iter), py_item(fields, nk[].k_tol), method, py_item(fields, nk[].k_random_state),
                                Bool(py=py_item(fields, nk[].k_positive_dict)), Bool(py=py_item(fields, nk[].k_positive_code)),
                                py_item(fields, nk[].k_transform_max_iter))
        var split = project and Bool(py=py_item(fields, nk[].k_split_sign))
        var fit = DictFit()
        var code = -1
        var py = Python()
        with GILReleased(py):
            fit = dl_dict_learning(kit, x, words, ci, di)
            if project:
                code = dl_split_sign(kit, fit.code) if split else fit.code
        py_setitem(fields, nk[].k_components_, decomp_kit_publish(door, kit, fit.d))
        py_setitem(fields, nk[].k_error_, py_list_f64(fit.errors))
        py_setitem(fields, nk[].k_n_iter_, PythonObject(fit.done))
        py_setitem(fields, nk[].k_n_features_in_, PythonObject(kit.ncols(x)))
        if not project:
            return model
        return decomp_kit_publish(door, kit, code)
    finally:
        kit.close()
        door.keep_alive()  # the borrowed owners outlive every RUN read


def minibatch_dictionary_fit(model: PythonObject, args: PythonObject, project: Bool,
                             context: PythonObject) raises -> PythonObject:
    var nk = dict_keys()
    # MiniBatchDictionaryLearning.fit / fit_transform (fit, then the transform
    # of the X and the dictionary the fit already holds).
    var fields = py_fields(model)
    var mode = decomp_mode(py_item(fields, nk[].k_numeric_mode), context)
    py_setitem(fields, nk[].k_numeric_mode_, mode)
    _check_algos(py_item(fields, nk[].k_fit_algorithm), py_item(fields, nk[].k_transform_algorithm))
    if py_get(fields, nk[].k_callback) is not None:
        raise Error(String(DL_CALLBACK_WORDS))
    var entries = decomp_entries_routed(model, fields, mode, context)
    var door = DecompDoor(entries, py_item(context, nk[].k_output_context))
    var kit = decomp_kit_unpack(entries)
    try:
        var x = _matrix(door, kit, args[0], "X", context)
        var requested = py_item(fields, nk[].k_n_components)
        var nc = kit.ncols(x) if requested is None else decomp_int(requested)
        var init = -1
        if py_item(fields, nk[].k_dict_init) is not None:
            # (after the batch size and the seed are read, as the Python fit did)
            _ = min(decomp_int(py_item(fields, nk[].k_batch_size)), kit.nrows(x))
            _ = decomp_seed(py_item(fields, nk[].k_random_state))
            init = _matrix(door, kit, py_item(fields, nk[].k_dict_init), "dict_init", context)
        var per_pass = 1
        var words = _minibatch_words(kit.nrows(x), py_item(fields, nk[].k_batch_size), py_item(fields, nk[].k_random_state), py_item(fields, nk[].k_shuffle),
                                     py_item(fields, nk[].k_max_iter), py_item(fields, nk[].k_max_no_improvement), py_item(fields, nk[].k_fit_algorithm),
                                     py_item(fields, nk[].k_transform_max_iter), py_item(fields, nk[].k_positive_dict), py_item(fields, nk[].k_positive_code),
                                     py_item(fields, nk[].k_alpha), py_item(fields, nk[].k_tol), per_pass)
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
        py_setitem(fields, nk[].k_n_steps_, PythonObject(n_steps))
        py_setitem(fields, nk[].k_n_iter_, PythonObject(-((-n_steps) // per_pass)))
        py_setitem(fields, nk[].k_components_, decomp_kit_publish(door, kit, d))
        py_setitem(fields, nk[].k_n_features_in_, PythonObject(kit.ncols(x)))
        if not project:
            return model
        return decomp_kit_publish(door, kit, code)
    finally:
        kit.close()
        door.keep_alive()  # the borrowed owners outlive every RUN read


def sparse_coder(model: PythonObject, operation: Int, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = dict_keys()
    # SparseCoder fit / transform (fit_transform) / inverse_transform.
    var fields = py_fields(model)
    if operation == DICT_OPN_CODER_FIT:
        var shape = decomp_shape(args[0], "X", context)
        var dictionary = _coder_dictionary(fields, context)
        var rows = Int(py=dictionary.shape[0])
        var cols = Int(py=dictionary.shape[1])
        if shape[1] != cols:
            raise Error("Dictionary and X have different numbers of features:dictionary.shape: ("
                        + String(rows) + ", " + String(cols) + ") X.shape: (" + String(shape[0]) + ", "
                        + String(shape[1]) + ")")
        py_setitem(fields, nk[].k_numeric_mode_, decomp_mode(py_item(fields, nk[].k_numeric_mode), context))
        py_setitem(fields, nk[].k_components_, _copy_out(dictionary, rows, cols, context))
        py_setitem(fields, nk[].k_n_components_, PythonObject(rows))
        py_setitem(fields, nk[].k_n_features_in_, PythonObject(cols))
        return model
    # The dictionary is the coder's fitted operand (SparseCoder learns
    # nothing). When it is an exact Array owning its words, its device copy
    # and its Gram D D^T stay in the model's resident record across calls,
    # as DictionaryLearning's components_ do (`coding_transform`) and as the
    # member keeps them per state (core/xdecomp_member.mojo handles[0] /
    # [2]): dropped when the dictionary object is replaced (set_params, a
    # new fit) or the model's generation moves; the Array is made read-only
    # at its first query (the read-only rule, decomp_resident_freeze). The
    # door recomputed the Gram per call (lane ma-pub-transform round 9).
    # Any other storage (NumPy, a view) keeps the per-call admission and
    # Gram exactly as before: one type check added, no record lookup. Same
    # RUN, same words: bits unchanged. The Gram entry's name is the one
    # `_gram` uses; a SparseCoder record holds no components_ entry.
    var source = py_item(fields, nk[].k_dictionary)
    var held = fitted_owns_store_any(source)
    var dictionary = PythonObject(None)
    if held:
        _coder_check(fields)
    else:
        dictionary = _coder_dictionary(fields, context)
    var entries = decomp_entries_routed(model, fields, py_item(fields, nk[].k_numeric_mode), context)
    var door = DecompDoor(entries, py_item(context, nk[].k_output_context))
    if held:
        door.owner = model
    var gram_source = source if held else PythonObject(None)
    var kit = decomp_kit_unpack(entries)
    try:
        var d = -1
        if held:
            d = _fitted_matrix(door, kit, source, "dictionary", context)
        else:
            d = decomp_door_borrow(door, kit, dictionary, Int(py=dictionary.shape[0]), Int(py=dictionary.shape[1]))
        var py = Python()
        var out = -1
        if operation == DICT_OPN_CODER_INVERSE:
            var c = _matrix(door, kit, args[0], "code", context)
            var split = Bool(py=py_item(fields, nk[].k_split_sign))
            with GILReleased(py):
                out = dl_inverse_run(kit, c, d, split)
            return decomp_kit_publish(door, kit, out)
        var x = _matrix(door, kit, args[0], "X", context)
        var words = _coding_words(kit.ncols(x), kit.nrows(d), fields)
        var gram = _gram(door, kit, gram_source, words)
        with GILReleased(py):
            out = dl_encode_run(kit, x, d, words, gram)
        _keep_gram(door, kit, gram_source, gram)
        return decomp_kit_publish(door, kit, out)
    finally:
        kit.close()
        door.keep_alive()  # the borrowed owners outlive every RUN read


def sparse_pca_fit(model: PythonObject, args: PythonObject, project: Bool, context: PythonObject) raises -> PythonObject:
    var nk = dict_keys()
    # SparsePCA / MiniBatchSparsePCA fit and fit_transform (the ridge
    # projection of the X, mean and components the fit holds).
    var fields = py_fields(model)
    var mode = decomp_mode(py_item(fields, nk[].k_numeric_mode), context)
    py_setitem(fields, nk[].k_numeric_mode_, mode)
    var minibatch = _is_a(model, py_item(context, nk[].k_minibatch_sparse_pca_class))
    if not minibatch and not _is_str(py_item(fields, nk[].k_method), "lars") and not _is_str(py_item(fields, nk[].k_method), "cd"):
        raise Error("method must be 'lars' or 'cd'")
    var entries = decomp_entries_routed(model, fields, mode, context)
    var door = DecompDoor(entries, py_item(context, nk[].k_output_context))
    var kit = decomp_kit_unpack(entries)
    try:
        # UNPACK: X, the inits and every option, once.
        var x = _matrix(door, kit, args[0], "X", context)
        var cols = kit.ncols(x)
        var requested = py_item(fields, nk[].k_n_components)
        var nc = cols if requested is None else decomp_int(requested)
        var vi = -1
        var ui = -1
        var dict_words = DictWords(nc, 0.0, 0, 0.0, 0, 0, False, False, 0)
        var mb_words = MiniBatchWords(1, 0, False, 0, -1, 0, 0, False, False, 0.0, 0.0)
        var encode = EncodeWords(0, 0, 1.0, 0, 0, False, False)
        var per_pass = 1
        if not minibatch:
            if py_item(fields, nk[].k_V_init) is not None:
                vi = _matrix(door, kit, py_item(fields, nk[].k_V_init), "V_init", context)
            if py_item(fields, nk[].k_U_init) is not None:
                ui = _matrix(door, kit, py_item(fields, nk[].k_U_init), "U_init", context)
            dict_words = _dict_words(nc, py_item(fields, nk[].k_alpha), py_item(fields, nk[].k_max_iter), py_item(fields, nk[].k_tol), _fit_index(py_item(fields, nk[].k_method)),
                                     py_item(fields, nk[].k_random_state), False, False, PythonObject(1000))
        else:
            # MiniBatchDictionaryLearning(n_components=nc, fit_algorithm=method,
            # transform_algorithm="lasso_lars", transform_alpha=alpha, ...) on X^T.
            _check_algos(py_item(fields, nk[].k_method), PythonObject("lasso_lars"))
            mb_words = _minibatch_words(cols, py_item(fields, nk[].k_batch_size), py_item(fields, nk[].k_random_state), py_item(fields, nk[].k_shuffle),
                                        py_item(fields, nk[].k_max_iter), py_item(fields, nk[].k_max_no_improvement), py_item(fields, nk[].k_method),
                                        PythonObject(1000), PythonObject(False), PythonObject(False), py_item(fields, nk[].k_alpha),
                                        py_item(fields, nk[].k_tol), per_pass)
            # lasso_lars on X^T (rows: the features) against the nc atoms
            encode = _encode_words(kit.nrows(x), nc, PythonObject("lasso_lars"), py_item(fields, nk[].k_alpha),
                                   PythonObject(None), PythonObject(1000), PythonObject(False), False)
        var ridge = Float32(decomp_float(py_item(fields, nk[].k_ridge_alpha))) if project else Float32(0.0)
        # RUN: the centering, the fit and the normalization (and the projection).
        var fit = SpcaFit()
        var py = Python()
        with GILReleased(py):
            fit = dl_spca_fit_run(kit, x, vi, ui, minibatch, dict_words, mb_words, encode, project, ridge)
        # PACK: the fitted fields, once.
        py_setitem(fields, nk[].k_components_, decomp_kit_publish(door, kit, fit.comps))
        py_setitem(fields, nk[].k_n_components_, PythonObject(kit.nrows(fit.comps)))
        if not minibatch:
            py_setitem(fields, nk[].k_error_, py_list_f64(fit.errors))
            py_setitem(fields, nk[].k_n_iter_, PythonObject(fit.done))
        else:
            py_setitem(fields, nk[].k_n_iter_, PythonObject(-((-fit.steps) // per_pass)))
        py_setitem(fields, nk[].k_mean_, decomp_kit_out_shape(door, kit, fit.mean, Python.tuple(cols)))
        py_setitem(fields, nk[].k_n_features_in_, PythonObject(cols))
        if not project:
            return model
        return decomp_kit_publish(door, kit, fit.projected)
    finally:
        kit.close()
        door.keep_alive()  # the borrowed owners outlive every RUN read


def _dict_learning_dispatch(model: PythonObject, operation: Int, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var result = PythonObject(None)
    if operation == DICT_OPN_DL_FIT or operation == DICT_OPN_DL_FIT_TRANSFORM:
        result = dictionary_learning_fit(model, args, operation == DICT_OPN_DL_FIT_TRANSFORM, context)
    elif operation == DICT_OPN_MBDL_FIT or operation == DICT_OPN_MBDL_FIT_TRANSFORM:
        result = minibatch_dictionary_fit(model, args, operation == DICT_OPN_MBDL_FIT_TRANSFORM, context)
    elif operation == DICT_OPN_CODING_TRANSFORM or operation == DICT_OPN_CODING_INVERSE:
        result = coding_transform(model, args, operation == DICT_OPN_CODING_INVERSE, context)
    elif operation == DICT_OPN_CODER_FIT or operation == DICT_OPN_CODER_TRANSFORM or operation == DICT_OPN_CODER_FIT_TRANSFORM or operation == DICT_OPN_CODER_INVERSE:
        result = sparse_coder(model, operation, args, context)
    elif operation == DICT_OPN_SPCA_FIT or operation == DICT_OPN_SPCA_FIT_TRANSFORM:
        result = sparse_pca_fit(model, args, operation == DICT_OPN_SPCA_FIT_TRANSFORM, context)
    elif operation == DICT_OPN_SPCA_TRANSFORM or operation == DICT_OPN_SPCA_INVERSE:
        result = sparse_pca_transform(model, args, operation == DICT_OPN_SPCA_INVERSE, context)
    elif operation == DICT_OPN_SPARSE_ENCODE:
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
        var nk = dict_keys()
        var operation = py_op_code(nk[].ops, call[1])
        var args = call[2]
        var context = call[3]
        classical_connect_numpy(context)
        if operation == DICT_OPN_CODER_TRANSFORM or operation == DICT_OPN_CODER_FIT_TRANSFORM:
            decomp_coder_width(model, args, context)
        # The public method's name comes from the decomposition table, which
        # still reads the name (the decomp family's own conversion owns it).
        var method = decomp_public_method(String(py=call[1]))
        var coder = (operation == DICT_OPN_CODER_FIT or operation == DICT_OPN_CODER_TRANSFORM
                     or operation == DICT_OPN_CODER_FIT_TRANSFORM or operation == DICT_OPN_CODER_INVERSE)
        var depth = decomp_public_begin(model, method, args, True, not coder, context)
        var result = PythonObject(None)
        try:
            if operation == DICT_OPN_DEVICE_SET or operation == DICT_OPN_VENDOR_USED:
                result = decomp_kit_device_operation(model, operation == DICT_OPN_DEVICE_SET, args, context)
            else:
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
