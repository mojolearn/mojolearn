# SPDX-License-Identifier: Apache-2.0
# PACK of the PLS family (the door is bindings/pls_native_door.mojo): the
# fitted fields of a fit (published once, into the settings snapshot and the
# instance state together), the public arrays of a query, and the instance
# marks of the fit envelope (the fit depth, the annotated one-row /
# one-feature refusal, the feature names, the fitted mark). Same fields,
# same order, same words as the door held before the split.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.optimizer_public_words_unpack import PYTHON_ERROR_PENDING
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_item
from bindings.metrics_native_support_door import metric_warn
from bindings.decomp_kit_native_door import DecompDoor, decomp_kit_publish, decomp_kit_out_shape
from bindings.decomp_kit_run import DKit
from bindings.pls_run import PlsFit
from bindings.pls_keys_unpack import pls_keys
from bindings.pls_boundary_unpack import PlsFitUnpacked, pls_store


def pls_fit_pack(model: PythonObject, mut door: DecompDoor, mut kit: DKit, fit: PlsFit, unpacked: PlsFitUnpacked,
                 transform: Bool, fields: PythonObject, state: PythonObject,
                 context: PythonObject) raises -> PythonObject:
    """The warning, the iteration counts and the fitted fields, once; the
    estimator, or fit_transform's projections of the X and Y the fit held."""
    var nk = pls_keys()
    var p = unpacked.p
    var q = unpacked.q
    if fit.done < unpacked.words.components:
        metric_warn("y residual is constant at iteration " + String(fit.done),
                    native_loaded_module("builtins").UserWarning, context)
    var all_iterations = Python.list()
    for index in range(len(fit.iterations)):  # driver-loop: publish completed component iteration counts
        _ = py_call(all_iterations.append, PythonObject(Int(fit.iterations[index])))
    pls_store(fields, state, "n_iter_", all_iterations)
    pls_store(fields, state, "x_weights_", decomp_kit_publish(door, kit, fit.outs[0]))
    pls_store(fields, state, "y_weights_", decomp_kit_publish(door, kit, fit.outs[1]))
    pls_store(fields, state, "x_loadings_", decomp_kit_publish(door, kit, fit.outs[4]))
    pls_store(fields, state, "y_loadings_", decomp_kit_publish(door, kit, fit.outs[5]))
    pls_store(fields, state, "x_rotations_", decomp_kit_publish(door, kit, fit.x_rotations))
    pls_store(fields, state, "y_rotations_", decomp_kit_publish(door, kit, fit.y_rotations))
    pls_store(fields, state, "coef_", decomp_kit_publish(door, kit, fit.coef))
    pls_store(fields, state, "intercept_", decomp_kit_out_shape(door, kit, fit.y_mean, Python.tuple(q)))
    pls_store(fields, state, "_x_scores", decomp_kit_publish(door, kit, fit.outs[2]))
    pls_store(fields, state, "_y_scores", decomp_kit_publish(door, kit, fit.outs[3]))
    pls_store(fields, state, "_x_mean", decomp_kit_out_shape(door, kit, fit.x_mean, Python.tuple(p)))
    pls_store(fields, state, "_x_std", decomp_kit_out_shape(door, kit, fit.x_std, Python.tuple(p)))
    pls_store(fields, state, "_y_mean", decomp_kit_out_shape(door, kit, fit.y_mean, Python.tuple(q)))
    pls_store(fields, state, "_y_std", decomp_kit_out_shape(door, kit, fit.y_std, Python.tuple(q)))
    pls_store(fields, state, "n_features_in_", PythonObject(p))
    if Bool(py=py_item(fields, nk[].k__pls_regression)):
        pls_store(fields, state, "x_scores_", py_item(fields, nk[].k__x_scores))
        pls_store(fields, state, "y_scores_", py_item(fields, nk[].k__y_scores))
    if not transform:
        return model
    # fit_transform: the projections of the X and Y the fit held.
    return Python.tuple(decomp_kit_publish(door, kit, fit.tx), decomp_kit_publish(door, kit, fit.ty))


def pls_pair_pack(mut door: DecompDoor, mut kit: DKit, first: Int, second: Int, paired: Bool) raises -> PythonObject:
    """transform / inverse_transform: the X result, or (X result, Y result)
    when the call carried the second operand."""
    if not paired:
        return decomp_kit_publish(door, kit, first)
    return Python.tuple(decomp_kit_publish(door, kit, first), decomp_kit_publish(door, kit, second))


def pls_predict_pack(mut door: DecompDoor, mut kit: DKit, prediction: Int, fields: PythonObject) raises -> PythonObject:
    """predict: 1-D when the fit's target was a vector."""
    var nk = pls_keys()
    if Bool(py=py_item(fields, nk[].k__predict_1d)):
        return decomp_kit_out_shape(door, kit, prediction, Python.tuple(kit.nrows(prediction)))
    return decomp_kit_publish(door, kit, prediction)


# ---------------------------------------------------------------- the fit envelope's instance marks


def pls_fit_enter(fields: PythonObject, state: PythonObject) raises -> Int:
    """Mark the instance as fitting (one level deeper); the depth before."""
    var nk = pls_keys()
    var depth_value = py_get(fields, nk[].k__mojolearn_fitting, PythonObject(0))
    if py_type(depth_value) is not py_type(PythonObject(0)):
        raise Error("TypeError: native fit depth must be an exact integer")
    var depth = Int(py=depth_value)
    py_setitem(state, nk[].k__mojolearn_fitting, PythonObject(depth + 1))
    return depth


def pls_fit_leave(state: PythonObject, depth: Int) raises:
    var nk = pls_keys()
    if depth > 0:
        py_setitem(state, nk[].k__mojolearn_fitting, PythonObject(depth))
    else:
        _ = py_call(state.pop, "_mojolearn_fitting", PythonObject(None))


def pls_fit_annotate(message: String, name: String, rows: Int, cols: Int) raises:
    """_sk_validate.reraise annotated failed one-row / one-feature fits:
    raises the annotated refusal, or returns for the caller to re-raise."""
    var device_fault = message.find("device_context") >= 0 or message.find("CUDA") >= 0 or message.find("HIP") >= 0 or message.find("hipError") >= 0 or message.find("Metal") >= 0 or message.find("MTL") >= 0
    if not device_fault and not message.startswith("TypeError:") and not message.startswith("RuntimeError:") and not message.startswith("NotImplementedError:") and not message.startswith("ImportError:") and message != PYTHON_ERROR_PENDING and not message.startswith("Input X contains NaN or infinity"):
        if rows == 1:
            raise Error(name + " cannot be fit on n_samples=1: " + message)
        if cols == 1:
            raise Error(name + " cannot be fit on n_features=1: " + message)


def pls_fit_mark(state: PythonObject, feature_names: PythonObject) raises:
    """A finished fit: the feature names it saw (or none) and the fitted mark."""
    var nk = pls_keys()
    if feature_names is None:
        _ = py_call(state.pop, "feature_names_in_", PythonObject(None))
    else:
        py_setitem(state, nk[].k_feature_names_in_, feature_names)
    py_setitem(state, nk[].k__mojolearn_fitted, PythonObject(True))
