# SPDX-License-Identifier: Apache-2.0
# Shared native tabular target admission and public fit error translation.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_type
from bindings.classical_validate import classical_view
from bindings.classical_arrays import classical_kind
from bindings.classical_labels import labels_shape_of
from bindings.classical_warnings import classical_warn_column
from bindings.family_common import family_f32
from bindings.cluster_fit_a_native import _pending, _pending_text, _drop_pending
from bindings.optimizer_public_words import PYTHON_ERROR_PENDING


def _target(source: PythonObject, rows: Int, name: String, context: PythonObject) raises:
    if source is None:
        return
    var b = native_loaded_module("builtins")
    var value = classical_view(source, context, "y")
    var kind = classical_kind(value, context)
    if kind == "c":
        raise Error("Complex data not supported: " + name + " got a complex y.")
    var shape = labels_shape_of(source, context)
    if len(shape) > 0 and shape[0] != rows:
        raise Error("Found input variables with inconsistent numbers of samples: [" + String(rows) + ", " + String(shape[0]) + "]")
    if len(shape) == 2 and shape[1] == 1 and py_type(source) is not b.list and py_type(source) is not b.tuple and py_type(source) is not b.memoryview:
        classical_warn_column(context)
    if kind == "f":
        # The native cast owns the finite-value scan; y is accepted/ignored by
        # these unsupervised algorithms, exactly as their public signatures say.
        _ = family_f32(source, -1, "y", context)


def _refusal(error: Error, fitting: Bool, nested: Bool, rows: Int, columns: Int, name: String) raises -> Error:
    var text = String(error)
    var pending = text == PYTHON_ERROR_PENDING
    var b = native_loaded_module("builtins")
    var original = _pending(0) if pending else PythonObject(None)
    # Only standard native exception classes may have their message read here.
    # A foreign exception retains the original pending exception unchanged.
    if pending and original is not b.Exception and original is not b.ValueError and original is not b.ZeroDivisionError:
        return Error(text)
    if pending:
        text = _pending_text(error)
    for fault in ["device_context", "CUDA", "HIP", "hipError", "Metal", "MTL"]:  # driver-loop: fixed device failure spellings
        if text.find(fault) >= 0:
            return Error(String(error))
    var message = String("")
    if not nested and pending and original is b.Exception:
        for spelling in ["NaN", "infinity", "Infinity", "infinite", "Infinite", "nonfinite", "non-finite", "not finite", "requires finite", "must be finite"]:  # driver-loop: former public nonfinite error classification
            if text.find(spelling) >= 0:
                message = (String("Input") if fitting else String("Input X")) + " contains NaN or infinity (" + name + ": " + text + ")"
                break
    if message.byte_length() == 0 and fitting and not nested:
        var value_error = original is b.ValueError or (not pending and not text.startswith("TypeError:") and not text.startswith("RuntimeError:") and not text.startswith("ImportError:") and not text.startswith("AttributeError:") and not text.startswith("OverflowError:") and not text.startswith("IndexError:"))
        if rows == 1 and (value_error or original is b.Exception or original is b.ZeroDivisionError):
            message = name + " cannot be fit on n_samples=1: " + text
        elif columns == 1 and value_error:
            message = name + " cannot be fit on n_features=1: " + text
    if message.byte_length() == 0:
        return Error(String(error))
    if pending:
        _drop_pending()
    return Error("ValueError: " + message)
