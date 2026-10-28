# SPDX-License-Identifier: Apache-2.0
# Actual public RNN/MLP envelope formerly installed by _mode's Python wrappers.
# Retain fit tracking, validation, fitted errors, feature names and public output
# dtype while calling the existing xseq native owners directly.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields
from bindings.native_loaded_module import native_loaded_module
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_pandas import classical_admit_value
from bindings.classical_validate import classical_view, classical_refuse_sparse, classical_estimator_name, classical_check_width, RESHAPE_HINT
from bindings.classical_exceptions import classical_require_fitted
from bindings.classical_labels import labels_shape_of
from bindings.classical_warnings import classical_warn_column
from bindings.prep_lda_admission import lda_native_labels
from bindings.classifier_public_lifecycle import classifier_label_list
from bindings.sequential_arrays import sequential_array, sequential_address
from bindings.xseq_native_common import xseq_base
from providers.python.estimator_lifecycle_pyapi import estimator_outside, estimator_feature_check, estimator_fit_begin, estimator_fit_end, estimator_fit_complete
from core.classical_ridge_impl import classical_widen_f32_f64


def xseq_public_labels(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    return classifier_label_list(fields["classes_"], context["xseq_lifecycle_context"])


def _finite(view: PythonObject, name: String, model_name: String, context: PythonObject) raises:
    var code = Int(py=view.dtype.num)
    if code != 11 and code != 12:
        return
    var input = view
    if not Bool(py=view.flags.c_contiguous):
        input = py_call(view.copy, "C")
    var entry = xseq_base(context, "all_finite_f32" if code == 11 else "all_finite_f64")
    if Int(py=py_call(entry, sequential_address(input), Int(py=input.size))) != 1:
        raise Error("Input " + name + " contains NaN or infinity (" + model_name + ").")
    _ = input


struct XseqPublicLifecycle(Movable):
    var source: PythonObject
    var arguments: PythonObject
    var lifecycle: PythonObject
    var fitting: Bool
    var classifier: Bool
    var nested: Bool
    var outside: Bool
    var name: String
    var method: String
    var shape: List[Int]

    def __init__(out self, model: PythonObject, operation: String, args: PythonObject,
                 context: PythonObject) raises:
        self.source = args[0]
        self.arguments = args
        self.lifecycle = context["xseq_lifecycle_context"]
        self.fitting = operation.endswith("_fit")
        self.method = String("fit" if self.fitting else ("score" if operation.endswith("score") else
            ("predict_proba" if operation.endswith("predict_proba") else
             ("hidden_sequence" if operation.endswith("hidden_sequence") else "predict"))))
        self.outside = estimator_outside()
        self.name = classical_estimator_name(model)
        self.shape = List[Int]()
        var b = native_loaded_module("builtins")
        self.classifier = String(py=py_call(b.getattr, py_type(model), "_estimator_type", "")) == "classifier"
        var fields = py_fields(model)
        var depth = py_get(fields, "_mojolearn_fitting", PythonObject(0))
        if py_type(depth) is not b.int:
            raise Error("TypeError: native fit depth must be an exact integer")
        self.nested = Int(py=depth) > 0
        if not self.fitting:
            classical_require_fitted(model, False, self.method, self.lifecycle)
        if self.nested:
            return
        classical_connect_numpy(self.lifecycle)
        classical_refuse_sparse(self.source, self.name)
        var x = classical_view(self.source, self.lifecycle, "X")
        var kind = String(py=x.dtype.kind)
        if kind == "c":
            raise Error("Complex data not supported: " + self.name + " got a complex X.")
        var dimensions = x.shape
        for axis in range(len(dimensions)):  # small-loop(public input rank metadata)
            self.shape.append(Int(py=dimensions[axis]))
        var rank = len(self.shape)
        if kind == "b" or kind == "i" or kind == "u" or kind == "f":
            if rank < 2:
                raise Error("Expected 2D array, got " + String("scalar" if rank == 0 else "1D") + " array instead (shape="
                    + String(py=py_call(b.repr, dimensions)) + "). " + RESHAPE_HINT)
            if rank == 2 and (self.shape[0] == 0 or self.shape[1] == 0):
                raise Error("Found array with 0 " + String("sample(s)" if self.shape[0] == 0 else "feature(s)")
                    + " (shape=" + String(py=py_call(b.repr, dimensions)) + ") while a minimum of 1 is required by " + self.name + ".")
        if operation.startswith("mlp_"):
            _finite(x, "X", self.name, context)
        if not self.fitting and rank == 2:
            var columns = py_get(fields, "n_features_in_")
            if py_type(columns) is b.int:
                classical_check_width(self.shape[1], Int(py=columns), self.name)
            estimator_feature_check(model, fields, self.source, self.lifecycle)
        var prepared = Python.list(x)
        if self.fitting:
            var source_y = args[1]
            if source_y is None:
                raise Error(self.name + " requires y to be passed, but the target y is None.")
            classical_admit_value(source_y, self.lifecycle)
            var y = source_y
            if self.classifier:
                lda_native_labels(source_y, self.lifecycle, True, self.name)
            var target_shape = labels_shape_of(source_y, self.lifecycle)
            if rank > 0 and len(target_shape) > 0 and self.shape[0] != target_shape[0]:
                raise Error("Found input variables with inconsistent numbers of samples: [" + String(self.shape[0]) + ", " + String(target_shape[0]) + "]")
            # String labels remain in their native container for the label
            # encoder. Numeric targets retain the former column-warning path.
            if not self.classifier:
                y = classical_view(source_y, self.lifecycle, "y")
                if String(py=y.dtype.kind) == "c":
                    raise Error("Complex data not supported: " + self.name + " got a complex y.")
                _finite(y, "y", self.name, context)
            if len(target_shape) == 2 and target_shape[1] == 1 and py_type(source_y) is not b.list and py_type(source_y) is not b.tuple and py_type(source_y) is not b.memoryview:
                if not self.classifier:
                    classical_warn_column(self.lifecycle)
                y = classical_view(source_y, self.lifecycle, "y")
                y = py_call(y.reshape, Python.tuple(-1))
            _ = py_call(prepared.append, y)
            for i in range(2, len(args)):  # small-loop(remaining admitted public arguments)
                classical_admit_value(args[i], self.lifecycle)
                _ = py_call(prepared.append, args[i])
        else:
            for i in range(1, len(args)):  # small-loop(score target and weight arguments)
                classical_admit_value(args[i], self.lifecycle)
                _ = py_call(prepared.append, args[i])
        self.arguments = prepared

    def refusal(self, problem: Error) -> Error:
        # Preserve _sk_validate.reraise's single-sample/feature explanation;
        # a device failure or a typed API refusal is never relabeled as data.
        var text = String(problem)
        if self.nested or not self.fitting or len(self.shape) != 2:
            return Error(text)
        if (text.find("device_context") >= 0 or text.find("CUDA") >= 0 or text.find("HIP") >= 0
            or text.find("hipError") >= 0 or text.find("Metal") >= 0 or text.find("MTL") >= 0):
            return Error(text)
        if (text.startswith("TypeError:") or text.startswith("RuntimeError:") or text.startswith("ImportError:")
            or text.startswith("NotImplementedError:") or text.startswith("AttributeError:")
            or text.startswith("IndexError:") or text.startswith("OverflowError:")):
            return Error(text)
        if self.shape[0] == 1:
            return Error(self.name + " cannot be fit on n_samples=1: " + text)
        if self.shape[1] == 1 and not text.startswith("ZeroDivisionError:"):
            return Error(self.name + " cannot be fit on n_features=1: " + text)
        return Error(text)

    def finish(self, model: PythonObject, result: PythonObject, context: PythonObject) raises -> PythonObject:
        if self.fitting:
            if self.nested:
                py_setitem(py_fields(model), "_mojolearn_fitted", PythonObject(True))
            else:
                estimator_fit_complete(model, self.source, self.classifier, self.lifecycle)
            return result
        if self.outside and self.method == "predict" and not self.classifier:
            # The former public-output wrapper widens external regression
            # predictions exactly. Internal score/identity calls retain f32.
            if py_type(result) is context["ndarray"] and String(py=result.dtype.kind) == "f" and Int(py=result.itemsize) == 4:
                var wide = sequential_array(result.shape, context["float64"], context)
                classical_widen_f32_f64(Int(py=sequential_address(result)), Int(py=sequential_address(wide)), Int(py=result.size))
                _ = result
                return wide
        return result
