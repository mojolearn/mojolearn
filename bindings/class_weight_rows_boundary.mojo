# SPDX-License-Identifier: Apache-2.0
"""One native class-weight pipeline behind the admitted family boundaries.

Profile selects only the existing scalar/dictionary admission and provider
format. Counts, cast, ratios, Float32 planning and row gather are shared.
"""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.admitted_input import admitted_storage_native
from bindings.class_weight_boundary import (
    CLASS_WEIGHT_CLASSICAL, CLASS_WEIGHT_SVC, CLASS_WEIGHT_TREES, class_weight_dictionary_values,
)
from bindings.label_weights_boundary import label_weights_run
from bindings.labels_descriptor_boundary import labels_class_ratio_f64
from core.admitted_cast_elements import admitted_cast_elements
from core.class_weight_plan import ClassWeightInputs, ClassWeightPlan, class_weight_all_one, class_weight_plan
from core.label_weights_native_api import LabelWeightsRequest, LABEL_WEIGHTS_COUNTS, LABEL_WEIGHTS_ROWS


def class_weight_rows_admitted[Profile: Int = CLASS_WEIGHT_CLASSICAL](
    class_weight: PythonObject, classes: PythonObject, codes: PythonObject,
    n: Int, k: Int, balanced: Bool, xtrees: PythonObject, context: PythonObject,
) raises -> PythonObject:
    var inputs = ClassWeightInputs()
    if balanced:
        var counts = List[Int32](unsafe_uninit_length=k)
        _ = label_weights_run[Profile == CLASS_WEIGHT_SVC](xtrees, LABEL_WEIGHTS_COUNTS,
            LabelWeightsRequest(Int(py=codes._addr) if n else 0, Int(counts.unsafe_ptr()), 0, n, k))
        var wide = List[Int64](unsafe_uninit_length=k)
        if k and admitted_cast_elements(Int(counts.unsafe_ptr()), 2, Int(wide.unsafe_ptr()), 3, k) != 0:
            raise Error("TypeError: mojolearn: an element does not fit <i8")
        inputs.values = List[Float64](unsafe_uninit_length=k)
        if k and labels_class_ratio_f64(context, Int(wide.unsafe_ptr()), k, n, 0,
                                       Int(inputs.values.unsafe_ptr())) != 0:
            raise Error("ZeroDivisionError: division by zero")
        _ = counts^
        _ = wide^
    else:
        inputs = class_weight_dictionary_values[Profile](class_weight, classes, context)
    var plan = ClassWeightPlan(True, List[Float32]())
    with GILReleased(Python()):
        if balanced:
            inputs.all_one = class_weight_all_one(inputs.values)
        plan = class_weight_plan(inputs.values, inputs.all_one)
    _ = inputs^
    if plan.all_one:
        return PythonObject(None)
    var output = admitted_storage_native(List[Int](length=1, fill=n), "<f4", context, as_array=True)
    if n:
        _ = label_weights_run[Profile == CLASS_WEIGHT_SVC](xtrees, LABEL_WEIGHTS_ROWS,
            LabelWeightsRequest(Int(py=codes._addr), Int(plan.words.unsafe_ptr()),
                                Int(py=output._addr), n, k))
    _ = plan^
    _ = codes
    _ = classes
    _ = xtrees
    return output
