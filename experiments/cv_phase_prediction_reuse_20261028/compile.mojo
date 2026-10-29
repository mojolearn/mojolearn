# SPDX-License-Identifier: Apache-2.0
# Focused compile-only actual cross_validate operation; never executed.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.msel_cv import msel_cross_validate


def cross_validate(args: PythonObject, context: PythonObject) raises -> PythonObject:
    return msel_cross_validate(args, context)


@export
def PyInit__cv_phase_prediction_reuse() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_cv_phase_prediction_reuse")
        module.def_function[cross_validate]("cross_validate")
        return module.finalize()
    except error:
        abort(String(error))
