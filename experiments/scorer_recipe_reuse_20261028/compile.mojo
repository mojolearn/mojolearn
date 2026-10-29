# SPDX-License-Identifier: Apache-2.0
# Actual CV/search/scorer owners, compile only; never execute this artifact.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.msel_cv import msel_cross_validate
from bindings.msel_search import msel_learning_curve
from bindings.msel_scorer_native import msel_scorer_call


def cross_validate(args: PythonObject, context: PythonObject) raises -> PythonObject:
    return msel_cross_validate(args, context)


def learning_curve(args: PythonObject, context: PythonObject) raises -> PythonObject:
    return msel_learning_curve(args, context)


def scorer(args: PythonObject, context: PythonObject) raises -> PythonObject:
    return msel_scorer_call(args[0], args[1], args[2], args[3], args[4], args[5], context)


@export
def PyInit__scorer_recipe_reuse() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_scorer_recipe_reuse")
        module.def_function[cross_validate]("cross_validate")
        module.def_function[learning_curve]("learning_curve")
        module.def_function[scorer]("scorer")
        return module.finalize()
    except error:
        abort(String(error))
