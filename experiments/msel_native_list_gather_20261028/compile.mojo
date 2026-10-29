# SPDX-License-Identifier: Apache-2.0
# Compile the affected public operations; never execute this fixture.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.msel_cv import msel_cross_validate
from bindings.msel_search import msel_learning_curve
from bindings.msel_split import msel_train_test_split


def cross_validate(args: PythonObject, context: PythonObject) raises -> PythonObject:
    return msel_cross_validate(args, context)


def learning_curve(args: PythonObject, context: PythonObject) raises -> PythonObject:
    return msel_learning_curve(args, context)


def train_test_split(args: PythonObject, context: PythonObject) raises -> PythonObject:
    return msel_train_test_split(args[0], args[1], args[2], args[3], args[4], args[5], context)


@export
def PyInit__msel_native_list_gather() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_msel_native_list_gather")
        module.def_function[cross_validate]("cross_validate")
        module.def_function[learning_curve]("learning_curve")
        module.def_function[train_test_split]("train_test_split")
        return module.finalize()
    except error:
        abort(String(error))
