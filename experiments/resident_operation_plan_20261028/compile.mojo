# Focused CPU compilation of private resident row integration; never executed.
from std.python import PythonObject
from bindings.msel_cv import msel_cross_validate, msel_cross_val_predict
from bindings.msel_search import msel_learning_curve, msel_permutation_test_score

@export
def compile_cv(args: PythonObject, context: PythonObject) raises -> PythonObject:
    return msel_cross_validate(args, context)

@export
def compile_predict(args: PythonObject, context: PythonObject) raises -> PythonObject:
    return msel_cross_val_predict(args, context)

@export
def compile_learning(args: PythonObject, context: PythonObject) raises -> PythonObject:
    return msel_learning_curve(args, context)

@export
def compile_permutation(args: PythonObject, context: PythonObject) raises -> PythonObject:
    return msel_permutation_test_score(args, context)
