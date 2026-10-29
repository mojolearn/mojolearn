# SPDX-License-Identifier: Apache-2.0
# Compile only the affected native implementation helpers. Never executed.
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.metrics_reg_native import _Prog
from bindings.msel_common import MselGroups, MselRng
from bindings.msel_split import _fold_rows_run, _select_folds, msel_strat_shuffle_split
from bindings.msel_search import _group_order


def group_plan(values: PythonObject, words: PythonObject, sizes: PythonObject,
               context: PythonObject) raises -> PythonObject:
    var plan = MselGroups(values, Int(py=words.size), "groups", context)
    var mapped = plan.mapped([0], context)
    var rows = plan.rows(words, 0, 0, context)
    var only = plan.only(words, 0, 0, context)
    var folds = plan.split_by_fold(words, Int(py=sizes.size), sizes, context)
    return Python.tuple(mapped, rows, only, folds)


def fold_rows(n: PythonObject, k: PythonObject, code: PythonObject, order: PythonObject,
              context: PythonObject) raises -> PythonObject:
    var prog = _Prog(context["family_metrics_ops"])
    return _fold_rows_run(prog, Int(py=n), Int(py=k), Int(py=code), Int(py=order), context)


def select_folds(store: PythonObject, counts: PythonObject, n: PythonObject,
                 k: PythonObject, context: PythonObject) raises -> PythonObject:
    return _select_folds(store, counts, Int(py=n), Int(py=k), context)


def approximate_modes(counts: PythonObject, train: PythonObject, test: PythonObject,
                      context: PythonObject) raises -> PythonObject:
    var sizes = List[Int]()
    for i in range(len(counts)):  # driver-loop: compile fixture scalar metadata
        sizes.append(Int(py=counts[i]))
    var rng = MselRng(PythonObject(0), context)
    var both = rng.approximate_modes(sizes, Int(py=train), Int(py=test), context)
    return PythonObject(len(both))


def group_order(values: PythonObject, n: PythonObject, context: PythonObject) raises -> PythonObject:
    return _group_order(values, Int(py=n), context)


def stratified_shuffle(values: PythonObject, count: PythonObject, test: PythonObject,
                       train: PythonObject, context: PythonObject) raises -> PythonObject:
    return msel_strat_shuffle_split(values, Int(py=count), test, train, PythonObject(0), PythonObject(0.1), context)


@export
def PyInit__model_selection_loop_setup() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_model_selection_loop_setup")
        module.def_function[group_plan]("group_plan")
        module.def_function[fold_rows]("fold_rows")
        module.def_function[select_folds]("select_folds")
        module.def_function[approximate_modes]("approximate_modes")
        module.def_function[group_order]("group_order")
        module.def_function[stratified_shuffle]("stratified_shuffle")
        return module.finalize()
    except error:
        abort(String(error))
