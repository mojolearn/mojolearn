# Focused compilation of parameter-grid admission/expansion and candidate routes.
# Never executed. Full model-selection dispatcher/kernel campaigns are excluded.
from std.python import PythonObject
from bindings.msel_search import (
    msel_grid_dicts, msel_grid_list, msel_grid_len, msel_sampler_check,
    msel_sampler_list, msel_sampler_len, msel_candidates,
)
from bindings.model_selection_native import msel_operation

@export
def compile_grid_admit(grid: PythonObject, context: PythonObject) raises -> PythonObject:
    return msel_grid_dicts(grid, context)

@export
def compile_grid_expand(grid: PythonObject, context: PythonObject) raises -> PythonObject:
    return msel_grid_list(grid, context)

@export
def compile_grid_count(grid: PythonObject, context: PythonObject) raises -> Int:
    return msel_grid_len(grid, context)

@export
def compile_sampler_admit(grid: PythonObject, context: PythonObject) raises -> PythonObject:
    return msel_sampler_check(grid, context)

@export
def compile_sampler(grid: PythonObject, count: PythonObject, seed: PythonObject,
                    context: PythonObject) raises -> PythonObject:
    return msel_sampler_list(grid, count, seed, context)

@export
def compile_sampler_count(grid: PythonObject, count: PythonObject, context: PythonObject) raises -> PythonObject:
    return msel_sampler_len(grid, count, context)

@export
def compile_candidates(model: PythonObject, context: PythonObject) raises -> PythonObject:
    return msel_candidates(model, context)
