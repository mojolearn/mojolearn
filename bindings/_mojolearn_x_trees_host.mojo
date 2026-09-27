# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding of the trees expansion lane (`x_trees` family).

HOST ONLY: no DeviceContext, no kernel. It exports the GPU binding's names
(`xtrees/api.mojo::register`, the SAME functions the GPU binding registers)
plus `x_trees_numeric_mode` (1) and `x_trees_vendor` ("cpu"), and the four
host read-backs `x_trees_host_{numeric_mode,vendor,column,sabotage}`, so
`_expansion_trees.py` runs unchanged on a CPU-only install through
`_backend._HOST_MODULES`."""
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE
from xtrees.api import (
    accumulate_binding,
    accumulate_cols_binding,
    accumulate_onehot_binding,
    apply_binding,
    argmax_rows_binding,
    argmax_rows_f32_binding,
    block_mean_binding,
    expected_value_binding,
    gather_f32_binding,
    gather_i32_binding,
    gradients_binding,
    isotonic_fit_binding,
    isotonic_predict_binding,
    kernel_solve_binding,
    leaf_newton_binding,
    log64_binding,
    mask_expand_binding,
    node_cover_binding,
    normalize_rows_binding,
    onehot_leaves_binding,
    platt_apply_binding,
    platt_fit_binding,
    put_f32_binding,
    r2_step_binding,
    samme_step_binding,
    sample_indices_binding,
    scale_binding,
    scale_to_f32_binding,
    scatter_binding,
    softmax_rows_binding,
    transpose_f32_binding,
    tree_score_add_binding,
    tree_shap_binding,
    uniform_binding,
    weighted_median_binding,
    weighted_sample_binding,
)
from xtrees.ops import XTREES_HOST_SABOTAGE


def x_trees_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(Int(GLOBAL_NUMERIC_MODE))


def x_trees_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def x_trees_host_column_binding() raises -> PythonObject:
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "x_trees host: this binding compiles the CPU column only; pass -D MOJOLEARN_COLUMN_CPU"
        " (bindings/build_x_trees_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


def x_trees_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary was built with -D MOJOLEARN_HOST_SABOTAGE=1 (the
    gate's negative control: xtrees/ops.mojo `scale_f64` divides by a
    perturbed divisor). The GPU binding never defines it."""
    return PythonObject(XTREES_HOST_SABOTAGE)


@export
def PyInit__mojolearn_x_trees_host() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_x_trees_host")
        m.def_function[x_trees_host_numeric_mode_binding]("x_trees_host_numeric_mode")
        m.def_function[x_trees_host_vendor_binding]("x_trees_host_vendor")
        m.def_function[x_trees_host_column_binding]("x_trees_host_column")
        m.def_function[x_trees_host_sabotage_binding]("x_trees_host_sabotage")
        m.def_function[x_trees_host_numeric_mode_binding]("x_trees_numeric_mode")
        m.def_function[x_trees_host_vendor_binding]("x_trees_vendor")
        m.def_function[sample_indices_binding]("x_trees_sample_indices")
        m.def_function[weighted_sample_binding]("x_trees_weighted_sample")
        m.def_function[gather_f32_binding]("x_trees_gather_f32")
        m.def_function[gather_i32_binding]("x_trees_gather_i32")
        m.def_function[accumulate_binding]("x_trees_accumulate")
        m.def_function[accumulate_onehot_binding]("x_trees_accumulate_onehot")
        m.def_function[accumulate_cols_binding]("x_trees_accumulate_cols")
        m.def_function[argmax_rows_binding]("x_trees_argmax_rows")
        m.def_function[argmax_rows_f32_binding]("x_trees_argmax_rows_f32")
        m.def_function[scale_binding]("x_trees_scale")
        m.def_function[softmax_rows_binding]("x_trees_softmax_rows")
        m.def_function[scale_to_f32_binding]("x_trees_scale_to_f32")
        m.def_function[put_f32_binding]("x_trees_put_f32")
        m.def_function[samme_step_binding]("x_trees_samme_step")
        m.def_function[r2_step_binding]("x_trees_r2_step")
        m.def_function[weighted_median_binding]("x_trees_weighted_median")
        m.def_function[apply_binding]("x_trees_apply")
        m.def_function[gradients_binding]("x_trees_gradients")
        m.def_function[leaf_newton_binding]("x_trees_leaf_newton")
        m.def_function[tree_score_add_binding]("x_trees_tree_score_add")
        m.def_function[uniform_binding]("x_trees_uniform")
        m.def_function[onehot_leaves_binding]("x_trees_onehot_leaves")
        m.def_function[transpose_f32_binding]("x_trees_transpose_f32")
        m.def_function[log64_binding]("x_trees_log64")
        m.def_function[normalize_rows_binding]("x_trees_normalize_rows")
        m.def_function[scatter_binding]("x_trees_scatter")
        m.def_function[platt_fit_binding]("x_trees_platt_fit")
        m.def_function[platt_apply_binding]("x_trees_platt_apply")
        m.def_function[isotonic_fit_binding]("x_trees_isotonic_fit")
        m.def_function[isotonic_predict_binding]("x_trees_isotonic_predict")
        m.def_function[node_cover_binding]("x_trees_node_cover")
        m.def_function[tree_shap_binding]("x_trees_tree_shap")
        m.def_function[expected_value_binding]("x_trees_expected_value")
        m.def_function[mask_expand_binding]("x_trees_mask_expand")
        m.def_function[block_mean_binding]("x_trees_block_mean")
        m.def_function[kernel_solve_binding]("x_trees_kernel_solve")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_x_trees_host: ", e))
