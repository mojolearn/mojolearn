# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""TreeSHAP cache ABI for the host binding only.

Backend selection belongs to the binding entry point, so the GPU module
never imports CPU TreeSHAP or its thread scheduling policy.
"""
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys.compile import is_defined
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from xtrees.api import _need, _i, _count
from xtrees import shap_host as shap_cpu


# T44/C51 retained snapshot ABI, shared by GPU and host registrations.
# NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
comptime _TREE_SHAP_CACHE = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and is_defined["MOJOLEARN_TREES_T44_METADATA_CACHE"]()


def tree_shap_cache_enabled_binding() raises -> PythonObject:
    return PythonObject(_TREE_SHAP_CACHE)


def tree_shap_cache_create_binding(forest: PythonObject, tscale: PythonObject,
                                   cover: PythonObject, params: PythonObject) raises -> PythonObject:
    comptime if not _TREE_SHAP_CACHE:
        raise Error("TreeSHAP cache requires the opt-in T44 binding")
    else:
        _need(params, 4, "x_trees_tree_shap_cache_create")
        if len(forest) != 5:
            raise Error("TreeSHAP cache requires five forest addresses")
        var addresses = List[Int]()
        for i in range(5):
            addresses.append(Int(py=forest[i]))
        var d = _i(params, 0)
        var trees = _i(params, 1)
        var k = _i(params, 2)
        var nodes = _i(params, 3)
        if d < 1 or trees < 1 or k < 1 or nodes < trees:
            raise Error("TreeSHAP cache requires positive model dimensions")
        return PythonObject(shap_cpu.shap_cache_create(addresses, Int(py=tscale), Int(py=cover), d, trees, k, nodes))


def tree_shap_cache_values_binding(handle: PythonObject, x: PythonObject,
                                   phi: PythonObject, params: PythonObject) raises -> PythonObject:
    _need(params, 3, "x_trees_tree_shap_cache_values")
    var n = _count(_i(params, 0), "x_trees_tree_shap_cache_values")
    var slots = _count(_i(params, 1), "x_trees_tree_shap_cache_values")
    var width = _i(params, 2)
    if width < 8 or width > 256 or (width & (width - 1)) != 0:
        raise Error("TreeSHAP cache path width must be 8, 16, ..., 256")
    if n > 0:
        shap_cpu.shap_cache_values(Int(py=handle), Int(py=x), Int(py=phi), n, slots, width)
    return PythonObject(n)


def tree_shap_cache_release_binding(handle: PythonObject) raises -> PythonObject:
    shap_cpu.shap_cache_release(Int(py=handle))
    return PythonObject(0)


def register_shap_cache(mut m: PythonModuleBuilder) raises:
    m.def_function[tree_shap_cache_enabled_binding]("x_trees_tree_shap_cache_enabled")
    m.def_function[tree_shap_cache_create_binding]("x_trees_tree_shap_cache_create")
    m.def_function[tree_shap_cache_values_binding]("x_trees_tree_shap_cache_values")
    m.def_function[tree_shap_cache_release_binding]("x_trees_tree_shap_cache_release")
