# SPDX-License-Identifier: Apache-2.0
# Logical host owner for classical. Release builds merge this builder
# with the other host builders; development builds give it an independent
# source/cache boundary. TreesResident is registered and finalized only in this owner
# (registered once per process; the registrars below).
# 2026-11-07 host split (docs/build/HOT_GROUPS.md section 6): the tree and
# family operations of classical_native_call run in their own doors
# (_mojolearn_runtime_trees_host, _mojolearn_runtime_family_host), and the
# preprocessing, decomposition and linear doors are their own bindings
# (_mojolearn_runtime_prep_host, _mojolearn_runtime_decomp_host,
# _mojolearn_runtime_linear_host), each its own merged group. The Python types
# their code allocates stay registered HERE, unchanged: import setup loads this
# owner before them (python/mojolearn/host_surface.py HOST_IMPORT_ALIASES order).
from core.experiment_guards_linear import LINEAR_GUARDS_OK
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from std.sys.compile import is_defined
from std.python.bindings import lookup_py_type_object
from std.reflection import reflect
from bindings.py_type_registry import py_type_absent, py_type_attach
from bindings.trees_lifetime import TreesResident
from bindings.trees_resident_table import TreesResidentDeath
from bindings.trees_explainer_native import TreeShapOwner
from providers.python.compiled_binding_snapshot_pyapi import CompiledBindingSnapshot
from bindings.classical_native import classical_c
from bindings.classical_attributes import classical_attribute_c, classical_attribute_descriptor_c
from bindings.host_archive_lifetime import host_archive_finalize_c
from providers.python.preprocessing_input_pyapi import register_scaler_snapshot
from bindings.linalg_ops_buffers import register_lin_buffer_pin

comptime RUNTIME_CLASSICAL_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()


def runtime_classical_host_numeric_mode_binding() raises -> PythonObject:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL, (
        "classical host requires MOJOLEARN_NUMERIC_IDENTICAL=1"
    )
    return PythonObject(GLOBAL_NUMERIC_MODE)


def runtime_classical_host_vendor_binding() raises -> PythonObject:
    return PythonObject("cpu")


def runtime_classical_host_column_binding() raises -> PythonObject:
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "classical host requires MOJOLEARN_COLUMN_CPU"
    )
    return PythonObject(column_name(TARGET_COLUMN))


def runtime_classical_host_sabotage_binding() raises -> PythonObject:
    return PythonObject(RUNTIME_CLASSICAL_SABOTAGE)


# This host's own Python types. Its PyInit can run twice in one process (see
# bindings/py_type_registry.mojo); each registrar creates the type on the first
# initialization and attaches the existing type object on any later one.
def register_trees_resident(mut module: PythonModuleBuilder) raises:
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[TreesResident]()
    except error:
        if not py_type_absent(error, String(reflect[TreesResident].name())):
            raise error
        _ = module.add_type[TreesResident]("_TreesResident")
        return
    py_type_attach(module.module, "_TreesResident", existing)


def register_trees_resident_death(mut module: PythonModuleBuilder) raises:
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[TreesResidentDeath]()
    except error:
        if not py_type_absent(error, String(reflect[TreesResidentDeath].name())):
            raise error
        _ = module.add_type[TreesResidentDeath]("_TreesResidentDeath")
        return
    py_type_attach(module.module, "_TreesResidentDeath", existing)


def register_tree_shap_owner(mut module: PythonModuleBuilder) raises:
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[TreeShapOwner]()
    except error:
        if not py_type_absent(error, String(reflect[TreeShapOwner].name())):
            raise error
        _ = module.add_type[TreeShapOwner]("_TreeShapOwner")
        return
    py_type_attach(module.module, "_TreeShapOwner", existing)


def register_compiled_binding_snapshot(mut module: PythonModuleBuilder) raises:
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[CompiledBindingSnapshot]()
    except error:
        if not py_type_absent(error, String(reflect[CompiledBindingSnapshot].name())):
            raise error
        _ = module.add_type[CompiledBindingSnapshot]("_CompiledBindingSnapshot")
        return
    py_type_attach(module.module, "_CompiledBindingSnapshot", existing)


def build__mojolearn_runtime_classical_host() -> PythonObject:
    comptime assert LINEAR_GUARDS_OK, "invalid integrated experiment configuration (linear guards)"
    try:
        var module = PythonModuleBuilder("_mojolearn_runtime_classical_host")
        module.def_py_c_function(classical_c, "classical_native_call")
        module.def_py_c_function(classical_attribute_c, "classical_attribute_call")
        module.def_py_c_function(classical_attribute_descriptor_c, "classical_attribute_descriptor")
        module.def_py_c_function(host_archive_finalize_c, "host_archive_finalize_call")
        # Values this host allocates with PythonObject(alloc=...): resident trees, the Tree SHAP
        # owner and the compiled-binding snapshots of the SVM, hierarchy and x_neighbors descriptor
        # owners. Registered once per process here (the manifold and tokenizer hosts allocate
        # snapshots through the same x_neighbors admission and must not register them a second
        # time). This PyInit can run twice in one process (`mojolearn._host.` at package connection,
        # then `mojolearn.host.` through the merged finder); a second run attaches the existing
        # type objects instead of aborting (bindings/py_type_registry.mojo).
        register_trees_resident(module)
        # The tree family's resident-table death record (ne1-a-trees).
        register_trees_resident_death(module)
        register_tree_shap_owner(module)
        register_compiled_binding_snapshot(module)
        register_scaler_snapshot(module)
        register_lin_buffer_pin(module)
        module.def_function[runtime_classical_host_numeric_mode_binding]("runtime_classical_host_numeric_mode")
        module.def_function[runtime_classical_host_vendor_binding]("runtime_classical_host_vendor")
        module.def_function[runtime_classical_host_column_binding]("runtime_classical_host_column")
        module.def_function[runtime_classical_host_sabotage_binding]("runtime_classical_host_sabotage")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_runtime_classical_host: ", error))
