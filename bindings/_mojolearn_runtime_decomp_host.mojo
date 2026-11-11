# SPDX-License-Identifier: Apache-2.0
# Host owner of the decomposition doors (PCA, the decomposition family, linear-algebra decompositions, ALS, MCD,
# PLS, dictionary learning, random projection, incremental PCA, truncated SVD). Split out of
# bindings/_mojolearn_runtime_classical_host.mojo (2026-11-07 host split, docs/build/HOT_GROUPS.md section 6) so
# it compiles as its own merged group: the same compiled functions, registered under the same names.
# bindings/_mojolearn_tokenizer_host.mojo keeps a native refusal under each name, and import setup replaces it
# with this module's entry (python/mojolearn/host_surface.py HOST_IMPORT_ALIASES). The Python types these
# entries allocate stay registered by their existing owner, which import setup loads first.
from core.experiment_guards_linear import LINEAR_GUARDS_OK
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder, lookup_py_type_object
from std.reflection import reflect
from bindings.py_type_registry import py_type_absent, py_type_attach
from providers.python.pca_query_cache_pyapi import PCAQueryCache
from bindings.tsvd_owner_boundary import TSVDResidentOwner
from bindings.ipca_owner_boundary import IPCAResidentOwner
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from std.sys.compile import is_defined
from bindings.pca_native import pca_c
from bindings.decomp_native import decomp_c
from bindings.linalg_decomp_native import linalg_decomp_c
from bindings.als_native import als_c
from bindings.mcd_native import mcd_c
from bindings.pls_native import pls_c
from bindings.dict_learning_native import dict_learning_c
from bindings.jl_native import jl_c
from bindings.ipca_native import ipca_c
from bindings.tsvd_native import tsvd_c

comptime RUNTIME_DECOMP_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()


def runtime_decomp_host_numeric_mode_binding() raises -> PythonObject:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL, (
        "runtime decomp host requires MOJOLEARN_NUMERIC_IDENTICAL=1"
    )
    return PythonObject(GLOBAL_NUMERIC_MODE)


def runtime_decomp_host_vendor_binding() raises -> PythonObject:
    return PythonObject("cpu")


def runtime_decomp_host_column_binding() raises -> PythonObject:
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "runtime decomp host requires MOJOLEARN_COLUMN_CPU"
    )
    return PythonObject(column_name(TARGET_COLUMN))


def runtime_decomp_host_sabotage_binding() raises -> PythonObject:
    return PythonObject(RUNTIME_DECOMP_SABOTAGE)


def register_pca_query_cache(mut module: PythonModuleBuilder) raises:
    var existing: PythonObject
    try:
        existing = lookup_py_type_object[PCAQueryCache]()
    except error:
        if not py_type_absent(error, String(reflect[PCAQueryCache].name())):
            raise error
        _ = module.add_type[PCAQueryCache]("_PCAQueryCache")
        return
    py_type_attach(module.module, "_PCAQueryCache", existing)


def register_tsvd_resident_owner(mut module: PythonModuleBuilder) raises:
    var existing: PythonObject
    try:
        existing = lookup_py_type_object[TSVDResidentOwner]()
    except error:
        if not py_type_absent(error, String(reflect[TSVDResidentOwner].name())):
            raise error
        _ = module.add_type[TSVDResidentOwner]("_TSVDResidentOwner")
        return
    py_type_attach(module.module, "_TSVDResidentOwner", existing)


def register_ipca_resident_owner(mut module: PythonModuleBuilder) raises:
    var existing: PythonObject
    try:
        existing = lookup_py_type_object[IPCAResidentOwner]()
    except error:
        if not py_type_absent(error, String(reflect[IPCAResidentOwner].name())):
            raise error
        _ = module.add_type[IPCAResidentOwner]("_IPCAResidentOwner")
        return
    py_type_attach(module.module, "_IPCAResidentOwner", existing)


def build__mojolearn_runtime_decomp_host() -> PythonObject:
    comptime assert LINEAR_GUARDS_OK, "invalid integrated experiment configuration (linear guards)"
    try:
        var module = PythonModuleBuilder("_mojolearn_runtime_decomp_host")
        register_pca_query_cache(module)
        register_tsvd_resident_owner(module)
        register_ipca_resident_owner(module)
        module.def_py_c_function(pca_c, "pca_native_call")
        module.def_py_c_function(decomp_c, "decomp_native_call")
        module.def_py_c_function(linalg_decomp_c, "linalg_decomp_native_call")
        module.def_py_c_function(als_c, "als_native_call")
        module.def_py_c_function(mcd_c, "mcd_native_call")
        module.def_py_c_function(pls_c, "pls_native_call")
        module.def_py_c_function(dict_learning_c, "dict_learning_native_call")
        module.def_py_c_function(jl_c, "jl_native_call")
        module.def_py_c_function(ipca_c, "ipca_native_call")
        module.def_py_c_function(tsvd_c, "tsvd_native_call")
        module.def_function[runtime_decomp_host_numeric_mode_binding]("runtime_decomp_host_numeric_mode")
        module.def_function[runtime_decomp_host_vendor_binding]("runtime_decomp_host_vendor")
        module.def_function[runtime_decomp_host_column_binding]("runtime_decomp_host_column")
        module.def_function[runtime_decomp_host_sabotage_binding]("runtime_decomp_host_sabotage")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_runtime_decomp_host: ", error))
