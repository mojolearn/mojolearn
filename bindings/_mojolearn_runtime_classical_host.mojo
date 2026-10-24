# SPDX-License-Identifier: Apache-2.0
# Logical host owner for classical. Release builds merge this builder
# with the other host builders; development builds give it an independent
# source/cache boundary. TreesResident is registered, allocated and finalized only in this owner.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from checks.kernel_matrix import COLUMN_CPU, TARGET_COLUMN, column_name
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from std.sys.compile import is_defined
from bindings.trees_lifetime import TreesResident
from bindings.classical_native import classical_c
from bindings.classical_attributes import classical_attribute_c, classical_attribute_descriptor_c
from bindings.host_archive_lifetime import host_archive_finalize_c
from bindings.host_svr_native import host_svr_attribute_c
from bindings.host_svc_native import host_svc_attribute_c
from bindings.pca_native import pca_c
from bindings.scaler_native import scaler_fit_c
from bindings.scaler_lifecycle_native import scaler_lifecycle_c, scaler_attribute_c
from bindings.minmax_native import minmax_lifecycle_c
from bindings.maxabs_native import maxabs_lifecycle_c
from bindings.prep_stateless_native import prep_stateless_c
from bindings.qn_native import qn_c
from bindings.logistic_native import logistic_c
from bindings.decomp_native import decomp_c
from bindings.linalg_decomp_native import linalg_decomp_c
from bindings.als_native import als_c
from bindings.mcd_native import mcd_c
from bindings.pls_native import pls_c
from bindings.dict_learning_native import dict_learning_c
from bindings.jl_native import jl_c
from bindings.ipca_native import ipca_c
from bindings.tsvd_native import tsvd_c
from bindings.prep_classify_native import prep_classify_c
from bindings.prep_owner_native import prep_owner_c
from bindings.x_linear_native import x_linear_c
from bindings.str_labels_native import str_labels_c
from bindings.robust_native import robust_lifecycle_c

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


def build__mojolearn_runtime_classical_host() -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_runtime_classical_host")
        module.def_py_c_function(classical_c, "classical_native_call")
        module.def_py_c_function(classical_attribute_c, "classical_attribute_call")
        module.def_py_c_function(classical_attribute_descriptor_c, "classical_attribute_descriptor")
        module.def_py_c_function(host_archive_finalize_c, "host_archive_finalize_call")
        module.def_py_c_function(host_svr_attribute_c, "host_svr_attribute_call")
        module.def_py_c_function(host_svc_attribute_c, "host_svc_attribute_call")
        module.def_py_c_function(pca_c, "pca_native_call")
        module.def_py_c_function(scaler_fit_c, "standard_scaler_fit_native")
        module.def_py_c_function(scaler_lifecycle_c, "standard_scaler_native_call")
        module.def_py_c_function(minmax_lifecycle_c, "minmax_scaler_native_call")
        module.def_py_c_function(maxabs_lifecycle_c, "maxabs_scaler_native_call")
        module.def_py_c_function(prep_stateless_c, "prep_stateless_native_call")
        module.def_py_c_function(qn_c, "qn_native_call")
        module.def_py_c_function(logistic_c, "logistic_native_call")
        module.def_py_c_function(decomp_c, "decomp_native_call")
        module.def_py_c_function(linalg_decomp_c, "linalg_decomp_native_call")
        module.def_py_c_function(als_c, "als_native_call")
        module.def_py_c_function(mcd_c, "mcd_native_call")
        module.def_py_c_function(pls_c, "pls_native_call")
        module.def_py_c_function(dict_learning_c, "dict_learning_native_call")
        module.def_py_c_function(jl_c, "jl_native_call")
        module.def_py_c_function(ipca_c, "ipca_native_call")
        module.def_py_c_function(tsvd_c, "tsvd_native_call")
        module.def_py_c_function(prep_classify_c, "prep_classify_native_call")
        module.def_py_c_function(prep_owner_c, "prep_encode_native_call")
        module.def_py_c_function(x_linear_c, "x_linear_native_call")
        module.def_py_c_function(str_labels_c, "str_labels_native_call")
        module.def_py_c_function(robust_lifecycle_c, "robust_scaler_native_call")
        module.def_py_c_function(scaler_attribute_c, "standard_scaler_attribute_call")
        _ = module.add_type[TreesResident]("_TreesResident")
        module.def_function[runtime_classical_host_numeric_mode_binding]("runtime_classical_host_numeric_mode")
        module.def_function[runtime_classical_host_vendor_binding]("runtime_classical_host_vendor")
        module.def_function[runtime_classical_host_column_binding]("runtime_classical_host_column")
        module.def_function[runtime_classical_host_sabotage_binding]("runtime_classical_host_sabotage")
        return module.finalize()
    except error:
        abort(String("failed to create _mojolearn_runtime_classical_host: ", error))
