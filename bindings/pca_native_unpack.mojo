# SPDX-License-Identifier: Apache-2.0
# The native PCA owner: every public PCA operation is one call into pca_c.
from std.python import Python, PythonObject
from bindings.cpython_table import cpython_table
from bindings.native_loaded_module import native_loaded_module
from bindings.optimizer_public_words_unpack import PYTHON_ERROR_PENDING, optimizer_python_error
from std.python._cpython import PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.py_balanced import py_call, py_get, py_type
from bindings.sequential_admission import sequential_admit_value
from providers.python.linear_metadata_boundary_door import classical_is_fitted
from bindings.classical_attributes import classical_connect_numpy
from bindings.causal_exception_unpack import causal_borrowed_item, causal_set_exception
from bindings.pca_metadata import pca_dictionary, pca_fields, pca_initialize, pca_get_params, pca_set_params, pca_entries
from bindings.pca_fit_native import pca_fit_operation
from bindings.pca_query_native_unpack import pca_query_operation
from bindings.pca_descriptor_boundary import pca_mode, pca_vendor
from bindings.pca_archive_pack import pca_archive_save, pca_archive_load


def _pca_admitted(b: PythonObject, model: PythonObject, loading: Bool, context: PythonObject) raises -> Bool:
    # PCA itself or an ordinary subclass of it (HostPCA, a user's class):
    # the class's MRO, read through type's own descriptor, holds PCA.
    # `b` is the builtins module the entry already admitted this call (one
    # module check per call, not two).
    var cls = model if loading else py_type(model)
    if py_type(cls) is not b.type:
        return False
    var target = context["pca_class"]
    if py_type(target) is not b.type:
        return False
    # CPython's own subtype test walks the same tp_mro by identity in one
    # native call (no MRO tuple fetch or per-row Python indexing).
    # Resolved once per process in the shared table (pass 2), not by name per call.
    var t = cpython_table()
    return t[].PyType_IsSubtype(cls._obj_ptr, target._obj_ptr) != 0
