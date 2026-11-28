# SPDX-License-Identifier: Apache-2.0
# AgglomerativeClustering's public lifecycle, setting admission and captured X.
# Public values remain pins until their established conversion points. Typed
# native options and requests carry the fit/query execution between providers.
# The PACK half of the former `providers/python/hierarchy_public_lifecycle_pyapi.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `providers/python/hierarchy_public_lifecycle_pyapi_unpack.mojo`.
from std.math import isfinite, floor
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.native_array_view_pack import native_array_ndarray_view
from providers.python.estimator_lifecycle_pyapi_unpack import (
    estimator_outside, estimator_feature_check, estimator_fit_begin, estimator_fit_end, estimator_fit_complete,
)


from providers.python.hierarchy_public_lifecycle_pyapi_unpack import (
    HierarchyPublicCall,
)
from bindings.cluster_keys_unpack import cluster_keys
from bindings.py_balanced import py_item


def hierarchy_public_finish(model: PythonObject, result: PythonObject,
                            call: HierarchyPublicCall, context: PythonObject) raises -> PythonObject:
    if call.fitting:
        estimator_fit_complete(model, call.source, False, context)
    var nk = cluster_keys()
    if call.outside and py_type(result) is py_item(context, nk[].k_array_class) and py_get(context, nk[].k_ndarray) is not None:
        return native_array_ndarray_view(result, context)
    return result
