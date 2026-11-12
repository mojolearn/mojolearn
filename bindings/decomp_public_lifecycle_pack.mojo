# SPDX-License-Identifier: Apache-2.0
# The public lifecycle of the decomposition expansion's estimators (the
# former `_mode` class-creation wrappers), at each of their native doors:
# decomp_c, ipca_c, dict_learning_c, mcd_c and als_c. Every public method
# of IncrementalPCA, the random projections, NMF, FastICA, FactorAnalysis,
# LatentDirichletAllocation, the dictionary learners, the sparse PCAs,
# SparseCoder, MinCovDet, EllipticEnvelope and AlternatingLeastSquares is
# one native call that runs the fitted check (NotFittedError), scikit-learn's
# X refusals and width / feature-name checks, fit bookkeeping and NumPy
# results for callers outside the package (bindings/public_lifecycle_owner.mojo).
# The PACK half of the former `bindings/decomp_public_lifecycle.mojo` (fence-flip split, 2026-11-12);
# the UNPACK half is `bindings/decomp_public_lifecycle_unpack.mojo`.
from std.python import Python, PythonObject
from bindings.py_balanced import py_get, py_fields, py_contains
from bindings.public_lifecycle_owner_door import (
    lifecycle_begin, lifecycle_end, lifecycle_complete, lifecycle_result, lifecycle_check_width,
    lifecycle_check_x, lifecycle_x_shape, lifecycle_depth_value,
)
from bindings.classical_validate import classical_estimator_name


def decomp_public_finish(model: PythonObject, method: String, args: PythonObject, result: PythonObject,
                         kind: String, context: PythonObject) raises -> PythonObject:
    """After a successful algorithm: fit bookkeeping and the caller's result."""
    if method.byte_length() == 0:
        return result
    var source = args[0] if len(args) > 0 else PythonObject(None)
    lifecycle_complete(model, method, source, False, context)
    return lifecycle_result(model, method, result, kind, context)
