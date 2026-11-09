# SPDX-License-Identifier: Apache-2.0
# The decomposition owners' X admission (PCA, TruncatedSVD, IncrementalPCA,
# KernelPCA, FactorAnalysis, FastICA, NMF, LatentDirichletAllocation, the
# random projections): the one admission path (bindings/admitted_input.mojo,
# admit_x) with scikit-learn's refusals naming the estimator, after an exact
# SciPy sparse container is densified from its storage, as the Python shells
# densified it before converting. No cast or scan of its own.
from std.python import PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_get, py_type
from bindings.classical_arrays import classical_address
from bindings.admitted_input import AdmittedX, admit_x, admit_x_unrecorded
from bindings.host_sparse_native import host_sparse_format, host_sparse_dense

# TOMBSTONE: decomp_copied (a second classical_view / memoryview of X to rediscover the input_copied_ flag the admission already knows) deleted 2026-11-08 by lane a3-admit-decomp-prep-metrics; code recoverable at f7d10d2b09.
# Restore: experiments/removed/A3_ADMISSION_HELPERS.patch. The flag is decomp_input_copied below.


def decomp_admit(source: PythonObject, est: String, context: PythonObject,
                 device_x: Bool = False) raises -> AdmittedX:
    """X as C-order float32 (the caller's buffer when it already is one,
    `borrowed`); refusals name `est`. A SciPy sparse X is densified first
    and admitted with the plain rank/empty/finite wording. `device_x`
    (lane a6): a model-selection fold keeps no host words when its store
    ids resolve (`x.store.active()`): the door's device entry gathers it."""
    var format = host_sparse_format(source)
    if format.byte_length() > 0:
        var dense = host_sparse_dense(source, format, context)
        var array = admit_x(dense, 2, "X", context)
        var address = Int(py=classical_address(array))
        var borrowed = address == Int(py=classical_address(dense))
        var x = AdmittedX(array, source, address, Int(py=array.shape[0]), Int(py=array.shape[1]), 2, borrowed)
        _ = dense^
        return x^
    return admit_x_unrecorded(source, est, context, device_x=device_x)


def decomp_input(source: PythonObject, est: String, context: PythonObject) raises -> PythonObject:
    """X as C-order float32 (the caller's buffer when it already is one);
    refusals name `est`."""
    var x = decomp_admit(source, est, context)
    return x.array


def decomp_input_copied(x: AdmittedX, context: PythonObject) raises -> Bool:
    """The public `input_copied_` flag of an admitted X: False only when the
    kernels read the caller's own ndarray, Array or native buffer in place.
    Sequences, pandas and Torch inputs count as copied (they are
    materialized); a densified sparse X counts as the dense array its storage
    holds (copied unless that is already float32)."""
    var b = native_loaded_module("builtins")
    if host_sparse_format(x.source).byte_length() > 0:
        return not x.borrowed
    var t = py_type(x.source)
    # NumPy's keys are in the context only once NumPy is loaded
    var ndarray = py_get(context, "ndarray")
    var direct = ((ndarray is not None and t is ndarray) or t is context["array_class"] or t is b.memoryview or t is context["array_type"]
                  or t is b.bytearray)
    return not direct or not x.borrowed
