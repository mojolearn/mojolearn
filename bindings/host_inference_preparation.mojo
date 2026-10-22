# SPDX-License-Identifier: Apache-2.0
# Validation of pandas' promoted native storage before the f32 prediction
# cast. Finite f64 values that overflow f32 remain valid, as in the existing
# wrapper's separate pre-cast validation and inference preparation.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get
from bindings.classical_pandas import _pd_class, _pd_fields, _pd_get, _pd_values
from bindings.classical_arrays import classical_address
from bindings.sequential_inputs import _promote
from core.host_inference_impl import host_inference_raw_finite


def host_inference_pandas_finite(source: PythonObject, context: PythonObject) raises:
    # A Series is rejected by the ordinary 2-D input check. Only DataFrames
    # can reach a valid CD prediction and need this pre-cast matrix check.
    if py_type(source) is not _pd_class("pandas.core.frame", "DataFrame"):
        return
    var manager = py_get(_pd_fields(source), "_mgr")
    if py_type(manager) is not _pd_class("pandas.core.internals.managers", "BlockManager"):
        raise Error("TypeError: pandas prediction requires native block metadata")
    var blocks = _pd_get("pandas._libs.internals", "BlockManager", "blocks", manager)
    if py_type(blocks) is not Python.import_module("builtins").tuple:
        raise Error("TypeError: pandas prediction requires native storage blocks")
    var owned = Python.list()
    var kind = String()
    var width = 0
    var objects = False
    var booleans = False
    var numbers = False
    for i in range(len(blocks)):  # driver-loop: native pandas promotion metadata
        var block = blocks[i]
        var cls = py_type(block)
        if cls is not _pd_class("pandas.core.internals.blocks", "NumpyBlock") and cls is not _pd_class("pandas.core.internals.blocks", "NumericBlock") and cls is not _pd_class("pandas.core.internals.blocks", "ObjectBlock") and cls is not _pd_class("pandas.core.internals.blocks", "ExtensionBlock"):
            raise Error("TypeError: pandas requires ordinary native blocks")
        var pair = _pd_values(_pd_get("pandas._libs.internals", "Block", "values", block), False, context)
        var value = pair[0]
        var incoming = String(value.dtype.kind)
        objects = objects or incoming == "O" or Bool(py=pair[1])
        if incoming != "O":
            if incoming == "c":
                raise Error("Complex data not supported")
            _promote(kind, width, incoming, Int(py=value.itemsize))
            booleans = booleans or incoming == "b"
            numbers = numbers or incoming != "b"
        _ = py_call(owned.append, value)
    # The existing validator scans only promoted float32/float64 matrices;
    # object promotion and an unpromoted float16 matrix are not pre-scanned.
    if objects or (booleans and numbers) or kind != "f" or width < 4:
        return
    for i in range(len(owned)):  # driver-loop: one native finite scan per float block
        var value = owned[i]
        if String(value.dtype.kind) != "f":
            continue
        var rank = Int(py=value.ndim)
        if rank != 1 and rank != 2:
            raise Error("pandas numeric block rank must be one or two")
        host_inference_raw_finite(Int(py=classical_address(value)), Int(py=value.strides[0]),
            0 if rank == 1 else Int(py=value.strides[1]), Int(py=value.shape[0]),
            1 if rank == 1 else Int(py=value.shape[1]), Int(py=value.itemsize), Bool(py=value.dtype.isnative))
        _ = value^
    _ = owned^
    _ = manager^
