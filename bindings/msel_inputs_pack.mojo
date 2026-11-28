# SPDX-License-Identifier: Apache-2.0
# Model-selection input preparation owns admission, byte copies and metadata.
# Never ask caller objects for shape, len, __array__, or DLPack protocols.
# The PACK half of the former `bindings/msel_inputs.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `bindings/msel_inputs_unpack.mojo`.
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get
from bindings.classical_pandas_door import (
    classical_admit_value, native_pandas_exact, _pd_class, _pd_fields, _pd_get,
    _pd_values, _pd_locations,
)


def _pandas_blocks(value: PythonObject) raises -> PythonObject:
    var series = py_type(value) is _pd_class("pandas.core.series", "Series")
    var manager = py_get(_pd_fields(value), "_mgr")
    var blocks = PythonObject(None)
    if series:
        if py_type(manager) is not _pd_class("pandas.core.internals.managers", "SingleBlockManager"):
            raise Error("TypeError: model selection requires ordinary pandas Series storage")
        blocks = py_get(_pd_fields(manager), "blocks")
    else:
        if py_type(manager) is not _pd_class("pandas.core.internals.managers", "BlockManager"):
            raise Error("TypeError: model selection requires ordinary pandas DataFrame storage")
        blocks = _pd_get("pandas._libs.internals", "BlockManager", "blocks", manager)
    if py_type(blocks) is not py_type(Python.tuple()) or (series and len(blocks) != 1):
        raise Error("ValueError: pandas input has no supported block storage")
    for i in range(len(blocks)):  # driver-loop: exact storage classes before native descriptor reads
        var cls = py_type(blocks[i])
        if (cls is not _pd_class("pandas.core.internals.blocks", "NumpyBlock")
                and cls is not _pd_class("pandas.core.internals.blocks", "NumericBlock")
                and cls is not _pd_class("pandas.core.internals.blocks", "ObjectBlock")
                and cls is not _pd_class("pandas.core.internals.blocks", "ExtensionBlock")):
            raise Error("TypeError: model selection cannot invoke pandas block subclass hooks")
    return blocks
