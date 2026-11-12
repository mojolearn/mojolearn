# SPDX-License-Identifier: Apache-2.0
# Model-selection input preparation owns admission, byte copies and metadata.
# Never ask caller objects for shape, len, __array__, or DLPack protocols.
# The PACK half of the former `bindings/msel_inputs.mojo` (fence-flip split, 2026-11-12);
# the UNPACK half is `bindings/msel_inputs_unpack.mojo`.
from bindings.native_array_storage_door import native_array_from_bytes, native_array_from_values
from bindings.native_loaded_module import native_loaded_module
from std.memory import unsafe_memcpy
from core.msel_column_rows import msel_block_columns_to_rows
from bindings.msel_inputs_run import _pandas_promoted_dtype
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get
from bindings.py_attr import py_attr
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_arrays_door import classical_input
from bindings.classical_pandas import (
    classical_admit_value, native_pandas_exact, _pd_class, _pd_fields, _pd_get,
    _pd_values, _pd_locations,
)
from bindings.sequential_inputs import sequential_sequence, _promote
from bindings.native_torch_admission import native_torch_exact, native_torch_numpy
from bindings.native_buffer_owner_pack import native_buffer_view
from bindings.arena_array_metadata_pack import arena_array_metadata
from bindings.arena_materialize_native_door import _prepare, _owned, _empty, _cast, _slot
from bindings.pandas_arrow_native_unpack import pandas_arrow_exact, pandas_arrow_length, pandas_arrow_ndarray


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
