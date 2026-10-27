# SPDX-License-Identifier: Apache-2.0
"""Import-time metadata for the native reader, without requiring NumPy or Session."""
from types import BuiltinFunctionType
from ._native_import_bridge import _layout


def connect(graph, reader, node, tensor, resolved, unsupported):
    # glue: connect the reader's own native table, not an execution Session.
    import ctypes
    import pathlib
    from array import array
    from collections import OrderedDict, defaultdict
    from .. import _backend
    from .._native_family_bridge import _unavailable
    scalars = (str, bytes, bool, int, float)
    containers = (list, range, str, bytes, bytearray, memoryview, array,
                  type(iter([])), type(iter(())), type(iter(range(0))), type(reversed([])))
    try:
        import numpy as np
    except ModuleNotFoundError as error:
        if error.name != 'numpy':
            raise
    else:
        scalar_types = tuple(np.dtype(name).type for name in  # glue: optional native scalar metadata at setup
            ('bool', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64',
             'uint64', 'float16', 'float32', 'float64', 'longdouble')) + (np.str_, np.bytes_)
        scalars += scalar_types
        containers += (np.ndarray,)
    _backend.default_mode()
    bindings, aliases, entry = {}, [], None
    for mode in ('identical', 'fast'):  # glue: the two existing numeric mode connections
        try:
            binding = _backend.binding('_mojolearn_x_sequence', mode)
            native = binding.ops_import_model
            factory = binding.onnx_native_api_v1
            if type(native) is not BuiltinFunctionType or type(factory) is not BuiltinFunctionType:
                raise TypeError('ONNX reader requires compiled provider entries')
            actual = factory.__self__
            aliases.append((binding, actual))
            bindings[mode] = actual
            if entry is None:
                entry = native
        except (ImportError, AttributeError, TypeError) as error:
            bindings[mode] = type(error).__name__ + ': ' + str(error)
    if entry is None:
        entry = _unavailable(ImportError('rebuild x_sequence with onnx_native_api_v1'))
    return entry, dict(shape_scalars=scalars, shape_names=scalars, shape_containers=containers,
                      shape_mappings=(dict, OrderedDict, defaultdict), bindings=bindings, aliases=tuple(aliases),
                      backend=vars(_backend), graph=graph, reader=reader, node=node, tensor=tensor,
                      resolved=resolved, unsupported=unsupported, byte_type=ctypes.c_ubyte,
                      dtypes=('f32', 'f16', 'bf16', 'f64', 'i64', 'i32', 'bool'), providers={},
                      layouts={cls: _layout(cls) for cls in (graph, reader)})  # glue: fixed native storage layouts
