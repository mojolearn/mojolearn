# SPDX-License-Identifier: Apache-2.0
"""Import-time metadata for the native reader, without requiring NumPy or Session."""
from types import BuiltinFunctionType
from ._native_import_bridge import _layout


def connect(graph, reader, node, tensor, resolved, unsupported, attrs_default, dtype_names, refused_names):
    # glue: connect the reader's own native table, not an execution Session.
    from types import MethodType
    from dataclasses import FrozenInstanceError
    from ._table import SUPPORTED_OPS
    import ctypes
    import pathlib
    from array import array
    from collections import OrderedDict, defaultdict
    from .. import _backend
    from .._native_family_bridge import _unavailable
    scalars = (str, bytes, bool, int, float)
    containers = (list, range, str, bytes, bytearray, memoryview, array,
                  type(iter([])), type(iter(())), type(iter(range(0))), type(reversed([])))
    np = None
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
    record_entry = None
    for mode in ('identical', 'fast'):  # glue: the two existing numeric mode connections
        try:
            binding = _backend.binding('_mojolearn_x_sequence', mode)
            native = binding.ops_import_model
            record = binding.ops_graph_record
            factory = binding.onnx_native_api_v1
            if type(native) is not BuiltinFunctionType or type(factory) is not BuiltinFunctionType or type(record) is not BuiltinFunctionType:
                raise TypeError('ONNX reader requires compiled provider entries')
            actual = factory.__self__
            aliases.append((binding, actual))
            bindings[mode] = actual
            if entry is None:
                entry = native
                record_entry = record
        except (ImportError, AttributeError, TypeError) as error:
            bindings[mode] = type(error).__name__ + ': ' + str(error)
    if entry is None:
        entry = _unavailable(ImportError('rebuild x_sequence with onnx_native_api_v1'))
    context = dict(shape_scalars=scalars, shape_names=scalars, shape_containers=containers,
                      shape_mappings=(dict, OrderedDict, defaultdict), bindings=bindings, aliases=tuple(aliases),
                      backend=vars(_backend), graph=graph, reader=reader, node=node, tensor=tensor,
                      resolved=resolved, unsupported=unsupported, byte_type=ctypes.c_ubyte,
                      dtypes=('f32', 'f16', 'bf16', 'f64', 'i64', 'i32', 'bool'), providers={},
                      layouts={}, records={
                          tensor: ('name', 'dtype', 'shape', 'data'),
                          node: ('op', 'domain', 'name', 'inputs', 'outputs', 'attrs'),
                          graph: ('opset', 'inputs', 'outputs', 'nodes', 'initializers', 'source'),
                          resolved: ('shapes', 'dtypes', 'folded', 'no_rule', 'problems')},
                      attrs_default=attrs_default, dtype_names=dtype_names, refused_names=refused_names,
                      method_type=MethodType, native_entry=entry,
                      supported_ops=SUPPORTED_OPS, numpy=np, frozen_error=FrozenInstanceError, array_type=array,
                      record_names={cls: cls.__qualname__ for cls in (tensor, node, graph, resolved)},  # glue: native record names
                      numpy_array_interface=None if np is None else vars(np.ndarray)["__array_interface__"].__get__,
                      numpy_frombuffer=None if np is None else np.frombuffer, numpy_uint8=None if np is None else np.uint8)  # glue: fixed native storage layouts

    if type(entry) is BuiltinFunctionType:
        for cls in (tensor, node, graph, resolved):  # glue: compiled native metadata protocols
            names = ('__repr__', '__eq__', '__hash__', '__setattr__', '__delattr__') if cls in (tensor, node) else ('__repr__', '__eq__')
            for name in names:  # glue: native hooks replace generated Python dataclass methods
                bound = MethodType(record_entry, (name, context))
                setattr(cls, name, entry('record_descriptor', bound, None, context))
    context['layouts'] = {cls: _layout(cls) for cls in (graph, reader, node, tensor, resolved)}  # glue: pin final native record storage
    for cls in (graph, reader, node, tensor, resolved):  # glue: pin record protocols and fields after native hook setup
        layout = context['layouts'][cls]
        names = context['records'].get(cls, ('b', 'h', 'views')) + ('__repr__', '__eq__', '__hash__', '__delattr__')
        layout['hooks'] += tuple((base, name, vars(base).get(name)) for base in cls.__mro__ for name in names)  # glue: fixed record hooks
    return entry, context
