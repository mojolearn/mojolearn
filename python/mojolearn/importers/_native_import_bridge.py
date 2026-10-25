# SPDX-License-Identifier: Apache-2.0
"""Import-time connections for the compiled ONNX artifact import owner."""
from types import BuiltinFunctionType


def _layout(cls):
    # glue: exact storage and descriptor identities, read once at import.
    mro = type.__dict__['__mro__'].__get__(cls)
    fields = tuple(type.__dict__['__dict__'].__get__(base) for base in mro)  # glue: class metadata
    descriptor = next(row['__dict__'] for row in fields if '__dict__' in row)  # glue: dictionary storage
    names = ('__getattribute__', '__setattr__', '__getattr__', '__del__', '__dict__',
             '_id', '_b', '_native', 'source', 'h', 'b', '_session_native', '_lifetime',
             '_program_execute', '_program_inspect', '_program_control', '_program_fit',
             '_program_class', 'inputs', 'outputs')
    hooks = tuple((base, name, row.get(name)) for base, row in zip(mro, fields) for name in names)  # glue: fixed storage hooks
    return dict(cls=cls, mro=mro, fields_get=descriptor.__get__, hooks=hooks)


def connect(session, program, imported, graph, unsupported):
    # glue: all functions retained here are compiled entries; no operation
    # calls this connector or asks a lazy module to resolve an attribute.
    import pathlib  # preload native_path's standard path storage metadata
    from .. import ops, _backend
    from .._native_family_bridge import shared_context
    from .onnx import _Native
    modes = {}
    entry = None
    for mode, connection in ops._SESSION_CONNECTIONS.items():  # glue: two fixed numeric modes
        if isinstance(connection, str):
            modes[mode] = connection
            continue
        try:
            binding = connection.binding
            connected = {name: getattr(binding, name) for name in  # glue: fixed native import exports
                         ('ops_import_model', 'onnx_import_file', 'onnx_import', 'onnx_buffers',
                          'ops_imported_run')}
            connected['ops_imported_run_device'] = getattr(binding, 'ops_imported_run_device', None)
            for name, function in connected.items():  # glue: compiled export admission at import
                if function is not None and type(function) is not BuiltinFunctionType:
                    raise TypeError('native artifact import requires compiled ' + name)
            modes[mode] = connected
            if entry is None:
                entry = connected['ops_import_model']
        except (ImportError, AttributeError, TypeError) as error:
            modes[mode] = type(error).__name__ + ': ' + str(error)
    if entry is None:
        from .._native_family_bridge import _unavailable
        entry = _unavailable(ImportError('rebuild x_sequence with ops_import_model'))
    return entry, dict(admission=shared_context(), modes=modes, connections=ops._SESSION_CONNECTIONS, backend=_backend,
        session=session, program=program, imported=imported, graph=graph, reader=_Native,
        unsupported=unsupported, layouts={cls: _layout(cls) for cls in  # glue: fixed native object layouts
            (session, program, imported, graph, _Native)},
        dtype_names=dict(f32='float32', f16='float16', bf16='bfloat16', f64='float64',
                         i64='int64', i32='int32', bool='bool'))
