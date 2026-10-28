# SPDX-License-Identifier: Apache-2.0
"""Source-only artifact boundary checks, without importing the product."""
import ast
import importlib.util
from pathlib import Path
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'python/mojolearn'


def test_artifact_setup_passes_raw_values_to_one_native_owner():
    source = (PACKAGE / 'importers/_program.py').read_text()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_import')
    assert len(node.body) == 1 and isinstance(node.body[0], ast.Return)
    calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)]
    assert len(calls) == 1 and calls[0].func.id == '_IMPORT_NATIVE'
    seen = []
    context = object()
    namespace = {'_IMPORT_CONTEXT': context, '_IMPORT_NATIVE': lambda *args: seen.append(args) or 'native'}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<import boundary>', 'exec'), namespace)
    args = tuple(object() for _ in range(4))
    assert namespace['_import'](*args) == 'native'
    assert seen == [args + (context,)]
    assert 's.close()' not in ast.get_source_segment(source, node)
    # Preserve supported Torch and ModelProto behavior visibly; don't turn
    # the native helper into a false claim about the whole public boundary.
    public = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'import_model')
    assert any(isinstance(n, ast.If) for n in ast.walk(public))
    assert '_export_torch(source, path, shapes, example_inputs)' in source
    assert 'source.SerializeToString()' in source


def test_both_sequence_bindings_register_the_same_native_setup():
    for name in ('_mojolearn_x_sequence.mojo', '_mojolearn_x_sequence_host.mojo'):
        source = (ROOT / 'bindings' / name).read_text()
        assert 'from providers.python.autograd_import_frontend_pyapi import ops_import_model_c' in source
        assert 'm.def_py_c_function(ops_import_model_c, "ops_import_model")' in source
    source = (ROOT / 'providers/python/autograd_import_frontend_pyapi.mojo').read_text()
    for prohibited in ('torch.onnx', 'SerializeToString', 'session._sid(', 'ImportedProgram(', 'Session(', 'buffer_address(', 'Python.import_module('):
        assert prohibited not in source
    assert 'ops_session_initialize_py(session,' in source
    assert 'neural_descriptor_execute(owner[].provider, PROGRAM_IMPORT,' in source
    assert '_program_initialize(target, session, info)' in source
    assert 'downcast_value_ptr[NativeDescriptorResource]()[].close()' in source
    owner = (ROOT / 'autograd/program_import_request.mojo').read_text()
    assert 'onnx_import_file[E, KIND]' in owner and 'onnx_buffers(request[].reader)' in owner
    metadata = (ROOT / 'providers/python/autograd_import_metadata_pyapi.mojo').read_text()
    assert 'py_call(b.ValueError.__new__, category,' in metadata
    assert '"PyErr_SetObject"' in metadata


def test_import_connections_retain_only_native_entries_and_real_class_layouts(monkeypatch):
    package = types.ModuleType('_import_fixture')
    package.__path__ = []
    importers = types.ModuleType(package.__name__ + '.importers')
    importers.__path__ = []
    session, program, imported, graph, reader = [type(name, (), {}) for name in
        ('Session', 'Program', 'ImportedProgram', 'GraphIR', '_Native')]
    binding = types.SimpleNamespace(**{name: len for name in ('ops_import_model', 'onnx_import_file',
        'onnx_import', 'onnx_buffers', 'ops_imported_run', 'ops_imported_run_device')})
    ops = types.ModuleType(package.__name__ + '.ops')
    ops._SESSION_CONNECTIONS = {'identical': types.SimpleNamespace(binding=binding),
                                'fast': 'ImportError: optional mode absent'}
    backend = types.ModuleType(package.__name__ + '._backend')
    family = types.ModuleType(package.__name__ + '._native_family_bridge')
    family.shared_context = lambda: {'admission': 'test'}
    import numpy as np
    optional_numpy = types.ModuleType(package.__name__ + '._optional_numpy')
    optional_numpy.require_numpy = lambda _operation: np
    onnx = types.ModuleType(importers.__name__ + '.onnx')
    onnx._Native = reader
    for module in (package, importers, ops, backend, family, optional_numpy, onnx):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    package.ops, package._backend = ops, backend
    spec = importlib.util.spec_from_file_location(importers.__name__ + '.bridge', PACKAGE / 'importers/_native_import_bridge.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    entry, context = module.connect(session, program, imported, graph, ValueError)
    assert entry is len
    assert context['modes']['fast'].startswith('ImportError:')
    assert set(context['layouts']) == {session, program, imported, graph, reader}
    for cls, layout in context['layouts'].items():
        value = cls()
        assert layout['fields_get'](value, cls) is value.__dict__
        assert layout['mro'] == cls.__mro__
    assert all(type(fn) is types.BuiltinFunctionType for fn in context['modes']['identical'].values())


def test_native_shape_admission_precedes_protocol_sensitive_conversion():
    source = (ROOT / 'providers/python/autograd_import_metadata_pyapi.mojo').read_text()
    body = source[source.index('def _shape_words('):source.index('def _raise_object(')]
    assert 'Python mapping callbacks are unsupported' in body
    assert body.index('_scalar(key, context["shape_names"])') < body.index('py_setitem(provided, key, pair[1])')
    assert body.index('dimensions refuse Python iteration hooks') < body.index('py_call(b.tuple.__dict__["__iter__"], dims)')
    assert body.index('_scalar(dims[j], context["shape_scalars"])') < body.index('graph_int(dims[j])')
    assert 'context["shape_containers"]' in body
    assert 'context["shape_scalars"]' in body
    assert 'from bindings.classical_' not in source
