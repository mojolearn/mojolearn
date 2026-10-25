# SPDX-License-Identifier: Apache-2.0
"""Runtime probes for artifact setup; generic export remains separate debt."""
import sys

import numpy as np
import pytest

from mojolearn.importers import UnsupportedModel, _program, onnx
from mojolearn.tests import _onnx_fixture as fx


def test_artifact_setup_has_no_python_session_or_program_construction(tmp_path):
    fixture = fx.write_fixture('linear_layernorm_linear', tmp_path)
    frames = []
    def trace(frame, event, arg):
        if event == 'call' and frame.f_globals.get('__name__', '').startswith('mojolearn.'):
            frames.append((frame.f_globals['__name__'], frame.f_code.co_name))
    sys.setprofile(trace)
    try:
        program = _program._import(fixture.path, None, {'x': (np.int64(3), np.int32(6))}, 'identical')
    finally:
        sys.setprofile(None)
    assert frames == [('mojolearn.importers._program', '_import')]
    assert program.inputs[0][1] == (3, 6)
    assert program.run(np.ones((3, 6), np.float32)).shape[0] == 3


def test_path_and_native_graph_preserve_metadata_and_outputs(tmp_path):
    fixture = fx.write_fixture('linear_layernorm_linear', tmp_path)
    graph = onnx.read(fixture.path)
    direct = _program._import(fixture.path, None, {'x': (3, 6)}, None)
    retained = _program._import(graph, None, {'x': np.array([3, 6])}, None)
    assert direct.inputs == retained.inputs and direct.outputs == retained.outputs
    assert direct.input_shapes == retained.input_shapes
    assert direct.input_kinds == retained.input_kinds
    X = np.arange(18, dtype=np.float32).reshape(3, 6)
    assert np.asarray(direct.run(X)).tobytes() == np.asarray(retained.run(X)).tobytes()


def test_native_import_keeps_missing_file_error_metadata(tmp_path):
    path = tmp_path / 'missing.onnx'
    with pytest.raises(FileNotFoundError) as error:
        _program._import(path, None, None, None)
    assert error.value.errno == 2
    assert error.value.filename == str(path)


def test_native_import_keeps_unsupported_model_class_and_problem_records(tmp_path):
    path = fx.make_model(str(tmp_path / 'unsupported.onnx'),
        [fx.node('Sin', ['x'], ['y'], name='wrong_domain', domain='unsupported.test')],
        [('x', 'f32', (2, 3))], [('y', 'f32', (2, 3))], extra_opsets=(('unsupported.test', 1),))
    with pytest.raises(UnsupportedModel) as error:
        _program._import(path, None, None, None)
    assert error.value.source == path
    assert isinstance(error.value.problems, tuple) and error.value.problems
    assert 'wrong_domain' in str(error.value)
    assert 'cannot be imported' in str(error.value)


@pytest.mark.parametrize('field', ['path', 'dimensions', 'shape_pairs'])
def test_native_artifact_admission_does_not_execute_protocols(tmp_path, field):
    calls = []
    class Poison:
        def __fspath__(self):
            calls.append('path')
            raise AssertionError('path callback executed')
        def __iter__(self):
            calls.append('iter')
            raise AssertionError('iteration callback executed')
        def __int__(self):
            calls.append('int')
            raise AssertionError('scalar callback executed')
    source = tmp_path / 'unused.onnx'
    shapes = None
    if field == 'path':
        source = Poison()
    elif field == 'dimensions':
        shapes = {'x': (Poison(),)}
    else:
        shapes = [Poison()]
    with pytest.raises(TypeError, match='native|Python|path|shape'):
        _program._import(source, None, shapes, None)
    assert calls == []
