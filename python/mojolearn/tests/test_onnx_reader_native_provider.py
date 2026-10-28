"""Authored bounded public ONNX read/resolve probes; run after integration build."""
import gc
import numpy as np
import pytest


def _fixture(path):
    onnx = pytest.importorskip('onnx')
    h, t = onnx.helper, onnx.TensorProto
    weights = np.arange(6, dtype='f4').reshape(3, 2) / np.float32(8)
    graph = h.make_graph([h.make_node('MatMul', ['x', 'w'], ['y'], name='projection')], 'fixture',
                        [h.make_tensor_value_info('x', t.FLOAT, ['batch', 3])],
                        [h.make_tensor_value_info('y', t.FLOAT, ['batch', 2])],
                        [onnx.numpy_helper.from_array(weights, 'w')])
    model = h.make_model(graph, opset_imports=[h.make_opsetid('', 17)])
    path.write_bytes(model.SerializeToString())
    return weights


def test_public_reader_resolve_views_and_import(tmp_path):
    from mojolearn.importers import onnx as reader
    from mojolearn.importers._graph import resolve, resolve_shapes
    from mojolearn._verify_native_runtime import audit_native_call
    from mojolearn import import_model
    path = tmp_path / 'fixture.onnx'
    weights = _fixture(path)
    graph = audit_native_call(reader.read, path).assert_native()
    assert graph.inputs == (('x', 'f32', ('batch', 3)),)
    result = audit_native_call(resolve, graph, {'x': (2, 3)}).assert_native()
    assert result.shapes['y'] == (2, 2) and not result.problems
    assert audit_native_call(resolve_shapes, graph, {'x': (2, 3)}).assert_native()['y'] == (2, 2)
    assert np.frombuffer(graph.initializers['w'].data, dtype='<f4').tobytes() == weights.tobytes()
    model = import_model(graph, shapes={'x': (2, 3)})
    value = np.ones((2, 3), dtype='f4')
    assert np.allclose(np.asarray(model.run(value)), value @ weights)
    view = graph.initializers['w'].data
    del model, graph
    gc.collect()
    assert np.frombuffer(view, dtype='<f4').tobytes() == weights.tobytes()
    with pytest.raises(FileNotFoundError) as failure:
        reader.read(tmp_path / 'missing.onnx')
    assert failure.value.errno == 2


def test_reader_refuses_replaced_factory_without_invoking_it(tmp_path):
    from mojolearn.importers import onnx as reader
    path = tmp_path / 'fixture.onnx'
    _fixture(path)
    graph = reader.read(path)
    binding = graph._native.b
    factory = binding.onnx_native_api_v1
    calls = []
    binding.onnx_native_api_v1 = lambda: calls.append('factory')
    try:
        with pytest.raises(TypeError):
            graph._native.resolve({'x': (2, 3)})
    finally:
        binding.onnx_native_api_v1 = factory
    assert not calls
