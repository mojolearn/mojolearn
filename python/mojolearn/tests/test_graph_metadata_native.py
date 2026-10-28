"""Bounded public metadata probes for the orchestrator's compiled Mac check."""
from dataclasses import FrozenInstanceError
import pytest


def test_graph_metadata_constructors_and_nested_native_protocols():
    from mojolearn.importers._graph import TensorInfo, Node, GraphIR, Resolved
    from mojolearn._verify_native_runtime import audit_native_call
    tensor = audit_native_call(TensorInfo, 'w', 'f32', (2,), memoryview(b'12345678')).assert_native()
    node = audit_native_call(Node, 'Identity', '', 'copy', ('x',), ('y',)).assert_native()
    other = Node('Identity', '', 'copy', ('x',), ('y',))
    assert node.attrs is not other.attrs
    assert audit_native_call(repr, node).assert_native() == "Node(op='Identity', domain='', name='copy', inputs=('x',), outputs=('y',), attrs={})"
    assert audit_native_call(node.__eq__, other).assert_native() is True
    assert audit_native_call(hash, tensor).assert_native() == hash(('w', 'f32', (2,), memoryview(b'12345678')))
    with pytest.raises(FrozenInstanceError):
        tensor.name = 'different'
    graph = audit_native_call(GraphIR, 13, (), (), (node,), {'w': tensor}, 'model.onnx').assert_native()
    assert 'TensorInfo(' in audit_native_call(repr, graph).assert_native()
    assert audit_native_call(graph.__eq__, GraphIR(13, (), (), (other,), {'w': tensor}, 'model.onnx')).assert_native()
    assert audit_native_call(Resolved, {}, {}, {}, (), ()).assert_native().problems == ()


def test_graph_helpers_and_handmade_graph_refusal():
    from mojolearn.importers import supported_ops, UnsupportedModel
    from mojolearn.importers._graph import GraphIR, resolve, onnx_dtype_name, bytes_view, buffer_address
    from mojolearn.importers._table import session_shape
    from mojolearn._verify_native_runtime import audit_native_call
    import numpy as np
    assert audit_native_call(onnx_dtype_name, 1).assert_native() == 'f32'
    assert audit_native_call(onnx_dtype_name, 123).assert_native() == 'data type 123'
    assert audit_native_call(session_shape, ()).assert_native() == (1, 1)
    assert audit_native_call(session_shape, (3,)).assert_native() == (1, 3)
    assert ('Abs', 'any input') in audit_native_call(supported_ops).assert_native()
    error = audit_native_call(UnsupportedModel, '/tmp/model.onnx', [('x', None, 'missing')]).assert_native()
    assert str(error) == '1 item of model.onnx cannot be imported:\n  x: missing'
    data = bytearray(b'abcd')
    view = audit_native_call(bytes_view, data).assert_native()
    data[0] = 99
    assert view[0] == 99
    noncontiguous = np.arange(10, dtype='u1')[::2]
    assert audit_native_call(buffer_address, noncontiguous).assert_native() == noncontiguous.ctypes.data
    with pytest.raises(TypeError, match='graph read by'):
        resolve(GraphIR(13, (), (), (), {}, 'manual'))


def test_record_rejects_python_protocol_before_invoking_it():
    from mojolearn.importers._graph import Node
    calls = []
    class Callback:
        def __repr__(self):
            calls.append('repr')
            raise AssertionError('Python metadata hook ran')
    record = Node('Identity', '', 'copy', (), (), {'bad': Callback()})
    with pytest.raises(TypeError, match='Python hooks are unsupported'):
        repr(record)
    assert calls == []


def test_public_reader_errors_keep_native_exception_categories():
    from mojolearn.importers.onnx import reader_error
    with pytest.raises(ValueError, match='not an ONNX model'):
        reader_error(1, 'bad header', False, 'bad.onnx')
    with pytest.raises(FileNotFoundError) as error:
        reader_error(2, 'missing', True, 'absent.onnx')
    assert error.value.filename == 'absent.onnx'
    with pytest.raises(RuntimeError, match='returned status 99'):
        reader_error(99, '', False, 'bad.onnx')
