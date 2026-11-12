# SPDX-License-Identifier: Apache-2.0
# Python doors for the ONNX model reader (import_reader.mojo): each converts
# its arguments and the reader's typed results, nothing else.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call
from autograd.import_reader_dispatch import ONNX_READER_TABLE
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_capsule
from autograd.import_reader import onnx_read, onnx_info, onnx_buffers, onnx_close


def onnx_native_api_py() raises -> PythonObject:
    var table = ONNX_READER_TABLE.get_or_create_ptr()
    return neural_descriptor_capsule(table[].name, table[].entries)
