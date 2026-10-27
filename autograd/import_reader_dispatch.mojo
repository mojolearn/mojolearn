# SPDX-License-Identifier: Apache-2.0
# Reader/resolve ownership through the common native descriptor transport.
# This does not allocate an execution Session or GPU context merely to parse
# an artifact. All graph handles belong to the selected compiled image.
from std.ffi import _Global
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_words, native_descriptor_report
from autograd.import_reader import OnnxInfo, onnx_read, onnx_info, onnx_buffers, onnx_close, onnx_registry_token
from autograd.import_resolve import OnnxResolved, onnx_resolve

comptime ONNX_READER_ABI = "mojolearn.autograd.onnx.v1"
comptime ONNX_READER_COUNT = 4


struct NativeReaderRequest(Movable):
    var path: String
    var names: List[String]
    var dimensions: List[Int]
    var info: OnnxInfo
    var resolved: OnnxResolved
    var buffers: List[Int]

    def __init__(out self):
        self.path = String("")
        self.names = List[String]()
        self.dimensions = List[Int]()
        self.info = OnnxInfo(List[Int](), List[String](), List[Float64]())
        self.resolved = OnnxResolved(List[Int](), List[String]())
        self.buffers = List[Int]()


def reader_native_operation(op: Int, addresses: List[Int], words: List[Int]) raises -> Int:
    if op == 0:
        if len(addresses) != 2 or addresses[0] == 0 or addresses[1] == 0 or len(words) != 0:
            raise Error("native reader: invalid open descriptor")
        var request = MutPointer[NativeReaderRequest, MutUntrackedOrigin](unsafe_from_address=addresses[0])
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=addresses[1])[0] = onnx_registry_token()
        return onnx_read(request[].path)
    if len(words) != 1 or words[0] != onnx_registry_token() or len(addresses) < 1:
        raise Error("native reader: graph belongs to another registry")
    if op == 3:
        onnx_close(addresses[0])
        return 0
    if len(addresses) != 2 or addresses[1] == 0:
        raise Error("native reader: missing request storage")
    var request = MutPointer[NativeReaderRequest, MutUntrackedOrigin](unsafe_from_address=addresses[1])
    if op == 1:
        request[].info = onnx_info(addresses[0])
        request[].buffers = onnx_buffers(addresses[0])
        return 0
    if op == 2:
        request[].resolved = onnx_resolve(addresses[0], request[].names, request[].dimensions)
        return 0
    raise Error("native reader: unknown operation")


def reader_native_v1(op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                     result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        if op < 0 or op >= ONNX_READER_COUNT or result == 0 or nf != 0:
            raise Error("native reader: invalid descriptor")
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = reader_native_operation(
            op, native_descriptor_words(aa, na), native_descriptor_words(pp, np))
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


struct NativeReaderTable(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(ONNX_READER_ABI)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(reader_native_v1)


comptime ONNX_READER_TABLE = _Global[StorageType=NativeReaderTable,
    name="MojoNativeOnnxReaderTableV1", init_fn=NativeReaderTable.__init__]
