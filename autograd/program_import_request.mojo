# SPDX-License-Identifier: Apache-2.0
# Native request/result storage shared by the whole-model provider's ONNX import
# operation. The boundary retains this initialized owner until synchronous return.
# This private same-build descriptor contains native Mojo state, never Python
# objects, callbacks or user protocols. The selected image owns all registries.
from sequence.exec_trait import Exec
from autograd.ops_api import ops_session_registry_token
from autograd.import_lower import OnnxImport, onnx_import, onnx_import_file
from autograd.import_reader import onnx_buffers


struct NativeProgramImport(Movable):
    var from_reader: Bool
    var reader: Int
    var path: String
    var source: String
    var names: List[String]
    var dimensions: List[Int]
    var result: OnnxImport

    def __init__(out self, from_reader: Bool, reader: Int, var path: String,
                 var source: String, var names: List[String], var dimensions: List[Int]):
        self.from_reader = from_reader
        self.reader = reader
        self.path = path^
        self.source = source^
        self.names = names^
        self.dimensions = dimensions^
        self.result = OnnxImport()


def program_import_request[E: Exec & Movable & Deinitable, KIND: Int](
    addresses: List[Int], words: List[Int],
) raises -> Int:
    if len(addresses) != 1 or addresses[0] == 0 or len(words) != 3:
        raise Error("native import: invalid owned request descriptor")
    if words[0] != ops_session_registry_token() or words[1] != KIND:
        raise Error("native import: Session belongs to another registry or executor")
    var request = MutPointer[NativeProgramImport, MutUntrackedOrigin](unsafe_from_address=addresses[0])
    if request[].from_reader:
        # Read the buffer addresses from the graph owner in this same image.
        # They stay borrowed from that graph throughout lowering and capture.
        var addresses = onnx_buffers(request[].reader)
        request[].result = onnx_import[E, KIND](request[].reader, words[2], request[].names,
            request[].dimensions, addresses^, request[].source)
    else:
        request[].result = onnx_import_file[E, KIND](request[].path, words[2], request[].names,
            request[].dimensions, request[].source)
    return request[].result.status
