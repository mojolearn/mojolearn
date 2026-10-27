# SPDX-License-Identifier: Apache-2.0
# Target-specific ownership only; both doors share the same whole-model engine.
from std.ffi import _Global
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from autograd.model_descriptor import MODEL_ABI_NAME, MODEL_FIT_LABELS, NativeModelRequest, model_request_execute
from sequence.exec import HostExec
from core.label_cast import host_unique_inverse_cast


def model_native_v1(op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        if result == 0:
            raise Error("native model: null result")
        var request = NativeModelRequest(op, aa, na, pp, np, ff, nf)
        var ex = HostExec()
        var value = 0
        if op == MODEL_FIT_LABELS:
            var kind = request.label_kind()
            var codes = ex.alloc(request.sources[1].rows)
            var count = host_unique_inverse_cast(request.sources[1], kind, request.classes, Int(codes))
            value = model_request_execute(ex, op, request, Int(codes), count)
        else:
            value = model_request_execute(ex, op, request)
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = value
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct NativeModelAPI(Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(MODEL_ABI_NAME)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(model_native_v1)


comptime NATIVE_MODEL_API = _Global[StorageType=NativeModelAPI,
    name="MojoWholeModelHostNativeAPIv1", init_fn=NativeModelAPI.__init__]
