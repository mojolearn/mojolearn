# SPDX-License-Identifier: Apache-2.0
# Target-specific ownership only; both doors share the same whole-model engine.
from std.ffi import _Global
from checks.numerics import GLOBAL_NUMERIC_MODE
from autograd.ops_api import ops_session_open, ops_session_close, ops_session_registry_token, OPS_DEVICE
from autograd.program_import_request import program_import_request
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report, native_descriptor_words
from autograd.model_descriptor import MODEL_ABI_NAME, MODEL_FIT_LABELS, NativeModelRequest, model_request_execute
from sequence.neural_model_dispatch import PROGRAM_IMPORT, PROGRAM_DEVICE_RUN, SESSION_OPEN, SESSION_CLOSE, RNN_FIT, MODEL_LABELS, MODEL_ARGMAX, NeuralModelRequest, neural_model_execute
from sequence.model_outputs_device import model_output_device
from autograd.program_device_execution import program_device_request
from sequence.exec_device import DeviceExec
from core.label_cast_device import device_unique_inverse_cast


def model_native_v1(op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        if result == 0:
            raise Error("native model: null result")
        if op == PROGRAM_IMPORT:
            if nf != 0:
                raise Error("native import: operation takes no scalar descriptors")
            var value = program_import_request[DeviceExec, OPS_DEVICE](native_descriptor_words(aa, na), native_descriptor_words(pp, np))
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = value
            return 0
        if op == PROGRAM_DEVICE_RUN:
            if nf != 0:
                raise Error("native program: device execution takes no scalar descriptors")
            var value = program_device_request(native_descriptor_words(aa, na), native_descriptor_words(pp, np))
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = value
            return 0
        if op == SESSION_OPEN:
            var addresses = native_descriptor_words(aa, na)
            var options = native_descriptor_words(pp, np)
            if na != 1 or np != 2 or nf != 0 or addresses[0] == 0:
                raise Error("native session: invalid open descriptor")
            if options[1] != Int(GLOBAL_NUMERIC_MODE):
                raise Error("ValueError: selected session binding does not implement the requested numeric mode")
            var scope = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=addresses[0])
            scope[0] = ops_session_registry_token()
            scope[1] = OPS_DEVICE
            var sid = ops_session_open[DeviceExec, OPS_DEVICE](DeviceExec(), options[0] != 0)
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = sid
            return 0
        if op == SESSION_CLOSE:
            var addresses = native_descriptor_words(aa, na)
            var options = native_descriptor_words(pp, np)
            if na != 1 or np != 2 or nf != 0:
                raise Error("native session: invalid close descriptor")
            if options[0] != ops_session_registry_token() or options[1] != OPS_DEVICE:
                raise Error("native session: resource belongs to another registry or executor")
            ops_session_close[DeviceExec, OPS_DEVICE](addresses[0])
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = 0
            return 0
        var ex = DeviceExec()
        var value = 0
        if op >= RNN_FIT:
            var auxiliary = NeuralModelRequest(op, aa, na, pp, np, ff, nf)
            if op == MODEL_LABELS:
                var labels = auxiliary.labels()
                value = device_unique_inverse_cast(ex.ctx, labels[0], labels[1], auxiliary.addresses[0], auxiliary.addresses[1])
                if value == -2:
                    raise Error("mojolearn: y contains a NaN label; NaN is not a class")
            elif op >= MODEL_ARGMAX:
                value = model_output_device(ex, op, auxiliary)
            else:
                value = neural_model_execute(ex, op, auxiliary)
        else:
            var request = NativeModelRequest(op, aa, na, pp, np, ff, nf)
            if op == MODEL_FIT_LABELS:
                var kind = request.label_kind()
                var codes = ex.alloc(request.sources[1].rows)
                var count = device_unique_inverse_cast(ex.ctx, request.sources[1], kind, request.classes, Int(codes), True)
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
    name="MojoWholeModelDeviceNativeAPIv1", init_fn=NativeModelAPI.__init__]
