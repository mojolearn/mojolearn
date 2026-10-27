# SPDX-License-Identifier: Apache-2.0
# Target-specific ownership only; both doors share the same whole-model engine.
from std.ffi import _Global
from checks.numerics import GLOBAL_NUMERIC_MODE
from autograd.ops_api import ops_session_open, ops_session_close, ops_session_registry_token, OPS_HOST
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report, native_descriptor_words
from autograd.model_descriptor import MODEL_ABI_NAME, MODEL_FIT_LABELS, NativeModelRequest, model_request_execute
from sequence.neural_model_dispatch import SESSION_OPEN, SESSION_CLOSE, RNN_FIT, MODEL_LABELS, MODEL_ARGMAX, NeuralModelRequest, neural_model_execute
from sequence.model_outputs_host import model_output_host
from sequence.exec import HostExec
from core.label_cast import host_unique_inverse_cast


def model_native_v1(op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        if result == 0:
            raise Error("native model: null result")
        if op == SESSION_OPEN:
            var addresses = native_descriptor_words(aa, na)
            var options = native_descriptor_words(pp, np)
            if na != 1 or np != 2 or nf != 0 or addresses[0] == 0:
                raise Error("native session: invalid open descriptor")
            if options[1] != Int(GLOBAL_NUMERIC_MODE):
                raise Error("ValueError: selected session binding does not implement the requested numeric mode")
            var scope = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=addresses[0])
            scope[0] = ops_session_registry_token()
            scope[1] = OPS_HOST
            var sid = ops_session_open[HostExec, OPS_HOST](HostExec(), options[0] != 0)
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = sid
            return 0
        if op == SESSION_CLOSE:
            var addresses = native_descriptor_words(aa, na)
            var options = native_descriptor_words(pp, np)
            if na != 1 or np != 2 or nf != 0:
                raise Error("native session: invalid close descriptor")
            if options[0] != ops_session_registry_token() or options[1] != OPS_HOST:
                raise Error("native session: resource belongs to another registry or executor")
            ops_session_close[HostExec, OPS_HOST](addresses[0])
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = 0
            return 0
        var ex = HostExec()
        var value = 0
        if op >= RNN_FIT:
            var auxiliary = NeuralModelRequest(op, aa, na, pp, np, ff, nf)
            if op == MODEL_LABELS:
                var labels = auxiliary.labels()
                value = host_unique_inverse_cast(labels[0], labels[1], auxiliary.addresses[0], auxiliary.addresses[1])
                if value == -2:
                    raise Error("mojolearn: y contains a NaN label; NaN is not a class")
            elif op >= MODEL_ARGMAX:
                value = model_output_host(op, auxiliary)
            else:
                value = neural_model_execute(ex, op, auxiliary)
        else:
            var request = NativeModelRequest(op, aa, na, pp, np, ff, nf)
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
