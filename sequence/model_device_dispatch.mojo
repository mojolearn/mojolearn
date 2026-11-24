# SPDX-License-Identifier: Apache-2.0
# Target-specific ownership only; both doors share the same whole-model engine.
from checks.numerics import GLOBAL_NUMERIC_MODE
from autograd.ops_api import ops_session_open, ops_session_close, ops_session_registry_token, OPS_DEVICE
from autograd.program_import_request import program_import_request
from autograd.program_host_request import program_host_request
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report, native_descriptor_words
from autograd.model_descriptor import MODEL_ABI_NAME, MODEL_FIT_LABELS, NativeModelRequest, model_request_execute
from sequence.neural_model_dispatch import PROGRAM_HOST_RUN, PROGRAM_IMPORT, PROGRAM_DEVICE_RUN, SESSION_OPEN, SESSION_CLOSE, RNN_FIT, RNN_PREDICT, MLP_FIT, MLP_PREDICT, MODEL_LABELS, MODEL_ARGMAX, NeuralModelRequest, neural_model_execute
from sequence.neural_model_dispatch import CNN_FIT, CNN_PREDICT, CNN_OPEN
from sequence.neural_model_dispatch import MODEL_IMPORT_PARAMS, MODEL_EXPORT_PARAMS
from sequence.model_file_dispatch import model_file_operation, model_file_execute, model_import_execute, model_export_execute
from sequence.sequence_ops_dispatch import sequence_ops_operation
from sequence.sequence_ops_device_dispatch import sequence_ops_device_execute
from sequence.cnn_classifier_program import cnn_fit_program, cnn_open_program, cnn_predict_program
from x_cnn.device import cnn_ctx

from sequence.perf3_seq_resident import seq_resident_resolve, seq_resident_unpin
from sequence.seq_member_store import seq_member_store_ptr
from sequence.model_outputs_device import model_output_device
from autograd.program_device_execution import program_device_request
from sequence.exec_device import DeviceExec
from core.label_cast_device import device_unique_inverse_cast
from core.shared_slot import SharedSlot


def seq_x_slot_resolve(op: Int, mut r: NeuralModelRequest) raises:
    """Lane vdf-scope-neural: an MLP_FIT / MLP_PREDICT whose X word is
    `-(slot + 1)` reads slot `slot` of the image's member store
    (sequence/seq_member_store.mojo) in place: the word becomes the slot's
    device address (checked to hold the N x D words the options name) and
    `r.x_device` is set. Any other X word is left as it is."""
    if len(r.addresses) == 0 or r.addresses[0] >= 0:
        return
    # MLP: ip = [N, D, ...]; RNN: ip = [cell, D, H, L, O, task, N, T, ...].
    var words = 0
    if op == RNN_FIT or op == RNN_PREDICT:
        if len(r.options) < 8:
            raise Error("native model: a device-store X needs the row, step and feature words")
        words = r.options[6] * r.options[7] * r.options[1]
    else:
        if len(r.options) < 2:
            raise Error("native model: a device-store X needs the row and column words")
        words = r.options[0] * r.options[1]
    r.addresses[0] = seq_member_store_ptr(-r.addresses[0] - 1, words)
    r.x_device = True


def _cnn_exec() raises -> DeviceExec:
    """The CNN sessions' executor. It wraps the x_cnn context, so the session's
    launches and the x_cnn entries the tuned ops call (autograd/cnn_block_op.mojo)
    share one in-order queue."""
    # TOMBSTONE: MOJOLEARN_CNN_GENERIC_OPS (opt-in arm) named here; deleted 2026-11-15 by lane/expdel2-23-neural (owner); this
    # docstring note removed by lane/expdel2-29-neural, code recoverable at cdff18f6c8.
    # Restore: git apply experiments/removed/MOJOLEARN_CNN_GENERIC_OPS.expdel2-29-neural.patch
    # RESTORED 2026-11-15 by lane/expdel2-restore-owed: owed B arm (owed-pre0867.json), under four weeks old; the deletion was premature.
    # TOMBSTONE: MOJOLEARN_CNN_GENERIC_OPS (LOSER: NV L40S cnn-clf fit 161 -> 395 ms (2.45), infer 1.45) deleted 2026-11-24 by lane/ab2-nseq (ab-outcome-2); code recoverable at ba83d4655f.
    # Restore: git apply ~/mojolearn-evidence/experiments-removed/MOJOLEARN_CNN_GENERIC_OPS.ab-outcome-2.patch
    return DeviceExec(cnn_ctx())


def model_native_v1(op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
                    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    try:
        if result == 0:
            raise Error("native model: null result")
        # lane dw-autograd-seq: the Sequential model-file operations
        # (sequence/model_file_dispatch.mojo); only the weight import takes
        # an executor, so the other eight build none.
        if model_file_operation(op):
            var value = model_file_execute(op, aa, na, pp, np, ff, nf)
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = value
            return 0
        if op == MODEL_IMPORT_PARAMS:
            var import_ex = DeviceExec()
            var value = model_import_execute(import_ex, aa, na, pp, np, ff, nf)
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = value
            return 0
        if op == MODEL_EXPORT_PARAMS:
            var export_ex = DeviceExec()
            var value = model_export_execute(export_ex, aa, na, pp, np, ff, nf)
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = value
            return 0
        # lane dw-autograd-seq: the sequence layer and resident-tensor
        # operations (sequence/sequence_ops_device_dispatch.mojo).
        if sequence_ops_operation(op):
            var value = sequence_ops_device_execute(op, aa, na, pp, np, ff, nf)
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = value
            return 0
        if op == PROGRAM_HOST_RUN:
            if nf != 0:
                raise Error("native program: host execution takes no scalar descriptors")
            var value = program_host_request[DeviceExec, OPS_DEVICE](native_descriptor_words(aa, na), native_descriptor_words(pp, np))
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = value
            return 0
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
        if op == CNN_FIT or op == CNN_PREDICT or op == CNN_OPEN:
            # lane x8-cnn-on-executor: CNNClassifier as one captured program
            # (Program.fit) on a fresh executor; lane x8b: predict replays the
            # forward on the resident registry session CNN_OPEN made (the
            # binding opens it with the GIL held, as SESSION_OPEN)
            var cnn = NeuralModelRequest(op, aa, na, pp, np, ff, nf)
            var written = 0
            if op == CNN_FIT:
                written = cnn_fit_program[DeviceExec](_cnn_exec(), cnn.addresses, cnn.options, cnn.floats)
            elif op == CNN_OPEN:
                if nf != 0:
                    raise Error("native CNN: the resident open takes no scalar descriptors")
                written = cnn_open_program[DeviceExec, OPS_DEVICE](_cnn_exec(), cnn.addresses, cnn.options)
            else:
                if nf != 0:
                    raise Error("native CNN: predict takes no scalar descriptors")
                written = cnn_predict_program[DeviceExec, OPS_DEVICE](cnn.addresses, cnn.options)
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = written
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
                # perf3-seq-neural: a predict that names its fitted block
                # reads the block's resident device copy (uploaded on the
                # model's first predict) instead of uploading it per call.
                # The slot stays pinned from the lookup until the launches
                # that read its address are queued (and here run), so no
                # other thread frees it in between (ne1-a-cluster-spectral).
                # Lane vdf-scope-neural: an MLP X held in the member store
                # (a fold taken on the device) is read in place.
                if op == MLP_FIT or op == MLP_PREDICT or op == RNN_FIT or op == RNN_PREDICT:
                    seq_x_slot_resolve(op, auxiliary)
                var pinned = 0
                if op == RNN_PREDICT:
                    pinned = seq_resident_resolve(ex, auxiliary.addresses, 4)
                elif op == MLP_PREDICT:
                    pinned = seq_resident_resolve(ex, auxiliary.addresses, 3)
                try:
                    value = neural_model_execute(ex, op, auxiliary)
                except launch_error:
                    seq_resident_unpin(pinned)
                    raise launch_error^
                seq_resident_unpin(pinned)
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


comptime NATIVE_MODEL_API = SharedSlot[NativeModelAPI, ("MojolearnWholeModelDeviceNativeAPIIdenticalV3" if GLOBAL_NUMERIC_MODE == 1 else "MojolearnWholeModelDeviceNativeAPIFastV3")]
