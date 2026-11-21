# SPDX-License-Identifier: Apache-2.0
# Whole-model/consumed-output operation names for the common descriptor provider.
# These are internal native engine entries, not public per-operation adapters.
from core.cast_in import CastIn, label_kind_of
from core.native_descriptor_call import native_descriptor_words
from sequence.exec_trait import Exec
from sequence.ops_impl import rnn_fit_impl, rnn_predict_impl, mlp_fit_impl, mlp_predict_impl

comptime RNN_FIT = 4
comptime RNN_PREDICT = 5
comptime MLP_FIT = 6
comptime MLP_PREDICT = 7
comptime MODEL_LABELS = 8
comptime MODEL_ARGMAX = 9
comptime MODEL_MIN = 10
comptime MODEL_THRESHOLD = 11
comptime MODEL_UNIFORM = 12
comptime SESSION_OPEN = 13
comptime SESSION_CLOSE = 14
comptime PROGRAM_DEVICE_RUN = 15
comptime PROGRAM_IMPORT = 16
comptime PROGRAM_HOST_RUN = 17
# lane x8-cnn-on-executor: CNNClassifier fit and predict on the program
# executor (sequence/cnn_classifier_program.mojo)
comptime CNN_FIT = 18
# lane x8b-cnn-resident-predict: CNN_PREDICT runs on the resident session
# CNN_OPEN made (the fitted parameters uploaded once)
comptime CNN_PREDICT = 19
comptime CNN_OPEN = 20
# lane dw-autograd-seq (2026-11-21): the Sequential model-file operations
# (checkpoint save / info / load, the verify sidecar's four numbers, the
# parameter count and the weight import) are operations of this descriptor,
# served by sequence/model_file_dispatch.mojo in both images. Before, the
# Sequential door reached each through the x_sequence binding's Python entry
# of the same name (a CPython call with Python lists of words inside RUN).
# The same typed functions run (autograd/model_checkpoint.mojo,
# autograd/model_verify.mojo, autograd/weight_import.mojo): no bit changes.
comptime MODEL_SAVE_CHECKPOINT = 21
comptime MODEL_CHECKPOINT_INFO = 22
comptime MODEL_LOAD_CHECKPOINT = 23
comptime MODEL_VERIFY_DIGEST = 24
comptime MODEL_VERIFY_FILL = 25
comptime MODEL_VERIFY_FILE_SHA256 = 26
comptime MODEL_VERIFY_MODEL_INFO = 27
comptime MODEL_N_PARAMS = 28
comptime MODEL_IMPORT_PARAMS = 29
# lane dw-autograd-seq (2026-11-21, continuation): the sequence layer and
# resident-tensor operations (LayerNorm, the sparse MoE block, the resident
# sequence tensors and the MoE resident weights) are operations of this
# descriptor in both images. SEQ_LAYER_NORM, SEQ_MOE_FORWARD and
# SEQ_MOE_BACKWARD run on either executor (sequence/sequence_ops_dispatch.mojo);
# SEQ_TENSOR_ALLOC .. SEQ_MOE_WEIGHTS_FREE need device tensors and are served
# by the device image only (sequence/sequence_ops_device_dispatch.mojo): the
# host image has no host kernel for them and refuses each in words. Before,
# the doors reached each through the binding's Python entry of the same name
# and decided device against host by whether that NAME was present. The same
# typed functions run: no bit changes.
comptime SEQ_LAYER_NORM = 30
comptime SEQ_MOE_FORWARD = 31
comptime SEQ_MOE_BACKWARD = 32
comptime SEQ_TENSOR_ALLOC = 33
comptime SEQ_TENSOR_FREE = 34
comptime SEQ_TENSOR_UPLOAD = 35
comptime SEQ_TENSOR_DOWNLOAD = 36
comptime SEQ_TENSOR_COPY = 37
comptime SEQ_LAYER_NORM_DEV = 38
comptime SEQ_LAYER_NORM_DEV_ALLOC = 39
comptime SEQ_MOE_FORWARD_DEV = 40
comptime SEQ_MOE_FORWARD_DEV_ALLOC = 41
comptime SEQ_MOE_WEIGHTS_PUT = 42
comptime SEQ_MOE_WEIGHTS_FREE = 43
# lane api-torch round 2 (2026-11-21): Sequential.state_dict's weight export
# (autograd/weight_import.mojo model_export_params), on either executor.
comptime MODEL_EXPORT_PARAMS = 44
comptime MODEL_OPERATION_COUNT = 45


struct NeuralModelRequest(Movable):
    var addresses: List[Int]
    var options: List[Int]
    var floats: List[Float64]

    def __init__(out self, op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int) raises:
        if op < RNN_FIT or (op >= SESSION_OPEN and op != CNN_FIT and op != CNN_PREDICT and op != CNN_OPEN) or op >= MODEL_SAVE_CHECKPOINT:
            raise Error("native model: unknown operation")
        self.addresses = native_descriptor_words(aa, na)
        self.options = native_descriptor_words(pp, np)
        self.floats = List[Float64]()
        if nf < 0 or (nf > 0 and ff == 0):
            raise Error("native model: invalid scalar span")
        var scalars = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
        for i in range(nf):  # driver-loop: admitted execution metadata
            self.floats.append(scalars[i])
        if ((op == MODEL_LABELS and (na != 2 or np != 7 or nf != 0)) or
            (op == MODEL_ARGMAX and (na != 2 or np != 2 or nf != 0)) or
            (op == MODEL_MIN and (na != 2 or np != 1 or nf != 0)) or
            (op == MODEL_THRESHOLD and (na != 2 or np != 1 or nf != 0)) or
            (op == MODEL_UNIFORM and (na != 1 or np != 4 or nf != 2))):
            raise Error("native model: invalid consumed-output descriptor")

    def labels(self) raises -> Tuple[CastIn, Int]:
        ref p = self.options
        var src = CastIn(p[0], p[1], p[2], p[3], p[4], p[5], p[6])
        src.check("y")
        if src.cols != 1:
            raise Error("model_labels: the labels must be one per row")
        if self.addresses[0] == 0 or self.addresses[1] == 0:
            raise Error("model_labels: null buffer address")
        return (src, label_kind_of(src.kind, src.itemsize, "y"))


def neural_model_execute[E: Exec](mut ex: E, op: Int, r: NeuralModelRequest) raises -> Int:
    if op == RNN_FIT:
        return rnn_fit_impl(ex, r.addresses, r.options, r.floats)
    if op == RNN_PREDICT:
        return rnn_predict_impl(ex, r.addresses, r.options)
    if op == MLP_FIT:
        return mlp_fit_impl(ex, r.addresses, r.options, r.floats)
    if op == MLP_PREDICT:
        return mlp_predict_impl(ex, r.addresses, r.options)
    raise Error("native model: operation requires target output owner")
