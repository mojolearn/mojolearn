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
comptime MODEL_OPERATION_COUNT = 18


struct NeuralModelRequest(Movable):
    var addresses: List[Int]
    var options: List[Int]
    var floats: List[Float64]

    def __init__(out self, op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int) raises:
        if op < RNN_FIT or op >= SESSION_OPEN:
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
