# SPDX-License-Identifier: Apache-2.0
# Whole-model schema over the shared NativeDescriptorCall provider interface.
# Metadata words: program length, option length, CastIn count, classes address,
# classification output width/target, native class-count result address; then
# program words, options, and seven native CastIn words per input. This is an
# internal engine table, not a separate Python/Torch/MAX provider implementation.
from core.cast_in import CastIn, label_kind_of
from core.native_descriptor_call import native_descriptor_words
from autograd.model_execution import model_fit_native, model_predict_native, model_class_count
from sequence.exec_trait import Exec
from sequence.ops import FP

comptime MODEL_ABI_NAME = "mojolearn.model.execution.v1"
comptime MODEL_FIT = 0
comptime MODEL_FIT_LABELS = 1
comptime MODEL_PREDICT = 2
comptime MODEL_PREDICT_LABELS = 3


def model_descriptor_words(prog: List[Int], options: List[Int], sources: List[CastIn],
                           classes: Int = 0, outputs: Int = 0, target: Int = 0, count_result: Int = 0) -> List[Int]:
    var words: List[Int] = [len(prog), len(options), len(sources), classes, outputs, target, count_result]
    for word in prog:  # driver-loop: native model description metadata
        words.append(word)
    for word in options:  # driver-loop: native model execution options
        words.append(word)
    for src in sources:  # driver-loop: already admitted input descriptors
        words.append(src.addr)
        words.append(src.kind)
        words.append(src.itemsize)
        words.append(src.rows)
        words.append(src.cols)
        words.append(src.s0)
        words.append(src.s1)
    return words^


struct NativeModelRequest(Movable):
    var addresses: List[Int]
    var program: List[Int]
    var options: List[Int]
    var sources: List[CastIn]
    var floats: List[Float64]
    var classes: Int
    var outputs: Int
    var target: Int
    var count_result: Int

    def __init__(out self, op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int) raises:
        if op < MODEL_FIT or op > MODEL_PREDICT_LABELS:
            raise Error("native model: unknown operation")
        var words = native_descriptor_words(pp, np)
        if len(words) < 7:
            raise Error("native model: missing metadata header")
        var nprog = words[0]
        var nopt = words[1]
        var nsrc = words[2]
        if nprog < 0 or nopt < 0 or nsrc < 0 or len(words) != 7 + nprog + nopt + 7 * nsrc:
            raise Error("native model: invalid metadata span")
        if nf < 0 or (nf > 0 and ff == 0):
            raise Error("native model: invalid scalar span")
        self.addresses = native_descriptor_words(aa, na)
        self.program = List[Int]()
        self.options = List[Int]()
        self.sources = List[CastIn]()
        self.floats = List[Float64]()
        self.classes = words[3]
        self.outputs = words[4]
        self.target = words[5]
        self.count_result = words[6]
        var at = 7
        for _ in range(nprog):  # driver-loop: borrow admitted model words
            self.program.append(words[at])
            at += 1
        for _ in range(nopt):  # driver-loop: borrow admitted operation options
            self.options.append(words[at])
            at += 1
        for _ in range(nsrc):  # driver-loop: borrow admitted CastIn records
            self.sources.append(CastIn(words[at], words[at + 1], words[at + 2], words[at + 3],
                                       words[at + 4], words[at + 5], words[at + 6]))
            at += 7
        var scalars = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=ff)
        for i in range(nf):  # driver-loop: native scalar metadata
            self.floats.append(scalars[i])
        if op <= MODEL_FIT_LABELS:
            if nsrc != (5 if na == 13 else 2):
                raise Error("native model: missing fit input descriptors")
        elif nsrc != 1:
            raise Error("native model: prediction requires one input descriptor")

    def label_kind(self) raises -> Int:
        var src = self.sources[1]
        src.check("y")
        if src.cols != 1:
            raise Error("model_labels: the labels must be one per row")
        if self.classes == 0 or self.count_result == 0:
            raise Error("model_labels: null buffer address")
        if self.target != 1 and self.target != 2:
            raise Error("model_fit_labels: target must be 1 (cross_entropy) or 2 (bce_with_logits)")
        return label_kind_of(src.kind, src.itemsize, "y")


def model_request_execute[E: Exec](mut ex: E, op: Int, request: NativeModelRequest,
                                  codes: Int = 0, classes: Int = 0) raises -> Int:
    if op == MODEL_FIT_LABELS:
        var count = model_class_count(classes, request.outputs, request.target)
        var epochs = model_fit_native(ex, request.addresses, request.sources, request.program,
                                      request.options, request.floats, codes, count)
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=request.count_result)[0] = count
        return epochs
    if op == MODEL_FIT:
        return model_fit_native(ex, request.addresses, request.sources, request.program,
                                request.options, request.floats)
    return model_predict_native(ex, request.addresses, request.sources[0], request.program,
                                request.floats, request.options, op == MODEL_PREDICT_LABELS)
