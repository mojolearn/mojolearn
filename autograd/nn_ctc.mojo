# SPDX-License-Identifier: Apache-2.0
"""The CTC loss of `Sequential.compile(loss="ctc")` and of the tape (lane nn-losses-2, 2026-11-23).

torch.nn.CTCLoss(blank, reduction, zero_infinity) on log-probabilities: the negative log-likelihood of each target
summed over every alignment of the sample's first `input length` steps, by the log-space forward-backward
recursion (Graves et al. 2006). The bodies are autograd/nn_ctc_ops.mojo's (one scan per batch row, rows in
parallel; host column and every device run the same bodies).

Launches. Forward: OP_CTC_FWD (one launch, every row's alpha scan and loss value), then the batch reduction of
the row values (Tape.mean / Tape.sum, the blocked fold: one launch for at most one leaf block of rows, else two).
Backward: the fold's VJP (one launch) and OP_CTC_VJP (one launch: every row's beta scan and gradient). Nothing is
launched per time step or per state. The scratch (alpha (B, T, 2 S + 1), beta (B, 2, 2 S + 1), the raw
likelihoods (B)) is tape storage, freed at the tape's reset.

The fit (autograd/model.mojo, `# nn-losses-2` groups): the model's output sample is (T, C) log-probabilities
(a final log_softmax over the classes); the targets are packed once per fit into (N, 2 + S) rows
[input length, target length, labels] on the executor (OP_CTC_PACK), so the fit's per-batch row gather carries
them with no extra launch; the input length is the sample's `lengths` value, or T without lengths.
"""
from sequence.exec_trait import Exec
from sequence.ops import FP, Args, dummy_ptr
from autograd.tape import Tape, Tensor, BackwardRules
from autograd.losses import REDUCTION_MEAN, REDUCTION_SUM
from autograd.nn_ctc_ops import OP_CTC_FWD, OP_CTC_VJP, OP_CTC_PACK, CTC_LABELS

#: the custom-node tag of the per-row CTC values (autograd/model_blocks.mojo ModelRules, `# nn-losses-2` group)
comptime NN_CTC_TAG = 3420


def ctc_blank_word(blank: Int, zero_infinity: Bool) -> Float32:
    """The fit's loss scalar word of the ctc loss: the blank index, or -1 - blank with zero_infinity."""
    return Float32(-1 - blank) if zero_infinity else Float32(blank)


def ctc_blank_of(word: Float32) -> Int:
    var w = Int(word)
    return -1 - w if w < 0 else w


def ctc_zero_infinity_of(word: Float32) -> Bool:
    return word < Float32(0)


def ctc_loss[E: Exec](
    mut tape: Tape, mut ex: E, log_probs: Tensor, targets: Tensor, T: Int, C: Int, blank: Int,
    reduction: Int = REDUCTION_MEAN, zero_infinity: Bool = False,
) raises -> Tensor:
    """The CTC loss of log-probabilities (B, T, C) (any view whose elements are the B samples' (T, C) rows in
    row-major order) against target rows (B, 2 + S): [input length, target length, S labels], lengths and labels
    as float values (labels past the target length are ignored). 'mean' divides each sample's loss by its target
    length (at least 1) and averages over the batch (PyTorch's); 'sum' adds the samples' losses. zero_infinity
    gives 0 (and gradient 0) for a sample no alignment can produce; otherwise its loss is +inf. Only the
    log-probabilities take a gradient.

    Example::

        var lp = tape.log_softmax(ex, tape.linear(ex, h, w, b))   # (B * T, C)
        var loss = ctc_loss(tape, ex, lp, rows, T, C, 0)
        tape.backward_scalar_with_rules(ex, loss, NnCtcRules())
    """
    if T < 1 or C < 2:
        raise Error("ctc: the log-probabilities need at least one step and two classes")
    if blank < 0 or blank >= C:
        raise Error("ctc: blank must be a class index in [0, " + String(C) + ")")
    if reduction != REDUCTION_MEAN and reduction != REDUCTION_SUM:
        raise Error("ctc: reduction must be 'mean' or 'sum'")
    var words = tape.rows(log_probs) * tape.cols(log_probs)
    if words % (T * C) != 0:
        raise Error("ctc: the log-probabilities are not a whole number of (" + String(T) + ", " + String(C)
                    + ") samples")
    var B = words // (T * C)
    if tape.rows(targets) != B or tape.cols(targets) < CTC_LABELS:
        raise Error("ctc: the targets must be (" + String(B) + ", 2 + max target length) rows")
    var TW = tape.cols(targets)
    var SP = 2 * (TW - CTC_LABELS) + 1
    var grad = tape.is_recording() and tape.requires_grad(log_probs)
    var a = Args()
    a.p0 = tape.value_of(ex, log_probs)
    a.p1 = tape.value_of(ex, targets)
    a.p2 = tape.reserve(ex, B)
    a.p3 = tape.reserve(ex, B * T * SP)
    a.p4 = tape.reserve(ex, B)
    a.p7 = tape.reserve(ex, B * 2 * SP) if grad else dummy_ptr()
    a.i0 = T
    a.i1 = C
    a.i2 = TW
    a.i3 = blank
    a.i4 = 1 if reduction == REDUCTION_SUM else 0
    a.i5 = 1 if zero_infinity else 0
    a.i6 = 1 if grad else 0
    a.i7 = B
    ex.launch[OP_CTC_FWD](a, B)
    var rows = tape.custom(ex, [log_probs, targets], a.p2, B, 1, NN_CTC_TAG, a)
    if reduction == REDUCTION_SUM:
        return tape.sum(ex, rows)
    return tape.mean(ex, rows)


@fieldwise_init
struct NnCtcRules(BackwardRules, ImplicitlyCopyable, Movable):
    """The backward of the per-row CTC values (NN_CTC_TAG): OP_CTC_VJP writes every word of the
    log-probabilities' destination (zeroed_destination False); the targets take no gradient and their
    destination is never touched."""

    @staticmethod
    def owns(tag: Int) -> Bool:
        return tag == NN_CTC_TAG

    def supports(self, tag: Int) -> Bool:
        return tag == NN_CTC_TAG

    def zeroed_destination(self, tag: Int, saved: Args, edge: Int) -> Bool:
        return False

    def writes_total(self, tag: Int, saved: Args, edge: Int) -> Bool:
        return False

    def apply[E: Exec](
        self, mut ex: E, tag: Int, saved: Args, output: FP, upstream: FP,
        inputs: List[FP], gradients: List[FP], rows: List[Int], cols: List[Int],
    ) raises:
        if tag != NN_CTC_TAG:
            raise Error("nn_ctc rules: unknown tag")
        if saved.i6 == 0:
            return
        var a = saved
        a.p5 = upstream
        a.p6 = gradients[0]
        ex.launch[OP_CTC_VJP](a, saved.i7)


def ctc_pack_targets[E: Exec](
    mut ex: E, dst: FP, labels: FP, target_lengths: FP, input_lengths: Optional[FP], N: Int, S: Int, T: Int,
) raises:
    """The fit's (N, 2 + S) target rows in dst from the uploaded labels (N, S), target lengths (N) and input
    lengths (N; None: every sample's T steps), one launch once per fit."""
    var a = Args()
    a.p0 = labels
    a.p1 = target_lengths
    a.p2 = input_lengths.value() if input_lengths else dummy_ptr()
    a.p3 = dst
    a.i0 = S + CTC_LABELS
    a.i1 = T
    a.i2 = 1 if input_lengths else 0
    ex.launch[OP_CTC_PACK](a, N * (S + CTC_LABELS))


def ctc_admit(labels: FP, target_lengths: FP, input_lengths: Optional[FP], N: Int, S: Int, T: Int, C: Int,
              blank: Int) raises:
    """Admission of a ctc fit's targets on the host, before anything is uploaded (the words a native fit reads):
    each target length a whole number in [0, S], each label before it a whole class index in [0, C) other than
    blank, and each input length (when given) at most T. Clear refusals; no alignment check (an impossible one is
    the loss's +inf, or 0 under zero_infinity)."""
    for r in range(N):  # admission: one pass over the caller's target words, before the fit
        var tl = target_lengths.unsafe_load(r)
        if not (tl >= Float32(0) and tl <= Float32(S) and Float32(Int(tl)) == tl):
            raise Error("ValueError: model fit: target_lengths must be whole numbers from 0 to the targets' width ("
                        + String(S) + "); sample " + String(r) + " has " + String(tl))
        for k in range(Int(tl)):  # admission: the sample's labels
            var v = labels.unsafe_load(r * S + k)
            if not (v >= Float32(0) and v < Float32(C) and Float32(Int(v)) == v):
                raise Error("ValueError: model fit: ctc labels must be class indices in [0, " + String(C)
                            + "); sample " + String(r) + " label " + String(k) + " is " + String(v))
            if Int(v) == blank:
                raise Error("ValueError: model fit: ctc labels must not hold the blank index (" + String(blank)
                            + "); sample " + String(r) + " label " + String(k))
        if input_lengths:
            var il = input_lengths.value().unsafe_load(r)
            if il > Float32(T):
                raise Error("ValueError: model fit: ctc input lengths must be at most the output's "
                            + String(T) + " steps; sample " + String(r) + " has " + String(il))
