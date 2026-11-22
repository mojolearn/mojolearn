# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""The executor op OPS_LM_HEAD_V2 (autograd/lm_head_v2_op.mojo, the chunked
LM head as one session op with its VJP) against the oracle
(training/checks/chunked_lm_head_oracle.mojo), word for word, on the host
column and on the device column.

Each case records `loss = lm_head_v2(hidden, weight, codes)` on a session of
the column's executor, runs the backward from the loss and compares the loss
word and every dHidden and dWeight word with the oracle's:

  0 rows 7, vocab 3 x 1024 + 1, width 17: three whole vocabulary panels and
    a one-token tail (the panel loop, the dHidden fold over panels and, under
    the arm, a merged pair plus a carried tail and an odd last leaf).
  1 rows 300, vocab 130, width 5: the mean-loss fold crosses its 256-row leaf
    (a second leaf and one tree level), the vocabulary ends inside a panel.
  2 the upstream factor: backward from `3 * loss`; every gradient word is
    `ftz(oracle word * 3)` (the op's own scale pass; at a loss root the
    factor is 1 and hides a dropped scale).
  3 a code that names no vocabulary entry (-100), one row: the oracle refuses
    such a code, so the expected word is built from the oracle's denominator
    of the same row: the row loss is `-(0 - log(denominator))`, the op's own
    rule. A recording and an inference session must give that word.

The op, its host arm and the oracle all run the GEMM-panel fold (the only
profile since 2026-11-20; the scalar serial profile is deleted), so the same
check decides it.

What a sabotage of the kernels changes here: a vocabulary or row fold out of
ascending order (cases 0 and 1: dHidden, dWeight or the denominator move), a
panel boundary that resets the maximum or the denominator (case 0), a loss
leaf boundary off by one or a tree level skipped (case 1), a dropped or
doubled upstream scale (case 2), an off-vocabulary code read as an index or
as a one-hot (case 3), and any difference between the device kernels and the
host arm (both columns compare with the one oracle).

The session uploads and downloads through its own doors; this file queues no
copy of its own.
"""
from autograd.lm_head_v2_op import OPS_LM_HEAD_V2
from autograd.ops_api import OpsSession
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from checks.numerics import ftz, identical_log, identical_mul
from training.checks.loss_contract import neg_by_bits
from training.checks.chunked_lm_head_oracle import chunked_lm_head_v2_oracle, lm_head_v2_loss_fold
from checks.native_tests.ops_higher_common import tensor, scale, values, gradient
from checks.native_tests.core_ops_common import uniform, dims2
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, CPU_COLUMN_BUILD, check_cases_serial, require


comptime OPS_LM_HEAD_V2_CASES = 4
# case 3: one whole panel and a tail inside the second
comptime OFF_VOCAB = 1024 + 77
comptime OFF_WIDTH = 9


def _same(got: List[Float32], want: List[Float32], what: String) raises:
    require(len(got) == len(want), what + ": extent " + String(len(got)) + " vs " + String(len(want)))
    for i in range(len(got)):
        if got[i].to_bits() != want[i].to_bits():
            raise Error(what + ": at " + String(i) + " got bits " + String(got[i].to_bits())
                        + ", oracle bits " + String(want[i].to_bits()))


def _codes(targets: List[Int32]) -> List[Float32]:
    """The class codes as float values, the op's input convention."""
    var codes = List[Float32](length=len(targets), fill=0)
    for i in range(len(targets)):
        codes[i] = Float32(Int(targets[i]))
    return codes^


def _against_oracle[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], rows: Int, vocab: Int, width: Int, seed: Int, factor: Float32
) raises:
    """The recorded op and its backward against the oracle. `factor` 1 runs
    the backward from the loss; another factor from `factor * loss`."""
    var hidden = uniform(rows * width, seed, -1.0, 1.0)
    var weight = uniform(vocab * width, seed + 1, -0.5, 0.5)
    var targets = List[Int32](length=rows, fill=0)
    for row in range(rows):
        # spread over the vocabulary, the first and the last entry included
        targets[row] = Int32((row * 127) % vocab)
    targets[rows - 1] = Int32(vocab - 1)
    var codes = _codes(targets)
    var h = tensor(s, hidden, dims2(rows, width), True)
    var w = tensor(s, weight, dims2(vocab, width), True)
    var c = tensor(s, codes, dims2(rows, 1), False)
    var hs: List[Int] = [h, w, c]
    var loss = s.op(OPS_LM_HEAD_V2, hs, List[Int](), List[Float32]())
    var loss_word = values(s, loss, dims2(1, 1))
    var root = loss
    if factor != Float32(1.0):
        root = scale(s, loss, factor)
    s.backward(root, False)
    var d_hidden = gradient(s, h, dims2(rows, width))
    var d_weight = gradient(s, w, dims2(vocab, width))

    var want = chunked_lm_head_v2_oracle(hidden, weight, targets, rows, vocab, width)
    var want_loss = List[Float32](length=1, fill=want.loss)
    var want_dh = want.d_hidden.copy()
    var want_dw = want.d_weight.copy()
    if factor != Float32(1.0):
        for i in range(len(want_dh)):
            want_dh[i] = ftz(identical_mul(want_dh[i], factor))
        for i in range(len(want_dw)):
            want_dw[i] = ftz(identical_mul(want_dw[i], factor))
    var shape = "(" + String(rows) + ", " + String(vocab) + ", " + String(width) + ")"
    _same(loss_word, want_loss, "lm_head_v2 loss " + shape)
    _same(d_hidden, want_dh, "lm_head_v2 dHidden " + shape)
    _same(d_weight, want_dw, "lm_head_v2 dWeight " + shape)


def _off_hidden() -> List[Float32]:
    return uniform(OFF_WIDTH, 41, -1.0, 1.0)


def _off_weight() -> List[Float32]:
    return uniform(OFF_VOCAB * OFF_WIDTH, 42, -0.5, 0.5)


def _off_vocabulary_want() raises -> List[Float32]:
    """One row whose code names no entry: no token matches, so the target's
    shifted logit stays +0.0 and the row loss is the negated
    `0 - log(denominator)`; the mean of one row is the fold of that word.
    The maximum and the denominator do not read the target, so the oracle's
    row with any valid target supplies the denominator."""
    var hidden = _off_hidden()
    var weight = _off_weight()
    var targets = List[Int32](length=1, fill=Int32(0))
    var valid = chunked_lm_head_v2_oracle(hidden, weight, targets, 1, OFF_VOCAB, OFF_WIDTH)
    var row_loss = List[Float32](
        length=1, fill=neg_by_bits(ftz(ftz(Float32(0.0)) - ftz(identical_log(ftz(valid.row_denom[0])))))
    )
    return List[Float32](length=1, fill=lm_head_v2_loss_fold(row_loss, 1))


def _off_vocabulary[E: Exec & Movable & Deinitable](mut s: OpsSession[E], what: String) raises:
    var hidden = _off_hidden()
    var weight = _off_weight()
    var codes = List[Float32](length=1, fill=Float32(-100.0))
    var record = s.recording()
    var h = tensor(s, hidden, dims2(1, OFF_WIDTH), record)
    var w = tensor(s, weight, dims2(OFF_VOCAB, OFF_WIDTH), record)
    var c = tensor(s, codes, dims2(1, 1), False)
    var hs: List[Int] = [h, w, c]
    var loss = s.op(OPS_LM_HEAD_V2, hs, List[Int](), List[Float32]())
    _same(values(s, loss, dims2(1, 1)), _off_vocabulary_want(), "lm_head_v2 off-vocabulary loss, " + what)


def ops_lm_head_v2_records(which: Int, second: Bool) -> Bool:
    """Whether the case's session records: all but case 3's second session."""
    return not (which == 3 and second)


def run_ops_lm_head_v2_case[E: Exec & Movable & Deinitable](
    mut s: OpsSession[E], mut again: OpsSession[E], which: Int
) raises:
    if which == 0:
        _against_oracle(s, 7, 3 * 1024 + 1, 17, 11, Float32(1.0))
    elif which == 1:
        _against_oracle(s, 300, 130, 5, 21, Float32(1.0))
    elif which == 2:
        _against_oracle(s, 6, 1024 + 3, 8, 31, Float32(3.0))
    else:
        _off_vocabulary(s, "recording session")
        _off_vocabulary(again, "inference session")


# The host column: the cases across host cores, each on its own two host sessions. The device column: DeviceExec
# shares sequence_ctx() and its pool, so concurrent sessions would race that context's ownership (as in
# core_ops_session): the cases run one after another and each kernel keeps its own parallelism.
def test_lm_head_v2() raises:
    comptime if CPU_COLUMN_BUILD:
        def host_case(which: Int) raises:
            var s = OpsSession[HostExec](HostExec(), ops_lm_head_v2_records(which, False))
            var again = OpsSession[HostExec](HostExec(), ops_lm_head_v2_records(which, True))
            try:
                run_ops_lm_head_v2_case(s, again, which)
            except error:
                s.close()
                again.close()
                raise error
            s.close()
            again.close()
        check_cases("lm_head_v2", OPS_LM_HEAD_V2_CASES, host_case)
    else:
        def device_case(which: Int) raises:
            var s = OpsSession[DeviceExec](DeviceExec(), ops_lm_head_v2_records(which, False))
            var again = OpsSession[DeviceExec](DeviceExec(), ops_lm_head_v2_records(which, True))
            try:
                run_ops_lm_head_v2_case(s, again, which)
            except error:
                s.close()
                again.close()
                raise error
            s.close()
            again.close()
        check_cases_serial("lm_head_v2", OPS_LM_HEAD_V2_CASES, device_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
