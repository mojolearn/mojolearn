# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# native-check: cpu gpu
"""Native check of the streamed fits (lane stream-fits, 2026-11-25): `Sequential.fit(source=)` (autograd/model.mojo,
the streamed instance of the fit loop, through autograd/model_execution.mojo model_fit_source_native) and the
recurrent estimators' fit(source=) / partial_fit(source=) (sequence/recurrent.mojo rnn_fit_stream).

The fixture is the batch source's (checks/native_tests/batch_source.mojo, verify lane ops-batch-source-resume): 7
records of lengths 1 to 4 in two sealed shards, F = 2, batch 2, two batches a bucket window, shuffled from seed 5
(4 batches an epoch); here each record also has a target row of 4 words (one a step, padded past its length).

Units (one column each build: the host column with -D MOJOLEARN_COLUMN_CPU=1, else the device column):
  0  SEQUENTIAL. A GRU(4, final state) -> flatten -> dropout(0.25) -> linear(4) regressor on the targets (4 words a
     record). Uninterrupted: 2 epochs, 8 steps. Interrupted: 2 epochs capped at 3 steps: the position handed back
     is (0, 3) and the state's source part (FS_SOURCE) holds it with the source's fingerprint, while the source has
     already read the fourth batch ahead; then a resume from those buffers with a source reopened at (0, 0) (the
     state's position wins) capped at 5 steps: the parameters and both optimizer buffers are the uninterrupted
     run's words, and the curve's last iteration (epoch 1) is the same word. Every loss is finite and the mean loss
     falls.
  1  RECURRENT ESTIMATOR. rnn_fit_stream with a GRU(4) and output="sequence" (a target a step, masked by the
     lengths), Adam with the optimizer state in and out: 8 steps uninterrupted against 3 steps, the saved source
     part (position (0, 3)), and 5 steps from a source reopened at the saved position with the saved state: the
     same parameters and optimizer state, word for word; the step losses of the resumed run are the uninterrupted
     run's last five.
Sabotage arms coverage-autograd-batch_source_fit_read_ahead_{gpu,cpu} fail both units on their own column (the saved
position is (0, 4)).
"""
from std.ffi import c_int, external_call
from std.memory import bitcast
from std.testing import TestSuite
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from sequence.ops import FP, OPT_ADAM, CELL_GRU
from sequence.recurrent import Net, OptConfig, RnnOptions, TASK_MSE, rnn_fit_stream
from autograd.rules_recurrent import AutogradRnnStep
from autograd.model import FIT_STATE_WORDS, FS_SOURCE, IP64, LAYER_FLATTEN, LOSS_MSE, TARGET_DENSE, parse_model
from autograd.model_blocks import LAYER_RECURRENT
from autograd.model_execution import model_fit_source_native, model_float_words
from autograd.batch_source import (
    BatchSource, BatchSourceSpec, BS_FIELD_STEPS, BS_FIELD_TARGETS, SOURCE_PART_WORDS, batch_source_open,
    batch_source_seal, batch_source_fit_words, source_part_load,
)
from bindings.byte_checkpoint_native_run import byte_file_write, ByteCheckpointWriteFailure
from checks.native_tests.training_sequential_common import SeqModel, L_DROPOUT
from checks.native_tests.testsuite_cases import check_cases, check_cases_serial, CPU_COLUMN_BUILD, require

comptime STREAM_FITS_UNITS = 2
comptime _F = 2
comptime _T = 4  # the longest record: every batch is padded to it
comptime _W = 4  # target words a record: one a step (unit 1), or the regressor's four outputs (unit 0)
comptime _I64P = UnsafePointer[Int64, MutUntrackedOrigin]
comptime _SEQ_GRU = 2  # Sequential's recurrent cell word for a GRU (bindings/sequential_description_pack.mojo)


def _fail(message: String) raises:
    raise Error("stream_fits: " + message)


def _dir(unit: Int) raises -> String:
    var pid = Int(external_call["getpid", c_int]())
    var path = String("/tmp/mojolearn-stream-fits-") + String(pid) + "-" + String(unit)
    _ = external_call["mkdir", c_int](path.as_c_string_slice().unsafe_ptr(), c_int(448))
    return path


def _put(path: String, bytes: List[UInt8]) raises:
    var failure = ByteCheckpointWriteFailure()
    byte_file_write(path, Int(bytes.unsafe_ptr()), len(bytes), failure)


def _npy(descr: String, shape: String, payload_address: Int, payload_bytes: Int) -> List[UInt8]:
    """A version 1.0 .npy file as numpy.save writes it (header padded to 64 bytes, ending in a newline)."""
    var text = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + shape + ", }"
    var length = text.byte_length() + 1
    while (10 + length) % 64 != 0:  # driver-loop: over header padding bytes
        length += 1
    var out = List[UInt8]()
    var magic: List[UInt8] = [0x93, 78, 85, 77, 80, 89, 1, 0]
    out.extend(magic^)
    out.append(UInt8(length & 255))
    out.append(UInt8(length >> 8))
    for b in text.as_bytes():  # driver-loop: over header bytes
        out.append(b)
    while len(out) < 10 + length - 1:  # driver-loop: over header padding bytes
        out.append(32)
    out.append(10)
    var src = UnsafePointer[UInt8, MutUntrackedOrigin](unsafe_from_address=payload_address)
    for k in range(payload_bytes):  # driver-loop: over payload bytes (a tiny fixture)
        out.append(src[k])
    return out^


def _lengths(shard: Int) -> List[Int]:
    if shard == 0:
        return [1, 3, 2, 4]
    return [2, 1, 3]


def _write_sealed(root: String) raises:
    """The fixture's two shards with targets under `root`, then sealed."""
    for k in range(2):  # driver-loop: over the two shards
        var lengths = _lengths(k)
        var total = 0
        var offsets: List[Int] = [0]
        for i in range(len(lengths)):  # driver-loop: over the shard's records
            total += lengths[i]
            offsets.append(total)
        var steps = List[Float32]()
        for i in range(total * _F):  # driver-loop: over the shard's step words (a fixed draw)
            steps.append(Float32(((i * 37 + k * 11) % 17) - 8) / Float32(8))
        var targets = List[Float32]()
        for i in range(len(lengths) * _W):  # driver-loop: over the shard's target words (a fixed draw)
            targets.append(Float32(((i * 29 + k * 7) % 13) - 6) / Float32(12))
        var stem = root + "/part" + String(k)
        _put(stem + ".steps.npy", _npy("<f4", "(" + String(total) + ", " + String(_F) + ")",
                                       Int(steps.unsafe_ptr()), total * _F * 4))
        _put(stem + ".offsets.npy", _npy("<i8", "(" + String(len(offsets)) + ",)", Int(offsets.unsafe_ptr()),
                                         len(offsets) * 8))
        _put(stem + ".targets.npy", _npy("<f4", "(" + String(len(lengths)) + ", " + String(_W) + ")",
                                         Int(targets.unsafe_ptr()), len(lengths) * _W * 4))
        _ = steps^
        _ = offsets^
        _ = targets^
    var index = String("mojolearn-batch-shards v1\nshard part0\nshard part1\n")
    var bytes = List[UInt8]()
    for b in index.as_bytes():  # driver-loop: over the index's bytes
        bytes.append(b)
    _put(root + "/index.txt", bytes)
    require(batch_source_seal(root) == 2, "stream_fits: the fixture's shards were not both sealed")


def _spec(root: String, epoch: Int, consumed: Int) raises -> BatchSourceSpec:
    """The handle's words (BatchSource(root, 2, inputs=("steps",), target="targets", seed=5, bucket_batches=2)) at
    a position, with the fingerprint and the seal of an opening, as the door reads them."""
    var spec = BatchSourceSpec()
    spec.path = root
    spec.batch = 2
    spec.inputs = [BS_FIELD_STEPS]
    spec.target = BS_FIELD_TARGETS
    spec.shuffle = True
    spec.seed = 5
    spec.bucket = 2
    var probe = batch_source_open(spec.copy())
    spec.fingerprint = probe.fingerprint
    spec.sealed = probe.sealed
    probe.close()
    spec.epoch = epoch
    spec.consumed = consumed
    return spec^


def _same(name: String, a: List[Float32], b: List[Float32]) raises:
    if len(a) != len(b):
        _fail(name + ": lengths differ")
    for i in range(len(a)):  # small-loop(compared words)
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            _fail(name + ": word " + String(i) + " differs (" + String(a[i]) + " vs " + String(b[i]) + ")")


def _finite(name: String, a: List[Float32]) raises:
    for i in range(len(a)):  # small-loop(checked words)
        if a[i] != a[i] or abs(a[i]) > Float32(1e30):
            _fail(name + ": word " + String(i) + " is not finite")


# ---------------------------------------------------------------- unit 0: Sequential.fit(source=)
struct _SeqRun(Movable):
    var params: List[Float32]
    var m: List[Float32]
    var v: List[Float32]
    var state: List[Int64]
    var curve: List[Float32]
    var position: List[Int64]
    var n_iter: Int

    def __init__(out self, n_params: Int, epochs: Int):
        self.params = List[Float32](length=n_params, fill=Float32(0))
        self.m = List[Float32](length=n_params, fill=Float32(0))
        self.v = List[Float32](length=n_params, fill=Float32(0))
        self.state = List[Int64](length=FIT_STATE_WORDS, fill=Int64(0))
        self.curve = List[Float32](length=epochs, fill=Float32(0))
        self.position = List[Int64](length=2, fill=Int64(0))
        self.n_iter = 0


def _seq_model() -> SeqModel:
    var m = SeqModel()
    m.sample = [_T, _F]
    m.layer(LAYER_RECURRENT, 4, _SEQ_GRU, 0, 1)  # GRU(4), forward, its final state
    m.layer(LAYER_FLATTEN)
    m.layer(L_DROPOUT, f=0.25)
    m.linear(_W)
    m.loss = LOSS_MSE
    m.lr = 0.05
    return m^


def _seq_fit[E: Exec](
    mut ex: E, m: SeqModel, mut run: _SeqRun, root: String, epochs: Int, steps: Int, resume: Bool,
) raises:
    """One `fit(source=..., epochs=, steps=, resume=)` through the streamed entry (the door's words)."""
    var prog = m.program(_T * _F)
    var ip: List[Int] = [7, m.loss, TARGET_DENSE, m.opt, m.flags, m.sched, m.step_size, 2, epochs, 1, 11, 0,
                         m.patience, 0 if resume else 1, 1 if resume else 0]
    var fp = m._fit_floats()
    var previous = run.curve.copy()
    if resume:
        run.curve = List[Float32](length=run.n_iter + epochs, fill=Float32(0))
    var addrs: List[Int] = [
        0, 0, Int(run.params.unsafe_ptr()), Int(run.curve.unsafe_ptr()), Int(run.m.unsafe_ptr()),
        Int(run.v.unsafe_ptr()), Int(run.state.unsafe_ptr()), Int(previous.unsafe_ptr()) if resume else 0,
    ]
    var words = batch_source_fit_words(_spec(root, 0, 0), steps)
    run.n_iter = model_fit_source_native(ex, addrs, prog, ip, fp, root, words, Int(run.position.unsafe_ptr()))
    ex.sync()
    _ = previous^


def run_sequential_unit[E: Exec](root: String, var ex: E) raises:
    var m = _seq_model()
    var spec = parse_model(m.program(_T * _F), model_float_words(m.table()))
    var whole = _SeqRun(spec.n_params, 2)
    _seq_fit(ex, m, whole, root, 2, 0, False)
    require(Int(whole.position[0]) == 1 and Int(whole.position[1]) == 4,
            "stream_fits: two epochs end at (1, 4), not (" + String(whole.position[0]) + ", "
            + String(whole.position[1]) + ")")
    _finite("sequential loss curve", whole.curve)
    require(whole.n_iter == 2 and whole.curve[1] < whole.curve[0], "stream_fits: the streamed Sequential loss "
            + "did not fall over its two epochs")
    var part = _SeqRun(spec.n_params, 2)
    _seq_fit(ex, m, part, root, 2, 3, False)
    require(Int(part.position[0]) == 0 and Int(part.position[1]) == 3,
            "stream_fits: three steps hand back the consumed position (0, 3)")
    var saved = source_part_load(_I64P(unsafe_from_address=Int(part.state.unsafe_ptr())) + FS_SOURCE)
    require(saved.present and saved.epoch == 0 and saved.consumed == 3,
            "stream_fits: the training state saves the consumed position (0, 3), not (" + String(saved.epoch)
            + ", " + String(saved.consumed) + ")")
    _seq_fit(ex, m, part, root, 2, 5, True)
    _same("sequential resumed parameters", part.params, whole.params)
    _same("sequential resumed first moments", part.m, whole.m)
    _same("sequential resumed second moments", part.v, whole.v)
    var a: List[Float32] = [whole.curve[1]]
    var b: List[Float32] = [part.curve[part.n_iter - 1]]
    _same("sequential last epoch's loss", b, a)
    print("STREAM_FITS sequential loss", whole.curve[0], "->", whole.curve[1])


# ---------------------------------------------------------------- unit 1: the recurrent estimators' streamed fit
struct _RnnRun(Movable):
    var params: List[Float32]
    var state: List[Float32]
    var losses: List[Float32]
    var saved: List[Int64]
    var position: List[Int64]

    def __init__(out self, n_params: Int):
        self.params = List[Float32](length=n_params, fill=Float32(0))
        for i in range(n_params):  # small-loop(a fixed initial draw)
            self.params[i] = Float32(((i * 53) % 19) - 9) / Float32(40)
        self.state = List[Float32](length=3 * n_params + 4, fill=Float32(0))
        self.state[3 * n_params] = Float32(1)
        self.state[3 * n_params + 1] = Float32(1)
        self.state[3 * n_params + 2] = Float32(1)
        self.losses = List[Float32]()
        self.saved = List[Int64](length=SOURCE_PART_WORDS, fill=Int64(0))
        self.position = List[Int64](length=2, fill=Int64(0))


def _rnn_fit[E: Exec](mut ex: E, net: Net, mut run: _RnnRun, root: String, epoch: Int, consumed: Int,
                      total: Int) raises:
    var opts = RnnOptions()
    opts.seq_out = True
    opts.has_state = True
    opts.state = FP(unsafe_from_address=Int(run.state.unsafe_ptr()))
    opts.epochs = 1
    var cfg = OptConfig(OPT_ADAM, 0, Float32(0.9), Float32(0.999), Float32(1e-8), Float32(0), Float32(0))
    var lrs = List[Float32](length=total, fill=Float32(0.05))
    run.losses = List[Float32](length=total, fill=Float32(0))
    var src = batch_source_open(_spec(root, epoch, consumed))
    rnn_fit_stream[AutogradRnnStep](ex, net, TASK_MSE, src, total, _T, UInt64(17),
                                    FP(unsafe_from_address=Int(run.params.unsafe_ptr())),
                                    FP(unsafe_from_address=Int(run.losses.unsafe_ptr())),
                                    FP(unsafe_from_address=Int(lrs.unsafe_ptr())), cfg, Float32(0), 0, opts,
                                    Int(run.saved.unsafe_ptr()), Int(run.position.unsafe_ptr()))
    ex.sync()
    src.close()
    _ = lrs^


def run_recurrent_unit[E: Exec](root: String, var ex: E) raises:
    var net = Net(CELL_GRU, _F, 4, 1, 1)
    var whole = _RnnRun(net.n_params())
    _rnn_fit(ex, net, whole, root, 0, 0, 8)
    _finite("recurrent step losses", whole.losses)
    var part = _RnnRun(net.n_params())
    _rnn_fit(ex, net, part, root, 0, 0, 3)
    var saved = source_part_load(_I64P(unsafe_from_address=Int(part.saved.unsafe_ptr())))
    require(saved.present and saved.epoch == 0 and saved.consumed == 3,
            "stream_fits: the estimator saves the consumed position (0, 3), not (" + String(saved.epoch) + ", "
            + String(saved.consumed) + ")")
    _rnn_fit(ex, net, part, root, saved.epoch, saved.consumed, 5)
    _same("recurrent resumed parameters", part.params, whole.params)
    _same("recurrent resumed optimizer state", part.state, whole.state)
    var tail = List[Float32]()
    for k in range(3, 8):  # small-loop(the uninterrupted run's last five step losses)
        tail.append(whole.losses[k])
    _same("recurrent resumed step losses", part.losses, tail)
    print("STREAM_FITS recurrent loss", whole.losses[0], "->", whole.losses[7])


def test_stream_fits() raises:
    comptime if CPU_COLUMN_BUILD:
        def host_unit(unit: Int) raises:
            var root = _dir(unit)
            _write_sealed(root)
            if unit == 0:
                run_sequential_unit[HostExec](root, HostExec())
            else:
                run_recurrent_unit[HostExec](root, HostExec())
        check_cases("stream_fits", STREAM_FITS_UNITS, host_unit)
    else:
        def device_unit(unit: Int) raises:
            var root = _dir(unit)
            _write_sealed(root)
            if unit == 0:
                run_sequential_unit[DeviceExec](root, DeviceExec())
            else:
                run_recurrent_unit[DeviceExec](root, DeviceExec())
        check_cases_serial("stream_fits", STREAM_FITS_UNITS, device_unit)
    print("PASS STREAM_FITS units", STREAM_FITS_UNITS)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
