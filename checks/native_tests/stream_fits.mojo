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
  2  CLASS WEIGHTS AND VALIDATION (lane fable-finalize). A GRU(4) classifier of three classes with
     output="sequence" on class targets (one a step). Class weights of 1 give the unweighted fit's words (the
     weight of a row is its label's weight, times nothing else); weights (1, 3, 0.5) move the parameters; weights
     of 0 leave the parameters at their initial words (every valid step's weight comes from its label: one row
     weighed by a stale or foreign word would move them); validation sequences leave the training words unchanged
     and give one finite loss for each of the two passes over the source, with the passes completed and the best
     one.
  3  RECORDS AND THE MONITOR (lane stream-fits-2). The classifier of unit 2: sample weights of 1 for every record
     are the unweighted words, of 0 leave the parameters; a weight a record equals that weight at each of its steps
     (both gathered by the batch's record ids); a nonzero h_0 a record moves the fit;
     the 'balanced' counts of the one admission read equal the fixture's valid-step counts; early stopping
     (patience 1, every later pass stalls) runs 8 of 12 steps, and 3 steps then the rest from the saved position,
     state and monitor words stop at the same step with the same parameters, state and monitor; restore_best keeps
     the best pass's parameters (the 4-step run's when pass 1 is best, and the source at (0, 4)).
  4  SEQUENTIAL VALIDATION (lane stream-fits-2). Unit 0's regressor with validation data (3 sequences with their
     lengths) through the validated streamed form: one finite validation word a pass and the unvalidated fit's
     parameters; restore_best_weights gives the parameters of a fit of best-pass epochs and the source at that
     pass's end.
Sabotage arms coverage-autograd-batch_source_fit_read_ahead_{gpu,cpu} fail units 0 and 1 on their own column (the saved
position is (0, 4)).
"""
from std.ffi import c_int, external_call
from std.memory import bitcast
from std.testing import TestSuite
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from sequence.ops import FP, OPT_ADAM, CELL_GRU
from sequence.recurrent import Net, OptConfig, RnnOptions, TASK_MSE, TASK_CE, rnn_fit_stream
from autograd.rules_recurrent import AutogradRnnStep
from autograd.model import FIT_STATE_WORDS, FS_SOURCE, IP64, LAYER_FLATTEN, LOSS_MSE, TARGET_DENSE, parse_model
from autograd.model_blocks import LAYER_RECURRENT
from autograd.model_execution import model_fit_source_native, model_float_words
from autograd.batch_source import (
    BatchSource, BatchSourceSpec, BS_FIELD_STEPS, BS_FIELD_TARGETS, SOURCE_PART_WORDS, batch_source_open,
    batch_source_seal, batch_source_fit_words, source_part_load, batch_source_class_counts, BatchSourcePtr,
    RNN_STREAM_MONITOR_WORDS, RNN_STREAM_POSITION_WORDS,
)
from bindings.byte_checkpoint_native_run import byte_file_write, ByteCheckpointWriteFailure
from core.cast_in import CastIn
from checks.native_tests.training_sequential_common import SeqModel, L_DROPOUT
from checks.native_tests.testsuite_cases import check_cases, check_cases_serial, CPU_COLUMN_BUILD, require

comptime STREAM_FITS_UNITS = 5
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


def _write_sealed(root: String, classes: Bool = False) raises:
    """The fixture's two shards with targets under `root`, then sealed. `classes`: the targets are class indices
    in [0, 3), one a step."""
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
            if classes:
                targets.append(Float32((i * 5 + k) % 3))
            else:
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
    # the source a door keeps on the handle (here opened for the call at the handle's (0, 0)), by address
    var src = batch_source_open(_spec(root, 0, 0))
    var words = batch_source_fit_words(BatchSourcePtr(unsafe_from_address=Int(UnsafePointer(to=src))), steps)
    run.n_iter = model_fit_source_native(ex, addrs, prog, ip, fp, words, Int(run.position.unsafe_ptr()))
    ex.sync()
    src.close()
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


def _seq_fit_validated[E: Exec](
    mut ex: E, m: SeqModel, mut run: _SeqRun, root: String, epochs: Int, restore: Bool,
    mut vcurve: List[Float32], mut vinfo: List[Int64],
) raises:
    """`fit(source=..., epochs=, validation_data=(X, y, lengths), restore_best_weights=)` through the streamed
    entry's validated form (15 addresses, 16 integers, five input descriptors; lane stream-fits-2)."""
    var prog = m.program(_T * _F)
    var ip: List[Int] = [7, m.loss, TARGET_DENSE, m.opt, m.flags, m.sched, m.step_size, 2, epochs, 1, 11, 0,
                         m.patience, 1, 0, 1 if restore else 0]
    var fp = m._fit_floats()
    var vx = List[Float32]()
    for i in range(3 * _T * _F):  # small-loop(a fixed draw of validation steps)
        vx.append(Float32(((i * 31) % 11) - 5) / Float32(6))
    var vy = List[Float32]()
    for i in range(3 * _W):  # small-loop(a fixed draw of validation targets)
        vy.append(Float32(((i * 13) % 7) - 3) / Float32(5))
    var vl: List[Float32] = [4.0, 2.0, 3.0]
    var addrs: List[Int] = [
        0, 0, Int(run.params.unsafe_ptr()), Int(run.curve.unsafe_ptr()), Int(run.m.unsafe_ptr()),
        Int(run.v.unsafe_ptr()), Int(run.state.unsafe_ptr()), 0, 0, 0, 0, Int(vcurve.unsafe_ptr()),
        Int(vinfo.unsafe_ptr()), 0, Int(vl.unsafe_ptr()),
    ]
    var casts: List[CastIn] = [CastIn.f32_c(0, 1, 1), CastIn.f32_c(0, 1, 1),
                               CastIn.f32_c(Int(vx.unsafe_ptr()), 3, _T * _F),
                               CastIn.f32_c(Int(vy.unsafe_ptr()), 3, _W), CastIn.f32_c(0, 1, 1)]
    var src = batch_source_open(_spec(root, 0, 0))
    var words = batch_source_fit_words(BatchSourcePtr(unsafe_from_address=Int(UnsafePointer(to=src))), 0)
    run.n_iter = model_fit_source_native(ex, addrs, prog, ip, fp, words, Int(run.position.unsafe_ptr()),
                                         0, 0, Optional[List[CastIn]](casts^))
    ex.sync()
    src.close()
    _ = vx^
    _ = vy^
    _ = vl^
    _ = vcurve
    _ = vinfo


def run_sequential_validation_unit[E: Exec](root: String, var ex: E) raises:
    """Unit 4 (lane stream-fits-2): Sequential.fit(source=) with validation data: one validation loss a pass, the
    training words of the unvalidated fit; restore_best_weights keeps the best pass's parameters (those of a fit
    of that many epochs) and the source at that pass's end."""
    var m = _seq_model()
    var spec = parse_model(m.program(_T * _F), model_float_words(m.table()))
    var plain = _SeqRun(spec.n_params, 3)
    _seq_fit(ex, m, plain, root, 3, 0, False)
    var curve = List[Float32](length=3, fill=Float32(0))
    var info = List[Int64](length=2, fill=Int64(0))
    var seen = _SeqRun(spec.n_params, 3)
    _seq_fit_validated(ex, m, seen, root, 3, False, curve, info)
    _same("sequential validation leaves the parameters", seen.params, plain.params)
    _finite("sequential validation losses", curve)
    require(Int(info[0]) == 3, "stream_fits: three passes give three validation words, not " + String(info[0]))
    var kept = _SeqRun(spec.n_params, 3)
    var curve2 = List[Float32](length=3, fill=Float32(0))
    var info2 = List[Int64](length=2, fill=Int64(0))
    _seq_fit_validated(ex, m, kept, root, 3, True, curve2, info2)
    var best = Int(info2[1])
    var reference = _SeqRun(spec.n_params, best)
    _seq_fit(ex, m, reference, root, best, 0, False)
    _same("sequential restore_best_weights: parameters", kept.params, reference.params)
    require(Int(kept.position[0]) == best - 1 and Int(kept.position[1]) == 4,
            "stream_fits: restore_best_weights leaves the source at the best pass's end (" + String(best - 1)
            + ", 4), not (" + String(kept.position[0]) + ", " + String(kept.position[1]) + ")")
    print("STREAM_FITS sequential validation best pass", best)


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
        self.position = List[Int64](length=RNN_STREAM_POSITION_WORDS, fill=Int64(0))


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


# ---------------------------------------------------------------- unit 2: class weights and validation
def _rnn_fit_classes[E: Exec](
    mut ex: E, net: Net, mut run: _RnnRun, root: String, total: Int, weights: List[Float32], validate: Bool,
    mut vcurve: List[Float32],
) raises:
    """`total` streamed steps of the classifier from position (0, 0); `weights` empty: no class weights; `validate`:
    three validation sequences, their losses into vcurve (2 passes, then the passes run and the best one)."""
    var opts = RnnOptions()
    opts.seq_out = True
    opts.has_state = True
    opts.state = FP(unsafe_from_address=Int(run.state.unsafe_ptr()))
    opts.epochs = 2
    if len(weights) > 0:
        opts.has_cw = True
        opts.cw = FP(unsafe_from_address=Int(weights.unsafe_ptr()))
    var vx = List[Float32]()
    var vy = List[Float32]()
    var vl: List[Float32] = [4.0, 2.0, 3.0]
    for i in range(3 * _T * _F):  # small-loop(a fixed draw of validation steps)
        vx.append(Float32(((i * 31) % 11) - 5) / Float32(6))
    for i in range(3 * _T):  # small-loop(a fixed draw of validation class indices)
        vy.append(Float32((i * 7) % 3))
    if validate:
        opts.nv = 3
        opts.vx = FP(unsafe_from_address=Int(vx.unsafe_ptr()))
        opts.vy = FP(unsafe_from_address=Int(vy.unsafe_ptr()))
        opts.has_vl = True
        opts.vl = FP(unsafe_from_address=Int(vl.unsafe_ptr()))
        opts.vcurve = FP(unsafe_from_address=Int(vcurve.unsafe_ptr()))
    var cfg = OptConfig(OPT_ADAM, 0, Float32(0.9), Float32(0.999), Float32(1e-8), Float32(0), Float32(0))
    var lrs = List[Float32](length=total, fill=Float32(0.05))
    run.losses = List[Float32](length=total, fill=Float32(0))
    var src = batch_source_open(_spec(root, 0, 0))
    rnn_fit_stream[AutogradRnnStep](ex, net, TASK_CE, src, total, _T, UInt64(17),
                                    FP(unsafe_from_address=Int(run.params.unsafe_ptr())),
                                    FP(unsafe_from_address=Int(run.losses.unsafe_ptr())),
                                    FP(unsafe_from_address=Int(lrs.unsafe_ptr())), cfg, Float32(0), 0, opts,
                                    Int(run.saved.unsafe_ptr()), Int(run.position.unsafe_ptr()))
    ex.sync()
    src.close()
    _ = lrs^
    _ = vx^
    _ = vy^
    _ = vl^
    _ = weights
    _ = vcurve


def run_weights_unit[E: Exec](root: String, var ex: E) raises:
    var net = Net(CELL_GRU, _F, 4, 1, 3)
    var none = List[Float32]()
    var unused = List[Float32](length=4, fill=Float32(0))
    var plain = _RnnRun(net.n_params())
    var start = plain.params.copy()
    _rnn_fit_classes(ex, net, plain, root, 8, none, False, unused)
    _finite("classifier step losses", plain.losses)
    var ones: List[Float32] = [1.0, 1.0, 1.0]
    var unit_w = _RnnRun(net.n_params())
    _rnn_fit_classes(ex, net, unit_w, root, 8, ones, False, unused)
    _same("class weights of 1: parameters", unit_w.params, plain.params)
    _same("class weights of 1: step losses", unit_w.losses, plain.losses)
    var mixed: List[Float32] = [1.0, 3.0, 0.5]
    var weighted = _RnnRun(net.n_params())
    _rnn_fit_classes(ex, net, weighted, root, 8, mixed, False, unused)
    _finite("weighted classifier step losses", weighted.losses)
    var moved = False
    for i in range(len(weighted.params)):  # small-loop(check values)
        if bitcast[DType.uint32](weighted.params[i]) != bitcast[DType.uint32](plain.params[i]):
            moved = True
    require(moved, "stream_fits: class weights (1, 3, 0.5) left the parameters of the unweighted fit")
    var zeros: List[Float32] = [0.0, 0.0, 0.0]
    var still = _RnnRun(net.n_params())
    _rnn_fit_classes(ex, net, still, root, 8, zeros, False, unused)
    _same("class weights of 0: the parameters do not move", still.params, start)
    # validation: the training words of the weighted fit, and one loss a pass (2 passes, 2 result words, the
    # monitor words: all 0, no earlier call)
    var curve = List[Float32](length=4 + RNN_STREAM_MONITOR_WORDS, fill=Float32(0))
    var watched = _RnnRun(net.n_params())
    _rnn_fit_classes(ex, net, watched, root, 8, mixed, True, curve)
    _same("validation leaves the training parameters", watched.params, weighted.params)
    _same("validation leaves the step losses", watched.losses, weighted.losses)
    var seen: List[Float32] = [curve[0], curve[1]]
    _finite("validation losses", seen)
    require(curve[0] > Float32(0) and curve[1] > Float32(0), "stream_fits: a validation loss is not positive")
    require(curve[2] == Float32(2), "stream_fits: two passes over the source give two validation losses, not "
            + String(curve[2]))
    var want_best = Float32(2) if curve[1] < curve[0] else Float32(1)
    require(curve[3] == want_best, "stream_fits: the best pass is " + String(curve[3]) + ", not "
            + String(want_best))
    print("STREAM_FITS validation", curve[0], "->", curve[1])


# ---------------------------------------------------------------- unit 3: per-record weights and states, the monitor
struct _Opt(Copyable, Movable):
    """The options of one unit-3 fit: per-record sample weights (`sw`, n or n T words; `per_step`), class weights,
    h_0 (L n H words), validation with patience / restore, the curve words (passes, 2 results, the monitor)."""
    var sw: List[Float32]
    var per_step: Bool
    var cw: List[Float32]
    var h0: List[Float32]
    var validate: Bool
    var patience: Int
    var tol: Float32
    var restore: Bool
    var passes: Int

    def __init__(out self):
        self.sw = List[Float32]()
        self.per_step = False
        self.cw = List[Float32]()
        self.h0 = List[Float32]()
        self.validate = False
        self.patience = 0
        self.tol = Float32(0)
        self.restore = False
        self.passes = 3


def _rnn_fit_opts[E: Exec](
    mut ex: E, net: Net, mut run: _RnnRun, root: String, epoch: Int, consumed: Int, total: Int, o: _Opt,
    mut vcurve: List[Float32],
) raises:
    """`total` streamed classifier steps from (epoch, consumed) with the options `o` (validation curve `vcurve`:
    o.passes + 2 + RNN_STREAM_MONITOR_WORDS words, its monitor words the carried ones)."""
    var opts = RnnOptions()
    opts.seq_out = True
    opts.has_state = True
    opts.state = FP(unsafe_from_address=Int(run.state.unsafe_ptr()))
    opts.epochs = o.passes
    if len(o.sw) > 0:
        opts.has_sw = True
        opts.sw_steps = o.per_step
        opts.sw = FP(unsafe_from_address=Int(o.sw.unsafe_ptr()))
    if len(o.cw) > 0:
        opts.has_cw = True
        opts.cw = FP(unsafe_from_address=Int(o.cw.unsafe_ptr()))
    if len(o.h0) > 0:
        opts.has_h0 = True
        opts.h0 = FP(unsafe_from_address=Int(o.h0.unsafe_ptr()))
    var vx = List[Float32]()
    var vy = List[Float32]()
    var vl: List[Float32] = [4.0, 2.0, 3.0]
    for i in range(3 * _T * _F):  # small-loop(a fixed draw of validation steps)
        vx.append(Float32(((i * 31) % 11) - 5) / Float32(6))
    for i in range(3 * _T):  # small-loop(a fixed draw of validation class indices)
        vy.append(Float32((i * 7) % 3))
    if o.validate:
        opts.nv = 3
        opts.vx = FP(unsafe_from_address=Int(vx.unsafe_ptr()))
        opts.vy = FP(unsafe_from_address=Int(vy.unsafe_ptr()))
        opts.has_vl = True
        opts.vl = FP(unsafe_from_address=Int(vl.unsafe_ptr()))
        opts.vcurve = FP(unsafe_from_address=Int(vcurve.unsafe_ptr()))
        opts.patience = o.patience
        opts.tol = o.tol
        opts.restore = o.restore
    var cfg = OptConfig(OPT_ADAM, 0, Float32(0.9), Float32(0.999), Float32(1e-8), Float32(0), Float32(0))
    var lrs = List[Float32](length=total, fill=Float32(0.05))
    run.losses = List[Float32](length=total, fill=Float32(0))
    var src = batch_source_open(_spec(root, epoch, consumed))
    rnn_fit_stream[AutogradRnnStep](ex, net, TASK_CE, src, total, _T, UInt64(17),
                                    FP(unsafe_from_address=Int(run.params.unsafe_ptr())),
                                    FP(unsafe_from_address=Int(run.losses.unsafe_ptr())),
                                    FP(unsafe_from_address=Int(lrs.unsafe_ptr())), cfg, Float32(0), 0, opts,
                                    Int(run.saved.unsafe_ptr()), Int(run.position.unsafe_ptr()))
    ex.sync()
    src.close()
    _ = lrs^
    _ = vx^
    _ = vy^
    _ = vl^
    _ = vcurve


def _moved(a: List[Float32], b: List[Float32]) -> Bool:
    for i in range(len(a)):  # small-loop(check values)
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            return True
    return False


def run_records_unit[E: Exec](root: String, var ex: E) raises:
    """Unit 3 (lane stream-fits-2): sample weights and initial states by record, 'balanced' counts, early stopping,
    restore_best and the monitor carried across calls, on the classifier fixture of unit 2 (7 records)."""
    var net = Net(CELL_GRU, _F, 4, 1, 3)
    var n = 7
    var cv = List[Float32](length=1, fill=Float32(0))
    var plain = _RnnRun(net.n_params())
    var start = plain.params.copy()
    _rnn_fit_opts(ex, net, plain, root, 0, 0, 8, _Opt(), cv)
    # (a) a weight of 1 for every record is the unweighted fit
    var o = _Opt()
    o.sw = List[Float32](length=n, fill=Float32(1))
    var ones = _RnnRun(net.n_params())
    _rnn_fit_opts(ex, net, ones, root, 0, 0, 8, o, cv)
    _same("sample weights of 1: parameters", ones.params, plain.params)
    _same("sample weights of 1: step losses", ones.losses, plain.losses)
    # (b) a weight of 0 for every record leaves the parameters
    o.sw = List[Float32](length=n, fill=Float32(0))
    var still = _RnnRun(net.n_params())
    _rnn_fit_opts(ex, net, still, root, 0, 0, 8, o, cv)
    _same("sample weights of 0: the parameters do not move", still.params, start)
    # (c) a weight a record equals the same weight at each of its steps (the record's id selects both rows)
    o.sw = List[Float32]()
    for r in range(n):  # small-loop(records)
        o.sw.append(Float32(r + 1) / Float32(4))
    var by_record = _RnnRun(net.n_params())
    _rnn_fit_opts(ex, net, by_record, root, 0, 0, 8, o, cv)
    require(_moved(by_record.params, plain.params), "stream_fits: per-record weights left the unweighted words")
    var per = _Opt()
    per.per_step = True
    for r in range(n):  # small-loop(records)
        for _ in range(_T):  # small-loop(steps)
            per.sw.append(Float32(r + 1) / Float32(4))
    var by_step = _RnnRun(net.n_params())
    _rnn_fit_opts(ex, net, by_step, root, 0, 0, 8, per, cv)
    _same("per-record and per-step sample weights", by_step.params, by_record.params)
    # (d) a nonzero h_0 a record moves the fit and keeps it finite
    var h = _Opt()
    h.h0 = List[Float32](length=n * 4, fill=Float32(0))
    for i in range(n * 4):  # small-loop(state words)
        h.h0[i] = Float32((i % 5) - 2) / Float32(4)
    var some_h = _RnnRun(net.n_params())
    _rnn_fit_opts(ex, net, some_h, root, 0, 0, 8, h, cv)
    require(_moved(some_h.params, plain.params), "stream_fits: a nonzero h_0 left the zero-state words")
    _finite("h_0 step losses", some_h.losses)
    # (e) 'balanced': the counts of the one admission read, valid steps only, against the fixture's own count
    var src = batch_source_open(_spec(root, 0, 0))
    var counts = batch_source_class_counts(src, 3, True, "stream_fits")
    src.close()
    var want = List[Int](length=3, fill=0)
    for k in range(2):  # driver-loop: over the two shards
        var lengths = _lengths(k)
        for r in range(len(lengths)):  # small-loop(records)
            for t in range(lengths[r]):  # small-loop(valid steps)
                want[((r * _W + t) * 5 + k) % 3] += 1
    for c in range(3):  # small-loop(classes)
        require(counts[c] == want[c], "stream_fits: class " + String(c) + " counted " + String(counts[c])
                + " valid steps, not " + String(want[c]))
    # (f) early stopping (patience 1, tol huge: the second pass stalls) stops after two passes, 8 of 12 steps;
    # three steps, then the rest with the carried state and monitor, stop at the same step with the same words
    var stop = _Opt()
    stop.validate = True
    stop.patience = 1
    stop.tol = Float32(1e9)
    var vw = List[Float32](length=3 + 2 + RNN_STREAM_MONITOR_WORDS, fill=Float32(0))
    var whole = _RnnRun(net.n_params())
    _rnn_fit_opts(ex, net, whole, root, 0, 0, 12, stop, vw)
    require(Int(whole.position[2]) == 8, "stream_fits: early stopping ran " + String(whole.position[2])
            + " steps, not 8")
    require(vw[3] == Float32(2), "stream_fits: early stopping completed " + String(vw[3]) + " passes, not 2")
    var vp = List[Float32](length=3 + 2 + RNN_STREAM_MONITOR_WORDS, fill=Float32(0))
    var split = _RnnRun(net.n_params())
    _rnn_fit_opts(ex, net, split, root, 0, 0, 3, stop, vp)
    var saved = source_part_load(_I64P(unsafe_from_address=Int(split.saved.unsafe_ptr())))
    var vq = List[Float32](length=3 + 2 + RNN_STREAM_MONITOR_WORDS, fill=Float32(0))
    for i in range(RNN_STREAM_MONITOR_WORDS):  # small-loop(the carried monitor words)
        vq[5 + i] = vp[5 + i]
    _rnn_fit_opts(ex, net, split, root, saved.epoch, saved.consumed, 9, stop, vq)
    require(Int(split.position[2]) == 5, "stream_fits: the resumed call ran " + String(split.position[2])
            + " steps, not 5")
    _same("early stopping resumed: parameters", split.params, whole.params)
    _same("early stopping resumed: optimizer state", split.state, whole.state)
    var monitor_a: List[Float32] = [vw[5], vw[6], vw[7], vw[8]]
    var monitor_b: List[Float32] = [vq[5], vq[6], vq[7], vq[8]]
    _same("early stopping resumed: monitor words", monitor_b, monitor_a)
    # (g) restore_best over two passes: the best pass's parameters (the 4- or 8-step run's)
    var keep = _Opt()
    keep.validate = True
    keep.restore = True
    keep.passes = 2
    var vr = List[Float32](length=2 + 2 + RNN_STREAM_MONITOR_WORDS, fill=Float32(0))
    var restored = _RnnRun(net.n_params())
    _rnn_fit_opts(ex, net, restored, root, 0, 0, 8, keep, vr)
    var four = _RnnRun(net.n_params())
    _rnn_fit_opts(ex, net, four, root, 0, 0, 4, _Opt(), cv)
    if vr[3] == Float32(1):
        _same("restore_best (pass 1): parameters", restored.params, four.params)
        require(Int(restored.position[0]) == 0 and Int(restored.position[1]) == 4,
                "stream_fits: restore_best leaves the source at the best pass's end (0, 4)")
    else:
        _same("restore_best (pass 2): parameters", restored.params, plain.params)
    print("STREAM_FITS records: early stop at", whole.position[2], "best pass", vr[3])


def test_stream_fits() raises:
    comptime if CPU_COLUMN_BUILD:
        def host_unit(unit: Int) raises:
            var root = _dir(unit)
            _write_sealed(root, unit == 2 or unit == 3)
            if unit == 0:
                run_sequential_unit[HostExec](root, HostExec())
            elif unit == 1:
                run_recurrent_unit[HostExec](root, HostExec())
            elif unit == 2:
                run_weights_unit[HostExec](root, HostExec())
            elif unit == 3:
                run_records_unit[HostExec](root, HostExec())
            else:
                run_sequential_validation_unit[HostExec](root, HostExec())
        check_cases("stream_fits", STREAM_FITS_UNITS, host_unit)
    else:
        def device_unit(unit: Int) raises:
            var root = _dir(unit)
            _write_sealed(root, unit == 2 or unit == 3)
            if unit == 0:
                run_sequential_unit[DeviceExec](root, DeviceExec())
            elif unit == 1:
                run_recurrent_unit[DeviceExec](root, DeviceExec())
            elif unit == 2:
                run_weights_unit[DeviceExec](root, DeviceExec())
            elif unit == 3:
                run_records_unit[DeviceExec](root, DeviceExec())
            else:
                run_sequential_validation_unit[DeviceExec](root, DeviceExec())
        check_cases_serial("stream_fits", STREAM_FITS_UNITS, device_unit)
    print("PASS STREAM_FITS units", STREAM_FITS_UNITS)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
