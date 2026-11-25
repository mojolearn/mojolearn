# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Native check of the batch source and the one-call training checkpoint (lane batch-source, 2026-11-25;
autograd/batch_source.mojo, autograd/program_training_checkpoint.mojo; verify lane ops-batch-source-resume).

The fixture is the verify lane's shape, written here as .npy files: 7 records of lengths 1 to 4 in two shards,
F = 2, batch 2, two batches a bucket window, shuffled from seed 5 (4 batches an epoch).

Units (one column each build: the host column with -D MOJOLEARN_COLUMN_CPU=1, else the device column):
  0  THE PLAN. For epochs 0 to 2: every record exactly once; inside each bucket window the batches are cut from a
     length-sorted window (each batch's lengths are at most the next batch's of the same window, in window order);
     the same seed gives the same plan twice; seed 6 gives another plan for at least one epoch (a plan that ignored
     the seed would not move).
  1  THE RESUME (Andrew's deciding test, in Mojo). Program.fit's streamed loop trains three steps (the fourth batch
     is read ahead: held_batch 3, consumed 3), program_training_save writes the checkpoint, two more steps run
     uninterrupted; a FRESH session (and a source reopened from the saved position alone) loads it with
     program_training_load and runs two steps: the same next record ids, the same step losses (the dropout masks)
     and the same parameters, word for word.
     Lane fable-finalize, in the same unit: THE SEAL (a twin directory with other step values under the same record
     lengths has the same fingerprint unsealed and another one sealed; a resume on it is refused by name, and so
     are other input fields, another target field and another seed); THE REPLACEMENT (a hard link taken on the
     first checkpoint still loads as the first checkpoint after a second save to the same name, which then holds
     the second: the writer replaces the file and never opens the previous one, so an interrupted save cannot
     destroy it).
  2  ONE OPENING (lane fr2-followups). One opened source serves 100 `Program.fit(source=, steps=1)` calls (the
     handle's native owner keeps it across calls, autograd/batch_source.mojo THE OWNER): the shards are never
     reopened (the same descriptors), every epoch is planned exactly once (the counter is here, not in the product:
     a plan is seen as a new record-order list or a new plan epoch after a call), and after every call the batch
     the next call trains is already read ahead (so a call reads one batch, not the epoch's index and plan).
Sabotage arms coverage-autograd-batch_source_read_ahead_{gpu,cpu}, coverage-autograd-program_training_drop_group_
{gpu,cpu} and coverage-autograd-program_training_in_place_{gpu,cpu} fail unit 1 on their own column.
"""
from std.ffi import c_int, external_call
from std.memory import bitcast
from std.testing import TestSuite
from sequence.exec_trait import Exec
from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from sequence.ops import OPT_ADAMW, FP
from core.hostptr import f32_ptr
from autograd.ops_api import (
    OpsSession, ops_session, ops_session_open, ops_session_close, OPS_HOST, OPS_DEVICE, OPS_MATMUL, OPS_ADD,
    OPS_DROPOUT, OPS_RESHAPE_N, OPS_SUM_N,
)
from autograd.optimizer_requests import OpsOptimizerRequest, OpsOptimizerExtras
from autograd.batch_source import (
    BatchSource, BatchSourceSpec, BS_FIELD_STEPS, BS_FIELD_LENGTHS, BS_FIELD_MASK, BS_FIELD_TARGETS,
    batch_source_open, batch_source_saved, batch_source_seal,
)
from autograd.program_runtime import train_program_source
from autograd.program_training_checkpoint import program_training_save, program_training_load, ProgramTrainingSource
from bindings.byte_checkpoint_native_run import byte_file_write, ByteCheckpointWriteFailure
from checks.native_tests.ops_higher_common import tensor
from checks.native_tests.testsuite_cases import check_cases, check_cases_serial, CPU_COLUMN_BUILD, require

comptime BATCH_SOURCE_UNITS = 3
comptime _STEP_CALLS = 100
comptime _F = 2


def _fail(message: String) raises:
    raise Error("batch_source: " + message)


def _dir(unit: Int) raises -> String:
    var base = String("/tmp")
    var pid = Int(external_call["getpid", c_int]())
    var path = base + "/mojolearn-batch-source-" + String(pid) + "-" + String(unit)
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


def _write_shards(root: String, salt: Int = 0) raises:
    """The fixture's two shards under `root`, unsealed. Another `salt` gives other step values under the same
    record lengths (same-shaped replacement data)."""
    for k in range(2):  # driver-loop: over the two shards
        var lengths = _lengths(k)
        var total = 0
        var offsets: List[Int] = [0]
        for i in range(len(lengths)):  # driver-loop: over the shard's records
            total += lengths[i]
            offsets.append(total)
        var steps = List[Float32]()
        for i in range(total * _F):  # driver-loop: over the shard's step words (a fixed draw)
            steps.append(Float32(((i * 37 + k * 11 + salt) % 17) - 8) / Float32(8))
        var stem = root + "/part" + String(k)
        _put(stem + ".steps.npy", _npy("<f4", "(" + String(total) + ", " + String(_F) + ")",
                                       Int(steps.unsafe_ptr()), total * _F * 4))
        _put(stem + ".offsets.npy", _npy("<i8", "(" + String(len(offsets)) + ",)", Int(offsets.unsafe_ptr()),
                                         len(offsets) * 8))
        _ = steps^
        _ = offsets^
    var index = String("mojolearn-batch-shards v1\nshard part0\nshard part1\n")
    var bytes = List[UInt8]()
    for b in index.as_bytes():  # driver-loop: over the index's bytes
        bytes.append(b)
    _put(root + "/index.txt", bytes)


def _spec(root: String, seed: UInt64) -> BatchSourceSpec:
    var spec = BatchSourceSpec()
    spec.path = root
    spec.batch = 2
    spec.inputs = [BS_FIELD_STEPS, BS_FIELD_LENGTHS]
    spec.shuffle = True
    spec.seed = seed
    spec.bucket = 2
    return spec^


def _plan(mut src: BatchSource, epoch: Int) -> List[Int]:
    var out = List[Int]()
    for b in range(src.n_batches):  # driver-loop: over the epoch's batches
        out.extend(src.records(epoch, b))
        out.append(-1)
    return out^


# ---------------------------------------------------------------- unit 0: the plan
def run_plan_unit(root: String) raises:
    var a = batch_source_open(_spec(root, 5))
    var b = batch_source_open(_spec(root, 5))
    var c = batch_source_open(_spec(root, 6))
    require(a.n == 7 and a.n_batches == 4 and a.max_len == 4 and a.width == _F, "the fixture's counts")
    var moved = False
    for e in range(3):  # driver-loop: over three epochs
        var pa = _plan(a, e)
        var pb = _plan(b, e)
        if pa != pb:
            _fail("the same seed gave two plans for epoch " + String(e))
        if pa != _plan(c, e):
            moved = True
        var seen = List[Int](length=7, fill=0)
        for i in range(len(pa)):  # driver-loop: over the plan's words
            if pa[i] >= 0:
                seen[pa[i]] += 1
        for r in range(7):  # driver-loop: over records
            if seen[r] != 1:
                _fail("epoch " + String(e) + " trains record " + String(r) + " " + String(seen[r]) + " times")
    if not moved:
        _fail("seed 6 drew seed 5's plan in every epoch (the plan ignores the seed)")
    a.close()
    b.close()
    c.close()


# ---------------------------------------------------------------- unit 1: the resume
def _request(lr: Float32, decay: Float32) -> OpsOptimizerRequest:
    return OpsOptimizerRequest(OPT_ADAMW, False, 0, 1, lr, Float32(0), Float32(0.9), Float32(0.999), Float32(1e-8),
                               decay, Float32(0.1), Float32(0))


def _extras() -> OpsOptimizerExtras:
    return OpsOptimizerExtras(Float32(0.99), False, Float32(0), Float32(0), Float32(0), Float32(0))


def _words[E: Exec & Movable & Deinitable](mut s: OpsSession[E], h: Int, n: Int) raises -> List[Float32]:
    var out = List[Float32](length=n, fill=0)
    s.download(h, f32_ptr(Int(out.unsafe_ptr())))
    return out^


def _same(name: String, a: List[Float32], b: List[Float32]) raises:
    if len(a) != len(b):
        _fail(name + ": lengths differ")
    for i in range(len(a)):  # driver-loop: over compared words
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            _fail(name + ": word " + String(i) + " differs (" + String(a[i]) + " vs " + String(b[i]) + ")")


def _model[E: Exec & Movable & Deinitable](mut c: OpsSession[E]) raises -> Tuple[Int, Int, Int, Int]:
    """(program, optimizer, W, v): steps -> reshape (-1, 2) -> dropout -> matmul W, plus lengths matmul v."""
    var w_v: List[Float32] = [0.25, -0.375]
    var v_v: List[Float32] = [0.125]
    var w = tensor(c, w_v, [_F, 1])
    var v = tensor(c, v_v, [1, 1])
    var xs = tensor(c, List[Float32](length=2 * 4 * _F, fill=0), [2, 4, _F])
    var xl = tensor(c, List[Float32](length=2, fill=0), [2, 1])
    var flat = c.op(OPS_RESHAPE_N, [xs], [0, -1, _F], List[Float32]())
    var df: List[Float32] = [Float32(0.5)]
    var d = c.op(OPS_DROPOUT, [flat], [11, 0, 0], df)
    var y = c.op(OPS_SUM_N, [c.op(OPS_MATMUL, [d, w], [], List[Float32]())], [0], List[Float32]())
    var z = c.op(OPS_SUM_N, [c.op(OPS_MATMUL, [xl, v], [], List[Float32]())], [0], List[Float32]())
    var loss = c.op(OPS_ADD, [y, z], [], List[Float32]())
    var ins: List[Int] = [xs, xl]
    var outs: List[Int] = [loss]
    var pid = c.capture(ins, outs)
    var reqs: List[OpsOptimizerRequest] = [_request(Float32(0.05), Float32(0.01)), _request(Float32(0.02), Float32(0))]
    var exts: List[OpsOptimizerExtras] = [_extras(), _extras()]
    var groups = List[List[Int]]()
    groups.append([w])
    groups.append([v])
    var opt = c.param_group_optimizer(_request(Float32(0.05), Float32(0.01)), _extras(), reqs, exts, groups)
    return (pid, opt, w, v)


def _step[E: Exec & Movable & Deinitable](mut s: OpsSession[E], pid: Int, mut src: BatchSource, opt: Int,
                                         params: List[Int]) raises -> Float32:
    """One streamed step (Program.fit(source=, steps=1)); returns its loss."""
    var curve = List[Float32](length=10, fill=0)
    _ = train_program_source(s, pid, src, 0, Float32(1.0), opt, params, 10, 1,
                             FP(unsafe_from_address=Int(curve.unsafe_ptr())), 0, 0)
    return curve[0]


def _load_refused[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, file: String, expect: ProgramTrainingSource, needle: String,
) raises -> Bool:
    """Is a load of `file` with the source words `expect` refused with a message holding `needle`?"""
    try:
        _ = program_training_load[E, KIND](sid, file, expect)
    except error:
        return needle in String(error)
    return False


def run_resume_unit[E: Exec & Movable & Deinitable, KIND: Int](root: String, var first: E, var second: E) raises:
    """`first` runs the uninterrupted side, `second` the fresh one."""
    var file = root + "/run.ckpt"
    var previous = root + "/run.ckpt.prev"
    # THE SEAL (lane fable-finalize). A twin directory holds other step values under the same record lengths:
    # unsealed, the two fingerprints are equal (the defect: a position applied to other data); sealed, they differ.
    var twin = root + "-twin"
    _ = external_call["mkdir", c_int](twin.as_c_string_slice().unsafe_ptr(), c_int(448))
    _write_shards(twin, 3)
    var plain = batch_source_open(_spec(root, 5))
    var plain_twin = batch_source_open(_spec(twin, 5))
    if plain.sealed or plain_twin.sealed or plain.fingerprint != plain_twin.fingerprint:
        _fail("the twin fixture must be unsealed and share the unsealed fingerprint (same record lengths)")
    plain.close()
    plain_twin.close()
    if batch_source_seal(root) != 2 or batch_source_seal(twin) != 2:
        _fail("sealing the fixture must seal two shards")
    var sealed_twin = batch_source_open(_spec(twin, 5))
    var twin_fingerprint = sealed_twin.fingerprint
    if not sealed_twin.sealed:
        _fail("a sealed directory must open as sealed")
    sealed_twin.close()
    var later_epoch = -1
    var later_consumed = -1
    var sid = ops_session_open[E, KIND](first^, False)
    var straight_ids = List[List[Int]]()
    var straight = List[List[Float32]]()
    var saved = ProgramTrainingSource()
    try:
        var s = ops_session[E, KIND](sid)
        var m = _model(s[])
        var params: List[Int] = [m[2], m[3]]
        var src = batch_source_open(_spec(root, 5))
        var curve = List[Float32](length=10, fill=0)
        _ = train_program_source(s[], m[0], src, 0, Float32(1.0), m[1], params, 10, 3,
                                 FP(unsafe_from_address=Int(curve.unsafe_ptr())), 0, 0)
        _ = curve^  # the fit wrote the curve through its address
        if src.consumed != 3 or src.held_batch != 3:
            _fail("after three steps: consumed " + String(src.consumed) + ", read ahead " + String(src.held_batch)
                  + " (want 3 and 3)")
        var at = batch_source_saved[E.IS_DEVICE](src)
        saved.present = True
        saved.epoch = at[0]
        saved.consumed = at[1]
        saved.fingerprint = src.fingerprint
        saved.seed = 5
        saved.batch = 2
        saved.bucket = 2
        saved.shuffle = True
        saved.inputs = [BS_FIELD_STEPS, BS_FIELD_LENGTHS]
        saved.target = 0
        if not src.sealed or src.fingerprint == twin_fingerprint:
            _fail("sealed shards with other content must have another fingerprint")
        _ = program_training_save[E, KIND](sid, m[0], m[1], file, saved)
        # THE REPLACEMENT (lane fable-finalize). A hard link on the first checkpoint: the second save below must
        # replace `file` (a new file renamed over the name) and never write into the first checkpoint's file, so
        # the link keeps the first checkpoint whole whatever happens to the second save.
        if external_call["link", c_int](file.as_c_string_slice().unsafe_ptr(),
                                        previous.as_c_string_slice().unsafe_ptr()) != 0:
            _fail("cannot hard-link the first checkpoint")
        for k in range(2):  # driver-loop: over two uninterrupted steps
            var next = src.pending()
            straight_ids.append(src.records(next[0], next[1]))
            var row: List[Float32] = [_step(s[], m[0], src, m[1], params)]
            row.extend(_words(s[], m[2], _F))
            row.extend(_words(s[], m[3], 1))
            straight.append(row^)
        var later_at = batch_source_saved[E.IS_DEVICE](src)
        var later = saved.copy()
        later.epoch = later_at[0]
        later.consumed = later_at[1]
        later_epoch = later_at[0]
        later_consumed = later_at[1]
        if later_epoch == saved.epoch and later_consumed == saved.consumed:
            _fail("two more steps must move the consumed position")
        _ = program_training_save[E, KIND](sid, m[0], m[1], file, later)
        src.close()
    except error:
        ops_session_close[E, KIND](sid)
        raise error
    ops_session_close[E, KIND](sid)
    # the fresh side: a new session and a source opened from the saved words alone
    var fresh = ops_session_open[E, KIND](second^, False)
    try:
        # lane fable-review-2: a load refused for another source's words changes nothing in the session
        var wrong = ProgramTrainingSource()
        wrong.present = True
        wrong.fingerprint = 1
        var before = ops_session[E, KIND](fresh)
        var opts_before = len(before[].opts)
        var refused = _load_refused[E, KIND](fresh, previous, wrong, "not the data the checkpoint was saved with")
        # same-shaped replacement data (the sealed twin): refused by name
        var replaced = saved.copy()
        replaced.fingerprint = twin_fingerprint
        if not _load_refused[E, KIND](fresh, previous, replaced, "not the data the checkpoint was saved with"):
            _fail("a resume on same-shaped replacement data must be refused by name")
        # a changed field mapping: other input fields, then another target
        var remapped = saved.copy()
        remapped.inputs = [BS_FIELD_STEPS, BS_FIELD_MASK]
        if not _load_refused[E, KIND](fresh, previous, remapped, "are not the fields the checkpoint was trained on"):
            _fail("a resume with other input fields must be refused by name")
        var retargeted = saved.copy()
        retargeted.target = BS_FIELD_TARGETS
        if not _load_refused[E, KIND](fresh, previous, retargeted, "is not the checkpoint's"):
            _fail("a resume with another target field must be refused by name")
        var replanned = saved.copy()
        replanned.seed = 6
        if not _load_refused[E, KIND](fresh, previous, replanned, "so its batches would differ"):
            _fail("a resume with another seed must be refused by name")
        if (not refused or len(before[].opts) != opts_before or len(before[].opt_states) != 0
                or len(before[].pg_tables) != 0):
            _fail("a load refused for the wrong source must leave the session unchanged")
        # the first checkpoint, through the hard link taken before the second save replaced `file`
        var loaded = program_training_load[E, KIND](fresh, previous, saved)
        if loaded.source.epoch != saved.epoch or loaded.source.consumed != saved.consumed:
            _fail("the second save wrote into the first checkpoint's file (position (" + String(loaded.source.epoch)
                  + ", " + String(loaded.source.consumed) + ")): a save must replace the file, never overwrite it")
        var spec = _spec(root, loaded.source.seed)
        spec.epoch = loaded.source.epoch
        spec.consumed = loaded.source.consumed
        spec.fingerprint = loaded.source.fingerprint
        var src = batch_source_open(spec^)
        var s = ops_session[E, KIND](fresh)
        for k in range(2):  # driver-loop: over two resumed steps
            var next = src.pending()
            if src.records(next[0], next[1]) != straight_ids[k]:
                _fail("resumed step " + String(k) + " trains other records than the uninterrupted run")
            var row: List[Float32] = [_step(s[], loaded.program, src, loaded.optimizer, loaded.params)]
            row.extend(_words(s[], loaded.params[0], _F))
            row.extend(_words(s[], loaded.params[1], 1))
            _same("step " + String(k) + " loss and parameters, uninterrupted vs resumed", straight[k], row)
        src.close()
        # the name itself holds the second checkpoint
        var second_load = program_training_load[E, KIND](fresh, file, ProgramTrainingSource())
        if second_load.source.epoch != later_epoch or second_load.source.consumed != later_consumed:
            _fail("the checkpoint's name does not hold the second save")
    except error:
        ops_session_close[E, KIND](fresh)
        raise error
    ops_session_close[E, KIND](fresh)


# ---------------------------------------------------------------- unit 2: one opening, one plan an epoch
def run_steps_unit[E: Exec & Movable & Deinitable, KIND: Int](root: String, var ex: E) raises:
    """100 one-step fits on one opened source: no reopen, each epoch planned once, the next batch read ahead."""
    var sid = ops_session_open[E, KIND](ex^, False)
    try:
        var s = ops_session[E, KIND](sid)
        var m = _model(s[])
        var params: List[Int] = [m[2], m[3]]
        var src = batch_source_open(_spec(root, 5))
        var fds = src.steps_fd.copy()
        var plans = 0
        var order_at = Int(src.order.unsafe_ptr())
        var plan_epoch = src.plan_epoch
        for k in range(_STEP_CALLS):  # driver-loop: over the one-step fit calls
            _ = _step(s[], m[0], src, m[1], params)
            var now_at = Int(src.order.unsafe_ptr())
            if now_at != order_at or src.plan_epoch != plan_epoch:
                plans += 1
                if src.plan_epoch != plan_epoch + 1:
                    _fail("call " + String(k) + " planned epoch " + String(src.plan_epoch) + " after epoch "
                          + String(plan_epoch) + " (an epoch planned twice or skipped)")
                order_at = now_at
                plan_epoch = src.plan_epoch
            var done = k + 1
            var next = src.pending()
            if next[0] != done // src.n_batches or next[1] != done % src.n_batches:
                _fail("after " + String(done) + " one-step calls the next batch is (" + String(next[0]) + ", "
                      + String(next[1]) + ")")
            if src.held_epoch != next[0] or src.held_batch != next[1]:
                _fail("after call " + String(k) + " the batch the next call trains is not read ahead")
            for f in range(len(fds)):  # driver-loop: over shard descriptors
                if src.steps_fd[f] != fds[f]:
                    _fail("call " + String(k) + " reopened shard " + String(f))
        if plans != plan_epoch + 1:
            _fail(String(plans) + " plans for epochs 0 to " + String(plan_epoch) + " (want one an epoch)")
        src.close()
    except error:
        ops_session_close[E, KIND](sid)
        raise error
    ops_session_close[E, KIND](sid)


def test_batch_source() raises:
    comptime if CPU_COLUMN_BUILD:
        def host_unit(unit: Int) raises:
            var root = _dir(unit)
            _write_shards(root)
            if unit == 0:
                run_plan_unit(root)
            elif unit == 1:
                run_resume_unit[HostExec, OPS_HOST](root, HostExec(), HostExec())
            else:
                run_steps_unit[HostExec, OPS_HOST](root, HostExec())
        check_cases("batch_source", BATCH_SOURCE_UNITS, host_unit)
    else:
        def device_unit(unit: Int) raises:
            var root = _dir(unit)
            _write_shards(root)
            if unit == 0:
                run_plan_unit(root)
            elif unit == 1:
                run_resume_unit[DeviceExec, OPS_DEVICE](root, DeviceExec(), DeviceExec())
            else:
                run_steps_unit[DeviceExec, OPS_DEVICE](root, DeviceExec())
        check_cases_serial("batch_source", BATCH_SOURCE_UNITS, device_unit)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
