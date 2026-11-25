# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE ONE-CALL TRAINING CHECKPOINT of a Program (lane batch-source, Andrew 2026-11-25): `Program.save_training`
writes, in one file, everything the next `Program.fit` step reads, and `Session.load_training` restores it in a
fresh session (a fresh process included: nothing is kept outside the file).

What it holds: the captured graph (autograd/program.mojo `program_words`, the words `Program.save` and the control
checkpoint already write), the parameter table (each unique parameter's shape, dtype and values), the optimizer
(its kind, schedule, schedule epoch and clips; with parameter groups every group's settings and members, so the
groups are rebuilt and no handle is remapped by hand), every parameter's optimizer state (step count, the
bias-correction products and the moment buffers), and, when a batch source fed the fit, the source's CONSUMED
position, its fingerprint and its plan settings (autograd/batch_source.mojo THE POSITION). The RNG progress is the
optimizer's step counts: a dropout's key and a scan's key under Program.fit are drawn from the step count
(autograd/program_runtime.mojo `step`), and the source's plan is a pure function of (seed, epoch); so the saved
counts and position reproduce every later mask and batch.

This replaces the manual dance of the composed tagger (tools/identity_lanes/nn_composed.py): `Program.save`, then
`optimizer_state_dict` with every moment read back, then `load_program`, a parameter-handle remap, the groups
rebuilt by hand and `load_optimizer_state_dict`.

Container: training/checkpoint.mojo, section CKPT_SECTION_PROGRAM_TRAINING (its header, payload hash and file hash
are the container's): `param` holds the parameter values, `m_state` / `v_state` the first and second state buffers;
the records below live in the extension's integers and floats, in this order (version 1):

    ints: version, n_params; per parameter its shape (autograd/program.mojo `_put_shape`) and dtype (0 float, 1 ids);
          the graph: n_words_int, n_words_float, n_refs, then the graph's parameter references (table indices), then its
          integer words; the optimizer: grouped (0/1), n_groups, the schedule epoch; per group: kind, nesterov,
          schedule, step_size, centered, n_members, member table indices; per parameter: has_state, step count;
          the source: present (0/1), then epoch, consumed, fingerprint (hi, lo), seed (hi, lo), batch, bucket,
          shuffle, n_inputs, the input field codes in capture order, the target field code (0: none)
    floats: the graph's float words; per group: lr, momentum, beta1, beta2, eps, weight_decay, gamma, dampening,
            alpha, lr_decay, initial_accumulator_value, clip_grad_norm, clip_grad_value; per parameter: pw1, pw2,
            mu_prod; then, for each parameter in table order that has a state under a group with a third state
            buffer (RMSprop centered: the gradient average), that buffer's words (the parameter's element count)

THE SOURCE BINDING (lane fable-finalize, 2026-11-25; version 2). A checkpoint saved with a batch source resumes
only on that source: the same DATA (the fingerprint of a sealed source holds the content digests its shard index
recorded when the shards were sealed, autograd/batch_source.mojo THE SEAL, so same-shaped replacement data is
refused; nothing of the corpus is read at save or at load), the same PLAN (seed, batch, bucket, shuffle) and the
same FIELD MAPPING (the ordered input fields and the target field: a source that would feed "mask" where "lengths"
was trained is refused). `program_training_source_refuse` is the one comparison; the load calls it before the
session changes and the handle's restore calls it again. Version 1 files (counts-only fingerprint, no mapping) are
refused as an unknown version.

An optimizer without groups is one group holding every parameter of the table. The container has slots for two
state buffers; the third (lane fable-review-2: it was refused by name) rides in the extension's floats, so a file
without one is the bytes it was.

THE FILE IS REPLACED WHOLE (lane fable-finalize): the container writes a sibling temporary, fsyncs, closes and
renames it onto the destination, so a save that is interrupted leaves the previous checkpoint loadable
(training/checkpoint.mojo `_save_checkpoint`).

Parallel plan: serial by nature. The file is one ordered write through one descriptor (training/checkpoint.mojo);
the per-parameter downloads are executor copies the device queues, not host work to split.
"""
from std.sys.compile import is_defined
from sequence.exec_trait import Exec
from sequence.ops import FP, OPT_RMSPROP
from autograd.ops_api import ops_session, OpsSession
from autograd.optim import optimizer_slots
from autograd.optimizer_requests import OpsOptimizerRequest, OpsOptimizerExtras
from autograd.program import (
    OpsProgram, _Reader, _put_shape, program_words, program_from_words, same_shape, PROGRAM_PARAM_WEIGHT,
    PROGRAM_PARAM_STATE_WORDS,
)
from autograd.rankn import ShapeN
from autograd.batch_source import batch_source_field_name, BS_FIELD_STEPS, BS_FIELD_TARGETS
from core.cast_in import CastIn, CAST_KIND_FLOAT
from training.checkpoint import Checkpoint, CKPT_SECTION_PROGRAM_TRAINING, read_checkpoint_unchecked, save_checkpoint

# Sabotage arms (tools/identity_lanes/coverage_sabotage.tsv coverage-autograd-program_training_*; lane
# ops-batch-source-resume). Off in every build; an arm's patch switches one on. The plain name faults the device
# executor only, _HOST the host executor only.
#   DROP_GROUP: a resume restores the optimizer state of group 0 only: every later group's moments and step counts
#   start over, so the first resumed update differs from the uninterrupted one.
comptime PT_SABOTAGE_DROP_GROUP = is_defined["MOJOLEARN_PT_SABOTAGE_DROP_GROUP"]()
comptime PT_SABOTAGE_DROP_GROUP_HOST = is_defined["MOJOLEARN_PT_SABOTAGE_DROP_GROUP_HOST"]()
#   IN_PLACE (lane fable-finalize): Program.save_training opens and overwrites the destination directly, the form
#   before 2026-11-25, instead of replacing it whole: a hard link taken on the previous checkpoint then shows the NEW
#   bytes (one file, rewritten) where it must keep the previous checkpoint (the old file, never touched).
comptime PT_SABOTAGE_IN_PLACE = is_defined["MOJOLEARN_PT_SABOTAGE_IN_PLACE"]()
comptime PT_SABOTAGE_IN_PLACE_HOST = is_defined["MOJOLEARN_PT_SABOTAGE_IN_PLACE_HOST"]()
#   SOURCE_UNBOUND (lane fable-finalize): Session.load_training(source=) does not compare the source with the saved
#   one, so a resume on same-shaped replacement data or with other input fields is accepted where it must be refused.
comptime PT_SABOTAGE_SOURCE_UNBOUND = is_defined["MOJOLEARN_PT_SABOTAGE_SOURCE_UNBOUND"]()
comptime PT_SABOTAGE_SOURCE_UNBOUND_HOST = is_defined["MOJOLEARN_PT_SABOTAGE_SOURCE_UNBOUND_HOST"]()

comptime _PT_VERSION = 2
comptime _PT_SOURCE_FIELDS_MAX = 64
comptime _PT_GROUP_FLOATS = 13


struct ProgramTrainingSource(Copyable, Movable):
    """The batch source part of a training checkpoint (present False: none)."""

    var present: Bool
    var epoch: Int
    var consumed: Int
    var fingerprint: UInt64
    var seed: UInt64
    var batch: Int
    var bucket: Int
    var shuffle: Bool
    var inputs: List[Int]
    var target: Int

    def __init__(out self):
        self.inputs = List[Int]()
        self.target = 0
        self.present = False
        self.epoch = 0
        self.consumed = 0
        self.fingerprint = 0
        self.seed = 0
        self.batch = 0
        self.bucket = 0
        self.shuffle = False


def _field_names(codes: List[Int]) -> String:
    var out = String("(")
    for k in range(len(codes)):  # small-loop(codes: a source's input fields)
        if k > 0:
            out += ", "
        out += batch_source_field_name(codes[k])
    return out + ")"


def _same_fields(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for k in range(len(a)):  # small-loop(codes: a source's input fields)
        if a[k] != b[k]:
            return False
    return True


def program_training_source_refuse(
    expect: ProgramTrainingSource, saved: ProgramTrainingSource, who: String,
) raises:
    """Refuse, by name, a batch source (`expect`: the handle's words) that is not the one a checkpoint or a saved
    position (`saved`) was taken with: other data, another plan, or another field mapping (module note THE SOURCE
    BINDING). The one comparison of every resume door."""
    if expect.fingerprint != saved.fingerprint:
        raise Error("ValueError: " + who + ": the source's shards are not the data the checkpoint was saved with "
                    + "(their content digests or record lengths differ); resume on the shards that were trained")
    if (expect.seed != saved.seed or expect.batch != saved.batch or expect.bucket != saved.bucket
            or expect.shuffle != saved.shuffle):
        raise Error("ValueError: " + who + ": the source is not the one the checkpoint was saved with (another "
                    + "seed, batch_size, bucket_batches or shuffle), so its batches would differ")
    if not _same_fields(expect.inputs, saved.inputs):
        raise Error("ValueError: " + who + ": the source's inputs " + _field_names(expect.inputs)
                    + " are not the fields the checkpoint was trained on " + _field_names(saved.inputs))
    if expect.target != saved.target:
        var mine = batch_source_field_name(expect.target) if expect.target != 0 else String("none")
        var theirs = batch_source_field_name(saved.target) if saved.target != 0 else String("none")
        raise Error("ValueError: " + who + ": the source's target (" + mine + ") is not the checkpoint's ("
                    + theirs + ")")


def _hi(v: UInt64) -> Int64:
    return Int64(Int(v >> 32))


def _lo(v: UInt64) -> Int64:
    return Int64(Int(v & UInt64(0xFFFFFFFF)))


def _word(hi: Int, lo: Int) raises -> UInt64:
    if hi < 0 or lo < 0 or hi > 0xFFFFFFFF or lo > 0xFFFFFFFF:
        raise Error("ValueError: malformed training checkpoint: a 64-bit word is out of range")
    return (UInt64(hi) << 32) | UInt64(lo)


def _put_group[E: Exec & Movable & Deinitable](mut ck: Checkpoint, s: OpsSession[E], oi: Int, members: List[Int]):
    var m = s.opts[oi]
    var c = m.cfg
    ck.ext_ints.append(Int64(c.kind))
    ck.ext_ints.append(Int64(1) if c.nesterov else Int64(0))
    ck.ext_ints.append(Int64(m.sched))
    ck.ext_ints.append(Int64(m.sched_step))
    ck.ext_ints.append(Int64(1) if c.centered else Int64(0))
    ck.ext_ints.append(Int64(len(members)))
    for k in range(len(members)):  # driver-loop: over one group's member parameters (table indices)
        ck.ext_ints.append(Int64(members[k]))
    var fs: List[Float32] = [c.lr, c.momentum, c.beta1, c.beta2, c.eps, c.weight_decay, m.gamma, c.dampening,
                             c.alpha, c.lr_decay, c.init_acc, c.clip if c.clip_global else Float32(0), c.clip_value]
    ck.ext_floats.extend(fs^)


def program_training_save[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, pid: Int, optimizer: Int, file: String, source: ProgramTrainingSource,
) raises -> Int:
    """Write program `pid`'s training checkpoint (module note) to `file`; returns the byte count."""
    var s = ops_session[E, KIND](sid)
    var p = s[].program(pid)
    var oi = s[]._opt_index(optimizer)
    var grouped = s[].opts[oi].pg >= 0
    if source.present and (len(source.inputs) < 1 or len(source.inputs) > _PT_SOURCE_FIELDS_MAX):
        raise Error("ValueError: Program.save_training: the source part names no input field (or too many)")
    # the parameter table: the graph's parameters, each once, in graph order
    var handles = List[Int]()
    var shapes = List[ShapeN]()
    var kinds = List[Int]()
    var index = Dict[Int, Int]()
    for k in range(len(p.params)):  # driver-loop: over the graph's parameter references
        var h = p.params[k]
        var slot = s[]._slot(h)
        if s[].owned[slot] == 0 or not same_shape(s[].shapes[slot], p.param_shapes[k]):
            raise Error("ValueError: Program.save_training: a parameter is not an unchanged session tensor")
        if h not in index:
            index[h] = len(handles)
            handles.append(h)
            shapes.append(s[].shapes[slot])
            kinds.append(Int(s[].int_words[slot]))
    var ck = Checkpoint()
    ck.section_kind = CKPT_SECTION_PROGRAM_TRAINING
    ck.offsets.append(0)
    var n = 0
    ck.ext_ints.append(Int64(_PT_VERSION))
    ck.ext_ints.append(Int64(len(handles)))
    for k in range(len(handles)):  # driver-loop: over the parameter table
        n += shapes[k].count()
        ck.offsets.append(n)
        ck.names.append("p" + String(k))
        ck.buf_initialized.append(False)
        _put_shape(ck.ext_ints, shapes[k])
        ck.ext_ints.append(Int64(kinds[k]))
    if len(handles) == 0:
        raise Error("ValueError: Program.save_training: the program has no parameter")
    ck.param = List[Float32](length=n, fill=Float32(0))
    ck.m_state = List[Float32](length=n, fill=Float32(0))
    ck.v_state = List[Float32](length=n, fill=Float32(0))
    for k in range(len(handles)):  # driver-loop: over parameter downloads
        s[].download(handles[k], FP(unsafe_from_address=Int(ck.param.unsafe_ptr())) + ck.offsets[k])
    var words = program_words(p)
    ck.ext_ints.append(Int64(len(words[0])))
    ck.ext_ints.append(Int64(len(words[1])))
    ck.ext_ints.append(Int64(len(p.params)))
    for k in range(len(p.params)):  # driver-loop: over the graph's parameter references
        ck.ext_ints.append(Int64(index[p.params[k]]))
    for k in range(len(words[0])):  # driver-loop: over serialized graph integer words
        ck.ext_ints.append(words[0][k])
    for k in range(len(words[1])):  # driver-loop: over serialized graph float words
        ck.ext_floats.append(words[1][k])
    # the optimizer: its groups (one group of every table parameter without groups), the schedule epoch
    var member_of = List[Int](length=len(handles), fill=oi)
    ck.ext_ints.append(Int64(1) if grouped else Int64(0))
    if grouped:
        var o = oi + 1
        var groups = s[].pg_groups(o)
        ck.ext_ints.append(Int64(groups))
        ck.ext_ints.append(Int64(s[].opts[oi].epoch))
        for g in range(groups):  # driver-loop: over parameter groups
            var members = List[Int]()
            var hs = s[].pg_group_params(o, g)
            for i in range(len(hs)):  # driver-loop: over one group's parameters
                if hs[i] not in index:
                    raise Error("ValueError: Program.save_training: parameter group " + String(g)
                                + " holds a tensor that is not a parameter of this program")
                members.append(index[hs[i]])
                member_of[index[hs[i]]] = s[].pg_tables[s[].opts[oi].pg].members[g]
            _put_group(ck, s[], s[].pg_tables[s[].opts[oi].pg].members[g], members)
    else:
        ck.ext_ints.append(Int64(1))
        ck.ext_ints.append(Int64(s[].opts[oi].epoch))
        var members = List[Int]()
        for k in range(len(handles)):  # driver-loop: over the parameter table (the one group)
            members.append(k)
        _put_group(ck, s[], oi, members)
    var state_of = List[Int](length=len(handles), fill=-1)
    for k in range(len(handles)):  # driver-loop: over every parameter's optimizer state
        var m = member_of[k]
        var slots = optimizer_slots(s[].opts[m].cfg)
        var si = -1
        for j in range(len(s[].opt_states)):  # driver-loop: over retained optimizer states
            if s[].opt_states[j].opt == m and s[].opt_states[j].param == handles[k]:
                si = j
        if si < 0:
            ck.ext_ints.append(Int64(0))
            ck.ext_ints.append(Int64(0))
            ck.ext_floats.append(Float32(1))
            ck.ext_floats.append(Float32(1))
            ck.ext_floats.append(Float32(1))
            continue
        var state = s[].opt_states[si]
        state_of[k] = si
        ck.buf_initialized[k] = True
        ck.t = max(ck.t, state.t)
        ck.ext_ints.append(Int64(1))
        ck.ext_ints.append(Int64(state.t))
        ck.ext_floats.append(state.pw1)
        ck.ext_floats.append(state.pw2)
        ck.ext_floats.append(state.mu_prod)
        if slots[0]:
            s[].ex.download(FP(unsafe_from_address=Int(ck.m_state.unsafe_ptr())) + ck.offsets[k], state.s1, state.n)
        if slots[1]:
            s[].ex.download(FP(unsafe_from_address=Int(ck.v_state.unsafe_ptr())) + ck.offsets[k], state.s2, state.n)
    for k in range(len(handles)):  # driver-loop: over the third state buffers (RMSprop centered), after the scalars
        if state_of[k] < 0 or not optimizer_slots(s[].opts[member_of[k]].cfg)[2]:
            continue
        var third = s[].opt_states[state_of[k]]
        var at = len(ck.ext_floats)
        ck.ext_floats.resize(at + third.n, Float32(0))
        s[].ex.download(FP(unsafe_from_address=Int(ck.ext_floats.unsafe_ptr())) + at, third.s3, third.n)
    ck.ext_ints.append(Int64(1) if source.present else Int64(0))
    if source.present:
        ck.ext_ints.append(Int64(source.epoch))
        ck.ext_ints.append(Int64(source.consumed))
        ck.ext_ints.append(_hi(source.fingerprint))
        ck.ext_ints.append(_lo(source.fingerprint))
        ck.ext_ints.append(_hi(source.seed))
        ck.ext_ints.append(_lo(source.seed))
        ck.ext_ints.append(Int64(source.batch))
        ck.ext_ints.append(Int64(source.bucket))
        ck.ext_ints.append(Int64(1) if source.shuffle else Int64(0))
        ck.ext_ints.append(Int64(len(source.inputs)))
        for k in range(len(source.inputs)):  # small-loop(inputs: the source's input fields, in capture order)
            ck.ext_ints.append(Int64(source.inputs[k]))
        ck.ext_ints.append(Int64(source.target))
    s[].ex.sync()
    # the file is replaced whole (training/checkpoint.mojo): an interrupted save leaves the previous checkpoint
    comptime if (PT_SABOTAGE_IN_PLACE and E.IS_DEVICE) or (PT_SABOTAGE_IN_PLACE_HOST and not E.IS_DEVICE):
        var direct = save_checkpoint(file, ck, True)  # SABOTAGE: the previous checkpoint is overwritten in place
        return direct.bytes
    else:
        var written = save_checkpoint(file, ck)
        return written.bytes


struct ProgramTrainingLoad(Movable):
    """program_training_load's answer: the new program id, its parameter handles in graph order, the new optimizer
    id and the saved source part."""

    var program: Int
    var params: List[Int]
    var optimizer: Int
    var source: ProgramTrainingSource

    def __init__(out self):
        self.program = 0
        self.params = List[Int]()
        self.optimizer = 0
        self.source = ProgramTrainingSource()


struct _Group(Copyable, Movable):
    var ints: List[Int]
    var floats: List[Float32]
    var members: List[Int]

    def __init__(out self):
        self.ints = List[Int]()
        self.floats = List[Float32]()
        self.members = List[Int]()


def _request(g: _Group) -> OpsOptimizerRequest:
    return OpsOptimizerRequest(g.ints[0], g.ints[1] != 0, g.ints[2], g.ints[3], g.floats[0], g.floats[1],
                               g.floats[2], g.floats[3], g.floats[4], g.floats[5], g.floats[6], g.floats[7])


def _extras(g: _Group) -> OpsOptimizerExtras:
    return OpsOptimizerExtras(g.floats[8], g.ints[4] != 0, g.floats[9], g.floats[10], g.floats[11], g.floats[12])


def program_training_load[E: Exec & Movable & Deinitable, KIND: Int](
    sid: Int, file: String, expect: ProgramTrainingSource,
) raises -> ProgramTrainingLoad:
    """Read a training checkpoint (module note) into inference session `sid`: new parameter tensors, the program,
    the optimizer with its groups, schedule epoch and every state. Every record is checked before the session
    changes; a failure after that removes what was added. `expect` (present: the caller resumes with a batch
    source) is that source's data fingerprint, plan settings and field mapping: they are compared with the saved ones here, before
    the session changes, so a load refused for the wrong source leaves no program, tensor or optimizer behind
    (lane fable-review-2: the door compared them after the load and freed the program and tensors only, so the
    refused load's optimizer, its group members and its group table stayed in the session)."""
    var ck = read_checkpoint_unchecked(file)
    var what = String("ValueError: malformed training checkpoint")
    if ck.section_kind != CKPT_SECTION_PROGRAM_TRAINING:
        raise Error("ValueError: not a Program training checkpoint (Program.save_training writes one)")
    var r = _Reader(len(ck.ext_ints))
    var float_at = 0
    if r.take(ck.ext_ints, what) != _PT_VERSION:
        raise Error(what + ": unknown version")
    var count = r.take(ck.ext_ints, what)
    if count < 1 or ck.j_count() != count:
        raise Error(what + ": wrong parameter table length")
    var shapes = List[ShapeN]()
    var kinds = List[Int]()
    for k in range(count):  # driver-loop: over parameter metadata
        var sh = r.shape(ck.ext_ints, what)
        var dtype = r.take(ck.ext_ints, what)
        if (dtype != 0 and dtype != 1) or ck.names[k] != "p" + String(k) or ck.offsets[k + 1] - ck.offsets[k] != sh.count():
            raise Error(what + ": invalid parameter shape, kind or name")
        shapes.append(sh)
        kinds.append(dtype)
    var ni = r.take(ck.ext_ints, what)
    var nf = r.take(ck.ext_ints, what)
    if ni < 1 or nf < 0 or nf > len(ck.ext_floats):
        raise Error(what + ": invalid graph lengths")
    var n_refs = r.take(ck.ext_ints, what)
    if n_refs < count or n_refs > len(ck.ext_ints) - r.at or ni > len(ck.ext_ints) - r.at - n_refs:
        raise Error(what + ": invalid or truncated graph")
    var refs = List[Int]()
    var xi = List[Int64]()
    var xf = List[Float32]()
    for k in range(n_refs):  # driver-loop: over the graph's parameter references
        var idx = r.take(ck.ext_ints, what)
        if idx < 0 or idx >= count:
            raise Error(what + ": a parameter reference is out of range")
        refs.append(idx)
    for k in range(ni):  # driver-loop: over serialized graph integer words
        xi.append(Int64(r.take(ck.ext_ints, what)))
    for k in range(nf):  # driver-loop: over serialized graph float words
        xf.append(ck.ext_floats[float_at + k])
    float_at += nf
    var p = program_from_words(xi, xf, what)
    if len(p.params) != n_refs:
        raise Error(what + ": the graph's parameter count does not match its references")
    for k in range(n_refs):  # driver-loop: over validated parameter references
        if not same_shape(p.param_shapes[k], shapes[refs[k]]):
            raise Error(what + ": a parameter shape disagrees with the graph")
        p.params[k] = refs[k]
    var grouped = r.take(ck.ext_ints, what)
    var n_groups = r.take(ck.ext_ints, what)
    var epoch = r.take(ck.ext_ints, what)
    if (grouped != 0 and grouped != 1) or n_groups < 1 or (grouped == 0 and n_groups != 1) or epoch < 0:
        raise Error(what + ": invalid optimizer record")
    var groups = List[_Group]()
    var group_of = List[Int](length=count, fill=-1)
    for g in range(n_groups):  # driver-loop: over saved parameter groups
        var grp = _Group()
        for j in range(5):  # small-loop(5: a group's integer settings)
            grp.ints.append(r.take(ck.ext_ints, what))
        var members = r.take(ck.ext_ints, what)
        if members < 0 or members > count:
            raise Error(what + ": invalid group member count")
        for j in range(members):  # driver-loop: over one group's members
            var m = r.take(ck.ext_ints, what)
            if m < 0 or m >= count or group_of[m] >= 0:
                raise Error(what + ": a group member is out of range or in two groups")
            group_of[m] = g
            grp.members.append(m)
        if float_at + _PT_GROUP_FLOATS > len(ck.ext_floats):
            raise Error(what + ": truncated group settings")
        for j in range(_PT_GROUP_FLOATS):  # small-loop(13: a group's float settings)
            grp.floats.append(ck.ext_floats[float_at + j])
        float_at += _PT_GROUP_FLOATS
        groups.append(grp^)
    var has_state = List[Bool]()
    var steps = List[Int]()
    var scalars_at = float_at
    for k in range(count):  # driver-loop: over saved optimizer states
        var flag = r.take(ck.ext_ints, what)
        var step = r.take(ck.ext_ints, what)
        if (flag != 0 and flag != 1) or step < 0 or (flag == 0 and step != 0) or (flag == 1 and group_of[k] < 0):
            raise Error(what + ": invalid parameter state record")
        has_state.append(flag == 1)
        steps.append(step)
    if float_at + 3 * count > len(ck.ext_floats):
        raise Error(what + ": truncated optimizer scalars")
    float_at += 3 * count
    # the third state buffers (module note): located and bounded here, before the session changes
    var third_at = List[Int](length=count, fill=-1)
    for k in range(count):  # driver-loop: over saved optimizer states (which carry a third buffer)
        if not has_state[k]:
            continue
        if groups[group_of[k]].ints[0] != OPT_RMSPROP or groups[group_of[k]].ints[4] == 0:
            continue
        if shapes[k].count() > len(ck.ext_floats) - float_at:
            raise Error(what + ": truncated third optimizer state buffer")
        third_at[k] = float_at
        float_at += shapes[k].count()
    var out = ProgramTrainingLoad()
    var present = r.take(ck.ext_ints, what)
    if present != 0 and present != 1:
        raise Error(what + ": invalid source flag")
    if present == 1:
        out.source.present = True
        out.source.epoch = r.take(ck.ext_ints, what)
        out.source.consumed = r.take(ck.ext_ints, what)
        var fh = r.take(ck.ext_ints, what)
        out.source.fingerprint = _word(fh, r.take(ck.ext_ints, what))
        var sh = r.take(ck.ext_ints, what)
        out.source.seed = _word(sh, r.take(ck.ext_ints, what))
        out.source.batch = r.take(ck.ext_ints, what)
        out.source.bucket = r.take(ck.ext_ints, what)
        var shuffle = r.take(ck.ext_ints, what)
        out.source.shuffle = shuffle == 1
        if (out.source.epoch < 0 or out.source.consumed < 0 or out.source.batch < 1 or out.source.bucket < 1
                or (shuffle != 0 and shuffle != 1)):
            raise Error(what + ": invalid source position")
        var n_inputs = r.take(ck.ext_ints, what)
        if n_inputs < 1 or n_inputs > _PT_SOURCE_FIELDS_MAX:
            raise Error(what + ": invalid source field count")
        for k in range(n_inputs):  # small-loop(inputs: the saved input fields)
            var code = r.take(ck.ext_ints, what)
            if code < BS_FIELD_STEPS or code > BS_FIELD_TARGETS:
                raise Error(what + ": invalid source input field")
            out.source.inputs.append(code)
        out.source.target = r.take(ck.ext_ints, what)
        if out.source.target != 0 and out.source.target != BS_FIELD_TARGETS:
            raise Error(what + ": invalid source target field")
    if r.at != len(ck.ext_ints) or float_at != len(ck.ext_floats):
        raise Error(what + ": trailing records")
    if expect.present:
        if not out.source.present:
            raise Error("ValueError: Session.load_training(source=): the checkpoint was saved without a batch source")
        # (the sabotage arms SOURCE_UNBOUND skip this comparison, one side each)
        comptime if not ((PT_SABOTAGE_SOURCE_UNBOUND and E.IS_DEVICE)
                         or (PT_SABOTAGE_SOURCE_UNBOUND_HOST and not E.IS_DEVICE)):
            program_training_source_refuse(expect, out.source, "Session.load_training(source=)")
    var s = ops_session[E, KIND](sid)
    if s[].recording():
        raise Error("ValueError: Session.load_training loads into an inference session (record=False)")
    var handles = List[Int]()
    var pid = 0
    var added_opts = len(s[].opts)
    var added_tables = len(s[].pg_tables)
    try:
        for k in range(count):  # driver-loop: over restored parameter buffers
            var sh = shapes[k]
            # the words as they were saved (int32 words keep their bit patterns), as ops_load_program uploads them
            var source = CastIn(Int(ck.param.unsafe_ptr()) + 4 * ck.offsets[k], CAST_KIND_FLOAT,
                                4, sh.rows(), sh.cols(), sh.cols() * 4, 4)
            handles.append(s[].upload_nd(source, sh, False))
        for k in range(len(p.params)):  # driver-loop: over the graph's handle rebinding (and its parameter kinds)
            var h = handles[p.params[k]]
            var pk = p.param_kind(k)
            if pk == PROGRAM_PARAM_STATE_WORDS or kinds[p.params[k]] == 1:
                s[].int_words[s[]._slot(h)] = True
            if pk != PROGRAM_PARAM_WEIGHT:
                s[].declare_state(h)
            p.params[k] = h
        pid = s[].add_program(p.copy())
        var opt: Int
        if grouped == 1:
            var reqs = List[OpsOptimizerRequest]()
            var exts = List[OpsOptimizerExtras]()
            var members = List[List[Int]]()
            for g in range(n_groups):  # driver-loop: over rebuilt parameter groups
                reqs.append(_request(groups[g]))
                exts.append(_extras(groups[g]))
                var hs = List[Int]()
                for j in range(len(groups[g].members)):  # driver-loop: over one group's members
                    hs.append(handles[groups[g].members[j]])
                members.append(hs^)
            opt = s[].param_group_optimizer(reqs[0], exts[0], reqs, exts, members)
            s[].pg_set_epoch(opt, epoch)
        else:
            opt = s[].optimizer_with(_request(groups[0]), _extras(groups[0]))
            s[].opts[opt - 1].epoch = epoch
        var oi = opt - 1
        for k in range(count):  # driver-loop: over restored optimizer states
            if not has_state[k]:
                continue
            comptime if (PT_SABOTAGE_DROP_GROUP and E.IS_DEVICE) or (PT_SABOTAGE_DROP_GROUP_HOST and not E.IS_DEVICE):
                if group_of[k] > 0:
                    continue  # SABOTAGE: only group 0's optimizer state is restored
            var m = s[]._pg_member(oi, handles[k], "Session.load_training") if grouped == 1 else oi
            s[].wants_grad[s[]._slot(handles[k])] = True
            var si = s[]._state_of(m, handles[k], shapes[k].count())
            var state = s[].opt_states[si]
            var slots = optimizer_slots(s[].opts[m].cfg)
            if slots[2] != (third_at[k] >= 0):
                raise Error(what + ": a third optimizer state buffer disagrees with its group's settings")
            if slots[2]:
                s[].ex.upload(state.s3, FP(unsafe_from_address=Int(ck.ext_floats.unsafe_ptr())) + third_at[k], state.n)
            if slots[0]:
                s[].ex.upload(state.s1, FP(unsafe_from_address=Int(ck.m_state.unsafe_ptr())) + ck.offsets[k], state.n)
            if slots[1]:
                s[].ex.upload(state.s2, FP(unsafe_from_address=Int(ck.v_state.unsafe_ptr())) + ck.offsets[k], state.n)
            s[].opt_states[si].t = steps[k]
            s[].opt_states[si].pw1 = ck.ext_floats[scalars_at + 3 * k]
            s[].opt_states[si].pw2 = ck.ext_floats[scalars_at + 3 * k + 1]
            s[].opt_states[si].mu_prod = ck.ext_floats[scalars_at + 3 * k + 2]
        s[].ex.sync()
        _ = ck^  # the uploads have consumed the payload
        out.program = pid
        out.optimizer = opt
        for k in range(len(p.params)):  # driver-loop: over the program's parameter handles
            out.params.append(p.params[k])
    except error:
        while len(s[].opts) > added_opts:  # driver-loop: over optimizers this load added
            _ = s[].opts.pop()
        while len(s[].pg_tables) > added_tables:  # driver-loop: over group tables this load added
            _ = s[].pg_tables.pop()
        if pid > 0:
            s[].program_free(pid)
        for k in range(len(handles)):  # driver-loop: over partial parameter uploads
            s[].free(handles[k])
        raise error
    return out^
