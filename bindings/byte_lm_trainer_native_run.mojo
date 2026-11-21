# SPDX-License-Identifier: Apache-2.0
"""RUN of the byte-LM trainer door: the trainer's native state owner and its
operations. Typed values only; no Python object is read, built or called here.

`ByteLmTrainerState` is the whole fitted state of one `LanguageModelTrainer`:
the model shape words, the parameters, both optimizer moments and the flags
(native buffers), the step counter, the optimizer scalars, the data schedule
(its canonical JSON text) and the session lifecycle (the resident session,
open / lost, the last export step, the gradient step, the stateless route's
kept gradient). It lives in the process registry `BYTE_LM_TRAINER_STATES`; the
trainer object holds only `ByteLmTrainerOwner`, the two handle words (index,
generation). The boundary door (`bindings/byte_lm_trainer_native_door.mojo`)
acquires the lease ONCE per public operation (one generation check), unpacks
the call's buffers into addresses, calls the functions below and packs the
public result. A step performs no handle lookup, lock or state copy: it reads
the leased state's fields and makes one entry call.

One trainer operation is one call of the image's compiled entry with three
typed argument lists: buffer addresses, the integer plan words and the
optimizer scalars (the `byte_lm_trainer_api_v1` numbering,
training/byte_lm_native_dispatch.mojo). `ByteLmPlan` is that request.

Not split across cores: one call is one entry call into the session, which
runs its own device (or the host column's own workers); the state transitions
around it are a handful of scalar writes.
"""
from std.memory import ArcPointer, unsafe_memcpy
from core.entry_table import EntryRef
from core.state_registry import StateRegistry, StateHandle
from core.byte_lm_entry_table import ByteLmEntryKey
from core.byte_lm_entry_table_sessions import byte_lm_resident_run, byte_lm_sessions_published
from training.byte_lm_session_owner_run import ByteLmSessionOwner
from training.byte_lm_glue_abi import byte_lm_trainer_call

# The plan's integer layout: ten shape words (nine dimensions, chunked head),
# then train, completed steps, optimizer kind, nesterov, batch (the step
# count of operation 1), length, release.
comptime BYTE_PLAN_SHAPE_WORDS = 10
# lr, beta1, beta2, eps, weight_decay, momentum, dampening, max_norm.
comptime BYTE_PLAN_SCALARS = 8
comptime BYTE_MAX_STEP = 999999

comptime _F32 = MutPointer[Float32, MutUntrackedOrigin]


struct ByteLmPlan(Movable):
    """One operation's integer words and optimizer scalars."""
    var ints: List[Int]
    var floats: List[Float64]

    def __init__(out self, shape_words: List[Int], train: Bool, completed: Int, kind: Int, nesterov: Int,
                 batch: Int, length: Int, release: Bool, var scalars: List[Float64]):
        var ints = List[Int](capacity=BYTE_PLAN_SHAPE_WORDS + 7)
        for i in range(BYTE_PLAN_SHAPE_WORDS):  # driver-loop: over the admitted model shape words
            ints.append(shape_words[i])
        ints.append(Int(train))
        ints.append(completed)
        ints.append(kind)
        ints.append(nesterov)
        ints.append(batch)
        ints.append(length)
        ints.append(Int(release))
        self.ints = ints^
        self.floats = scalars^


struct ByteLmStateData(Movable):
    """The transferable training state: what `export_state()` publishes and a
    restore replaces whole. `parameters`, `m` and `v` are empty while a
    resident session is open (the device holds the only copy)."""
    var shape: List[Int]
    var n_total: Int
    var n_tensors: Int
    var has_model_shape: Bool
    var profile: String
    var mode: Int
    var parameters: List[Float32]
    var m: List[Float32]
    var v: List[Float32]
    var flags: List[Int32]
    var completed: Int
    var scalars: List[Float64]
    var kind: Int
    var nesterov: Int
    var schedule: String

    def __init__(out self, var shape: List[Int], n_total: Int, n_tensors: Int, has_model_shape: Bool,
                 var profile: String, mode: Int, zero_moments: Bool):
        """Buffers sized for the shape. `parameters` is unfilled: the caller
        writes all `n_total` floats before the state is used. The moments are
        zero-filled for a new trainer and unfilled for a restore (which writes
        them in full)."""
        self.shape = shape^
        self.n_total = n_total
        self.n_tensors = n_tensors
        self.has_model_shape = has_model_shape
        self.profile = profile^
        self.mode = mode
        self.parameters = List[Float32](unsafe_uninit_length=n_total)
        if zero_moments:
            self.m = List[Float32](length=n_total, fill=0.0)
            self.v = List[Float32](length=n_total, fill=0.0)
        else:
            self.m = List[Float32](unsafe_uninit_length=n_total)
            self.v = List[Float32](unsafe_uninit_length=n_total)
        self.flags = List[Int32](length=n_tensors, fill=0)
        self.completed = 0
        self.scalars = List[Float64](length=BYTE_PLAN_SCALARS, fill=0.0)
        self.kind = 2
        self.nesterov = 0
        self.schedule = String("")


struct ByteLmTrainerState(Movable):
    """One trainer's state and session lifecycle. Mutated only under the
    trainer's own lock, by one caller at a time."""
    var data: ByteLmStateData
    # The outputs of the step in flight (the next flags; the stateless
    # route's next parameters and moments), installed by its commit.
    var next_flags: List[Int32]
    var next_p: List[Float32]
    var next_m: List[Float32]
    var next_v: List[Float32]
    # The stateless route's last lean-step gradient, kept for
    # export_gradients(): the flat vector and the per-tensor copy.
    var grad_flat: List[Float32]
    var grad_named: List[Float32]
    var next_grad_flat: List[Float32]
    var next_grad_named: List[Float32]
    var grad_present: Bool
    var grad_valid: Bool
    # One loss word for a single step or evaluation.
    var loss: List[Float32]
    # The resident request words, refilled in place by every step.
    var words: List[Int]
    # -1 automatic (resolved from the admitted image), 0 stateless, 1 resident.
    var resident: Int
    var lean: Bool
    # The admitted image's table key; -1 before the first admission.
    var key_mode: Int
    var key_vendor: Int
    var session: Optional[ArcPointer[ByteLmSessionOwner]]
    var session_open: Bool
    var lost: Bool
    var lost_step: Int
    var lost_export: Int
    var grad_step: Int
    var last_export_step: Int

    def __init__(out self, var data: ByteLmStateData, resident: Int, lean: Bool):
        self.last_export_step = data.completed
        self.data = data^
        self.next_flags = List[Int32]()
        self.next_p = List[Float32]()
        self.next_m = List[Float32]()
        self.next_v = List[Float32]()
        self.grad_flat = List[Float32]()
        self.grad_named = List[Float32]()
        self.next_grad_flat = List[Float32]()
        self.next_grad_named = List[Float32]()
        self.grad_present = False
        self.grad_valid = False
        self.loss = List[Float32](length=1, fill=0.0)
        self.words = List[Int]()
        self.resident = resident
        self.lean = lean
        self.key_mode = -1
        self.key_vendor = -1
        self.session = Optional[ArcPointer[ByteLmSessionOwner]]()
        self.session_open = False
        self.lost = False
        self.lost_step = 0
        self.lost_export = 0
        self.grad_step = -1

    def key(self) raises -> ByteLmEntryKey:
        """The table key the door admitted for this trainer."""
        if self.key_mode < 0:
            raise Error("RuntimeError: Byte-LM native binding is not admitted")
        return ByteLmEntryKey(self.key_mode, self.key_vendor)

    def plan(self, train: Bool, batch: Int, length: Int, release: Bool) -> ByteLmPlan:
        return ByteLmPlan(self.data.shape, train, self.data.completed, self.data.kind, self.data.nesterov,
                          batch, length, release, self.data.scalars.copy())

    def has_session(self) -> Bool:
        return Bool(self.session)


comptime BYTE_LM_TRAINER_STATES = StateRegistry[ByteLmTrainerState, "MojolearnByteLmTrainerStatesV1"]


struct ByteLmTrainerOwner(Movable, Writable):
    """What the trainer object holds: the registry handle of its state.
    Dropping it (the trainer's destruction) releases the state, and with it
    the resident session."""
    var index: Int
    var generation: UInt64

    def __init__(out self, var state: ByteLmTrainerState) raises:
        var handle = BYTE_LM_TRAINER_STATES.insert(state^)
        self.index = handle.index
        self.generation = handle.generation

    def handle(self) -> StateHandle:
        return StateHandle(self.index, self.generation)

    def __deinit__(deinit self):
        try:
            BYTE_LM_TRAINER_STATES.release(StateHandle(self.index, self.generation))
        except:
            pass  # Quiet finalization: a released handle has nothing to free.

    def write_to(self, mut writer: Some[Writer]):
        writer.write("ByteLmTrainerOwner(handle=", self.index, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def byte_trainer_acquire(handle: StateHandle) raises -> ArcPointer[ByteLmTrainerState]:
    """The one generation check of a public operation."""
    return BYTE_LM_TRAINER_STATES.acquire(handle)


def byte_trainer_state_ptr(lease: ArcPointer[ByteLmTrainerState]) -> MutPointer[ByteLmTrainerState, MutUntrackedOrigin]:
    """The leased state, mutable; valid while the caller names `lease`."""
    return MutPointer[ByteLmTrainerState, MutUntrackedOrigin](unsafe_from_address=Int(UnsafePointer(to=lease[])))


# ---------------------------------------------------------------- the session lifecycle

def _session_run(mut s: ByteLmTrainerState, operation: Int, addresses: List[Int], ints: List[Int],
                 floats: List[Float64]) raises -> Int:
    """One trainer operation on the resident session. `addresses` are the
    descriptors after the session word. The request words are refilled in
    place, so a step allocates none after the first."""
    var lease = s.session.value().copy()
    var entry = lease[].entry  # resolved once, at the session's construction
    var handle = lease[].handle()
    var result = byte_lm_resident_run(entry, handle, operation, addresses, ints, floats, s.words)
    _ = lease  # the session outlives the synchronous call
    return result


def _lost_refusal(s: ByteLmTrainerState) -> String:
    return ("RuntimeError: Byte-LM session lost at step " + String(s.lost_step) + "; last exported state is step "
            + String(s.lost_export))


def byte_trainer_require_live(s: ByteLmTrainerState) raises:
    if s.lost:
        raise Error(_lost_refusal(s))


def byte_trainer_is_resident(mut s: ByteLmTrainerState) raises -> Bool:
    """Automatic (`resident=None`): resident when the admitted image
    publishes its session entry; resolved once."""
    if s.resident < 0:
        s.resident = 1 if byte_lm_sessions_published(s.key()) else 0
    return s.resident == 1


def byte_trainer_session_ensure(mut s: ByteLmTrainerState) raises:
    """The resident session of the admitted table key, created on first use
    (a handle only: no device work until the first call admits the state)."""
    if not s.session:
        s.session = Optional[ArcPointer[ByteLmSessionOwner]](ArcPointer[ByteLmSessionOwner](ByteLmSessionOwner(s.key())))
        s.session_open = False
        s.grad_step = -1


def byte_trainer_forget(mut s: ByteLmTrainerState):
    """Drop the session (its finalizer releases the handle)."""
    s.session = Optional[ArcPointer[ByteLmSessionOwner]]()
    s.session_open = False
    s.grad_step = -1


def byte_trainer_mark_lost(mut s: ByteLmTrainerState):
    if not s.lost:
        s.lost = True
        s.lost_step = s.data.completed
        s.lost_export = s.last_export_step


def byte_trainer_adopt_open(mut s: ByteLmTrainerState):
    """The device holds the only copy now: the host buffers are freed."""
    if not s.session_open:
        s.session_open = True
        s.grad_step = -1
        s.grad_valid = False
        s.last_export_step = s.data.completed
        s.data.parameters = List[Float32]()
        s.data.m = List[Float32]()
        s.data.v = List[Float32]()


def byte_trainer_after_failure(mut s: ByteLmTrainerState):
    """The native call's verdict read back after a failed operation."""
    s.grad_step = -1
    if not s.session or s.lost:
        return
    var is_open = s.session_open
    var usable = False
    var completed = -1
    try:
        var info = List[Int](length=4, fill=0)
        var a: List[Int] = [Int(info.unsafe_ptr())]
        _ = _session_run(s, 5, a, List[Int](), List[Float64]())
        is_open = info[3] != 0
        usable = info[2] != 0
        completed = info[0]
        _ = info^
    except:
        is_open = s.session_open
        usable = False
        completed = -1
    if is_open:
        byte_trainer_adopt_open(s)
    if s.session_open and (not usable or completed != s.data.completed):
        byte_trainer_mark_lost(s)


def _commit_step(mut s: ByteLmTrainerState, completed: Int):
    swap(s.data.flags, s.next_flags)
    s.next_flags = List[Int32]()
    s.data.completed = completed


def _steps_reconcile(mut s: ByteLmTrainerState, completed: Int, n_steps: Int):
    """After a failed multi-step call: the steps the session reports as
    committed (it rolled the failed step back) are committed here too, with
    the flags the native call wrote after the last of them; then the usual
    failure verdict."""
    try:
        var info = List[Int](length=4, fill=0)
        var a: List[Int] = [Int(info.unsafe_ptr())]
        _ = _session_run(s, 5, a, List[Int](), List[Float64]())
        var done = info[0]
        if info[3] != 0 and info[2] != 0 and done > completed and done <= completed + n_steps:
            _commit_step(s, done)
        _ = info^
    except:
        pass
    byte_trainer_after_failure(s)


def byte_trainer_loss_address(mut s: ByteLmTrainerState) -> Int:
    return Int(s.loss.unsafe_ptr())


def byte_trainer_flags_address(s: ByteLmTrainerState) -> Int:
    return Int(s.data.flags.unsafe_ptr())


# ---------------------------------------------------------------- the resident operations

def byte_trainer_step(mut s: ByteLmTrainerState, tokens: Int, n_steps: Int, train: Bool, loss: Int,
                      flat: Int, named: List[Int]) raises -> Int:
    """One native call: `n_steps` training steps (`tokens` the (n_steps,
    batch, length + 1) stack) or one evaluation. `loss` receives one word per
    step; `flat` and `named` (0 / empty for a lean step) receive the last
    step's gradient. Returns the committed step counter."""
    byte_trainer_require_live(s)
    if not s.session and not byte_lm_sessions_published(s.key()):
        raise Error("ImportError: Byte-LM binding lacks owned sessions (its native session entry); rebuild bindings/build_byte_lm.sh")
    var completed = s.data.completed
    if train and completed + n_steps > BYTE_MAX_STEP:
        raise Error("ValueError: Byte-LM native call step counter is exhausted")
    byte_trainer_session_ensure(s)
    if train:
        s.next_flags = List[Int32](length=s.data.n_tensors, fill=0)
    var a = List[Int](capacity=8 + len(named))
    a.append(tokens)
    a.append(Int(s.data.flags.unsafe_ptr()))
    if s.session_open:
        a.append(0)
        a.append(0)
        a.append(0)
    else:
        a.append(Int(s.data.parameters.unsafe_ptr()))
        a.append(Int(s.data.m.unsafe_ptr()))
        a.append(Int(s.data.v.unsafe_ptr()))
    a.append(loss)
    a.append(Int(s.next_flags.unsafe_ptr()) if train else 0)
    a.append(flat)
    for i in range(len(named)):  # driver-loop: one address per model parameter tensor
        a.append(named[i])
    s.grad_step = -1
    var returned = 0
    try:
        # batch slot = the call's step count (operation 1)
        var plan = s.plan(train, n_steps, 0, False)
        returned = _session_run(s, 1, a, plan.ints, plan.floats)
    except error:
        if n_steps > 1 and train:
            # a column that commits step by step (the host column) may
            # stand past `completed`; the device column commits none
            _steps_reconcile(s, completed, n_steps)
        else:
            byte_trainer_after_failure(s)
        raise error^
    byte_trainer_adopt_open(s)
    var expected = completed + (n_steps if train else 0)
    if returned != expected:
        byte_trainer_mark_lost(s)
        raise Error("RuntimeError: Byte-LM returned an invalid completed-step counter")
    if train:
        _commit_step(s, expected)
        s.grad_step = expected
    return expected


def byte_trainer_export(mut s: ByteLmTrainerState, out_p: Int, out_m: Int, out_v: Int, out_flags: Int,
                        release: Bool) raises:
    """The open session's state downloaded into the four buffers (operation
    3); `release` also releases the device state."""
    var completed = s.data.completed
    var a: List[Int] = [Int(s.data.flags.unsafe_ptr()), out_p, out_m, out_v, out_flags]
    var plan = s.plan(False, 0, 0, release)
    var returned = _session_run(s, 3, a, plan.ints, plan.floats)
    if returned != completed:
        raise Error("RuntimeError: Byte-LM state export returned a different completed step")
    s.last_export_step = completed


def byte_trainer_export_release(mut s: ByteLmTrainerState) raises:
    """Suspend's export: the device state comes back as the host state. The
    three float buffers are unfilled: the export writes all of each before it
    returns, and a failed export raises before they are installed."""
    var n = s.data.n_total
    var p = List[Float32](unsafe_uninit_length=n)
    var m = List[Float32](unsafe_uninit_length=n)
    var v = List[Float32](unsafe_uninit_length=n)
    var flags = List[Int32](length=s.data.n_tensors, fill=0)
    byte_trainer_export(s, Int(p.unsafe_ptr()), Int(m.unsafe_ptr()), Int(v.unsafe_ptr()),
                        Int(flags.unsafe_ptr()), True)
    s.data.parameters = p^
    s.data.m = m^
    s.data.v = v^
    s.data.flags = flags^
    s.grad_valid = False


def byte_trainer_session_close(mut s: ByteLmTrainerState) raises:
    """Close the session without export (operation 4); the caller then
    forgets it."""
    if s.session:
        _ = _session_run(s, 4, List[Int](), List[Int](), List[Float64]())


def byte_trainer_logits(mut s: ByteLmTrainerState, tokens: Int, output: Int, row_lengths: Int, direct_next: Bool,
                        batch: Int, length: Int) raises -> Int:
    """The resident forward (operation 2), or its greedy pick on the device
    (operation 6; `row_lengths` 0 for a full batch). Returns the count of
    values written."""
    byte_trainer_require_live(s)
    byte_trainer_session_ensure(s)
    var a = List[Int](capacity=7)
    a.append(tokens)
    a.append(output)
    a.append(Int(s.data.flags.unsafe_ptr()))
    if s.session_open:
        a.append(0)
        a.append(0)
        a.append(0)
    else:
        a.append(Int(s.data.parameters.unsafe_ptr()))
        a.append(Int(s.data.m.unsafe_ptr()))
        a.append(Int(s.data.v.unsafe_ptr()))
    if row_lengths != 0:
        a.append(row_lengths)
    var written = 0
    try:
        var plan = s.plan(False, batch, length, False)
        written = _session_run(s, 6 if direct_next else 2, a, plan.ints, plan.floats)
    except problem:
        byte_trainer_after_failure(s)
        raise problem^
    byte_trainer_adopt_open(s)
    return written


def byte_trainer_logits_stateless(s: ByteLmTrainerState, trainer: EntryRef, tokens: Int, output: Int, batch: Int,
                                  length: Int) raises -> Int:
    """The stateless forward (operation 7) through the image's table entry."""
    var a: List[Int] = [Int(s.data.parameters.unsafe_ptr()), tokens, output]
    var plan = s.plan(False, batch, length, False)
    return byte_lm_trainer_call(trainer, 7, a, plan.ints, plan.floats)


def byte_trainer_gradients_check(s: ByteLmTrainerState) raises:
    """export_gradients() refusals, before any output is allocated."""
    if s.lost:
        raise Error("RuntimeError: Byte-LM session lost; restore state before exporting gradients")
    if not s.session_open:
        if not s.grad_present:
            raise Error("RuntimeError: Byte-LM export_gradients requires an open resident session; the stateless train_step returns its gradients")
        if not s.grad_valid or s.grad_step != s.data.completed:
            raise Error("RuntimeError: Byte-LM has no gradient to export: no step has completed since open, restore, import or rollback")
    elif s.grad_step != s.data.completed:
        raise Error("RuntimeError: Byte-LM has no gradient to export: no step has completed since open, restore, import or rollback")


def byte_trainer_export_gradients(mut s: ByteLmTrainerState, flat: Int, named: List[Int],
                                  sizes: List[Int]) raises:
    """The last completed step's gradient into `flat` and, when given, one
    buffer per tensor (`sizes` their element counts): from the open session
    (operation 10), or the copy a lean stateless step kept."""
    byte_trainer_gradients_check(s)
    if not s.session_open:
        unsafe_memcpy(dest=_F32(unsafe_from_address=flat), src=_F32(unsafe_from_address=Int(s.grad_flat.unsafe_ptr())),
                      count=s.data.n_total)
        var base = Int(s.grad_named.unsafe_ptr())
        var offset = 0
        for i in range(len(named)):  # driver-loop: one copy per model parameter tensor
            unsafe_memcpy(dest=_F32(unsafe_from_address=named[i]), src=_F32(unsafe_from_address=base + offset * 4),
                          count=sizes[i])
            offset += sizes[i]
        return
    var completed = s.data.completed
    var a = List[Int](capacity=1 + len(named))
    a.append(flat)
    for i in range(len(named)):  # driver-loop: named parameter gradient descriptors
        a.append(named[i])
    var plan = s.plan(False, 0, 0, False)
    var returned = _session_run(s, 10, a, plan.ints, plan.floats)
    if returned != completed:
        raise Error("RuntimeError: Byte-LM gradient export returned a different step")


def byte_trainer_stage_info(mut s: ByteLmTrainerState) raises -> List[Int]:
    """The open session's attention stage words (operation 11); empty when
    no session is open."""
    var info = List[Int]()
    if not s.session_open or not s.session:
        return info^
    var values = List[Int](length=17 + 5 * s.data.shape[7], fill=0)
    var a: List[Int] = [Int(values.unsafe_ptr())]
    var p: List[Int] = [len(values)]
    var size = _session_run(s, 11, a, p, List[Float64]())
    for i in range(size):  # driver-loop: over completed native stage metadata
        info.append(values[i])
    _ = values^
    return info^


# ---------------------------------------------------------------- the stateless step
#
# Operation 8 runs on a call-scoped native session the door creates and frees
# around it (and finalizes with operation 9 on the locked-teardown build), so
# the step is three RUN pieces: the descriptors, the entry call, the commit.

def byte_trainer_stateless_begin(mut s: ByteLmTrainerState, tokens: Int, train: Bool, loss: Int, flat: Int,
                                 named: List[Int], sizes: List[Int]) raises -> List[Int]:
    """The stateless step's descriptors: slot 0 (the session, filled by the
    caller), the eleven state / output buffers, then the named gradient
    buffers. A lean training step (`flat` 0 with `sizes` given) writes its
    gradient into the state's own buffers, kept for export_gradients()."""
    if train and s.data.completed >= BYTE_MAX_STEP:
        raise Error("ValueError: Byte-LM native call step counter is exhausted")
    var n = s.data.n_total
    s.next_p = List[Float32](length=n, fill=0.0)
    s.next_m = List[Float32](length=n, fill=0.0)
    s.next_v = List[Float32](length=n, fill=0.0)
    s.next_flags = List[Int32](length=s.data.n_tensors, fill=0)
    var kept = train and flat == 0
    if kept:
        s.next_grad_flat = List[Float32](length=n, fill=0.0)
        s.next_grad_named = List[Float32](length=n, fill=0.0)
    var d = List[Int](capacity=12 + len(named) + len(sizes))
    d.append(0)
    d.append(Int(s.data.parameters.unsafe_ptr()))
    d.append(Int(s.data.m.unsafe_ptr()))
    d.append(Int(s.data.v.unsafe_ptr()))
    d.append(Int(s.data.flags.unsafe_ptr()))
    d.append(tokens)
    d.append(Int(s.next_p.unsafe_ptr()))
    d.append(Int(s.next_m.unsafe_ptr()))
    d.append(Int(s.next_v.unsafe_ptr()))
    d.append(Int(s.next_grad_flat.unsafe_ptr()) if kept else flat)
    d.append(Int(s.next_flags.unsafe_ptr()))
    d.append(loss)
    if kept:
        var base = Int(s.next_grad_named.unsafe_ptr())
        var offset = 0
        for i in range(len(sizes)):  # driver-loop: one address per model parameter tensor
            d.append(base + offset * 4)
            offset += sizes[i]
    else:
        for i in range(len(named)):  # driver-loop: one address per model parameter tensor
            d.append(named[i])
    return d^


def byte_trainer_stateless_call(trainer: EntryRef, operation: Int, descriptors: List[Int], ints: List[Int],
                                floats: List[Float64]) raises -> Int:
    """One trainer operation through the image's table entry: the stateless
    step (operation 8, its call-scoped session first among `descriptors`)
    and its locked finalization (9)."""
    return byte_lm_trainer_call(trainer, operation, descriptors, ints, floats)


def byte_trainer_stateless_commit(mut s: ByteLmTrainerState, returned: Int, train: Bool, kept: Bool) raises -> Int:
    """Install the step's outputs as the state. Returns the step counter."""
    var expected = s.data.completed + Int(train)
    if returned != expected:
        byte_trainer_stateless_discard(s)
        raise Error("RuntimeError: Byte-LM returned an invalid completed-step counter")
    if not train:
        byte_trainer_stateless_discard(s)
        return expected
    swap(s.data.parameters, s.next_p)
    swap(s.data.m, s.next_m)
    swap(s.data.v, s.next_v)
    swap(s.data.flags, s.next_flags)
    s.data.completed = expected
    if kept:
        swap(s.grad_flat, s.next_grad_flat)
        swap(s.grad_named, s.next_grad_named)
        s.next_grad_flat = List[Float32]()
        s.next_grad_named = List[Float32]()
        s.grad_present = True
        s.grad_valid = True
        s.grad_step = expected
    else:
        s.grad_present = False
        s.grad_valid = False
        s.grad_flat = List[Float32]()
        s.grad_named = List[Float32]()
    byte_trainer_stateless_discard(s)
    return expected


def byte_trainer_stateless_discard(mut s: ByteLmTrainerState):
    """Free the outputs of a step that did not commit."""
    s.next_p = List[Float32]()
    s.next_m = List[Float32]()
    s.next_v = List[Float32]()
    s.next_flags = List[Int32]()
    s.next_grad_flat = List[Float32]()
    s.next_grad_named = List[Float32]()


# ---------------------------------------------------------------- replacement

def byte_trainer_replace(mut s: ByteLmTrainerState, var data: ByteLmStateData):
    """`load_state_dict`: the validated replacement becomes the state. The
    caller has closed and forgotten the session."""
    s.last_export_step = data.completed
    s.data = data^
    s.lost = False
    s.grad_valid = False
