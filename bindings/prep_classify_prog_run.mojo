# SPDX-License-Identifier: Apache-2.0
"""RUN of the classifier x_prep program (lane entangled-programs, 2026-11-12).

`ClsProgram` is the typed program of the native classifier owner
(bindings/prep_classify_native.mojo): `_expansion_prep._Prog`'s layout rules
(alloc, work, put, put_list, put_codes, put_words, put_ints, stage) on native
words. Constant inputs (`put_list`, `put_scalar`, `put_ints`) are program-owned
native words (were a Python array each); a caller buffer is an input record
whose owner the door (bindings/prep_classify_prog.mojo `ClsProg`) keeps alive.

`cls_prog_execute` is the run the shared arena runner made for this program
(bindings/arena_prog_native.mojo `_prep_run_impl` with native stage words and
no device output), on the image's x_prep transport (`EntryRef`,
x_prep/transport_abi.mojo) instead of the Python `x_prep_run*` /
`x_prep_dev_*` entries: the same executors (x_prep/transport_*.mojo call the
bodies the Python entries called), the same arena layout, input copies,
direct and gathered device inputs, range descriptors and stage words, the same
route choice. The door resolves the Python-held parts (the selected image, its
entry presence words, the resident cache slots, DeviceRows fields) in UNPACK
and releases the GIL around this call. No bit moves.
"""
from std.memory import unsafe_memcpy
from core.entry_table import EntryRef
from core.prep_entry_table import xp_transport_call
from x_prep.transport_abi import XP_RUN, XP_RUN_SCRATCH, XP_RUN_OUT, XP_RUN_RANGES, XP_DEV_PUT, XP_DEV_FREE, XP_DEV_TAKE_ROWS
from bindings.arena_prog_run import _add_input, _merge_pairs, _pad

comptime CLS_F32P = MutPointer[Float32, MutUntrackedOrigin]
# Words of one stage record: op, total and the fourteen parameters.
comptime CLS_STAGE_WORDS = 16
comptime CLS_PARAMS = 14
# `_expansion_prep._NONE`: an absent stage operand.
comptime CLS_NONE = -1

#: input record kinds
comptime CLS_IN_BUFFER = 0   # a door-held buffer (float32 or int32 words), or a DeviceRows
comptime CLS_IN_CONST_F = 1  # program-owned float32 words (`put_list`)
comptime CLS_IN_CONST_I = 2  # program-owned int32 words (`put_ints`)


@fieldwise_init
struct ClsInput(ImplicitlyCopyable, Movable):
    """One input record: its arena offset and word count, its kind, the
    constant's index (constants) and whether a stage rewrites it and the
    caller reads it back (`in_out`)."""
    var offset: Int
    var words: Int
    var kind: Int
    var index: Int
    var in_out: Bool


struct ClsProgram(Movable):
    """One x_prep program: its host words, its device scratch words, the
    input records, the native constants and the stage words (a scratch word
    s is the stage word -(s + 2) until the run)."""
    var size: Int
    var scratch_size: Int
    var work_on: Bool
    var inputs: List[ClsInput]
    var inout: List[Int]
    var stages: List[Int]
    var consts_f: List[List[Float32]]
    var consts_i: List[List[Int32]]

    def __init__(out self, work_on: Bool):
        self.size = 0
        self.scratch_size = 0
        self.work_on = work_on
        self.inputs = List[ClsInput]()
        self.inout = List[Int]()
        self.stages = List[Int]()
        self.consts_f = List[List[Float32]]()
        self.consts_i = List[List[Int32]]()

    def alloc(mut self, n: Int) -> Int:
        var offset = self.size
        self.size += max(n, 0)
        return offset

    def scratch(mut self, n: Int) -> Int:
        var offset = self.scratch_size
        self.scratch_size += max(n, 0)
        return -(offset + 2)

    def work(mut self, n: Int) -> Int:
        """`_Prog.work`: device scratch under the prep-apple3 `work` change,
        else arena words."""
        if not self.work_on:
            return self.alloc(n)
        return self.scratch(n)

    def input(mut self, words: Int, kind: Int, index: Int, in_out: Bool) -> Int:
        """`_Prog.put`'s layout: `words` arena words for one input record."""
        var offset = self.alloc(words)
        self.inputs.append(ClsInput(offset, words, kind, index, in_out))
        if in_out and words > 0:
            self.inout.append(offset)
            self.inout.append(offset + words)
        return offset

    def put_list(mut self, values: List[Float64], in_out: Bool = False) -> Int:
        """`_Prog.put_list`: float32 words of a parameter list ([0.0] when empty)."""
        var words = List[Float32](length=max(len(values), 1), fill=0.0)
        for i in range(len(values)):  # driver-loop: over a parameter list (priors, scalars, model-structure words)
            words[i] = Float32(values[i])
        var count = len(words)
        self.consts_f.append(words^)
        return self.input(count, CLS_IN_CONST_F, len(self.consts_f) - 1, in_out)

    def put_ints(mut self, values: List[Int]) -> Int:
        """`_Prog.put_ints`: int32 words read by `ldi` ([0] when empty)."""
        var words = List[Int32](length=max(len(values), 1), fill=0)
        for i in range(len(values)):  # driver-loop: over a parameter list of int words
            words[i] = Int32(values[i])
        var count = len(words)
        self.consts_i.append(words^)
        return self.input(count, CLS_IN_CONST_I, len(self.consts_i) - 1, False)

    def stage(mut self, op: Int, total: Int, params: List[Int]):
        """`_Prog.stage`: op, total, the parameters, zero padding."""
        self.stages.append(op)
        self.stages.append(total)
        for i in range(len(params)):  # driver-loop: over at most fourteen stage parameters
            self.stages.append(params[i])
        for _ in range(CLS_PARAMS - len(params)):  # driver-loop: over the padding of the fixed fourteen-word stage ABI
            self.stages.append(0)

    def const_address(self, record: ClsInput) -> Int:
        if record.kind == CLS_IN_CONST_F:
            return Int(self.consts_f[record.index].unsafe_ptr())
        return Int(self.consts_i[record.index].unsafe_ptr())

    def stage_words(self) -> List[Int32]:
        """The stage records as the runner's Int32 words; a scratch word s
        sits past the host arena (no device output region)."""
        var host = self.size
        var words = List[Int32](capacity=max(len(self.stages), 1))
        for i in range(len(self.stages) // CLS_STAGE_WORDS):  # driver-loop: over the program's stages
            for k in range(CLS_STAGE_WORDS):  # small-loop(params: the fixed sixteen-word stage record)
                var word = self.stages[i * CLS_STAGE_WORDS + k]
                if k >= 2 and word < -1:
                    word = host - word - 2
                words.append(Int32(word))
        if len(words) == 0:
            words.append(0)  # the runner always receives a valid pointer
        return words^


@fieldwise_init
struct ClsRunInput(ImplicitlyCopyable, Movable):
    """One input as the door resolved it. `source`: a resident slot the
    door's cache holds (-1: none). `direct`: put from its own buffer into a
    slot this run owns. `gather`: a DeviceRows selection taken on the device
    out of its base (the base's resident slot `base_slot`, or put from
    `base_address` / `base_words` into a slot this run owns)."""
    var offset: Int
    var words: Int
    var address: Int
    var source: Int
    var direct: Bool
    var gather: Bool
    var base_slot: Int
    var base_address: Int
    var base_words: Int
    var indices: Int
    var n_indices: Int
    var row_words: Int


@fieldwise_init
struct ClsRoute(ImplicitlyCopyable, Movable):
    """The run's route words, read by the door from the selected image:
    device scratch, the `x_prep_run_out` entry, the range runner."""
    var dev_scratch: Bool
    var has_out: Bool
    var ranges: Bool


def cls_prog_execute(key: EntryRef, program: ClsProgram, inputs: List[ClsRunInput], arena: Int,
                     route: ClsRoute, mut owned: List[Int]) raises:
    """RUN: the inputs into the arena (or onto the device), then one program
    run. Every slot this run acquires is appended to `owned` (the door
    releases them, `cls_prog_release`). The door keeps every addressed
    buffer alive and has released the GIL."""
    var ha = program.size
    var sc = program.scratch_size
    var host_words = ha + (0 if route.dev_scratch else sc)
    var ins = List[Int32]()
    for k in range(len(inputs)):  # driver-loop: over the program's input buffers
        var r = inputs[k]
        if r.words == 0:
            continue
        var src = r.source
        if r.gather:
            var base_slot = r.base_slot
            if base_slot < 0:
                var put_src: List[Int] = [r.base_address]
                base_slot = xp_transport_call(key, XP_DEV_PUT, put_src, [r.base_words])
                owned.append(base_slot)
            var idx: List[Int] = [r.indices]
            src = xp_transport_call(key, XP_DEV_TAKE_ROWS, idx, [base_slot, r.row_words, r.n_indices])
            owned.append(src)
        elif r.direct:
            var put_src: List[Int] = [r.address]
            src = xp_transport_call(key, XP_DEV_PUT, put_src, [r.words])
            owned.append(src)
        if src < 0:
            unsafe_memcpy(dest=CLS_F32P(unsafe_from_address=arena + 4 * r.offset),
                          src=CLS_F32P(unsafe_from_address=r.address), count=r.words)
        _add_input(ins, r.offset, r.offset + r.words, src)
    var stages = program.stage_words()
    var nst = len(program.stages) // CLS_STAGE_WORDS
    var pa = Int(stages.unsafe_ptr())
    if route.ranges:
        # the shared ranges runner (core/arena_io.mojo): the inputs go up,
        # the rest of the host arena starts zero on the device, and
        # everything but the inputs (and the in-out inputs) comes back
        var pairs = List[Int]()
        var at = 0
        for i in range(len(ins) // 3):  # driver-loop: complement of the input ranges over the host arena
            var lo = Int(ins[3 * i])
            if lo > at:
                pairs.append(at)
                pairs.append(lo)
            at = max(at, Int(ins[3 * i + 1]))
        if at < ha:
            pairs.append(at)
            pairs.append(ha)
        for j in range(len(program.inout)):  # driver-loop: the in-out input ranges come back too
            pairs.append(program.inout[j])
        var outs = _merge_pairs(pairs^)
        var n_in = len(ins) // 3
        var n_out = len(outs) // 4
        var ia = _pad(ins.copy(), [Int32(0), Int32(0), Int32(-1)])
        var oa = _pad(outs^, [Int32(0), Int32(0), Int32(-1), Int32(1)])
        var addresses: List[Int] = [arena, pa, 0, Int(ia.unsafe_ptr()), Int(oa.unsafe_ptr())]
        _ = xp_transport_call(key, XP_RUN_RANGES, addresses, [ha, sc if route.dev_scratch else 0, 0, nst, n_in, n_out])
        # The address words are not owners: keep both descriptor lists live.
        _ = ia^
        _ = oa^
    elif route.dev_scratch:
        if route.has_out:
            var addresses: List[Int] = [arena, pa, 0]
            _ = xp_transport_call(key, XP_RUN_OUT, addresses, [ha, sc, 0, nst])
        else:
            var addresses: List[Int] = [arena, pa]
            _ = xp_transport_call(key, XP_RUN_SCRATCH, addresses, [ha, sc, nst])
    else:
        var addresses: List[Int] = [arena, pa]
        _ = xp_transport_call(key, XP_RUN, addresses, [host_words, nst])
    _ = ins^
    _ = stages^


def cls_prog_release(key: EntryRef, owned: List[Int]) -> String:
    """Free every slot one run acquired, newest first, even when one fails;
    the failures as text (empty on success)."""
    var failures = String()
    for i in range(len(owned)):  # driver-loop: over the device slots one program run acquired
        var slot = owned[len(owned) - 1 - i]
        try:
            _ = xp_transport_call(key, XP_DEV_FREE, List[Int](), [slot])
        except error:
            failures += ("" if failures.byte_length() == 0 else "; ") + "slot " + String(slot) + ": " + String(error)
    return failures
