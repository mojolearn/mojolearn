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
from std.memory import unsafe_memcpy, ArcPointer
from core.entry_table import EntryRef
from core.prep_entry_table import xp_transport_call
from x_prep.transport_abi import XP_RUN, XP_RUN_SCRATCH, XP_RUN_OUT, XP_RUN_RANGES, XP_DEV_PUT, XP_DEV_FREE, XP_DEV_TAKE_ROWS
from bindings.arena_prog_run import _add_input, _merge_pairs, _pad, _switch
from core.entry_table import entry_ref_unresolved
from core.arena_state import ARENA_CACHES, ARENA_HANDLE_NONE, DeviceCacheState, arena_handle
from bindings.arena_cache_run import (
    ArenaCachePut, arena_cache_upload, arena_cache_drop, arena_scope_cache, arena_scope_cache_found,
)

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


# ---------------------------------------------------------------- the typed run
# Lane ma-pub-kernel round 5 (2026-11-21): the run policy of one classifier
# program (the image's route words, the Int32 bound, which input stays
# resident, goes up directly or is copied, the release of the run's slots
# and the drop of stale resident copies) was written in the door over Python
# objects (`ClsProg.run`). It is typed here so the door (UNPACK: addresses,
# the thread's scope word, the store) and a member (core/naive_bayes_member.mojo:
# borrowed views, the caller's scope word) run the SAME policy. Same route
# choice, same input choices, same executor calls in the same order.
comptime CLS_INT32_MAX = 2147483647
# An input of at least this many words goes up from its own buffer into a
# slot freed after the run (`_expansion_prep._DIRECT_MIN_WORDS`).
comptime CLS_DIRECT_MIN_WORDS = 1048576

#: `cls_input_choice` answers
comptime CLS_INPUT_HOST = 0      # copied into the host arena
comptime CLS_INPUT_RESIDENT = 1  # the scope's cache holds (or takes) its device copy
comptime CLS_INPUT_DIRECT = 2    # put from its own buffer into a slot this run owns


@fieldwise_init
struct ClsRunPlan(ImplicitlyCopyable, Movable):
    """The selected image as one run reads it: its x_prep transport entry and
    its capabilities (`has_ranges` is the `x_prep_run_ranges` entry alone; the
    MOJOLEARN_ARENA_RANGES switch is read by `cls_run_route`). `dev_store`:
    both `x_prep_dev_put` and `x_prep_dev_free`. `take_rows`: those two and
    `x_prep_dev_take_rows`."""
    var key: EntryRef
    var has_out: Bool
    var has_scratch: Bool
    var has_ranges: Bool
    var dev_store: Bool
    var take_rows: Bool


def cls_run_route(plan: ClsRunPlan, program: ClsProgram) raises -> ClsRoute:
    """The run's route words for `program` on the image of `plan`, and the
    refusal of a program past the native Int32 indexing bound."""
    var sc = program.scratch_size
    var has_scratch_entry = sc > 0 and not plan.has_out and plan.has_scratch
    var dev_scratch = sc > 0 and (plan.has_out or has_scratch_entry)
    if program.size + sc > CLS_INT32_MAX:
        raise Error("ValueError: x_prep: the program exceeds the native Int32 indexing bound")
    var ranges = (dev_scratch or sc == 0) and plan.has_ranges and _switch("MOJOLEARN_ARENA_RANGES")
    return ClsRoute(dev_scratch, plan.has_out, ranges)


def cls_host_words(program: ClsProgram, route: ClsRoute) -> Int:
    """The words of the host arena of one run (the scratch region too when
    the image keeps no device scratch)."""
    return program.size + (0 if route.dev_scratch else program.scratch_size)


def _cls_cache_find(lease: ArcPointer[DeviceCacheState], address: Int, words: Int) raises -> Int:
    return lease[].find(address, words, True)


struct ClsResident(Movable):
    """The caller's `resident()` scope for ONE program run: the scope's handle
    word (`ARENA_HANDLE_NONE` outside a scope, or when the image has no device
    store), its size floor and the image's transport entry. The scope's cache
    on that entry is created and leased at the first input the run keeps
    resident (one scope acquire, one lease per run; a hit is one typed lookup)."""
    var scope: Int
    var min_words: Int
    var store: EntryRef
    var cache: Int
    var lease: Optional[ArcPointer[DeviceCacheState]]

    def __init__(out self):
        self.scope = ARENA_HANDLE_NONE
        self.min_words = 0
        self.store = entry_ref_unresolved()
        self.cache = ARENA_HANDLE_NONE
        self.lease = Optional[ArcPointer[DeviceCacheState]]()

    def __init__(out self, scope: Int, min_words: Int, store: EntryRef):
        self.scope = scope
        self.min_words = min_words
        self.store = store
        self.cache = ARENA_HANDLE_NONE
        self.lease = Optional[ArcPointer[DeviceCacheState]]()

    def active(self) -> Bool:
        return self.scope >= 0

    def keeps(self, rows: Bool, words: Int) -> Bool:
        """Whether the run keeps an input of `words` words resident (a
        DeviceRows base at any size)."""
        return self.scope >= 0 and (rows or words >= self.min_words)

    def find(mut self, address: Int, words: Int) raises -> Int:
        """The store id of the resident copy of (`address`, `words`), counted
        as a hit, or -1."""
        if not self.lease:
            self.cache = arena_scope_cache(self.scope, self.store, "x_prep")
            self.lease = Optional[ArcPointer[DeviceCacheState]](ARENA_CACHES.acquire(arena_handle(self.cache)))
        return _cls_cache_find(self.lease.value(), address, words)

    def put(self, address: Int, words: Int, pin: Int) raises -> ArenaCachePut:
        """After a `find` miss: upload the buffer and publish it with the pin
        token `pin` (a reference the caller took for the cache; `pinned`
        False in the answer hands it back). No Python is touched: the door
        releases the GIL around this call."""
        return arena_cache_upload(self.cache, self.store, address, words, pin)


def cls_input_choice(record: ClsInput, route: ClsRoute, plan: ClsRunPlan, resident: ClsResident) -> Int:
    """Where one caller buffer (not a device gather) goes: the scope's cache,
    a direct put, or the host arena. An in-out input and a run without the
    range runner always copy."""
    if not route.ranges or record.in_out:
        return CLS_INPUT_HOST
    if resident.keeps(False, record.words):
        return CLS_INPUT_RESIDENT
    if plan.dev_store and record.words >= CLS_DIRECT_MIN_WORDS:
        return CLS_INPUT_DIRECT
    return CLS_INPUT_HOST


struct ClsRunDone(Movable):
    """What one run hands back to its boundary: the run's own failure
    (`failed`, `error`), the failures of the cleanup after it (`cleanup`,
    empty: none) and the pin tokens of resident copies the run dropped (the
    boundary unpins them last, with the interpreter held)."""
    var failed: Bool
    var error: String
    var cleanup: String
    var pins: List[Int]

    def __init__(out self):
        self.failed = False
        self.error = String("")
        self.cleanup = String("")
        self.pins = List[Int]()

    def keep_cleanup(mut self, text: String):
        if text.byte_length() > 0:
            self.cleanup += ("" if self.cleanup.byte_length() == 0 else "; ") + text


def cls_prog_run(plan: ClsRunPlan, program: ClsProgram, records: List[ClsRunInput], arena: Int, route: ClsRoute,
                 resident: ClsResident, inout_addresses: List[Int], inout_words: List[Int]) -> ClsRunDone:
    """RUN of one program: the inputs and the stages (`cls_prog_execute`),
    then, whether or not it failed, the release of every slot the run
    acquired and the drop of the resident copies of its in-out inputs (a
    stage rewrote their host words; `inout_addresses` / `inout_words` are
    their buffers' identities, empty when the caller has none that could be
    resident). Never raises: the boundary reports from the answer. The
    caller keeps every addressed buffer alive and holds no interpreter lock."""
    var done = ClsRunDone()
    var owned = List[Int](capacity=2 * len(records))
    try:
        cls_prog_execute(plan.key, program, records, arena, route, owned)
    except error:
        done.failed = True
        done.error = String(error)
    done.keep_cleanup(cls_prog_release(plan.key, owned))
    if resident.active() and len(inout_addresses) > 0:
        try:
            var cache = arena_scope_cache_found(resident.scope, resident.store, "x_prep")
            if cache >= 0:
                var released = arena_cache_drop(cache, resident.store, inout_addresses, inout_words)
                for i in range(len(released.pins)):  # driver-loop: over the pins of the dropped in-out copies
                    done.pins.append(released.pins[i])
                done.keep_cleanup(released.failure)
        except drop_error:
            done.keep_cleanup("resident in-out drop: " + String(drop_error))
    return done^


# ---------------------------------------------------------------- typed stage builders
# Lane door-run-split (2026-11-14): the class-count stages took the door's
# Python-holding `ClsProg` twice over (bindings/prep_classify_native_door.mojo
# `_class_counts` and an uncalled copy, bindings/prep_classify_common_door.mojo
# `p2m_class_counts`); one typed builder on `ClsProgram` replaces both. Same
# allocations, same stage words: the bits do not move.
comptime CLS_OP_P2M_CCOUNT = 200
comptime CLS_OP_P2M_CSCAN = 201
# `_p2m_chunks`' minimum rows per chunk (_II_CH).
comptime CLS_II_CH = 1024


def cls_class_counts(mut program: ClsProgram, co: Int, n: Int, K: Int) -> Int:
    """`_p2m_class_rows(pr, co, n, K, rows=False)`: the class counts' (int32
    words) offset."""
    var ch = max(CLS_II_CH, (n + CLS_II_CH - 1) // CLS_II_CH, (n * K + 16777215) // 16777216, 1)
    var nch = max(1, (n + ch - 1) // ch)
    var cnt = program.alloc(nch * K)
    var tot = program.alloc(K)
    var count_words: List[Int] = [co, n, K, ch, cnt]
    program.stage(CLS_OP_P2M_CCOUNT, nch, count_words)
    var scan_words: List[Int] = [cnt, nch, K, tot]
    program.stage(CLS_OP_P2M_CSCAN, K, scan_words)
    return tot
