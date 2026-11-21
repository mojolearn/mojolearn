# SPDX-License-Identifier: Apache-2.0
"""Typed cross-vendor replicated training (RUN): the coordinator's lockstep
step, the worker's command handling, the batch feed, the rate table, the
readiness flag, the ordered host fold and the state digests.

No Python object is read here and the GIL is not touched. The door
(`bindings/cross_vendor_native_door.mojo`) unpacks sockets to descriptors
(`transport_unpack`), the trainer to its `ParallelByteSession`, and packs
rows and events. The frames are the ones the boundary transport writes:
`TransportHeader` fields in key order, then the payload, through
`transport_send_words` / `transport_receive_into`.

Parallel plan: one training step is a network round in a fixed order (step
requests, replies in worker order, the fold in logical shard order, the
apply, the witnesses), so the round is serial by nature; the fold's
arithmetic and the session's work run on the device or in the base image's
compiled entry. The state digest is one SHA-256 chain over four buffers in
a fixed order, serial by definition.
"""
from std.ffi import c_int, external_call
from std.math import isfinite
from std.memory import unsafe_memcpy, bitcast
from core.parallel_entry_table import ParallelEntryKey, parallel_ref, parallel_fold_pair_f32, PARALLEL_TABLE_BASE
from core.sha256 import Sha256, sha256_hex_addr, sha256_sliced8_hex, _hex32
from core.transport_header import (
    TransportHeader, transport_header_find, transport_header_has, transport_header_int,
    transport_header_bool, transport_header_words, transport_header_text,
)
from bindings.native_transport_run import transport_clock, transport_send_words, transport_receive_into
from bindings.parallel_byte_native_run import (
    ParallelByteSession, PARALLEL_REPLICA, PARALLEL_SCALAR_LR,
    parallel_session_openable, parallel_session_require_split, parallel_session_set_lr,
    parallel_session_shard_gradient, parallel_session_apply_gradient, parallel_session_fold,
    parallel_session_fold_export, parallel_session_export,
)

comptime CROSS_PROTOCOL = "mojolearn.cross-vendor.v1"
comptime CROSS_SCHEME_V1 = 0
comptime CROSS_SCHEME_SLICED = 1
# What the door does after one worker command.
comptime CROSS_CONTINUE = 0
comptime CROSS_COMMIT = 1
comptime CROSS_DONE = 2
comptime CROSS_HOST_FOLD = 3
# The first thing wrong with one peer's declaration that only UNPACK can see.
comptime CROSS_DEFECT_NONE = 0
comptime CROSS_DEFECT_NAME = 1
comptime CROSS_DEFECT_IDS = 2
comptime CROSS_DEFECT_ID = 3
comptime _CrossBytes = MutPointer[UInt8, MutUntrackedOrigin]


# ------------------------------------------------------------ header helpers
# SHARED CHANGE WANTED: a float list field belongs in core/transport_header.mojo
# (`put_floats`, `transport_header_floats`); until then these two write and
# read the `losses` field with the header's own open/close.
def cross_header_put_floats(mut header: TransportHeader, key: String, values: List[Float64]) raises:
    for i in range(len(values)):  # driver-loop: over one frame's logical shard losses
        if not isfinite(values[i]):
            raise Error("transport: JSON numbers must be finite")
    header._open(key)
    header.values.append(UInt8(91))
    for i in range(len(values)):  # driver-loop: over one frame's logical shard losses
        if i != 0:
            header.values.append(UInt8(44))
        # The shortest decimal that reads back as the same float64.
        var text = String(values[i])
        var view = text.as_bytes()
        for b in range(len(view)):  # small-loop(digits of one number)
            header.values.append(view[b])
    header.values.append(UInt8(93))
    header._close()


def cross_header_floats(address: Int, count: Int, key: String) raises -> List[Float64]:
    var span = transport_header_find(address, count, key)
    var text = _CrossBytes(unsafe_from_address=address)
    if span[0] < 0 or span[1] - span[0] < 2 or Int(text[span[0]]) != 91 or Int(text[span[1] - 1]) != 93:
        raise Error("transport: header field " + key + " is not a list of numbers")
    var values = List[Float64]()
    var token = List[UInt8](length=64, fill=UInt8(0))
    var stop = List[Int](length=1, fill=0)
    var i = span[0] + 1
    var end = span[1] - 1
    while i < end:  # driver-loop: over one frame's logical shard losses
        var b = Int(text[i])
        if b == 44 or b == 32 or b == 9 or b == 10 or b == 13:
            i += 1
            continue
        var size = 0
        while i < end:  # small-loop(digits of one number)
            b = Int(text[i])
            if b == 44 or b == 32 or b == 9 or b == 10 or b == 13:
                break
            var digit = (b >= 48 and b <= 57) or b == 43 or b == 45 or b == 46 or b == 69 or b == 101
            if not digit or size >= 63:
                raise Error("transport: header field " + key + " is not a list of numbers")
            token[size] = UInt8(b)
            size += 1
            i += 1
        token[size] = UInt8(0)
        var value = external_call["strtod", Float64](token.unsafe_ptr(), stop.unsafe_ptr())
        if stop[0] != Int(token.unsafe_ptr()) + size or not isfinite(value):
            raise Error("transport: header field " + key + " is not a list of numbers")
        values.append(value)
    _ = len(token)
    _ = len(stop)
    return values^


def _cross_flag(address: Int, count: Int, key: String) raises -> Bool:
    if not transport_header_has(address, count, key):
        return False
    return transport_header_bool(address, count, key)


def _cross_int_is(address: Int, count: Int, key: String, expected: Int) -> Bool:
    try:
        if not transport_header_has(address, count, key):
            return False
        return transport_header_int(address, count, key) == expected
    except:
        return False


def _cross_need(address: Int, count: Int, key: String) raises:
    """A field the frame must carry. Its absence is refused in the words the
    subscript read gave (the missing key, quoted), as a ValueError."""
    if not transport_header_has(address, count, key):
        raise Error("'" + key + "'")


def _cross_send(fd: Int, timeout: Float64, header: TransportHeader, payload_address: Int, payload_bytes: Int) raises:
    var text = header.encode(payload_bytes)
    transport_send_words(fd, timeout, Int(text.unsafe_ptr()), len(text), payload_address, payload_bytes)
    _ = len(text)


# ------------------------------------------------------------ fold and digests
def cross_fold_into(mode: Int, vendor: Int, total_address: Int, first_address: Int, parts_address: Int,
                    first_part: Int, parts: Int, n: Int) raises:
    """`total` = the float32[n] at `first_address`, then the base image's
    fold_pair_f32 of parts `first_part..parts-1` (float32[n] each, contiguous
    from `parts_address`) in that order."""
    var entry = parallel_ref(PARALLEL_TABLE_BASE, ParallelEntryKey(mode, vendor))
    if n > 0 and first_address != total_address:
        unsafe_memcpy(dest=_CrossBytes(unsafe_from_address=total_address),
                      src=_CrossBytes(unsafe_from_address=first_address), count=n * 4)
    for k in range(first_part, parts):  # driver-loop: over logical shard gradient buffers in immutable order
        _ = parallel_fold_pair_f32(entry, total_address, parts_address + k * n * 4, n)


def cross_state_hash(addresses: List[Int], n: Int, nt: Int) -> String:
    """SHA-256 over parameters, m, v (float32[n]) and flags (int32[nt])."""
    var hasher = Sha256()
    for i in range(4):  # small-loop(four state buffers)
        hasher.update(addresses[i], (nt if i == 3 else n) * 4)
    return hasher.hexdigest()


def cross_scheme_name(scheme: Int) -> String:
    return String("sha256.v1") if scheme == CROSS_SCHEME_V1 else String("sliced-sha256-8.v2")


def cross_float_word(value: Float64) raises -> String:
    if not isfinite(value):
        raise Error("recorded float32 must be finite")
    return _hex32(bitcast[DType.uint32](Float32(value)))


# ------------------------------------------------------------ feed, rates, readiness
struct CrossFeed(Movable):
    """int32[steps, shards, batch, width] at `address`; the owner pins it."""
    var address: Int
    var steps: Int
    var shards: Int
    var batch: Int
    var width: Int
    var start: Int

    def __init__(out self):
        self.address = 0
        self.steps = 0
        self.shards = 0
        self.batch = 0
        self.width = 0
        self.start = 0


def cross_feed_address(feed: CrossFeed, step: Int, shard: Int) raises -> Int:
    var local = step - feed.start
    if local < 0 or local >= feed.steps or shard < 0 or shard >= feed.shards:
        raise Error("BatchFeed step or shard falls outside its retained data")
    return feed.address + (local * feed.shards + shard) * feed.batch * feed.width * 4


struct CrossRates(Movable):
    var values: List[Float32]
    var start: Int

    def __init__(out self):
        self.values = List[Float32]()
        self.start = 0


def cross_rate(rates: CrossRates, step: Int) raises -> Float32:
    var at = step - rates.start
    if at < 0 or at >= len(rates.values):
        raise Error("LearningRateTable step falls outside its retained schedule")
    return rates.values[at]


def _cross_nap(seconds: Float64):
    var request = List[Int64](length=2, fill=Int64(0))
    var whole = Int64(seconds)
    request[0] = whole
    request[1] = Int64((seconds - Float64(whole)) * 1e9)
    var remaining = List[Int64](length=2, fill=Int64(0))
    _ = external_call["nanosleep", c_int](request.unsafe_ptr(), remaining.unsafe_ptr())
    _ = len(request)
    _ = len(remaining)


def cross_ready_wait(cell_address: Int, unlimited: Bool, timeout: Float64) raises -> Bool:
    """Poll the readiness word at `cell_address` (0 clear, 1 set) each
    millisecond; False when `timeout` seconds pass first."""
    var cell = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=cell_address)
    var start = transport_clock()
    while cell[0] == 0:  # driver-loop: over native readiness wait notifications
        if not unlimited and transport_clock() - start >= timeout:
            return False
        _cross_nap(0.001)
    return True


# ------------------------------------------------------------ coordinator
struct CrossCoordinatorState(Movable):
    """One Coordinator: its admitted configuration, the connected group as
    typed words (filled by the door before each drive) and the last step's row."""
    var workers: Int
    var shards: Int
    var steps: Int
    var max_bytes: Int
    var chained: Bool
    var host: String
    var port: Int
    var timeout: Float64
    var accept_timeout: Float64
    var fold_mode: Int
    var fold_vendor: Int
    # The group: one entry per peer, in connection order; `order` is the
    # order the steps use (by first shard when chained).
    var fds: List[Int]
    var timeouts: List[Float64]
    var names: List[String]
    var protocols: List[String]
    var defects: List[Int]
    var same: List[Int]
    var peer_chained: List[Bool]
    var starts: List[Int]
    var ids: List[Int]
    var order: List[Int]
    var n_total: Int
    var completed: Int
    var head: List[UInt8]
    var gather: List[Float32]
    var total: List[Float32]
    # The row of the step just driven; states and timing in `order`.
    var row_losses: List[Float64]
    var row_states: List[String]
    var row_agreed: String
    var row_total_hash: String
    var timing_keys: List[String]
    var timing_values: List[Float64]

    def __init__(out self):
        self.workers = 0
        self.shards = 0
        self.steps = 0
        self.max_bytes = 0
        self.chained = False
        self.host = String("")
        self.port = 0
        self.timeout = Float64(0)
        self.accept_timeout = Float64(0)
        self.fold_mode = -1
        self.fold_vendor = -1
        self.fds = List[Int]()
        self.timeouts = List[Float64]()
        self.names = List[String]()
        self.protocols = List[String]()
        self.defects = List[Int]()
        self.same = List[Int]()
        self.peer_chained = List[Bool]()
        self.starts = List[Int]()
        self.ids = List[Int]()
        self.order = List[Int]()
        self.n_total = 0
        self.completed = 0
        self.head = List[UInt8]()
        self.gather = List[Float32]()
        self.total = List[Float32]()
        self.row_losses = List[Float64]()
        self.row_states = List[String]()
        self.row_agreed = String("")
        self.row_total_hash = String("")
        self.timing_keys = List[String]()
        self.timing_values = List[Float64]()


def cross_coordinator_clear(mut c: CrossCoordinatorState):
    c.fds = List[Int]()
    c.timeouts = List[Float64]()
    c.names = List[String]()
    c.protocols = List[String]()
    c.defects = List[Int]()
    c.same = List[Int]()
    c.peer_chained = List[Bool]()
    c.starts = List[Int]()
    c.starts.append(0)
    c.ids = List[Int]()
    c.order = List[Int]()


def cross_coordinator_refuse(c: CrossCoordinatorState, reason: String) raises:
    """Tell every connected worker, then raise the mismatch."""
    var header = TransportHeader()
    header.put_text("cmd", "refuse")
    header.put_text("reason", reason)
    var text = header.encode(0)
    for i in range(len(c.fds)):  # driver-loop: over connected worker refusal notices
        try:
            transport_send_words(c.fds[i], c.timeouts[i], Int(text.unsafe_ptr()), len(text), 0, 0)
        except:
            pass
    _ = len(text)
    raise Error("CrossVendorMismatch: " + reason)


def cross_coordinator_admit(mut c: CrossCoordinatorState) raises:
    """The group is a valid split of one step, or every worker is told why not."""
    var count = len(c.fds)
    if count != c.workers:
        cross_coordinator_refuse(c, "worker count differs from the admitted group")
    var owned = List[Int]()
    for i in range(count):  # driver-loop: over worker protocol and shard ownership declarations
        if c.defects[i] == CROSS_DEFECT_NAME:
            cross_coordinator_refuse(c, "workers require distinct string names")
        for j in range(i):  # small-loop(earlier workers of the group)
            if c.names[j] == c.names[i]:
                cross_coordinator_refuse(c, "workers require distinct string names")
        if c.protocols[i] != CROSS_PROTOCOL:
            cross_coordinator_refuse(c, "protocol mismatch: " + c.names[i])
        if c.defects[i] == CROSS_DEFECT_IDS or c.starts[i + 1] == c.starts[i]:
            cross_coordinator_refuse(c, "workers require nonempty shard lists")
        if c.defects[i] == CROSS_DEFECT_ID:
            cross_coordinator_refuse(c, "shard IDs must be integers")
        for k in range(c.starts[i], c.starts[i + 1]):  # driver-loop: over one worker's declared shard IDs
            # Kept sorted: a stable insertion of one id.
            owned.append(c.ids[k])
            var at = len(owned) - 1
            while at > 0 and owned[at - 1] > owned[at]:  # small-loop(ids admitted so far)
                var held = owned[at - 1]
                owned[at - 1] = owned[at]
                owned[at] = held
                at -= 1
        if (c.same[i] & 1) == 0:
            cross_coordinator_refuse(c, "workers start from different state")
        if (c.same[i] & 2) == 0:
            cross_coordinator_refuse(c, "workers start from different completed")
        if (c.same[i] & 4) == 0:
            cross_coordinator_refuse(c, "workers start from different n_total")
    if len(owned) != c.shards:
        cross_coordinator_refuse(c, "workers must own every logical shard exactly once")
    for k in range(c.shards):  # driver-loop: over logical shards (ownership check)
        if owned[k] != k:
            cross_coordinator_refuse(c, "workers must own shards 0.." + String(c.shards - 1) + " exactly once")
    if c.n_total < 0 or c.max_bytes < 0 or c.n_total > c.max_bytes // 4:
        cross_coordinator_refuse(c, "gradient exceeds the coordinator byte bound")
    c.order = List[Int]()
    for i in range(count):  # small-loop(workers of the group)
        c.order.append(i)
    if c.chained:
        # Stable insertion by the first shard.
        for i in range(1, count):  # driver-loop: over worker blocks in immutable shard order
            var peer = c.order[i]
            var j = i
            while j > 0 and c.ids[c.starts[c.order[j - 1]]] > c.ids[c.starts[peer]]:  # driver-loop: over worker blocks in immutable shard order
                c.order[j] = c.order[j - 1]
                j -= 1
            c.order[j] = peer
        for i in range(count):  # driver-loop: over workers (contiguous block check)
            var p = c.order[i]
            for k in range(c.starts[p], c.starts[p + 1]):  # driver-loop: over one worker's declared shard IDs
                if c.ids[k] != c.ids[c.starts[p]] + (k - c.starts[p]):
                    cross_coordinator_refuse(c, "chained fold: worker must own a contiguous block")
            if not c.peer_chained[p]:
                cross_coordinator_refuse(c, "chained fold: every worker must start with chained=True")
    c.head = List[UInt8](length=1048576, fill=UInt8(0))
    c.total = List[Float32](length=c.n_total, fill=Float32(0))
    c.gather = List[Float32]()


def _cross_timing(mut c: CrossCoordinatorState, key: String, seconds: Float64):
    c.timing_keys.append(key)
    c.timing_values.append(seconds)


def cross_coordinator_step(mut c: CrossCoordinatorState, step: Int) raises:
    """Drive one lockstep step and leave its row in `c`."""
    var started = transport_clock()
    var count = len(c.fds)
    var n = c.n_total
    var total_bytes = n * 4
    var head = Int(c.head.unsafe_ptr())
    var capacity = len(c.head)
    var total = Int(c.total.unsafe_ptr())
    c.timing_keys = List[String]()
    c.timing_values = List[Float64]()
    c.row_states = List[String]()
    for i in range(count):  # driver-loop: over worker step requests
        var p = c.order[i]
        var request = TransportHeader()
        request.put_text("cmd", "step")
        request.put_int("step", step)
        request.put_bool("chained", c.chained)
        request.put_bool("first", c.chained and i == 0)
        _cross_send(c.fds[p], c.timeouts[p], request, 0, 0)
    if not c.chained:
        for i in range(count):  # driver-loop: over worker gradient payload bounds
            var p = c.order[i]
            if total_bytes != 0 and c.starts[p + 1] - c.starts[p] > c.max_bytes // total_bytes:
                cross_coordinator_refuse(c, "worker gradient payload exceeds coordinator byte bound")
        if len(c.gather) != c.shards * n:
            c.gather = List[Float32](length=c.shards * n, fill=Float32(0))
    var gather = Int(c.gather.unsafe_ptr())
    var losses = List[Float64](length=c.shards, fill=Float64(0))
    var span = List[Int](length=c.shards, fill=0)
    var block = 0
    for i in range(count):  # driver-loop: over worker gradient replies
        var p = c.order[i]
        var first = c.starts[p]
        var own = c.starts[p + 1] - first
        # Each worker's shards land in place in the gather area (no copy).
        var got = transport_receive_into(c.fds[p], c.timeouts[p], head, capacity,
                                         gather + block * total_bytes, 0 if c.chained else own * total_bytes)
        var answered = _cross_int_is(head, got[0], "step", step)
        if answered:
            try:
                var declared = transport_header_words(head, got[0], "shards")
                answered = len(declared) == own
                for k in range(min(own, len(declared))):  # driver-loop: over one worker's declared shard IDs
                    answered = answered and declared[k] == c.ids[first + k]
            except:
                answered = False
        if not answered:
            cross_coordinator_refuse(c, c.names[p] + " answered the wrong step or shards")
        _cross_need(head, got[0], "losses")
        var values = List[Float64]()
        var counted = True
        try:
            values = cross_header_floats(head, got[0], "losses")
        except:
            counted = False
        if not counted or len(values) != own or (not c.chained and got[1] != own * total_bytes):
            cross_coordinator_refuse(c, c.names[p] + " sent a wrong gradient or loss count")
        for k in range(own):  # driver-loop: over one worker's logical shard replies
            losses[c.ids[first + k]] = values[k]
            span[c.ids[first + k]] = block + k
        block += own
        if c.chained:
            _cross_timing(c, "gradients_" + c.names[p], transport_clock() - started)
    if c.chained:
        var have = 0
        for i in range(count):  # driver-loop: over chained prefix handoffs in fixed shard order
            var p = c.order[i]
            var request = TransportHeader()
            request.put_text("cmd", "fold")
            request.put_int("step", step)
            var folded_at = transport_clock()
            _cross_send(c.fds[p], c.timeouts[p], request, total, have)
            var got = transport_receive_into(c.fds[p], c.timeouts[p], head, capacity, total, total_bytes)
            if not _cross_int_is(head, got[0], "step", step) or got[1] != total_bytes:
                cross_coordinator_refuse(c, c.names[p] + " returned a wrong fold")
            have = total_bytes
            _cross_timing(c, "fold_" + c.names[p], transport_clock() - folded_at)
    else:
        # The ordered fold reads every shard's received span in place.
        var entry = parallel_ref(PARALLEL_TABLE_BASE, ParallelEntryKey(c.fold_mode, c.fold_vendor))
        if n > 0:
            unsafe_memcpy(dest=_CrossBytes(unsafe_from_address=total),
                          src=_CrossBytes(unsafe_from_address=gather + span[0] * total_bytes), count=total_bytes)
        for k in range(1, c.shards):  # driver-loop: over logical shard gradients in fixed order
            _ = parallel_fold_pair_f32(entry, total, gather + span[k] * total_bytes, n)
    c.row_total_hash = sha256_hex_addr(total, total_bytes)
    var apply = TransportHeader()
    apply.put_text("cmd", "apply")
    apply.put_int("step", step)
    apply.put_text("total_sha256", c.row_total_hash)
    cross_header_put_floats(apply, "losses", losses)
    var frame = apply.encode(total_bytes)
    var last = apply.encode(0)
    for i in range(count):  # driver-loop: over identical optimizer commit requests
        var p = c.order[i]
        if c.chained and i == count - 1:
            # The last worker of the chain holds the total it just returned.
            transport_send_words(c.fds[p], c.timeouts[p], Int(last.unsafe_ptr()), len(last), 0, 0)
        else:
            transport_send_words(c.fds[p], c.timeouts[p], Int(frame.unsafe_ptr()), len(frame), total, total_bytes)
    _ = len(frame)
    _ = len(last)
    _cross_timing(c, "apply_sent", transport_clock() - started)
    c.row_agreed = String("")
    for i in range(count):  # driver-loop: over committed worker state witnesses
        var p = c.order[i]
        var got = transport_receive_into(c.fds[p], c.timeouts[p], head, capacity, 0, 0)
        if not _cross_int_is(head, got[0], "completed", step + 1):
            cross_coordinator_refuse(c, c.names[p] + " did not commit step " + String(step + 1))
        _cross_need(head, got[0], "state")
        var state = transport_header_text(head, got[0], "state")
        if i == 0:
            c.row_agreed = state
        elif state != c.row_agreed:
            cross_coordinator_refuse(c, "replicas disagree after step " + String(step + 1))
        c.row_states.append(state)
        _cross_timing(c, "applied_" + c.names[p], transport_clock() - started)
    _cross_timing(c, "step_seconds", transport_clock() - started)
    c.row_losses = losses^
    _ = len(span)
    _ = len(c.head)
    _ = len(c.gather)
    _ = len(c.total)


def cross_coordinator_done(c: CrossCoordinatorState) raises:
    var done = TransportHeader()
    done.put_text("cmd", "done")
    for i in range(len(c.fds)):  # driver-loop: over group completion notices
        _cross_send(c.fds[c.order[i]], c.timeouts[c.order[i]], done, 0, 0)


# ------------------------------------------------------------ worker
struct CrossWorkerState(Movable):
    """One Worker: its admitted configuration, the buffers of one run and the
    record of the step it committed last."""
    var name: String
    var vendor: String
    var shards: List[Int]
    var chained: Bool
    var host: String
    var port: Int
    var connect_timeout: Float64
    var timeout: Float64
    var scheme: Int
    var retain: Bool
    var checkpoint_steps: List[Int]
    # Configured expectations, one entry per field, keys ascending per step.
    var expected_step: List[Int]
    var expected_key: List[String]
    var expected_value: List[String]
    var fold_mode: Int
    var fold_vendor: Int
    # One run (set by the door's UNPACK at run).
    var fd: Int
    var socket_timeout: Float64
    var n: Int
    var nt: Int
    var feed: CrossFeed
    var has_rates: Bool
    var rates: CrossRates
    var snapshot: List[Int]
    var head: List[UInt8]
    var head_bytes: Int
    var payload: List[Float32]
    var payload_bytes: Int
    var held: List[Float32]
    var held_count: Int
    var total: List[Float32]
    var has_total: Bool
    var on_device: Bool
    var reply: List[UInt8]
    var reply_address: Int
    var reply_bytes: Int
    var reply_pending: Bool
    var committed: Int
    var step_started: Float64
    # The step committed last (PACK reads it).
    var applied_address: Int
    var commit_step: Int
    var commit_state: String
    var commit_has_total_hash: Bool
    var commit_total_hash: String
    var commit_losses: List[Float64]
    var commit_loss_words: List[String]
    var commit_state_sha: String
    var commit_has_gradient_sha: Bool
    var commit_gradient_sha: String
    var commit_lr_word: String
    var commit_hash_seconds: Float64
    var commit_mismatch: String
    var commit_checkpoint: Bool

    def __init__(out self):
        self.name = String("")
        self.vendor = String("")
        self.shards = List[Int]()
        self.chained = False
        self.host = String("")
        self.port = 0
        self.connect_timeout = Float64(0)
        self.timeout = Float64(0)
        self.scheme = CROSS_SCHEME_V1
        self.retain = False
        self.checkpoint_steps = List[Int]()
        self.expected_step = List[Int]()
        self.expected_key = List[String]()
        self.expected_value = List[String]()
        self.fold_mode = -1
        self.fold_vendor = -1
        self.fd = -1
        self.socket_timeout = Float64(-1)
        self.n = 0
        self.nt = 0
        self.feed = CrossFeed()
        self.has_rates = False
        self.rates = CrossRates()
        self.snapshot = List[Int]()
        self.head = List[UInt8]()
        self.head_bytes = 0
        self.payload = List[Float32]()
        self.payload_bytes = 0
        self.held = List[Float32]()
        self.held_count = 0
        self.total = List[Float32]()
        self.has_total = False
        self.on_device = False
        self.reply = List[UInt8]()
        self.reply_address = 0
        self.reply_bytes = 0
        self.reply_pending = False
        self.committed = 0
        self.step_started = Float64(0)
        self.applied_address = 0
        self.commit_step = 0
        self.commit_state = String("")
        self.commit_has_total_hash = False
        self.commit_total_hash = String("")
        self.commit_losses = List[Float64]()
        self.commit_loss_words = List[String]()
        self.commit_state_sha = String("")
        self.commit_has_gradient_sha = False
        self.commit_gradient_sha = String("")
        self.commit_lr_word = String("")
        self.commit_hash_seconds = Float64(0)
        self.commit_mismatch = String("")
        self.commit_checkpoint = False


def cross_worker_begin(mut w: CrossWorkerState, n: Int, nt: Int, var snapshot: List[Int]) raises:
    """The buffers of one run: the frame header, one received payload, this
    worker's shard gradients (also the gathered reply's payload, in place)
    and the folded total. Allocated once per run, not per step."""
    w.n = n
    w.nt = nt
    w.snapshot = snapshot^
    w.head = List[UInt8](length=1048576, fill=UInt8(0))
    w.head_bytes = 0
    w.payload = List[Float32](length=n, fill=Float32(0))
    w.payload_bytes = 0
    w.held = List[Float32](length=len(w.shards) * n, fill=Float32(0))
    w.held_count = 0
    w.total = List[Float32](length=n, fill=Float32(0))
    w.has_total = False
    w.on_device = False
    w.reply_pending = False
    w.committed = 0
    w.step_started = transport_clock()


def cross_worker_end(mut w: CrossWorkerState):
    w.fd = -1
    w.head = List[UInt8]()
    w.payload = List[Float32]()
    w.held = List[Float32]()
    w.total = List[Float32]()
    w.snapshot = List[Int]()
    w.reply = List[UInt8]()
    w.reply_pending = False


def cross_worker_receive(mut w: CrossWorkerState) raises:
    var got = transport_receive_into(w.fd, w.socket_timeout, Int(w.head.unsafe_ptr()), len(w.head),
                                     Int(w.payload.unsafe_ptr()), w.n * 4)
    w.head_bytes = got[0]
    w.payload_bytes = got[1]


def cross_worker_hello(mut w: CrossWorkerState, completed: Int) raises:
    """Declare this worker to the coordinator and receive its first command."""
    var hello = TransportHeader()
    hello.put_text("protocol", CROSS_PROTOCOL)
    hello.put_text("name", w.name)
    hello.put_text("vendor", w.vendor)
    hello.put_words("shards", w.shards)
    hello.put_int("completed", completed)
    hello.put_text("state", cross_state_hash(w.snapshot, w.n, w.nt))
    hello.put_int("n_total", w.n)
    hello.put_bool("chained", w.chained)
    _cross_send(w.fd, w.socket_timeout, hello, 0, 0)
    cross_worker_receive(w)


def cross_worker_reply(mut w: CrossWorkerState) raises:
    if not w.reply_pending:
        return
    w.reply_pending = False
    transport_send_words(w.fd, w.socket_timeout, Int(w.reply.unsafe_ptr()), len(w.reply), w.reply_address, w.reply_bytes)


def _cross_reply(mut w: CrossWorkerState, header: TransportHeader, payload_address: Int, payload_bytes: Int) raises:
    w.reply = header.encode(payload_bytes)
    w.reply_address = payload_address
    w.reply_bytes = payload_bytes
    w.reply_pending = True


def _cross_live(session: ParallelByteSession) raises:
    # The door opened the session before the run; RUN cannot open one.
    if parallel_session_openable(session):
        raise Error("RuntimeError: parallel trainer has no open session")


def _cross_shard_ids(w: CrossWorkerState, session: ParallelByteSession, step: Int, shard: Int) raises -> Int:
    var address = cross_feed_address(w.feed, step, shard)
    if w.feed.batch != session.dims[0] or w.feed.width != session.dims[1] + 1:
        raise Error("parallel input has wrong shape")
    return address


def _cross_fold_reply(mut w: CrossWorkerState) raises:
    w.held_count = 0
    w.on_device = False
    w.has_total = True
    var reply = TransportHeader()
    _cross_need(Int(w.head.unsafe_ptr()), w.head_bytes, "step")
    reply.put_int("step", transport_header_int(Int(w.head.unsafe_ptr()), w.head_bytes, "step"))
    _cross_reply(w, reply, Int(w.total.unsafe_ptr()), w.n * 4)


def _cross_prefix(w: CrossWorkerState) raises -> Int:
    if w.payload_bytes == 0:
        return 0
    if w.payload_bytes != w.n * 4:
        raise Error("ordered_fold: gradients differ in length")
    return Int(w.payload.unsafe_ptr())


def cross_worker_host_fold(mut w: CrossWorkerState) raises:
    """The chained fold on the host (a trainer without a device fold): the
    received prefix, then the held shard gradients in order."""
    var prefix = _cross_prefix(w)
    var held = Int(w.held.unsafe_ptr())
    cross_fold_into(w.fold_mode, w.fold_vendor, Int(w.total.unsafe_ptr()), prefix if prefix != 0 else held,
                    held, 0 if prefix != 0 else 1, w.held_count, w.n)
    _cross_fold_reply(w)


def _cross_commit(mut w: CrossWorkerState, session: ParallelByteSession, done: Int, source: Int) raises:
    var head = Int(w.head.unsafe_ptr())
    var hash_start = transport_clock()
    w.applied_address = source
    w.commit_step = done
    w.commit_state = cross_state_hash(w.snapshot, w.n, w.nt)
    # The refusal words of the boundary reads this replaced: a `losses` that
    # is absent, null or not a list; then an entry that is not a number.
    var listed = transport_header_find(head, w.head_bytes, "losses")
    if listed[0] < 0 or Int(_CrossBytes(unsafe_from_address=head)[listed[0]]) != 91:
        raise Error("committed losses requires list, tuple or range; Python iteration hooks are not accepted")
    var numbers = True
    try:
        w.commit_losses = cross_header_floats(head, w.head_bytes, "losses")
    except:
        numbers = False
    if not numbers:
        raise Error("recorded float32 requires a native number; Python conversion hooks are not accepted")
    if w.scheme == CROSS_SCHEME_V1:
        w.commit_state_sha = w.commit_state
        w.commit_has_gradient_sha = w.commit_has_total_hash
        w.commit_gradient_sha = w.commit_total_hash
    else:
        var chain = Sha256()
        for i in range(4):  # small-loop(four state buffers): one digest update per named state field
            chain.update_text(sha256_sliced8_hex(w.snapshot[i], (w.nt if i == 3 else w.n) * 4))
        w.commit_state_sha = chain.hexdigest()
        w.commit_has_gradient_sha = True
        w.commit_gradient_sha = sha256_sliced8_hex(source, w.n * 4)
    w.commit_lr_word = cross_float_word(session.scalars[PARALLEL_SCALAR_LR])
    w.commit_loss_words = List[String]()
    var joined = String("")
    for i in range(len(w.commit_losses)):  # driver-loop: over logical shard loss metadata
        var word = cross_float_word(w.commit_losses[i])
        if i != 0:
            joined += ","
        joined += word
        w.commit_loss_words.append(word)
    w.commit_hash_seconds = transport_clock() - hash_start
    w.commit_mismatch = String("")
    for i in range(len(w.expected_step)):  # driver-loop: over configured committed-state checks
        if w.expected_step[i] != done:
            continue
        var key = w.expected_key[i]
        var present = True
        var value = String("")
        if key == "state":
            value = w.commit_state
        elif key == "total_sha256":
            present = w.commit_has_total_hash
            value = w.commit_total_hash
        elif key == "state_sha256":
            value = w.commit_state_sha
        elif key == "gradient_sha256":
            present = w.commit_has_gradient_sha
            value = w.commit_gradient_sha
        elif key == "losses_f32_hex":
            value = joined
        elif key == "lr_f32_hex":
            value = w.commit_lr_word
        else:
            value = cross_scheme_name(w.scheme)
        if not present or value != w.expected_value[i]:
            w.commit_mismatch = key
            break
    w.commit_checkpoint = False
    if w.commit_mismatch.byte_length() == 0:
        for i in range(len(w.checkpoint_steps)):  # small-loop(configured checkpoint steps)
            if w.checkpoint_steps[i] == done:
                w.commit_checkpoint = True


def cross_worker_command(mut w: CrossWorkerState, mut session: ParallelByteSession) raises -> Int:
    """Handle the received command up to its reply (left pending for
    `cross_worker_reply`) and say what the door does next."""
    var head = Int(w.head.unsafe_ptr())
    var count = w.head_bytes
    var n = w.n
    var command = String("None")
    if transport_header_has(head, count, "cmd"):
        command = transport_header_text(head, count, "cmd")
    if command == "step":
        w.step_started = transport_clock()
        _cross_need(head, count, "step")
        var step = transport_header_int(head, count, "step")
        if session.completed != step:
            raise Error("CrossVendorMismatch: worker completed step differs from coordinator")
        if _cross_flag(head, count, "chained") != w.chained:
            raise Error("CrossVendorMismatch: coordinator and worker disagree on the protocol")
        if w.has_rates:
            parallel_session_set_lr(session, cross_rate(w.rates, step))
        _cross_live(session)
        var losses = List[Float64]()
        var held = Int(w.held.unsafe_ptr())
        var device_fold = w.chained and session.schedule == PARALLEL_REPLICA
        if device_fold and _cross_flag(head, count, "first"):
            parallel_session_require_split(session, "fold_reset")
            parallel_session_fold(session, False, 0)
            for k in range(len(w.shards)):  # driver-loop: over this worker's immutable logical shards
                var ids = _cross_shard_ids(w, session, step, w.shards[k])
                parallel_session_require_split(session, "shard_gradient_fold")
                losses.append(parallel_session_shard_gradient(session, ids, 0, True))
            w.held_count = 0
            w.has_total = False
            w.on_device = True
        else:
            for k in range(len(w.shards)):  # driver-loop: over this worker's immutable logical shards
                # The batch is read in place from the feed and the gradient
                # is written in place into the held area.
                var ids = _cross_shard_ids(w, session, step, w.shards[k])
                parallel_session_require_split(session, "shard_gradient")
                losses.append(parallel_session_shard_gradient(session, ids, held + k * n * 4, False))
            w.on_device = False
            if w.chained:
                w.held_count = len(w.shards)
                w.has_total = False
        var reply = TransportHeader()
        reply.put_int("step", step)
        reply.put_words("shards", w.shards)
        cross_header_put_floats(reply, "losses", losses)
        if w.chained:
            _cross_reply(w, reply, 0, 0)
        else:
            _cross_reply(w, reply, held, len(w.shards) * n * 4)
        return CROSS_CONTINUE
    if command == "fold":
        if not w.chained or (w.held_count == 0 and not w.on_device):
            raise Error("CrossVendorMismatch: worker was asked to fold with nothing held")
        _cross_live(session)
        var total = Int(w.total.unsafe_ptr())
        if w.on_device:
            parallel_session_require_split(session, "fold_export")
            parallel_session_fold_export(session, total)
        elif session.schedule == PARALLEL_REPLICA:
            if w.payload_bytes != 0 and w.payload_bytes != n * 4:
                raise Error("parallel input byte count differs from shape")
            parallel_session_require_split(session, "fold_reset")
            parallel_session_fold(session, False, Int(w.payload.unsafe_ptr()) if w.payload_bytes != 0 else 0)
            for k in range(w.held_count):  # driver-loop: over held logical gradients continuing the prefix
                parallel_session_require_split(session, "fold_add")
                parallel_session_fold(session, True, Int(w.held.unsafe_ptr()) + k * n * 4)
            parallel_session_require_split(session, "fold_export")
            parallel_session_fold_export(session, total)
        else:
            return CROSS_HOST_FOLD
        _cross_fold_reply(w)
        return CROSS_CONTINUE
    if command == "apply":
        var source = 0
        var size = -1
        if w.payload_bytes != 0:
            source = Int(w.payload.unsafe_ptr())
            size = w.payload_bytes
        elif w.has_total:
            source = Int(w.total.unsafe_ptr())
            size = n * 4
        if size != n * 4:
            raise Error("CrossVendorMismatch: summed gradient has the wrong size")
        w.commit_has_total_hash = False
        w.commit_total_hash = String("")
        var named = transport_header_find(head, count, "total_sha256")
        if named[0] >= 0 and Int(_CrossBytes(unsafe_from_address=head)[named[0]]) != 110:
            var expected = String("")
            try:
                expected = transport_header_text(head, count, "total_sha256")
            except:
                raise Error("CrossVendorMismatch: total hash differs from the coordinator")
            if sha256_hex_addr(source, n * 4) != expected:
                raise Error("CrossVendorMismatch: total hash differs from the coordinator")
            w.commit_has_total_hash = True
            w.commit_total_hash = expected
        _cross_live(session)
        parallel_session_require_split(session, "apply_gradient")
        var done = parallel_session_apply_gradient(session, source)
        w.has_total = False
        w.committed += 1
        # The witness export is written into the same four buffers every step.
        parallel_session_export(session, 0, False, w.snapshot.copy())
        _cross_commit(w, session, done, source)
        var reply = TransportHeader()
        reply.put_int("completed", done)
        reply.put_text("state", w.commit_state)
        _cross_reply(w, reply, 0, 0)
        return CROSS_COMMIT
    if command == "done":
        return CROSS_DONE
    if command == "refuse":
        var reason = String("")
        if transport_header_has(head, count, "reason"):
            reason = transport_header_text(head, count, "reason")
        raise Error("CrossVendorMismatch: coordinator refused: " + reason)
    raise Error("cross_vendor: unknown coordinator command " + command)
