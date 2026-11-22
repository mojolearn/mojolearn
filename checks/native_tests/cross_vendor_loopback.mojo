# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""Loopback contracts of the typed cross-vendor run
(bindings/cross_vendor_native_run.mojo): one gathered step and one chained
step of a real coordinator and real workers over local socket pairs.

Each worker runs its typed command handling on a `ParallelByteSession` whose
schedule table is the stand-in of the byte session suite (it answers each
operation and computes no model arithmetic; the gradients and folded totals
are fixture values placed in the worker's buffers). The gathered fold is the
base image's host `fold_pair_f32`.

One step of the coordinator is one call that waits for its workers, so the
two sides cannot alternate inside one thread. The fixture closes the loop
without a second thread: the workers first consume the frames a coordinator
sends (built here, field for field) and their replies stay queued in the
socket pair; the coordinator then drives its step on those queued replies;
last, every frame the coordinator actually sent is read back and required to
be byte-identical, header and payload, to the frame the worker consumed. So
each side ran on exactly the bytes the other side produced.

A third fixture holds the refusal words of malformed frames on the same
path. Host column; three independent fixtures across the host's cores. No
device is touched.
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require
from std.ffi import c_int, external_call
from core.sha256 import sha256_hex_addr
from core.transport_header import TransportHeader
from checks.numerics import GLOBAL_NUMERIC_MODE
from bindings.parallel_base_entry_host import publish_parallel_base_host
from bindings.native_posix_transport_run import posix_close, posix_socket_prepare
from bindings.native_transport_run import transport_send_words, transport_receive_into
from bindings.parallel_byte_native_run import ParallelByteSession, PARALLEL_REPLICA
from bindings.cross_vendor_native_run import (
    CROSS_PROTOCOL, CROSS_SCHEME_V1, CROSS_DEFECT_NONE, CROSS_CONTINUE, CROSS_COMMIT, CROSS_DONE, CROSS_HOST_FOLD,
    CrossCoordinatorState, CrossWorkerState,
    cross_header_put_floats, cross_fold_into,
    cross_coordinator_clear, cross_coordinator_admit, cross_coordinator_step, cross_coordinator_done,
    cross_worker_begin, cross_worker_end, cross_worker_receive, cross_worker_command, cross_worker_host_fold,
    cross_worker_reply,
)
from checks.native_tests.parallel_byte_stand_in import (
    stand_in, provider, opened, STEP_AT_OPEN, session as stand_in_session,
)

comptime CROSS_VENDOR_LOOPBACK_FIXTURES = 3
# The stand-in session's sizes (parallel_byte_stand_in.session).
comptime N = 64
comptime NT = 12
comptime BATCH = 2
comptime WIDTH = 3
comptime WAIT = 20.0


def same(a: Int, b: Int, count: Int) -> Bool:
    var left = Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=a)
    var right = Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=b)
    for i in range(count):  # small-loop(bytes of one fixture frame)
        if left[i] != right[i]:
            return False
    return True


def pair() raises -> List[Int]:
    """A connected local stream pair: [coordinator end, worker end]."""
    var fds = Array[c_int, 2](fill=c_int(-1))
    require(external_call["socketpair", c_int](c_int(1), c_int(1), c_int(0), fds.unsafe_ptr()) == 0,
            "cross loopback: socketpair failed")
    var ends: List[Int] = [Int(fds[0]), Int(fds[1])]
    posix_socket_prepare(ends[0])
    posix_socket_prepare(ends[1])
    return ends^


def put(fd: Int, header: TransportHeader, payload: Int, payload_bytes: Int) raises -> List[UInt8]:
    """Send one frame and return its header bytes."""
    var text = header.encode(payload_bytes)
    transport_send_words(fd, WAIT, Int(text.unsafe_ptr()), len(text), payload, payload_bytes)
    return text^


def took(fd: Int, text: List[UInt8], payload: Int, payload_bytes: Int, what: String) raises:
    """The next frame on `fd` is exactly `text` with exactly these payload bytes."""
    var head = List[UInt8](length=4096, fill=UInt8(0))
    var body = List[UInt8](length=4 * N * 4, fill=UInt8(0))
    var got = transport_receive_into(fd, WAIT, Int(head.unsafe_ptr()), 4096, Int(body.unsafe_ptr()), 4 * N * 4)
    require(got[0] == len(text) and same(Int(head.unsafe_ptr()), Int(text.unsafe_ptr()), got[0]),
            what + ": the coordinator's header differs from the frame the worker consumed")
    require(got[1] == payload_bytes and same(Int(body.unsafe_ptr()), payload, payload_bytes),
            what + ": the coordinator's payload differs from the frame the worker consumed")
    _ = len(head)
    _ = len(body)
    _ = len(text)


def step_frame(step: Int, chained: Bool, first: Bool) raises -> TransportHeader:
    var header = TransportHeader()
    header.put_text("cmd", "step")
    header.put_int("step", step)
    header.put_bool("chained", chained)
    header.put_bool("first", first)
    return header^


def fold_frame(step: Int) raises -> TransportHeader:
    var header = TransportHeader()
    header.put_text("cmd", "fold")
    header.put_int("step", step)
    return header^


def apply_frame(step: Int, total_hash: String, shards: Int) raises -> TransportHeader:
    var header = TransportHeader()
    header.put_text("cmd", "apply")
    header.put_int("step", step)
    header.put_text("total_sha256", total_hash)
    # The stand-in answers every shard gradient with the loss 2.5.
    cross_header_put_floats(header, "losses", List[Float64](length=shards, fill=2.5))
    return header^


def witness(state: Int) -> List[Int]:
    """The four state arrays inside one area: parameters, m, v, flags."""
    var addresses: List[Int] = [state, state + 4 * N, state + 8 * N, state + 12 * N]
    return addresses^


def worker(name: String, var shards: List[Int], chained: Bool, fd: Int, feed: Int, feed_shards: Int,
           var snapshot: List[Int]) raises -> CrossWorkerState:
    """A worker as the door's UNPACK leaves it at the start of a run."""
    var w = CrossWorkerState()
    w.name = name
    w.vendor = String("cpu")
    w.shards = shards^
    w.chained = chained
    w.scheme = CROSS_SCHEME_V1
    w.fold_mode = Int(GLOBAL_NUMERIC_MODE)
    w.fold_vendor = 0
    w.fd = fd
    w.socket_timeout = WAIT
    w.feed.address = feed
    w.feed.steps = 1
    w.feed.shards = feed_shards
    w.feed.batch = BATCH
    w.feed.width = WIDTH
    w.feed.start = STEP_AT_OPEN
    w.has_rates = False
    cross_worker_begin(w, N, NT, snapshot^)
    return w^


def serve(mut w: CrossWorkerState, mut trainer: ParallelByteSession) raises -> Int:
    """One command of the door's serve loop: receive, handle, reply."""
    cross_worker_receive(w)
    var code = cross_worker_command(w, trainer)
    if code == CROSS_HOST_FOLD:
        cross_worker_host_fold(w)
    cross_worker_reply(w)
    return code


def serve_refusal(mut w: CrossWorkerState, mut trainer: ParallelByteSession) -> String:
    try:
        _ = serve(w, trainer)
    except problem:
        return String(problem)
    return String("")


def step_refusal(mut c: CrossCoordinatorState, step: Int) -> String:
    try:
        cross_coordinator_step(c, step)
    except problem:
        return String(problem)
    return String("")


def group(mut c: CrossCoordinatorState, workers: Int, shards: Int, chained: Bool):
    c.workers = workers
    c.shards = shards
    c.chained = chained
    c.max_bytes = 1048576
    c.n_total = N
    c.completed = STEP_AT_OPEN
    c.fold_mode = Int(GLOBAL_NUMERIC_MODE)
    c.fold_vendor = 0
    cross_coordinator_clear(c)


def join(mut c: CrossCoordinatorState, name: String, fd: Int, ids: List[Int], chained: Bool):
    """One admitted hello, as the door's peer UNPACK records it."""
    c.fds.append(fd)
    c.timeouts.append(WAIT)
    c.names.append(name)
    c.protocols.append(String(CROSS_PROTOCOL))
    c.defects.append(CROSS_DEFECT_NONE)
    c.same.append(7)
    c.peer_chained.append(chained)
    for id in ids:  # metadata: one peer's declared shard ids
        c.ids.append(id)
    c.starts.append(len(c.ids))


def committed(w: CrossWorkerState, step: Int, total_hash: String, shards: Int, who: String) raises:
    require(w.committed == 1 and w.commit_step == step + 1, who + ": the apply commits the next step once")
    require(w.commit_has_total_hash and w.commit_total_hash == total_hash, who + ": the commit records the coordinator's total hash")
    require(len(w.commit_losses) == shards and len(w.commit_loss_words) == shards, who + ": one committed loss per logical shard")
    for i in range(shards):  # small-loop(logical shards of the fixture)
        require(w.commit_losses[i] == 2.5 and w.commit_loss_words[i] == "40200000", who + ": committed loss and its float32 word")
    require(w.commit_lr_word == "3e000000", who + ": the commit records the session's learning rate word")
    require(w.commit_mismatch.byte_length() == 0 and not w.commit_checkpoint, who + ": no expectation and no checkpoint configured")
    require(w.commit_state_sha == w.commit_state, who + ": sha256.v1 records the state digest itself")


def gathered(ca: Int, wa: Int, cb: Int, wb: Int) raises:
    """Three logical shards on two workers (a: 0 and 2, b: 1), gathered."""
    var step = STEP_AT_OPEN
    var a_words = stand_in()
    var a_records = provider()
    var a_trainer = stand_in_session(Int(a_records.unsafe_ptr()), PARALLEL_REPLICA, 1, False)
    opened(a_trainer, Int(a_words.unsafe_ptr()))
    var b_words = stand_in()
    var b_records = provider()
    var b_trainer = stand_in_session(Int(b_records.unsafe_ptr()), PARALLEL_REPLICA, 1, False)
    opened(b_trainer, Int(b_words.unsafe_ptr()))
    require(a_trainer.n_total == N and a_trainer.n_tensors == NT and a_trainer.completed == step,
            "gathered: the stand-in session's sizes")
    var a_state = List[Float32](length=3 * N + NT, fill=Float32(0.5))
    var b_state = List[Float32](length=3 * N + NT, fill=Float32(0.5))
    var ids = List[Int32](length=3 * BATCH * WIDTH, fill=Int32(1))
    var a_shards: List[Int] = [0, 2]
    var b_shards: List[Int] = [1]
    var a = worker(String("a"), a_shards.copy(), False, wa, Int(ids.unsafe_ptr()), 3, witness(Int(a_state.unsafe_ptr())))
    var b = worker(String("b"), b_shards.copy(), False, wb, Int(ids.unsafe_ptr()), 3, witness(Int(b_state.unsafe_ptr())))
    # The shard gradients (the stand-in writes none): an order-sensitive
    # float32 sum, so a fold in any order but 0, 1, 2 gives another total.
    var grads = List[Float32](length=3 * N, fill=Float32(0))
    for i in range(N):  # small-loop(elements of one fixture gradient)
        grads[i] = Float32(100000000.0)
        grads[N + i] = Float32(1.0)
        grads[2 * N + i] = Float32(-100000000.0)
        a.held[i] = grads[i]
        a.held[N + i] = grads[2 * N + i]
        b.held[i] = grads[N + i]
    var expected = List[Float32](length=N, fill=Float32(0))
    var total = Int(expected.unsafe_ptr())
    cross_fold_into(Int(GLOBAL_NUMERIC_MODE), 0, total, Int(grads.unsafe_ptr()), Int(grads.unsafe_ptr()), 1, 3, N)
    var total_hash = sha256_hex_addr(total, 4 * N)

    # The workers, on the frames a coordinator sends.
    var a_step = put(ca, step_frame(step, False, False), 0, 0)
    require(serve(a, a_trainer) == CROSS_CONTINUE, "gathered: a answers the step")
    var b_step = put(cb, step_frame(step, False, False), 0, 0)
    require(serve(b, b_trainer) == CROSS_CONTINUE, "gathered: b answers the step")
    var a_apply = put(ca, apply_frame(step, total_hash, 3), total, 4 * N)
    require(serve(a, a_trainer) == CROSS_COMMIT, "gathered: a commits")
    var b_apply = put(cb, apply_frame(step, total_hash, 3), total, 4 * N)
    require(serve(b, b_trainer) == CROSS_COMMIT, "gathered: b commits")
    committed(a, step, total_hash, 3, String("gathered a"))
    committed(b, step, total_hash, 3, String("gathered b"))
    require(a.commit_state == b.commit_state, "gathered: equal state arrays give equal witnesses")

    # The coordinator, on the workers' queued replies.
    var c = CrossCoordinatorState()
    group(c, 2, 3, False)
    join(c, String("a"), ca, a_shards, False)
    join(c, String("b"), cb, b_shards, False)
    cross_coordinator_admit(c)
    cross_coordinator_step(c, step)
    require(c.row_total_hash == total_hash and same(Int(c.total.unsafe_ptr()), total, 4 * N),
            "gathered: the coordinator folds the shards in logical order 0, 1, 2")
    require(len(c.row_losses) == 3 and c.row_losses[0] == 2.5 and c.row_losses[1] == 2.5 and c.row_losses[2] == 2.5,
            "gathered: one loss per logical shard in the row")
    require(len(c.row_states) == 2 and c.row_agreed == a.commit_state and c.row_states[1] == b.commit_state,
            "gathered: the row carries each worker's committed witness")

    # Every frame the coordinator sent is the frame its worker consumed.
    took(wa, a_step, 0, 0, String("gathered a step"))
    took(wa, a_apply, total, 4 * N, String("gathered a apply"))
    took(wb, b_step, 0, 0, String("gathered b step"))
    took(wb, b_apply, total, 4 * N, String("gathered b apply"))

    cross_coordinator_done(c)
    require(serve(a, a_trainer) == CROSS_DONE and serve(b, b_trainer) == CROSS_DONE, "gathered: both workers take the completion notice")
    cross_worker_end(a)
    cross_worker_end(b)
    # The storage the sessions, workers and frames read by address.
    _ = len(a_words)
    _ = len(b_words)
    _ = len(a_records)
    _ = len(b_records)
    _ = len(a_state)
    _ = len(b_state)
    _ = len(ids)
    _ = len(grads)
    _ = len(expected)


def chained(ca: Int, wa: Int, cb: Int, wb: Int) raises:
    """Two logical shards on two workers (a: 0, b: 1), the fold handed along."""
    var step = STEP_AT_OPEN
    var a_words = stand_in()
    var a_records = provider()
    var a_trainer = stand_in_session(Int(a_records.unsafe_ptr()), PARALLEL_REPLICA, 1, False)
    opened(a_trainer, Int(a_words.unsafe_ptr()))
    var b_words = stand_in()
    var b_records = provider()
    var b_trainer = stand_in_session(Int(b_records.unsafe_ptr()), PARALLEL_REPLICA, 1, False)
    opened(b_trainer, Int(b_words.unsafe_ptr()))
    var a_state = List[Float32](length=3 * N + NT, fill=Float32(0.25))
    var b_state = List[Float32](length=3 * N + NT, fill=Float32(0.25))
    var ids = List[Int32](length=2 * BATCH * WIDTH, fill=Int32(1))
    var a_shards: List[Int] = [0]
    var b_shards: List[Int] = [1]
    var a = worker(String("a"), a_shards.copy(), True, wa, Int(ids.unsafe_ptr()), 2, witness(Int(a_state.unsafe_ptr())))
    var b = worker(String("b"), b_shards.copy(), True, wb, Int(ids.unsafe_ptr()), 2, witness(Int(b_state.unsafe_ptr())))
    # What each worker's device fold exports (the stand-in writes none): a's
    # prefix, then b's total of the whole step.
    var prefix = List[Float32](length=N, fill=Float32(3.0))
    var folded = List[Float32](length=N, fill=Float32(5.0))
    var first_total = Int(prefix.unsafe_ptr())
    var total = Int(folded.unsafe_ptr())
    var total_hash = sha256_hex_addr(total, 4 * N)

    # The workers, on the frames a coordinator sends.
    var a_step = put(ca, step_frame(step, True, True), 0, 0)
    require(serve(a, a_trainer) == CROSS_CONTINUE and a.on_device, "chained: the first worker folds on its device")
    var b_step = put(cb, step_frame(step, True, False), 0, 0)
    require(serve(b, b_trainer) == CROSS_CONTINUE and b.held_count == 1 and not b.on_device, "chained: a later worker holds its shard gradient")
    for i in range(N):  # small-loop(elements of one fixture gradient)
        a.total[i] = prefix[i]
        b.total[i] = folded[i]
    var a_fold = put(ca, fold_frame(step), 0, 0)
    require(serve(a, a_trainer) == CROSS_CONTINUE and a.has_total, "chained: a exports its prefix")
    var b_fold = put(cb, fold_frame(step), first_total, 4 * N)
    require(serve(b, b_trainer) == CROSS_CONTINUE and b.has_total and b.held_count == 0, "chained: b continues the prefix")
    var a_apply = put(ca, apply_frame(step, total_hash, 2), total, 4 * N)
    require(serve(a, a_trainer) == CROSS_COMMIT, "chained: a commits the received total")
    var b_apply = put(cb, apply_frame(step, total_hash, 2), 0, 0)
    require(serve(b, b_trainer) == CROSS_COMMIT, "chained: the last worker commits the total it holds")
    committed(a, step, total_hash, 2, String("chained a"))
    committed(b, step, total_hash, 2, String("chained b"))

    # The coordinator, on the workers' queued replies.
    var c = CrossCoordinatorState()
    group(c, 2, 2, True)
    # Connected in the other order: the steps go by first shard.
    join(c, String("b"), cb, b_shards, True)
    join(c, String("a"), ca, a_shards, True)
    cross_coordinator_admit(c)
    require(c.order[0] == 1 and c.order[1] == 0, "chained: the group is ordered by first shard")
    cross_coordinator_step(c, step)
    require(c.row_total_hash == total_hash and same(Int(c.total.unsafe_ptr()), total, 4 * N),
            "chained: the coordinator's total is the last worker's fold")
    require(len(c.row_losses) == 2 and c.row_losses[0] == 2.5 and c.row_losses[1] == 2.5, "chained: one loss per logical shard")
    require(len(c.row_states) == 2 and c.row_agreed == a.commit_state and c.row_states[1] == b.commit_state,
            "chained: the row carries each worker's committed witness")

    # Every frame the coordinator sent is the frame its worker consumed.
    took(wa, a_step, 0, 0, String("chained a step"))
    took(wa, a_fold, 0, 0, String("chained a fold"))
    took(wa, a_apply, total, 4 * N, String("chained a apply"))
    took(wb, b_step, 0, 0, String("chained b step"))
    took(wb, b_fold, first_total, 4 * N, String("chained b fold"))
    took(wb, b_apply, 0, 0, String("chained b apply"))

    cross_coordinator_done(c)
    require(serve(a, a_trainer) == CROSS_DONE and serve(b, b_trainer) == CROSS_DONE, "chained: both workers take the completion notice")
    cross_worker_end(a)
    cross_worker_end(b)
    _ = len(a_words)
    _ = len(b_words)
    _ = len(a_records)
    _ = len(b_records)
    _ = len(a_state)
    _ = len(b_state)
    _ = len(ids)
    _ = len(prefix)
    _ = len(folded)


def malformed(ca: Int, wa: Int) raises:
    """The refusal words of frames that lack a field."""
    var step = STEP_AT_OPEN
    var words = stand_in()
    var records = provider()
    var trainer = stand_in_session(Int(records.unsafe_ptr()), PARALLEL_REPLICA, 1, False)
    opened(trainer, Int(words.unsafe_ptr()))
    var state = List[Float32](length=3 * N + NT, fill=Float32(0.5))
    var ids = List[Int32](length=BATCH * WIDTH, fill=Int32(1))
    var shards: List[Int] = [0]
    var w = worker(String("a"), shards.copy(), False, wa, Int(ids.unsafe_ptr()), 1, witness(Int(state.unsafe_ptr())))
    var gradient = List[Float32](length=N, fill=Float32(2.0))

    # A step command without its step: the missing key, before any work.
    var bare = TransportHeader()
    bare.put_text("cmd", "step")
    _ = put(ca, bare, 0, 0)
    var text = serve_refusal(w, trainer)
    require(text.find("'step'") >= 0 and trainer.completed == step, "a step command without its step is refused by the key's name: " + text)
    # An unknown command.
    var other = TransportHeader()
    other.put_text("cmd", "pause")
    _ = put(ca, other, 0, 0)
    text = serve_refusal(w, trainer)
    require(text.find("cross_vendor: unknown coordinator command pause") >= 0, "an unknown command is refused: " + text)
    # A refusal from the coordinator.
    var refuse = TransportHeader()
    refuse.put_text("cmd", "refuse")
    refuse.put_text("reason", "replicas disagree")
    _ = put(ca, refuse, 0, 0)
    text = serve_refusal(w, trainer)
    require(text.find("CrossVendorMismatch: coordinator refused: replicas disagree") >= 0, "a coordinator refusal carries its reason: " + text)
    # An apply whose total is not one gradient.
    var short = TransportHeader()
    short.put_text("cmd", "apply")
    short.put_int("step", step)
    _ = put(ca, short, 0, 0)
    text = serve_refusal(w, trainer)
    require(text.find("CrossVendorMismatch: summed gradient has the wrong size") >= 0 and trainer.completed == step,
            "an apply without a total is refused before the trainer: " + text)
    # An apply whose hash is not the payload's.
    var wrong = TransportHeader()
    wrong.put_text("cmd", "apply")
    wrong.put_int("step", step)
    wrong.put_text("total_sha256", "00")
    _ = put(ca, wrong, Int(gradient.unsafe_ptr()), 4 * N)
    text = serve_refusal(w, trainer)
    require(text.find("CrossVendorMismatch: total hash differs from the coordinator") >= 0 and trainer.completed == step,
            "an apply with another hash is refused before the trainer: " + text)
    # An apply without losses: the words of the committed-losses admission.
    var silent = TransportHeader()
    silent.put_text("cmd", "apply")
    silent.put_int("step", step)
    _ = put(ca, silent, Int(gradient.unsafe_ptr()), 4 * N)
    text = serve_refusal(w, trainer)
    require(text.find("committed losses requires list, tuple or range") >= 0, "an apply without losses is refused in the admission's words: " + text)

    # The coordinator: a gradient reply without losses, then a commit reply
    # without its state.
    var c = CrossCoordinatorState()
    group(c, 1, 1, False)
    join(c, String("a"), ca, shards, False)
    cross_coordinator_admit(c)
    var reply = TransportHeader()
    reply.put_int("step", step)
    reply.put_words("shards", shards)
    _ = put(wa, reply, Int(gradient.unsafe_ptr()), 4 * N)
    text = step_refusal(c, step)
    require(text.find("'losses'") >= 0, "a gradient reply without losses is refused by the key's name: " + text)
    # Drain the step request that attempt sent, then answer a whole step but
    # for the state.
    var head = List[UInt8](length=4096, fill=UInt8(0))
    _ = transport_receive_into(wa, WAIT, Int(head.unsafe_ptr()), 4096, 0, 0)
    var whole = TransportHeader()
    whole.put_int("step", step)
    whole.put_words("shards", shards)
    cross_header_put_floats(whole, "losses", List[Float64](length=1, fill=2.5))
    _ = put(wa, whole, Int(gradient.unsafe_ptr()), 4 * N)
    var done = TransportHeader()
    done.put_int("completed", step + 1)
    _ = put(wa, done, 0, 0)
    text = step_refusal(c, step)
    require(text.find("'state'") >= 0, "a commit reply without its state is refused by the key's name: " + text)
    # A reply for another step: the mismatch, told to the worker.
    _ = transport_receive_into(wa, WAIT, Int(head.unsafe_ptr()), 4096, 0, 0)
    var body = List[Float32](length=N, fill=Float32(0))
    _ = transport_receive_into(wa, WAIT, Int(head.unsafe_ptr()), 4096, Int(body.unsafe_ptr()), 4 * N)
    var late = TransportHeader()
    late.put_int("step", step + 3)
    late.put_words("shards", shards)
    cross_header_put_floats(late, "losses", List[Float64](length=1, fill=2.5))
    _ = put(wa, late, Int(gradient.unsafe_ptr()), 4 * N)
    text = step_refusal(c, step)
    require(text.find("CrossVendorMismatch: a answered the wrong step or shards") >= 0, "a reply for another step is the mismatch: " + text)
    _ = transport_receive_into(wa, WAIT, Int(head.unsafe_ptr()), 4096, 0, 0)
    text = serve_refusal(w, trainer)
    require(text.find("CrossVendorMismatch: coordinator refused: a answered the wrong step or shards") >= 0,
            "the worker is told the coordinator's refusal: " + text)
    cross_worker_end(w)
    _ = len(words)
    _ = len(records)
    _ = len(state)
    _ = len(ids)
    _ = len(gradient)
    _ = len(head)
    _ = len(body)


def fixture(unit: Int) raises:
    var one = pair()
    var two = List[Int](length=2, fill=-1)
    try:
        if unit == 2:
            malformed(one[0], one[1])
        else:
            two = pair()
            if unit == 0:
                gathered(one[0], one[1], two[0], two[1])
            else:
                chained(one[0], one[1], two[0], two[1])
    finally:
        posix_close(one[0])
        posix_close(one[1])
        posix_close(two[0])
        posix_close(two[1])


def test_loopback() raises:
    # The base image's host fold entry, published once before the fixtures (idempotent).
    publish_parallel_base_host()
    check_cases("cross_vendor_loopback", CROSS_VENDOR_LOOPBACK_FIXTURES, fixture)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
