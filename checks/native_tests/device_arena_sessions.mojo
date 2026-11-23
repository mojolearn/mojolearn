# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""Concurrent sessions, each with its own device arena (core/device_arena.mojo; lane arena-session, 2026-11-23).

The arena was ONE process-wide "active session": every allocation helper consulted it on every call, so two
sessions constructing at once (or one session's step allocating while another constructed) took views of each
other's chunks and released each other's chunks. A session now owns a `DeviceArena` and hands its helpers a pointer
to it (`arena_ptr`); only the chunk pool is shared, under its lock.

DEVICE_ARENA_ON is False on the CPU column, so these checks switch their arenas on by hand (`DeviceArena(on=True)`)
and run the same `take` / `arena_buffer` / `release` code the Apple column runs; the chunks are CPU-device buffers.

    test_sessions_concurrent   ARENA_SESSIONS sessions at once, ARENA_ROUNDS rounds (chunks returned and reused
                               between rounds), each session's words equal to the same session run alone first
    test_sessions_interleaved  two arenas alive at once on one context, views taken in alternation: each arena
                               holds its own chunk and each session's words equal its words alone
    test_arena_off             an arena that is off gives no handle; `arena_buffer` then allocates a fresh buffer
"""
from std.memory import bitcast
from std.testing import TestSuite
from max.gpu.host import DeviceBuffer, DeviceContext
from core.device_arena import DeviceArena, arena_buffer, arena_ptr
from core.host_parallel import host_parallelize
from checks.native_tests.testsuite_cases import require

comptime ARENA_SESSIONS = 2
comptime ARENA_ROUNDS = 8
comptime _VIEWS = 48


def _size(unit: Int, j: Int) -> Int:
    """View `j` of session `unit`: 1 to 4000 floats, not a multiple of the 64-float alignment for most views."""
    return 1 + ((unit * 131 + j * 977) % 4000)


def _word(unit: Int, j: Int, i: Int) -> Float32:
    return Float32(unit * 100000 + j * 1000 + i % 1000) * Float32(0.25)


def _fill(ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], unit: Int, j: Int, n: Int) raises:
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    for i in range(n):  # small-loop(fixture: at most 4000 words of one view)
        host.unsafe_ptr().unsafe_store(i, _word(unit, j, i))
    ctx.enqueue_copy(dst_buf=buf, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    _ = host^


def _read(ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int, mut words: List[Float32]) raises:
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=buf)
    ctx.synchronize()
    for i in range(n):  # small-loop(fixture: at most 4000 words of one view)
        words.append(host.unsafe_ptr().unsafe_load(i))
    _ = host^


def _session(ctx: DeviceContext, unit: Int) raises -> List[Float32]:
    """One session: its own arena, `_VIEWS` buffers through `arena_buffer` (the helpers' path), each filled with
    this session's words, then every buffer read back in order once all are written."""
    var arena = DeviceArena(on=True)
    var ap = arena_ptr(arena)
    require(True if ap else False, "an arena that is on gives a handle")
    var views = List[DeviceBuffer[DType.float32]]()
    for j in range(_VIEWS):  # small-loop(fixture: the session's views)
        var n = _size(unit, j)
        var buf = arena_buffer(ctx, n, ap)
        _fill(ctx, buf, unit, j, n)
        views.append(buf^)
    var words = List[Float32]()
    for j in range(_VIEWS):  # small-loop(fixture: the session's views)
        _read(ctx, views[j], _size(unit, j), words)
    var stats = arena.stats()
    require(stats[1] == _VIEWS, "session " + String(unit) + ": " + String(stats[1]) + " views taken, want "
        + String(_VIEWS))
    require(stats[0] >= 1, "session " + String(unit) + " holds no chunk")
    _ = views^
    arena.release()
    _ = arena^
    return words^


def _same(got: List[Float32], want: List[Float32], label: String) raises:
    if len(got) != len(want):
        raise Error(label + ": " + String(len(got)) + " words, alone " + String(len(want)))
    for i in range(len(got)):  # small-loop(fixture: one session's words)
        if bitcast[DType.uint32](got[i]) != bitcast[DType.uint32](want[i]):
            raise Error(label + " word " + String(i) + ": " + String(got[i]) + " alone " + String(want[i]))


def test_sessions_concurrent() raises:
    """Each session on its own context (as each neural binding runs its own), all at once, every round."""
    var ctxs = List[DeviceContext]()
    for _ in range(ARENA_SESSIONS):  # small-loop(fixture: one context per session)
        ctxs.append(DeviceContext())
    var alone = List[List[Float32]]()
    for unit in range(ARENA_SESSIONS):  # driver-loop: the reference sessions, one after another
        alone.append(_session(ctxs[unit], unit))
    var failed = List[Int](length=ARENA_SESSIONS, fill=0)
    var messages = List[String](length=ARENA_SESSIONS, fill=String(""))
    var fp = failed.unsafe_ptr()
    var mp = messages.unsafe_ptr()
    var ap = alone.unsafe_ptr()
    var cp = ctxs.unsafe_ptr()
    for rnd in range(ARENA_ROUNDS):  # driver-loop: rounds; each round's sessions run at once
        def task(unit: Int) {imm fp, imm mp, imm ap, imm cp, imm rnd}:
            try:
                var got = _session(cp[unsafe_offset=unit], unit)
                _same(got, ap[unsafe_offset=unit], "round " + String(rnd) + " session " + String(unit))
            except problem:
                mp[unsafe_offset=unit] = String(problem)
                fp.unsafe_store(unit, 1)
        host_parallelize(task, ARENA_SESSIONS)
    var count = 0
    var first = String("")
    for i in range(ARENA_SESSIONS):  # small-loop(the tasks' failure words)
        if failed[i] != 0:
            if count == 0:
                first = messages[i]
            count += 1
    _ = messages^
    _ = failed^
    _ = alone^
    _ = ctxs^
    require(count == 0, String(count) + " of " + String(ARENA_SESSIONS)
        + " concurrent sessions read back other words; first: " + first)


def test_sessions_interleaved() raises:
    """Two arenas alive at once on ONE context, their views taken in alternation (through the one process-wide
    active arena, B's open superseded A's and both took views of one chunk)."""
    var ctx = DeviceContext()
    var want_a = _session(ctx, 0)
    var want_b = _session(ctx, 1)
    var a = DeviceArena(on=True)
    var b = DeviceArena(on=True)
    var pa = arena_ptr(a)
    var pb = arena_ptr(b)
    var va = List[DeviceBuffer[DType.float32]]()
    var vb = List[DeviceBuffer[DType.float32]]()
    for j in range(_VIEWS):  # small-loop(fixture: the sessions' views, alternated)
        var na = _size(0, j)
        var ba = arena_buffer(ctx, na, pa)
        var nb = _size(1, j)
        var bb = arena_buffer(ctx, nb, pb)
        _fill(ctx, ba, 0, j, na)
        _fill(ctx, bb, 1, j, nb)
        va.append(ba^)
        vb.append(bb^)
    require(len(a.held) >= 1 and len(b.held) >= 1, "each arena holds a chunk")
    for i in range(len(a.held)):  # small-loop(fixture: a handful of chunk indices)
        for k in range(len(b.held)):  # small-loop(fixture: a handful of chunk indices)
            require(a.held[i] != b.held[k], "two live arenas hold the same pool chunk " + String(a.held[i]))
    var got_a = List[Float32]()
    var got_b = List[Float32]()
    for j in range(_VIEWS):  # small-loop(fixture: the sessions' views)
        _read(ctx, va[j], _size(0, j), got_a)
        _read(ctx, vb[j], _size(1, j), got_b)
    _same(got_a, want_a, "interleaved session A")
    _same(got_b, want_b, "interleaved session B")
    _ = va^
    _ = vb^
    a.release()
    b.release()
    _ = a^
    _ = b^
    _ = ctx^


def test_arena_off() raises:
    var ctx = DeviceContext()
    var off = DeviceArena(on=False)
    var p = arena_ptr(off)
    require(False if p else True, "an arena that is off gives no handle")
    var buf = arena_buffer(ctx, 100, p)
    require(len(buf) == 100, "a fresh buffer of the asked length when no arena is given")
    var refused = False
    try:
        _ = off.take(ctx, 10)
    except:
        refused = True
    require(refused, "an arena that is off refuses a take")
    require(len(off.held) == 0, "an arena that is off holds no chunk")
    _ = buf^
    _ = off^
    _ = ctx^


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
