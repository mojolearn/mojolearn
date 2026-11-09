# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Mamba-1, Mamba-2 and Mamba-3 blocks as one executor program each (lane
x5-mamba-on-executor, 2026-11-09; docs/plans/EXECUTOR_CONSOLIDATION_AND_CLEANUP.md
row X5, the API note docs/plans/EXECUTOR_X1_API.md).

`MambaBlockProgram` is one inference `OpsSession` on the device holding

- the block's weights as ONE uploaded parameter tensor (`upload_block`, the
  block struct's field order), read in place at every run;
- the carried float state as ONE declared state tensor and, for Mamba-2/3,
  the carried integers as one int32-word state tensor;
- the x it last ran on, reused while the token count is the same.

A run is ONE session op, `OPS_M1_STEP` / `OPS_M2_STEP` / `OPS_M3_STEP`
(autograd/block_step_ops.mojo): L = 1 is the decode step, L > 1 a prefill or
a chunked continuation. That op's body is the only Mamba block body: the
resident decode sessions (Mamba{1,2,3}DecodeSession), the per-call
forward/step of the public blocks and the generation engine's layer calls
all run it. The per-call bodies that uploaded nine or ten weights, the state
and x on every call (`_mamba{1,2,3}_run` and their arena twins) are deleted.

Two owners:

- a decode session owns its program (weights and state uploaded once at
  open, the caller's state arrays stale until export);
- the per-call doors (`mamba_block_call`) share a small process-wide cache of
  programs keyed by (block kind, B, d_model, weight addresses), bounded by
  their device bytes. Lane x45c: the weights are keyed by the caller's
  addresses and the process weight epoch (core/weight_epoch.mojo): the
  binding's weight door freezes every weight it hands over and moves the
  epoch on a new freeze, so equal addresses at an unchanged epoch hold the
  same bytes. A changed epoch rewrites the device copy in place once; no
  call compares weight bytes. The caller's state is the
  truth on every call: it is written into the state tensor in place before
  the run and read back after it, so the caller's arrays always hold the
  post-call state (DEVIATION 792).

Same kernels, same launch order and same bytes as the resident session
bodies the op lifted. No arithmetic here.
"""
from std.ffi import _Global
from std.atomic import Atomic, Ordering
from core.weight_epoch import weight_epoch
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.cast_in import CastIn
from core.decode_slab import decode_slab_check, decode_slab_put, decode_slab_take
from core.hostptr import f32_ptr
from core.neural_context import neural_ctx
from sequence.ops import FP
from sequence.exec_device import DeviceExec
from autograd.ops_api import OpsSession
from autograd.block_step_ops import (
    OPS_M1_STEP, OPS_M2_STEP, OPS_M3_STEP, BLOCK_PART_WEIGHTS, BLOCK_PART_STATE, BLOCK_PART_WORDS,
    block_weight_sizes, block_state_sizes, block_int_words, block_step_words, block_step_reports_to_host,
    block_buffer_view, block_part_floats,
)

# The process-lifetime DeviceContext of the Mamba bindings (the name
# mamba/impl/mamba_shell_impl.mojo has used since the neural lane).
comptime _NEURAL_CTX = "MojolearnNeuralMambaContextIdenticalV2" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojolearnNeuralMambaContextFastV2"


def mamba_step_code(kind: Int) raises -> Int:
    """The session op of Mamba-`kind` (1, 2 or 3)."""
    if kind == 1:
        return OPS_M1_STEP
    if kind == 2:
        return OPS_M2_STEP
    if kind == 3:
        return OPS_M3_STEP
    raise Error("mamba: unknown block kind " + String(kind))


def mamba_report_count(code: Int) -> Int:
    """Report arrays a run writes: Mamba-2 h_last; Mamba-3 h_last, k_last,
    v_last, theta_last; Mamba-1 none."""
    if code == OPS_M2_STEP:
        return 1
    if code == OPS_M3_STEP:
        return 4
    return 0


def _pieces(addrs: List[Int], sizes: List[Int], what: String, allow_null: Bool) raises -> List[CastIn]:
    """One float32 C-order source per piece; a 0 address (when allowed) is an
    empty source, which `load_block` leaves as it is."""
    if len(addrs) != len(sizes):
        raise Error(what + ": expected " + String(len(sizes)) + " arrays, got " + String(len(addrs)))
    var out = List[CastIn](capacity=len(sizes))
    for i in range(len(sizes)):  # driver-loop: over the block's weight or state pieces (model structure)
        if addrs[i] == 0:
            if not allow_null:
                raise Error(what + ": null buffer address at piece " + String(i))
            out.append(CastIn.f32_c(0, 0, 1))
        else:
            out.append(CastIn.f32_c(addrs[i], sizes[i], 1))
    return out^


struct MambaBlockProgram(Movable):
    """One Mamba block on the executor: weights parameter, state, step op."""
    var s: OpsSession[DeviceExec]
    var code: Int
    var b: Int
    var dm: Int
    var ints: List[Int]
    var fs: List[Float32]
    var w: Int
    var st: Int
    var q: Int
    var x: Int
    var x_n: Int
    #: the host weight addresses the parameter was loaded from (cache key)
    var w_addrs: List[Int]
    #: the weight epoch at the last weight load (lane x45c: the version key)
    var epoch: Int
    #: the device floats this program holds (the cache's byte bound)
    var floats: Int
    var busy: Bool
    var stamp: Int

    def __init__(out self, code: Int, b: Int, dm: Int, fs: List[Float32], weights: List[Int]) raises:
        """A program holding `weights` (one host float32 array per piece,
        `block_weight_sizes` order) and a zero state for `b` rows."""
        if b < 1 or dm < 1:
            raise Error("mamba: B and d_model must be positive")
        var epoch = weight_epoch()
        var ex = DeviceExec(neural_ctx[_NEURAL_CTX]())
        var s = OpsSession[DeviceExec](ex^, False)
        var ints: List[Int] = [b, dm]
        var wsizes = block_weight_sizes(code, ints)
        var w = s.upload_block(code, ints, BLOCK_PART_WEIGHTS, _pieces(weights, wsizes, "mamba weights", False),
                               List[Int]())
        # The certified zero state (allocate_inference_cache's semantics).
        var ssizes = block_state_sizes(code, ints)
        var most = 1
        for i in range(len(ssizes)):  # driver-loop: over the block's state pieces (model structure)
            most = max(most, ssizes[i])
        var zero = List[Float32](length=most, fill=Float32(0))
        var zsrc = List[CastIn](capacity=len(ssizes))
        for i in range(len(ssizes)):  # driver-loop: over the block's state pieces (model structure)
            zsrc.append(CastIn.f32_c(Int(zero.unsafe_ptr()), ssizes[i], 1))
        var st = s.upload_block(code, ints, BLOCK_PART_STATE, zsrc, List[Int]())
        _ = len(zero)  # keep-alive past the staging copies
        var q = 0
        var k = block_int_words(code)
        if k > 0:
            q = s.upload_block(code, ints, BLOCK_PART_WORDS, List[CastIn](), List[Int](length=k, fill=0))
        self.epoch = epoch
        self.floats = (block_part_floats(code, ints, BLOCK_PART_WEIGHTS)
                       + block_part_floats(code, ints, BLOCK_PART_STATE) + k)
        self.s = s^
        self.code = code
        self.b = b
        self.dm = dm
        self.ints = ints^
        self.fs = fs.copy()
        self.w = w
        self.st = st
        self.q = q
        self.x = 0
        self.x_n = 0
        self.w_addrs = weights.copy()
        self.busy = False
        self.stamp = 0

    # ------------------------------------------------------------ weights
    def refresh_weights(mut self, weights: List[Int]) raises:
        """The per-call rule (lane x45c): the same addresses at the same
        weight epoch keep the parameter (no pass over the weights);
        otherwise the caller's arrays are written into it in place and the
        block's non-finite refusal re-arms."""
        var wsizes = block_weight_sizes(self.code, self.ints)
        if len(weights) != len(wsizes):
            raise Error("mamba weights: expected " + String(len(wsizes)) + " arrays, got " + String(len(weights)))
        var now = weight_epoch()
        if now == self.epoch and weights == self.w_addrs:
            return
        self.s.load_block(self.w, self.code, self.ints, BLOCK_PART_WEIGHTS,
                          _pieces(weights, wsizes, "mamba weights", False), List[Int]())
        self.epoch = now
        self.w_addrs = weights.copy()

    def export_weights(mut self, dsts: List[Int]) raises:
        """The parameter into the caller's arrays (piece order), one wait."""
        self.s.download_block(self.w, self.code, self.ints, BLOCK_PART_WEIGHTS, dsts)

    # ------------------------------------------------------------ state
    def load_state(mut self, pieces: List[Int], words: List[Int]) raises:
        """The caller's state into the state tensors, in place (a 0 address
        keeps that piece): the carried integers are checked by the step op's
        own rules before anything is written."""
        var k = block_int_words(self.code)
        if len(words) != k:
            raise Error("mamba state: this block carries " + String(k) + " integers, got " + String(len(words)))
        if k > 0:
            self.s.load_block(self.q, self.code, self.ints, BLOCK_PART_WORDS, List[CastIn](), words)
        var ssizes = block_state_sizes(self.code, self.ints)
        self.s.load_block(self.st, self.code, self.ints, BLOCK_PART_STATE,
                          _pieces(pieces, ssizes, "mamba state", True), List[Int]())

    def zero_state(mut self) raises:
        """A fresh sequence: the state tensor zeroed on the device (one fill)
        and the carried integers zero."""
        var n = block_part_floats(self.code, self.ints, BLOCK_PART_STATE)
        var view = block_buffer_view(self.s.ex, self.s.buffer(self.st), n)
        view.enqueue_fill(Float32(0))
        var k = block_int_words(self.code)
        if k > 0:
            self.s.load_block(self.q, self.code, self.ints, BLOCK_PART_WORDS, List[CastIn](),
                              List[Int](length=k, fill=0))

    def export_state(mut self, pieces: List[Int]) raises -> List[Int]:
        """The state into the caller's arrays (a 0 address skips that piece)
        behind one wait; returns the carried integers."""
        self.s.download_block(self.st, self.code, self.ints, BLOCK_PART_STATE, pieces)
        return self.words()

    def words(self) raises -> List[Int]:
        """The carried integers after the last run (host-side, no transfer):
        Mamba-2 [buf_len], Mamba-3 [buf_len, pending]; before any run, the
        ones last loaded."""
        if self.code == OPS_M1_STEP:
            return List[Int]()
        try:
            return block_step_words(self.s.steps, self.code, Int(self.s.buffer(self.w)), Int(self.s.buffer(self.st)))
        except:
            pass
        # No resident entry yet (no run since the last write): read the words.
        var k = block_int_words(self.code)
        var host = List[Int32](length=k, fill=0)
        var view = block_buffer_view(self.s.ex, self.s.buffer(self.q), k)
        self.s.ex.ctx.enqueue_copy(dst_ptr=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(host.unsafe_ptr())),
                                   src_buf=view)
        self.s.ex.ctx.synchronize()
        var out = List[Int](capacity=k)
        for i in range(k):  # small-loop(k: at most two state words): word decode, no data
            out.append(Int(host[i]))
        return out^

    # ------------------------------------------------------------ runs
    def _x_for(mut self, px: Int, n: Int) raises -> Int:
        """The x tensor of n floats holding the caller's x (kept while n is
        the same: one in-place write per call)."""
        var src = CastIn.f32_c(px, 1, n)
        if self.x != 0 and self.x_n == n:
            self.s.load_into(self.x, src)
        else:
            if self.x != 0:
                self.s.free(self.x)
                self.x = 0
            self.x = self.s.upload(src, False)
            self.x_n = n
        return self.x

    def _hs(self, x: Int) -> List[Int]:
        var hs: List[Int] = [x, self.w, self.st]
        if self.q != 0:
            hs.append(self.q)
        return hs^

    def run(mut self, px: Int, py: Int, l: Int, reports: List[Int]) raises:
        """L tokens per row (L = 1 a decode step): x (B, L, d_model) at px,
        the block output into py, the reports (`mamba_report_count`, 0
        addresses skipped) into their arrays. Enqueued; `finish` waits."""
        if l < 1:
            raise Error("mamba: L must be positive")
        if px == 0 or py == 0:
            raise Error("mamba: null x or y address")
        var n = self.b * l * self.dm
        var x = self._x_for(px, n)
        var hs = self._hs(x)
        var ints = self.ints.copy()
        var fs = self.fs.copy()
        var y = self.s.op(self.code, hs, ints, fs)
        var yt = self.s.tensor(y)
        var yp = self.s.tape.value_of(self.s.ex, yt)
        self.s.ex.download_async(FP(unsafe_from_address=py), yp, n)
        self._reports(reports)

    def _reports(mut self, reports: List[Int]) raises:
        var k = mamba_report_count(self.code)
        if k == 0 or len(reports) == 0:
            return
        if len(reports) != k:
            raise Error("mamba: expected " + String(k) + " report addresses")
        var ctx = self.s.ex.ctx.copy()
        var w_addr = Int(self.s.buffer(self.w))
        var s_addr = Int(self.s.buffer(self.st))
        block_step_reports_to_host(ctx, self.s.steps, self.code, w_addr, s_addr, reports)

    def finish(mut self) raises:
        """Wait for the queued run and its copies, then drop the run's
        computed tensors (the uploads, the state and the resident structs
        stay)."""
        self.s.ex.sync()
        self.s.steps.release_parked()
        self.s.reset()

    def finish_with_state(mut self, pieces: List[Int]) raises:
        """`finish`, with the post-run state copied into the caller's arrays
        first (the one wait covers the run, y, the reports and the state)."""
        var ints = self.ints.copy()
        self.s.download_block(self.st, self.code, ints, BLOCK_PART_STATE, pieces)
        self.s.reset()

    def step(mut self, px: Int, py: Int, reports: List[Int]) raises -> List[Int]:
        """One decode token, one wait; returns the carried integers."""
        self.run(px, py, 1, reports)
        self.finish()
        return self.words()

    def decode(mut self, px: Int, py: Int, steps: Int, reports: List[Int]) raises -> List[Int]:
        """N decode tokens (decode plan D2): x (B, N, d_model) uploaded once,
        the step op once per token with token t moved on the device, y
        (B, N, d_model) down in one copy with the last token's reports, behind
        one wait. Same kernels, order and bytes as N `step` calls."""
        decode_slab_check(self.b, steps, self.dm, "mamba session decode")
        var row = self.b * self.dm
        var total = row * steps
        var xs = self.s.ex.alloc(total)
        var ys = self.s.ex.alloc(total)
        try:
            self.s.ex.upload(xs, FP(unsafe_from_address=px), total)
            var xt = self._x_for(px, row)
            var ctx = self.s.ex.ctx.copy()
            var xs_view = block_buffer_view(self.s.ex, xs, total)
            var ys_view = block_buffer_view(self.s.ex, ys, total)
            var xt_view = block_buffer_view(self.s.ex, self.s.buffer(xt), row)
            var hs = self._hs(xt)
            var ints = self.ints.copy()
            var fs = self.fs.copy()
            for t in range(steps):  # driver-loop: over autoregressive decode steps (each reads the state the previous step left), the one step op each
                decode_slab_take(ctx, xt_view, xs_view, self.b, steps, self.dm, t)
                var y = self.s.op(self.code, hs, ints, fs)
                var yt = self.s.tensor(y)
                var yp = self.s.tape.value_of(self.s.ex, yt)
                var y_view = block_buffer_view(self.s.ex, yp, row)
                decode_slab_put(ctx, ys_view, y_view, self.b, steps, self.dm, t)
            self.s.ex.download_async(FP(unsafe_from_address=py), ys, total)
            self._reports(reports)
            self.finish()
        finally:
            self.s.ex.free(xs)
            self.s.ex.free(ys)
        return self.words()

    def close(mut self) raises:
        self.s.close()


# TOMBSTONE: _host_copy / _same_bits (REFACTORED: the per-call memcmp of the caller's weights against a host copy of every cached program) deleted 2026-11-09 by lane/x45c-causal-generate-program-decode; code recoverable at 5b5490960d.
# The programs key their weights by address and the process weight epoch (core/weight_epoch.mojo).
# Restore: git apply experiments/removed/WEIGHT_BYTE_COMPARE.patch; record in docs/TOMBSTONES.md.


# ---------------------------------------------------------------- the per-call cache
# One program per live block (kind, B, d_model, weight addresses): the per-call
# doors stop uploading the weights on every call. Lane x45c: the cache is
# bounded by the device bytes its idle programs hold (weights, state and
# words), not by a count: past MAMBA_PROGRAM_CACHE_BYTES the least recently
# used idle programs go (a program larger than the bound alone runs as a
# temporary). 1 GiB is a small fraction of any supported GPU's memory (the
# smallest, Apple's unified pool, starts at 8 GiB) and holds every layer of
# a deep stack at a decode batch; it comes from device memory, not a shape.
comptime MAMBA_PROGRAM_CACHE_BYTES = 1 << 30


struct MambaProgramCache(Movable):
    var programs: List[Int]
    var tick: Int
    # One lock word: the doors run with the GIL released.
    var lock: List[Int64]

    def __init__(out self):
        self.programs = List[Int]()
        self.tick = 0
        self.lock = List[Int64](length=1, fill=Int64(0))


comptime MAMBA_PROGRAM_CACHE = _Global[
    StorageType=MambaProgramCache,
    name=("MojoMambaProgramCacheIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoMambaProgramCacheFast"),
    init_fn=MambaProgramCache.__init__,
]

comptime _ProgramPtr = UnsafePointer[MambaBlockProgram, MutUntrackedOrigin]


def _cache() raises -> UnsafePointer[MambaProgramCache, MutAnyOrigin]:
    return MAMBA_PROGRAM_CACHE.get_or_create_ptr().unsafe_origin_cast[MutAnyOrigin]()


def _lock(c: UnsafePointer[MambaProgramCache, MutAnyOrigin]):
    var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](c[].lock.unsafe_ptr())
    while True:  # driver-loop: spin on the cache's lock word (held for one list step only)
        var expected = Int64(0)
        if Atomic.compare_exchange[
            success_ordering=Ordering.SEQUENTIAL,
            failure_ordering=Ordering.RELAXED,
        ](word, expected, Int64(1)):
            return


def _unlock(c: UnsafePointer[MambaProgramCache, MutAnyOrigin]):
    var word = rebind[MutPointer[Int64, MutUntrackedOrigin]](c[].lock.unsafe_ptr())
    Atomic.store[ordering=Ordering.RELEASE](word, Int64(0))


def _destroy(p: _ProgramPtr):
    try:
        p[].close()
    except:
        pass
    p.destroy_pointee()
    p.free()


def _claim(code: Int, b: Int, dm: Int, fs: List[Float32], weights: List[Int]) raises -> Tuple[_ProgramPtr, Bool]:
    """An idle cached program for this block (marked busy), or a new one:
    cached while the cache's device bytes stay within
    MAMBA_PROGRAM_CACHE_BYTES (least recently used idle programs go to make
    room), else a temporary (second flag False) the caller destroys. A
    program matches by its weight addresses; its weights are reloaded by
    `refresh_weights` when the weight epoch moved (lane x45c: the byte-match
    fallback for moved arrays is gone with the byte compare)."""
    var c = _cache()
    _lock(c)
    try:
        c[].tick += 1
        for i in range(len(c[].programs)):  # driver-loop: over cached Mamba programs
            var p = _ProgramPtr(unsafe_from_address=c[].programs[i])
            if not p[].busy and p[].code == code and p[].b == b and p[].dm == dm and p[].w_addrs == weights:
                p[].busy = True
                p[].stamp = c[].tick
                p[].fs = fs.copy()
                _unlock(c)
                return (p, True)
    except error:
        _unlock(c)
        raise error
    _unlock(c)
    var p = alloc[MambaBlockProgram](1)
    try:
        p.init_pointee_move(MambaBlockProgram(code, b, dm, fs, weights))
    except error:
        p.free()
        raise error
    var owned = _ProgramPtr(unsafe_from_address=Int(p))
    owned[].busy = True
    var need = 4 * owned[].floats
    if need > MAMBA_PROGRAM_CACHE_BYTES:
        return (owned, False)
    var evicted = List[Int]()
    _lock(c)
    c[].tick += 1
    owned[].stamp = c[].tick
    var held = 0
    for i in range(len(c[].programs)):  # driver-loop: over cached Mamba programs (their device bytes)
        held += 4 * _ProgramPtr(unsafe_from_address=c[].programs[i])[].floats
    while held + need > MAMBA_PROGRAM_CACHE_BYTES:  # driver-loop: evict least recently used idle programs until the new one fits
        var victim = -1
        for i in range(len(c[].programs)):  # driver-loop: over cached Mamba programs
            var q = _ProgramPtr(unsafe_from_address=c[].programs[i])
            if q[].busy:
                continue
            if victim < 0 or q[].stamp < _ProgramPtr(unsafe_from_address=c[].programs[victim])[].stamp:
                victim = i
        if victim < 0:
            break
        var gone = c[].programs.pop(victim)
        held -= 4 * _ProgramPtr(unsafe_from_address=gone)[].floats
        evicted.append(gone)
    var cached = held + need <= MAMBA_PROGRAM_CACHE_BYTES
    if cached:
        c[].programs.append(Int(p))
    _unlock(c)
    for i in range(len(evicted)):  # driver-loop: over evicted Mamba programs
        _destroy(_ProgramPtr(unsafe_from_address=evicted[i]))
    return (owned, cached)


def _release(p: _ProgramPtr, cached: Bool, failed: Bool):
    """Back to the cache idle; a temporary, or a program whose run failed
    (its state is whatever the failure left), is destroyed."""
    if not cached:
        _destroy(p)
        return
    try:
        var c = _cache()
        _lock(c)
        if failed:
            for i in range(len(c[].programs)):  # driver-loop: over cached Mamba programs (at most the cap)
                if c[].programs[i] == Int(p):
                    _ = c[].programs.pop(i)
                    break
        else:
            p[].busy = False
        _unlock(c)
        if failed:
            _destroy(p)
    except:
        pass


def mamba_block_call(
    code: Int, weights: List[Int], state: List[Int], words: List[Int], px: Int, py: Int, reports: List[Int],
    b: Int, l: Int, dm: Int, fs: List[Float32], fresh: Bool,
) raises -> List[Int]:
    """One per-call block call on host arrays (the public forward/step door
    and the generation engine's layer call): the caller's weights (refreshed
    on the cached program when their bytes changed), its state written in
    (or a zero state when `fresh`), L tokens through the block's step op,
    y and the reports out, the post-call state written back into the
    caller's arrays (skipped when `fresh`: the discard-state prefill), one
    wait. Returns the post-call carried integers."""
    var claimed = _claim(code, b, dm, fs, weights)
    var p = claimed[0]
    var cached = claimed[1]
    var out = List[Int]()
    try:
        p[].refresh_weights(weights)
        if fresh:
            p[].zero_state()
        else:
            p[].load_state(state, words)
        p[].run(px, py, l, List[Int]())
        if mamba_report_count(code) > 0:
            # The reports of this run, queued on the program's device context
            # behind y (the wait below covers them).
            var ctx = p[].s.ex.ctx.copy()
            block_step_reports_to_host(ctx, p[].s.steps, code, Int(p[].s.buffer(p[].w)),
                                       Int(p[].s.buffer(p[].st)), reports)
        if fresh:
            p[].finish()
        else:
            p[].finish_with_state(state)
        out = p[].words()
    except error:
        _release(p, cached, True)
        raise error
    _release(p, cached, False)
    return out^


def mamba_block_cache_clear() raises:
    """Drop every idle cached program (their device memory goes)."""
    var c = _cache()
    var drop = List[Int]()
    _lock(c)
    var keep = List[Int]()
    for i in range(len(c[].programs)):  # driver-loop: over cached Mamba programs (at most the cap)
        var p = _ProgramPtr(unsafe_from_address=c[].programs[i])
        if p[].busy:
            keep.append(c[].programs[i])
        else:
            drop.append(c[].programs[i])
    c[].programs = keep^
    _unlock(c)
    for i in range(len(drop)):  # driver-loop: over dropped Mamba programs
        _destroy(_ProgramPtr(unsafe_from_address=drop[i]))


def mamba_block_call_abi(kind: Int, a: List[Int], b: Int, l: Int, dm: Int, q0: Int, pend: Int,
                         dt_lo: Float32, dt_hi: Float32, fresh: Bool = False) raises -> Int:
    """`mamba_block_call` over the per-call ABI's one address list (the
    order every `mamba{1,2,3}_forward` / `_decode_step` docstring gives):

    - Mamba-1, 14: x, ten weights, conv_window, h, y_out;
    - Mamba-2, 16: x, nine weights, conv_window, h, buffer_xbc, buffer_dtraw,
      y_out, h_last_out;
    - Mamba-3, 25: x, nine weights, the ten state pieces, y_out, h_last_out,
      k_last_out, v_last_out, theta_last_out.

    Returns the post-call buf_len (Mamba-2/3; 0 for Mamba-1). `fresh` runs
    from a zero state and leaves the state slots unread and unwritten (the
    discard-state prefill)."""
    var code = mamba_step_code(kind)
    var nw = 10 if kind == 1 else 9
    var ns = 2 if kind == 1 else (4 if kind == 2 else 10)
    var nr = mamba_report_count(code)
    if len(a) != 1 + nw + ns + 1 + nr:
        raise Error("mamba" + String(kind) + ": expected " + String(1 + nw + ns + 1 + nr) + " addresses, got "
                    + String(len(a)))
    var weights = List[Int](capacity=nw)
    for i in range(nw):  # small-loop(nw: the block's weight addresses): address list, no data
        weights.append(a[1 + i])
    var state = List[Int](capacity=ns)
    for i in range(ns):  # small-loop(ns: the block's state addresses): address list, no data
        state.append(a[1 + nw + i])
    var y = a[1 + nw + ns]
    var reports = List[Int](capacity=nr)
    for i in range(nr):  # small-loop(nr: the block's report addresses): address list, no data
        reports.append(a[2 + nw + ns + i])
    var words = List[Int]()
    var fs = List[Float32]()
    if kind == 2:
        words.append(q0)
        fs.append(dt_lo)
        fs.append(dt_hi)
    elif kind == 3:
        words.append(q0)
        words.append(pend)
    var out = mamba_block_call(code, weights, state, words, a[0], y, reports, b, l, dm, fs, fresh)
    return out[0] if len(out) > 0 else 0
