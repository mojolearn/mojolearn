# SPDX-License-Identifier: Apache-2.0
"""GENERATION OPS of the operation session (lane x45c-causal-generate-program-decode,
2026-11-09; plan docs/plans/EXECUTOR_CONSOLIDATION_AND_CLEANUP.md row X4,
API note docs/plans/EXECUTOR_X1_API.md section 4).

Two session operation codes make a language model's token pick a node of a
captured decode body, so `CausalLM.generate` is ONE `Program.decode` over ONE
session (transformer/causal_lm_program.mojo):

  OPS_LM_PICK  85  hs [logits, rows]   ints [b, vocab, pad, mode, top_k, seed_lo, seed_hi]
                                        floats [temperature, top_p]
  OPS_LM_ALIVE 86  hs [ids, rows]      ints [b]

`logits` holds b * L * vocab values; the pick reads each row's LAST
position. `rows` is ONE int32-word tensor declared state (`upload_ids` +
`declare_state`) of `lm_rows_words(b, n_eos)` words, written in place:

  [0, b)          row status: 0 active, 1 finished (an eos id was picked)
  [b, 2b)         row lengths (picks made while active)
  [2b, 3b)        the pick's per-row first non-finite token, -1 none (scratch)
  3b              alive: 1 while some row is active and no row failed
  3b + 1          the first failing row's first non-finite token, -1 none
  3b + 2          the output position of the next pick (the sampler's counter)
  3b + 3          n_eos
  [3b + 4, ...)   the eos ids

The pick's result is a (b, 1) tensor of int32 ids (`pad` for a finished row).
The rule is the documented one of `core/generation_policy_impl.mojo` and the
resident read it replaces (transformer_binding_impl `_lm_read_argmax_kernel`,
`_lm_read_sample_kernel`, `_lm_read_status_kernel`, deleted by this lane):
greedy = the lowest-index maximum by the float-order key (no float compare,
-0.0 keyed as +0.0); any non-finite logit of an active row refuses that row;
sampling = `sampling_pick_f32` (the one FP32 policy) at (seed, row,
position); then, in row order, rows before the first failing row commit
(length + 1, an eos id finishes the row) and the alive word drops to 0 when
no row stays active or a row failed. ALIVE returns the alive word as a (1, 1)
float: the decode loop's condition, so the loop stops on the device and a
chunk of iterations needs no host read in between.

Device only: a host session refuses these codes. Not differentiable.
"""
from std.memory import bitcast, stack_allocation
from std.gpu import block_idx, thread_idx
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from sequence.ops import FP
from sequence.exec_device import DeviceExec
from core.generation_sampling_f32 import sampling_pick_f32
from autograd.block_step_ops import block_buffer_view

comptime OPS_LM_PICK = 85
comptime OPS_LM_ALIVE = 86
comptime OPS_GEN_FIRST = OPS_LM_PICK
comptime OPS_GEN_LAST = OPS_LM_ALIVE

comptime LM_PICK_INTS = 7
comptime LM_PICK_GREEDY = 0
comptime LM_PICK_SAMPLE = 1

comptime _TPB = 256
comptime _NONE: Int32 = 2147483647


def is_generation_op(code: Int) -> Bool:
    return code >= OPS_GEN_FIRST and code <= OPS_GEN_LAST


def lm_rows_words(b: Int, n_eos: Int) -> Int:
    """Words of the rows state tensor for b rows and n_eos eos ids."""
    return 3 * b + 4 + n_eos


def generation_op_name(code: Int) -> StaticString:
    if code == OPS_LM_PICK:
        return "lm_pick"
    return "lm_alive"


@always_inline
def _order_key(value: Float32) -> UInt32:
    """An unsigned key whose order is the float order (by bits, never a float
    compare: Metal flushes compare operands); -0.0 keys as +0.0."""
    var bits = bitcast[DType.uint32](value)
    if (bits & UInt32(0x7FFFFFFF)) == UInt32(0):
        bits = UInt32(0)
    if (bits & UInt32(0x80000000)) != UInt32(0):
        return ~bits
    return bits | UInt32(0x80000000)


@always_inline
def _w(p: MutPointer[Float32, MutAnyOrigin], i: Int) -> Int32:
    return bitcast[DType.int32](p[i])


@always_inline
def _put(p: MutPointer[Float32, MutAnyOrigin], i: Int, v: Int32):
    p[i] = bitcast[DType.float32](v)


def _lm_greedy_kernel(
    logits: MutPointer[Float32, MutAnyOrigin], rows: MutPointer[Float32, MutAnyOrigin],
    ids: MutPointer[Float32, MutAnyOrigin], batch_in: Int32, cur_l_in: Int32, vocab_in: Int32, pad: Int32,
):
    # One block per row: the lowest-index maximum and the first non-finite
    # token of the row's last position (an integer (key, index) reduction:
    # the same pick on every vendor and at any block size).
    var r = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var batch = Int(batch_in)
    var cur_l = Int(cur_l_in)
    var vocab = Int(vocab_in)
    if _w(rows, r) != 0:
        if tid == 0:
            _put(ids, r, pad)
            _put(rows, 2 * batch + r, Int32(-1))
        return
    var keys = stack_allocation[_TPB, UInt32, address_space=AddressSpace.SHARED]()
    var indices = stack_allocation[_TPB, Int32, address_space=AddressSpace.SHARED]()
    var bad = stack_allocation[_TPB, Int32, address_space=AddressSpace.SHARED]()
    var best_key = UInt32(0)
    var best_index = _NONE
    var first_bad = _NONE
    var base = (r * cur_l + cur_l - 1) * vocab
    var c = tid
    while c < vocab:
        var value = logits[base + c]
        if (bitcast[DType.uint32](value) & UInt32(0x7F800000)) == UInt32(0x7F800000):
            first_bad = min(first_bad, Int32(c))
        var key = _order_key(value)
        if best_index == _NONE or key > best_key:
            best_key = key
            best_index = Int32(c)
        c += _TPB
    keys[tid] = best_key
    indices[tid] = best_index
    bad[tid] = first_bad
    barrier()
    var active = _TPB // 2
    while active > 0:
        if tid < active:
            var oi = indices[tid + active]
            var ok = keys[tid + active]
            var mi = indices[tid]
            var mk = keys[tid]
            if oi != _NONE and (mi == _NONE or ok > mk or (ok == mk and oi < mi)):
                keys[tid] = ok
                indices[tid] = oi
            bad[tid] = min(bad[tid], bad[tid + active])
        barrier()
        active //= 2
    if tid == 0:
        if bad[0] != _NONE:
            _put(rows, 2 * batch + r, bad[0])
            _put(ids, r, pad)
        else:
            _put(rows, 2 * batch + r, Int32(-1))
            _put(ids, r, indices[0])


def _lm_sample_kernel(
    logits: MutPointer[Float32, MutAnyOrigin], rows: MutPointer[Float32, MutAnyOrigin],
    ids: MutPointer[Float32, MutAnyOrigin], order: MutPointer[Int32, MutAnyOrigin],
    merge: MutPointer[Int32, MutAnyOrigin], weights: MutPointer[Float32, MutAnyOrigin],
    batch_in: Int32, cur_l_in: Int32, vocab_in: Int32, pad: Int32,
    temperature: Float32, top_k: Int32, top_p: Float32, seed: UInt64,
):
    # One worker per row runs the SAME FP32 policy as the host sampler.
    var r = Int(block_idx.x)
    if Int(thread_idx.x) != 0:
        return
    var b = Int(batch_in)
    var v = Int(vocab_in)
    var cur_l = Int(cur_l_in)
    if _w(rows, r) != 0:
        _put(ids, r, pad)
        _put(rows, 2 * b + r, Int32(-1))
        return
    var position = Int(_w(rows, 3 * b + 2))
    var selected = sampling_pick_f32(
        logits + (r * cur_l + cur_l - 1) * v, order + r * v, merge + r * v, weights + r * v,
        v, temperature, Int(top_k), top_p, seed, r, position)
    _put(rows, 2 * b + r, Int32(selected[1]))
    _put(ids, r, Int32(selected[0]) if selected[1] < 0 else pad)


def _lm_status_kernel(rows: MutPointer[Float32, MutAnyOrigin], ids: MutPointer[Float32, MutAnyOrigin],
                      batch_in: Int32):
    # Rows before the first failing row commit, in row order, exactly as
    # the source-ordered host picker; then the summary words.
    var batch = Int(batch_in)
    var tid = Int(thread_idx.x)
    var first = stack_allocation[_TPB, Int32, address_space=AddressSpace.SHARED]()
    var pending = stack_allocation[_TPB, Int32, address_space=AddressSpace.SHARED]()
    var n_eos = Int(_w(rows, 3 * batch + 3))
    var first_bad = _NONE
    var r = tid
    while r < batch:
        if _w(rows, 2 * batch + r) >= 0:
            first_bad = min(first_bad, Int32(r))
        r += _TPB
    first[tid] = first_bad
    barrier()
    var width = _TPB // 2
    while width > 0:
        if tid < width:
            first[tid] = min(first[tid], first[tid + width])
        barrier()
        width //= 2
    var stop = Int(first[0])
    var active = Int32(0)
    r = tid
    while r < batch:
        if _w(rows, r) == 0 and r < stop:
            _put(rows, batch + r, _w(rows, batch + r) + 1)
            var pick = _w(ids, r)
            for i in range(n_eos):
                if pick == _w(rows, 3 * batch + 4 + i):
                    _put(rows, r, Int32(1))
        if _w(rows, r) == 0:
            active = 1
        r += _TPB
    pending[tid] = active
    barrier()
    width = _TPB // 2
    while width > 0:
        if tid < width:
            pending[tid] = max(pending[tid], pending[tid + width])
        barrier()
        width //= 2
    if tid == 0:
        var failed = first[0] != _NONE
        _put(rows, 3 * batch, Int32(0) if failed else pending[0])
        _put(rows, 3 * batch + 1, _w(rows, 2 * batch + stop) if failed else Int32(-1))
        _put(rows, 3 * batch + 2, _w(rows, 3 * batch + 2) + 1)


def _lm_alive_kernel(rows: MutPointer[Float32, MutAnyOrigin], out: MutPointer[Float32, MutAnyOrigin],
                     batch_in: Int32):
    if Int(thread_idx.x) == 0:
        out[0] = Float32(1.0) if _w(rows, 3 * Int(batch_in)) != 0 else Float32(0.0)


def generation_op_check(code: Int, n_tensors: Int, ints: List[Int], fs: List[Float32]) raises:
    var name = String(generation_op_name(code))
    if n_tensors != 2:
        raise Error("ValueError: ops: " + name + " takes two tensors")
    if code == OPS_LM_PICK:
        if len(ints) != LM_PICK_INTS or len(fs) != 2:
            raise Error("ValueError: ops: lm_pick takes [b, vocab, pad, mode, top_k, seed_lo, seed_hi] and"
                        + " [temperature, top_p]")
        if ints[0] < 1 or ints[1] < 1:
            raise Error("ValueError: ops: lm_pick needs b >= 1 and vocab >= 1")
        if ints[3] != LM_PICK_GREEDY and ints[3] != LM_PICK_SAMPLE:
            raise Error("ValueError: ops: lm_pick mode is 0 (greedy) or 1 (sample)")
    else:
        if len(ints) != 1 or len(fs) != 0 or ints[0] < 1:
            raise Error("ValueError: ops: lm_alive takes [b]")


def lm_pick_run(
    mut ex: DeviceExec, logits: FP, n_logits: Int, rows: FP, n_rows: Int, ints: List[Int], fs: List[Float32],
    ids: FP, order: FP, merge: FP, weights: FP,
) raises:
    """Enqueue the pick (no wait). `order`, `merge`, `weights` are b * vocab
    words of scratch (sampling only; greedy passes any buffer)."""
    var b = ints[0]
    var v = ints[1]
    if n_logits % (b * v) != 0 or n_logits == 0:
        raise Error("ValueError: ops: lm_pick: the logits hold " + String(n_logits)
                    + " values, not a multiple of b * vocab = " + String(b * v))
    var l = n_logits // (b * v)
    if n_rows < 3 * b + 4:
        raise Error("ValueError: ops: lm_pick: the rows state holds " + String(n_rows) + " words, fewer than 3 * b + 4")
    var ctx = ex.ctx.copy()
    var lp = block_buffer_view(ex, logits, n_logits).unsafe_ptr()
    var rp = block_buffer_view(ex, rows, n_rows).unsafe_ptr()
    var ip = block_buffer_view(ex, ids, b).unsafe_ptr()
    if ints[3] == LM_PICK_SAMPLE:
        var temperature = fs[0]
        var top_p = fs[1]
        if not (temperature > 0) or top_p <= 0 or top_p > 1 or ints[4] < 0 or ints[4] > v:
            raise Error("generation: temperature must be finite and positive; top_k must be in [0, vocabulary];"
                        + " top_p in (0, 1]")
        var seed = (UInt64(ints[6]) << 32) | UInt64(ints[5])
        var op = block_buffer_view(ex, order, b * v).unsafe_ptr().bitcast[Int32]()
        var mp = block_buffer_view(ex, merge, b * v).unsafe_ptr().bitcast[Int32]()
        var wp = block_buffer_view(ex, weights, b * v).unsafe_ptr()
        ctx.enqueue_function[_lm_sample_kernel](
            lp, rp, ip, op, mp, wp, Int32(b), Int32(l), Int32(v), Int32(ints[2]),
            temperature, Int32(ints[4]), top_p, seed,
            grid_dim=(b, 1, 1), block_dim=(1, 1, 1))
    else:
        ctx.enqueue_function[_lm_greedy_kernel](
            lp, rp, ip, Int32(b), Int32(l), Int32(v), Int32(ints[2]),
            grid_dim=(b, 1, 1), block_dim=(_TPB, 1, 1))
    ctx.enqueue_function[_lm_status_kernel](rp, ip, Int32(b), grid_dim=(1, 1, 1), block_dim=(_TPB, 1, 1))


def lm_alive_run(mut ex: DeviceExec, rows: FP, n_rows: Int, b: Int, out: FP) raises:
    if n_rows < 3 * b + 4:
        raise Error("ValueError: ops: lm_alive: the rows state holds " + String(n_rows) + " words, fewer than 3 * b + 4")
    var ctx = ex.ctx.copy()
    var rp = block_buffer_view(ex, rows, n_rows).unsafe_ptr()
    var op = block_buffer_view(ex, out, 1).unsafe_ptr()
    ctx.enqueue_function[_lm_alive_kernel](rp, op, Int32(b), grid_dim=(1, 1, 1), block_dim=(1, 1, 1))
