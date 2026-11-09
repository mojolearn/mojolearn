# SPDX-License-Identifier: Apache-2.0
"""The HOST ARM of the generation ops (lane x56d-mamba-samba-train-and-host,
2026-11-09; autograd/generation_ops.mojo has the device bodies and the rule).

A host session (`OpsSession[HostExec]`) runs OPS_LM_PICK and OPS_LM_ALIVE
through these bodies over host storage (an FP is the floats themselves; the
int32 words are stored as float bit patterns, as on the device): the order
key, the lowest index on a tie, the first non-finite token, the one FP32
sampler `sampling_pick_f32` at (seed, row, position), the row-order commit.
The rows' picks are independent and run over host tasks; the commit is in
row order. So the CPU column's `CausalLM.generate` can be the same captured
`Program.decode` as the GPUs'.

This is a HOST module: no GPU binding imports it. Generic session code
reaches it through the executor (`Exec.gen_pick`, `Exec.gen_alive`;
HostExec, sequence/exec.mojo), as the step ops' host arm
(autograd/block_step_host.mojo).
"""
from std.memory import bitcast
from sequence.ops import FP
from core.generation_sampling_f32 import sampling_pick_f32
from core.host_parallel import host_parallelize
from autograd.generation_ops import LM_PICK_SAMPLE

comptime _NONE: Int32 = 2147483647


@always_inline
def _order_key(value: Float32) -> UInt32:
    """generation_ops' order key (by bits; -0.0 keys as +0.0)."""
    var bits = bitcast[DType.uint32](value)
    if (bits & UInt32(0x7FFFFFFF)) == UInt32(0):
        bits = UInt32(0)
    if (bits & UInt32(0x80000000)) != UInt32(0):
        return ~bits
    return bits | UInt32(0x80000000)


@always_inline
def _hw(p: FP, i: Int) -> Int32:
    return bitcast[DType.int32](p[i])


@always_inline
def _hput(p: FP, i: Int, v: Int32):
    p[i] = bitcast[DType.float32](v)


def lm_pick_host(logits: FP, n_logits: Int, rows: FP, n_rows: Int, ints: List[Int], fs: List[Float32],
                 ids: FP) raises:
    """`lm_pick_run` on a host session: each row's pick (independent rows,
    host tasks), then the status commit in row order."""
    var b = ints[0]
    var v = ints[1]
    if n_logits % (b * v) != 0 or n_logits == 0:
        raise Error("ValueError: ops: lm_pick: the logits hold " + String(n_logits)
                    + " values, not a multiple of b * vocab = " + String(b * v))
    var l = n_logits // (b * v)
    if n_rows < 3 * b + 4:
        raise Error("ValueError: ops: lm_pick: the rows state holds " + String(n_rows) + " words, fewer than 3 * b + 4")
    var pad = Int32(ints[2])
    var lp = logits.unsafe_origin_cast[MutAnyOrigin]()
    if ints[3] == LM_PICK_SAMPLE:
        var temperature = fs[0]
        var top_p = fs[1]
        var top_k = ints[4]
        if not (temperature > 0) or top_p <= 0 or top_p > 1 or top_k < 0 or top_k > v:
            raise Error("generation: temperature must be finite and positive; top_k must be in [0, vocabulary];"
                        + " top_p in (0, 1]")
        var seed = (UInt64(ints[6]) << 32) | UInt64(ints[5])
        var position = Int(_hw(rows, 3 * b + 2))
        var order = List[Int32](length=b * v, fill=Int32(0))
        var merge = List[Int32](length=b * v, fill=Int32(0))
        var weights = List[Float32](length=b * v, fill=Float32(0))
        var op = order.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var mp = merge.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var wp = weights.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()

        def _sample(r: Int) {imm lp, imm op, imm mp, imm wp, imm rows, imm ids, imm b, imm v, imm l, imm pad,
                             imm temperature, imm top_k, imm top_p, imm seed, imm position}:
            if _hw(rows, r) != 0:
                _hput(ids, r, pad)
                _hput(rows, 2 * b + r, Int32(-1))
                return
            var selected = sampling_pick_f32(
                lp + (r * l + l - 1) * v, op + r * v, mp + r * v, wp + r * v,
                v, temperature, top_k, top_p, seed, r, position)
            _hput(rows, 2 * b + r, Int32(selected[1]))
            _hput(ids, r, Int32(selected[0]) if selected[1] < 0 else pad)

        host_parallelize(_sample, b)
        _ = len(order)
        _ = len(merge)
        _ = len(weights)
    else:

        def _greedy(r: Int) {imm lp, imm rows, imm ids, imm b, imm v, imm l, imm pad}:
            if _hw(rows, r) != 0:
                _hput(ids, r, pad)
                _hput(rows, 2 * b + r, Int32(-1))
                return
            var base = (r * l + l - 1) * v
            var best_key = UInt32(0)
            var best_index = _NONE
            var first_bad = _NONE
            for c in range(v):
                var value = lp[base + c]
                if (bitcast[DType.uint32](value) & UInt32(0x7F800000)) == UInt32(0x7F800000) and first_bad == _NONE:
                    first_bad = Int32(c)
                var key = _order_key(value)
                if best_index == _NONE or key > best_key:
                    best_key = key
                    best_index = Int32(c)
            if first_bad != _NONE:
                _hput(rows, 2 * b + r, first_bad)
                _hput(ids, r, pad)
            else:
                _hput(rows, 2 * b + r, Int32(-1))
                _hput(ids, r, best_index)

        host_parallelize(_greedy, b)
    _lm_status_host(rows, ids, b)


def _lm_status_host(rows: FP, ids: FP, b: Int):
    """`_lm_status_kernel` on the host: rows before the first failing row
    commit, in row order; then the summary words."""
    var n_eos = Int(_hw(rows, 3 * b + 3))
    var stop = b
    for r in range(b):  # driver-loop: over the batch rows' status words (the first failing row; b words)
        if _hw(rows, 2 * b + r) >= 0:
            stop = r
            break
    var active = Int32(0)
    for r in range(b):  # driver-loop: over the batch rows' status words (the row-order commit; b words)
        if _hw(rows, r) == 0 and r < stop:
            _hput(rows, b + r, _hw(rows, b + r) + 1)
            var pick = _hw(ids, r)
            for i in range(n_eos):  # small-loop(n_eos: the generation's eos ids): id compare, no data
                if pick == _hw(rows, 3 * b + 4 + i):
                    _hput(rows, r, Int32(1))
        if _hw(rows, r) == 0:
            active = 1
    var failed = stop < b
    _hput(rows, 3 * b, Int32(0) if failed else active)
    _hput(rows, 3 * b + 1, _hw(rows, 2 * b + stop) if failed else Int32(-1))
    _hput(rows, 3 * b + 2, _hw(rows, 3 * b + 2) + 1)


def lm_alive_host(rows: FP, n_rows: Int, b: Int, dst: FP) raises:
    """`lm_alive_run` on a host session."""
    if n_rows < 3 * b + 4:
        raise Error("ValueError: ops: lm_alive: the rows state holds " + String(n_rows) + " words, fewer than 3 * b + 4")
    dst[0] = Float32(1.0) if _hw(rows, 3 * b) != 0 else Float32(0.0)
