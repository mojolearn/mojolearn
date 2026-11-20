# SPDX-License-Identifier: Apache-2.0
"""HOST CHECK: the Mamba-3 backward S17 chunk-end tail (`d_dacs_rec`) of the
host column against a float64 restatement of the same sum, on a three-chunk
row with a partial last chunk, and the tail's tree against the GEMM
contract's own fold.

Why it exists: the stored-word checks of the Mamba-3 backward
(training/checks/samba_tape_step_check.mojo, L = 32) run one chunk, where the
next chunk's carried state is zero and the tail is +0.0 on every arm, so they
cannot see the tail's order. This check runs l > qs.

What it requires, in either build (the default: lane folds and the tree;
_OFF: the serial chain):

  1. every `d_dacs_rec` cell is within `S17_TOL` x (the sum of the absolute
     terms) of the float64 sum of the same terms: the same mathematics, a
     different summation order;
  2. a non-chunk-end token's cell and the last chunk's cells carry no tail
     (the float64 tail is exactly zero there);

and in the default build only:

  3. `_m3_s17_tail_tree` returns the bits of
     `gemm/host/gemm_oracle.mojo::fold_balanced_tree` (contract 7.2) on
     M3_D_STATE partials, for several seeds.

It prints the FNV-1a 64 digest of the `d_dacs_rec` words so the arm's host
words can be read beside a device dump of the same operands. No GPU.
Serial by nature: one row of 20 cells, under a second; nothing to split.

    mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_CPU -I . -I bindings mamba/checks/mamba3_s17_tail_tree_check.mojo

Exit 1 on any failure.
"""
from std.math import exp
from std.memory import bitcast

from gemm.host.gemm_oracle import fold_balanced_tree
from mamba.checks.mamba3_fixture import M3_D_STATE, M3_HEADDIM
from mamba.host.device_shim import DeviceBuffer
from mamba.host.gen.mamba3_backward import (
    _m3_s17_tail_tree,
    mamba3_s17_operands_kernel,
)

#: the bound on |host - float64| as a fraction of the sum of absolute terms.
#: A float32 chain of T fused steps is within about T x 2^-24 of the exact
#: sum relative to that scale; the longest chain here is
#: (qs + 1) x M3_HEADDIM x M3_D_STATE = 40,960 steps (2.4e-3 worst case, about
#: sqrt(T) x 2^-24 = 1.2e-5 typical). 1e-3 passes both orders and fails a
#: dropped or doubled term (one term of 40,960 equal ones is 2.4e-5; the
#: operands here are spread over [-1, 1), so a lost chunk or lane is far over).
comptime S17_TOL: Float64 = 1e-3

comptime CHK_B = 1
comptime CHK_NH = 2
comptime CHK_QS = 4
comptime CHK_L = 10


def _next(mut state: UInt64) -> Float32:
    """A fixed 32-bit LCG mapped to [-1, 1) on a 2^-23 grid."""
    state = (state * UInt64(1664525) + UInt64(1013904223)) & UInt64(0xFFFFFFFF)
    var top = Int(state >> UInt64(8))
    return Float32(top) / Float32(8388608.0) - Float32(1.0)


def _fill(buf: DeviceBuffer[DType.float32], count: Int, mut state: UInt64):
    var ptr = buf.unsafe_ptr()
    for i in range(count):
        ptr.unsafe_store(i, _next(state))


def _fnv(buf: DeviceBuffer[DType.float32], count: Int) -> UInt64:
    var ptr = buf.unsafe_ptr()
    var hash = UInt64(14695981039346656037)
    for i in range(count):
        hash = (hash ^ UInt64(bitcast[DType.uint32](ptr.unsafe_load(i)))) * UInt64(1099511628211)
    return hash


def run_checks() raises:
    var b = CHK_B
    var nh = CHK_NH
    var qs = CHK_QS
    var l = CHK_L
    var nc = (l + qs - 1) // qs
    var rows = b * l * nh
    var chunk_cells = b * nc * nh * M3_HEADDIM * M3_D_STATE
    var d_q_read = DeviceBuffer[DType.float32](rows * M3_D_STATE)
    var d_dacs_read = DeviceBuffer[DType.float32](rows)
    var d_k_rec = DeviceBuffer[DType.float32](rows * M3_D_STATE)
    var d_v_rec = DeviceBuffer[DType.float32](rows * M3_HEADDIM)
    var d_dacs_rec = DeviceBuffer[DType.float32](rows)
    var d_y = DeviceBuffer[DType.float32](rows * M3_HEADDIM)
    var q = DeviceBuffer[DType.float32](rows * M3_D_STATE)
    var k = DeviceBuffer[DType.float32](rows * M3_D_STATE)
    var v = DeviceBuffer[DType.float32](rows * M3_HEADDIM)
    var dacs = DeviceBuffer[DType.float32](b * nh * nc * qs)
    var states = DeviceBuffer[DType.float32](chunk_cells)
    var d_states = DeviceBuffer[DType.float32](chunk_cells)
    var seed = UInt64(20261118)
    _fill(d_y, rows * M3_HEADDIM, seed)
    _fill(q, rows * M3_D_STATE, seed)
    _fill(k, rows * M3_D_STATE, seed)
    _fill(v, rows * M3_HEADDIM, seed)
    _fill(states, chunk_cells, seed)
    _fill(d_states, chunk_cells, seed)
    # dACS is a chunk-local inclusive cumsum of negative steps
    var dacs_ptr = dacs.unsafe_ptr()
    for cell in range(b * nh * nc):
        var run = Float32(0.0)
        for j in range(qs):
            run = run - Float32(0.03125) * Float32(1 + ((cell + j) % 3))
            dacs_ptr.unsafe_store(cell * qs + j, run)

    for th in range(rows):
        mamba3_s17_operands_kernel(
            th,
            d_q_read.unsafe_ptr(), d_dacs_read.unsafe_ptr(), d_k_rec.unsafe_ptr(), d_v_rec.unsafe_ptr(), d_dacs_rec.unsafe_ptr(),
            d_y.unsafe_ptr(), q.unsafe_ptr(), k.unsafe_ptr(), v.unsafe_ptr(),
            dacs.unsafe_ptr(), states.unsafe_ptr(), d_states.unsafe_ptr(),
            Int32(b), Int32(l), Int32(nh), Int32(qs),
        )

    var bad = 0
    var k_ptr = k.unsafe_ptr()
    var v_ptr = v.unsafe_ptr()
    var st_ptr = states.unsafe_ptr()
    var ds_ptr = d_states.unsafe_ptr()
    var rec_ptr = d_dacs_rec.unsafe_ptr()
    for th in range(rows):
        var h = th % nh
        var token = (th // nh) % l
        var bb = th // (nh * l)
        var c = token // qs
        var inner = token % qs
        var base = ((bb * nh + h) * nc + c) * qs
        var last = Float64(dacs_ptr.unsafe_load(base + qs - 1))
        var dec = exp(last - Float64(dacs_ptr.unsafe_load(base + inner)))
        var has_carry = c + 1 < nc
        var carry_base = ((bb * nc + c + 1) * nh + h) * M3_HEADDIM
        var state_base = ((bb * nc + c) * nh + h) * M3_HEADDIM
        var total = Float64(0.0)
        var scale = Float64(0.0)
        var tail = Float64(0.0)
        if has_carry:
            for p in range(M3_HEADDIM):
                var dv = Float64(0.0)
                for n in range(M3_D_STATE):
                    dv += Float64(ds_ptr.unsafe_load((carry_base + p) * M3_D_STATE + n)) * Float64(k_ptr.unsafe_load(th * M3_D_STATE + n))
                var term = dv * dec * Float64(v_ptr.unsafe_load(th * M3_HEADDIM + p))
                total -= term
                scale += abs(term)
            if inner == qs - 1 or token == l - 1:
                for j in range(qs):
                    var tj = c * qs + j
                    if tj < l:
                        var de = exp(last - Float64(dacs_ptr.unsafe_load(base + j)))
                        var row = (bb * l + tj) * nh + h
                        for p in range(M3_HEADDIM):
                            for n in range(M3_D_STATE):
                                var term = (
                                    Float64(ds_ptr.unsafe_load((carry_base + p) * M3_D_STATE + n))
                                    * Float64(v_ptr.unsafe_load(row * M3_HEADDIM + p))
                                    * Float64(k_ptr.unsafe_load(row * M3_D_STATE + n))
                                    * de
                                )
                                tail += term
                                scale += abs(term)
                var el = exp(last)
                for p in range(M3_HEADDIM):
                    for n in range(M3_D_STATE):
                        var term = (
                            Float64(ds_ptr.unsafe_load((carry_base + p) * M3_D_STATE + n))
                            * Float64(st_ptr.unsafe_load((state_base + p) * M3_D_STATE + n))
                            * el
                        )
                        tail += term
                        scale += abs(term)
        var want = total + tail
        var got = Float64(rec_ptr.unsafe_load(th))
        var bound = S17_TOL * scale
        if not has_carry and got != 0.0:
            print("FAIL last-chunk cell", th, "carries no state and must be zero, got", got)
            bad += 1
        elif abs(got - want) > bound:
            print("FAIL cell", th, "token", token, "head", h, "got", got, "float64", want, "bound", bound)
            bad += 1
    print("d_dacs_rec", "tree", "words", rows, "fnv1a64", _fnv(d_dacs_rec, rows))

    var tree_seed = UInt64(7)
    for trial in range(8):
        var lanes = Array[Float32, M3_D_STATE](fill=Float32(0))
        var listed = List[Float32]()
        for i in range(M3_D_STATE):
            var value = _next(tree_seed)
            # one trial in two spreads the magnitudes so the pairing matters
            if (trial & 1) == 1 and (i % 5) == 0:
                value = value * Float32(4096.0)
            lanes[i] = value
            listed.append(value)
        var mine = _m3_s17_tail_tree(lanes)
        var theirs = fold_balanced_tree(listed)
        if bitcast[DType.uint32](mine) != bitcast[DType.uint32](theirs):
            print("FAIL tree trial", trial, "tail tree", mine, "contract 7.2 fold", theirs)
            bad += 1

    # the buffers the pointers above read stay alive to here
    _ = len(d_q_read) + len(d_dacs_read) + len(d_k_rec) + len(d_v_rec) + len(d_dacs_rec) + len(d_y)
    _ = len(q) + len(k) + len(v) + len(dacs) + len(states) + len(d_states)
    if bad != 0:
        print("mamba3 S17 tail check: FAIL", bad)
        raise Error("mamba3 S17 tail check: FAIL")
    print("mamba3 S17 tail check: PASS")
