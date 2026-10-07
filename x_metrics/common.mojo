# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE METRICS LANE'S ONE-SOURCE PROGRAM MODEL (lane/metrics, 2026-09-27).

The shape is the prep lane's (x_prep/common.mojo), kept in this lane's own
directory so neither lane can move the other's bits: every kernel is a UNIT,
a plain function of `(t, f, q)` computing work item `t` of one stage over the
float32 ARENA `f`, at offsets carried in the stage's Int32 parameters `q`.
The GPU binding launches one thread per unit (x_metrics/device.mojo); the
host binding runs the same units in ascending `t` (x_metrics/host/program.mojo).
CPU == GPU bit for bit is a property of construction.

Every unit is written for IDENTICAL: each loop runs in ascending row order,
every float sum is the fixed-shape pairwise fold `PairSum` below (DEVIATION
6100), every operand is flushed on load (DEVIATION 6104), every product goes
through `identical_mul` (DEVIATION 6105), and no computed NaN reaches an output
(DEVIATION 6103). Integer slots are read and written with `ldi`/`sti`: a small
integer's bits are a subnormal float and `ftz` would zero them.
"""
from std.memory import bitcast
from std.sys.compile import is_defined
from checks.numerics import ftz

#: The host binding's negative control (MOJOLEARN_HOST_SABOTAGE, the CPU
#: gate's routed-set sabotage): every PairSum leaf add is moved by one ulp
#: upward. Only the host build ever defines it.
comptime X_METRICS_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()

#: lane apple-fast-py2mojo-core (2026-10-03): the label layouts and class sums
#: of `_expansion_metrics.py` run here (x_metrics/onehot.mojo) unless the
#: build defines MOJOLEARN_PY2MOJO_core_OFF, which restores the Python
#: layouts for the A/B (the binding reports it as `x_metrics_py2mojo_core`).
comptime PY2MOJO_CORE_ON = not is_defined["MOJOLEARN_PY2MOJO_core_OFF"]()

comptime FP = MutPointer[Float32, MutAnyOrigin]
comptime IP = MutPointer[Int32, MutAnyOrigin]
comptime STAGE_INTS = 16
comptime PARAMS = 14
#: DEVIATION 6100: the leaf width of the pairwise fold. A constant of the
#: source, never of a launch, a core count or a vendor.
# C01_LEAF (lane classical-decomp split, 2026-10-07): classical metrics only,
# an integer sweep -D MOJOLEARN_CLASSICAL_C01_LEAF=32|64|128 (absent = 32).
# Fixed leaves, adjacent binary carries and FTZ use the same PairSum/parallel
# planner on host and all GPU vendors. NOT MEASURED.
from experiments.classical_identical_ideas.shared_controls import C01_LEAF, C01_LEAF_LEGAL
comptime LEAF = C01_LEAF
comptime STACK = 48


@always_inline
def p(q: IP, k: Int) -> Int:
    return Int(q.unsafe_load(k))


@always_inline
def ld(f: FP, i: Int) -> Float32:
    """DEVIATION 6104: every float operand a unit loads is flushed."""
    return ftz(f.unsafe_load(i))


@always_inline
def st(f: FP, i: Int, v: Float32):
    f.unsafe_store(i, ftz(v))


@always_inline
def ldi(f: FP, i: Int) -> Int:
    return Int(bitcast[DType.int32](f.unsafe_load(i)))


@always_inline
def sti(f: FP, i: Int, v: Int):
    f.unsafe_store(i, bitcast[DType.float32](Int32(v)))


@always_inline
def ldu(f: FP, i: Int) -> UInt32:
    """A UInt32 word (a sort key half) stored as its bits."""
    return bitcast[DType.uint32](f.unsafe_load(i))


@always_inline
def stu(f: FP, i: Int, v: UInt32):
    f.unsafe_store(i, bitcast[DType.float32](v))


@always_inline
def is_nan(x: Float32) -> Bool:
    return x != x


@always_inline
def key(x: Float32) -> UInt32:
    """DEVIATION 6101: a total order on float32 words: negatives below
    positives, -0.0 below +0.0, NaN last. Every sort in the lane orders by
    it, ties broken by the row index."""
    if x != x:
        return UInt32(0xFFFFFFFF)
    var b = bitcast[DType.uint32](x)
    if (b & UInt32(0x80000000)) != UInt32(0):
        return ~b
    return b | UInt32(0x80000000)


@always_inline
def fadd(a: Float32, b: Float32) -> Float32:
    """One flushed add: the only way two partial sums meet."""
    return ftz(ftz(a) + ftz(b))


@always_inline
def leaf_add(leaf: Float32, v: Float32) -> Float32:
    """One add inside a PairSum leaf (the sequential part of the fold). The
    parallel fold (x_metrics/par.mojo) and PairSum both add through here, so
    the host sabotage moves both."""
    comptime if X_METRICS_HOST_SABOTAGE:
        return fadd(leaf, v) + Float32(1.1920929e-07) * abs(leaf)
    else:
        return fadd(leaf, v)


struct PairSum(Movable):
    """DEVIATION 6100: THE FOLD SHAPE. Values are added sequentially in
    leaves of LEAF, and the leaves are merged as a binary counter merges its
    carries (the older operand on the left), so the tree is a function of the
    COUNT of values alone. The error grows with log2(n / LEAF) rather than n,
    which a sequential Float32 fold over a million rows cannot afford, and no
    compensation term is carried (gemm/IDENTICAL_FP32_CONTRACT.md section 1)."""
    var stack: InlineArray[Float32, STACK]
    var depth: Int
    var count: Int
    var leaf: Float32
    var nleaf: Int

    def __init__(out self):
        comptime assert C01_LEAF_LEGAL, "MOJOLEARN_CLASSICAL_C01_LEAF must be 32, 64 or 128"
        self.stack = InlineArray[Float32, STACK](fill=Float32(0))
        self.depth = 0
        self.count = 0
        self.leaf = Float32(0)
        self.nleaf = 0

    @always_inline
    def add(mut self, v: Float32):
        self.leaf = leaf_add(self.leaf, v)
        self.nleaf += 1
        if self.nleaf == LEAF:
            self._push(self.leaf)
            self.leaf = Float32(0)
            self.nleaf = 0

    def _push(mut self, x_in: Float32):
        var x = x_in
        var c = self.count
        while (c & 1) == 1:
            self.depth -= 1
            x = fadd(self.stack[self.depth], x)
            c >>= 1
        self.stack[self.depth] = x
        self.depth += 1
        self.count += 1

    def result(self) -> Float32:
        var acc = self.leaf
        var i = self.depth - 1
        while i >= 0:
            acc = fadd(self.stack[i], acc)
            i -= 1
        return ftz(acc)
