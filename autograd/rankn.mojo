# SPDX-License-Identifier: Apache-2.0
# ARBITRARY-RANK SHAPES FOR THE TAPE (lane ag-rankn-core, 2026-10-20;
# docs/plans/AUTOGRAD_RANKN.md sections 1.1, 1.2 and 1.4).
#
# A tape tensor's storage stays one flat row-major float32 buffer. Its rank-N
# shape is metadata: Tape keeps one ShapeN per node beside _nodes, and every
# existing method keeps working on the MATRIX VIEW, rows = the product of the
# leading dims and cols = the last dim. Rank 2 is the special case
# shape = (rows, cols); a scalar is rank 0 with rows = cols = 1.
#
# Kernels that need shapes and strides read them from a DESCRIPTOR: tape
# storage holding int32 bit patterns (Tape.descriptor), bound to Args.p11 and
# read with rn_word (a bitcast, never ld: ftz would zero a small integer's
# pattern, as autograd/embedding_ops.mojo _ld_i notes). A launch's descriptor
# is a concatenation of these records, at offsets its body names:
#   RN_DESC_SHAPE  : [ndim, d0 .. d7]   RN_DESC_SHAPE_WORDS = 9 words
#                    (unused dims hold 1)
#   RN_DESC_STRIDE : [s0 .. s7]         RN_DESC_STRIDE_WORDS = 8 words
#                    (element strides; 0 on a broadcast axis and unused axes)
# A SHAPE record followed by a STRIDE record is a RN_DESC_VIEW
# (RN_DESC_VIEW_WORDS = 17 words): the strided read of a dense output.
#
# rn_collapse merges adjacent axes whose strides are compatible on every
# operand and drops extent-1 axes before a descriptor is uploaded: metadata
# over at most 8 axes. Each output element still reads the same operand
# words, so it changes no word; it only shortens the decode.
from std.memory import bitcast
from sequence.ops import FP

comptime RN_MAX_RANK = 8
comptime RN_DESC_SHAPE_WORDS = 1 + RN_MAX_RANK
comptime RN_DESC_STRIDE_WORDS = RN_MAX_RANK
comptime RN_DESC_VIEW_WORDS = RN_DESC_SHAPE_WORDS + RN_DESC_STRIDE_WORDS
# Element-count bound of one tensor: Tape._shape's (addressable float32 storage).
comptime RN_MAX_COUNT = (1 << 60) // 4


@fieldwise_init
struct ShapeN(ImplicitlyCopyable, Movable, Writable):
    """The shape of a tape tensor: `ndim` dims, row-major.

    `d` holds the dims in its first `ndim` slots; the unused slots hold 1.
    A rank-0 shape is a scalar. `rows()` and `cols()` give the matrix view
    every 2-D operation works on.
    """
    var ndim: Int
    var d: InlineArray[Int, RN_MAX_RANK]

    def __init__(out self, *, copy: Self):
        self.ndim = copy.ndim
        self.d = copy.d.copy()

    @staticmethod
    def matrix(rows: Int, cols: Int) -> ShapeN:
        """The rank-2 shape (rows, cols)."""
        var d = InlineArray[Int, RN_MAX_RANK](fill=1)
        d[0] = rows
        d[1] = cols
        return ShapeN(2, d^)

    @staticmethod
    def scalar() -> ShapeN:
        """The rank-0 shape."""
        return ShapeN(0, InlineArray[Int, RN_MAX_RANK](fill=1))

    @staticmethod
    def of(dims: List[Int]) raises -> ShapeN:
        """The shape with these dims (at most RN_MAX_RANK, each positive)."""
        if len(dims) > RN_MAX_RANK:
            raise Error("tensor rank must be at most 8")
        var d = InlineArray[Int, RN_MAX_RANK](fill=1)
        for k in range(len(dims)):  # small-loop(ndim: tensor rank, at most 8): shape metadata, no data
            d[k] = dims[k]
        var s = ShapeN(len(dims), d^)
        rn_check(s)
        return s

    def dims(self) -> List[Int]:
        """The dims as a list of length ndim."""
        var r = List[Int]()
        for k in range(self.ndim):  # small-loop(ndim: tensor rank, at most 8): shape metadata, no data
            r.append(self.d[k])
        return r^

    def count(self) -> Int:
        """The number of elements (1 for a scalar)."""
        var c = 1
        for k in range(self.ndim):  # small-loop(ndim: tensor rank, at most 8): shape metadata, no data
            c *= self.d[k]
        return c

    def rows(self) -> Int:
        """The matrix view's rows: the product of every dim but the last."""
        var c = 1
        for k in range(self.ndim - 1):  # small-loop(ndim: tensor rank, at most 8): shape metadata, no data
            c *= self.d[k]
        return c

    def cols(self) -> Int:
        """The matrix view's cols: the last dim (1 for a scalar)."""
        if self.ndim == 0:
            return 1
        return self.d[self.ndim - 1]

    def stride(self, k: Int) -> Int:
        """The row-major element stride of axis k."""
        var s = 1
        for j in range(k + 1, self.ndim):  # small-loop(ndim: tensor rank, at most 8): shape metadata, no data
            s *= self.d[j]
        return s

    def strides(self) -> InlineArray[Int, RN_MAX_RANK]:
        """Every axis's row-major element stride (0 in the unused slots)."""
        var s = InlineArray[Int, RN_MAX_RANK](fill=0)
        var run = 1
        for j in range(self.ndim):  # small-loop(ndim: tensor rank, at most 8): shape metadata, no data
            var k = self.ndim - 1 - j
            s[k] = run
            run *= self.d[k]
        return s^

    def same(self, o: ShapeN) -> Bool:
        """True when both shapes have the same rank and dims."""
        if self.ndim != o.ndim:
            return False
        for k in range(self.ndim):  # small-loop(ndim: tensor rank, at most 8): shape metadata, no data
            if self.d[k] != o.d[k]:
                return False
        return True

    def with_last(self, cols: Int) -> ShapeN:
        """This shape with its last dim replaced by `cols` (a scalar
        becomes the rank-1 shape (cols,))."""
        var d = self.d.copy()
        if self.ndim == 0:
            d[0] = cols
            return ShapeN(1, d^)
        d[self.ndim - 1] = cols
        return ShapeN(self.ndim, d^)

    def write_to(self, mut writer: Some[Writer]):
        writer.write("(")
        for k in range(self.ndim):  # small-loop(ndim: tensor rank, at most 8): shape metadata, no data
            if k > 0:
                writer.write(", ")
            writer.write(self.d[k])
        writer.write(")")


def rn_check(s: ShapeN) raises:
    """Refuse a rank above RN_MAX_RANK, a dim below 1, or more elements than
    a tensor's addressable storage holds."""
    if s.ndim < 0 or s.ndim > RN_MAX_RANK:
        raise Error("tensor rank must be between 0 and 8")
    var c = 1
    for k in range(s.ndim):  # small-loop(ndim: tensor rank, at most 8): shape metadata, no data
        if s.d[k] < 1:
            raise Error("tensor dimensions must be positive")
        if s.d[k] > RN_MAX_COUNT // c:
            raise Error("tensor dimensions must fit in addressable storage")
        c *= s.d[k]


def rn_broadcast(a: ShapeN, b: ShapeN) raises -> ShapeN:
    """The NumPy broadcast of two shapes: dims aligned from the right, a dim
    of 1 stretches, any other mismatch raises."""
    var nd = max(a.ndim, b.ndim)
    var d = InlineArray[Int, RN_MAX_RANK](fill=1)
    for j in range(nd):  # small-loop(ndim: tensor rank, at most 8): shape metadata, no data
        var da = a.d[a.ndim - 1 - j] if j < a.ndim else 1
        var db = b.d[b.ndim - 1 - j] if j < b.ndim else 1
        if da != db and da != 1 and db != 1:
            raise Error("incompatible tensor shapes for broadcasting")
        d[nd - 1 - j] = max(da, db)
    return ShapeN(nd, d^)


def rn_normalize_axis(axis: Int, ndim: Int) raises -> Int:
    """axis in [-ndim, ndim) as an index in [0, ndim)."""
    if axis < -ndim or axis >= ndim:
        raise Error("axis " + String(axis) + " is out of range for a tensor of rank " + String(ndim))
    return axis + ndim if axis < 0 else axis


def rn_collapse(mut shape: ShapeN, mut strides: List[InlineArray[Int, RN_MAX_RANK]]):
    """Drop extent-1 axes and merge each pair of adjacent axes whose outer
    stride equals the inner stride times the inner extent on EVERY operand
    (strides[o] is operand o's element strides over `shape`; 0 marks a
    broadcast axis, which merges with a broadcast neighbour). A shape with
    no axis left becomes (1,) with stride 0. Every element still maps to the
    same operand words: no word changes, the decode is shorter."""
    var n_ops = len(strides)
    var nd = 0
    var dims = InlineArray[Int, RN_MAX_RANK](fill=1)
    var out = List[InlineArray[Int, RN_MAX_RANK]]()
    for _ in range(n_ops):  # small-loop(operands of one launch, at most a few): shape metadata
        out.append(InlineArray[Int, RN_MAX_RANK](fill=0))
    for k in range(shape.ndim):  # small-loop(ndim: tensor rank, at most 8): shape metadata, no data
        var e = shape.d[k]
        if e == 1:
            continue
        var merge = nd > 0
        if merge:
            for o in range(n_ops):  # small-loop(operands of one launch): shape metadata
                if out[o][nd - 1] != strides[o][k] * e:
                    merge = False
        if merge:
            dims[nd - 1] = dims[nd - 1] * e
            for o in range(n_ops):  # small-loop(operands of one launch): shape metadata
                out[o][nd - 1] = strides[o][k]
        else:
            dims[nd] = e
            for o in range(n_ops):  # small-loop(operands of one launch): shape metadata
                out[o][nd] = strides[o][k]
            nd += 1
    if nd == 0:
        nd = 1
    shape = ShapeN(nd, dims^)
    strides = out^


@always_inline
def rn_word(p: FP, i: Int) -> Int:
    """The int32 held as bits in descriptor word i (a bitcast, never ld)."""
    return Int(bitcast[DType.int32](p.unsafe_load(i)))


def rn_desc_shape(mut words: List[Int], s: ShapeN):
    """Append a RN_DESC_SHAPE record: [ndim, d0 .. d7], unused dims 1."""
    words.append(s.ndim)
    for k in range(RN_MAX_RANK):  # small-loop(RN_MAX_RANK = 8: descriptor slots)
        words.append(s.d[k] if k < s.ndim else 1)


def rn_desc_strides(mut words: List[Int], s: InlineArray[Int, RN_MAX_RANK]):
    """Append a RN_DESC_STRIDE record: [s0 .. s7]."""
    for k in range(RN_MAX_RANK):  # small-loop(RN_MAX_RANK = 8: descriptor slots)
        words.append(s[k])


def rn_desc_view(mut words: List[Int], s: ShapeN, strides: InlineArray[Int, RN_MAX_RANK]):
    """Append a RN_DESC_VIEW record (a SHAPE record, then a STRIDE record)
    for a strided read of a dense output of shape s, collapsed first
    (rn_collapse: the same words, a shorter decode)."""
    var shape = s
    var st = List[InlineArray[Int, RN_MAX_RANK]]()
    st.append(strides.copy())
    rn_collapse(shape, st)
    rn_desc_shape(words, shape)
    rn_desc_strides(words, st[0])
