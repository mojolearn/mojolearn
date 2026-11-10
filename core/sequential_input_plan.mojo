# SPDX-License-Identifier: Apache-2.0
"""Typed numeric promotion and preparation plans for admitted sequence inputs."""
from std.memory import bitcast
from core.borrowed_buffer import BorrowedBufferLayout
from core.cast_in import cast_w64_bits, f64_to_f32_bits, CAST_KIND_UNSIGNED, CAST_KIND_SIGNED, CAST_KIND_FLOAT, CAST_KIND_BOOL
from core.classical_pandas_impl import pd_raw_at
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from core.admitted_convert import _lin_store

def sequence_promote(mut kind: String, mut size: Int, incoming: String, width: Int) raises:
    if (incoming != "b" and incoming != "i" and incoming != "u" and incoming != "f") or (
        incoming == "f" and width != 2 and width != 4 and width != 8
    ) or (incoming != "f" and width != 1 and width != 2 and width != 4 and width != 8):
        raise Error("TypeError: inputs must contain supported numeric scalars")
    if size == 0 or kind == "b":
        kind = incoming
        size = width
    elif incoming == "b":
        return
    elif kind == incoming:
        size = max(size, width)
    elif kind == "f" or incoming == "f":
        var integer_width = width if kind == "f" else size
        var float_width = size if kind == "f" else width
        # Numeric dtype promotion: half holds 8-bit integers, single holds
        # 16-bit integers; 32/64-bit integer mixtures use double precision.
        size = max(float_width, 2 if integer_width <= 1 else (4 if integer_width <= 2 else 8))
        kind = "f"
    else:
        var signed_width = size if kind == "i" else width
        var unsigned_width = size if kind == "u" else width
        if signed_width > unsigned_width:
            kind = "i"
            size = signed_width
        elif unsigned_width < 8:
            kind = "i"
            size = unsigned_width * 2
        else:
            kind = "f"
            size = 8


def sequence_kind_code(kind: String) -> Int:
    return CAST_KIND_FLOAT if kind == "f" else (CAST_KIND_SIGNED if kind == "i" else (CAST_KIND_BOOL if kind == "b" else CAST_KIND_UNSIGNED))


struct SequenceBufferFormat(Copyable, Movable):
    var kind: String
    var prefix: String
    var itemsize: Int
    var native: Bool

    def __init__(out self, var format: String, itemsize: Int) raises:
        self.prefix = String()
        if format.byte_length() > 1:
            var first = String(format[byte=0:1])
            if first == "@" or first == "=" or first == "<" or first == ">" or first == "!":
                if first == "<" or first == ">" or first == "!":
                    self.prefix = ">" if first == "!" else first
                format = String(format[byte=1:])
        self.kind = String()
        if format == "?":
            self.kind = "b"
        elif format == "e" or format == "f" or format == "d":
            self.kind = "f"
        elif format == "b" or format == "h" or format == "i" or format == "l" or format == "q" or format == "n":
            self.kind = "i"
        elif format == "B" or format == "H" or format == "I" or format == "L" or format == "Q" or format == "N":
            self.kind = "u"
        else:
            raise Error("TypeError: inputs must expose a numeric buffer")
        var kind = String()
        var width = 0
        sequence_promote(kind, width, self.kind, itemsize)
        self.itemsize = itemsize
        # All supported host/GPU platforms expose little-endian native words.
        self.native = self.prefix != ">"


def sequence_buffer_shape(rank: Int, c_order: Bool, samples: Bool) raises:
    if samples and rank > 2 and not c_order:
        raise Error("X with more than two dimensions must be C-contiguous")


def sequence_array_typestr(kind: String, width: Int, name: String) raises -> String:
    var typestr = String("<f8")
    if kind == "b":
        typestr = "<i1"
    elif kind == "i":
        typestr = "<i8"
    elif kind != "f":
        raise Error("TypeError: " + name + " holds integers beyond the int64 range")
    if width != (1 if kind == "b" else 8):
        raise Error("TypeError: " + name + " holds unsupported numbers")
    return typestr^


struct SequenceArray(Copyable, Movable):
    var layout: BorrowedBufferLayout
    var kind: Int
    var native: Bool

    def __init__(out self, var layout: BorrowedBufferLayout, kind: Int, native: Bool):
        self.layout = layout^
        self.kind = kind
        self.native = native


struct SequenceSegment(Copyable, Movable):
    var start: Int
    var count: Int
    var source: Int
    var scalar_start: Int

    def __init__(out self, start: Int, count: Int, source: Int, scalar_start: Int = 0):
        self.start = start
        self.count = count
        self.source = source
        self.scalar_start = scalar_start


struct SequenceInputPlan(Movable):
    var shape: List[Int]
    var kind: String
    var itemsize: Int
    var leaf_depth: Int
    var count: Int
    var words: List[UInt64]
    var kinds: List[UInt8]
    var arrays: List[SequenceArray]
    var segments: List[SequenceSegment]

    def __init__(out self):
        self.shape = List[Int]()
        self.kind = String()
        self.itemsize = 0
        self.leaf_depth = -1
        self.count = 0
        self.words = List[UInt64]()
        self.kinds = List[UInt8]()
        self.arrays = List[SequenceArray]()
        self.segments = List[SequenceSegment]()

    def container(mut self, depth: Int, count: Int) raises:
        if depth >= 8 or count == 0:
            raise Error("input sequences must be nonempty and have rank at most eight")
        if self.leaf_depth >= 0 and depth >= self.leaf_depth:
            raise Error("input sequences must be rectangular")
        if depth == len(self.shape):
            self.shape.append(count)
        elif self.shape[depth] != count:
            raise Error("input sequences must be rectangular")

    def leaf(mut self, depth: Int, shape: List[Int]) raises:
        var rank = len(shape)
        if depth + rank > 8:
            raise Error("input sequences must have rank at most eight")
        if self.leaf_depth >= 0 and self.leaf_depth != depth + rank:
            raise Error("input sequences must be rectangular")
        self.leaf_depth = depth + rank
        for axis in range(rank):  # driver-loop: over admitted sequence leaf rank metadata
            if shape[axis] == 0:
                raise Error("input sequences must be nonempty")
            if depth + axis == len(self.shape):
                self.shape.append(shape[axis])
            elif self.shape[depth + axis] != shape[axis]:
                raise Error("input sequences must be rectangular")
        if len(self.shape) != self.leaf_depth:
            raise Error("input sequences must be rectangular")

    def scalar(mut self, depth: Int, word: UInt64, kind: String, width: Int) raises:
        self.leaf(depth, List[Int]())
        sequence_promote(self.kind, self.itemsize, kind, width)
        var source = len(self.words)
        self.words.append(word)
        self.kinds.append(UInt8(sequence_kind_code(kind)))
        if len(self.segments) > 0 and self.segments[len(self.segments) - 1].source == -1:
            self.segments[len(self.segments) - 1].count += 1
        else:
            self.segments.append(SequenceSegment(self.count, 1, -1, source))
        self.count += 1

    def array(mut self, depth: Int, var layout: BorrowedBufferLayout,
        kind: String, native: Bool) raises:
        self.leaf(depth, layout.shape)
        sequence_promote(self.kind, self.itemsize, kind, layout.itemsize)
        if layout.count > 9223372036854775807 - self.count:
            raise Error("input sequence element count exceeds native storage")
        self.segments.append(SequenceSegment(self.count, layout.count, len(self.arrays)))
        self.count += layout.count
        self.arrays.append(SequenceArray(layout^, sequence_kind_code(kind), native))


def _sequence_store(word: UInt64, kind: Int, target_kind: Int, width: Int,
    destination: Int, index: Int):
    var bits = word
    if target_kind == CAST_KIND_FLOAT:
        var number = bitcast[DType.float64](word) if kind == CAST_KIND_FLOAT else (
            Float64(bitcast[DType.int64](word)) if kind == CAST_KIND_SIGNED else Float64(word))
        # array.array constructs a float from a Python number via C double;
        # preserve that double rounding for large integer inputs.
        bits = bitcast[DType.uint64](number)
        if width == 4:
            bits = UInt64(f64_to_f32_bits(bits))
        elif width == 2:
            bits = UInt64(bitcast[DType.uint16](Float16(number)))
    _lin_store(destination, index, width, bits)


def sequence_write(plan: SequenceInputPlan, destination: Int, float32_target: Bool = False):
    """Write independent C-order ranges across all available host cores.

    Python scalar words occupy compact word/tag vectors; consecutive scalars
    share one segment. Array leaves stay borrowed native descriptors until
    this parallel write, avoiding packed-array and Python-number staging.
    """
    var width = 4 if float32_target else plan.itemsize
    var target_kind = CAST_KIND_FLOAT if float32_target else sequence_kind_code(plan.kind)
    var tasks = host_predict_task_count_work(plan.count, plan.count * max(width, 8))
    var chunk = host_predict_chunk(plan.count, tasks)
    var words = plan.words.unsafe_ptr()
    var kinds = plan.kinds.unsafe_ptr()
    var arrays = plan.arrays.unsafe_ptr()
    var segments = plan.segments.unsafe_ptr()
    var segment_count = len(plan.segments)
    var count = plan.count

    def _write(task: Int) {imm words, imm kinds, imm arrays, imm segments, imm segment_count,
        imm count, imm chunk, imm destination, imm target_kind, imm width}:
        var first = task * chunk
        var last = min(first + chunk, count)
        var coordinates = List[Int](length=8, fill=0)
        var low = 0
        var high = segment_count
        while low < high:  # driver-loop: binary-search the native segment containing this worker range
            var middle = (low + high) // 2
            if segments[middle].start + segments[middle].count <= first:
                low = middle + 1
            else:
                high = middle
        var segment = low
        while segment < segment_count and segments[segment].start < last:  # driver-loop: over native array/scalar spans intersecting the task
            var span = segments[segment].copy()
            var lo = max(first, span.start)
            var hi = min(last, span.start + span.count)
            for index in range(lo, hi):
                var offset = index - span.start
                var word = UInt64(0)
                var kind = 0
                if span.source < 0:
                    word = words[span.scalar_start + offset]
                    kind = Int(kinds[span.scalar_start + offset])
                else:
                    ref source = arrays[span.source]
                    var rank = len(source.layout.shape)
                    var remainder = offset
                    for reverse in range(rank):  # small-loop(sequence leaf rank is at most eight)
                        var axis = rank - reverse - 1
                        coordinates[axis] = remainder % source.layout.shape[axis]
                        remainder //= source.layout.shape[axis]
                    var address = source.layout.address
                    for axis in range(rank):  # small-loop(admitted direct or indirect buffer coordinates)
                        address += coordinates[axis] * source.layout.strides[axis]
                        if source.layout.suboffsets[axis] >= 0:
                            address = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=address)[0] + source.layout.suboffsets[axis]
                    word = cast_w64_bits(pd_raw_at(address, 0, source.layout.itemsize, source.native), source.kind, source.layout.itemsize)
                    kind = source.kind
                _sequence_store(word, kind, target_kind, width, destination, index)
            segment += 1

    if tasks <= 1:
        _write(0)
    else:
        host_parallelize(_write, tasks)
    _ = plan

