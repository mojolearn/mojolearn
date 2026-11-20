# SPDX-License-Identifier: Apache-2.0
"""The typed frame of a parallel worker RPC (lane parallel-payload-transport,
2026-11-12).

A frame carries one typed request (`core/parallel_worker_request.mojo`) or one
reply, and a typed record: the value tree that rides with it, as plain Int64
words. Tensors travel as a descriptor (dtype word, order word, readonly word,
shape words, byte length) followed by their raw bytes, written straight from
the sender's storage and read straight into the receiver's. No Python object,
dictionary, tuple or JSON text crosses the pipe; the bindings side only
UNPACKs a Python value into a `WireRecord` before the send and PACKs a
received record into Python values after it
(`bindings/native_process_wire.mojo`).

Wire layout, native little-endian Int64 words (local workers on one host):
  header  [MAGIC, LAYOUT, kind, sequence, request_words, node_words,
           text_count, text_words, tensor_count, descriptor_words, tensor_bytes]
  body    request words | nodes | text lengths | text bytes (zero padded to
          8) | tensor descriptors
  tensors the raw bytes of each tensor, in descriptor order, back to back.

The record is a prefix-order tree of (tag, value) node pairs:
  NONE 0 | BOOL 0/1 | INT value | FLOAT the Float64 bits | TEXT text index |
  TENSOR tensor index | BF16 tensor index of its uint16 bits |
  INT8 tensor index of its int8 codes (its int32 exponents are the next
  tensor) | LIST/TUPLE child count | MAP pair count (key node, value node).

Encoding and decoding walk a few header words and one node pair per value:
no independent work to split across cores. The tensor bytes are one
transfer per tensor.
"""
from std.memory import unsafe_memcpy, bitcast

comptime WIRE_FRAME_MAGIC = 0x46574C4D  # "MLWF": low byte nonzero, unlike a JSON frame's length prefix
comptime WIRE_FRAME_LAYOUT = 1
comptime WIRE_FRAME_HEADER_WORDS = 11
# Bounds a decoded body before allocation (512 MiB of descriptor words).
comptime WIRE_FRAME_MAX_BODY_WORDS = 1 << 26
comptime WIRE_RECORD_MAX_DEPTH = 64
comptime WIRE_TENSOR_MAX_RANK = 64

# Frame kinds.
comptime WIRE_REQUEST = 1
comptime WIRE_RESULT = 2
comptime WIRE_ERROR = 3
comptime WIRE_CLOSE = 4

# Record node tags.
comptime WIRE_NONE = 0
comptime WIRE_BOOL = 1
comptime WIRE_INT = 2
comptime WIRE_FLOAT = 3
comptime WIRE_TEXT = 4
comptime WIRE_TENSOR = 5
comptime WIRE_BF16 = 6
comptime WIRE_INT8 = 7
comptime WIRE_LIST = 8
comptime WIRE_TUPLE = 9
comptime WIRE_MAP = 10

# Tensor dtype words (the native Array dtype strings they stand for).
comptime WIRE_F4 = 1   # "<f4"
comptime WIRE_F8 = 2   # "<f8"
comptime WIRE_I4 = 3   # "<i4"
comptime WIRE_I8 = 4   # "<i8"
comptime WIRE_U4 = 5   # "<u4"
comptime WIRE_U1 = 6   # "<u1"
comptime WIRE_F2 = 7   # "<f2"
comptime WIRE_U2 = 8   # "<u2"
comptime WIRE_I1 = 9   # "<i1"


def wire_dtype_word(dtype: String) raises -> Int:
    if dtype == "<f4":
        return WIRE_F4
    if dtype == "<f8":
        return WIRE_F8
    if dtype == "<i4":
        return WIRE_I4
    if dtype == "<i8":
        return WIRE_I8
    if dtype == "<u4":
        return WIRE_U4
    if dtype == "<u1":
        return WIRE_U1
    if dtype == "<f2":
        return WIRE_F2
    if dtype == "<u2":
        return WIRE_U2
    if dtype == "<i1":
        return WIRE_I1
    raise Error("parallel wire: unsupported tensor dtype " + dtype)


def wire_dtype_name(word: Int) raises -> String:
    if word == WIRE_F4:
        return "<f4"
    if word == WIRE_F8:
        return "<f8"
    if word == WIRE_I4:
        return "<i4"
    if word == WIRE_I8:
        return "<i8"
    if word == WIRE_U4:
        return "<u4"
    if word == WIRE_U1:
        return "<u1"
    if word == WIRE_F2:
        return "<f2"
    if word == WIRE_U2:
        return "<u2"
    if word == WIRE_I1:
        return "<i1"
    raise Error("parallel wire: unknown tensor dtype word " + String(word))


def wire_dtype_itemsize(word: Int) raises -> Int:
    if word == WIRE_F8 or word == WIRE_I8:
        return 8
    if word == WIRE_F4 or word == WIRE_I4 or word == WIRE_U4:
        return 4
    if word == WIRE_F2 or word == WIRE_U2:
        return 2
    if word == WIRE_U1 or word == WIRE_I1:
        return 1
    raise Error("parallel wire: unknown tensor dtype word " + String(word))


struct WireTensor(Copyable, Movable):
    """One tensor's descriptor. `order` is 0 for C order and 1 for F order:
    the bytes are the tensor's physical store in that order."""
    var dtype: Int
    var order: Int
    var readonly: Bool
    var shape: List[Int]
    var byte_count: Int

    def __init__(out self, dtype: Int, order: Int, readonly: Bool, var shape: List[Int]) raises:
        if order != 0 and order != 1:
            raise Error("parallel wire: tensor order must be C or F")
        if len(shape) > WIRE_TENSOR_MAX_RANK:
            raise Error("parallel wire: tensor rank exceeds its bound")
        var itemsize = wire_dtype_itemsize(dtype)
        var count = 1
        for axis in range(len(shape)):  # small-loop(tensor rank)
            if shape[axis] < 0 or (shape[axis] != 0 and count > 9223372036854775807 // shape[axis]):
                raise Error("parallel wire: invalid tensor shape")
            count *= shape[axis]
        if count > 9223372036854775807 // itemsize:
            raise Error("parallel wire: tensor byte count overflows")
        self.dtype = dtype
        self.order = order
        self.readonly = readonly
        self.shape = shape^
        self.byte_count = count * itemsize


struct WireRecord(Copyable, Movable):
    """A typed value tree: node pairs, the texts they name and the tensors
    they name."""
    var nodes: List[Int64]
    var texts: List[String]
    var tensors: List[WireTensor]

    def __init__(out self):
        self.nodes = List[Int64]()
        self.texts = List[String]()
        self.tensors = List[WireTensor]()

    def node(mut self, tag: Int, value: Int):
        self.nodes.append(Int64(tag))
        self.nodes.append(Int64(value))

    def none(mut self):
        self.node(WIRE_NONE, 0)

    def boolean(mut self, value: Bool):
        self.node(WIRE_BOOL, 1 if value else 0)

    def integer(mut self, value: Int):
        self.node(WIRE_INT, value)

    def real(mut self, value: Float64):
        self.node(WIRE_FLOAT, Int(bitcast[DType.int64](value)))

    def text(mut self, value: String):
        self.node(WIRE_TEXT, len(self.texts))
        self.texts.append(value.copy())

    def tensor(mut self, var value: WireTensor) -> Int:
        """Appends a tensor descriptor (no node); returns its index."""
        self.tensors.append(value^)
        return len(self.tensors) - 1

    def node_count(self) -> Int:
        return len(self.nodes) // 2

    def tag(self, index: Int) -> Int:
        return Int(self.nodes[2 * index])

    def value(self, index: Int) -> Int:
        return Int(self.nodes[2 * index + 1])

    def real_value(self, index: Int) -> Float64:
        return bitcast[DType.float64](self.nodes[2 * index + 1])

    def text_value(self, index: Int) raises -> String:
        var t = self.value(index)
        if t < 0 or t >= len(self.texts):
            raise Error("parallel wire: text index outside its record")
        return self.texts[t].copy()


def wire_text_record(message: String) -> WireRecord:
    """The record of an error reply: one text."""
    var record = WireRecord()
    record.text(message)
    return record^


def _wire_text_words(text: String) -> Int:
    return (text.byte_length() + 7) // 8


def wire_frame_words(kind: Int, sequence: Int, request: List[Int64], record: WireRecord) raises -> List[Int64]:
    """Header and body words of one frame: everything but the tensor bytes,
    which the sender writes after these words from each tensor's storage."""
    var text_words = 0
    for i in range(len(record.texts)):  # small-loop(record texts: option keys and names)
        text_words += _wire_text_words(record.texts[i])
    var descriptor_words = 0
    var tensor_bytes = 0
    for i in range(len(record.tensors)):  # small-loop(record tensors: one descriptor each)
        descriptor_words += 5 + len(record.tensors[i].shape)
        tensor_bytes += record.tensors[i].byte_count
    var body = len(request) + len(record.nodes) + len(record.texts) + text_words + descriptor_words
    if body > WIRE_FRAME_MAX_BODY_WORDS:
        raise Error("parallel wire: frame descriptors exceed their bound")
    var words = List[Int64](capacity=WIRE_FRAME_HEADER_WORDS + body)
    words.append(Int64(WIRE_FRAME_MAGIC))
    words.append(Int64(WIRE_FRAME_LAYOUT))
    words.append(Int64(kind))
    words.append(Int64(sequence))
    words.append(Int64(len(request)))
    words.append(Int64(len(record.nodes)))
    words.append(Int64(len(record.texts)))
    words.append(Int64(text_words))
    words.append(Int64(len(record.tensors)))
    words.append(Int64(descriptor_words))
    words.append(Int64(tensor_bytes))
    for i in range(len(request)):  # small-loop(request words)
        words.append(request[i])
    for i in range(len(record.nodes)):  # small-loop(record node words: two per value)
        words.append(record.nodes[i])
    for i in range(len(record.texts)):  # small-loop(record text lengths)
        words.append(Int64(record.texts[i].byte_length()))
    for i in range(len(record.texts)):  # small-loop(record texts packed into zero-padded words)
        var bytes = record.texts[i].as_bytes()
        var start = len(words)
        for _ in range(_wire_text_words(record.texts[i])):  # small-loop(words of one text)
            words.append(Int64(0))
        var target = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(words.unsafe_ptr()) + start * 8)
        for b in range(len(bytes)):  # small-loop(bytes of one short text)
            target[b] = bytes[b]
    for i in range(len(record.tensors)):  # small-loop(record tensor descriptors)
        ref t = record.tensors[i]
        words.append(Int64(t.dtype))
        words.append(Int64(t.order))
        words.append(Int64(1 if t.readonly else 0))
        words.append(Int64(len(t.shape)))
        for axis in range(len(t.shape)):  # small-loop(tensor rank)
            words.append(Int64(t.shape[axis]))
        words.append(Int64(t.byte_count))
    return words^


@fieldwise_init
struct WireFrameHeader(Copyable, Movable):
    var kind: Int
    var sequence: Int
    var request_words: Int
    var node_words: Int
    var text_count: Int
    var text_words: Int
    var tensor_count: Int
    var descriptor_words: Int
    var tensor_bytes: Int

    def body_words(self) -> Int:
        return self.request_words + self.node_words + self.text_count + self.text_words + self.descriptor_words


def wire_frame_header(words: List[Int64], max_tensor_bytes: Int) raises -> WireFrameHeader:
    """The admitted header from its WIRE_FRAME_HEADER_WORDS words."""
    if len(words) != WIRE_FRAME_HEADER_WORDS:
        raise Error("parallel wire: truncated frame header")
    if Int(words[0]) != WIRE_FRAME_MAGIC or Int(words[1]) != WIRE_FRAME_LAYOUT:
        raise Error("parallel wire: frame layout differs from this process")
    var kind = Int(words[2])
    if kind < WIRE_REQUEST or kind > WIRE_CLOSE:
        raise Error("parallel wire: unknown frame kind")
    var header = WireFrameHeader(kind=kind, sequence=Int(words[3]), request_words=Int(words[4]),
        node_words=Int(words[5]), text_count=Int(words[6]), text_words=Int(words[7]),
        tensor_count=Int(words[8]), descriptor_words=Int(words[9]), tensor_bytes=Int(words[10]))
    if (header.request_words < 0 or header.node_words < 0 or header.node_words % 2 != 0 or header.text_count < 0
            or header.text_words < 0 or header.tensor_count < 0 or header.descriptor_words < 0):
        raise Error("parallel wire: negative frame extent")
    var limit = WIRE_FRAME_MAX_BODY_WORDS
    if (header.request_words > limit or header.node_words > limit or header.text_count > limit
            or header.text_words > limit or header.descriptor_words > limit or header.body_words() > limit):
        raise Error("parallel wire: frame descriptors exceed their bound")
    if header.tensor_bytes < 0 or header.tensor_bytes > max_tensor_bytes:
        raise Error("parallel wire: frame tensor bytes exceed their bound")
    return header^


struct WireFrame(Movable):
    """A received frame before its tensor bytes are read: the request words
    (empty for a reply) and the record with its tensor descriptors."""
    var kind: Int
    var sequence: Int
    var request: List[Int64]
    var record: WireRecord

    def __init__(out self, kind: Int, sequence: Int):
        self.kind = kind
        self.sequence = sequence
        self.request = List[Int64]()
        self.record = WireRecord()


def wire_frame_body(header: WireFrameHeader, body: List[Int64]) raises -> WireFrame:
    """The frame whose body words were read after `header`; checks every
    extent against the body and the header's tensor byte total."""
    if len(body) != header.body_words():
        raise Error("parallel wire: frame body length differs from its header")
    var frame = WireFrame(header.kind, header.sequence)
    var at = 0
    for i in range(header.request_words):  # small-loop(request words)
        frame.request.append(body[at + i])
    at += header.request_words
    for i in range(header.node_words):  # small-loop(record node words)
        frame.record.nodes.append(body[at + i])
    at += header.node_words
    var lengths = List[Int](capacity=header.text_count)
    var text_words = 0
    for i in range(header.text_count):  # small-loop(record text lengths)
        var n = Int(body[at + i])
        if n < 0 or n > header.text_words * 8:
            raise Error("parallel wire: text length outside its frame")
        lengths.append(n)
        text_words += (n + 7) // 8
    at += header.text_count
    if text_words != header.text_words:
        raise Error("parallel wire: text words differ from their lengths")
    var base = Int(body.unsafe_ptr())
    for i in range(header.text_count):  # small-loop(record texts)
        if lengths[i] == 0:
            frame.record.texts.append(String(""))
        else:
            var bytes = Span(unsafe_ptr=Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=base + at * 8), length=lengths[i])
            frame.record.texts.append(String(from_utf8=bytes))
        at += (lengths[i] + 7) // 8
    var end = at + header.descriptor_words
    var total = 0
    for _ in range(header.tensor_count):  # small-loop(record tensor descriptors)
        if at + 4 > end:
            raise Error("parallel wire: truncated tensor descriptor")
        var dtype = Int(body[at])
        var order = Int(body[at + 1])
        var readonly = Int(body[at + 2]) != 0
        var rank = Int(body[at + 3])
        if rank < 0 or rank > WIRE_TENSOR_MAX_RANK or at + 5 + rank > end:
            raise Error("parallel wire: truncated tensor descriptor")
        var shape = List[Int](capacity=rank)
        for axis in range(rank):  # small-loop(tensor rank)
            shape.append(Int(body[at + 4 + axis]))
        var tensor = WireTensor(dtype, order, readonly, shape^)
        if tensor.byte_count != Int(body[at + 4 + rank]):
            raise Error("parallel wire: tensor byte length differs from its shape and dtype")
        if tensor.byte_count > header.tensor_bytes - total:
            raise Error("parallel wire: tensor bytes exceed the frame total")
        total += tensor.byte_count
        _ = frame.record.tensor(tensor^)
        at += 5 + rank
    if at != end or total != header.tensor_bytes:
        raise Error("parallel wire: tensor descriptors differ from the frame header")
    _ = body
    return frame^
