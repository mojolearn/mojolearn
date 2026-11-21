# SPDX-License-Identifier: Apache-2.0
"""Typed frames over the native POSIX transport (lane parallel-payload-transport,
2026-11-12).

UNPACK: a Python value (None, bool, int, float, str, bytes, bytearray, list,
tuple, dict with str/int keys, native Array or contiguous buffer, packed
bfloat16/int8 weight) becomes a `WireRecord` once (`core/parallel_worker_frame.mojo`); each tensor
keeps only its buffer exporter, and its bytes are written to the socket from
that exporter's storage (no payload copy for contiguous storage).
PACK: a received record becomes Python values once; each tensor's bytes are
read from the socket straight into a fresh native Array store.

Every pool sends these frames (lane pools-frame, 2026-11-21: the byte-string
node carries the serialized member states the codec path was kept for; that
path, a JSON header dict, a JSON structural descriptor and a joined bytearray
payload, is removed from the pool). A frame is serial by nature (one socket, one
byte stream): the tensors go one after another; independent workers are
addressed in parallel by the pool, not here.
"""
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_contains, py_setitem, py_type
from bindings.native_codec_door import CODEC_BOUNDARY, _codec_keys, _codec_packed_class, _codec_packed_fields, _codec_dtype, _codec_code
from bindings.native_array_snapshot_door import native_array_snapshot, native_array_slots
from bindings.native_array_storage_door import native_array_unfilled
from bindings.native_buffer_address_unpack import native_buffer_address
from bindings.native_instance_item_pack import native_instance_item
from bindings.native_transport_door import _transport_state
from bindings.native_posix_transport_unpack import posix_transfer
from bindings.native_posix_transport_run import PosixBytes
from providers.python.borrowed_buffer_pyapi_unpack import admit_native_buffer_exporter, AdmittedBorrowedBuffer
from core.parallel_worker_frame import (
    WireRecord, WireTensor, WireFrame, wire_frame_words, wire_frame_header, wire_frame_body,
    wire_dtype_word, wire_dtype_name, wire_dtype_itemsize, WIRE_FRAME_HEADER_WORDS, WIRE_RECORD_MAX_DEPTH,
    WIRE_NONE, WIRE_BOOL, WIRE_INT, WIRE_FLOAT, WIRE_TEXT, WIRE_TENSOR, WIRE_BF16, WIRE_INT8,
    WIRE_LIST, WIRE_TUPLE, WIRE_MAP, WIRE_BYTES, WIRE_BYTEARRAY, WIRE_U1,
)
from bindings.native_keys_unpack import native_keys


comptime WIRE_MAX_TENSOR_BYTES = (1 << 62)
comptime WIRE_MAX_NODES = 1000000


struct WireOutgoing(Movable):
    """A record ready to send: the typed tree and, per tensor, the buffer
    exporter whose storage holds its bytes."""
    var record: WireRecord
    var exporters: List[PythonObject]

    def __init__(out self):
        self.record = WireRecord()
        self.exporters = List[PythonObject]()


struct WireMessage(Movable):
    """A received frame after PACK: kind, sequence, the request words (empty
    for a reply) and the record as Python values."""
    var kind: Int
    var sequence: Int
    var request: List[Int64]
    var value: PythonObject

    def __init__(out self, kind: Int, sequence: Int, var request: List[Int64], value: PythonObject):
        self.kind = kind
        self.sequence = sequence
        self.request = request^
        self.value = value


def _wire_tensor(value: PythonObject, mut outgoing: WireOutgoing) raises -> Int:
    """UNPACK of one tensor: its descriptor and its exporter."""
    var c = CODEC_BOUNDARY.get_or_create_ptr()
    if c[].array_cls is None:
        raise Error("parallel wire: native storage classes are not connected")
    if py_type(value) is c[].array_cls:
        var captured = native_array_snapshot(value, Python.dict(array_class=c[].array_cls), True)
        var shape = List[Int]()
        var dims = captured.shape
        for axis in range(len(dims)):  # small-loop(tensor rank)
            shape.append(Int(py=dims[axis]))
        var order = 1 if String(captured.order) == "F" else 0
        var tensor = WireTensor(wire_dtype_word(String(captured.dtype)), order, Bool(py=captured._readonly), shape^)
        # The admitted _mv describes the Array's physical store (C or F order).
        outgoing.exporters.append(captured._mv)
        _ = captured
        return outgoing.record.tensor(tensor^)
    # Admission precedes any buffer protocol (as the codec's).
    admit_native_buffer_exporter(value)
    var pin = AdmittedBorrowedBuffer(value)
    var index = 0
    try:
        var layout = pin.layout.value().copy()
        var shape = layout.shape.copy()
        var tensor = WireTensor(wire_dtype_word(_codec_dtype(pin.format, layout.itemsize)), 0, layout.readonly, shape^)
        outgoing.exporters.append(value)
        index = outgoing.record.tensor(tensor^)
    except error:
        pin.release()
        raise error
    pin.release()
    return index


def _wire_unpack(value: PythonObject, mut outgoing: WireOutgoing, depth: Int, mut budget: Int) raises:
    budget -= 1
    if budget < 0 or depth > WIRE_RECORD_MAX_DEPTH:
        raise Error("parallel wire: value exceeds its depth or node budget")
    var c = CODEC_BOUNDARY.get_or_create_ptr()
    var bi = c[].bi
    if value is None:
        outgoing.record.none()
        return
    var cls = py_type(value)
    if cls is bi.bool:
        outgoing.record.boolean(Bool(py=value))
    elif cls is bi.int:
        outgoing.record.integer(Int(py=value))
    elif cls is bi.float:
        outgoing.record.real(Float64(py=value))
    elif cls is bi.str:
        outgoing.record.text(String(value))
    elif cls is bi.list or cls is bi.tuple:
        var count = len(value)
        outgoing.record.node(WIRE_LIST if cls is bi.list else WIRE_TUPLE, count)
        for i in range(count):  # driver-loop: over the entries of one request or reply sequence
            var item = value[i]
            _wire_unpack(item, outgoing, depth + 1, budget)
    elif cls is bi.dict:
        var keys = _codec_keys(value)
        outgoing.record.node(WIRE_MAP, len(keys))
        for i in range(len(keys)):  # driver-loop: over the admitted keys of one request or reply mapping
            var key = keys[i]
            var item = value[key]
            _wire_unpack(key, outgoing, depth + 1, budget)
            _wire_unpack(item, outgoing, depth + 1, budget)
    elif c[].bf16_cls is not None and cls is c[].bf16_cls:
        var fields = _codec_packed_fields(value, c[].bf16_cls, Python.tuple("bits", "shape"))
        var index = _wire_tensor(native_instance_item(fields, PythonObject("bits")), outgoing)
        outgoing.record.node(WIRE_BF16, index)
    elif c[].int8_cls is not None and cls is c[].int8_cls:
        var fields = _codec_packed_fields(value, c[].int8_cls, Python.tuple("codes", "exponents", "shape"))
        var index = _wire_tensor(native_instance_item(fields, PythonObject("codes")), outgoing)
        _ = _wire_tensor(native_instance_item(fields, PythonObject("exponents")), outgoing)
        outgoing.record.node(WIRE_INT8, index)
    elif cls is bi.bytes or cls is bi.bytearray:
        # The exact built-in byte strings export their own storage; the send
        # writes it in place and checks the length again under its pin.
        outgoing.exporters.append(value)
        outgoing.record.byte_string(len(value), cls is bi.bytearray)
    else:
        outgoing.record.node(WIRE_TENSOR, _wire_tensor(value, outgoing))


def wire_unpack(value: PythonObject) raises -> WireOutgoing:
    """UNPACK of one request payload or reply value into its typed record."""
    var outgoing = WireOutgoing()
    var budget = WIRE_MAX_NODES
    _wire_unpack(value, outgoing, 0, budget)
    return outgoing^


def wire_text(message: String) -> WireOutgoing:
    """The record of an error reply: one text, no tensor."""
    var outgoing = WireOutgoing()
    outgoing.record.text(message)
    return outgoing^


def _wire_socket(socket: PythonObject, mut fd: Int, mut timeout: Float64) raises:
    var state = _transport_state(socket)
    fd = state[0]
    timeout = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(state))[1]


def wire_send(socket: PythonObject, kind: Int, sequence: Int, request: List[Int64], outgoing: WireOutgoing) raises:
    """One typed frame on an owned native socket (a worker's connection)."""
    var fd = -1
    var timeout = Float64(-1)
    _wire_socket(socket, fd, timeout)
    wire_send_fd(fd, timeout, kind, sequence, request, outgoing)
    _ = socket


def wire_send_fd(fd: Int, timeout: Float64, kind: Int, sequence: Int, request: List[Int64], outgoing: WireOutgoing) raises:
    """One typed frame: header and descriptor words in one write, then each
    tensor's bytes from its exporter's storage. `fd` and `timeout` are the
    typed pool record's (bindings/native_process_pool_run.mojo) or a
    socket's; the caller keeps the descriptor open through the call."""
    if fd < 0:
        raise Error("transport: socket is closed")
    var words = wire_frame_words(kind, sequence, request, outgoing.record)
    posix_transfer(fd, PosixBytes(unsafe_from_address=Int(words.unsafe_ptr())), len(words) * 8, timeout, True)
    _ = words^
    for i in range(len(outgoing.record.tensors)):  # driver-loop: over the tensors of one frame, in descriptor order
        var expected = outgoing.record.tensors[i].byte_count
        # The export pins the storage (an Array store cannot resize while
        # exported) through the GIL-released socket writes.
        var pin = AdmittedBorrowedBuffer(outgoing.exporters[i])
        try:
            var layout = pin.layout.value().copy()
            if layout.count * layout.itemsize != expected:
                raise Error("parallel wire: tensor storage changed size before its send")
            if expected > 0:
                if layout.contiguous():
                    posix_transfer(fd, PosixBytes(unsafe_from_address=layout.address), expected, timeout, True)
                else:
                    # A strided foreign buffer: one C-order copy, as the codec made for every tensor.
                    var staged = List[UInt8](length=expected, fill=0)
                    layout.copy_to(Int(staged.unsafe_ptr()))
                    posix_transfer(fd, PosixBytes(unsafe_from_address=Int(staged.unsafe_ptr())), expected, timeout, True)
                    _ = staged^
        except error:
            pin.release()
            raise error
        pin.release()


def _wire_store(fd: Int, timeout: Float64, tensor: WireTensor) raises -> PythonObject:
    """The destination of one received tensor: an unfilled native store
    (one allocation, no Array object) that receives the tensor's bytes from
    the socket. `_wire_array` wraps it at PACK."""
    var code = _codec_code(wire_dtype_name(tensor.dtype))
    var count = tensor.byte_count // wire_dtype_itemsize(tensor.dtype)
    # lane fix-output-zero-fill (2026-11-14): unfilled; posix_transfer writes the tensor's bytes
    # (a short read raises before the store escapes).
    var store = native_array_unfilled(code, count)
    if count > 0:
        posix_transfer(fd, PosixBytes(unsafe_from_address=native_buffer_address(store)), tensor.byte_count, timeout, False)
    return store


def _wire_array(store: PythonObject, tensor: WireTensor) raises -> PythonObject:
    """PACK of one tensor: the native Array over the store that received its
    bytes (`_wire_store`)."""
    var c = CODEC_BOUNDARY.get_or_create_ptr()
    var bi = c[].bi
    var dtype = wire_dtype_name(tensor.dtype)
    var itemsize = wire_dtype_itemsize(tensor.dtype)
    var count = tensor.byte_count // itemsize
    var shape = bi.list()
    for axis in range(len(tensor.shape)):  # small-loop(tensor rank)
        _ = py_call(shape.append, PythonObject(tensor.shape[axis]))
    var strides = List[Int](length=len(tensor.shape), fill=0)
    var stride = itemsize
    for i in range(len(tensor.shape)):  # small-loop(tensor rank)
        var axis = len(tensor.shape) - i - 1 if tensor.order == 0 else i
        strides[axis] = stride
        var size = tensor.shape[axis]
        if size != 0 and stride > 9223372036854775807 // size:
            raise Error("parallel wire: tensor stride overflows native indexing")
        stride *= size
    var stride_list = bi.list()
    for axis in range(len(strides)):  # small-loop(tensor rank)
        _ = py_call(stride_list.append, PythonObject(strides[axis]))
    var slots = native_array_slots(Python.dict(array_class=c[].array_cls))
    var result = py_call(bi.object.__new__, c[].array_cls)
    var fields = Python.dict(_store=store, _base=PythonObject(None), _pin=PythonObject(None),
        _mv=py_call(bi.memoryview, store), _addr=PythonObject(native_buffer_address(store)),
        _readonly=PythonObject(tensor.readonly), shape=py_call(bi.tuple, shape), dtype=PythonObject(dtype),
        ndim=PythonObject(len(tensor.shape)), size=PythonObject(count), nbytes=PythonObject(tensor.byte_count),
        itemsize=PythonObject(itemsize), order=PythonObject("F" if tensor.order == 1 else "C"),
        strides=py_call(bi.tuple, stride_list))
    var names = py_call(bi.list, fields.keys())
    for i in range(len(names)):  # small-loop(publish admitted Array metadata through native member setters)
        var name = names[i]
        _ = py_call(slots[name].__set__, result, fields[name])
    return result


def _wire_bytes(fd: Int, timeout: Float64, tensor: WireTensor) raises -> PythonObject:
    """PACK of one byte string's storage: a bytearray that receives the
    bytes in place (`_wire_pack` hands it on, or its immutable copy, the one
    copy the codec's decoder made as well)."""
    if tensor.dtype != WIRE_U1 or len(tensor.shape) != 1 or tensor.order != 0:
        raise Error("parallel wire: a byte string requires a one-dimensional uint8 descriptor")
    var bi = CODEC_BOUNDARY.get_or_create_ptr()[].bi
    var store = py_call(bi.bytearray, PythonObject(tensor.byte_count))
    if tensor.byte_count > 0:
        posix_transfer(fd, PosixBytes(unsafe_from_address=native_buffer_address(store)), tensor.byte_count, timeout, False)
    return store


def _wire_pack(record: WireRecord, mut cursor: Int, tensors: List[PythonObject], mut next_tensor: Int,
               depth: Int) raises -> PythonObject:
    if cursor >= record.node_count() or depth > WIRE_RECORD_MAX_DEPTH:
        raise Error("parallel wire: record tree exceeds its nodes or depth")
    var c = CODEC_BOUNDARY.get_or_create_ptr()
    var bi = c[].bi
    var at = cursor
    cursor += 1
    var tag = record.tag(at)
    var word = record.value(at)
    if tag == WIRE_NONE:
        return PythonObject(None)
    if tag == WIRE_BOOL:
        return PythonObject(word != 0)
    if tag == WIRE_INT:
        return PythonObject(word)
    if tag == WIRE_FLOAT:
        return PythonObject(record.real_value(at))
    if tag == WIRE_TEXT:
        return PythonObject(record.text_value(at))
    if tag == WIRE_TENSOR or tag == WIRE_BF16 or tag == WIRE_INT8 or tag == WIRE_BYTES or tag == WIRE_BYTEARRAY:
        # Tensors are claimed in descriptor order, each exactly once.
        var width = 2 if tag == WIRE_INT8 else 1
        if word != next_tensor or word + width > len(tensors):
            raise Error("parallel wire: tensor references must follow descriptor order")
        next_tensor += width
        # A byte-string node names a bytearray store and nothing else does:
        # the receive built each store from the node kinds (`_wire_byte_strings`).
        var stored = py_type(tensors[word]) is bi.bytearray
        if stored != (tag == WIRE_BYTES or tag == WIRE_BYTEARRAY):
            raise Error("parallel wire: byte string and tensor references are mixed")
        if tag == WIRE_BYTEARRAY:
            return tensors[word]
        if tag == WIRE_BYTES:
            return py_call(bi.bytes, tensors[word])
        if tag == WIRE_TENSOR:
            return tensors[word]
        var nk = native_keys()
        if tag == WIRE_BF16:
            if c[].bf16_cls is None:
                raise Error("parallel wire: bfloat16 weight class is not connected")
            var bits = tensors[word]
            if String(bits.dtype) != "<u2" or len(bits.shape) != 2:
                raise Error("parallel wire: bfloat16 weights require a two-dimensional uint16 tensor")
            _codec_packed_class(c[].bf16_cls, Python.tuple("bits", "shape"))
            var packed = py_call(bi.object.__new__, c[].bf16_cls)
            var fields = _codec_packed_fields(packed, c[].bf16_cls, Python.tuple("bits", "shape"))
            py_setitem(fields, nk[].k_bits, bits)
            py_setitem(fields, nk[].k_shape, bits.shape)
            return packed
        if c[].int8_cls is None:
            raise Error("parallel wire: int8 weight class is not connected")
        var codes = tensors[word]
        var exponents = tensors[word + 1]
        if String(codes.dtype) != "<i1" or len(codes.shape) != 2:
            raise Error("parallel wire: int8 weights require a two-dimensional int8 tensor")
        if String(exponents.dtype) != "<i4" or len(exponents.shape) != 1 or Int(py=exponents.shape[0]) != Int(py=codes.shape[0]):
            raise Error("parallel wire: int8 exponents must be int32 with one element per row")
        _codec_packed_class(c[].int8_cls, Python.tuple("codes", "exponents", "shape"))
        var packed = py_call(bi.object.__new__, c[].int8_cls)
        var fields = _codec_packed_fields(packed, c[].int8_cls, Python.tuple("codes", "exponents", "shape"))
        py_setitem(fields, nk[].k_codes, codes)
        py_setitem(fields, nk[].k_exponents, exponents)
        py_setitem(fields, nk[].k_shape, codes.shape)
        return packed
    if tag == WIRE_LIST or tag == WIRE_TUPLE:
        if word < 0 or word > record.node_count() - cursor:
            raise Error("parallel wire: sequence length outside its record")
        var items = bi.list()
        for _ in range(word):  # driver-loop: over the entries of one received sequence
            _ = py_call(items.append, _wire_pack(record, cursor, tensors, next_tensor, depth + 1))
        if tag == WIRE_TUPLE:
            return py_call(bi.tuple, items)
        return items
    if tag == WIRE_MAP:
        if word < 0 or 2 * word > record.node_count() - cursor:
            raise Error("parallel wire: mapping length outside its record")
        var result = bi.dict()
        for _ in range(word):  # driver-loop: over the pairs of one received mapping
            var key = _wire_pack(record, cursor, tensors, next_tensor, depth + 1)
            var key_cls = py_type(key)
            if key_cls is not bi.str and key_cls is not bi.int and key_cls is not bi.bool:
                raise Error("parallel wire: invalid mapping key")
            if py_contains(result, key):
                raise Error("parallel wire: duplicate mapping key")
            py_setitem(result, key, _wire_pack(record, cursor, tensors, next_tensor, depth + 1))
        return result
    raise Error("parallel wire: unknown record tag " + String(tag))


def _wire_byte_strings(record: WireRecord) raises -> List[Bool]:
    """Per tensor descriptor: whether a byte-string node names it. One pass
    over the node pairs, which carry no nesting a tag scan could misread
    (every pair is a node; containers hold only counts)."""
    var marked = List[Bool](length=len(record.tensors), fill=False)
    for i in range(record.node_count()):  # small-loop(record nodes: one pair per value of one frame)
        var tag = record.tag(i)
        if tag == WIRE_BYTES or tag == WIRE_BYTEARRAY:
            var index = record.value(i)
            if index < 0 or index >= len(marked):
                raise Error("parallel wire: byte string index outside its record")
            marked[index] = True
    return marked^


# Removed 2026-11-21 (lane pools-frame): `wire_receive(socket)`, the PACKing
# receive of a worker's connection. Its last caller, the causal worker loop,
# reads typed turns (`process_worker_next`, bindings/native_process_pool_door.mojo).


struct WireReply(Movable):
    """A received frame before PACK: kind, sequence, the request words (empty
    for a reply), the typed record and, per tensor descriptor, the store that
    received its bytes (an unfilled native array store, or a bytearray for a
    byte-string node). The stores are the destinations of the socket reads
    and nothing else: no Array object, no Python container and no record
    value exists until `wire_reply_pack`."""
    var kind: Int
    var sequence: Int
    var request: List[Int64]
    var record: WireRecord
    var stores: List[PythonObject]

    def __init__(out self, kind: Int, sequence: Int):
        self.kind = kind
        self.sequence = sequence
        self.request = List[Int64]()
        self.record = WireRecord()
        self.stores = List[PythonObject]()

    def text(self) raises -> String:
        """The text of an error reply ("" when the record is not one text)."""
        if self.record.node_count() == 1 and self.record.tag(0) == WIRE_TEXT:
            return self.record.text_value(0)
        return String()


def wire_receive_typed_fd(fd: Int, timeout: Float64) raises -> WireReply:
    """One typed frame, not PACKed: header, descriptor words, then each
    tensor's bytes into its own store. A wave collects these and PACKs them
    once (`wire_reply_pack`), after its last reply."""
    if fd < 0:
        raise Error("transport: socket is closed")
    var head = List[Int64](length=WIRE_FRAME_HEADER_WORDS, fill=0)
    posix_transfer(fd, PosixBytes(unsafe_from_address=Int(head.unsafe_ptr())), WIRE_FRAME_HEADER_WORDS * 8, timeout, False)
    var header = wire_frame_header(head, WIRE_MAX_TENSOR_BYTES)
    var body = List[Int64](length=header.body_words(), fill=0)
    if len(body) > 0:
        posix_transfer(fd, PosixBytes(unsafe_from_address=Int(body.unsafe_ptr())), len(body) * 8, timeout, False)
    var frame = wire_frame_body(header, body)
    _ = body^
    var reply = WireReply(frame.kind, frame.sequence)
    # The frame's words and record change owner (no copy); the frame keeps empty ones.
    swap(reply.request, frame.request)
    swap(reply.record, frame.record)
    var byte_strings = _wire_byte_strings(reply.record)
    for i in range(len(reply.record.tensors)):  # driver-loop: over the tensors of one frame, in descriptor order
        if byte_strings[i]:
            reply.stores.append(_wire_bytes(fd, timeout, reply.record.tensors[i]))
        else:
            reply.stores.append(_wire_store(fd, timeout, reply.record.tensors[i]))
    return reply^


def wire_reply_pack(reply: WireReply) raises -> PythonObject:
    """PACK of one received record, once: each tensor store becomes its
    native Array (a byte-string store is handed on as it is), then the record
    tree becomes Python values."""
    var byte_strings = _wire_byte_strings(reply.record)
    var tensors = List[PythonObject](capacity=len(reply.stores))
    for i in range(len(reply.stores)):  # driver-loop: over the tensors of one frame, in descriptor order
        if byte_strings[i]:
            tensors.append(reply.stores[i])
        else:
            tensors.append(_wire_array(reply.stores[i], reply.record.tensors[i]))
    var value = PythonObject(None)
    if reply.record.node_count() > 0:
        var cursor = 0
        var next_tensor = 0
        value = _wire_pack(reply.record, cursor, tensors, next_tensor, 0)
        if cursor != reply.record.node_count() or next_tensor != len(tensors):
            raise Error("parallel wire: unclaimed nodes or tensors at the end of the record")
    elif len(tensors) != 0:
        raise Error("parallel wire: tensors without a record")
    return value


def wire_record_pack(record: WireRecord) raises -> PythonObject:
    """PACK of a record kept typed after its frame (a worker's identity
    reply in the shard trace). Such a record names no tensor."""
    if len(record.tensors) != 0:
        raise Error("parallel wire: a kept record names no tensor")
    var value = PythonObject(None)
    if record.node_count() > 0:
        var tensors = List[PythonObject]()
        var cursor = 0
        var next_tensor = 0
        value = _wire_pack(record, cursor, tensors, next_tensor, 0)
        if cursor != record.node_count():
            raise Error("parallel wire: unclaimed nodes at the end of the record")
    return value


def wire_receive_fd(fd: Int, timeout: Float64) raises -> WireMessage:
    """One typed frame and the PACK of its record (one addressed request's
    reply, a worker's next request)."""
    var reply = wire_receive_typed_fd(fd, timeout)
    var value = wire_reply_pack(reply)
    var words = List[Int64]()
    swap(words, reply.request)
    return WireMessage(reply.kind, reply.sequence, words^, value)


def wire_append(value: PythonObject, mut outgoing: WireOutgoing) raises:
    """UNPACK of one value appended under a node the caller already wrote
    (a reply record built field by field, without a Python container)."""
    var budget = WIRE_MAX_NODES
    _wire_unpack(value, outgoing, 1, budget)


def wire_key(mut outgoing: WireOutgoing, key: String):
    """A text mapping key under a WIRE_MAP node the caller wrote."""
    outgoing.record.text(key)
