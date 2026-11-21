# SPDX-License-Identifier: Apache-2.0
"""RUN of the generation buffer door (`bindings/generation_buffers_door.mojo`): typed words only, no Python.

The id conversion, the layout words of a new Array, the shape check of a state buffer and the state copy. The door
keeps only what needs a Python object: buffer admission (UNPACK) and wrapping a native store as an Array (PACK).

`generation_ids_into` is one pass over the ids of one call (B x L token ids; it refuses at the first invalid id, by
flat index, before anything runs on them). It is not split across cores: the ids of a forward or a step are a few
thousand words, less than the cost of starting workers, and the first-invalid-index refusal is defined by the order.
"""
from std.memory import unsafe_memcpy



def generation_ids_into(
    addr: Int, item: Int, signed: Bool, b: Int, l: Int, column_major: Bool, vocab: Int,
    dst: MutPointer[Int32, MutUntrackedOrigin],
) raises:
    """Native little-endian integer ids (`item` bytes each, C order in the buffer) to int32,
    refusing any id outside [0, vocab) before conversion."""
    var bytes = Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=addr)
    for i in range(b * l):
        var source = (i // l) + (i % l) * b if column_major else i
        var raw = UInt64(0)
        for k in range(item):
            raw |= UInt64(bytes[source * item + k]) << UInt64(8 * k)
        var negative = signed and ((raw >> UInt64(8 * item - 1)) & UInt64(1)) == UInt64(1)
        if negative or raw >= UInt64(vocab):
            raise Error("generation: ids must be in [0, " + String(vocab) + "); invalid id at flat index " + String(i))
        dst[i] = Int32(Int(raw))


def generation_array_layout(dims: List[Int], itemsize: Int, mut strides: List[Int]) raises -> Int:
    """The element count of a C-order Array of `dims`, with its row-major byte strides written to `strides`
    (one word per dimension, in dimension order). Refuses a negative dimension and a count past signed 32-bit
    kernel indexing."""
    var n = 1
    for i in range(len(dims)):  # small-loop(array shape dimension words)
        if dims[i] < 0 or (dims[i] != 0 and n > 2147483647 // dims[i]):
            raise Error("generation: invalid shape or signed 32-bit kernel indexing overflow")
        n *= dims[i]
    strides.clear()
    for _ in range(len(dims)):  # small-loop(array shape dimension words)
        strides.append(0)
    var stride = itemsize
    for i in range(len(dims)):  # small-loop(array shape dimension words)
        var axis = len(dims) - i - 1
        strides[axis] = stride
        stride *= dims[axis]
    return n


def generation_shape_size(shape: List[Int], expected: List[Int]) -> Int:
    """The element count of `expected` when `shape` is exactly it, else -1."""
    if len(shape) != len(expected):
        return -1
    var size = 1
    for i in range(len(expected)):  # small-loop(state buffer shape dims)
        if shape[i] != expected[i]:
            return -1
        size *= expected[i]
    return size


def generation_shape_text(expected: List[Int]) -> String:
    """`expected` as the refusal prints it: (a, b, c)."""
    var want = String("(")
    for i in range(len(expected)):  # small-loop(state buffer shape dims)
        if i > 0:
            want += ", "
        want += String(expected[i])
    want += ")"
    return want


def generation_bytes_copy(target: Int, source: Int, count: Int):
    """One copy of `count` bytes between two live native stores (a state snapshot)."""
    var dst = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=target)
    var src = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=source)
    unsafe_memcpy(dest=dst, src=src, count=count)
