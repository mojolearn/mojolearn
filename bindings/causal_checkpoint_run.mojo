# SPDX-License-Identifier: Apache-2.0
"""Typed decisions of the safetensors checkpoint reader (RUN, no Python).

`checkpoint_header_length` decodes the 8-byte little-endian header length and
bounds it; `checkpoint_offsets_check` bounds one tensor's data offsets inside
the data section; `checkpoint_info_count` validates a TensorInfo's shape,
offsets and dtype size and returns its element count; `checkpoint_read_plan`
admits a read (dtype by name, the widen / bits form, offsets against the
retained file) and names the storage code and the conversion;
`checkpoint_row_major` gives a C-order shape's element count and byte strides.
The half-to-float widening itself is `core/causal_checkpoint_impl.mojo`.

The boundary door (`bindings/causal_checkpoint_door.mojo`) is UNPACK and PACK:
it reads the file, the JSON header and the TensorInfo records into the plain
values these take, and publishes TensorInfo records and Arrays from the words
they return.

Parallel plan: none here. Each function is a handful of integer checks for
one tensor; the door's loops over header entries and tensor names are the
units, and each publishes a Python record under the GIL.
"""

comptime _INT_MAX = 9223372036854775807
comptime CHECKPOINT_HEADER_LIMIT = 100 * 1024 * 1024


def checkpoint_header_length(head: List[UInt8]) raises -> Int:
    """The header length of a safetensors file from its first eight bytes."""
    if len(head) != 8:
        raise Error("safetensors: file is shorter than its 8-byte header length")
    var length = UInt64(0)
    for i in range(8):  # driver-loop: over the little-endian 8-byte header length
        length |= UInt64(head[i]) << UInt64(8 * i)
    if length > UInt64(CHECKPOINT_HEADER_LIMIT):
        raise Error("safetensors: header length exceeds 104857600")
    return Int(length)


def checkpoint_offsets_check(name: String, begin: Int, end: Int, total: Int, header: Int) raises:
    """One tensor's `data_offsets` lie inside the data section of a `total`-byte file."""
    if begin < 0 or end < begin or end > total - 8 - header:
        raise Error("safetensors: tensor " + name + " offsets fall outside the data section")


def checkpoint_itemsize(dtype: String) -> Int:
    """Bytes per element of a safetensors dtype the reader sizes (0 when unsized)."""
    return 8 if dtype == "I64" else (4 if dtype == "F32" or dtype == "I32" else (2 if dtype == "BF16" or dtype == "F16" else 0))


def checkpoint_info_count(dtype: String, shape: List[Int], begin: Int, end: Int) raises -> Int:
    """A TensorInfo's element count, after its shape, offsets and byte size check."""
    var count = 1
    for size in shape:  # driver-loop: over tensor shape metadata
        if size < 0 or (size != 0 and count > _INT_MAX // size):
            raise Error("TensorInfo shape must hold non-negative dimensions")
        count *= size
    if begin < 0 or end < begin:
        raise Error("TensorInfo offsets must satisfy 0 <= begin <= end")
    var itemsize = checkpoint_itemsize(dtype)
    if itemsize != 0 and (count > _INT_MAX // itemsize or count * itemsize != end - begin):
        raise Error("TensorInfo bytes differ from dtype/shape")
    return count


@fieldwise_init
struct CheckpointReadPlan(Copyable, Movable):
    """How one admitted tensor is read: storage code, absolute byte range,
    whether it is widened to float32 or returned as packed bfloat16 bits."""
    var code: String
    var first: Int
    var last: Int
    var widen: Bool
    var packed_bits: Bool


def checkpoint_read_plan(name: String, dtype: String, format: String, begin: Int, end: Int, data_start: Int,
                         data_size: Int, ndim: Int) raises -> CheckpointReadPlan:
    """Admit one read and name its storage and conversion."""
    if format != "widen" and format != "bits":
        raise Error("safetensors: bf16 must be widen or bits")
    if dtype != "F32" and dtype != "F16" and dtype != "BF16" and dtype != "I32" and dtype != "I64":
        raise Error("TypeError: tensor " + name + " dtype " + dtype + " refuses by name (the reader admits F32, F16, BF16, I32 and I64)")
    if data_start < 0 or data_size < 0 or end > data_size or data_start > _INT_MAX - end:
        raise Error("ValueError: safetensors data offsets exceed the retained file")
    var code = "f" if dtype == "F32" else ("q" if dtype == "I64" else ("i" if dtype == "I32" else "H"))
    var widen = dtype == "F16" or (dtype == "BF16" and format == "widen")
    var packed = dtype == "BF16" and format == "bits" and ndim == 2
    return CheckpointReadPlan(code, data_start + begin, data_start + end, widen, packed)


def checkpoint_mapped_check(plan: CheckpointReadPlan, mapped: Int) raises:
    """The retained mapping still covers the admitted range."""
    if plan.last > mapped:
        raise Error("ValueError: safetensors file changed after header admission")


def checkpoint_row_major(shape: List[Int], itemsize: Int, stored: Int) raises -> List[Int]:
    """Byte strides of a C-order `shape` whose element count must equal `stored`."""
    var n = len(shape)
    var strides = List[Int](length=n, fill=0)
    var count = 1
    var stride = itemsize
    for i in range(n):  # driver-loop: over tensor shape and row-major stride metadata
        var size = shape[n - i - 1]
        if size < 0 or (size != 0 and stride > _INT_MAX // size):
            raise Error("safetensors: tensor shape exceeds native indexing")
        count *= size
        strides[n - i - 1] = stride
        stride *= size
    if count != stored:
        raise Error("safetensors: tensor shape does not match stored bytes")
    return strides^
