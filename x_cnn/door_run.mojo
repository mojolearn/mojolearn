# SPDX-License-Identifier: Apache-2.0
"""RUN half shared by the CNN layer doors (lane doorsplit-cnn-xseq,
2026-11-11; docs/plans/PYTHON_AT_THE_BOUNDARY.md section 2.2).

The doors in `bindings/xcnn_*.mojo` are the UNPACK/PACK boundary: they read
the layer's Python fields, admit the caller's arrays and publish the results.
Everything they hand to this directory is a typed Mojo value: the route of the
layer's connected x_cnn image (`XcnnRoute`: its native table entry, resolved
once per door operation as an `EntryRef`, and the lane switch words the bridge
recorded at import), plain address and handle words,
shape words and scalars. No module under `x_cnn/` imports Python
(`tools/python_free_dirs.sh`).

GIL: RUN never touches the GIL. The boundary releases it around a RUN call
(`bindings/xcnn_common.mojo` `xcnn_call_f`), exactly where the door released
it before the split, so the release points are unchanged.
"""
from std.memory import unsafe_memcpy
from core.entry_table import EntryRef, entry_ref_call

comptime XCNN_F_ADAPT = 1
comptime XCNN_F_GRAPH = 2
comptime XCNN_F_PAD = 4
comptime XCNN_F_GROUP = 8
comptime XCNN_F_LAYER_IO = 16
comptime XCNN_F_PIN = 32
comptime XCNN_F_RESIDENT = 128
# `x_cnn_idn2_flags` bits: `-D MOJOLEARN_IDN_ALL_OFF` sets 16
comptime XCNN_F2_ALL_OFF = 16

comptime XcnnF32P = MutPointer[Float32, MutUntrackedOrigin]


@fieldwise_init
struct XcnnRoute(Copyable, Movable):
    """The layer's connected x_cnn image as typed words (UNPACK reads them
    once from the bridge's entries record): the resolved native table entry
    (lane entry-table-zero-alloc: the slot is read once, never per call), whether
    the image has the mixed-residency (`_m`) operations, the two lane switch
    words and the import-time environment switch of the resident I/O route."""
    var entry: EntryRef
    var mixed: Bool
    var idn: Int
    var idn2: Int
    var io_off_env: Bool

    def has(self, bit: Int) -> Bool:
        return (self.idn & bit) != 0

    def has2(self, bit: Int) -> Bool:
        return (self.idn2 & bit) != 0

    def device_io_off(self) -> Bool:
        """The resident I/O route is off: the environment switches read at
        import, or a `-D MOJOLEARN_IDN_ALL_OFF` build of the binding."""
        return self.io_off_env or self.has2(XCNN_F2_ALL_OFF)

    def resident(self) -> Bool:
        """Whether the binding runs the G3 resident glue."""
        return self.mixed and self.has(XCNN_F_RESIDENT) and not self.device_io_off()


def xcnn_run_call_f(entry: EntryRef, op: Int, addrs: List[Int], ints: List[Int],
                    floats: List[Float64]) raises -> Int:
    """One x_cnn operation (`XC_*`) through the image's resolved native
    entry: one indirect call. Plain words only; the caller keeps every
    addressed buffer alive and has released the GIL."""
    return entry_ref_call(entry, op, addrs, ints, floats)


def xcnn_run_call(entry: EntryRef, op: Int, addrs: List[Int], ints: List[Int]) raises -> Int:
    """`xcnn_run_call_f` for the operations that read no Float64 scalar (no
    empty float List is built)."""
    var result = entry_ref_call(entry, op, Int(addrs.unsafe_ptr()), len(addrs), Int(ints.unsafe_ptr()), len(ints), 0, 0)
    # Raw span addresses do not themselves retain their List owners.
    _ = addrs
    _ = ints
    return result


# ---------------------------------------------------------------- shape words

def xcnn_size(shape: List[Int]) -> Int:
    var n = 1
    for i in range(len(shape)):  # driver-loop: over shape words
        n *= shape[i]
    return n


def xcnn_shape_text(shape: List[Int]) -> String:
    var text = String("(")
    for i in range(len(shape)):  # driver-loop: over shape words of a refusal message
        if i > 0:
            text += ", "
        text += String(shape[i])
    if len(shape) == 1:
        text += ","
    return text + ")"


def xcnn_run_reshape_check(old: List[Int], shape: List[Int]) raises:
    """`DeviceTensor.reshape`'s element-count refusal."""
    if xcnn_size(old) != xcnn_size(shape):
        raise Error("mojolearn: cannot reshape " + xcnn_shape_text(old) + " to " + xcnn_shape_text(shape))


def xcnn_run_infer_dims(var dims: List[Int], total: Int) raises -> List[Int]:
    """`DeviceTensor.reshape` shape inference: at most one -1 word, filled
    from the element count `total` of the tensor."""
    var missing = -1
    var known = 1
    for i in range(len(dims)):  # driver-loop: shape inference from admitted integer metadata
        if dims[i] == -1:
            if missing >= 0:
                raise Error("ValueError: CNN reshape accepts only one inferred dimension")
            missing = i
        else:
            known *= dims[i]
    if missing >= 0:
        dims[missing] = total // known if known else 0
    return dims^


# ---------------------------------------------------------------- host words

def xcnn_run_copy_f32(dst: Int, src: Int, count: Int):
    """`count` float32 words from address `src` to address `dst`."""
    if count > 0:
        unsafe_memcpy(dest=XcnnF32P(unsafe_from_address=dst), src=XcnnF32P(unsafe_from_address=src), count=count)


def xcnn_run_fill_f32(dst: Int, count: Int, value: Float32):
    """`count` float32 words at address `dst` set to `value`."""
    var words = XcnnF32P(unsafe_from_address=dst)
    for i in range(count):  # driver-loop: over a small per-channel parameter block
        words[i] = value
