# SPDX-License-Identifier: Apache-2.0
# Typed view of one admitted array: address (CastIn), shape, strides, dtype
# word, contiguity and ownership flags. No Python here: RUN code takes this
# view. The storage owners that keep `source.addr` valid are the door's
# ArrayPin (providers/python/array_metadata_pyapi.mojo); the door holds the pin
# for the whole call and names it again after the last RUN read (AGENTS.md:
# Mojo frees a local at its last named use).
from core.cast_in import CastIn, CAST_KIND_FLOAT, NPY_FLOAT, NPY_DOUBLE


struct ArrayMetadata(Copyable, Movable):
    var shape: List[Int]
    var source: CastIn
    var count: Int
    var dtype_number: Int
    var native_byte_order: Bool
    var c_order: Bool
    var is_array: Bool
    # True after replace_f32/replace_f64: the door's pin `keep` then owns the
    # converted buffer and its `owner` still holds the original storage.
    var copied: Bool
    var owned_writable: Bool
    # Writability is independent of exclusive ownership: a borrowed Array
    # view may be writable, while copy=False must respect a readonly view.
    var readonly: Bool
    # Admission-only snapshots may retain higher rank/unsupported dtype
    # metadata for public error ordering. Their CastIn is NOT executable.
    var castable: Bool
    var strides: List[Int]
    # Native descriptor spelling, including datetime/timedelta units. Owners
    # may inspect it without rereading the caller's dtype after admission.
    var dtype_name: String

    def __init__(out self, var shape: List[Int], source: CastIn,
                 count: Int, dtype_number: Int, native_byte_order: Bool,
                 c_order: Bool, is_array: Bool, owned_writable: Bool,
                 readonly: Bool = False, castable: Bool = True,
                 var strides: List[Int] = List[Int](), dtype_name: String = ""):
        self.shape = shape^
        self.source = source
        self.count = count
        self.dtype_number = dtype_number
        self.native_byte_order = native_byte_order
        self.c_order = c_order
        self.is_array = is_array
        self.copied = False
        self.owned_writable = owned_writable
        self.readonly = readonly
        self.castable = castable
        self.strides = strides^
        self.dtype_name = dtype_name
        if len(self.strides) == 0:
            for axis in range(len(self.shape)):  # small-loop(native rank metadata for direct constructors)
                self.strides.append(source.s0 if axis == 0 else source.s1)

    def f32_c(self) -> Bool:
        return self.castable and self.dtype_number == NPY_FLOAT and self.source.itemsize == 4 and self.c_order

    def replace_f32(mut self, address: Int):
        # A new packed buffer's metadata is known from the allocation. Never
        # fetch it back from the caller to rediscover the shape/dtype we
        # supplied. The door stores the new buffer in its pin (`keep`) and
        # leaves the pin's original storage owners in place.
        self.copied = True
        self.owned_writable = True
        self.readonly = False
        self.source = CastIn(address, CAST_KIND_FLOAT, 4, self.source.rows, self.source.cols,
                             self.source.cols * 4 if len(self.shape) > 0 else 0,
                             4 if len(self.shape) > 1 else 0)
        self.strides = List[Int]()
        for axis in range(len(self.shape)):  # small-loop(replacement matrix/vector stride metadata)
            self.strides.append(self.source.s0 if axis == 0 else self.source.s1)
        self.dtype_number = NPY_FLOAT
        self.dtype_name = "<f4"
        self.native_byte_order = True
        self.c_order = True
        self.is_array = True

    def replace_f64(mut self, address: Int):
        # Same ownership rule as replace_f32.
        self.copied = True
        self.owned_writable = True
        self.readonly = False
        self.source = CastIn(address, CAST_KIND_FLOAT, 8, self.source.rows, self.source.cols,
                             self.source.cols * 8 if len(self.shape) > 0 else 0,
                             8 if len(self.shape) > 1 else 0)
        self.strides = List[Int]()
        for axis in range(len(self.shape)):  # small-loop(replacement matrix/vector stride metadata)
            self.strides.append(self.source.s0 if axis == 0 else self.source.s1)
        self.dtype_number = NPY_DOUBLE
        self.dtype_name = "<f8"
        self.native_byte_order = True
        self.c_order = True
        self.is_array = True
