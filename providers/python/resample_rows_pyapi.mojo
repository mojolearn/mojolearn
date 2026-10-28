# SPDX-License-Identifier: Apache-2.0
# Stable input metadata and retained storage for general row resampling.
from std.memory import memcpy
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.py_attr import py_attr
from bindings.native_loaded_module import native_loaded_module
from bindings.native_numpy_metadata import native_numpy_dtype
from bindings.native_owner_lifetime import native_release_admit, _released_dtype
from bindings.classical_attributes import classical_connect_numpy
from bindings.classical_arrays import classical_is_array
from bindings.classical_pandas import classical_admit_value
from bindings.linalg_ops_buffers import LinAdmission, LinBuf, lin_new, lin_tuple
from providers.python.borrowed_buffer_pyapi import AdmittedBorrowedBuffer
from providers.python.neural_descriptor_pyapi import neural_descriptor_entry
from core.borrowed_buffer import BorrowedBufferLayout


def _row_dtype_admit(dtype: PythonObject, context: PythonObject) raises:
    if _released_dtype(dtype, context, 0):
        return
    # NumPy's native variable-length StringDType cannot be copied as raw
    # allocator bytes. Accept its exact builtin class, with callback-free
    # missing-value/metadata owners, and use native scalar storage copying.
    var b = native_loaded_module("builtins")
    var string_dtype = PythonObject(None)
    try:
        string_dtype = py_call(native_numpy_dtype(context), "T")
    except:
        pass
    if string_dtype is None or py_type(dtype) is not py_type(string_dtype):
        raise Error("TypeError: resample requires a native NumPy dtype")
    native_release_admit(py_call(b.getattr, dtype, "na_object", PythonObject(None)), context)
    var metadata = dtype.metadata
    if metadata is not None:
        var pairs = py_call(b.list, metadata.items())
        for i in range(len(pairs)):  # driver-loop: native dtype metadata cannot retain callback owners
            native_release_admit(pairs[i][0], context)
            native_release_admit(pairs[i][1], context)


struct ResampleRows(Movable):
    var source: PythonObject
    var sequence: PythonObject
    var dtype: PythonObject
    var pin: Optional[AdmittedBorrowedBuffer]
    var array_owner: Optional[LinAdmission]
    var layout: BorrowedBufferLayout
    var is_array: Bool
    var has_objects: Bool
    var zero_width: Bool
    var gpu: Bool
    var n: Int

    def __init__(out self, source: PythonObject, context: PythonObject, gpu: Bool = True) raises:
        var b = native_loaded_module("builtins")
        self.source = source
        self.sequence = PythonObject(None)
        self.dtype = PythonObject(None)
        self.pin = None
        self.array_owner = None
        self.layout = BorrowedBufferLayout(0, 1, False, [0], [1], [-1])
        self.is_array = False
        self.has_objects = False
        self.zero_width = False
        self.gpu = False
        self.n = 0
        var cls = py_type(source)
        if cls is py_get(context, "ndarray"):
            # ndarray.view is the exact immutable C descriptor. Its native
            # view freezes this operation's shape/strides/dtype and keeps the
            # original owner alive without Array/Python conversion callbacks.
            self.source = py_call(py_attr(context["ndarray"], "view"), source)
            self.dtype = self.source.dtype
            _row_dtype_admit(self.dtype, context)
            self.has_objects = Bool(py=self.dtype.hasobject) or String(self.dtype.kind) == "T"
            self.is_array = True
            var item = Int(py=self.dtype.itemsize)
            self.zero_width = item == 0
            try:
                self.pin = AdmittedBorrowedBuffer(self.source)
                self.layout = self.pin.value().layout.value().copy()
            except:
                # Native datetime/timedelta and zero-width dtypes need not
                # export PEP3118. Their admitted ndarray view still owns its
                # storage; read its immutable C metadata without a protocol.
                var shape = self.source.shape
                var strides = self.source.strides
                var sizes = List[Int]()
                var steps = List[Int]()
                for axis in range(len(shape)):  # driver-loop: one native-view metadata capture for non-buffer dtypes
                    sizes.append(Int(py=shape[axis]))
                    steps.append(Int(py=strides[axis]))
                var address = Int(py=self.source.__array_interface__["data"][0])
                self.layout = BorrowedBufferLayout(max(address, 1) if item == 0 else address,
                    max(item, 1), not Bool(py=self.source.flags.writeable), sizes^, steps^,
                    List[Int](length=len(shape), fill=-1))
            self.gpu = gpu and String(self.dtype.str) == "<f4" and self.layout.contiguous()
        elif classical_is_array(source, context):
            if py_get(context, "ndarray") is None:
                raise Error("ImportError: mojolearn: resample requires optional NumPy support for Array outputs")
            self.array_owner = LinAdmission(source, context)
            ref view = self.array_owner.value().view
            self.layout = BorrowedBufferLayout(view.addr, view.itemsize, view.readonly,
                view.shape.copy(), view.strides.copy(), List[Int](length=len(view.shape), fill=-1))
            self.dtype = py_call(native_numpy_dtype(context), view.dtype)
            self.is_array = True
        else:
            # Only exact native sequence slots can execute during snapshotting.
            # The retained tuple also protects membership while kernels release
            # the GIL. Admit its owners before it can be destroyed here.
            classical_admit_value(source, context)
            if (cls is not b.list and cls is not b.tuple and cls is not b.range
                    and cls is not b.str and cls is not b.bytes and cls is not b.bytearray
                    and cls is not b.memoryview and cls is not context["array_type"]):
                raise Error("TypeError: resample requires a native array or sequence without Python indexing hooks")
            self.sequence = py_call(b.tuple, source)
            native_release_admit(self.sequence, context)
            self.n = len(self.sequence)
            return
        if len(self.layout.shape) == 0:
            raise Error("TypeError: len() of unsized object")
        self.n = self.layout.shape[0]
        self.gpu = self.gpu and (len(self.layout.shape) == 1 or len(self.layout.shape) == 2)
        if len(self.layout.shape) == 2:
            self.gpu = self.gpu and self.layout.shape[1] > 0 and self.layout.shape[1] <= 2147483647

    def width(self) -> Int:
        return self.layout.shape[1] if len(self.layout.shape) == 2 else 1

    def output(self, count: Int, context: PythonObject) raises -> PythonObject:
        var shape = self.layout.shape.copy()
        shape[0] = count
        # Exact ndarray allocation with a retained native dtype is storage
        # construction only. No asarray, conversion, or caller protocol runs.
        return py_call(context["ndarray"], lin_tuple(shape), dtype=self.dtype)

    def take(self, indices: LinBuf, context: PythonObject) raises -> PythonObject:
        var rows = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=indices.addr)
        var count = indices.dim(0)
        if not self.is_array:
            var out = Python.list()
            for i in range(count):  # driver-loop: native sequence row references, without caller indexing hooks
                var row = Int(rows[i])
                if row < 0 or row >= self.n:
                    raise Error("IndexError: resample returned an invalid row")
                py_call(out.append, self.sequence[row])
            _ = indices
            return out
        var out = self.output(count, context)
        # The output's exact native ndarray metadata is read once as well;
        # datetime and zero-width dtypes need no PEP3118 exporter.
        var dst = Int(py=out.__array_interface__["data"][0])
        var rank = len(self.layout.shape)
        var row_items = 1
        for axis in range(1, rank):  # driver-loop: captured trailing dimensions of one row
            row_items *= self.layout.shape[axis]
        var coordinates = List[Int](length=rank, fill=0)
        var out_coordinates = List[Int](length=rank, fill=0)
        for i in range(count):  # driver-loop: native byte/reference row gathering
            var row = Int(rows[i])
            if row < 0 or row >= self.n:
                raise Error("IndexError: resample returned an invalid row")
            coordinates[0] = row
            out_coordinates[0] = i
            for item in range(0 if self.zero_width else row_items):  # driver-loop: copy every captured trailing element without numeric work
                if self.has_objects:
                    # Exact ndarray scalar storage access preserves object and
                    # structured-dtype reference semantics. The source dtype
                    # and result dtype are identical: no dtype conversion runs.
                    var value = py_call(py_attr(context["ndarray"], "__getitem__"), self.source, lin_tuple(coordinates))
                    py_call(py_attr(context["ndarray"], "__setitem__"), out, lin_tuple(out_coordinates), value)
                else:
                    memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=dst + (i * row_items + item) * self.layout.itemsize),
                        src=Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=self.layout.item_address(coordinates)), count=self.layout.itemsize)
                for reverse in range(rank - 1):  # driver-loop: native carry over captured trailing axes
                    var axis = rank - reverse - 1
                    coordinates[axis] += 1
                    out_coordinates[axis] += 1
                    if coordinates[axis] < self.layout.shape[axis]:
                        break
                    coordinates[axis] = 0
                    out_coordinates[axis] = 0
        _ = indices
        _ = self
        return out


def resample_rows_indices(binding: PythonObject, n: Int, count: Int, replace: Bool,
                          seed: PythonObject, context: PythonObject) raises -> LinBuf:
    var indices = lin_new([max(count, 0)], "<i4", context)
    py_call(neural_descriptor_entry(binding, "resample_indices"), Python.list(indices.addr),
            Python.list(n, count, 1 if replace else 0, seed))
    _ = indices
    return indices^


def resample_rows_gpu(rows: List[ResampleRows], binding: PythonObject, n: Int, count: Int,
                      replace: Bool, seed: PythonObject, context: PythonObject) raises -> PythonObject:
    if n <= 0 or count <= 0:
        return PythonObject(None)
    var enabled = neural_descriptor_entry(binding, "resample_gpu_gather_enabled", False)
    if enabled is None or Int(py=py_call(enabled)) == 0:
        return PythonObject(None)
    for i in range(len(rows)):  # driver-loop: native eligibility from each input's single metadata snapshot
        if not rows[i].gpu:
            return PythonObject(None)
    var route = neural_descriptor_entry(binding, "resample_gather_gpu" if replace else "resample_permutation_gather_gpu", False)
    if route is None:
        return PythonObject(None)
    var outputs = Python.list()
    var output_pins = List[AdmittedBorrowedBuffer]()
    var addresses = Python.list()
    var params = Python.list(n, count, seed)
    for i in range(len(rows)):  # driver-loop: retain every output allocation through the compiled gather
        var output = rows[i].output(count, context)
        output_pins.append(AdmittedBorrowedBuffer(output, True))
        py_call(outputs.append, output)
        py_call(addresses.append, rows[i].layout.address)
        py_call(addresses.append, output_pins[len(output_pins) - 1].layout.value().address)
        py_call(params.append, rows[i].width())
    var done = Int(py=py_call(route, addresses, params))
    _ = output_pins^
    _ = rows
    return outputs if done != 0 else PythonObject(None)


def resample_rows_narrow(rows: List[ResampleRows], binding: PythonObject, n: Int, count: Int,
                         seed: PythonObject, context: PythonObject) raises -> PythonObject:
    # Retain the existing optional fast gather path. No new tuning/default.
    if n <= 0 or count <= 0:
        return PythonObject(None)
    var defines = neural_descriptor_entry(binding, "resample_fast_defines", False)
    if defines is None or (Int(py=py_call(defines)) & 128) == 0:
        return PythonObject(None)
    var route = neural_descriptor_entry(binding, "resample_gather_narrow", False)
    if route is None:
        return PythonObject(None)
    var bound = Int(py=py_call(neural_descriptor_entry(binding, "resample_gather_narrow_bytes")))
    var indices = lin_new([count], "<i4", context)
    var outputs = Python.list()
    var output_pins = List[AdmittedBorrowedBuffer]()
    var addresses = Python.list(indices.addr)
    var params = Python.list(n, count, seed)
    for i in range(len(rows)):  # driver-loop: retain the existing byte-width eligibility from snapshot metadata
        var width = rows[i].width() if rows[i].gpu else 0
        if 4 * width > bound:
            width = 0
        var output = PythonObject(None)
        var src = 0
        var dst = 0
        if width > 0:
            output = rows[i].output(count, context)
            output_pins.append(AdmittedBorrowedBuffer(output, True))
            src = rows[i].layout.address
            dst = output_pins[len(output_pins) - 1].layout.value().address
        py_call(outputs.append, output)
        py_call(addresses.append, src)
        py_call(addresses.append, dst)
        py_call(params.append, width)
    var done = Int(py=py_call(route, addresses, params))
    _ = output_pins^
    if done < 0:
        return PythonObject(None)
    var result = Python.list()
    for i in range(len(rows)):  # driver-loop: finish rows not handled by the retained compiled narrow route
        py_call(result.append, outputs[i] if ((done >> i) & 1) != 0 else rows[i].take(indices, context))
    _ = indices^
    return result
