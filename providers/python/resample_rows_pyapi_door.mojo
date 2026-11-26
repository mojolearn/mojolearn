# SPDX-License-Identifier: Apache-2.0
# Stable input metadata and retained storage for general row resampling.
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.py_attr import py_attr
from bindings.native_loaded_module_door import native_loaded_module
from bindings.native_numpy_metadata_unpack import native_numpy_dtype
from bindings.native_owner_lifetime_door import native_release_admit
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.classical_arrays_door import classical_is_array
from bindings.classical_pandas_door import classical_admit_value
from bindings.admitted_input_unpack import AdmittedTypedPin, admitted_typed_once, AdmittedBuf, admitted_dtype_admit, admitted_zeros, admitted_shape_tuple
from providers.python.borrowed_buffer_pyapi_unpack import AdmittedBorrowedBuffer
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_connect_once, neural_descriptor_execute
from providers.python.resample_resident_pyapi_door import RsResident, rs_resident, rs_resident_slot_at
from bindings.metric_resident_source_door import metric_exact_buffer
from bindings.classical_arrays_door import classical_address

# ORDER MATCHES resample/resample_native.mojo (bindings/linalg_ops_resample_door.mojo RS_OP_*).
comptime _RS_OP_RESAMPLE_INDICES = 4
#: lane gd-rf-resample: the row resample, drawn and gathered by the image (on the device in a GPU image).
comptime _RS_OP_RESAMPLE_ROWS = 8
comptime _RS_OPERATIONS = 9
comptime _INT32_MAX = 2147483647
from core.borrowed_buffer import BorrowedBufferLayout
from core.resample_rows_gather import resample_gather_rows
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from bindings.py_cpython_table import cpython_table
from bindings.native_buffer_address_unpack import native_buffer_address
from bindings.resample_keys_unpack import resample_keys
from bindings.py_balanced import py_item


# _row_dtype_admit moved to bindings/admitted_input_unpack.mojo admitted_dtype_admit (lane a6-device-x-family-classical; word for word).


def _take_rows_contiguous(src: Int, row_stride: Int, row_bytes: Int, n: Int,
                          rows: MutPointer[Int32, MutUntrackedOrigin], count: Int, dst: Int) raises:
    """`count` rows of `row_bytes` contiguous bytes each, row `rows[i]` of the
    source (at `src + rows[i] * row_stride`) to output row `i` (at
    `dst + i * row_bytes`): one memcpy per row, the rows split across the
    host's cores (lane ne2-b-rounds, N052). No dtype work: the bytes as they
    are. A row index outside [0, n) is refused after the join."""
    if count <= 0:
        return
    var tasks = host_predict_task_count_work(count, count * max(row_bytes, 1))
    var chunk = host_predict_chunk(count, tasks)
    var failed = List[Int](length=tasks, fill=0)
    var fp = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(failed.unsafe_ptr()))

    def _rows_task(c: Int) {imm src, imm row_stride, imm row_bytes, imm n, imm rows, imm count,
                            imm dst, imm chunk, imm fp}:
        var lo = c * chunk
        var hi = lo + chunk
        if hi > count:
            hi = count
        for i in range(lo, hi):  # driver-loop: native byte row gathering, one memcpy per row of one task
            var row = Int(rows[i])
            if row < 0 or row >= n:
                fp[c] = 1
                return
            unsafe_memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=dst + i * row_bytes),
                src=Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=src + row * row_stride), count=row_bytes)

    host_parallelize(_rows_task, tasks)
    for c in range(tasks):  # small-loop(tasks: the row tasks, at most one per core): the failure words of the join
        if fp[c] != 0:
            raise Error("IndexError: resample returned an invalid row")
    _ = failed^


struct ResampleRows(Movable):
    var source: PythonObject
    var sequence: PythonObject
    var dtype: PythonObject
    var pin: Optional[AdmittedBorrowedBuffer]
    var array_owner: AdmittedTypedPin
    var layout: BorrowedBufferLayout
    var is_array: Bool
    var has_objects: Bool
    var zero_width: Bool
    var n: Int

    def __init__(out self, source: PythonObject, context: PythonObject) raises:
        var nk = resample_keys()
        var b = native_loaded_module("builtins")
        self.source = source
        self.sequence = PythonObject(None)
        self.dtype = PythonObject(None)
        self.pin = None
        self.array_owner = AdmittedTypedPin()
        self.layout = BorrowedBufferLayout(0, 1, False, [0], [1], [-1])
        self.is_array = False
        self.has_objects = False
        self.zero_width = False
        self.n = 0
        var cls = py_type(source)
        if cls is py_get(context, nk[].k_ndarray):
            # ndarray.view is the exact immutable C descriptor. Its native
            # view freezes this operation's shape/strides/dtype and keeps the
            # original owner alive without Array/Python conversion callbacks.
            self.source = py_call(py_attr(py_item(context, nk[].k_ndarray), "view"), source)
            self.dtype = self.source.dtype
            admitted_dtype_admit(self.dtype, context)
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
                var address = Int(py=py_item(self.source.__array_interface__, nk[].k_data)[0])
                self.layout = BorrowedBufferLayout(max(address, 1) if item == 0 else address,
                    max(item, 1), not Bool(py=self.source.flags.writeable), sizes^, steps^,
                    List[Int](length=len(shape), fill=-1))
        elif classical_is_array(source, context):
            if py_get(context, nk[].k_ndarray) is None:
                raise Error("ImportError: mojolearn: resample requires optional NumPy support for Array outputs")
            var view = admitted_typed_once(source, context, self.array_owner)
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
                    and cls is not b.memoryview and cls is not py_item(context, nk[].k_array_type)):
                raise Error("TypeError: resample requires a native array or sequence without Python indexing hooks")
            self.sequence = py_call(b.tuple, source)
            native_release_admit(self.sequence, context)
            self.n = len(self.sequence)
            return
        if len(self.layout.shape) == 0:
            raise Error("TypeError: len() of unsized object")
        self.n = self.layout.shape[0]

    def _rows_contiguous(self) -> Bool:
        """Whether each row (axes 1..rank-1) is one C-contiguous run of bytes
        reached without an indirection: no suboffset on any axis and every
        trailing axis at its C stride (a length-0 or length-1 axis is never
        stepped)."""
        var rank = len(self.layout.shape)
        if rank == 0 or self.layout.suboffsets[0] >= 0:
            return False
        var stride = self.layout.itemsize
        for reverse in range(rank - 1):  # small-loop(rank: the admitted array's axes): the trailing strides
            var axis = rank - reverse - 1
            if self.layout.suboffsets[axis] >= 0 or (self.layout.shape[axis] > 1 and self.layout.strides[axis] != stride):
                return False
            stride *= self.layout.shape[axis]
        return True

    def row_words(self) -> Int:
        """The 4-byte words of one row when the image gathers this array
        itself (on the device in a GPU image, lane gd-rf-resample): an array
        of plain values (no object references, not zero-width) whose whole
        buffer is one C-contiguous run and whose row is a whole number of
        4-byte words, every dtype alike (a word is copied, never converted).
        0 otherwise: a sequence, an object array, a strided view or an
        odd-width row is gathered here with the image's row numbers."""
        if not self.is_array or self.has_objects or self.zero_width or not self.layout.contiguous():
            return 0
        var row_bytes = self.layout.itemsize
        for axis in range(1, len(self.layout.shape)):  # small-loop(rank: the admitted array's axes)
            row_bytes *= self.layout.shape[axis]
        if row_bytes <= 0 or row_bytes % 4 != 0:
            return 0
        return row_bytes // 4

    def output(self, count: Int, context: PythonObject) raises -> PythonObject:
        var nk = resample_keys()
        var shape = self.layout.shape.copy()
        shape[0] = count
        # Exact ndarray allocation with a retained native dtype is storage
        # construction only. No asarray, conversion, or caller protocol runs.
        return py_call(py_item(context, nk[].k_ndarray), admitted_shape_tuple(shape), dtype=self.dtype)

    def take(self, indices: AdmittedBuf, context: PythonObject) raises -> PythonObject:
        var nk = resample_keys()
        var rows = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=indices.addr)
        var count = indices.dim(0)
        if not self.is_array:
            # lane ne2-b-rounds (N052): the list is built natively at its
            # final length (PyList_New + PyList_SetItem of a new reference to
            # each retained tuple member), not appended row by row through a
            # bound-method call
            ref cpy = Python().cpython()
            var raw = cpython_table()[].PyList_New(count)
            if not raw:
                raise cpy.unsafe_get_error()
            var out = PythonObject(from_owned=raw)
            for i in range(count):  # driver-loop: native sequence row references, without caller indexing hooks
                var row = Int(rows[i])
                if row < 0 or row >= self.n:
                    raise Error("IndexError: resample returned an invalid row")
                var item = self.sequence[row]
                if cpython_table()[].PyList_SetItem(out._obj_ptr, i, cpy.Py_NewRef(item._obj_ptr)) != 0:
                    raise cpy.unsafe_get_error()
                _ = item
            _ = indices
            return out
        for i in range(count):  # driver-loop: refuse an invalid row index before any output is written
            var row = Int(rows[i])
            if row < 0 or row >= self.n:
                raise Error("IndexError: resample returned an invalid row")
        if self.has_objects:
            # TOMBSTONE: P1_RESAMPLE_OBJECT_GETITEM (removed: one ndarray __getitem__/__setitem__ per row x element for object and structured arrays) deleted 2026-11-09 by lane p1-python-callbacks-worst; code recoverable at e68654a757.
            # Restore: experiments/removed/P1_RESAMPLE_OBJECT_GETITEM.patch.
            # Object, structured-object and StringDType rows (lane
            # i3-move-data-loops, 2026-11-09): one exact C ndarray.take over a
            # native index array copies every row, NumPy's own C transfer
            # taking each object reference (StringDType storage is the source
            # array's own arena). No per-element call, no Python code, and no
            # loop over rows here: the per-leaf C-API reference counting P1
            # wrote in this file (the object-leaf offsets walk and the row x
            # element x leaf loops) is gone.
            var index_array = py_call(py_item(context, nk[].k_ndarray), admitted_shape_tuple([count]),
                                      dtype=py_call(native_numpy_dtype(context), "<i4"))
            unsafe_memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=native_buffer_address(index_array)),
                src=Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=indices.addr), count=count * 4)
            _ = indices
            return py_call(py_attr(py_item(context, nk[].k_ndarray), "take"), self.source, index_array, 0)
        var out = self.output(count, context)
        # The output's exact native ndarray metadata is read once as well;
        # datetime and zero-width dtypes need no PEP3118 exporter.
        var dst = Int(py=py_item(out.__array_interface__, nk[].k_data)[0])
        var rank = len(self.layout.shape)
        var row_items = 1
        for axis in range(1, rank):  # driver-loop: captured trailing dimensions of one row
            row_items *= self.layout.shape[axis]
        if not self.zero_width and self._rows_contiguous():
            # lane ne2-b-rounds (N052): every row's items are contiguous in
            # the source, so a row is one memcpy of row_items * itemsize
            # bytes, the rows split across the host's cores
            _take_rows_contiguous(self.layout.address, self.layout.strides[0],
                                  row_items * self.layout.itemsize, self.n, rows, count, dst)
            _ = indices
            _ = self
            return out
        # A strided or indirect source without object references: the
        # general byte gather, split across the host's cores
        # (core/resample_rows_gather.mojo). A zero-width item copies nothing.
        if not self.zero_width:
            resample_gather_rows(self.layout, rows, count, row_items, dst)
        _ = indices
        _ = self
        return out


def resample_rows_indices(binding: PythonObject, n: Int, count: Int, replace: Bool,
                          seed: PythonObject, context: PythonObject) raises -> AdmittedBuf:
    # The binding's named `resample_indices` entry became operation 4 of its typed table
    # (resample/resample_native.mojo; 067cc9c9bd); the table is connected once per process and binding.
    var indices = admitted_zeros([max(count, 0)], "<i4", context)
    var provider = neural_descriptor_connect_once(binding, "resample_native_api_v1", "mojolearn.resample.v1",
                                                  _RS_OPERATIONS)
    _ = neural_descriptor_execute(provider, _RS_OP_RESAMPLE_INDICES, [indices.addr],
                                  [n, count, 1 if replace else 0, Int(py=seed)], List[Float64]())
    _ = indices
    return indices^


def resample_rows_run(rows: List[ResampleRows], arrays: PythonObject, binding: PythonObject, n: Int, count: Int,
                      replace: Bool, seed: PythonObject, context: PythonObject) raises -> PythonObject:
    """`resample(*arrays)`'s finished rows, in the caller's argument order (lane gd-rf-resample, 2026-11-25).
    One call of the image's operation 8 draws the rows and gathers every array of plain words
    (`ResampleRows.row_words`) ON THE DEVICE in a GPU image (resample/estimator.mojo `resample_rows_device`;
    the host image gathers them on the CPU column), straight into the outputs allocated here; the row numbers
    come home only when an array needs them here (a sequence, an object array, a strided or odd-width one).
    Inside a `resident()` scope each array that is the caller's own buffer is kept once in the image's store
    and read there in place. Before, every non-Apple-FAST GPU call drew the rows on the device, downloaded
    them and gathered every array on the host. The rows are copies: the bytes do not change."""
    var provider = neural_descriptor_connect_once(binding, "resample_native_api_v1", "mojolearn.resample.v1",
                                                  _RS_OPERATIONS)
    var rs = rs_resident(binding, provider, context)
    var outputs = Python.list()
    var output_pins = List[AdmittedBorrowedBuffer]()
    var addresses: List[Int] = [0]
    var ints: List[Int] = [n, count, 1 if replace else 0, Int(py=seed)]
    var here = False
    for i in range(len(rows)):  # driver-loop: over the arrays passed to resample (a handful): metadata and outputs
        var w = rows[i].row_words()
        if w > 0 and (n > _INT32_MAX // w or count > _INT32_MAX // w):
            w = 0  # the gather kernels index words with int32
        if w == 0:
            here = True
            py_call(outputs.append, PythonObject(None))
            addresses.append(0)
            addresses.append(0)
            ints.append(0)
            ints.append(-1)
            continue
        var output = rows[i].output(count, context)
        var output_address = 0
        var kind = String(rows[i].dtype.kind)
        if kind != "M" and kind != "m":
            output_pins.append(AdmittedBorrowedBuffer(output, True))
            output_address = output_pins[len(output_pins) - 1].layout.value().address
        else:
            # A source admitted without PEP3118 (native datetime/timedelta:
            # NumPy exports no buffer for 'M'/'m'): its output is the
            # operation's own new ndarray of the same dtype, held in
            # `outputs` for the run; read its data address from the same
            # immutable C metadata the source used (w3-integrate-11: the op 8
            # path pinned every output and refused every datetime column).
            output_address = Int(py=py_item(output.__array_interface__, resample_keys()[].k_data)[0])
        py_call(outputs.append, output)
        addresses.append(rows[i].layout.address)
        addresses.append(output_address)
        ints.append(w)
        ints.append(_rows_slot(rs, arrays[i], rows[i], n * w))
    if here:
        var indices = admitted_zeros([count], "<i4", context)
        addresses[0] = indices.addr
        _ = neural_descriptor_execute(provider, _RS_OP_RESAMPLE_ROWS, addresses, ints, List[Float64]())
        for i in range(len(rows)):  # driver-loop: finish the arrays the image does not gather, in argument order
            if outputs[i] is None:
                py_setitem(outputs, PythonObject(i), rows[i].take(indices, context))
        _ = indices^
    else:
        _ = neural_descriptor_execute(provider, _RS_OP_RESAMPLE_ROWS, addresses, ints, List[Float64]())
    _ = output_pins^
    _ = rs^  # the store provider stays connected through the run above
    _ = rows
    return outputs


def _rows_slot(rs: RsResident, original: PythonObject, row: ResampleRows, words: Int) raises -> Int:
    """The resident store slot of the caller's array `original` (inside a `resident()` scope), when the rows
    gathered are its own buffer: an exact ndarray or mojolearn Array whose data address is the admitted
    layout's (a pandas or Torch argument is converted per call and never kept); -1 otherwise."""
    if not rs.active() or not metric_exact_buffer(original):
        return -1
    var address = Int(py=classical_address(original))
    if address != row.layout.address:
        return -1
    return rs_resident_slot_at(rs, original, address, words)
