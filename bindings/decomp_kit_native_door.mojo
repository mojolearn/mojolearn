# SPDX-License-Identifier: Apache-2.0
# The UNPACK and PACK half of the native decomposition owners' kit (lane
# entangled-decompkit, 2026-11-12): the admission helpers (decomp_input,
# decomp_shape, decomp_seed, decomp_entries, ...), the door's typed-kit
# construction (decomp_kit_unpack: the image's dispatch and optional entries
# read once), the borrowed inputs and resident fitted operands a door adds
# (DecompDoor, decomp_door_*), PACK's copies into public Arrays
# (decomp_kit_publish / decomp_kit_out_shape) and the model's resident records.
# The RUN half, python/mojolearn/_expansion_decomp.py's `_Kit` / `_M` with no
# Python, is bindings/decomp_kit_run.mojo (DKit).
from std.python import Python, PythonObject
from bindings.py_cpython_table import cpython_table
from bindings.native_loaded_module_door import native_loaded_module, NativeModules
from bindings.native_prepared_owner_unpack import prepared_get, prepared_put, prepared_drop
from bindings.native_resident_capsule_pack import resident_capsule
from bindings.py_model_records import records_open, model_generation
from bindings.p3nc_fitted_freeze_unpack import fitted_is_freezable_any, fitted_freeze_any
from std.memory import unsafe_memcpy
from providers.python.array_metadata_pyapi_door import ArrayMetadata, ArrayPin
from bindings.py_balanced import py_call, py_get, py_type, py_contains, py_setitem, py_setattr
from bindings.classical_arrays_door import classical_typed, classical_address, classical_kind
from bindings.admitted_input import admit_x
from std.python._cpython import GILReleased
from core.linear_entry_table import LinearEntryKey
from core.entry_table import EntryRef, entry_ref_unresolved
from core.decomp_entry_table import decomp_ref, decomp_table_call
from x_decomp.descriptor_schema import XD_DEV_ALLOC, XD_DEV_FREE, XD_DEV_UPLOAD
from bindings.classical_validate_unpack import classical_view
from bindings.host_sparse_native_door import host_sparse_format, host_sparse_dense
from bindings.decomp_kit_run import (
    DKit, DK_HAS_DEV_SVD, DK_HAS_DEV_LU, DK_HAS_DEV_CODE_ROWS, DK_HAS_DEV_EIGH_TO, DK_HAS_S_FLAGS,
    DK_HAS_QFIX_FLAGS, DK_HAS_IDN_FLAGS,
)

comptime DecompF32 = MutPointer[Float32, MutUntrackedOrigin]

# TOMBSTONE: DecompKit's elementwise and move code copies (XD_ADD .. XD_SELECT, XD_TAKE_ROWS ..
# XD_I32_MAX) moved with the kit to bindings/decomp_kit_run.mojo (DK_*) by lane entangled-decompkit, 2026-11-12.
comptime XD_SPARSE_ALGOS = "('lasso_lars', 'lasso_cd', 'lars', 'omp', 'threshold')"
comptime XD_RESHAPE_HINT = ". Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample."


def _repr(value: PythonObject) raises -> String:
    return String(py_call(native_loaded_module("builtins").repr, value))


def decomp_int(value: PythonObject) raises -> Int:
    """`int(value)`."""
    return Int(py=py_call(native_loaded_module("builtins").int, value))


def decomp_float(value: PythonObject) raises -> Float64:
    """`float(value)`."""
    return Float64(py=py_call(native_loaded_module("builtins").float, value))


def decomp_seed(random_state: PythonObject) raises -> PythonObject:
    """`_expansion_decomp._seed_of`: None is 0; anything but an int refused."""
    var b = native_loaded_module("builtins")
    if random_state is None:
        return PythonObject(0)
    if Bool(py=py_call(b.isinstance, random_state, b.bool)) or not Bool(py=py_call(b.isinstance, random_state, b.int)):
        raise Error("TypeError: random_state must be None or an int (the counter-based RNG takes an integer seed)")
    return random_state


def decomp_seed_word(seed: PythonObject) raises -> Int:
    """`int(seed) & 0xFFFFFFFF`, with Python's integer semantics."""
    var masked = py_call(native_loaded_module("operator").and_, py_call(native_loaded_module("builtins").int, seed),
                         PythonObject(4294967295))
    return Int(py=masked)


def decomp_is_a(model: PythonObject, cls: PythonObject) raises -> Bool:
    """The model's class MRO holds cls."""
    # CPython's own subtype test walks the same tp_mro by identity in one
    # native call (no builtins module check, MRO tuple fetch or per-entry
    # Python indexing per call). An MRO holds classes only, so anything
    # that is not a class is never found, as before.
    var meta = py_type(py_type(PythonObject(0)))  # the builtin `type`
    # Resolved once per process in the shared table (pass 2), not by name per call.
    var t = cpython_table()
    if t[].PyType_IsSubtype(py_type(cls)._obj_ptr, meta._obj_ptr) == 0:
        return False
    return t[].PyType_IsSubtype(py_type(model)._obj_ptr, cls._obj_ptr) != 0


def decomp_class_attr(model: PythonObject, name: String) raises -> PythonObject:
    """A class attribute (a fixed estimator constant such as `_deflation`),
    read from the class dictionaries along the MRO, None when absent."""
    var b = native_loaded_module("builtins")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, py_type(model))
    for k in range(len(mro)):  # driver-loop: over the model class's MRO
        var fields = py_call(b.type.__dict__["__dict__"].__get__, mro[k])
        if py_contains(fields, PythonObject(name)):
            return fields[name]
    return PythonObject(None)


def decomp_mode(value: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_expansion_decomp._mode`: the estimator's numeric_mode, None the
    process default (the value the estimator records as numeric_mode_)."""
    if value is None:
        return context["backend_fields"]["_DEFAULT_MODE"]
    return value


def decomp_entries(mode_value: PythonObject, context: PythonObject) raises -> PythonObject:
    """The connected x_decomp entries of a numeric mode, refused as
    `_backend.binding` refuses the mode (or as it refused at connection)."""
    var b = native_loaded_module("builtins")
    var mode = decomp_mode(mode_value, context)
    if py_type(mode) is not b.str:
        raise Error("numeric_mode must be fast, identical or None")
    # An exact canonical spelling needs no strip()/lower() calls (both would
    # return it unchanged); any other spelling is normalized as before.
    var name = String(py=mode)
    if name != "identical" and name != "fast":
        name = String(py=py_call(py_call(mode.strip).lower))
    if name != "identical" and name != "fast":
        raise Error("numeric_mode must be fast, identical or None")
    var entries = context["decomp_bindings"][name]
    if py_type(entries) is b.str:
        raise Error(String(py=entries))
    return entries


def decomp_input(source: PythonObject, name: String, context: PythonObject,
                 defer_finite: Bool = False) raises -> PythonObject:
    """`_M.from_input`: a SciPy sparse X densified (exact), then a nonempty
    2-D input cast once to C-order float32 (the input itself when it is one),
    non-finite values refused, in `_buffer.as_f32_c`'s and `_M`'s words.
    `defer_finite` (randomized_svd under RSVD_FAST_DEVSCAN, restored
    2026-11-09): a borrowed input is not scanned here; the caller's native
    call tests its device copy."""
    var value = source
    var format = host_sparse_format(source)
    if format.byte_length() > 0:
        value = host_sparse_dense(source, format, context)
    var view = classical_view(value, context, name)
    var nd = Int(py=view.ndim)
    # The shape's repr names a refusal only; it is built on that path alone
    # (it was formatted through builtins.repr on every call).
    if nd != 2:
        raise Error("mojolearn: " + name + " must be 2-D, got " + String(nd) + "-D shape " + _repr(view.shape)
                    + (String(XD_RESHAPE_HINT) if nd < 2 else String("")))
    var rows = Int(py=view.shape[0])
    var cols = Int(py=view.shape[1])
    if rows * cols == 0:
        var shape = _repr(view.shape)
        var what = String("feature(s)") if rows > 0 else String("sample(s)")
        raise Error("mojolearn: " + name + " is empty, shape " + shape + ": Found array with 0 " + what
                    + " (shape=" + shape + ") while a minimum of 1 is required.")
    var kind = classical_kind(view, context)
    if kind != "b" and kind != "i" and kind != "u" and kind != "f":
        raise Error("TypeError: mojolearn: " + name + " has a non-numeric dtype")
    try:
        return admit_x(view, 2, name, context, admitted=True, defer_finite=defer_finite)
    except error:
        if String(error) == "Input " + name + " contains NaN or infinity":
            raise Error(name + ": input must be finite; NaN/inf are unsupported")
        raise Error(String(error))


def decomp_shape(source: PythonObject, name: String, context: PythonObject) raises -> Tuple[Int, Int]:
    """The (rows, cols) `_buffer.as_f32_c(source, ndim=2)` accepts, with its
    refusals, and no conversion (SparseCoder.fit reads only the width)."""
    var view = classical_view(source, context, name)
    var nd = Int(py=view.ndim)
    # The shape's repr names a refusal only; it is built on that path alone
    # (it was formatted through builtins.repr on every call).
    if nd != 2:
        raise Error("mojolearn: " + name + " must be 2-D, got " + String(nd) + "-D shape " + _repr(view.shape)
                    + (String(XD_RESHAPE_HINT) if nd < 2 else String("")))
    var rows = Int(py=view.shape[0])
    var cols = Int(py=view.shape[1])
    if rows * cols == 0:
        var shape = _repr(view.shape)
        var what = String("feature(s)") if rows > 0 else String("sample(s)")
        raise Error("mojolearn: " + name + " is empty, shape " + shape + ": Found array with 0 " + what
                    + " (shape=" + shape + ") while a minimum of 1 is required.")
    return (rows, cols)


def decomp_freeze_array(array: PythonObject) raises:
    """Make an owned mojolearn Array read-only at the buffer level: its
    `_readonly` flag and a read-only memoryview (`memoryview.toreadonly()`),
    so NumPy views made from it from now on are writeable=False and cannot
    be turned writable again. Native kernels still write by address."""
    var view = array._mv
    if not Bool(py=view.readonly):
        py_setattr(array, "_mv", py_call(view.toreadonly))
    if not Bool(py=array._readonly):
        py_setattr(array, "_readonly", PythonObject(True))


# Device-resident fitted operands of a query model (lane perf2-pca-decomp):
# one prepared-registry record per model (a dict: operand name -> (stored
# source, alloc entry, free entry, rows, cols, handle, owning capsule)). The
# record lives outside the model's __dict__ and is dropped with the model.
comptime _DK_RESIDENT_SLOT = "decomp_resident_operands_v1"


# The capsule that returns a resident device buffer to its minting binding
# when the record holding it is dropped lives in
# bindings/native_resident_capsule.mojo (shared with the TSA and projection
# owners, lane ne1-a-decomp-tsa).


# TOMBSTONE: DecompMatrix and the Python-holding DecompKit (its entries dict, output context,
# one Python Array per matrix, input owners and query model) deleted 2026-11-12 by lane
# entangled-decompkit; the typed DKit (bindings/decomp_kit_run.mojo) is the kit, the door
# half (DecompDoor, decomp_kit_unpack, decomp_door_*, decomp_kit_publish) is below. Code
# recoverable at 2999c0855c (git show 2999c0855c:bindings/decomp_kit_native.mojo).


# ---------------------------------------------------------------------------
# Device-resident fitted operands of the decomposition owners (lane
# perf2-pca-decomp; shared with IncrementalPCA, AlternatingLeastSquares and
# the random projections by lane ne1-a-decomp-resident, workstream A of
# docs/plans/NATIVE_ENTRY_POINTS_AFTER_0866.md). One prepared-registry
# record per model (`_DK_RESIDENT_SLOT`, outside its __dict__, dropped with
# the model by the native_prepared_owner weakref): a dict, operand name ->
#   (source, alloc entry, free entry, rows, cols, pool id, capsule,
#    own words, generation)
# `source` is the fitted Array the copy was taken from (kept alive, so its
# address is never recycled under the entry), `alloc` / `free` the minting
# binding's entries, the capsule (bindings/native_resident_capsule.mojo)
# returns the pool buffer when the entry is replaced or dropped, `own words`
# is True when the buffer holds the source's own words (False: words a
# query derived from them), and `generation` the model's generation word
# (bindings/py_model_records.mojo) when the entry was taken. The key is
# so (model identity, generation, source identity): no content hash. The
# read-only rule (bindings/p3nc_fitted_freeze.mojo) makes the source
# read-only before an entry is taken. No byte cap here: an entry is one
# copy of a fitted array the model holds anyway.
# ---------------------------------------------------------------------------


def decomp_resident_generation(owner: PythonObject) raises -> Int:
    """`owner`'s generation word, its record entry opened first (a first
    entry taken at word 0 would otherwise miss once when another door
    opens the entry)."""
    _ = records_open(owner)
    return model_generation(owner)


def decomp_resident_frozen(source: PythonObject) raises -> Bool:
    """An exact mojolearn Array that is read-only at the buffer level (its
    flag and its memoryview)."""
    if not fitted_is_freezable_any(source):
        return False
    return Bool(py=source._readonly) and Bool(py=source._mv.readonly)


def decomp_resident_freeze(source: PythonObject) raises -> Bool:
    """The read-only rule: an exact mojolearn Array owning its words
    (`_base` None; a view's base could still be written) is made read-only
    (fitted_freeze_any's flag plus decomp_freeze_array's read-only
    memoryview) and True is returned; any other storage is left alone and
    keeps the per-call upload (False)."""
    if not fitted_is_freezable_any(source):
        return False
    if source._base is not None:
        return False
    _ = fitted_freeze_any(source)
    decomp_freeze_array(source)
    return True


def decomp_resident_find(owner: PythonObject, alloc: PythonObject, free: PythonObject, source: PythonObject,
                         name: String, own_words: Bool, check_generation: Bool = True) raises -> PythonObject:
    """`owner`'s entry `name` when it was taken from exactly this `source`
    object, still read-only, on the binding whose `alloc` / `free` entries
    minted it, at `owner`'s current generation (and, with `own_words`,
    holding the source's own words); else None.

    An entry a fitting call adopted (`decomp_resident_adopt`: generation
    word -1, pending) is stamped with the generation of its first query:
    the fit's own bookkeeping writes the instance dictionary after the
    state is published, so the word at adoption would never match. The
    source's identity and read-only flag prove the words until then.
    `check_generation` False: a fitting call reading the state it extends
    (its lifecycle already wrote the dictionary in this call), proven by
    the same identity and read-only rule."""
    if owner is None:
        return PythonObject(None)
    var mods = NativeModules()
    var record = prepared_get(mods, _DK_RESIDENT_SLOT, owner)
    if record is None:
        return PythonObject(None)
    var key = PythonObject(name)
    var held = py_get(record, key)
    if held is None:
        return PythonObject(None)
    if (held[0] is not source or held[1] is not alloc or held[2] is not free
            or not decomp_resident_frozen(source)):
        # A replaced array or another binding: the stale copy goes now, not
        # at the model's death (the entry also kept the old array alive).
        _ = py_call(record.pop, key, PythonObject(None))
        return PythonObject(None)
    if own_words and not Bool(py=held[7]):
        return PythonObject(None)
    if not check_generation:
        return held
    var generation = decomp_resident_generation(owner)
    var taken = Int(py=held[8])
    if taken == -1:
        var stamped = Python.tuple(held[0], held[1], held[2], held[3], held[4], held[5], held[6], held[7],
                                   PythonObject(generation))
        py_setitem(record, key, stamped)
        return stamped
    if taken != generation:
        _ = py_call(record.pop, key, PythonObject(None))
        return PythonObject(None)
    return held


def decomp_resident_hold(owner: PythonObject, alloc: PythonObject, free: PythonObject, source: PythonObject,
                         name: String, rows: Int, cols: Int, handle: Int, own_words: Bool,
                         pending: Bool = False) raises -> PythonObject:
    """Give the pool buffer `handle` (rows x cols words, already written) to
    `owner`'s record under `name` and return the entry; the previous entry
    of that name (and so its buffer) goes. The caller keeps the entry alive
    through its call: where `owner` takes no weak reference no record is
    kept and the entry's capsule frees the buffer when the caller drops it.
    On a failure the buffer is returned at once. `pending`: the generation
    is stamped by the first query (see `decomp_resident_find`)."""
    var capsule = resident_capsule(free, handle)
    var generation = -1 if pending else decomp_resident_generation(owner)
    var held = Python.tuple(source, alloc, free, PythonObject(rows), PythonObject(cols), PythonObject(handle),
                            capsule, PythonObject(own_words), PythonObject(generation))
    var mods = NativeModules()
    var record = prepared_get(mods, _DK_RESIDENT_SLOT, owner)
    if record is None:
        record = Python.dict()
        prepared_put(mods, _DK_RESIDENT_SLOT, owner, record)
    py_setitem(record, PythonObject(name), held)
    return held


def decomp_resident_drop(owner: PythonObject) raises:
    """Drop every resident entry of `owner` (a fit is about to publish new
    fitted arrays): the device buffers go back to their bindings now
    rather than at the next query."""
    var mods = NativeModules()
    prepared_drop(mods, _DK_RESIDENT_SLOT, owner)


def _decomp_key(entries: PythonObject) raises -> EntryRef:
    """The dispatch of a connected x_decomp entries dict, keyed by the
    bridge's `x_decomp_native_key` read at the boundary and resolved once per
    operation (lane entry-table-zero-alloc)."""
    var key = py_get(entries, "x_decomp_native_key")
    if key is None:
        raise Error("ImportError: mojolearn: the x_decomp binding publishes no native entry table")
    return decomp_ref(LinearEntryKey(Int(py=key[0]), Int(py=key[1])))


def _decomp_dev(key: EntryRef, op: Int, addresses: List[Int], ints: List[Int], release: Bool) raises -> Int:
    """One pool operation (XD_DEV_ALLOC / _FREE / _UPLOAD) of the image `key`
    names, outside a kit; `release` as the door the call replaces."""
    var result = 0
    var floats = List[Float64]()
    if release:
        var py = Python()
        with GILReleased(py):
            result = decomp_table_call(key, op, addresses, ints, floats)
    else:
        result = decomp_table_call(key, op, addresses, ints, floats)
    return result


def decomp_resident_binding(entries: PythonObject) raises -> Bool:
    """A GPU x_decomp binding's entries dict: the pool entries a resident
    operand is held through (`DecompKit.resident` reads the same marker)."""
    return (py_get(entries, "x_decomp_dev_ew") is not None and py_get(entries, "x_decomp_dev_alloc") is not None
            and py_get(entries, "x_decomp_dev_upload") is not None and py_get(entries, "x_decomp_dev_free") is not None)


def decomp_resident_cached(owner: PythonObject, entries: PythonObject, source: PythonObject, name: String,
                           check_generation: Bool = True) raises -> PythonObject:
    """`owner`'s entry `name` when it holds the own words of exactly this
    frozen `source` on the binding of `entries` (a connected x_decomp
    entries dict): the caller may then skip the per-call admission of
    `source` (neither its values nor its shape can have changed). None on
    any miss, a host binding, or no `source`."""
    if source is None or not decomp_resident_binding(entries):
        return PythonObject(None)
    return decomp_resident_find(owner, entries["x_decomp_dev_alloc"], entries["x_decomp_dev_free"], source, name,
                                True, check_generation)


def decomp_resident_operand(owner: PythonObject, entries: PythonObject, source: PythonObject, name: String,
                            rows: Int, cols: Int) raises -> PythonObject:
    """The entry (word 5: the pool id) of the device copy of the fitted
    float32 array `source` (rows x cols words, its own words: the caller
    passes the stored object, not a converted copy): served from `owner`'s
    record, else uploaded once and recorded (the read-only rule first).
    None on a host binding, an empty array, or storage that cannot be
    frozen: the caller then uploads it per call, as before. The caller
    keeps the entry alive through its native call."""
    var n = rows * cols
    if owner is None or n < 1 or not decomp_resident_binding(entries):
        return PythonObject(None)
    # alloc / free name the minting binding in the record (identity only).
    var alloc = entries["x_decomp_dev_alloc"]
    var free = entries["x_decomp_dev_free"]
    var held = decomp_resident_find(owner, alloc, free, source, name, True)
    if held is not None and Int(py=held[3]) == rows and Int(py=held[4]) == cols:
        return held
    if not decomp_resident_freeze(source):
        return PythonObject(None)
    var key = _decomp_key(entries)
    var handle = _decomp_dev(key, XD_DEV_ALLOC, List[Int](), [n], False)
    try:
        var address: List[Int] = [Int(py=classical_address(source))]
        _ = _decomp_dev(key, XD_DEV_UPLOAD, address, [handle, n], True)
    except error:
        _ = _decomp_dev(key, XD_DEV_FREE, List[Int](), [handle], False)
        raise error^
    _ = source  # keep the Python object alive through the native call above
    return decomp_resident_hold(owner, alloc, free, source, name, rows, cols, handle, True)


def decomp_resident_adopt(owner: PythonObject, entries: PythonObject, source: PythonObject, name: String,
                          rows: Int, cols: Int, handle: Int) raises -> PythonObject:
    """Record the pool buffer `handle` a fitting driver kept on the device
    (it holds the words just published into the fresh Array `source`, rows
    x cols) as `owner`'s entry `name`, its generation pending until the
    first query. When `source` cannot be frozen, or there is no owner, the
    buffer is returned at once and None comes back. Never leaks `handle`."""
    if handle < 0:
        return PythonObject(None)
    var free = entries["x_decomp_dev_free"]
    var key = _decomp_key(entries)
    var frozen = False
    try:
        frozen = owner is not None and decomp_resident_freeze(source)
    except error:
        _ = _decomp_dev(key, XD_DEV_FREE, List[Int](), [handle], False)
        raise error^
    if not frozen:
        _ = _decomp_dev(key, XD_DEV_FREE, List[Int](), [handle], False)
        return PythonObject(None)
    return decomp_resident_hold(owner, entries["x_decomp_dev_alloc"], free, source, name, rows, cols, handle, True,
                                True)


# ---------------------------------------------------------------------------
# The typed kit's door half (lane entangled-decompkit, 2026-11-12). A public
# operation of a decomposition owner is UNPACK (read the Python arrays,
# options, the image and the model's resident records into a `DKit` and
# plain words), RUN (bindings/<estimator>_run.mojo over the `DKit`, the GIL
# released, no Python) and PACK (fresh public Arrays, fitted fields
# published once, resident records written). `DecompDoor` is what UNPACK
# and PACK hold that the RUN must not: the connected entries dict (the
# resident records name their binding by its pool entries), the output
# context, the query model, and every Python owner a borrowed matrix reads,
# kept alive until the door is dropped (after RUN and `kit.close()`).
# ---------------------------------------------------------------------------


struct DecompDoor(Movable):
    var entries: PythonObject
    var output: PythonObject    # the NumPy-free output context: every buffer is an Array
    var owner: PythonObject     # the fitted model of a query, or None
    var keeps: List[PythonObject]

    def __init__(out self, entries: PythonObject, output: PythonObject):
        self.entries = entries
        self.output = output
        self.owner = PythonObject(None)
        self.keeps = List[PythonObject]()

    def keep(mut self, value: PythonObject):
        self.keeps.append(value)

    def keep_alive(self):
        """A named use after RUN and `kit.close()`: Mojo frees a local at its
        last use, and the kit reads the kept owners' words by address."""
        pass


def decomp_kit_unpack(entries: PythonObject) raises -> DKit:
    """UNPACK: the typed kit over a connected x_decomp entries dict: its image
    dispatch (`x_decomp_native_key`, resolved once), whether it is resident
    (`x_decomp_dev_ew`) and which optional entries it exports."""
    var key = py_get(entries, "x_decomp_native_key")
    var entry = entry_ref_unresolved()
    var keyed = key is not None
    if keyed:
        entry = decomp_ref(LinearEntryKey(Int(py=key[0]), Int(py=key[1])))
    var present = 0
    if py_get(entries, "x_decomp_dev_svd") is not None:
        present |= DK_HAS_DEV_SVD
    if py_get(entries, "x_decomp_dev_lu") is not None:
        present |= DK_HAS_DEV_LU
    if py_get(entries, "x_decomp_dev_code_rows") is not None:
        present |= DK_HAS_DEV_CODE_ROWS
    if py_get(entries, "x_decomp_dev_eigh_to") is not None:
        present |= DK_HAS_DEV_EIGH_TO
    if py_get(entries, "x_decomp_s_flags") is not None:
        present |= DK_HAS_S_FLAGS
    if py_get(entries, "x_decomp_qfix_flags") is not None:
        present |= DK_HAS_QFIX_FLAGS
    if py_get(entries, "x_decomp_idn_flags") is not None:
        present |= DK_HAS_IDN_FLAGS
    return DKit(entry, keyed, py_get(entries, "x_decomp_dev_ew") is not None, present)


def decomp_door_borrow(mut door: DecompDoor, mut kit: DKit, array: PythonObject, rows: Int, cols: Int) raises -> Int:
    """UNPACK `kit.input`: a C-order float32 Array the door converted
    (decomp_input), read in place; the door keeps it alive."""
    door.keep(array)
    return kit.borrow(Int(py=classical_address(array)), rows, cols)


def decomp_door_input(mut door: DecompDoor, mut kit: DKit, source: PythonObject, name: String,
                      context: PythonObject) raises -> Int:
    """UNPACK `_M.from_input` into the kit: decomp_input, then borrowed."""
    var array = decomp_input(source, name, context)
    return decomp_door_borrow(door, kit, array, Int(py=array.shape[0]), Int(py=array.shape[1]))


def decomp_door_snapshot(mut door: DecompDoor, mut kit: DKit, snapshot: ArrayMetadata, pin: ArrayPin, rows: Int,
                         cols: Int) raises -> Int:
    """UNPACK `input_snapshot`: the provider's admitted address (the typed
    view), the pin's storage owners kept by the door for the call."""
    door.keep(pin.owner)
    door.keep(pin.keep)
    return kit.borrow(snapshot.source.addr, rows, cols)


def decomp_door_tested(mut door: DecompDoor, mut kit: DKit, snapshot: ArrayMetadata, pin: ArrayPin, rows: Int,
                       cols: Int, est: String) raises -> Int:
    """UNPACK `input_tested`: `decomp_door_snapshot` of an X admitted with
    `defer_finite`, tested on its upload (the GIL released, as before)."""
    door.keep(pin.owner)
    door.keep(pin.keep)
    var index = -1
    var py = Python()
    with GILReleased(py):
        index = kit.borrow_tested(snapshot.source.addr, rows, cols, est)
    return index


def decomp_door_resident_hit(mut door: DecompDoor, mut kit: DKit, source: PythonObject, name: String) raises -> Int:
    """UNPACK `resident_hit`: a fitted operand already resident on this
    image (the stored `source` is the same frozen Array whose own words were
    uploaded), served before the per-call admission. -1 on any miss."""
    if door.owner is None or not kit.resident:
        return -1
    var held = decomp_resident_find(door.owner, door.entries["x_decomp_dev_alloc"],
                                    door.entries["x_decomp_dev_free"], source, name, True)
    if held is None:
        return -1
    door.keep(held)
    door.keep(source)
    return kit.add_resident(Int(py=source._addr), Int(py=held[5]), Int(py=held[3]), Int(py=held[4]))


def decomp_door_resident_input(mut door: DecompDoor, mut kit: DKit, source: PythonObject, name: String,
                               host: PythonObject, address: Int, keep: PythonObject, rows: Int,
                               cols: Int) raises -> Int:
    """UNPACK `resident_input`: a fitted operand of the query model whose
    device copy is kept across calls in the model's record (uploaded at the
    first query after a fit, reused while the stored `source` is the same
    frozen object of the same shape on the minting binding at the model's
    generation). -1 when nothing is kept (a host image, no query model, an
    empty operand, storage that cannot be frozen): the caller borrows the
    operand as before. Same words, same routes (a host matrix at first)."""
    var n = rows * cols
    if door.owner is None or not kit.resident or n < 1:
        return -1
    # Hazard class A: a device copy is served only while the host words
    # cannot change under it (the read-only rule, decomp_resident_freeze).
    if not decomp_resident_freeze(source):
        return -1
    var alloc = door.entries["x_decomp_dev_alloc"]
    var free = door.entries["x_decomp_dev_free"]
    var held = decomp_resident_find(door.owner, alloc, free, source, name, False)
    if held is not None and (Int(py=held[3]) != rows or Int(py=held[4]) != cols):
        held = PythonObject(None)
    if held is None:
        var handle = kit.dev_alloc(n)
        try:
            var py = Python()
            with GILReleased(py):
                kit.dev_upload(handle, address, n)
        except error:
            kit.dev_free(handle)
            raise error^
        _ = keep  # keep the host owner alive through the upload above
        # From here the record's capsule owns the device buffer. Word 7:
        # the uploaded words are the frozen source's own (no cast copy).
        held = decomp_resident_hold(door.owner, alloc, free, source, name, rows, cols, handle, host is source)
    door.keep(held)
    door.keep(keep)
    door.keep(host)
    return kit.add_resident(address, Int(py=held[5]), rows, cols)


def decomp_door_resident_derived(mut door: DecompDoor, mut kit: DKit, source: PythonObject,
                                 name: String) raises -> Int:
    """UNPACK `resident_derived`: a device matrix an earlier query computed
    from the fitted operand `source` and kept, served while `source` is the
    same frozen Array at the same generation (the words this call would
    compute again), added as that call left it. -1 on any miss."""
    if door.owner is None or not kit.resident:
        return -1
    var held = decomp_resident_find(door.owner, door.entries["x_decomp_dev_alloc"],
                                    door.entries["x_decomp_dev_free"], source, name, False)
    if held is None or Bool(py=held[7]):
        return -1
    door.keep(held)
    return kit.add_device(Int(py=held[5]), Int(py=held[3]), Int(py=held[4]))


def decomp_door_resident_keep(mut door: DecompDoor, mut kit: DKit, source: PythonObject, name: String,
                              a: Int) raises -> Bool:
    """PACK `resident_keep`: matrix `a`, computed by RUN from the frozen
    fitted operand `source` and left on the device, goes to the model's
    record for later queries (its pool buffer leaves the kit's ledger for a
    record capsule). False (nothing kept) for a host image, no query model,
    an unfrozen source, or a buffer the kit does not own."""
    if door.owner is None or not kit.resident:
        return False
    if not decomp_resident_frozen(source):
        return False
    var handle = kit.release_device(a)
    if handle < 0:
        return False
    var held = decomp_resident_hold(door.owner, door.entries["x_decomp_dev_alloc"], door.entries["x_decomp_dev_free"],
                                    source, name, kit.nrows(a), kit.ncols(a), handle, False)
    door.keep(held)
    return True


def decomp_kit_out_shape(mut door: DecompDoor, mut kit: DKit, a: Int, shape: PythonObject) raises -> PythonObject:
    """PACK `_M.out(shape)`: a fresh float32 Array of matrix a's words (a
    device matrix comes down once, the GIL released)."""
    var n = kit.size(a)
    var result = classical_typed(shape, "<f4", door.output)
    if n:
        var src = 0
        var py = Python()
        with GILReleased(py):
            src = kit.at(a)
        unsafe_memcpy(dest=DecompF32(unsafe_from_address=Int(py=classical_address(result))),
                      src=DecompF32(unsafe_from_address=src), count=n)
        _ = result  # keep the Python object alive through the copy above
    return result


def decomp_kit_publish(mut door: DecompDoor, mut kit: DKit, a: Int) raises -> PythonObject:
    """PACK `_M.out()`: a fresh (rows, cols) float32 Array of matrix a."""
    return decomp_kit_out_shape(door, kit, a, Python.tuple(kit.nrows(a), kit.ncols(a)))
