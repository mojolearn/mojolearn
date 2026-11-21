# SPDX-License-Identifier: Apache-2.0
"""`byte_lm_trainer_call(trainer, operation, args, context)`: the one native
call of every `LanguageModelTrainer` method
(python/mojolearn/_byte_lm_impl.py `SmallByteLanguageModelTrainer`).

UNPACK / RUN / PACK. The trainer's whole fitted state (weights, optimizer
moments, flags, step counter, optimizer scalars, data schedule, the resident
session and its lifecycle) is ONE native owner in the process registry
(`ByteLmTrainerState`, bindings/byte_lm_trainer_native_run.mojo); the trainer
object holds `_native`, the handle of that state, and no state of its own.

Under the trainer's own lock each operation: UNPACK leases the state once
(one generation check, `_lease`), admits and copies the call's buffers (the
ids as int32 of the exact batch shape) and resolves the binding as `_load`
checks it (a binding change suspends the open session first); RUN is one
function of the run file on the leased state with the GIL released (the
resident step with the session's create / admit / adopt / failure lifecycle,
or the stateless step); PACK builds the public result. The state dictionary
exists only as a PACK result (`export_state`, the checkpoint codecs) and an
UNPACK input (`load_state_dict`, the three checkpoint restores), never as
held state. The same compiled entries with the same argument words as
before, so the step keeps its bits.
"""
from std.memory import ArcPointer, unsafe_memcpy
from std.os import getenv
from std.sys import is_defined
from std.python import Python, PythonObject
from std.python._cpython import GILReleased, PyObjectPtr
from std.python.bindings import ExceptionType, raise_python_exception
from bindings.native_error_class import error_without_class
from bindings.native_loaded_module_door import native_loaded_module, NativeModules
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields, py_contains, py_setattr
from bindings.classical_arrays_door import classical_input, classical_address, classical_cast_kind, classical_is_array, classical_wrap_admitted_layout, classical_dtype_str
from core.cast_in import CAST_KIND_SIGNED
from core.native_array_layout import NativeArrayLayout
from providers.python.array_layout_boundary_pack import native_shape_pack
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.classical_pandas_door import classical_admit_value
from bindings.native_class_admission_door import class_record, class_admit
from bindings.native_buffer_owner_pack import native_buffer_view, native_storage_allocate
from bindings.neural_block_public_door import _neural_lengths
from core.neural_block_public_impl import neural_copy_row_prefixes
from core.host_helpers_impl import argmax_rows_f32
from providers.python.byte_lm_metadata_pyapi_door import _Shape, _shape_of, byte_lm_config_operation
from providers.python.training_metadata_pyapi_unpack import training_f32_scalar, training_schedule_json
from providers.python.training_buffer_pyapi_door import training_words32_into
from core.training_input_copy import training_copy_f32
from core.training_state_validation import training_admit_moments_and_flags, training_byte_shape_validate
from bindings.native_json_door import native_json_parse, native_path_exists
from bindings.native_transport_door import _transport_json
from core.sha256_file import sha256_file
from bindings.byte_checkpoint_io_unpack import ByteCheckpointIOError
from bindings.optimizer_public_words_unpack import optimizer_python_error, PYTHON_ERROR_PENDING
from bindings.byte_checkpoint_native_door import byte_checkpoint_save, byte_checkpoint_load
from providers.python.byte_lm_json_checkpoint_pyapi_door import BYTE_JSON_LIMIT, byte_json_payload, byte_json_arrays, byte_json_encode, byte_json_read, byte_json_write
from bindings.tsa_hw_native_door import tsa_array, tsa_array_unfilled
from bindings.native_array_storage_door import native_scratch_bytes, native_scratch_address
from core.admitted_ids import admitted_ids_i32
from bindings.byte_perf_helpers_pack import byte_int_list_from_i64
from core.byte_lm_teardown_switch import byte_lm_teardown_with_gil
from core.byte_lm_entry_table import ByteLmEntryKey, byte_lm_table_published, byte_lm_vendor_text_word, BYTE_LM_TABLE_TRAINER
from bindings.byte_lm_trainer_native_run import (
    BYTE_PLAN_SCALARS, ByteLmStateData, ByteLmTrainerState, ByteLmTrainerOwner,
    byte_trainer_acquire, byte_trainer_state_ptr, byte_trainer_require_live, byte_trainer_is_resident,
    byte_trainer_forget, byte_trainer_mark_lost, byte_trainer_loss_address, byte_trainer_flags_address,
    byte_trainer_step, byte_trainer_export, byte_trainer_export_release, byte_trainer_session_close,
    byte_trainer_logits, byte_trainer_logits_stateless, byte_trainer_gradients_check,
    byte_trainer_export_gradients, byte_trainer_stage_info, byte_trainer_stateless_begin,
    byte_trainer_stateless_call, byte_trainer_stateless_commit, byte_trainer_stateless_discard,
    byte_trainer_replace,
)
from bindings.byte_lm_trainer_owner_pack import byte_lm_trainer_owner_new
from core.entry_table import EntryRef
from core.device_route import device_route_plan, device_mode_code
from providers.python.device_route_boundary_door import (
    DeviceRouteOwner, device_admit, device_request, device_unpack, device_route_held, device_route_keep, device_route_clear,
)
from training.byte_lm_glue_abi import (
    BYTE_LM_GLUE_ABI, BG_OP_COUNT, BG_ABSENT, BG_NUMERIC_MODE, BG_VENDOR, BG_PROFILE, BG_SUFFIX,
    BG_CONFIG_PROFILE, BG_ATTENTION_ARM, BG_MEMORY_PROFILE, BG_STEP_GLUE_ARM,
    byte_lm_trainer_ref, byte_lm_glue_ref, byte_lm_glue_read_text, byte_lm_glue_read_int, byte_lm_session_new, byte_lm_session_free,
)

comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _TOKEN_REFUSAL = "token ID outside configured vocabulary"
comptime _MAX_STEP = 999999
comptime _PROFILE_REFUSAL = ("RuntimeError: Byte-LM requires the exact native profile, the selected IDENTICAL/FAST mode and "
                             + "CUDA/HIP/Metal vendor (or, on a CPU-only install, the CPU byte LM binding)")


def _b(mut mods: NativeModules) raises -> PythonObject:
    return mods.module("builtins")




def _ids(mut mods: NativeModules, source: PythonObject, shape: _Shape, context: PythonObject) raises -> PythonObject:
    """`_array(ids, (batch, length + 1), "ids", "<i4")`: an owned C-ordered
    int32 copy of an int32 buffer of exactly that shape."""
    var b = _b(mods)
    var refusal = String("TypeError: Byte-LM ids must be a int32 array (a NumPy array, an array.array or a mojolearn Array)")
    classical_admit_value(source, context)
    var buffer = classical_is_array(source, context)
    if not buffer and py_type(source) is not b.list and py_type(source) is not b.tuple:
        try:
            _ = native_buffer_view(source)
            buffer = True
        except:
            mods.released()
            buffer = False
    if not buffer:
        raise Error(refusal)
    var value = classical_input(source, context, "ids")
    # NumPy dtype.__str__ dispatches into Python. Read the admitted native
    # typestr once; the strict int32 contract fixes the cast kind directly.
    var dtype = classical_dtype_str(value, context)
    var item = Int(py=value.itemsize)
    if not ((dtype == "<i4" or dtype == "int32") and item == 4):
        raise Error(refusal)
    var rows = shape.batch()
    var cols = shape.length() + 1
    var dimensions = value.shape
    if len(dimensions) != 2 or Int(py=dimensions[0]) != rows or Int(py=dimensions[1]) != cols:
        raise Error("ValueError: Byte-LM ids requires finite shape (" + String(rows) + ", " + String(cols) + ")")
    var strides = value.strides
    var tokens = tsa_array("i", "<i4", Python.tuple(rows, cols), rows * cols, 4, context)
    _ = admitted_ids_i32(Int(py=classical_address(value)), Int(py=strides[0]), Int(py=strides[1]),
                         rows, cols, item, True, False, _I32P(unsafe_from_address=Int(py=classical_address(tokens))))
    _ = value  # keep the Python object alive through the native call above
    _ = tokens  # keep the Python object alive through the native call above
    _ = value^
    return tokens



def _f32(n: Int, context: PythonObject) raises -> PythonObject:
    return tsa_array("f", "<f4", Python.tuple(n), n, 4, context)


def _f32_unfilled(n: Int, context: PythonObject) raises -> PythonObject:
    """`_f32` with no fill pass, for an output the native call fills in full."""
    return tsa_array_unfilled("f", "<f4", Python.tuple(n), n, 4, context)


def _i32(n: Int, context: PythonObject) raises -> PythonObject:
    return tsa_array("i", "<i4", Python.tuple(n), n, 4, context)


def _gradient_outputs(mut mods: NativeModules, shape: _Shape, named: Bool, context: PythonObject) raises -> PythonObject:
    """`_gradient_outputs(shape, named)`: (flat, {name: Array})."""
    var flat = _f32(shape.n_total(), context)
    var by_name = _b(mods).dict()
    if named:
        var names = shape.names()
        var shapes = shape.shapes()
        for i in range(len(names)):  # driver-loop: one output array per model parameter tensor
            var dims = Python.list()
            var count = 1
            for j in range(len(shapes[i])):  # driver-loop: over one tensor's shape words
                _ = py_call(dims.append, shapes[i][j])
                count *= shapes[i][j]
            py_setitem(by_name, names[i], tsa_array("f", "<f4", py_call(_b(mods).tuple, dims), count, 4, context))
    return Python.tuple(flat, by_name)


def _copy_i32(source: Int, n: Int, context: PythonObject) raises -> PythonObject:
    """PACK: an owned int32 Array copy of `n` words at `source`."""
    # Unfilled (lane fix-output-zero-fill): the memcpy below writes all n words.
    var out = tsa_array_unfilled("i", "<i4", Python.tuple(n), n, 4, context)
    if n > 0:
        unsafe_memcpy(dest=_I32P(unsafe_from_address=Int(py=classical_address(out))),
               src=_I32P(unsafe_from_address=source), count=n)
        _ = out  # keep the Python object alive through the native call above
    return out




# ---------------------------------------------------------------- the binding and the session

def _mode(mut mods: NativeModules, context: PythonObject) raises -> PythonObject:
    """`_mode()`: the process-selected tier."""
    # the backend's own answers, as `_backend.default_mode()` /
    # `_backend.numeric_mode()` gave them (a setting changed after import
    # is seen; the bare `_DEFAULT_MODE` constant is not that answer)
    var backend = context["backend"]
    # Import-time connection resolves the initial default. Read its native
    # configuration once here; a product operation never invokes backend.py.
    var mode = backend["_DEFAULT_MODE"]
    if mode is None or py_type(mode) is not _b(mods).str or (String(py=mode) != "identical" and String(py=mode) != "fast"):
        raise Error("RuntimeError: SmallByteLanguageModelTrainer requires process-selected IDENTICAL or FAST mode")
    return mode


# ---------------------------------------------------------------- the device route
#
# A trainer's device (`device=None | "gpu" | "cpu"`) and the process tier are
# resolved ONCE into the one `DeviceRouteOwner`
# (providers/python/device_route_boundary_door.mojo) with the rule and words
# of core/device_route.mojo. The owner pins `images` = the mode -> image table
# of the route's device (the process set, or the CPU identical image loaded
# beside a GPU set at import: python/mojolearn/_byte_lm_impl.py
# `_connect_trainer`) and `scope` = the vendor names an image of that device
# may report. No operation reads or compares the device: a hit is the mode
# read, one record read and three pointer compares.

@fieldwise_init
struct _ByteTier(Movable):
    """The images one operation selects from: `mode` the admitted tier word,
    `bindings` the mode -> image (or connection error) table of the route's
    device, `vendors` the vendor names such an image may report."""
    var mode: PythonObject
    var bindings: PythonObject
    var vendors: PythonObject


def _process_vendor(context: PythonObject) raises -> Int:
    """This process's selected vendor word for the device rule: 0 on a
    CPU-only installation (its admitted vendor names are `('cpu',)`)."""
    var vendors = context["byte_lm_vendors"]
    return 0 if (len(vendors) == 1 and String(py=vendors[0]) == "cpu") else 1


def _tier_process(mut mods: NativeModules, context: PythonObject) raises -> _ByteTier:
    """The process tier: what an operation without a trainer selects from
    (the "binding" read of tools, a state validated for the host model)."""
    return _ByteTier(_mode(mods, context), context["byte_lm_bindings"], context["byte_lm_vendors"])


def _tier_build(mut mods: NativeModules, requested: Int, mode: PythonObject, context: PythonObject) raises -> DeviceRouteOwner:
    """Resolve a trainer's images once (its first operation after
    construction or restore). A metadata step with no independent units to
    split."""
    var owner = DeviceRouteOwner(requested)
    var mode_code = device_mode_code(String(py=mode))
    var plan = device_route_plan(requested, _process_vendor(context), mode_code)
    if plan.cpu_table:
        # A GPU installation running `device="cpu"`: the CPU identical image
        # connected beside the device images at import.
        var cpu = py_get(context, "byte_lm_cpu_bindings")
        var cpu_vendors = py_get(context, "byte_lm_cpu_vendors")
        if cpu is None or cpu_vendors is None:
            raise Error("ImportError: device='cpu' is not connected in this installation")
        owner.images = cpu
        owner.scope = cpu_vendors
    else:
        owner.images = context["byte_lm_bindings"]
        owner.scope = context["byte_lm_vendors"]
    owner.route.device = plan.device
    owner.route.mode = mode_code
    owner.route.vendor = plan.expected_vendor
    owner.route.flags = 1 if plan.cpu_table else 0
    owner.mode = mode
    owner.default_mode = mode
    owner.context = context
    owner.resolved = True
    return owner^


def _tier(mut mods: NativeModules, trainer: PythonObject, context: PythonObject) raises -> _ByteTier:
    """The trainer's tier for this operation. The device is read only on a
    miss (first operation, a changed process mode object, a constructor run
    again)."""
    var mode = context["backend"]["_DEFAULT_MODE"]
    var owner = device_route_held(trainer, mode, mode, context)
    if owner is None:
        mode = _mode(mods, context)
        var requested = device_request(py_fields(trainer), _b(mods))
        owner = device_route_keep(trainer, _tier_build(mods, requested, mode, context))
    var native = owner.downcast_value_ptr[DeviceRouteOwner]()
    var bindings = native[].images
    var vendors = native[].scope
    _ = owner  # the owner pins both tables through the reads above
    return _ByteTier(mode, bindings, vendors)


def _native_key(mode: PythonObject, vendors: PythonObject) raises -> ByteLmEntryKey:
    """The trainer table key (core/byte_lm_entry_table.mojo) of the image
    the route's device connected: the first of its admitted vendor names
    whose table is published for the mode. The table's glue entry answers
    what the binding's Python metadata entries answered (lane
    calltable-byte-lm); an unpublished table refuses as the profile check
    did."""
    var word = 1 if String(py=mode) == "identical" else 0
    for i in range(len(vendors)):  # small-loop(the admitted vendor names)
        var vendor = byte_lm_vendor_text_word(String(py=vendors[i]))
        if vendor < 0:
            continue
        var key = ByteLmEntryKey(word, vendor)
        if byte_lm_table_published[BYTE_LM_TABLE_TRAINER](key, 1, String(BYTE_LM_GLUE_ABI), BG_OP_COUNT):
            return key^
    raise Error(_PROFILE_REFUSAL)



def _shape_words(shape: _Shape) -> List[Int]:
    """The ten shape words of the glue ABI (nine dimensions, chunked head)."""
    var words = List[Int]()
    for i in range(9):  # driver-loop: over the nine dimensions
        words.append(shape.d[i])
    words.append(Int(shape.chunked))
    return words^


def _glue_text_words(gl: EntryRef, op: Int, var words: List[Int]) raises -> String:
    var text = byte_lm_glue_read_text(gl, op, words^)
    if not text:
        raise Error("ImportError: Byte-LM binding lacks its native metadata entry; rebuild bindings/build_byte_lm.sh")
    return text.value().copy()


def _glue_text(gl: EntryRef, op: Int) raises -> String:
    return _glue_text_words(gl, op, List[Int]())


def _load_snapshot(mut mods: NativeModules, shape: _Shape, context: PythonObject, tier: _ByteTier) raises -> Tuple[PythonObject, PythonObject, String, PythonObject, ByteLmEntryKey]:
    """`_load(shape)`: the byte LM binding of `tier` (the trainer's route, or
    the process tier), its profile, mode and vendor read back through the
    image's native glue entry."""
    var mode = tier.mode
    var binding = tier.bindings[mode]
    if py_type(binding) is _b(mods).str:
        raise Error(String(py=binding))
    var key = _native_key(mode, tier.vendors)
    var gl = byte_lm_glue_ref(key)  # resolved once for the admission reads below
    var suffix = _glue_text(gl, BG_SUFFIX)
    var code = byte_lm_glue_read_int(gl, BG_NUMERIC_MODE, List[Int]())
    var expected = 1 if String(py=mode) == "identical" else 0
    var vendor = _glue_text(gl, BG_VENDOR)
    var vendors = tier.vendors
    var vendor_ok = False
    for i in range(len(vendors)):  # small-loop(the admitted vendor names)
        vendor_ok = vendor_ok or String(py=vendors[i]) == vendor
    var profile = String(py=context["byte_lm_profile"])
    if code != expected or _glue_text(gl, BG_PROFILE) != profile + suffix or not vendor_ok:
        raise Error(_PROFILE_REFUSAL)
    # The table's trainer entry carries the run and runtime-shape operations
    # the former `byte_lm_run` / `byte_lm_run_configured` presence checks
    # asked for; `_native_key` refused an image without it.
    var shape_profile = shape.profile(profile)
    if shape_profile != profile:
        if _glue_text_words(gl, BG_CONFIG_PROFILE, _shape_words(shape)) != shape_profile + suffix:
            raise Error("RuntimeError: Byte-LM native runtime shape/profile mismatch")
    var snapshot = _b(mods).dict()
    py_setitem(snapshot, "native_profile", profile + suffix)
    py_setitem(snapshot, "native_vendor", vendor)
    py_setitem(snapshot, "native_numeric_mode", code)
    return (binding, mode, suffix, snapshot, key^)


def _load(mut mods: NativeModules, shape: _Shape, context: PythonObject) raises -> PythonObject:
    var tier = _tier_process(mods, context)
    var snapshot = _load_snapshot(mods, shape, context, tier)
    return snapshot[0]


# ---------------------------------------------------------------- the native state (UNPACK side)
#
# The trainer object holds `_native`, the `ByteLmTrainerOwner` handle of its
# state in the registry (bindings/byte_lm_trainer_native_run.mojo). Each public
# operation leases the state ONCE here, under the trainer's lock (one
# generation check), and passes the leased state down; nothing below looks the
# handle up again.

def _lease(fields: PythonObject) raises -> ArcPointer[ByteLmTrainerState]:
    var holder = py_get(fields, "_native")
    if holder is None:
        raise Error("RuntimeError: Byte-LM trainer has no native state; construct it or restore a checkpoint")
    var native = holder.downcast_value_ptr[ByteLmTrainerOwner]()
    var handle = native[].handle()
    _ = holder  # the holder pins the handle words through the read above
    return byte_trainer_acquire(handle)


def _state_shape(data: ByteLmStateData) -> _Shape:
    """The admitted model shape, from the state's ten shape words."""
    var d = List[Int](capacity=9)
    for i in range(9):  # driver-loop: over the nine dimensions
        d.append(data.shape[i])
    return _Shape(d^, data.shape[9] != 0)


def _tensor_sizes(shape: _Shape) -> List[Int]:
    """The element count of every parameter tensor, in packing order."""
    var shapes = shape.shapes()
    var sizes = List[Int](capacity=len(shapes))
    for i in range(len(shapes)):  # driver-loop: over the parameter tensors (model structure)
        var count = 1
        for j in range(len(shapes[i])):  # small-loop(one tensor's shape words)
            count *= shapes[i][j]
        sizes.append(count)
    return sizes^


def _table_execute(trainer: EntryRef, operation: Int, addresses: List[Int], ints: List[Int],
                   floats: List[Float64], release_gil: Bool = True) raises -> Int:
    """One trainer operation through the image's table entry (no Python
    provider object): computation releases the GIL as the provider did;
    session finalization keeps it."""
    var result = 0
    # RUN: typed values only (bindings/byte_lm_trainer_native_run.mojo).
    if release_gil:
        with GILReleased(Python()):
            result = byte_trainer_stateless_call(trainer, operation, addresses, ints, floats)
    else:
        result = byte_trainer_stateless_call(trainer, operation, addresses, ints, floats)
    return result


def _reset_pool(mut mods: NativeModules, fields: PythonObject, context: PythonObject) raises:
    """A fresh, empty logits output pool (PACK storage of `logits()`)."""
    var pool = py_call(_b(mods).object.__new__, context["byte_lm_logits_pool"])
    var pool_fields = py_fields(pool)
    py_setitem(pool_fields, "keep", 2)
    py_setitem(pool_fields, "free", _b(mods).list())
    py_setitem(fields, "_logits_pool", pool)


def _forget(mut mods: NativeModules, fields: PythonObject, mut s: ByteLmTrainerState, context: PythonObject) raises:
    """`_forget_session`: the session is dropped with the GIL held (its
    finalizer releases the handle), and the logits pool is emptied."""
    byte_trainer_forget(s)
    _reset_pool(mods, fields, context)


def _close_forget(mut mods: NativeModules, fields: PythonObject, mut s: ByteLmTrainerState,
                  context: PythonObject) raises:
    """Close the session without export (operation 4), then drop it. The
    session is forgotten even when the close refuses."""
    try:
        with GILReleased(Python()):
            byte_trainer_session_close(s)
        mods.released()
    except error:
        mods.released()
        _forget(mods, fields, s, context)
        raise error^
    _forget(mods, fields, s, context)


def _suspend(mut mods: NativeModules, fields: PythonObject, mut s: ByteLmTrainerState, context: PythonObject) raises:
    """`_suspend_session`: an open session's state comes back to the host
    (one export that releases the device), then the session is dropped."""
    if s.session_open and not s.lost:
        try:
            with GILReleased(Python()):
                byte_trainer_export_release(s)
            mods.released()
        except error:
            mods.released()
            # the native export closed the session on its way out
            byte_trainer_mark_lost(s)
            _forget(mods, fields, s, context)
            raise error^
        _forget(mods, fields, s, context)
        return
    _close_forget(mods, fields, s, context)


def _binding(mut mods: NativeModules, trainer: PythonObject, fields: PythonObject, mut s: ByteLmTrainerState,
             shape: _Shape, context: PythonObject) raises -> PythonObject:
    """`_binding()`: a binding change suspends the open session first; the
    run metadata is then computed on its next read (`run_metadata`)."""
    # A compiled binding's profile, vendor and numeric-mode answers are fixed
    # for the module object. Once this trainer admitted the same binding for
    # the same mode and shape profile, reuse that admission instead of five
    # native queries per train step / generated token.
    var tier = _tier(mods, trainer, context)
    var mode = tier.mode
    var selected = tier.bindings[mode]
    var shape_profile = shape.profile(String(py=context["byte_lm_profile"]))
    var key = py_get(fields, "_admitted_binding_key")
    if (key is not None and s.key_mode >= 0 and selected is fields["_runtime_binding"] and key[0] is selected
            and String(py=key[1]) == String(py=mode) and String(py=key[2]) == shape_profile
            and py_get(fields, "_admitted_binding_metadata") is not None):
        return selected
    var admitted = _load_snapshot(mods, shape, context, tier)
    var binding = admitted[0]
    py_setitem(fields, "_admitted_binding_metadata", admitted[3])
    s.key_mode = admitted[4].mode
    s.key_vendor = admitted[4].vendor
    if binding is not fields["_runtime_binding"]:
        _suspend(mods, fields, s, context)
        py_setitem(fields, "_runtime", PythonObject(None))
        py_setitem(fields, "_runtime_binding", binding)
    py_setitem(fields, "_admitted_binding_key", Python.tuple(binding, admitted[1], PythonObject(shape_profile)))
    return binding


def _gradient_addresses(mut mods: NativeModules, grads: PythonObject) raises -> List[Int]:
    """The named gradient outputs' addresses, in registry order."""
    var named = List[Int]()
    var by_name = grads[1]
    var keys = py_call(_b(mods).list, by_name)
    for i in range(len(keys)):  # driver-loop: one address per model parameter tensor
        named.append(Int(py=classical_address(by_name[keys[i]])))
    return named^


def _step_result(mut mods: NativeModules, s: ByteLmTrainerState, shape: _Shape, expected: Int, loss: Float64,
                 full: Bool, grads: PythonObject, context: PythonObject) raises -> PythonObject:
    """PACK: a training step's public result."""
    var result = _b(mods).dict()
    py_setitem(result, "loss", PythonObject(loss))
    py_setitem(result, "step", expected)
    py_setitem(result, "completed_steps", expected)
    py_setitem(result, "next_batch_index", expected)
    if full:
        py_setitem(result, "flat_gradients", grads[0])
        py_setitem(result, "gradients", grads[1])
    else:
        py_setitem(result, "flags", _copy_i32(byte_trainer_flags_address(s), shape.n_tensors(), context))
    return result


def _run_resident(mut mods: NativeModules, mut s: ByteLmTrainerState, tokens: PythonObject, shape: _Shape,
                  context: PythonObject, train: Bool, n_steps: Int = 1, steps_call: Bool = False) raises -> PythonObject:
    """One native call: `n_steps` training steps (`tokens` the (n_steps,
    batch, length + 1) stack; one Program.fit on the device column) or one
    evaluation. `steps_call`: the result also carries the step `losses`."""
    var b = _b(mods)
    byte_trainer_require_live(s)
    # UNPACK: the call's buffers as addresses (plan row N104). A single
    # step's loss is the state's own word; `train_steps` returns its losses.
    var full = train and not s.lean
    var out_loss = PythonObject(None)
    var loss_address = byte_trainer_loss_address(s)
    if steps_call:
        out_loss = _f32(n_steps, context)
        loss_address = Int(py=classical_address(out_loss))
    var grads = _gradient_outputs(mods, shape, True, context) if full else Python.tuple(PythonObject(None), b.dict())
    var named = _gradient_addresses(mods, grads)
    var flat = Int(py=classical_address(grads[0])) if full else 0
    var token_address = Int(py=classical_address(tokens))
    var expected = 0
    # RUN: the state's flags, buffers, step counter and optimizer scalars are
    # read from the leased native state; no Python object is read.
    with GILReleased(Python()):
        expected = byte_trainer_step(s, token_address, n_steps, train, loss_address, flat, named)
    mods.released()
    _ = tokens  # keep the Python objects alive through the native call above
    _ = out_loss
    # PACK
    var last_loss = Float64(_F32P(unsafe_from_address=loss_address)[n_steps - 1])
    if not train:
        return PythonObject(last_loss)
    var result = _step_result(mods, s, shape, expected, last_loss, full, grads, context)
    if steps_call:
        py_setitem(result, "losses", out_loss)
    return result


def _run_stateless(mut mods: NativeModules, mut s: ByteLmTrainerState, tokens: PythonObject, shape: _Shape,
                   context: PythonObject, train: Bool) raises -> PythonObject:
    var b = _b(mods)
    # A lean step's gradient stays in the native state for export_gradients()
    # (same bytes as the full result's); a full step writes the result arrays.
    var kept = train and s.lean
    var full = train and not s.lean
    var grads = _gradient_outputs(mods, shape, True, context) if full else Python.tuple(PythonObject(None), b.dict())
    var named = _gradient_addresses(mods, grads)
    var flat = Int(py=classical_address(grads[0])) if full else 0
    var sizes = List[Int]()
    if kept:
        sizes = _tensor_sizes(shape)
    var loss_address = byte_trainer_loss_address(s)
    # The stateless step's session is native and call-scoped (glue entry
    # BG_SESSION_NEW / BG_SESSION_FREE): no Python session object, handle
    # factory or provider per call (lane calltable-byte-lm). Descriptor
    # order: the session, the eleven stateless buffers, then the named
    # gradient buffers (training/byte_lm_native_dispatch.mojo operation 8).
    var descriptors = byte_trainer_stateless_begin(s, Int(py=classical_address(tokens)), train, loss_address, flat,
                                                   named, sizes)
    var completed = 0
    try:
        var locked_teardown = byte_lm_teardown_with_gil()
        var plan = s.plan(train, 0, 0, locked_teardown)
        var key = s.key()
        var gl = byte_lm_glue_ref(key)
        var trainer = byte_lm_trainer_ref(key)
        var session = byte_lm_session_new(gl)
        try:
            descriptors[0] = session
            completed = _table_execute(trainer, 8, descriptors, plan.ints, plan.floats)
            if locked_teardown:
                var finalize: List[Int] = [session]
                _ = _table_execute(trainer, 9, finalize, List[Int](), List[Float64](), False)
        finally:
            byte_lm_session_free(gl, session)
    except error:
        byte_trainer_stateless_discard(s)
        raise error^
    mods.released()
    _ = tokens  # keep the Python object alive through the native calls above
    var expected = byte_trainer_stateless_commit(s, completed, train, kept)
    var loss = Float64(_F32P(unsafe_from_address=loss_address)[0])
    if not train:
        return PythonObject(loss)
    return _step_result(mods, s, shape, expected, loss, full, grads, context)


def _optimizer_scalars(mut mods: NativeModules, lr: PythonObject, beta_values: PythonObject, eps: PythonObject,
                       weight_decay: PythonObject, context: PythonObject) raises -> List[Float64]:
    """UNPACK: the optimizer configuration as the plan's eight scalars (lr,
    beta1, beta2, eps, weight_decay, then momentum, dampening and max_norm,
    which the AdamW profile holds at zero)."""
    var bi = _b(mods)
    var betas: PythonObject
    try:
        # Only already-admitted native containers/buffers reach iteration.
        betas = py_call(bi.tuple, beta_values)
    except:
        raise Error("ValueError: Byte-LM betas must contain two scalars")
    if len(betas) != 2:
        raise Error("ValueError: Byte-LM betas must contain two scalars")
    var keys = [String("lr"), String("beta1"), String("beta2"), String("eps"), String("weight_decay")]
    var scalars: List[PythonObject] = [lr, betas[0], betas[1], eps, weight_decay]
    var numeric = List[Float64](capacity=BYTE_PLAN_SCALARS)
    for i in range(5):  # driver-loop: over the five optimizer configuration scalars
        numeric.append(training_f32_scalar(scalars[i], "Byte-LM " + keys[i], context, True))
        mods.released()
    if numeric[0] <= 0 or numeric[3] <= 0 or numeric[4] < 0:
        raise Error("ValueError: Byte-LM requires lr > 0, eps > 0 and weight_decay >= 0")
    if numeric[1] < 0 or numeric[1] >= 1 or numeric[2] < 0 or numeric[2] >= 1:
        raise Error("ValueError: Byte-LM betas must remain in [0, 1) in float32")
    numeric.append(0.0)
    numeric.append(0.0)
    numeric.append(0.0)
    return numeric^


def _require_training_profile(mut mods: NativeModules, context: PythonObject) raises:
    var bi = _b(mods)
    var profile = py_get(context["byte_lm_profile_defaults"], "training", PythonObject("fp32_v1"))
    if py_type(profile) is not bi.str or not Bool(py=context["byte_lm_profiles"][profile]["training"]):
        raise Error("NotImplementedError: the selected numeric profile is unavailable for Byte-LM training")


def _construction_options(mut mods: NativeModules, resident: PythonObject, var step_result: PythonObject) raises -> PythonObject:
    var bi = _b(mods)
    if resident is not None and py_type(resident) is not bi.bool:
        raise Error("TypeError: resident must be a bool")
    # Default (Andrew, 2026-11-06): 'lean' for every trainer that is not
    # explicitly stateless, so a resident step copies only the loss and the
    # flags to the host; the gradient is exported on request. The same
    # default holds on every vendor and on the CPU column. An explicitly
    # stateless trainer (resident=False) keeps 'full'. A trainer whose
    # automatic choice resolves to the stateless route under 'lean' returns
    # the same lean keys and keeps the step's gradient for export_gradients().
    var explicit_stateless = resident is not None and not Bool(py=resident)
    if step_result is None:
        step_result = PythonObject("full" if explicit_stateless else "lean")
    if py_type(step_result) is not bi.str or (String(py=step_result) != "full" and String(py=step_result) != "lean"):
        raise Error("ValueError: step_result must be 'full' or 'lean'")
    if String(py=step_result) == "lean" and explicit_stateless:
        raise Error("ValueError: step_result='lean' requires a resident session; resident=False returns the full result")
    return step_result


def _schedule_vocabulary(mut mods: NativeModules, schedule: PythonObject, shape: _Shape) raises:
    var bi = _b(mods)
    var vocabulary = py_get(schedule, "vocabulary")
    if vocabulary is not None:
        if py_type(vocabulary) is not bi.dict or py_type(py_get(vocabulary, "n_vocab")) is not bi.int or py_type(py_get(vocabulary, "sha256")) is not bi.str:
            raise Error("ValueError: Byte-LM data_schedule vocabulary must carry an int n_vocab and a str sha256")
        if Int(py=vocabulary["n_vocab"]) != shape.vocab():
            raise Error("ValueError: Byte-LM data_schedule vocabulary does not match the configured vocabulary")


def _initialize_fields(mut mods: NativeModules, trainer: PythonObject, var data: ByteLmStateData,
                       resident: PythonObject, step_result: PythonObject, context: PythonObject) raises:
    """PACK of construction and restore: the state enters the registry and
    the trainer keeps its handle, its lock and its (empty) binding record."""
    var fields = py_fields(trainer)
    var resident_word = -1 if resident is None else (1 if Bool(py=resident) else 0)
    var state = ByteLmTrainerState(data^, resident_word, String(py=step_result) == "lean")
    py_setitem(fields, "_lock", py_call(context["byte_lm_lock"]))
    py_setitem(fields, "_runtime", PythonObject(None))
    py_setitem(fields, "_runtime_binding", PythonObject(None))
    # Replacing `_native` (a constructor run again) drops the former state
    # and its session through the holder's finalizer.
    py_setitem(fields, "_native", byte_lm_trainer_owner_new(state^))
    _reset_pool(mods, fields, context)


def byte_lm_construct(mut mods: NativeModules, trainer: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    var bi = _b(mods)
    if py_type(trainer) is not context["byte_lm_trainer_class"]:
        raise Error("TypeError: Byte-LM requires its native trainer class without Python subclass hooks")
    _require_training_profile(mods, context)
    # The device is checked where it is given, in the shared words, before
    # any field of the trainer changes; the route is resolved below, once.
    device_admit(args[9], _process_vendor(context), bi)
    # Every external value is admitted before scalar, buffer or JSON protocols.
    for i in range(6):  # small-loop(parameter data, schedule and five optimizer arguments)
        classical_admit_value(args[i], context)
    var resident = args[7]
    var step_result = _construction_options(mods, resident, args[8])
    var config_value = args[6]
    var shape_state = bi.dict()
    if config_value is not None:
        if py_type(config_value) is not context["byte_lm_config_class"]:
            raise Error("TypeError: shape must be a ByteLanguageModelConfig")
        var config_fields = py_fields(config_value)
        classical_admit_value(config_fields, context)
        py_setitem(shape_state, "model_shape", config_fields)
    var shape = _shape_of(shape_state)
    # The selected native binding validates configured dimensions before any
    # allocation; a host-only glue binary must not impose its own numeric mode.
    py_setitem(py_fields(trainer), "device", args[9])
    device_route_clear(trainer)
    var tier = _tier(mods, trainer, context)
    var loaded = _load_snapshot(mods, shape, context, tier)
    var names = shape.names()
    var shapes = shape.shapes()
    var total = shape.n_total()
    var base_profile = String(py=context["byte_lm_profile"])
    var shape_profile = shape.profile(base_profile)
    # The state's buffers are native from the start: the parameters are
    # admitted (dtype, shape, finite) straight into the state's own buffer,
    # one copy; the moments and the flags start at zero.
    var data = ByteLmStateData(_shape_words(shape), total, shape.n_tensors(), shape_profile != base_profile,
                               shape_profile + loaded[2], 1 if String(py=loaded[1]) == "identical" else 0, True)
    var dest = Int(data.parameters.unsafe_ptr())
    var parameters = args[0]
    if py_type(parameters) is bi.dict:
        if len(parameters) != len(names):
            raise Error("ValueError: Byte-LM named parameters must contain the exact configured tensor registry")
        var offset = 0
        for i in range(len(names)):  # driver-loop: admit each configured parameter tensor once
            if not py_contains(parameters, names[i]):
                raise Error("ValueError: Byte-LM named parameters must contain the exact configured tensor registry")
            training_words32_into(parameters[names[i]], shapes[i], "Byte-LM " + names[i], context, False,
                                  dest + offset * 4)
            var count = 1
            for width in shapes[i]:  # small-loop(parameter rank)
                count *= width
            offset += count
    else:
        training_words32_into(parameters, [total], "Byte-LM parameters", context, False, dest)
    data.scalars = _optimizer_scalars(mods, args[2], args[3], args[4], args[5], context)
    data.schedule = training_schedule_json(args[1], 256, 16384, "Byte-LM data_schedule", context)
    mods.released()
    _schedule_vocabulary(mods, native_json_parse(PythonObject(data.schedule)), shape)
    mods.released()
    _initialize_fields(mods, trainer, data^, resident, step_result, context)
    return PythonObject(None)



def _state_step(mut mods: NativeModules, value: PythonObject) raises -> Int:
    var bi = _b(mods)
    if py_type(value) is bi.bool or String(py=py_type(value).__name__) == "bool_":
        raise Error("ValueError: Byte-LM completed_steps must be an integer")
    var step = 0
    try:
        step = Int(py=py_call(mods.module("operator").index, value))
    except:
        raise Error("ValueError: Byte-LM completed_steps must be an integer")
    if step < 0 or step > _MAX_STEP:
        raise Error("ValueError: Byte-LM completed_steps must be in [0, 999999]")
    return step


def byte_state_unpack(mut mods: NativeModules, source: PythonObject, context: PythonObject,
                      tier: _ByteTier) raises -> ByteLmStateData:
    """UNPACK of a public state dictionary (`load_state_dict`, the three
    checkpoint restores, the host model's restore) into the typed state.

    Admission and every copy complete before any trainer or session is
    touched: the result is a detached native value. Each of the four arrays
    is admitted (dtype, shape, finite) straight into the state's own buffer,
    one copy each."""
    classical_admit_value(source, context)
    var bi = _b(mods)
    if py_type(source) is not bi.dict:
        raise Error("ValueError: Byte-LM state has missing or unknown fields")
    var keys = Python.list("schema", "profile", "numeric_mode", "parameter_names", "parameter_shapes",
        "parameter_offsets", "parameters", "m", "v", "flags", "completed_steps",
        "next_batch_index", "config", "data_schedule")
    var has_model_shape = py_contains(source, "model_shape")
    if has_model_shape:
        _ = py_call(keys.append, "model_shape")
    if not Bool(py=py_call(bi.set, source) == py_call(bi.set, keys)):
        raise Error("ValueError: Byte-LM state has missing or unknown fields")
    var shape = _shape_of(source)
    training_byte_shape_validate(shape.d, shape.chunked)
    var mode = source["numeric_mode"]
    if py_type(mode) is not bi.str or (String(py=mode) != "identical" and String(py=mode) != "fast"):
        raise Error("ValueError: Byte-LM state profile/registry/mode mismatch")
    # The state's own tier word selects among the images of the route's device.
    var binding = py_get(tier.bindings, mode)
    if binding is None:
        # the route's device has no image of the state's tier (a `fast`
        # state on a CPU route)
        raise Error("ValueError: Byte-LM state profile/registry/mode mismatch")
    if py_type(binding) is bi.str:
        raise Error("ImportError: " + String(py=binding))
    var suffix = _glue_text(byte_lm_glue_ref(_native_key(mode, tier.vendors)), BG_SUFFIX)
    var expected_profile = shape.profile(String(py=context["byte_lm_profile"])) + suffix
    var registry = _registry(mods, shape)
    var names = Python.list()
    var shapes = Python.list()
    var offsets = Python.list(0)
    for i in range(len(registry)):  # driver-loop: over configured state registry captured once
        var item = registry[i]
        _ = py_call(names.append, item["name"])
        _ = py_call(shapes.append, py_call(bi.list, item["shape"]))
        _ = py_call(offsets.append, Int(py=item["offset"]) + Int(py=item["size"]))
    if (not Bool(py=source["schema"] == PythonObject("mojolearn.small-byte-lm-state.v1"))
            or not Bool(py=source["profile"] == PythonObject(expected_profile))
            or not Bool(py=source["parameter_names"] == names)
            or not Bool(py=source["parameter_shapes"] == shapes)
            or not Bool(py=source["parameter_offsets"] == offsets)):
        raise Error("ValueError: Byte-LM state profile/registry/mode mismatch")
    var cfg = source["config"]
    var config_keys = Python.list("lr", "beta1", "beta2", "eps", "weight_decay", "kind",
        "momentum", "dampening", "nesterov", "max_norm")
    if py_type(cfg) is not bi.dict or not Bool(py=py_call(bi.set, cfg) == py_call(bi.set, config_keys)):
        raise Error("ValueError: Byte-LM state optimizer configuration mismatch")
    var scalars = _optimizer_scalars(mods, cfg["lr"], Python.tuple(cfg["beta1"], cfg["beta2"]),
                                     cfg["eps"], cfg["weight_decay"], context)
    if py_type(cfg["kind"]) is not bi.int or Int(py=cfg["kind"]) != 2 or py_type(cfg["nesterov"]) is not bi.bool or Bool(py=cfg["nesterov"]):
        raise Error("ValueError: Byte-LM first profile supports AdamW without clipping or SGD options")
    for name in [String("momentum"), String("dampening"), String("max_norm")]:  # small-loop(the three unused optimizer slots)
        if training_f32_scalar(cfg[name], name, context, True) != 0:
            mods.released()
            raise Error("ValueError: Byte-LM first profile supports AdamW without clipping or SGD options")
        mods.released()
    var completed = _state_step(mods, source["completed_steps"])
    if _state_step(mods, source["next_batch_index"]) != completed:
        raise Error("ValueError: Byte-LM schedule cursor must match completed_steps")
    var count = shape.n_total()
    var tensors = shape.n_tensors()
    var data = ByteLmStateData(_shape_words(shape), count, tensors, has_model_shape, expected_profile.copy(),
                               1 if String(py=mode) == "identical" else 0, False)
    training_words32_into(source["parameters"], [count], "Byte-LM parameters", context, False,
                          Int(data.parameters.unsafe_ptr()))
    training_words32_into(source["m"], [count], "Byte-LM m", context, False, Int(data.m.unsafe_ptr()))
    training_words32_into(source["v"], [count], "Byte-LM v", context, False, Int(data.v.unsafe_ptr()))
    training_words32_into(source["flags"], [tensors], "Byte-LM flags", context, True, Int(data.flags.unsafe_ptr()))
    mods.released()
    training_admit_moments_and_flags(Int(data.v.unsafe_ptr()), count, Int(data.flags.unsafe_ptr()), tensors)
    data.schedule = training_schedule_json(source["data_schedule"], 256, 16384, "Byte-LM data_schedule", context)
    mods.released()
    data.completed = completed
    data.scalars = scalars^
    return data^


def byte_lm_load_state(mut mods: NativeModules, trainer: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    if py_type(trainer) is not context["byte_lm_trainer_class"]:
        raise Error("TypeError: Byte-LM requires its native trainer class without Python subclass hooks")
    var fields = py_fields(trainer)
    var lock = fields["_lock"]
    _ = py_call(lock.acquire)
    mods.released()
    try:
        var lease = _lease(fields)
        var s = byte_trainer_state_ptr(lease)
        var tier = _tier(mods, trainer, context)
        # UNPACK completes (every refusal raised) before the live state changes.
        var replacement = byte_state_unpack(mods, args[0], context, tier)
        # close (operation 4) without export, then the session drops
        _close_forget(mods, fields, s[], context)
        byte_trainer_replace(s[], replacement^)
        _ = lease  # the lease keeps the state alive through the writes above
    except problem:
        mods.released()
        _ = py_call(lock.release)
        raise problem
    _ = py_call(lock.release)
    return trainer


def byte_lm_checkpoint(mut mods: NativeModules, trainer: PythonObject, operation: String, args: PythonObject,
                       context: PythonObject, mut io_error: ByteCheckpointIOError) raises -> PythonObject:
    classical_connect_numpy(context)
    if operation == "from_checkpoint_binary" or operation == "from_checkpoint" or operation == "from_checkpoint_bytes":
        if trainer is not context["byte_lm_trainer_class"]:
            raise Error("TypeError: checkpoint restoration requires the connected native trainer class")
        # `device=` is checked where it is given, in the shared words, before
        # the file is read.
        device_admit(args[2], _process_vendor(context), _b(mods))
        var source = PythonObject(None)
        if operation == "from_checkpoint_binary":
            source = byte_checkpoint_load(args[0], io_error)
            mods.released()
        else:
            var encoded = args[0] if operation == "from_checkpoint_bytes" else byte_json_read(args[0], io_error)
            mods.released()
            source = byte_json_payload(encoded)
            mods.released()
            var described = _shape_of(source)
            training_byte_shape_validate(described.d, described.chunked)
            byte_json_arrays(source, described.n_total(), described.n_tensors())
            mods.released()
        # The restored trainer's device selects the images its state is
        # validated against; the trainer's own route is resolved at its
        # first operation, from the `device` field written below.
        var route_mode = _mode(mods, context)
        var route = _tier_build(mods, device_unpack(args[2], _b(mods)), route_mode, context)
        var tier = _ByteTier(route_mode, route.images, route.scope)
        # UNPACK: the codec's dictionary into the typed state, once.
        var data = byte_state_unpack(mods, source, context, tier)
        var shape = _state_shape(data)
        _require_training_profile(mods, context)
        var resident = args[1]
        var step_result = _construction_options(mods, resident, PythonObject(None))
        _schedule_vocabulary(mods, native_json_parse(PythonObject(data.schedule)), shape)
        mods.released()
        # Preserve constructor binding/profile admission without constructing
        # temporary parameter arrays or invoking a Python class constructor.
        _ = _load_snapshot(mods, shape, context, tier)
        var restored = native_storage_allocate(trainer)
        py_setitem(py_fields(restored), "device", args[2])
        _initialize_fields(mods, restored, data^, resident, step_result, context)
        return restored
    if operation != "export_checkpoint_binary" and operation != "export_checkpoint":
        raise Error("RuntimeError: unknown Byte-LM checkpoint operation")
    if py_type(trainer) is not context["byte_lm_trainer_class"]:
        raise Error("TypeError: checkpoint export requires the connected native trainer class")
    var fields = py_fields(trainer)
    var lock = fields["_lock"]
    _ = py_call(lock.acquire)
    mods.released()
    var state = PythonObject(None)
    try:
        var lease = _lease(fields)
        var s = byte_trainer_state_ptr(lease)
        var shape = _state_shape(s[].data)
        if operation == "export_checkpoint" and shape.n_total() * 24 + shape.n_tensors() * 8 > BYTE_JSON_LIMIT:
            raise Error("Byte-LM checkpoint exceeds 2 MiB; export state_dict arrays")
        state = _export_snapshot(mods, s[], shape, context)
        _ = lease
    except problem:
        mods.released()
        _ = py_call(lock.release)
        raise problem
    _ = py_call(lock.release)
    # The export owns detached arrays. File I/O follows lock release exactly
    # as before, using the existing atomic codec and original I/O exceptions.
    if operation == "export_checkpoint":
        byte_json_write(args[0], byte_json_encode(state), io_error)
        mods.released()
        return PythonObject(None)
    var windowed_result1 = byte_checkpoint_save(args[0], state, io_error)
    mods.released()
    return windowed_result1^


def byte_lm_train_step(mut mods: NativeModules, trainer: PythonObject, args: PythonObject, context: PythonObject, train: Bool = True) raises -> PythonObject:
    """`train_step(ids)`: `with self._lock: return self._run(ids, True)`."""
    classical_connect_numpy(context)
    var fields = py_fields(trainer)
    var lock = fields["_lock"]
    _ = py_call(lock.acquire)
    mods.released()
    var result = PythonObject(None)
    try:
        # UNPACK: the state lease (one generation check), the ids, the binding.
        var lease = _lease(fields)
        var s = byte_trainer_state_ptr(lease)
        var shape = _state_shape(s[].data)
        var tokens = _ids(mods, args[0], shape, context)
        _ = _binding(mods, trainer, fields, s[], shape, context)
        try:
            if byte_trainer_is_resident(s[]):
                result = _run_resident(mods, s[], tokens, shape, context, train)
            else:
                result = _run_stateless(mods, s[], tokens, shape, context, train)
        except error:
            mods.released()
            var text = String(error)
            if not text.startswith("ValueError:") and text.find(_TOKEN_REFUSAL) >= 0:
                raise Error("ValueError: Byte-LM IDs must be in [0, " + String(shape.vocab()) + ")")
            raise error^
        _ = lease  # the lease keeps the state alive through the operation
    except error:
        mods.released()
        _ = py_call(lock.release)
        raise error^
    _ = py_call(lock.release)
    return result



# ---------------------------------------------------------------- the multi-step entry
#
# Plan section 2C (rows N107, N109, N101, N104; 2026-11-07) and lane
# x7-byte-lm-on-executor (2026-11-09): `train_steps(ids)` takes n_steps
# batches in one call. The lock, the state admission, the shape, the ids
# and the binding are admitted ONCE, then ONE native call (operation 1 with
# the step count) runs every step: the hand step per batch (training/byte_lm.mojo).
# TOMBSTONE: the binding's `while k < n_steps` step driver (the opening step, 'full' results and the stateless route one call each, the rest through operation 12) deleted 2026-11-09 by lane x7-byte-lm-on-executor; code recoverable at the deletion commit's parent.
# Restore: git apply experiments/removed/BYTE_LM_HAND_STEP.patch


def _ids_steps(mut mods: NativeModules, source: PythonObject, shape: _Shape,
               context: PythonObject) raises -> Tuple[PythonObject, Int]:
    """The `train_steps` ids: an int32 buffer of shape (n_steps, batch,
    length + 1), admitted into one owned C-ordered int32 copy (the step
    batches one after the other); returns the copy and n_steps."""
    var b = _b(mods)
    var refusal = String("TypeError: Byte-LM ids must be a int32 array (a NumPy array, an array.array or a mojolearn Array)")
    classical_admit_value(source, context)
    var buffer = classical_is_array(source, context)
    if not buffer and py_type(source) is not b.list and py_type(source) is not b.tuple:
        try:
            _ = native_buffer_view(source)
            buffer = True
        except:
            mods.released()
            buffer = False
    if not buffer:
        raise Error(refusal)
    var value = classical_input(source, context, "ids")
    var dtype = classical_dtype_str(value, context)
    var item = Int(py=value.itemsize)
    if not ((dtype == "<i4" or dtype == "int32") and item == 4):
        raise Error(refusal)
    var rows = shape.batch()
    var cols = shape.length() + 1
    var dimensions = value.shape
    if (len(dimensions) != 3 or Int(py=dimensions[0]) < 1 or Int(py=dimensions[0]) > _MAX_STEP
            or Int(py=dimensions[1]) != rows or Int(py=dimensions[2]) != cols):
        raise Error("ValueError: Byte-LM train_steps ids requires shape (n_steps, " + String(rows) + ", "
                    + String(cols) + ") with n_steps >= 1")
    var n_steps = Int(py=dimensions[0])
    var strides = value.strides
    var s0 = Int(py=strides[0])
    var s1 = Int(py=strides[1])
    var s2 = Int(py=strides[2])
    var tokens = tsa_array("i", "<i4", Python.tuple(n_steps, rows, cols), n_steps * rows * cols, 4, context)
    var source_address = Int(py=classical_address(value))
    var target_address = Int(py=classical_address(tokens))
    for k in range(n_steps):  # driver-loop: over the step batches (one strided 2-D cast each)
        _ = admitted_ids_i32(source_address + k * s0, s1, s2, rows, cols, item, True, False,
                             _I32P(unsafe_from_address=target_address + k * rows * cols * 4))
    _ = value  # keep the Python object alive through the native calls above
    _ = tokens  # keep the Python object alive through the native calls above
    _ = value^
    return (tokens, n_steps)


def _ids_batch(stack: PythonObject, k: Int, shape: _Shape, context: PythonObject) raises -> PythonObject:
    """Batch `k` of an admitted `train_steps` stack as its own (batch,
    length + 1) int32 array, for the per-step entry."""
    var cells = shape.batch() * (shape.length() + 1)
    var out = tsa_array("i", "<i4", Python.tuple(shape.batch(), shape.length() + 1), cells, 4, context)
    unsafe_memcpy(dest=_I32P(unsafe_from_address=Int(py=classical_address(out))),
           src=_I32P(unsafe_from_address=Int(py=classical_address(stack)) + k * cells * 4), count=cells)
    _ = stack  # keep the Python object alive through the native call above
    return out


def _run_steps(mut mods: NativeModules, mut s: ByteLmTrainerState, stack: PythonObject, n_steps: Int, shape: _Shape,
               context: PythonObject) raises -> PythonObject:
    """`train_steps`: on a resident trainer ONE native call (operation 1 with
    the step count: the session opens from the host state when it is not
    open, the steps run as one Program.fit on the device column, a 'full'
    result exports the last step's gradient). The stateless route
    (`resident=False`) is one whole-state native call per step by design
    (the state crosses the boundary every step), so it keeps one call per
    batch."""
    if byte_trainer_is_resident(s):
        return _run_resident(mods, s, stack, shape, context, True, n_steps, True)
    var losses = _f32(n_steps, context)
    var loss_out = _F32P(unsafe_from_address=Int(py=classical_address(losses)))
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var last = Optional[PythonObject]()
    for k in range(n_steps):  # driver-loop: over the stateless route's steps (each one whole-state native call)
        var batch = _ids_batch(stack, k, shape, context)
        var step = _run_stateless(mods, s, batch, shape, context, True)
        loss_out[k] = Float32(Float64(py=step["loss"]))
        last = step
    if not last:
        raise Error("TypeError: 'NoneType' object does not support item assignment")
    var result = last.value()
    py_setitem(result, "losses", losses)
    _ = losses  # keep the Python object alive through the native writes above
    return result


def byte_lm_train_steps(mut mods: NativeModules, trainer: PythonObject, args: PythonObject,
                        context: PythonObject) raises -> PythonObject:
    """`train_steps(ids)`: `with self._lock:` n_steps training steps."""
    classical_connect_numpy(context)
    var fields = py_fields(trainer)
    var lock = fields["_lock"]
    _ = py_call(lock.acquire)
    mods.released()
    var result = PythonObject(None)
    try:
        var lease = _lease(fields)
        var s = byte_trainer_state_ptr(lease)
        var shape = _state_shape(s[].data)
        var admitted = _ids_steps(mods, args[0], shape, context)
        var stack = admitted[0]
        var n_steps = admitted[1]
        _ = _binding(mods, trainer, fields, s[], shape, context)
        try:
            result = _run_steps(mods, s[], stack, n_steps, shape, context)
        except error:
            mods.released()
            var text = String(error)
            if not text.startswith("ValueError:") and text.find(_TOKEN_REFUSAL) >= 0:
                raise Error("ValueError: Byte-LM IDs must be in [0, " + String(shape.vocab()) + ")")
            raise error^
        _ = lease  # the lease keeps the state alive through the operation
    except error:
        mods.released()
        _ = py_call(lock.release)
        raise error^
    _ = py_call(lock.release)
    return result



def _logits_input(mut mods: NativeModules, source: PythonObject, lengths: PythonObject, shape: _Shape,
                  context: PythonObject) raises -> Tuple[PythonObject, Int, Int, List[Int]]:
    classical_admit_value(source, context)
    classical_admit_value(lengths, context)
    var value = classical_input(source, context, "ids")
    var dims = value.shape
    if len(dims) != 2:
        raise Error("ValueError: ids must be [batch, 1.." + String(shape.length()) + "]")
    var batch = Int(py=dims[0])
    var length = Int(py=dims[1])
    if batch < 1 or batch > 1024 or length < 1 or length > shape.length() or batch * length > 268435456 // shape.vocab():
        raise Error("ValueError: Byte-LM logits input shape exceeds its admitted bounds")
    var item = Int(py=value.itemsize)
    var kind = classical_cast_kind(value, item, "ids", context)
    var strides = value.strides
    var source_addr = Int(py=classical_address(value))
    var tokens = tsa_array("i", "<i4", Python.tuple(batch, length), batch * length, 4, context)
    var token_addr = Int(py=classical_address(tokens))
    _ = admitted_ids_i32(source_addr, Int(py=strides[0]), Int(py=strides[1]), batch, length,
        item, kind == CAST_KIND_SIGNED, False, _I32P(unsafe_from_address=token_addr))
    _ = value^
    var sizes = List[Int](length=batch, fill=length)
    if lengths is not None:
        sizes = _neural_lengths(mods, lengths, batch, length, context)
        var padded = tsa_array("i", "<i4", Python.tuple(batch, length), batch * length, 4, context)
        neural_copy_row_prefixes(Int(py=classical_address(padded)), token_addr, sizes, batch, length, 1)
        _ = tokens^
        tokens = padded
    return (tokens, batch, length, sizes^)


comptime _POOLED_LAYOUT_SLOT = "byte_lm_pooled_layout_v1"


def _admit_pooled_logits_layout(mut mods: NativeModules, context: PythonObject) raises:
    # Only the exact connected pooled class is admitted. Read type dictionaries
    # through pinned native descriptors; do not trigger class/Python hooks while
    # checking whether a caller replaced one. This precedes slot writes too.
    var cls = context["byte_lm_pooled_class"]
    if py_type(cls) is not _b(mods).type:
        raise Error("TypeError: Byte-LM pooled logits require the connected native storage class")
    # The check is class-level: keep its verdict per class epoch (type
    # watcher; any change to the class or an ancestor re-runs it).
    if class_record(mods, _POOLED_LAYOUT_SLOT, cls) is not None:
        return
    var mro = py_call(context["byte_lm_type_mro_getter"], cls)
    var expected = context["byte_lm_pooled_mro"]
    if len(mro) != len(expected):
        raise Error("TypeError: Byte-LM pooled logits ancestry changed")
    var layout = context["byte_lm_pooled_layout"]
    for i in range(len(expected)):  # driver-loop: over fixed pooled Array ancestry metadata
        if mro[i] is not expected[i]:
            raise Error("TypeError: Byte-LM pooled logits ancestry changed")
        var namespace = py_call(context["byte_lm_type_dict_getter"], layout[i][0])
        var hooks = layout[i][1]
        for j in range(len(hooks)):  # driver-loop: over fixed connected native storage descriptors
            if py_get(namespace, hooks[j][0]) is not hooks[j][1]:
                raise Error("TypeError: Byte-LM pooled logits storage hook changed")
    class_admit(mods, _POOLED_LAYOUT_SLOT, cls, PythonObject(True))


def _logits_output_address(mut mods: NativeModules, output: PythonObject, context: PythonObject) raises -> Int:
    # Both outputs are internally allocated, exact Array storage. The pooled
    # subclass was admitted before allocation above; no arbitrary subclasses or
    # array-interface/input protocols are accepted as output storage here.
    var cls = py_type(output)
    if cls is not context["array_class"] and cls is not context["byte_lm_pooled_class"]:
        raise Error("TypeError: Byte-LM logits output requires exact native Array storage")
    var address = py_call(context["byte_lm_addr_getter"], output, cls)
    if py_type(address) is not _b(mods).int or Int(py=address) == 0:
        raise Error("TypeError: Byte-LM logits output has an invalid native address slot")
    return Int(py=address)


def _pooled_logits(mut mods: NativeModules, fields: PythonObject, batch: Int, length: Int, vocab: Int,
                   context: PythonObject) raises -> PythonObject:
    _admit_pooled_logits_layout(mods, context)
    var pool = py_get(fields, "_logits_pool")
    if pool is None:
        pool = native_storage_allocate(context["byte_lm_logits_pool"])
        py_setitem(py_fields(pool), "keep", 2)
        py_setitem(py_fields(pool), "free", Python.list())
        py_setitem(fields, "_logits_pool", pool)
    if py_type(pool) is not context["byte_lm_logits_pool"]:
        raise Error("TypeError: Byte-LM logits requires its native output pool")
    var free = py_fields(pool)["free"]
    if py_type(free) is not _b(mods).list:
        raise Error("TypeError: Byte-LM logits pool requires a native list")
    var count = batch * length * vocab
    var store_type = _pooled_store_type(mods, context)
    # The pool counts an entry in float32 items: the B arm's array.array
    # store, or the default arm's "f" view of its unfilled bytearray store
    # (lane fix-wave2-leftovers: the pool keeps the view, so a reused store
    # builds no memoryview).
    var store_len = count
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var pooled = Optional[PythonObject]()
    for i in range(len(free)):  # driver-loop: over the bounded reusable host-buffer pool
        if py_type(free[i]) is not store_type:
            raise Error("TypeError: Byte-LM logits pool contains a foreign buffer")
        if len(free[i]) == store_len:
            pooled = py_call(free.pop, i)
            break
    # The pooled class was admitted above (_admit_pooled_logits_layout: exact
    # MRO and every Array slot descriptor pinned at import), so the slots are
    # written directly; the shared Array admission refuses any subclass.
    var out: PythonObject
    # TOMBSTONE: MOJOLEARN_BYTE_LM_LOGITS_ZERO_FILL (opt-in arm, no A/B recorded) deleted 2026-11-15 by lane/expdel2-02-base; code recoverable at 337c096a5b.
    # Restore: git apply experiments/removed/MOJOLEARN_BYTE_LM_LOGITS_ZERO_FILL.patch; record in docs/TOMBSTONES.md.
    # RESTORED 2026-11-15 by lane/expdel2-restore-owed: owed B arm (owed-pre0867.json), under four weeks old; the deletion was premature.
    comptime if BYTE_LM_LOGITS_ZERO_FILL:
        var store: PythonObject
        if pooled:
            store = pooled.value()
        else:
            store = py_call(context["array_type"], "f", py_call(context["native_bytes_type"], count * 4))
        out = classical_wrap_admitted_layout(context["byte_lm_pooled_class"], store,
            NativeArrayLayout([batch, length, vocab], 4, "C"), "<f4")
    else:
        var store: PythonObject
        var view: PythonObject
        if pooled:
            # Lane fix-wave2-leftovers (audit door:1451 leftover): a pooled
            # entry is the "f" view the store was first wrapped with (one
            # C getter for its exporter), not two new memoryviews per call.
            view = pooled.value()
            store = py_call(context["byte_lm_view_obj"], view)
            if py_type(store) is not context["byte_lm_store_type"]:
                raise Error("TypeError: Byte-LM logits pool contains a foreign buffer")
        else:
            store = native_scratch_bytes(count * 4)
            # TOMBSTONE: MOJOLEARN_BYTE_LM_FAST_LOGITS_HUGEPAGE (LOSS, A/B round 2 NVIDIA L40S: lm-forward 21.96 -> 25.73 ms, 1.17) deleted 2026-11-19 by lane/fx-verdicts4; code recoverable at 8f3f639b13.
            # Restore: git apply experiments/removed/MOJOLEARN_BYTE_LM_FAST_LOGITS_HUGEPAGE.patch; record in docs/TOMBSTONES.md.
            # The store's memoryview is one C-API call (PyMemoryView_FromObject
            # through the CPython table, as `classical_wrap_admitted_layout`
            # makes `_mv`), its "f" cast the `memoryview.cast` descriptor with
            # the "f" format object, both pinned once at connect: built once
            # per store, then pooled with it (hot-path audit 2026-11-13, BLOCK
            # door:1451).
            view = py_call(context["byte_lm_view_cast"], native_buffer_view(store), context["byte_lm_view_format"])
        out = _wrap_pooled_unfilled(context, context["byte_lm_pooled_class"], store, view, batch, length, vocab)
    py_setitem(py_fields(out), "_pool", pool)
    return out


#: lane fix-lm-forward-door (2026-11-13; hist-neural REPORT row 10: lm-forward
#: NVIDIA 44.9 -> 60.3 ms, AMD 58.5 -> 66.9 ms, 0.8.55 -> 0.8.66, same kernel).
#: The door migration (6cc681830d) made a fresh logits store as
#: `array.array("f", bytes(count * 4))`: a zero-filled bytes object, then a
#: COPY of all of it into the array (at the board's L2048 x V8192 shape a
#: 64 MiB zero read, a 64 MiB first-touch write and the bytes' free), before
#: the native download overwrote every byte again. The board's lm-forward
#: holds the previous result while it calls (`self.out = trainer.logits(ids)`),
#: so the scored call always misses the pool and pays that. 0.8.55 paid one
#: zero-fill pass (`array.array("f", [0]) * n`). The default now allocates the
#: store UNFILLED (`native_scratch_bytes`, an uninitialized bytearray, no fill
#: pass; `_mv` its memoryview cast to "f", as `generation_array_unfilled`):
#: the one device-to-host download is the first and only write, and it writes
#: all batch * length * vocab floats before the call returns (the native
#: owner checks `written == expected`; a failure raises and the Array is never
#: returned). No output byte moves: same kernel, same fold, same download.
#: `-D MOJOLEARN_BYTE_LM_LOGITS_ZERO_FILL` keeps the zero-filled array.array
#: store (B arm of the owed A/B, bench/results/paper_compare/owed-pre0867.json
#: row wave2-lm-forward-door). Only the resident pooled output changes: the
#: ragged result (padding must read +0.0) and the stateless output keep
#: `tsa_array`'s zeros.
comptime BYTE_LM_LOGITS_ZERO_FILL = is_defined["MOJOLEARN_BYTE_LM_LOGITS_ZERO_FILL"]()

# TOMBSTONE: MOJOLEARN_BYTE_LM_FAST_LOGITS_HUGEPAGE (LOSS: a fresh resident logits store advised for transparent huge
# pages before the download first-touches it; A/B round 2, NVIDIA L40S, wheel 0c68d6cd27, job
# nvc-ab-20261119T154232Z-0642a5, base and arm in one job, digests and quality equal: lm-forward 21.96 -> 25.73 ms
# (1.17; the row's three bases in the job 21.35-21.96), lm-train-step 1.00, transformer-forward 0.99; fresh-store
# driver shapes 32 MiB 13.73 -> 14.31, 64 MiB 28.23 -> 29.03, 2.3 MiB 4.77 -> 4.44) deleted 2026-11-19 by
# lane/fx-verdicts4; code recoverable at 8f3f639b13.
# Restore: git apply experiments/removed/MOJOLEARN_BYTE_LM_FAST_LOGITS_HUGEPAGE.patch; record in docs/TOMBSTONES.md.


def _pooled_store_type(mut mods: NativeModules, context: PythonObject) raises -> PythonObject:
    """The exact type of a logits pool entry under the compiled arm: the
    array.array store (B arm), or the "f" memoryview of the unfilled
    bytearray store (default; lane fix-wave2-leftovers)."""
    comptime if BYTE_LM_LOGITS_ZERO_FILL:
        return context["array_type"]
    else:
        # pinned once at connect (hot-path audit NOTE door:1437)
        return context["byte_lm_view_type"]


def _wrap_pooled_unfilled(context: PythonObject, cls: PythonObject, store: PythonObject, view: PythonObject,
                          batch: Int, length: Int, vocab: Int) raises -> PythonObject:
    """`classical_wrap_admitted_layout` over an uninitialized bytearray store
    (`native_scratch_bytes`): the same slots, with `_mv` the caller's "f"
    view of the store (the export pins the bytearray against resizing for
    as long as the view lives, the pool's hold included) and `_addr` its
    data address. `cls` is the pooled class the caller already admitted."""
    var layout = NativeArrayLayout([batch, length, vocab], 4, "C")
    if layout.nbytes != len(store) or layout.count != len(view):
        raise Error("RuntimeError: a Byte-LM logits store does not match its shape")
    var array = native_storage_allocate(cls)
    py_setattr(array, "_store", store)
    py_setattr(array, "_base", PythonObject(None))
    py_setattr(array, "_pin", PythonObject(None))
    py_setattr(array, "_mv", view)
    py_setattr(array, "_addr", PythonObject(native_scratch_address(store)))
    py_setattr(array, "_readonly", PythonObject(False))
    py_setattr(array, "shape", native_shape_pack(layout.shape))
    py_setattr(array, "dtype", PythonObject("<f4"))
    py_setattr(array, "ndim", PythonObject(len(layout.shape)))
    py_setattr(array, "size", PythonObject(layout.count))
    py_setattr(array, "nbytes", PythonObject(layout.nbytes))
    py_setattr(array, "itemsize", PythonObject(layout.itemsize))
    py_setattr(array, "strides", native_shape_pack(layout.strides))
    py_setattr(array, "order", PythonObject(layout.order))
    return array


def byte_lm_pooled_finalize(mut mods: NativeModules, owner: PythonObject, context: PythonObject) raises:
    # Installed as a compiled instance-method descriptor. A decref during a
    # native operation never invokes _PooledLogits.__del__ or pool.give Python.
    if py_type(owner) is not context["byte_lm_pooled_class"]:
        return
    var pool = py_get(py_fields(owner), "_pool")
    if pool is None or py_type(pool) is not context["byte_lm_logits_pool"]:
        return
    # The pool entry: the store (B arm) or the store's "f" view, the `_mv`
    # slot (default arm, lane fix-wave2-leftovers), through the slot getter
    # pinned at connect.
    var store: PythonObject
    comptime if BYTE_LM_LOGITS_ZERO_FILL:
        store = py_call(context["byte_lm_store_getter"], owner, context["byte_lm_pooled_class"])
    else:
        store = py_call(context["byte_lm_view_getter"], owner, context["byte_lm_pooled_class"])
    if py_type(store) is not _pooled_store_type(mods, context):
        return
    var pool_fields = py_fields(pool)
    var free = py_get(pool_fields, "free")
    var keep = py_get(pool_fields, "keep")
    if py_type(free) is not _b(mods).list or py_type(keep) is not _b(mods).int or Int(py=keep) < 1:
        return
    for i in range(len(free)):  # small-loop(the bounded reusable host-buffer pool)
        if free[i] is store:
            return
    if len(free) >= Int(py=keep):
        _ = py_call(free.pop, 0)
    _ = py_call(free.append, store)


def byte_lm_pooled_finalize_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    var mods = NativeModules()
    try:
        var args = PythonObject(from_borrowed=args_ptr)
        byte_lm_pooled_finalize(mods, args[1], args[0])
    except:
        # Finalization is best effort, as the previous Python __del__; a
        # malformed/expired pool must not lose the Array's owned storage.
        pass
    return PythonObject(None).steal_data()


def byte_lm_logits(mut mods: NativeModules, trainer: PythonObject, args: PythonObject, context: PythonObject,
                   next_bytes: Bool = False) raises -> PythonObject:
    classical_connect_numpy(context)
    var fields = py_fields(trainer)
    var lock = fields["_lock"]
    _ = py_call(lock.acquire)
    mods.released()
    var result = PythonObject(None)
    try:
        var lease = _lease(fields)
        var s = byte_trainer_state_ptr(lease)
        var shape = _state_shape(s[].data)
        var admitted = _logits_input(mods, args[0], args[1], shape, context)
        var tokens = admitted[0]
        var batch = admitted[1]
        var length = admitted[2]
        _ = _binding(mods, trainer, fields, s[], shape, context)
        var resident = byte_trainer_is_resident(s[])
        # The greedy pick runs on the device for a ragged batch too (plan row
        # N091): operation 6 takes the row lengths as a seventh descriptor.
        var direct_next = next_bytes and resident
        # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
        var row_lengths = Optional[PythonObject]()
        var lengths_address = 0
        if direct_next and args[1] is not None:
            var lengths = _i32(batch, context)
            row_lengths = lengths
            lengths_address = Int(py=classical_address(lengths))
            var lengths_out = _I32P(unsafe_from_address=lengths_address)
            for row in range(batch):  # small-loop(batch: one admitted length per row)
                lengths_out[row] = Int32(admitted[3][row])
        var expected = batch if direct_next else batch * length * shape.vocab()
        var out = _i32(batch, context) if direct_next else (_pooled_logits(mods, fields, batch, length, shape.vocab(), context) if resident else tsa_array("f", "<f4", Python.tuple(batch, length, shape.vocab()), expected, 4, context))
        var output_address = _logits_output_address(mods, out, context)
        var token_address = Int(py=classical_address(tokens))
        var written = 0
        # RUN: the state's buffers and plan words come from the leased native
        # state; the session lifecycle (admit, adopt, failure verdict) is typed.
        if resident:
            with GILReleased(Python()):
                written = byte_trainer_logits(s[], token_address, output_address, lengths_address, direct_next,
                                              batch, length)
            mods.released()
        else:
            # Operation 7 (stateless logits) takes no session: the image's
            # table entry runs it with plain words, no provider object.
            var entry = byte_lm_trainer_ref(s[].key())
            with GILReleased(Python()):
                written = byte_trainer_logits_stateless(s[], entry, token_address, output_address, batch, length)
            mods.released()
        _ = tokens
        _ = row_lengths  # keep the Python object alive through the native call above
        if written != expected:
            raise Error("RuntimeError: byte LM returned an unexpected number of logits")
        if next_bytes:
            var indices = List[Int64](length=batch, fill=0)
            if direct_next:
                var source = _I32P(unsafe_from_address=output_address)
                for row in range(batch):  # cpu-route: consume the native greedy output
                    indices[row] = Int64(source[row])
            else:
                var source = _F32P(unsafe_from_address=output_address)
                for row in range(batch):  # cpu-route: same strict first-max owner on each last real row
                    argmax_rows_f32(source + (row * length + admitted[3][row] - 1) * shape.vocab(), 1, shape.vocab(), MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(indices.unsafe_ptr()) + row * 8))
            result = byte_int_list_from_i64(Int(indices.unsafe_ptr()), batch)
            _ = indices^
        elif args[1] is not None:
            result = tsa_array("f", "<f4", Python.tuple(batch, length, shape.vocab()), expected, 4, context)
            neural_copy_row_prefixes(Int(py=classical_address(result)), output_address, admitted[3], batch, length, shape.vocab())
        else:
            result = out
        _ = out
        _ = lease  # the lease keeps the state alive through the operation
    except problem:
        mods.released()
        _ = py_call(lock.release)
        raise problem
    _ = py_call(lock.release)
    return result




def _registry(mut mods: NativeModules, shape: _Shape) raises -> PythonObject:
    var names = shape.names()
    var shapes = shape.shapes()
    var result = Python.list()
    var offset = 0
    for i in range(len(names)):  # driver-loop: configured parameter tensors
        var dims = Python.list()
        var count = 1
        for width in shapes[i]:  # driver-loop: over parameter rank
            _ = py_call(dims.append, width)
            count *= width
        var record = _b(mods).dict()
        py_setitem(record, "name", names[i])
        py_setitem(record, "shape", py_call(_b(mods).tuple, dims))
        py_setitem(record, "offset", offset)
        py_setitem(record, "size", count)
        _ = py_call(result.append, record)
        offset += count
    return result


def _config_dict(mut mods: NativeModules, data: ByteLmStateData) raises -> PythonObject:
    """PACK: the optimizer configuration as its public dictionary."""
    var config = _b(mods).dict()
    var keys = [String("lr"), String("beta1"), String("beta2"), String("eps"), String("weight_decay")]
    for i in range(5):  # small-loop(the five optimizer configuration scalars)
        py_setitem(config, keys[i], data.scalars[i])
    py_setitem(config, "kind", data.kind)
    py_setitem(config, "momentum", data.scalars[5])
    py_setitem(config, "dampening", data.scalars[6])
    py_setitem(config, "nesterov", data.nesterov != 0)
    py_setitem(config, "max_norm", data.scalars[7])
    return config


def _state_dict(mut mods: NativeModules, data: ByteLmStateData, shape: _Shape, parameters: PythonObject,
                moment: PythonObject, variance: PythonObject, flags: PythonObject) raises -> PythonObject:
    """PACK: the public state dictionary (`export_state()`), built from the
    typed state and the four arrays the caller exported or copied."""
    var b = _b(mods)
    var result = b.dict()
    py_setitem(result, "schema", "mojolearn.small-byte-lm-state.v1")
    py_setitem(result, "profile", PythonObject(data.profile))
    py_setitem(result, "numeric_mode", "identical" if data.mode == 1 else "fast")
    var registry = _registry(mods, shape)
    var names = Python.list()
    var shapes = Python.list()
    var offsets = Python.list(0)
    for i in range(len(registry)):  # driver-loop: detached parameter metadata
        var item = registry[i]
        _ = py_call(names.append, item["name"])
        _ = py_call(shapes.append, py_call(b.list, item["shape"]))
        _ = py_call(offsets.append, Int(py=item["offset"]) + Int(py=item["size"]))
    py_setitem(result, "parameter_names", names)
    py_setitem(result, "parameter_shapes", shapes)
    py_setitem(result, "parameter_offsets", offsets)
    py_setitem(result, "parameters", parameters)
    py_setitem(result, "m", moment)
    py_setitem(result, "v", variance)
    py_setitem(result, "flags", flags)
    py_setitem(result, "completed_steps", data.completed)
    py_setitem(result, "next_batch_index", data.completed)
    py_setitem(result, "config", _config_dict(mods, data))
    py_setitem(result, "data_schedule", native_json_parse(PythonObject(data.schedule)))
    mods.released()
    if data.has_model_shape:
        var dims = b.dict()
        var dim_names = [String("batch"), String("length"), String("d_model"), String("n_heads"), String("n_kv"),
            String("head_dim"), String("intermediate"), String("n_layers"), String("vocab_size")]
        for i in range(9):  # small-loop(the nine normalized model dimensions)
            py_setitem(dims, dim_names[i], shape.d[i])
        py_setitem(dims, "chunked_lm_head_v2", shape.chunked)
        py_setitem(result, "model_shape", dims)
    return result


def _host_copy(source: Int, n: Int, name: String, context: PythonObject) raises -> PythonObject:
    """PACK: an owned Array copy of one host state buffer (with the finite
    check the export has always applied to a closed trainer's state)."""
    var out = _f32_unfilled(n, context)
    training_copy_f32(source, Int(py=classical_address(out)), [n], [4], name)
    return out


def _export_snapshot(mut mods: NativeModules, mut s: ByteLmTrainerState, shape: _Shape,
                     context: PythonObject) raises -> PythonObject:
    """`export_state()`: a detached public state. An open session downloads
    its state into the result arrays (one export, the device keeps its
    state); a closed trainer's native buffers are copied once each."""
    byte_trainer_require_live(s)
    var n = shape.n_total()
    if s.session_open:
        # lane fix-output-zero-fill (2026-11-14): parameters, m and v are
        # unfilled. byte_session_export_state (training/byte_lm_session_ops.mojo)
        # copy_f32-s all n floats of each from its staged download before
        # returning, and a failed export raises before this dict escapes. The
        # flags stay zero-filled (n_tensors words; `_write_flags` writes the
        # session's own flag count). No output bit changes.
        var out_p = _f32_unfilled(n, context)
        var out_m = _f32_unfilled(n, context)
        var out_v = _f32_unfilled(n, context)
        var out_flags = _i32(shape.n_tensors(), context)
        var ap = Int(py=classical_address(out_p))
        var am = Int(py=classical_address(out_m))
        var av = Int(py=classical_address(out_v))
        var af = Int(py=classical_address(out_flags))
        mods.released()
        with GILReleased(Python()):
            byte_trainer_export(s, ap, am, av, af, False)
        mods.released()
        return _state_dict(mods, s.data, shape, out_p, out_m, out_v, out_flags)
    var parameters = _host_copy(Int(s.data.parameters.unsafe_ptr()), n, "parameters", context)
    var moment = _host_copy(Int(s.data.m.unsafe_ptr()), n, "m", context)
    var variance = _host_copy(Int(s.data.v.unsafe_ptr()), n, "v", context)
    var flags = _copy_i32(byte_trainer_flags_address(s), shape.n_tensors(), context)
    return _state_dict(mods, s.data, shape, parameters, moment, variance, flags)


def _runtime_metadata(mut mods: NativeModules, trainer: PythonObject, fields: PythonObject, mut s: ByteLmTrainerState,
                      shape: _Shape, context: PythonObject) raises -> PythonObject:
    var binding = _binding(mods, trainer, fields, s, shape, context)
    var runtime = fields["_runtime"]
    if runtime is None:
        runtime = py_call(_b(mods).dict, fields["_admitted_binding_metadata"])
        var inventory = _b(mods).dict()
        var root = String(py=context["byte_lm_source_root"])
        var names = context["byte_lm_source_names"]
        for i in range(len(names)):  # driver-loop: over 6 direct source paths
            var path = root + "/" + String(py=names[i])
            if native_path_exists(PythonObject(path)):
                mods.released()
                py_setitem(inventory, names[i], sha256_file(path))
            mods.released()
        py_setitem(inventory, "loaded_python_wrapper", sha256_file(String(py=context["byte_lm_wrapper_file"])))
        var binding_file = binding.__file__
        py_setitem(runtime, "binding_file", binding_file)
        py_setitem(runtime, "binding_sha256", sha256_file(String(py=binding_file)))
        # Build read-backs through the image's glue entry; an image without
        # them (the CPU image) answers BG_ABSENT, as its missing entry did.
        var key = s.key()
        var gl = byte_lm_glue_ref(key)
        var attention = PythonObject(None)
        var arm_name = byte_lm_glue_read_text(gl, BG_ATTENTION_ARM, [0])
        if arm_name:
            attention = _b(mods).dict()
            py_setitem(attention, "arm", PythonObject(arm_name.value()))
            py_setitem(attention, "default", _glue_text_words(gl, BG_ATTENTION_ARM, [1]))
            py_setitem(attention, "trial_build", byte_lm_glue_read_int(gl, BG_ATTENTION_ARM, [2]) != 0)
            py_setitem(attention, "resolved_hd64", _glue_text_words(gl, BG_ATTENTION_ARM, [3]))
            var profile_words: List[Int] = [0]
            profile_words.extend(_shape_words(shape))
            var memory = byte_lm_glue_read_text(gl, BG_MEMORY_PROFILE, profile_words^)
            if memory:
                var bytes_words: List[Int] = [1]
                bytes_words.extend(_shape_words(shape))
                py_setitem(attention, "memory_profile", PythonObject(memory.value()))
                py_setitem(attention, "retained_exp_bytes_per_layer", byte_lm_glue_read_int(gl, BG_MEMORY_PROFILE, bytes_words^))
        py_setitem(runtime, "native_attention_arm", attention)
        var glue = PythonObject(None)
        var glue_name = byte_lm_glue_read_text(gl, BG_STEP_GLUE_ARM, [0])
        if glue_name:
            glue = _b(mods).dict()
            py_setitem(glue, "arm", PythonObject(glue_name.value()))
            py_setitem(glue, "trial_build", byte_lm_glue_read_int(gl, BG_STEP_GLUE_ARM, [1]) != 0)
        py_setitem(runtime, "native_step_glue_arm", glue)
        py_setitem(runtime, "source_sha256", inventory)
        py_setitem(runtime, "source_scope", "available direct source files; binding SHA identifies the compiled artifact")
        py_setitem(fields, "_runtime", runtime)
    var result = native_json_parse(PythonObject(_transport_json(runtime)))
    mods.released()
    py_setitem(result, "schema", "small-byte-lm.run-metadata.v1")
    py_setitem(result, "qualification", "authored/unqualified")
    py_setitem(result, "profile", PythonObject(s.data.profile))
    py_setitem(result, "device_state_lifetime", "resident" if byte_trainer_is_resident(s) else "call")
    py_setitem(result, "step_result", "lean" if s.lean else "full")
    py_setitem(result, "last_export_step", s.last_export_step)
    py_setitem(result, "config", _config_dict(mods, s.data))
    py_setitem(result, "data_schedule", native_json_parse(PythonObject(s.data.schedule)))
    mods.released()
    py_setitem(result, "completed_steps", s.data.completed)
    py_setitem(result, "next_batch_index", s.data.completed)
    return result


def _attention_stage_report(mut mods: NativeModules, mut s: ByteLmTrainerState, shape: _Shape) raises -> PythonObject:
    # RUN: the open session's stage words (operation 11, GIL held as before).
    var info = byte_trainer_stage_info(s)
    mods.released()
    if len(info) < 10:
        return PythonObject(None)
    var fwd = info[4]
    var aexp = info[5]
    var bwd = info[6]
    var report = _b(mods).dict()
    py_setitem(report, "forward_eager_cells", fwd)
    py_setitem(report, "forward_aexp_cells", aexp)
    py_setitem(report, "backward_eager_cells", bwd)
    py_setitem(report, "forward_eager_bytes", fwd * 4)
    py_setitem(report, "forward_aexp_bytes", aexp * 4)
    py_setitem(report, "backward_eager_bytes", bwd * 4)
    py_setitem(report, "eager_bytes", (fwd + bwd) * 4)
    py_setitem(report, "total_bytes", (fwd + aexp + bwd) * 4)
    py_setitem(report, "layers_grown_forward", info[7])
    py_setitem(report, "layers_grown_backward", info[8])
    py_setitem(report, "layers_full_aexp", info[9])
    var layers = shape.layers()
    py_setitem(report, "layers", layers)
    if len(info) >= 12:
        var forward = info[10]
        var backward = info[11]
        py_setitem(report, "stage_lists", Python.tuple(forward, backward))
        layers = min(layers, min(forward, backward))
    if len(info) >= 12 + 3 * layers:
        var names = [String("NOT_ATTEMPTED"), String("FUSED_RAN"), String("FUSED_REFUSED_REGIME"),
            String("FUSED_CORNER"), String("FUSED_SKIPPED_STICKY")]
        for offset in range(2):  # driver-loop: over forward and backward status fields
            var values = Python.list()
            var counts = List[Int](length=5, fill=0)
            for layer in range(layers):  # driver-loop: over admitted layer metadata
                var code = info[12 + 3 * layer + offset]
                _ = py_call(values.append, code)
                if code >= -1 and code <= 3:
                    counts[code + 1] += 1
            var mapped = _b(mods).dict()
            for i in range(5):  # driver-loop: over fixed status names
                py_setitem(mapped, names[i], counts[i])
            var key = String("forward_status" if offset == 0 else "backward_status")
            py_setitem(report, key, values)
            py_setitem(report, key + "_counts", mapped)
        var materialized = Python.list()
        for layer in range(layers):  # driver-loop: over admitted layer metadata
            _ = py_call(materialized.append, Bool(info[14 + 3 * layer]))
        py_setitem(report, "attn_materialized", materialized)
    if len(info) > 12 + 3 * layers:
        py_setitem(report, "exact_tail_guard", Bool(info[12 + 3 * layers]))
    if len(info) > 14 + 3 * layers:
        py_setitem(report, "release_eager", Bool(info[13 + 3 * layers]))
        py_setitem(report, "released_eager_bytes", info[14 + 3 * layers] * 4)
    if len(info) >= 16 + 4 * layers:
        py_setitem(report, "sticky_eager", Bool(info[15 + 3 * layers]))
        var preferred = Python.list()
        for layer in range(layers):  # driver-loop: over admitted layer metadata
            _ = py_call(preferred.append, Bool(info[16 + 3 * layers + layer]))
        py_setitem(report, "layers_prefer_eager", preferred)
    if len(info) >= 17 + 5 * layers:
        py_setitem(report, "repair_masked_tail", Bool(info[16 + 4 * layers]))
        var sites = Python.list()
        for layer in range(layers):  # driver-loop: over admitted layer metadata
            _ = py_call(sites.append, info[17 + 4 * layers + layer])
        py_setitem(report, "backward_repair_sites", sites)
    return report


def _shape_argument(mut mods: NativeModules, value: PythonObject, context: PythonObject) raises -> _Shape:
    """A `ByteLanguageModelConfig` or None (the regression fixture's shape)
    as the admitted native shape: `require_shape(value)` then
    `state_shape`, the dimension refusals included."""
    var state = _b(mods).dict()
    if value is not None:
        if py_type(value) is not context["byte_lm_config_class"]:
            raise Error("TypeError: shape must be a ByteLanguageModelConfig")
        var config = py_fields(value)
        classical_admit_value(config, context)
        py_setitem(state, "model_shape", config)
    return _shape_of(state)


def _session_info(mut mods: NativeModules, s: ByteLmTrainerState) raises -> PythonObject:
    """PACK: the session lifecycle words, for tools and tests (they replaced
    reads of the trainer's former `_state` / `_session_open` fields)."""
    var info = _b(mods).dict()
    py_setitem(info, "session", s.has_session())
    # The session's handle words (index, generation): a new session differs.
    var session_id = PythonObject(None)
    if s.has_session():
        var held = s.session.value().copy()
        session_id = Python.tuple(held[].index, Int(held[].generation))
    py_setitem(info, "session_id", session_id)
    py_setitem(info, "open", s.session_open)
    py_setitem(info, "lost", s.lost)
    py_setitem(info, "host_state", len(s.data.parameters) == s.data.n_total)
    py_setitem(info, "grad_step", s.grad_step)
    py_setitem(info, "last_export_step", s.last_export_step)
    py_setitem(info, "completed_steps", s.data.completed)
    return info


def byte_lm_consume(mut mods: NativeModules, trainer: PythonObject, operation: String, args: PythonObject,
                    context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    if operation == "parameter_registry":
        return _registry(mods, _shape_argument(mods, args[0], context))
    if operation == "binding":
        # `_byte_lm_impl._load(shape)` (tools and tests read the binding):
        # the process tier, the image's profile, mode and vendor admission and
        # the runtime-shape profile check, all through `_load_snapshot`; the
        # Python twin (`_mode`, the profile/vendor/entry probes) is gone
        # (lane py-lm-neural, 2026-11-14). No trainer, no lock.
        return _load(mods, _shape_argument(mods, args[0], context), context)
    var fields = py_fields(trainer)
    var lock = fields["_lock"]
    _ = py_call(lock.acquire)
    mods.released()
    var result = PythonObject(None)
    try:
        classical_admit_value(args, context)
        var lease = _lease(fields)
        var s = byte_trainer_state_ptr(lease)
        var shape = _state_shape(s[].data)
        if operation == "close":
            _suspend(mods, fields, s[], context)
        elif operation == "data_schedule":
            result = native_json_parse(PythonObject(s[].data.schedule))
            mods.released()
        elif operation == "step":
            result = PythonObject(s[].data.completed)
        elif operation == "attention_stage_report":
            result = _attention_stage_report(mods, s[], shape)
        elif operation == "run_metadata":
            result = _runtime_metadata(mods, trainer, fields, s[], shape, context)
        elif operation == "export_state":
            result = _export_snapshot(mods, s[], shape, context)
        elif operation == "state_metadata":
            # The state's metadata with no array (None in their places): no
            # device download and no copy.
            result = _state_dict(mods, s[].data, shape, PythonObject(None), PythonObject(None), PythonObject(None),
                                 PythonObject(None))
        elif operation == "session_info":
            result = _session_info(mods, s[])
        elif operation == "trainer_binding":
            # The process-tier binding for this trainer's shape (fault-hook tools).
            result = _load(mods, shape, context)
        elif operation == "parameters":
            if s[].session_open:
                result = _export_snapshot(mods, s[], shape, context)["parameters"]
            else:
                if s[].lost:
                    raise Error("RuntimeError: Byte-LM session lost; restore state before exporting parameters")
                result = _host_copy(Int(s[].data.parameters.unsafe_ptr()), shape.n_total(), "parameters", context)
        elif operation == "export_gradients":
            # RUN refuses first (lost, no completed step); then the outputs
            # are allocated and filled from the open session (operation 10)
            # or from the gradient a lean stateless step kept.
            byte_trainer_gradients_check(s[])
            var named = Bool(py=args[0])
            var grads = _gradient_outputs(mods, shape, named, context)
            var addresses = _gradient_addresses(mods, grads)
            var sizes = List[Int]()
            if named:
                sizes = _tensor_sizes(shape)
            var flat = Int(py=classical_address(grads[0]))
            mods.released()
            with GILReleased(Python()):
                byte_trainer_export_gradients(s[], flat, addresses, sizes)
            mods.released()
            result = _b(mods).dict()
            py_setitem(result, "flat_gradients", grads[0])
            if named:
                py_setitem(result, "gradients", grads[1])
        else:
            raise Error("RuntimeError: unknown Byte-LM consumed-output operation")
        _ = lease  # the lease keeps the state alive through the operation
    except problem:
        mods.released()
        _ = py_call(lock.release)
        raise problem
    _ = py_call(lock.release)
    return result



def byte_lm_trainer_c(py_self: PyObjectPtr, args_ptr: PyObjectPtr) abi("C") -> PyObjectPtr:
    var mods = NativeModules()
    var io_error = ByteCheckpointIOError()
    try:
        var call = PythonObject(from_borrowed=args_ptr)
        var operation = String(py=call[1])
        if operation.startswith("config_"):
            return byte_lm_config_operation(call[0], operation, call[2], call[3]).steal_data()
        if operation == "from_checkpoint_binary" or operation == "from_checkpoint" or operation == "from_checkpoint_bytes" or operation == "export_checkpoint_binary" or operation == "export_checkpoint":
            return byte_lm_checkpoint(mods, call[0], operation, call[2], call[3], io_error).steal_data()
        if operation == "load_state":
            return byte_lm_load_state(mods, call[0], call[2], call[3]).steal_data()
        if operation == "construct":
            return byte_lm_construct(mods, call[0], call[2], call[3]).steal_data()
        if operation == "train_steps":
            return byte_lm_train_steps(mods, call[0], call[2], call[3]).steal_data()
        if operation == "train_step" or operation == "evaluate":
            return byte_lm_train_step(mods, call[0], call[2], call[3], operation == "train_step").steal_data()
        if operation == "logits" or operation == "next_bytes":
            return byte_lm_logits(mods, call[0], call[2], call[3], operation == "next_bytes").steal_data()
        if operation == "attention_stage_report" or operation == "state_metadata" or operation == "session_info" or operation == "trainer_binding" or operation == "parameter_registry" or operation == "binding" or operation == "run_metadata" or operation == "export_state" or operation == "export_gradients" or operation == "close" or operation == "parameters" or operation == "step" or operation == "data_schedule":
            return byte_lm_consume(mods, call[0], operation, call[2], call[3]).steal_data()
        raise Error("RuntimeError: unknown byte LM trainer operation " + operation)
    except error:
        if io_error.pending:
            return io_error^.restore()
        if String(error) == PYTHON_ERROR_PENDING:
            return optimizer_python_error(error)
        var text = String(error)
        if text.startswith("TypeError:"):
            return raise_python_exception(error_without_class(error, "TypeError"), ExceptionType.TypeError)
        if text.startswith("RuntimeError:"):
            return raise_python_exception(error_without_class(error, "RuntimeError"), ExceptionType("PyExc_RuntimeError"))
        if text.startswith("ImportError:"):
            return raise_python_exception(error_without_class(error, "ImportError"), ExceptionType("PyExc_ImportError"))
        if text.startswith("NotImplementedError:"):
            return raise_python_exception(error_without_class(error, "NotImplementedError"), ExceptionType("PyExc_NotImplementedError"))
        return raise_python_exception(error_without_class(error, "ValueError"), ExceptionType.ValueError)
