# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python door of the public optimizers (`mojolearn.SGD`, `Adam`,
`AdamW`, `clip_grad_norm_`, `accumulate_grads`; lane optimizers-thin-glue,
2026-10-21). Each public method is ONE native call: the Python method passes
the optimizer object and the caller's arguments as they are, and this door
reads them, refuses by name what the Python glue used to refuse, calls ONE
typed kernel entry and writes the results back onto the optimizer.

What the door does is argument conversion only:

  * the caller's parameter and gradient tensors are read through the
    buffer protocol (`PyObject_GetBuffer`, retained through native execution) or, for a `mojolearn.Array`,
    off its own storage fields; a strided read-only input is made
    C-contiguous by the parallel native optimizer input plan;
  * dtype, emptiness, layout, count and shape are refused with the
    messages the Python glue raised;
  * built-in schedules supply admitted native configuration words; Python
    learning-rate callbacks are refused before lookup or invocation;
  * the optimizer state's host mirrors (`exp_avg`, `exp_avg_sq`) are
    `mojolearn.Array`s assembled from native storage and slots.

The kernels are training/optimizer_public.mojo (the GPU binding),
training/host/optimizer_public_host.mojo (a CPU-only install) and
training/optimizer_public_common.mojo (shared, no device code); `OptimizerNativeDriver`
names the entries a binding provides. Ownership (docs/MODULAR_ASKS.md):
every Python call and attribute or item assignment goes through
bindings/py_balanced.mojo, so no caller object is retained."""
from std.memory import memcpy
from std.ffi import c_int
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.cpython_table import cpython_table
from std.python._cpython import ExternalFunction, GILReleased, PyObjectPtr
from bindings.admitted_input import admitted_storage_native
from training.optimizer_config import optimizer_keyword_refusal
from training.optimizer_owner_control import OptimizerOwnerControl
from training.optimizer_native_api import OptimizerNativeDriver
from bindings.optimizer_owner_boundary import OptimizerOwner, OptimizerOwnerLease, optimizer_owner_new, optimizer_owner_idle
from training.optimizer_state_plan import (
    OptimizerStateSnapshot, optimizer_state_span, optimizer_checkpoint_spans,
    optimizer_state_download_needed, optimizer_host_moment_plan, OptimizerRegistryPlan, optimizer_registry_plan,
)

from bindings.optimizer_public_words import optimizer_python_error, stash_python_error
from bindings.py_balanced import py_call, py_setitem, py_type, py_get, py_fields
from bindings.tensor_view import (
    TensorView, shape_text, _contiguous, base_format, _format_name, _typestr_name, _array_code,
)
from bindings.optimizer_admission import opt_guards, opt_owner, opt_value, opt_scalar, opt_schedule
from sequence.lr_table import LrSpec
from training.optimizer_schedule_plan import optimizer_decay_readback
from training.optimizer_public_common import (
    GRADS_DEVICE, GRADS_HOST, GRADS_RESIDENT, OptHyper, PublicStep, PublicStepOut, zero_tensors,
)
from core.shared_slot import SharedSlot
from bindings.optimizer_descriptor_boundary import optimizer_provider, optimizer_driver
from bindings.optimizer_scalar_boundary import optimizer_float, optimizer_truth, optimizer_integer
from bindings.optimizer_container_boundary import optimizer_sequence_items, optimizer_pack_list, OptimizerInputOwners, OptimizerBufferLease
from training.optimizer_input_plan import (
    check_views, _check_view, _addrs_of, optimizer_gradient_count, optimizer_gradient_shape, optimizer_accumulation_shapes, optimizer_contiguous_address,
)

comptime IP = MutPointer[Int, MutUntrackedOrigin]
comptime BP = MutPointer[UInt8, MutUntrackedOrigin]
comptime PYBUF_FULL_RO = 0x11C

#: `optimizer_step`'s operations
comptime OP_STEP = 0
comptime OP_STEP_DEVICE = 1
comptime OP_STEP_ACCUMULATED = 2

#: `optimizer_state`'s readings
comptime STATE_DICT = 0
comptime STATE_EXP_AVG = 1
comptime STATE_EXP_AVG_SQ = 2

comptime _SRC = " (python/mojolearn/_training_impl.py)"


# ---------------------------------------------------------------------------
# the kernels a binding provides


# ---------------------------------------------------------------------------
# the classes and tables the door reads, resolved once


comptime _OptimizerRepr = ExternalFunction[
    "PyObject_Repr", def(PyObjectPtr) thin abi("C") -> PyObjectPtr]
comptime _OptimizerDictClear = ExternalFunction[
    "PyDict_Clear", def(PyObjectPtr) thin abi("C") -> NoneType]
comptime _OptimizerDictUpdate = ExternalFunction[
    "PyDict_Update", def(PyObjectPtr, PyObjectPtr) thin abi("C") -> c_int]


struct OptDoor(Defaultable, Movable):
    var context: PythonObject
    var ready: Bool
    var scalar_repr: _OptimizerRepr.type
    var dict_clear: _OptimizerDictClear.type
    var dict_update: _OptimizerDictUpdate.type
    var array_cls: PythonObject
    var dev_cls: PythonObject

    def __init__(out self):
        self.context = PythonObject(None)
        self.ready = False
        # This owner's error formatter is resolved once with its existing
        # connection; it never consults the mutable builtins.repr binding.
        ref cpy = Python().cpython()
        self.scalar_repr = _OptimizerRepr.load(cpy.lib.borrow())
        # Publication and rollback operate only on admitted exact dictionaries.
        # Their pointers belong to this owner, not the process-wide API table.
        self.dict_clear = _OptimizerDictClear.load(cpy.lib.borrow())
        self.dict_update = _OptimizerDictUpdate.load(cpy.lib.borrow())
        self.array_cls = PythonObject(None)
        self.dev_cls = PythonObject(None)


comptime OPT_DOOR = SharedSlot[OptDoor, "MojolearnOptimizerPublicDoorV3"]


def door_ready() raises:
    door_ready(PythonObject(None))


def door_ready(guarded: PythonObject) raises:
    """Connect loaded native type/owner identities once per process. `guarded` is a
    context its caller has just run `opt_guards` on (the router): when it
    is the door's own context the guards are not walked a second time."""
    var d = OPT_DOOR.get_or_create_ptr()
    if d[].ready:
        if d[].context is not guarded:
            opt_guards(d[].context)
        return
    var training = native_loaded_module("mojolearn._training_impl")
    d[].context = py_fields(training)["_OPT_CONTEXT"]
    if d[].context is not guarded:
        opt_guards(d[].context)
    # Keep the import connection's guarded Array layout and native storage
    # type even when a module field is replaced before the first operation.
    d[].array_cls = d[].context["array_class"]
    d[].dev_cls = d[].context["device_class"]
    d[].ready = True


def getattr_or(obj: PythonObject, name: String, default: PythonObject) raises -> PythonObject:
    return py_get(py_fields(obj), PythonObject(name), default)


def as_float(value: PythonObject) raises -> Float64:
    return optimizer_float(value, OPT_DOOR.get_or_create_ptr()[].context)


def truth(value: PythonObject) raises -> Bool:
    return optimizer_truth(value, OPT_DOOR.get_or_create_ptr()[].context)


def py_repr(value: PythonObject) raises -> String:
    var door = OPT_DOOR.get_or_create_ptr()
    # Repr can invoke a type slot: admit exact native scalars (and their
    # recursively admitted containers) before entering it under the GIL.
    opt_scalar(value, door[].context)
    var pointer = door[].scalar_repr(value._obj_ptr)
    if not pointer:
        var failure = stash_python_error()
        raise failure^
    return String(py=PythonObject(from_owned=pointer))


def type_name(value: PythonObject) raises -> String:
    return String(py=py_call(OPT_DOOR.get_or_create_ptr()[].context["type_name"], py_type(value)))



# shape_text, TensorView, _contiguous, base_format, _format_name, _typestr_name and _array_code moved to bindings/tensor_view.mojo (lane a6-device-x-family-classical).


def new_array(dims: List[Int], dtype: String) raises -> PythonObject:
    """PACK an owned Array through the shared native layout/storage boundary."""
    return admitted_storage_native(dims, dtype, OPT_DOOR.get_or_create_ptr()[].context["input_context"], as_array=True)


def set_field(opt: PythonObject, name: String, value: PythonObject) raises:
    py_setitem(py_fields(opt), PythonObject(name), value)


def owner_kind(opt: PythonObject) raises -> Int:
    var classes = OPT_DOOR.get_or_create_ptr()[].context["owners"]
    for i in range(len(classes)):  # driver-loop: over three native optimizer kinds
        if py_type(opt) is classes[i]:
            return i
    raise Error("TypeError: optimizer requires an exact native optimizer owner")


# ---------------------------------------------------------------------------
# one tensor, read through the buffer protocol




def dtype_name(obj: PythonObject, fmt: String, itemsize: Int) raises -> String:
    # Buffer format is native metadata; querying arbitrary dtype properties is
    # not needed and could call a Python descriptor.
    return _format_name(base_format(fmt), itemsize, fmt)


def _get_buffer(obj: PythonObject, mut view: List[Int]) raises -> Bool:
    """`PyObject_GetBuffer(obj, view, PyBUF_FULL_RO)`; False (indicator
    cleared) when `obj` does not export a buffer."""
    # Symbols from the process-wide table (bindings/cpython_table.mojo),
    # resolved once, not a dlsym per tensor; the clear runs only on refusal.
    var p = view.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var rc = cpython_table()[].PyObject_GetBuffer(obj._obj_ptr, p, Int32(PYBUF_FULL_RO))
    if rc != 0:
        ref cpy = Python().cpython()
        cpython_table()[].PyErr_Clear()
        return False
    return True


def _array_view(obj: PythonObject, context: PythonObject, mut v: TensorView, mut pins: OptimizerInputOwners) raises:
    """`opt_value`'s Array admission and `tensor_view`'s Array facts from one
    read of each retained native slot descriptor (same errors, same values)."""
    var getters = context["array_getters"]
    var shape = py_call(getters["shape"], obj)
    var dtype = py_call(getters["dtype"], obj)
    var addr = py_call(getters["_addr"], obj)
    var readonly = py_call(getters["_readonly"], obj)
    var nbytes = py_call(getters["nbytes"], obj)
    var itemsize = py_call(getters["itemsize"], obj)
    var strides = py_call(getters["strides"], obj)
    opt_scalar(shape, context, 1)
    opt_scalar(dtype, context, 1)
    opt_scalar(py_call(getters["order"], obj), context, 1)
    opt_scalar(addr, context, 1)
    opt_scalar(readonly, context, 1)
    opt_scalar(py_call(getters["ndim"], obj), context, 1)
    opt_scalar(py_call(getters["size"], obj), context, 1)
    opt_scalar(nbytes, context, 1)
    opt_scalar(itemsize, context, 1)
    opt_scalar(strides, context, 1)
    var mv = py_call(getters["_mv"], obj)
    if py_type(mv) is not context["container_types"][5]:
        raise Error("TypeError: optimizer Array requires native memoryview storage")
    opt_value(mv, context, 1)
    pins.append(mv)
    v.is_array = True
    v.addr = Int(py=addr)
    v.nbytes = Int(py=nbytes)
    v.itemsize = Int(py=itemsize)
    v.readonly = optimizer_truth(readonly, context)
    var t = String(py=dtype)
    v.fmt = _array_code(t)
    for i in range(len(shape)):  # driver-loop: over shape words
        v.shape.append(Int(py=shape[i]))
    for i in range(len(strides)):  # driver-loop: over stride words
        v.strides.append(Int(py=strides[i]))
    v.has_strides = True
    v.c_contig = _contiguous(v.shape, v.strides, True, v.itemsize)
    v.dtype = dtype_name(obj, v.fmt, v.itemsize)


def tensor_view(obj: PythonObject, mut v: TensorView) raises -> Bool:
    var pins = OptimizerInputOwners()
    return tensor_view(obj, v, pins)


def tensor_view(obj: PythonObject, mut v: TensorView, mut pins: OptimizerInputOwners) raises -> Bool:
    """Fill `v` with `obj`'s buffer facts; False when `obj` is not a buffer
    (`_training_impl._is_buffer`). The admitted export stays in `pins`
    until the boundary has completed native execution."""
    var d = OPT_DOOR.get_or_create_ptr()
    v = TensorView()
    pins.append(obj)
    if py_type(obj) is d[].array_cls:
        # An exact Array: admit its native slots (what opt_value's Array arm
        # does) and read the facts off the same slot values, in one pass.
        # Before, opt_value read all ten slots and this function then read
        # seven of them again by attribute, plus a builtins.isinstance call.
        _array_view(obj, d[].context, v, pins)
        return True
    opt_value(obj, d[].context)
    if py_type(obj) is d[].dev_cls:
        return False
    var view = List[Int](length=10, fill=0)
    if not _get_buffer(obj, view):
        _ = view^
        return False
    var lease = OptimizerBufferLease(view^, cpython_table()[].PyBuffer_Release)
    var p = lease.words.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var words32 = p.bitcast[Int32]()
    var indirect = False
    v.addr = p[0]
    v.nbytes = p[2]
    v.itemsize = p[3]
    v.readonly = words32[8] != 0
    var ndim = Int(words32[9])
    if p[5] != 0:
        var fp = BP(unsafe_from_address=p[5])
        var text = String("")
        for k in range(64):  # driver-loop: over format characters of one buffer
            if fp[k] == 0:
                break
            text += chr(Int(fp[k]))
        v.fmt = text
    if ndim > 0:
        if p[6] != 0:
            var sp = IP(unsafe_from_address=p[6])
            for k in range(ndim):  # driver-loop: over shape words of one buffer
                v.shape.append(sp[k])
        else:
            v.shape.append(v.nbytes // max(v.itemsize, 1))
        if p[7] != 0:
            var tp = IP(unsafe_from_address=p[7])
            for k in range(ndim):  # driver-loop: over stride words of one buffer
                v.strides.append(tp[k])
            v.has_strides = True
        if p[8] != 0:
            var op = IP(unsafe_from_address=p[8])
            for k in range(ndim):  # driver-loop: over suboffset words of one buffer
                if op[k] >= 0:
                    indirect = True
    if indirect:
        # `Buf._fill` refuses an indirect buffer with a TypeError, which
        # `_is_buffer` reads as "not a buffer"
        return False
    v.c_contig = _contiguous(v.shape, v.strides, v.has_strides, v.itemsize)
    v.dtype = dtype_name(obj, v.fmt, v.itemsize)
    pins.buffers.append(lease^)
    return True



# ---------------------------------------------------------------------------
# the argument checks of the Python glue, message for message


def as_seq_views(x: PythonObject, name: String, where: String, mut views: List[TensorView], mut pins: OptimizerInputOwners) raises -> List[PythonObject]:
    """`as_seq`, keeping each element's view in `views` so `check_views`
    checks the same facts without reading every buffer a second time. Native
    owner vectors and export leases remain in UNPACK until RUN completes."""
    views = List[TensorView]()
    var v = TensorView()
    if tensor_view(x, v, pins):
        var one = List[PythonObject]()
        one.append(x)
        views.append(v^)
        return one^
    var items = List[PythonObject]()
    try:
        items = optimizer_sequence_items(x, OPT_DOOR.get_or_create_ptr()[].context)
    except:
        raise Error(
            "TypeError: mojolearn." + where + ": " + name + " must be a float32 array (a numpy array, an "
            + "array.array or a mojolearn.Array -- anything supporting the buffer protocol) or an iterable "
            + "of them, got " + type_name(x)
        )
    if len(items) == 0:
        raise Error(
            "ValueError: mojolearn." + where + ": " + name + " is empty; there is nothing to do and an "
            + "empty parameter registry is refused rather than answered (training/estimator.mojo)"
        )
    for i in range(len(items)):  # driver-loop: over parameter buffers
        var item = TensorView()
        if not tensor_view(items[i], item, pins):
            raise Error(
                "TypeError: mojolearn." + where + ": " + name + "[" + String(i) + "] is "
                + type_name(items[i]) + ", not an array (it does not support the buffer protocol)"
            )
        views.append(item^)
    return items^


def check_tensors(seq: PythonObject, name: String, where: String, inplace: Bool, mut pins: OptimizerInputOwners) raises -> List[TensorView]:
    """`_check_dtype`: float32 or refused by name, never cast; non-empty;
    with `inplace`, C-contiguous and writable. Returns every tensor's view."""
    var views = List[TensorView]()
    for i in range(len(seq)):  # driver-loop: over parameter buffers
        var v = TensorView()
        if not tensor_view(seq[i], v, pins):
            raise Error("TypeError: mojolearn: object does not support the buffer protocol")
        _check_view(v, i, name, where, inplace)
        views.append(v^)
    return views^


def write_addr(obj: PythonObject, name: String) raises -> Int:
    var pins = OptimizerInputOwners()
    return write_addr(obj, name, pins)


def write_addr(obj: PythonObject, name: String, mut pins: OptimizerInputOwners) raises -> Int:
    """`_buffer.addr`: the address of a buffer the kernel writes."""
    var v = TensorView()
    if not tensor_view(obj, v, pins):
        raise Error("TypeError: mojolearn: " + name + " does not support the buffer protocol")
    if v.readonly:
        raise Error("ValueError: mojolearn: " + name + " is read-only, refusing to write to it")
    return v.addr


def read_addr(obj: PythonObject, name: String) raises -> Int:
    var pins = OptimizerInputOwners()
    return read_addr(obj, name, pins)


def read_addr(obj: PythonObject, name: String, mut pins: OptimizerInputOwners) raises -> Int:
    """`_buffer.addr_ro`."""
    var v = TensorView()
    if not tensor_view(obj, v, pins):
        raise Error("TypeError: mojolearn: " + name + " does not support the buffer protocol")
    return v.addr


def _check_grads(params: PythonObject, gs: List[PythonObject], views: List[TensorView], owhere: String, where: String,
                 mut pviews: List[TensorView], mut have_pviews: Bool, mut pins: OptimizerInputOwners) raises:
    """Unpack each parameter view once; native plans decide count and shape."""
    optimizer_gradient_count(len(params), len(gs), owhere, where)
    if not have_pviews:
        pviews = List[TensorView]()
        for j in range(len(params)):  # driver-loop: admit parameter buffer owners
            var pv = TensorView()
            if not tensor_view(params[j], pv, pins):
                raise Error("TypeError: mojolearn: object does not support the buffer protocol")
            optimizer_gradient_shape(pv, views[j], j, owhere, where)
            pviews.append(pv^)
        have_pviews = True
        return
    for j in range(len(pviews)):  # driver-loop: native shapes of parameter buffers
        optimizer_gradient_shape(pviews[j], views[j], j, owhere, where)


def refuse_kwargs(kwargs: PythonObject, where: String) raises:
    var context = OPT_DOOR.get_or_create_ptr()[].context
    opt_value(kwargs, context)
    var keys = optimizer_sequence_items(kwargs, context)
    if len(keys) != 0:
        optimizer_keyword_refusal(String(keys[0]), where)


# ---------------------------------------------------------------------------
# the optimizer object's words


def _resident_word(value: PythonObject) raises -> Int:
    """`_RESIDENT_WORD.get(resident, 1)`: None -1, False 0, True 1."""
    if value is None:
        return -1
    if py_type(value) is OPT_DOOR.get_or_create_ptr()[].context["scalar_types"][1] or py_type(value) is OPT_DOOR.get_or_create_ptr()[].context["scalar_types"][2]:
        return 0 if Int(py=value) == 0 else 1
    return 1


def state_address(value: PythonObject, count: Int, name: String, mut pins: OptimizerInputOwners) raises -> Int:
    var view = TensorView()
    if not tensor_view(value, view, pins):
        raise Error("TypeError: optimizer " + name + " requires a native state buffer")
    return optimizer_state_span(view, count, name)


def _state_into(opt: PythonObject, control: OptimizerOwnerControl, mut req: PublicStep, mut pins: OptimizerInputOwners) raises:
    var fields = py_fields(opt)
    var first = fields["_m"]
    var second = fields["_v"]
    var state = OptimizerStateSnapshot(control.handle,
        _resident_word(fields["_resident_wanted"]), control.pristine,
        0 if first is None else state_address(first, req.n_total, "exp_avg", pins),
        0 if second is None else state_address(second, req.n_total, "exp_avg_sq", pins))
    state.apply(req)


def host_moments(opt: PythonObject) raises:
    """`_host_moments`: the host mirrors, allocated (zero filled) on first need."""
    # The instance dictionary is fetched once; set_field writes the same dict.
    var fields = py_fields(opt)
    var n = Int(py=fields["n_total"])
    var plan = optimizer_host_moment_plan(n, fields["_m"] is not None, fields["_v"] is not None)
    if plan[0]:
        set_field(opt, "_m", new_array([n], "<f4"))
    if plan[1]:
        set_field(opt, "_v", new_array([n], "<f4"))


def keep_slot(mut record: OptimizerOwner, handle: Int, provider: PythonObject,
    driver: OptimizerNativeDriver):
    record.attach(handle, driver, provider)


def _hyper(opt: PythonObject, kind: Int, maximize: Bool, max_norm: Float64) raises -> OptHyper:
    """`_config()` of SGD (kind 0) and Adam / AdamW, as the step's floats."""
    # The instance dictionary is fetched once; set_field writes the same dict.
    var fields = py_fields(opt)
    if kind == 0:
        return OptHyper(
            kind, 1 if truth(fields["nesterov"]) else 0, maximize, as_float(fields["lr"]), 0.9, 0.999, 1e-8,
            as_float(fields["weight_decay"]), as_float(fields["momentum"]), as_float(fields["dampening"]), max_norm,
        )
    var betas = fields["betas"]
    return OptHyper(
        kind, 0, maximize, as_float(fields["lr"]), as_float(betas[0]), as_float(betas[1]), as_float(fields["eps"]),
        as_float(fields["weight_decay"]), 0.0, 0.0, max_norm,
    )


def schedule_spec(schedule: PythonObject) raises -> LrSpec:
    opt_schedule(schedule, OPT_DOOR.get_or_create_ptr()[].context)
    return schedule_spec_admitted(schedule)


def schedule_spec_admitted(schedule: PythonObject) raises -> LrSpec:
    """`schedule_spec` for a schedule its caller has just admitted with
    `opt_schedule` (`opt_owner` admits the optimizer's `lr_schedule`)."""
    var context = OPT_DOOR.get_or_create_ptr()[].context
    var fields = py_fields(schedule)
    var kind = 0
    var classes = context["schedules"]
    for i in range(len(classes)):  # driver-loop: over three built-in schedule configurations
        if py_type(schedule) is classes[i]:
            kind = i
    if kind < 3:
        var total = fields["total_steps"]
        return LrSpec(kind, Int(py=fields["warmup_steps"]), -1 if total is None else Int(py=total), 0,
                      Float64(py=fields["peak_lr"]), Float64(py=fields["min_lr"]), 0.0, 0.0)
    if kind < 5:
        var base = Float64(py=fields["base_lr"])
        var gamma = Float64(py=fields["gamma"])
        var readback = optimizer_decay_readback(base, gamma)
        return readback.spec(Int(py=fields["_step"]) if kind == 3 else 1)
    return LrSpec(10, 1 if String(py=fields["anneal_strategy"]) == "linear" else 0,
                  1 if Bool(py=fields["three_phase"]) else 0, Int(py=fields["total_steps"]),
                  Float64(py=fields["max_lr"]), Float64(py=fields["div_factor"]),
                  Float64(py=fields["final_div_factor"]), Float64(py=fields["pct_start"]))


# ---------------------------------------------------------------------------
# the public methods


@fieldwise_init
struct OptimizerRegistryPrepared(Movable):
    # PACK owners for the one prepared registry; algorithm metadata is native.
    var params: PythonObject
    var offsets: PythonObject
    var initialized: PythonObject
    var first: PythonObject
    var second: PythonObject
    var plan: OptimizerRegistryPlan


def optimizer_prepare_registry(params: PythonObject, where: String) raises -> OptimizerRegistryPrepared:
    var pins = OptimizerInputOwners()
    var views = List[TensorView]()
    var owners = as_seq_views(params, "params", where, views, pins)
    var published = optimizer_pack_list(owners)
    check_views(views, "params", where, True)
    var nbytes = List[Int]()
    for k in range(len(views)):  # driver-loop: native parameter registry descriptors
        nbytes.append(views[k].nbytes)
    var offsets = new_array([len(views) + 1], "<i4")
    var plan = optimizer_registry_plan(nbytes, Int(py=offsets._addr))
    var first = PythonObject(None)
    var second = PythonObject(None)
    if plan.eager:
        first = new_array([plan.count], "<f4")
        second = new_array([plan.count], "<f4")
    var initialized = new_array([plan.parameters], "<i4")
    _ = pins
    return OptimizerRegistryPrepared(published, offsets, initialized, first, second, plan)


def optimizer_publish_registry(owner: PythonObject, registry: OptimizerRegistryPrepared) raises:
    var fields = py_fields(owner)
    py_setitem(fields, "params", registry.params)
    py_setitem(fields, "offsets", registry.offsets)
    py_setitem(fields, "n_total", PythonObject(registry.plan.count))
    py_setitem(fields, "_m", registry.first)
    py_setitem(fields, "_v", registry.second)
    py_setitem(fields, "buf_initialized", registry.initialized)


def door_init(opt: PythonObject, params: PythonObject) raises -> PythonObject:
    """Legacy registry export using the same native preparation and publication."""
    door_ready()
    opt_owner(opt, OPT_DOOR.get_or_create_ptr()[].context)
    var kind = owner_kind(opt)
    var name = String("SGD") if kind == 0 else (String("Adam") if kind == 1 else String("AdamW"))
    var registry = optimizer_prepare_registry(params, name)
    optimizer_owner_idle(opt)
    var owner_state = optimizer_owner_new(opt, owner_kind(opt), not registry.plan.eager)
    var construction = OptimizerOwnerLease(opt, owner_state)
    set_field(opt, "_native_optimizer_owner", owner_state)
    optimizer_publish_registry(opt, registry)
    _ = construction
    return PythonObject(registry.plan.count)


def door_step(
    module: PythonObject, provider: PythonObject, opt: PythonObject, grads: PythonObject, max_norm: PythonObject, op: Int,
    tokens: PythonObject,
) raises -> PythonObject:
    """`step` (op 0), `step_device` (op 1) and `step_accumulated` (op 2;
    `grads` is the microbatch gradient lists): the argument checks of the
    Python glue, the ONE kernel call, the assignments of its result.
    Returns the pre-clip total norm, or None."""
    door_ready()
    var buffer_pins = OptimizerInputOwners()
    # Host parameters are admitted one by one by tensor_view below, before
    # any read; only device-resident parameters need the list walk here.
    opt_owner(opt, OPT_DOOR.get_or_create_ptr()[].context, True)
    var lease = OptimizerOwnerLease(opt)
    var record = lease.value.downcast_value_ptr[OptimizerOwner]()
    # The instance dictionary is fetched once; set_field writes the same dict.
    var fields = py_fields(opt)
    var d = OPT_DOOR.get_or_create_ptr()
    var owhere = record[].control.name()
    var where = String("step") if op == OP_STEP else (
        String("step_device") if op == OP_STEP_DEVICE else String("step_accumulated"))
    var p_dev = record[].control.parameters_device
    if p_dev:
        opt_value(fields["params"], d[].context)
    var parts = List[PythonObject]()
    var t_tokens = -1
    if op == OP_STEP_DEVICE and not p_dev:
        raise Error("RuntimeError: mojolearn." + owhere + ".step_device: call params_to_device() first")
    opt_value(grads, d[].context)
    opt_scalar(tokens, d[].context)
    if op == OP_STEP_ACCUMULATED:
        parts = optimizer_sequence_items(grads, d[].context)
        var want = Int(py=fields["accumulation_steps"])
        if len(parts) != want:
            raise Error(
                "ValueError: mojolearn." + owhere + ".step_accumulated: " + String(len(parts))
                + " microbatch gradient lists for accumulation_steps=" + String(want) + "; the count is "
                + "part of the run's numerical specification (optimizer contract clause 9.2)" + _SRC
            )
        if not (tokens is None):
            t_tokens = optimizer_integer(tokens, OPT_DOOR.get_or_create_ptr()[].context)
        if len(parts) > 1 and not (tokens is None) and t_tokens < 1:
            raise Error("ValueError: mojolearn.accumulate_grads: tokens must be >= 1")
    var mn = 0.0
    if not (max_norm is None):
        mn = as_float(max_norm)
        if not (mn > 0.0):
            raise Error(
                "ValueError: mojolearn." + owhere + "." + where + ": max_norm must be > 0 or None, got "
                + py_repr(max_norm) + "; None is how 'no clipping' is spelled" + _SRC
            )
    var params = fields["params"]
    var j = len(params)
    var n_total = Int(py=fields["n_total"])
    var req = PublicStep()
    var keep = List[List[Float32]]()
    req.g_mode = GRADS_HOST
    req.a_count = 1
    var pviews = List[TensorView]()
    var have_pviews = False
    if op == OP_STEP_ACCUMULATED:
        req.a_count = len(parts)
        for i in range(len(parts)):  # driver-loop: over microbatch arguments
            var label = "microbatch_grads[" + String(i) + "]"
            var views = List[TensorView]()
            var gs = as_seq_views(parts[i], label, owhere, views, buffer_pins)
            check_views(views, label, owhere, req.a_count == 1)
            _check_grads(params, gs, views, owhere, where, pviews, have_pviews, buffer_pins)
            with GILReleased(Python()):
                for k in range(len(views)):  # driver-loop: over parameter buffers
                    if req.a_count > 1:
                        # read-only inputs: strided buffers are made contiguous
                        req.g_addr.append(optimizer_contiguous_address(views[k], keep))
                    else:
                        if views[k].readonly:
                            raise Error("ValueError: mojolearn: tensor is read-only, refusing to write to it")
                        req.g_addr.append(views[k].addr)
                    req.g_bytes.append(views[k].nbytes)
    elif p_dev and not (grads is None) and (
        (py_type(grads) is d[].dev_cls) or (
            (py_type(grads) is d[].context["container_types"][0] or py_type(grads) is d[].context["container_types"][1]) and len(grads) > 0
            and (py_type(grads[0]) is d[].dev_cls))):
        var gs = List[PythonObject]()
        if (py_type(grads) is d[].dev_cls):
            gs.append(grads)
        else:
            gs = optimizer_sequence_items(grads, d[].context)
        if len(gs) != j:
            raise Error(
                "ValueError: mojolearn." + owhere + "." + where + ": " + String(len(gs)) + " gradients for "
                + String(j) + " parameter tensors"
            )
        for k in range(len(gs)):  # driver-loop: over parameter buffers
            if not (py_type(gs[k]) is d[].dev_cls):
                raise Error("ValueError: mojolearn." + owhere + "." + where + ": grads[" + String(k)
                            + "] must be a device array")
        req.g_mode = GRADS_DEVICE
        for k in range(len(gs)):  # driver-loop: over parameter buffers: one device handle each
            buffer_pins.append(gs[k])
            req.g_addr.append(Int(py=py_fields(gs[k])["h"]))
            req.g_bytes.append(Int(py=py_fields(gs[k])["size"]))
    elif p_dev and grads is None:
        req.g_mode = GRADS_RESIDENT
    else:
        var views = List[TensorView]()
        var gs = as_seq_views(grads, "grads", owhere, views, buffer_pins)
        check_views(views, "grads", owhere, True)
        if p_dev and len(gs) != j:
            raise Error(
                "ValueError: mojolearn." + owhere + ": grads does not match the optimizer's registry ("
                + String(j) + " tensors, " + String(n_total) + " floats)"
            )
        if not p_dev:
            _check_grads(params, gs, views, owhere, where, pviews, have_pviews, buffer_pins)
        _addrs_of(views, req.g_addr, req.g_bytes)
    if not p_dev:
        # Re-read every parameter buffer's layout on EVERY step: a caller
        # can reshape or resize their own array between steps.
        if have_pviews:
            # Read this step by _check_grads; the dtype/layout refusals run
            # here in the same order without exporting each buffer again.
            check_views(pviews, "params", owhere, True)
        else:
            pviews = check_tensors(params, "params", owhere, True, buffer_pins)
        _addrs_of(pviews, req.p_addr, req.p_bytes)
    var schedule = fields["lr_schedule"]
    req.has_sched = False
    if not (schedule is None):
        req.has_sched = True
        # opt_owner above admitted this lr_schedule field with opt_schedule.
        req.sched = schedule_spec_admitted(schedule)
    var resident = fields["_resident_wanted"]
    if py_type(resident) is d[].context["scalar_types"][1] and not truth(resident):
        # `resident=False`: the state lives in these host arrays (no slot)
        host_moments(opt)
    var maximize = truth(getattr_or(opt, "maximize", PythonObject(False)))
    req.j = j
    req.t = Int(py=fields["t"])
    req.hyper = _hyper(opt, record[].control.kind, maximize, mn)
    req.n_total = n_total
    req.offsets = read_addr(fields["offsets"], "offsets", buffer_pins)
    req.init = write_addr(fields["buf_initialized"], "buf_initialized", buffer_pins)
    req.tokens = t_tokens
    _state_into(opt, record[].control, req, buffer_pins)
    req.p_dev = p_dev
    req.g_dev = record[].control.gradients_device
    req.device_op = op == OP_STEP_DEVICE
    var out = PublicStepOut(0, -1, False, False, Float32(0), Float32(0), Float32(0), -1, False)
    var driver = optimizer_driver(provider)
    var has_sched = req.has_sched
    with GILReleased(Python()):
        out = driver.step(req^)
    _ = provider
    _ = buffer_pins
    _ = keep^
    # the assignments of the native result
    set_field(opt, "t", PythonObject(out.t))
    keep_slot(record[], out.handle, provider, driver)
    record[].control.host_fresh = record[].control.handle < 0
    var total = PythonObject(None)
    if out.clip_ran:
        total = PythonObject(Float64(out.total_norm))
        set_field(opt, "total_norm_", total)
        set_field(opt, "clip_coef_", PythonObject(Float64(out.coef)))
    else:
        set_field(opt, "total_norm_", PythonObject(None))
        set_field(opt, "clip_coef_", PythonObject(None))
    var lr = PythonObject(Float64(out.lr))
    set_field(opt, "lr_", lr)
    if has_sched:
        set_field(opt, "lr", lr)
    if out.packed >= 0:
        set_field(opt, "packed_", PythonObject(out.packed != 0))
    record[].control.gradients_device = out.g_dev
    _ = lease
    return total


def optimizer_io(module: PythonObject, provider: PythonObject, opt: PythonObject,
    op_code: Int, tensors: PythonObject, name: String, tensor: Int) raises -> Int:
    door_ready()
    opt_owner(opt, OPT_DOOR.get_or_create_ptr()[].context)
    var lease = OptimizerOwnerLease(opt)
    var record = lease.value.downcast_value_ptr[OptimizerOwner]()
    var value = _optimizer_io_admitted(module, provider, opt, op_code, tensors, name, tensor, record[])
    _ = lease
    return value


def _optimizer_io_admitted(module: PythonObject, provider: PythonObject, opt: PythonObject,
    op_code: Int, tensors: PythonObject, name: String, tensor: Int, mut record: OptimizerOwner) raises -> Int:
    var buffer_pins = OptimizerInputOwners()
    # The instance dictionary is fetched once; set_field writes the same dict.
    var fields = py_fields(opt)
    var owhere = record.control.name()
    var j = len(fields["params"])
    var n_total = Int(py=fields["n_total"])
    var req = PublicStep()
    if not (tensors is None):
        var views = List[TensorView]()
        var seq = as_seq_views(tensors, name, owhere, views, buffer_pins)
        check_views(views, name, owhere, True)
        if len(seq) != j:
            raise Error(
                "ValueError: mojolearn." + owhere + ": " + name + " does not match the optimizer's registry ("
                + String(j) + " tensors, " + String(n_total) + " floats)"
            )
        _addrs_of(views, req.g_addr, req.g_bytes)
    req.j = j
    req.n_total = n_total
    req.offsets = read_addr(fields["offsets"], "offsets", buffer_pins)
    _state_into(opt, record.control, req, buffer_pins)
    var r = (-1, 0)
    var driver = optimizer_driver(provider)
    if driver.host:
        r = driver.io(req^, op_code, tensor)
    else:
        with GILReleased(Python()):
            r = driver.io(req^, op_code, tensor)
    _ = provider
    _ = buffer_pins
    try:
        keep_slot(record, r[0], provider, driver)
        # A device allocation's handle stays an integer through publication.
        # The caller has prepared its owner before asking for this allocation.
        return r[1]
    except error:
        if op_code == 5:
            try:
                with GILReleased(Python()):
                    _ = driver.free_device(r[1])
            except:
                pass
        raise error^

def optimizer_pack_io(provider: PythonObject, operation: Int, value: Int) raises -> PythonObject:
    """PACK a public legacy IO result; release an unpublished device handle."""
    try:
        return PythonObject(value) if operation == 5 else PythonObject(Float64(value))
    except error:
        if operation == 5:
            var driver = optimizer_driver(provider)
            try:
                with GILReleased(Python()):
                    _ = driver.free_device(value)
            except:
                pass
        raise error^


def door_io(module: PythonObject, provider: PythonObject, opt: PythonObject,
    operation: Int, tensors: PythonObject, name: String, tensor: Int) raises -> PythonObject:
    var value = optimizer_io(module, provider, opt, operation, tensors, name, tensor)
    return optimizer_pack_io(provider, operation, value)



def door_zero(opt: PythonObject, grads: PythonObject) raises -> PythonObject:
    """`zero_grad`: +0.0 into every gradient buffer."""
    door_ready()
    var buffer_pins = OptimizerInputOwners()
    opt_owner(opt, OPT_DOOR.get_or_create_ptr()[].context)
    var lease = OptimizerOwnerLease(opt)
    var owhere = lease.value.downcast_value_ptr[OptimizerOwner]()[].control.name()
    var views = List[TensorView]()
    _ = as_seq_views(grads, "grads", owhere, views, buffer_pins)
    check_views(views, "grads", owhere, True)
    var addrs = List[Int]()
    var nbytes = List[Int]()
    _addrs_of(views, addrs, nbytes)
    zero_tensors(addrs, nbytes)
    _ = buffer_pins
    _ = lease
    return PythonObject(None)


def _sync_host(opt: PythonObject, mut record: OptimizerOwner) raises:
    """`_download`: the slot's state into the mirrors, when they are stale,
    through the module that owns the slot."""
    var fields = py_fields(opt)
    var handle = record.control.handle
    if not optimizer_state_download_needed(handle, record.control.host_fresh):
        return
    host_moments(opt)
    var buffer_pins = OptimizerInputOwners()
    var h = handle
    var n = Int(py=fields["n_total"])
    var m = state_address(fields["_m"], n, "exp_avg", buffer_pins)
    var v = state_address(fields["_v"], n, "exp_avg_sq", buffer_pins)
    var driver = record.control.driver.value().copy()
    if driver.host:
        _ = driver.download(h, m, v, n)
    else:
        with GILReleased(Python()):
            _ = driver.download(h, m, v, n)
    _ = buffer_pins
    record.control.host_fresh = True


def _upload(record: OptimizerOwner, m: Int, v: Int, n: Int) raises:
    if record.control.handle < 0:
        return
    var driver = record.control.driver.value().copy()
    if driver.host:
        _ = driver.upload(record.control.handle, m, v, n)
    else:
        with GILReleased(Python()):
            _ = driver.upload(record.control.handle, m, v, n)


def copy_array(source: PythonObject) raises -> PythonObject:
    var v = TensorView()
    if not tensor_view(source, v) or not v.c_contig or v.itemsize != 4:
        raise Error("TypeError: optimizer state copy requires a contiguous float32/int32 buffer")
    var result = new_array(v.shape, "<f4" if base_format(v.fmt) == "f" else "<i4")
    memcpy(dest=BP(unsafe_from_address=Int(py=result._addr)), src=BP(unsafe_from_address=v.addr), count=v.nbytes)
    return result


def door_state(module: PythonObject, provider: PythonObject, opt: PythonObject, which: Int) raises -> PythonObject:
    """`state_dict()` (which 0) and the `exp_avg` / `exp_avg_sq` readings
    (1 / 2): the slot's state downloaded once per stale read."""
    door_ready()
    opt_owner(opt, OPT_DOOR.get_or_create_ptr()[].context)
    var lease = OptimizerOwnerLease(opt)
    var record = lease.value.downcast_value_ptr[OptimizerOwner]()
    _sync_host(opt, record[])
    record[].control.pristine = False
    host_moments(opt)
    if which == STATE_EXP_AVG:
        _ = lease
        return py_fields(opt)["_m"]
    if which == STATE_EXP_AVG_SQ:
        _ = lease
        return py_fields(opt)["_v"]
    var out = Python.dict()
    py_setitem(out, PythonObject("t"), PythonObject(Int(py=py_fields(opt)["t"])))
    py_setitem(out, PythonObject("exp_avg"), copy_array(py_fields(opt)["_m"]))
    py_setitem(out, PythonObject("exp_avg_sq"), copy_array(py_fields(opt)["_v"]))
    py_setitem(out, PythonObject("buf_initialized"), copy_array(py_fields(opt)["buf_initialized"]))
    _ = lease
    return out


def optimizer_export_state(opt: PythonObject) raises -> PythonObject:
    """PACK copy/pickle state with host buffers and no process-local handle.

    The owner lease spans synchronization and field capture. A restored copy
    opens its own native slot lazily; public buffers keep ordinary Python
    shallow/deep-copy behavior outside this operation.
    """
    opt_owner(opt, OPT_DOOR.get_or_create_ptr()[].context)
    var lease = OptimizerOwnerLease(opt)
    var record = lease.value.downcast_value_ptr[OptimizerOwner]()
    _sync_host(opt, record[])
    var fields = py_fields(opt)
    if record[].control.parameters_device:
        _ = _optimizer_io_admitted(PythonObject(None), record[].provider, opt, 2, fields["params"], "params", 0, record[])
    var result = Python.dict()
    var table = cpython_table()
    var position = List[Int](length=1, fill=0)
    var pair = List[PyObjectPtr](length=2, fill=PyObjectPtr())
    while table[].PyDict_Next(fields._obj_ptr, Int(position.unsafe_ptr()),
            Int(pair.unsafe_ptr()), Int(pair.unsafe_ptr() + 1)):  # driver-loop: PACK public serialized fields
        var key = PythonObject(from_borrowed=pair[0])
        if String(key) != "_native_optimizer_owner":
            py_setitem(result, key, PythonObject(from_borrowed=pair[1]))
    py_setitem(result, "_native_optimizer_pristine", PythonObject(record[].control.serialized_pristine(
        fields["_m"] is not None, fields["_v"] is not None)))
    _ = lease
    return result


def door_state_set(module: PythonObject, provider: PythonObject, opt: PythonObject, which: Int, value: PythonObject) raises -> PythonObject:
    """`exp_avg = value` (which 1) and `exp_avg_sq = value` (2)."""
    door_ready()
    opt_owner(opt, OPT_DOOR.get_or_create_ptr()[].context)
    var lease = OptimizerOwnerLease(opt)
    var record = lease.value.downcast_value_ptr[OptimizerOwner]()
    opt_value(value, OPT_DOOR.get_or_create_ptr()[].context)
    var name = String("exp_avg") if which == STATE_EXP_AVG else String("exp_avg_sq")
    var buffer_pins = OptimizerInputOwners()
    var count = Int(py=py_fields(opt)["n_total"])
    var a = 0 if value is None else state_address(value, count, name, buffer_pins)
    if which == STATE_EXP_AVG:
        _upload(record[], a, 0, count)
    else:
        _upload(record[], 0, a, count)
    set_field(opt, String("_m") if which == STATE_EXP_AVG else String("_v"), value)
    record[].control.pristine = False
    _ = buffer_pins
    _ = lease
    return PythonObject(None)


def door_load_state(
    module: PythonObject, provider: PythonObject, opt: PythonObject, t: PythonObject, ea: PythonObject, c1: PythonObject,
    es: PythonObject, c2: PythonObject, bi: PythonObject, c3: PythonObject,
) raises -> PythonObject:
    """`load_state_dict` after the conversion of the state's arrays
    (`_buffer.as_f32_c` / `as_i32_c`, which may copy): the optimizer keeps
    its own copies, the sizes are checked, the state is uploaded."""
    door_ready()
    opt_owner(opt, OPT_DOOR.get_or_create_ptr()[].context)
    opt_scalar(t, OPT_DOOR.get_or_create_ptr()[].context)
    opt_value(ea, OPT_DOOR.get_or_create_ptr()[].context)
    opt_value(es, OPT_DOOR.get_or_create_ptr()[].context)
    opt_value(bi, OPT_DOOR.get_or_create_ptr()[].context)
    var m = ea if truth(c1) else copy_array(ea)
    var v = es if truth(c2) else copy_array(es)
    optimizer_load_buffers(module, provider, opt, m, v, bi)
    set_field(opt, "buf_initialized", bi if truth(c3) else copy_array(bi))
    set_field(opt, "t", t)
    return PythonObject(None)


def optimizer_load_buffers(module: PythonObject, provider: PythonObject, opt: PythonObject,
    m: PythonObject, v: PythonObject, bi: PythonObject) raises:
    """UNPACK owned checkpoint outputs, run the native span/upload plan, PACK mirrors.

    Only API buffer owners cross this helper. Converted step/count/copy-policy
    words stay native in the public router; legacy entry flags are decoded by
    their own external UNPACK wrapper above.
    """
    # Both callers admit the current owner immediately before this boundary.
    var lease = OptimizerOwnerLease(opt)
    var record = lease.value.downcast_value_ptr[OptimizerOwner]()
    var n_total = Int(py=py_fields(opt)["n_total"])
    var owhere = record[].control.name()
    var buffer_pins = OptimizerInputOwners()
    var first_view = TensorView()
    var second_view = TensorView()
    var init_view = TensorView()
    if not tensor_view(m, first_view, buffer_pins) or not tensor_view(v, second_view, buffer_pins) or not tensor_view(bi, init_view, buffer_pins):
        raise Error("TypeError: optimizer checkpoint requires native state buffers")
    optimizer_checkpoint_spans(first_view, second_view, init_view, n_total, len(py_fields(opt)["params"]), owhere)
    set_field(opt, "_m", m)
    set_field(opt, "_v", v)
    record[].control.pristine = False
    _upload(record[], first_view.addr, second_view.addr, n_total)
    _ = buffer_pins
    record[].control.host_fresh = True
    _ = lease


comptime _NORM_TYPE_REFUSAL = (
    " is REFUSED BY NAME. The profile has ONE norm, the two-level L2 of optimizer contract 3.1; an L-inf or "
    "L1 clip is a different reduction with its own clause, its own fixture and its own sabotage, and none "
    "of them exist (python/mojolearn/_training_impl.py)"
)
comptime _NONFINITE_REFUSAL = (
    "mojolearn.clip_grad_norm_: error_if_nonfinite=False is REFUSED BY NAME. The profile ALWAYS refuses a "
    "non-finite total norm (optimizer contract 8a, refuse_nonfinite_scalar in "
    "training/checks/optimizer_oracle.mojo) and there is no build in which it does not, so accepting False "
    "would be accepting a promise this surface cannot keep. Skipping a step whose gradient norm is "
    "non-finite is a useful thing to do and the contract says where it belongs: OUTSIDE the pinned region, "
    "as an explicit recorded branch in your own loop (python/mojolearn/_training_impl.py)"
)


def door_clip(
    provider: PythonObject, grads: PythonObject, max_norm: PythonObject, norm_type: PythonObject, error_if_nonfinite: PythonObject,
    kwargs: PythonObject,
) raises -> PythonObject:
    """`clip_grad_norm_`: the refusals, the tensors, the in-place scale.
    Returns the pre-clip total norm."""
    door_ready()
    var buffer_pins = OptimizerInputOwners()
    refuse_kwargs(kwargs, "clip_grad_norm_")
    if as_float(norm_type) != 2.0:
        raise Error("NotImplementedError: mojolearn.clip_grad_norm_: norm_type=" + py_repr(norm_type)
                    + _NORM_TYPE_REFUSAL)
    if not truth(error_if_nonfinite):
        raise Error("NotImplementedError: " + _NONFINITE_REFUSAL)
    var views = List[TensorView]()
    _ = as_seq_views(grads, "grads", "clip_grad_norm_", views, buffer_pins)
    check_views(views, "grads", "clip_grad_norm_", True)
    var mn = as_float(max_norm)
    if not (mn > 0.0):
        raise Error(
            "ValueError: mojolearn.clip_grad_norm_: max_norm must be > 0, got " + py_repr(max_norm)
            + "; reaching this function IS the clip running, and 'no clipping' is spelled by not calling "
            + "it (training/estimator.mojo)"
        )
    var addrs = List[Int]()
    var nbytes = List[Int]()
    _addrs_of(views, addrs, nbytes)
    var r = (Float32(0), Float32(0))
    var driver = optimizer_driver(provider)
    var maximum = Float32(mn)
    with GILReleased(Python()):
        r = driver.clip(addrs^, nbytes^, maximum)
    _ = provider
    _ = buffer_pins
    return PythonObject(Float64(r[0]))


def door_accumulate(provider: PythonObject, microbatch_grads: PythonObject, tokens: PythonObject) raises -> PythonObject:
    """`accumulate_grads`: the microbatches checked, made contiguous where
    strided, combined by the clause 9.2 tree into new arrays."""
    door_ready()
    var buffer_pins = OptimizerInputOwners()
    opt_value(microbatch_grads, OPT_DOOR.get_or_create_ptr()[].context)
    opt_scalar(tokens, OPT_DOOR.get_or_create_ptr()[].context)
    var parts = optimizer_sequence_items(microbatch_grads, OPT_DOOR.get_or_create_ptr()[].context)
    if len(parts) == 0:
        raise Error("ValueError: mojolearn.accumulate_grads: no microbatches")
    var first = TensorView()
    var single = tensor_view(parts[0], first, buffer_pins)
    var probes = List[List[TensorView]]()
    for i in range(len(parts)):  # driver-loop: over microbatch arguments
        var views = List[TensorView]()
        _ = as_seq_views(parts[i], "microbatch_grads[" + String(i) + "]", "accumulate_grads", views, buffer_pins)
        probes.append(views^)
    for i in range(len(parts)):  # driver-loop: over microbatch arguments
        check_views(probes[i], "microbatch_grads[" + String(i) + "]", "accumulate_grads", False)
    var j0 = optimizer_accumulation_shapes(probes)
    var t_tokens = -1
    if not (tokens is None):
        t_tokens = optimizer_integer(tokens, OPT_DOOR.get_or_create_ptr()[].context)
        if t_tokens < 1:
            raise Error("ValueError: mojolearn.accumulate_grads: tokens must be >= 1")
    var keep = List[List[Float32]]()
    var in_addrs = List[Int]()
    with GILReleased(Python()):
        for i in range(len(probes)):  # driver-loop: admitted native microbatch plans
            for k in range(j0):  # driver-loop: parameter buffers
                in_addrs.append(optimizer_contiguous_address(probes[i][k], keep))
    var nbytes = List[Int]()
    var out_addrs = List[Int]()
    var result = List[PythonObject]()
    for k in range(j0):  # driver-loop: over parameter buffers: one result array each
        var a = new_array(probes[0][k].shape, "<f4")
        result.append(a)
        out_addrs.append(Int(py=a._addr))
        nbytes.append(probes[0][k].nbytes)
    var a_count = len(parts)
    var driver = optimizer_driver(provider)
    with GILReleased(Python()):
        _ = driver.accumulate(in_addrs^, nbytes^, out_addrs^, a_count, t_tokens)
    _ = provider
    _ = buffer_pins
    _ = keep^
    if single:
        return result[0]
    return optimizer_pack_list(result)


# ---------------------------------------------------------------------------
# the C boundary of each entry (shared by the GPU and the CPU binding)


def entry_args(packed: PyObjectPtr, count: Int, name: String) raises -> PythonObject:
    var args = PythonObject(from_borrowed=packed)
    if len(args) != count:
        raise Error("TypeError: " + name + " expects " + String(count) + " arguments")
    return args


def entry_step(py_self: PyObjectPtr, packed: PyObjectPtr) -> PyObjectPtr:
    """optimizer_step(opt, grads, max_norm, op, tokens)."""
    try:
        var a = entry_args(packed, 5, "optimizer_step_public")
        var module = PythonObject(from_borrowed=py_self)
        var provider = optimizer_provider(module)
        return door_step(module, provider, a[0], a[1], a[2], Int(py=a[3]), a[4]).steal_data()
    except error:
        return optimizer_python_error(error)


def entry_io(py_self: PyObjectPtr, packed: PyObjectPtr) -> PyObjectPtr:
    """optimizer_io(opt, op, tensors, name, tensor)."""
    try:
        var a = entry_args(packed, 5, "optimizer_io")
        var module = PythonObject(from_borrowed=py_self)
        var provider = optimizer_provider(module)
        return door_io(module, provider, a[0], Int(py=a[1]), a[2], String(py=a[3]),
                          Int(py=a[4])).steal_data()
    except error:
        return optimizer_python_error(error)


def entry_zero(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """optimizer_zero(opt, grads)."""
    try:
        var a = entry_args(packed, 2, "optimizer_zero")
        return door_zero(a[0], a[1]).steal_data()
    except error:
        return optimizer_python_error(error)


def entry_state(py_self: PyObjectPtr, packed: PyObjectPtr) -> PyObjectPtr:
    """optimizer_state(opt, which)."""
    try:
        var a = entry_args(packed, 2, "optimizer_state")
        var module = PythonObject(from_borrowed=py_self)
        var provider = optimizer_provider(module)
        return door_state(module, provider, a[0], Int(py=a[1])).steal_data()
    except error:
        return optimizer_python_error(error)


def entry_state_set(py_self: PyObjectPtr, packed: PyObjectPtr) -> PyObjectPtr:
    """optimizer_state_set(opt, which, value)."""
    try:
        var a = entry_args(packed, 3, "optimizer_state_set")
        var module = PythonObject(from_borrowed=py_self)
        var provider = optimizer_provider(module)
        return door_state_set(module, provider, a[0], Int(py=a[1]), a[2]).steal_data()
    except error:
        return optimizer_python_error(error)


def entry_load_state(py_self: PyObjectPtr, packed: PyObjectPtr) -> PyObjectPtr:
    """optimizer_load_state(opt, t, exp_avg, copied, exp_avg_sq, copied, buf_initialized, copied)."""
    try:
        var a = entry_args(packed, 8, "optimizer_load_state")
        var module = PythonObject(from_borrowed=py_self)
        var provider = optimizer_provider(module)
        return door_load_state(module, provider, a[0], a[1], a[2], a[3], a[4], a[5], a[6],
                                  a[7]).steal_data()
    except error:
        return optimizer_python_error(error)


def entry_clip(py_self: PyObjectPtr, packed: PyObjectPtr) -> PyObjectPtr:
    """optimizer_clip(grads, max_norm, norm_type, error_if_nonfinite, kwargs)."""
    try:
        var a = entry_args(packed, 5, "optimizer_clip")
        var module = PythonObject(from_borrowed=py_self)
        var provider = optimizer_provider(module)
        return door_clip(provider, a[0], a[1], a[2], a[3], a[4]).steal_data()
    except error:
        return optimizer_python_error(error)


def entry_accumulate(py_self: PyObjectPtr, packed: PyObjectPtr) -> PyObjectPtr:
    """optimizer_accumulate(microbatch_grads, tokens)."""
    try:
        var a = entry_args(packed, 2, "optimizer_accumulate")
        var module = PythonObject(from_borrowed=py_self)
        var provider = optimizer_provider(module)
        return door_accumulate(provider, a[0], a[1]).steal_data()
    except error:
        return optimizer_python_error(error)


def entry_init(py_self: PyObjectPtr, packed: PyObjectPtr) abi("C") -> PyObjectPtr:
    """optimizer_init(opt, params)."""
    try:
        var a = entry_args(packed, 2, "optimizer_init")
        return door_init(a[0], a[1]).steal_data()
    except error:
        return optimizer_python_error(error)
