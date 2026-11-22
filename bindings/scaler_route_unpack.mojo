# SPDX-License-Identifier: Apache-2.0
"""The scaler family's resolver on the ONE device route owner (lane dw-prep, round 2).

`DeviceRouteOwner` (providers/python/device_route_boundary_door.mojo) holds an
estimator's route; the rule and the words are core/device_route.mojo. This
file is only the family's resolver: what the route of a StandardScaler,
MinMaxScaler, MaxAbsScaler or RobustScaler holds, and how a door reads it.

What the route holds, resolved once per estimator:
  * `context` (pin, compared by identity): the connected context of the door;
  * `scope`: the connected context the estimator runs in: the door's own, the
    CPU one import connected beside it on a GPU installation
    (`prep_cpu_context`), or an explicit Host* class's own;
  * `cls`: the exact `str` type (the mode word's type test of a selection);
  * `images`: the entry dictionary of the selected numeric mode in that
    context, with `mode` / `default_mode` the mode objects it was selected
    for (identity hits) and `route.mode` its word (1 identical, 0 fast);
  * `route.entry`: the x_prep transport entry; `route.second`: the scaler
    kernel entry (StandardScaler, MinMaxScaler); `route.flags`: the image's
    capability bits (SCALER_CAP_*) and the two route bits.

The device is read ONCE, by `_scaler_route_build`, at the estimator's first
data operation. No public call reads or compares it. A data operation reads
the owner (one record read, one pointer compare) for its context, and each
entry selection, calls construction and x_prep run of the operation reads
the owner again for the held selection in place of the dictionary reads, the
mode probes, the by-name capability reads and the table slot read it used to
make (the per-call cost goes down; the report lists each site).

An estimator without an owner for the context it is run in (the scratch batch
model of MinMaxScaler.partial_fit, a shard worker's model) selects as before:
every function here answers "not held" and the door's old read runs.

Resolution is a metadata operation with no independent numerical work to
split; it runs once per estimator, never per row.
"""
from std.python import PythonObject
from bindings.py_balanced import py_type, py_get, py_fields, py_call, py_setitem
from bindings.prep_entry_cache_unpack import prep_entry_word, prep_entry_text
from bindings.x_prep_native_owner_run import XprepCaps
from core.device_route import (
    DEVICE_GPU, DEVICE_CPU, device_installed, device_cpu_fast, device_mode_code,
)
from core.entry_table import EntryRef, entry_ref_call, entry_ref_unresolved
from core.prep_entry_table import PrepEntryKey, xp_transport_ref, scaler_ref
from core.scaler_native_api import SCALER_OP_DIRECT
from providers.python.device_route_boundary_door import (
    DeviceRouteOwner, device_request, device_admit, device_route_owner, device_route_keep, device_route_clear,
    device_cpu_connect_late,
)
from x_prep.transport_abi import xp_vendor_word

# Capability bits of the selected image (route.flags).
comptime SCALER_CAP_RUN_OUT = 1
comptime SCALER_CAP_RUN_SCRATCH = 2
comptime SCALER_CAP_RUN_RANGES = 4
comptime SCALER_CAP_DEV_PUT = 8
comptime SCALER_CAP_DEV_FREE = 16
comptime SCALER_CAP_DEV_TAKE = 32
comptime SCALER_CAP_MAXABS_DIRECT = 64
comptime SCALER_CAP_TRANSPORT = 128
comptime SCALER_CAP_KERNEL = 256
comptime SCALER_CAP_KERNEL_DIRECT = 512
comptime SCALER_CAP_STANDARD_FIT = 1024
comptime SCALER_CAP_MINMAX_FIT = 2048
comptime _SCALER_CAP_MASK = 4095
# Route bits (kept across selections).
comptime SCALER_ROUTE_CPU_TABLE = 1048576
comptime SCALER_ROUTE_HOST = 2097152


def _scaler_process_vendor(context: PythonObject) raises -> Int:
    """The selected vendor word of the process (0: a CPU-only process)."""
    return xp_vendor_word(String(context["vendor"]))


def _scaler_explicit_host(cls: PythonObject, context: PythonObject) raises -> PythonObject:
    """The explicit host context of `cls` (a mojolearn.host_model class), or None."""
    var hosts = py_get(context, "scaler_host_contexts")
    if hosts is None:
        return PythonObject(None)
    var host = py_get(hosts, cls)
    return host


def scaler_device_admit(value: PythonObject, cls: PythonObject, context: PythonObject) raises:
    """The shared `device_admit` where a scaler's device is given (the
    constructor, `set_params`): refused in the shared words before the
    estimator changes. The default None is one identity compare and reads
    nothing. The explicit Host* classes run on the CPU by class and take any
    admitted value. The builtins module is the cell import connected
    (python/mojolearn/_native_prep_device.py), read only for a given value."""
    if value is None:
        return
    var b = context["prep_builtins"]
    if _scaler_explicit_host(cls, context) is not None:
        device_admit(value, 1, b)
        return
    device_admit(value, _scaler_process_vendor(context), b)


def scaler_route_clear(model: PythonObject) raises:
    """Forget the estimator's route (a new `__init__`, `set_params`)."""
    device_route_clear(model)


def _scaler_cpu_sync(cpu: PythonObject, context: PythonObject, b: PythonObject) raises:
    """Copy the CPU context again from its source context when the source
    changed since the copy (another family connecting to the owner's
    context, or NumPy connecting at the first construction, which adds keys
    AND replaces the type tuples `native_value_types`, `scalar_types`, ...):
    the source's size at the copy is recorded (`prep_cpu_source_size`,
    python/mojolearn/_native_prep_device.py). Every value is the source's
    except the CPU context's own (`prep_cpu_overrides`: the CPU entries, the
    vendor). The former form copied only the keys the source gained and kept
    the stale type tuples, so a device="cpu" scaler whose CPU context was
    copied before NumPy connected refused a NumPy X ("Sequential requires
    native input types", proving suite 2026-11-22). In place: estimators
    already resolved to the CPU context see the values too. Once per
    estimator resolution; one dictionary read and one size compare when
    nothing changed."""
    var held = py_get(cpu, "prep_cpu_source_size")
    var size = len(context)
    if held is not None and Int(py=held) == size:
        return
    var merged = py_call(b.dict, context)
    var overrides = py_get(cpu, "prep_cpu_overrides")
    if overrides is not None:
        for k in range(len(overrides)):  # small-loop(overrides): the CPU context's few own keys
            var key = overrides[k]
            py_setitem(merged, key, cpu[key])
        py_setitem(merged, PythonObject("prep_cpu_overrides"), overrides)
    py_setitem(merged, PythonObject("prep_cpu_source_size"), PythonObject(size))
    _ = py_call(cpu.update, merged)


def _scaler_route_build(model: PythonObject, context: PythonObject) raises -> DeviceRouteOwner:
    var b = context["prep_builtins"]
    var owner = DeviceRouteOwner(DEVICE_GPU)
    var host = _scaler_explicit_host(py_type(model), context)
    if host is not None:
        owner.route.device = DEVICE_CPU
        owner.route.flags = SCALER_ROUTE_HOST
        owner.scope = host
    else:
        # The one read of `device`. None, and an instance without the field,
        # run this installation's device; an explicit 'gpu' in a CPU-only
        # process is refused in the shared words (an instance that reached
        # here without its constructor's check).
        var process_vendor = _scaler_process_vendor(context)
        owner.route.device = device_installed(device_request(py_fields(model), b), process_vendor)
        if owner.route.device == DEVICE_CPU and process_vendor != 0:
            var cpu = py_get(context, "prep_cpu_context")
            # Built at the first device="cpu" admission (fx-host-import:
            # python/mojolearn/_native_prep_device.py, no longer at import);
            # an estimator restored without construction connects it here,
            # once, at its first route resolution.
            if cpu is None and device_cpu_connect_late(context):
                cpu = py_get(context, "prep_cpu_context")
            if cpu is None:
                raise Error("ImportError: device='cpu' is not connected in this installation")
            if py_type(cpu) is b.str:
                raise Error("ImportError: " + String(cpu))
            _scaler_cpu_sync(cpu, context, b)
            owner.route.flags = SCALER_ROUTE_CPU_TABLE
            owner.scope = cpu
        else:
            owner.scope = context
    owner.cls = b.str
    owner.context = context
    owner.resolved = True
    return owner^


def scaler_route_context(model: PythonObject, context: PythonObject) raises -> PythonObject:
    """The connected context this estimator's data operation runs in. A hit
    is one record read and one pointer compare; a miss (the estimator's
    first data operation, after `set_params` or a new `__init__`) resolves
    and stores the route. `model` is an estimator instance, never a class."""
    var owner = device_route_owner(model)
    if owner is not None:
        var native = owner.downcast_value_ptr[DeviceRouteOwner]()
        if native[].resolved and native[].context is context:
            var held = native[].scope
            _ = owner  # the owner pins the context through the copy above
            return held
    var fresh = device_route_keep(model, _scaler_route_build(model, context))
    var run = fresh.downcast_value_ptr[DeviceRouteOwner]()[].scope
    _ = fresh
    return run


def scaler_route_refuse_fast(model: PythonObject, mode: PythonObject, context: PythonObject) raises:
    """`device="cpu"` beside a GPU set runs identical only: the shared
    refusal (core/device_route.mojo `device_cpu_fast`), said before the CPU
    context's entries are read. A selection miss path only."""
    var owner = device_route_owner(model)
    if owner is None:
        return
    var native = owner.downcast_value_ptr[DeviceRouteOwner]()
    if native[].scope is not context or (native[].route.flags & SCALER_ROUTE_CPU_TABLE) == 0:
        return
    if py_type(mode) is native[].cls and String(mode) == "fast":
        device_cpu_fast()


def scaler_selection_held(model: PythonObject, mode: PythonObject, context: PythonObject) raises -> PythonObject:
    """The entry dictionary the estimator's route holds for `mode` in the
    run context `context`, or None (no owner for this context, nothing
    selected yet, another mode word, a mode that is not a `str`). A hit by
    identity is one record read and three pointer compares; a mode object
    not seen before costs one type compare and one word compare more."""
    var owner = device_route_owner(model)
    if owner is None:
        return owner
    var native = owner.downcast_value_ptr[DeviceRouteOwner]()
    if not native[].resolved or native[].scope is not context or native[].images is None:
        return PythonObject(None)
    if mode is native[].mode or mode is native[].default_mode:
        var held = native[].images
        _ = owner
        return held
    if py_type(mode) is not native[].cls:
        return PythonObject(None)
    if device_mode_code(String(mode)) != native[].route.mode:
        return PythonObject(None)
    # The same word as the held selection, in a new object (a door that
    # lower-cases its mode per call): keep it as the second identity pin.
    native[].default_mode = mode
    var same = native[].images
    _ = owner
    return same


def scaler_selection_keep(model: PythonObject, mode: PythonObject, entries: PythonObject,
    context: PythonObject) raises:
    """Record the selection a door just made by its own checks (`_entries`,
    `scaler_fit_entries`, `_minmax_entries`): the entries of `mode`, the
    image's capability bits, its transport entry and, where the image
    publishes one, its scaler kernel entry. Nothing is kept for an estimator
    without an owner for this run context. A selection miss path only."""
    var owner = device_route_owner(model)
    if owner is None:
        return
    var native = owner.downcast_value_ptr[DeviceRouteOwner]()
    if not native[].resolved or native[].scope is not context:
        return
    var code = device_mode_code(String(mode))
    if code < 0:
        return
    var flags = native[].route.flags & (SCALER_ROUTE_CPU_TABLE | SCALER_ROUTE_HOST)
    if py_get(entries, "x_prep_run_out") is not None:
        flags |= SCALER_CAP_RUN_OUT
    if py_get(entries, "x_prep_run_scratch") is not None:
        flags |= SCALER_CAP_RUN_SCRATCH
    if py_get(entries, "x_prep_run_ranges") is not None:
        flags |= SCALER_CAP_RUN_RANGES
    if py_get(entries, "x_prep_dev_put") is not None:
        flags |= SCALER_CAP_DEV_PUT
    if py_get(entries, "x_prep_dev_free") is not None:
        flags |= SCALER_CAP_DEV_FREE
    if py_get(entries, "x_prep_dev_take_rows") is not None:
        flags |= SCALER_CAP_DEV_TAKE
    if py_get(entries, "x_prep_maxabs_fit_direct") is not None:
        flags |= SCALER_CAP_MAXABS_DIRECT
    if py_get(entries, "standard_fit") is not None:
        flags |= SCALER_CAP_STANDARD_FIT
    if py_get(entries, "minmax_fit") is not None:
        flags |= SCALER_CAP_MINMAX_FIT
    var transport = entry_ref_unresolved()
    var kernel = entry_ref_unresolved()
    var image_mode = prep_entry_word(entries, "x_prep_numeric_mode", -1)
    var image_vendor = xp_vendor_word(prep_entry_text(entries, "x_prep_vendor"))
    if image_mode >= 0 and image_mode <= 1 and image_vendor >= 0:
        var key = PrepEntryKey(image_mode, image_vendor)
        try:
            transport = xp_transport_ref(key)
            flags |= SCALER_CAP_TRANSPORT
        except:
            # Said by the run that needs it, in the table's own words
            # (the door's old `xprep_key` read).
            pass
        if (flags & SCALER_ROUTE_HOST) == 0 and (flags & (SCALER_CAP_STANDARD_FIT | SCALER_CAP_MINMAX_FIT)) != 0:
            try:
                kernel = scaler_ref(key)
                flags |= SCALER_CAP_KERNEL
                var a: Array[Int, 0] = []
                var p: Array[Int, 0] = []
                var f: Array[Float64, 0] = []
                if entry_ref_call(kernel, SCALER_OP_DIRECT, a, p, f) == 1:
                    flags |= SCALER_CAP_KERNEL_DIRECT
            except:
                # Said by the calls construction that needs it (`ScalerDoorCalls`).
                pass
    native[].mode = mode
    native[].default_mode = PythonObject(None)
    native[].images = entries
    native[].route.mode = code
    native[].route.vendor = image_vendor
    native[].route.entry = transport
    native[].route.second = kernel
    native[].route.has_second = (flags & SCALER_CAP_KERNEL) != 0
    native[].route.flags = flags


@fieldwise_init
struct ScalerHeld(ImplicitlyCopyable, Movable):
    """The held selection of `entries`, by value: `held` False when the
    estimator's route does not hold exactly this entry dictionary."""
    var held: Bool
    var flags: Int
    var transport: EntryRef
    var kernel: EntryRef


def scaler_route_held(model: PythonObject, entries: PythonObject) raises -> ScalerHeld:
    """The capability bits and resolved entries the estimator's route holds
    for `entries` (one record read, one pointer compare). The caller keeps
    `entries` named until its operation returns: it pins the images."""
    var owner = device_route_owner(model)
    if owner is not None:
        var native = owner.downcast_value_ptr[DeviceRouteOwner]()
        if native[].resolved and native[].images is entries:
            var found = ScalerHeld(True, native[].route.flags, native[].route.entry, native[].route.second)
            _ = owner
            return found
    return ScalerHeld(False, 0, entry_ref_unresolved(), entry_ref_unresolved())


def scaler_held_flags(held: ScalerHeld, entries: PythonObject) raises -> Int:
    """The capability bits of `entries`: the route's when it holds this
    selection, else read by name as the doors read them before."""
    if held.held:
        return held.flags
    var flags = 0
    if py_get(entries, "x_prep_run_out") is not None:
        flags |= SCALER_CAP_RUN_OUT
    if py_get(entries, "x_prep_run_scratch") is not None:
        flags |= SCALER_CAP_RUN_SCRATCH
    if py_get(entries, "x_prep_maxabs_fit_direct") is not None:
        flags |= SCALER_CAP_MAXABS_DIRECT
    return flags


def scaler_held_caps(held: ScalerHeld) -> XprepCaps:
    """The x_prep run capabilities of a held selection. `held` False in the
    result (no route holds the selection, or its image publishes no
    transport table): the run reads them from its entries as before."""
    if held.held and (held.flags & SCALER_CAP_TRANSPORT) != 0:
        return XprepCaps(True, held.transport, (held.flags & SCALER_CAP_RUN_OUT) != 0,
                         (held.flags & SCALER_CAP_RUN_SCRATCH) != 0, (held.flags & SCALER_CAP_RUN_RANGES) != 0,
                         (held.flags & SCALER_CAP_DEV_PUT) != 0 and (held.flags & SCALER_CAP_DEV_FREE) != 0)
    return XprepCaps(False, entry_ref_unresolved(), False, False, False, False)


def scaler_route_vendor_used(model: PythonObject, context: PythonObject) raises -> PythonObject:
    """`vendor_used()`: the vendor word the compiled x_prep image of this
    estimator's route reports about itself (`cpu`, `metal`, `cuda`, `hip`),
    or None for an image built before the read-back. A metadata operation,
    in no loop."""
    var run = scaler_route_context(model, context)
    var str_type = context["prep_builtins"].str
    var mode = py_get(py_fields(model), "numeric_mode")
    if mode is None:
        mode = run["backend_fields"]["_DEFAULT_MODE"]
    if py_type(mode) is not str_type:
        raise Error("numeric_mode must be fast, identical or None")
    var word = py_call(py_call(mode.strip).lower)
    scaler_route_refuse_fast(model, word, run)
    var entries = py_get(run["scaler_bindings"], word)
    if entries is None:
        raise Error("numeric_mode must be fast, identical or None")
    if py_type(entries) is str_type:
        raise Error("ImportError: " + String(entries))
    var text = prep_entry_text(entries, "x_prep_vendor")
    if text.byte_length() == 0:
        return PythonObject(None)
    return PythonObject(text)
