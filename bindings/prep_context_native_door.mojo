# SPDX-License-Identifier: Apache-2.0
# Stateless preprocessing admits and snapshots its native provider context
# before lazy NumPy setup, dispatch, metadata access, or numerical execution.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.classical_attributes_door import classical_connect_numpy
from core.shared_slot import SharedSlot
from bindings.prep_keys_unpack import prep_keys
from bindings.py_balanced import py_item, py_contains
from providers.python.device_route_boundary_door import device_cpu_connect_late


def _exact_dict(mapping: PythonObject, b: PythonObject) raises:
    if py_type(mapping) is not b.dict:
        raise Error("TypeError: preprocessing provider metadata requires an exact dictionary")


def _copy_records(records: PythonObject, b: PythonObject) raises -> PythonObject:
    # The operation's copy is built from the immutable import-time records
    # alone (one C-level dict construction), so a later change to the import
    # context can never reach it. The former per-key walk of the source
    # context (key admission plus an identity test per key, every operation)
    # only re-proved that, and is gone. `b` is the operation's admitted
    # builtins module (one stamp per operation, not one per record copy).
    if py_type(records) is not b.tuple:
        raise Error("TypeError: preprocessing provider identity records must remain immutable")
    return py_call(b.dict, records)


def _fixed_maps(destination: PythonObject, records: PythonObject, b: PythonObject) raises:
    if py_type(records) is not b.tuple:
        raise Error("TypeError: preprocessing fixed provider maps must remain immutable")
    for i in range(len(records)):  # driver-loop: fixed class/descriptor/dtype-code maps
        var record = records[i]
        if py_type(record) is not b.tuple or len(record) != 3 or py_type(record[0]) is not b.str:
            raise Error("TypeError: preprocessing provider map identity is invalid")
        py_setitem(destination, record[0], _copy_records(record[2], b))


# The context keys classical_connect_numpy publishes. The first operation
# after NumPy is loaded connects them on its copy; later operations reuse
# them instead of rebuilding about forty NumPy dtypes per call.
comptime _NUMPY_CACHE = "prep_operation_numpy"


def _numpy_names() raises -> PythonObject:
    return Python.tuple("ndarray", "scalar_type", "integer", "scalar_types", "integer_types",
                        "native_value_types", "buffer_owner_types", "numeric_dtypes", "class_dtypes",
                        "archive_npy_dtypes", "float_types", "int_types", "bool_types", "float32",
                        "float64", "int32", "int64", "uint64", "bool")


def _connect_numpy_once(source: PythonObject, context: PythonObject) raises:
    var nk = prep_keys()
    if py_get(context, nk[].k_ndarray) is not None:
        return
    var cached = py_get(source, _NUMPY_CACHE)
    if cached is not None:
        _ = py_call(context.update, cached)
        return
    classical_connect_numpy(context)
    if py_get(context, nk[].k_ndarray) is None:
        return  # NumPy is not loaded; the next operation looks again
    var names = _numpy_names()
    var kept = Python.dict()
    for i in range(len(names)):  # small-loop(the fixed NumPy admission key names)
        py_setitem(kept, names[i], context[names[i]])
    # A new key of the import context, outside the pinned records: the
    # records (and so every operation copy) are unchanged by it.
    py_setitem(source, _NUMPY_CACHE, kept)


struct _ContextTemplate(Defaultable, Movable):
    # The operation context and output context built once from the immutable
    # CONNECTION records (records copied, fixed maps built), kept with a
    # strong reference to the connection tuple and reused while the bridge
    # still exports that same tuple (p3-prep). Each operation takes its own
    # shallow copy of both dicts; the fixed lookup maps (array_fields,
    # array_storage_slots, buffer_codes, prep_stateless_kinds,
    # warning_category_slots) are shared read-only tables that no operation
    # writes. The templates never leave this module.
    # `cpu_context` is the template of an estimator with `device="cpu"` on a
    # GPU installation: the same records with the host x_prep image as the
    # identical binding and `vendor="cpu"`, frozen at import as the
    # connection's seventh member and built here once, at the first such
    # operation. It shares the output template and the fixed maps' records.
    var connection: PythonObject
    var context: PythonObject
    var output: PythonObject
    var cpu_context: PythonObject

    def __init__(out self):
        self.connection = PythonObject(None)
        self.context = PythonObject(None)
        self.output = PythonObject(None)
        self.cpu_context = PythonObject(None)


comptime _CONTEXT_TEMPLATE = SharedSlot[_ContextTemplate, "MojolearnPrepOperationContextTemplateV3"]


def _connection(source: PythonObject, b: PythonObject) raises -> PythonObject:
    # native_loaded_module verifies the bridge module and the exact immutable
    # CONNECTION export captured at import. Replacing a context's own alleged
    # identity table cannot authorize callbacks or new descriptor references.
    var connection = native_loaded_module("mojolearn._native_prep_stateless_bridge").CONTEXT_CONNECTION
    if py_type(connection) is not b.tuple or len(connection) != 7 or source is not connection[0]:
        raise Error("TypeError: preprocessing requires its originally connected native provider context")
    return connection


def _cpu_template_records(connection: PythonObject, b: PythonObject) raises -> PythonObject:
    """The CPU template's records from the connection's cell (python/mojolearn/
    _native_prep_stateless_bridge.py `_cpu_cell`): None in a CPU-only
    process, the refusal text when the host x_prep image is not built, else
    the frozen records. The cell is filled at the first device="cpu"
    admission (fx-audit 9: never at import); an estimator restored without
    construction connects it here, once (`device_cpu_connect_late`). Read
    only for a CPU route of a GPU process, once per estimator."""
    var cell = connection[6]
    if py_type(cell) is not b.dict:
        raise Error("TypeError: preprocessing requires its originally connected CPU template cell")
    var key = PythonObject("records")
    if not py_contains(cell, key):
        _ = device_cpu_connect_late(cell)
    if not py_contains(cell, key):
        raise Error("ImportError: device='cpu' is not connected in this installation")
    return py_item(cell, key)


def prep_cpu_connected(source: PythonObject, b: PythonObject) raises -> Bool:
    """True when this installation holds a CPU template beside its own (a GPU
    installation whose host x_prep image is connected). False in a
    CPU-only process: its own template already is the CPU one. A host image
    that is not built is refused here in its import words. Read once per
    estimator, when its device route is resolved; never per call. `b` is
    the caller's admitted builtins module."""
    var records = _cpu_template_records(_connection(source, b), b)
    if records is None:
        return False
    if py_type(records) is b.str:
        raise Error("ImportError: " + String(records))
    return True


def prep_operation_context(source: PythonObject, cpu: Bool) raises -> PythonObject:
    """The operation's own context, copied from the template of the device
    the estimator's route names (`cpu`: the caller's route, resolved once per
    estimator; this function reads no device)."""
    var nk = prep_keys()
    var b = native_loaded_module("builtins")
    var connection = _connection(source, b)
    var g = _CONTEXT_TEMPLATE.get_or_create_ptr()
    if g[].connection is not connection or g[].context is None:
        var built = _copy_records(connection[1], b)
        _fixed_maps(built, connection[2], b)
        var built_output = _copy_records(connection[4], b)
        _fixed_maps(built_output, connection[5], b)
        g[].context = built
        g[].output = built_output
        g[].cpu_context = PythonObject(None)
        g[].connection = connection
    var template = g[].context
    if cpu:
        if g[].cpu_context is None:
            var records = _cpu_template_records(connection, b)
            if py_type(records) is b.str:
                raise Error("ImportError: " + String(records))
            if records is not None:
                var built_cpu = _copy_records(records, b)
                _fixed_maps(built_cpu, connection[2], b)
                g[].cpu_context = built_cpu
        # None still: a CPU-only process, whose own template is the CPU one.
        if g[].cpu_context is not None:
            template = g[].cpu_context
    var context = py_call(b.dict, template)
    var output = py_call(b.dict, g[].output)
    # The process mode is mutable process data: the backend namespace (a
    # small module dict) is copied in C before a readback or input
    # preparation can release the GIL, so one operation reads one mode.
    var process = py_item(context, nk[].k_backend_fields)
    var modules = py_item(context, nk[].k_modules)
    _exact_dict(process, b)
    _exact_dict(modules, b)
    var process_copy = py_call(b.dict, process)
    py_setitem(context, nk[].k_backend_fields, process_copy)
    py_setitem(output, nk[].k_backend_fields, process_copy)
    # sys.modules (~1000-3000 keys with NumPy loaded) is no longer copied per
    # operation (perf2-prep): the stateless chain only looks modules up by
    # name (`py_get(context["modules"], name)`), exactly as every other native
    # family reads the live sys.modules its import context holds, and a
    # lookup after a GIL window sees the current module, as theirs do.
    py_setitem(output, nk[].k_modules, modules)
    py_setitem(context, nk[].k_output_context, output)
    _connect_numpy_once(source, context)
    return context
