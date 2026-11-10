# SPDX-License-Identifier: Apache-2.0
"""MinMaxScaler archive UNPACK/PACK over typed native members and plans."""
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_fields, py_get, py_setitem, py_contains
from bindings.classical_arrays import classical_is_array, classical_c_order, classical_numpy
from bindings.classical_pandas import classical_admit_value
from bindings.classical_archive import (
    _archive_unpack_member, _archive_class_name_unpack,
    _archive_open, _archive_methods, _archive_pack_members, _archive_file_call,
)
from bindings.classical_archive_read import _archive_read_input
from bindings.classical_archive_output import archive_events_pack, archive_numeric_pack
from bindings.classical_archive_linear import archive_print_mode_unpack
from bindings.native_buffer_owner import native_storage_allocate
from bindings.minmax_metadata import _flag, minmax_range_values
from core.native_archive_write import NativeArchiveMember
from core.native_archive_read import NativeArchiveDecoded, native_archive_read_status
from core.native_archive_minmax import (
    NativeMinMaxArchiveClass, NativeMinMaxArchivePlan, native_minmax_statistic_names,
    native_minmax_archive_plan, native_minmax_saved_statistic,
    native_minmax_archive_members, native_minmax_statistic_view,
)


def _minmax_archive_class_unpack(cls: PythonObject, context: PythonObject,
                                 generic: Bool = False) raises -> NativeMinMaxArchiveClass:
    var result = NativeMinMaxArchiveClass()
    var native_type = py_type(py_type(PythonObject("")))
    if py_type(cls) is not native_type:
        result.layout_error = 1
        return result^
    var type_fields = native_type.__dict__
    var mro = py_call(type_fields["__mro__"].__get__, cls)
    var descriptor_type = py_type(context["archive_memoryview_type"].obj)
    if generic:
        descriptor_type = context["host_archive_dict_descriptor_type"]
    var dictionary = False
    var finalizer = False
    for i in range(len(mro)):  # unpack: exact native class namespace metadata
        result.names.append(_archive_class_name_unpack(mro[i]))
        var fields = py_call(type_fields["__dict__"].__get__, mro[i])
        if (not generic or not finalizer) and py_contains(fields, "__del__"):
            if not generic or (fields["__del__"] is not py_get(context, "host_archive_finalizer_descriptor")
                    and fields["__del__"] is not py_get(context, "host_archive_peer_finalizer_descriptor")):
                if not result.layout_error:
                    result.layout_error = 2
            finalizer = True
        if not dictionary and py_contains(fields, "__dict__"):
            if py_type(fields["__dict__"]) is not descriptor_type and not result.layout_error:
                result.layout_error = 3
            dictionary = True
    if not dictionary and not result.layout_error:
        result.layout_error = 4
    return result^


def minmax_archive_save(model: PythonObject, path: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = py_fields(model)
    if py_get(fields, "numeric_mode_") is None or py_get(fields, "scale_") is None:
        raise Error("RuntimeError: this estimator is not fitted yet")
    var integers = py_type(PythonObject(0))
    if py_type(fields["n_features_in_"]) is not integers or py_type(fields["n_samples_seen_"]) is not integers:
        raise Error("TypeError: MinMaxScaler dimensions and counts must remain native integers")
    var features = Int(py=fields["n_features_in_"])
    var samples = Int64(py=fields["n_samples_seen_"])
    var clip = _flag(fields["clip_"], "clip_", context)
    var endpoints = minmax_range_values(fields["feature_range_"], context)
    var names = native_minmax_statistic_names()
    var values = List[PythonObject]()
    for name in names:  # unpack: five public statistic buffers, in refusal order
        var value = fields[PythonObject(name + "_")]
        classical_admit_value(value, context)
        if not classical_is_array(value, context):
            raise Error("MinMaxScaler saved statistics must remain contiguous Float32 feature vectors")
        native_minmax_saved_statistic(String(value.dtype), Int(py=value.ndim), Int(py=value.size),
                                     features, classical_c_order(value, context))
        values.append(value)
    # Preserve the existing file-open boundary after saved-state admission.
    # Mode storage and member buffers are unpacked only after that open.
    var opened = _archive_open(path, context)
    var file = opened[0]
    var methods = _archive_methods(file, context)
    var pins = List[PythonObject]()
    var members = List[NativeArchiveMember]()
    try:
        var statistics = List[NativeArchiveMember]()
        for i in range(len(names)):  # unpack: native descriptors and borrowed owner pins
            statistics.append(_archive_unpack_member(names[i], values[i], context, pins))
        var mode = _archive_unpack_member("numeric_mode", fields["numeric_mode_"], context, pins)
        members = native_minmax_archive_members(statistics^, mode^, _archive_class_name_unpack(py_type(model)),
            features, samples, clip, endpoints[0], endpoints[1])
        _archive_pack_members(file, methods, members)
    finally:
        if Bool(py=opened[1]):
            _ = _archive_file_call(methods["close"], file)
        _ = values
        _ = pins
        _ = members
    return path


def _minmax_archive_pack(cls: PythonObject, plan: NativeMinMaxArchivePlan,
                         members: List[NativeArchiveDecoded], context: PythonObject,
                         generic: Bool = False) raises -> PythonObject:
    var model = native_storage_allocate(cls)
    var fields = py_fields(model)
    py_setitem(fields, "copy", True)
    py_setitem(fields, "numeric_mode", PythonObject(None))
    py_setitem(fields, "clip", plan.clip)
    py_setitem(fields, "feature_range", Python.tuple(plan.lower, plan.upper))
    var output_context = context
    if not generic:
        output_context = py_get(context, "output_context", context)
    var names = native_minmax_statistic_names()
    for i in range(len(names)):  # pack: fixed public statistic owners, under the GIL
        var value = PythonObject(None)
        if generic:
            # The generic map used to publish the saved rank-one order flag.
            # Direct scaler loading deliberately publishes a C-order vector.
            value = archive_numeric_pack(members[plan.statistics[i]], output_context)
        else:
            var view = native_minmax_statistic_view(members[plan.statistics[i]])
            value = archive_numeric_pack(view, output_context)
        py_setitem(fields, PythonObject(names[i] + "_"), value)
    py_setitem(fields, "n_features_in_", plan.features)
    py_setitem(fields, "n_samples_seen_", plan.samples)
    py_setitem(fields, "numeric_mode_", PythonObject(plan.mode))
    py_setitem(fields, "feature_range_", Python.tuple(plan.lower, plan.upper))
    py_setitem(fields, "clip_", plan.clip)
    _ = members
    return model


def minmax_archive_decoded(cls: PythonObject, members: List[NativeArchiveDecoded],
                           context: PythonObject, print_mode: Int) raises -> PythonObject:
    """Generic loader PACK over its still-owned native decoded archive."""
    var metadata = _minmax_archive_class_unpack(cls, context, True)
    var plan = native_minmax_archive_plan(members, metadata, classical_numpy(context), print_mode, True)
    return _minmax_archive_pack(cls, plan, members, context, True)


def minmax_archive_load(cls: PythonObject, path: PythonObject, context: PythonObject) raises -> PythonObject:
    var admitted = _archive_read_input(path, context)
    var minor = Int(py=context["archive_python_minor"])
    var micro = Int(py=context["archive_python_micro"])
    try:
        var members = native_archive_read_status(admitted.span, minor, micro)
        archive_events_pack(members, context)
        var metadata = _minmax_archive_class_unpack(cls, context)
        var plan = native_minmax_archive_plan(members, metadata, classical_numpy(context),
                                              archive_print_mode_unpack(members, context))
        return _minmax_archive_pack(cls, plan, members, context)
    finally:
        admitted.close()
