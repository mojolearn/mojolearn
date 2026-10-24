# SPDX-License-Identifier: Apache-2.0
# The model-selection resident owner's entries (`_Resident._entry`/`take`)
# and DeviceRows.materialize. They sit above both the Array preparation
# owner (arena_materialize_native) and the model-selection input owner
# (msel_inputs, which itself imports arena_materialize_native), so foreign
# containers reach the native materializer without an import cycle.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.arena_materialize_native import (
    _native_call, _prepare, _prepared_info, _cast, _slot, _empty, _resident_fields,
)
from bindings.msel_inputs import msel_input


def _entry(values: PythonObject, entries: PythonObject, indices: Bool, context: PythonObject) raises -> PythonObject:
    var b = context["arena_native_types"]
    var key = py_call(context["arena_id"],values)
    var hit = py_get(entries,key)
    if hit is not None:
        if py_type(hit) is not b["tuple"] or len(hit) != 3:
            raise Error("TypeError: native resident entries require their native owner tuple")
        if hit[0] is values:
            _ = _prepared_info(hit[1],context)
            if py_type(hit[2]) is not b["int"]:
                raise Error("TypeError: native resident entries require an integer store id")
            return hit
        raise Error("TypeError: resident identity entry does not retain its original input")
    var source = values
    if py_type(values) is not context["array_class"] and py_type(values) is not py_get(context,"ndarray"):
        # A foreign container (pandas, Arrow-backed pandas, list, tuple,
        # range, memoryview, array.array, torch CPU tensor) is materialized
        # once by the dtype-preserving model-selection input owner: native
        # admission, no __array__/DLPack/iteration hooks. Array and exact
        # ndarray stay on the zero-copy preparation below.
        var inputs = py_get(context,"arena_msel_input_context")
        if inputs is None:
            raise Error("RuntimeError: native resident rows require the model-selection input connection")
        source = msel_input(values,"fold data",inputs)
    var array = _prepare(source,context)
    if indices and String(_slot(array,"dtype",context)) != "<i8":
        array = _cast(array,"<i8",context)
    var sid = -1
    var prepared = _prepared_info(array,context)
    var nbytes = Int(py=prepared["nbytes"])
    if nbytes:
        sid = Int(py=_native_call(context["arena_msel_put"],Python.tuple(prepared["address"],nbytes)))
    try:
        var result = Python.tuple(values,array,sid)
        py_setitem(entries,key,result)
        return result
    except error:
        if sid >= 0:
            # Publication failed after allocation. Free the unpublished id,
            # preserving the primary failure if the backend also refuses
            # cleanup; do not overwrite the shared pending-exception receipt.
            try:
                _ = py_call(context["arena_msel_free"],sid)
            except cleanup_error:
                pass
        raise error^


def arena_resident_take(fields: PythonObject, values: PythonObject, indices_value: PythonObject,
                        context: PythonObject) raises -> PythonObject:
    # _Resident.take: rows `indices_value` of `values`, gathered on the device
    # out of the resident store entries (each put once per owner).
    var entries = fields["_entries"]
    var b = context["arena_native_types"]
    if py_type(entries) is not b["dict"]:
        raise Error("TypeError: resident entries require a native dictionary")
    var keys = py_call(b["list"],entries)
    for i in range(len(keys)):  # driver-loop: identity-keyed cache lookups must not invoke application equality
        if py_type(keys[i]) is not b["int"]:
            raise Error("TypeError: resident entries require native identity keys")
    var base_entry = _entry(values,entries,False,context)
    var index_entry = _entry(indices_value,entries,True,context)
    var array = _prepared_info(base_entry[1],context)
    var indices = _prepared_info(index_entry[1],context)
    var shape = Python.list(indices["size"])
    var base_shape = array["shape"]
    for i in range(1,len(base_shape)):  # driver-loop: generic row shape preserves all trailing dimensions
        _ = py_call(shape.append,base_shape[i])
    var output = _empty(py_call(b["tuple"],shape),String(array["dtype"]),context)
    var n = Int(py=base_shape[0])
    var row_bytes = Int(py=array["nbytes"])//n if n else 0
    if Int(py=indices["size"]) and row_bytes:
        _ = _native_call(context["arena_msel_take_rows"],Python.tuple(base_entry[2],n,row_bytes,index_entry[2],indices["size"],_slot(output,"_addr",context)))
    return output


def arena_resident_entry(fields: PythonObject, values: PythonObject, indices: Bool,
                         context: PythonObject) raises -> PythonObject:
    # _Resident._entry: the (owner, prepared Array, store id) entry of
    # `values`, put once; `indices` casts to the int64 index layout.
    var entries = fields["_entries"]
    var b = context["arena_native_types"]
    if py_type(entries) is not b["dict"]:
        raise Error("TypeError: resident entries require a native dictionary")
    var keys = py_call(b["list"],entries)
    for i in range(len(keys)):  # driver-loop: identity-keyed cache lookups must not invoke application equality
        if py_type(keys[i]) is not b["int"]:
            raise Error("TypeError: resident entries require native identity keys")
    return _entry(values,entries,indices,context)


def arena_materialize(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    var cached = fields["_host"]
    if cached is not None:
        return cached
    var take = fields["_take"]
    if py_type(take) is not context["owner_method_class"] or take.__func__ is not context["arena_resident_take"]:
        raise Error("TypeError: DeviceRows.materialize requires model_selection._Resident.take; Python gather callbacks cannot execute")
    var owner = take.__self__
    var output = arena_resident_take(_resident_fields(owner,context),fields["base"],fields["indices"],context)
    py_setitem(fields,"_host",output)
    return output
