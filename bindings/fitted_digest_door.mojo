# SPDX-License-Identifier: Apache-2.0
# `model.digest_`: the hash of a fitted model's state, one native call
# (`classical_native_call(model, "fitted_digest", (), context)`, the context
# of python/mojolearn/_native_archive_bridge.py `connect()`).
#
# NO CACHE. Nothing here runs at fit and nothing is kept: EVERY read of
# `digest_` describes the fitted fields (UNPACK), hashes them with the GIL
# released across the host cores (RUN, core/fitted_digest.mojo) and returns
# the hex string (PACK). A kept string would be a wrong answer after an
# in-place write into a fitted array (`model.coef_[0] = 1` changes no
# generation word), so the digest is always the hash of the state as it is at
# the read.
#
# WHICH FIELDS (one walk, the saved-model writer's describers; the table
# docs/api/DIGEST_FIELDS.md is generated from the same three sources):
#   1. A class with a connected save schema (`archive_save_schemas`, the
#      schema `save` writes): the schema's members, built by `_schema_run`
#      and described by `_archive_unpack_member`, exactly as `save` does
#      before it writes. Nothing is written. A class may declare a digest
#      schema (`digest_schemas`) that is run in its place: a decision tree's
#      is its save schema without the refusal of a splitter='random' tree,
#      whose fitted arrays need no file to be hashed.
#   2. Any other class: its public fitted attributes, the rule `is_fitted`
#      reads (instance fields whose name ends in "_" and does not start with
#      "_", value not None, name not in `digest_skip`: the few public names
#      that record how a call ran, not what was fitted), plus the private
#      fields its class names in `digest_extra`
#      (python/mojolearn/_digest_fields.py), each described by
#      the npz codec's `_npz_value` (dtype, shape, C-order bytes).
#      A list, tuple or dict that is not one flat run of scalars is entered:
#      `name[i]`, `name{key}` and `name.len`. An integer wider than 64 bits
#      is hashed as its decimal text under `name.bigint`.
#      A layer object (a class under `digest_layers`: the CNN layers) is
#      entered: its class name, then its own public learned fields. An
#      unfitted estimator kept beside fitted members gives its class name. A member estimator contributes
#      its own digest, as text, in member order.
#   3. A class named in `digest_refused` has no digest yet: the read raises
#      AttributeError with the reason in words.
# A field holding a value with no canonical bytes (an object that is not an
# array, a scalar, text, a container of those or an estimator) is refused by
# name: never skipped, never hashed by address or repr.
#
# ONE MODEL, ONE DIGEST. Nothing of the route, the device or the binary is
# hashed: the class name is the public estimator's (a CPU twin is hashed under
# the name of the class it loads), fields go in name order, a device-resident
# array is described by its dtype, shape and bytes exactly as a host array is,
# and the public names that record how a call ran are skipped (`digest_skip`).
#
# Refusals are AttributeError, in words: an unfitted model (like any fitted
# attribute), a model in `fast` mode (fast promises no identical bits), and
# the two above. `hasattr(model, "digest_")` is then False.
#
# No mutual recursion (AGENTS.md part 6): `fitted_digest_text` calls only
# itself, for a member estimator; containers are walked with an explicit
# stack.
from std.builtin.sort import sort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.py_cpython_table import cpython_table
from bindings.py_balanced import py_call, py_type, py_get, py_item, py_contains
from bindings.native_loaded_module_door import native_loaded_module
from bindings.optimizer_public_words_unpack import PYTHON_ERROR_PENDING
from bindings.mode_metadata_native_door import _fields, _fitted
from bindings.classical_arrays_door import classical_is_array
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.classical_archive_door import _archive_unpack_member, _archive_class_name_unpack
from bindings.archive_npz_door import _npz_value, _npz_is
from bindings.archive_schema_native_door import archive_schema_lookup, _schema_run, _schema_gp_state, _schema_mode
from providers.python.classical_family_admission_pyapi_unpack import classical_family_admit
from core.native_archive_write import NativeArchiveMember
from core.fitted_digest import fitted_digest_hex

#: Member estimators inside member estimators, and containers inside containers.
comptime _DIGEST_DEPTH = 8


def _digest_unfitted(name: String) -> Error:
    return Error("AttributeError: '" + name + "' object has no attribute 'digest_': the model is not fitted yet")


def _digest_scalar(value: PythonObject, b: PythonObject, bytes_type: PythonObject) raises -> Int:
    """1 for a str, 2 for bytes, 3 for a bool / int / float, 0 otherwise
    (PyType_IsSubtype on the value's type: no Python hook)."""
    if _npz_is(value, b.str):
        return 1
    if _npz_is(value, bytes_type):
        return 2
    if _npz_is(value, b.bool) or _npz_is(value, b.int) or _npz_is(value, b.float):
        return 3
    return 0


def _digest_flat(value: PythonObject, is_list: Bool, b: PythonObject, bytes_type: PythonObject) raises -> Bool:
    """True when an exact list / tuple is one flat run of scalars of one kind
    (or empty): the npz codec describes it as one member."""
    var t = cpython_table()
    var n = t[].PyList_Size(value._obj_ptr) if is_list else t[].PyTuple_Size(value._obj_ptr)
    var kind = 0
    for i in range(n):  # driver-loop: over one stored metadata sequence's items (type words only)
        var item = value[i]
        var found = _digest_scalar(item, b, bytes_type)
        if found == 0 or (kind != 0 and found != kind):
            return False
        kind = found
    return True


def fitted_digest_text(model: PythonObject, context: PythonObject, depth: Int) raises -> String:
    """The digest of the fitted `model` (64 lowercase hex digits), hashed
    from its state as it is now (nothing is kept between reads)."""
    if depth > _DIGEST_DEPTH:
        raise Error("AttributeError: digest_ is not available: member estimators are nested deeper than "
                    + String(_DIGEST_DEPTH) + " levels")
    var b = native_loaded_module("builtins")
    var cls = py_type(model)
    var class_name = _archive_class_name_unpack(cls)
    # The name that is hashed: the public estimator's. A CPU twin (a `Host*`
    # class, python/mojolearn/_digest_fields.py CANONICAL) is hashed under the
    # name of the class whose file it loads, in the hash head and in the save
    # schema's `estimator` member, so a model and its loaded copy have one
    # digest on any device. `class_name` stays the real class for refusals.
    var hashed_name = class_name
    var canonical = py_get(context, "digest_canonical")
    if canonical is not None and py_contains(canonical, cls):
        var named = py_item(canonical, cls)
        if py_type(named) is py_type(PythonObject("")):
            hashed_name = String(py=named)
        else:
            # A twin of several public classes (HostForest, HostGBDT;
            # `_digest_fields.twin_field`): a one-element tuple naming the
            # instance field that holds the public class name the file was
            # written by (`estimator`).
            var held = py_get(_fields(model), named[0])
            if held is None or py_type(held) is not py_type(PythonObject("")):
                raise _digest_unfitted(class_name)
            hashed_name = String(py=held)
    var refused = py_get(context, "digest_refused")
    if refused is not None:
        var reason = archive_schema_lookup(cls, refused)
        if reason is not None:
            raise Error("AttributeError: " + class_name + " has no digest_ yet: " + String(py=reason))
    # A fitted field may be an ndarray when the caller loaded NumPy.
    classical_connect_numpy(context)
    # The class's digest schema when it declares one (a tree: its fitted
    # arrays without the save refusal), else the schema `save` writes.
    var steps = PythonObject(None)
    var own_schemas = py_get(context, "digest_schemas")
    if own_schemas is not None:
        steps = archive_schema_lookup(cls, own_schemas)
    if steps is None:
        steps = archive_schema_lookup(cls, context["archive_save_schemas"])
    var schema = steps is not None and py_type(steps) is not b.str
    var pins = List[PythonObject]()
    var members = List[NativeArchiveMember]()
    if schema:
        # 1. The saved-model schema's members, as `save` builds them.
        var values = Python.dict()
        var failure = String("")
        try:
            var admitted = classical_family_admit(model)
            var fields = py_call(b.dict, admitted)
            if String(py=_schema_mode(fields, PythonObject("mojolearn: "), context)) == "fast":
                failure = "fast"
            else:
                _schema_gp_state(fields, steps, context)
                _schema_run(model, fields, steps, values, hashed_name, context)
        except problem:
            failure = String(problem)
            if failure == PYTHON_ERROR_PENDING:
                raise Error(failure)
            if failure.byte_length() == 0:
                failure = "the saved-model schema refused this model"
        if failure == "fast":
            raise Error("AttributeError: digest_ exists in identical mode only: fast mode promises no identical "
                        + "bits, so a fast model has no digest")
        if failure.byte_length() > 0:
            if not Bool(py=_fitted(model)):
                raise _digest_unfitted(class_name)
            raise Error("AttributeError: " + class_name + ".digest_ is not available for this model: " + failure)
        var keys = py_call(b.list, values.keys())
        var names = List[String]()
        for i in range(len(keys)):  # driver-loop: over the schema's member names (model fields)
            names.append(String(py=keys[i]))
        sort(names)
        for name in names:  # driver-loop: over the schema's members (model fields), described once
            members.append(_archive_unpack_member(name, values[PythonObject(name)], context, pins))
        pins.append(values)
    else:
        # 2. The public fitted attributes (the `is_fitted` rule) and the
        #    class's declared private fitted fields.
        if not Bool(py=_fitted(model)):
            raise _digest_unfitted(class_name)
        var fields = _fields(model)
        if String(py=_schema_mode(fields, PythonObject("mojolearn: "), context)) == "fast":
            raise Error("AttributeError: digest_ exists in identical mode only: fast mode promises no identical "
                        + "bits, so a fast model has no digest")
        var bytes_type = context["archive_bytes_type"]
        var ndarray = py_get(context, "ndarray")
        var numpy_scalar = py_get(context, "scalar_type")
        var base = py_get(context, "digest_base")
        var layers = py_get(context, "digest_layers")
        # A classifier's `classes_` is a `ClassLabels` (python/mojolearn/
        # _labels.py), a `list` subclass that only adds an integer-array
        # index. Its items are the class list: hashed as that list (copied
        # once through list's own iterator, no Python hook), so a model whose
        # classes_ is a plain list (a loaded copy) has the same digest.
        var labels_class = py_get(context, "classifier_labels_class")
        var work_names = List[String]()
        var work_values = List[PythonObject]()
        var work_depths = List[Int]()
        var skip = py_get(context, "digest_skip")
        var keys = py_call(b.list, fields.keys())
        for i in range(len(keys)):  # driver-loop: over the model's stored field names
            var name = String(py=keys[i])
            if name.endswith("_") and not name.startswith("_"):
                # A record of how a call ran (was the input copied), not fitted state.
                if skip is not None and py_contains(skip, keys[i]):
                    continue
                var value = fields[keys[i]]
                if value is not None:
                    work_names.append(name)
                    work_values.append(value)
                    work_depths.append(0)
        var extra = py_get(context, "digest_extra")
        if extra is not None:
            var declared = archive_schema_lookup(cls, extra)
            if declared is not None:
                for i in range(len(declared)):  # driver-loop: over the class's declared private fitted fields
                    if py_contains(fields, declared[i]):
                        var value = fields[declared[i]]
                        if value is not None:
                            work_names.append(String(py=declared[i]))
                            work_values.append(value)
                            work_depths.append(0)
        var leaf_names = List[String]()
        var leaf_values = List[PythonObject]()
        while len(work_names) > 0:  # driver-loop: over the model's fitted fields and their containers (structure only)
            var name = work_names.pop()
            var value = work_values.pop()
            var level = work_depths.pop()
            if value is None:
                # None inside a container keeps its place.
                leaf_names.append(name + ".none")
                leaf_values.append(PythonObject(True))
                continue
            var kind = py_type(value)
            if labels_class is not None and kind is labels_class:
                value = py_call(b.list, value)
                kind = py_type(value)
            if kind is b.int and Int(py=py_call(value.bit_length)) > 63:
                # A stored integer wider than a signed 64-bit word (a 64-bit
                # generator state) has no 8-byte form: its decimal text.
                # `bit_length` and `str` of an exact int run no Python hook.
                leaf_names.append(name + ".bigint")
                leaf_values.append(py_call(b.str, value))
                continue
            var leaf = classical_is_array(value, context) or _digest_scalar(value, b, bytes_type) != 0
            if not leaf and ndarray is not None and kind is ndarray:
                leaf = True
            if not leaf and numpy_scalar is not None and _npz_is(value, numpy_scalar):
                leaf = True
            var is_list = kind is b.list
            if not leaf and (is_list or kind is b.tuple):
                leaf = _digest_flat(value, is_list, b, bytes_type)
                if not leaf:
                    if level >= _DIGEST_DEPTH:
                        raise Error("AttributeError: " + class_name + ".digest_ is not available: field '" + name
                                    + "' nests containers deeper than " + String(_DIGEST_DEPTH) + " levels")
                    var n = len(value)
                    leaf_names.append(name + ".len")
                    leaf_values.append(PythonObject(n))
                    for i in range(n):  # driver-loop: over one stored sequence's items (member estimators, per-layer arrays)
                        work_names.append(name + "[" + String(i) + "]")
                        work_values.append(value[i])
                        work_depths.append(level + 1)
                    continue
            if leaf:
                leaf_names.append(name)
                leaf_values.append(value)
                continue
            if kind is b.dict:
                if level >= _DIGEST_DEPTH:
                    raise Error("AttributeError: " + class_name + ".digest_ is not available: field '" + name
                                + "' nests containers deeper than " + String(_DIGEST_DEPTH) + " levels")
                # items() of an exact dict never hashes or compares keys.
                var items = py_call(b.list, py_call(value.items))
                leaf_names.append(name + ".len")
                leaf_values.append(PythonObject(len(items)))
                for i in range(len(items)):  # driver-loop: over one stored mapping's entries
                    var key = items[i][0]
                    if py_type(key) is not b.str:
                        raise Error("AttributeError: " + class_name + ".digest_ is not available: field '" + name
                                    + "' is a dictionary with keys that are not text")
                    work_names.append(name + "{" + String(py=key) + "}")
                    work_values.append(items[i][1])
                    work_depths.append(level + 1)
                continue
            if base is not None and _npz_is(value, base):
                if not Bool(py=_fitted(value)):
                    # An unfitted template a fitted model keeps beside its
                    # members (the base estimator of a batched bagging fit):
                    # it has no fitted state; its class keeps its place.
                    leaf_names.append(name + ".unfitted")
                    leaf_values.append(PythonObject(_archive_class_name_unpack(kind)))
                    continue
                # A member estimator: its own digest, as text, under its place.
                var inner = fitted_digest_text(value, context, depth + 1)
                leaf_names.append(name)
                leaf_values.append(PythonObject(inner))
                continue
            var is_layer = False
            if layers is not None:
                for i in range(len(layers)):  # driver-loop: over the declared layer base classes (one or two)
                    if _npz_is(value, layers[i]):
                        is_layer = True
            if is_layer:
                # A layer object of a layered model: its class, then its own
                # learned fields by the same public rule (`weight_`, `bias_`,
                # running statistics), each under `name.<field>`.
                if level >= _DIGEST_DEPTH:
                    raise Error("AttributeError: " + class_name + ".digest_ is not available: field '" + name
                                + "' nests layers deeper than " + String(_DIGEST_DEPTH) + " levels")
                leaf_names.append(name + ".layer")
                leaf_values.append(PythonObject(_archive_class_name_unpack(kind)))
                var layer_fields = _fields(value)
                var layer_keys = py_call(b.list, layer_fields.keys())
                for i in range(len(layer_keys)):  # driver-loop: over one layer's stored field names
                    var field = String(py=layer_keys[i])
                    if field.endswith("_") and not field.startswith("_"):
                        if skip is not None and py_contains(skip, layer_keys[i]):
                            continue
                        var held = layer_fields[layer_keys[i]]
                        if held is not None:
                            work_names.append(name + "." + field)
                            work_values.append(held)
                            work_depths.append(level + 1)
                continue
            raise Error("AttributeError: " + class_name + ".digest_ is not available yet: field '" + name
                        + "' holds a " + _archive_class_name_unpack(kind) + ", which has no canonical bytes")
        # Name order, whatever order the dictionary or the walk gave.
        var place = Dict[String, Int]()
        for i in range(len(leaf_names)):  # driver-loop: over the described fields (name index)
            place[leaf_names[i]] = i
        var names = leaf_names.copy()
        sort(names)
        for name in names:  # driver-loop: over the model's fitted fields, described once (no bytes copied)
            var described = _npz_value(name, leaf_values[place[name]], True, context, pins)
            members.append(described^.take())
        for i in range(len(leaf_values)):  # driver-loop: keep every described value alive to the end of the hash
            pins.append(leaf_values[i])
    # RUN: typed members only, the interpreter lock released.
    var result = String("")
    try:
        with GILReleased(Python()):
            result = fitted_digest_hex(hashed_name, members)
    finally:
        _ = pins
    _ = members
    return result


def fitted_digest_call(model: PythonObject, context: PythonObject) raises -> PythonObject:
    """`model.digest_`: see the module note."""
    var text = fitted_digest_text(model, context, 0)
    return PythonObject(text)
