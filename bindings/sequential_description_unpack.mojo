# SPDX-License-Identifier: Apache-2.0
"""Native Sequential model-description and option metadata."""
from bindings.native_loaded_module_door import native_loaded_module, NativeModules
from std.python import Python, PythonObject
from bindings.sequential_admission_door import sequential_integer, sequential_admit_value
from bindings.py_balanced import py_call, py_contains, py_fields, py_setitem, py_type
from bindings.native_identity_key_door import NativeIdentityKey
from bindings.native_prepared_owner_unpack import prepared_get, prepared_put


def sequential_fields(value: PythonObject) raises -> PythonObject:
    return py_fields(value)


def _integer(value: PythonObject, context: PythonObject) raises -> Bool:
    return sequential_integer(value, context)


def _positive(value: PythonObject, name: String, low: Int, context: PythonObject) raises -> Int:
    if not _integer(value, context) or Bool(py=value < PythonObject(low)):
        raise Error(name + " must be an integer >= " + String(low))
    return Int(py=value)


def sequential_append(mut mods: NativeModules, model: PythonObject, code: PythonObject, a0: PythonObject,
                      a1: PythonObject, f: PythonObject, supplied: PythonObject) raises:
    var b = mods.module("builtins")
    var args = Python.list(a0, a1, 0, 0, 0, 0) if supplied is None else py_call(b.list, supplied)
    var fields = sequential_fields(model)
    _ = py_call(fields["_layers"].append, Python.tuple(code, a0, a1, f, py_call(b.tuple, args)))


struct SequentialDescription(Movable, Writable):
    # The execution ABI owns native words. Retain them across calls rather than
    # rebuilding Python lists and immediately copying their elements back into
    # Mojo. The public description methods keep their existing Python results.
    var identities: NativeIdentityKey
    var input_width: Int
    var cacheable: Bool
    var ranked: Bool
    var sample_dims: List[Int]
    var int_type: PythonObject
    var float_type: PythonObject
    var bool_type: PythonObject
    var counts_ready: Bool
    var parameter_count: Int
    var output_count: Int
    var program: List[Int]
    var scalars: List[Float64]

    def __init__(out self, mut mods: NativeModules, layers: PythonObject,
                 width: Int, sample: PythonObject) raises:
        self.identities = NativeIdentityKey(mods)
        self.cacheable = True
        try:
            self.identities.add(layers)
        except:
            # Uncacheable extra metadata must not change ordinary execution.
            self.cacheable = False
        var b = mods.module("builtins")
        self.int_type = b.int
        self.float_type = b.float
        self.bool_type = b.bool
        for i in range(len(self.identities.objects)):  # driver-loop: over retained metadata identities to establish immutable scalar ownership
            var value = self.identities.objects[i]
            var cls = py_type(value)
            if value is not None and cls is not self.identities.list_type and cls is not self.identities.tuple_type and cls is not self.int_type and cls is not self.float_type and cls is not self.bool_type:
                # Writable native buffers can convert to scalars while retaining
                # their identity. Preserve that behavior by rebuilding for those
                # descriptions, rather than caching their converted contents.
                self.cacheable = False
                break
        self.input_width = width
        self.ranked = sample is not None
        self.sample_dims = List[Int]()
        self.counts_ready = False
        self.parameter_count = 0
        self.output_count = 0
        self.program = List[Int]()
        self.scalars = List[Float64]()
        var count = len(layers)
        if sample is None:
            self.program.append(1)
            self.program.append(width)
            self.program.append(count)
        else:
            var cls = py_type(sample)
            if cls is not self.identities.list_type and cls is not self.identities.tuple_type:
                self.cacheable = False
            var rank = len(sample)
            self.program.append(2)
            self.program.append(rank)
            for k in range(rank):  # driver-loop: over sample dimensions captured as native ABI words
                var value = sample[k]
                var value_type = py_type(value)
                if value_type is not self.int_type and value_type is not self.float_type and value_type is not self.bool_type:
                    self.cacheable = False
                var dim = Int(py=value)
                self.sample_dims.append(dim)
                self.program.append(dim)
            for k in range(rank, 8):  # driver-loop: over unused native sample dimension slots
                self.program.append(1)
            self.program.append(count)
        for k in range(count):  # driver-loop: over admitted layers once per changed description
            var layer = layers[k]
            self.program.append(Int(py=layer[0]))
            if sample is None:
                self.program.append(Int(py=layer[1]))
                self.program.append(Int(py=layer[2]))
            else:
                var args = layer[4]
                for j in range(len(args)):  # driver-loop: over rank-aware native layer arguments
                    self.program.append(Int(py=args[j]))
            self.program.append(0)
            self.scalars.append(Float64(py=layer[3]))

    def matches_sample(self, sample: PythonObject) raises -> Bool:
        # Fit infers a fresh tuple even when the sample schema did not change.
        # Compare the native ABI dimensions, not that temporary tuple's identity.
        # Exact containers and immutable scalar types retain the existing buffer
        # opt-out; mutable scalar conversions must rebuild on every operation.
        if sample is None:
            return not self.ranked
        if not self.ranked:
            return False
        var cls = py_type(sample)
        if cls is not self.identities.list_type and cls is not self.identities.tuple_type:
            return False
        if len(sample) != len(self.sample_dims):
            return False
        for k in range(len(self.sample_dims)):  # driver-loop: over retained native sample dimensions
            var value = sample[k]
            var value_type = py_type(value)
            if value_type is not self.int_type and value_type is not self.float_type and value_type is not self.bool_type:
                return False
            if Int(py=value) != self.sample_dims[k]:
                return False
        return True

    def write_to(self, mut writer: Some[Writer]):
        writer.write("SequentialDescription(native)")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)
