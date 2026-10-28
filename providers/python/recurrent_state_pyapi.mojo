# SPDX-License-Identifier: Apache-2.0
# Whole-operation recurrent state admission, native packing and finished output.
from std.memory import memcpy
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.classical_pandas import classical_admit_value
from bindings.xseq_native_common import xseq_input, xseq_f32, xseq_refuse_f64, xseq_zeros, xseq_f32p


def _state_fields(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var cls = py_type(model)
    if py_type(cls) is not b.type:
        raise Error("TypeError: recurrent state requires native class storage")
    var layouts = context["xseq_rnn_state_layouts"]
    for i in range(len(layouts)):  # small-loop(connected recurrent classes)
        var layout = layouts[i]
        if cls is not layout[0]:
            continue
        var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
        if mro is not layout[1]:
            raise Error("TypeError: recurrent state class ancestry changed")
        var hooks = layout[3]
        for j in range(len(hooks)):  # small-loop(pinned native storage hooks)
            var fields = py_call(b.type.__dict__["__dict__"].__get__, hooks[j][0])
            if py_get(fields, hooks[j][1]) is not hooks[j][2]:
                raise Error("TypeError: recurrent state refuses replaced Python storage hooks")
        var fields = py_call(layout[2], model, cls)
        var keys = py_call(b.list, fields)
        for j in range(len(keys)):  # small-loop(instance metadata keys)
            if py_type(keys[j]) is not b.str:
                raise Error("TypeError: recurrent state fields require exact string keys")
        return Python.tuple(fields, layout[4])
    raise Error("TypeError: recurrent state requires a connected native recurrent model")


def _size(fields: PythonObject, name: String, context: PythonObject) raises -> Int:
    var value = py_get(fields, name)
    if value is None:
        raise Error("AttributeError: recurrent model has no " + name + "; fit it before accessing state")
    classical_admit_value(value, context["xseq_lifecycle_context"])
    var result = Int(py=value)
    if result <= 0:
        raise Error(name + " must be positive")
    return result


def recurrent_state(model: PythonObject, operation: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var admitted = _state_fields(model, context)
    var fields = admitted[0]
    var cell = String(py=admitted[1])
    if cell == "rnn":
        var nonlinearity = py_get(fields, "nonlinearity")
        if py_type(nonlinearity) is not b.str:
            raise Error("TypeError: nonlinearity requires an exact string")
        var kind = String(py=nonlinearity)
        if kind != "tanh" and kind != "relu":
            raise Error("nonlinearity must be 'tanh' or 'relu' (nn.RNN)")
    var gates = 4 if cell == "lstm" else (3 if cell == "gru" else 1)
    var hidden = _size(fields, "hidden_size", context)
    var layers = _size(fields, "num_layers", context)
    var features = _size(fields, "n_features_in_", context)
    var outputs = _size(fields, "n_outputs_", context)
    var names = List[String]()
    var dimensions = List[List[Int]]()
    var counts = List[Int]()
    var total = 0
    for layer in range(layers):  # driver-loop: over recurrent parameter block layout
        var input_width = features if layer == 0 else hidden
        names.append("weight_ih_l" + String(layer))
        dimensions.append(List[Int](gates * hidden, input_width))
        names.append("weight_hh_l" + String(layer))
        dimensions.append(List[Int](gates * hidden, hidden))
        names.append("bias_ih_l" + String(layer))
        dimensions.append(List[Int](gates * hidden))
        names.append("bias_hh_l" + String(layer))
        dimensions.append(List[Int](gates * hidden))
    names.append("head.weight")
    dimensions.append(List[Int](outputs, hidden))
    names.append("head.bias")
    dimensions.append(List[Int](outputs))
    for i in range(len(names)):  # driver-loop: over named parameter buffers
        var count = 1
        for j in range(len(dimensions[i])):  # small-loop(parameter shape words)
            count *= dimensions[i][j]
        counts.append(count)
        total += count
    var loading = operation == "rnn_load_state_dict"
    var state = PythonObject(None)
    var packed = PythonObject(None)
    var result = Python.dict()
    if loading:
        state = args[0]
        # Do not invoke arbitrary mapping __getitem__ or tensor conversion
        # callbacks. Ordinary dictionaries retain the existing name contract.
        var mappings = context["xseq_rnn_state_mappings"]
        if py_type(state) is not mappings[0] and py_type(state) is not mappings[1]:
            raise Error("TypeError: recurrent state requires an exact dict or OrderedDict of native arrays")
        var keys = py_call(b.list, state)
        for i in range(len(keys)):  # small-loop(parameter-name metadata)
            if py_type(keys[i]) is not b.str:
                raise Error("TypeError: recurrent parameter names require exact strings")
        packed = xseq_zeros(Python.tuple(total), context)
    else:
        packed = xseq_f32(xseq_input(fields["params_"], context), "params_", False, context)
        if Int(py=packed.size) != total:
            raise Error("params_ size differs from the recurrent parameter layout")
    var base = xseq_f32p(packed)
    var offset = 0
    for i in range(len(names)):  # driver-loop: over named parameter buffers
        var shape = Python.list()
        for j in range(len(dimensions[i])):  # small-loop(parameter shape words)
            _ = py_call(shape.append, dimensions[i][j])
        var tuple_shape = py_call(b.tuple, shape)
        if loading:
            var source = xseq_input(state[PythonObject(names[i])], context)
            xseq_refuse_f64(source, names[i] + ": float64 is refused; pass float32 (no float64 on the device)")
            var actual_shape = source.shape
            if len(actual_shape) != len(dimensions[i]):
                raise Error(names[i] + ": shape differs from the recurrent parameter layout")
            for j in range(len(dimensions[i])):  # small-loop(admitted parameter shape words)
                if Int(py=actual_shape[j]) != dimensions[i][j]:
                    raise Error(names[i] + ": shape differs from the recurrent parameter layout")
            var input = xseq_f32(source, names[i], False, context)
            memcpy(dest=base + offset, src=xseq_f32p(input), count=counts[i])
            _ = input
        else:
            var output = xseq_zeros(tuple_shape, context)
            memcpy(dest=xseq_f32p(output), src=base + offset, count=counts[i])
            py_setitem(result, names[i], output)
        offset += counts[i]
    _ = packed
    if loading:
        # Commit only after every parameter has passed admission and shape
        # checks. A malformed later tensor never partially replaces the model.
        py_setitem(fields, "params_", packed)
        return model
    return result
