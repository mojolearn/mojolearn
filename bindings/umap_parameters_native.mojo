# SPDX-License-Identifier: Apache-2.0
# UMAP constructor/fit validation over admitted native instance dictionaries.
# Shared by the constructor/reset owners and manifold execution, so preserving
# constructor checks does not require invoking UMAP._parameters or _extras.
from std.memory import bitcast
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains
from bindings.native_loaded_module import native_loaded_module
from bindings.native_instance_item import native_instance_item
from bindings.classical_attributes import classical_connect_numpy
from bindings.sequential_admission import _one_of
from providers.python.array_metadata_pyapi import ArrayMetadata, array_shape_once, array_metadata_once
from core.cast_in import CAST_KIND_SIGNED, CAST_KIND_UNSIGNED, CAST_KIND_FLOAT, cast_w64_bits

comptime UMAP_F32_MAX = 3.4028234663852886e+38
comptime UMAP_I64_MAX = 9223372036854775807


def _field(fields: PythonObject, name: String) raises -> PythonObject:
    return native_instance_item(fields, PythonObject(name))


def _scalar_admitted(value: PythonObject, context: PythonObject) raises -> Bool:
    var b = native_loaded_module("builtins")
    var cls = py_type(value)
    if cls is b.int or cls is b.float or cls is b.bool or cls is b.str:
        return True
    classical_connect_numpy(context)
    return _one_of(cls, context["scalar_types"])


def _scalar_array(value: PythonObject, name: String, context: PythonObject) raises -> ArrayMetadata:
    var shape = array_shape_once(value, context)
    if len(shape) != 0:
        raise Error("TypeError: UMAP " + name + " must be a scalar")
    return array_metadata_once(value, shape^, name, context, True)


def _scalar_word(value: ArrayMetadata) -> UInt64:
    var width = value.source.itemsize
    var pointer = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=value.source.addr)
    var raw = UInt64(0)
    for i in range(width):  # small-loop(itemsize: supported numeric scalar byte width): at most eight bytes
        var at = i if value.native_byte_order else width - i - 1
        raw |= UInt64(pointer[at]) << UInt64(8 * i)
    return cast_w64_bits(raw, value.source.kind, width)


def _parameter_array(value: PythonObject, name: String, integer: Bool,
                     context: PythonObject) raises -> ArrayMetadata:
    # The former constructor translated index/float failures to its own
    # ValueError. Preserve that public error for exact ndarrays without ever
    # invoking object elements' conversion protocols.
    try:
        return _scalar_array(value, name, context)
    except:
        var detail = " must be an integer" if integer else " must be a finite float32 value"
        raise Error("ValueError: UMAP " + name + detail)


def umap_integer(value: PythonObject, name: String, minimum: Int, maximum: Int,
                 context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    classical_connect_numpy(context)
    var cls = py_type(value)
    # Exact integer identities admit the native NumPy scalar variants while
    # refusing user __index__/__int__ and bool, before their protocols run.
    if cls is py_get(context, "ndarray"):
        var scalar = _parameter_array(value, name, True, context)
        var word = _scalar_word(scalar)
        var signed = scalar.source.kind == CAST_KIND_SIGNED
        if not signed and scalar.source.kind != CAST_KIND_UNSIGNED:
            raise Error("ValueError: UMAP " + name + " must be an integer")
        if not signed and word > UInt64(UMAP_I64_MAX):
            raise Error("ValueError: UMAP " + name + " must be in [" + String(minimum) + ", " + String(maximum) + "]")
        var integer = Int(bitcast[DType.int64](word))
        if integer < minimum or integer > maximum:
            raise Error("ValueError: UMAP " + name + " must be in [" + String(minimum) + ", " + String(maximum) + "]")
        _ = scalar
        return PythonObject(integer)
    if cls is b.bool or not _one_of(cls, context["integer_types"]):
        raise Error("ValueError: UMAP " + name + " must be an integer")
    var result: Int
    try:
        result = Int(py=value)
    except:
        raise Error("ValueError: UMAP " + name + " must be in [" + String(minimum) + ", " + String(maximum) + "]")
    if result < minimum or result > maximum:
        raise Error("ValueError: UMAP " + name + " must be in [" + String(minimum) + ", " + String(maximum) + "]")
    return PythonObject(result)


def umap_scalar(value: PythonObject, name: String, context: PythonObject) raises -> Float64:
    classical_connect_numpy(context)
    var result: Float64
    if py_type(value) is py_get(context, "ndarray"):
        var scalar = _parameter_array(value, name, False, context)
        var word = _scalar_word(scalar)
        if scalar.source.kind == CAST_KIND_FLOAT:
            result = bitcast[DType.float64](word)
        elif scalar.source.kind == CAST_KIND_SIGNED:
            result = Float64(bitcast[DType.int64](word))
        else:
            result = Float64(word)
        _ = scalar
    else:
        if not _scalar_admitted(value, context):
            raise Error("ValueError: UMAP " + name + " must be a finite float32 value")
        try:
            # Scalar marshaling only, after exact native-type admission. No
            # Python-defined __float__ or string-subclass parser can run here.
            result = Float64(py=py_call(native_loaded_module("builtins").float, value))
        except:
            raise Error("ValueError: UMAP " + name + " must be a finite float32 value")
    if not (result - result == 0.0) or abs(result) > UMAP_F32_MAX:
        raise Error("ValueError: UMAP " + name + " must be a finite float32 value")
    return Float64(Float32(result))


def umap_is_string(value: PythonObject) raises -> Bool:
    ref cpy = Python().cpython()
    var cls = py_type(value)
    var flags = cpy.lib.get_function[UInt64]("PyType_GetFlags")(cls._obj_ptr)
    # Ordinary unicode and NumPy's static unicode scalar retain native string
    # storage. Heap subclasses can override comparison/format hooks.
    return (flags & UInt64(1 << 28)) != 0 and (flags & UInt64(1 << 9)) == 0


def _string(value: PythonObject, name: String) raises -> String:
    if not umap_is_string(value):
        raise Error("TypeError: UMAP " + name + " requires a native string; Python protocols are not supported")
    return String(value)


def _repr(value: PythonObject) raises -> String:
    return String(py_call(native_loaded_module("builtins").repr, value))


def umap_extras_fields(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var densmap = py_get(fields, "densmap", PythonObject(False))
    var kind = py_type(densmap)
    classical_connect_numpy(context)
    var dense_flag = False
    if kind is py_get(context, "ndarray"):
        var scalar = _scalar_array(densmap, "densmap", context)
        var word = _scalar_word(scalar)
        dense_flag = bitcast[DType.float64](word) != 0.0 if scalar.source.kind == CAST_KIND_FLOAT else word != 0
    elif (kind is not b.bool and kind is not b.int and kind is not b.float and kind is not b.str
            and kind is not b.list and kind is not b.tuple and kind is not b.dict
            and kind is not py_type(PythonObject(None)) and not _one_of(kind, context["scalar_types"])):
        raise Error("TypeError: UMAP densmap requires a native boolean or scalar")
    else:
        dense_flag = Bool(py=densmap)
    if dense_flag:
        raise Error("NotImplementedError: UMAP densmap=True is not implemented: the density-augmented "
                    + "objective (umap-learn's densMAP) is a different optimizer")
    var output = py_get(fields, "output_metric", PythonObject("euclidean"))
    var output_name = _string(output, "output_metric")
    if output_name != "euclidean" and output_name != "l2":
        raise Error("NotImplementedError: UMAP output_metric=" + _repr(output) + " is not implemented; "
                    + "the layout optimizer's gradient is the euclidean one")
    var metric = _field(fields, "metric")
    var name = _string(metric, "metric")
    var code = -2
    if name == "euclidean" or name == "l2":
        code = -1
    elif name == "sqeuclidean":
        code = 0
    elif name == "cosine":
        code = 2
    elif name == "manhattan" or name == "l1" or name == "cityblock" or name == "taxicab":
        code = 3
    elif name == "chebyshev" or name == "linf" or name == "linfinity" or name == "infinity":
        code = 7
    elif name == "minkowski":
        code = 9
    if code == -2:
        raise Error("ValueError: UMAP metric=" + _repr(metric) + " is not supported; the kNN computes "
                    + "['chebyshev', 'cityblock', 'cosine', 'euclidean', 'infinity', 'l1', 'l2', 'linf', 'linfinity', 'manhattan', 'minkowski', 'sqeuclidean', 'taxicab']"
                    + " (a callable, and every other name, is refused by name)")
    var kw = py_get(fields, "metric_kwds")
    var p = Float64(2.0)
    if kw is not None:
        if py_type(kw) is not b.dict:
            raise Error("TypeError: UMAP metric_kwds requires an exact native dictionary")
        var names = py_call(b.list, kw)
        for i in range(len(names)):  # driver-loop: native configuration key admission before hashing/lookup
            if py_type(names[i]) is not b.str:
                raise Error("TypeError: UMAP metric_kwds requires exact string keys")
        if code == 9:
            p = umap_scalar(py_get(kw, "p", PythonObject(2.0)), "metric_kwds['p']", context)
            if not p > 0:
                raise Error("ValueError: UMAP minkowski p must be positive")
        var unused = Python.list()
        for i in range(len(names)):  # driver-loop: names of unsupported native configuration fields
            if code != 9 or String(names[i]) != "p":
                _ = py_call(unused.append, names[i])
        if len(unused) > 0:
            # Error metadata contains exact strings only; sorting invokes no
            # user comparison protocol and performs no numerical operation.
            raise Error("ValueError: UMAP metric_kwds " + _repr(py_call(b.sorted, unused))
                        + " are not used by metric=" + _repr(metric))
    var a_param = py_get(fields, "a")
    var b_param = py_get(fields, "b")
    if (a_param is None) != (b_param is None):
        raise Error("ValueError: UMAP a and b must be given together")
    var a = Float64(0.0)
    var bb = Float64(0.0)
    if a_param is not None:
        a = umap_scalar(a_param, "a", context)
        bb = umap_scalar(b_param, "b", context)
        if not (a > 0 and bb > 0):
            raise Error("ValueError: UMAP a and b must be positive")
    var init = _field(fields, "init")
    if umap_is_string(init):
        var text = String(init)
        if text != "spectral" and text != "random" and text != "pca":
            raise Error("ValueError: UMAP init=" + _repr(init) + " must be 'spectral', 'random', 'pca' or an array")
    return Python.tuple(PythonObject(code), PythonObject(p), PythonObject(a), PythonObject(bb))


def umap_parameters_fields(fields: PythonObject, context: PythonObject) raises -> Tuple[PythonObject, PythonObject]:
    var neighbors = umap_integer(_field(fields, "n_neighbors"), "n_neighbors", 2, UMAP_I64_MAX, context)
    var components = umap_integer(_field(fields, "n_components"), "n_components", 1, 32, context)
    var epochs_param = _field(fields, "n_epochs")
    var epochs = PythonObject(0)
    if epochs_param is not None:
        epochs = umap_integer(epochs_param, "n_epochs", 1, UMAP_I64_MAX, context)
    var seed = umap_integer(_field(fields, "random_state"), "random_state", 0, UMAP_I64_MAX, context)
    var min_dist = umap_scalar(_field(fields, "min_dist"), "min_dist", context)
    var spread = umap_scalar(_field(fields, "spread"), "spread", context)
    var mix = umap_scalar(_field(fields, "set_op_mix_ratio"), "set_op_mix_ratio", context)
    var connectivity = umap_scalar(_field(fields, "local_connectivity"), "local_connectivity", context)
    var rate = umap_scalar(_field(fields, "learning_rate"), "learning_rate", context)
    var repulsion = umap_scalar(_field(fields, "repulsion_strength"), "repulsion_strength", context)
    var negatives = umap_integer(_field(fields, "negative_sample_rate"), "negative_sample_rate", 0, (1 << 31) - 1, context)
    if rate <= 0:
        raise Error("ValueError: UMAP learning_rate must be positive")
    if repulsion < 0:
        raise Error("ValueError: UMAP repulsion_strength must be nonnegative")
    if not (min_dist >= 0 and min_dist <= spread) or spread <= 0:
        raise Error("ValueError: UMAP requires 0 <= min_dist <= spread and spread > 0")
    if not (mix >= 0 and mix <= 1):
        raise Error("ValueError: UMAP set_op_mix_ratio must be in [0, 1]")
    if connectivity < 0:
        raise Error("ValueError: UMAP local_connectivity must be >= 0")
    var extras = umap_extras_fields(fields, context)
    var config = Python.list(neighbors, components, epochs, PythonObject(min_dist), PythonObject(spread), PythonObject(mix),
                             PythonObject(connectivity), seed, PythonObject(rate), PythonObject(repulsion), negatives)
    return (config, extras)
