# SPDX-License-Identifier: Apache-2.0
"""`CNNClassifier.fit`, `predict` and `predict_proba`
(python/mojolearn/_expansion_cnn.py) as one native call each, through
`xcnn_native_call` (bindings/xcnn_native.mojo).

fit: the images admitted, the labels encoded (the sequence owner's label
sort, bindings/xseq_neural_native.mojo `xseq_labels`), the layer stack built
natively (Conv2d / ReLU / MaxPool2d blocks and the Linear head, each with the
constructor's fields and its seeded Mojo draws), then the whole training as
ONE call of the x_sequence whole-model door of the model's tier (CNN_FIT,
sequence/cnn_classifier_program.mojo): the layer stack captured as one
program on the program executor and trained by Program.fit; the trained
words come back into the layers' arrays. predict_proba / predict: one
CNN_PREDICT call, the same forward with the softmax (and argmax) in an
inference session, then classes_[codes] by the base gathers.

Lane x8-cnn-on-executor (2026-11-09): the hand epoch on the x_cnn resident
entries (x_cnn_fit_d / x_cnn_fit_epoch_d / x_cnn_fit_epoch_r, the Python
step-loop arm, the resident predict record and x_cnn_predict_r) is deleted.
TOMBSTONE: the hand CNN classifier epoch and resident predict deleted 2026-11-09 by lane/x8-cnn-on-executor; code recoverable at f59d913b0b.
Restore: git apply experiments/removed/X8_CNN_HAND_EPOCH.patch
"""
from std.math import sqrt
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields, py_contains
from bindings.xcnn_common import xcnn_builtins, xcnn_entries, xcnn_f32, xcnn_addr
from bindings.xseq_native_common import XseqStream, xseq_neural_provider
from bindings.neural_labels_native import neural_encode_labels, neural_decode_labels
from bindings.mode_metadata_native import _fields
from bindings.neural_block_admission import neural_plain_value
from bindings.native_loaded_module import NativeModules
from bindings.sequential_admission import sequential_admit_value
from bindings.sequential_inputs import sequential_input
from bindings.sequential_arrays import sequential_array
from bindings.xcnn_common import xcnn_shape, xcnn_tuple, xcnn_reshape, xcnn_numpy
from providers.python.borrowed_buffer_pyapi import AdmittedBorrowedBuffer, admit_native_buffer_exporter
from providers.python.neural_descriptor_pyapi import neural_descriptor_execute
from bindings.native_model_records import (
    RECORDS_PREPARED, ContentRecord, content_fields, content_object, records_stampable, records_stamped_get,
    records_stamped_put,
)
from sequence.neural_model_dispatch import CNN_FIT, CNN_PREDICT

#: rows per predict pass on the device (every op is per row, so the chunk
#: never changes a word)
comptime _PREDICT_ROWS = 2048


def _b() raises -> PythonObject:
    return xcnn_builtins()


def _empty(shape: PythonObject, dtype: String, context: PythonObject) raises -> PythonObject:
    xcnn_numpy(context)
    var dims = Python.tuple(shape) if py_type(shape) is _b().int else shape
    return sequential_array(dims, context[PythonObject(dtype)], context)


def _contiguous(source: PythonObject, context: PythonObject) raises -> PythonObject:
    xcnn_numpy(context)
    sequential_admit_value(source, context)
    var array = sequential_input(source, context)
    admit_native_buffer_exporter(array)
    var borrowed = AdmittedBorrowedBuffer(array)
    var layout = borrowed.layout.value().copy()
    if Int(py=array.dtype.num) != 5 and Int(py=array.dtype.num) != 11:
        raise Error("TypeError: CNN resident transfers require float32 or int32")
    if layout.contiguous():
        return py_call(context["ndarray"], xcnn_tuple(layout.shape), dtype=array.dtype, buffer=array)
    var out = sequential_array(xcnn_tuple(layout.shape), array.dtype, context)
    layout.copy_to(Int(py=xcnn_addr(out)))
    _ = array
    return out


# N135 (ne3-c-cnn, 2026-11-07): the admission record. A call on a classifier
# whose generation word (bindings/native_model_records.mojo: the watched
# instance dictionary) and content words (exact list items of `layers_` /
# `_blocks`, every held object's class, and each layer's own generation and
# field contents) are the ones the last full admission saw skips the walk
# below: 18 settings, 16 options and both learned buffers per layer. Any
# fit, set_params, setattr or layer field write changes a generation and
# misses; Python 3.10 / 3.11 (generation 0) always take the full walk. The
# payload is whether the fitted attributes were all present then. Not
# noticed, as in the other records: an ndarray header edited in place
# (`layer.weight_.shape = ...`); the shape reads downstream still check it.
comptime _ADMIT_STAMP = "xcnn_clf_admission_v1"


def _admission_stamp(model: PythonObject, fields: PythonObject, fitted_ok: Bool) raises:
    if not (py_fields(model) is fields) or not records_stampable(model):
        return
    var content = ContentRecord()
    content_fields(content, fields, 4)
    var layers = py_get(fields, "layers_")
    if layers is not None and py_type(layers) is _b().list:
        for i in range(len(layers)):  # driver-loop: over the classifier layers (content words, once per record)
            content_object(content, layers[i], 4)
    var head = py_get(fields, "head_")
    if head is not None:
        content_object(content, head, 4)
    var blocks = py_get(fields, "_blocks")
    if blocks is not None and py_type(blocks) is _b().list:
        for i in range(len(blocks)):  # driver-loop: over the block pairs (content words, once per record)
            content_object(content, blocks[i][0], 4)
            if blocks[i][1] is not None:
                content_object(content, blocks[i][1], 4)
    _ = records_stamped_put(RECORDS_PREPARED, model, _ADMIT_STAMP, content^, PythonObject(fitted_ok))


def _classifier_fields(model: PythonObject, context: PythonObject, fitted: Bool = False) raises -> PythonObject:
    var fields = _fields(model)
    var stamped = records_stamped_get(RECORDS_PREPARED, model, _ADMIT_STAMP)
    if stamped is not None and py_fields(model) is fields and (not fitted or Bool(py=stamped)):
        return fields
    var mods = NativeModules()   # one module view for every field and child layer below
    for name in ["input_shape", "conv_channels", "kernel_size", "pool_size", "learning_rate",  # driver-loop: over passive classifier settings
                 "momentum", "weight_decay", "batch_size", "max_iter", "shuffle", "random_state", "numeric_mode",
                 "optimizer", "dampening", "nesterov", "betas", "eps", "_flat"]:
        neural_plain_value(mods, py_get(fields, PythonObject(name)), 0, context)
    if fitted and (not py_contains(fields, "classes_") or not py_contains(fields, "layers_") or not py_contains(fields, "head_")):
        raise Error("RuntimeError: call fit before classifier inference")
    var classes = py_get(fields, "classes_")
    if classes is not None and py_type(classes) is not context["ndarray"]:
        raise Error("TypeError: CNN classes_ requires its native ndarray")
    var layers = py_get(fields, "layers_")
    if layers is not None:
        if py_type(layers) is not _b().list:
            raise Error("TypeError: CNN classifier layers require an exact native list")
        var owners = py_call(_b().list, layers)
        if py_contains(fields, "head_"):
            _ = py_call(owners.append, fields["head_"])
        for i in range(len(owners)):  # driver-loop: retained native classifier layer metadata
            var layer = owners[i]
            var cls = py_type(layer)
            if (cls is not context["xcnn_conv2d_class"] and cls is not context["xcnn_relu_class"]
                    and cls is not context["xcnn_maxpool_class"] and cls is not context["xcnn_linear_class"]):
                raise Error("TypeError: CNN classifier requires its connected native layer classes")
            var state = _fields(layer)
            for key in ["in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "groups",  # driver-loop: over compiled child options
                        "bias", "_conv_pad", "_pads", "_explicit", "ceil_mode", "count_include_pad", "divisor_override",
                        "in_features", "out_features"]:
                neural_plain_value(mods, py_get(state, PythonObject(key)), 0, context)
            for key in ["weight_", "bias_"]:  # driver-loop: native learned buffers before shape/address reads
                if py_contains(state, PythonObject(key)):
                    var buffer = state[PythonObject(key)]
                    sequential_admit_value(buffer, context)
                    if py_type(buffer) is not context["ndarray"]:
                        raise Error("TypeError: CNN learned buffers require native ndarray storage")
    var blocks = py_get(fields, "_blocks")
    if blocks is not None:
        if py_type(blocks) is not _b().list:
            raise Error("TypeError: CNN block plans require a native list")
        for i in range(len(blocks)):  # driver-loop: exact native model block-pair metadata
            if py_type(blocks[i]) is not _b().tuple or len(blocks[i]) != 2:
                raise Error("TypeError: CNN block plans require native layer pairs")
            if py_type(blocks[i][0]) is not context["xcnn_conv2d_class"]:
                raise Error("TypeError: CNN blocks require their native convolution class")
            if blocks[i][1] is not None and py_type(blocks[i][1]) is not context["xcnn_maxpool_class"]:
                raise Error("TypeError: CNN blocks require their native pooling class")
    _admission_stamp(model, fields, py_contains(fields, "classes_") and py_contains(fields, "layers_")
                     and py_contains(fields, "head_"))
    return fields


def cnnclf_initialize(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    neural_plain_value(args, 0, context)
    var b = _b()
    var fields = _fields(model)
    if py_type(args[0]) is not b.tuple and py_type(args[0]) is not b.list:
        raise Error("TypeError: input_shape requires a native tuple or list")
    if len(args[0]) != 3:
        raise Error("ValueError: input_shape must contain C, H and W")
    if py_type(args[1]) is not b.tuple and py_type(args[1]) is not b.list:
        raise Error("TypeError: conv_channels requires a native tuple or list")
    var input_shape = Python.list()
    var channels = Python.list()
    for i in range(3):  # small-loop(the three declared image dimensions)
        var value = Int(py=args[0][i])
        if value < 1:
            raise Error("ValueError: input_shape dimensions must be positive")
        _ = py_call(input_shape.append, PythonObject(value))
    for i in range(len(args[1])):  # driver-loop: declared convolution layer widths
        var value = Int(py=args[1][i])
        if value < 1:
            raise Error("ValueError: convolution channels must be positive")
        _ = py_call(channels.append, PythonObject(value))
    py_setitem(fields, "input_shape", py_call(b.tuple, input_shape))
    py_setitem(fields, "conv_channels", py_call(b.tuple, channels))
    var integers = Python.tuple("kernel_size", "pool_size", "batch_size", "max_iter")
    var positions: List[Int] = [2, 3, 7, 8]
    for i in range(4):  # small-loop(integer classifier settings)
        var value = Int(py=args[positions[i]])
        if value < (0 if i == 3 else 1):
            raise Error("ValueError: invalid classifier size or iteration setting")
        py_setitem(fields, integers[i], PythonObject(value))
    var floats = Python.tuple("learning_rate", "momentum", "weight_decay", "dampening", "eps")
    var indices: List[Int] = [4, 5, 6, 13, 16]
    for i in range(5):  # small-loop(floating classifier optimizer settings)
        py_setitem(fields, floats[i], PythonObject(Float64(py=args[indices[i]])))
    py_setitem(fields, "shuffle", PythonObject(Bool(py=args[9])))
    py_setitem(fields, "random_state", args[10])
    py_setitem(fields, "numeric_mode", args[11])
    var optimizer = String(args[12])
    if optimizer != "sgd" and optimizer != "adam" and optimizer != "adamw":
        raise Error("ValueError: optimizer must be sgd, adam or adamw")
    py_setitem(fields, "optimizer", PythonObject(optimizer))
    var nesterov = Bool(py=args[14])
    if nesterov and (Float64(py=args[5]) <= 0 or Float64(py=args[13]) != 0):
        raise Error("ValueError: Nesterov momentum requires a momentum and zero dampening")
    py_setitem(fields, "nesterov", PythonObject(nesterov))
    if (py_type(args[15]) is not b.tuple and py_type(args[15]) is not b.list) or len(args[15]) != 2:
        raise Error("ValueError: betas must contain two scalars")
    py_setitem(fields, "betas", Python.tuple(Float64(py=args[15][0]), Float64(py=args[15][1])))
    return PythonObject(None)


def cnnclf_weights(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _classifier_fields(model, context, True)
    var params = _params(fields)
    var out = Python.list()
    for i in range(len(params)):  # driver-loop: publish each learned native array in layer order
        _ = py_call(out.append, _fields(params[i][0])[params[i][1]])
    return out


# ---------------------------------------------------------------- the layers

def _kaiming(mut stream: XseqStream, shape: PythonObject, fan_in: Int, context: PythonObject) raises -> PythonObject:
    """`_kaiming_uniform(rng, shape, fan_in)`: float32 U(-b, b) draws, b =
    1 / sqrt(fan_in) (correctly rounded), from the layer's stream."""
    var out = _empty(shape, "float32", context)
    var bound = 1.0 / sqrt(Float64(fan_in))
    stream.fill_uniform(context, xcnn_addr(out), Int(py=out.size), -bound, bound)
    return out


def _conv_new(c: Int, oc: Int, k: Int, seed: Int, mode: PythonObject, context: PythonObject) raises -> PythonObject:
    """`Conv2d(c, oc, k, padding=k // 2, random_state=seed, numeric_mode=mode)`."""
    var conv = py_call(_b().object.__new__, context["xcnn_conv2d_class"])
    var f = _fields(conv)
    var p = k // 2
    py_setitem(f, "in_channels", c)
    py_setitem(f, "out_channels", oc)
    py_setitem(f, "kernel_size", Python.tuple(k, k))
    py_setitem(f, "stride", Python.tuple(1, 1))
    py_setitem(f, "dilation", Python.tuple(1, 1))
    py_setitem(f, "groups", 1)
    py_setitem(f, "padding_mode", "zeros")
    py_setitem(f, "padding", Python.tuple(p, p))
    py_setitem(f, "_pads", Python.tuple(p, p, p, p))
    py_setitem(f, "_explicit", False)
    py_setitem(f, "_conv_pad", Python.tuple(p, p))
    py_setitem(f, "bias", True)
    py_setitem(f, "random_state", seed)
    py_setitem(f, "input_shape", PythonObject(None))
    py_setitem(f, "numeric_mode", mode)
    var fan_in = c * k * k
    var stream = XseqStream(PythonObject(seed), context)
    py_setitem(f, "weight_", _kaiming(stream, Python.tuple(oc, c, k, k), fan_in, context))
    py_setitem(f, "bias_", _kaiming(stream, Python.tuple(oc), fan_in, context))
    return conv


def _relu_new(mode: PythonObject, context: PythonObject) raises -> PythonObject:
    var relu = py_call(_b().object.__new__, context["xcnn_relu_class"])
    py_setitem(_fields(relu), "numeric_mode", mode)
    return relu


def _maxpool_new(p: Int, mode: PythonObject, context: PythonObject) raises -> PythonObject:
    """`MaxPool2d(p, numeric_mode=mode)`."""
    var pool = py_call(_b().object.__new__, context["xcnn_maxpool_class"])
    var f = _fields(pool)
    py_setitem(f, "ceil_mode", False)
    py_setitem(f, "divisor_override", PythonObject(None))
    py_setitem(f, "kernel_size", Python.tuple(p, p))
    py_setitem(f, "stride", Python.tuple(p, p))
    py_setitem(f, "padding", Python.tuple(0, 0))
    py_setitem(f, "dilation", Python.tuple(1, 1))
    py_setitem(f, "count_include_pad", True)
    py_setitem(f, "input_shape", PythonObject(None))
    py_setitem(f, "numeric_mode", mode)
    return pool


def _linear_new(d_in: Int, d_out: Int, seed: Int, mode: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_Linear(d_in, d_out, random_state=seed, numeric_mode=mode)`."""
    var lin = py_call(_b().object.__new__, context["xcnn_linear_class"])
    var f = _fields(lin)
    py_setitem(f, "in_features", d_in)
    py_setitem(f, "out_features", d_out)
    py_setitem(f, "numeric_mode", mode)
    var stream = XseqStream(PythonObject(seed), context)
    py_setitem(f, "weight_", _kaiming(stream, Python.tuple(d_out, d_in), d_in, context))
    py_setitem(f, "bias_", _kaiming(stream, Python.tuple(d_out), d_in, context))
    return lin


def _build(fields: PythonObject, k: Int, context: PythonObject) raises:
    """`CNNClassifier._build(n_classes)`."""
    var b = _b()
    var rs = fields["random_state"]
    var seed = 0 if rs is None else Int(py=py_call(b.int, rs))
    var shape = fields["input_shape"]
    var c = Int(py=shape[0])
    var h = Int(py=shape[1])
    var w = Int(py=shape[2])
    var ks = Int(py=fields["kernel_size"])
    var ps = Int(py=fields["pool_size"])
    var mode = fields["numeric_mode"]
    var layers = Python.list()
    var blocks = Python.list()
    var channels = fields["conv_channels"]
    for i in range(len(channels)):  # driver-loop: over the conv blocks (model structure)
        var oc = Int(py=channels[i])
        var conv = _conv_new(c, oc, ks, seed + 101 * i, mode, context)
        _ = py_call(layers.append, conv)
        h = h + 2 * (ks // 2) - ks + 1
        w = w + 2 * (ks // 2) - ks + 1
        _ = py_call(layers.append, _relu_new(mode, context))
        var pool = PythonObject(None)
        if h >= ps and w >= ps and ps > 1:
            pool = _maxpool_new(ps, mode, context)
            _ = py_call(layers.append, pool)
            h = h // ps
            w = w // ps
        _ = py_call(blocks.append, Python.tuple(conv, pool))
        c = oc
    py_setitem(fields, "_flat", c * h * w)
    py_setitem(fields, "head_", _linear_new(c * h * w, k, seed + 997, mode, context))
    py_setitem(fields, "layers_", layers)
    py_setitem(fields, "_blocks", blocks)


def _params(fields: PythonObject) raises -> PythonObject:
    """`_params()`: (layer, weight attr, grad attr) of every parametrized layer."""
    var out = Python.list()
    var layers = py_call(_b().list, fields["layers_"])
    _ = py_call(layers.append, fields["head_"])
    for i in range(len(layers)):  # driver-loop: over the layer list (model structure)
        if py_contains(_fields(layers[i]), "weight_"):
            _ = py_call(out.append, Python.tuple(layers[i], "weight_", "grad_weight_"))
            _ = py_call(out.append, Python.tuple(layers[i], "bias_", "grad_bias_"))
    return out


# ---------------------------------------------------------------- the images

def _images(fields: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_images(X)`."""
    var x = xcnn_f32(source, "X", context)
    var shape = xcnn_shape(x)
    var given = fields["input_shape"]
    if len(shape) < 1:
        raise Error("ValueError: classifier images require a row dimension")
    var full: List[Int] = [shape[0], Int(py=given[0]), Int(py=given[1]), Int(py=given[2])]
    if len(shape) == 2:
        return xcnn_reshape(x, full, context)
    if len(shape) != 4:
        raise Error("ValueError: classifier images must have rank two or four")
    for i in range(1, 4):  # small-loop(image channel/height/width metadata)
        if shape[i] != full[i]:
            raise Error("ValueError: images do not match the model input_shape")
    return x


# ---------------------------------------------------------------- the program descriptor

def _engine(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    """The x_sequence whole-model door of the model's tier (the provider the
    label sort already uses)."""
    var labels = context["xcnn_label_bindings"][fields["numeric_mode_"]]
    if py_type(labels) is _b().str:
        raise Error(String(py=labels))
    return labels


def _geometry(fields: PythonObject, n: Int, k: Int) raises -> List[Int]:
    """[n, C, H, W, k, blocks, kernel_size, pool_size]: the head of a CNN_FIT /
    CNN_PREDICT descriptor."""
    var shape = fields["input_shape"]
    var out: List[Int] = [n, Int(py=shape[0]), Int(py=shape[1]), Int(py=shape[2]), k, len(fields["_blocks"]),
                          Int(py=fields["kernel_size"]), Int(py=fields["pool_size"])]
    return out^


def _block_words(fields: PythonObject, mut ints: List[Int]) raises:
    """Append every block's width, then every block's pool flag (a block
    pools when _build kept its MaxPool2d)."""
    var blocks = fields["_blocks"]
    for i in range(len(blocks)):  # driver-loop: over the conv blocks (model structure)
        ints.append(Int(py=_fields(blocks[i][0])["out_channels"]))
    for i in range(len(blocks)):  # driver-loop: over the conv blocks (model structure)
        ints.append(0 if blocks[i][1] is None else 1)


def _param_arrays(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    """Every learned array in descriptor order (per block weight_, bias_,
    then the head's), each a float32 C-contiguous ndarray."""
    var out = _b().list()
    var params = _params(fields)
    for i in range(len(params)):  # driver-loop: over the layer parameter list (model structure)
        _ = py_call(out.append, _contiguous(_fields(params[i][0])[params[i][1]], context))
    return out


# ---------------------------------------------------------------- fit

def cnnclf_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = _b()
    var fields = _classifier_fields(model, context)
    var x = _contiguous(_images(fields, args[0], context), context)
    # the identical-mode profile check, and `numeric_mode_`
    _ = xcnn_entries(fields, context)
    # sorted classes and the inverse codes (the sequence owner's label sort,
    # `model_labels` of the x_sequence binding of the model's tier)
    var labels = _engine(fields, context)
    var encoded = neural_encode_labels(args[1], labels, context)
    py_setitem(fields, "classes_", encoded[0])
    var yi = encoded[1]
    var n = Int(py=x.shape[0])
    if Int(py=yi.size) != n:
        raise Error("ValueError: X and y must contain the same number of rows")
    var k = len(encoded[0])
    _build(fields, k, context)
    var optimizer = String(py=fields["optimizer"])
    var seed64 = XseqStream(fields["random_state"], context).seed
    var batch = Int(py=fields["batch_size"])
    var max_iter = Int(py=fields["max_iter"])
    var losses_all = b.list()
    var curve_out = b.list()
    if max_iter > 0:
        var nsteps = (n + batch - 1) // batch
        var curve = _empty(Python.tuple(max_iter), "float32", context)
        var steps = _empty(Python.tuple(max_iter * nsteps), "float32", context)
        var arrays = _param_arrays(fields, context)
        var addresses: List[Int] = [Int(py=xcnn_addr(x)), Int(py=xcnn_addr(yi)), Int(py=xcnn_addr(curve)),
                                    Int(py=xcnn_addr(steps))]
        for i in range(len(arrays)):  # driver-loop: over the layer parameter arrays (model structure)
            addresses.append(Int(py=xcnn_addr(arrays[i])))
        var ints = _geometry(fields, n, k)
        ints.append(batch)
        ints.append(max_iter)
        ints.append(1 if Bool(py=fields["shuffle"]) else 0)
        ints.append(Int(seed64 & 0xFFFFFFFF))
        ints.append(Int(seed64 >> 32))
        ints.append(0 if optimizer == "sgd" else (1 if optimizer == "adam" else 2))
        ints.append(1 if Bool(py=fields["nesterov"]) else 0)
        _block_words(fields, ints)
        var betas = fields["betas"]
        var floats: List[Float64] = [
            Float64(py=fields["learning_rate"]), Float64(py=fields["momentum"]), Float64(py=betas[0]),
            Float64(py=betas[1]), Float64(py=fields["eps"]), Float64(py=fields["weight_decay"]),
            Float64(py=fields["dampening"]),
        ]
        _ = neural_descriptor_execute(xseq_neural_provider(labels), CNN_FIT, addresses, ints, floats)
        _ = x  # keep the Python objects alive through the native call above
        _ = yi
        # the trained words are in `arrays`; publish them on their layers
        var params = _params(fields)
        for i in range(len(params)):  # driver-loop: over the layer parameter list (model structure)
            py_setitem(_fields(params[i][0]), params[i][1], arrays[i])
        losses_all = py_call(steps.tolist)
        curve_out = py_call(curve.tolist)
    py_setitem(fields, "losses_", losses_all)
    py_setitem(fields, "loss_curve_", curve_out)
    var ishape = fields["input_shape"]
    py_setitem(fields, "n_features_in_", Int(py=ishape[0]) * Int(py=ishape[1]) * Int(py=ishape[2]))
    return model


# ---------------------------------------------------------------- predict

def _run_predict(model: PythonObject, source: PythonObject, codes: Bool, context: PythonObject) raises -> PythonObject:
    """`_run_predict(X, codes)`: probabilities (n, k) float32, or with
    `codes` each row's class index (int32): one CNN_PREDICT call."""
    var fields = _classifier_fields(model, context, True)
    var x = _contiguous(_images(fields, source, context), context)
    var n = Int(py=x.shape[0])
    var k = len(fields["classes_"])
    _ = xcnn_entries(fields, context)
    var out = _empty(Python.tuple(n), "int32", context) if codes else _empty(Python.tuple(n, k), "float32", context)
    if n == 0:
        return out
    var labels = _engine(fields, context)
    var arrays = _param_arrays(fields, context)
    var addresses: List[Int] = [Int(py=xcnn_addr(x)), Int(py=xcnn_addr(out))]
    for i in range(len(arrays)):  # driver-loop: over the layer parameter arrays (model structure)
        addresses.append(Int(py=xcnn_addr(arrays[i])))
    var ints = _geometry(fields, n, k)
    ints.append(1 if codes else 0)
    ints.append(_PREDICT_ROWS)
    _block_words(fields, ints)
    _ = neural_descriptor_execute(xseq_neural_provider(labels), CNN_PREDICT, addresses, ints, List[Float64]())
    _ = x  # keep the Python objects alive through the native call above
    _ = arrays
    return out


def cnnclf_predict_proba(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    return _run_predict(model, args[0], False, context)


def cnnclf_predict(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var codes = _run_predict(model, args[0], True, context)
    return neural_decode_labels(_fields(model)["classes_"], codes, context)
