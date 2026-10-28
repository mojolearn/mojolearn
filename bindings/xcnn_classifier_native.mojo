# SPDX-License-Identifier: Apache-2.0
"""`CNNClassifier.fit`, `predict` and `predict_proba`
(python/mojolearn/_expansion_cnn.py) as one native call each, through
`xcnn_native_call` (bindings/xcnn_native.mojo).

fit: the images admitted, the labels encoded (the sequence owner's label
sort, bindings/xseq_neural_native.mojo `xseq_labels`), the layer stack built
natively (Conv2d / ReLU / MaxPool2d blocks and the Linear head, each with the
constructor's fields and its seeded Mojo draws), the resident arrays planned
and allocated, the whole X and its labels uploaded once, then every epoch as
the binding's device epoch (`x_cnn_fit_epoch_d`), or (MOJOLEARN_XCNN_PY_STEPS=1,
or a binding without the epoch entry) the step loop on the same `_r`
entries; the parameters, gradients and optimizer buffers come down at the
end. predict_proba / predict: the resident forward in row chunks and the
device softmax (and argmax), then classes_[codes] by the base gathers.

The same compiled entries with the same argument words, residency and
launch order as the Python bodies, so the bits do not move.
"""
from std.math import sqrt
from std.memory import memcpy
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_setattr, py_fields, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.xcnn_common import xcnn_builtins, xcnn_entries, xcnn_entry, xcnn_f32, xcnn_addr
from bindings.xseq_native_common import XseqStream
from bindings.xcnn_classifier_labels import xcnn_encode_labels, xcnn_decode_labels
from bindings.mode_metadata_native import _fields
from bindings.neural_block_admission import neural_plain_value
from bindings.sequential_admission import sequential_admit_value
from bindings.sequential_inputs import sequential_input
from bindings.sequential_arrays import sequential_array
from bindings.xcnn_common import xcnn_shape, xcnn_tuple, xcnn_reshape
from providers.python.borrowed_buffer_pyapi import AdmittedBorrowedBuffer, admit_native_buffer_exporter
from core.array_helpers_impl import nsum_f64

comptime _PREDICT_ROWS = 2048


def _b() raises -> PythonObject:
    return xcnn_builtins()


def _e(entries: PythonObject, name: String) raises -> PythonObject:
    return xcnn_entry(entries, name)


def _has(entries: PythonObject, name: String) raises -> Bool:
    return py_contains(entries["symbols"], name)


def _ints(values: List[Int]) raises -> PythonObject:
    var out = Python.list()
    for i in range(len(values)):  # small-loop(launch-plan words)
        _ = py_call(out.append, values[i])
    return out


def _prod(values: List[Int]) -> Int:
    var n = 1
    for i in range(len(values)):  # small-loop(shape words)
        n *= values[i]
    return n


def _empty(shape: PythonObject, dtype: String, context: PythonObject) raises -> PythonObject:
    var dims = Python.tuple(shape) if py_type(shape) is _b().int else shape
    return sequential_array(dims, context[PythonObject(dtype)], context)


def _contiguous(source: PythonObject, context: PythonObject) raises -> PythonObject:
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


def _float_words(values: PythonObject) raises -> PythonObject:
    var count = Int(py=values.size)
    var words = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=xcnn_addr(values)))
    var result = Python.list()
    for i in range(count):  # driver-loop: native loss/optimizer scalar result publication
        _ = py_call(result.append, PythonObject(words[i]))
    _ = values
    return result


def _minus_one(count: Int, context: PythonObject) raises -> PythonObject:
    var array = _empty(Python.tuple(count), "int32", context)
    var words = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(py=xcnn_addr(array)))
    for i in range(count):  # driver-loop: native absent-target metadata for prediction
        words[i] = -1
    _ = array
    return array


def _classifier_fields(model: PythonObject, context: PythonObject, fitted: Bool = False) raises -> PythonObject:
    var fields = _fields(model)
    for name in ["input_shape", "conv_channels", "kernel_size", "pool_size", "learning_rate",  # driver-loop: over passive classifier settings
                 "momentum", "weight_decay", "batch_size", "max_iter", "shuffle", "random_state", "numeric_mode",
                 "optimizer", "dampening", "nesterov", "betas", "eps", "_flat"]:
        neural_plain_value(py_get(fields, PythonObject(name)), 0, context)
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
                neural_plain_value(py_get(state, PythonObject(key)), 0, context)
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


# ---------------------------------------------------------------- the plan

def _conv_params(conv: PythonObject, shape: List[Int]) raises -> List[Int]:
    """`Conv2d._params(shape)`."""
    var f = _fields(conv)
    var g = Int(py=f["groups"])
    var ks = f["kernel_size"]
    var st = f["stride"]
    var cp = f["_conv_pad"]
    var dl = f["dilation"]
    return [shape[0], shape[1] // g, shape[2], shape[3], Int(py=f["out_channels"]) // g, Int(py=ks[0]), Int(py=ks[1]),
            Int(py=st[0]), Int(py=st[1]), Int(py=cp[0]), Int(py=cp[1]), Int(py=dl[0]), Int(py=dl[1]), 0, 0,
            1 if Bool(py=f["bias"]) else 0, 0]


def _pool_params(pool: PythonObject, shape: List[Int]) raises -> List[Int]:
    """`_Pool2d._params(shape)`."""
    var f = _fields(pool)
    var ks = f["kernel_size"]
    var st = f["stride"]
    var pd = f["padding"]
    var dl = f["dilation"]
    var div = f["divisor_override"]
    return [shape[0], shape[1], shape[2], shape[3], Int(py=ks[0]), Int(py=ks[1]), Int(py=st[0]), Int(py=st[1]),
            Int(py=pd[0]), Int(py=pd[1]), Int(py=dl[0]), Int(py=dl[1]), 0, 0,
            1 if Bool(py=f["count_include_pad"]) else 0, 0, 1 if Bool(py=f["ceil_mode"]) else 0,
            0 if div is None else Int(py=div)]


struct _BlockPlan(Copyable, Movable):
    var prm: List[Int]
    var pprm: List[Int]
    var in_size: Int
    var out_size: Int
    var im2col: Int
    var ny: Int

    def __init__(out self, var prm: List[Int], var pprm: List[Int], in_size: Int, out_size: Int, im2col: Int, ny: Int):
        self.prm = prm^
        self.pprm = pprm^
        self.in_size = in_size
        self.out_size = out_size
        self.im2col = im2col
        self.ny = ny


def _plan(fields: PythonObject, entries: PythonObject, n: Int) raises -> List[_BlockPlan]:
    """`_plan(n)`: per block (conv params, pool params or [], input size,
    output size, im2col size, conv output size) at batch size n."""
    var plans = List[_BlockPlan]()
    var ishape = fields["input_shape"]
    var shape: List[Int] = [n, Int(py=ishape[0]), Int(py=ishape[1]), Int(py=ishape[2])]
    var blocks = fields["_blocks"]
    for i in range(len(blocks)):  # driver-loop: one launch plan per conv block (model structure)
        var conv = blocks[i][0]
        var pool = blocks[i][1]
        var prm = _conv_params(conv, shape)
        var hw = py_call(_e(entries, "x_cnn_conv_shape"), _ints(prm))
        var cshape: List[Int] = [shape[0], Int(py=_fields(conv)["out_channels"]), Int(py=hw[0]), Int(py=hw[1])]
        var pprm = List[Int]()
        var oshape = cshape.copy()
        if pool is not None:
            pprm = _pool_params(pool, cshape)
            var phw = py_call(_e(entries, "x_cnn_pool_shape"), _ints(pprm))
            oshape = [cshape[0], cshape[1], Int(py=phw[0]), Int(py=phw[1])]
        var ny = _prod(cshape)
        var wsize = Int(py=_fields(conv)["weight_"].size) // Int(py=_fields(conv)["out_channels"])
        plans.append(_BlockPlan(prm^, pprm^, _prod(shape), _prod(oshape), ny // Int(py=_fields(conv)["out_channels"]) * wsize, ny))
        shape = oshape^
    return plans^


# ---------------------------------------------------------------- resident arrays

def _new(entries: PythonObject, handles: PythonObject, n: Int) raises -> PythonObject:
    """`_Res.new(n)`."""
    var h = py_call(_e(entries, "x_cnn_res_alloc"), Int(n))
    _ = py_call(handles.append, h)
    return h


def _put(entries: PythonObject, h: PythonObject, arr: PythonObject, context: PythonObject) raises:
    """`_Res.put(h, arr)`: a float32 or int32 host array's words up."""
    var a = _contiguous(arr, context)
    _ = py_call(_e(entries, "x_cnn_res_upload"), h, xcnn_addr(a), a.size)
    _ = a  # keep the Python object alive through the native call above


def _get(entries: PythonObject, h: PythonObject, shape: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_Res.get(h, shape)`: float32 words down."""
    var out = _empty(shape, "float32", context)
    _ = py_call(_e(entries, "x_cnn_res_download"), h, xcnn_addr(out), out.size)
    _ = out  # keep the Python object alive through the native call above
    return out


def _free_all(entries: PythonObject, handles: PythonObject) raises:
    for i in range(len(handles)):  # driver-loop: over the resident buffer handles
        _ = py_call(_e(entries, "x_cnn_res_free"), handles[i])


def _resident(fields: PythonObject, entries: PythonObject, handles: PythonObject, n: Int, save: Bool, train: Bool,
              context: PythonObject) raises -> PythonObject:
    """`_resident(R, n, save, train)`: the step's resident arrays."""
    var b = _b()
    var plans = _plan(fields, entries, n)
    var k = len(fields["classes_"])
    var flat = Int(py=fields["_flat"])
    var a = b.dict()
    py_setitem(a, "x", _new(entries, handles, plans[0].in_size if len(plans) else n * flat))
    py_setitem(a, "y", _new(entries, handles, n))
    py_setitem(a, "logits", _new(entries, handles, n * k))
    py_setitem(a, "glog", _new(entries, handles, n * k))
    py_setitem(a, "proba", _new(entries, handles, n * k))
    var outs = Python.list()
    var idxs = Python.list()
    var gouts = Python.list()
    for i in range(len(plans)):  # driver-loop: one buffer per conv block
        _ = py_call(outs.append, _new(entries, handles, plans[i].out_size))
    for i in range(len(plans)):  # driver-loop: one buffer per conv block
        _ = py_call(idxs.append, _new(entries, handles, plans[i].out_size))
    if train:
        for i in range(len(plans)):  # driver-loop: one buffer per conv block
            _ = py_call(gouts.append, _new(entries, handles, plans[i].out_size))
    py_setitem(a, "out", outs)
    py_setitem(a, "idx", idxs)
    py_setitem(a, "gout", gouts)
    var bounded = _has(entries, "x_cnn_bounded_im2col") and Bool(py=py_call(_e(entries, "x_cnn_bounded_im2col")))
    var budget = Int(py=py_call(_e(entries, "x_cnn_neural_tape_budget_bytes"))) if _has(entries, "x_cnn_neural_tape_budget_bytes") else -1
    var saved = Python.list()
    for i in range(len(plans)):  # driver-loop: optional buffers per conv block
        var tape = 4 * ((1 if bounded else plans[i].im2col) + plans[i].ny)
        var keep = save and (budget < 0 or tape <= budget)
        if keep:
            _ = py_call(saved.append, Python.list(_new(entries, handles, 1 if bounded else plans[i].im2col),
                                                  _new(entries, handles, plans[i].ny)))
            if budget >= 0:
                budget -= tape
        elif save:
            _ = py_call(saved.append, Python.list(0, 0))
        else:
            _ = py_call(saved.append, Python.list())
    py_setitem(a, "saved", saved)
    if len(plans) == 0:
        py_setitem(a, "ghead", _new(entries, handles, n * flat))
    return a


def _forward_r(fields: PythonObject, entries: PythonObject, rw: PythonObject, a: PythonObject, n: Int) raises -> List[_BlockPlan]:
    """`_forward_r(b, a, n)`: logits of the n rows in a["x"], resident."""
    var b = _b()
    var plans = _plan(fields, entries, n)
    var src = a["x"]
    var blocks = fields["_blocks"]
    for i in range(len(plans)):  # driver-loop: one device launch per conv block
        var conv = blocks[i][0]
        var hw = rw[py_call(b.id, conv)]
        _ = py_call(_e(entries, "x_cnn_conv_block_forward_r"), src, hw[0], hw[1], a["out"][i], a["idx"][i],
                    _ints(plans[i].prm), _ints(plans[i].pprm), a["saved"][i])
        src = a["out"][i]
    var head = rw[py_call(b.id, fields["head_"])]
    _ = py_call(_e(entries, "x_cnn_linear_forward_r"), src, head[0], head[1], a["logits"],
                Python.list(n, fields["_flat"], len(fields["classes_"])))
    return plans^


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
        return xcnn_reshape(x, full)
    if len(shape) != 4:
        raise Error("ValueError: classifier images must have rank two or four")
    for i in range(1, 4):  # small-loop(image channel/height/width metadata)
        if shape[i] != full[i]:
            raise Error("ValueError: images do not match the model input_shape")
    return x


# ---------------------------------------------------------------- fit

def cnnclf_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = _b()
    var fields = _classifier_fields(model, context)
    var x = _images(fields, args[0], context)
    var entries = xcnn_entries(fields, context)
    # sorted classes and the inverse codes (the sequence owner's label sort,
    # `model_labels` of the x_sequence binding of the model's tier)
    var key = fields["numeric_mode_"]
    var labels = context["xcnn_label_bindings"][key]
    if py_type(labels) is b.str:
        raise Error(String(py=labels))
    var encoded = xcnn_encode_labels(args[1], labels, context)
    py_setitem(fields, "classes_", encoded[0])
    var yi = encoded[1]
    if Int(py=yi.size) != Int(py=x.shape[0]):
        raise Error("ValueError: X and y must contain the same number of rows")
    var k = len(encoded[0])
    _build(fields, k, context)
    var optimizer = String(py=fields["optimizer"])
    var per = 1 if optimizer == "sgd" else 2
    var params = _params(fields)
    var order = XseqStream(fields["random_state"], context)
    var seed64 = order.seed
    var n = Int(py=x.shape[0])
    var batch = Int(py=fields["batch_size"])
    var cap = min(batch, n)
    var losses_all = b.list()
    var curve = b.list()
    py_setitem(fields, "losses_", losses_all)
    py_setitem(fields, "loss_curve_", curve)
    var handles = b.list()
    var rw = b.dict()
    py_setitem(fields, "_rw", rw)
    try:
        var hp = b.list()
        var hg = b.list()
        var hbuf = b.list()
        var sizes = b.list()
        for i in range(len(params)):  # driver-loop: over the layer parameter list
            var arr = _fields(params[i][0])[params[i][1]]
            var size = Int(py=arr.size)
            var p = _new(entries, handles, size)
            _put(entries, p, arr, context)
            _ = py_call(hp.append, p)
            _ = py_call(hg.append, _new(entries, handles, size))
            _ = py_call(hbuf.append, _new(entries, handles, per * size))
            _ = py_call(sizes.append, size)
        for i in range(len(params)):  # driver-loop: over the layer parameter list
            if String(py=params[i][1]) == "weight_":
                py_setitem(rw, py_call(b.id, params[i][0]), Python.tuple(hp[i], hp[i + 1], hg[i], hg[i + 1]))
        var a = _resident(fields, entries, handles, cap, True, True, context)
        var lists = String(py=py_call(b.str, py_call(_e(entries, "x_cnn_vendor")))) != "cpu"
        var ishape = fields["input_shape"]
        var row = Int(py=ishape[0]) * Int(py=ishape[1]) * Int(py=ishape[2])
        var xall = _new(entries, handles, Int(py=x.size))
        var yall = _new(entries, handles, n)
        _put(entries, xall, x, context)
        _put(entries, yall, yi, context)
        var step = 0
        var epoch_entry = Bool(py=context["xcnn_epoch_entry"]) and _has(entries, "x_cnn_fit_epoch_r")
        var seed_lo = Int(seed64 & 0xFFFFFFFF)
        var seed_hi = Int(seed64 >> 32)
        var max_iter = Int(py=fields["max_iter"])
        var shuffle = 1 if Bool(py=fields["shuffle"]) else 0
        var sgd = optimizer == "sgd"
        var spec = PythonObject(None)
        var nsteps = 0
        if epoch_entry:
            nsteps = (n + batch - 1) // batch
            var m_last = n - (nsteps - 1) * batch
            var head = rw[py_call(b.id, fields["head_"])]
            var blocks = b.list()
            var bl = fields["_blocks"]
            for j in range(len(bl)):  # driver-loop: over the conv blocks
                var hw = rw[py_call(b.id, bl[j][0])]
                var entry = Python.list(hw[0], hw[1], hw[2], hw[3], a["out"][j], a["idx"][j], a["gout"][j])
                _ = py_call(entry.extend, a["saved"][j])
                _ = py_call(blocks.append, entry)
            var full_plan = b.list()
            var last_plan = b.list()
            var pf = _plan(fields, entries, cap)
            var pl = _plan(fields, entries, m_last)
            for j in range(len(pf)):  # driver-loop: launch plan of every conv block
                _ = py_call(full_plan.append, Python.list(_ints(pf[j].prm), _ints(pf[j].pprm)))
            for j in range(len(pl)):  # driver-loop: launch plan of every conv block
                _ = py_call(last_plan.append, Python.list(_ints(pl[j].prm), _ints(pl[j].pprm)))
            spec = b.dict()
            py_setitem(spec, "blocks", blocks)
            py_setitem(spec, "head", Python.list(head[0], head[1], head[2], head[3]))
            py_setitem(spec, "a", Python.list(a["x"], a["y"], a["logits"], a["glog"], a["proba"], py_get(a, "ghead", PythonObject(0))))
            py_setitem(spec, "opt", Python.list(hp, hg, hbuf, sizes))
            py_setitem(spec, "data", Python.list(xall, yall, row))
            py_setitem(spec, "dims", Python.list(fields["_flat"], k))
            py_setitem(spec, "plan_full", full_plan)
            py_setitem(spec, "plan_last", last_plan)
        for ep in range(max_iter):  # driver-loop: over the device epochs
            var epoch = b.list()
            if epoch_entry:
                var losses = _empty(Python.tuple(nsteps), "float64", context)
                var fpar = Python.list()
                if sgd:
                    fpar = Python.list(py_call(b.float, fields["learning_rate"]), py_call(b.float, fields["momentum"]),
                                       py_call(b.float, fields["weight_decay"]), py_call(b.float, fields["dampening"]),
                                       1.0 if Bool(py=fields["nesterov"]) else 0.0, 0.0)
                else:
                    var betas = fields["betas"]
                    fpar = Python.list(py_call(b.float, fields["learning_rate"]), py_call(b.float, betas[0]),
                                       py_call(b.float, betas[1]), py_call(b.float, fields["eps"]),
                                       py_call(b.float, fields["weight_decay"]), 1.0 if optimizer == "adamw" else 0.0)
                _ = py_call(_e(entries, "x_cnn_fit_epoch_d"), spec, xcnn_addr(losses),
                            Python.list(n, batch, 0 if sgd else 1, step, ep, shuffle, seed_lo, seed_hi), fpar)
                _ = losses  # keep the Python object alive through the native call above
                step += nsteps
                epoch = _float_words(losses)
            else:
                var order_rows = _empty(Python.tuple(n), "int32", context)
                _ = py_call(_e(entries, "x_cnn_epoch_rows"), xcnn_addr(order_rows), Python.list(n, ep, shuffle, seed_lo, seed_hi))
                _ = order_rows  # keep the Python object alive through the native call above
                var s = 0
                var k_out = k
                while s < n:  # driver-loop: over the device minibatch steps
                    var m = min(batch, n - s)
                    var rows = py_call(context["ndarray"], Python.tuple(m), dtype=context["int32"], buffer=order_rows, offset=s * 4)
                    if not lists:
                        _ = py_call(_e(entries, "x_cnn_res_gather"), a["x"], xall, xcnn_addr(rows), Python.list(m, row))
                        _ = rows  # keep the Python object alive through the native call above
                        _ = py_call(_e(entries, "x_cnn_res_gather"), a["y"], yall, xcnn_addr(rows), Python.list(m, 1))
                        _ = rows  # keep the Python object alive through the native call above
                    else:
                        _ = py_call(_e(entries, "x_cnn_res_gather"), Python.list(a["x"], a["y"]), Python.list(xall, yall),
                                    xcnn_addr(rows), Python.list(m, row, 1))
                        _ = rows  # keep the Python object alive through the native call above
                    var plans = _forward_r(fields, entries, rw, a, m)
                    var loss = py_call(b.float, py_call(_e(entries, "x_cnn_softmax_xent_r"), a["logits"], a["y"], a["glog"],
                                                       a["proba"], Python.list(m, k_out)))
                    var head = rw[py_call(b.id, fields["head_"])]
                    var last = a["out"][len(plans) - 1] if len(plans) else a["x"]
                    var glast = a["gout"][len(plans) - 1] if len(plans) else a["ghead"]
                    _ = py_call(_e(entries, "x_cnn_linear_backward_r"), last, head[0], a["glog"], glast, head[2], head[3],
                                Python.list(m, fields["_flat"], k_out))
                    var bl = fields["_blocks"]
                    var j = len(plans) - 1
                    while j >= 0:  # driver-loop: the backward launch per conv block
                        var hw = rw[py_call(b.id, bl[j][0])]
                        var src = a["out"][j - 1] if j > 0 else a["x"]
                        var dx = a["gout"][j - 1] if j > 0 else PythonObject(0)
                        var tail = Python.list(dx, hw[2], hw[3])
                        _ = py_call(tail.extend, a["saved"][j])
                        _ = py_call(_e(entries, "x_cnn_conv_block_backward_r"), src, hw[0], hw[1], a["gout"][j], a["idx"][j],
                                    tail, _ints(plans[j].prm), _ints(plans[j].pprm))
                        j -= 1
                    step += 1
                    if sgd:
                        var hyper = Python.list(py_call(b.float, fields["learning_rate"]), py_call(b.float, fields["momentum"]),
                                                py_call(b.float, fields["weight_decay"]), py_call(b.float, fields["dampening"]),
                                                1.0 if Bool(py=fields["nesterov"]) else 0.0, 1.0 if step == 1 else 0.0)
                        if lists:
                            _ = py_call(_e(entries, "x_cnn_sgd_r"), hp, hg, hbuf, sizes, hyper)
                        else:
                            for q in range(len(params)):  # driver-loop: one optimizer launch per parameter array
                                _ = py_call(_e(entries, "x_cnn_sgd_r"), hp[q], hg[q], hbuf[q], Python.list(sizes[q]), hyper)
                    else:
                        var block = _empty(Python.tuple(1, 9), "float64", context)
                        var betas = fields["betas"]
                        _ = py_call(_e(entries, "x_cnn_adam_hyper_d"), xcnn_addr(block), Python.list(step, 1),
                                    Python.list(py_call(b.float, fields["learning_rate"]), py_call(b.float, betas[0]),
                                                py_call(b.float, betas[1]), py_call(b.float, fields["eps"]),
                                                py_call(b.float, fields["weight_decay"]), 1.0 if optimizer == "adamw" else 0.0))
                        _ = block  # keep the Python object alive through the native call above
                        var hyper = _float_words(block)
                        if lists:
                            _ = py_call(_e(entries, "x_cnn_adam_r"), hp, hg, hbuf, sizes, hyper)
                        else:
                            for q in range(len(params)):  # driver-loop: one optimizer launch per parameter array
                                _ = py_call(_e(entries, "x_cnn_adam_r"), hp[q], hg[q], hbuf[q], Python.list(sizes[q]), hyper)
                    _ = py_call(epoch.append, loss)
                    s += m
            _ = py_call(losses_all.extend, epoch)
            # CPython 3.12+'s sum spelled out (`_portable_math.nsum`: the same
            # Neumaier fold in the same order, core/array_helpers_impl.mojo)
            var vals = py_call(context["array_type"], "d", epoch)
            var total = nsum_f64(MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=py_call(vals.buffer_info)[0])),
                                 len(vals)) if len(epoch) else 0.0
            _ = py_call(curve.append, PythonObject(total / Float64(len(epoch))))
        for i in range(len(params)):  # driver-loop: reads back each parameter array handle
            var layer = params[i][0]
            var lf = _fields(layer)
            var arr = lf[params[i][1]]
            py_setitem(lf, params[i][1], _get(entries, hp[i], arr.shape, context))
            py_setitem(lf, params[i][2], _get(entries, hg[i], arr.shape, context))
        var bufs = b.list()
        for i in range(len(params)):  # driver-loop: reads back each optimizer buffer handle
            _ = py_call(bufs.append, _get(entries, hbuf[i], Python.tuple(per * Int(py=sizes[i])), context))
        py_setitem(fields, "_bufs", bufs)
        py_setitem(fields, "_rw", b.dict())
    except error:
        _free_all(entries, handles)
        raise error^
    _free_all(entries, handles)
    var ishape = fields["input_shape"]
    py_setitem(fields, "n_features_in_", Int(py=ishape[0]) * Int(py=ishape[1]) * Int(py=ishape[2]))
    return model


# ---------------------------------------------------------------- predict

def _run_predict(model: PythonObject, source: PythonObject, codes: Bool, context: PythonObject) raises -> PythonObject:
    """`_run_predict(X, codes)`: probabilities (n, k) float32, or with
    `codes` each row's class index (int32), on the device."""
    var b = _b()
    var fields = _classifier_fields(model, context, True)
    var x = _images(fields, source, context)
    var n = Int(py=x.shape[0])
    var k = len(fields["classes_"])
    var entries = xcnn_entries(fields, context)
    var cap = max(1, min(n, _PREDICT_ROWS))
    var proba = PythonObject(None) if codes else _empty(Python.tuple(n, k), "float32", context)
    var lab = _empty(Python.tuple(n), "int32", context) if codes else PythonObject(None)
    var handles = b.list()
    var rw = b.dict()
    py_setitem(fields, "_rw", rw)
    try:
        var layers = b.list()
        var blocks = fields["_blocks"]
        for i in range(len(blocks)):  # driver-loop: over the conv layers
            _ = py_call(layers.append, blocks[i][0])
        _ = py_call(layers.append, fields["head_"])
        for i in range(len(layers)):  # driver-loop: the layer list for device copies
            var lf = _fields(layers[i])
            var hw = _new(entries, handles, Int(py=lf["weight_"].size))
            var hb = _new(entries, handles, Int(py=lf["bias_"].size))
            _put(entries, hw, lf["weight_"], context)
            _put(entries, hb, lf["bias_"], context)
            py_setitem(rw, py_call(b.id, layers[i]), Python.tuple(hw, hb))
        var a = _resident(fields, entries, handles, cap, False, False, context)
        _put(entries, a["y"], _minus_one(cap, context), context)
        var ishape = fields["input_shape"]
        var row = Int(py=ishape[0]) * Int(py=ishape[1]) * Int(py=ishape[2])
        var base = Int(py=xcnn_addr(x))
        _ = x  # keep the Python object alive through the native call above
        var s = 0
        while s < n:  # driver-loop: over the device prediction batches (cap-sized)
            var m = min(cap, n - s)
            _ = py_call(_e(entries, "x_cnn_res_upload"), a["x"], base + 4 * s * row, m * row)
            _ = x  # keep the Python object alive through the native call above
            _ = _forward_r(fields, entries, rw, a, m)
            _ = py_call(_e(entries, "x_cnn_softmax_xent_r"), a["logits"], a["y"], a["glog"], a["proba"], Python.list(m, k))
            if codes:
                _ = py_call(_e(entries, "x_cnn_res_argmax"), a["proba"], Int(py=xcnn_addr(lab)) + 4 * s, Python.list(m, k))
                _ = lab  # keep the Python object alive through the native call above
            else:
                _ = py_call(_e(entries, "x_cnn_res_download"), a["proba"], Int(py=xcnn_addr(proba)) + 4 * s * k, m * k)
                _ = proba  # keep the Python object alive through the native call above
            s += m
        py_setitem(fields, "_rw", b.dict())
    except error:
        _free_all(entries, handles)
        raise error^
    _free_all(entries, handles)
    return lab if codes else proba


def cnnclf_predict_proba(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    return _run_predict(model, args[0], False, context)


def cnnclf_predict(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var codes = _run_predict(model, args[0], True, context)
    return xcnn_decode_labels(_fields(model)["classes_"], codes, context)
