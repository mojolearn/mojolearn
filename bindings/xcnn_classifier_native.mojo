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
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_setattr, py_fields, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.xcnn_common import xcnn_builtins, xcnn_entries, xcnn_entry, xcnn_f32, xcnn_addr
from bindings.xseq_native_common import XseqStream
from bindings.xseq_neural_native import xseq_labels
from core.array_helpers_impl import nsum_f64

comptime _PREDICT_ROWS = 2048


def _np() raises -> PythonObject:
    return Python.import_module("numpy")


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


# ---------------------------------------------------------------- the layers

def _kaiming(mut stream: XseqStream, shape: PythonObject, fan_in: Int, context: PythonObject) raises -> PythonObject:
    """`_kaiming_uniform(rng, shape, fan_in)`: float32 U(-b, b) draws, b =
    1 / sqrt(fan_in) (correctly rounded), from the layer's stream."""
    var np = _np()
    var out = py_call(np.empty, shape, dtype=np.float32)
    var bound = 1.0 / sqrt(Float64(fan_in))
    stream.fill_uniform(context, out.ctypes.data, Int(py=out.size), -bound, bound)
    return out


def _conv_new(c: Int, oc: Int, k: Int, seed: Int, mode: PythonObject, context: PythonObject) raises -> PythonObject:
    """`Conv2d(c, oc, k, padding=k // 2, random_state=seed, numeric_mode=mode)`."""
    var conv = py_call(_b().object.__new__, context["xcnn_conv2d_class"])
    var f = py_fields(conv)
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
    py_setitem(py_fields(relu), "numeric_mode", mode)
    return relu


def _maxpool_new(p: Int, mode: PythonObject, context: PythonObject) raises -> PythonObject:
    """`MaxPool2d(p, numeric_mode=mode)`."""
    var pool = py_call(_b().object.__new__, context["xcnn_maxpool_class"])
    var f = py_fields(pool)
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
    var f = py_fields(lin)
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
        if py_contains(py_fields(layers[i]), "weight_"):
            _ = py_call(out.append, Python.tuple(layers[i], "weight_", "grad_weight_"))
            _ = py_call(out.append, Python.tuple(layers[i], "bias_", "grad_bias_"))
    return out


# ---------------------------------------------------------------- the plan

def _conv_params(conv: PythonObject, shape: List[Int]) raises -> List[Int]:
    """`Conv2d._params(shape)`."""
    var f = py_fields(conv)
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
    var f = py_fields(pool)
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
        var cshape: List[Int] = [shape[0], Int(py=py_fields(conv)["out_channels"]), Int(py=hw[0]), Int(py=hw[1])]
        var pprm = List[Int]()
        var oshape = cshape.copy()
        if pool is not None:
            pprm = _pool_params(pool, cshape)
            var phw = py_call(_e(entries, "x_cnn_pool_shape"), _ints(pprm))
            oshape = [cshape[0], cshape[1], Int(py=phw[0]), Int(py=phw[1])]
        var ny = _prod(cshape)
        var wsize = Int(py=py_fields(conv)["weight_"][0].size)
        plans.append(_BlockPlan(prm^, pprm^, _prod(shape), _prod(oshape), ny // Int(py=py_fields(conv)["out_channels"]) * wsize, ny))
        shape = oshape^
    return plans^


# ---------------------------------------------------------------- resident arrays

def _new(entries: PythonObject, handles: PythonObject, n: Int) raises -> PythonObject:
    """`_Res.new(n)`."""
    var h = py_call(_e(entries, "x_cnn_res_alloc"), Int(n))
    _ = py_call(handles.append, h)
    return h


def _put(entries: PythonObject, h: PythonObject, arr: PythonObject) raises:
    """`_Res.put(h, arr)`: a float32 or int32 host array's words up."""
    var a = py_call(_np().ascontiguousarray, arr)
    _ = py_call(_e(entries, "x_cnn_res_upload"), h, xcnn_addr(a), a.size)


def _get(entries: PythonObject, h: PythonObject, shape: PythonObject) raises -> PythonObject:
    """`_Res.get(h, shape)`: float32 words down."""
    var np = _np()
    var out = py_call(np.empty, shape, np.float32)
    _ = py_call(_e(entries, "x_cnn_res_download"), h, xcnn_addr(out), out.size)
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
    var b = _b()
    var x = xcnn_f32(source, "X", context)
    var ishape = py_call(b.tuple, fields["input_shape"])
    if Int(py=x.ndim) == 2:
        var full = py_call(b.tuple, Python.list(x.shape[0]))
        return py_call(py_call(_np().ascontiguousarray, x).reshape, full + ishape)
    var tail = py_call(b.tuple, x.shape).__getitem__(py_call(b.slice, 1, PythonObject(None)))
    if not Bool(py=tail == ishape):
        raise Error("ValueError: mojolearn: images of shape " + String(py=py_call(b.str, tail)) + ", the model "
                    + String(py=py_call(b.str, ishape)))
    return x


# ---------------------------------------------------------------- fit

def cnnclf_fit(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var np = _np()
    var b = _b()
    var fields = py_fields(model)
    var x = _images(fields, args[0], context)
    var entries = xcnn_entries(fields, context)
    # sorted classes and the inverse codes (the sequence owner's label sort,
    # `model_labels` of the x_sequence binding of the model's tier)
    var key = py_call(py_call(fields["numeric_mode_"].strip).lower)
    var labels = context["xcnn_label_bindings"][key]
    if py_type(labels) is b.str:
        raise Error(String(py=labels))
    if not py_contains(context, "np_asarray"):
        py_setitem(context, "np_asarray", np.asarray)
    var encoded = xseq_labels(args[1], labels, context)
    py_setitem(fields, "classes_", encoded[0])
    var yi = py_call(np.ascontiguousarray, encoded[1], np.int32)
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
            var arr = py_fields(params[i][0])[params[i][1]]
            var size = Int(py=arr.size)
            var p = _new(entries, handles, size)
            _put(entries, p, arr)
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
        _put(entries, xall, x)
        _put(entries, yall, yi)
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
                var losses = py_call(np.empty, nsteps, dtype=np.float64)
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
                step += nsteps
                epoch = py_call(losses.tolist)
            else:
                var order_rows = py_call(np.empty, n, dtype=np.int32)
                _ = py_call(_e(entries, "x_cnn_epoch_rows"), xcnn_addr(order_rows), Python.list(n, ep, shuffle, seed_lo, seed_hi))
                var s = 0
                var k_out = k
                while s < n:  # driver-loop: over the device minibatch steps
                    var m = min(batch, n - s)
                    var rows = py_call(np.ascontiguousarray, order_rows.__getitem__(py_call(b.slice, s, s + m)), dtype=np.int32)
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
                        var block = py_call(np.empty, Python.tuple(1, 9), dtype=np.float64)
                        var betas = fields["betas"]
                        _ = py_call(_e(entries, "x_cnn_adam_hyper_d"), xcnn_addr(block), Python.list(step, 1),
                                    Python.list(py_call(b.float, fields["learning_rate"]), py_call(b.float, betas[0]),
                                                py_call(b.float, betas[1]), py_call(b.float, fields["eps"]),
                                                py_call(b.float, fields["weight_decay"]), 1.0 if optimizer == "adamw" else 0.0))
                        var hyper = py_call(py_call(block.reshape, -1).tolist)
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
            var lf = py_fields(layer)
            var arr = lf[params[i][1]]
            py_setitem(lf, params[i][1], _get(entries, hp[i], arr.shape))
            py_setitem(lf, params[i][2], _get(entries, hg[i], arr.shape))
        var bufs = b.list()
        for i in range(len(params)):  # driver-loop: reads back each optimizer buffer handle
            _ = py_call(bufs.append, _get(entries, hbuf[i], Python.tuple(per * Int(py=sizes[i]))))
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
    var np = _np()
    var b = _b()
    var fields = py_fields(model)
    var x = _images(fields, source, context)
    var n = Int(py=x.shape[0])
    var k = len(fields["classes_"])
    var entries = xcnn_entries(fields, context)
    var cap = max(1, min(n, _PREDICT_ROWS))
    var proba = PythonObject(None) if codes else py_call(np.empty, Python.tuple(n, k), np.float32)
    var lab = py_call(np.empty, n, np.int32) if codes else PythonObject(None)
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
            var lf = py_fields(layers[i])
            var hw = _new(entries, handles, Int(py=lf["weight_"].size))
            var hb = _new(entries, handles, Int(py=lf["bias_"].size))
            _put(entries, hw, lf["weight_"])
            _put(entries, hb, lf["bias_"])
            py_setitem(rw, py_call(b.id, layers[i]), Python.tuple(hw, hb))
        var a = _resident(fields, entries, handles, cap, False, False, context)
        _put(entries, a["y"], py_call(np.full, cap, -1, np.int32))
        var ishape = fields["input_shape"]
        var row = Int(py=ishape[0]) * Int(py=ishape[1]) * Int(py=ishape[2])
        var base = Int(py=xcnn_addr(x))
        var s = 0
        while s < n:  # driver-loop: over the device prediction batches (cap-sized)
            var m = min(cap, n - s)
            _ = py_call(_e(entries, "x_cnn_res_upload"), a["x"], base + 4 * s * row, m * row)
            _ = _forward_r(fields, entries, rw, a, m)
            _ = py_call(_e(entries, "x_cnn_softmax_xent_r"), a["logits"], a["y"], a["glog"], a["proba"], Python.list(m, k))
            if codes:
                _ = py_call(_e(entries, "x_cnn_res_argmax"), a["proba"], Int(py=xcnn_addr(lab)) + 4 * s, Python.list(m, k))
            else:
                _ = py_call(_e(entries, "x_cnn_res_download"), a["proba"], Int(py=xcnn_addr(proba)) + 4 * s * k, m * k)
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
    var np = _np()
    var codes = _run_predict(model, args[0], True, context)
    var cls = py_fields(model)["classes_"]
    var kind = String(py=cls.dtype.kind)
    # classes_[codes]: ndarray's own C `take`, a copy of the class words (the
    # widened-table gather it replaces wrote the same values)
    _ = kind
    return py_call(cls.take, codes)
