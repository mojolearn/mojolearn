# SPDX-License-Identifier: Apache-2.0
"""`CNNClassifier.fit`, `predict` and `predict_proba`
(python/mojolearn/_expansion_cnn.py) as one native call each, through
`xcnn_native_call` (bindings/xcnn_native.mojo).

fit: the images admitted, the labels encoded (the sequence owner's label
sort, bindings/xseq_neural_native.mojo `xseq_labels`), the layer stack built
from typed RUN (x_cnn/classifier_run.mojo `cnnclf_run_layout`: the block
planes, seeds and fan-ins; `cnnclf_run_init`: the seeded Kaiming draws on the
model's image, GIL released), then its Conv2d / ReLU / MaxPool2d blocks and
Linear head published as layer objects (PACK), then the whole training as
ONE call of the x_sequence whole-model door of the model's tier (CNN_FIT,
sequence/cnn_classifier_program.mojo): the layer stack captured as one
program on the program executor and trained by Program.fit; the trained
words come back into the layers' arrays. predict_proba / predict: one
CNN_PREDICT call on the model's resident session (the fitted parameters
uploaded once, by CNN_OPEN at the first predict; lane
x8b-cnn-resident-predict), the captured forward with the softmax (and
argmax), then classes_[codes] by the base gathers.

Lane x8-cnn-on-executor (2026-11-09): the hand epoch on the x_cnn resident
entries (x_cnn_fit_d / x_cnn_fit_epoch_d / x_cnn_fit_epoch_r, the Python
step-loop arm, the resident predict record and x_cnn_predict_r) is deleted.
TOMBSTONE: the hand CNN classifier epoch and resident predict deleted 2026-11-09 by lane/x8-cnn-on-executor; code recoverable at f59d913b0b.
Restore: git apply experiments/removed/X8_CNN_HAND_EPOCH.patch
"""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from providers.python.neural_descriptor_pyapi_door import NeuralDescriptorProvider
from sequence.xseq_neural_run import XseqNeuralEntry
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_fields, py_contains
from bindings.xcnn_common_door import xcnn_builtins, xcnn_labels, xcnn_f32, xcnn_addr, xcnn_readonly
from bindings.xcnn_common_door import xcnn_device_init, xcnn_device_inherit
from bindings.xseq_native_common_door import XseqStream, xseq_neural_provider
from bindings.neural_labels_native_unpack import neural_encode_labels, neural_decode_labels
from bindings.mode_metadata_native_door import _fields
from bindings.neural_block_admission_door import neural_plain_value
from bindings.native_loaded_module_door import NativeModules
from bindings.sequential_admission_door import sequential_admit_value
from bindings.sequential_inputs_door import sequential_input
from bindings.sequential_arrays_door import sequential_array
from bindings.xcnn_common_door import xcnn_shape, xcnn_tuple, xcnn_reshape, xcnn_numpy
from providers.python.borrowed_buffer_pyapi_unpack import AdmittedBorrowedBuffer, admit_native_buffer_exporter
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_execute
from bindings.py_model_records import (
    RECORDS_PREPARED, ContentRecord, content_fields, content_object, records_stampable, records_stamped_get,
    records_stamped_put,
)
from sequence.neural_model_dispatch import CNN_FIT, CNN_PREDICT, CNN_OPEN, SESSION_CLOSE
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_adopt, NativeDescriptorResource
from bindings.native_prepared_owner_unpack import prepared_get, prepared_put, prepared_drop
from bindings.native_identity_key_door import NativeIdentityKey
from x_cnn.classifier_run import (
    CnnClfPlan, CnnClfFitSettings, CnnClfLayout, cnnclf_run_layout, cnnclf_run_init, cnnclf_run_check_input_rank, cnnclf_run_check_input_dim,
    cnnclf_run_check_channels, cnnclf_run_check_setting, cnnclf_run_check_optimizer, cnnclf_run_optimizer_word,
    cnnclf_run_check_nesterov, cnnclf_run_check_rows, cnnclf_run_images,
    cnnclf_run_features, cnnclf_run_check_labels, cnnclf_run_fit_steps, cnnclf_run_fit_addresses,
    cnnclf_run_fit_ints, cnnclf_run_fit_floats, cnnclf_run_open_addresses, cnnclf_run_open_ints,
    cnnclf_run_check_session, cnnclf_run_predict_ints,
)
from bindings.xcnn_keys_unpack import xcnn_keys
from bindings.py_balanced import py_item


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
        var nk = xcnn_keys()
        return py_call(py_item(context, nk[].k_ndarray), xcnn_tuple(layout.shape), dtype=array.dtype, buffer=array)
    var out = sequential_array(xcnn_tuple(layout.shape), array.dtype, context)
    layout.copy_to(Int(py=xcnn_addr(out)))
    _ = array
    return out


# N135 (ne3-c-cnn, 2026-11-07): the admission record. A call on a classifier
# whose generation word (bindings/py_model_records.mojo: the watched
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
    var nk = xcnn_keys()
    var layers = py_get(fields, nk[].k_layers_)
    if layers is not None and py_type(layers) is _b().list:
        for i in range(len(layers)):  # driver-loop: over the classifier layers (content words, once per record)
            content_object(content, layers[i], 4)
    var head = py_get(fields, nk[].k_head_)
    if head is not None:
        content_object(content, head, 4)
    var blocks = py_get(fields, nk[].k__blocks)
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
    var nk = xcnn_keys()
    if fitted and (not py_contains(fields, nk[].k_classes_) or not py_contains(fields, nk[].k_layers_) or not py_contains(fields, nk[].k_head_)):
        raise Error("RuntimeError: call fit before classifier inference")
    var classes = py_get(fields, nk[].k_classes_)
    if classes is not None and py_type(classes) is not py_item(context, nk[].k_ndarray):
        raise Error("TypeError: CNN classes_ requires its native ndarray")
    var layers = py_get(fields, nk[].k_layers_)
    if layers is not None:
        if py_type(layers) is not _b().list:
            raise Error("TypeError: CNN classifier layers require an exact native list")
        var owners = py_call(_b().list, layers)
        if py_contains(fields, nk[].k_head_):
            _ = py_call(owners.append, py_item(fields, nk[].k_head_))
        for i in range(len(owners)):  # driver-loop: retained native classifier layer metadata
            var layer = owners[i]
            var cls = py_type(layer)
            if (cls is not py_item(context, nk[].k_xcnn_conv2d_class) and cls is not py_item(context, nk[].k_xcnn_relu_class)
                    and cls is not py_item(context, nk[].k_xcnn_maxpool_class) and cls is not py_item(context, nk[].k_xcnn_linear_class)):
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
                    if py_type(buffer) is not py_item(context, nk[].k_ndarray):
                        raise Error("TypeError: CNN learned buffers require native ndarray storage")
    var blocks = py_get(fields, nk[].k__blocks)
    if blocks is not None:
        if py_type(blocks) is not _b().list:
            raise Error("TypeError: CNN block plans require a native list")
        for i in range(len(blocks)):  # driver-loop: exact native model block-pair metadata
            if py_type(blocks[i]) is not _b().tuple or len(blocks[i]) != 2:
                raise Error("TypeError: CNN block plans require native layer pairs")
            if py_type(blocks[i][0]) is not py_item(context, nk[].k_xcnn_conv2d_class):
                raise Error("TypeError: CNN blocks require their native convolution class")
            if blocks[i][1] is not None and py_type(blocks[i][1]) is not py_item(context, nk[].k_xcnn_maxpool_class):
                raise Error("TypeError: CNN blocks require their native pooling class")
    _admission_stamp(model, fields, py_contains(fields, nk[].k_classes_) and py_contains(fields, nk[].k_layers_)
                     and py_contains(fields, nk[].k_head_))
    return fields


def cnnclf_initialize(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    neural_plain_value(args, 0, context)
    var b = _b()
    var fields = _fields(model)
    xcnn_device_init(model, fields, args[17], context)
    if py_type(args[0]) is not b.tuple and py_type(args[0]) is not b.list:
        raise Error("TypeError: input_shape requires a native tuple or list")
    cnnclf_run_check_input_rank(len(args[0]))
    if py_type(args[1]) is not b.tuple and py_type(args[1]) is not b.list:
        raise Error("TypeError: conv_channels requires a native tuple or list")
    var input_shape = Python.list()
    var channels = Python.list()
    for i in range(3):  # small-loop(the three declared image dimensions)
        var value = Int(py=args[0][i])
        cnnclf_run_check_input_dim(value)
        _ = py_call(input_shape.append, PythonObject(value))
    for i in range(len(args[1])):  # driver-loop: declared convolution layer widths
        var value = Int(py=args[1][i])
        cnnclf_run_check_channels(value)
        _ = py_call(channels.append, PythonObject(value))
    var nk = xcnn_keys()
    py_setitem(fields, nk[].k_input_shape, py_call(b.tuple, input_shape))
    py_setitem(fields, nk[].k_conv_channels, py_call(b.tuple, channels))
    var integers = Python.tuple("kernel_size", "pool_size", "batch_size", "max_iter")
    var positions: List[Int] = [2, 3, 7, 8]
    for i in range(4):  # small-loop(integer classifier settings)
        var value = Int(py=args[positions[i]])
        cnnclf_run_check_setting(i, value)
        py_setitem(fields, integers[i], PythonObject(value))
    var floats = Python.tuple("learning_rate", "momentum", "weight_decay", "dampening", "eps")
    var indices: List[Int] = [4, 5, 6, 13, 16]
    for i in range(5):  # small-loop(floating classifier optimizer settings)
        py_setitem(fields, floats[i], PythonObject(Float64(py=args[indices[i]])))
    py_setitem(fields, nk[].k_shuffle, PythonObject(Bool(py=args[9])))
    py_setitem(fields, nk[].k_random_state, args[10])
    py_setitem(fields, nk[].k_numeric_mode, args[11])
    var optimizer = String(args[12])
    cnnclf_run_check_optimizer(optimizer)
    py_setitem(fields, nk[].k_optimizer, PythonObject(optimizer))
    var nesterov = Bool(py=args[14])
    if nesterov:  # the scalars are read only when they are checked (as before)
        cnnclf_run_check_nesterov(nesterov, Float64(py=args[5]), Float64(py=args[13]))
    py_setitem(fields, nk[].k_nesterov, PythonObject(nesterov))
    if (py_type(args[15]) is not b.tuple and py_type(args[15]) is not b.list) or len(args[15]) != 2:
        raise Error("ValueError: betas must contain two scalars")
    py_setitem(fields, nk[].k_betas, Python.tuple(Float64(py=args[15][0]), Float64(py=args[15][1])))
    return PythonObject(None)


def cnnclf_weights(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var fields = _classifier_fields(model, context, True)
    var params = _params(fields)
    var out = Python.list()
    for i in range(len(params)):  # driver-loop: publish each learned native array in layer order
        _ = py_call(out.append, _fields(params[i][0])[params[i][1]])
    return out


# ---------------------------------------------------------------- the layers

def _conv_new(c: Int, oc: Int, k: Int, seed: Int, mode: PythonObject, weight: PythonObject, bias: PythonObject,
              context: PythonObject) raises -> PythonObject:
    """`Conv2d(c, oc, k, padding=k // 2, random_state=seed, numeric_mode=mode)`
    holding the learned arrays `weight` and `bias` (drawn by
    `cnnclf_run_init`)."""
    var nk = xcnn_keys()
    var conv = py_call(_b().object.__new__, py_item(context, nk[].k_xcnn_conv2d_class))
    var f = _fields(conv)
    var p = k // 2
    py_setitem(f, nk[].k_in_channels, c)
    py_setitem(f, nk[].k_out_channels, oc)
    py_setitem(f, nk[].k_kernel_size, Python.tuple(k, k))
    py_setitem(f, nk[].k_stride, Python.tuple(1, 1))
    py_setitem(f, nk[].k_dilation, Python.tuple(1, 1))
    py_setitem(f, nk[].k_groups, 1)
    py_setitem(f, nk[].k_padding_mode, "zeros")
    py_setitem(f, nk[].k_padding, Python.tuple(p, p))
    py_setitem(f, nk[].k__pads, Python.tuple(p, p, p, p))
    py_setitem(f, nk[].k__explicit, False)
    py_setitem(f, nk[].k__conv_pad, Python.tuple(p, p))
    py_setitem(f, nk[].k_bias, True)
    py_setitem(f, nk[].k_random_state, seed)
    py_setitem(f, nk[].k_input_shape, PythonObject(None))
    py_setitem(f, nk[].k_numeric_mode, mode)
    py_setitem(f, nk[].k_weight_, weight)
    py_setitem(f, nk[].k_bias_, bias)
    return conv


def _relu_new(mode: PythonObject, context: PythonObject) raises -> PythonObject:
    var nk = xcnn_keys()
    var relu = py_call(_b().object.__new__, py_item(context, nk[].k_xcnn_relu_class))
    py_setitem(_fields(relu), nk[].k_numeric_mode, mode)
    return relu


def _maxpool_new(p: Int, mode: PythonObject, context: PythonObject) raises -> PythonObject:
    """`MaxPool2d(p, numeric_mode=mode)`."""
    var nk = xcnn_keys()
    var pool = py_call(_b().object.__new__, py_item(context, nk[].k_xcnn_maxpool_class))
    var f = _fields(pool)
    py_setitem(f, nk[].k_ceil_mode, False)
    py_setitem(f, nk[].k_divisor_override, PythonObject(None))
    py_setitem(f, nk[].k_kernel_size, Python.tuple(p, p))
    py_setitem(f, nk[].k_stride, Python.tuple(p, p))
    py_setitem(f, nk[].k_padding, Python.tuple(0, 0))
    py_setitem(f, nk[].k_dilation, Python.tuple(1, 1))
    py_setitem(f, nk[].k_count_include_pad, True)
    py_setitem(f, nk[].k_input_shape, PythonObject(None))
    py_setitem(f, nk[].k_numeric_mode, mode)
    return pool


def _linear_new(d_in: Int, d_out: Int, mode: PythonObject, weight: PythonObject, bias: PythonObject,
                context: PythonObject) raises -> PythonObject:
    """`_Linear(d_in, d_out, numeric_mode=mode)` holding `weight`, `bias`."""
    var nk = xcnn_keys()
    var lin = py_call(_b().object.__new__, py_item(context, nk[].k_xcnn_linear_class))
    var f = _fields(lin)
    py_setitem(f, nk[].k_in_features, d_in)
    py_setitem(f, nk[].k_out_features, d_out)
    py_setitem(f, nk[].k_numeric_mode, mode)
    py_setitem(f, nk[].k_weight_, weight)
    py_setitem(f, nk[].k_bias_, bias)
    return lin


def cnnclf_entry(provider: PythonObject) raises -> XseqNeuralEntry:
    """UNPACK: the admitted model provider's descriptor entry and the address
    of its retained call workspace (as bindings/xseq_neural_native_door.mojo
    `_entry`); the caller keeps `provider` referenced through every call."""
    var owner = provider.downcast_value_ptr[NeuralDescriptorProvider]()
    return XseqNeuralEntry(owner[].record[0].entry, Int(owner[].record[0].workspace.unsafe_ptr()))


def _layout(fields: PythonObject, k: Int) raises -> CnnClfLayout:
    """UNPACK `_build(n_classes)`'s words; RUN derives the stack."""
    var b = _b()
    var nk = xcnn_keys()
    var rs = py_item(fields, nk[].k_random_state)
    var seed = 0 if rs is None else Int(py=py_call(b.int, rs))
    var shape = py_item(fields, nk[].k_input_shape)
    var channels = py_item(fields, nk[].k_conv_channels)
    var widths = List[Int]()
    for i in range(len(channels)):  # driver-loop: over the conv blocks (model structure)
        widths.append(Int(py=channels[i]))
    return cnnclf_run_layout(Int(py=shape[0]), Int(py=shape[1]), Int(py=shape[2]),
                             Int(py=py_item(fields, nk[].k_kernel_size)), Int(py=py_item(fields, nk[].k_pool_size)),
                             widths, k, seed)


def cnnclf_publish_layers(fields: PythonObject, layout: CnnClfLayout, arrays: PythonObject,
                          context: PythonObject) raises:
    """PACK `_build`'s layer stack from `layout` (cnnclf_run_layout) onto
    `fields`: Conv2d / ReLU / MaxPool2d blocks and the Linear head, holding
    the learned arrays `arrays` (descriptor order; None entries when the
    caller assigns them next)."""
    var nk = xcnn_keys()
    var shape = py_item(fields, nk[].k_input_shape)
    var c = Int(py=shape[0])
    var ks = Int(py=py_item(fields, nk[].k_kernel_size))
    var ps = Int(py=py_item(fields, nk[].k_pool_size))
    var mode = py_item(fields, nk[].k_numeric_mode)
    var layers = Python.list()
    var blocks = Python.list()
    for i in range(len(layout.widths)):  # driver-loop: over the conv blocks (model structure)
        var oc = layout.widths[i]
        var conv = _conv_new(c, oc, ks, layout.seeds[2 * i], mode, arrays[2 * i], arrays[2 * i + 1], context)
        _ = py_call(layers.append, conv)
        _ = py_call(layers.append, _relu_new(mode, context))
        var pool = PythonObject(None)
        if layout.pools[i] != 0:
            pool = _maxpool_new(ps, mode, context)
            _ = py_call(layers.append, pool)
        _ = py_call(blocks.append, Python.tuple(conv, pool))
        c = oc
    var last = 2 * len(layout.widths)
    var k = layout.shapes[last][0]
    py_setitem(fields, nk[].k__flat, layout.flat)
    py_setitem(fields, nk[].k_head_, _linear_new(layout.flat, k, mode, arrays[last], arrays[last + 1], context))
    py_setitem(fields, nk[].k_layers_, layers)
    py_setitem(fields, nk[].k__blocks, blocks)
    # Every layer of the stack takes the classifier's device (checked at its
    # constructor), as it took its numeric mode above.
    xcnn_device_inherit(_fields(py_item(fields, nk[].k_head_)), fields)
    for i in range(len(layers)):  # driver-loop: over the layer list (model structure)
        xcnn_device_inherit(_fields(layers[i]), fields)


def _build(fields: PythonObject, k: Int, provider: PythonObject, draw: Bool, context: PythonObject) raises:
    """`CNNClassifier._build(n_classes)`: the stack's words in RUN
    (`cnnclf_run_layout`), the learned arrays allocated here and, when
    `draw`, their initial Kaiming words drawn in RUN (`cnnclf_run_init`,
    GIL released) on the model's image `provider`; then the layer objects.
    `draw` is False when the caller assigns every learned array next (adopt):
    nothing is allocated or drawn for them."""
    var layout = _layout(fields, k)
    var arrays = Python.list()
    if draw:
        for i in range(len(layout.shapes)):  # driver-loop: over the learned arrays (model structure)
            var dims = Python.list()
            for j in range(len(layout.shapes[i])):  # small-loop(one array's dimensions)
                _ = py_call(dims.append, layout.shapes[i][j])
            _ = py_call(arrays.append, _empty(py_call(_b().tuple, dims), "float32", context))
        var addresses = _param_addresses(arrays)
        var entry = cnnclf_entry(provider)
        var py = Python()
        with GILReleased(py):
            cnnclf_run_init(entry, layout, addresses)
        _ = provider  # pins the entry's workspace through the call above
    else:
        for i in range(len(layout.shapes)):  # driver-loop: over the learned arrays (model structure)
            _ = py_call(arrays.append, PythonObject(None))
    cnnclf_publish_layers(fields, layout, arrays, context)


def _params(fields: PythonObject) raises -> PythonObject:
    """`_params()`: (layer, weight attr, grad attr) of every parametrized layer."""
    var out = Python.list()
    var nk = xcnn_keys()
    var layers = py_call(_b().list, py_item(fields, nk[].k_layers_))
    _ = py_call(layers.append, py_item(fields, nk[].k_head_))
    for i in range(len(layers)):  # driver-loop: over the layer list (model structure)
        if py_contains(_fields(layers[i]), nk[].k_weight_):
            _ = py_call(out.append, Python.tuple(layers[i], "weight_", "grad_weight_"))
            _ = py_call(out.append, Python.tuple(layers[i], "bias_", "grad_bias_"))
    return out


# ---------------------------------------------------------------- the images

def _images(fields: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_images(X)`."""
    var x = xcnn_f32(source, "X", context)
    var shape = xcnn_shape(x)
    var nk = xcnn_keys()
    var given = py_item(fields, nk[].k_input_shape)
    cnnclf_run_check_rows(shape)
    var images = cnnclf_run_images(shape, [Int(py=given[0]), Int(py=given[1]), Int(py=given[2])])
    if images.reshape:
        return xcnn_reshape(x, images.full, context)
    return x


# ---------------------------------------------------------------- the program descriptor

def _engine(labels: PythonObject) raises -> PythonObject:
    """The x_sequence whole-model door of the model's tier and device (the
    provider the label sort already uses), as the model's route holds it
    (`xcnn_labels`: the identical-mode profile check and `numeric_mode_`
    ran there); an image that did not connect raises its recorded text."""
    # dw-neural-b (2026-11-21): this read `xcnn_label_bindings` by the
    # recorded mode at every fit and predict (the process set, whatever the
    # model's device); the route holds the image of the model's device.
    if labels is None:
        raise Error("ImportError: CNNClassifier requires the compiled label provider")
    if py_type(labels) is _b().str:
        raise Error(String(py=labels))
    return labels


def _plan(fields: PythonObject, n: Int, k: Int) raises -> CnnClfPlan:
    """UNPACK the descriptor head of CNN_FIT / CNN_OPEN: the input plane,
    kernel and pool sizes, and every block's width and pool flag (a block
    pools when _build kept its MaxPool2d)."""
    var nk = xcnn_keys()
    var shape = py_item(fields, nk[].k_input_shape)
    var blocks = py_item(fields, nk[].k__blocks)
    var widths = List[Int]()
    var pools = List[Int]()
    for i in range(len(blocks)):  # driver-loop: over the conv blocks (model structure)
        widths.append(Int(py=py_item(_fields(blocks[i][0]), nk[].k_out_channels)))
    for i in range(len(blocks)):  # driver-loop: over the conv blocks (model structure)
        pools.append(0 if blocks[i][1] is None else 1)
    return CnnClfPlan(n, Int(py=shape[0]), Int(py=shape[1]), Int(py=shape[2]), k, Int(py=py_item(fields, nk[].k_kernel_size)),
                      Int(py=py_item(fields, nk[].k_pool_size)), widths^, pools^)


def _param_addresses(arrays: PythonObject) raises -> List[Int]:
    """The address word of every learned array (descriptor order)."""
    var out = List[Int]()
    for i in range(len(arrays)):  # driver-loop: over the layer parameter arrays (model structure)
        out.append(Int(py=xcnn_addr(arrays[i])))
    return out^


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
    # the identical-mode profile check, and `numeric_mode_`; then the
    # sorted classes and the inverse codes (the sequence owner's label sort,
    # `model_labels` of the x_sequence binding of the model's tier and device)
    var labels = _engine(xcnn_labels(model, fields, context))
    var encoded = neural_encode_labels(args[1], labels, context)
    var nk = xcnn_keys()
    py_setitem(fields, nk[].k_classes_, encoded[0])
    var yi = encoded[1]
    var n = Int(py=x.shape[0])
    cnnclf_run_check_labels(Int(py=yi.size), n)
    var k = len(encoded[0])
    # lane x8b: the refit publishes new weights; close the old resident
    # predict session now rather than at the next predict
    var drop_mods = NativeModules()
    prepared_drop(drop_mods, _PREDICT_SLOT, model)
    _build(fields, k, xseq_neural_provider(labels), True, context)
    var optimizer = String(py=py_item(fields, nk[].k_optimizer))
    var seed64 = XseqStream(py_item(fields, nk[].k_random_state), context).seed
    var batch = Int(py=py_item(fields, nk[].k_batch_size))
    var max_iter = Int(py=py_item(fields, nk[].k_max_iter))
    var losses_all = b.list()
    var curve_out = b.list()
    if max_iter > 0:
        var nsteps = cnnclf_run_fit_steps(n, batch)
        var curve = _empty(Python.tuple(max_iter), "float32", context)
        var steps = _empty(Python.tuple(max_iter * nsteps), "float32", context)
        var arrays = _param_arrays(fields, context)
        var addresses = cnnclf_run_fit_addresses(Int(py=xcnn_addr(x)), Int(py=xcnn_addr(yi)),
                                                 Int(py=xcnn_addr(curve)), Int(py=xcnn_addr(steps)),
                                                 _param_addresses(arrays))
        var plan = _plan(fields, n, k)
        var betas = py_item(fields, nk[].k_betas)
        var settings = CnnClfFitSettings(
            batch, max_iter, Bool(py=py_item(fields, nk[].k_shuffle)), seed64, cnnclf_run_optimizer_word(optimizer),
            Bool(py=py_item(fields, nk[].k_nesterov)), Float64(py=py_item(fields, nk[].k_learning_rate)), Float64(py=py_item(fields, nk[].k_momentum)),
            Float64(py=betas[0]), Float64(py=betas[1]), Float64(py=py_item(fields, nk[].k_eps)), Float64(py=py_item(fields, nk[].k_weight_decay)),
            Float64(py=py_item(fields, nk[].k_dampening)),
        )
        var ints = cnnclf_run_fit_ints(plan, settings)
        var floats = cnnclf_run_fit_floats(settings)
        _ = neural_descriptor_execute(xseq_neural_provider(labels), CNN_FIT, addresses, ints, floats)
        _ = x  # keep the Python objects alive through the native call above
        _ = yi
        # the trained words are in `arrays`; publish them on their layers,
        # read-only (lane x8b: the resident predict session keys on these
        # objects; an in-place edit raises, an assigned array is uploaded
        # again at the next predict)
        var params = _params(fields)
        for i in range(len(params)):  # driver-loop: over the layer parameter list (model structure)
            py_setitem(_fields(params[i][0]), params[i][1], xcnn_readonly(arrays[i], PythonObject(None), context))
        losses_all = py_call(steps.tolist)
        curve_out = py_call(curve.tolist)
    py_setitem(fields, nk[].k_losses_, losses_all)
    py_setitem(fields, nk[].k_loss_curve_, curve_out)
    var ishape = py_item(fields, nk[].k_input_shape)
    py_setitem(fields, nk[].k_n_features_in_, cnnclf_run_features([Int(py=ishape[0]), Int(py=ishape[1]), Int(py=ishape[2])]))
    return model


# ---------------------------------------------------------------- the fitted state on another device

def cnnclf_adopt(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """`model.check`: hand the fitted state of `args[0]` (a fitted classifier
    of the same class and settings on the other device) to `model`. The
    layer stack is built again for `model` (so every layer holds `model`'s
    device) and takes the source's learned arrays themselves (no copy: the
    same words answer on both devices); `classes_` and the loss records are
    the same objects. A structure step with no independent units to split."""
    var source = args[0]
    if py_type(source) is not py_type(model):
        raise Error("TypeError: the fitted state of a CNN classifier goes to a classifier of its own class")
    var fields = _classifier_fields(model, context)
    var given = _classifier_fields(source, context, True)
    var nk = xcnn_keys()
    var classes = py_item(given, nk[].k_classes_)
    var drop_mods = NativeModules()
    prepared_drop(drop_mods, _PREDICT_SLOT, model)
    _build(fields, len(classes), PythonObject(None), False, context)
    var learned = _params(given)
    var mine = _params(fields)
    if len(learned) != len(mine):
        raise Error("ValueError: the two CNN classifiers do not have the same layers")
    for i in range(len(mine)):  # driver-loop: over the layer parameter list (model structure)
        py_setitem(_fields(mine[i][0]), mine[i][1], _fields(learned[i][0])[learned[i][1]])
    py_setitem(fields, nk[].k_classes_, classes)
    _carry(fields, given, nk[].k_losses_)
    _carry(fields, given, nk[].k_loss_curve_)
    _carry(fields, given, nk[].k_n_features_in_)
    return PythonObject(None)


def _carry(fields: PythonObject, given: PythonObject, key: PythonObject) raises:
    # one fit record of `given`, the same object, when it has it
    if py_contains(given, key):
        py_setitem(fields, key, py_item(given, key))


# ---------------------------------------------------------------- predict

# Lane x8b-cnn-resident-predict: the resident predict record. The fitted
# parameters live ONCE in a resident registry session of the model's tier
# (sequence/cnn_classifier_program.mojo CNN_OPEN: every parameter uploaded,
# the forward and its softmax captured as the session's one program). The
# record is (identity key of the tier's binding entries, `_blocks`, `head_`,
# `classes_`, every predict layer and its `weight_` / `bias_`; the session's
# lifetime owner, which closes it through SESSION_CLOSE when dropped). It is
# kept in the prepared registry (weak on the model, outside its `__dict__`):
# released with the model, never pickled or deep-copied (a copy opens its
# own at its first predict), dropped by a refit. It is reused while the key
# matches and every weight is still read-only (the first predict marks them
# so: an in-place edit raises NumPy's read-only error, an assigned array
# misses the key and opens a new session). The record is taken out of the
# registry while a call uses it, so two concurrent predicts never share one
# session (the second opens its own; the record put back last stays).
comptime _PREDICT_SLOT = "xcnn_clf_predict_v2"


def _predict_layers(fields: PythonObject) raises -> PythonObject:
    """The layers whose weights the resident session holds: every block's
    conv, then the head (`_params` order)."""
    var layers = _b().list()
    var nk = xcnn_keys()
    var blocks = py_item(fields, nk[].k__blocks)
    for i in range(len(blocks)):  # driver-loop: over the conv layers
        _ = py_call(layers.append, blocks[i][0])
    _ = py_call(layers.append, py_item(fields, nk[].k_head_))
    return layers


def _weights_frozen(layers: PythonObject, freeze: Bool, context: PythonObject) raises -> Bool:
    """Whether every layer's `weight_` / `bias_` is a read-only exact ndarray
    (marked so first when `freeze`). One flags read per array."""
    var b = _b()
    var nk = xcnn_keys()
    for i in range(len(layers)):  # driver-loop: over the predict layers (model structure)
        var lf = _fields(layers[i])
        for name in ["weight_", "bias_"]:  # small-loop(the two learned arrays)
            var arr = lf[PythonObject(name)]
            if py_type(arr) is not py_item(context, nk[].k_ndarray):
                return False
            var flags = arr.flags
            if freeze and Bool(py=flags.writeable):
                _ = py_call(b.setattr, flags, "writeable", PythonObject(False))
            if Bool(py=flags.writeable):
                return False
    return True


def _predict_key(mut mods: NativeModules, fields: PythonObject, layers: PythonObject,
                 labels: PythonObject) raises -> NativeIdentityKey:
    var key = NativeIdentityKey(mods)
    key.add_identity(labels)
    var nk = xcnn_keys()
    key.add_identity(py_item(fields, nk[].k__blocks))
    key.add_identity(py_item(fields, nk[].k_head_))
    key.add_identity(py_item(fields, nk[].k_classes_))
    for i in range(len(layers)):  # driver-loop: over the predict layers (model structure)
        var lf = _fields(layers[i])
        key.add_identity(layers[i])
        key.add_identity(py_item(lf, nk[].k_weight_))
        key.add_identity(py_item(lf, nk[].k_bias_))
    return key^


def _predict_key_matches(record: PythonObject, fields: PythonObject, layers: PythonObject,
                         labels: PythonObject) raises -> Bool:
    if py_type(record) is not _b().tuple or len(record) != 2:
        return False
    var key = record[0].downcast_value_ptr[NativeIdentityKey]()
    var offset = 0
    var nk = xcnn_keys()
    if not key[].matches_identity(labels, offset) or not key[].matches_identity(py_item(fields, nk[].k__blocks), offset):
        return False
    if not key[].matches_identity(py_item(fields, nk[].k_head_), offset) or not key[].matches_identity(py_item(fields, nk[].k_classes_), offset):
        return False
    for i in range(len(layers)):  # driver-loop: over the predict layers (model structure)
        var lf = _fields(layers[i])
        if (not key[].matches_identity(layers[i], offset) or not key[].matches_identity(py_item(lf, nk[].k_weight_), offset)
                or not key[].matches_identity(py_item(lf, nk[].k_bias_), offset)):
            return False
    return key[].complete(offset)


def _open_resident(fields: PythonObject, provider: PythonObject, k: Int, context: PythonObject) raises -> PythonObject:
    """One CNN_OPEN call: the resident session holding every fitted
    parameter; returns its lifetime owner (SESSION_CLOSE on release). The
    registry mutation keeps the GIL, as SESSION_OPEN does."""
    var arrays = _param_arrays(fields, context)
    var scope = List[Int](length=2, fill=0)
    var addresses = cnnclf_run_open_addresses(Int(scope.unsafe_ptr()), _param_addresses(arrays))
    var ints = cnnclf_run_open_ints(_plan(fields, 1, k))
    var sid = neural_descriptor_execute(provider, CNN_OPEN, addresses, ints, List[Float64](), False)
    _ = arrays  # keep the Python objects alive through the native call above
    return neural_descriptor_adopt(provider, sid, SESSION_CLOSE, scope^)


def _run_predict(model: PythonObject, source: PythonObject, codes: Bool, context: PythonObject) raises -> PythonObject:
    """`_run_predict(X, codes)`: probabilities (n, k) float32, or with
    `codes` each row's class index (int32): one CNN_PREDICT call on the
    model's resident session (opened first when the record misses)."""
    var fields = _classifier_fields(model, context, True)
    var x = _contiguous(_images(fields, source, context), context)
    var n = Int(py=x.shape[0])
    var nk = xcnn_keys()
    var k = len(py_item(fields, nk[].k_classes_))
    var held = xcnn_labels(model, fields, context)
    var out = _empty(Python.tuple(n), "int32", context) if codes else _empty(Python.tuple(n, k), "float32", context)
    if n == 0:
        return out
    var labels = _engine(held)
    var provider = xseq_neural_provider(labels)
    var mods = NativeModules()
    var layers = _predict_layers(fields)
    var record = prepared_get(mods, _PREDICT_SLOT, model)
    if record is not None:
        prepared_drop(mods, _PREDICT_SLOT, model)  # this call owns it now
        if not _predict_key_matches(record, fields, layers, labels) or not _weights_frozen(layers, False, context):
            record = PythonObject(None)  # its lifetime owner closes the stale session
    var keep = True
    if record is None:
        # read-only first, so the words uploaded are the words kept
        keep = _weights_frozen(layers, True, context)
        var lifetime = _open_resident(fields, provider, k, context)
        var key = _predict_key(mods, fields, layers, labels)
        keep = keep and not key.retains(model)
        record = Python.tuple(PythonObject(alloc=key^), lifetime)
    var resource = record[1].downcast_value_ptr[NativeDescriptorResource]()
    cnnclf_run_check_session(resource[].handle, len(resource[].context))
    var addresses: List[Int] = [Int(py=xcnn_addr(x)), Int(py=xcnn_addr(out))]
    var ints = cnnclf_run_predict_ints(resource[].context[0], resource[].context[1], resource[].handle, n, codes)
    _ = neural_descriptor_execute(provider, CNN_PREDICT, addresses, ints, List[Float64]())
    _ = x  # keep the Python objects alive through the native call above
    # the session stays for the next predict while every weight is read-only;
    # otherwise the record goes here and its owner closes the session
    if keep:
        prepared_put(mods, _PREDICT_SLOT, model, record)
    return out


def cnnclf_predict_proba(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    return _run_predict(model, args[0], False, context)


def cnnclf_predict(model: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var codes = _run_predict(model, args[0], True, context)
    var nk = xcnn_keys()
    return neural_decode_labels(py_item(_fields(model), nk[].k_classes_), codes, context)
