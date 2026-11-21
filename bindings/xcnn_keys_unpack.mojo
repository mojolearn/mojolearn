# SPDX-License-Identifier: Apache-2.0
# The xcnn family's held name keys and operation codes (bindings/py_keys.mojo
# has the form and its cost rule). Written by the boundary codemod of lane
# fx-boundary-core; a name the family starts to read is added in the three
# places below (field, None, fill), and the slot name keeps its V<n> until a
# wheel that holds the old layout could meet this one in a process (bump
# `V<n>` and the version together then: core/shared_slot.mojo).
from std.python import Python, PythonObject
from core.shared_slot import SharedSlot
from bindings.py_keys import py_key, py_op_add

comptime XCNN_OPN_TENSOR_FINALIZE = 0
comptime XCNN_OPN_TENSOR_TO = 1
comptime XCNN_OPN_TENSOR_NEW = 2
comptime XCNN_OPN_TENSOR_NEW_OUT = 3
comptime XCNN_OPN_TENSOR_WRAP = 4
comptime XCNN_OPN_TENSOR_INIT = 5
comptime XCNN_OPN_TENSOR_SIZE = 6
comptime XCNN_OPN_TENSOR_NDIM = 7
comptime XCNN_OPN_TENSOR_NUMPY = 8
comptime XCNN_OPN_TENSOR_ARRAY = 9
comptime XCNN_OPN_TENSOR_RESHAPE = 10
comptime XCNN_OPN_DEV_FINALIZE = 11
comptime XCNN_OPN_BLOCK_INIT = 12
comptime XCNN_OPN_BLOCK_TRAIN = 13
comptime XCNN_OPN_BLOCK_BACKWARD = 14
comptime XCNN_OPN_BLOCK_PARAMETERS = 15
comptime XCNN_OPN_BLOCK_FIT = 16
comptime XCNN_OPN_BLOCK_FORWARD = 17
comptime XCNN_OPN_BLOCK_TRANSFORM = 18
comptime XCNN_OPN_CONV_INIT = 19
comptime XCNN_OPN_CONV_INIT1D = 20
comptime XCNN_OPN_CONV_BACKWARD = 21
comptime XCNN_OPN_CONV_SET_WEIGHTS = 22
comptime XCNN_OPN_CONV_SET_WEIGHTS1D = 23
comptime XCNN_OPN_CONV_PIN = 24
comptime XCNN_OPN_CONV_UNPIN = 25
comptime XCNN_OPN_CONV_GRAD_WEIGHT = 26
comptime XCNN_OPN_CONV_GRAD_BIAS = 27
comptime XCNN_OPN_CONV_GRAD_WEIGHT_SET = 28
comptime XCNN_OPN_CONV_GRAD_BIAS_SET = 29
comptime XCNN_OPN_CONV_WEIGHT1D = 30
comptime XCNN_OPN_CONV_FIT = 31
comptime XCNN_OPN_CONV_FORWARD = 32
comptime XCNN_OPN_CONV_TRANSFORM = 33
comptime XCNN_OPN_CNNCLF_INIT = 34
comptime XCNN_OPN_CNNCLF_WEIGHTS = 35
comptime XCNN_OPN_CNNCLF_FIT = 36
comptime XCNN_OPN_CNNCLF_PREDICT = 37
comptime XCNN_OPN_CNNCLF_PREDICT_PROBA = 38
comptime XCNN_OPN_GCN_INIT = 39
comptime XCNN_OPN_SAGE_INIT = 40
comptime XCNN_OPN_GCN_CACHED = 41
comptime XCNN_OPN_SAGE_CACHED = 42
comptime XCNN_OPN_GCN_CACHED_SET = 43
comptime XCNN_OPN_SAGE_CACHED_SET = 44
comptime XCNN_OPN_GCN_FORWARD = 45
comptime XCNN_OPN_SAGE_FORWARD = 46
comptime XCNN_OPN_GCN_BACKWARD = 47
comptime XCNN_OPN_SAGE_BACKWARD = 48
comptime XCNN_OPN_LINEAR_INIT = 49
comptime XCNN_OPN_LINEAR_FORWARD = 50
comptime XCNN_OPN_LINEAR_BACKWARD = 51
comptime XCNN_OPN_RELU_INIT = 52
comptime XCNN_OPN_RELU_FORWARD = 53
comptime XCNN_OPN_RELU_BACKWARD = 54
comptime XCNN_OPN_BN_INIT = 55
comptime XCNN_OPN_BN_TRAIN = 56
comptime XCNN_OPN_BN_RUNNING_GET = 57
comptime XCNN_OPN_BN_RUNNING_SET = 58
comptime XCNN_OPN_BN_GRAD_GET = 59
comptime XCNN_OPN_BN_GRAD_SET = 60
comptime XCNN_OPN_BN_BACKWARD = 61
comptime XCNN_OPN_BN_FORWARD = 62
comptime XCNN_OPN_BN_TRANSFORM = 63
comptime XCNN_OPN_BN_FIT = 64
comptime XCNN_OPN_DROPOUT_INIT = 65
comptime XCNN_OPN_DROPOUT_TRAIN = 66
comptime XCNN_OPN_DROPOUT_MASK_GET = 67
comptime XCNN_OPN_DROPOUT_MASK_SET = 68
comptime XCNN_OPN_DROPOUT_BACKWARD = 69
comptime XCNN_OPN_DROPOUT_FORWARD = 70
comptime XCNN_OPN_DROPOUT_TRANSFORM = 71
comptime XCNN_OPN_ADAPTIVE_INIT = 72
comptime XCNN_OPN_ADAPTIVE_INDICES_GET = 73
comptime XCNN_OPN_ADAPTIVE_BACKWARD = 74
comptime XCNN_OPN_ADAPTIVE_DEVICE_IO = 75
comptime XCNN_OPN_ADAPTIVE_FORWARD = 76
comptime XCNN_OPN_ADAPTIVE_TRANSFORM = 77
comptime XCNN_OPN_POOL_INIT = 78
comptime XCNN_OPN_POOL_BACKWARD = 79
comptime XCNN_OPN_POOL_FIT = 80
comptime XCNN_OPN_POOL_INDICES_GET = 81
comptime XCNN_OPN_POOL_INDICES_SET = 82
comptime XCNN_OPN_POOL_FORWARD = 83
comptime XCNN_OPN_POOL_TRANSFORM = 84
comptime XCNN_OPN_CNNCLF_ADOPT = 85
# lane api-torch round 2: state_dict / load_state_dict of Conv, BatchNorm and BasicBlock
comptime XCNN_OPN_LAYER_STATE_DICT = 86
comptime XCNN_OPN_LAYER_LOAD_STATE_DICT = 87
# lane api-torch round 3: torch-form parameters() / named_parameters() of Conv, BatchNorm and BasicBlock
comptime XCNN_OPN_LAYER_PARAMETERS = 88
comptime XCNN_OPN_LAYER_NAMED_PARAMETERS = 89
# lane api-torch round 3: GCNConv / SAGEConv state_dict / load_state_dict (PyG names)
comptime XCNN_OPN_GRAPH_STATE_DICT = 90
comptime XCNN_OPN_GRAPH_LOAD_STATE_DICT = 91


struct XcnnKeys(Defaultable, Movable):
    # Immutable interface names only: no model, state or user data.
    var ready: Bool
    var ops: PythonObject
    var k___del__: PythonObject
    var k___dict__: PythonObject
    var k___getattr__: PythonObject
    var k___mro__: PythonObject
    var k___name__: PythonObject
    var k__aux: PythonObject
    var k__aux_dev: PythonObject
    var k__aux_k: PythonObject
    var k__blocks: PythonObject
    var k__chain: PythonObject
    var k__conv_pad: PythonObject
    var k__den: PythonObject
    var k__dev: PythonObject
    var k__explicit: PythonObject
    var k__flat: PythonObject
    var k__g: PythonObject
    var k__g_dev: PythonObject
    var k__gcache: PythonObject
    var k__gdev_for: PythonObject
    var k__gdev_g: PythonObject
    var k__gw_1d: PythonObject
    var k__gw_dev: PythonObject
    var k__idx_dev: PythonObject
    var k__idx_host: PythonObject
    var k__idx_shape: PythonObject
    var k__layer: PythonObject
    var k__mask_dev: PythonObject
    var k__mask_host: PythonObject
    var k__mask_shape: PythonObject
    var k__max: PythonObject
    var k__max_aux: PythonObject
    var k__mode: PythonObject
    var k__nograph: PythonObject
    var k__pads: PythonObject
    var k__pin_src: PythonObject
    var k__pins: PythonObject
    var k__pre: PythonObject
    var k__relu_p: PythonObject
    var k__run_dev: PythonObject
    var k__run_stale: PythonObject
    var k__shape: PythonObject
    var k__vals: PythonObject
    var k__x: PythonObject
    var k__x3shape: PythonObject
    var k__xdev: PythonObject
    var k__xdev_dev: PythonObject
    var k__xs: PythonObject
    var k__y: PythonObject
    var k__ydev: PythonObject
    var k_add_self_loops: PythonObject
    var k_admission_context: PythonObject
    var k_affine: PythonObject
    var k_aggr: PythonObject
    var k_b: PythonObject
    var k_b1: PythonObject
    var k_batch_size: PythonObject
    var k_betas: PythonObject
    var k_bias: PythonObject
    var k_bias_: PythonObject
    var k_bn1: PythonObject
    var k_bn2: PythonObject
    var k_c1: PythonObject
    var k_c2: PythonObject
    var k_cached: PythonObject
    var k_calls_: PythonObject
    var k_ceil_mode: PythonObject
    var k_classes_: PythonObject
    var k_conv1: PythonObject
    var k_conv2: PythonObject
    var k_conv_channels: PythonObject
    var k_count_include_pad: PythonObject
    var k_csr_f: PythonObject
    var k_csr_t: PythonObject
    var k_d1: PythonObject
    var k_dampening: PythonObject
    var k_dilation: PythonObject
    var k_divisor_override: PythonObject
    var k_downsample: PythonObject
    var k_eps: PythonObject
    var k_float32: PythonObject
    var k_grad_bias_: PythonObject
    var k_grad_weight_: PythonObject
    var k_grad_weight_r_: PythonObject
    var k_groups: PythonObject
    var k_h: PythonObject
    var k_head_: PythonObject
    var k_improved: PythonObject
    var k_in_channels: PythonObject
    var k_in_features: PythonObject
    var k_input_shape: PythonObject
    var k_int32: PythonObject
    var k_kernel_size: PythonObject
    var k_layers_: PythonObject
    var k_learning_rate: PythonObject
    var k_lin: PythonObject
    var k_lin_l: PythonObject
    var k_loss_curve_: PythonObject
    var k_losses_: PythonObject
    var k_max_iter: PythonObject
    var k_module: PythonObject
    var k_momentum: PythonObject
    var k_n: PythonObject
    var k_n_features_in_: PythonObject
    var k_ndarray: PythonObject
    var k_nesterov: PythonObject
    var k_nnz: PythonObject
    var k_normalize: PythonObject
    var k_num_batches_tracked_: PythonObject
    var k_num_features: PythonObject
    var k_numeric_mode: PythonObject
    var k_numeric_mode_: PythonObject
    var k_ones: PythonObject
    var k_optimizer: PythonObject
    var k_order_f: PythonObject
    var k_order_t: PythonObject
    var k_out_channels: PythonObject
    var k_out_features: PythonObject
    var k_p: PythonObject
    var k_padding: PythonObject
    var k_padding_mode: PythonObject
    var k_pool_size: PythonObject
    var k_project: PythonObject
    var k_r1: PythonObject
    var k_random_state: PythonObject
    var k_relu1: PythonObject
    var k_relu2: PythonObject
    var k_root_weight: PythonObject
    var k_running_mean_: PythonObject
    var k_running_var_: PythonObject
    var k_s: PythonObject
    var k_s1: PythonObject
    var k_s2: PythonObject
    var k_scalar_types: PythonObject
    var k_sd: PythonObject
    var k_shape: PythonObject
    var k_shuffle: PythonObject
    var k_stride: PythonObject
    var k_track_running_stats: PythonObject
    var k_training: PythonObject
    var k_weight_: PythonObject
    var k_weight_decay: PythonObject
    var k_weight_r_: PythonObject
    var k_x: PythonObject
    var k_xcnn_array_context: PythonObject
    var k_xcnn_avgpool_class: PythonObject
    var k_xcnn_bn_class: PythonObject
    var k_xcnn_conv1d_class: PythonObject
    var k_xcnn_conv2d_class: PythonObject
    var k_xcnn_dev_class: PythonObject
    var k_xcnn_linear_class: PythonObject
    var k_xcnn_maxpool_class: PythonObject
    var k_xcnn_numpy_import_tried: PythonObject
    var k_xcnn_relu_class: PythonObject
    var k_xcnn_tensor_class: PythonObject
    var k_xcnn_tensor_finalizer: PythonObject
    var k_xcnn_tensor_slots: PythonObject
    var k_xh: PythonObject
    var k_xt: PythonObject

    def __init__(out self):
        self.ready = False
        self.ops = PythonObject(None)
        self.k___del__ = PythonObject(None)
        self.k___dict__ = PythonObject(None)
        self.k___getattr__ = PythonObject(None)
        self.k___mro__ = PythonObject(None)
        self.k___name__ = PythonObject(None)
        self.k__aux = PythonObject(None)
        self.k__aux_dev = PythonObject(None)
        self.k__aux_k = PythonObject(None)
        self.k__blocks = PythonObject(None)
        self.k__chain = PythonObject(None)
        self.k__conv_pad = PythonObject(None)
        self.k__den = PythonObject(None)
        self.k__dev = PythonObject(None)
        self.k__explicit = PythonObject(None)
        self.k__flat = PythonObject(None)
        self.k__g = PythonObject(None)
        self.k__g_dev = PythonObject(None)
        self.k__gcache = PythonObject(None)
        self.k__gdev_for = PythonObject(None)
        self.k__gdev_g = PythonObject(None)
        self.k__gw_1d = PythonObject(None)
        self.k__gw_dev = PythonObject(None)
        self.k__idx_dev = PythonObject(None)
        self.k__idx_host = PythonObject(None)
        self.k__idx_shape = PythonObject(None)
        self.k__layer = PythonObject(None)
        self.k__mask_dev = PythonObject(None)
        self.k__mask_host = PythonObject(None)
        self.k__mask_shape = PythonObject(None)
        self.k__max = PythonObject(None)
        self.k__max_aux = PythonObject(None)
        self.k__mode = PythonObject(None)
        self.k__nograph = PythonObject(None)
        self.k__pads = PythonObject(None)
        self.k__pin_src = PythonObject(None)
        self.k__pins = PythonObject(None)
        self.k__pre = PythonObject(None)
        self.k__relu_p = PythonObject(None)
        self.k__run_dev = PythonObject(None)
        self.k__run_stale = PythonObject(None)
        self.k__shape = PythonObject(None)
        self.k__vals = PythonObject(None)
        self.k__x = PythonObject(None)
        self.k__x3shape = PythonObject(None)
        self.k__xdev = PythonObject(None)
        self.k__xdev_dev = PythonObject(None)
        self.k__xs = PythonObject(None)
        self.k__y = PythonObject(None)
        self.k__ydev = PythonObject(None)
        self.k_add_self_loops = PythonObject(None)
        self.k_admission_context = PythonObject(None)
        self.k_affine = PythonObject(None)
        self.k_aggr = PythonObject(None)
        self.k_b = PythonObject(None)
        self.k_b1 = PythonObject(None)
        self.k_batch_size = PythonObject(None)
        self.k_betas = PythonObject(None)
        self.k_bias = PythonObject(None)
        self.k_bias_ = PythonObject(None)
        self.k_bn1 = PythonObject(None)
        self.k_bn2 = PythonObject(None)
        self.k_c1 = PythonObject(None)
        self.k_c2 = PythonObject(None)
        self.k_cached = PythonObject(None)
        self.k_calls_ = PythonObject(None)
        self.k_ceil_mode = PythonObject(None)
        self.k_classes_ = PythonObject(None)
        self.k_conv1 = PythonObject(None)
        self.k_conv2 = PythonObject(None)
        self.k_conv_channels = PythonObject(None)
        self.k_count_include_pad = PythonObject(None)
        self.k_csr_f = PythonObject(None)
        self.k_csr_t = PythonObject(None)
        self.k_d1 = PythonObject(None)
        self.k_dampening = PythonObject(None)
        self.k_dilation = PythonObject(None)
        self.k_divisor_override = PythonObject(None)
        self.k_downsample = PythonObject(None)
        self.k_eps = PythonObject(None)
        self.k_float32 = PythonObject(None)
        self.k_grad_bias_ = PythonObject(None)
        self.k_grad_weight_ = PythonObject(None)
        self.k_grad_weight_r_ = PythonObject(None)
        self.k_groups = PythonObject(None)
        self.k_h = PythonObject(None)
        self.k_head_ = PythonObject(None)
        self.k_improved = PythonObject(None)
        self.k_in_channels = PythonObject(None)
        self.k_in_features = PythonObject(None)
        self.k_input_shape = PythonObject(None)
        self.k_int32 = PythonObject(None)
        self.k_kernel_size = PythonObject(None)
        self.k_layers_ = PythonObject(None)
        self.k_learning_rate = PythonObject(None)
        self.k_lin = PythonObject(None)
        self.k_lin_l = PythonObject(None)
        self.k_loss_curve_ = PythonObject(None)
        self.k_losses_ = PythonObject(None)
        self.k_max_iter = PythonObject(None)
        self.k_module = PythonObject(None)
        self.k_momentum = PythonObject(None)
        self.k_n = PythonObject(None)
        self.k_n_features_in_ = PythonObject(None)
        self.k_ndarray = PythonObject(None)
        self.k_nesterov = PythonObject(None)
        self.k_nnz = PythonObject(None)
        self.k_normalize = PythonObject(None)
        self.k_num_batches_tracked_ = PythonObject(None)
        self.k_num_features = PythonObject(None)
        self.k_numeric_mode = PythonObject(None)
        self.k_numeric_mode_ = PythonObject(None)
        self.k_ones = PythonObject(None)
        self.k_optimizer = PythonObject(None)
        self.k_order_f = PythonObject(None)
        self.k_order_t = PythonObject(None)
        self.k_out_channels = PythonObject(None)
        self.k_out_features = PythonObject(None)
        self.k_p = PythonObject(None)
        self.k_padding = PythonObject(None)
        self.k_padding_mode = PythonObject(None)
        self.k_pool_size = PythonObject(None)
        self.k_project = PythonObject(None)
        self.k_r1 = PythonObject(None)
        self.k_random_state = PythonObject(None)
        self.k_relu1 = PythonObject(None)
        self.k_relu2 = PythonObject(None)
        self.k_root_weight = PythonObject(None)
        self.k_running_mean_ = PythonObject(None)
        self.k_running_var_ = PythonObject(None)
        self.k_s = PythonObject(None)
        self.k_s1 = PythonObject(None)
        self.k_s2 = PythonObject(None)
        self.k_scalar_types = PythonObject(None)
        self.k_sd = PythonObject(None)
        self.k_shape = PythonObject(None)
        self.k_shuffle = PythonObject(None)
        self.k_stride = PythonObject(None)
        self.k_track_running_stats = PythonObject(None)
        self.k_training = PythonObject(None)
        self.k_weight_ = PythonObject(None)
        self.k_weight_decay = PythonObject(None)
        self.k_weight_r_ = PythonObject(None)
        self.k_x = PythonObject(None)
        self.k_xcnn_array_context = PythonObject(None)
        self.k_xcnn_avgpool_class = PythonObject(None)
        self.k_xcnn_bn_class = PythonObject(None)
        self.k_xcnn_conv1d_class = PythonObject(None)
        self.k_xcnn_conv2d_class = PythonObject(None)
        self.k_xcnn_dev_class = PythonObject(None)
        self.k_xcnn_linear_class = PythonObject(None)
        self.k_xcnn_maxpool_class = PythonObject(None)
        self.k_xcnn_numpy_import_tried = PythonObject(None)
        self.k_xcnn_relu_class = PythonObject(None)
        self.k_xcnn_tensor_class = PythonObject(None)
        self.k_xcnn_tensor_finalizer = PythonObject(None)
        self.k_xcnn_tensor_slots = PythonObject(None)
        self.k_xh = PythonObject(None)
        self.k_xt = PythonObject(None)


comptime _XCNN_KEYS = SharedSlot[XcnnKeys, "MojolearnXcnnKeysV1"]


def _xcnn_keys_fill(nk: UnsafePointer[XcnnKeys, MutUntrackedOrigin]) raises:
    # Once per process. `ready` is set last, so a fill that raised is run
    # again whole by the next call.
    nk[].k___del__ = py_key("__del__")
    nk[].k___dict__ = py_key("__dict__")
    nk[].k___getattr__ = py_key("__getattr__")
    nk[].k___mro__ = py_key("__mro__")
    nk[].k___name__ = py_key("__name__")
    nk[].k__aux = py_key("_aux")
    nk[].k__aux_dev = py_key("_aux_dev")
    nk[].k__aux_k = py_key("_aux_k")
    nk[].k__blocks = py_key("_blocks")
    nk[].k__chain = py_key("_chain")
    nk[].k__conv_pad = py_key("_conv_pad")
    nk[].k__den = py_key("_den")
    nk[].k__dev = py_key("_dev")
    nk[].k__explicit = py_key("_explicit")
    nk[].k__flat = py_key("_flat")
    nk[].k__g = py_key("_g")
    nk[].k__g_dev = py_key("_g_dev")
    nk[].k__gcache = py_key("_gcache")
    nk[].k__gdev_for = py_key("_gdev_for")
    nk[].k__gdev_g = py_key("_gdev_g")
    nk[].k__gw_1d = py_key("_gw_1d")
    nk[].k__gw_dev = py_key("_gw_dev")
    nk[].k__idx_dev = py_key("_idx_dev")
    nk[].k__idx_host = py_key("_idx_host")
    nk[].k__idx_shape = py_key("_idx_shape")
    nk[].k__layer = py_key("_layer")
    nk[].k__mask_dev = py_key("_mask_dev")
    nk[].k__mask_host = py_key("_mask_host")
    nk[].k__mask_shape = py_key("_mask_shape")
    nk[].k__max = py_key("_max")
    nk[].k__max_aux = py_key("_max_aux")
    nk[].k__mode = py_key("_mode")
    nk[].k__nograph = py_key("_nograph")
    nk[].k__pads = py_key("_pads")
    nk[].k__pin_src = py_key("_pin_src")
    nk[].k__pins = py_key("_pins")
    nk[].k__pre = py_key("_pre")
    nk[].k__relu_p = py_key("_relu_p")
    nk[].k__run_dev = py_key("_run_dev")
    nk[].k__run_stale = py_key("_run_stale")
    nk[].k__shape = py_key("_shape")
    nk[].k__vals = py_key("_vals")
    nk[].k__x = py_key("_x")
    nk[].k__x3shape = py_key("_x3shape")
    nk[].k__xdev = py_key("_xdev")
    nk[].k__xdev_dev = py_key("_xdev_dev")
    nk[].k__xs = py_key("_xs")
    nk[].k__y = py_key("_y")
    nk[].k__ydev = py_key("_ydev")
    nk[].k_add_self_loops = py_key("add_self_loops")
    nk[].k_admission_context = py_key("admission_context")
    nk[].k_affine = py_key("affine")
    nk[].k_aggr = py_key("aggr")
    nk[].k_b = py_key("b")
    nk[].k_b1 = py_key("b1")
    nk[].k_batch_size = py_key("batch_size")
    nk[].k_betas = py_key("betas")
    nk[].k_bias = py_key("bias")
    nk[].k_bias_ = py_key("bias_")
    nk[].k_bn1 = py_key("bn1")
    nk[].k_bn2 = py_key("bn2")
    nk[].k_c1 = py_key("c1")
    nk[].k_c2 = py_key("c2")
    nk[].k_cached = py_key("cached")
    nk[].k_calls_ = py_key("calls_")
    nk[].k_ceil_mode = py_key("ceil_mode")
    nk[].k_classes_ = py_key("classes_")
    nk[].k_conv1 = py_key("conv1")
    nk[].k_conv2 = py_key("conv2")
    nk[].k_conv_channels = py_key("conv_channels")
    nk[].k_count_include_pad = py_key("count_include_pad")
    nk[].k_csr_f = py_key("csr_f")
    nk[].k_csr_t = py_key("csr_t")
    nk[].k_d1 = py_key("d1")
    nk[].k_dampening = py_key("dampening")
    nk[].k_dilation = py_key("dilation")
    nk[].k_divisor_override = py_key("divisor_override")
    nk[].k_downsample = py_key("downsample")
    nk[].k_eps = py_key("eps")
    nk[].k_float32 = py_key("float32")
    nk[].k_grad_bias_ = py_key("grad_bias_")
    nk[].k_grad_weight_ = py_key("grad_weight_")
    nk[].k_grad_weight_r_ = py_key("grad_weight_r_")
    nk[].k_groups = py_key("groups")
    nk[].k_h = py_key("h")
    nk[].k_head_ = py_key("head_")
    nk[].k_improved = py_key("improved")
    nk[].k_in_channels = py_key("in_channels")
    nk[].k_in_features = py_key("in_features")
    nk[].k_input_shape = py_key("input_shape")
    nk[].k_int32 = py_key("int32")
    nk[].k_kernel_size = py_key("kernel_size")
    nk[].k_layers_ = py_key("layers_")
    nk[].k_learning_rate = py_key("learning_rate")
    nk[].k_lin = py_key("lin")
    nk[].k_lin_l = py_key("lin_l")
    nk[].k_loss_curve_ = py_key("loss_curve_")
    nk[].k_losses_ = py_key("losses_")
    nk[].k_max_iter = py_key("max_iter")
    nk[].k_module = py_key("module")
    nk[].k_momentum = py_key("momentum")
    nk[].k_n = py_key("n")
    nk[].k_n_features_in_ = py_key("n_features_in_")
    nk[].k_ndarray = py_key("ndarray")
    nk[].k_nesterov = py_key("nesterov")
    nk[].k_nnz = py_key("nnz")
    nk[].k_normalize = py_key("normalize")
    nk[].k_num_batches_tracked_ = py_key("num_batches_tracked_")
    nk[].k_num_features = py_key("num_features")
    nk[].k_numeric_mode = py_key("numeric_mode")
    nk[].k_numeric_mode_ = py_key("numeric_mode_")
    nk[].k_ones = py_key("ones")
    nk[].k_optimizer = py_key("optimizer")
    nk[].k_order_f = py_key("order_f")
    nk[].k_order_t = py_key("order_t")
    nk[].k_out_channels = py_key("out_channels")
    nk[].k_out_features = py_key("out_features")
    nk[].k_p = py_key("p")
    nk[].k_padding = py_key("padding")
    nk[].k_padding_mode = py_key("padding_mode")
    nk[].k_pool_size = py_key("pool_size")
    nk[].k_project = py_key("project")
    nk[].k_r1 = py_key("r1")
    nk[].k_random_state = py_key("random_state")
    nk[].k_relu1 = py_key("relu1")
    nk[].k_relu2 = py_key("relu2")
    nk[].k_root_weight = py_key("root_weight")
    nk[].k_running_mean_ = py_key("running_mean_")
    nk[].k_running_var_ = py_key("running_var_")
    nk[].k_s = py_key("s")
    nk[].k_s1 = py_key("s1")
    nk[].k_s2 = py_key("s2")
    nk[].k_scalar_types = py_key("scalar_types")
    nk[].k_sd = py_key("sd")
    nk[].k_shape = py_key("shape")
    nk[].k_shuffle = py_key("shuffle")
    nk[].k_stride = py_key("stride")
    nk[].k_track_running_stats = py_key("track_running_stats")
    nk[].k_training = py_key("training")
    nk[].k_weight_ = py_key("weight_")
    nk[].k_weight_decay = py_key("weight_decay")
    nk[].k_weight_r_ = py_key("weight_r_")
    nk[].k_x = py_key("x")
    nk[].k_xcnn_array_context = py_key("xcnn_array_context")
    nk[].k_xcnn_avgpool_class = py_key("xcnn_avgpool_class")
    nk[].k_xcnn_bn_class = py_key("xcnn_bn_class")
    nk[].k_xcnn_conv1d_class = py_key("xcnn_conv1d_class")
    nk[].k_xcnn_conv2d_class = py_key("xcnn_conv2d_class")
    nk[].k_xcnn_dev_class = py_key("xcnn_dev_class")
    nk[].k_xcnn_linear_class = py_key("xcnn_linear_class")
    nk[].k_xcnn_maxpool_class = py_key("xcnn_maxpool_class")
    nk[].k_xcnn_numpy_import_tried = py_key("xcnn_numpy_import_tried")
    nk[].k_xcnn_relu_class = py_key("xcnn_relu_class")
    nk[].k_xcnn_tensor_class = py_key("xcnn_tensor_class")
    nk[].k_xcnn_tensor_finalizer = py_key("xcnn_tensor_finalizer")
    nk[].k_xcnn_tensor_slots = py_key("xcnn_tensor_slots")
    nk[].k_xh = py_key("xh")
    nk[].k_xt = py_key("xt")
    var ops = Python.dict()
    py_op_add(ops, "tensor_finalize", XCNN_OPN_TENSOR_FINALIZE)
    py_op_add(ops, "tensor_to", XCNN_OPN_TENSOR_TO)
    py_op_add(ops, "tensor_new", XCNN_OPN_TENSOR_NEW)
    py_op_add(ops, "tensor_new_out", XCNN_OPN_TENSOR_NEW_OUT)
    py_op_add(ops, "tensor_wrap", XCNN_OPN_TENSOR_WRAP)
    py_op_add(ops, "tensor_init", XCNN_OPN_TENSOR_INIT)
    py_op_add(ops, "tensor_size", XCNN_OPN_TENSOR_SIZE)
    py_op_add(ops, "tensor_ndim", XCNN_OPN_TENSOR_NDIM)
    py_op_add(ops, "tensor_numpy", XCNN_OPN_TENSOR_NUMPY)
    py_op_add(ops, "tensor_array", XCNN_OPN_TENSOR_ARRAY)
    py_op_add(ops, "tensor_reshape", XCNN_OPN_TENSOR_RESHAPE)
    py_op_add(ops, "dev_finalize", XCNN_OPN_DEV_FINALIZE)
    py_op_add(ops, "block_init", XCNN_OPN_BLOCK_INIT)
    py_op_add(ops, "block_train", XCNN_OPN_BLOCK_TRAIN)
    py_op_add(ops, "block_backward", XCNN_OPN_BLOCK_BACKWARD)
    py_op_add(ops, "block_parameter_descriptors", XCNN_OPN_BLOCK_PARAMETERS)
    py_op_add(ops, "block_fit", XCNN_OPN_BLOCK_FIT)
    py_op_add(ops, "block_forward", XCNN_OPN_BLOCK_FORWARD)
    py_op_add(ops, "block_transform", XCNN_OPN_BLOCK_TRANSFORM)
    py_op_add(ops, "conv_init", XCNN_OPN_CONV_INIT)
    py_op_add(ops, "conv_init1d", XCNN_OPN_CONV_INIT1D)
    py_op_add(ops, "conv_backward", XCNN_OPN_CONV_BACKWARD)
    py_op_add(ops, "conv_set_weights", XCNN_OPN_CONV_SET_WEIGHTS)
    py_op_add(ops, "conv_set_weights1d", XCNN_OPN_CONV_SET_WEIGHTS1D)
    py_op_add(ops, "conv_pin", XCNN_OPN_CONV_PIN)
    py_op_add(ops, "conv_unpin", XCNN_OPN_CONV_UNPIN)
    py_op_add(ops, "conv_grad_weight", XCNN_OPN_CONV_GRAD_WEIGHT)
    py_op_add(ops, "conv_grad_bias", XCNN_OPN_CONV_GRAD_BIAS)
    py_op_add(ops, "conv_grad_weight_set", XCNN_OPN_CONV_GRAD_WEIGHT_SET)
    py_op_add(ops, "conv_grad_bias_set", XCNN_OPN_CONV_GRAD_BIAS_SET)
    py_op_add(ops, "conv_weight1d", XCNN_OPN_CONV_WEIGHT1D)
    py_op_add(ops, "conv_fit", XCNN_OPN_CONV_FIT)
    py_op_add(ops, "conv_forward", XCNN_OPN_CONV_FORWARD)
    py_op_add(ops, "conv_transform", XCNN_OPN_CONV_TRANSFORM)
    py_op_add(ops, "cnnclf_init", XCNN_OPN_CNNCLF_INIT)
    py_op_add(ops, "cnnclf_weights", XCNN_OPN_CNNCLF_WEIGHTS)
    py_op_add(ops, "cnnclf_fit", XCNN_OPN_CNNCLF_FIT)
    py_op_add(ops, "cnnclf_predict", XCNN_OPN_CNNCLF_PREDICT)
    py_op_add(ops, "cnnclf_predict_proba", XCNN_OPN_CNNCLF_PREDICT_PROBA)
    py_op_add(ops, "gcn_init", XCNN_OPN_GCN_INIT)
    py_op_add(ops, "sage_init", XCNN_OPN_SAGE_INIT)
    py_op_add(ops, "gcn_cached", XCNN_OPN_GCN_CACHED)
    py_op_add(ops, "sage_cached", XCNN_OPN_SAGE_CACHED)
    py_op_add(ops, "gcn_cached_set", XCNN_OPN_GCN_CACHED_SET)
    py_op_add(ops, "sage_cached_set", XCNN_OPN_SAGE_CACHED_SET)
    py_op_add(ops, "gcn_forward", XCNN_OPN_GCN_FORWARD)
    py_op_add(ops, "sage_forward", XCNN_OPN_SAGE_FORWARD)
    py_op_add(ops, "gcn_backward", XCNN_OPN_GCN_BACKWARD)
    py_op_add(ops, "sage_backward", XCNN_OPN_SAGE_BACKWARD)
    py_op_add(ops, "linear_init", XCNN_OPN_LINEAR_INIT)
    py_op_add(ops, "linear_forward", XCNN_OPN_LINEAR_FORWARD)
    py_op_add(ops, "linear_backward", XCNN_OPN_LINEAR_BACKWARD)
    py_op_add(ops, "relu_init", XCNN_OPN_RELU_INIT)
    py_op_add(ops, "relu_forward", XCNN_OPN_RELU_FORWARD)
    py_op_add(ops, "relu_backward", XCNN_OPN_RELU_BACKWARD)
    py_op_add(ops, "bn_init", XCNN_OPN_BN_INIT)
    py_op_add(ops, "bn_train", XCNN_OPN_BN_TRAIN)
    py_op_add(ops, "bn_running_get", XCNN_OPN_BN_RUNNING_GET)
    py_op_add(ops, "bn_running_set", XCNN_OPN_BN_RUNNING_SET)
    py_op_add(ops, "bn_grad_get", XCNN_OPN_BN_GRAD_GET)
    py_op_add(ops, "bn_grad_set", XCNN_OPN_BN_GRAD_SET)
    py_op_add(ops, "bn_backward", XCNN_OPN_BN_BACKWARD)
    py_op_add(ops, "bn_forward", XCNN_OPN_BN_FORWARD)
    py_op_add(ops, "bn_transform", XCNN_OPN_BN_TRANSFORM)
    py_op_add(ops, "bn_fit", XCNN_OPN_BN_FIT)
    py_op_add(ops, "dropout_init", XCNN_OPN_DROPOUT_INIT)
    py_op_add(ops, "dropout_train", XCNN_OPN_DROPOUT_TRAIN)
    py_op_add(ops, "dropout_mask_get", XCNN_OPN_DROPOUT_MASK_GET)
    py_op_add(ops, "dropout_mask_set", XCNN_OPN_DROPOUT_MASK_SET)
    py_op_add(ops, "dropout_backward", XCNN_OPN_DROPOUT_BACKWARD)
    py_op_add(ops, "dropout_forward", XCNN_OPN_DROPOUT_FORWARD)
    py_op_add(ops, "dropout_transform", XCNN_OPN_DROPOUT_TRANSFORM)
    py_op_add(ops, "adaptive_init", XCNN_OPN_ADAPTIVE_INIT)
    py_op_add(ops, "adaptive_indices_get", XCNN_OPN_ADAPTIVE_INDICES_GET)
    py_op_add(ops, "adaptive_backward", XCNN_OPN_ADAPTIVE_BACKWARD)
    py_op_add(ops, "adaptive_device_io", XCNN_OPN_ADAPTIVE_DEVICE_IO)
    py_op_add(ops, "adaptive_forward", XCNN_OPN_ADAPTIVE_FORWARD)
    py_op_add(ops, "adaptive_transform", XCNN_OPN_ADAPTIVE_TRANSFORM)
    py_op_add(ops, "pool_init", XCNN_OPN_POOL_INIT)
    py_op_add(ops, "pool_backward", XCNN_OPN_POOL_BACKWARD)
    py_op_add(ops, "pool_fit", XCNN_OPN_POOL_FIT)
    py_op_add(ops, "pool_indices_get", XCNN_OPN_POOL_INDICES_GET)
    py_op_add(ops, "pool_indices_set", XCNN_OPN_POOL_INDICES_SET)
    py_op_add(ops, "pool_forward", XCNN_OPN_POOL_FORWARD)
    py_op_add(ops, "pool_transform", XCNN_OPN_POOL_TRANSFORM)
    py_op_add(ops, "cnnclf_adopt", XCNN_OPN_CNNCLF_ADOPT)
    py_op_add(ops, "layer_state_dict", XCNN_OPN_LAYER_STATE_DICT)
    py_op_add(ops, "layer_load_state_dict", XCNN_OPN_LAYER_LOAD_STATE_DICT)
    py_op_add(ops, "layer_parameters", XCNN_OPN_LAYER_PARAMETERS)
    py_op_add(ops, "layer_named_parameters", XCNN_OPN_LAYER_NAMED_PARAMETERS)
    py_op_add(ops, "graph_state_dict", XCNN_OPN_GRAPH_STATE_DICT)
    py_op_add(ops, "graph_load_state_dict", XCNN_OPN_GRAPH_LOAD_STATE_DICT)
    nk[].ops = ops
    nk[].ready = True


@always_inline
def xcnn_keys() raises -> UnsafePointer[XcnnKeys, MutUntrackedOrigin]:
    """The family's keys: one slot fetch and one flag test. Fetch once per
    function, outside loops."""
    var nk = _XCNN_KEYS.get_or_create_ptr()
    if not nk[].ready:
        _xcnn_keys_fill(nk)
    return nk
