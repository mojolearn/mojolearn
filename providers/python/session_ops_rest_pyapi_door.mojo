# SPDX-License-Identifier: Apache-2.0
# UNPACK of the Session operations of lane session-ops-rest (2026-11-25): batch_norm, group_norm, instance_norm,
# prelu, mish, multi_head_attention, positional_encoding, smooth_l1_loss, nll_loss, kl_div,
# binary_cross_entropy_with_logits, ctc_loss, selective_scan, ssd_scan, trapezoid_scan (frontend codes
# 1160..1174, session codes 160..174; autograd/session_ops_rest.mojo). Every argument is read here once, from the
# caller's own objects, into the session call's handles, integers and floats; the operation itself runs in Mojo
# (OpsSession._layer_rest). Options Sequential's layer does not support are refused by name.
from std.python import PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_type
from providers.python.autograd_program_pyapi_door import _integer
from providers.python.autograd_program_arrays_pyapi_door import (
    is_integer_input, uint64_input, program_numeric_scalar,
)
from autograd.session_ops_rest import (
    SOR_BATCH_NORM, SOR_GROUP_NORM, SOR_INSTANCE_NORM, SOR_PRELU, SOR_MISH, SOR_MHA, SOR_POS_ENC, SOR_SMOOTH_L1,
    SOR_NLL, SOR_KL_DIV, SOR_BCE_POS_WEIGHT, SOR_CTC, SOR_SELECTIVE_SCAN, SOR_SSD_SCAN, SOR_TRAPEZOID_SCAN,
)


def _nil(x: PythonObject) -> Bool:
    return x is PythonObject(None)


def _handle(x: PythonObject, name: String) raises -> Int:
    return _integer(x, name, 1)


def _number(x: PythonObject, name: String) raises -> Float32:
    if not program_numeric_scalar(x):
        raise Error("TypeError: " + name + " must be a real number")
    return Float32(py=x)


def _flag(x: PythonObject, name: String) raises -> Int:
    """An exact bool (or the integers 0 and 1) as 0 / 1."""
    var b = native_loaded_module("builtins")
    var kind = py_type(x)
    if kind is b.bool:
        return 1 if Bool(py=x) else 0
    if kind is b.int:
        var v = Int(py=x)
        if v == 0 or v == 1:
            return v
    raise Error("TypeError: " + name + " must be True or False")


def _reduction(x: PythonObject, name: String, batchmean: Bool) raises -> Int:
    """'mean' 0, 'sum' 1, 'batchmean' 2 (kl_div only); 'none' and anything else refused."""
    var b = native_loaded_module("builtins")
    if not (py_type(x) is b.str):
        raise Error("TypeError: " + name + ": reduction must be a string")
    var r = String(x)
    if r == "mean":
        return 0
    if r == "sum":
        return 1
    if batchmean and r == "batchmean":
        return 2
    var choices = String("'mean' or 'sum'")
    if batchmean:
        choices = String("'mean', 'sum' or 'batchmean'")
    if r == "none":
        raise Error("ValueError: " + name + ": reduction='none' is not supported (the loss is one value); use "
                    + choices)
    raise Error("ValueError: " + name + ": reduction must be " + choices)


def _seed_words(x: PythonObject, mut ints: List[Int]) raises:
    if not is_integer_input(x):
        raise Error("TypeError: seed must be an integer")
    var seed = uint64_input(x, "seed")
    ints.append(Int(seed & 0xffffffff))
    ints.append(Int(seed >> 32))


def _dropout_p(x: PythonObject, name: String) raises -> Float32:
    var p = _number(x, "dropout_p")
    if not (p >= 0 and p < 1):
        raise Error("ValueError: " + name + ": dropout_p must be in [0, 1)")
    return p


def _eps(x: PythonObject, name: String) raises -> Float32:
    var e = _number(x, "eps")
    if not e > 0:
        raise Error("ValueError: " + name + ": eps must be positive")
    return e


def _affine(weight: PythonObject, bias: PythonObject, name: String, mut hs: List[Int]) raises -> Int:
    """weight and bias both given (1, appended) or both None (0)."""
    if _nil(weight) and _nil(bias):
        return 0
    if _nil(weight) or _nil(bias):
        raise Error("ValueError: " + name + ": weight and bias must be given together (pass both or neither)")
    hs.append(_handle(weight, "weight"))
    hs.append(_handle(bias, "bias"))
    return 1


def _running(mean: PythonObject, var_: PythonObject, name: String, mut hs: List[Int]) raises -> Int:
    if _nil(mean) and _nil(var_):
        return 0
    if _nil(mean) or _nil(var_):
        raise Error("ValueError: " + name + ": running_mean and running_var must be given together")
    hs.append(_handle(mean, "running_mean"))
    hs.append(_handle(var_, "running_var"))
    return 1


def sor_unpack(op: Int, args: PythonObject, mut hs: List[Int], mut ints: List[Int], mut fs: List[Float32]) raises -> Int:
    """The session code of frontend code `op` and its operands, integers and floats (OpsSession._layer_rest's
    words); the arguments in the Python method's order (python/mojolearn/ops.py, session-ops-rest group)."""
    var code = op - 1000
    if code == SOR_BATCH_NORM or code == SOR_INSTANCE_NORM:
        # batch_norm (input, running_mean, running_var, weight, bias, training, momentum, eps);
        # instance_norm (input, running_mean, running_var, weight, bias, use_input_stats, momentum, eps)
        var name = String("batch_norm")
        var stats_word = String("training")
        if code == SOR_INSTANCE_NORM:
            name = String("instance_norm")
            stats_word = String("use_input_stats")
        hs.append(_handle(args[0], "input"))
        var running = _running(args[1], args[2], name, hs)
        var affine = _affine(args[3], args[4], name, hs)
        var training = _flag(args[5], stats_word)
        if _nil(args[6]):
            raise Error("ValueError: " + name + ": momentum=None (a cumulative average) is not supported; pass a number")
        var momentum = _number(args[6], "momentum")
        if not (momentum >= 0 and momentum <= 1):
            raise Error("ValueError: " + name + ": momentum must be in [0, 1]")
        if code == SOR_BATCH_NORM and training == 0 and running == 0:
            raise Error("ValueError: batch_norm: training=False normalizes with running_mean and running_var; pass both")
        ints = [training, running, affine]
        fs = [momentum, _eps(args[7], name)]
        return code
    if code == SOR_GROUP_NORM:
        # (input, num_groups, weight, bias, eps)
        hs.append(_handle(args[0], "input"))
        var groups = _integer(args[1], "num_groups", 1)
        var affine = _affine(args[2], args[3], "group_norm", hs)
        ints = [groups, affine]
        fs = [_eps(args[4], "group_norm")]
        return code
    if code == SOR_PRELU:
        hs = [_handle(args[0], "input"), _handle(args[1], "weight")]
        return code
    if code == SOR_MISH:
        hs = [_handle(args[0], "input")]
        return code
    if code == SOR_MHA:
        # (query, key, value, num_heads, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias,
        #  key_lengths, attn_mask, is_causal, dropout_p, training, seed, counter)
        var name = String("multi_head_attention")
        var q = _handle(args[0], "query")
        var k = _handle(args[1], "key")
        var v = _handle(args[2], "value")
        if k != v:
            raise Error("ValueError: multi_head_attention: key and value must be the same tensor (its keys and "
                        + "values are projections of one sequence); for separate keys and values use Session.attention")
        var self_attn = q == k
        var heads = _integer(args[3], "num_heads", 1)
        hs = [q, k]
        if self_attn:
            if not is_integer_input(args[4]):
                raise Error("TypeError: multi_head_attention: self-attention takes one in_proj_weight tensor "
                            + "(embed_dim, 3 * embed_dim)")
            hs.append(_handle(args[4], "in_proj_weight"))
        else:
            if is_integer_input(args[4]) or len(args[4]) != 2:
                raise Error("TypeError: multi_head_attention: cross-attention (key is not query) takes in_proj_weight "
                            + "as a pair (query weight (embed_dim, embed_dim), key/value weight (kdim, 2 * embed_dim))")
            hs.append(_handle(args[4][0], "in_proj_weight[0]"))
            hs.append(_handle(args[4][1], "in_proj_weight[1]"))
        hs.append(_handle(args[6], "out_proj_weight"))
        var bias = 0
        if _nil(args[5]) != _nil(args[7]):
            raise Error("ValueError: multi_head_attention: in_proj_bias and out_proj_bias must be given together")
        if not _nil(args[5]):
            bias = 1
            if self_attn:
                hs.append(_handle(args[5], "in_proj_bias"))
            else:
                if is_integer_input(args[5]) or len(args[5]) != 2:
                    raise Error("TypeError: multi_head_attention: cross-attention takes in_proj_bias as a pair "
                                + "(query bias (embed_dim), key/value bias (2 * embed_dim))")
                hs.append(_handle(args[5][0], "in_proj_bias[0]"))
                hs.append(_handle(args[5][1], "in_proj_bias[1]"))
            hs.append(_handle(args[7], "out_proj_bias"))
        var lens = 0
        if not _nil(args[8]):
            lens = 1
            hs.append(_handle(args[8], "key_lengths"))
        if not _nil(args[9]):
            raise Error("ValueError: multi_head_attention: attn_mask is not supported; use is_causal=True for a causal "
                        + "mask and key_lengths for key padding, or Session.attention for any other mask")
        ints = [heads, 1 if self_attn else 0, bias, _flag(args[10], "is_causal"), lens, _flag(args[12], "training")]
        _seed_words(args[13], ints)
        ints.append(_integer(args[14], "counter", 0))
        fs = [_dropout_p(args[11], name)]
        return code
    if code == SOR_POS_ENC:
        # (x, dropout_p, training, seed, counter)
        hs = [_handle(args[0], "x")]
        ints = [_flag(args[2], "training")]
        _seed_words(args[3], ints)
        ints.append(_integer(args[4], "counter", 0))
        fs = [_dropout_p(args[1], "positional_encoding")]
        return code
    if code == SOR_SMOOTH_L1:
        # (input, target, reduction, beta)
        hs = [_handle(args[0], "input"), _handle(args[1], "target")]
        ints = [_reduction(args[2], "smooth_l1_loss", False)]
        var beta = _number(args[3], "beta")
        if not (beta >= 0):
            raise Error("ValueError: smooth_l1_loss: beta must be a non-negative number")
        fs = [beta]
        return code
    if code == SOR_NLL:
        # (input, target, weight, ignore_index, reduction)
        if not _nil(args[2]):
            raise Error("ValueError: nll_loss: weight (class weights) is not supported; pass weight=None")
        if not is_integer_input(args[3]) or Int(py=args[3]) != -100:
            raise Error("ValueError: nll_loss: ignore_index is not supported; every target must be a class id")
        hs = [_handle(args[0], "input"), _handle(args[1], "target")]
        ints = [_reduction(args[4], "nll_loss", False)]
        return code
    if code == SOR_KL_DIV:
        # (input, target, reduction, log_target)
        if _flag(args[3], "log_target") != 0:
            raise Error("ValueError: kl_div: log_target=True is not supported; pass the target as probabilities")
        hs = [_handle(args[0], "input"), _handle(args[1], "target")]
        ints = [_reduction(args[2], "kl_div", True)]
        return code
    if code == SOR_BCE_POS_WEIGHT:
        # (input, target, weight, reduction, pos_weight)
        if not _nil(args[2]):
            raise Error("ValueError: binary_cross_entropy_with_logits: weight is not supported here; "
                        + "Session.bce_with_logits takes a per-element weight")
        hs = [_handle(args[0], "input"), _handle(args[1], "target")]
        ints = [_reduction(args[3], "binary_cross_entropy_with_logits", False)]
        var pw = Float32(1)
        if not _nil(args[4]):
            var b = native_loaded_module("builtins")
            if not (py_type(args[4]) is b.float):
                raise Error("TypeError: binary_cross_entropy_with_logits: pos_weight must be a float, one weight "
                            + "for every output (a tensor of per-class weights is not supported)")
            pw = Float32(py=args[4])
            if not (pw > 0):
                raise Error("ValueError: binary_cross_entropy_with_logits: pos_weight must be positive")
        fs = [pw]
        return code
    if code == SOR_CTC:
        # (log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity)
        hs = [_handle(args[0], "log_probs"), _handle(args[1], "targets"), _handle(args[2], "input_lengths"),
              _handle(args[3], "target_lengths")]
        ints = [_integer(args[4], "blank", 0), _reduction(args[5], "ctc_loss", False), _flag(args[6], "zero_infinity")]
        return code
    if code == SOR_SELECTIVE_SCAN:
        # (u, delta, A_log, B, C, D)
        hs = [_handle(args[0], "u"), _handle(args[1], "delta"), _handle(args[2], "A_log"), _handle(args[3], "B"),
              _handle(args[4], "C"), _handle(args[5], "D")]
        return code
    if code == SOR_SSD_SCAN:
        # (x, dt, A_log, D)
        hs = [_handle(args[0], "x"), _handle(args[1], "dt"), _handle(args[2], "A_log"), _handle(args[3], "D")]
        return code
    if code == SOR_TRAPEZOID_SCAN:
        # (v, k, q, dt, A, trap, D, heads)
        hs = [_handle(args[0], "v"), _handle(args[1], "k"), _handle(args[2], "q"), _handle(args[3], "dt"),
              _handle(args[4], "A"), _handle(args[5], "trap"), _handle(args[6], "D")]
        ints = [_integer(args[7], "heads", 1)]
        return code
    raise Error("unknown native session operation code " + String(op))
