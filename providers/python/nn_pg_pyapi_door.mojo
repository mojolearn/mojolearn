# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# UNPACK / PACK of the session optimizers (lane nn-blocks-pg, 2026-11-25): the optimizer request of
# `Session.optimizer`, the per-parameter state of `optimizer_state` / `load_optimizer_state` (moved here from
# providers/python/autograd_ops_frontend_pyapi_door.mojo, the same words, so the grouped optimizer's state dict
# packs every parameter exactly as `optimizer_state` does), and the parameter-group operations
# `param_group_optimizer`, `learning_rates`, `optimizer_state_dict`, `load_optimizer_state_dict`
# (autograd/nn_pg.mojo). Python objects only here; RUN is OpsSession's typed methods.
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_item, py_setitem, py_type, py_contains
from bindings.autograd_keys_unpack import autograd_keys
from sequence.exec_trait import Exec
from autograd.ops_api import OpsSession
from autograd.optimizer_requests import (
    OpsOptimizerRequest, OpsOptimizerExtras, OpsOptimizerRestoreRequest, OpsOptimizerStateValue,
    ops_optimizer_codes,
)
from autograd.nn_pg import pg_admit_key
from autograd.autograd_ops_frontend_run import (
    _SOP_PG_OPTIMIZER, _SOP_PG_LEARNING_RATES, _SOP_PG_STATE_DICT, _SOP_PG_LOAD_STATE_DICT,
)
from sequence.ops import OPT_SGD, OPT_LION, OPT_RMSPROP, OPT_ADAGRAD
from providers.python.autograd_program_arrays_pyapi_door import (
    program_admit_value, program_numeric_scalar, program_builtin_types, is_integer_input,
)


def _pg_nil(x: PythonObject) -> Bool:
    return x is PythonObject(None)


def _pg_float(x: PythonObject, name: String) raises -> Float32:
    # The door's `_float`: a real scalar without conversion hooks, refused naming the argument.
    if not program_numeric_scalar(x):
        raise Error("TypeError: " + name + " must be a native real scalar without conversion hooks")
    try:
        return Float32(py=x)
    except error:
        raise Error("TypeError: " + name + " must be a real number")


def _pg_int(x: PythonObject, name: String, low: Int) raises -> Int:
    if not is_integer_input(x):
        raise Error("TypeError: " + name + " must be an integer")
    var value = Int(py=x)
    if value < low:
        raise Error("ValueError: " + name + " must be >= " + String(low))
    return value


def _pg_bool(x: PythonObject, name: String) raises -> Bool:
    var types = program_builtin_types()
    if py_type(x) is not py_type(PythonObject(True)) and py_type(x) is not types[].int_type:
        raise Error("TypeError: " + name + " must be a bool")
    return Bool(py=x)


def _pg_sequence(x: PythonObject) raises -> Bool:
    var types = program_builtin_types()
    var cls = py_type(x)
    return cls is types[].list_type or cls is types[].tuple_type


# ---------------------------------------------------------------- the single-group optimizer's words
def ops_optimizer_unpack(args: PythonObject) raises -> Tuple[OpsOptimizerRequest, OpsOptimizerExtras]:
    """`Session.optimizer`'s arguments (kind, lr, momentum, nesterov, beta1, beta2, eps, weight_decay,
    schedule, step_size, gamma, alpha, centered, lr_decay, initial_accumulator_value, clip_grad_norm,
    clip_grad_value) as the typed request. UNPACK: exact supported scalar/container admission precedes every
    conversion; no user conversion hook can enter optimizer execution."""
    program_admit_value(args)
    var codes = ops_optimizer_codes(String(args[0]), String(args[8]))
    var request = OpsOptimizerRequest(codes[0], Bool(py=args[3]), codes[1],
        _pg_int(args[9], "step_size", 1), _pg_float(args[1], "lr"), _pg_float(args[2], "momentum"),
        _pg_float(args[4], "beta1"), _pg_float(args[5], "beta2"), _pg_float(args[6], "eps"),
        _pg_float(args[7], "weight_decay"), _pg_float(args[10], "gamma"), Float32(0))
    # lane fx-ops-session-blocks: args[11..16] = alpha, centered, lr_decay,
    # initial_accumulator_value, clip_grad_norm, clip_grad_value (None: no
    # clip); a clip given must be positive.
    var clip_norm = Float32(0)
    var clip_value = Float32(0)
    if not _pg_nil(args[15]):
        clip_norm = _pg_float(args[15], "clip_grad_norm")
        if not (clip_norm > Float32(0)):
            raise Error("ValueError: clip_grad_norm must be positive")
    if not _pg_nil(args[16]):
        clip_value = _pg_float(args[16], "clip_grad_value")
        if not (clip_value > Float32(0)):
            raise Error("ValueError: clip_grad_value must be positive")
    var extras = OpsOptimizerExtras(_pg_float(args[11], "alpha"), Bool(py=args[12]), _pg_float(args[13], "lr_decay"),
        _pg_float(args[14], "initial_accumulator_value"), clip_norm, clip_value)
    return (request, extras)


def ops_optimizer_state_pack(snapshot: OpsOptimizerStateValue) raises -> PythonObject:
    """PACK one (optimizer, parameter) state snapshot into the documented mapping (torch's state names)."""
    var state = Python.dict()
    py_setitem(state, "step", snapshot.step)
    if snapshot.kind == 3 or snapshot.kind == 4:
        # RMSprop / Adagrad (sequence/ops.mojo OPT_RMSPROP 3,
        # OPT_ADAGRAD 4), torch's state names
        if snapshot.first != 0:
            py_setitem(state, "momentum_buffer", snapshot.first)
        if snapshot.second != 0:
            py_setitem(state, "square_avg" if snapshot.kind == 3 else "sum", snapshot.second)
        if snapshot.third != 0:
            py_setitem(state, "grad_avg", snapshot.third)
    elif snapshot.kind != 0:
        py_setitem(state, "beta1_power", Float64(snapshot.beta1_power))
        py_setitem(state, "beta2_power", Float64(snapshot.beta2_power))
        if snapshot.kind == 9:
            # NAdam (sequence/ops.mojo OPT_NADAM): the running mu product
            py_setitem(state, "mu_product", Float64(snapshot.mu_product))
        if snapshot.first != 0:
            py_setitem(state, "exp_avg", snapshot.first)
        if snapshot.second != 0:
            # Adamax (OPT_ADAMAX 8) keeps the infinity norm there
            py_setitem(state, "exp_inf" if snapshot.kind == 8 else "exp_avg_sq", snapshot.second)
    elif snapshot.first != 0:
        py_setitem(state, "momentum_buffer", snapshot.first)
    return state


def ops_optimizer_state_unpack(state: PythonObject, opt_kind: Int) raises -> OpsOptimizerRestoreRequest:
    """UNPACK the mapping `ops_optimizer_state_pack` makes (tensor handles for the buffers) for an optimizer of
    kind `opt_kind`."""
    var nk = autograd_keys()
    program_admit_value(state)
    var step = _pg_int(py_item(state, nk[].k_step), "step", 0)
    var first = 0
    var second = 0
    var beta1 = Float32(1)
    var beta2 = Float32(1)
    var mu = Float32(1)
    var third = 0
    if opt_kind == 0:
        if py_contains(state, nk[].k_momentum_buffer):
            first = _pg_int(py_item(state, nk[].k_momentum_buffer), "momentum_buffer", 1)
    elif opt_kind == 3 or opt_kind == 4:
        if py_contains(state, nk[].k_momentum_buffer):
            first = _pg_int(py_item(state, nk[].k_momentum_buffer), "momentum_buffer", 1)
        if opt_kind == 3:
            if py_contains(state, nk[].k_square_avg):
                second = _pg_int(py_item(state, nk[].k_square_avg), "square_avg", 1)
        elif py_contains(state, nk[].k_sum):
            second = _pg_int(py_item(state, nk[].k_sum), "sum", 1)
        if py_contains(state, nk[].k_grad_avg):
            third = _pg_int(py_item(state, nk[].k_grad_avg), "grad_avg", 1)
    else:
        if py_contains(state, nk[].k_exp_avg):
            first = _pg_int(py_item(state, nk[].k_exp_avg), "exp_avg", 1)
        if opt_kind == 8:
            # Adamax: the keys `optimizer_state` returns for it
            if py_contains(state, nk[].k_exp_inf):
                second = _pg_int(py_item(state, nk[].k_exp_inf), "exp_inf", 1)
        elif py_contains(state, nk[].k_exp_avg_sq):
            second = _pg_int(py_item(state, nk[].k_exp_avg_sq), "exp_avg_sq", 1)
        beta1 = Float32(py=state["beta1_power"])
        beta2 = Float32(py=state["beta2_power"])
        if opt_kind == 9:
            mu = Float32(py=py_item(state, nk[].k_mu_product))
    return OpsOptimizerRestoreRequest(step, first, second, third, beta1, beta2, mu)


# ---------------------------------------------------------------- parameter groups
def _pg_handles(x: PythonObject, where: String) raises -> List[Int]:
    """A group's `params`: one tensor handle or a list / tuple of them."""
    var out = List[Int]()
    if is_integer_input(x):
        out.append(_pg_int(x, where + "['params']", 1))
        return out^
    if not _pg_sequence(x):
        raise Error("TypeError: " + where + "['params'] must be a tensor handle or a list of tensor handles")
    for k in range(len(x)):  # driver-loop: over one group's parameter handles (construction, once)
        out.append(_pg_int(x[k], where + "['params']", 1))
    return out^


def _pg_keys(group: PythonObject, where: String) raises -> PythonObject:
    """The keys of a group dict, as a list of exact str (refused otherwise)."""
    var types = program_builtin_types()
    if py_type(group) is not types[].dict_type:
        raise Error("TypeError: " + where + " must be a dict")
    var keys = py_call(native_loaded_module("builtins").list, group)
    for k in range(len(keys)):  # driver-loop: over one group's keys (construction, once)
        if py_type(keys[k]) is not types[].str_type:
            raise Error("TypeError: " + where + " has a key that is not a str")
    return keys


def _pg_apply(kind: Int, group: PythonObject, gi: Int, mut req: OpsOptimizerRequest,
              mut ext: OpsOptimizerExtras) raises:
    """Group gi's own keys over the optimizer-wide settings, each admitted by `pg_admit_key` (clear words for
    an unknown key, a key of another optimizer kind and an optimizer-wide key)."""
    var where = "param_groups[" + String(gi) + "]"
    var keys = _pg_keys(group, where)
    for k in range(len(keys)):  # driver-loop: over one group's keys (construction, once)
        var key = String(keys[k])
        pg_admit_key(kind, key, gi)
        var v = py_item(group, keys[k])
        var name = where + "['" + key + "']"
        if key == "params":
            continue
        elif key == "lr":
            req.lr = _pg_float(v, name)
        elif key == "weight_decay":
            req.weight_decay = _pg_float(v, name)
        elif key == "momentum":
            req.momentum = _pg_float(v, name)
        elif key == "nesterov":
            req.nesterov = _pg_bool(v, name)
        elif key == "dampening":
            req.dampening = _pg_float(v, name)
        elif key == "betas":
            if not _pg_sequence(v) or len(v) != 2:
                raise Error("ValueError: " + name + " must be a pair (beta1, beta2)")
            req.beta1 = _pg_float(v[0], name)
            req.beta2 = _pg_float(v[1], name)
        elif key == "beta1":
            req.beta1 = _pg_float(v, name)
        elif key == "beta2":
            req.beta2 = _pg_float(v, name)
        elif key == "eps":
            req.eps = _pg_float(v, name)
        elif key == "alpha":
            ext.alpha = _pg_float(v, name)
        elif key == "centered":
            ext.centered = _pg_bool(v, name)
        elif key == "lr_decay":
            ext.lr_decay = _pg_float(v, name)
        elif key == "initial_accumulator_value":
            ext.initial_accumulator_value = _pg_float(v, name)


def _pg_group_dict[E: Exec & Movable & Deinitable](
    s: UnsafePointer[OpsSession[E], MutUntrackedOrigin], o: Int, gi: Int, params: PythonObject,
) raises -> PythonObject:
    """PACK group gi's settings under the keys a group takes (`pg_key_taken`), so the dict loads back."""
    var cfg = s[].pg_group_config(o, gi)
    var d = Python.dict()
    py_setitem(d, "params", params)
    py_setitem(d, "lr", Float64(cfg.lr))
    py_setitem(d, "weight_decay", Float64(cfg.weight_decay))
    var kind = cfg.kind
    if kind == OPT_SGD:
        py_setitem(d, "momentum", Float64(cfg.momentum))
        py_setitem(d, "nesterov", PythonObject(cfg.nesterov))
        py_setitem(d, "dampening", Float64(cfg.dampening))
    elif kind == OPT_RMSPROP:
        py_setitem(d, "alpha", Float64(cfg.alpha))
        py_setitem(d, "eps", Float64(cfg.eps))
        py_setitem(d, "momentum", Float64(cfg.momentum))
        py_setitem(d, "centered", PythonObject(cfg.centered))
    elif kind == OPT_ADAGRAD:
        py_setitem(d, "eps", Float64(cfg.eps))
        py_setitem(d, "lr_decay", Float64(cfg.lr_decay))
        py_setitem(d, "initial_accumulator_value", Float64(cfg.init_acc))
    else:
        var betas = Python.list()
        _ = py_call(betas.append, Float64(cfg.beta1))
        _ = py_call(betas.append, Float64(cfg.beta2))
        py_setitem(d, "betas", py_call(native_loaded_module("builtins").tuple, betas))
        if kind != OPT_LION:
            py_setitem(d, "eps", Float64(cfg.eps))
    return d


def pg_session_call_py[E: Exec & Movable & Deinitable](
    s: UnsafePointer[OpsSession[E], MutUntrackedOrigin], op: Int, args: PythonObject,
) raises -> PythonObject:
    """The parameter-group Session operations (codes _SOP_PG_*)."""
    if op == _SOP_PG_OPTIMIZER:
        # args: Session.optimizer's seventeen arguments, then param_groups (a list of dicts)
        var base = ops_optimizer_unpack(args)
        var groups_obj = args[17]
        if not _pg_sequence(groups_obj):
            raise Error("TypeError: param_groups must be a list of dicts")
        var requests = List[OpsOptimizerRequest]()
        var extras = List[OpsOptimizerExtras]()
        var groups = List[List[Int]]()
        for gi in range(len(groups_obj)):  # driver-loop: over parameter groups (construction, once)
            var group = groups_obj[gi]
            var where = "param_groups[" + String(gi) + "]"
            _ = _pg_keys(group, where)
            if not py_contains(group, PythonObject("params")):
                raise Error("ValueError: " + where + " has no 'params'")
            var req = base[0]
            var ext = base[1]
            _pg_apply(base[0].kind, group, gi, req, ext)
            requests.append(req)
            extras.append(ext)
            groups.append(_pg_handles(py_item(group, PythonObject("params")), where))
        return PythonObject(s[].param_group_optimizer(base[0], base[1], requests, extras, groups))
    var o = _pg_int(args[0], "optimizer", 1)
    if op == _SOP_PG_LEARNING_RATES:
        var rates = s[].pg_learning_rates(o)
        var out = Python.list()
        for k in range(len(rates)):  # driver-loop: over parameter groups
            _ = py_call(out.append, Float64(rates[k]))
        return out
    if op == _SOP_PG_STATE_DICT:
        # torch's layout: {"state": {index: per-parameter state}, "param_groups": [{"params": [index, ...],
        # "lr": ..., ...}]}, indices in group order; plus the schedule's "epoch".
        var state = Python.dict()
        var param_groups = Python.list()
        var index = 0
        for gi in range(s[].pg_groups(o)):  # driver-loop: over parameter groups
            var hs = s[].pg_group_params(o, gi)
            var params = Python.list()
            for k in range(len(hs)):  # driver-loop: over one group's parameters (state snapshots)
                _ = py_call(params.append, index)
                py_setitem(state, PythonObject(index), ops_optimizer_state_pack(s[].optimizer_state_value(o, hs[k])))
                index += 1
            _ = py_call(param_groups.append, _pg_group_dict(s, o, gi, params))
        var sd = Python.dict()
        py_setitem(sd, "state", state)
        py_setitem(sd, "param_groups", param_groups)
        py_setitem(sd, "epoch", s[].pg_epoch(o))
        return sd
    if op == _SOP_PG_LOAD_STATE_DICT:
        var sd = args[1]
        program_admit_value(sd)
        var types = program_builtin_types()
        if py_type(sd) is not types[].dict_type or not py_contains(sd, PythonObject("param_groups")) or not py_contains(sd, PythonObject("state")):
            raise Error("ValueError: load_optimizer_state_dict takes the dict optimizer_state_dict returned "
                        + "(keys 'state' and 'param_groups')")
        var saved_groups = py_item(sd, PythonObject("param_groups"))
        var saved_state = py_item(sd, PythonObject("state"))
        if not _pg_sequence(saved_groups):
            raise Error("TypeError: the state's 'param_groups' must be a list of dicts")
        var n = s[].pg_groups(o)
        if len(saved_groups) != n:
            raise Error("ValueError: the state has " + String(len(saved_groups)) + " parameter groups and this "
                        + "optimizer has " + String(n))
        var requests = List[OpsOptimizerRequest]()
        var extras = List[OpsOptimizerExtras]()
        var params = List[List[Int]]()
        var saved_index = List[PythonObject]()
        for gi in range(n):  # driver-loop: over parameter groups (every check before any change)
            var group = saved_groups[gi]
            var where = "param_groups[" + String(gi) + "]"
            _ = _pg_keys(group, where)
            if not py_contains(group, PythonObject("params")):
                raise Error("ValueError: the state's " + where + " has no 'params'")
            var indices = py_item(group, PythonObject("params"))
            var hs = s[].pg_group_params(o, gi)
            if not _pg_sequence(indices) or len(indices) != len(hs):
                raise Error("ValueError: the state's " + where + " has " + String(len(indices)) + " parameters "
                            + "and this optimizer's has " + String(len(hs)))
            var now = s[].pg_group_request(o, gi)
            var req = now[0]
            var ext = now[1]
            _pg_apply(req.kind, group, gi, req, ext)
            s[].pg_load_group(o, gi, req, ext, False)
            requests.append(req)
            extras.append(ext)
            params.append(hs^)
            saved_index.append(indices)
        var epoch = -1
        if py_contains(sd, PythonObject("epoch")):
            epoch = _pg_int(py_item(sd, PythonObject("epoch")), "epoch", 0)
        for gi in range(n):  # driver-loop: over parameter groups (the settings, then each parameter's state)
            s[].pg_load_group(o, gi, requests[gi], extras[gi], True)
            for k in range(len(params[gi])):  # driver-loop: over one group's parameters (state restores)
                var key = saved_index[gi][k]
                if py_contains(saved_state, key):
                    var st = ops_optimizer_state_unpack(py_item(saved_state, key), requests[gi].kind)
                    s[].load_optimizer_state(o, params[gi][k], st)
        if epoch >= 0:
            s[].pg_set_epoch(o, epoch)
        return PythonObject(None)
    raise Error("unknown native session operation code " + String(op))
