# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""UNPACK of `Sequential.compile(param_groups=..., no_decay=...)` and the fit / save doors' group words (lane
nn-blocks-pgseq, 2026-11-25; autograd/nn_pgseq.mojo).

compile: each group's keys admitted by part 1's words (autograd/nn_pg.mojo pg_admit_key) after Sequential's own
(max_grad_norm, trust_clip and always_adapt are optimizer-wide; Sequential has no dampening; 'sgd' has no
momentum), each group's optimizer words computed by compile's own body (bindings/nn_optim_compile_pack.mojo
nn_optim_compile_words) over compile's options with the group's values put in, and stored in the compile options
as `param_groups`:
    {"layers": ((i, ...), ...), "names": (("0.weight", ...), ...), "no_decay": ("bias", ...),
     "words": (PGSEQ_GROUP_WORDS floats a group: the U user groups, the default group, then their no-decay twins)}
A checkpoint's groups come back as {"resolved": (the fit words)}: the per-tensor table it saved.

fit / save: `pgseq_door_words` resolves them against the model's tensor layout once per call
(autograd/nn_pgseq.mojo pgseq_resolve). Nothing here runs inside a fit.
"""
from std.python import Python, PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_contains, py_get, py_item, py_setitem, py_type
from bindings.sequential_description_unpack import _integer
from bindings.nn_optim_compile_pack import nn_optim_compile_words
from autograd.model import parse_model
from autograd.model_execution import model_float_words
from autograd.model_checkpoint import model_layout
from autograd.nn_optim import nn_positive_lr
from autograd.nn_pg import pg_admit_key
from autograd.nn_pgseq import PGSEQ_GROUP_WORDS, pgseq_admit_schedule, pgseq_resolve

# compile's option positions a group may set (bindings/nn_optim_compile_pack.mojo's table)
comptime _AT_LR = 2
comptime _AT_MOMENTUM = 3
comptime _AT_NESTEROV = 4
comptime _AT_BETA1 = 5
comptime _AT_BETA2 = 6
comptime _AT_EPS = 7
comptime _AT_WEIGHT_DECAY = 8
comptime _AT_ALPHA = 20
comptime _AT_CENTERED = 21
comptime _AT_LR_DECAY = 22
comptime _AT_INITIAL_ACC = 23


def _where(g: Int) -> String:
    return "param_groups[" + String(g) + "]"


def _group_words(
    options: PythonObject, optimizer: String, sched: Int, b: PythonObject, no_decay_twin: Bool,
) raises -> List[Float64]:
    """One group's PGSEQ_GROUP_WORDS words from compile's options with the group's values put in."""
    var scratch = Python.dict()
    nn_optim_compile_words(scratch, py_call(b.tuple, options), optimizer, sched, b)
    var lr = Float64(py=py_call(b.float, options[_AT_LR]))
    if not nn_positive_lr(Float32(lr)):
        raise Error("ValueError: a parameter group's lr must be positive and finite in float32")
    var w = List[Float64]()
    w.append(lr)
    w.append(Float64(Int(py=py_item(scratch, PythonObject("flags")))))
    w.append(Float64(py=py_item(scratch, PythonObject("f1"))))
    w.append(Float64(py=py_item(scratch, PythonObject("f2"))))
    w.append(Float64(py=py_item(scratch, PythonObject("eps"))))
    w.append(0.0 if no_decay_twin else Float64(py=py_item(scratch, PythonObject("weight_decay"))))
    w.append(Float64(py=py_item(scratch, PythonObject("f7"))))
    w.append(0.0)
    return w^


def _put(options: PythonObject, g: Int, key: String, value: PythonObject, optimizer: String, kind: Int,
         b: PythonObject) raises:
    """Put group g's `key` into its copy of compile's options, refusing in words what a group cannot set."""
    var where = _where(g)
    if key == "max_grad_norm" or key == "trust_clip" or key == "always_adapt":
        raise Error("ValueError: " + where + ": '" + key + "' is optimizer-wide: the clip is computed over the "
                    + "gradients of every group; pass it to compile, not to a group")
    if key == "dampening":
        raise Error("ValueError: " + where + ": Sequential.compile has no 'dampening'")
    if optimizer == "sgd" and (key == "momentum" or key == "nesterov"):
        raise Error("ValueError: " + where + ": the sgd optimizer has no momentum; compile with "
                    + "optimizer='momentum' to give groups their own momentum")
    # part 1's words: optimizer-wide schedule / clip keys, another kind's keys, unknown keys
    pg_admit_key(kind, key, g)
    if key == "lr":
        py_setitem(options, _AT_LR, value)
    elif key == "weight_decay":
        py_setitem(options, _AT_WEIGHT_DECAY, value)
    elif key == "momentum":
        py_setitem(options, _AT_MOMENTUM, value)
    elif key == "nesterov":
        py_setitem(options, _AT_NESTEROV, value)
    elif key == "betas":
        if len(value) != 2:
            raise Error("ValueError: " + where + ": betas must be two values (beta1, beta2)")
        py_setitem(options, _AT_BETA1, value[0])
        py_setitem(options, _AT_BETA2, value[1])
    elif key == "beta1":
        py_setitem(options, _AT_BETA1, value)
    elif key == "beta2":
        py_setitem(options, _AT_BETA2, value)
    elif key == "eps":
        py_setitem(options, _AT_EPS, value)
    elif key == "alpha":
        py_setitem(options, _AT_ALPHA, value)
    elif key == "centered":
        py_setitem(options, _AT_CENTERED, value)
    elif key == "lr_decay":
        py_setitem(options, _AT_LR_DECAY, value)
    elif key == "initial_accumulator_value":
        py_setitem(options, _AT_INITIAL_ACC, value)
    else:
        raise Error("ValueError: " + where + ": unknown parameter group key '" + key + "'")


def pgseq_compile_groups(
    dst: PythonObject, options: PythonObject, optimizer: String, sched: Int, kind: Int, groups: PythonObject,
    no_decay: PythonObject, b: PythonObject, context: PythonObject,
) raises:
    """Admit compile's `param_groups` and `no_decay` and store them in the compile options `dst` (nothing when
    both are None: the model trains without groups and pays nothing)."""
    if groups is None and no_decay is None:
        return
    pgseq_admit_schedule(sched)
    var given = Python.list()
    if groups is not None:
        if not (Bool(py=py_call(b.isinstance, groups, Python.tuple(b.list, b.tuple)))):
            raise Error("TypeError: param_groups must be a list of dicts")
        given = groups
    var suffixes = Python.list()
    if no_decay is not None:
        var items = Python.list(no_decay) if Bool(py=py_call(b.isinstance, no_decay, b.str)) else no_decay
        if not (Bool(py=py_call(b.isinstance, items, Python.tuple(b.list, b.tuple)))) or len(items) < 1:
            raise Error("TypeError: no_decay must be a parameter-name suffix such as 'bias', or a list of them")
        for k in range(len(items)):  # driver-loop: over the no_decay suffixes (compile, once)
            if not Bool(py=py_call(b.isinstance, items[k], b.str)) or len(items[k]) < 1:
                raise Error("TypeError: no_decay must hold parameter-name suffixes such as 'bias'")
            _ = py_call(suffixes.append, items[k])
    var layers = Python.list()
    var names = Python.list()
    var words = Python.list()
    var twins = len(suffixes) > 0
    for g in range(len(given)):  # driver-loop: over parameter groups (compile, once)
        var group = given[g]
        if py_type(group) is not b.dict:
            raise Error("TypeError: param_groups must be a list of dicts; " + _where(g) + " is not a dict")
        var has_layers = py_contains(group, PythonObject("layers"))
        var has_params = py_contains(group, PythonObject("params"))
        if has_layers == has_params:
            raise Error("ValueError: " + _where(g) + " takes 'layers' (layer indices) or 'params' (parameter names "
                        + "such as '0.weight'), one of the two")
        var members = py_item(group, PythonObject("layers") if has_layers else PythonObject("params"))
        if not (Bool(py=py_call(b.isinstance, members, Python.tuple(b.list, b.tuple)))):
            var field = String("layers") if has_layers else String("params")
            raise Error("TypeError: " + _where(g) + "['" + field + "'] must be a list")
        if len(members) < 1:
            raise Error("ValueError: " + _where(g) + " has no parameters")
        var ls = Python.list()
        var ns = Python.list()
        for i in range(len(members)):  # driver-loop: over the group's members (compile, once)
            var m = members[i]
            if has_layers:
                if not _integer(m, context) or Bool(py=m < PythonObject(0)):
                    raise Error("TypeError: " + _where(g) + "['layers'] holds layer indices (integers >= 0)")
                _ = py_call(ls.append, py_call(b.int, m))
            else:
                if not Bool(py=py_call(b.isinstance, m, b.str)):
                    raise Error("TypeError: " + _where(g) + "['params'] holds parameter names such as '0.weight' "
                                + "(model.state_dict()'s keys)")
                _ = py_call(ns.append, m)
        _ = py_call(layers.append, py_call(b.tuple, ls))
        _ = py_call(names.append, py_call(b.tuple, ns))
        var opts = py_call(b.list, options)
        var keys = py_call(b.list, group)
        for k in range(len(keys)):  # driver-loop: over the group's keys (compile, once)
            var key = String(py=py_call(b.str, keys[k]))
            if key == "layers" or key == "params":
                continue
            _put(opts, g, key, py_item(group, keys[k]), optimizer, kind, b)
        var w = _group_words(opts, optimizer, sched, b, False)
        for k in range(PGSEQ_GROUP_WORDS):  # small-loop(8: a group's words)
            _ = py_call(words.append, PythonObject(w[k]))
    # the default group: compile's own settings
    var base = py_call(b.list, options)
    var d = _group_words(base, optimizer, sched, b, False)
    for k in range(PGSEQ_GROUP_WORDS):  # small-loop(8: a group's words)
        _ = py_call(words.append, PythonObject(d[k]))
    if twins:
        # each group's no-decay twin (its words with weight_decay 0), the user groups' then the default's
        var U = len(given)
        for g in range(U + 1):  # driver-loop: over parameter groups (compile, once)
            for k in range(PGSEQ_GROUP_WORDS):  # small-loop(8: a group's words)
                var v = Float64(py=words[g * PGSEQ_GROUP_WORDS + k])
                _ = py_call(words.append, PythonObject(0.0 if k == 5 else v))
    var stored = Python.dict()
    py_setitem(stored, PythonObject("layers"), py_call(b.tuple, layers))
    py_setitem(stored, PythonObject("names"), py_call(b.tuple, names))
    py_setitem(stored, PythonObject("no_decay"), py_call(b.tuple, suffixes))
    py_setitem(stored, PythonObject("words"), py_call(b.tuple, words))
    py_setitem(dst, PythonObject("param_groups"), stored)


def pgseq_door_words(stored: PythonObject, program: List[Int], scalars: List[Float64]) raises -> List[Float64]:
    """The fit words of the stored groups against this description's tensors (once per fit or save call)."""
    if py_contains(stored, PythonObject("resolved")):
        var saved = py_item(stored, PythonObject("resolved"))
        var out = List[Float64]()
        for k in range(len(saved)):  # driver-loop: over the saved group words (once per call)
            out.append(Float64(py=saved[k]))
        return out^
    var spec = parse_model(program, model_float_words(scalars))
    var offsets = List[Int]()
    var layout = List[String]()
    model_layout(spec, offsets, layout)
    var pl = py_item(stored, PythonObject("layers"))
    var pn = py_item(stored, PythonObject("names"))
    var layers = List[List[Int]]()
    var names = List[List[String]]()
    for g in range(len(pl)):  # driver-loop: over parameter groups (once per call)
        var ls = List[Int]()
        for i in range(len(pl[g])):  # driver-loop: over the group's layers (once per call)
            ls.append(Int(py=pl[g][i]))
        layers.append(ls^)
        var ns = List[String]()
        for i in range(len(pn[g])):  # driver-loop: over the group's names (once per call)
            ns.append(String(py=pn[g][i]))
        names.append(ns^)
    var nd = py_item(stored, PythonObject("no_decay"))
    var no_decay = List[String]()
    for k in range(len(nd)):  # driver-loop: over the no_decay suffixes (once per call)
        no_decay.append(String(py=nd[k]))
    var pw = py_item(stored, PythonObject("words"))
    var words = List[Float64]()
    for k in range(len(pw)):  # driver-loop: over the groups' words (once per call)
        words.append(Float64(py=pw[k]))
    return pgseq_resolve(layout, layers, names, no_decay, words)


def pgseq_words_tuple(words: List[Float64]) raises -> PythonObject:
    """The group words as a tuple of floats (the `_param_groups` field and a checkpoint's resolved options)."""
    var b = native_loaded_module("builtins")
    var items = Python.list()
    for k in range(len(words)):  # driver-loop: over the group words (once per fit or load)
        _ = py_call(items.append, PythonObject(words[k]))
    return py_call(b.tuple, items)


def pgseq_resolved_options(words: List[Float64]) raises -> PythonObject:
    """A checkpoint's group words as the compile options' `param_groups` ({"resolved": words})."""
    var stored = Python.dict()
    py_setitem(stored, PythonObject("resolved"), pgseq_words_tuple(words))
    return stored
