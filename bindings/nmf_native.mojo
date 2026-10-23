# SPDX-License-Identifier: Apache-2.0
# NMF's native operations (dispatched by bindings/decomp_native.mojo): the
# parameter checks `_validate` / `_fit_transform` made, the regularization
# `_reg` computed (the same binary64 products), X converted once, then one
# compiled driver (x_decomp/nmf_owner.mojo: x_decomp_nmf_fit / _inverse, the
# kit steps the Python shell sequenced, the same entries, operands and order,
# so the same bits), and the fitted attributes stored as the shell stored them.
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_get, py_type, py_fields, py_setitem, py_contains
from bindings.py_typed_call import py_call_typed
from bindings.classical_arrays import classical_address, classical_host_f32
from bindings.decomp_inputs import decomp_input
from bindings.decomp_owner_common import (
    decomp_attr, decomp_check, decomp_entries, decomp_eq, decomp_f32, decomp_float, decomp_gemm_check, decomp_int,
    decomp_le, decomp_repr, decomp_seed, decomp_set_mode,
)

#: x_decomp/nmf_owner.mojo's init codes
comptime _INIT_RANDOM = 0
comptime _INIT_NNDSVD = 1
comptime _INIT_NNDSVDA = 2
comptime _INIT_NNDSVDAR = 3
comptime _INIT_GIVEN = 4
comptime _TRANSFORM = 5


def _contains(options: PythonObject, value: PythonObject) raises -> Bool:
    return Bool(py=py_call(Python.import_module("operator").contains, options, value))


def _beta(fields: PythonObject) raises -> Float64:
    """`NMF._beta`: the loss name's beta, or the value itself, one of 2, 1, 0."""
    var names = Python.dict()
    names["frobenius"] = 2.0
    names["kullback-leibler"] = 1.0
    names["itakura-saito"] = 0.0
    var loss = fields["beta_loss"]
    var value = py_call(names.get, loss, loss)
    if not _contains(Python.tuple(0.0, 1.0, 2.0), value):
        raise Error("beta_loss: 'frobenius', 'kullback-leibler' and 'itakura-saito' are carried; "
                    + "a general float beta is not")
    return decomp_float(value)


def _validate(fields: PythonObject) raises -> Float64:
    """`NMF._validate`'s parameter checks (the negativity count runs in the
    driver, on the cells); returns beta."""
    var beta = _beta(fields)
    var solver = fields["solver"]
    if beta != 2.0 and not decomp_eq(solver, PythonObject("mu")):
        raise Error("Invalid beta_loss parameter: solver 'cd' does not handle beta_loss other than 'frobenius'")
    if not _contains(Python.tuple("cd", "mu"), solver):
        raise Error("Invalid solver parameter: got " + decomp_repr(solver) + " instead of one of {'cd', 'mu'}")
    return beta


def _regs(fields: PythonObject, n: Int, d: Int) raises -> PythonObject:
    """`NMF._reg(n, d)`: [l1W, l1H, l2W, l2H] as the same binary64 products."""
    var alpha_h = fields["alpha_H"]
    var a_w = decomp_float(fields["alpha_W"])
    var a_h = a_w if decomp_eq(alpha_h, PythonObject("same")) else decomp_float(alpha_h)
    var l1 = decomp_float(fields["l1_ratio"])
    var fd = Float64(d)
    var fn_obj = Float64(n)
    return Python.list(fd * a_w * l1, fn_obj * a_h * l1, fd * a_w * (1.0 - l1), fn_obj * a_h * (1.0 - l1))


def _run(model: PythonObject, fields: PythonObject, x: PythonObject, nc: Int, init: Int, w_in: PythonObject,
         h_in: PythonObject, seed: Int, update_h: Bool, beta: Float64, context: PythonObject) raises -> PythonObject:
    """One x_decomp_nmf_fit call: (W, H, n_iter_, error)."""
    var entries = decomp_entries(model, fields, Python.tuple("x_decomp_nmf_fit"), context)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var regs = _regs(fields, n, d)
    var w = decomp_f32(Python.tuple(n, nc), context)
    var h = decomp_f32(Python.tuple(nc, d), context)
    var spare = classical_address(h)
    _ = h  # keep the Python object alive through the native call above
    var params = Python.list(
        classical_address(x), n, d, nc, init, classical_address(w_in) if w_in is not None else spare,
        classical_address(h_in) if h_in is not None else spare, seed,
        0 if decomp_eq(fields["solver"], PythonObject("cd")) else 1, 1 if update_h else 0,
        decomp_int(fields["max_iter"]), 1 if Bool(py=fields["shuffle"]) else 0, classical_address(w),
        classical_address(h))
    _ = w  # keep the Python object alive through the native call above
    _ = h  # keep the Python object alive through the native call above
    var floats = Python.list(beta, decomp_float(fields["tol"]), regs[0], regs[1], regs[2], regs[3])
    var got = py_call_typed(entries["x_decomp_nmf_fit"], params, floats)
    _ = h  # keep the Python object alive through the native call above
    _ = w  # keep the Python object alive through the native call above
    return Python.tuple(w, h, got[0], got[1])


def nmf_fit_transform(model: PythonObject, source: PythonObject, w_given: PythonObject, h_given: PythonObject,
                      context: PythonObject) raises -> PythonObject:
    """`NMF.fit_transform(X, W=None, H=None)`: W."""
    var fields = py_fields(model)
    decomp_set_mode(fields, context)
    var x = decomp_input(source, "NMF", context)
    var n = Int(py=x.shape[0])
    var d = Int(py=x.shape[1])
    var beta = _validate(fields)
    var init_value = fields["init"]
    var result: PythonObject
    if decomp_eq(init_value, PythonObject("custom")):
        var wm = decomp_input(w_given, "NMF", context)
        var hm = decomp_input(h_given, "NMF", context)
        var nc = Int(py=hm.shape[0])
        if Int(py=wm.shape[0]) != n or Int(py=wm.shape[1]) != nc or Int(py=hm.shape[1]) != d:
            raise Error("NMF: W must be (n_samples, n_components) and H (n_components, n_features) for init='custom'")
        result = _run(model, fields, x, nc, _INIT_GIVEN, wm, hm, decomp_seed(fields["random_state"]), True, beta,
                      context)
        _ = wm^
        _ = hm^
    else:
        var requested = fields["n_components"]
        var nc: Int
        if requested is None or decomp_eq(requested, PythonObject("auto")):
            nc = d
        else:
            nc = decomp_int(requested)
        var smaller = min(n, d)
        var name = init_value
        if not Bool(py=init_value):
            name = PythonObject("nndsvda") if nc <= smaller else PythonObject("random")
        if not _contains(Python.tuple("random", "nndsvd", "nndsvda", "nndsvdar"), name):
            raise Error("init=" + decomp_repr(name) + " is not supported here (random, nndsvd, nndsvda, nndsvdar)")
        var code = _INIT_RANDOM
        if decomp_eq(name, PythonObject("nndsvd")):
            code = _INIT_NNDSVD
        elif decomp_eq(name, PythonObject("nndsvda")):
            code = _INIT_NNDSVDA
        elif decomp_eq(name, PythonObject("nndsvdar")):
            code = _INIT_NNDSVDAR
        if code != _INIT_RANDOM and nc > smaller:
            raise Error("init = 'nndsvd' can only be used when n_components <= min(n_samples, n_features)")
        result = _run(model, fields, x, nc, code, PythonObject(None), PythonObject(None),
                      decomp_seed(fields["random_state"]), True, beta, context)
    var h = result[1]
    py_setitem(fields, "components_", h)
    py_setitem(fields, "n_components_", PythonObject(Int(py=h.shape[0])))
    py_setitem(fields, "n_iter_", PythonObject(Int(py=result[2])))
    py_setitem(fields, "n_features_in_", PythonObject(d))
    py_setitem(fields, "reconstruction_err_", PythonObject(Float64(py=result[3])))
    _ = x^
    return result[0]


def nmf_fit(model: PythonObject, source: PythonObject, params: PythonObject, context: PythonObject) raises -> PythonObject:
    """`NMF.fit(X, **params)`: fit_transform with the W / H keywords."""
    var b = Python.import_module("builtins")
    var keys = py_call(b.list, params.keys())
    for k in range(len(keys)):  # driver-loop: over the caller's keyword names
        if not _contains(Python.tuple("W", "H", "y"), keys[k]):
            raise Error("TypeError: fit_transform() got an unexpected keyword argument " + decomp_repr(keys[k]))
    _ = nmf_fit_transform(model, source, py_call(params.get, "W"), py_call(params.get, "H"), context)
    return model


def nmf_transform(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`NMF.transform`: W for the fitted components (H fixed)."""
    var fields = py_fields(model)
    decomp_check(model, fields, "components_")
    var x = decomp_input(source, "NMF", context)
    var d = Int(py=x.shape[1])
    var fitted = decomp_int(decomp_attr(model, fields, "n_features_in_"))
    if d != fitted:
        raise Error("X has " + String(d) + " features, but NMF is expecting " + String(fitted))
    var beta = _validate(fields)
    var h = classical_host_f32(fields["components_"], 2, "components_", context)
    var result = _run(model, fields, x, Int(py=h.shape[0]), _TRANSFORM, PythonObject(None), h,
                      decomp_seed(fields["random_state"]), False, beta, context)
    _ = x^
    _ = h^
    return result[0]


def nmf_inverse_transform(model: PythonObject, source: PythonObject, context: PythonObject) raises -> PythonObject:
    """`NMF.inverse_transform`: W components_."""
    var fields = py_fields(model)
    decomp_check(model, fields, "components_")
    var entries = decomp_entries(model, fields, Python.tuple("x_decomp_nmf_inverse"), context)
    var w = decomp_input(source, "NMF", context)
    var h = classical_host_f32(fields["components_"], 2, "components_", context)
    var n = Int(py=w.shape[0])
    var nc = Int(py=w.shape[1])
    decomp_gemm_check(nc, Int(py=h.shape[0]))
    var d = Int(py=h.shape[1])
    var out = decomp_f32(Python.tuple(n, d), context)
    _ = py_call_typed(entries["x_decomp_nmf_inverse"], Python.list(
        classical_address(w), n, nc, classical_address(h), d, classical_address(out)))
    _ = out  # keep the Python object alive through the native call above
    _ = h  # keep the Python object alive through the native call above
    _ = w  # keep the Python object alive through the native call above
    _ = w^
    _ = h^
    return out
