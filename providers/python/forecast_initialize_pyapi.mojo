# SPDX-License-Identifier: Apache-2.0
# Constructor admission, normalization and state commit for forecast families.
from std.python import Python, PythonObject
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_setitem, py_contains
from bindings.sequential_admission import _one_of
from providers.python.autoarima_lifecycle_pyapi import _fields, _integer
from providers.python.forecast_owner_pyapi import forecast_commit


def _scalar(value: PythonObject, context: PythonObject) raises:
    var cls = py_type(value)
    if value is None or cls is py_type(PythonObject("")) or cls is py_type(PythonObject(True)) or cls is py_type(PythonObject(0)) or cls is py_type(PythonObject(0.0)) or _one_of(cls, context["scalar_types"]):
        return
    raise Error("TypeError: forecast constructor settings require native scalar values")


def forecast_initialize(model: PythonObject, kind: String, args: PythonObject, context: PythonObject) raises -> PythonObject:
    var records = context["xseq_" + kind + "_classes"]
    var cls = py_type(model)
    var found = False
    var fixed_model = PythonObject(None)
    for i in range(len(records)):  # driver-loop: over connected forecast class choices
        if cls is records[i][0]:
            found = True
            fixed_model = records[i][1]
    if not found:
        raise Error("TypeError: forecast constructor requires an exact connected model class")
    var fields = _fields(model, cls)
    var updates = Python.dict()
    var lifecycle = context["xseq_lifecycle_context"]
    for i in range(len(args)):  # driver-loop: over constructor scalar argument words
        _scalar(args[i], lifecycle)
    if kind == "croston":
        py_setitem(updates, "alias", args[0])
        py_setitem(updates, "numeric_mode", args[1])
        forecast_commit(fields, updates, context["xseq_forecast_ownership"])
        return PythonObject(None)
    var b = native_loaded_module("builtins")
    if kind == "theta":
        var decomposition = args[1]
        if not Bool(py=decomposition == "multiplicative") and not Bool(py=decomposition == "additive"):
            raise Error("ValueError: decomposition_type must be 'multiplicative' or 'additive'")
        var choice = args[2]
        if choice is not None and not py_contains(context["xseq_theta_codes"], choice):
            raise Error("ValueError: model must be STM, OTM, DSTM, DOTM or None")
        py_setitem(updates, "season_length", _integer(args[0], lifecycle))
        py_setitem(updates, "decomposition_type", decomposition)
        py_setitem(updates, "model", choice if fixed_model is None else fixed_model)
        py_setitem(updates, "initial_smoothed", args[3])
        py_setitem(updates, "alpha", args[4])
        py_setitem(updates, "theta", args[5])
        py_setitem(updates, "numeric_mode", args[6])
        forecast_commit(fields, updates, context["xseq_forecast_ownership"])
        return PythonObject(None)
    if kind == "garch":
        if not Bool(py=args[3] == "Constant") and not Bool(py=args[3] == "Zero"):
            raise Error("NotImplementedError: GARCH: mean must be Constant or Zero")
        if not Bool(py=args[4] == "normal"):
            raise Error("NotImplementedError: GARCH: dist must be normal")
        if not Bool(py=args[5] == PythonObject(2.0)):
            raise Error("NotImplementedError: GARCH: power != 2 is not implemented")
        var p = _integer(args[0], lifecycle)
        var o = _integer(args[1], lifecycle)
        var q = _integer(args[2], lifecycle)
        if min(p, min(o, q)) < 0 or p + o + q < 1:
            raise Error("ValueError: GARCH: p, o, q >= 0 with at least one term")
        if o > 0 and p == 0:
            raise Error("ValueError: GARCH: o > 0 needs p > 0 (arch refuses it too)")
        py_setitem(updates, "p", p)
        py_setitem(updates, "o", o)
        py_setitem(updates, "q", q)
        py_setitem(updates, "mean", args[3])
        py_setitem(updates, "dist", args[4])
        py_setitem(updates, "power", args[5])
        py_setitem(updates, "numeric_mode", args[6])
        forecast_commit(fields, updates, context["xseq_forecast_ownership"])
        return PythonObject(None)
    # ETS and its DampedETS constructor share the same complete normalization.
    var damped_ctor = len(args) == 6
    var error = args[1]
    if damped_ctor and py_type(error) is not py_type(PythonObject("")):
        raise Error("TypeError: DampedETS error requires a native string")
    var text = String(error) + "AN" if damped_ctor else String(py=py_call(b.str, args[1]))
    if text.byte_length() != 3:
        raise Error("ValueError: ETS: model is three letters, error / trend / season")
    var e = String(text[byte=0:1])
    var tr = String(text[byte=1:2])
    var season = String(text[byte=2:3])
    if text.find("Z") >= 0:
        raise Error("NotImplementedError: ETS: automatic ('Z') model selection is not implemented")
    if tr == "M":
        raise Error("NotImplementedError: ETS: multiplicative trend is not implemented")
    if (e != "A" and e != "M") or (tr != "N" and tr != "A") or (season != "N" and season != "A" and season != "M"):
        raise Error("ValueError: ETS: invalid model (error A|M, trend N|A, season N|A|M)")
    var damped = True if damped_ctor else Bool(py=args[2])
    if damped and tr == "N":
        raise Error("ValueError: ETS: a damped model needs a trend")
    var period = _integer(args[0], lifecycle)
    if season != "N" and period == 1:
        raise Error("ValueError: ETS: Nonseasonal data (a seasonal model needs season_length > 1)")
    if season != "N" and period > 58:
        raise Error("NotImplementedError: ETS: season_length above 58 is not implemented (Nelder-Mead over at most 63 coordinates)")
    if e == "A" and season == "M":
        raise Error("ValueError: ETS: Forbidden model combination (additive errors with a multiplicative season)")
    py_setitem(updates, "season_length", period)
    py_setitem(updates, "model", text)
    py_setitem(updates, "damped", damped)
    py_setitem(updates, "alpha", args[2] if damped_ctor else args[3])
    py_setitem(updates, "beta", args[3] if damped_ctor else args[4])
    py_setitem(updates, "phi", args[4] if damped_ctor else args[5])
    py_setitem(updates, "gamma", PythonObject(None) if damped_ctor else args[6])
    py_setitem(updates, "numeric_mode", args[5] if damped_ctor else args[7])
    if damped_ctor:
        py_setitem(updates, "error", error)
    forecast_commit(fields, updates, context["xseq_forecast_ownership"])
    return PythonObject(None)
