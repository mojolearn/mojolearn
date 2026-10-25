# SPDX-License-Identifier: Apache-2.0
# Inner-estimator calls of the prep owners (RFE, IterativeImputer) and the
# TargetEncoder's splitter, without a Python frame.
#
# A wrapped estimator is reached through its fixed native ABI plan, the one
# native cross-validation uses (python/mojolearn/_native_msel_dispatch.py
# `register`, bindings/msel_estimator_native.mojo): the model-selection
# owner's compiled entry clones it from its constructor schema
# (`family_msel_estimator_clone`) and runs its fit / predict / score as the
# compiled operation its plan names (`family_msel_estimator_call`). The
# estimator's Python `fit`, `predict`, `get_params` or `__init__` is never
# called. The entry and its context are connected at package import
# (python/mojolearn/__init__.py) as `native_estimator_entry` and
# `native_estimator_context` of the encode owner's context.
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.native_loaded_module import native_loaded_module
from bindings.optimizer_public_words import stash_python_error
from bindings.py_balanced import py_call, py_type, py_get, py_contains


def prep_entry_call(entry: PythonObject, args: PythonObject) raises -> PythonObject:
    """`entry(*args)` on a compiled entry; the exception it sets is restored
    at the door unchanged."""
    ref cpy = Python().cpython()
    var result = cpy.PyObject_Call(entry._obj_ptr, args._obj_ptr, PyObjectPtr())
    if not result:
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=result)


def prep_native_entry(context: PythonObject) raises -> PythonObject:
    """The model-selection owner's compiled entry (a builtin function, never a
    Python callable)."""
    var b = native_loaded_module("builtins")
    var entry = py_get(context, "native_estimator_entry")
    if entry is None or py_type(entry) is not py_type(b.len):
        raise Error("ImportError: mojolearn: the native estimator entry is unavailable")
    return entry


def prep_native_context(context: PythonObject) raises -> PythonObject:
    var found = py_get(context, "native_estimator_context")
    if found is None:
        raise Error("ImportError: mojolearn: the native estimator plans are not connected")
    return found


def prep_estimator_plan(est: PythonObject, context: PythonObject) raises -> PythonObject:
    """The connected native ABI plan of est's exact class, or None."""
    var b = native_loaded_module("builtins")
    var cls = py_type(est)
    if py_type(cls) is not b.type:
        return PythonObject(None)
    return py_get(prep_native_context(context)["family_msel_estimators"], cls)


def prep_estimator_admit(est: PythonObject, owner: String, context: PythonObject) raises:
    """Refuse an estimator without a connected native plan: a user class, a
    subclass of a mojolearn estimator, or a built-in whose native schema is
    still pending. Its methods would be Python code in the runtime."""
    if py_type(py_type(est)) is not native_loaded_module("builtins").type:
        raise Error("TypeError: native estimator admission refuses Python metaclass hooks")
    if prep_estimator_plan(est, context) is None:
        raise Error("TypeError: mojolearn: " + owner + " requires a mojolearn estimator; got "
                    + String(py_type(est).__name__))


def prep_estimator_has(est: PythonObject, method: String, context: PythonObject) raises -> Bool:
    """Whether est's native plan serves `method` (the plan's fixed method table)."""
    var plan = prep_estimator_plan(est, context)
    return plan is not None and py_contains(plan["methods"], PythonObject(method))


def prep_estimator_clone(est: PythonObject, context: PythonObject) raises -> PythonObject:
    """`type(est)(**est.get_params(deep=False))`, built natively from the
    constructor schema (no constructor or get_params call)."""
    return prep_entry_call(prep_native_entry(context),
                       Python.tuple(est, "family_msel_estimator_clone", Python.tuple(), prep_native_context(context)))


def prep_estimator_call(est: PythonObject, method: String, args: PythonObject,
                        context: PythonObject, keywords: PythonObject = PythonObject(None)) raises -> PythonObject:
    """`est.<method>(*args)` as the compiled operation est's plan names."""
    return prep_entry_call(prep_native_entry(context),
                       Python.tuple(est, "family_msel_estimator_call", Python.tuple(method, args, keywords),
                                    prep_native_context(context)))


def prep_native_split(cv: PythonObject, x: PythonObject, y: PythonObject, context: PythonObject) raises -> PythonObject:
    """`cv.split(x, y)` of a mojolearn splitter as the model-selection owner's
    `family_msel_split` operation (the operation its `split` calls)."""
    return prep_entry_call(prep_native_entry(context),
                       Python.tuple(cv, "family_msel_split", Python.tuple(x, y, PythonObject(None)),
                                    prep_native_context(context)))


def prep_estimator_attribute(est: PythonObject, name: PythonObject, context: PythonObject) raises -> PythonObject:
    # RFE's attribute path reads only connected native estimator state or
    # an explicitly registered native attribute operation (SVM coef_).
    # operator.attrgetter/getattr could invoke arbitrary Python descriptors.
    var b = native_loaded_module("builtins")
    if py_type(name) is not b.str:
        raise Error("TypeError: native estimator attribute names must be exact strings")
    var parts = py_call(name.split, ".")
    var current = est
    for k in range(len(parts)):  # driver-loop: the explicit attribute path
        var plan = prep_estimator_plan(current, context)
        if plan is None:
            raise Error("TypeError: estimator attribute paths require connected native estimator state")
        if prep_estimator_has(current, String(parts[k]), context):
            current = prep_estimator_call(current, String(parts[k]), Python.tuple(), context)
        else:
            var fields = py_call(plan["fields_get"], current, py_type(current))
            current = py_get(fields, parts[k])
        if current is None:
            return current
    return current
