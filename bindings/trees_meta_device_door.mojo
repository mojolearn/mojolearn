# SPDX-License-Identifier: Apache-2.0
"""The device of a meta-estimator's members (lane vdf-meta, 2026-11-24):
VotingClassifier / VotingRegressor, StackingClassifier / StackingRegressor,
MultiOutputClassifier / MultiOutputRegressor, OneVsRestClassifier and
CalibratedClassifierCV (python/mojolearn/_expansion_trees.py
`_TreesWrapperBase`, `_mojolearn_device = True`).

The meta-estimator's own operations already run in its device route's
context (bindings/trees_route_boundary_door.mojo `trees_route_context`: the
x_trees glue, the label steps through `labels_scope`, the multi-output ridge
table). Its members are fitted as their own estimators, on their own route:
so a member clone stores the device the meta-estimator runs it on.

THE RULE (core/function_device.mojo `function_device_member`, the one rule a
tool hands its device to its member fits by): the meta-estimator's explicit
`device` is handed to a member that names none; a member constructed with a
different explicit `device` is refused in the shared words before anything
of the fit runs. `device=None` on the meta-estimator is today's route
exactly: each member keeps the device it names, and nothing here is read
past one identity compare.

A member that names no device and is handed `'cpu'` beside a GPU set must be
a class whose CPU route is connected: the ONE table the constructor's check
reads (python/mojolearn/_device_table.py, the trees context's `device_cpu`),
refused in the constructor's own words (`device_cpu_absent`) otherwise, as
if the user had constructed it with `device='cpu'`.

WHERE IT IS READ: the meta-estimator's `device` once per fit (the door's fit
functions), each member's once per fit (`meta_members_device`, before any
member is cloned or fitted), and each clone's once at its clone
(`meta_member_device_set`). Never per row, fold, output or member call.

THE META MEMBER KIND (providers/python/member_ensemble_boundary_door.mojo
`ensemble_member_admit`: a meta-estimator that is itself a member of a
search, a calibration or another meta-estimator): its nested estimators are
admitted on the device each of them names. So the admission reads a
DEVICE VIEW of the meta-estimator's fields (`meta_kind_device_fields`): with
an explicit `device`, every nested estimator that names none (and the
default base or final estimator the admission would make) is replaced by a
clone storing that word; the view is read for the work-list's children and
for the admission alike, so nested meta-estimators hand it down at any
depth. PACK stores the same word on every packed clone
(`meta_device_handed`, `meta_device_base`). Bagging and AdaBoost hand theirs
on in their own admission (`_bag_template`); a search and an RFE are not
this lane's (a search's candidate meta-estimator carries its own word).

WHERE THE WORD CAME FROM: a clone handed a meta-estimator's word records the
meta-estimator it came from (`_mojolearn_device_from`, the outermost one that
was given it), so a nested refusal names it: "<Outer> was given
device='cpu', handed to <Inner>, but its estimator ...".

Admission metadata only: no numerical work, nothing to split across cores.
"""
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_get, py_type, py_fields, py_setitem
from bindings.prep_native_estimator_door import prep_estimator_plan, prep_estimator_clone
from core.device_route import DEVICE_CPU, DEVICE_INSTALL, device_cpu_absent, device_name
from core.function_device import function_device_member
from core.ensemble_member import (
    META_STACK_CLASSIFIER, META_STACK_REGRESSOR, META_OVR, META_CALIBRATED, META_MULTI_CLASSIFIER,
    META_MULTI_REGRESSOR,
)
from providers.python.device_route_boundary_door import device_unpack


def meta_cpu_only(context: PythonObject) raises -> Bool:
    """This installation has no GPU set (the backend's `_CPU_ONLY`, as the
    constructor's device check reads it)."""
    var backend = py_get(context, PythonObject("backend_fields"))
    return backend is not None and py_get(backend, PythonObject("_CPU_ONLY")) is not None


def _class_cpu(cls: PythonObject, context: PythonObject, b: PythonObject) raises:
    """The constructor's CPU answer for a member class handed `'cpu'` beside
    a GPU set: the one table (`device_cpu`) and its words."""
    var table = py_get(context, PythonObject("device_cpu"))
    var answer = py_get(table, cls) if table is not None else PythonObject(None)
    if answer is None:
        device_cpu_absent(String(cls.__name__), String("its CPU route is not connected in this installation"))
    if py_type(answer) is b.str:
        device_cpu_absent(String(cls.__name__), String(answer))


comptime _FROM = "_mojolearn_device_from"


def meta_device_origin(model: PythonObject, device: PythonObject) raises -> String:
    """Where a meta-estimator's explicit `device` came from: the meta-estimator
    that handed it on (recorded on this one when it was cloned as a nested
    member), else its own class. "" for `device=None` (one identity compare)."""
    if device is None:
        return String("")
    var handed = py_get(py_fields(model), PythonObject(_FROM))
    if handed is not None:
        return String(handed)
    return String(py_type(model).__name__)


def _refuse_member(meta: String, origin: String, request: Int, own: Int, child: String) raises:
    """A member whose explicit device is not the meta-estimator's: the shared
    words when the meta-estimator was given the word itself, else naming the
    meta-estimator that handed it down."""
    if origin.byte_length() == 0 or origin == meta:
        _ = function_device_member(request, own, meta, child)
        return
    if own == DEVICE_INSTALL or own == request:
        return
    raise Error("ValueError: " + origin + " was given device='" + device_name(request) + "', handed to " + meta
        + ", but its estimator " + child + " was constructed with device='" + device_name(own)
        + "'; give both the same device, or leave one of them None")


def meta_members_device(device: PythonObject, ests: PythonObject, owner: String, origin: String,
                        context: PythonObject, b: PythonObject) raises:
    """Once per fit, before any member is cloned or fitted: every member of a
    meta-estimator whose `device` is the explicit word `device` either names
    the same word or none (refused in words otherwise, `_refuse_member`); one
    that names none and is handed `'cpu'` beside a GPU set is a class whose
    CPU route is connected (`_class_cpu`). `device=None`: returns at the
    identity compare."""
    if device is None:
        return
    var request = device_unpack(device, b)
    var cpu_beside = request == DEVICE_CPU and not meta_cpu_only(context)
    for i in range(len(ests)):  # driver-loop: over the members (once each per fit)
        var est = ests[i]
        var cls = py_type(est)
        var own = py_get(py_fields(est), PythonObject("device"))
        _refuse_member(owner, origin, request, device_unpack(own, b), String(cls.__name__))
        if own is None and cpu_beside:
            _class_cpu(cls, context, b)


def meta_member_device_set(device: PythonObject, origin: String, clone: PythonObject) raises:
    """A member clone that names no device stores the meta-estimator's
    explicit word and where it came from (admitted by `meta_members_device`
    for its original): its fit, its predictions and its later public calls
    run on that side, and a meta-estimator clone hands it on. One field read
    per clone."""
    var fields = py_fields(clone)
    if py_get(fields, PythonObject("device")) is None:
        py_setitem(fields, PythonObject("device"), device)
        py_setitem(fields, PythonObject(_FROM), PythonObject(origin))


def _handed(device: PythonObject, request: Int, cpu_beside: Bool, est: PythonObject, meta: String,
            origin: String, context: PythonObject, b: PythonObject) raises -> PythonObject:
    """`est` as the admission reads it under the meta-estimator's explicit
    word: itself when it names that word, a clone storing the word when it
    names none (its class checked for 'cpu' beside a GPU set), refused in
    words when it names another. A value that is no Mojolearn estimator is
    returned as given (its admission refuses it in its own words)."""
    if est is None or prep_estimator_plan(est, context) is None:
        return est
    var cls = py_type(est)
    var own = py_get(py_fields(est), PythonObject("device"))
    _refuse_member(meta, origin, request, device_unpack(own, b), String(cls.__name__))
    if own is not None:
        return est
    if cpu_beside:
        _class_cpu(cls, context, b)
    var clone = prep_estimator_clone(est, context)
    meta_member_device_set(device, origin, clone)
    return clone


def meta_kind_device_fields(est: PythonObject, fields: PythonObject, kind: Int, context: PythonObject,
                            b: PythonObject) raises -> PythonObject:
    """THE DEVICE VIEW (module note), once per admission of a nested or member
    meta-estimator `est` of META kind `kind`: `fields` itself for
    `device=None` (one identity compare) and for a kind this does not cover
    (Bagging, AdaBoost, a search, an RFE); else a shallow copy whose
    `estimators`, `final_estimator` and `estimator` entries are the members
    as `_handed` gives them, a default base or final estimator included."""
    if kind > META_CALIBRATED and kind != META_MULTI_CLASSIFIER and kind != META_MULTI_REGRESSOR:
        return fields
    var device = py_get(fields, PythonObject("device"))
    if device is None:
        return fields
    var request = device_unpack(device, b)
    var cpu_beside = request == DEVICE_CPU and not meta_cpu_only(context)
    var meta = String(py_type(est).__name__)
    var origin = meta_device_origin(est, device)
    var view = py_call(b.dict, fields)
    if kind == META_OVR or kind == META_CALIBRATED or kind == META_MULTI_CLASSIFIER or kind == META_MULTI_REGRESSOR:
        var base = py_get(fields, PythonObject("estimator"))
        if base is None and kind == META_CALIBRATED:
            var defaults = py_get(context, PythonObject("native_estimator_defaults"))
            if defaults is not None:
                base = py_get(defaults, PythonObject("LinearSVC"))
        py_setitem(view, PythonObject("estimator"), _handed(device, request, cpu_beside, base, meta, origin, context, b))
        return view
    var given = py_get(fields, PythonObject("estimators"))
    if given is not None and (py_type(given) is b.list or py_type(given) is b.tuple):
        var handed = Python.list()
        for i in range(len(given)):  # driver-loop: over the named members (once each per admission)
            var entry = given[i]
            if (py_type(entry) is b.tuple or py_type(entry) is b.list) and len(entry) == 2:
                var member = _handed(device, request, cpu_beside, entry[1], meta, origin, context, b)
                if member is entry[1]:
                    _ = py_call(handed.append, entry)
                else:
                    _ = py_call(handed.append, Python.tuple(entry[0], member))
            else:
                _ = py_call(handed.append, entry)  # its admission refuses it in its own words
        py_setitem(view, PythonObject("estimators"), handed)
    if kind == META_STACK_CLASSIFIER or kind == META_STACK_REGRESSOR:
        var final = py_get(fields, PythonObject("final_estimator"))
        if final is None:
            var defaults = py_get(context, PythonObject("native_estimator_defaults"))
            if defaults is not None:
                final = py_get(defaults, PythonObject(String("LogisticRegression") if kind == META_STACK_CLASSIFIER else String("Ridge")))
        py_setitem(view, PythonObject("final_estimator"),
                   _handed(device, request, cpu_beside, final, meta, origin, context, b))
    return view


def meta_device_handed(fields: PythonObject, meta: PythonObject, clone: PythonObject) raises:
    """PACK: a packed nested clone stores the meta-estimator's explicit word
    (as its admission's device view read it). `device=None`: one identity
    compare."""
    var device = py_get(fields, PythonObject("device"))
    if device is None:
        return
    meta_member_device_set(device, meta_device_origin(meta, device), clone)


def meta_device_base(fields: PythonObject, meta: PythonObject, base: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    """PACK of a unit kind (OneVsRest, Calibrated): the base its units are
    cloned from, storing the meta-estimator's explicit word when it names
    none (one clone per pack). `device=None`: `base` itself."""
    var device = py_get(fields, PythonObject("device"))
    if device is None or base is None:
        return base
    if py_get(py_fields(base), PythonObject("device")) is not None:
        return base
    var clone = prep_estimator_clone(base, context)
    meta_member_device_set(device, meta_device_origin(meta, device), clone)
    return clone
