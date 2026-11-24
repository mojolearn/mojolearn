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
admitted on the device each of them names, and that admission does not hand
the meta-estimator's device on. `meta_kind_device` therefore admits an
explicit device there only when every nested estimator runs on it already
(named equal, or None where None resolves to the same side), and refuses in
words otherwise, before anything runs: nothing runs on a device its caller
did not name. Bagging and AdaBoost hand theirs on in their own admission
(`_bag_template`); a search and an RFE are not this lane's.

Admission metadata only: no numerical work, nothing to split across cores.
"""
from std.python import PythonObject
from bindings.py_balanced import py_get, py_type, py_fields, py_setitem
from bindings.prep_native_estimator_door import prep_estimator_plan
from core.device_route import DEVICE_CPU, DEVICE_GPU, device_cpu_absent, device_installed, device_name
from core.function_device import function_device_member
from core.ensemble_member import (
    META_STACK_CLASSIFIER, META_STACK_REGRESSOR, META_CALIBRATED, META_MULTI_CLASSIFIER, META_MULTI_REGRESSOR,
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


def meta_members_device(device: PythonObject, ests: PythonObject, owner: String, context: PythonObject,
                        b: PythonObject) raises:
    """Once per fit, before any member is cloned or fitted: every member of a
    meta-estimator whose `device` is the explicit word `device` either names
    the same word or none (`function_device_member`'s words otherwise); one
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
        _ = function_device_member(request, device_unpack(own, b), owner, String(cls.__name__))
        if own is None and cpu_beside:
            _class_cpu(cls, context, b)


def meta_member_device_set(device: PythonObject, clone: PythonObject) raises:
    """A member clone that names no device stores the meta-estimator's
    explicit word (admitted by `meta_members_device` for its original):
    its fit, its predictions and its later public calls run on that side.
    One field read per clone."""
    var fields = py_fields(clone)
    if py_get(fields, PythonObject("device")) is None:
        py_setitem(fields, PythonObject("device"), device)


def _nested_refuse(meta: String, word: String, child: String) raises:
    raise Error("ValueError: " + meta + " was given device='" + word + "', but inside another estimator its "
        + "estimator " + child + " runs on the device it names itself (None: this installation's device). "
        + "Construct " + child + " with device='" + word + "' too, or leave " + meta + "'s device None")


def meta_kind_device(est: PythonObject, fields: PythonObject, kind: Int, children: PythonObject,
                     context: PythonObject, b: PythonObject) raises:
    """THE META MEMBER KIND (module note), once per admission of a nested or
    member meta-estimator `est` of META kind `kind` whose nested estimators
    are `children` (`_meta_children`): an explicit `device` is admitted only
    when every nested estimator runs on its side already; otherwise refused
    in words before anything runs. A kind that hands its device on itself
    (Bagging, AdaBoost) or that is not this lane's (a search, an RFE) and
    `device=None` return at once."""
    if kind > META_CALIBRATED and kind != META_MULTI_CLASSIFIER and kind != META_MULTI_REGRESSOR:
        return
    var device = py_get(fields, PythonObject("device"))
    if device is None:
        return
    var request = device_unpack(device, b)
    var cpu_only = meta_cpu_only(context)
    var runs = device_installed(request, 0 if cpu_only else 1)
    var install = DEVICE_CPU if cpu_only else DEVICE_GPU
    var meta = String(py_type(est).__name__)
    var word = device_name(request)
    for i in range(len(children)):  # driver-loop: over the nested estimators (once each per admission)
        var child = children[i]
        if child is None or prep_estimator_plan(child, context) is None:
            continue  # not a Mojolearn estimator: its admission refuses it in its own words
        var child_name = String(py_type(child).__name__)
        var own = py_get(py_fields(child), PythonObject("device"))
        _ = function_device_member(request, device_unpack(own, b), meta, child_name)
        if own is None and runs != install:
            _nested_refuse(meta, word, child_name)
    if runs == install:
        return
    # The defaults the admission makes itself name no device either.
    if kind == META_CALIBRATED and py_get(fields, PythonObject("estimator")) is None:
        _nested_refuse(meta, word, String("LinearSVC"))
    if kind == META_STACK_CLASSIFIER and py_get(fields, PythonObject("final_estimator")) is None:
        _nested_refuse(meta, word, String("LogisticRegression"))
    if kind == META_STACK_REGRESSOR and py_get(fields, PythonObject("final_estimator")) is None:
        _nested_refuse(meta, word, String("Ridge"))
