# SPDX-License-Identifier: Apache-2.0
"""UNPACK of a member estimator's `device` for the orchestration owners
(model selection, the multi-GPU drivers): the stored constructor value read
out of the member's instance dictionary into the rule's code
(core/device_route_dw_msel_parallel.mojo).

A member is never asked through a Python method. Its `device` is the plain
stored value: None, `'gpu'` or `'cpu'`; a class that has no `device`
parameter yet has no such key and reads as None. Any other value is refused
in the pilot's words (core/linear_route.mojo `linear_device_refusal`).

Each read is once per public operation of the owner (one cross-validation,
one search fit, one driver call), never per fold, candidate or row.
"""
from std.python import PythonObject
from bindings.family_common_unpack import family_repr
from bindings.py_balanced import py_call, py_get, py_type
from core.linear_route import linear_device_code, linear_device_refusal
from core.device_route_dw_msel_parallel import (
    MEMBER_DEVICE_INSTALL, MEMBER_VENDOR_UNKNOWN, member_folds_on_host, member_own_device_admit,
)


def member_device_unpack(value: PythonObject) raises -> Int:
    """The code of a stored `device` value: None is this installation's
    device; exact `'gpu'` or `'cpu'`; anything else is refused in words."""
    if value is None:
        return MEMBER_DEVICE_INSTALL
    # An exact str only (no subclass, no conversion hook); the class is read
    # off a literal, and only for a value that is not None.
    if py_type(value) is py_type(PythonObject("")):
        var code = linear_device_code(String(value))
        if code >= 0:
            return code
    linear_device_refusal(family_repr(value))
    return MEMBER_DEVICE_INSTALL


def msel_process_vendor(context: PythonObject) raises -> Int:
    """The vendor word of this process's base binding (0 CPU only), from the
    `(mode, vendor)` key the model-selection bridge recorded at import
    (python/mojolearn/_native_msel_inputs.py `msel_native_key`). A process
    whose base binding recorded a refusal instead answers
    MEMBER_VENDOR_UNKNOWN; the first helper call raises that refusal."""
    var key = py_get(context["family_msel_entries"], PythonObject("msel_native_key"))
    if key is None:
        return MEMBER_VENDOR_UNKNOWN
    if py_type(key) is py_type(PythonObject("")):
        return MEMBER_VENDOR_UNKNOWN
    return Int(py=key[1])


def msel_member_host_folds(fields: PythonObject, context: PythonObject) raises -> Bool:
    """True when the estimator whose instance dictionary is `fields` takes
    its fold rows gathered on the host: its `device` is `'cpu'` and this
    process has a GPU set. The default (no key, or None) is one dictionary
    read and one compare, and reads nothing else."""
    var value = py_get(fields, PythonObject("device"))
    if value is None:
        return False
    var device = member_device_unpack(value)
    if device == MEMBER_DEVICE_INSTALL:
        return False
    return member_folds_on_host(device, msel_process_vendor(context))


def msel_own_device_admit(value: PythonObject, context: PythonObject) raises:
    """Check a search's own `device` where it is given (constructor,
    `set_params`): an unknown value, and an explicit `'gpu'` in a CPU-only
    process, are refused in words before the search changes. The default
    None is one compare and reads nothing."""
    if value is None:
        return
    var device = member_device_unpack(value)
    member_own_device_admit(device, msel_process_vendor(context))


def msel_params_name_device(params: PythonObject, b: PythonObject) raises -> Bool:
    """True when one candidate's parameter dict sets a `device` (its own
    `device`, or a nested member's `<step>__device`). `b` is the builtins
    module the calling door already admitted."""
    var names = py_call(b.list, params)
    for k in range(len(names)):  # driver-loop: over the parameter names of one candidate
        if py_type(names[k]) is not b.str:
            continue
        var name = String(names[k])
        if name == "device" or name.endswith("__device"):
            return True
    return False
