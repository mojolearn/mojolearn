# SPDX-License-Identifier: Apache-2.0
"""UNPACK of a member estimator's `device` for the orchestration owners
(model selection, the multi-GPU drivers), through the ONE route: the rule
and the words are core/device_route.mojo, the read and the checks are the
shared door's (providers/python/device_route_boundary_door.mojo
`device_request`, `device_admit`). This file holds no rule and no refusal
text of its own; it only says what an orchestration owner does with the
answer:

* model selection gathers the fold rows of a member that says
  `device="cpu"` beside a GPU set on the host (rows resident on the device
  are for members that run there);
* a multi-GPU driver, which has no CPU form, refuses such a member with the
  shared `device_cpu_absent` words.

A member is never asked through a Python method, and a member whose class
has no `device` parameter reads as None (`device_request`). The owners have
no estimator of their own to hold a route for a member they are handed per
call, so each read is once per public operation of the owner (one
cross-validation, one search fit, one driver call), never per fold,
candidate or row.
"""
from std.python import PythonObject
from bindings.py_balanced import py_call, py_get, py_type
from core.device_route import DEVICE_CPU, device_cpu_absent
from providers.python.device_route_boundary_door import device_admit, device_request

# The vendor word of a process whose base binding recorded no table key: it
# is not 0, so nothing is treated as CPU only on a guess.
comptime MEMBER_VENDOR_UNKNOWN = -1


def msel_process_vendor(context: PythonObject) raises -> Int:
    """The vendor word of this process's base binding (0 CPU only), from the
    `(mode, vendor)` key the model-selection bridge recorded at import
    (python/mojolearn/_native_msel_inputs.py `msel_native_key`). A process
    whose base binding recorded a refusal instead answers
    MEMBER_VENDOR_UNKNOWN; the first helper call raises that refusal."""
    var entries = context["family_msel_entries"]
    # Model selection's CPU scope beside a GPU set (bindings/
    # msel_function_device_door.mojo `msel_cpu_scope`, lane vdf-msel) points
    # `msel_native_key` at the CPU helper column and keeps this process's own
    # key here; every other context has none (one more read, on the paths
    # that ask the vendor: an explicit device, a search's vendor_used).
    var key = py_get(entries, PythonObject("msel_process_key"))
    if key is None:
        key = py_get(entries, PythonObject("msel_native_key"))
    if key is None:
        return MEMBER_VENDOR_UNKNOWN
    if py_type(key) is py_type(PythonObject("")):
        return MEMBER_VENDOR_UNKNOWN
    return Int(py=key[1])


def member_says_cpu(fields: PythonObject, b: PythonObject) raises -> Bool:
    """True when the estimator whose instance dictionary is `fields` says
    `device='cpu'` (the shared one read: no key, None and `'gpu'` answer
    False; any other value is refused in the shared words). `b` is the
    builtins module the calling door already admitted."""
    return device_request(fields, b) == DEVICE_CPU


def msel_member_host_folds(fields: PythonObject, b: PythonObject, context: PythonObject) raises -> Bool:
    """True when the estimator (or search) whose instance dictionary is
    `fields` takes its fold rows gathered on the host: its `device` is
    `'cpu'` and this process has a GPU set. The process vendor is read only
    for one that says `'cpu'`."""
    if not member_says_cpu(fields, b):
        return False
    return msel_process_vendor(context) != 0


def msel_own_device_admit(value: PythonObject, b: PythonObject, context: PythonObject) raises:
    """Check a search's own `device` where it is given (constructor,
    `set_params`) with the shared `device_admit`: an unknown value, and an
    explicit `'gpu'` in a CPU-only process, are refused in the shared words
    before the search changes. The default None is one compare and reads
    nothing."""
    if value is None:
        return
    device_admit(value, msel_process_vendor(context), b)


def member_multi_gpu_refusal(driver: String) raises:
    """A multi-GPU driver has no CPU form: the shared refusal of an explicit
    `device="cpu"` (core/device_route.mojo `device_cpu_absent`), with the
    driver as the thing that cannot run there and this sentence as why."""
    device_cpu_absent("parallel " + driver, "it spreads one fit or query across the GPUs listed in devices= "
        + "and has no CPU form; call the estimator's own method to run it on the CPU")


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
