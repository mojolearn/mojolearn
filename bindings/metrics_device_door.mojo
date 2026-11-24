# SPDX-License-Identifier: Apache-2.0
# `device=` of the public metrics (`mojolearn.metrics`, lane vdf-metrics,
# 2026-11-24): the family-local UNPACK helpers over the ONE function device
# rule (core/function_device.mojo, providers/python/function_device_boundary_door.mojo).
#
# Every public metric shell passes its `device` as the LAST item of its
# argument tuple. The three family doors (bindings/metrics_cls_native_unpack.mojo
# `metrics_cls_public`, bindings/metrics_exp_native_unpack.mojo
# `metrics_exp_public`, bindings/metrics_reg_door.mojo `metrics_reg_public`)
# read it once, first, with `metric_device_admit`, and run the metric with
#   * the binding `estimator_binding_device` selects (the `_mojolearn_metrics`
#     kernel entries; their CPU image is `_mojolearn_metrics_host`), and
#   * the context `function_device_scope` gives (the x_metrics programs and
#     the label steps; the CPU scope carries the `_mojolearn_x_metrics_host`
#     runner entries and the host label kernels).
# Every public metric has both implementations: METRICS_HAS_GPU and
# METRICS_HAS_CPU are True for the whole family (no metric calls a kernel
# entry of a binding without a CPU image).
#
# `device=None` costs one identity compare here and one Bool test in each of
# `function_device_scope` and `estimator_binding_device`: nothing is looked
# up, no String is built, and the door runs what it ran before.
from std.python import PythonObject
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_get, py_type
from core.device_route import DEVICE_CPU, device_cpu_fast
from core.function_device import FunctionDevice, function_device_default
from core.mode_answer import MODE_FAMILY_RESOLVER, answer_mode_word
from providers.python.function_device_boundary_door import function_device_admit

comptime METRICS_HAS_GPU = True
comptime METRICS_HAS_CPU = True


def metric_device_admit(value: PythonObject, name: StaticString, context: PythonObject) raises -> FunctionDevice:
    """The admitted `device` of one public metric call. `value` is the shell's
    last argument item; `name` leads the refusal words (built only for an
    explicit device)."""
    if value is None:
        return function_device_default()
    return function_device_admit(value, String(name), METRICS_HAS_GPU, METRICS_HAS_CPU, context,
                                 native_loaded_module("builtins"))


def metric_device_identical(mode: PythonObject, context: PythonObject) raises:
    """Only for an explicit `device="cpu"` beside a GPU set: the CPU images are
    the identical ones, so a metric whose effective `numeric_mode` is 'fast'
    is refused in the estimators' words (`device_cpu_fast`), exactly as
    `estimator_binding_device` refuses it for a kernel entry. A word that is
    not a mode is left to the door's own numeric_mode refusal."""
    var word = mode
    if word is None:
        word = context["backend_fields"]["_DEFAULT_MODE"]
    var b = native_loaded_module("builtins")
    if py_type(word) is not b.str:
        return
    var selected = String(py=py_call(py_call(word.strip).lower))
    if selected != "identical" and selected != "fast":
        return
    answer_mode_word(selected, MODE_FAMILY_RESOLVER)
    if selected == "fast":
        device_cpu_fast()


def metric_device_inputs(device: FunctionDevice, args: PythonObject, end: Int, context: PythonObject) raises:
    """UNPACK refusal of a device-resident input (`mojolearn._arena_io.DeviceRows`,
    the fold view model selection hands an estimator) with an explicit
    `device="cpu"` (addendum 3 item 4): the rows live in a GPU binding's
    store and the CPU side would have to download them. Called only for an
    explicit device word (never for `device=None`); one type compare per
    argument of the call."""
    if device.request != DEVICE_CPU:
        return
    var programs = py_get(context, "family_metrics_prog_context")
    if programs is None:
        return
    var rows_class = py_get(programs, "maxabs_rows_class")
    if rows_class is None:
        return
    for index in range(end):  # driver-loop: over the public metric's arguments
        if py_type(args[index]) is rows_class:
            raise Error("ValueError: device='cpu' cannot take a device-resident input (DeviceRows, argument "
                + String(index) + "): its rows live on the GPU. Pass a host array, or leave device=None")
