# SPDX-License-Identifier: Apache-2.0
"""The device of one public FUNCTION call, as a typed native value: the ONE rule
every public function that computes shares (lane vd-functions, 2026-11-24;
docs/plans/VERIFY_THROUGH_DEVICE.md item 1).

A public function (a metric, a model-selection tool, a linear-algebra or
resampling function) takes the keyword-only `device=None | "gpu" | "cpu"`
with the estimators' meaning (core/device_route.mojo):

* None keeps today's route exactly: the door reads nothing else and runs
  what it ran before the parameter existed;
* "gpu" runs the device images; on a CPU-only installation it is refused
  in the estimators' words (`device_gpu_absent`); a function with no GPU
  implementation refuses it in words naming the function, and never runs
  its CPU route in its place (the selector refuses, never widens);
* "cpu" runs the CPU identical images. On a CPU-only installation, and for
  a function whose only implementation is the host one, that is today's
  route; on a GPU installation it is the CPU images loaded beside the GPU
  set (`host_beside`). A function with no CPU implementation refuses it in
  the estimators' `device_cpu_absent` words.

The word is read and admitted in the function's UNPACK
(providers/python/function_device_boundary_door.mojo `function_device_admit`)
into a `FunctionDevice`; RUN reads the one Bool `host_beside` and sees no
Python object. Nothing here is per row: one decision per public call, made
from two words the door already holds.

A model-selection tool hands its `device` to the member fits it runs; a
member estimator constructed with a different explicit `device` is refused
before anything runs (`function_device_member`).

Pure rule: no Python, no image; checked by checks/native_tests/function_device.mojo.
"""
from core.device_route import (
    DEVICE_CPU, DEVICE_GPU, DEVICE_INSTALL, device_cpu_absent, device_gpu_absent, device_name,
)


@fieldwise_init
struct FunctionDevice(ImplicitlyCopyable, Movable):
    """The admitted device of one public function call. `request` is the
    caller's word as given (DEVICE_INSTALL for None, DEVICE_GPU, DEVICE_CPU);
    `host_beside` is True only for an explicit "cpu" on a GPU installation
    for a function that has both implementations: its door then runs the
    CPU identical images loaded beside the GPU set. Every other admitted
    value runs the door's route of today."""
    var request: Int
    var host_beside: Bool


def function_device_default() -> FunctionDevice:
    """`device=None`: today's route."""
    return FunctionDevice(DEVICE_INSTALL, False)


def function_device_gpu_unimplemented(name: String) raises:
    """The one refusal of an explicit `device="gpu"` for a function that has no
    GPU implementation (its CPU route never runs in its place)."""
    raise Error("ValueError: device='gpu' is not available for " + name + ": it has no GPU implementation. "
        + "Use device='cpu', or leave device=None")


def function_device_decide(request: Int, cpu_only: Bool, has_gpu: Bool, has_cpu: Bool,
    name: String) raises -> FunctionDevice:
    """The device rule of a public function. `request` is the admitted code of
    the caller's word; `cpu_only` says this installation has no GPU set;
    `has_gpu` / `has_cpu` say which implementations the function has (a door
    constant, never looked up); `name` leads the refusal words."""
    if request == DEVICE_INSTALL:
        return function_device_default()
    if request == DEVICE_GPU:
        if not has_gpu:
            function_device_gpu_unimplemented(name)
        if cpu_only:
            device_gpu_absent()
        return FunctionDevice(DEVICE_GPU, False)
    if request != DEVICE_CPU:
        raise Error("RuntimeError: unknown device code " + String(request))
    if cpu_only:
        return FunctionDevice(DEVICE_CPU, False)
    if not has_cpu:
        device_cpu_absent(name, String("it has no CPU implementation"))
    # A function whose only implementation is the host one already runs
    # there; only a function with both runs the CPU images beside the GPU set.
    return FunctionDevice(DEVICE_CPU, has_gpu)


def function_device_member(request: Int, member: Int, tool: String, member_class: String) raises -> Int:
    """The device a model-selection tool's member fits run on: the tool's
    explicit `device` is handed to a member that names none; a member whose
    own explicit `device` differs is refused before anything runs. Codes are
    core/device_route.mojo's (DEVICE_INSTALL for None)."""
    if request == DEVICE_INSTALL:
        return member
    if member == DEVICE_INSTALL or member == request:
        return request
    raise Error("ValueError: " + tool + " was given device='" + device_name(request) + "' but its estimator "
        + member_class + " was constructed with device='" + device_name(member)
        + "'; give both the same device, or leave one of them None")
